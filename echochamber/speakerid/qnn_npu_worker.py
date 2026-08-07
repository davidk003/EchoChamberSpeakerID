"""ARM64 NPU inference worker for CAM++ (QNN HTP backend).

Deliberately minimal: no wav/fbank logic, no torch (PyTorch ships no Windows
ARM64 wheel, but ``onnxruntime-qnn``'s HTP backend only loads inside a native
ARM64 process -- so this worker only does ONNX Runtime inference and nothing
else). Run under a native ARM64 interpreter:

    python -m echochamber.speakerid.qnn_npu_worker --check   # confirm QNN EP is active
    python -m echochamber.speakerid.qnn_npu_worker            # frame-protocol worker

Driven by ``echochamber/speakerid/qnn_driver_worker.py`` (in the x64 driver
venv) over :mod:`echochamber.speakerid.protocol` frames on a *duplicated*
stdout handle -- see :func:`_protect_stdout` for why not the real one.
Ported from the sibling ``cam-script`` repository's ``campplus_qnn_infer.py``,
which hands data across via temp files instead specifically to dodge this
same problem; confirmed on real Hexagon NPU hardware that the frame protocol
works too, once stdout is protected.

**QNN's native HTP graph compiler writes its progress bar directly to the
process's stdout file descriptor, bypassing Python's I/O layer entirely.**
This is not a logging-configuration problem -- ``sys.stdout`` and
``logging.basicConfig(stream=...)`` never see it, because the native library
(invoked from inside :func:`make_session`, during graph compilation on
Windows ARM64 hardware) writes straight to OS file descriptor 1.  On this
project's own frame protocol, that means literal "Starting stage: ..."
progress text lands *inside* a length-prefixed binary stream, and every frame
after it desyncs.  This was verified end-to-end on a Snapdragon device: the
worker looked like it hung (the ``NpuWorkerHandle`` reader thread waiting for
a READY frame that never parses cleanly), when what had actually happened was
the compiler's own stdout writes corrupting the channel before the READY
frame's header could be read intact.  :func:`_protect_stdout` -- called
before any session is created -- duplicates the *real* file descriptor 1 to a
private one this module keeps for itself, then repoints fd 1 at ``NUL``, so
whatever the native compiler writes there is discarded and the frame protocol
rides on a channel the compiler cannot reach.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from typing import Any, BinaryIO, Sequence

from echochamber.speakerid.protocol import (
    FrameKind,
    ProtocolError,
    encode_json,
    read_frame,
    write_frame,
)

__all__ = ["build_parser", "main", "make_session", "run"]

_LOGGER = logging.getLogger("echochamber.speakerid.qnn_npu_worker")
"""Logger for this module.  Writes to stderr; the protected stdout duplicate
is the frame channel -- see :func:`_protect_stdout`.
"""

_EP_NAME: str = "QNNExecutionProvider"

_registered = False


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the worker."""
    parser = argparse.ArgumentParser(
        prog="python -m echochamber.speakerid.qnn_npu_worker",
        description=(
            "QNN HTP inference worker for CAM++. Reads length-prefixed fbank "
            "frames on stdin and writes embedding frames on stdout; not meant "
            "to be run by hand."
        ),
    )
    parser.add_argument(
        "--onnx-path",
        required=True,
        help="path to the exported static-shape ONNX graph",
    )
    parser.add_argument(
        "--htp-performance-mode",
        default="burst",
        help="onnxruntime-qnn htp_performance_mode (default: burst)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print active providers and exit, without serving frames",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose QNN EP node-placement logging, written to stderr",
    )
    return parser


def set_binary_mode(stream: BinaryIO) -> BinaryIO:
    """Put ``stream``'s file descriptor into binary mode on Windows.

    See :func:`echochamber.voicegate.worker.set_binary_mode` for why this
    matters: raw float32 bytes contain ``0x0A`` constantly, and text-mode
    ``\\n``/``\\r\\n`` translation corrupts them silently.
    """
    try:
        import msvcrt  # noqa: PLC0415 - Windows only; absent elsewhere
    except ImportError:
        return stream
    try:
        msvcrt.setmode(stream.fileno(), os.O_BINARY)
    except (AttributeError, OSError, ValueError):
        pass
    return stream


def _protect_stdout() -> BinaryIO:
    """Move the frame channel off the OS-level stdout file descriptor.

    Must be called **before** :func:`make_session` -- the QNN native graph
    compiler that runs during session creation writes its progress bar
    straight to file descriptor 1, and once that has happened any frame
    already in flight on the real stdout is corrupted beyond recovery.  See
    the module docstring for how this was diagnosed.

    Returns:
        A binary stream backed by a private duplicate of the original fd 1.
        Callers must use this, not ``sys.stdout``, for every frame written
        from here on; the real fd 1 is repointed at ``NUL`` and is from this
        point permanently unusable as a data channel for this process.
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
    finally:
        os.close(devnull_fd)
    return os.fdopen(saved_fd, "wb")


def _register_qnn_once() -> None:
    """Register the QNN execution provider plugin library, once per process.

    ``onnxruntime-qnn`` 2.x ships QNN as a plugin EP: it must be explicitly
    registered (as a shared library path) before it shows up as a selectable
    device, unlike the older bundled-EP packaging model.
    """
    global _registered  # noqa: PLW0603
    if _registered:
        return
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    ort.register_execution_provider_library(_EP_NAME, qnn_ep.get_library_path())
    _registered = True


def make_session(
    onnx_path: str, htp_performance_mode: str = "burst", verbose: bool = False
) -> Any:
    """Build an ONNX Runtime session with the QNN HTP execution provider.

    Args:
        onnx_path: Path to the exported static-shape ONNX graph.
        htp_performance_mode: ``onnxruntime-qnn``'s performance mode knob.
        verbose: Enable QNN EP partition/fallback logging on stderr.

    Returns:
        A ready ``onnxruntime.InferenceSession``.

    Raises:
        RuntimeError: If no QNN execution provider device is found after
            registration -- this process is not running on ARM64 hardware
            with the Hexagon NPU driver installed.
    """
    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    _register_qnn_once()

    so = ort.SessionOptions()
    if verbose:
        so.log_severity_level = 0  # VERBOSE: prints QNN EP partition/fallback decisions

    qnn_devices = [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME]
    if not qnn_devices:
        raise RuntimeError("no QNN execution provider device found after registration")
    ep_options = {
        "backend_path": qnn_ep.get_qnn_htp_path(),
        "htp_performance_mode": htp_performance_mode,
        "enable_htp_fp16_precision": "1",
    }
    so.add_provider_for_devices(qnn_devices, ep_options)

    return ort.InferenceSession(onnx_path, sess_options=so)


def run(
    stdin: BinaryIO,
    stdout: BinaryIO,
    onnx_path: str,
    htp_performance_mode: str = "burst",
    verbose: bool = False,
) -> int:
    """Load the QNN session, announce readiness, then serve frames until told to stop.

    Args:
        stdin: Binary stream carrying frames from the driver process.
        stdout: Binary stream result frames are written to.  Callers running
            for real must pass the stream :func:`_protect_stdout` returns,
            obtained *before* this function runs -- session creation below is
            exactly the call that would otherwise corrupt it.  Tests may pass
            any binary stream, since nothing here touches file descriptor 1
            directly.
        onnx_path: Path to the exported static-shape ONNX graph.
        htp_performance_mode: ``onnxruntime-qnn``'s performance mode knob.
        verbose: Enable QNN EP partition/fallback logging on stderr.

    Returns:
        ``0`` after a clean shutdown, ``1`` if session creation failed or the
        stream desynchronised.
    """
    try:
        session = make_session(onnx_path, htp_performance_mode, verbose)
    except BaseException as exc:  # noqa: BLE001 - reported to the driver as ERROR
        _send_error(stdout, exc)
        return 1

    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    feat_dim = session.get_inputs()[0].shape[-1]
    t_max = session.get_inputs()[0].shape[-2]
    _LOGGER.info("loaded %r: t_max=%d feat_dim=%d", onnx_path, t_max, feat_dim)

    try:
        write_frame(
            stdout,
            FrameKind.READY,
            encode_json({"onnx_path": onnx_path, "t_max": t_max, "feat_dim": feat_dim}),
        )
        return _serve(stdin, stdout, session, in_name, out_name, feat_dim)
    except BaseException as exc:  # noqa: BLE001 - last-ditch report to the driver
        _send_error(stdout, exc)
        return 1


def _serve(
    stdin: BinaryIO,
    stdout: BinaryIO,
    session: Any,
    in_name: str,
    out_name: str,
    feat_dim: int,
) -> int:
    """Read EMBED frames until shutdown, end of stream, or a protocol failure."""
    import numpy as np

    while True:
        try:
            frame = read_frame(stdin)
        except ProtocolError as exc:
            _send_error(stdout, exc)
            return 1

        if frame is None:
            _LOGGER.info("stdin closed; exiting")
            return 0

        kind, payload = frame
        if kind is FrameKind.EMBED:
            feat = np.frombuffer(payload, dtype=np.float32).reshape(1, -1, feat_dim)
            (raw,) = session.run([out_name], {in_name: feat})
            emb = raw[0].astype(np.float32)
            write_frame(stdout, FrameKind.RESULT, emb.tobytes())
        elif kind is FrameKind.SHUTDOWN:
            _LOGGER.info("shutdown requested; exiting")
            return 0
        elif kind is FrameKind.RESET:
            pass  # Stateless model; nothing to discard.
        else:
            _LOGGER.warning("ignoring unexpected %s frame from the driver", kind.name)


def _send_error(stdout: BinaryIO, exc: BaseException) -> bool:
    """Try to report ``exc`` to the driver process as an ERROR frame."""
    message = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).strip()
    if not message:
        message = f"{type(exc).__name__}: {exc}"
    try:
        write_frame(stdout, FrameKind.ERROR, message.encode("utf-8", "replace"))
    except BaseException:  # noqa: BLE001 - the pipe is gone; stderr still has it
        _LOGGER.exception("could not report the failure to the driver")
        return False
    return True


def _configure_logging(verbose: bool) -> None:
    """Point the root logger at stderr, at DEBUG if ``verbose`` else INFO."""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, switch stdio to binary, and run the frame loop.

    Args:
        argv: Argument list *without* the program name, or ``None`` to take
            ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` on a clean shutdown, non-zero otherwise.
    """
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    _configure_logging(args.verbose)

    # Must happen before make_session() in every path below: that call is
    # what triggers the QNN native graph compiler, and once it has written to
    # the real fd 1 there is no recovering a frame already in flight there.
    # --check does not serve frames, but still creates a session, so it goes
    # through the same protection -- its own diagnostic print() below targets
    # the *protected* stream deliberately, not the now-discarded real stdout.
    protected_stdout = _protect_stdout()
    set_binary_mode(protected_stdout)

    if args.check:
        session = make_session(args.onnx_path, args.htp_performance_mode, args.verbose)
        providers = session.get_providers()
        protected_stdout.write(f"active providers: {providers}\n".encode("utf-8"))
        protected_stdout.flush()
        if providers[0] != _EP_NAME:
            sys.exit(f"error: {_EP_NAME} is not the active provider: {providers}")
        return 0

    stdin = set_binary_mode(sys.stdin.buffer)
    return run(stdin, protected_stdout, args.onnx_path, args.htp_performance_mode, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
