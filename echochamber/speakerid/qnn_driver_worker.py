"""The x64 driver process: fbank extraction, then drives the ARM64 NPU worker.

This is the *middle* hop of the three-process chain
:class:`echochamber.speakerid.qnn_subprocess.QnnEmbedder` sits in front of:

    main process (ARM64, native)
        <-- speakerid.protocol frames -->
    this driver process (x64, has torch)
        <-- speakerid.protocol frames -->
    qnn_npu_worker (ARM64, native, no torch)

It exists because CAM++'s feature extraction (waveform -> 80-dim Kaldi fbank)
needs ``torch``/``torchaudio``, which has no Windows ARM64 wheel, while the
Hexagon NPU inference step needs ``onnxruntime-qnn``'s HTP backend, which only
loads inside a *native* ARM64 process. Neither constraint can be satisfied by
one process, so this one sits in the middle: it runs under an x64 interpreter
(so torch installs), and spawns ``qnn_npu_worker`` as its own child under a
*separate*, native-ARM64 interpreter (so the HTP backend loads), forwarding
frames between them. This mirrors the sibling ``cam-script`` repository's
``campplus_qnn.py`` -- which does the same split, but hands data to its worker
over temp files rather than a pipe.

Run under an **x64** interpreter, exactly like the Vosk recogniser worker:

    python -m echochamber.speakerid.qnn_driver_worker --onnx-path ... --npu-python ...

Driven by :mod:`echochamber.speakerid.qnn_subprocess` (in the main ARM64
process) over :mod:`echochamber.speakerid.protocol` frames on stdin/stdout.

**Stdout carries binary frames, so nothing else may ever touch it** -- see
:mod:`echochamber.voicegate.worker`'s docstring, which this module mirrors
frame-for-frame reasoning.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, BinaryIO, Sequence

from echochamber.speakerid.protocol import (
    FrameKind,
    ProtocolError,
    encode_json,
    read_frame,
    write_frame,
)

__all__ = ["build_parser", "main", "run"]

_LOGGER = logging.getLogger("echochamber.speakerid.qnn_driver_worker")

_DEFAULT_NPU_MODULE: str = "echochamber.speakerid.qnn_npu_worker"

_NPU_READY_TIMEOUT_S: float = 60.0
"""How long to wait for the NPU worker's own READY frame.

Generous: the QNN HTP backend compiles the graph on session creation, which
can take several seconds the first time a device's cache is cold.
"""

_NPU_CALL_TIMEOUT_S: float = 120.0
"""Per-call ceiling for the NPU worker to return a RESULT frame.

Matches the sibling ``cam-script`` repository's ``campplus_qnn.py``, whose
comment notes this covers HTP graph dispatch, not just a memcpy.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the driver process."""
    parser = argparse.ArgumentParser(
        prog="python -m echochamber.speakerid.qnn_driver_worker",
        description=(
            "CAM++ QNN driver: extracts fbank features and forwards them to "
            "a native-ARM64 NPU inference worker. Reads length-prefixed "
            "sample frames on stdin and writes embedding frames on stdout; "
            "not meant to be run by hand."
        ),
    )
    parser.add_argument(
        "--onnx-path",
        required=True,
        help="path to the exported static-shape ONNX graph, passed through to the NPU worker",
    )
    parser.add_argument(
        "--npu-python",
        default=None,
        help=(
            "native ARM64 interpreter to run the NPU worker under; falls "
            "back to the CAMPPLUS_QNN_PYTHON environment variable, then to "
            "the interpreter currently running this driver"
        ),
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=600,
        help="fixed fbank-frame length the exported graph accepts (default: 600)",
    )
    parser.add_argument(
        "--npu-module",
        default=_DEFAULT_NPU_MODULE,
        help=argparse.SUPPRESS,  # testing seam: point at tests.fake_qnn_npu_worker
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="logging level for this process's own diagnostics, written to stderr",
    )
    return parser


def set_binary_mode(stream: BinaryIO) -> BinaryIO:
    """Put ``stream``'s file descriptor into binary mode on Windows.

    See :func:`echochamber.voicegate.worker.set_binary_mode` for why raw
    float32 sample bytes need this.
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


class NpuWorkerHandle:
    """Owns the child NPU-inference process and its frame protocol.

    A small, single-purpose cousin of
    :class:`echochamber.voicegate.subprocess_recognizer.SubprocessRecognizer`:
    one reader thread drains stdout so a full pipe cannot deadlock the two
    processes against each other, and one stderr thread keeps the tail of
    whatever QNN's native graph compiler printed, for the same reason that
    class keeps one.
    """

    def __init__(
        self,
        npu_python: str,
        onnx_path: str,
        npu_module: str = _DEFAULT_NPU_MODULE,
        ready_timeout_s: float = _NPU_READY_TIMEOUT_S,
    ) -> None:
        """Launch the NPU worker and block until it reports readiness.

        Args:
            npu_python: Interpreter to launch the worker under.
            onnx_path: Path to the exported ONNX graph.
            npu_module: Module to run with ``-m``.  Overridable so a test can
                point this at a fake worker
                (``tests/fake_qnn_npu_worker.py``) with no NPU or ONNX Runtime
                anywhere, mirroring
                ``SubprocessRecognizer``/``WORKER_MODULE``'s own seam.
            ready_timeout_s: How long to wait for the READY frame.

        Raises:
            RuntimeError: If the worker could not be launched, failed before
                reporting readiness, or did not report readiness in time.
        """
        self._lock = threading.Lock()
        self._results: list[bytes] = []
        self._ready = threading.Event()
        self._ready_ok = False
        self._worker_error: str | None = None
        self._stderr_lines: list[str] = []
        self._error: BaseException | None = None

        argv = [
            npu_python,
            "-m",
            npu_module,
            "--onnx-path",
            onnx_path,
        ]
        try:
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(
                f"could not launch the NPU worker {npu_python!r}: {exc}"
            ) from exc

        self._reader = threading.Thread(
            target=self._read_results, name="speakerid-npu-reader", daemon=True
        )
        self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, name="speakerid-npu-stderr", daemon=True
        )
        self._stderr_reader.start()

        if not self._ready.wait(ready_timeout_s):
            self._fail_startup(
                f"the NPU worker did not report readiness within {ready_timeout_s:g} s"
            )
        if not self._ready_ok:
            reported = self._worker_error
            self._fail_startup(
                "the NPU worker reported a failure while creating its session"
                if reported
                else "the NPU worker exited before reporting readiness",
                reported=reported,
            )

    def embed(self, feat: Any, timeout_s: float = _NPU_CALL_TIMEOUT_S) -> bytes:
        """Send one ``(T, 80)`` float32 fbank array and block for its embedding.

        Args:
            feat: A ``(t_max, feat_dim)`` numpy array, already padded/truncated
                to the exported graph's fixed shape.
            timeout_s: How long to wait for the RESULT frame.

        Returns:
            The raw ``(192,)`` float32 embedding bytes, unnormalized.

        Raises:
            RuntimeError: If the worker has died or reports an error.
            TimeoutError: If no result arrives in time.
        """
        if self._process.poll() is not None:
            raise RuntimeError("NPU worker subprocess exited unexpectedly")
        write_frame(self._process.stdin, FrameKind.EMBED, feat.tobytes())

        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._results:
                    return self._results.pop(0)
                error = self._error
                worker_error = self._worker_error
            if error is not None:
                raise RuntimeError(f"NPU worker reader failed: {error}")
            if worker_error is not None:
                raise RuntimeError(f"NPU worker reported an error: {worker_error}")
            if self._process.poll() is not None:
                raise RuntimeError("NPU worker subprocess exited unexpectedly")
            if time.monotonic() > deadline:
                raise TimeoutError("NPU worker did not return a result in time")
            time.sleep(0.005)

    def close(self) -> None:
        """Shut the NPU worker down.  Idempotent, never raises."""
        process = self._process
        try:
            if process.poll() is None:
                write_frame(process.stdin, FrameKind.SHUTDOWN)
        except (OSError, ValueError):
            pass
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                process.terminate()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                except OSError:
                    pass
        for thread in (self._reader, self._stderr_reader):
            if thread is not threading.current_thread():
                thread.join(1.0)

    def _read_results(self) -> None:
        stream = self._process.stdout
        try:
            while True:
                frame = read_frame(stream)
                if frame is None:
                    return
                kind, payload = frame
                if kind is FrameKind.RESULT:
                    with self._lock:
                        self._results.append(payload)
                elif kind is FrameKind.READY:
                    self._ready_ok = True
                    self._ready.set()
                elif kind is FrameKind.ERROR:
                    self._worker_error = payload.decode("utf-8", "replace")
                    self._ready.set()
                    return
        except (ProtocolError, ValueError, OSError) as exc:
            with self._lock:
                if self._error is None:
                    self._error = exc
        finally:
            self._ready.set()

    def _read_stderr(self) -> None:
        stream = self._process.stderr
        try:
            while True:
                line = stream.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                with self._lock:
                    self._stderr_lines.append(text)
                    del self._stderr_lines[:-50]
        except BaseException:  # noqa: BLE001 - diagnostics only; never fatal
            return

    def _fail_startup(self, reason: str, reported: str | None = None) -> None:
        deadline = time.monotonic() + 0.5
        while not self._stderr_lines and time.monotonic() < deadline:
            time.sleep(0.01)
        tail = "\n".join(self._stderr_lines)
        self.close()

        parts = [f"the NPU worker failed to start: {reason}"]
        if reported:
            parts.append(f"  worker reported:\n{reported}")
        if tail:
            parts.append(f"  worker stderr:\n{tail}")
        raise RuntimeError("\n".join(parts))


def run(
    stdin: BinaryIO,
    stdout: BinaryIO,
    onnx_path: str,
    npu_python: str,
    t_max: int,
    npu_module: str = _DEFAULT_NPU_MODULE,
) -> int:
    """Launch the NPU worker, announce readiness, then serve frames until told to stop.

    Args:
        stdin: Binary stream carrying frames from the main process.
        stdout: Binary stream result frames are written to.
        onnx_path: Path to the exported static-shape ONNX graph.
        npu_python: Interpreter to launch the NPU worker under.
        t_max: Fixed fbank-frame length the exported graph accepts.
        npu_module: Module to run the NPU worker with ``-m``; overridable for
            tests, see :class:`NpuWorkerHandle`.

    Returns:
        ``0`` after a clean shutdown, ``1`` if the NPU worker could not be
        started or the stream desynchronised.
    """
    try:
        npu = NpuWorkerHandle(npu_python, onnx_path, npu_module=npu_module)
    except BaseException as exc:  # noqa: BLE001 - reported to the main process as ERROR
        _send_error(stdout, exc)
        return 1

    _LOGGER.info("NPU worker ready: onnx_path=%r t_max=%d", onnx_path, t_max)

    try:
        write_frame(
            stdout, FrameKind.READY, encode_json({"onnx_path": onnx_path, "t_max": t_max})
        )
        return _serve(stdin, stdout, npu, t_max)
    except BaseException as exc:  # noqa: BLE001 - last-ditch report to the main process
        _send_error(stdout, exc)
        return 1
    finally:
        npu.close()


def _serve(stdin: BinaryIO, stdout: BinaryIO, npu: NpuWorkerHandle, t_max: int) -> int:
    """Read EMBED frames of raw mono samples, extract fbank, forward, respond."""
    import numpy as np

    from echochamber.speakerid.campplus import FEAT_DIM, compute_fbank, prepare_wav

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
            try:
                samples = np.frombuffer(payload, dtype=np.float32)
                wav = prepare_wav(samples, 16_000)
                feat = compute_fbank(wav).numpy().astype(np.float32)
                feat = _pad_or_truncate(feat, t_max, FEAT_DIM)
                raw = npu.embed(feat)
            except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the driver
                write_frame(
                    stdout,
                    FrameKind.ERROR,
                    f"{type(exc).__name__}: {exc}".encode("utf-8", "replace"),
                )
                continue
            write_frame(stdout, FrameKind.RESULT, raw)
        elif kind is FrameKind.SHUTDOWN:
            _LOGGER.info("shutdown requested; exiting")
            return 0
        elif kind is FrameKind.RESET:
            pass
        else:
            _LOGGER.warning("ignoring unexpected %s frame from the main process", kind.name)


def _pad_or_truncate(feat: Any, t_max: int, feat_dim: int) -> Any:
    """Pad or truncate ``(T, feat_dim)`` fbank features to exactly ``(t_max, feat_dim)``.

    Padding happens after :func:`~echochamber.speakerid.campplus.compute_fbank`'s
    mean-subtraction, not before -- mixing zeros in before the mean is
    computed would skew every real frame's value for short clips. Ported from
    the sibling ``cam-script`` repository's ``campplus_qnn.py``.
    """
    import numpy as np

    t = feat.shape[0]
    if t >= t_max:
        return feat[:t_max]
    pad = np.zeros((t_max - t, feat_dim), dtype=feat.dtype)
    return np.concatenate([feat, pad], axis=0)


def _send_error(stdout: BinaryIO, exc: BaseException) -> bool:
    """Try to report ``exc`` to the main process as an ERROR frame."""
    message = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).strip()
    if not message:
        message = f"{type(exc).__name__}: {exc}"
    try:
        write_frame(stdout, FrameKind.ERROR, message.encode("utf-8", "replace"))
    except BaseException:  # noqa: BLE001 - the pipe is gone; stderr still has it
        _LOGGER.exception("could not report the failure to the main process")
        return False
    return True


def _configure_logging(level: str | None) -> None:
    if not level:
        return
    resolved = logging.getLevelName(str(level).upper())
    if not isinstance(resolved, int):
        resolved = logging.INFO
    logging.basicConfig(
        stream=sys.stderr,
        level=resolved,
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
    _configure_logging(args.log_level)

    npu_python = args.npu_python or os.environ.get("CAMPPLUS_QNN_PYTHON") or sys.executable

    stdin = set_binary_mode(sys.stdin.buffer)
    stdout = set_binary_mode(sys.stdout.buffer)
    return run(stdin, stdout, args.onnx_path, npu_python, args.t_max, args.npu_module)


if __name__ == "__main__":
    sys.exit(main())
