"""The recogniser subprocess: Vosk on the far side of a pipe.

This module is the *child* half of
:mod:`echochamber.voicegate.subprocess_recognizer`.  It is launched as
``python -m echochamber.voicegate.worker`` **by a different interpreter from
the one running the GUI** -- an x64 one on an ARM64 machine -- which is the
entire reason it exists.  See the sibling module for that story; what matters
here is that this file may only ever import :mod:`vosk` lazily, because the
parent process imports it too (to run ``main`` in tests, and to reason about
the protocol) and the parent is exactly the interpreter where ``import vosk``
fails.

**Stdout carries binary frames, so nothing else may ever touch it.**  Every
diagnostic goes to stderr, including the logging this module configures.  A
stray ``print`` here would be indistinguishable from a corrupt frame at the
other end -- the parent would read the first bytes of the message as a kind
byte and a length prefix, and then either raise a ``ProtocolError`` about an
"unknown frame kind" or block forever waiting for a payload that will never
arrive.

**On Windows, stdin and stdout must be switched to binary mode explicitly.**
This is the single nastiest thing about the design.  The C runtime opens the
standard handles in *text* mode, where every ``\\n`` written becomes ``\\r\\n``
and every ``\\r\\n`` read becomes ``\\n``.  Raw ``int16`` PCM contains ``0x0A``
bytes constantly, so without :func:`set_binary_mode` roughly one frame in a
hundred grows a byte on the way out and shrinks on the way in.  The failure
mode is not a clean error: the length prefixes stop lining up, and the symptom
is a recogniser that works on the developer's Linux box, works for the first
few seconds on Windows, and then reports garbage or hangs.  ``msvcrt.setmode``
on both file descriptors is the fix, and it must happen before a single byte
moves.

**Importing this module must not run anything**, for the same reason
:mod:`echochamber.app` refuses to build a ``QApplication`` at import time: the
test suite imports it in-process to exercise :func:`run` against fake streams,
and an import with side effects would take stdio away from pytest.  Everything
lives inside :func:`main`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from typing import BinaryIO, Sequence

from echochamber.voicegate.protocol import (
    FrameKind,
    ProtocolError,
    encode_json,
    read_frame,
    write_frame,
)
from echochamber.voicegate.recognizer import (
    Recognition,
    Recognizer,
    load_vosk_recognizer,
)

__all__ = ["build_parser", "main", "run", "set_binary_mode"]

_LOGGER = logging.getLogger("echochamber.voicegate.worker")
"""Logger for this module.  Writes to **stderr**; stdout is the frame channel."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the worker.

    Split out from :func:`main` so the argument surface -- which is a
    compatibility contract with
    :class:`~echochamber.voicegate.subprocess_recognizer.SubprocessRecognizer`,
    living in another interpreter -- can be asserted on directly.

    Returns:
        A parser accepting ``--model``, ``--sample-rate``, a repeatable
        ``--phrase`` and ``--log-level``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m echochamber.voicegate.worker",
        description=(
            "Vosk recognition worker. Reads length-prefixed audio frames on "
            "stdin and writes result frames on stdout; not meant to be run by "
            "hand."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="directory of an unpacked Vosk model",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="sample rate of the incoming PCM, in Hz (default: 16000)",
    )
    parser.add_argument(
        "--phrase",
        action="append",
        default=None,
        dest="phrases",
        help=(
            "wake phrase to constrain the decoder to; repeat for several, omit "
            "entirely for open vocabulary"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help=(
            "logging level for the worker's own diagnostics, written to stderr "
            "(e.g. DEBUG, INFO); logging is off when omitted"
        ),
    )
    return parser


def set_binary_mode(stream: BinaryIO) -> BinaryIO:
    """Put ``stream``'s file descriptor into binary mode on Windows.

    A no-op everywhere else: :mod:`msvcrt` does not exist off Windows, and no
    other platform translates line endings on a file descriptor in the first
    place.  See the module docstring for why skipping this corrupts PCM.

    Failures are swallowed rather than raised.  The descriptor may not be a
    real file at all -- a test hands this a :class:`io.BytesIO`, and a service
    harness may hand it a socket -- and in both cases there is nothing to fix
    and nothing to complain about.

    Args:
        stream: The binary stream to switch, normally ``sys.stdin.buffer`` or
            ``sys.stdout.buffer``.

    Returns:
        ``stream`` itself, so this can be used inline where the stream is
        bound.
    """
    try:
        import msvcrt  # noqa: PLC0415 - Windows only; absent elsewhere
    except ImportError:
        return stream
    try:
        msvcrt.setmode(stream.fileno(), os.O_BINARY)
    except (AttributeError, OSError, ValueError):
        # Not backed by a real descriptor (BytesIO, a pipe wrapper, a socket).
        # Nothing to set, and nothing that would benefit from an exception.
        pass
    return stream


def run(
    stdin: BinaryIO,
    stdout: BinaryIO,
    model_path: str,
    sample_rate: int,
    phrases: tuple[str, ...] = (),
) -> int:
    """Load the model, announce readiness, then serve frames until told to stop.

    The whole protocol lives here, against plain binary streams rather than
    ``sys.stdin``/``sys.stdout``, so the loop can be driven from a test with
    two :class:`io.BytesIO` objects and no subprocess at all.

    A failure to load the model is reported as a
    :attr:`~echochamber.voicegate.protocol.FrameKind.ERROR` frame carrying the
    traceback *before* exiting non-zero.  Dying silently would leave the parent
    to time out after ``startup_timeout_s`` and report "the worker did not
    become ready", which says nothing about the missing model directory or the
    ``ImportError`` that actually happened.

    Args:
        stdin: Binary stream carrying frames from the parent.  Must already be
            in binary mode; see :func:`set_binary_mode`.
        stdout: Binary stream frames are written to.  Nothing else may write
            here.
        model_path: Directory of an unpacked Vosk model.
        sample_rate: Sample rate of the PCM the parent will send, in Hz.
        phrases: Wake phrases to constrain the decoder to; empty for open
            vocabulary.

    Returns:
        ``0`` after a clean shutdown -- a
        :attr:`~echochamber.voicegate.protocol.FrameKind.SHUTDOWN` frame or an
        end of stream -- and ``1`` if the model could not be loaded, the stream
        desynchronised, or anything else went wrong.
    """
    try:
        recognizer = load_vosk_recognizer(model_path, sample_rate, phrases)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent as ERROR
        _send_error(stdout, exc)
        return 1

    _LOGGER.info("loaded model %r at %d Hz", model_path, sample_rate)

    try:
        write_frame(
            stdout,
            FrameKind.READY,
            encode_json(
                {
                    "model": model_path,
                    "sample_rate": sample_rate,
                    "phrases": list(phrases),
                }
            ),
        )
        return _serve(stdin, stdout, recognizer)
    except BaseException as exc:  # noqa: BLE001 - last-ditch report to the parent
        # Anything reaching here (a broken pipe, an OOM, a bug in this file)
        # would otherwise surface at the other end as an unexplained EOF.
        _send_error(stdout, exc)
        return 1
    finally:
        recognizer.close()


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

    # Binary mode first, before anything can write a byte.  See the module
    # docstring for what text-mode translation does to PCM.
    stdin = set_binary_mode(sys.stdin.buffer)
    stdout = set_binary_mode(sys.stdout.buffer)

    phrases = tuple(args.phrases or ())
    return run(stdin, stdout, args.model, int(args.sample_rate), phrases)


def _serve(stdin: BinaryIO, stdout: BinaryIO, recognizer: Recognizer) -> int:
    """Read frames until shutdown, end of stream, or a protocol failure.

    Args:
        stdin: Binary stream carrying frames from the parent.
        stdout: Binary stream result frames are written to.
        recognizer: The loaded recogniser.

    Returns:
        ``0`` for a clean end, ``1`` if the stream desynchronised.
    """
    while True:
        try:
            frame = read_frame(stdin)
        except ProtocolError as exc:
            # The stream is desynchronised: there is no way to resync a
            # length-prefixed protocol, so say so and stop rather than read
            # further garbage.
            _send_error(stdout, exc)
            return 1

        if frame is None:
            # Clean EOF: the parent closed our stdin, which is what `close()`
            # does after its SHUTDOWN frame and what process death looks like.
            _LOGGER.info("stdin closed; exiting")
            return 0

        kind, payload = frame
        if kind is FrameKind.AUDIO:
            for result in recognizer.accept_pcm(payload):
                write_frame(stdout, FrameKind.RESULT, _encode_result(result))
        elif kind is FrameKind.RESET:
            recognizer.reset()
        elif kind is FrameKind.SHUTDOWN:
            _LOGGER.info("shutdown requested; exiting")
            return 0
        else:
            # READY, RESULT and ERROR travel the other way.  Ignoring them is
            # kinder than dying: a future parent may send something this
            # worker predates, and dropping an unknown control frame keeps the
            # audio flowing.
            _LOGGER.warning("ignoring unexpected %s frame from the parent", kind.name)


def _encode_result(result: Recognition) -> bytes:
    """Serialise one recognition for a ``RESULT`` frame.

    Args:
        result: What the recogniser produced.

    Returns:
        The compact JSON payload the parent reconstructs a
        :class:`~echochamber.voicegate.recognizer.Recognition` from.
    """
    return encode_json(
        {
            "text": result.text,
            "final": bool(result.final),
            "confidence": float(result.confidence),
        }
    )


def _send_error(stdout: BinaryIO, exc: BaseException) -> bool:
    """Try to report ``exc`` to the parent as an ``ERROR`` frame.

    The traceback is included, not just the message.  The parent surfaces this
    verbatim in
    :class:`~echochamber.voicegate.subprocess_recognizer.RecognizerStartupError`,
    and "model failed to load" without a traceback is a bug report nobody can
    act on.

    Args:
        stdout: Binary stream to write the frame to.
        exc: The failure to report.

    Returns:
        ``True`` if the frame was written, ``False`` if it could not be --
        which normally means the parent is already gone, and is not itself
        worth raising over.
    """
    message = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).strip()
    if not message:
        message = f"{type(exc).__name__}: {exc}"
    try:
        write_frame(stdout, FrameKind.ERROR, message.encode("utf-8", "replace"))
    except BaseException:  # noqa: BLE001 - the pipe is gone; stderr still has it
        _LOGGER.exception("could not report the failure to the parent")
        return False
    return True


def _configure_logging(level: str | None) -> None:
    """Point the root logger at stderr at ``level``, or leave it alone.

    Explicitly ``stream=sys.stderr``: :func:`logging.basicConfig` already
    defaults there, but stdout is the frame channel and relying on a default
    for that is not a risk worth taking.

    Args:
        level: A level name such as ``"DEBUG"``, case-insensitive, or ``None``
            to configure nothing.  An unrecognised name falls back to
            ``INFO`` rather than raising: a bad ``--log-level`` must not stop
            the worker from recognising speech.
    """
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


if __name__ == "__main__":
    sys.exit(main())
