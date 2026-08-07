"""Recognition in a child process, because Vosk has no ARM64 wheel.

**This class exists for a packaging reason, not a design preference.**  The
deployment target is Windows on ARM64.  Vosk publishes no ``win_arm64`` wheel
and its Kaldi core is not something this project is going to cross-compile, so
``pip install vosk`` into the application's own (native ARM64) interpreter
simply fails -- and that is the end of in-process recognition on the target
machine.

The workaround is to move *only the decode* into a second interpreter.  A
separate **x64** ``python.exe`` -- the ordinary "Windows installer (64-bit)"
build from python.org -- is installed alongside the native one, ``vosk`` is
pip-installed into a venv made from it, and
:class:`SubprocessRecognizer` launches
:mod:`echochamber.voicegate.worker` under that interpreter, talking to it over
pipes (see :mod:`echochamber.voicegate.protocol`).  Windows 11's Prism
emulator runs the x64 child transparently; nothing in the parent knows or
cares.

**What must not be emulated is exactly what stays in the parent.**  The
PortAudio/WASAPI capture stream, the real-time audio callback, the ring buffer
and the chunker all run native ARM64, where the callback's deadline is a few
milliseconds and emulation overhead would show up as xruns.  What crosses into
the emulated process is the Kaldi decode, whose latency budget is the wake
phrase's -- hundreds of milliseconds, and already asynchronous.  Paying a
process boundary and an emulator's overhead there costs nothing anyone can
hear.  ``scripts/setup_voice_gate.py`` builds the x64 venv and prints the
``worker_python`` path to configure.

**A reader thread is mandatory here, not a nicety.**  Both directions of the
pipe are finite OS buffers.  If the parent wrote audio and only then read
results, then the moment the worker's stdout buffer filled -- which it does as
soon as results outpace reads -- the worker would block in ``write`` and stop
reading stdin, the parent's stdin buffer would fill, and the parent would block
in ``write`` too.  Two processes, each blocked writing to a full pipe, each
waiting for the other to read: a textbook deadlock, and one that only appears
under load.  So a daemon thread does nothing but drain stdout into a deque, and
:meth:`SubprocessRecognizer.accept_pcm` never reads the pipe at all -- it
writes, then takes whatever the reader has already collected.

**Stderr gets its own drainer for the same reason and one more.**  It is a
third finite buffer that would eventually wedge the worker, and it is where the
useful evidence lives: Vosk writes Kaldi diagnostics there, and a worker that
cannot load its model writes a Python traceback there.  The last
:data:`STDERR_LINES` lines are kept and folded into
:class:`RecognizerStartupError`, because a bare "startup failed" with the
traceback discarded is the least actionable error message this project could
produce.

**Nothing on the audio path raises.**  :meth:`SubprocessRecognizer.accept_pcm`
runs on the pipeline's consumer thread; an exception there kills the consumer
and takes capture down with it.  A dead worker therefore degrades to "no
recognitions ever again", recorded in :attr:`SubprocessRecognizer.error`, which
is the same failure mode as
:class:`~echochamber.voicegate.recognizer.NullRecognizer` -- and vastly
preferable to losing the recording.
"""

from __future__ import annotations

import collections
import subprocess
import threading
import time
from typing import Any, BinaryIO, Callable

from echochamber.voicegate.protocol import (
    FrameKind,
    ProtocolError,
    decode_json,
    read_frame,
    write_frame,
)
from echochamber.voicegate.recognizer import Recognition, parse_word_timings

__all__ = [
    "STDERR_LINES",
    "WORKER_MODULE",
    "RecognizerStartupError",
    "SubprocessRecognizer",
]

WORKER_MODULE: str = "echochamber.voicegate.worker"
"""Module the child interpreter is asked to run with ``-m``.

The child needs this package importable, which it is whenever the x64 venv was
made inside the checkout or the project is installed into it.  Naming a module
rather than a script path means the child resolves it the same way the parent
would, with no path arithmetic across two interpreters.
"""

STDERR_LINES: int = 50
"""How many trailing stderr lines are kept for :attr:`SubprocessRecognizer.stderr_tail`.

Enough for a Python traceback plus the Kaldi banner above it, and bounded so a
worker that logs every frame cannot grow the parent's memory without limit.
"""

_SHUTDOWN_WAIT_S: float = 2.0
"""Seconds :meth:`SubprocessRecognizer.close` waits for a voluntary exit."""

_TERMINATE_WAIT_S: float = 1.0
"""Seconds :meth:`SubprocessRecognizer.close` waits after ``terminate()``."""

_STDERR_SETTLE_S: float = 0.5
"""How long startup failure handling waits for the worker's traceback.

The process dying and its traceback reaching the stderr drainer are two
different events, and reporting the failure before the second one happens is
how a perfectly good traceback gets thrown away.
"""


class RecognizerStartupError(RuntimeError):
    """The worker process never reported readiness.

    Raised by :meth:`SubprocessRecognizer.start` and always carrying whatever
    the worker managed to say on stderr; see the module docstring.
    """


class SubprocessRecognizer:
    """Run a Vosk recogniser in a child interpreter and forward PCM to it.

    Satisfies :class:`~echochamber.voicegate.recognizer.Recognizer`, so the
    gate cannot tell it apart from
    :class:`~echochamber.voicegate.recognizer.VoskRecognizer` beyond the fact
    that results arrive a beat later -- ``accept_pcm`` returns whatever the
    reader thread has collected *by now*, not the result for the buffer just
    written.  Nothing downstream cares: the gate acts on finals, which lag the
    audio containing them regardless of backend.

    Single-use.  A worker that has been closed cannot be restarted; build a new
    instance, which is also what a configuration change wants.

    Threading: two daemon threads (``voicegate-reader``, ``voicegate-stderr``)
    own the child's stdout and stderr for its whole life.  :meth:`accept_pcm`
    and :meth:`reset` write to stdin from the caller's thread -- the pipeline's
    consumer thread in practice -- and a single lock guards only the result and
    stderr deques, never a write, so a slow pipe cannot block the reader.
    """

    __slots__ = (
        "_python_executable",
        "_model_path",
        "_sample_rate",
        "_phrases",
        "_startup_timeout_s",
        "_popen",
        "_process",
        "_lock",
        "_results",
        "_stderr_lines",
        "_reader",
        "_stderr_reader",
        "_ready",
        "_ready_ok",
        "_worker_error",
        "_error",
        "_started",
        "_closed",
    )

    def __init__(
        self,
        python_executable: str,
        model_path: str,
        sample_rate: int,
        phrases: tuple[str, ...] = (),
        startup_timeout_s: float = 30.0,
        popen: Callable[..., Any] | None = None,
    ) -> None:
        """Prepare a worker; nothing is launched until :meth:`start`.

        Args:
            python_executable: Interpreter to run the worker with.  On Windows
                ARM64 this is an **x64** ``python.exe`` with ``vosk``
                installed; see the module docstring.  Not validated here -- a
                bad path surfaces from :meth:`start` as a
                :class:`RecognizerStartupError`, where it can be reported with
                everything else that might have gone wrong at launch.
            model_path: Directory of an unpacked Vosk model, interpreted by the
                *child*, so it must make sense to that interpreter.
            sample_rate: Capture sample rate in Hz; must match the PCM fed to
                :meth:`accept_pcm`.
            phrases: Wake phrases to constrain the decoder to, or empty for
                open vocabulary.
            startup_timeout_s: How long :meth:`start` waits for the worker to
                load its model and report readiness.  Model loading is seconds,
                not milliseconds, and the first run on a cold file cache is far
                slower than every run after it.
            popen: Factory used instead of :func:`subprocess.Popen`.  The
                testing seam, mirroring ``sd_module`` on
                :class:`~echochamber.audio.sources.sounddevice_source.SoundDeviceSource`:
                supply a fake and the whole protocol, both threads and the
                shutdown sequence run with no child process anywhere.

        Raises:
            ValueError: If ``sample_rate`` is not positive or
                ``startup_timeout_s`` is not positive.
        """
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        startup_timeout_s = float(startup_timeout_s)
        if startup_timeout_s <= 0.0:
            raise ValueError(
                f"startup_timeout_s must be > 0, got {startup_timeout_s}"
            )

        self._python_executable: str = str(python_executable)
        self._model_path: str = str(model_path)
        self._sample_rate: int = sample_rate
        self._phrases: tuple[str, ...] = tuple(phrases)
        self._startup_timeout_s: float = startup_timeout_s
        self._popen: Callable[..., Any] = (
            subprocess.Popen if popen is None else popen
        )

        self._process: Any = None
        self._lock: threading.Lock = threading.Lock()
        self._results: collections.deque[Recognition] = collections.deque()
        self._stderr_lines: collections.deque[str] = collections.deque(
            maxlen=STDERR_LINES
        )
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._ready: threading.Event = threading.Event()
        # Set only by a READY frame.  `_ready` is also set when the worker dies
        # or errors, so that start() wakes up; this flag is what distinguishes
        # "ready" from "gave up waiting".
        self._ready_ok: bool = False
        self._worker_error: str | None = None
        self._error: BaseException | None = None
        self._started: bool = False
        self._closed: bool = False

    @property
    def python_executable(self) -> str:
        """Interpreter the worker runs under."""
        return self._python_executable

    @property
    def model_path(self) -> str:
        """Model directory passed to the worker."""
        return self._model_path

    @property
    def sample_rate(self) -> int:
        """Sample rate the worker was configured with, in Hz."""
        return self._sample_rate

    @property
    def phrases(self) -> tuple[str, ...]:
        """Wake phrases the decoder is constrained to; empty for open vocabulary."""
        return self._phrases

    @property
    def command(self) -> tuple[str, ...]:
        """The exact argument vector :meth:`start` launches.

        Exposed because "what did it actually run?" is the first question when
        a worker will not start, and reconstructing it from the configuration
        by hand gets the ``--phrase`` repetition wrong.
        """
        argv: list[str] = [
            self._python_executable,
            "-m",
            WORKER_MODULE,
            "--model",
            self._model_path,
            "--sample-rate",
            str(self._sample_rate),
        ]
        for phrase in self._phrases:
            argv.extend(("--phrase", phrase))
        return tuple(argv)

    @property
    def running(self) -> bool:
        """``True`` while the child process is alive."""
        process = self._process
        if process is None:
            return False
        try:
            return process.poll() is None
        except BaseException:  # noqa: BLE001 - a fake or a reaped process
            return False

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has run."""
        return self._closed

    @property
    def error(self) -> BaseException | None:
        """First failure recorded by a background thread or a failed write.

        The reader threads never re-raise -- they are daemons, and re-raising
        would only scribble on the parent's stderr -- and neither does
        :meth:`accept_pcm`, so this is the only way to learn that the worker
        has stopped producing results.
        """
        return self._error

    @property
    def stderr_tail(self) -> str:
        """The last :data:`STDERR_LINES` lines the worker wrote to stderr.

        Returns:
            The lines joined by newlines, or an empty string if the worker has
            said nothing.
        """
        with self._lock:
            return "\n".join(self._stderr_lines)

    @property
    def pending(self) -> int:
        """Results the reader thread has collected but nobody has taken yet."""
        with self._lock:
            return len(self._results)

    def start(self) -> None:
        """Launch the worker and block until it reports readiness.

        Blocking is the point: model loading takes seconds, and audio sent
        before the decoder exists is audio nobody will ever recognise.  The
        wait is bounded by ``startup_timeout_s``, and every way of failing --
        the interpreter not existing, the model not loading, the process dying,
        the worker simply never answering -- ends as a
        :class:`RecognizerStartupError` carrying :attr:`stderr_tail`.

        Raises:
            RuntimeError: If called twice, or after :meth:`close`.
            RecognizerStartupError: If the worker could not be launched, failed
                before reporting readiness, or did not report readiness within
                ``startup_timeout_s``.  The worker is cleaned up first, so a
                failed start leaves no orphan process behind.
        """
        if self._started:
            raise RuntimeError(
                "SubprocessRecognizer has already been started and cannot be "
                "restarted; create a new SubprocessRecognizer"
            )
        self._started = True

        argv = list(self.command)
        try:
            process = self._popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Unbuffered: `write_frame` flushes, but an intermediate
                # buffer between us and the pipe is one more place a frame can
                # sit while both ends wait for each other.
                bufsize=0,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised as a startup error
            self._error = exc
            self._closed = True
            raise RecognizerStartupError(
                f"could not launch the recogniser worker "
                f"{self._python_executable!r}: {exc}"
            ) from exc

        self._process = process
        self._reader = self._spawn(self._read_results, "voicegate-reader")
        self._stderr_reader = self._spawn(self._read_stderr, "voicegate-stderr")

        if not self._ready.wait(self._startup_timeout_s):
            self._fail_startup(
                f"the worker did not report readiness within "
                f"{self._startup_timeout_s:g} s"
            )
        if not self._ready_ok:
            reported = self._worker_error
            self._fail_startup(
                "the worker reported a failure while loading the model"
                if reported
                else "the worker exited before reporting readiness",
                reported=reported,
            )

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        """Send ``pcm`` to the worker and return whatever has come back.

        Never blocks on a result and **never raises**: this runs on the
        pipeline's consumer thread, where an exception kills the consumer and
        stops capture.  A dead or closed worker quietly consumes the audio and
        returns nothing, with the reason left in :attr:`error`.

        Args:
            pcm: 16-bit little-endian mono bytes at :attr:`sample_rate`.

        Returns:
            The recognitions the reader thread has collected since the last
            call -- usually the ones for slightly earlier audio, and routinely
            an empty list.
        """
        if pcm:
            self._send(FrameKind.AUDIO, pcm)
        with self._lock:
            if not self._results:
                return []
            results = list(self._results)
            self._results.clear()
        return results

    def reset(self) -> None:
        """Ask the worker to discard decoder state.  Never raises."""
        self._send(FrameKind.RESET)

    def close(self) -> None:
        """Shut the worker down, escalating until it is actually gone.

        SHUTDOWN frame, then close stdin (which the worker sees as a clean
        EOF), then wait, then ``terminate()``, then ``kill()``.  Each step is
        bounded, so a wedged worker -- one stuck inside Kaldi, say -- delays
        shutdown by a few seconds rather than hanging the application.

        Idempotent and never raises, because this is called from the sink's own
        ``close``, which the pipeline calls during a shutdown that must not be
        derailed by a child process behaving badly.
        """
        if self._closed:
            return
        process = self._process
        if process is not None:
            self._send(FrameKind.SHUTDOWN)
        self._closed = True
        if process is None:
            return

        try:
            self._close_stream(getattr(process, "stdin", None))
            if not self._wait(process, _SHUTDOWN_WAIT_S):
                self._signal(process, "terminate")
                if not self._wait(process, _TERMINATE_WAIT_S):
                    self._signal(process, "kill")
                    self._wait(process, _TERMINATE_WAIT_S)
            # Only now are the pipes certain to be at EOF, so the drainers can
            # be joined without waiting out a blocking read.
            for thread in (self._reader, self._stderr_reader):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(_TERMINATE_WAIT_S)
            self._close_stream(getattr(process, "stdout", None))
            self._close_stream(getattr(process, "stderr", None))
        except BaseException as exc:  # noqa: BLE001 - close must not raise
            if self._error is None:
                self._error = exc

    def _read_results(self) -> None:
        """Reader thread body: drain stdout into the result deque until EOF."""
        stream: BinaryIO | None = getattr(self._process, "stdout", None)
        try:
            if stream is None:
                return
            while True:
                frame = read_frame(stream)
                if frame is None:
                    return
                kind, payload = frame
                if kind is FrameKind.RESULT:
                    recognition = _decode_result(payload)
                    with self._lock:
                        self._results.append(recognition)
                elif kind is FrameKind.READY:
                    self._ready_ok = True
                    self._ready.set()
                elif kind is FrameKind.ERROR:
                    self._worker_error = payload.decode("utf-8", "replace")
                    # Unblock a start() that is still waiting; there is nothing
                    # left to wait for once the worker has reported a failure.
                    self._ready.set()
                    return
                # AUDIO, RESET and SHUTDOWN travel the other way; a worker
                # sending one is confused, but dropping it is harmless.
        except (ProtocolError, ValueError, OSError) as exc:
            # A desynchronised stream or a pipe torn down under us.  Recorded
            # rather than raised: this is a daemon thread, and `error` is how
            # the owner finds out.
            if self._error is None:
                self._error = exc
        except BaseException as exc:  # noqa: BLE001 - reported via `error`
            if self._error is None:
                self._error = exc
        finally:
            # Whatever happened, no READY is coming now.  Wake start() so it
            # fails on `_ready_ok` instead of waiting out the whole timeout.
            self._ready.set()

    def _read_stderr(self) -> None:
        """Stderr thread body: keep the tail of what the worker complains about.

        Draining is not optional even when nobody reads :attr:`stderr_tail`: a
        full stderr pipe blocks the worker inside its own ``write`` just as
        surely as a full stdout pipe does.
        """
        stream: BinaryIO | None = getattr(self._process, "stderr", None)
        try:
            if stream is None:
                return
            while True:
                line = stream.readline()
                if not line:
                    return
                text = line.decode("utf-8", "replace").rstrip("\r\n")
                with self._lock:
                    self._stderr_lines.append(text)
        except BaseException:  # noqa: BLE001 - diagnostics only; never fatal
            # Losing stderr costs us a nicer error message, nothing more, and
            # it must not overwrite a real failure recorded in `error`.
            return

    def _send(self, kind: FrameKind, payload: bytes = b"") -> bool:
        """Write one frame to the worker, degrading to a no-op on failure.

        Args:
            kind: Frame kind to send.
            payload: Frame body.

        Returns:
            ``True`` if the frame was written, ``False`` if there was nowhere
            to write it -- no process, already closed, or a broken pipe
            (:class:`BrokenPipeError` is an :class:`OSError`) because the
            worker has died.  A failure is recorded in :attr:`error`.
        """
        process = self._process
        if process is None or self._closed:
            return False
        stream: BinaryIO | None = getattr(process, "stdin", None)
        if stream is None:
            return False
        try:
            write_frame(stream, kind, payload)
        except (OSError, ValueError, ProtocolError) as exc:
            # ValueError: writing to an already-closed stream. OSError covers
            # BrokenPipeError, which is simply what a dead worker looks like.
            if self._error is None:
                self._error = exc
            return False
        return True

    def _fail_startup(self, reason: str, reported: str | None = None) -> None:
        """Tear the worker down and raise a :class:`RecognizerStartupError`.

        Args:
            reason: What went wrong, in one line.
            reported: The body of the worker's ``ERROR`` frame, if it sent one.
                Kept out of ``reason`` because it is a whole traceback, and a
                traceback spliced into the middle of a sentence is unreadable.

        Raises:
            RecognizerStartupError: Always, quoting the command, the worker's
                own report and :attr:`stderr_tail`.
        """
        # Give the stderr drainer a moment: the process can be gone before its
        # traceback has been read, and the traceback is the whole point.
        deadline = time.monotonic() + _STDERR_SETTLE_S
        while not self.stderr_tail and time.monotonic() < deadline:
            time.sleep(0.01)
        tail = self.stderr_tail

        self.close()

        parts = [
            f"the recogniser worker failed to start: {reason}",
            f"  command: {' '.join(self.command)}",
        ]
        if reported:
            parts.append(f"  worker reported:\n{_indent(reported)}")
        if tail:
            parts.append(f"  worker stderr:\n{_indent(tail)}")
        else:
            parts.append("  worker stderr: (nothing)")
        raise RecognizerStartupError("\n".join(parts))

    def _spawn(self, target: Callable[[], None], name: str) -> threading.Thread:
        """Start one daemon thread.

        Args:
            target: Thread body.
            name: Thread name, which is what shows up in a stack dump.

        Returns:
            The started thread.
        """
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def _wait(self, process: Any, timeout: float) -> bool:
        """Wait up to ``timeout`` for ``process`` to exit.

        Args:
            process: The child process handle.
            timeout: Seconds to wait.

        Returns:
            ``True`` if the process has exited, ``False`` on timeout or if the
            handle does not support waiting (a fake, in a test).
        """
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        except BaseException:  # noqa: BLE001 - a fake process, or already reaped
            return True
        return True

    def _signal(self, process: Any, method: str) -> None:
        """Call ``terminate`` or ``kill`` on ``process``, tolerating absence.

        Args:
            process: The child process handle.
            method: ``"terminate"`` or ``"kill"``.
        """
        action = getattr(process, method, None)
        if action is None:
            return
        try:
            action()
        except BaseException:  # noqa: BLE001 - already dead, or a fake process
            pass

    def _close_stream(self, stream: BinaryIO | None) -> None:
        """Close one of the child's pipes, tolerating a broken one.

        Args:
            stream: The stream to close, or ``None``.
        """
        if stream is None:
            return
        try:
            stream.close()
        except BaseException:  # noqa: BLE001 - closing a dead pipe is not news
            pass

    def __repr__(self) -> str:
        """Return a debugging representation of the worker's state."""
        return (
            f"{type(self).__name__}(python={self._python_executable!r}, "
            f"model={self._model_path!r}, sample_rate={self._sample_rate}, "
            f"phrases={len(self._phrases)}, running={self.running}, "
            f"pending={self.pending}, closed={self._closed})"
        )


def _decode_result(payload: bytes) -> Recognition:
    """Rebuild a :class:`Recognition` from a ``RESULT`` frame.

    Every field is coerced rather than trusted.  The payload crossed a process
    boundary from an interpreter this one cannot import, so a version skew
    between parent and worker is a real possibility, and a malformed result
    must degrade to "nothing recognised" rather than kill the reader thread.

    Args:
        payload: The frame body.

    Returns:
        The decoded recognition; empty text when the payload made no sense.

    Raises:
        ProtocolError: If the payload is not a JSON object at all, which means
            the stream itself is suspect rather than one result.
    """
    parsed = decode_json(payload)

    text = parsed.get("text", "")
    if not isinstance(text, str):
        text = ""

    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        confidence = 0.0

    return Recognition(
        text=text,
        final=bool(parsed.get("final", False)),
        confidence=float(confidence),
        # Reuses the same coercion the in-process backend applies to Vosk's own
        # JSON, so a malformed timing degrades to "no timings" -- and therefore
        # to a wider clip -- rather than to a confidently wrong one.
        words=parse_word_timings(parsed.get("words")),
    )


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent every line of ``text``.

    Args:
        text: The block to indent.
        prefix: What to put in front of each line.

    Returns:
        The indented block, so a captured traceback reads as one nested item
        inside the error message rather than as a second error.
    """
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())
