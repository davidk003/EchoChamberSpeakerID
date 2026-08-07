"""Tests for echochamber.voicegate.subprocess_recognizer, with no child process.

``SubprocessRecognizer`` takes a ``popen=`` factory for exactly the reason
``SoundDeviceSource`` takes ``sd_module``: the thing it drives -- here an x64
interpreter with Vosk installed, on a machine that may not have one -- is not
available where the tests run.  :class:`FakeProcess` below is the counterpart of
``tests/fake_sounddevice.py``: it exposes the three pipes and the four lifecycle
methods the class actually touches (``poll``, ``wait``, ``terminate``,
``kill``), counts them, and lets a test pre-load what the "worker" will say.

**The reader thread is what makes this delicate, so the fakes are built around
it.**  ``start()`` spawns a daemon thread that loops on ``read_frame(stdout)``,
and blocks on a ``threading.Event`` until a READY frame arrives.  Two
consequences shape every test here:

* stdout is pre-loaded with its whole conversation *before* ``start()`` is
  called, READY first.  A test that started the recogniser and then wrote to
  stdout would be racing its own reader thread.
* stdout is an :class:`io.BytesIO`, so it hits EOF the moment the pre-loaded
  frames run out and the reader thread exits.  That is deliberate: nothing here
  can hang, because no fake stream ever blocks indefinitely.  The one test that
  *needs* a stream that does not end -- the readiness timeout -- uses
  :class:`SlowStream`, which sleeps for a bounded 0.3 s and then ends anyway.

Results arrive on that thread, so the tests that observe them poll with
:func:`wait_until` rather than sleeping a guessed interval.
"""

from __future__ import annotations

import io
import re
import subprocess
import threading
import time
from typing import Any, Callable, Iterator

import pytest

from echochamber.voicegate.protocol import (
    FrameKind,
    encode_frame,
    encode_json,
    read_frame,
)
from echochamber.voicegate.recognizer import Recognition, Recognizer
from echochamber.voicegate.subprocess_recognizer import (
    STDERR_LINES,
    WORKER_MODULE,
    RecognizerStartupError,
    SubprocessRecognizer,
    _decode_result,
)


PYTHON = "/opt/x64-venv/Scripts/python.exe"
MODEL = "/models/vosk-model-small-en-us-0.15"
SR = 16_000

# Generous: every wait below is event-driven, so these only bound failures.
TIMEOUT = 5.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Poll ``pred`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def ready_frame(model: str = MODEL, sample_rate: int = SR) -> bytes:
    """The READY frame the worker sends once its model is loaded."""
    return encode_frame(
        FrameKind.READY,
        encode_json({"model": model, "sample_rate": sample_rate, "phrases": []}),
    )


def result_frame(text: str, final: bool = True, confidence: float = 0.0) -> bytes:
    """A RESULT frame, encoded exactly as the worker's _encode_result does."""
    return encode_frame(
        FrameKind.RESULT,
        encode_json({"text": text, "final": final, "confidence": confidence}),
    )


def error_frame(message: str) -> bytes:
    """An ERROR frame carrying a worker-side traceback."""
    return encode_frame(FrameKind.ERROR, message.encode("utf-8"))


def frames_in(blob: bytes) -> list[tuple[FrameKind, bytes]]:
    """Decode a whole byte blob into frames, for inspecting what was written."""
    stream = io.BytesIO(blob)
    out: list[tuple[FrameKind, bytes]] = []
    while True:
        frame = read_frame(stream)
        if frame is None:
            return out
        out.append(frame)


# --------------------------------------------------------------------------
# the fake child process
# --------------------------------------------------------------------------

class FakeStdin:
    """The child's stdin: keeps every byte written and survives close().

    A plain :class:`io.BytesIO` would work until ``close()`` ran, after which
    ``getvalue()`` raises -- and inspecting what was sent *after* shutdown is
    exactly what the close tests do.  ``raise_on_write`` turns it into a broken
    pipe on demand, which is what a dead worker looks like from this side.
    """

    def __init__(self, raise_on_write: BaseException | None = None) -> None:
        self.written = bytearray()
        self.flushes = 0
        self.closed = False
        self.raise_on_write = raise_on_write

    def write(self, data: bytes) -> int:
        if self.closed:
            raise ValueError("write to closed file")
        if self.raise_on_write is not None:
            raise self.raise_on_write
        self.written += data
        return len(data)

    def flush(self) -> None:
        if self.closed:
            raise ValueError("flush of closed file")
        self.flushes += 1

    def close(self) -> None:
        self.closed = True

    @property
    def frames(self) -> list[tuple[FrameKind, bytes]]:
        """Everything the parent sent, decoded."""
        return frames_in(bytes(self.written))

    @property
    def kinds(self) -> list[FrameKind]:
        """Just the kinds of everything the parent sent."""
        return [kind for kind, _ in self.frames]


class SlowStream:
    """A stdout that answers slowly, then ends -- never forever.

    Used only by the readiness-timeout test.  Blocking indefinitely would leave
    a wedged daemon thread behind on every run, so this sleeps a bounded
    ``delay`` and then reports EOF, which lets ``close()`` join its reader.
    """

    def __init__(self, delay: float = 0.3) -> None:
        self.delay = delay
        self.read_calls = 0
        self.closed = False

    def read(self, n: int = -1) -> bytes:
        self.read_calls += 1
        time.sleep(self.delay)
        return b""

    def close(self) -> None:
        self.closed = True


class FakeProcess:
    """Stand-in for ``subprocess.Popen``: three pipes plus lifecycle counters.

    Only the surface ``SubprocessRecognizer`` actually touches is implemented:
    ``stdin``/``stdout``/``stderr``, ``poll``, ``wait``, ``terminate``,
    ``kill``.  ``wait_raises`` makes ``wait`` time out the way a wedged worker
    does, which is what drives the terminate/kill escalation.
    """

    def __init__(
        self,
        argv: list[str],
        stdout_bytes: bytes = b"",
        stderr_bytes: bytes = b"",
        stdin: FakeStdin | None = None,
        stdout: Any = None,
        returncode: int | None = None,
        wait_raises: BaseException | None = None,
        popen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.argv = list(argv)
        self.popen_kwargs = dict(popen_kwargs or {})
        self.stdin: FakeStdin = FakeStdin() if stdin is None else stdin
        self.stdout: Any = io.BytesIO(stdout_bytes) if stdout is None else stdout
        self.stderr = io.BytesIO(stderr_bytes)
        self._returncode = returncode
        self.wait_raises = wait_raises
        self.poll_calls = 0
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        self.poll_calls += 1
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.wait_raises is not None:
            raise self.wait_raises
        self._returncode = 0
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = -9

    def __repr__(self) -> str:
        return f"FakeProcess(argv={self.argv!r}, returncode={self._returncode!r})"


class FakePopen:
    """The ``popen=`` factory: records how it was called, hands back a FakeProcess."""

    def __init__(
        self,
        raises: BaseException | None = None,
        **process_kwargs: Any,
    ) -> None:
        self.raises = raises
        self.process_kwargs = process_kwargs
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.process: FakeProcess | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        # The stdin/stdout/stderr kwargs are subprocess.PIPE markers, recorded
        # for inspection but never forwarded: the fake's pipes are pre-loaded.
        self.calls.append((list(argv), dict(kwargs)))
        if self.raises is not None:
            raise self.raises
        self.process = FakeProcess(argv, **self.process_kwargs, popen_kwargs=kwargs)
        return self.process


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def make_recognizer() -> Iterator[Callable[..., SubprocessRecognizer]]:
    """Factory registering every recogniser for guaranteed close() on teardown.

    close() is what joins the two daemon threads, so a test that fails partway
    cannot leave a reader thread wedged on a fake stream.
    """
    created: list[SubprocessRecognizer] = []

    def _make(
        popen: Any,
        python_executable: str = PYTHON,
        model_path: str = MODEL,
        sample_rate: int = SR,
        phrases: tuple[str, ...] = (),
        startup_timeout_s: float = TIMEOUT,
    ) -> SubprocessRecognizer:
        rec = SubprocessRecognizer(
            python_executable,
            model_path,
            sample_rate,
            phrases,
            startup_timeout_s,
            popen=popen,
        )
        created.append(rec)
        return rec

    yield _make

    for rec in created:
        try:
            rec.close()
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


@pytest.fixture
def started(
    make_recognizer: Callable[..., SubprocessRecognizer]
) -> Callable[..., tuple[SubprocessRecognizer, FakePopen]]:
    """Build and start a recogniser whose worker says READY, then whatever follows."""

    def _start(
        *extra_frames: bytes,
        stderr_bytes: bytes = b"",
        **kwargs: Any,
    ) -> tuple[SubprocessRecognizer, FakePopen]:
        popen = FakePopen(
            stdout_bytes=ready_frame() + b"".join(extra_frames),
            stderr_bytes=stderr_bytes,
        )
        rec = make_recognizer(popen, **kwargs)
        rec.start()
        return rec, popen

    return _start


# ==========================================================================
# construction and the argument vector
# ==========================================================================

class TestConstruction:
    """Nothing is launched until start(), but the arguments are checked now."""

    def test_stores_its_configuration(self) -> None:
        """Every constructor argument is readable back off the instance."""
        rec = SubprocessRecognizer(PYTHON, MODEL, 48_000, ("ok chamber",))
        assert rec.python_executable == PYTHON
        assert rec.model_path == MODEL
        assert rec.sample_rate == 48_000
        assert rec.phrases == ("ok chamber",)

    def test_phrases_are_stored_as_a_tuple(self) -> None:
        """A list would let a caller mutate the decoder's grammar after the fact."""
        rec = SubprocessRecognizer(PYTHON, MODEL, SR, ["a", "b"])  # type: ignore[arg-type]
        assert rec.phrases == ("a", "b")
        assert isinstance(rec.phrases, tuple)

    def test_nothing_is_launched_by_the_constructor(self) -> None:
        """Construction is cheap and total; start() is where a process appears."""
        popen = FakePopen()
        rec = SubprocessRecognizer(PYTHON, MODEL, SR, popen=popen)
        assert popen.calls == []
        assert rec.running is False
        assert rec.closed is False
        assert rec.error is None
        assert rec.pending == 0

    @pytest.mark.parametrize("sample_rate", [0, -1, -16_000])
    def test_non_positive_sample_rate_raises(self, sample_rate: int) -> None:
        """A zero rate would divide by zero downstream; reject it at the door."""
        with pytest.raises(ValueError, match=r"sample_rate must be > 0"):
            SubprocessRecognizer(PYTHON, MODEL, sample_rate)

    @pytest.mark.parametrize("timeout", [0.0, -1.0, -0.001])
    def test_non_positive_startup_timeout_raises(self, timeout: float) -> None:
        """A zero timeout means start() can never succeed, which is a config bug."""
        with pytest.raises(ValueError, match=r"startup_timeout_s must be > 0"):
            SubprocessRecognizer(PYTHON, MODEL, SR, startup_timeout_s=timeout)

    def test_sample_rate_is_coerced_to_int(self) -> None:
        """A float rate from a config file must not reach the worker as '16000.0'."""
        rec = SubprocessRecognizer(PYTHON, MODEL, 16_000.0)  # type: ignore[arg-type]
        assert rec.sample_rate == 16_000
        assert isinstance(rec.sample_rate, int)

    def test_satisfies_the_recognizer_protocol(self) -> None:
        """The gate must not be able to tell this apart from VoskRecognizer."""
        assert isinstance(SubprocessRecognizer(PYTHON, MODEL, SR), Recognizer)

    def test_repr_summarises_the_state(self) -> None:
        """The repr is what shows up in a log when a worker misbehaves."""
        rec = SubprocessRecognizer(PYTHON, MODEL, SR, ("a", "b"))
        text = repr(rec)
        assert "SubprocessRecognizer(" in text
        assert "phrases=2" in text and "running=False" in text and "closed=False" in text


class TestCommand:
    """The argv is a contract with worker.build_parser, in another interpreter."""

    def test_builds_the_documented_argument_vector(self) -> None:
        """Interpreter, -m WORKER_MODULE, --model, --sample-rate, in that order."""
        rec = SubprocessRecognizer(PYTHON, MODEL, SR)
        assert rec.command == (
            PYTHON, "-m", WORKER_MODULE, "--model", MODEL, "--sample-rate", "16000"
        )

    def test_repeats_phrase_once_per_phrase(self) -> None:
        """--phrase is append-style; one flag per phrase, values kept in order."""
        rec = SubprocessRecognizer(
            PYTHON, MODEL, SR, ("ok chamber", "hey chamber", "start recording")
        )
        assert rec.command == (
            PYTHON, "-m", WORKER_MODULE,
            "--model", MODEL,
            "--sample-rate", "16000",
            "--phrase", "ok chamber",
            "--phrase", "hey chamber",
            "--phrase", "start recording",
        )

    def test_no_phrases_means_no_phrase_flag(self) -> None:
        """Open vocabulary is the absence of --phrase, not an empty value."""
        assert "--phrase" not in SubprocessRecognizer(PYTHON, MODEL, SR).command

    def test_sample_rate_is_stringified(self) -> None:
        """argv is strings; an int would blow up inside Popen."""
        rec = SubprocessRecognizer(PYTHON, MODEL, 48_000)
        assert "48000" in rec.command
        assert all(isinstance(arg, str) for arg in rec.command)

    def test_command_is_an_immutable_tuple(self) -> None:
        """Callers log it; they must not be able to edit what will be run."""
        assert isinstance(SubprocessRecognizer(PYTHON, MODEL, SR).command, tuple)

    def test_worker_module_is_the_dotted_module_name(self) -> None:
        """`-m` resolves the module the same way in both interpreters."""
        assert WORKER_MODULE == "echochamber.voicegate.worker"

    def test_start_launches_exactly_the_command_property(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """`command` must be what was actually run, or it is a lie in a bug report."""
        popen = FakePopen(stdout_bytes=ready_frame())
        rec = make_recognizer(popen, phrases=("ok chamber",))
        rec.start()

        argv, kwargs = popen.calls[0]
        assert argv == list(rec.command)
        assert kwargs["stdin"] is subprocess.PIPE
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
        assert kwargs["bufsize"] == 0, (
            "an intermediate buffer is one more place a frame can sit while both "
            "ends wait for each other"
        )


# ==========================================================================
# start()
# ==========================================================================

class TestStart:
    """Every way of failing to start ends as a RecognizerStartupError."""

    def test_succeeds_when_the_worker_sends_ready(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """READY is the handshake; receiving it is the whole success condition."""
        popen = FakePopen(stdout_bytes=ready_frame())
        rec = make_recognizer(popen)

        rec.start()

        assert len(popen.calls) == 1, "exactly one process must be launched"
        assert rec.error is None
        assert rec.closed is False

    def test_running_reflects_poll(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """`running` is poll() is None -- nothing more, so a fake drives it."""
        popen = FakePopen(stdout_bytes=ready_frame())
        rec = make_recognizer(popen)
        rec.start()

        assert rec.running is True, "poll() returning None means still alive"
        assert popen.process is not None
        popen.process._returncode = 0
        assert rec.running is False, "a returncode means the child has exited"

    def test_running_is_false_before_start(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """There is no process yet, so there is nothing to poll."""
        assert make_recognizer(FakePopen()).running is False

    def test_starting_twice_raises(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """Single-use: a second worker over the same object would leak the first."""
        rec = make_recognizer(FakePopen(stdout_bytes=ready_frame()))
        rec.start()

        with pytest.raises(RuntimeError, match="already been started"):
            rec.start()

    def test_starting_twice_does_not_launch_a_second_process(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """The guard must run before Popen, not after."""
        popen = FakePopen(stdout_bytes=ready_frame())
        rec = make_recognizer(popen)
        rec.start()
        with pytest.raises(RuntimeError):
            rec.start()
        assert len(popen.calls) == 1

    def test_a_popen_failure_becomes_a_startup_error_naming_the_interpreter(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """A bad `worker_python` is the most likely misconfiguration there is."""
        boom = FileNotFoundError(2, "No such file or directory", PYTHON)
        rec = make_recognizer(FakePopen(raises=boom))

        with pytest.raises(RecognizerStartupError, match=re.escape(PYTHON)):
            rec.start()

        assert rec.error is boom, "the original exception must be kept for inspection"
        assert rec.closed is True, "a failed launch must not leave the object half-open"

    def test_a_popen_failure_chains_the_original_exception(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """`raise ... from exc` keeps the OSError visible in the traceback."""
        boom = OSError("exec format error")
        rec = make_recognizer(FakePopen(raises=boom))

        with pytest.raises(RecognizerStartupError) as excinfo:
            rec.start()
        assert excinfo.value.__cause__ is boom
        assert "could not launch the recogniser worker" in str(excinfo.value)

    def test_an_error_frame_instead_of_ready_reports_stderr_and_the_command(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """The headline diagnostic: the worker's own traceback must reach the user.

        A bare 'startup failed' with the traceback discarded is the least
        actionable error message this project could produce.
        """
        traceback_text = (
            "Traceback (most recent call last):\n"
            '  File "worker.py", line 190, in run\n'
            "FileNotFoundError: Vosk model directory not found: '/models/gone'"
        )
        stderr_text = b"LOG (VoskAPI:Model()): no such directory\nfatal: giving up\n"
        popen = FakePopen(
            stdout_bytes=error_frame(traceback_text), stderr_bytes=stderr_text
        )
        rec = make_recognizer(popen, phrases=("ok chamber",))

        with pytest.raises(RecognizerStartupError) as excinfo:
            rec.start()

        message = str(excinfo.value)
        assert "the worker reported a failure while loading the model" in message
        assert "FileNotFoundError: Vosk model directory not found" in message, (
            "the worker's ERROR frame must be quoted verbatim"
        )
        assert "LOG (VoskAPI:Model()): no such directory" in message, (
            "the worker's stderr tail must be included; it is where Kaldi complains"
        )
        assert "fatal: giving up" in message
        assert " ".join(rec.command) in message, (
            "'what did it actually run?' is the first question when a worker "
            "will not start"
        )

    def test_a_worker_that_exits_without_ready_is_reported_as_such(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """A silent death is distinguished from an explicit ERROR frame."""
        popen = FakePopen(stdout_bytes=b"", stderr_bytes=b"segfault\n")

        with pytest.raises(RecognizerStartupError) as excinfo:
            make_recognizer(popen).start()

        message = str(excinfo.value)
        assert "exited before reporting readiness" in message
        assert "segfault" in message

    def test_a_startup_failure_with_no_stderr_says_so(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """'(nothing)' is more honest than an empty section."""
        with pytest.raises(RecognizerStartupError, match=re.escape("(nothing)")):
            make_recognizer(FakePopen(stdout_bytes=b"")).start()

    def test_a_desynchronised_stdout_is_a_startup_failure(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """Garbage on stdout kills the reader thread, which must still wake start()."""
        popen = FakePopen(
            stdout_bytes=b"\x99\x00\x00\x00\x00garbage", stderr_bytes=b"confused\n"
        )

        with pytest.raises(RecognizerStartupError, match="readiness"):
            make_recognizer(popen).start()

    def test_timeout_when_readiness_never_arrives(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """A worker wedged inside Kaldi must not hang the application forever.

        SlowStream never yields a frame, so the only thing that ends the wait is
        ``startup_timeout_s``.  It is kept small here on purpose; the message,
        not the duration, is what is under test.
        """
        popen = FakePopen(stdout_bytes=b"", stdout=SlowStream(0.3), stderr_bytes=b"...\n")
        rec = make_recognizer(popen, startup_timeout_s=0.2)

        t0 = time.monotonic()
        with pytest.raises(RecognizerStartupError) as excinfo:
            rec.start()
        elapsed = time.monotonic() - t0

        assert "did not report readiness within 0.2 s" in str(excinfo.value), (
            f"the message must quote the timeout that expired: {excinfo.value}"
        )
        assert elapsed >= 0.2, "start() must actually wait out the timeout"
        assert elapsed < TIMEOUT, f"the timeout was not honoured, waited {elapsed:.2f}s"

    def test_a_failed_start_closes_the_worker(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """A failed start must leave no orphan process behind."""
        popen = FakePopen(stdout_bytes=error_frame("nope"), stderr_bytes=b"nope\n")
        rec = make_recognizer(popen)

        with pytest.raises(RecognizerStartupError):
            rec.start()

        assert rec.closed is True
        assert popen.process is not None
        assert popen.process.stdin.closed, "the child's stdin must be closed"

    def test_accept_pcm_before_start_is_silently_ignored(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """There is nowhere to write yet, and the audio path never raises."""
        rec = make_recognizer(FakePopen())
        assert rec.accept_pcm(b"\x00\x01") == []
        assert rec.error is None


# ==========================================================================
# accept_pcm
# ==========================================================================

class TestAcceptPcm:
    """Writes audio, takes whatever the reader thread already collected."""

    def test_writes_a_well_formed_audio_frame(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """The PCM reaches stdin as one AUDIO frame with the bytes unchanged."""
        rec, popen = started()
        pcm = bytes(range(256)) * 4

        rec.accept_pcm(pcm)

        assert popen.process is not None
        assert popen.process.stdin.frames == [(FrameKind.AUDIO, pcm)]

    def test_each_call_writes_one_frame(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Buffers are not coalesced; the worker sees the same chunking we do."""
        rec, popen = started()
        rec.accept_pcm(b"\x01\x02")
        rec.accept_pcm(b"\x03\x04")

        assert popen.process is not None
        assert popen.process.stdin.frames == [
            (FrameKind.AUDIO, b"\x01\x02"),
            (FrameKind.AUDIO, b"\x03\x04"),
        ]

    def test_each_frame_is_flushed(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """An unflushed AUDIO frame is a deadlock, not a delay."""
        rec, popen = started()
        rec.accept_pcm(b"\x01\x02")
        assert popen.process is not None
        assert popen.process.stdin.flushes == 1

    def test_an_empty_buffer_writes_nothing(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Polling for results must not put empty frames on the wire."""
        rec, popen = started()
        rec.accept_pcm(b"")
        assert popen.process is not None
        assert popen.process.stdin.written == b""

    def test_returns_the_results_the_reader_thread_collected(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Results arrive on another thread, so they are polled for, not awaited."""
        rec, _ = started(
            result_frame("hey chamber", final=True, confidence=0.75),
            result_frame("start recording", final=False, confidence=0.25),
        )

        collected: list[Recognition] = []
        assert wait_until(
            lambda: bool(collected.extend(rec.accept_pcm(b"")) or len(collected) >= 2)
        ), f"only {len(collected)} of 2 results arrived: {collected!r}"

        assert collected == [
            Recognition("hey chamber", True, 0.75),
            Recognition("start recording", False, 0.25),
        ]

    def test_results_are_drained_and_not_repeated(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Taking a result removes it; a second call must not see it again."""
        rec, _ = started(result_frame("once"))

        assert wait_until(lambda: rec.pending == 1)
        assert rec.accept_pcm(b"") == [Recognition("once", True, 0.0)]
        assert rec.accept_pcm(b"") == []
        assert rec.pending == 0

    def test_pending_counts_undelivered_results(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """`pending` is how a test (or the GUI) sees the reader's backlog."""
        rec, _ = started(result_frame("a"), result_frame("b"), result_frame("c"))
        assert wait_until(lambda: rec.pending == 3)
        rec.accept_pcm(b"")
        assert rec.pending == 0

    def test_a_broken_pipe_never_raises_and_is_recorded(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """An exception here kills the consumer thread and stops capture.

        A dead worker must degrade to 'no recognitions ever again', recorded in
        `error` -- the same failure mode as NullRecognizer, and vastly
        preferable to losing the recording.
        """
        rec, popen = started()
        assert popen.process is not None
        boom = BrokenPipeError(32, "Broken pipe")
        popen.process.stdin.raise_on_write = boom

        assert rec.accept_pcm(b"\x00\x01\x02\x03") == []
        assert rec.error is boom, f"the failure must be recorded, got {rec.error!r}"

    def test_it_keeps_not_raising_on_every_later_call(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """The audio path stays quiet for the rest of the process's life."""
        rec, popen = started()
        assert popen.process is not None
        popen.process.stdin.raise_on_write = BrokenPipeError("gone")

        for _ in range(5):
            assert rec.accept_pcm(b"\x00\x01") == []

    def test_the_first_error_is_the_one_kept(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A later, less informative failure must not overwrite the real cause."""
        rec, popen = started()
        assert popen.process is not None
        first = BrokenPipeError("the original failure")
        popen.process.stdin.raise_on_write = first

        rec.accept_pcm(b"\x00")
        popen.process.stdin.raise_on_write = ValueError("a later, vaguer failure")
        rec.accept_pcm(b"\x00")

        assert rec.error is first

    def test_a_closed_recognizer_consumes_audio_quietly(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """The pipeline may still be draining chunks when the gate is shut down."""
        rec, popen = started()
        rec.close()
        assert popen.process is not None
        before = bytes(popen.process.stdin.written)

        assert rec.accept_pcm(b"\x00\x01\x02") == []
        assert popen.process.stdin.written == before, (
            "a closed recogniser must not write to a stream it already closed"
        )


class TestReset:
    """RESET is fire-and-forget, and shares accept_pcm's never-raise rule."""

    def test_writes_a_reset_frame(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A discontinuity in the capture must reach the decoder."""
        rec, popen = started()
        rec.reset()
        assert popen.process is not None
        assert popen.process.stdin.frames == [(FrameKind.RESET, b"")]

    def test_reset_frames_have_no_payload(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A control frame is exactly its five-byte header."""
        rec, popen = started()
        rec.reset()
        assert popen.process is not None
        assert bytes(popen.process.stdin.written) == encode_frame(FrameKind.RESET)

    def test_reset_never_raises_on_a_broken_pipe(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """reset() runs on the same thread as accept_pcm and has the same rule."""
        rec, popen = started()
        assert popen.process is not None
        popen.process.stdin.raise_on_write = BrokenPipeError("gone")
        rec.reset()
        assert isinstance(rec.error, BrokenPipeError)

    def test_reset_before_start_is_a_noop(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """There is no worker to tell, and nothing to complain about."""
        make_recognizer(FakePopen()).reset()


# ==========================================================================
# close()
# ==========================================================================

class TestClose:
    """Shutdown escalates until the child is actually gone, and never raises."""

    def test_writes_a_shutdown_frame(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """The polite request comes first, before any signal."""
        rec, popen = started()
        rec.close()
        assert popen.process is not None
        assert popen.process.stdin.frames == [(FrameKind.SHUTDOWN, b"")]

    def test_shutdown_is_sent_before_stdin_is_closed(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Closing stdin first would make the frame unwritable."""
        rec, popen = started()
        rec.accept_pcm(b"\x01\x02")
        rec.close()

        assert popen.process is not None
        assert popen.process.stdin.kinds == [FrameKind.AUDIO, FrameKind.SHUTDOWN]
        assert popen.process.stdin.closed is True

    def test_marks_itself_closed(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """`closed` is the flag the owning sink checks."""
        rec, _ = started()
        assert rec.closed is False
        rec.close()
        assert rec.closed is True

    def test_is_idempotent(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Called from a sink's close(), which the pipeline calls idempotently."""
        rec, popen = started()
        rec.close()
        rec.close()
        rec.close()

        assert popen.process is not None
        assert popen.process.stdin.kinds == [FrameKind.SHUTDOWN], (
            "only the first close() may send a SHUTDOWN frame"
        )

    def test_close_before_start_is_harmless(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """There is no process, so there is nothing to shut down."""
        rec = make_recognizer(FakePopen())
        rec.close()
        assert rec.closed is True
        assert rec.error is None

    def test_waits_for_a_voluntary_exit_before_signalling(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A worker that exits on its own must never be terminated or killed."""
        rec, popen = started()
        rec.close()

        assert popen.process is not None
        assert popen.process.wait_calls >= 1
        assert popen.process.terminate_calls == 0, (
            "a cooperative worker must not be signalled"
        )
        assert popen.process.kill_calls == 0

    def test_escalates_to_terminate_and_kill_when_wait_times_out(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """A worker wedged inside Kaldi delays shutdown, it does not prevent it."""
        popen = FakePopen(
            stdout_bytes=ready_frame(),
            wait_raises=subprocess.TimeoutExpired(cmd="worker", timeout=2.0),
        )
        rec = make_recognizer(popen)
        rec.start()

        rec.close()

        assert popen.process is not None
        assert popen.process.terminate_calls == 1, (
            "a wait() that times out must escalate to terminate()"
        )
        assert popen.process.kill_calls == 1, (
            "a terminate() that also times out must escalate to kill()"
        )

    def test_does_not_raise_when_every_pipe_is_broken(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """Shutdown must not be derailed by a child process behaving badly."""
        popen = FakePopen(
            stdout_bytes=ready_frame(),
            stdin=FakeStdin(raise_on_write=BrokenPipeError("gone")),
            wait_raises=OSError("no such process"),
        )
        rec = make_recognizer(popen)
        rec.start()

        rec.close()          # must not raise
        rec.close()
        assert rec.closed is True

    def test_joins_the_reader_threads(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A leaked daemon thread per gate restart is a slow leak in a long session."""
        before = {t.name for t in threading.enumerate()}
        rec, _ = started(result_frame("a"))
        rec.close()

        assert wait_until(
            lambda: not {
                t.name for t in threading.enumerate() if t.name.startswith("voicegate-")
            }
            - before
        ), "close() must join voicegate-reader and voicegate-stderr"

    def test_closes_all_three_pipes(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Leaked pipe handles keep the child's file descriptors alive."""
        rec, popen = started()
        rec.close()

        assert popen.process is not None
        assert popen.process.stdin.closed is True
        assert popen.process.stdout.closed is True
        assert popen.process.stderr.closed is True


# ==========================================================================
# stderr draining
# ==========================================================================

class TestStderrTail:
    """Drained because a full pipe wedges the worker, kept because it is evidence."""

    def test_is_empty_before_start(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """Nothing has been said yet, and an empty string is not an error."""
        assert make_recognizer(FakePopen()).stderr_tail == ""

    def test_collects_what_the_worker_wrote(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Kaldi diagnostics and Python tracebacks both land here."""
        rec, _ = started(stderr_bytes=b"LOG (VoskAPI): loaded\nWARNING: slow\n")
        assert wait_until(lambda: rec.stderr_tail.count("\n") == 1)
        assert rec.stderr_tail == "LOG (VoskAPI): loaded\nWARNING: slow"

    def test_strips_line_endings(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """The child may be a Windows interpreter writing CRLF."""
        rec, _ = started(stderr_bytes=b"first\r\nsecond\r\n")
        assert wait_until(lambda: rec.stderr_tail.count("\n") == 1)
        assert rec.stderr_tail == "first\nsecond"
        assert "\r" not in rec.stderr_tail

    def test_is_bounded_to_stderr_lines(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A worker that logs every frame must not grow the parent without limit."""
        noisy = b"".join(f"line {i}\n".encode("ascii") for i in range(500))
        rec, _ = started(stderr_bytes=noisy)

        assert wait_until(lambda: len(rec.stderr_tail.splitlines()) == STDERR_LINES), (
            f"expected {STDERR_LINES} retained lines, got "
            f"{len(rec.stderr_tail.splitlines())}"
        )
        lines = rec.stderr_tail.splitlines()
        assert lines[-1] == "line 499", "the tail must keep the NEWEST lines"
        assert lines[0] == f"line {500 - STDERR_LINES}"

    def test_undecodable_bytes_do_not_kill_the_drainer(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """Kaldi can emit locale-encoded bytes; losing stderr must not be fatal."""
        rec, _ = started(stderr_bytes=b"bad \xff\xfe bytes\nstill going\n")
        assert wait_until(lambda: "still going" in rec.stderr_tail)
        assert rec.error is None


# ==========================================================================
# _decode_result -- coercion of a payload from another interpreter
# ==========================================================================

class TestDecodeResult:
    """Version skew between parent and worker must degrade, not crash the reader."""

    def test_decodes_a_well_formed_payload(self) -> None:
        """The happy path: exactly what worker._encode_result produces."""
        payload = encode_json({"text": "hey chamber", "final": True, "confidence": 0.5})
        assert _decode_result(payload) == Recognition("hey chamber", True, 0.5)

    def test_missing_fields_default_safely(self) -> None:
        """An empty object is a valid, empty, non-final result."""
        assert _decode_result(b"{}") == Recognition("", False, 0.0)

    @pytest.mark.parametrize("text", [123, None, ["a"], {"b": 1}, True, 1.5])
    def test_a_non_string_text_becomes_empty(self, text: object) -> None:
        """Recognition.text is typed str; downstream code slices and lowercases it."""
        got = _decode_result(encode_json({"text": text, "final": True}))
        assert got.text == "" and isinstance(got.text, str)

    def test_a_bool_confidence_becomes_zero(self) -> None:
        """bool is a subclass of int, so it must be excluded explicitly.

        Without the isinstance(confidence, bool) guard, True would silently
        become a confidence of 1.0 -- the most confident result in the log.
        """
        assert _decode_result(
            encode_json({"text": "x", "final": True, "confidence": True})
        ).confidence == 0.0
        assert _decode_result(
            encode_json({"text": "x", "final": True, "confidence": False})
        ).confidence == 0.0

    @pytest.mark.parametrize("confidence", ["0.9", None, [0.9], {"a": 1}])
    def test_a_non_numeric_confidence_becomes_zero(self, confidence: object) -> None:
        """float('0.9') would work; trusting it would not, so it is refused."""
        got = _decode_result(encode_json({"text": "x", "confidence": confidence}))
        assert got.confidence == 0.0

    def test_an_integer_confidence_is_accepted_as_a_float(self) -> None:
        """JSON has one number type; 1 and 1.0 are the same value."""
        got = _decode_result(encode_json({"text": "x", "confidence": 1}))
        assert got.confidence == 1.0 and isinstance(got.confidence, float)

    @pytest.mark.parametrize(
        ("final", "expected"),
        [(True, True), (False, False), (1, True), (0, False), ("yes", True),
         (None, False), ([], False)],
    )
    def test_final_is_coerced_to_a_bool(self, final: object, expected: bool) -> None:
        """The gate branches on `final`; it must never be a truthy string."""
        got = _decode_result(encode_json({"text": "x", "final": final}))
        assert got.final is expected

    def test_unknown_extra_fields_are_ignored(self) -> None:
        """A newer worker may send more; that is not a reason to fail."""
        payload = encode_json(
            {"text": "x", "final": True, "confidence": 0.1, "words": [1, 2], "v": 2}
        )
        assert _decode_result(payload) == Recognition("x", True, 0.1)

    def test_a_malformed_result_does_not_stop_the_reader_thread(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A coerced result still arrives, and the frames behind it still do too."""
        rec, _ = started(
            encode_frame(FrameKind.RESULT, encode_json({"text": 42, "final": "yes"})),
            result_frame("after the bad one"),
        )

        collected: list[Recognition] = []
        assert wait_until(
            lambda: bool(collected.extend(rec.accept_pcm(b"")) or len(collected) >= 2)
        ), f"the reader thread stopped after the malformed result: {collected!r}"
        assert collected == [
            Recognition("", True, 0.0),
            Recognition("after the bad one", True, 0.0),
        ]


# ==========================================================================
# the reader thread's own error handling
# ==========================================================================

class TestReaderThread:
    """A daemon thread that never re-raises; `error` is how the owner finds out."""

    def test_a_desync_after_ready_is_recorded_not_raised(
        self, make_recognizer: Callable[..., SubprocessRecognizer]
    ) -> None:
        """Garbage mid-stream stops results without taking the process down."""
        popen = FakePopen(
            stdout_bytes=ready_frame() + result_frame("good") + b"\x99\x00\x00\x00\x00"
        )
        rec = make_recognizer(popen)
        rec.start()

        assert wait_until(lambda: rec.error is not None), "the desync must be recorded"
        assert "unknown frame kind 153" in str(rec.error)
        assert rec.accept_pcm(b"") == [Recognition("good", True, 0.0)], (
            "results decoded before the desync must not be lost"
        )

    def test_frames_that_travel_the_other_way_are_dropped(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """A confused worker sending AUDIO back is harmless, not fatal."""
        rec, _ = started(
            encode_frame(FrameKind.AUDIO, b"\x00\x01"),
            encode_frame(FrameKind.RESET),
            result_frame("still here"),
        )

        assert wait_until(lambda: rec.pending == 1)
        assert rec.accept_pcm(b"") == [Recognition("still here", True, 0.0)]
        assert rec.error is None

    def test_a_late_error_frame_is_recorded_without_raising(
        self, started: Callable[..., tuple[SubprocessRecognizer, FakePopen]]
    ) -> None:
        """An ERROR after READY ends the reader; it must not crash the thread."""
        rec, _ = started(result_frame("last one"), error_frame("worker died"))

        assert wait_until(lambda: rec.pending == 1)
        assert rec.accept_pcm(b"") == [Recognition("last one", True, 0.0)]
        assert rec.error is None, "an ERROR frame is the worker's news, not an exception"
