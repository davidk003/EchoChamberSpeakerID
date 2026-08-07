"""``QnnEmbedder``: the main process's handle on the two-hop QNN subprocess chain.

Lives in the main application process (native ARM64 in production) and
launches ``qnn_driver_worker`` as its own child under an **x64** interpreter --
mirroring
:class:`echochamber.voicegate.subprocess_recognizer.SubprocessRecognizer`'s
shape closely enough that this class's docstrings point back to that one
rather than re-deriving the same reasoning: a mandatory reader thread to avoid
a two-process deadlock on full pipes, a separate stderr drainer that also
keeps the tail for a useful startup error, and a never-raise contract on the
per-call path because this runs on the pipeline's consumer thread.

The one thing genuinely different from the Vosk case: the driver process
launches a *second* subprocess of its own
(``qnn_npu_worker``, on native ARM64) before it can do anything, so this
class's startup timeout has to cover both hops' initialization, not just one
process's model load.
"""

from __future__ import annotations

import collections
import subprocess
import threading
import time
from typing import Any, BinaryIO, Callable

import numpy as np

from echochamber.speakerid.protocol import (
    FrameKind,
    ProtocolError,
    decode_json,
    read_frame,
    write_frame,
)

__all__ = ["QnnEmbedder", "QnnStartupError"]

DRIVER_MODULE: str = "echochamber.speakerid.qnn_driver_worker"
"""Module the x64 child interpreter is asked to run with ``-m``."""

STDERR_LINES: int = 50
"""How many trailing stderr lines are kept for :attr:`QnnEmbedder.stderr_tail`."""

EMBEDDING_SIZE: int = 192

_SHUTDOWN_WAIT_S: float = 2.0
_TERMINATE_WAIT_S: float = 2.0
"""Longer than the Vosk worker's: shutting down cleanly here means the driver
process must first shut down its own NPU-worker child."""

_STDERR_SETTLE_S: float = 0.5
_CALL_TIMEOUT_S: float = 130.0
"""Slightly above the driver's own ``_NPU_CALL_TIMEOUT_S`` (120 s), so a
genuinely slow HTP dispatch is reported by the driver's own timeout rather
than this class's, which would otherwise race it and report a less specific
error."""


class QnnStartupError(RuntimeError):
    """The driver process (or the NPU worker inside it) never reported readiness."""


class QnnEmbedder:
    """Run CAM++ inference on the Hexagon NPU via a two-hop subprocess chain.

    Satisfies :class:`echochamber.speakerid.verifier.Embedder`.  Single-use: a
    closed embedder cannot be restarted; build a new instance, which is also
    what a configuration change wants.
    """

    __slots__ = (
        "_driver_python",
        "_onnx_path",
        "_npu_python",
        "_t_max",
        "_npu_module",
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
        driver_python: str,
        onnx_path: str,
        npu_python: str | None = None,
        t_max: int = 600,
        startup_timeout_s: float = 30.0,
        popen: Callable[..., Any] | None = None,
        npu_module: str | None = None,
    ) -> None:
        """Prepare an embedder; nothing is launched until :meth:`start`.

        Args:
            driver_python: Interpreter for the x64 driver process (has
                torch). ``SpeakerIdConfig.qnn_worker_python``.
            onnx_path: Path to the exported static-shape ONNX graph,
                interpreted by the *NPU worker*, so it must make sense to
                that interpreter (in practice the same filesystem, since both
                hops run on the same machine).
            npu_python: Interpreter for the native ARM64 NPU worker, or
                ``None`` to let the driver fall back to its own default (the
                ``CAMPPLUS_QNN_PYTHON`` environment variable, then its own
                interpreter).
            t_max: Fixed fbank-frame length the exported graph accepts.
            startup_timeout_s: How long :meth:`start` waits for both hops to
                report readiness. Generous by default because the HTP
                backend compiles the graph on session creation.
            popen: Factory used instead of :func:`subprocess.Popen`. The
                testing seam; mirrors
                ``SubprocessRecognizer``'s own ``popen`` parameter.
            npu_module: Module the driver runs the NPU worker with ``-m``, or
                ``None`` for its own default
                (``echochamber.speakerid.qnn_npu_worker``). Passed through as
                the driver's own ``--npu-module`` flag; lets a test exercise
                the *real* driver subprocess end to end against
                ``tests/fake_qnn_npu_worker.py`` with no NPU or ONNX Runtime
                anywhere.

        Raises:
            ValueError: If ``startup_timeout_s`` is not positive.
        """
        startup_timeout_s = float(startup_timeout_s)
        if startup_timeout_s <= 0.0:
            raise ValueError(
                f"startup_timeout_s must be > 0, got {startup_timeout_s}"
            )

        self._driver_python: str = str(driver_python)
        self._onnx_path: str = str(onnx_path)
        self._npu_python: str | None = npu_python
        self._t_max: int = int(t_max)
        self._npu_module: str | None = npu_module
        self._startup_timeout_s: float = startup_timeout_s
        self._popen: Callable[..., Any] = (
            subprocess.Popen if popen is None else popen
        )

        self._process: Any = None
        self._lock: threading.Lock = threading.Lock()
        self._results: collections.deque[bytes] = collections.deque()
        self._stderr_lines: collections.deque[str] = collections.deque(
            maxlen=STDERR_LINES
        )
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._ready: threading.Event = threading.Event()
        self._ready_ok: bool = False
        self._worker_error: str | None = None
        self._error: BaseException | None = None
        self._started: bool = False
        self._closed: bool = False

    @property
    def command(self) -> tuple[str, ...]:
        """The exact argument vector :meth:`start` launches."""
        argv: list[str] = [
            self._driver_python,
            "-m",
            DRIVER_MODULE,
            "--onnx-path",
            self._onnx_path,
            "--t-max",
            str(self._t_max),
        ]
        if self._npu_python:
            argv.extend(("--npu-python", self._npu_python))
        if self._npu_module:
            argv.extend(("--npu-module", self._npu_module))
        return tuple(argv)

    @property
    def running(self) -> bool:
        """``True`` while the driver process is alive."""
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
        """First failure recorded by a background thread or a failed write."""
        return self._error

    @property
    def stderr_tail(self) -> str:
        """The last :data:`STDERR_LINES` lines the driver wrote to stderr."""
        with self._lock:
            return "\n".join(self._stderr_lines)

    def start(self) -> None:
        """Launch the driver process and block until the whole chain is ready.

        Raises:
            RuntimeError: If called twice, or after :meth:`close`.
            QnnStartupError: If the driver could not be launched, failed
                before reporting readiness (including a failure to start its
                own NPU-worker child), or did not report readiness within
                ``startup_timeout_s``.
        """
        if self._started:
            raise RuntimeError(
                "QnnEmbedder has already been started and cannot be "
                "restarted; create a new QnnEmbedder"
            )
        self._started = True

        argv = list(self.command)
        try:
            process = self._popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised as a startup error
            self._error = exc
            self._closed = True
            raise QnnStartupError(
                f"could not launch the QNN driver {self._driver_python!r}: {exc}"
            ) from exc

        self._process = process
        self._reader = self._spawn(self._read_results, "speakerid-reader")
        self._stderr_reader = self._spawn(self._read_stderr, "speakerid-stderr")

        if not self._ready.wait(self._startup_timeout_s):
            self._fail_startup(
                f"the QNN driver did not report readiness within "
                f"{self._startup_timeout_s:g} s"
            )
        if not self._ready_ok:
            reported = self._worker_error
            self._fail_startup(
                "the QNN driver reported a failure while starting"
                if reported
                else "the QNN driver exited before reporting readiness",
                reported=reported,
            )

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Embed mono samples via the QNN subprocess chain.

        Args:
            samples: 1-D mono ``float32`` samples in ``[-1, 1]``.
            sample_rate: Sample rate of ``samples`` in Hz.  Resampled to
                16 kHz here, before crossing into the driver process, which
                assumes its input already is -- this project's own capture
                pipeline always delivers 16 kHz, so that hop is never
                exercised in practice, but a caller from a different sample
                rate still gets a correct embedding rather than a silently
                wrong one.

        Returns:
            A ``(192,)`` L2-normalized ``float32`` numpy array.

        Raises:
            RuntimeError: If the chain has died or the driver reports a
                failure processing this clip (a malformed clip, an NPU
                inference error).
            TimeoutError: If no result arrives within the call timeout.
        """
        if self._closed or self._process is None:
            raise RuntimeError("QnnEmbedder is closed")
        if sample_rate != 16_000:
            samples = _resample(samples, sample_rate, 16_000)
        payload = np.ascontiguousarray(samples, dtype=np.float32).tobytes()

        if not self._send(FrameKind.EMBED, payload):
            raise RuntimeError("could not send audio to the QNN driver: pipe is gone")

        deadline = time.monotonic() + _CALL_TIMEOUT_S
        while True:
            with self._lock:
                if self._results:
                    raw = self._results.popleft()
                    break
            if self._error is not None:
                raise RuntimeError(f"QNN driver reader failed: {self._error}")
            if self._worker_error is not None:
                raise RuntimeError(f"QNN driver reported an error: {self._worker_error}")
            if not self.running:
                raise RuntimeError("QNN driver subprocess exited unexpectedly")
            if time.monotonic() > deadline:
                raise TimeoutError("QNN driver did not return a result in time")
            time.sleep(0.005)

        emb = np.frombuffer(raw, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if not np.isfinite(emb).all() or norm == 0.0:
            raise RuntimeError("embedding contains non-finite values or is zero")
        return (emb / norm).astype(np.float32, copy=False)

    def close(self) -> None:
        """Shut the driver process down, escalating until it is actually gone.

        Idempotent and never raises; see
        :meth:`echochamber.voicegate.subprocess_recognizer.SubprocessRecognizer.close`,
        which this mirrors.
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
            for thread in (self._reader, self._stderr_reader):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(_TERMINATE_WAIT_S)
            self._close_stream(getattr(process, "stdout", None))
            self._close_stream(getattr(process, "stderr", None))
        except BaseException as exc:  # noqa: BLE001 - close must not raise
            if self._error is None:
                self._error = exc

    def _read_results(self) -> None:
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
            if self._error is None:
                self._error = exc
        except BaseException as exc:  # noqa: BLE001 - reported via `error`
            if self._error is None:
                self._error = exc
        finally:
            self._ready.set()

    def _read_stderr(self) -> None:
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
            return

    def _send(self, kind: FrameKind, payload: bytes = b"") -> bool:
        process = self._process
        if process is None or self._closed:
            return False
        stream: BinaryIO | None = getattr(process, "stdin", None)
        if stream is None:
            return False
        try:
            write_frame(stream, kind, payload)
        except (OSError, ValueError, ProtocolError) as exc:
            if self._error is None:
                self._error = exc
            return False
        return True

    def _fail_startup(self, reason: str, reported: str | None = None) -> None:
        deadline = time.monotonic() + _STDERR_SETTLE_S
        while not self.stderr_tail and time.monotonic() < deadline:
            time.sleep(0.01)
        tail = self.stderr_tail

        self.close()

        parts = [
            f"the QNN embedder failed to start: {reason}",
            f"  command: {' '.join(self.command)}",
        ]
        if reported:
            parts.append(f"  driver reported:\n{_indent(reported)}")
        if tail:
            parts.append(f"  driver stderr:\n{_indent(tail)}")
        else:
            parts.append("  driver stderr: (nothing)")
        raise QnnStartupError("\n".join(parts))

    def _spawn(self, target: Callable[[], None], name: str) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def _wait(self, process: Any, timeout: float) -> bool:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        except BaseException:  # noqa: BLE001 - a fake process, or already reaped
            return True
        return True

    def _signal(self, process: Any, method: str) -> None:
        action = getattr(process, method, None)
        if action is None:
            return
        try:
            action()
        except BaseException:  # noqa: BLE001 - already dead, or a fake process
            pass

    def _close_stream(self, stream: BinaryIO | None) -> None:
        if stream is None:
            return
        try:
            stream.close()
        except BaseException:  # noqa: BLE001 - closing a dead pipe is not news
            pass

    def __repr__(self) -> str:
        """Return a debugging representation of this embedder's state."""
        return (
            f"{type(self).__name__}(driver={self._driver_python!r}, "
            f"onnx_path={self._onnx_path!r}, running={self.running}, "
            f"closed={self._closed})"
        )


def _resample(samples: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono samples with plain numpy linear interpolation.

    Only reached if a caller ever hands this embedder audio at a rate other
    than 16 kHz, which the capture pipeline in this project never does; kept
    dependency-free (no torchaudio here -- this runs in the main process,
    which has no torch) rather than unreachable-and-untested.
    """
    if sample_rate == target_rate or samples.size == 0:
        return np.asarray(samples, dtype=np.float32)
    duration = samples.shape[-1] / sample_rate
    n_out = max(1, int(round(duration * target_rate)))
    x_old = np.linspace(0.0, duration, num=samples.shape[-1], endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32, copy=False)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())
