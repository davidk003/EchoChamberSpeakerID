"""Tests for echochamber.speakerid.qnn_subprocess, with no child process.

Same fake-Popen shape as tests/test_voicegate_subprocess.py: QnnEmbedder
takes a popen= factory for exactly the reason SubprocessRecognizer does --
the thing it drives (an x64 driver process, itself spawning a native ARM64
NPU worker) is not available where the tests run. FakeProcess/FakePopen below
mirror that file's classes, adapted to this module's own frame protocol
(echochamber.speakerid.protocol, not echochamber.voicegate.protocol) and its
EMBED/RESULT/READY/ERROR frame shapes.
"""

from __future__ import annotations

import io
import re
import time
from typing import Any, Callable, Iterator

import numpy as np
import pytest

from echochamber.speakerid.protocol import (
    FrameKind,
    encode_frame,
    encode_json,
    read_frame,
)
from echochamber.speakerid.qnn_subprocess import QnnEmbedder, QnnStartupError

DRIVER_PYTHON = "/opt/x64-venv/Scripts/python.exe"
ONNX_PATH = "/models/speakerid/campplus_qnn.onnx"

TIMEOUT = 5.0


def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def ready_frame() -> bytes:
    return encode_frame(FrameKind.READY, encode_json({"onnx_path": ONNX_PATH, "t_max": 600}))


def result_frame(embedding: np.ndarray) -> bytes:
    return encode_frame(FrameKind.RESULT, embedding.astype(np.float32).tobytes())


def error_frame(message: str) -> bytes:
    return encode_frame(FrameKind.ERROR, message.encode("utf-8"))


def frames_in(blob: bytes) -> list[tuple[FrameKind, bytes]]:
    stream = io.BytesIO(blob)
    out: list[tuple[FrameKind, bytes]] = []
    while True:
        frame = read_frame(stream)
        if frame is None:
            return out
        out.append(frame)


# --------------------------------------------------------------------------
# fake child process
# --------------------------------------------------------------------------

class FakeStdin:
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
        return frames_in(bytes(self.written))

    @property
    def kinds(self) -> list[FrameKind]:
        return [kind for kind, _ in self.frames]


class FakeProcess:
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
        self.stderr: Any = io.BytesIO(stderr_bytes)
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


class FakePopen:
    def __init__(self, raises: BaseException | None = None, **process_kwargs: Any) -> None:
        self.raises = raises
        self.process_kwargs = process_kwargs
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.process: FakeProcess | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> FakeProcess:
        self.calls.append((list(argv), dict(kwargs)))
        if self.raises is not None:
            raise self.raises
        self.process = FakeProcess(argv, **self.process_kwargs, popen_kwargs=kwargs)
        return self.process


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def make_embedder() -> Iterator[Callable[..., QnnEmbedder]]:
    created: list[QnnEmbedder] = []

    def _make(
        popen: Any,
        driver_python: str = DRIVER_PYTHON,
        onnx_path: str = ONNX_PATH,
        startup_timeout_s: float = TIMEOUT,
        **kwargs: Any,
    ) -> QnnEmbedder:
        emb = QnnEmbedder(
            driver_python,
            onnx_path,
            startup_timeout_s=startup_timeout_s,
            popen=popen,
            **kwargs,
        )
        created.append(emb)
        return emb

    yield _make

    for emb in created:
        try:
            emb.close()
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


@pytest.fixture
def started(
    make_embedder: Callable[..., QnnEmbedder]
) -> Callable[..., tuple[QnnEmbedder, FakePopen]]:
    def _start(*extra_frames: bytes, stderr_bytes: bytes = b"", **kwargs: Any) -> tuple[QnnEmbedder, FakePopen]:
        popen = FakePopen(stdout_bytes=ready_frame() + b"".join(extra_frames), stderr_bytes=stderr_bytes)
        emb = make_embedder(popen, **kwargs)
        emb.start()
        return emb, popen

    return _start


# ==========================================================================
# construction and the argument vector
# ==========================================================================

class TestConstruction:
    def test_nothing_is_launched_by_the_constructor(self) -> None:
        popen = FakePopen()
        emb = QnnEmbedder(DRIVER_PYTHON, ONNX_PATH, popen=popen)
        assert popen.calls == []
        assert emb.running is False
        assert emb.closed is False
        assert emb.error is None

    @pytest.mark.parametrize("timeout", [0.0, -1.0, -0.001])
    def test_non_positive_startup_timeout_raises(self, timeout: float) -> None:
        with pytest.raises(ValueError, match=r"startup_timeout_s must be > 0"):
            QnnEmbedder(DRIVER_PYTHON, ONNX_PATH, startup_timeout_s=timeout)

    def test_command_names_the_driver_module(self) -> None:
        emb = QnnEmbedder(DRIVER_PYTHON, ONNX_PATH, popen=FakePopen())
        assert emb.command[0] == DRIVER_PYTHON
        assert "echochamber.speakerid.qnn_driver_worker" in emb.command
        assert ONNX_PATH in emb.command

    def test_npu_python_is_forwarded_when_given(self) -> None:
        emb = QnnEmbedder(DRIVER_PYTHON, ONNX_PATH, npu_python="/arm64/python.exe", popen=FakePopen())
        assert "--npu-python" in emb.command
        assert "/arm64/python.exe" in emb.command

    def test_npu_python_omitted_when_not_given(self) -> None:
        emb = QnnEmbedder(DRIVER_PYTHON, ONNX_PATH, popen=FakePopen())
        assert "--npu-python" not in emb.command


# ==========================================================================
# start()
# ==========================================================================

class TestStart:
    def test_becomes_running_once_ready_arrives(self, started: Callable[..., Any]) -> None:
        emb, popen = started()
        assert emb.running is True
        assert emb.error is None

    def test_double_start_raises(self, started: Callable[..., Any]) -> None:
        emb, _ = started()
        with pytest.raises(RuntimeError, match="already been started"):
            emb.start()

    def test_popen_launch_failure_raises_startup_error(self, make_embedder: Callable[..., Any]) -> None:
        popen = FakePopen(raises=OSError("no such file"))
        emb = make_embedder(popen)
        with pytest.raises(QnnStartupError, match="could not launch"):
            emb.start()
        assert emb.closed is True

    def test_worker_error_frame_before_ready_raises_startup_error(
        self, make_embedder: Callable[..., Any]
    ) -> None:
        popen = FakePopen(stdout_bytes=error_frame("Traceback: boom"))
        emb = make_embedder(popen)
        with pytest.raises(QnnStartupError, match="reported a failure while starting"):
            emb.start()

    def test_startup_error_includes_the_traceback(self, make_embedder: Callable[..., Any]) -> None:
        popen = FakePopen(stdout_bytes=error_frame("ModuleNotFoundError: no onnxruntime_qnn"))
        emb = make_embedder(popen)
        with pytest.raises(QnnStartupError, match="ModuleNotFoundError"):
            emb.start()

    def test_eof_before_ready_raises_startup_error(self, make_embedder: Callable[..., Any]) -> None:
        popen = FakePopen(stdout_bytes=b"")
        emb = make_embedder(popen)
        with pytest.raises(QnnStartupError, match="exited before reporting readiness"):
            emb.start()

    def test_startup_timeout_raises_and_reports_it(self, make_embedder: Callable[..., Any]) -> None:
        class NeverEndingStream:
            def read(self, n: int = -1) -> bytes:
                time.sleep(0.3)
                return b""

            def close(self) -> None:
                pass

        popen = FakePopen(stdout=NeverEndingStream())
        emb = make_embedder(popen, startup_timeout_s=0.15)
        with pytest.raises(QnnStartupError, match="did not report readiness within 0.15 s"):
            emb.start()


# ==========================================================================
# embed()
# ==========================================================================

class TestEmbed:
    def test_returns_normalized_embedding(self, started: Callable[..., Any]) -> None:
        raw = np.array([3.0, 4.0] + [0.0] * 190, dtype=np.float32)
        emb, popen = started(result_frame(raw))
        result = emb.embed(np.zeros(16_000, dtype=np.float32), 16_000)
        assert result.shape == (192,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-5)
        assert result[0] == pytest.approx(0.6, abs=1e-5)
        assert result[1] == pytest.approx(0.8, abs=1e-5)

    def test_sends_an_embed_frame_with_the_sample_bytes(self, started: Callable[..., Any]) -> None:
        raw = np.ones(192, dtype=np.float32)
        emb, popen = started(result_frame(raw))
        samples = np.linspace(-0.1, 0.1, 8000, dtype=np.float32)
        emb.embed(samples, 16_000)
        kinds = popen.process.stdin.kinds
        assert FrameKind.EMBED in kinds
        sent = dict(popen.process.stdin.frames)[FrameKind.EMBED]
        assert np.frombuffer(sent, dtype=np.float32).tolist() == pytest.approx(samples.tolist())

    def test_resamples_non_16k_input_before_sending(self, started: Callable[..., Any]) -> None:
        raw = np.ones(192, dtype=np.float32)
        emb, popen = started(result_frame(raw))
        samples = np.linspace(-0.1, 0.1, 8000, dtype=np.float32)  # 8 kHz
        emb.embed(samples, 8_000)
        sent = dict(popen.process.stdin.frames)[FrameKind.EMBED]
        sent_samples = np.frombuffer(sent, dtype=np.float32)
        # Resampled to 16 kHz means roughly double the sample count.
        assert sent_samples.size == pytest.approx(16_000, rel=0.05)

    def test_multiple_calls_consume_results_in_order(self, started: Callable[..., Any]) -> None:
        raw1 = np.array([1.0, 0.0] + [0.0] * 190, dtype=np.float32)
        raw2 = np.array([0.0, 1.0] + [0.0] * 190, dtype=np.float32)
        emb, popen = started(result_frame(raw1), result_frame(raw2))
        r1 = emb.embed(np.zeros(1000, dtype=np.float32), 16_000)
        r2 = emb.embed(np.zeros(1000, dtype=np.float32), 16_000)
        assert r1[0] == pytest.approx(1.0, abs=1e-5)
        assert r2[1] == pytest.approx(1.0, abs=1e-5)

    def test_zero_embedding_raises(self, started: Callable[..., Any]) -> None:
        raw = np.zeros(192, dtype=np.float32)
        emb, popen = started(result_frame(raw))
        with pytest.raises(RuntimeError, match="non-finite|zero"):
            emb.embed(np.zeros(1000, dtype=np.float32), 16_000)

    def test_non_finite_embedding_raises(self, started: Callable[..., Any]) -> None:
        raw = np.full(192, np.nan, dtype=np.float32)
        emb, popen = started(result_frame(raw))
        with pytest.raises(RuntimeError, match="non-finite"):
            emb.embed(np.zeros(1000, dtype=np.float32), 16_000)

    def test_error_frame_instead_of_result_raises(self, started: Callable[..., Any]) -> None:
        emb, popen = started(error_frame("bad clip"))
        with pytest.raises(RuntimeError, match="bad clip"):
            emb.embed(np.zeros(1000, dtype=np.float32), 16_000)

    def test_dead_process_raises(self, started: Callable[..., Any]) -> None:
        emb, popen = started()  # no RESULT frame ever comes
        popen.process._returncode = 1
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            emb.embed(np.zeros(1000, dtype=np.float32), 16_000)

    def test_closed_embedder_raises(self, started: Callable[..., Any]) -> None:
        emb, popen = started()
        emb.close()
        with pytest.raises(RuntimeError, match="closed"):
            emb.embed(np.zeros(1000, dtype=np.float32), 16_000)


# ==========================================================================
# close()
# ==========================================================================

class TestClose:
    def test_sends_shutdown_frame(self, started: Callable[..., Any]) -> None:
        emb, popen = started()
        emb.close()
        assert FrameKind.SHUTDOWN in popen.process.stdin.kinds

    def test_idempotent(self, started: Callable[..., Any]) -> None:
        emb, popen = started()
        emb.close()
        emb.close()  # must not raise
        assert popen.process.stdin.closed is True

    def test_close_before_start_is_safe(self) -> None:
        emb = QnnEmbedder(DRIVER_PYTHON, ONNX_PATH, popen=FakePopen())
        emb.close()  # must not raise; nothing was ever started
        assert emb.closed is True

    def test_escalates_to_terminate_when_wait_times_out(self, started: Callable[..., Any]) -> None:
        import subprocess

        emb, popen = started()
        popen.process.wait_raises = subprocess.TimeoutExpired(cmd="x", timeout=1.0)
        emb.close()
        assert popen.process.terminate_calls >= 1
