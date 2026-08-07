"""Tests for echochamber.speakerid.qnn_npu_worker.

Only the parts reachable without onnxruntime-qnn or a Hexagon NPU: argument
parsing, binary-mode setup, and the frame-serving loop's control flow
(SHUTDOWN/RESET handling) against a stubbed session. make_session()/run()'s
actual QNN session creation is exercised for real only on ARM64 hardware --
see scripts/setup_speakerid_qnn.py's printed follow-up for that manual step.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from echochamber.speakerid.protocol import FrameKind, read_frame, write_frame
from echochamber.speakerid.qnn_npu_worker import _send_error, _serve, build_parser


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: echoes a fixed embedding."""

    def __init__(self, embedding: np.ndarray) -> None:
        self._embedding = embedding

    def run(self, output_names: list[str], feed: dict) -> list[np.ndarray]:
        return [self._embedding.reshape(1, -1)]


class TestBuildParser:
    def test_requires_onnx_path(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_onnx_path(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--onnx-path", "model.onnx"])
        assert args.onnx_path == "model.onnx"
        assert args.htp_performance_mode == "burst"
        assert args.check is False
        assert args.verbose is False

    def test_check_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--onnx-path", "model.onnx", "--check"])
        assert args.check is True


class TestSendError:
    def test_writes_an_error_frame_with_the_traceback(self) -> None:
        stream = io.BytesIO()
        try:
            raise ValueError("boom")
        except ValueError as exc:
            assert _send_error(stream, exc) is True
        stream.seek(0)
        kind, payload = read_frame(stream)
        assert kind is FrameKind.ERROR
        assert "ValueError: boom" in payload.decode("utf-8")

    def test_returns_false_when_the_pipe_is_gone(self) -> None:
        class DeadStream:
            def write(self, data: bytes) -> int:
                raise BrokenPipeError("gone")

            def flush(self) -> None:
                pass

        assert _send_error(DeadStream(), RuntimeError("x")) is False


class TestServe:
    def test_embed_frame_returns_a_result(self) -> None:
        session = FakeSession(np.arange(192, dtype=np.float32))
        stdin = io.BytesIO()
        feat = np.zeros((600, 80), dtype=np.float32)
        write_frame(stdin, FrameKind.EMBED, feat.tobytes())
        write_frame(stdin, FrameKind.SHUTDOWN)
        stdin.seek(0)
        stdout = io.BytesIO()

        code = _serve(stdin, stdout, session, "fbank", "embedding", feat_dim=80)
        assert code == 0

        stdout.seek(0)
        kind, payload = read_frame(stdout)
        assert kind is FrameKind.RESULT
        out = np.frombuffer(payload, dtype=np.float32)
        assert np.array_equal(out, np.arange(192, dtype=np.float32))

    def test_shutdown_frame_ends_the_loop_cleanly(self) -> None:
        session = FakeSession(np.zeros(192, dtype=np.float32))
        stdin = io.BytesIO()
        write_frame(stdin, FrameKind.SHUTDOWN)
        stdin.seek(0)
        stdout = io.BytesIO()
        code = _serve(stdin, stdout, session, "fbank", "embedding", feat_dim=80)
        assert code == 0

    def test_eof_ends_the_loop_cleanly(self) -> None:
        session = FakeSession(np.zeros(192, dtype=np.float32))
        stdin = io.BytesIO(b"")
        stdout = io.BytesIO()
        code = _serve(stdin, stdout, session, "fbank", "embedding", feat_dim=80)
        assert code == 0

    def test_reset_frame_is_ignored_without_ending_the_loop(self) -> None:
        session = FakeSession(np.ones(192, dtype=np.float32))
        stdin = io.BytesIO()
        write_frame(stdin, FrameKind.RESET)
        feat = np.zeros((600, 80), dtype=np.float32)
        write_frame(stdin, FrameKind.EMBED, feat.tobytes())
        write_frame(stdin, FrameKind.SHUTDOWN)
        stdin.seek(0)
        stdout = io.BytesIO()
        code = _serve(stdin, stdout, session, "fbank", "embedding", feat_dim=80)
        assert code == 0
        stdout.seek(0)
        kind, _ = read_frame(stdout)
        assert kind is FrameKind.RESULT
