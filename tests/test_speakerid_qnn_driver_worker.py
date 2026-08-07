"""Tests for echochamber.speakerid.qnn_driver_worker.

Exercises the driver's own frame loop (run()/_serve()/_pad_or_truncate())
against tests/fake_qnn_npu_worker.py as its NPU-worker child -- a real
subprocess, but one with no ONNX Runtime or NPU anywhere, mirroring how the
speakerid package's own smoke tests during development validated the whole
chain. NpuWorkerHandle itself (the class that manages that child) is
exercised here as an integration point; see test_speakerid_qnn_subprocess.py
for the fully-faked unit tests of the outer QnnEmbedder <-> driver hop.
"""

from __future__ import annotations

import sys
import threading
from typing import Any

import numpy as np
import pytest

from echochamber.speakerid.protocol import (
    FrameKind,
    decode_json,
    read_frame,
    write_frame,
)
from echochamber.speakerid.qnn_driver_worker import NpuWorkerHandle, _pad_or_truncate, run

FAKE_NPU_MODULE = "tests.fake_qnn_npu_worker"


class TestPadOrTruncate:
    def test_pads_short_features_with_zeros(self) -> None:
        feat = np.ones((10, 80), dtype=np.float32)
        out = _pad_or_truncate(feat, t_max=20, feat_dim=80)
        assert out.shape == (20, 80)
        assert np.array_equal(out[:10], feat)
        assert np.array_equal(out[10:], np.zeros((10, 80), dtype=np.float32))

    def test_truncates_long_features(self) -> None:
        feat = np.arange(30 * 80, dtype=np.float32).reshape(30, 80)
        out = _pad_or_truncate(feat, t_max=20, feat_dim=80)
        assert out.shape == (20, 80)
        assert np.array_equal(out, feat[:20])

    def test_exact_length_is_unchanged(self) -> None:
        feat = np.ones((20, 80), dtype=np.float32)
        out = _pad_or_truncate(feat, t_max=20, feat_dim=80)
        assert out.shape == (20, 80)
        assert np.array_equal(out, feat)


class TestNpuWorkerHandle:
    def test_starts_and_reports_ready(self) -> None:
        npu = NpuWorkerHandle(sys.executable, "unused.onnx", npu_module=FAKE_NPU_MODULE, ready_timeout_s=15)
        try:
            assert npu._ready_ok is True
        finally:
            npu.close()

    def test_embed_round_trips_a_result(self) -> None:
        npu = NpuWorkerHandle(sys.executable, "unused.onnx", npu_module=FAKE_NPU_MODULE, ready_timeout_s=15)
        try:
            feat = np.zeros((600, 80), dtype=np.float32)
            raw = npu.embed(feat)
            assert len(raw) == 192 * 4
        finally:
            npu.close()

    def test_different_features_produce_different_results(self) -> None:
        npu = NpuWorkerHandle(sys.executable, "unused.onnx", npu_module=FAKE_NPU_MODULE, ready_timeout_s=15)
        try:
            r1 = npu.embed(np.full((600, 80), 0.1, dtype=np.float32))
            r2 = npu.embed(np.full((600, 80), 0.9, dtype=np.float32))
            assert r1 != r2
        finally:
            npu.close()

    def test_launch_failure_raises(self) -> None:
        with pytest.raises(RuntimeError, match="could not launch"):
            NpuWorkerHandle("/no/such/python.exe", "unused.onnx", npu_module=FAKE_NPU_MODULE, ready_timeout_s=5)

    def test_unimportable_module_raises_with_traceback(self) -> None:
        with pytest.raises(RuntimeError, match="failed to start"):
            NpuWorkerHandle(
                sys.executable, "unused.onnx", npu_module="tests.nonexistent_worker_module", ready_timeout_s=10
            )

    def test_close_is_idempotent(self) -> None:
        npu = NpuWorkerHandle(sys.executable, "unused.onnx", npu_module=FAKE_NPU_MODULE, ready_timeout_s=15)
        npu.close()
        npu.close()  # must not raise


class TestRun:
    """Exercises run()'s frame loop end to end over real OS pipes."""

    def test_serves_an_embed_request_and_shuts_down_cleanly(self) -> None:
        import os

        r_to_driver, w_to_driver = os.pipe()
        r_from_driver, w_from_driver = os.pipe()

        stdin = os.fdopen(r_to_driver, "rb")
        stdout = os.fdopen(w_from_driver, "wb")

        result: dict[str, Any] = {}

        def _run_driver() -> None:
            result["code"] = run(stdin, stdout, "unused.onnx", sys.executable, 600, FAKE_NPU_MODULE)

        thread = threading.Thread(target=_run_driver, daemon=True)
        thread.start()

        writer = os.fdopen(w_to_driver, "wb")
        reader = os.fdopen(r_from_driver, "rb")

        kind, payload = read_frame(reader)
        assert kind is FrameKind.READY
        assert decode_json(payload)["t_max"] == 600

        samples = np.linspace(-0.1, 0.1, 32_000, dtype=np.float32)
        write_frame(writer, FrameKind.EMBED, samples.tobytes())
        kind, payload = read_frame(reader)
        assert kind is FrameKind.RESULT
        assert len(payload) == 192 * 4

        write_frame(writer, FrameKind.SHUTDOWN)
        writer.close()
        thread.join(timeout=5)
        assert result["code"] == 0

    def test_a_malformed_clip_reports_an_error_frame_without_killing_the_driver(self) -> None:
        import os

        r_to_driver, w_to_driver = os.pipe()
        r_from_driver, w_from_driver = os.pipe()

        stdin = os.fdopen(r_to_driver, "rb")
        stdout = os.fdopen(w_from_driver, "wb")

        result: dict[str, Any] = {}

        def _run_driver() -> None:
            result["code"] = run(stdin, stdout, "unused.onnx", sys.executable, 600, FAKE_NPU_MODULE)

        thread = threading.Thread(target=_run_driver, daemon=True)
        thread.start()

        writer = os.fdopen(w_to_driver, "wb")
        reader = os.fdopen(r_from_driver, "rb")

        kind, _ = read_frame(reader)
        assert kind is FrameKind.READY

        # Too short to embed (CAM++'s MIN_SAMPLES floor is 720 samples).
        too_short = np.zeros(10, dtype=np.float32)
        write_frame(writer, FrameKind.EMBED, too_short.tobytes())
        kind, payload = read_frame(reader)
        assert kind is FrameKind.ERROR
        assert "audio too short" in payload.decode("utf-8")

        # The driver must still be alive and able to serve a good clip after.
        samples = np.linspace(-0.1, 0.1, 32_000, dtype=np.float32)
        write_frame(writer, FrameKind.EMBED, samples.tobytes())
        kind, payload = read_frame(reader)
        assert kind is FrameKind.RESULT

        write_frame(writer, FrameKind.SHUTDOWN)
        writer.close()
        thread.join(timeout=5)
        assert result["code"] == 0
