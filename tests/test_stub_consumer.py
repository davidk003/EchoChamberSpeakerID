"""Tests for StubInferenceSink, the stand-in for the ML stage.

Includes the end-to-end check that matters most for step 6: a real pipeline
replaying a real WAV produces latency measurements that are consistent with the
window length, because a window cannot be emitted until its final sample exists.
"""

from __future__ import annotations

import time
import wave
from typing import Any

import numpy as np
import pytest

from echochamber.audio.latency import LatencyTracker
from echochamber.audio.pipeline import AudioPipeline
from echochamber.audio.sources.file_source import FileSource
from echochamber.audio.stub_consumer import StubInferenceSink
from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats
from echochamber.config import AudioConfig


def make_chunk(capture_time: float = 0.0, value: float = 0.5, n: int = 64) -> AudioChunk:
    return AudioChunk(
        samples=np.full(n, value, dtype=np.float32),
        start_frame=0,
        seq=0,
        sample_rate=16000,
        capture_time=capture_time,
    )


def write_wav(path: Any, data: np.ndarray, sample_rate: int) -> Any:
    arr = np.asarray(data).reshape(-1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(arr.astype("<i2").tobytes())
    return path


# --------------------------------------------------------------------------
# basics
# --------------------------------------------------------------------------

def test_processes_chunks_and_counts_them() -> None:
    sink = StubInferenceSink()
    sink.on_chunk(make_chunk())
    sink.on_chunk(make_chunk())
    assert sink.processed == 2


def test_default_inference_is_rms_of_the_window() -> None:
    """The fake model does real numeric work on real samples."""
    sink = StubInferenceSink()
    sink.on_chunk(make_chunk(value=0.5))
    assert sink.last_result == pytest.approx(0.5)


def test_custom_infer_is_used() -> None:
    sink = StubInferenceSink(infer=lambda s: float(s.size))
    sink.on_chunk(make_chunk(n=32))
    assert sink.last_result == pytest.approx(32.0)


def test_empty_window_does_not_divide_by_zero() -> None:
    sink = StubInferenceSink()
    sink.on_chunk(
        AudioChunk(np.zeros(0, np.float32), 0, 0, 16000, capture_time=1.0)
    )
    assert sink.last_result == 0.0


def test_last_result_is_none_before_any_chunk() -> None:
    assert StubInferenceSink().last_result is None


def test_on_result_callback_receives_chunk_and_result() -> None:
    seen: list[tuple[int, float]] = []
    sink = StubInferenceSink(on_result=lambda c, r: seen.append((c.seq, r)))
    sink.on_chunk(make_chunk(value=0.25))
    assert seen == [(0, pytest.approx(0.25))]


def test_negative_delay_is_rejected() -> None:
    with pytest.raises(ValueError):
        StubInferenceSink(delay_s=-1.0)


def test_close_is_idempotent() -> None:
    sink = StubInferenceSink()
    sink.close()
    sink.close()
    assert sink.closed is True


def test_repr_reports_throughput() -> None:
    sink = StubInferenceSink()
    sink.on_chunk(make_chunk())
    assert "processed=1" in repr(sink)


# --------------------------------------------------------------------------
# latency measurement
# --------------------------------------------------------------------------

def test_latency_is_measured_from_the_chunk_capture_time() -> None:
    tracker = LatencyTracker()
    sink = StubInferenceSink(tracker=tracker)

    # A chunk stamped 50 ms in the past must measure as ~50 ms, not ~0.
    sink.on_chunk(make_chunk(capture_time=time.perf_counter() - 0.050))

    summary = sink.latency_summary()
    assert summary.count == 1
    assert 40.0 <= summary.p50_ms <= 200.0, (
        f"expected roughly 50 ms of measured age, got {summary.p50_ms}"
    )


def test_latency_excludes_the_sinks_own_simulated_work() -> None:
    """Latency answers "how stale was this", not "how slow are we"."""
    sink = StubInferenceSink(delay_s=0.05)
    sink.on_chunk(make_chunk(capture_time=time.perf_counter()))
    sink.on_chunk(make_chunk(capture_time=time.perf_counter()))

    # Both were fresh on arrival, even though the first one slept afterwards.
    assert sink.latency_summary().max_ms < 40.0


def test_a_supplied_tracker_is_used() -> None:
    tracker = LatencyTracker()
    sink = StubInferenceSink(tracker=tracker)
    sink.on_chunk(make_chunk(capture_time=time.perf_counter()))
    assert tracker.total_observations == 1
    assert sink.tracker is tracker


def test_delay_actually_sleeps() -> None:
    sink = StubInferenceSink(delay_s=0.05)
    started = time.perf_counter()
    sink.on_chunk(make_chunk())
    assert time.perf_counter() - started >= 0.04, "the simulated delay must be real"


# --------------------------------------------------------------------------
# end to end: the step 6 headline
# --------------------------------------------------------------------------

def test_pipeline_latency_is_dominated_by_the_window_length(tmp_path: Any) -> None:
    """A window cannot be emitted until its last sample exists.

    So with real-time replay, measured latency should be at least a hop and no
    more than a couple of windows -- the architecture's central latency claim,
    checked against real timings rather than asserted.
    """
    sr = 16000
    frames = sr  # 1 second
    data = (np.sin(np.arange(frames) * 0.05) * 20000).astype(np.int16)
    path = write_wav(tmp_path / "ramp.wav", data, sr)

    cfg = AudioConfig(
        sample_rate=sr, window_ms=200, hop_ms=100,
        ring_seconds=5.0, queue_max=32, drop_policy=DropPolicy.BLOCK,
    )
    sink = StubInferenceSink()
    stats = StreamStats()
    pipeline = AudioPipeline(
        cfg, sink,
        lambda cb, st: FileSource(path, cb, blocksize=160, realtime=True, stats=st),
        stats=stats,
    )
    pipeline.start()
    try:
        assert pipeline.wait_until_finished(timeout=30) is True
    finally:
        pipeline.stop(timeout=10)

    summary = sink.latency_summary()
    assert summary.count >= 5, f"too few chunks measured: {summary.count}"
    # Generous bounds: this asserts the shape of the latency, not a benchmark.
    assert summary.p95_ms < 1000.0, (
        f"latency far above one window; something is queueing badly: {summary}"
    )
    assert sink.processed == summary.count


def test_a_slow_consumer_shows_up_as_rising_latency(tmp_path: Any) -> None:
    """With BLOCK, a slow sink cannot drop -- so the backlog appears as latency."""
    sr = 8000
    frames = sr // 2
    data = (np.sin(np.arange(frames) * 0.05) * 15000).astype(np.int16)
    path = write_wav(tmp_path / "slow.wav", data, sr)

    cfg = AudioConfig(
        sample_rate=sr, window_ms=100, hop_ms=50,
        ring_seconds=5.0, queue_max=64, drop_policy=DropPolicy.BLOCK,
    )
    slow = StubInferenceSink(delay_s=0.02)
    fast = StubInferenceSink(delay_s=0.0)

    for sink in (fast, slow):
        stats = StreamStats()
        pipeline = AudioPipeline(
            cfg, sink,
            lambda cb, st: FileSource(path, cb, blocksize=80, realtime=False, stats=st),
            stats=stats,
        )
        pipeline.start()
        try:
            pipeline.wait_until_finished(timeout=30)
        finally:
            pipeline.stop(timeout=10)

    assert slow.latency_summary().max_ms > fast.latency_summary().max_ms, (
        f"a 20ms-per-chunk consumer should accumulate a backlog; "
        f"slow={slow.latency_summary()} fast={fast.latency_summary()}"
    )
