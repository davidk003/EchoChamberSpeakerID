"""Tests for LatencyTracker, LatencySummary and AudioChunk.age_s.

The tracker takes observations rather than reading a clock, so every assertion
here is exact -- no sleeping, no tolerance windows, no flakiness.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from echochamber.audio.latency import LatencySummary, LatencyTracker
from echochamber.audio.types import AudioChunk


def make_chunk(capture_time: float = 0.0, n: int = 8) -> AudioChunk:
    return AudioChunk(
        samples=np.zeros(n, dtype=np.float32),
        start_frame=0,
        seq=0,
        sample_rate=16000,
        capture_time=capture_time,
    )


# --------------------------------------------------------------------------
# AudioChunk.age_s
# --------------------------------------------------------------------------

def test_age_is_the_difference_from_capture_time() -> None:
    chunk = make_chunk(capture_time=100.0)
    assert chunk.age_s(100.25) == pytest.approx(0.25)


def test_age_is_zero_when_capture_time_was_never_set() -> None:
    """Chunks built by hand in tests carry 0.0 and must not report a huge age."""
    assert make_chunk(capture_time=0.0).age_s(12345.0) == 0.0


def test_age_never_goes_negative() -> None:
    """Clamped, so a clock oddity cannot produce a negative latency."""
    assert make_chunk(capture_time=100.0).age_s(99.0) == 0.0


def test_capture_time_defaults_to_zero() -> None:
    chunk = AudioChunk(np.zeros(4, np.float32), 0, 0, 16000)
    assert chunk.capture_time == 0.0


# --------------------------------------------------------------------------
# LatencySummary
# --------------------------------------------------------------------------

def test_empty_summary_is_flagged_empty() -> None:
    summary = LatencySummary()
    assert summary.is_empty is True
    assert summary.count == 0
    assert summary.p95_ms == 0.0


def test_summary_is_frozen() -> None:
    with pytest.raises(Exception):
        LatencySummary().count = 5  # type: ignore[misc]


# --------------------------------------------------------------------------
# LatencyTracker
# --------------------------------------------------------------------------

def test_new_tracker_summarises_as_empty() -> None:
    assert LatencyTracker().summary().is_empty is True


def test_records_are_converted_from_seconds_to_milliseconds() -> None:
    t = LatencyTracker()
    t.record(0.025)
    summary = t.summary()
    assert summary.count == 1
    assert summary.p50_ms == pytest.approx(25.0)
    assert summary.max_ms == pytest.approx(25.0)


def test_percentiles_over_a_known_distribution() -> None:
    """1..100 ms: nearest-rank percentiles are exactly the ranked values."""
    t = LatencyTracker()
    t.record_many([n / 1000.0 for n in range(1, 101)])
    s = t.summary()

    assert s.count == 100
    assert s.mean_ms == pytest.approx(50.5)
    assert s.p50_ms == pytest.approx(50.0)
    assert s.p95_ms == pytest.approx(95.0)
    assert s.p99_ms == pytest.approx(99.0)
    assert s.max_ms == pytest.approx(100.0)


def test_percentiles_are_real_observations_not_interpolated() -> None:
    """Nearest-rank: every reported figure actually happened."""
    t = LatencyTracker()
    t.record_many([0.010, 0.010, 0.010, 0.500])
    s = t.summary()
    assert s.p50_ms in (10.0, 500.0)
    assert s.max_ms == pytest.approx(500.0)


def test_a_single_spike_dominates_p95_but_not_the_median() -> None:
    """The reason percentiles are shown instead of a mean."""
    t = LatencyTracker()
    t.record_many([0.010] * 99 + [2.0])
    s = t.summary()
    assert s.p50_ms == pytest.approx(10.0)
    assert s.max_ms == pytest.approx(2000.0)
    assert s.mean_ms > s.p50_ms, "the mean is dragged up; the median is not"


def test_window_bounds_memory_and_ages_out_old_samples() -> None:
    t = LatencyTracker(window=10)
    t.record_many([1.0] * 10)      # 1000 ms each
    t.record_many([0.001] * 10)    # then 1 ms each

    s = t.summary()
    assert s.count == 10, "only the window is retained"
    assert s.max_ms == pytest.approx(1.0), "the old slow samples aged out"


def test_worst_ever_survives_the_rolling_window() -> None:
    """A spike that scrolled out still happened, and is worth surfacing."""
    t = LatencyTracker(window=5)
    t.record(3.0)
    t.record_many([0.001] * 10)

    assert t.summary().max_ms == pytest.approx(1.0)
    assert t.worst_ever_ms == pytest.approx(3000.0)


def test_total_observations_counts_everything_ever_recorded() -> None:
    t = LatencyTracker(window=4)
    t.record_many([0.01] * 20)
    assert t.total_observations == 20
    assert t.summary().count == 4


def test_negative_latency_is_clamped_not_rejected() -> None:
    t = LatencyTracker()
    t.record(-5.0)
    assert t.summary().max_ms == 0.0
    assert t.summary().count == 1


def test_reset_clears_everything_including_worst_ever() -> None:
    t = LatencyTracker()
    t.record_many([0.5, 0.25])
    t.reset()
    assert t.summary().is_empty is True
    assert t.worst_ever_ms == 0.0
    assert t.total_observations == 0


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LatencyTracker(window=0)


def test_repr_names_the_percentiles() -> None:
    t = LatencyTracker()
    t.record(0.042)
    assert "p95" in repr(t)


def test_concurrent_recording_loses_nothing() -> None:
    """The consumer thread records while the GUI thread summarises."""
    t = LatencyTracker(window=10_000)
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            for _ in range(500):
                t.record(0.01)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(500):
                t.summary()
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=20)

    assert not errors, f"threading errors: {errors}"
    assert t.total_observations == 2000, "every observation must be counted"


# --------------------------------------------------------------------------
# clock choice
#
# On Windows time.monotonic() is GetTickCount64 with 15.625 ms resolution. Using
# it for capture_time quantises a sub-millisecond consumer handoff into
# alternating 0 ms and 16 ms readings -- noise indistinguishable from real
# latency spikes. perf_counter is QueryPerformanceCounter (~100 ns) and is
# equally monotonic, so there is no reason to use anything else here.
# --------------------------------------------------------------------------


def test_perf_counter_is_fine_enough_to_measure_this_pipeline() -> None:
    import time as _time

    perf = _time.get_clock_info("perf_counter")
    assert perf.monotonic, "capture_time must come from a monotonic clock"
    assert perf.resolution <= 1e-5, (
        f"perf_counter resolution {perf.resolution * 1000:.4f} ms is too coarse "
        "to measure a sub-millisecond consumer handoff"
    )


def test_chunker_stamps_capture_time_from_perf_counter() -> None:
    """A chunk's capture_time must be comparable to perf_counter(), not monotonic.

    The two clocks have unrelated origins, so mixing them yields nonsense rather
    than a merely imprecise number. Comparing the stamp against both is the
    cheapest way to pin which clock produced it.
    """
    import time as _time

    from echochamber.audio.chunker import WindowChunker
    from echochamber.audio.ringbuffer import RingBuffer
    from echochamber.config import AudioConfig

    cfg = AudioConfig(sample_rate=1000, window_ms=10, hop_ms=10, ring_seconds=1.0)
    ring = RingBuffer(cfg.ring_frames)
    got: list[AudioChunk] = []
    chunker = WindowChunker(ring, cfg, got.append)
    chunker.start()
    try:
        ring.write(np.zeros(cfg.window_frames, dtype=np.float32))
        deadline = _time.perf_counter() + 5.0
        while not got and _time.perf_counter() < deadline:
            _time.sleep(0.005)
    finally:
        ring.close()
        chunker.stop(timeout=5.0)

    assert got, "the chunker never emitted a window"
    stamp = got[0].capture_time
    assert stamp > 0.0, "capture_time must be stamped"
    assert abs(_time.perf_counter() - stamp) < 5.0, (
        f"capture_time={stamp} is not on the perf_counter timeline "
        f"(perf_counter now={_time.perf_counter()}, "
        f"monotonic now={_time.monotonic()})"
    )
