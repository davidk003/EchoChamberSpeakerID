"""Tests for echochamber.audio.ringbuffer, written against the step-1 API contract.

The ring is single-producer / single-consumer with a doubled-write layout: the
backing store is 2 * capacity and every block is written twice, so a read of
n <= capacity frames is always one contiguous slice.  These tests hammer that
invariant across the wraparound seam, then prove it end-to-end with a threaded
producer/consumer stress test over a monotonic sample-index ramp.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from echochamber.audio.ringbuffer import OverrunError, RingBuffer


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def ramp(start: int, n: int, dtype: type = np.float32) -> np.ndarray:
    """samples[i] == start + i -- the canonical test signal."""
    return np.arange(start, start + n, dtype=dtype)


def assert_ramp(actual: np.ndarray, start: int, n: int, what: str) -> None:
    expected = ramp(start, n)
    assert actual.shape == (n,), f"{what}: expected shape ({n},), got {actual.shape}"
    assert np.array_equal(actual, expected), (
        f"{what}: read of {n} frames at {start} did not match the sample-index ramp; "
        f"first mismatch at index {int(np.argmax(actual != expected))} "
        f"(got {actual[actual != expected][:4]!r})"
    )


# --------------------------------------------------------------------------
# construction / properties
# --------------------------------------------------------------------------

@pytest.mark.parametrize("capacity", [1, 16, 160, 4096, 160_000])
def test_capacity_property(capacity: int) -> None:
    rb = RingBuffer(capacity)
    assert rb.capacity == capacity, "capacity must report the constructed capacity"


def test_initial_state_is_empty() -> None:
    rb = RingBuffer(64)
    assert rb.write_frames == 0, "a fresh ring has written nothing"
    assert rb.oldest_frame == 0, "a fresh ring's oldest frame is 0"
    assert rb.available_from(0) == 0


def test_default_dtype_is_float32() -> None:
    rb = RingBuffer(32)
    rb.write(ramp(0, 8, dtype=np.float64))
    out = rb.read(0, 8)
    assert out.dtype == np.float32, "the default ring dtype must be float32"


def test_custom_dtype_is_honoured() -> None:
    rb = RingBuffer(32, dtype=np.int16)
    rb.write(np.arange(8, dtype=np.int16))
    out = rb.read(0, 8)
    assert out.dtype == np.int16
    assert np.array_equal(out, np.arange(8, dtype=np.int16))


def test_write_casts_to_buffer_dtype() -> None:
    rb = RingBuffer(32)
    rb.write(np.arange(8, dtype=np.float64))
    out = rb.read(0, 8)
    assert out.dtype == np.float32
    assert np.array_equal(out, np.arange(8, dtype=np.float32))


# --------------------------------------------------------------------------
# basic write / read roundtrip
# --------------------------------------------------------------------------

def test_roundtrip_exact_values() -> None:
    rb = RingBuffer(64)
    data = np.array([0.0, -1.0, 0.5, 0.25, -0.125], dtype=np.float32)
    rb.write(data)

    out = rb.read(0, 5)
    assert np.array_equal(out, data), "read must return exactly what was written"


def test_roundtrip_ramp() -> None:
    rb = RingBuffer(128)
    rb.write(ramp(0, 100))
    assert_ramp(rb.read(0, 100), 0, 100, "full roundtrip")


@pytest.mark.parametrize(("start", "n"), [(0, 1), (0, 10), (5, 10), (17, 3), (99, 1), (0, 100)])
def test_partial_reads(start: int, n: int) -> None:
    rb = RingBuffer(128)
    rb.write(ramp(0, 100))
    assert_ramp(rb.read(start, n), start, n, "partial read")


def test_read_of_zero_frames_is_empty() -> None:
    rb = RingBuffer(64)
    rb.write(ramp(0, 10))
    out = rb.read(4, 0)
    assert out.shape == (0,), "a zero-length read must return an empty array"


def test_empty_write_is_a_noop() -> None:
    rb = RingBuffer(64)
    rb.write(ramp(0, 10))
    rb.write(np.empty(0, dtype=np.float32))
    assert rb.write_frames == 10, "writing zero frames must not advance write_frames"


def test_read_returns_a_view_not_a_copy() -> None:
    """Contract: read() hands back a view into the backing store (zero copy)."""
    rb = RingBuffer(64)
    rb.write(ramp(0, 32))
    out = rb.read(4, 8)
    assert out.base is not None, "read() must return a view into the ring, not a fresh array"


# --------------------------------------------------------------------------
# write_frames / oldest_frame / available_from
# --------------------------------------------------------------------------

def test_write_frames_is_monotonic_across_writes() -> None:
    rb = RingBuffer(32)
    total = 0
    for m in (5, 7, 11, 13, 3, 29, 31):
        rb.write(ramp(total, m))
        total += m
        assert rb.write_frames == total, "write_frames counts every frame ever written"


@pytest.mark.parametrize(
    ("capacity", "written", "expected_oldest"),
    [
        (16, 0, 0),
        (16, 1, 0),
        (16, 15, 0),
        (16, 16, 0),
        (16, 17, 1),
        (16, 100, 84),
        (64, 200, 136),
    ],
)
def test_oldest_frame(capacity: int, written: int, expected_oldest: int) -> None:
    rb = RingBuffer(capacity)
    if written:
        # write in capacity-sized bites so no single write exceeds capacity
        done = 0
        while done < written:
            m = min(capacity, written - done)
            rb.write(ramp(done, m))
            done += m
    assert rb.oldest_frame == expected_oldest, (
        "oldest_frame must be max(0, write_frames - capacity)"
    )


@pytest.mark.parametrize(
    ("start_frame", "expected"),
    [(0, 50), (10, 40), (49, 1), (50, 0), (60, -10)],
)
def test_available_from(start_frame: int, expected: int) -> None:
    rb = RingBuffer(128)
    rb.write(ramp(0, 50))
    assert rb.available_from(start_frame) == expected, (
        "available_from must be write_frames - start_frame"
    )


def test_available_from_tracks_writes() -> None:
    rb = RingBuffer(128)
    assert rb.available_from(0) == 0
    rb.write(ramp(0, 7))
    assert rb.available_from(0) == 7
    assert rb.available_from(7) == 0
    rb.write(ramp(7, 13))
    assert rb.available_from(0) == 20
    assert rb.available_from(7) == 13


# --------------------------------------------------------------------------
# wraparound seam -- the doubled-write invariant
# --------------------------------------------------------------------------

def test_read_spanning_the_seam_is_contiguous_and_correct() -> None:
    """capacity 16: after 19 frames the seam sits between frame 15 and 16."""
    rb = RingBuffer(16)
    rb.write(ramp(0, 10))
    rb.write(ramp(10, 9))

    assert rb.write_frames == 19
    assert rb.oldest_frame == 3

    assert_ramp(rb.read(12, 7), 12, 7, "read starting before the seam, ending after")
    assert_ramp(rb.read(9, 10), 9, 10, "read straddling the seam")
    assert_ramp(rb.read(15, 4), 15, 4, "read starting on the last pre-seam frame")
    assert_ramp(rb.read(16, 3), 16, 3, "read starting exactly on the seam")


@pytest.mark.parametrize("capacity", [16, 17, 64, 100])
@pytest.mark.parametrize("block_sizes", [(7, 13, 5, 11, 3), (1, 2, 3, 5, 8, 13), (31, 29, 23)])
def test_odd_block_writes_past_capacity_keep_reads_correct(
    capacity: int, block_sizes: tuple[int, ...]
) -> None:
    """Write odd-sized blocks well past capacity; after each one, read back the
    entire valid region and assert it is the exact sample-index ramp.  Odd sizes
    guarantee the seam lands mid-read on most iterations."""
    rb = RingBuffer(capacity)
    written = 0
    sizes = [s for s in block_sizes if s <= capacity]
    if not sizes:
        pytest.skip(f"no block in {block_sizes} fits a capacity of {capacity}")

    for i in range(40):
        m = sizes[i % len(sizes)]
        rb.write(ramp(written, m))
        written += m

        assert rb.write_frames == written
        oldest = max(0, written - capacity)
        assert rb.oldest_frame == oldest

        n_valid = written - oldest
        assert_ramp(rb.read(oldest, n_valid), oldest, n_valid,
                    f"capacity={capacity} after {written} frames")

        # ...and a shorter read anchored mid-region, which lands across the seam
        if n_valid >= 3:
            mid = oldest + n_valid // 3
            assert_ramp(rb.read(mid, written - mid), mid, written - mid,
                        f"capacity={capacity} mid-region read after {written} frames")


def test_read_of_exactly_capacity_at_the_newest_valid_position() -> None:
    capacity = 64
    rb = RingBuffer(capacity)
    written = 0
    for m in (37, 41, 43, 47, 53):
        rb.write(ramp(written, m))
        written += m

    start = rb.write_frames - capacity
    assert start == rb.oldest_frame, "the newest valid full-capacity read starts at oldest_frame"

    out = rb.read(start, capacity)
    assert_ramp(out, start, capacity, "full-capacity read at the newest valid position")


def test_full_capacity_read_after_an_exact_wrap() -> None:
    """write_frames a multiple of capacity puts the write cursor at logical 0."""
    capacity = 32
    rb = RingBuffer(capacity)
    for k in range(4):
        rb.write(ramp(k * capacity, capacity))

    assert rb.write_frames == 4 * capacity
    assert rb.write_frames % capacity == 0
    assert_ramp(rb.read(3 * capacity, capacity), 3 * capacity, capacity,
                "full-capacity read at an exact wrap boundary")


def test_write_of_exactly_capacity_is_allowed() -> None:
    rb = RingBuffer(32)
    rb.write(ramp(0, 32))
    assert_ramp(rb.read(0, 32), 0, 32, "a write of exactly capacity frames")


def test_single_frame_capacity_ring() -> None:
    rb = RingBuffer(1)
    for i in range(5):
        rb.write(ramp(i, 1))
        assert rb.write_frames == i + 1
        assert rb.oldest_frame == i
        assert_ramp(rb.read(i, 1), i, 1, "capacity-1 ring")


# --------------------------------------------------------------------------
# overrun detection
# --------------------------------------------------------------------------

def test_overrun_when_start_frame_is_older_than_oldest_frame() -> None:
    capacity = 16
    rb = RingBuffer(capacity)
    for k in range(4):
        rb.write(ramp(k * 10, 10))

    assert rb.write_frames == 40
    assert rb.oldest_frame == 24

    with pytest.raises(OverrunError):
        rb.read(10, 5)


@pytest.mark.parametrize("start_frame", [0, 1, 10, 23])
def test_overrun_for_every_lapped_start(start_frame: int) -> None:
    capacity = 16
    rb = RingBuffer(capacity)
    for k in range(4):
        rb.write(ramp(k * 10, 10))
    assert rb.oldest_frame == 24

    with pytest.raises(OverrunError):
        rb.read(start_frame, 4)


def test_no_overrun_exactly_at_oldest_frame() -> None:
    capacity = 16
    rb = RingBuffer(capacity)
    for k in range(4):
        rb.write(ramp(k * 10, 10))

    assert_ramp(rb.read(24, 4), 24, 4, "read starting exactly at oldest_frame")


def test_overrun_error_is_a_runtimeerror() -> None:
    assert issubclass(OverrunError, RuntimeError), "OverrunError must subclass RuntimeError"


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------

def test_write_rejects_2d_input() -> None:
    rb = RingBuffer(64)
    with pytest.raises(ValueError):
        rb.write(np.zeros((8, 2), dtype=np.float32))


def test_write_rejects_0d_input() -> None:
    rb = RingBuffer(64)
    with pytest.raises(ValueError):
        rb.write(np.zeros((), dtype=np.float32))


@pytest.mark.parametrize("n", [65, 100, 10_000])
def test_write_rejects_more_than_capacity(n: int) -> None:
    rb = RingBuffer(64)
    with pytest.raises(ValueError):
        rb.write(ramp(0, n))


def test_failed_write_does_not_advance_write_frames() -> None:
    rb = RingBuffer(64)
    rb.write(ramp(0, 10))
    with pytest.raises(ValueError):
        rb.write(ramp(10, 65))
    assert rb.write_frames == 10, "a rejected write must leave write_frames untouched"


@pytest.mark.parametrize(
    ("start_frame", "n", "why"),
    [
        (0, 65, "n > capacity"),
        (0, 1_000, "n far beyond capacity"),
        (0, -1, "negative n"),
        (0, -100, "negative n"),
        (-1, 4, "negative start_frame"),
        (-100, 4, "negative start_frame"),
        (0, 51, "start_frame + n > write_frames"),
        (49, 2, "start_frame + n > write_frames"),
        (50, 1, "reading past the write head"),
        (60, 1, "start_frame beyond the write head"),
    ],
)
def test_read_rejects_bad_arguments(start_frame: int, n: int, why: str) -> None:
    rb = RingBuffer(64)
    rb.write(ramp(0, 50))
    with pytest.raises(ValueError):
        rb.read(start_frame, n)


def test_read_rejects_n_greater_than_capacity_even_when_data_exists() -> None:
    capacity = 16
    rb = RingBuffer(capacity)
    for k in range(4):
        rb.write(ramp(k * 10, 10))
    assert rb.write_frames == 40

    with pytest.raises(ValueError):
        rb.read(24, capacity + 1)


def test_read_on_empty_ring_raises() -> None:
    rb = RingBuffer(64)
    with pytest.raises(ValueError):
        rb.read(0, 1)


# --------------------------------------------------------------------------
# wait_for
# --------------------------------------------------------------------------

def test_wait_for_already_satisfied_returns_true_immediately() -> None:
    rb = RingBuffer(64)
    rb.write(ramp(0, 10))
    t0 = time.monotonic()
    assert rb.wait_for(10, timeout=5.0) is True
    assert rb.wait_for(1, timeout=5.0) is True
    assert time.monotonic() - t0 < 1.0, "an already-satisfied wait must not block"


def test_wait_for_returns_false_on_timeout() -> None:
    rb = RingBuffer(64)
    rb.write(ramp(0, 10))
    t0 = time.monotonic()
    result = rb.wait_for(11, timeout=0.05)
    elapsed = time.monotonic() - t0

    assert result is False, "wait_for must return False when the timeout expires"
    assert elapsed >= 0.04, "wait_for must actually wait for roughly the timeout"
    assert elapsed < 3.0, "wait_for must not wait far beyond the timeout"


def test_wait_for_returns_false_after_close() -> None:
    rb = RingBuffer(64)
    rb.close()
    t0 = time.monotonic()
    assert rb.wait_for(1_000, timeout=5.0) is False, "a closed ring must not block waiters"
    assert time.monotonic() - t0 < 2.0, "close() must make wait_for return promptly"


def test_wait_for_satisfied_data_wins_over_closed() -> None:
    """Contract ordering: check write_frames >= end_frame BEFORE the closed flag."""
    rb = RingBuffer(64)
    rb.write(ramp(0, 10))
    rb.close()
    assert rb.wait_for(10, timeout=1.0) is True


def test_close_wakes_a_blocked_waiter() -> None:
    rb = RingBuffer(64)
    result: list[bool] = []

    def waiter() -> None:
        result.append(rb.wait_for(1_000, timeout=10.0))

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.02)
    rb.close()
    t.join(timeout=5.0)

    assert not t.is_alive(), "close() must wake a blocked waiter"
    assert result == [False], "a waiter woken by close() must return False"


def test_close_is_idempotent() -> None:
    rb = RingBuffer(64)
    rb.close()
    rb.close()
    assert rb.wait_for(1, timeout=1.0) is False


def test_wait_for_returns_true_when_a_producer_thread_writes() -> None:
    rb = RingBuffer(1024)
    started = threading.Event()

    def producer() -> None:
        started.wait(timeout=5.0)
        time.sleep(0.02)
        rb.write(ramp(0, 512))

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    started.set()

    assert rb.wait_for(512, timeout=10.0) is True, (
        "wait_for must return True once the producer supplies the data"
    )
    assert rb.write_frames >= 512
    assert_ramp(rb.read(0, 512), 0, 512, "data delivered via wait_for")
    t.join(timeout=5.0)


def test_wait_for_with_no_timeout_blocks_until_data_arrives() -> None:
    """timeout=None means block indefinitely.  A watchdog closes the ring if the
    implementation is broken, so a failure reports rather than hangs."""
    rb = RingBuffer(1024)

    watchdog = threading.Timer(5.0, rb.close)
    watchdog.daemon = True
    watchdog.start()

    producer = threading.Thread(target=lambda: (time.sleep(0.02), rb.write(ramp(0, 256))),
                                daemon=True)
    producer.start()
    try:
        assert rb.wait_for(256, timeout=None) is True, (
            "wait_for(timeout=None) must block until write_frames reaches end_frame"
        )
    finally:
        watchdog.cancel()
    producer.join(timeout=5.0)


@pytest.mark.parametrize("iteration", range(50))
def test_wait_for_does_not_miss_a_wakeup_under_a_tight_race(iteration: int) -> None:
    """The producer's write and the consumer's wait are released simultaneously by
    a barrier, so the write frequently lands inside the clear/re-check window.
    A missed wakeup shows up as a timeout here."""
    rb = RingBuffer(256)
    gate = threading.Barrier(2, timeout=10.0)
    n = 64

    def producer() -> None:
        gate.wait()
        rb.write(ramp(0, n))

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    gate.wait()
    result = rb.wait_for(n, timeout=5.0)
    t.join(timeout=5.0)

    assert result is True, (
        f"iteration {iteration}: wait_for missed a wakeup -- the write landed inside "
        "the event.clear() window and was not re-checked"
    )
    assert rb.write_frames == n
    assert_ramp(rb.read(0, n), 0, n, f"iteration {iteration}")


def test_wait_for_wakes_only_when_the_threshold_is_reached() -> None:
    """Intermediate writes must not satisfy a wait for a larger end_frame."""
    rb = RingBuffer(1024)

    def producer() -> None:
        pos = 0
        for m in (10, 20, 30, 40):
            time.sleep(0.005)
            rb.write(ramp(pos, m))
            pos += m

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    assert rb.wait_for(100, timeout=10.0) is True
    assert rb.write_frames >= 100, "wait_for returned before end_frame was reached"
    t.join(timeout=5.0)


# --------------------------------------------------------------------------
# SPSC stress -- the test that proves the design
# --------------------------------------------------------------------------

def test_spsc_ramp_stress() -> None:
    """One producer thread writes a monotonic sample-index ramp in odd-sized
    blocks; one consumer thread reads overlapping windows and asserts every
    sample equals its absolute frame index.

    The producer is flow-controlled against the consumer's low-water mark so it
    can never lap unread data -- that keeps the test deterministic (no spurious
    overruns) while still forcing thousands of wraps through the seam.
    """
    capacity = 4096
    total = 40_000
    window = 1024
    hop = 384
    block_sizes = (97, 31, 251, 7, 163, 59, 401, 13)

    assert window + hop + max(block_sizes) <= capacity, "test configuration error"

    rb = RingBuffer(capacity)
    low = [0]                      # frames below this are no longer needed by the consumer
    errors: list[str] = []

    def producer() -> None:
        try:
            pos = 0
            i = 0
            while pos < total:
                m = min(block_sizes[i % len(block_sizes)], total - pos)
                i += 1
                deadline = time.monotonic() + 30.0
                while (pos + m) - low[0] > capacity:
                    if time.monotonic() > deadline:
                        errors.append("producer stalled waiting for the consumer")
                        return
                    time.sleep(0.001)
                rb.write(ramp(pos, m))
                pos += m
        except Exception as exc:                      # pragma: no cover - surfaced below
            errors.append(f"producer raised {exc!r}")
        finally:
            rb.close()

    n_windows = (total - window) // hop + 1
    t = threading.Thread(target=producer, daemon=True)
    t.start()
    try:
        start = 0
        for k in range(n_windows):
            assert rb.wait_for(start + window, timeout=30.0) is True, (
                f"window {k}: wait_for({start + window}) timed out "
                f"(write_frames={rb.write_frames}, errors={errors})"
            )
            view = rb.read(start, window)
            got = np.array(view, copy=True)

            assert_ramp(got, start, window, f"SPSC window {k}")

            low[0] = start + hop
            start += hop
    finally:
        low[0] = total             # release the producer no matter how we exit
        t.join(timeout=30.0)

    assert not errors, f"producer errors: {errors}"
    assert not t.is_alive(), "producer thread did not finish"
    assert rb.write_frames == total, "the producer must have written every frame"


def test_spsc_overrun_is_detected_when_the_consumer_falls_behind() -> None:
    """Without flow control a slow consumer must get an OverrunError, not silent
    corruption."""
    capacity = 256
    rb = RingBuffer(capacity)

    pos = 0
    for _ in range(20):
        rb.write(ramp(pos, 100))
        pos += 100

    assert rb.write_frames == 2000
    assert rb.oldest_frame == 2000 - capacity

    with pytest.raises(OverrunError):
        rb.read(0, 100)          # the consumer never moved off frame 0


# ==========================================================================
# read_copy: the seqlock read
#
# read() validates and then returns a view, so its validation is already stale
# by the time the caller copies. A writer lapping the ring during that copy
# produced the wrong audio under a start_frame that lied about it, with nothing
# raised anywhere. Silently misaligned audio is worse than missing audio because
# nothing downstream can detect it.
#
# Every test below writes in the gap explicitly rather than racing for it, so
# these cannot become flaky-but-passing.
# ==========================================================================


def test_read_returns_a_view_that_a_lapping_writer_corrupts() -> None:
    """Documents exactly why read_copy exists. read() alone is not safe to keep."""
    ring = RingBuffer(1000)
    ring.write(np.arange(0, 500, dtype=np.float32))

    view = ring.read(0, 500)
    assert view[0] == 0.0

    for _ in range(3):
        ring.write(np.full(1000, -999.0, dtype=np.float32))

    assert view[0] == -999.0, (
        "read() hands back a live view; this is the hazard read_copy closes"
    )


class _LappingRing(RingBuffer):
    """A ring whose writer always runs between validation and the copy.

    Subclassed rather than monkeypatched because RingBuffer uses __slots__, and
    this makes the race deterministic instead of something to spin for.
    """

    def read(self, start_frame: int, n: int) -> np.ndarray:
        view = super().read(start_frame, n)
        for _ in range(3):
            super().write(np.full(self.capacity, -999.0, dtype=np.float32))
        return view


def test_read_copy_raises_when_the_writer_laps_during_the_copy() -> None:
    ring = _LappingRing(1000)
    ring.write(np.arange(0, 500, dtype=np.float32))

    with pytest.raises(OverrunError):
        ring.read_copy(0, 500)


def test_read_copy_returns_an_owned_copy() -> None:
    ring = RingBuffer(1000)
    ring.write(np.arange(0, 500, dtype=np.float32))

    out = ring.read_copy(0, 500)
    assert out.base is None, "read_copy must return an owned array"

    ring.write(np.full(1000, -1.0, dtype=np.float32))
    assert out[0] == 0.0, "the copy must survive the writer lapping afterwards"


def test_read_copy_matches_read_when_nothing_laps() -> None:
    ring = RingBuffer(1000)
    ring.write(np.arange(0, 900, dtype=np.float32))
    assert np.array_equal(ring.read_copy(100, 500), np.arange(100, 600, dtype=np.float32))


def test_read_copy_still_rejects_already_lost_frames() -> None:
    ring = RingBuffer(100)
    for start in (0, 100, 200):  # write() rejects blocks larger than capacity
        ring.write(np.arange(start, start + 100, dtype=np.float32))
    with pytest.raises(OverrunError):
        ring.read_copy(0, 50)


def test_wait_for_none_timeout_returns_when_close_races_the_clear() -> None:
    """close() landing in the clear() gap must not strand an infinite waiter.

    The event is set by close() and then discarded by clear(); without a second
    _closed check the waiter blocks forever on an already-closed ring.
    """
    ring = RingBuffer(1000)
    original_clear = ring._event.clear  # type: ignore[attr-defined]
    fired = threading.Event()

    def clear_then_close() -> None:
        original_clear()
        if not fired.is_set():
            fired.set()
            ring.close()      # exactly the racing close, made deterministic

    ring._event.clear = clear_then_close  # type: ignore[attr-defined]

    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(ring.wait_for(10_000, timeout=None)),
        daemon=True,
    )
    worker.start()
    worker.join(timeout=5.0)

    assert not worker.is_alive(), (
        "wait_for(timeout=None) hung after a close() racing the event clear"
    )
    assert result == [False]
