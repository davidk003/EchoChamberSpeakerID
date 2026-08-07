"""Tests for echochamber.audio.chunker, written against the step-2 API contract.

These tests are written from the *spec*, not from the implementation.  They
drive a :class:`RingBuffer` directly from the test thread (no audio device) and
collect emitted chunks through the ``on_chunk`` callback.

The centrepiece is the **sample-index ramp**: the ring is fed ``samples[i] == i``
so every emitted chunk can be checked for absolute sample-exactness.  Chunk ``k``
must cover absolute frames ``[k*H, k*H + W)`` -- which catches off-by-one errors
that are completely invisible on real audio.

Determinism strategy: nothing here races the chunker.  Where the test needs the
chunker to be at a known point in its loop, it *gates the callback* (the chunker
is parked inside ``on_chunk`` and provably cannot advance ``next_start`` or
re-read the config), mutates the world, then releases.  Waits are event-driven
with multi-second timeouts, so a failure means a logic error, not scheduler
jitter.  Every chunker is stopped in a fixture teardown so a hung thread can
never wedge the suite.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterator

import numpy as np
import pytest

from echochamber.audio import chunker as chunker_module
from echochamber.audio.chunker import WindowChunker
from echochamber.audio.ringbuffer import RingBuffer
from echochamber.audio.types import AudioChunk, StreamStats
from echochamber.config import AudioConfig


# Generous: every wait below is event-driven, so these only bound failures.
TIMEOUT = 10.0

# How long to allow for the chunker to *not* do something (prove absence).
QUIET_S = 0.35


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def ramp(start: int, n: int, dtype: type = np.float32) -> np.ndarray:
    """samples[i] == start + i -- the canonical test signal."""
    return np.arange(start, start + n, dtype=dtype)


def make_config(
    window_frames: int,
    hop_frames: int,
    sample_rate: int = 1000,
    ring_seconds: float | None = None,
) -> AudioConfig:
    """Build an AudioConfig whose window/hop are exactly the requested frames.

    At sample_rate=1000, ms_to_frames(x, 1000) == x, so 1 ms == 1 frame and the
    frame geometry of each test is readable at a glance.
    """
    if ring_seconds is None:
        ring_seconds = max(1.0, 4.0 * (window_frames + hop_frames) / sample_rate)
    cfg = AudioConfig(
        sample_rate=sample_rate,
        window_ms=window_frames,
        hop_ms=hop_frames,
        ring_seconds=ring_seconds,
    )
    assert cfg.window_frames == window_frames, "test helper: window did not round to W"
    assert cfg.hop_frames == hop_frames, "test helper: hop did not round to H"
    return cfg


def write_ramp(ring: RingBuffer, start: int, n: int, block: int = 128) -> int:
    """Write ``n`` ramp frames starting at absolute frame ``start``.

    Splits into blocks so no single write ever exceeds the ring capacity.
    Returns the absolute frame index one past the last written frame.
    """
    block = max(1, min(block, ring.capacity))
    pos = start
    end = start + n
    while pos < end:
        m = min(block, end - pos)
        ring.write(ramp(pos, m))
        pos += m
    return end


def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Poll ``pred`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def assert_ramp_chunk(chunk: AudioChunk, start_frame: int, w: int, what: str) -> None:
    """Assert a chunk is exactly the ramp window [start_frame, start_frame + w)."""
    assert chunk.start_frame == start_frame, (
        f"{what}: start_frame {chunk.start_frame} != expected {start_frame}"
    )
    assert len(chunk.samples) == w, (
        f"{what}: len(samples) {len(chunk.samples)} != W {w}"
    )
    assert chunk.samples.ndim == 1, f"{what}: samples must be 1-D"
    expected = ramp(start_frame, w)
    assert np.array_equal(chunk.samples, expected), (
        f"{what}: samples are not the absolute sample-index ramp for "
        f"[{start_frame}, {start_frame + w}); first mismatch at index "
        f"{int(np.argmax(chunk.samples != expected))}"
    )


class Collector:
    """Thread-safe ``on_chunk`` callback that records every chunk it sees.

    ``hook(index, chunk)`` runs *outside* the lock after recording, so a hook may
    block (to park the chunker mid-loop) or raise (to test error handling)
    without deadlocking the test thread.
    """

    def __init__(self, hook: Callable[[int, AudioChunk], None] | None = None) -> None:
        self.chunks: list[AudioChunk] = []
        self._cv = threading.Condition()
        self._hook = hook

    def __call__(self, chunk: AudioChunk) -> None:
        with self._cv:
            self.chunks.append(chunk)
            index = len(self.chunks) - 1
            self._cv.notify_all()
        if self._hook is not None:
            self._hook(index, chunk)

    @property
    def count(self) -> int:
        with self._cv:
            return len(self.chunks)

    def snapshot(self) -> list[AudioChunk]:
        with self._cv:
            return list(self.chunks)

    def wait_for_count(self, n: int, timeout: float = TIMEOUT) -> bool:
        """Block until at least ``n`` chunks have been recorded."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while len(self.chunks) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return True


class Gate:
    """Hook that parks the chunker inside ``on_chunk`` for one chosen chunk.

    While the gate holds, the chunker provably has not advanced ``next_start``
    and has not re-read ``self._config`` -- which is what makes the reconfigure
    and overrun tests deterministic instead of racy.
    """

    def __init__(self, at: int = 0) -> None:
        self.at = at
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, index: int, chunk: AudioChunk) -> None:
        if index == self.at:
            self.entered.set()
            self.release.wait(timeout=30.0)

    def open(self) -> None:
        self.release.set()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def make_chunker() -> Iterator[Callable[..., WindowChunker]]:
    """Factory that registers every chunker for guaranteed teardown.

    Teardown releases any gates, closes any rings and stops every chunker, so a
    blocked or wedged worker thread can never hang the rest of the suite.
    """
    created: list[WindowChunker] = []
    rings: list[RingBuffer] = []
    gates: list[Gate] = []

    def _make(
        ring: RingBuffer,
        config: AudioConfig,
        on_chunk: Callable[[AudioChunk], None],
        stats: StreamStats | None = None,
        name: str = "chunker",
        gate: Gate | None = None,
        **kwargs: Any,
    ) -> WindowChunker:
        c = WindowChunker(ring, config, on_chunk, stats=stats, name=name, **kwargs)
        created.append(c)
        rings.append(ring)
        if gate is not None:
            gates.append(gate)
        return c

    yield _make

    for g in gates:
        g.open()
    for r in rings:
        try:
            r.close()
        except Exception:  # pragma: no cover - teardown must never mask failures
            pass
    for c in created:
        try:
            c.stop(timeout=5.0)
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------
# module surface
# --------------------------------------------------------------------------

def test_poll_interval_constant_exists_and_is_short() -> None:
    """POLL_INTERVAL_S bounds how long stop()/reconfigure() take to be noticed."""
    assert hasattr(chunker_module, "POLL_INTERVAL_S"), (
        "the module must expose POLL_INTERVAL_S"
    )
    poll = chunker_module.POLL_INTERVAL_S
    assert isinstance(poll, float)
    assert 0.0 < poll <= 0.25, (
        f"POLL_INTERVAL_S={poll} must be a short finite interval so stop() and "
        "reconfigure() are noticed promptly"
    )


def test_chunk_callback_alias_is_exported() -> None:
    assert hasattr(chunker_module, "ChunkCallback"), (
        "the module must export the ChunkCallback type alias"
    )


# --------------------------------------------------------------------------
# the ramp test -- sample-exact windowing
# --------------------------------------------------------------------------

RAMP_GEOMETRIES = [
    (100, 100),   # H == W: no overlap at all
    (100, 50),    # 50 % overlap
    (100, 10),    # heavy overlap, H << W
    (160, 20),    # heavy overlap, larger window
    (240, 80),    # the shipping 3000/1000 ratio, scaled down
    (64, 64),     # H == W again, non-round sizes
    (128, 1),     # pathological: one-frame hop
]


@pytest.mark.parametrize(("w", "h"), RAMP_GEOMETRIES)
def test_ramp_chunks_are_sample_exact(
    w: int, h: int, make_chunker: Callable[..., WindowChunker]
) -> None:
    """Chunk k must be exactly absolute frames [k*H, k*H + W) of the ramp.

    This is the assertion the whole step exists to satisfy.  The ring is sized
    larger than the entire test signal, so no overrun can occur and every chunk
    is fully determined.
    """
    n_chunks = 6
    total = w + h * (n_chunks - 1)
    ring = RingBuffer(total + w)          # oversized: no lapping possible
    cfg = make_config(w, h)
    collector = Collector()
    ch = make_chunker(ring, cfg, collector)

    ch.start()
    write_ramp(ring, 0, total)

    assert collector.wait_for_count(n_chunks), (
        f"W={w} H={h}: expected {n_chunks} chunks from {total} frames, got "
        f"{collector.count}"
    )
    chunks = collector.snapshot()[:n_chunks]

    for k, chunk in enumerate(chunks):
        assert_ramp_chunk(chunk, k * h, w, f"W={w} H={h} chunk {k}")
        assert chunk.samples[0] == k * h, (
            f"W={w} H={h} chunk {k}: first sample must equal k*H"
        )
        assert chunk.seq == k, f"W={w} H={h}: seq must equal the chunk index"
        assert chunk.sample_rate == cfg.sample_rate
        assert chunk.discontinuous is False, (
            f"W={w} H={h} chunk {k}: no overrun or reconfigure happened, so no "
            "chunk may be flagged discontinuous"
        )

    ring.close()
    assert ch.stop(timeout=5.0) is True


@pytest.mark.parametrize(("w", "h"), RAMP_GEOMETRIES)
def test_consecutive_chunks_overlap_by_exactly_w_minus_h_identical_samples(
    w: int, h: int, make_chunker: Callable[..., WindowChunker]
) -> None:
    """The tail of chunk k and the head of chunk k+1 are the same W-H samples."""
    n_chunks = 5
    total = w + h * (n_chunks - 1)
    ring = RingBuffer(total + w)
    cfg = make_config(w, h)
    collector = Collector()
    ch = make_chunker(ring, cfg, collector)

    ch.start()
    write_ramp(ring, 0, total)
    assert collector.wait_for_count(n_chunks), f"W={w} H={h}: too few chunks"
    chunks = collector.snapshot()[:n_chunks]

    overlap = w - h
    for k in range(n_chunks - 1):
        a, b = chunks[k], chunks[k + 1]
        assert b.start_frame - a.start_frame == h, (
            f"W={w} H={h}: consecutive start_frames must differ by exactly H"
        )
        if overlap == 0:
            assert h == w, "test configuration error"
            continue
        tail = a.samples[h:]
        head = b.samples[:overlap]
        assert len(tail) == overlap and len(head) == overlap, (
            f"W={w} H={h}: chunks must share exactly W-H = {overlap} samples"
        )
        assert np.array_equal(tail, head), (
            f"W={w} H={h}: the {overlap} overlapping samples of chunks {k}/{k + 1} "
            "must be element-wise identical -- they come from the same frames"
        )


def test_no_chunk_is_emitted_before_a_full_window_exists(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    w, h = 100, 50
    ring = RingBuffer(1000)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, w - 1)
    time.sleep(QUIET_S)
    assert collector.count == 0, "W-1 frames is not a full window; nothing may be emitted"

    write_ramp(ring, w - 1, 1)
    assert collector.wait_for_count(1), "the W-th frame must complete the first window"
    assert_ramp_chunk(collector.snapshot()[0], 0, w, "first window")


# --------------------------------------------------------------------------
# seq
# --------------------------------------------------------------------------

def test_seq_starts_at_zero_and_increments_by_one(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    w, h = 80, 20
    n = 8
    total = w + h * (n - 1)
    ring = RingBuffer(total + w)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, total)
    assert collector.wait_for_count(n), f"expected {n} chunks, got {collector.count}"

    seqs = [c.seq for c in collector.snapshot()[:n]]
    assert seqs == list(range(n)), f"seq must be 0,1,2,... got {seqs}"


def test_seq_never_resets_across_reconfigure(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """seq is a stream-lifetime counter; new geometry must not restart it."""
    ring = RingBuffer(4000)
    cfg1 = make_config(100, 100)
    cfg2 = make_config(50, 50)
    gate = Gate(at=0)
    collector = Collector(hook=gate)
    ch = make_chunker(ring, cfg1, collector, gate=gate)

    try:
        ch.start()
        write_ramp(ring, 0, 100)                       # -> chunk 0, parks in callback
        assert gate.entered.wait(TIMEOUT), "chunk 0 was never emitted"

        write_ramp(ring, 100, 400)                     # write head -> 500
        ch.reconfigure(cfg2)
        gate.open()                                    # chunker realigns to 500

        write_ramp(ring, 500, 200)                     # -> 500,550,600,650
        assert collector.wait_for_count(5), (
            f"expected 5 chunks after reconfigure, got {collector.count}"
        )
    finally:
        gate.open()

    seqs = [c.seq for c in collector.snapshot()[:5]]
    assert seqs == [0, 1, 2, 3, 4], (
        f"seq must keep counting across reconfigure(), got {seqs}"
    )


# --------------------------------------------------------------------------
# owned copy -- non-negotiable
# --------------------------------------------------------------------------

def test_chunk_samples_are_an_owned_copy_not_a_ring_view(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """Mutating the ring after emit must not change an already-emitted chunk."""
    w, h = 100, 100
    ring = RingBuffer(200)                 # small: the writer laps almost at once
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, w)
    assert collector.wait_for_count(1), "the first window was never emitted"
    chunk = collector.snapshot()[0]

    before = np.array(chunk.samples, copy=True)
    assert_ramp_chunk(chunk, 0, w, "first window")

    # Scribble over the whole ring, twice, so frames [0, W) are definitely gone.
    for _ in range(2):
        ring.write(np.full(200, -7.5, dtype=np.float32))

    assert np.array_equal(chunk.samples, before), (
        "the chunk's samples changed when the ring was overwritten -- the chunker "
        "retained a view into the ring instead of calling .copy()"
    )
    assert np.array_equal(chunk.samples, ramp(0, w)), (
        "the emitted chunk must still be the original ramp window"
    )


def test_chunk_samples_base_is_none(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """An owned copy has no base array; a ring view would."""
    w, h = 64, 32
    ring = RingBuffer(500)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, w + h)
    assert collector.wait_for_count(2), "expected two chunks"

    for chunk in collector.snapshot()[:2]:
        assert chunk.samples.base is None, (
            f"chunk seq={chunk.seq}: samples.base must be None -- it must own its "
            "memory, not alias the ring's backing store"
        )
        assert chunk.samples.dtype == np.float32


# --------------------------------------------------------------------------
# discontinuous
# --------------------------------------------------------------------------

def test_first_chunk_is_not_discontinuous(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """Nothing preceded chunk 0, so there is no continuity to have broken."""
    w, h = 100, 100
    ring = RingBuffer(1000)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, w)
    assert collector.wait_for_count(1)
    assert collector.snapshot()[0].discontinuous is False, (
        "the very first chunk must NOT be flagged discontinuous"
    )


def test_steady_state_chunks_are_not_discontinuous(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    w, h = 100, 25
    n = 10
    total = w + h * (n - 1)
    ring = RingBuffer(total + w)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, total)
    assert collector.wait_for_count(n)

    flags = [c.discontinuous for c in collector.snapshot()[:n]]
    assert flags == [False] * n, (
        f"no chunk may be discontinuous in an uninterrupted stream, got {flags}"
    )


# --------------------------------------------------------------------------
# overrun recovery
# --------------------------------------------------------------------------

def _overrun_scenario(
    make_chunker: Callable[..., WindowChunker],
) -> tuple[WindowChunker, Collector, RingBuffer, Gate]:
    """Park the consumer in chunk 0 while the writer laps the ring.

    Geometry: W=H=100, capacity=300.  After the writer reaches frame 2000 the
    oldest surviving frame is 2000-300 = 1700, so the chunker's pending read at
    frame 100 must raise OverrunError and resync to 1700.
    """
    w, h = 100, 100
    ring = RingBuffer(300)
    gate = Gate(at=0)
    collector = Collector(hook=gate)
    ch = make_chunker(ring, make_config(w, h), collector, gate=gate)

    ch.start()
    write_ramp(ring, 0, w, block=100)                 # -> chunk 0 at frame 0
    assert gate.entered.wait(TIMEOUT), "chunk 0 was never emitted"

    write_ramp(ring, w, 1900, block=100)              # lap the ring hard
    assert ring.write_frames == 2000
    assert ring.oldest_frame == 1700, "test setup: the writer must have lapped"

    gate.open()
    return ch, collector, ring, gate


def test_overrun_increments_stats_and_resyncs_to_oldest_frame(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch, collector, ring, gate = _overrun_scenario(make_chunker)
    try:
        assert collector.wait_for_count(4), (
            f"the chunker must recover from the overrun and keep emitting; got "
            f"{collector.count} chunks, error={ch.error!r}"
        )
    finally:
        gate.open()

    assert ch.error is None, f"the overrun must not kill the thread: {ch.error!r}"
    assert ch.is_running is True, "the chunker must survive an overrun"
    assert ch.stats.overruns == 1, (
        f"exactly one overrun happened, stats.overruns={ch.stats.overruns}"
    )

    chunks = collector.snapshot()[:4]
    assert_ramp_chunk(chunks[0], 0, 100, "pre-overrun chunk")
    assert_ramp_chunk(chunks[1], 1700, 100, "first chunk after the overrun (resync)")
    assert_ramp_chunk(chunks[2], 1800, 100, "second chunk after the overrun")
    assert_ramp_chunk(chunks[3], 1900, 100, "third chunk after the overrun")


def test_first_chunk_after_an_overrun_is_discontinuous(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch, collector, ring, gate = _overrun_scenario(make_chunker)
    try:
        assert collector.wait_for_count(3), f"got {collector.count} chunks"
    finally:
        gate.open()

    chunks = collector.snapshot()[:3]
    assert chunks[0].discontinuous is False, "the pre-overrun chunk was continuous"
    assert chunks[1].discontinuous is True, (
        "the first chunk emitted after an overrun must be flagged discontinuous"
    )
    assert chunks[2].discontinuous is False, (
        "discontinuous must be set once, not latched on for every later chunk"
    )


def test_seq_keeps_counting_across_an_overrun(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch, collector, ring, gate = _overrun_scenario(make_chunker)
    try:
        assert collector.wait_for_count(4), f"got {collector.count} chunks"
    finally:
        gate.open()

    seqs = [c.seq for c in collector.snapshot()[:4]]
    assert seqs == [0, 1, 2, 3], (
        f"a dropped/overrun window does not consume a seq number, got {seqs}"
    )


# --------------------------------------------------------------------------
# reconfigure()
# --------------------------------------------------------------------------

def _reconfigure_scenario(
    make_chunker: Callable[..., WindowChunker],
    cfg1: AudioConfig,
    cfg2: AudioConfig,
    tail_frames: int,
) -> tuple[WindowChunker, Collector, RingBuffer, Gate, int]:
    """Reconfigure while the chunker is parked inside chunk 0's callback.

    Because the chunker cannot re-read the config until the gate opens, the
    write head at the moment of realignment is pinned at exactly 500 frames.
    Returns the realignment frame so callers can assert against it.
    """
    ring = RingBuffer(4000)
    gate = Gate(at=0)
    collector = Collector(hook=gate)
    ch = make_chunker(ring, cfg1, collector, gate=gate)

    ch.start()
    write_ramp(ring, 0, cfg1.window_frames)
    assert gate.entered.wait(TIMEOUT), "chunk 0 was never emitted"

    write_head = write_ramp(ring, cfg1.window_frames, 500 - cfg1.window_frames)
    assert write_head == 500
    assert ring.write_frames == 500

    ch.reconfigure(cfg2)
    gate.open()

    # Realignment target is pinned: the writer is idle until the chunker has
    # necessarily observed the new config (it needs frames beyond 500 to emit).
    write_ramp(ring, 500, tail_frames)
    return ch, collector, ring, gate, 500


def test_reconfigure_realigns_to_the_write_head_and_uses_the_new_geometry(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    cfg1 = make_config(100, 100)
    cfg2 = make_config(50, 50)
    ch, collector, ring, gate, head = _reconfigure_scenario(
        make_chunker, cfg1, cfg2, tail_frames=200
    )
    try:
        assert collector.wait_for_count(5), (
            f"expected 4 post-reconfigure chunks, got {collector.count - 1}; "
            f"error={ch.error!r}"
        )
    finally:
        gate.open()

    chunks = collector.snapshot()[:5]
    assert_ramp_chunk(chunks[0], 0, 100, "pre-reconfigure chunk (old W)")
    # Realigned to the live write head, NOT to old_start + old_H (== 100).
    assert_ramp_chunk(chunks[1], head, 50, "first chunk after reconfigure")
    assert_ramp_chunk(chunks[2], head + 50, 50, "second chunk after reconfigure")
    assert_ramp_chunk(chunks[3], head + 100, 50, "third chunk after reconfigure")
    assert_ramp_chunk(chunks[4], head + 150, 50, "fourth chunk after reconfigure")

    for c in chunks[1:]:
        assert len(c.samples) == cfg2.window_frames, (
            "post-reconfigure chunks must use the NEW window length"
        )


def test_reconfigure_sets_discontinuous_exactly_once(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    cfg1 = make_config(100, 100)
    cfg2 = make_config(50, 50)
    ch, collector, ring, gate, head = _reconfigure_scenario(
        make_chunker, cfg1, cfg2, tail_frames=200
    )
    try:
        assert collector.wait_for_count(5), f"got {collector.count} chunks"
    finally:
        gate.open()

    flags = [c.discontinuous for c in collector.snapshot()[:5]]
    assert flags == [False, True, False, False, False], (
        f"reconfigure must flag exactly the next chunk discontinuous, got {flags}"
    )


def test_reconfigure_to_a_larger_window_is_still_sample_exact(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """Growing W and shrinking H (heavier overlap) must realign just as exactly."""
    cfg1 = make_config(100, 100)
    cfg2 = make_config(200, 40)
    ch, collector, ring, gate, head = _reconfigure_scenario(
        make_chunker, cfg1, cfg2, tail_frames=400
    )
    try:
        assert collector.wait_for_count(4), (
            f"got {collector.count} chunks; error={ch.error!r}"
        )
    finally:
        gate.open()

    chunks = collector.snapshot()[:4]
    for i, c in enumerate(chunks[1:]):
        assert_ramp_chunk(c, head + i * 40, 200, f"post-reconfigure chunk {i}")

    # ...and the new overlap is W-H = 160 identical samples.
    a, b = chunks[1], chunks[2]
    assert np.array_equal(a.samples[40:], b.samples[:160]), (
        "post-reconfigure chunks must overlap by the NEW W-H identical samples"
    )


def test_reconfigure_with_an_equal_but_distinct_config_still_realigns(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """Detection is by identity (`is not`), not equality."""
    cfg1 = make_config(100, 100)
    cfg2 = make_config(100, 100)         # equal in value, a different object
    assert cfg1 == cfg2, "test setup: the two configs must compare equal"
    assert cfg1 is not cfg2, "test setup: the two configs must be distinct objects"

    ch, collector, ring, gate, head = _reconfigure_scenario(
        make_chunker, cfg1, cfg2, tail_frames=300
    )
    try:
        assert collector.wait_for_count(3), (
            f"got {collector.count} chunks; error={ch.error!r}"
        )
    finally:
        gate.open()

    chunks = collector.snapshot()[:3]
    assert chunks[1].start_frame == head, (
        f"an equal-but-distinct config must still realign to the write head "
        f"({head}); got start_frame={chunks[1].start_frame} -- the implementation "
        "is comparing configs by == instead of by identity"
    )
    assert chunks[1].discontinuous is True, (
        "an equal-but-distinct config must still flag the next chunk discontinuous"
    )
    assert_ramp_chunk(chunks[1], head, 100, "realigned chunk")
    assert_ramp_chunk(chunks[2], head + 100, 100, "chunk after the realigned one")


def test_reconfigure_changes_the_sample_rate_reported_on_chunks(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """sample_rate comes from the config in force for that chunk."""
    cfg1 = make_config(100, 100, sample_rate=1000)
    cfg2 = AudioConfig(
        sample_rate=2000, window_ms=50, hop_ms=50, ring_seconds=1.0
    )
    assert cfg2.window_frames == 100 and cfg2.hop_frames == 100

    ch, collector, ring, gate, head = _reconfigure_scenario(
        make_chunker, cfg1, cfg2, tail_frames=200
    )
    try:
        assert collector.wait_for_count(2), f"got {collector.count} chunks"
    finally:
        gate.open()

    chunks = collector.snapshot()[:2]
    assert chunks[0].sample_rate == 1000
    assert chunks[1].sample_rate == 2000, (
        "chunks must report the sample rate of the config in force when emitted"
    )


def test_config_property_reflects_the_reconfigured_object(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(1000)
    cfg1 = make_config(100, 50)
    cfg2 = make_config(80, 20)
    ch = make_chunker(ring, cfg1, Collector())

    assert ch.config is cfg1, "config must expose the object passed to __init__"
    ch.reconfigure(cfg2)
    assert ch.config is cfg2, "config must expose the most recently set object"


def test_reconfigure_before_start_is_honoured(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """A config swapped in before start() is simply the starting geometry."""
    ring = RingBuffer(1000)
    cfg1 = make_config(100, 100)
    cfg2 = make_config(60, 60)
    collector = Collector()
    ch = make_chunker(ring, cfg1, collector)

    ch.reconfigure(cfg2)
    ch.start()
    write_ramp(ring, 0, 120)

    assert collector.wait_for_count(2), f"got {collector.count} chunks"
    chunks = collector.snapshot()[:2]
    assert len(chunks[0].samples) == 60, "the pre-start reconfigure must take effect"
    assert_ramp_chunk(chunks[1], 60, 60, "second chunk under the pre-start config")


# --------------------------------------------------------------------------
# no partial windows
# --------------------------------------------------------------------------

def test_fewer_than_w_frames_then_close_emits_nothing(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    w, h = 100, 50
    ring = RingBuffer(1000)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, w - 1)
    ring.close()

    assert wait_until(lambda: ch.is_running is False), (
        "the chunker must exit once the ring is closed and the target is unreachable"
    )
    assert collector.count == 0, (
        f"a {w - 1}-frame tail is shorter than the window and must be dropped, not "
        f"emitted as a partial chunk (got {collector.count} chunks)"
    )
    assert ch.error is None
    assert ch.stats.chunks_emitted == 0


@pytest.mark.parametrize("tail", [1, 17, 49, 99])
def test_partial_tail_after_a_full_window_is_dropped(
    tail: int, make_chunker: Callable[..., WindowChunker]
) -> None:
    """W + tail frames (tail < H, and W + tail < W + H) yields exactly one chunk."""
    w, h = 100, 100
    ring = RingBuffer(1000)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, w + tail)
    assert collector.wait_for_count(1), "the one full window must be emitted"
    ring.close()

    assert wait_until(lambda: ch.is_running is False), "the chunker must exit on close"
    assert collector.count == 1, (
        f"only the full window may be emitted; the {tail}-frame tail must be "
        f"dropped (got {collector.count} chunks)"
    )
    assert_ramp_chunk(collector.snapshot()[0], 0, w, "the single full window")
    assert ch.stats.chunks_emitted == 1


def test_two_full_windows_plus_a_partial_tail(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    w, h = 100, 50
    ring = RingBuffer(1000)
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector)

    ch.start()
    write_ramp(ring, 0, 199)          # windows at 0 and 50; 100..199 is only 99 frames
    ring.close()

    assert wait_until(lambda: ch.is_running is False), "the chunker must exit on close"
    chunks = collector.snapshot()
    assert len(chunks) == 2, (
        f"199 frames with W=100 H=50 gives exactly 2 full windows, got {len(chunks)}: "
        f"{[c.start_frame for c in chunks]}"
    )
    assert_ramp_chunk(chunks[0], 0, w, "window 0")
    assert_ramp_chunk(chunks[1], 50, w, "window 1")


def test_closing_the_ring_stops_the_thread(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(1000)
    ch = make_chunker(ring, make_config(100, 100), Collector())
    ch.start()
    assert ch.is_running is True

    ring.close()
    assert wait_until(lambda: ch.is_running is False, timeout=5.0), (
        "a closed ring with an unreachable target must end the run loop"
    )
    assert ch.stop(timeout=5.0) is True


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------

def test_is_running_is_false_before_start(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch = make_chunker(RingBuffer(500), make_config(100, 100), Collector())
    assert ch.is_running is False


def test_is_running_transitions_true_then_false(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(500)
    ch = make_chunker(ring, make_config(100, 100), Collector())

    ch.start()
    assert ch.is_running is True, "is_running must be True while the thread runs"
    assert ch.stop(timeout=5.0) is True
    assert ch.is_running is False, "is_running must be False once stopped"


def test_start_twice_raises_runtimeerror(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch = make_chunker(RingBuffer(500), make_config(100, 100), Collector())
    ch.start()
    with pytest.raises(RuntimeError):
        ch.start()


def test_start_after_stop_raises_runtimeerror(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """No restart: a finished chunker is finished."""
    ch = make_chunker(RingBuffer(500), make_config(100, 100), Collector())
    ch.start()
    assert ch.stop(timeout=5.0) is True
    with pytest.raises(RuntimeError):
        ch.start()


def test_stop_before_start_returns_true(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch = make_chunker(RingBuffer(500), make_config(100, 100), Collector())
    assert ch.stop(timeout=5.0) is True, (
        "stopping a never-started chunker is a no-op returning True"
    )
    assert ch.is_running is False


def test_stop_is_idempotent(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch = make_chunker(RingBuffer(500), make_config(100, 100), Collector())
    ch.start()
    assert ch.stop(timeout=5.0) is True
    assert ch.stop(timeout=5.0) is True, "a second stop() must also return True"
    assert ch.stop(timeout=5.0) is True, "and a third"
    assert ch.is_running is False


def test_stop_is_prompt_while_the_loop_is_blocked_waiting_for_data(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """POLL_INTERVAL_S guarantees a wake, so stop() must not sit out its timeout."""
    ring = RingBuffer(500)                      # never written to: the loop blocks
    ch = make_chunker(ring, make_config(100, 100), Collector())
    ch.start()
    assert wait_until(lambda: ch.is_running is True, timeout=5.0)

    t0 = time.monotonic()
    result = ch.stop(timeout=5.0)
    elapsed = time.monotonic() - t0

    assert result is True, "stop() must join the blocked thread successfully"
    assert elapsed < 2.0, (
        f"stop() took {elapsed:.2f}s while the loop was blocked in wait_for; the "
        "loop must use the finite POLL_INTERVAL_S timeout, never None"
    )
    assert ch.is_running is False


def test_stop_returns_false_when_the_thread_is_still_alive_at_timeout(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """A wedged callback keeps the thread alive; stop() reports that honestly."""
    ring = RingBuffer(500)
    gate = Gate(at=0)
    collector = Collector(hook=gate)
    ch = make_chunker(ring, make_config(100, 100), collector, gate=gate)

    try:
        ch.start()
        write_ramp(ring, 0, 100)
        assert gate.entered.wait(TIMEOUT), "chunk 0 was never emitted"

        assert ch.stop(timeout=0.1) is False, (
            "stop() must return False when the thread is still alive at timeout"
        )
    finally:
        gate.open()

    assert ch.stop(timeout=5.0) is True, "once unblocked the thread must join"
    assert ch.is_running is False


def test_stop_after_the_ring_closes_returns_true(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(500)
    ch = make_chunker(ring, make_config(100, 100), Collector())
    ch.start()
    ring.close()
    assert ch.stop(timeout=5.0) is True


def test_worker_thread_is_a_daemon_named_as_configured(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """A non-daemon worker would keep the interpreter alive after a crash."""
    ring = RingBuffer(500)
    ch = make_chunker(ring, make_config(100, 100), Collector(), name="ramp-chunker")
    ch.start()

    matching = [t for t in threading.enumerate() if t.name == "ramp-chunker"]
    assert matching, (
        "the worker thread must be named with the `name` argument; saw "
        f"{[t.name for t in threading.enumerate()]}"
    )
    assert all(t.daemon for t in matching), "the worker thread must be a daemon"


# --------------------------------------------------------------------------
# callback exceptions
# --------------------------------------------------------------------------

class Boom(Exception):
    """Sentinel raised by a deliberately broken on_chunk."""


def test_error_is_none_on_a_healthy_chunker(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(500)
    collector = Collector()
    ch = make_chunker(ring, make_config(100, 100), collector)

    assert ch.error is None, "a fresh chunker has no error"
    ch.start()
    write_ramp(ring, 0, 200)
    assert collector.wait_for_count(2)
    assert ch.error is None, "a healthy chunker must not record an error"


def test_raising_callback_stores_the_exception_in_error(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(500)

    def exploding(index: int, chunk: AudioChunk) -> None:
        raise Boom(f"callback failed on seq={chunk.seq}")

    collector = Collector(hook=exploding)
    ch = make_chunker(ring, make_config(100, 100), collector)

    ch.start()
    write_ramp(ring, 0, 300)

    assert wait_until(lambda: ch.error is not None), (
        "an exception from on_chunk must be stored in .error"
    )
    assert isinstance(ch.error, Boom), (
        f"error must be the exception the callback raised, got {ch.error!r}"
    )


def test_raising_callback_stops_the_thread(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(500)

    def exploding(index: int, chunk: AudioChunk) -> None:
        raise Boom("nope")

    collector = Collector(hook=exploding)
    ch = make_chunker(ring, make_config(100, 100), collector)

    ch.start()
    write_ramp(ring, 0, 500)

    assert wait_until(lambda: ch.is_running is False), (
        "a raising callback must end the run loop, not spin forever"
    )
    # Exactly one chunk got as far as the callback; the loop stopped there.
    assert collector.count == 1, (
        f"the loop must exit on the first callback failure, got {collector.count} "
        "callback invocations"
    )


def test_stop_does_not_reraise_a_callback_exception(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """Callers inspect .error; stop() itself stays quiet."""
    ring = RingBuffer(500)

    def exploding(index: int, chunk: AudioChunk) -> None:
        raise Boom("nope")

    ch = make_chunker(ring, make_config(100, 100), Collector(hook=exploding))
    ch.start()
    write_ramp(ring, 0, 200)
    assert wait_until(lambda: ch.error is not None)

    t0 = time.monotonic()
    result = ch.stop(timeout=5.0)                  # must not raise, must not hang
    assert time.monotonic() - t0 < 2.0, "stop() must not deadlock on a dead thread"
    assert result is True, "the thread already ended, so stop() must return True"
    assert isinstance(ch.error, Boom), "stop() must not clear .error"


def test_a_raising_callback_does_not_deadlock_a_later_stop(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ring = RingBuffer(500)

    def exploding(index: int, chunk: AudioChunk) -> None:
        raise Boom("nope")

    ch = make_chunker(ring, make_config(100, 100), Collector(hook=exploding))
    ch.start()
    write_ramp(ring, 0, 200)
    assert wait_until(lambda: ch.is_running is False)

    for _ in range(3):
        assert ch.stop(timeout=5.0) is True, "stop() stays idempotent after an error"


def test_baseexception_from_the_callback_is_also_captured(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """`error` is typed BaseException|None, so a KeyboardInterrupt counts too."""
    ring = RingBuffer(500)

    def exploding(index: int, chunk: AudioChunk) -> None:
        raise KeyboardInterrupt("simulated")

    ch = make_chunker(ring, make_config(100, 100), Collector(hook=exploding))
    ch.start()
    write_ramp(ring, 0, 200)

    assert wait_until(lambda: ch.error is not None), (
        "a BaseException from on_chunk must still be recorded in .error"
    )
    assert isinstance(ch.error, KeyboardInterrupt)
    assert wait_until(lambda: ch.is_running is False)
    assert ch.stop(timeout=5.0) is True


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------

def test_stats_defaults_to_a_fresh_streamstats(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    ch = make_chunker(RingBuffer(500), make_config(100, 100), Collector(), stats=None)
    assert isinstance(ch.stats, StreamStats), (
        "stats=None must make the chunker allocate its own StreamStats"
    )
    assert ch.stats.chunks_emitted == 0
    assert ch.stats.overruns == 0


def test_stats_property_is_the_supplied_live_object(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    stats = StreamStats()
    ch = make_chunker(RingBuffer(500), make_config(100, 100), Collector(), stats=stats)
    assert ch.stats is stats, (
        "the chunker must mutate the caller's StreamStats, not a copy of it"
    )


def test_chunks_emitted_matches_the_callback_count(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    w, h = 100, 25
    n = 9
    total = w + h * (n - 1)
    ring = RingBuffer(total + w)
    stats = StreamStats()
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector, stats=stats)

    ch.start()
    write_ramp(ring, 0, total)
    assert collector.wait_for_count(n), f"got {collector.count} chunks"

    ring.close()
    assert ch.stop(timeout=5.0) is True          # quiesce before comparing counters

    assert collector.count == n, f"expected exactly {n} chunks, got {collector.count}"
    assert stats.chunks_emitted == collector.count, (
        f"stats.chunks_emitted ({stats.chunks_emitted}) must match the number of "
        f"chunks the callback saw ({collector.count})"
    )


def test_chunker_does_not_touch_the_other_stats_fields(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """frames_captured / xruns belong to the source; chunks_dropped to the sinks."""
    w, h = 100, 50
    total = 400
    ring = RingBuffer(1000)
    stats = StreamStats()
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector, stats=stats)

    ch.start()
    write_ramp(ring, 0, total)
    assert collector.wait_for_count(7), f"got {collector.count} chunks"
    ring.close()
    assert ch.stop(timeout=5.0) is True

    assert stats.chunks_emitted > 0, "test setup: chunks must actually have been emitted"
    assert stats.frames_captured == 0, "the chunker must not touch frames_captured"
    assert stats.xruns == 0, "the chunker must not touch xruns"
    assert stats.chunks_dropped == 0, "the chunker must not touch chunks_dropped"
    assert stats.peak_level == 0.0, "the chunker must not touch peak_level"
    assert stats.rms_level == 0.0, "the chunker must not touch rms_level"


def test_overruns_is_the_only_other_counter_the_chunker_moves(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    stats = StreamStats()
    ring = RingBuffer(300)
    gate = Gate(at=0)
    collector = Collector(hook=gate)
    ch = make_chunker(ring, make_config(100, 100), collector, stats=stats, gate=gate)

    try:
        ch.start()
        write_ramp(ring, 0, 100, block=100)
        assert gate.entered.wait(TIMEOUT), "chunk 0 was never emitted"
        write_ramp(ring, 100, 1900, block=100)
        gate.open()
        assert collector.wait_for_count(3), f"got {collector.count} chunks"
    finally:
        gate.open()

    assert stats.overruns == 1, f"stats.overruns must be 1, got {stats.overruns}"
    assert stats.chunks_dropped == 0, (
        "an overrun is not a sink drop; chunks_dropped belongs to the sinks"
    )
    assert stats.frames_captured == 0
    assert stats.xruns == 0


def test_no_overruns_are_counted_on_a_healthy_stream(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    w, h = 100, 50
    total = 500
    ring = RingBuffer(total + w)
    stats = StreamStats()
    collector = Collector()
    ch = make_chunker(ring, make_config(w, h), collector, stats=stats)

    ch.start()
    write_ramp(ring, 0, total)
    assert collector.wait_for_count(9), f"got {collector.count} chunks"
    ring.close()
    assert ch.stop(timeout=5.0) is True

    assert stats.overruns == 0, (
        f"the ring was never lapped, so overruns must stay 0, got {stats.overruns}"
    )


# --------------------------------------------------------------------------
# Config/ring pairing guard
#
# AudioConfig validates its own derived ring_frames, but the caller supplies the
# actual RingBuffer separately -- nothing forces the two to agree. A window
# larger than the ring can never be read, and without an upfront check the
# failure surfaces as ValueError raised deep on the chunker thread: the stream
# just silently never produces a chunk.
# --------------------------------------------------------------------------


def test_window_larger_than_ring_is_rejected_at_construction() -> None:
    cfg = make_config(500, 100)
    too_small = RingBuffer(cfg.window_frames - 1)
    with pytest.raises(ValueError, match="exceeds ring capacity"):
        WindowChunker(too_small, cfg, lambda chunk: None)


def test_window_exactly_ring_capacity_is_allowed() -> None:
    """The boundary is inclusive: read(start, capacity) is a legal ring read."""
    cfg = make_config(500, 100)
    exact = RingBuffer(cfg.window_frames)
    WindowChunker(exact, cfg, lambda chunk: None)  # must not raise


def test_reconfigure_to_an_oversized_window_is_rejected_and_leaves_config_intact(
    make_chunker: Callable[..., WindowChunker],
) -> None:
    """A rejected reconfigure must not poison a running stream."""
    cfg = make_config(100, 50)
    ring = RingBuffer(200)
    collector = Collector()
    ch = make_chunker(ring, cfg, collector)

    oversized = make_config(400, 50)
    with pytest.raises(ValueError, match="exceeds ring capacity"):
        ch.reconfigure(oversized)

    assert ch.config is cfg, "rejected reconfigure must leave the config unchanged"

    ch.start()
    write_ramp(ring, 0, cfg.window_frames)
    assert collector.wait_for_count(1), "stream must still work after a rejected reconfigure"
    assert collector.chunks[0].start_frame == 0


# --------------------------------------------------------------------------
# __repr__
#
# Regression: __repr__ read a `_config` attribute that was renamed to `_state`,
# so repr() raised AttributeError. Debug helpers are exactly the code that is
# never exercised until you need it at 2am, so it gets a test.
# --------------------------------------------------------------------------


def test_repr_works_and_reports_geometry() -> None:
    cfg = make_config(100, 40)
    ring = RingBuffer(cfg.ring_frames)
    ch = WindowChunker(ring, cfg, lambda chunk: None, name="reprtest")

    text = repr(ch)  # must not raise

    assert "reprtest" in text
    assert "window_frames=100" in text
    assert "hop_frames=40" in text
    assert "running=False" in text


def test_repr_reflects_a_reconfigured_geometry() -> None:
    cfg = make_config(100, 40)
    ring = RingBuffer(cfg.ring_frames)
    ch = WindowChunker(ring, cfg, lambda chunk: None)

    ch.reconfigure(make_config(50, 25))

    text = repr(ch)
    assert "window_frames=50" in text and "hop_frames=25" in text
