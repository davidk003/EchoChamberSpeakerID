"""Tests for echochamber.audio.sinks, written against the step-3 API contract.

These tests are written from the *spec*, not from the implementation.  Nothing
here needs an audio device: chunks are constructed by hand and pushed straight
into a sink, which is exactly how the chunker would deliver them.

Three things earn most of the assertions:

* :func:`new_frame_count` is a **pure function** and is therefore tested as a
  table -- steady state, first chunk, post-overrun gap, and a stale chunk that
  contributes nothing.  Every de-overlapping bug the recorder can have is
  really a bug in this function.
* :class:`WavRecorderSink` must reconstruct the **original continuous signal**
  from overlapping windows.  The test replays a known ramp through overlapping
  chunks, reads the WAV back off disk, and compares sample-for-sample (modulo
  int16 quantisation).  Writing the windows verbatim would duplicate ``W - H``
  frames per chunk and this test would fail loudly.
* :class:`QueueSink` crosses a thread boundary, so its tests are event-driven:
  blocked calls run on helper threads and are proven blocked by *absence* over
  a short quiet period, then released.  Every sink is closed and every helper
  thread joined in fixture teardown, so a wedged sink cannot hang the suite.
"""

from __future__ import annotations

import threading
import time
import wave
from typing import Any, Callable, Iterator

import numpy as np
import pytest

from echochamber.audio.sinks import (
    CallableSink,
    ChunkSink,
    QueueSink,
    TeeSink,
    WavRecorderSink,
    new_frame_count,
)
from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats


# Generous: every wait below is event-driven, so these only bound failures.
TIMEOUT = 10.0

# How long to allow for something to *not* happen (proving a call is blocked).
QUIET_S = 0.35

SR = 16_000

# One int16 LSB, plus slack for whichever rounding the implementation picked
# (truncation vs round-half-away, and 32767 vs 32768 scaling).
QUANT_TOL = 2.5 / 32767.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_chunk(
    samples: np.ndarray,
    start_frame: int,
    seq: int = 0,
    sample_rate: int = SR,
    discontinuous: bool = False,
) -> AudioChunk:
    """Build an AudioChunk the way the chunker would."""
    return AudioChunk(
        np.asarray(samples, dtype=np.float32),
        start_frame,
        seq,
        sample_rate,
        discontinuous,
    )


def signal(n: int) -> np.ndarray:
    """A strictly increasing float32 ramp over [-0.9, 0.9].

    Strictly increasing means a duplicated or dropped frame in the recorder's
    reconstruction shows up as a mismatch, not as harmless-looking audio.
    """
    return np.linspace(-0.9, 0.9, n, dtype=np.float32)


def windows(sig: np.ndarray, w: int, h: int) -> list[AudioChunk]:
    """Cut ``sig`` into the same overlapping windows the chunker would emit."""
    chunks: list[AudioChunk] = []
    start = 0
    seq = 0
    while start + w <= len(sig):
        chunks.append(make_chunk(sig[start : start + w], start, seq))
        start += h
        seq += 1
    return chunks


def read_wav(path: Any) -> tuple[np.ndarray, int, int, int]:
    """Read a mono PCM WAV back as (float samples, rate, channels, sampwidth)."""
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    assert sampwidth == 2, f"the recorder must write 16-bit PCM, got {sampwidth} bytes"
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    return data, rate, channels, sampwidth


def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Poll ``pred`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


class Boom(Exception):
    """Sentinel raised by a deliberately broken sink."""


class CloseBoom(Exception):
    """Sentinel raised by a sink whose close() is broken."""


class RecordingSink:
    """A minimal ChunkSink that logs every call into a shared list.

    Used to prove TeeSink's fan-out order and its "keep going after a failure"
    policy.  ``raises`` / ``close_raises`` make it selectively broken.
    """

    def __init__(
        self,
        name: str,
        log: list[str],
        raises: BaseException | None = None,
        close_raises: BaseException | None = None,
    ) -> None:
        self.name = name
        self.log = log
        self.raises = raises
        self.close_raises = close_raises
        self.chunks: list[AudioChunk] = []
        self.close_calls = 0

    def on_chunk(self, chunk: AudioChunk) -> None:
        self.log.append(self.name)
        self.chunks.append(chunk)
        if self.raises is not None:
            raise self.raises

    def close(self) -> None:
        self.log.append(f"close:{self.name}")
        self.close_calls += 1
        if self.close_raises is not None:
            raise self.close_raises

    @property
    def closed(self) -> bool:
        return self.close_calls > 0


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def make_queue_sink() -> Iterator[Callable[..., QueueSink]]:
    """Factory registering every QueueSink for guaranteed close() on teardown.

    Closing wakes any blocked producer or consumer, which is what stops a
    misbehaving implementation from wedging the rest of the suite.
    """
    created: list[QueueSink] = []

    def _make(
        maxsize: int,
        policy: DropPolicy,
        stats: StreamStats | None = None,
    ) -> QueueSink:
        sink = QueueSink(maxsize, policy, stats=stats)
        created.append(sink)
        return sink

    yield _make

    for sink in created:
        try:
            sink.close()
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


@pytest.fixture
def spawn() -> Iterator[Callable[..., threading.Thread]]:
    """Start daemon helper threads and join them all (bounded) on teardown."""
    threads: list[threading.Thread] = []

    def _spawn(fn: Callable[[], None], name: str = "helper") -> threading.Thread:
        t = threading.Thread(target=fn, name=name, daemon=True)
        threads.append(t)
        t.start()
        return t

    yield _spawn

    for t in threads:
        t.join(timeout=5.0)


@pytest.fixture
def recorder(tmp_path: Any) -> Iterator[Callable[..., WavRecorderSink]]:
    """Factory registering every WavRecorderSink so the file is always closed."""
    created: list[WavRecorderSink] = []

    def _make(name: str = "rec.wav", sample_rate: int = SR) -> WavRecorderSink:
        sink = WavRecorderSink(tmp_path / name, sample_rate)
        created.append(sink)
        return sink

    yield _make

    for sink in created:
        try:
            sink.close()
        except Exception:  # pragma: no cover
            pass


# ==========================================================================
# new_frame_count -- the pure de-overlapping helper
# ==========================================================================

# (chunk_start, chunk_len, next_expected, expected_n_new, expected_gap, what)
NEW_FRAME_CASES = [
    # --- the very first chunk: next_expected starts at its own start_frame ---
    (0, 100, 0, 100, 0, "first chunk at frame 0 writes all of itself"),
    (500, 100, 500, 100, 0, "first chunk mid-stream writes all of itself"),

    # --- steady state, W=100 H=40: each chunk contributes exactly H frames ---
    (40, 100, 100, 40, 0, "steady state W=100 H=40 chunk 1"),
    (80, 100, 140, 40, 0, "steady state W=100 H=40 chunk 2"),
    (120, 100, 180, 40, 0, "steady state W=100 H=40 chunk 3"),

    # --- steady state with other geometries ---
    (1, 100, 100, 1, 0, "one-frame hop contributes exactly one frame"),
    (99, 100, 100, 99, 0, "H just below W"),
    (100, 100, 100, 100, 0, "H == W: no overlap, the whole chunk is new"),
    (3000, 3000, 3000, 3000, 0, "H == W at window scale"),

    # --- a gap after an overrun: the whole chunk is new and gap is reported ---
    (500, 100, 300, 100, 200, "overrun gap of 200 frames"),
    (301, 100, 300, 100, 1, "smallest possible gap"),
    (10_000, 3000, 25, 3000, 9975, "huge gap after a long stall"),

    # --- fully overlapping / stale chunk: nothing new, no gap ---
    (0, 100, 100, 0, 0, "chunk entirely before next_expected (exactly flush)"),
    (0, 100, 250, 0, 0, "stale chunk far behind next_expected"),
    (40, 100, 140, 0, 0, "duplicate delivery of an already-written window"),
    (40, 100, 500, 0, 0, "very stale chunk contributes nothing"),
]


@pytest.mark.parametrize(
    ("chunk_start", "chunk_len", "next_expected", "n_new", "gap", "what"),
    NEW_FRAME_CASES,
)
def test_new_frame_count_table(
    chunk_start: int,
    chunk_len: int,
    next_expected: int,
    n_new: int,
    gap: int,
    what: str,
) -> None:
    """new_frame_count is pure arithmetic; every branch gets a worked example."""
    got = new_frame_count(chunk_start, chunk_len, next_expected)
    assert isinstance(got, tuple) and len(got) == 2, (
        f"new_frame_count must return a (n_new, gap_frames) 2-tuple, got {got!r}"
    )
    assert got == (n_new, gap), (
        f"{what}: new_frame_count({chunk_start}, {chunk_len}, {next_expected}) "
        f"must be ({n_new}, {gap}), got {got}"
    )


def test_new_frame_count_steady_state_returns_the_hop() -> None:
    """In steady state each chunk contributes exactly H new frames, forever."""
    w, h = 3000, 1000
    next_expected = w                       # after chunk 0 was written whole
    for k in range(1, 20):
        n_new, gap = new_frame_count(k * h, w, next_expected)
        assert gap == 0, f"chunk {k}: an uninterrupted stream has no gaps"
        assert n_new == h, (
            f"chunk {k}: a steady-state chunk contributes exactly the hop "
            f"({h} frames), got {n_new} -- writing more would duplicate the "
            "overlap into the recording"
        )
        next_expected = max(next_expected, k * h + w)


def test_new_frame_count_never_exceeds_chunk_len() -> None:
    """n_new is a count of frames taken from the END of the chunk."""
    for chunk_len in (1, 7, 100, 3000):
        for chunk_start in range(0, 400, 13):
            for next_expected in range(0, 400, 7):
                n_new, gap = new_frame_count(chunk_start, chunk_len, next_expected)
                assert 0 <= n_new <= chunk_len, (
                    f"n_new={n_new} out of range for chunk_len={chunk_len} "
                    f"(start={chunk_start}, next_expected={next_expected})"
                )
                assert gap >= 0, f"gap must never be negative, got {gap}"
                if chunk_start > next_expected:
                    assert gap == chunk_start - next_expected
                    assert n_new == chunk_len
                else:
                    assert gap == 0, (
                        "a chunk starting at or before next_expected has no gap"
                    )


def test_new_frame_count_gap_and_n_new_are_ints() -> None:
    n_new, gap = new_frame_count(10, 5, 3)
    assert isinstance(n_new, int) and isinstance(gap, int), (
        "new_frame_count must return plain ints (frame counts), got "
        f"{type(n_new).__name__}/{type(gap).__name__}"
    )


# ==========================================================================
# WavRecorderSink
# ==========================================================================

def test_wav_recorder_creates_the_file_in_init(tmp_path: Any) -> None:
    """Opening happens in __init__, so a bad path fails at the call site."""
    path = tmp_path / "opened.wav"
    sink = WavRecorderSink(path, SR)
    try:
        assert path.exists(), "the WAV file must be opened by __init__"
    finally:
        sink.close()


def test_wav_recorder_header_is_mono_16bit_at_the_given_rate(
    recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    sink = recorder("hdr.wav", 8000)
    sink.on_chunk(make_chunk(signal(100), 0, 0, 8000))
    sink.close()

    _, rate, channels, sampwidth = read_wav(tmp_path / "hdr.wav")
    assert channels == 1, "the recorder writes mono"
    assert sampwidth == 2, "the recorder writes 16-bit PCM"
    assert rate == 8000, "the recorder writes at the sample_rate it was given"


@pytest.mark.parametrize(
    ("w", "h"),
    [
        (100, 40),      # 60 % overlap
        (100, 50),      # 50 % overlap
        (128, 1),       # pathological: one-frame hop, 127 frames of overlap
        (240, 80),      # the shipping 3000/1000 ratio, scaled down
        (100, 100),     # H == W: no overlap at all
        (64, 33),       # non-round sizes
    ],
)
def test_wav_recorder_reconstructs_the_original_signal_from_overlapping_chunks(
    w: int, h: int, recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    """The headline recorder test: de-overlapping must be sample-exact.

    Chunks overlap by ``W - H`` frames.  Writing them verbatim would repeat
    those frames and stretch the recording; writing only the last ``H`` frames
    of every chunk after the first reproduces the original continuous signal
    exactly.
    """
    total = 1000
    sig = signal(total)
    chunks = windows(sig, w, h)
    assert len(chunks) >= 4, "test setup: need several overlapping windows"

    sink = recorder("ramp.wav")
    for chunk in chunks:
        sink.on_chunk(chunk)

    covered = chunks[-1].start_frame + w      # last frame the windows reached
    assert sink.frames_written == covered, (
        f"W={w} H={h}: {len(chunks)} windows cover frames [0, {covered}); "
        f"frames_written={sink.frames_written} means the sink wrote the "
        "overlapping windows verbatim instead of de-overlapping them"
    )
    assert sink.gaps == 0, "a contiguous window grid has no gaps"

    sink.close()
    data, rate, _, _ = read_wav(tmp_path / "ramp.wav")

    assert rate == SR
    assert len(data) == covered, (
        f"W={w} H={h}: recorded {len(data)} frames, expected {covered}"
    )
    assert np.allclose(data, sig[:covered], atol=QUANT_TOL), (
        f"W={w} H={h}: the recording is not the original continuous signal; "
        f"first mismatch at frame "
        f"{int(np.argmax(np.abs(data - sig[:covered]) > QUANT_TOL))}"
    )


def test_wav_recorder_first_chunk_writes_all_its_frames(
    recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    """next_expected starts at the first chunk's start_frame -- no leading gap."""
    sig = signal(300)
    sink = recorder("first.wav")
    sink.on_chunk(make_chunk(sig[:100], 0, 0))

    assert sink.frames_written == 100, (
        "the very first chunk contributes all of its frames"
    )
    assert sink.gaps == 0, "the first chunk cannot be preceded by a gap"

    sink.close()
    data, _, _, _ = read_wav(tmp_path / "first.wav")
    assert np.allclose(data, sig[:100], atol=QUANT_TOL)


def test_wav_recorder_first_chunk_not_at_frame_zero_is_not_a_gap(
    recorder: Callable[..., WavRecorderSink]
) -> None:
    """A stream whose first delivered chunk starts late is not a discontinuity."""
    sink = recorder("late.wav")
    sink.on_chunk(make_chunk(signal(100), 5000, 0))

    assert sink.frames_written == 100
    assert sink.gaps == 0, (
        "next_expected starts at the FIRST chunk's start_frame, so a stream "
        "that begins at frame 5000 must not be reported as a 5000-frame gap"
    )


def test_wav_recorder_counts_a_gap_and_is_short_by_exactly_the_dropped_frames(
    recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    """After an overrun, gaps increments and the recording loses those frames.

    Gaps are deliberately not zero-filled, so recorded duration is shorter than
    wall clock; ``gaps`` is how the caller learns that happened.
    """
    w = 100
    dropped = 250
    a = signal(w)
    b = signal(w) * 0.5

    sink = recorder("gap.wav")
    sink.on_chunk(make_chunk(a, 0, 0))                       # frames [0, 100)
    sink.on_chunk(make_chunk(b, w + dropped, 1))             # frames [350, 450)

    assert sink.gaps == 1, (
        f"a {dropped}-frame hole between the chunks is exactly one gap, got "
        f"{sink.gaps}"
    )
    assert sink.frames_written == 2 * w, (
        "after a gap the whole chunk is new, so both chunks contribute W frames"
    )

    sink.close()
    data, _, _, _ = read_wav(tmp_path / "gap.wav")

    span = (w + dropped) + w                                  # 450 wall-clock frames
    assert len(data) == span - dropped, (
        f"the recording must be short by exactly the {dropped} dropped frames: "
        f"expected {span - dropped} frames, got {len(data)}"
    )
    assert np.allclose(data[:w], a, atol=QUANT_TOL), "pre-gap audio must survive"
    assert np.allclose(data[w:], b, atol=QUANT_TOL), (
        "post-gap audio must follow immediately -- gaps are not zero-filled"
    )


def test_wav_recorder_counts_each_gap_separately(
    recorder: Callable[..., WavRecorderSink]
) -> None:
    sink = recorder("gaps.wav")
    sink.on_chunk(make_chunk(signal(100), 0, 0))
    sink.on_chunk(make_chunk(signal(100), 200, 1))     # gap of 100
    sink.on_chunk(make_chunk(signal(100), 300, 2))     # contiguous
    sink.on_chunk(make_chunk(signal(100), 900, 3))     # gap of 500

    assert sink.gaps == 2, f"two discontinuities means gaps == 2, got {sink.gaps}"
    assert sink.frames_written == 400


def test_wav_recorder_ignores_a_fully_overlapping_stale_chunk(
    recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    """A chunk that ends at or before next_expected contributes nothing."""
    sig = signal(200)
    sink = recorder("stale.wav")
    sink.on_chunk(make_chunk(sig[0:100], 0, 0))
    sink.on_chunk(make_chunk(sig[100:200], 100, 1))
    before = sink.frames_written

    sink.on_chunk(make_chunk(sig[0:100], 0, 2))       # a replayed old window
    assert sink.frames_written == before, (
        "a chunk entirely behind next_expected must write nothing"
    )
    assert sink.gaps == 0, "a stale chunk is not a gap"

    sink.close()
    data, _, _, _ = read_wav(tmp_path / "stale.wav")
    assert len(data) == 200
    assert np.allclose(data, sig, atol=QUANT_TOL)


def test_wav_recorder_clips_out_of_range_samples(
    recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    """float32 -> int16 clips to [-1, 1] rather than wrapping around."""
    loud = np.array([2.0, -2.0, 1.0, -1.0, 0.0, 5.5, -9.0], dtype=np.float32)
    sink = recorder("clip.wav")
    sink.on_chunk(make_chunk(loud, 0, 0))
    sink.close()

    data, _, _, _ = read_wav(tmp_path / "clip.wav")
    assert len(data) == len(loud)
    assert np.all(data <= 1.0 + 1e-6) and np.all(data >= -1.0 - 1e-6), (
        f"out-of-range samples must clip, not wrap: got {data}"
    )
    assert data[0] > 0.99, "+2.0 must clip to full positive scale, not wrap negative"
    assert data[1] < -0.99, "-2.0 must clip to full negative scale"
    assert abs(data[4]) < QUANT_TOL, "silence must stay silent"


def test_wav_recorder_counters_start_at_zero(
    recorder: Callable[..., WavRecorderSink]
) -> None:
    sink = recorder("zero.wav")
    assert sink.frames_written == 0
    assert sink.gaps == 0


def test_wav_recorder_close_is_idempotent(
    recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    sig = signal(150)
    sink = recorder("idem.wav")
    sink.on_chunk(make_chunk(sig, 0, 0))

    sink.close()
    sink.close()                     # must not raise
    sink.close()

    data, _, _, _ = read_wav(tmp_path / "idem.wav")
    assert len(data) == 150, (
        "a repeated close() must not truncate or corrupt the finalized file"
    )
    assert np.allclose(data, sig, atol=QUANT_TOL)
    assert sink.frames_written == 150, "close() must not reset the counters"


def test_wav_recorder_empty_recording_closes_cleanly(
    recorder: Callable[..., WavRecorderSink], tmp_path: Any
) -> None:
    """A pipeline that never produced a chunk still leaves a valid WAV."""
    sink = recorder("empty.wav")
    sink.close()

    data, rate, channels, _ = read_wav(tmp_path / "empty.wav")
    assert len(data) == 0
    assert rate == SR and channels == 1


def test_wav_recorder_satisfies_the_chunksink_protocol(
    recorder: Callable[..., WavRecorderSink]
) -> None:
    assert isinstance(recorder("proto.wav"), ChunkSink)


# ==========================================================================
# QueueSink
# ==========================================================================

def test_queue_sink_preserves_order(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    sink = make_queue_sink(8, DropPolicy.BLOCK)
    for k in range(5):
        sink.on_chunk(make_chunk(signal(10), k * 10, k))

    seqs = [sink.get(timeout=TIMEOUT).seq for _ in range(5)]  # type: ignore[union-attr]
    assert seqs == [0, 1, 2, 3, 4], f"FIFO order must be preserved, got {seqs}"


def test_queue_sink_qsize_tracks_the_backlog(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    sink = make_queue_sink(8, DropPolicy.BLOCK)
    assert sink.qsize == 0

    for k in range(3):
        sink.on_chunk(make_chunk(signal(10), k * 10, k))
    assert sink.qsize == 3, f"three queued chunks, qsize={sink.qsize}"

    sink.get(timeout=TIMEOUT)
    assert sink.qsize == 2, "qsize must fall as the consumer drains"


def test_queue_sink_stats_defaults_to_a_fresh_streamstats(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    sink = make_queue_sink(4, DropPolicy.DROP_OLDEST, None)
    assert isinstance(sink.stats, StreamStats)
    assert sink.stats.chunks_dropped == 0


def test_queue_sink_stats_property_is_the_supplied_live_object(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    stats = StreamStats()
    sink = make_queue_sink(4, DropPolicy.DROP_OLDEST, stats)
    assert sink.stats is stats, (
        "the sink must mutate the caller's StreamStats so the GUI sees the drops"
    )


def test_queue_sink_satisfies_the_chunksink_protocol(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    assert isinstance(make_queue_sink(4, DropPolicy.BLOCK), ChunkSink)


# --- DROP_OLDEST ----------------------------------------------------------

def test_drop_oldest_discards_the_oldest_and_counts_it(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    """Freshness beats completeness: the survivors are the NEWEST chunks."""
    stats = StreamStats()
    sink = make_queue_sink(2, DropPolicy.DROP_OLDEST, stats)

    for k in range(5):                       # 0,1,2,3,4 into a 2-deep queue
        sink.on_chunk(make_chunk(signal(10), k * 10, k))

    assert sink.qsize == 2, f"the queue must stay bounded at 2, got {sink.qsize}"
    assert stats.chunks_dropped == 3, (
        f"3 of 5 chunks had to be discarded, stats.chunks_dropped="
        f"{stats.chunks_dropped}"
    )

    seqs = [sink.get(timeout=TIMEOUT).seq for _ in range(2)]  # type: ignore[union-attr]
    assert seqs == [3, 4], (
        f"DROP_OLDEST must discard the OLDEST chunks and keep the newest, "
        f"got seqs {seqs} -- {[0, 1]} would mean it dropped the new arrivals"
    )


def test_drop_oldest_counts_one_drop_per_discarded_chunk(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    stats = StreamStats()
    sink = make_queue_sink(1, DropPolicy.DROP_OLDEST, stats)

    sink.on_chunk(make_chunk(signal(10), 0, 0))
    assert stats.chunks_dropped == 0, "a chunk that fits is not a drop"

    for k in range(1, 11):
        sink.on_chunk(make_chunk(signal(10), k * 10, k))
    assert stats.chunks_dropped == 10, (
        f"ten overflowing puts means ten drops, got {stats.chunks_dropped}"
    )
    assert sink.qsize == 1


def test_drop_oldest_never_blocks(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    """The chunker calls on_chunk; blocking there would overrun the ring."""
    sink = make_queue_sink(2, DropPolicy.DROP_OLDEST)

    t0 = time.monotonic()
    for k in range(2000):
        sink.on_chunk(make_chunk(signal(10), k * 10, k))
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0, (
        f"2000 puts into a full DROP_OLDEST queue took {elapsed:.2f}s -- "
        "DROP_OLDEST must never block the producer"
    )
    assert sink.qsize == 2


def test_drop_oldest_does_not_touch_other_stats_fields(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    stats = StreamStats()
    sink = make_queue_sink(1, DropPolicy.DROP_OLDEST, stats)
    for k in range(6):
        sink.on_chunk(make_chunk(signal(10), k * 10, k))

    assert stats.chunks_dropped == 5
    assert stats.chunks_emitted == 0, "emitting is the chunker's counter"
    assert stats.frames_captured == 0, "capturing is the source's counter"
    assert stats.overruns == 0, "a queue drop is not a ring overrun"


# --- BLOCK ----------------------------------------------------------------

def test_block_policy_blocks_until_drained(
    make_queue_sink: Callable[..., QueueSink],
    spawn: Callable[..., threading.Thread],
) -> None:
    """BLOCK back-pressures the producer instead of losing a chunk."""
    stats = StreamStats()
    sink = make_queue_sink(1, DropPolicy.BLOCK, stats)
    sink.on_chunk(make_chunk(signal(10), 0, 0))          # queue is now full

    returned = threading.Event()

    def producer() -> None:
        sink.on_chunk(make_chunk(signal(10), 10, 1))
        returned.set()

    spawn(producer, "block-producer")

    assert not returned.wait(QUIET_S), (
        "on_chunk must block while the BLOCK-policy queue is full"
    )
    assert stats.chunks_dropped == 0, "BLOCK must never drop a chunk"

    first = sink.get(timeout=TIMEOUT)                    # make room
    assert first is not None and first.seq == 0

    assert returned.wait(TIMEOUT), (
        "the blocked producer must be released once the consumer drains a slot"
    )
    second = sink.get(timeout=TIMEOUT)
    assert second is not None and second.seq == 1, (
        "the chunk the producer was blocked on must be enqueued, not lost"
    )
    assert stats.chunks_dropped == 0


def test_block_policy_close_releases_a_blocked_producer(
    make_queue_sink: Callable[..., QueueSink],
    spawn: Callable[..., threading.Thread],
) -> None:
    """A producer parked in on_chunk must not survive close() as a zombie."""
    sink = make_queue_sink(1, DropPolicy.BLOCK)
    sink.on_chunk(make_chunk(signal(10), 0, 0))

    returned = threading.Event()

    def producer() -> None:
        try:
            sink.on_chunk(make_chunk(signal(10), 10, 1))
        except Exception:
            pass                                  # a raise is an acceptable exit
        returned.set()

    spawn(producer, "block-producer-close")
    assert not returned.wait(QUIET_S), "test setup: the producer must be blocked"

    sink.close()
    assert returned.wait(TIMEOUT), (
        "close() must release a producer blocked in on_chunk, otherwise stop() "
        "can never join the chunker thread"
    )


# --- get() / close() ------------------------------------------------------

def test_get_returns_none_after_close_and_drain(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    """None is the end-of-stream signal -- but only after the backlog is drained."""
    sink = make_queue_sink(8, DropPolicy.BLOCK)
    sink.on_chunk(make_chunk(signal(10), 0, 0))
    sink.on_chunk(make_chunk(signal(10), 10, 1))
    sink.close()

    a = sink.get(timeout=TIMEOUT)
    b = sink.get(timeout=TIMEOUT)
    assert a is not None and a.seq == 0, (
        "close() must not discard chunks that were already queued"
    )
    assert b is not None and b.seq == 1

    assert sink.get(timeout=TIMEOUT) is None, (
        "once closed AND drained, get() must return None to end the consumer loop"
    )
    assert sink.get(timeout=TIMEOUT) is None, (
        "the end-of-stream signal must be repeatable, not a one-shot"
    )


def test_get_returns_none_immediately_on_a_closed_empty_sink(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    sink = make_queue_sink(8, DropPolicy.BLOCK)
    sink.close()

    t0 = time.monotonic()
    assert sink.get(timeout=None) is None
    assert time.monotonic() - t0 < 2.0, (
        "get(timeout=None) on a closed, empty sink must return at once, not hang"
    )


def test_close_wakes_a_get_blocked_with_timeout_none(
    make_queue_sink: Callable[..., QueueSink],
    spawn: Callable[..., threading.Thread],
) -> None:
    """The consumer thread parks in get(None); close() is what ends it."""
    sink = make_queue_sink(8, DropPolicy.DROP_OLDEST)
    result: list[Any] = []
    returned = threading.Event()

    def consumer() -> None:
        result.append(sink.get(timeout=None))
        returned.set()

    spawn(consumer, "blocked-consumer")

    assert not returned.wait(QUIET_S), (
        "test setup: get() on an open, empty sink must block"
    )

    sink.close()
    assert returned.wait(TIMEOUT), (
        "close() must wake a get(timeout=None); otherwise stop() hangs forever "
        "joining the consumer thread"
    )
    assert result == [None], (
        f"the woken get() must return the None end-of-stream signal, got {result!r}"
    )


def test_close_wakes_a_blocked_get_and_still_delivers_a_late_backlog(
    make_queue_sink: Callable[..., QueueSink],
    spawn: Callable[..., threading.Thread],
) -> None:
    """A chunk enqueued while the consumer waits must be delivered, not skipped."""
    sink = make_queue_sink(8, DropPolicy.DROP_OLDEST)
    result: list[Any] = []
    returned = threading.Event()

    def consumer() -> None:
        result.append(sink.get(timeout=None))
        returned.set()

    spawn(consumer, "waiting-consumer")
    assert not returned.wait(QUIET_S), "test setup: the consumer must be blocked"

    sink.on_chunk(make_chunk(signal(10), 0, 7))
    assert returned.wait(TIMEOUT), "a queued chunk must wake a blocked get()"
    assert result[0] is not None and result[0].seq == 7


def test_get_on_an_open_empty_sink_returns_none_at_timeout(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    """get() is typed AudioChunk | None, so a timeout yields None, not an error."""
    sink = make_queue_sink(4, DropPolicy.DROP_OLDEST)

    t0 = time.monotonic()
    assert sink.get(timeout=0.05) is None
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"get(timeout=0.05) waited {elapsed:.2f}s"


def test_queue_sink_close_is_idempotent(
    make_queue_sink: Callable[..., QueueSink]
) -> None:
    sink = make_queue_sink(4, DropPolicy.DROP_OLDEST)
    sink.close()
    sink.close()
    sink.close()
    assert sink.get(timeout=TIMEOUT) is None


def test_close_from_a_third_thread_is_safe(
    make_queue_sink: Callable[..., QueueSink],
    spawn: Callable[..., threading.Thread],
) -> None:
    """One producer + one consumer + close() from a third thread must be safe."""
    sink = make_queue_sink(4, DropPolicy.DROP_OLDEST)
    received: list[AudioChunk] = []
    done = threading.Event()
    produced = threading.Event()

    def consumer() -> None:
        while True:
            chunk = sink.get(timeout=None)
            if chunk is None:
                break
            received.append(chunk)
        done.set()

    def producer() -> None:
        for k in range(200):
            sink.on_chunk(make_chunk(signal(10), k * 10, k))
        produced.set()

    spawn(consumer, "sink-consumer")
    spawn(producer, "sink-producer")

    assert produced.wait(TIMEOUT), "the producer must not block under DROP_OLDEST"
    sink.close()                                   # third thread: the test thread
    assert done.wait(TIMEOUT), (
        "the consumer must see the None end-of-stream signal after close()"
    )
    assert len(received) > 0, "at least some chunks must have made it through"
    seqs = [c.seq for c in received]
    assert seqs == sorted(seqs), f"delivery order must be monotonic, got {seqs}"


# ==========================================================================
# TeeSink
# ==========================================================================

def test_tee_fans_out_to_every_sink_in_order() -> None:
    log: list[str] = []
    a = RecordingSink("a", log)
    b = RecordingSink("b", log)
    c = RecordingSink("c", log)
    tee = TeeSink(a, b, c)

    chunk = make_chunk(signal(10), 0, 0)
    tee.on_chunk(chunk)

    assert log == ["a", "b", "c"], f"sinks must be called in order, got {log}"
    for sink in (a, b, c):
        assert sink.chunks == [chunk], f"{sink.name} did not receive the chunk"
        assert sink.chunks[0] is chunk, "the same chunk object is shared, not copied"


def test_tee_sinks_property_is_a_tuple() -> None:
    a = RecordingSink("a", [])
    b = RecordingSink("b", [])
    tee = TeeSink(a, b)

    assert isinstance(tee.sinks, tuple), "sinks must be an immutable tuple"
    assert tee.sinks == (a, b)


def test_empty_tee_is_harmless() -> None:
    tee = TeeSink()
    assert tee.sinks == ()
    tee.on_chunk(make_chunk(signal(10), 0, 0))     # must not raise
    tee.close()


def test_tee_still_delivers_to_the_others_when_one_sink_raises() -> None:
    """One broken sink must not starve the recorder or the meters."""
    log: list[str] = []
    boom = Boom("sink b is broken")
    a = RecordingSink("a", log)
    b = RecordingSink("b", log, raises=boom)
    c = RecordingSink("c", log)
    tee = TeeSink(a, b, c)

    chunk = make_chunk(signal(10), 0, 0)
    with pytest.raises(Boom) as excinfo:
        tee.on_chunk(chunk)

    assert excinfo.value is boom, "the first exception must be re-raised as-is"
    assert log == ["a", "b", "c"], (
        f"every sink must be attempted even after one raises, got {log}"
    )
    assert a.chunks == [chunk], "the sink before the failure must have received it"
    assert c.chunks == [chunk], (
        "the sink AFTER the failure must still receive the chunk -- a broken "
        "sink must not silently starve the ones behind it"
    )


def test_tee_reraises_the_first_exception_when_several_sinks_raise() -> None:
    log: list[str] = []
    first = Boom("first failure")
    second = Boom("second failure")
    a = RecordingSink("a", log, raises=first)
    b = RecordingSink("b", log)
    c = RecordingSink("c", log, raises=second)
    tee = TeeSink(a, b, c)

    with pytest.raises(Boom) as excinfo:
        tee.on_chunk(make_chunk(signal(10), 0, 0))

    assert excinfo.value is first, (
        f"the FIRST exception must be re-raised, got {excinfo.value!r}"
    )
    assert log == ["a", "b", "c"], f"all three sinks must be attempted, got {log}"
    assert b.chunks, "the healthy sink between two failures must still be fed"


def test_tee_keeps_working_for_later_chunks_after_a_sink_raised() -> None:
    """A transient sink failure must not permanently break the fan-out."""
    log: list[str] = []
    a = RecordingSink("a", log)
    b = RecordingSink("b", log, raises=Boom("always"))
    tee = TeeSink(a, b)

    for k in range(3):
        with pytest.raises(Boom):
            tee.on_chunk(make_chunk(signal(10), k * 10, k))

    assert [c.seq for c in a.chunks] == [0, 1, 2], (
        "the healthy sink must keep receiving every chunk"
    )


def test_tee_close_closes_every_sink_even_if_one_raises() -> None:
    log: list[str] = []
    boom = CloseBoom("close of b failed")
    a = RecordingSink("a", log)
    b = RecordingSink("b", log, close_raises=boom)
    c = RecordingSink("c", log)
    tee = TeeSink(a, b, c)

    with pytest.raises(CloseBoom) as excinfo:
        tee.close()

    assert excinfo.value is boom, "the first close() exception must be re-raised"
    assert log == ["close:a", "close:b", "close:c"], (
        f"every sink must be closed even after one raises, got {log}"
    )
    assert a.closed and b.closed and c.closed, (
        "a failing close() must not leak the other sinks' files/threads"
    )


def test_tee_close_reraises_the_first_of_several_close_failures() -> None:
    log: list[str] = []
    first = CloseBoom("first")
    a = RecordingSink("a", log, close_raises=first)
    b = RecordingSink("b", log, close_raises=CloseBoom("second"))
    tee = TeeSink(a, b)

    with pytest.raises(CloseBoom) as excinfo:
        tee.close()
    assert excinfo.value is first
    assert a.closed and b.closed


def test_tee_close_with_healthy_sinks_closes_each_once() -> None:
    log: list[str] = []
    a = RecordingSink("a", log)
    b = RecordingSink("b", log)
    TeeSink(a, b).close()

    assert a.close_calls == 1 and b.close_calls == 1
    assert log == ["close:a", "close:b"]


def test_tee_satisfies_the_chunksink_protocol() -> None:
    assert isinstance(TeeSink(RecordingSink("a", [])), ChunkSink)


def test_tee_over_real_sinks(
    make_queue_sink: Callable[..., QueueSink],
    recorder: Callable[..., WavRecorderSink],
    tmp_path: Any,
) -> None:
    """The shipping arrangement: one chunk stream, a queue and a recorder."""
    q = make_queue_sink(8, DropPolicy.BLOCK)
    rec = recorder("tee.wav")
    tee = TeeSink(q, rec)

    sig = signal(300)
    for chunk in windows(sig, 100, 50):
        tee.on_chunk(chunk)

    assert q.qsize == 5, f"5 windows fit in a 300-frame signal, got {q.qsize}"
    assert rec.frames_written == 300

    tee.close()
    assert q.get(timeout=TIMEOUT) is not None
    data, _, _, _ = read_wav(tmp_path / "tee.wav")
    assert np.allclose(data, sig, atol=QUANT_TOL), (
        "the recorder behind the tee must still reconstruct the original signal"
    )


# ==========================================================================
# CallableSink
# ==========================================================================

def test_callable_sink_adapts_a_function() -> None:
    seen: list[AudioChunk] = []
    sink = CallableSink(seen.append)

    chunk = make_chunk(signal(10), 0, 0)
    sink.on_chunk(chunk)
    assert seen == [chunk], "the wrapped function must receive the chunk"
    assert seen[0] is chunk


def test_callable_sink_forwards_every_chunk_in_order() -> None:
    seen: list[int] = []
    sink = CallableSink(lambda c: seen.append(c.seq))

    for k in range(4):
        sink.on_chunk(make_chunk(signal(10), k * 10, k))
    assert seen == [0, 1, 2, 3]


def test_callable_sink_close_is_a_harmless_noop() -> None:
    """There is nothing to release, but the protocol still requires close()."""
    calls: list[AudioChunk] = []
    sink = CallableSink(calls.append)

    sink.close()
    sink.close()                                   # idempotent
    assert calls == [], "close() must not invoke the wrapped function"


def test_callable_sink_propagates_the_functions_exception() -> None:
    """CallableSink is a plain adapter; it must not swallow errors."""
    def broken(chunk: AudioChunk) -> None:
        raise Boom("the wrapped function failed")

    with pytest.raises(Boom):
        CallableSink(broken).on_chunk(make_chunk(signal(10), 0, 0))


def test_callable_sink_satisfies_the_chunksink_protocol() -> None:
    assert isinstance(CallableSink(lambda c: None), ChunkSink)


# ==========================================================================
# the ChunkSink protocol itself
# ==========================================================================

def test_chunksink_protocol_is_runtime_checkable() -> None:
    assert isinstance(RecordingSink("a", []), ChunkSink), (
        "any object with on_chunk() and close() must satisfy ChunkSink"
    )


def test_a_bare_object_does_not_satisfy_chunksink() -> None:
    class NotASink:
        pass

    assert not isinstance(NotASink(), ChunkSink)


def test_an_object_with_only_on_chunk_does_not_satisfy_chunksink() -> None:
    class HalfASink:
        def on_chunk(self, chunk: AudioChunk) -> None:
            pass

    assert not isinstance(HalfASink(), ChunkSink), (
        "close() is part of the protocol -- the pipeline calls it on stop()"
    )


def test_wav_recorder_close_releases_the_file_handle(tmp_path: Any) -> None:
    """close() must actually close the WAV, not merely set a flag.

    Windows will not allow an open file to be replaced, so os.replace is a
    direct probe for a leaked handle. Under CPython refcounting a leak is
    invisible until something holds a reference -- which is exactly when it
    would start truncating recordings in the field.
    """
    import os

    path = tmp_path / "recorded.wav"
    sink = WavRecorderSink(path, 16000)
    sink.on_chunk(
        AudioChunk(np.full(128, 0.25, np.float32), 0, 0, 16000)
    )
    sink.close()

    other = tmp_path / "other.wav"
    other.write_bytes(b"placeholder")
    os.replace(other, path)  # raises PermissionError on Windows if still open
