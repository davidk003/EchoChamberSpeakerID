"""End-to-end tests for echochamber.audio.pipeline, from the step-3 contract.

Written from the *spec*, not the implementation.  This is the file that proves
the whole ingestion path works: a WAV goes in at one end, and the exact windows
the architecture promises come out the other, through a real ring buffer, a real
chunker thread, a real bounded queue and a real consumer thread.

The headline test is :func:`test_full_file_fidelity`: replay a known ramp with
``realtime=False`` and assert that **every** full window arrives, that chunk
``k`` starts at absolute frame ``k*H``, and that its samples are the file's
samples.  Everything else -- backpressure, reconfiguration, lifecycle, error
propagation -- is about making sure that guarantee survives contact with a slow
or broken consumer.

Determinism strategy: sample rates are chosen so ``1 ms == 1 frame`` and the
geometry is readable at a glance; the ring is sized far larger than the test
file so no overrun can perturb the grid; waits are event-driven with multi-second
timeouts.  Every pipeline is stopped in fixture teardown so a wedged thread can
never hang the suite.
"""

from __future__ import annotations

import threading
import time
import wave
from typing import Any, Callable, Iterator

import numpy as np
import pytest

from echochamber.audio.pipeline import AudioPipeline
from echochamber.audio.ringbuffer import RingBuffer
from echochamber.audio.sources.file_source import FileSource
from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats
from echochamber.config import AudioConfig


# Generous: every wait below is event-driven, so these only bound failures.
TIMEOUT = 20.0

# How long to allow for something to *not* happen (proving absence).
QUIET_S = 0.35

# int16 storage -> float32 tolerance (the /32767 vs /32768 convention is free).
TOL = 1e-4


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def ramp_int16(n: int) -> np.ndarray:
    """A distinct, strictly increasing int16 ramp -- off-by-ones are visible."""
    step = max(1, 60_000 // max(n, 1))
    idx = np.arange(n, dtype=np.int64)
    return ((idx * step) % 60_000 - 30_000).astype(np.int64)


def write_wav(path: Any, data: np.ndarray, sample_rate: int) -> Any:
    """Write a 16-bit PCM WAV; ``data`` is ``(n,)`` mono or ``(n, channels)``."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(arr.shape[1])
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(arr.astype("<i2").tobytes())
    return path


def expected_mono(data: np.ndarray) -> np.ndarray:
    """The mono float32 signal the pipeline must carry for ``data``."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return (arr / 32768.0).mean(axis=1).astype(np.float32)


def make_config(
    window_frames: int,
    hop_frames: int,
    sample_rate: int = 1000,
    ring_seconds: float = 10.0,
    queue_max: int = 8,
    drop_policy: DropPolicy = DropPolicy.BLOCK,
) -> AudioConfig:
    """Build a config whose window/hop are exactly the requested frame counts.

    At the default ``sample_rate=1000``, ``ms_to_frames(x, 1000) == x``, so 1 ms
    is 1 frame and the geometry of each test reads directly off the arguments.
    """
    def to_ms(frames: int) -> float | int:
        ms = frames * 1000 / sample_rate
        return int(ms) if ms == int(ms) else ms

    cfg = AudioConfig(
        sample_rate=sample_rate,
        window_ms=to_ms(window_frames),  # type: ignore[arg-type]
        hop_ms=to_ms(hop_frames),  # type: ignore[arg-type]
        ring_seconds=ring_seconds,
        queue_max=queue_max,
        drop_policy=drop_policy,
    )
    assert cfg.window_frames == window_frames, "test helper: window did not round to W"
    assert cfg.hop_frames == hop_frames, "test helper: hop did not round to H"
    return cfg


def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Poll ``pred`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def n_windows(n_frames: int, w: int, h: int) -> int:
    """How many *full* windows fit in ``n_frames``; partial tails are dropped."""
    if n_frames < w:
        return 0
    return (n_frames - w) // h + 1


class Boom(Exception):
    """Sentinel raised by a deliberately broken sink."""


class CollectingSink:
    """Thread-safe ChunkSink recording everything the consumer thread delivers.

    ``delay`` makes it the "slow ML stage" the bounded queue exists to isolate;
    ``raises`` makes it the broken sink that must not deadlock ``stop()``.  Both
    take effect *after* the chunk is recorded, so the record is always complete.
    """

    def __init__(self, delay: float = 0.0, raises: BaseException | None = None) -> None:
        self.chunks: list[AudioChunk] = []
        self.close_calls = 0
        self.delay = delay
        self.raises = raises
        self._cv = threading.Condition()

    def on_chunk(self, chunk: AudioChunk) -> None:
        with self._cv:
            self.chunks.append(chunk)
            self._cv.notify_all()
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises

    def close(self) -> None:
        with self._cv:
            self.close_calls += 1
            self._cv.notify_all()

    @property
    def count(self) -> int:
        with self._cv:
            return len(self.chunks)

    def snapshot(self) -> list[AudioChunk]:
        with self._cv:
            return list(self.chunks)

    def wait_for_count(self, n: int, timeout: float = TIMEOUT) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while len(self.chunks) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return True


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def make_pipeline() -> Iterator[Callable[..., AudioPipeline]]:
    """Factory registering every pipeline for guaranteed teardown.

    Teardown stops each pipeline with a bounded timeout, so a wedged consumer,
    chunker or replay thread cannot hang the rest of the suite.
    """
    created: list[AudioPipeline] = []

    def _make(
        config: AudioConfig,
        sink: Any,
        source_factory: Callable[..., Any],
        stats: StreamStats | None = None,
    ) -> AudioPipeline:
        p = AudioPipeline(config, sink, source_factory, stats=stats)
        created.append(p)
        return p

    yield _make

    for p in created:
        try:
            p.stop(timeout=5.0)
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


def file_factory(
    path: Any,
    blocksize: int = 64,
    realtime: bool = False,
    loop: bool = False,
    stats: StreamStats | None = None,
) -> Callable[[Callable[[np.ndarray], None], StreamStats], FileSource]:
    """A SourceFactory: the pipeline hands it ``ring.write`` and its StreamStats.

    An explicit ``stats=`` argument overrides the pipeline's, so a test can watch
    the source's counters in isolation; otherwise the pipeline's instance is
    passed straight through, which is the normal wiring.
    """
    def _factory(
        on_audio: Callable[[np.ndarray], None], pipeline_stats: StreamStats
    ) -> FileSource:
        return FileSource(
            path,
            on_audio,
            blocksize=blocksize,
            realtime=realtime,
            loop=loop,
            stats=stats if stats is not None else pipeline_stats,
        )

    return _factory


# ==========================================================================
# wiring
# ==========================================================================

def test_pipeline_wires_ring_chunker_source_and_sink(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """Everything is constructed in __init__, before any thread runs."""
    cfg = make_config(100, 40)
    path = write_wav(tmp_path / "wire.wav", ramp_int16(500), cfg.sample_rate)
    sink = CollectingSink()
    stats = StreamStats()

    p = make_pipeline(cfg, sink, file_factory(path, stats=stats), stats)

    assert p.config is cfg, "config must expose the object passed to __init__"
    assert p.stats is stats, "the pipeline must share the caller's StreamStats"
    assert p.sink is sink, "sink must expose the caller's sink"
    assert isinstance(p.ring, RingBuffer)
    assert p.ring.capacity == cfg.ring_frames, (
        f"the ring must be sized from config.ring_frames ({cfg.ring_frames}), "
        f"got {p.ring.capacity}"
    )
    assert isinstance(p.source, FileSource), (
        "the source must be built by source_factory in __init__, so that "
        "source.sample_rate is available before start()"
    )
    assert p.chunker is not None
    assert p.is_running is False
    assert p.error is None


def test_stats_defaults_to_a_fresh_streamstats(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 40)
    path = write_wav(tmp_path / "st.wav", ramp_int16(300), cfg.sample_rate)
    p = make_pipeline(cfg, CollectingSink(), file_factory(path), None)

    assert isinstance(p.stats, StreamStats)
    assert p.stats.chunks_emitted == 0
    assert p.chunker.stats is p.stats, (
        "the chunker must share the pipeline's stats so the GUI sees one record"
    )


def test_the_source_never_sees_the_ring(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """The factory receives a writer callable, not the RingBuffer itself."""
    cfg = make_config(100, 40)
    path = write_wav(tmp_path / "w.wav", ramp_int16(300), cfg.sample_rate)
    seen: list[Any] = []

    def factory(
        on_audio: Callable[[np.ndarray], None], pipeline_stats: StreamStats
    ) -> FileSource:
        seen.append((on_audio, pipeline_stats))
        return FileSource(
            path, on_audio, blocksize=64, realtime=False, stats=pipeline_stats
        )

    p = make_pipeline(cfg, CollectingSink(), factory)

    assert len(seen) == 1, "source_factory must be called exactly once, in __init__"
    on_audio, pipeline_stats = seen[0]
    assert callable(on_audio), "the factory is handed a callable, not the ring"
    assert not isinstance(on_audio, RingBuffer)
    assert pipeline_stats is p.stats, (
        "the factory must receive the pipeline's own StreamStats, otherwise the "
        "source's frames_captured and level meters go nowhere"
    )


# ==========================================================================
# the headline test: full-file fidelity
# ==========================================================================

FIDELITY_CASES = [
    (100, 40, 1000, 64),      # 60 % overlap, short final block
    (100, 50, 1000, 128),     # 50 % overlap
    (100, 100, 1000, 64),     # H == W: no overlap at all
    (240, 80, 1000, 100),     # the shipping 3000/1000 ratio, scaled down
    (128, 1, 300, 64),        # pathological: a one-frame hop
    (64, 33, 777, 50),        # non-round everything
]


@pytest.mark.parametrize(("w", "h", "n", "blocksize"), FIDELITY_CASES)
def test_full_file_fidelity(
    w: int,
    h: int,
    n: int,
    blocksize: int,
    tmp_path: Any,
    make_pipeline: Callable[..., AudioPipeline],
) -> None:
    """Every full window in the file must reach the sink, sample-exact.

    This is the assertion the entire step exists to satisfy.  ``BLOCK`` policy
    (correct for file replay) guarantees nothing is dropped, and the ring is
    sized far larger than the file so no overrun can perturb the grid.
    """
    cfg = make_config(w, h, drop_policy=DropPolicy.BLOCK)
    data = ramp_int16(n)
    path = write_wav(tmp_path / "fid.wav", data, cfg.sample_rate)
    want = expected_mono(data)
    expected_chunks = n_windows(n, w, h)
    assert expected_chunks >= 3, "test setup: the file must yield several windows"

    stats = StreamStats()
    sink = CollectingSink()
    p = make_pipeline(
        cfg, sink, file_factory(path, blocksize=blocksize, stats=stats), stats
    )

    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True, (
        f"W={w} H={h}: the replay never finished draining; sink got {sink.count} "
        f"of {expected_chunks} chunks, error={p.error!r}"
    )
    assert p.stop(timeout=10.0) is True, f"stop() did not complete: {p.error!r}"
    assert p.error is None, f"W={w} H={h}: pipeline error {p.error!r}"

    chunks = sink.snapshot()
    assert len(chunks) == expected_chunks, (
        f"W={w} H={h} n={n}: expected exactly {expected_chunks} full windows, "
        f"got {len(chunks)} -- wait_until_finished() then stop() must lose "
        "nothing, including the tail"
    )

    for k, chunk in enumerate(chunks):
        assert chunk.seq == k, f"W={w} H={h}: seq must be the chunk index"
        assert chunk.start_frame == k * h, (
            f"W={w} H={h} chunk {k}: start_frame {chunk.start_frame} != k*H "
            f"({k * h})"
        )
        assert chunk.n_frames == w, (
            f"W={w} H={h} chunk {k}: len(samples) {chunk.n_frames} != W {w}"
        )
        assert chunk.sample_rate == cfg.sample_rate
        assert chunk.discontinuous is False, (
            f"W={w} H={h} chunk {k}: nothing was lost or reconfigured, so no "
            "chunk may be flagged discontinuous"
        )
        segment = want[k * h : k * h + w]
        assert np.allclose(chunk.samples, segment, atol=TOL), (
            f"W={w} H={h} chunk {k}: samples do not match the file at frames "
            f"[{k * h}, {k * h + w}); first mismatch at index "
            f"{int(np.argmax(np.abs(chunk.samples - segment) > TOL))}"
        )

    overlap = w - h
    if overlap > 0:
        for k in range(len(chunks) - 1):
            assert np.array_equal(chunks[k].samples[h:], chunks[k + 1].samples[:overlap]), (
                f"W={w} H={h}: chunks {k}/{k + 1} must share exactly {overlap} "
                "element-wise identical samples"
            )


def test_stats_match_the_file_and_the_delivered_chunks(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """With nothing dropped, the counters and the sink must tell one story."""
    w, h, n = 100, 40, 1200
    cfg = make_config(w, h, drop_policy=DropPolicy.BLOCK)
    path = write_wav(tmp_path / "stats.wav", ramp_int16(n), cfg.sample_rate)

    stats = StreamStats()
    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64, stats=stats), stats)

    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True

    expected = n_windows(n, w, h)
    assert stats.frames_captured == n, (
        f"frames_captured must equal the file's {n} frames, got "
        f"{stats.frames_captured}"
    )
    assert stats.chunks_dropped == 0, "BLOCK policy must never drop a chunk"
    assert stats.overruns == 0, (
        "the ring is far larger than the file, so nothing may be overrun"
    )
    assert sink.count == expected, f"sink got {sink.count} of {expected} windows"
    assert stats.chunks_emitted == sink.count, (
        f"chunks_emitted ({stats.chunks_emitted}) must match the {sink.count} "
        "windows delivered when nothing was dropped"
    )
    assert stats.peak_level > 0.0, "the source must have set the level fields"


def test_stereo_file_is_downmixed_before_it_reaches_the_sink(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """Downmix happens once, in the source; chunks are always 1-D mono."""
    w, h, n = 100, 50, 800
    cfg = make_config(w, h, drop_policy=DropPolicy.BLOCK)
    data = np.stack([ramp_int16(n), -ramp_int16(n)], axis=1)
    path = write_wav(tmp_path / "stereo.wav", data, cfg.sample_rate)
    want = expected_mono(data)

    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))

    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True

    chunks = sink.snapshot()
    assert len(chunks) == n_windows(n, w, h)
    for k, chunk in enumerate(chunks):
        assert chunk.samples.ndim == 1 and chunk.samples.dtype == np.float32
        assert np.allclose(chunk.samples, want[k * h : k * h + w], atol=TOL)


def test_stop_closes_the_sink(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 50, drop_policy=DropPolicy.BLOCK)
    path = write_wav(tmp_path / "close.wav", ramp_int16(500), cfg.sample_rate)
    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))

    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert sink.close_calls == 0, "the sink must stay open until stop()"

    assert p.stop(timeout=10.0) is True
    assert sink.close_calls >= 1, "stop() must close the sink, releasing its files"


def test_wait_until_finished_returns_false_on_timeout(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """A looping source never finishes; the caller learns that from the return."""
    cfg = make_config(100, 50, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "loop.wav", ramp_int16(500), cfg.sample_rate)
    sink = CollectingSink()
    p = make_pipeline(
        cfg, sink, file_factory(path, blocksize=50, realtime=True, loop=True)
    )

    p.start()
    assert p.wait_until_finished(timeout=0.3) is False, (
        "a looping source never sets finished, so a bounded wait must return False"
    )
    assert p.stop(timeout=10.0) is True


# ==========================================================================
# sample-rate agreement
# ==========================================================================

def test_sample_rate_mismatch_raises_value_error_from_start(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """Resampling is out of scope, so a mismatched WAV must be loud, not silent."""
    cfg = make_config(100, 50, sample_rate=1000)
    path = write_wav(tmp_path / "wrongrate.wav", ramp_int16(2000), 2000)
    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))

    assert p.source.sample_rate == 2000, "test setup: the file must disagree"

    with pytest.raises(ValueError) as excinfo:
        p.start()
    message = str(excinfo.value)
    assert "2000" in message or "1000" in message, (
        f"the error must name the mismatched rates, got {message!r}"
    )

    assert p.is_running is False, "a rejected start() must leave nothing running"
    time.sleep(QUIET_S)
    assert sink.count == 0, "no audio may flow after a rejected start()"
    assert p.stop(timeout=5.0) is True, "stop() after a failed start() must be safe"


def test_matching_sample_rate_starts_normally(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    # make_config's "1 ms == 1 frame" identity only holds at sample_rate=1000,
    # so build this one directly: the point here is purely that the source rate
    # and the config rate agree, not the exact window geometry.
    cfg = AudioConfig(
        sample_rate=8000,
        window_ms=100,   # 800 frames
        hop_ms=50,       # 400 frames
        ring_seconds=10.0,
        queue_max=8,
        drop_policy=DropPolicy.BLOCK,
    )
    path = write_wav(tmp_path / "rate8k.wav", ramp_int16(4000), 8000)
    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=160))

    p.start()                                     # must not raise
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True
    assert sink.count > 0


# ==========================================================================
# lifecycle
# ==========================================================================

def test_start_twice_raises_runtimeerror(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 50, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "twice.wav", ramp_int16(4000), cfg.sample_rate)
    p = make_pipeline(
        cfg, CollectingSink(), file_factory(path, blocksize=50, realtime=True, loop=True)
    )

    p.start()
    with pytest.raises(RuntimeError):
        p.start()
    assert p.stop(timeout=10.0) is True


def test_stop_without_start_is_safe(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 50)
    path = write_wav(tmp_path / "nostart.wav", ramp_int16(500), cfg.sample_rate)
    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))

    assert p.stop(timeout=5.0) is True, (
        "stopping a never-started pipeline is a no-op returning True"
    )
    assert p.is_running is False
    assert sink.count == 0


def test_stop_is_idempotent(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 50, drop_policy=DropPolicy.BLOCK)
    path = write_wav(tmp_path / "idem.wav", ramp_int16(600), cfg.sample_rate)
    p = make_pipeline(cfg, CollectingSink(), file_factory(path, blocksize=64))

    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True
    assert p.stop(timeout=10.0) is True, "a second stop() must also return True"
    assert p.stop(timeout=10.0) is True, "and a third"
    assert p.is_running is False


def test_is_running_transitions_true_then_false(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 50, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "run.wav", ramp_int16(500), cfg.sample_rate)
    sink = CollectingSink()
    p = make_pipeline(
        cfg, sink, file_factory(path, blocksize=50, realtime=True, loop=True)
    )

    p.start()
    assert p.is_running is True, "is_running must be True while the threads run"
    assert sink.wait_for_count(1), "no chunk ever arrived"

    assert p.stop(timeout=10.0) is True
    assert p.is_running is False, "is_running must be False once stopped"


def test_stop_is_prompt_on_a_silent_pipeline(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """A file too short to fill a window leaves every thread parked; stop() still returns."""
    cfg = make_config(500, 250, drop_policy=DropPolicy.BLOCK)
    path = write_wav(tmp_path / "short.wav", ramp_int16(100), cfg.sample_rate)
    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))

    p.start()
    t0 = time.monotonic()
    assert p.stop(timeout=10.0) is True
    elapsed = time.monotonic() - t0

    assert elapsed < 5.0, f"stop() took {elapsed:.2f}s on an idle pipeline"
    assert sink.count == 0, "a sub-window file yields no full windows"
    assert p.error is None


def test_no_chunks_are_lost_at_startup(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """start() launches the consumer first, so window 0 can never be dropped."""
    w, h, n = 100, 50, 400
    cfg = make_config(w, h, queue_max=1, drop_policy=DropPolicy.BLOCK)
    data = ramp_int16(n)
    path = write_wav(tmp_path / "startup.wav", data, cfg.sample_rate)

    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))
    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True

    chunks = sink.snapshot()
    assert len(chunks) == n_windows(n, w, h)
    assert chunks[0].start_frame == 0 and chunks[0].seq == 0, (
        "the very first window must reach the sink"
    )
    assert np.allclose(chunks[0].samples, expected_mono(data)[:w], atol=TOL)


# ==========================================================================
# backpressure -- why the consumer thread exists
# ==========================================================================

def test_a_slow_sink_does_not_stall_the_chunker(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """The bounded queue + consumer thread isolate the chunker from the sink.

    A slow sink must cause *drops*, not a stalled chunker: a delayed chunker
    falls behind the ring's writer and the whole stream overruns instead of
    degrading gracefully.
    """
    w, h, n = 100, 20, 2000
    cfg = make_config(w, h, queue_max=2, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "slow.wav", ramp_int16(n), cfg.sample_rate)

    stats = StreamStats()
    sink = CollectingSink(delay=0.02)              # ~50 chunks/s at best
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64, stats=stats), stats)

    expected_windows = n_windows(n, w, h)
    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True, (
        f"the pipeline wedged behind the slow sink; emitted="
        f"{stats.chunks_emitted} delivered={sink.count} error={p.error!r}"
    )
    assert p.stop(timeout=10.0) is True, "a slow sink must not wedge stop()"

    assert p.error is None
    assert stats.chunks_emitted == expected_windows, (
        f"the chunker must keep producing all {expected_windows} windows "
        f"regardless of sink speed, got {stats.chunks_emitted}"
    )
    assert stats.chunks_dropped > 0, (
        "a sink 100x slower than the chunker with a 2-deep queue must cause "
        "drops under DROP_OLDEST -- zero drops means the chunker was stalled"
    )
    assert sink.count < stats.chunks_emitted, (
        "the slow sink cannot have seen every chunk if chunks were dropped"
    )
    assert stats.chunks_dropped + sink.count == stats.chunks_emitted, (
        f"every emitted chunk was either delivered ({sink.count}) or dropped "
        f"({stats.chunks_dropped}); emitted={stats.chunks_emitted}"
    )


def test_a_slow_sink_never_overruns_the_ring(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """Dropping at the queue is precisely what keeps the ring healthy."""
    w, h, n = 100, 20, 1500
    cfg = make_config(w, h, queue_max=2, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "slow2.wav", ramp_int16(n), cfg.sample_rate)

    stats = StreamStats()
    p = make_pipeline(
        cfg,
        CollectingSink(delay=0.02),
        file_factory(path, blocksize=64, stats=stats),
        stats,
    )

    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True
    assert stats.overruns == 0, (
        f"the ring holds {cfg.ring_frames} frames and the file is only {n}; a "
        "slow sink must not cause a ring overrun"
    )


def test_delivered_chunks_are_still_sample_exact_under_dropping(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """Dropping loses chunks; it must never corrupt the ones that survive."""
    w, h, n = 100, 20, 1000
    cfg = make_config(w, h, queue_max=1, drop_policy=DropPolicy.DROP_OLDEST)
    data = ramp_int16(n)
    path = write_wav(tmp_path / "drop.wav", data, cfg.sample_rate)
    want = expected_mono(data)

    stats = StreamStats()
    sink = CollectingSink(delay=0.01)
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64, stats=stats), stats)

    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True

    chunks = sink.snapshot()
    assert chunks, "at least some chunks must survive the dropping"
    seqs = [c.seq for c in chunks]
    assert seqs == sorted(seqs), f"surviving chunks must stay in order, got {seqs}"
    for chunk in chunks:
        start = chunk.start_frame
        assert start == chunk.seq * h, "start_frame must still be seq*H"
        assert np.allclose(chunk.samples, want[start : start + w], atol=TOL), (
            f"chunk seq={chunk.seq} was corrupted by the dropping"
        )


# ==========================================================================
# error propagation
# ==========================================================================

def test_a_raising_sink_does_not_deadlock_stop(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """The future ML stage will throw; stop() must still be able to tear down."""
    cfg = make_config(100, 50, queue_max=2, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "boom.wav", ramp_int16(1000), cfg.sample_rate)

    sink = CollectingSink(raises=Boom("the ML stage exploded"))
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))

    p.start()
    assert sink.wait_for_count(1), "the sink was never called"

    t0 = time.monotonic()
    result = p.stop(timeout=10.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 8.0, (
        f"stop() took {elapsed:.2f}s after the sink raised -- a broken sink must "
        "not deadlock teardown"
    )
    assert result is True, "every stage was bounded, so stop() must report success"
    assert p.is_running is False


def test_pipeline_error_surfaces_a_consumer_failure(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 50, queue_max=2, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "cboom.wav", ramp_int16(1000), cfg.sample_rate)

    boom = Boom("consumer side failure")
    sink = CollectingSink(raises=boom)
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=64))

    p.start()
    assert wait_until(lambda: p.error is not None), (
        "an exception raised by the sink runs on the consumer thread and must "
        "surface through pipeline.error"
    )
    assert p.error is boom, f"expected the sink's exception, got {p.error!r}"
    assert p.stop(timeout=10.0) is True


def test_pipeline_error_surfaces_a_source_failure(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """A source whose blocks are larger than the ring fails on its own thread."""
    cfg = make_config(100, 50, ring_seconds=1.0, drop_policy=DropPolicy.DROP_OLDEST)
    assert cfg.ring_frames == 1000, "test setup: the ring must be 1000 frames"
    path = write_wav(tmp_path / "sboom.wav", ramp_int16(4000), cfg.sample_rate)

    sink = CollectingSink()
    p = make_pipeline(cfg, sink, file_factory(path, blocksize=1500))

    p.start()
    assert wait_until(lambda: p.error is not None), (
        "a source-thread exception must surface through pipeline.error"
    )
    assert isinstance(p.error, ValueError), (
        f"ring.write rejects an oversized block with ValueError, got {p.error!r}"
    )
    assert p.source.error is p.error, "pipeline.error must be the source's error"
    assert p.stop(timeout=10.0) is True, "a dead source must not wedge stop()"


def test_error_is_none_on_a_healthy_run(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg = make_config(100, 50, drop_policy=DropPolicy.BLOCK)
    path = write_wav(tmp_path / "healthy.wav", ramp_int16(1000), cfg.sample_rate)
    p = make_pipeline(cfg, CollectingSink(), file_factory(path, blocksize=64))

    assert p.error is None, "a freshly constructed pipeline has no error"
    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True
    assert p.error is None, f"a clean replay must record no error: {p.error!r}"


# ==========================================================================
# reconfigure()
# ==========================================================================

def test_reconfigure_mid_replay_applies_the_new_geometry(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """A looping realtime source gives the test all the time it needs."""
    cfg1 = make_config(100, 50, drop_policy=DropPolicy.DROP_OLDEST)
    cfg2 = make_config(200, 50, drop_policy=DropPolicy.DROP_OLDEST)
    path = write_wav(tmp_path / "recfg.wav", ramp_int16(500), cfg1.sample_rate)

    sink = CollectingSink()
    p = make_pipeline(
        cfg1, sink, file_factory(path, blocksize=50, realtime=True, loop=True)
    )

    p.start()
    assert sink.wait_for_count(3), (
        f"the pre-reconfigure stream never got going; error={p.error!r}"
    )
    assert all(c.n_frames == 100 for c in sink.snapshot()[:3]), (
        "chunks before the reconfigure must use the OLD window length"
    )

    p.reconfigure(cfg2)
    assert p.config is cfg2, "config must expose the newly applied object"
    assert p.chunker.config is cfg2, "reconfigure must delegate to the chunker"

    assert wait_until(
        lambda: any(c.n_frames == 200 for c in sink.snapshot())
    ), (
        f"no chunk arrived with the NEW 200-frame window; sizes seen: "
        f"{sorted({c.n_frames for c in sink.snapshot()})}, error={p.error!r}"
    )

    new_chunks = [c for c in sink.snapshot() if c.n_frames == 200]
    assert new_chunks[0].discontinuous is True, (
        "the first chunk under the new geometry starts at the write head, so it "
        "does not follow the previous chunk's audio"
    )
    assert p.stop(timeout=10.0) is True
    assert p.error is None


def test_reconfigure_before_start_is_honoured(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    cfg1 = make_config(100, 50, drop_policy=DropPolicy.BLOCK)
    cfg2 = make_config(60, 60, drop_policy=DropPolicy.BLOCK)
    path = write_wav(tmp_path / "prestart.wav", ramp_int16(600), cfg1.sample_rate)

    sink = CollectingSink()
    p = make_pipeline(cfg1, sink, file_factory(path, blocksize=64))

    p.reconfigure(cfg2)
    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True

    chunks = sink.snapshot()
    assert chunks, "the reconfigured pipeline must still produce chunks"
    assert all(c.n_frames == 60 for c in chunks), (
        "a reconfigure before start() is simply the starting geometry"
    )


def test_reconfigure_to_a_window_larger_than_the_ring_raises_value_error(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """The ring is not resized; an unreadable window must be rejected loudly."""
    cfg = make_config(100, 50, ring_seconds=1.0, drop_policy=DropPolicy.DROP_OLDEST)
    assert cfg.ring_frames == 1000, "test setup: the ring must be 1000 frames"
    path = write_wav(tmp_path / "big.wav", ramp_int16(2000), cfg.sample_rate)

    sink = CollectingSink()
    p = make_pipeline(
        cfg, sink, file_factory(path, blocksize=50, realtime=True, loop=True)
    )
    p.start()
    assert sink.wait_for_count(1), f"the stream never got going; error={p.error!r}"

    oversized = make_config(
        2000, 100, ring_seconds=10.0, drop_policy=DropPolicy.DROP_OLDEST
    )
    assert oversized.window_frames > p.ring.capacity, "test setup"

    with pytest.raises(ValueError, match="exceeds ring capacity"):
        p.reconfigure(oversized)

    assert p.config is cfg, (
        "a rejected reconfigure must leave the running config untouched"
    )
    before = sink.count
    assert wait_until(lambda: sink.count > before), (
        "the stream must survive a rejected reconfigure"
    )
    assert p.stop(timeout=10.0) is True
    assert p.error is None


# ==========================================================================
# stats injection
#
# The pipeline owns the StreamStats, but only the SOURCE ever sees raw audio,
# so only it can fill in frames_captured and the level meters. If the pipeline
# does not inject its stats into the factory, a caller who supplies no stats
# gets a pipeline whose meters silently never move -- which looks like a dead
# input device rather than a wiring mistake.
# ==========================================================================


def test_default_constructed_pipeline_still_gets_capture_stats(
    tmp_path: Any, make_pipeline: Callable[..., AudioPipeline]
) -> None:
    """No stats argument anywhere: meters must still work."""
    cfg = make_config(100, 50)
    frames = 600
    path = write_wav(tmp_path / "meters.wav", ramp_int16(frames), cfg.sample_rate)

    # Note: no stats= passed to file_factory OR to the pipeline.
    p = make_pipeline(cfg, CollectingSink(), file_factory(path, blocksize=64))
    p.start()
    assert p.wait_until_finished(timeout=TIMEOUT) is True
    assert p.stop(timeout=10.0) is True

    assert p.stats.frames_captured == frames, (
        f"source must report into the pipeline's stats; got "
        f"{p.stats.frames_captured}, expected {frames}"
    )
    assert p.stats.peak_level > 0.0, "a non-silent ramp must move the peak meter"
    assert p.stats.rms_level > 0.0, "a non-silent ramp must move the RMS meter"
