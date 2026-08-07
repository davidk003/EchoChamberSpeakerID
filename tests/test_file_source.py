"""Tests for echochamber.audio.sources.file_source, from the step-3 contract.

Written from the *spec*, not the implementation.  ``FileSource`` is what makes
the whole pipeline testable without a microphone, so these tests hold it to a
higher standard than "it plays something": every sample the file contains must
arrive, in order, mono, float32, in [-1, 1], with the final short block handled
correctly.

WAV files are generated with stdlib :mod:`wave` into pytest's ``tmp_path``.  The
canonical signal is a strictly increasing integer ramp, so a dropped, repeated
or reordered frame is a visible mismatch rather than plausible-looking audio.

Determinism strategy: nothing polls a clock except the two tests that exist to
measure pacing.  Everything else waits on ``finished`` or on an event-driven
predicate with a multi-second timeout, and every source is stopped in fixture
teardown, so a wedged replay thread cannot hang the suite.
"""

from __future__ import annotations

import struct
import threading
import time
import wave
from typing import Any, Callable, Iterator

import numpy as np
import pytest

from echochamber.audio.sources.file_source import FileSource
from echochamber.audio.types import StreamStats


# Generous: every wait below is event-driven, so these only bound failures.
TIMEOUT = 10.0

# How long to allow for something to *not* happen (proving absence).
QUIET_S = 0.35

# Per-sample tolerance by sample width.  The contract fixes the range ([-1, 1])
# but not the exact normalising constant, so these allow both the /2**(n-1) and
# /(2**(n-1) - 1) conventions plus a little rounding slack.
TOL = {1: 0.02, 2: 1e-4, 3: 1e-4, 4: 1e-4}


# --------------------------------------------------------------------------
# WAV helpers
# --------------------------------------------------------------------------

def ramp_int(n: int, sampwidth: int = 2, offset: int = 0) -> np.ndarray:
    """A strictly increasing storage-domain ramp of ``n`` frames.

    Strictly increasing and distinct: an off-by-one in the block loop shows up
    as a mismatch instead of as harmless-looking audio.
    """
    if sampwidth == 1:                                   # unsigned 8-bit
        step = max(1, 240 // max(n, 1))
        return ((np.arange(offset, offset + n) * step) % 240 + 8).astype(np.int64)
    if sampwidth == 2:
        step = max(1, 60_000 // max(n, 1))
        idx = np.arange(offset, offset + n, dtype=np.int64)
        return ((idx * step) % 60_000 - 30_000).astype(np.int64)
    step = max(1, (4 * 10**9) // max(n, 1))
    idx = np.arange(offset, offset + n, dtype=np.int64)
    return ((idx * step) % (4 * 10**9) - 2 * 10**9).astype(np.int64)


def write_wav(
    path: Any,
    data: np.ndarray,
    sample_rate: int,
    sampwidth: int = 2,
) -> Any:
    """Write ``data`` (shape ``(n,)`` mono or ``(n, channels)``) as a PCM WAV."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    channels = arr.shape[1]

    if sampwidth == 1:
        raw = arr.astype(np.uint8).tobytes()
    elif sampwidth == 2:
        raw = arr.astype("<i2").tobytes()
    elif sampwidth == 4:
        raw = arr.astype("<i4").tobytes()
    else:
        raise AssertionError(f"test helper does not write sampwidth={sampwidth}")

    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(raw)
    return path


def write_24bit_wav(path: Any, n_frames: int, sample_rate: int) -> Any:
    """Write a *valid* PCM WAV at an unsupported width (24-bit)."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(sample_rate)
        w.writeframes(bytes(n_frames * 3))
    return path


def write_ieee_float_wav(path: Any, n_frames: int, sample_rate: int) -> Any:
    """Write a valid RIFF/WAVE file whose format tag is not PCM.

    ``WAVE_FORMAT_IEEE_FLOAT`` (3) is the friendliest non-PCM case: the file is
    structurally sound, so anything that rejects it is rejecting the *format*.
    """
    payload = struct.pack("<%df" % n_frames, *([0.25] * n_frames))
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    body = (
        b"fmt " + struct.pack("<I", len(fmt)) + fmt
        + b"data" + struct.pack("<I", len(payload)) + payload
    )
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    return path


def expected_mono(data: np.ndarray, sampwidth: int = 2) -> np.ndarray:
    """The mono float32 signal a correct FileSource must produce from ``data``."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if sampwidth == 1:
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = arr / 32768.0
    else:
        arr = arr / 2147483648.0
    return arr.mean(axis=1).astype(np.float32)


# --------------------------------------------------------------------------
# collector / fixtures
# --------------------------------------------------------------------------

class Blocks:
    """Thread-safe ``on_audio`` callback recording every block it is handed.

    Also asserts the invariants that must hold for *every* block, so a bad
    dtype or a 2-D stereo block fails the test that produced it rather than
    some later comparison.
    """

    def __init__(self, hook: Callable[[int, np.ndarray], None] | None = None) -> None:
        self.blocks: list[np.ndarray] = []
        self.violations: list[str] = []
        self._cv = threading.Condition()
        self._hook = hook

    def __call__(self, block: np.ndarray) -> None:
        if block.ndim != 1:
            self.violations.append(f"block ndim={block.ndim}, must be 1-D mono")
        if block.dtype != np.float32:
            self.violations.append(f"block dtype={block.dtype}, must be float32")
        with self._cv:
            self.blocks.append(np.array(block, copy=True))
            index = len(self.blocks) - 1
            self._cv.notify_all()
        if self._hook is not None:
            self._hook(index, block)

    @property
    def count(self) -> int:
        with self._cv:
            return len(self.blocks)

    @property
    def frames(self) -> int:
        with self._cv:
            return sum(len(b) for b in self.blocks)

    def snapshot(self) -> list[np.ndarray]:
        with self._cv:
            return list(self.blocks)

    def concat(self) -> np.ndarray:
        blocks = self.snapshot()
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks)

    def wait_for_frames(self, n: int, timeout: float = TIMEOUT) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while sum(len(b) for b in self.blocks) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
            return True


def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Poll ``pred`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


@pytest.fixture
def make_source() -> Iterator[Callable[..., FileSource]]:
    """Factory registering every FileSource for guaranteed teardown."""
    created: list[FileSource] = []

    def _make(path: Any, on_audio: Callable[[np.ndarray], None], **kwargs: Any) -> FileSource:
        src = FileSource(path, on_audio, **kwargs)
        created.append(src)
        return src

    yield _make

    for src in created:
        try:
            src.stop(timeout=5.0)
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


# ==========================================================================
# header parsing -- before start()
# ==========================================================================

@pytest.mark.parametrize("sample_rate", [8000, 16_000, 44_100, 1000])
def test_sample_rate_is_available_before_start(
    sample_rate: int, tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """The pipeline compares source.sample_rate to the config *before* start()."""
    path = write_wav(tmp_path / "hdr.wav", ramp_int(500), sample_rate)
    src = make_source(path, Blocks())

    assert src.sample_rate == sample_rate, (
        "sample_rate must come from the file header, read in __init__"
    )


@pytest.mark.parametrize("channels", [1, 2, 3])
def test_channels_is_the_source_channel_count_before_downmix(
    channels: int, tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    data = np.stack([ramp_int(400, offset=c * 50) for c in range(channels)], axis=1)
    path = write_wav(tmp_path / "ch.wav", data, 16_000)
    src = make_source(path, Blocks())

    assert src.channels == channels, (
        "channels reports the SOURCE channel count (pre-downmix), not 1"
    )


def test_header_is_read_without_starting_a_thread(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "quiet.wav", ramp_int(500), 16_000)
    blocks = Blocks()
    src = make_source(path, blocks)

    assert src.is_running is False, "no thread may run before start()"
    assert src.error is None
    assert src.finished.is_set() is False
    time.sleep(QUIET_S)
    assert blocks.count == 0, "__init__ must not emit audio"


def test_finished_is_a_threading_event(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "ev.wav", ramp_int(100), 16_000)
    src = make_source(path, Blocks())
    assert isinstance(src.finished, threading.Event)


# ==========================================================================
# format support
# ==========================================================================

@pytest.mark.parametrize("sampwidth", [1, 2, 4])
def test_all_supported_pcm_widths_load_and_produce_mono_float32(
    sampwidth: int, tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    n = 1000
    data = ramp_int(n, sampwidth=sampwidth)
    path = write_wav(tmp_path / f"pcm{sampwidth}.wav", data, 16_000, sampwidth)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=128, realtime=False)
    src.start()
    assert src.finished.wait(TIMEOUT), f"{sampwidth * 8}-bit replay never finished"
    assert src.error is None, f"{sampwidth * 8}-bit PCM must load: {src.error!r}"

    got = blocks.concat()
    assert blocks.violations == [], blocks.violations
    assert len(got) == n, f"{sampwidth * 8}-bit: got {len(got)} of {n} frames"
    assert got.dtype == np.float32
    assert np.all(np.abs(got) <= 1.0), (
        f"{sampwidth * 8}-bit samples must be normalised into [-1, 1]; "
        f"max |x| = {float(np.max(np.abs(got)))}"
    )
    assert np.allclose(got, expected_mono(data, sampwidth), atol=TOL[sampwidth]), (
        f"{sampwidth * 8}-bit PCM was not converted correctly"
    )


def test_24_bit_pcm_is_rejected_with_a_value_error(tmp_path: Any) -> None:
    """Only 8/16/32-bit are supported; 24-bit is a clear, loud caller error."""
    path = write_24bit_wav(tmp_path / "pcm24.wav", 500, 16_000)
    with pytest.raises(ValueError) as excinfo:
        FileSource(path, Blocks())
    assert str(excinfo.value), "the ValueError must carry a clear message"


def test_non_pcm_file_is_rejected_with_a_value_error(tmp_path: Any) -> None:
    """A compressed / float-format WAV must raise ValueError, not leak wave.Error."""
    path = write_ieee_float_wav(tmp_path / "float.wav", 500, 16_000)
    with pytest.raises(ValueError):
        FileSource(path, Blocks())


def test_a_missing_file_fails_at_construction(tmp_path: Any) -> None:
    with pytest.raises((OSError, ValueError)):
        FileSource(tmp_path / "nope.wav", Blocks())


# ==========================================================================
# sample fidelity
# ==========================================================================

def test_every_sample_arrives_in_order_and_matches_the_file(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """The whole point of FileSource: replay is bit-for-bit faithful."""
    n = 4321
    data = ramp_int(n)
    path = write_wav(tmp_path / "ramp.wav", data, 16_000)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=256, realtime=False)
    src.start()
    assert src.finished.wait(TIMEOUT), "the replay never reached EOF"

    got = blocks.concat()
    assert blocks.violations == [], blocks.violations
    assert len(got) == n, f"expected all {n} frames, got {len(got)}"

    want = expected_mono(data)
    assert np.allclose(got, want, atol=TOL[2]), (
        "the concatenated blocks are not the file's samples in order; first "
        f"mismatch at frame {int(np.argmax(np.abs(got - want) > TOL[2]))}"
    )
    assert np.all(np.diff(got) > 0), (
        "the file is a strictly increasing ramp, so the replay must be too -- "
        "a repeated or reordered block breaks this"
    )


@pytest.mark.parametrize(
    ("n", "blocksize"),
    [
        (1024, 256),      # exact multiple: no short block
        (1000, 256),      # short final block of 232
        (1000, 999),      # short final block of 1
        (1000, 1),        # every block is one frame
        (7, 1024),        # the whole file is shorter than one block
        (1, 16),          # a single-frame file
    ],
)
def test_block_sizes_including_the_short_final_block(
    n: int, blocksize: int, tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """The final block may be shorter; it must never be padded or dropped."""
    data = ramp_int(n)
    path = write_wav(tmp_path / "blk.wav", data, 16_000)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=blocksize, realtime=False)
    src.start()
    assert src.finished.wait(TIMEOUT), f"n={n} bs={blocksize}: never finished"

    sizes = [len(b) for b in blocks.snapshot()]
    full, tail = divmod(n, blocksize)
    expected_sizes = [blocksize] * full + ([tail] if tail else [])
    assert sizes == expected_sizes, (
        f"n={n} blocksize={blocksize}: block sizes must be {expected_sizes}, "
        f"got {sizes}"
    )
    assert sum(sizes) == n, "the short tail must be emitted, not padded or dropped"
    assert np.allclose(blocks.concat(), expected_mono(data), atol=TOL[2])


def test_stereo_is_downmixed_by_averaging_the_channels(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """Downmix is the source's job; on_audio only ever sees 1-D mono."""
    n = 800
    left = ramp_int(n)
    right = -ramp_int(n)                      # averages to (l + r) / 2, not to l
    data = np.stack([left, right], axis=1)
    path = write_wav(tmp_path / "stereo.wav", data, 16_000)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=100, realtime=False)
    assert src.channels == 2, "channels reports the file's channel count"

    src.start()
    assert src.finished.wait(TIMEOUT), "the stereo replay never finished"

    got = blocks.concat()
    assert blocks.violations == [], blocks.violations
    assert len(got) == n
    assert np.allclose(got, expected_mono(data), atol=TOL[2]), (
        "stereo must be downmixed by AVERAGING the channels"
    )
    assert not np.allclose(got, expected_mono(left), atol=TOL[2]), (
        "the downmix must not simply take the left channel"
    )


def test_three_channel_downmix_is_the_mean(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    n = 600
    data = np.stack(
        [ramp_int(n), np.zeros(n, dtype=np.int64), -ramp_int(n) // 2], axis=1
    )
    path = write_wav(tmp_path / "tri.wav", data, 16_000)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=64, realtime=False)
    src.start()
    assert src.finished.wait(TIMEOUT)

    assert np.allclose(blocks.concat(), expected_mono(data), atol=TOL[2]), (
        "an N-channel downmix must be the mean of all N channels"
    )


def test_full_scale_samples_stay_inside_the_unit_range(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """A file that hits int16 full scale must not overshoot [-1, 1]."""
    data = np.array([-32768, -32767, 0, 32767, 32767, -32768], dtype=np.int64)
    path = write_wav(tmp_path / "fs.wav", data, 16_000)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=4, realtime=False)
    src.start()
    assert src.finished.wait(TIMEOUT)

    got = blocks.concat()
    assert len(got) == 6
    assert np.all(got >= -1.0) and np.all(got <= 1.0), (
        f"samples must be clamped into [-1, 1], got {got}"
    )
    assert got[3] > 0.9, "int16 full scale must map near +1.0, not near 0"
    assert got[0] < -0.9, "int16 negative full scale must map near -1.0"


# ==========================================================================
# pacing
# ==========================================================================

def test_realtime_false_replays_much_faster_than_the_audio_clock(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    sample_rate = 8000
    n = 8000                                    # 1.0 s of audio
    path = write_wav(tmp_path / "fast.wav", ramp_int(n), sample_rate)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=160, realtime=False)

    t0 = time.monotonic()
    src.start()
    assert src.finished.wait(TIMEOUT), "the replay never finished"
    elapsed = time.monotonic() - t0

    assert blocks.frames == n, "realtime=False must still deliver every frame"
    assert elapsed < 0.5, (
        f"1.0 s of audio replayed in {elapsed:.2f}s with realtime=False; it must "
        "run as fast as possible, with no per-block sleeping"
    )


def test_realtime_true_paces_to_the_audio_clock(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """A short file, so the whole test stays quick; 70 % is a loose floor."""
    sample_rate = 8000
    duration = 0.3
    n = int(sample_rate * duration)             # 2400 frames
    path = write_wav(tmp_path / "paced.wav", ramp_int(n), sample_rate)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=160, realtime=True)

    t0 = time.monotonic()
    src.start()
    assert src.finished.wait(TIMEOUT), "the paced replay never finished"
    elapsed = time.monotonic() - t0

    assert blocks.frames == n, "pacing must not cost any frames"
    assert elapsed >= 0.7 * duration, (
        f"{duration}s of audio replayed in {elapsed:.3f}s with realtime=True; "
        "the source must pace to the audio clock, not run flat out"
    )
    assert elapsed < 5.0, (
        f"{duration}s of audio took {elapsed:.2f}s -- the pacing schedule must be "
        "absolute (start_t + frames/sr), not a per-block sleep that accumulates"
    )


# ==========================================================================
# EOF, looping, finished
# ==========================================================================

def test_finished_is_set_at_eof_when_not_looping(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "eof.wav", ramp_int(500), 16_000)
    blocks = Blocks()
    src = make_source(path, blocks, blocksize=64, realtime=False, loop=False)

    src.start()
    assert src.finished.wait(TIMEOUT), "finished must be set at EOF"
    assert wait_until(lambda: src.is_running is False), (
        "the replay thread must exit at EOF, leaving is_running False"
    )
    assert src.error is None
    assert blocks.frames == 500

    time.sleep(QUIET_S)
    assert blocks.frames == 500, "no audio may be emitted after EOF"


def test_loop_true_keeps_emitting_past_eof_and_does_not_set_finished(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """Looping is what lets a test run the pipeline for as long as it needs."""
    sample_rate = 8000
    n = 400                                     # 50 ms per pass
    data = ramp_int(n)
    path = write_wav(tmp_path / "loop.wav", data, sample_rate)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=80, realtime=True, loop=True)
    src.start()

    assert blocks.wait_for_frames(3 * n), (
        f"loop=True must keep emitting past EOF; only {blocks.frames} frames "
        f"arrived, error={src.error!r}"
    )
    assert src.finished.is_set() is False, (
        "finished must NOT be set at EOF while looping -- the source will "
        "produce more audio"
    )
    assert src.is_running is True, "the replay thread must still be alive"

    got = blocks.concat()[: 3 * n]
    want = np.tile(expected_mono(data), 3)
    assert np.allclose(got, want, atol=TOL[2]), (
        "each loop pass must restart from the beginning of the file"
    )

    assert src.stop(timeout=5.0) is True


def test_stop_sets_finished_while_looping(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """finished means "no more audio" -- stop() is the other way to get there."""
    path = write_wav(tmp_path / "loopstop.wav", ramp_int(400), 8000)
    blocks = Blocks()
    src = make_source(path, blocks, blocksize=80, realtime=True, loop=True)

    src.start()
    assert blocks.wait_for_frames(400), "the loop never produced audio"
    assert src.stop(timeout=5.0) is True

    assert src.finished.wait(TIMEOUT), (
        "stop() must set finished, otherwise a waiter on a looping source hangs"
    )
    assert src.is_running is False


# ==========================================================================
# stats
# ==========================================================================

def test_stats_frames_captured_accumulates_to_the_file_length(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    n = 3000
    path = write_wav(tmp_path / "stats.wav", ramp_int(n), 16_000)
    stats = StreamStats()
    blocks = Blocks()

    src = make_source(path, blocks, blocksize=128, realtime=False, stats=stats)
    assert stats.frames_captured == 0, "__init__ must not capture anything"

    src.start()
    assert src.finished.wait(TIMEOUT)
    assert wait_until(lambda: stats.frames_captured == n), (
        f"frames_captured must total the file's {n} frames, got "
        f"{stats.frames_captured}"
    )
    assert stats.frames_captured == blocks.frames, (
        "frames_captured must equal the frames actually handed to on_audio"
    )


def test_stats_frames_captured_keeps_accumulating_across_loops(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    n = 400
    path = write_wav(tmp_path / "loopstats.wav", ramp_int(n), 8000)
    stats = StreamStats()
    blocks = Blocks()
    src = make_source(
        path, blocks, blocksize=80, realtime=True, loop=True, stats=stats
    )

    src.start()
    assert blocks.wait_for_frames(2 * n), "the loop never produced enough audio"
    assert wait_until(lambda: stats.frames_captured >= 2 * n), (
        f"frames_captured is a stream-lifetime counter, got {stats.frames_captured}"
    )
    assert src.stop(timeout=5.0) is True


def test_stats_levels_are_set_from_the_audio(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """Sources own the level fields; they are the only place raw audio is seen."""
    n = 1600
    data = np.full(n, 16384, dtype=np.int64)     # a constant 0.5 full-scale tone
    path = write_wav(tmp_path / "level.wav", data, 16_000)

    stats = StreamStats()
    src = make_source(path, Blocks(), blocksize=160, realtime=False, stats=stats)
    assert stats.peak_level == 0.0 and stats.rms_level == 0.0

    src.start()
    assert src.finished.wait(TIMEOUT)
    assert wait_until(lambda: stats.peak_level > 0.0), (
        "peak_level must be updated from the emitted audio"
    )

    assert stats.peak_level == pytest.approx(0.5, abs=0.01), (
        f"a constant 0.5 full-scale signal has peak 0.5, got {stats.peak_level}"
    )
    assert stats.rms_level == pytest.approx(0.5, abs=0.01), (
        f"a constant signal's RMS equals its level, got {stats.rms_level}"
    )
    assert 0.0 <= stats.peak_level <= 1.0
    assert 0.0 <= stats.rms_level <= 1.0


def test_stats_defaults_to_a_private_streamstats(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "nostats.wav", ramp_int(200), 16_000)
    src = make_source(path, Blocks(), blocksize=64, realtime=False, stats=None)

    src.start()
    assert src.finished.wait(TIMEOUT), "stats=None must not break the replay"
    assert src.error is None


def test_source_does_not_touch_the_chunker_or_sink_counters(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "own.wav", ramp_int(1000), 16_000)
    stats = StreamStats()
    src = make_source(path, Blocks(), blocksize=64, realtime=False, stats=stats)

    src.start()
    assert src.finished.wait(TIMEOUT)
    assert wait_until(lambda: stats.frames_captured == 1000)

    assert stats.chunks_emitted == 0, "emitting chunks is the chunker's job"
    assert stats.chunks_dropped == 0, "dropping is the sink's job"
    assert stats.overruns == 0, "overruns are detected by the chunker"


# ==========================================================================
# lifecycle
# ==========================================================================

def test_is_running_is_false_before_start(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "life.wav", ramp_int(400), 8000)
    src = make_source(path, Blocks(), blocksize=80, realtime=True, loop=True)
    assert src.is_running is False


def test_is_running_transitions_true_then_false(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "life2.wav", ramp_int(400), 8000)
    blocks = Blocks()
    src = make_source(path, blocks, blocksize=80, realtime=True, loop=True)

    src.start()
    assert blocks.wait_for_frames(80), "the source never emitted anything"
    assert src.is_running is True

    assert src.stop(timeout=5.0) is True
    assert src.is_running is False


def test_stop_before_start_returns_true_and_emits_nothing(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "nostart.wav", ramp_int(400), 8000)
    blocks = Blocks()
    src = make_source(path, blocks, blocksize=80, realtime=True, loop=True)

    assert src.stop(timeout=5.0) is True, (
        "stopping a never-started source is a no-op returning True"
    )
    assert src.is_running is False
    assert blocks.count == 0


def test_stop_is_idempotent(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "idem.wav", ramp_int(400), 8000)
    src = make_source(path, Blocks(), blocksize=80, realtime=True, loop=True)

    src.start()
    assert src.stop(timeout=5.0) is True
    assert src.stop(timeout=5.0) is True, "a second stop() must also return True"
    assert src.stop(timeout=5.0) is True, "and a third"
    assert src.is_running is False


def test_stop_during_replay_halts_the_audio_promptly(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """A long realtime replay must stop quickly, not play itself out."""
    sample_rate = 8000
    n = 8000 * 30                               # 30 s of audio
    path = write_wav(tmp_path / "long.wav", ramp_int(n), sample_rate)

    blocks = Blocks()
    src = make_source(path, blocks, blocksize=160, realtime=True)
    src.start()
    assert blocks.wait_for_frames(160), "the replay never started"

    t0 = time.monotonic()
    assert src.stop(timeout=5.0) is True, "stop() must join the replay thread"
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, (
        f"stop() took {elapsed:.2f}s on a 30 s file -- the replay loop must check "
        "its stop flag between blocks"
    )

    assert src.is_running is False
    frames_at_stop = blocks.frames
    assert frames_at_stop < n, "test setup: the file must not have finished"

    time.sleep(QUIET_S)
    assert blocks.frames == frames_at_stop, "no audio may be emitted after stop()"


def test_stop_after_eof_returns_true(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "afeof.wav", ramp_int(300), 16_000)
    src = make_source(path, Blocks(), blocksize=64, realtime=False)

    src.start()
    assert src.finished.wait(TIMEOUT)
    assert src.stop(timeout=5.0) is True, "the thread already ended"


def test_worker_thread_is_a_daemon_named_file_source(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """A non-daemon worker would keep the interpreter alive after a crash."""
    path = write_wav(tmp_path / "named.wav", ramp_int(4000), 8000)
    blocks = Blocks()
    src = make_source(path, blocks, blocksize=80, realtime=True, loop=True)
    src.start()
    assert blocks.wait_for_frames(80), "the source never started emitting"

    matching = [t for t in threading.enumerate() if t.name == "file-source"]
    assert matching, (
        "the replay thread must be named 'file-source'; saw "
        f"{[t.name for t in threading.enumerate()]}"
    )
    assert all(t.daemon for t in matching), "the replay thread must be a daemon"


# ==========================================================================
# error capture
# ==========================================================================

class Boom(Exception):
    """Sentinel raised by a deliberately broken on_audio."""


def test_error_is_none_on_a_healthy_replay(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "ok.wav", ramp_int(500), 16_000)
    src = make_source(path, Blocks(), blocksize=64, realtime=False)

    assert src.error is None, "a fresh source has no error"
    src.start()
    assert src.finished.wait(TIMEOUT)
    assert src.error is None, "a healthy replay must not record an error"


def test_a_raising_on_audio_is_captured_in_error(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    """Exceptions go to .error, the thread exits cleanly, finished is set."""
    path = write_wav(tmp_path / "boom.wav", ramp_int(2000), 16_000)

    def exploding(block: np.ndarray) -> None:
        raise Boom("the consumer of the audio failed")

    src = make_source(path, exploding, blocksize=64, realtime=False)
    src.start()

    assert wait_until(lambda: src.error is not None), (
        "an exception from on_audio must be recorded in .error"
    )
    assert isinstance(src.error, Boom), f"got {src.error!r}"
    assert src.finished.wait(TIMEOUT), (
        "finished must be set when the thread dies -- otherwise every waiter hangs"
    )
    assert wait_until(lambda: src.is_running is False)


def test_stop_after_an_error_does_not_raise_or_hang(
    tmp_path: Any, make_source: Callable[..., FileSource]
) -> None:
    path = write_wav(tmp_path / "boom2.wav", ramp_int(2000), 16_000)

    def exploding(block: np.ndarray) -> None:
        raise Boom("nope")

    src = make_source(path, exploding, blocksize=64, realtime=False)
    src.start()
    assert wait_until(lambda: src.error is not None)

    t0 = time.monotonic()
    assert src.stop(timeout=5.0) is True, "the thread already ended"
    assert time.monotonic() - t0 < 2.0, "stop() must not wait out its timeout"
    assert isinstance(src.error, Boom), "stop() must not clear .error"
