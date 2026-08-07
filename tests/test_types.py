"""Tests for echochamber.audio.types, written against the step-1 API contract.

Covers AudioChunk (derived properties, immutability), DropPolicy (members and
wire values) and StreamStats (defaults, snapshot independence).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_chunk(
    n_frames: int = 4,
    start_frame: int = 0,
    seq: int = 0,
    sample_rate: int = 16_000,
    discontinuous: bool = False,
) -> AudioChunk:
    samples = np.arange(n_frames, dtype=np.float32)
    return AudioChunk(
        samples=samples,
        start_frame=start_frame,
        seq=seq,
        sample_rate=sample_rate,
        discontinuous=discontinuous,
    )


# --------------------------------------------------------------------------
# AudioChunk
# --------------------------------------------------------------------------

def test_audiochunk_stores_fields_verbatim() -> None:
    samples = np.arange(8, dtype=np.float32)
    chunk = AudioChunk(samples=samples, start_frame=32, seq=2, sample_rate=8_000)

    assert chunk.samples is samples, "samples must be stored as given"
    assert chunk.start_frame == 32
    assert chunk.seq == 2
    assert chunk.sample_rate == 8_000


def test_audiochunk_discontinuous_defaults_to_false() -> None:
    chunk = make_chunk()
    assert chunk.discontinuous is False, "discontinuous must default to False"


def test_audiochunk_discontinuous_is_settable_at_construction() -> None:
    chunk = make_chunk(discontinuous=True)
    assert chunk.discontinuous is True


@pytest.mark.parametrize("n_frames", [0, 1, 160, 48_000])
def test_audiochunk_n_frames_is_len_samples(n_frames: int) -> None:
    chunk = make_chunk(n_frames=n_frames)
    assert chunk.n_frames == n_frames, "n_frames must equal len(samples)"
    assert chunk.n_frames == len(chunk.samples)


@pytest.mark.parametrize(
    ("n_frames", "sample_rate", "expected"),
    [
        (16_000, 16_000, 1.0),
        (48_000, 16_000, 3.0),
        (8_000, 16_000, 0.5),
        (0, 16_000, 0.0),
        (160, 16_000, 0.01),
        (44_100, 44_100, 1.0),
    ],
)
def test_audiochunk_duration_s(n_frames: int, sample_rate: int, expected: float) -> None:
    chunk = make_chunk(n_frames=n_frames, sample_rate=sample_rate)
    assert chunk.duration_s == pytest.approx(expected), (
        f"duration_s for {n_frames} frames @ {sample_rate} Hz should be {expected}"
    )


@pytest.mark.parametrize(
    ("start_frame", "sample_rate", "expected"),
    [
        (0, 16_000, 0.0),
        (16_000, 16_000, 1.0),
        (48_000, 16_000, 3.0),
        (24_000, 16_000, 1.5),
        (160, 16_000, 0.01),
        (7, 8_000, 7 / 8_000),
    ],
)
def test_audiochunk_start_time_s(start_frame: int, sample_rate: int, expected: float) -> None:
    chunk = make_chunk(n_frames=4, start_frame=start_frame, sample_rate=sample_rate)
    assert chunk.start_time_s == pytest.approx(expected), (
        "start_time_s must be start_frame / sample_rate"
    )


def test_audiochunk_nonzero_start_frame_properties_are_independent() -> None:
    """start_time_s tracks start_frame; duration_s tracks len(samples)."""
    chunk = make_chunk(n_frames=48_000, start_frame=32_000, sample_rate=16_000)

    assert chunk.n_frames == 48_000
    assert chunk.duration_s == pytest.approx(3.0)
    assert chunk.start_time_s == pytest.approx(2.0)


def test_audiochunk_hop_alignment_matches_architecture_example() -> None:
    """Chunk k covers absolute frames [k*H, k*H + W) -- verify k=3, W=48000, H=16000."""
    hop, window, rate = 16_000, 48_000, 16_000
    k = 3
    chunk = make_chunk(n_frames=window, start_frame=k * hop, seq=k, sample_rate=rate)

    assert chunk.start_frame == 48_000
    assert chunk.start_time_s == pytest.approx(3.0)
    assert chunk.duration_s == pytest.approx(3.0)
    assert chunk.seq == k


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_frame", 99),
        ("seq", 5),
        ("sample_rate", 8_000),
        ("discontinuous", True),
        ("samples", np.zeros(2, dtype=np.float32)),
    ],
)
def test_audiochunk_is_frozen(field: str, value: object) -> None:
    chunk = make_chunk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(chunk, field, value)


def test_audiochunk_rejects_unknown_attribute() -> None:
    """slots=True means no ad-hoc attributes; frozen=True means no assignment at all.

    The exception *type* is deliberately not pinned. On CPython 3.11,
    ``@dataclass(frozen=True, slots=True)`` raises ``TypeError`` here rather than
    ``AttributeError``: the generated ``__setattr__`` short-circuits to
    ``FrozenInstanceError`` only for declared fields and otherwise defers to
    ``super().__setattr__``, but ``slots=True`` rebuilds the class and leaves the
    ``__class__`` cell pointing at the original, so the ``super()`` call fails.
    Later CPython versions fix this and raise ``AttributeError``. What actually
    matters -- and what is asserted -- is that the write is rejected and does not
    take effect, which holds on every version.
    """
    chunk = make_chunk()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        chunk.not_a_field = 1  # type: ignore[attr-defined]
    assert not hasattr(chunk, "not_a_field"), "rejected attribute must not stick"


# --------------------------------------------------------------------------
# DropPolicy
# --------------------------------------------------------------------------

def test_droppolicy_has_exactly_two_members() -> None:
    assert set(DropPolicy) == {DropPolicy.DROP_OLDEST, DropPolicy.BLOCK}, (
        "DropPolicy must define exactly DROP_OLDEST and BLOCK"
    )


@pytest.mark.parametrize(
    ("member", "value"),
    [
        (DropPolicy.DROP_OLDEST, "drop_oldest"),
        (DropPolicy.BLOCK, "block"),
    ],
)
def test_droppolicy_values(member: DropPolicy, value: str) -> None:
    assert member.value == value, f"{member.name} must carry the value {value!r}"


@pytest.mark.parametrize("value", ["drop_oldest", "block"])
def test_droppolicy_lookup_by_value(value: str) -> None:
    assert DropPolicy(value).value == value, "DropPolicy must be constructible from its value"


# --------------------------------------------------------------------------
# StreamStats
# --------------------------------------------------------------------------

DEFAULT_STATS = {
    "frames_captured": 0,
    "chunks_emitted": 0,
    "chunks_dropped": 0,
    "overruns": 0,
    "xruns": 0,
    "peak_level": 0.0,
    "rms_level": 0.0,
}


@pytest.mark.parametrize(("field", "expected"), sorted(DEFAULT_STATS.items()))
def test_streamstats_defaults(field: str, expected: float) -> None:
    stats = StreamStats()
    assert getattr(stats, field) == expected, f"{field} must default to {expected!r}"


def test_streamstats_is_mutable() -> None:
    """StreamStats is deliberately NOT frozen -- the audio threads update it in place."""
    stats = StreamStats()
    stats.frames_captured = 1234
    assert stats.frames_captured == 1234


def test_streamstats_snapshot_equals_source() -> None:
    stats = StreamStats(
        frames_captured=16_000,
        chunks_emitted=15,
        chunks_dropped=2,
        overruns=1,
        xruns=3,
        peak_level=0.9,
        rms_level=0.25,
    )
    snap = stats.snapshot()

    for field in DEFAULT_STATS:
        assert getattr(snap, field) == getattr(stats, field), (
            f"snapshot must copy {field} verbatim"
        )


def test_streamstats_snapshot_is_a_distinct_object() -> None:
    stats = StreamStats()
    snap = stats.snapshot()

    assert isinstance(snap, StreamStats), "snapshot() must return a StreamStats"
    assert snap is not stats, "snapshot() must not return self"


def test_streamstats_snapshot_is_independent_of_later_mutation() -> None:
    """The GUI reads a snapshot; the audio thread keeps mutating the original."""
    stats = StreamStats(frames_captured=100, chunks_emitted=5, peak_level=0.5)
    snap = stats.snapshot()

    stats.frames_captured = 999_999
    stats.chunks_emitted = 42
    stats.chunks_dropped = 7
    stats.overruns = 3
    stats.xruns = 4
    stats.peak_level = 1.0
    stats.rms_level = 0.75

    assert snap.frames_captured == 100, "mutating the original must not change the snapshot"
    assert snap.chunks_emitted == 5
    assert snap.chunks_dropped == 0
    assert snap.overruns == 0
    assert snap.xruns == 0
    assert snap.peak_level == pytest.approx(0.5)
    assert snap.rms_level == pytest.approx(0.0)


def test_streamstats_mutating_snapshot_does_not_touch_original() -> None:
    stats = StreamStats(frames_captured=10)
    snap = stats.snapshot()

    snap.frames_captured = 0
    snap.xruns = 99

    assert stats.frames_captured == 10, "the snapshot must be a detached copy"
    assert stats.xruns == 0


def test_streamstats_snapshot_of_snapshot_is_stable() -> None:
    stats = StreamStats(chunks_emitted=3)
    first = stats.snapshot()
    second = first.snapshot()

    assert second.chunks_emitted == 3
    assert second is not first
