"""Tests for echochamber.config, written against the step-1 API contract.

Covers ms_to_frames, AudioConfig defaults, derived properties, every documented
validation rule, with_window() and immutability.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from echochamber.audio.types import DropPolicy
from echochamber.config import AudioConfig, ms_to_frames


# --------------------------------------------------------------------------
# ms_to_frames
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("ms", "sample_rate", "expected"),
    [
        (0, 16_000, 0),
        (0.0, 16_000, 0),
        (1, 16_000, 16),
        (10, 16_000, 160),
        (1000, 16_000, 16_000),
        (3000, 16_000, 48_000),
        (50, 16_000, 800),
        (1000, 44_100, 44_100),
        (20, 44_100, 882),
        (23, 44_100, 1014),          # 1014.3 -> 1014
        (0.1, 16_000, 2),            # 1.6 -> 2
        (0.03, 16_000, 0),           # 0.48 -> 0
    ],
)
def test_ms_to_frames_basic(ms: float, sample_rate: int, expected: int) -> None:
    result = ms_to_frames(ms, sample_rate)
    assert result == expected, f"ms_to_frames({ms}, {sample_rate}) should be {expected}"
    assert isinstance(result, int), "ms_to_frames must return an int"


def test_ms_to_frames_zero_is_zero() -> None:
    assert ms_to_frames(0, 16_000) == 0
    assert ms_to_frames(0.0, 44_100) == 0


@pytest.mark.parametrize(
    ("ms", "sample_rate", "expected"),
    [
        # Contract says round().  Python's round() is banker's rounding: exact
        # .5 values go to the nearest EVEN integer.
        (0.5, 1_000, 0),
        (1.5, 1_000, 2),
        (2.5, 1_000, 2),
        (3.5, 1_000, 4),
        (4.5, 1_000, 4),
    ],
)
def test_ms_to_frames_half_cases_use_round(ms: float, sample_rate: int, expected: int) -> None:
    assert ms_to_frames(ms, sample_rate) == expected, (
        f"ms_to_frames({ms}, {sample_rate}) must follow round() (banker's) semantics"
    )


@pytest.mark.parametrize("ms", [-0.0001, -1, -10, -1000, -1e6])
def test_ms_to_frames_never_negative(ms: float) -> None:
    assert ms_to_frames(ms, 16_000) == 0, "ms_to_frames must clamp negative input to 0"


def test_ms_to_frames_is_monotonic_nondecreasing() -> None:
    values = [ms_to_frames(ms, 16_000) for ms in range(0, 200, 7)]
    assert values == sorted(values), "ms_to_frames must be non-decreasing in ms"


# --------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "sample_rate": 16_000,
    "channels": 1,
    "blocksize": 160,
    "ring_seconds": 10.0,
    "window_ms": 3000,
    "hop_ms": 1000,
    "queue_max": 8,
    "drop_policy": DropPolicy.DROP_OLDEST,
}


@pytest.mark.parametrize(("field", "expected"), sorted(DEFAULTS.items(), key=lambda kv: kv[0]))
def test_default_field_values(field: str, expected: Any) -> None:
    cfg = AudioConfig()
    assert getattr(cfg, field) == expected, f"{field} must default to {expected!r}"


def test_default_config_constructs_without_error() -> None:
    cfg = AudioConfig()
    assert isinstance(cfg, AudioConfig)


# --------------------------------------------------------------------------
# derived properties
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("prop", "expected"),
    [
        ("window_frames", 48_000),
        ("hop_frames", 16_000),
        ("overlap_frames", 32_000),
        ("ring_frames", 160_000),
    ],
)
def test_derived_integer_properties_at_defaults(prop: str, expected: int) -> None:
    cfg = AudioConfig()
    assert getattr(cfg, prop) == expected, f"{prop} at defaults must be {expected}"


def test_overlap_ratio_at_defaults() -> None:
    cfg = AudioConfig()
    assert cfg.overlap_ratio == pytest.approx(2 / 3, abs=1e-9), (
        "3000 ms window / 1000 ms hop is a 2/3 (~66.67 %) overlap"
    )


@pytest.mark.parametrize(
    ("window_ms", "hop_ms", "sample_rate", "window_frames", "hop_frames"),
    [
        (3000, 1000, 16_000, 48_000, 16_000),
        (1000, 500, 16_000, 16_000, 8_000),
        (500, 50, 16_000, 8_000, 800),
        (2000, 2000, 16_000, 32_000, 32_000),
        (1000, 250, 44_100, 44_100, 11_025),
        (100, 10, 8_000, 800, 80),
    ],
)
def test_window_and_hop_frames_track_config(
    window_ms: int, hop_ms: int, sample_rate: int,
    window_frames: int, hop_frames: int,
) -> None:
    cfg = AudioConfig(window_ms=window_ms, hop_ms=hop_ms, sample_rate=sample_rate)

    assert cfg.window_frames == window_frames
    assert cfg.hop_frames == hop_frames
    assert cfg.overlap_frames == window_frames - hop_frames, (
        "overlap_frames must be window_frames - hop_frames"
    )
    assert cfg.overlap_ratio == pytest.approx(
        (window_frames - hop_frames) / window_frames, abs=1e-12
    )


@pytest.mark.parametrize(
    ("ring_seconds", "sample_rate", "expected"),
    [
        (10.0, 16_000, 160_000),
        (5.0, 16_000, 80_000),
        (4.0, 16_000, 64_000),
        (10.0, 44_100, 441_000),
        (7.5, 16_000, 120_000),
        (10.25, 16_000, 164_000),
    ],
)
def test_ring_frames(ring_seconds: float, sample_rate: int, expected: int) -> None:
    cfg = AudioConfig(ring_seconds=ring_seconds, sample_rate=sample_rate)
    assert cfg.ring_frames == expected, "ring_frames must be round(ring_seconds * sample_rate)"


def test_zero_overlap_is_allowed_when_hop_equals_window() -> None:
    cfg = AudioConfig(window_ms=1000, hop_ms=1000)
    assert cfg.overlap_frames == 0
    assert cfg.overlap_ratio == pytest.approx(0.0)


def test_ring_exactly_window_plus_hop_is_valid_boundary() -> None:
    """ring_frames >= window_frames + hop_frames -- equality must be accepted."""
    cfg = AudioConfig(window_ms=3000, hop_ms=1000, ring_seconds=4.0)
    assert cfg.ring_frames == cfg.window_frames + cfg.hop_frames


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("rule", "kwargs"),
    [
        ("sample_rate<=0", {"sample_rate": 0}),
        ("sample_rate<=0", {"sample_rate": -16_000}),
        ("channels<1", {"channels": 0}),
        ("channels<1", {"channels": -1}),
        ("blocksize<=0", {"blocksize": 0}),
        ("blocksize<=0", {"blocksize": -160}),
        ("queue_max<1", {"queue_max": 0}),
        ("queue_max<1", {"queue_max": -8}),
        ("ring_seconds<=0", {"ring_seconds": 0.0}),
        ("ring_seconds<=0", {"ring_seconds": -10.0}),
        ("window_ms<=0", {"window_ms": 0}),
        ("window_ms<=0", {"window_ms": -3000}),
        ("hop_ms<=0", {"hop_ms": 0}),
        ("hop_ms<=0", {"hop_ms": -1000}),
        # hop must not exceed window: overlapping windows only.
        ("hop_frames>window_frames", {"window_ms": 100, "hop_ms": 200}),
        ("hop_frames>window_frames", {"window_ms": 1000, "hop_ms": 1001}),
        # ring must hold a window plus a hop of slack.
        ("ring too small", {"ring_seconds": 1.0}),
        ("ring too small", {"ring_seconds": 3.9}),
        ("ring too small", {"window_ms": 3000, "hop_ms": 1000, "ring_seconds": 2.0}),
    ],
)
def test_validation_rejects_bad_config(rule: str, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError) as excinfo:
        AudioConfig(**kwargs)
    assert str(excinfo.value), f"ValueError for {rule} must carry a message"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"sample_rate": 44_100},
        {"channels": 2},
        {"blocksize": 1},
        {"queue_max": 1},
        {"ring_seconds": 4.0},
        {"window_ms": 1, "hop_ms": 1},
        {"window_ms": 3000, "hop_ms": 3000, "ring_seconds": 6.0},
        {"drop_policy": DropPolicy.BLOCK},
    ],
)
def test_validation_accepts_valid_config(kwargs: dict[str, Any]) -> None:
    cfg = AudioConfig(**kwargs)
    assert isinstance(cfg, AudioConfig)


# --------------------------------------------------------------------------
# with_window
# --------------------------------------------------------------------------

def test_with_window_returns_a_new_instance() -> None:
    cfg = AudioConfig()
    new = cfg.with_window(window_ms=2000, hop_ms=500)

    assert new is not cfg, "with_window must return a new object"
    assert isinstance(new, AudioConfig)
    assert new.window_ms == 2000
    assert new.hop_ms == 500
    assert new.window_frames == 32_000
    assert new.hop_frames == 8_000


def test_with_window_leaves_the_original_unchanged() -> None:
    cfg = AudioConfig()
    cfg.with_window(window_ms=2000, hop_ms=500)

    assert cfg.window_ms == 3000, "the original config must not be mutated"
    assert cfg.hop_ms == 1000
    assert cfg.window_frames == 48_000
    assert cfg.hop_frames == 16_000


def test_with_window_preserves_unrelated_fields() -> None:
    cfg = AudioConfig(
        sample_rate=44_100,
        channels=2,
        blocksize=441,
        ring_seconds=8.0,
        queue_max=4,
        drop_policy=DropPolicy.BLOCK,
    )
    new = cfg.with_window(window_ms=1500)

    assert new.sample_rate == 44_100
    assert new.channels == 2
    assert new.blocksize == 441
    assert new.ring_seconds == pytest.approx(8.0)
    assert new.queue_max == 4
    assert new.drop_policy is DropPolicy.BLOCK


@pytest.mark.parametrize(
    ("kwargs", "exp_window_ms", "exp_hop_ms"),
    [
        ({"window_ms": 2000}, 2000, 1000),
        ({"hop_ms": 500}, 3000, 500),
        ({"window_ms": 4000, "hop_ms": 2000}, 4000, 2000),
        ({}, 3000, 1000),
    ],
)
def test_with_window_partial_updates(
    kwargs: dict[str, Any], exp_window_ms: int, exp_hop_ms: int
) -> None:
    cfg = AudioConfig()
    new = cfg.with_window(**kwargs)

    assert new.window_ms == exp_window_ms, "omitted arguments must keep the current value"
    assert new.hop_ms == exp_hop_ms


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_ms": 0},
        {"hop_ms": 0},
        {"window_ms": -100},
        {"hop_ms": -100},
        {"window_ms": 100, "hop_ms": 200},   # hop > window
        {"window_ms": 100},                  # hop (1000 ms) would exceed the new window
        {"window_ms": 9000, "hop_ms": 9000},  # 18 s needed, 10 s ring
    ],
)
def test_with_window_revalidates(kwargs: dict[str, Any]) -> None:
    cfg = AudioConfig()
    with pytest.raises(ValueError):
        cfg.with_window(**kwargs)


def test_with_window_is_keyword_only() -> None:
    cfg = AudioConfig()
    with pytest.raises(TypeError):
        cfg.with_window(2000, 500)  # type: ignore[misc]


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 44_100),
        ("channels", 2),
        ("blocksize", 320),
        ("ring_seconds", 5.0),
        ("window_ms", 1000),
        ("hop_ms", 250),
        ("queue_max", 2),
        ("drop_policy", DropPolicy.BLOCK),
    ],
)
def test_audioconfig_is_frozen(field: str, value: Any) -> None:
    cfg = AudioConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cfg, field, value)
    assert getattr(cfg, field) == DEFAULTS[field], "the failed assignment must not take effect"


def test_audioconfig_rejects_new_attributes() -> None:
    """Unknown attributes must be rejected; the exception type is version-dependent.

    See ``test_audiochunk_rejects_unknown_attribute`` for why ``TypeError`` is
    accepted here: it is a CPython 3.11 quirk of ``frozen=True, slots=True``, not
    a property of this class. The assertion that matters is that the write does
    not take effect.
    """
    cfg = AudioConfig()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        cfg.some_new_field = 1  # type: ignore[attr-defined]
    assert not hasattr(cfg, "some_new_field"), "rejected attribute must not stick"


@pytest.mark.parametrize(
    "kwargs",
    [
        # 1 ms at 100 Hz rounds to 0 frames; window_ms > 0 does not catch it.
        # 1 ms at 100 Hz -> 0.1 frames -> 0: the window itself vanishes.
        dict(sample_rate=100, window_ms=1, hop_ms=1, ring_seconds=100.0),
        # 100 ms window is fine (10 frames) but the 1 ms hop rounds to 0.
        dict(sample_rate=100, window_ms=100, hop_ms=1, ring_seconds=100.0),
    ],
)
def test_subframe_window_or_hop_rejected(kwargs: dict[str, object]) -> None:
    """A window or hop that rounds to zero frames must not construct.

    Otherwise the chunker emits empty chunks and overlap_ratio divides by zero.
    """
    with pytest.raises(ValueError, match="at least 1 frame"):
        AudioConfig(**kwargs)  # type: ignore[arg-type]


def test_overlap_ratio_never_divides_by_zero() -> None:
    """Every constructible config must have a usable overlap_ratio."""
    cfg = AudioConfig(sample_rate=8000, window_ms=20, hop_ms=10, ring_seconds=1.0)
    assert 0.0 <= cfg.overlap_ratio < 1.0


def test_dataclasses_replace_still_validates() -> None:
    """The GUI swaps whole config objects; replace() must not bypass __post_init__."""
    cfg = AudioConfig()
    with pytest.raises(ValueError):
        dataclasses.replace(cfg, hop_ms=5000)
