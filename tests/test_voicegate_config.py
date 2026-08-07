"""Tests for echochamber.voicegate.config, written against the API contract.

``VoiceGateConfig`` is the same shape as
:class:`~echochamber.config.AudioConfig` -- a frozen dataclass validated in
``__post_init__`` -- so it is tested the same way: documented defaults, every
rejection path with an assertion on the *message* rather than only the type,
frame conversions at a round rate and at an awkward one, and the ``with_*``
helpers proving they return a new validated object without touching the
original.

Two things here are not obvious and get their own attention:

* **Only ``enabled`` gates the "at least one usable phrase" rule.**  A disabled
  gate is allowed to hold junk phrases, because the default config must
  construct on a checkout with no model, and the GUI is expected to fix the
  phrases before switching the gate on -- which is why ``with_enabled(True)``
  is what raises, not the constructor that stored the junk.
* **``phrases`` must be a tuple, and that is a ``TypeError``, not a
  ``ValueError``.**  It is a type contract (the config has to stay hashable and
  immutable), not a range check, and the two are asserted separately so a
  regression that collapses them is visible.

Nothing here touches the filesystem: ``snippet_dir`` is validated as a string
and is only created when a snippet is actually written.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from echochamber.voicegate.config import DEFAULT_PHRASES, VoiceGateConfig


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _cfg(**kwargs: Any) -> VoiceGateConfig:
    """Build a VoiceGateConfig, overriding only the named fields."""
    return VoiceGateConfig(**kwargs)


DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "phrases": ("ok google", "hey google"),
    "model_path": None,
    "pre_roll_ms": 1500,
    "post_roll_ms": 3000,
    "max_snippet_ms": 15_000,
    "cooldown_ms": 1000,
    "snippet_dir": "snippets",
    "worker_python": None,
    "startup_timeout_s": 30.0,
}


class TestDefaults:
    """The out-of-the-box config is the one the docstring documents."""

    @pytest.mark.parametrize(("field", "expected"), sorted(DEFAULTS.items()))
    def test_default_field_values(self, field: str, expected: Any) -> None:
        """Every field defaults to the documented value."""
        cfg = VoiceGateConfig()
        assert getattr(cfg, field) == expected, (
            f"{field} must default to {expected!r}"
        )

    def test_enabled_defaults_to_false(self) -> None:
        """The gate is off by default: it needs a model this repo does not ship."""
        assert VoiceGateConfig().enabled is False, (
            "defaulting the gate on would make a fresh checkout fail at start"
        )

    def test_default_phrases_constant_is_what_the_dataclass_uses(self) -> None:
        """DEFAULT_PHRASES is the single source of the default phrase list."""
        assert VoiceGateConfig().phrases is DEFAULT_PHRASES
        assert DEFAULT_PHRASES == ("ok google", "hey google")

    def test_default_phrases_is_a_tuple(self) -> None:
        """A mutable default would leak edits across every config that shares it."""
        assert isinstance(DEFAULT_PHRASES, tuple)

    def test_the_default_config_constructs(self) -> None:
        """No argument is required; the defaults are internally consistent."""
        assert isinstance(VoiceGateConfig(), VoiceGateConfig)

    def test_the_defaults_leave_room_under_the_ceiling(self) -> None:
        """pre_roll + post_roll must be strictly usable inside max_snippet_ms."""
        cfg = VoiceGateConfig()
        assert cfg.pre_roll_ms + cfg.post_roll_ms < cfg.max_snippet_ms


class TestValidation:
    """__post_init__ rejects every configuration that cannot work."""

    @pytest.mark.parametrize("value", [-1, -1000, -15_000])
    def test_negative_pre_roll_is_rejected(self, value: int) -> None:
        """A pre-roll cannot run backwards."""
        with pytest.raises(ValueError, match=r"pre_roll_ms must be >= 0"):
            _cfg(pre_roll_ms=value)

    @pytest.mark.parametrize("value", [-1, -500, -30_000])
    def test_negative_post_roll_is_rejected(self, value: int) -> None:
        """A post-roll cannot run backwards."""
        with pytest.raises(ValueError, match=r"post_roll_ms must be >= 0"):
            _cfg(post_roll_ms=value)

    @pytest.mark.parametrize("value", [-1, -1000])
    def test_negative_cooldown_is_rejected(self, value: int) -> None:
        """A negative refractory period has no meaning."""
        with pytest.raises(ValueError, match=r"cooldown_ms must be >= 0"):
            _cfg(cooldown_ms=value)

    @pytest.mark.parametrize("value", [0, -1, -15_000])
    def test_non_positive_max_snippet_is_rejected(self, value: int) -> None:
        """A zero ceiling would make every snippet empty the moment it opened."""
        with pytest.raises(ValueError, match=r"max_snippet_ms must be > 0"):
            _cfg(max_snippet_ms=value, pre_roll_ms=0, post_roll_ms=0)

    @pytest.mark.parametrize("value", [0, 0.0, -0.1, -30.0])
    def test_non_positive_startup_timeout_is_rejected(self, value: float) -> None:
        """A zero timeout could never let the subprocess backend report readiness."""
        with pytest.raises(ValueError, match=r"startup_timeout_s must be > 0"):
            _cfg(startup_timeout_s=value)

    @pytest.mark.parametrize(
        ("pre", "post", "maximum"),
        [
            (1000, 1000, 1500),
            (1500, 3000, 4000),
            (1, 1, 1),
            (0, 5000, 4999),
            (5000, 0, 4999),
        ],
    )
    def test_pre_plus_post_over_the_ceiling_is_rejected(
        self, pre: int, post: int, maximum: int
    ) -> None:
        """A snippet whose two rolls alone exceed the ceiling is born truncated."""
        with pytest.raises(ValueError, match=r"must be <= max_snippet_ms"):
            _cfg(pre_roll_ms=pre, post_roll_ms=post, max_snippet_ms=maximum)

    @pytest.mark.parametrize(
        ("pre", "post", "maximum"),
        [
            (1000, 1000, 2000),      # exactly equal is the boundary, and is legal
            (1, 0, 1),
            (0, 1, 1),
            (1500, 3000, 4500),
        ],
    )
    def test_pre_plus_post_exactly_at_the_ceiling_is_accepted(
        self, pre: int, post: int, maximum: int
    ) -> None:
        """Equality is the boundary the error message describes as allowed."""
        cfg = _cfg(pre_roll_ms=pre, post_roll_ms=post, max_snippet_ms=maximum)
        assert cfg.pre_roll_ms + cfg.post_roll_ms == cfg.max_snippet_ms

    def test_empty_snippet_dir_is_rejected(self) -> None:
        """An empty directory would write snippets into the process's cwd."""
        with pytest.raises(ValueError, match=r"snippet_dir must not be empty"):
            _cfg(snippet_dir="")

    @pytest.mark.parametrize(
        "phrases",
        [
            (),
            ("",),
            ("   ",),
            ("!!!",),
            ("", "   ", "..."),
            ("-- --",),
        ],
    )
    def test_enabled_with_no_usable_phrase_is_rejected(
        self, phrases: tuple[str, ...]
    ) -> None:
        """An enabled gate with nothing to listen for could only ever be silent."""
        with pytest.raises(ValueError, match=r"survives normalisation"):
            _cfg(enabled=True, phrases=phrases)

    @pytest.mark.parametrize(
        "phrases", [(), ("",), ("!!!",), ("", "   ")]
    )
    def test_disabled_with_no_usable_phrase_is_accepted(
        self, phrases: tuple[str, ...]
    ) -> None:
        """A switched-off gate may hold junk; the GUI fixes it before enabling."""
        cfg = _cfg(enabled=False, phrases=phrases)
        assert cfg.normalized_phrases == ()

    def test_enabled_with_one_usable_phrase_among_junk_is_accepted(self) -> None:
        """One survivor is enough; the junk entries are simply dropped."""
        cfg = _cfg(enabled=True, phrases=("", "!!!", "Ok Google"))
        assert cfg.normalized_phrases == ("ok google",)

    @pytest.mark.parametrize(
        "phrases",
        [
            ["ok google"],
            ["ok google", "hey google"],
            [],
        ],
    )
    def test_a_list_of_phrases_is_a_typeerror(self, phrases: list[str]) -> None:
        """phrases must be a tuple: a type contract, not a range check.

        A list would make the frozen config unhashable and quietly mutable, so
        this is deliberately a ``TypeError`` and not one more ``ValueError``.
        """
        with pytest.raises(TypeError, match=r"phrases must be a tuple"):
            _cfg(phrases=phrases)  # type: ignore[arg-type]

    @pytest.mark.parametrize("phrases", [{"ok google"}, "ok google", None])
    def test_other_non_tuple_phrases_are_also_typeerrors(
        self, phrases: Any
    ) -> None:
        """Sets, bare strings and None are all rejected as the wrong type."""
        with pytest.raises(TypeError, match=r"phrases must be a tuple"):
            _cfg(phrases=phrases)

    def test_a_list_of_phrases_raises_typeerror_not_valueerror(self) -> None:
        """Pinned explicitly, because TypeError is not a subclass of ValueError."""
        with pytest.raises(TypeError):
            _cfg(phrases=["ok google"])  # type: ignore[arg-type]
        assert not issubclass(TypeError, ValueError)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"enabled": True},
            {"pre_roll_ms": 0},
            {"post_roll_ms": 0},
            {"cooldown_ms": 0},
            {"pre_roll_ms": 0, "post_roll_ms": 0, "max_snippet_ms": 1},
            {"startup_timeout_s": 0.001},
            {"model_path": "/nowhere/that/exists"},
            {"worker_python": "/usr/bin/python3"},
            {"snippet_dir": "."},
            {"phrases": ()},
        ],
    )
    def test_valid_configurations_construct(self, kwargs: dict[str, Any]) -> None:
        """Every legal combination constructs without complaint."""
        assert isinstance(_cfg(**kwargs), VoiceGateConfig)

    def test_dataclasses_replace_still_validates(self) -> None:
        """The GUI swaps whole config objects; replace() must not bypass validation."""
        cfg = VoiceGateConfig()
        with pytest.raises(ValueError, match=r"pre_roll_ms must be >= 0"):
            dataclasses.replace(cfg, pre_roll_ms=-1)


class TestNormalizedPhrases:
    """normalized_phrases is the list the gate actually matches against."""

    def test_phrases_are_normalised(self) -> None:
        """The stored spelling is preserved, but matching uses the folded form."""
        cfg = _cfg(phrases=("OK, Google!", "  Hey   Google  "))
        assert cfg.phrases == ("OK, Google!", "  Hey   Google  ")
        assert cfg.normalized_phrases == ("ok google", "hey google")

    def test_duplicates_after_normalisation_are_dropped(self) -> None:
        """Three spellings of one phrase are one phrase, not three."""
        cfg = _cfg(phrases=("ok google", "OK GOOGLE", "Ok, Google!"))
        assert cfg.normalized_phrases == ("ok google",)

    def test_order_is_preserved_across_deduplication(self) -> None:
        """The first occurrence of each phrase fixes its position."""
        cfg = _cfg(
            phrases=("hey google", "ok google", "HEY GOOGLE", "computer")
        )
        assert cfg.normalized_phrases == ("hey google", "ok google", "computer")

    @pytest.mark.parametrize("junk", ["", "   ", "!!!", ",", "-- --", "\t\n"])
    def test_entries_that_normalise_to_nothing_are_dropped(self, junk: str) -> None:
        """An empty entry is removed, not kept as "" which would match everything."""
        cfg = _cfg(phrases=("ok google", junk))
        assert cfg.normalized_phrases == ("ok google",), (
            f"{junk!r} must be dropped rather than retained as an empty phrase"
        )
        assert "" not in cfg.normalized_phrases

    def test_all_junk_normalises_to_an_empty_tuple(self) -> None:
        """Nothing usable in means an empty tuple out."""
        assert _cfg(phrases=("", "  ", "...")).normalized_phrases == ()

    def test_empty_phrases_normalises_to_an_empty_tuple(self) -> None:
        """An empty configuration is legal while the gate is off."""
        assert _cfg(phrases=()).normalized_phrases == ()

    def test_returns_a_tuple(self) -> None:
        """The result is a tuple, so a caller cannot mutate the gate's phrase set."""
        assert isinstance(VoiceGateConfig().normalized_phrases, tuple)

    def test_defaults_normalise_to_themselves(self) -> None:
        """The shipped defaults are already in normalised form."""
        assert VoiceGateConfig().normalized_phrases == ("ok google", "hey google")


class TestFrameConversions:
    """The *_frames methods take the rate rather than assuming one."""

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("pre_roll_frames", 24_000),
            ("post_roll_frames", 48_000),
            ("max_snippet_frames", 240_000),
            ("cooldown_frames", 16_000),
        ],
    )
    def test_defaults_at_16_khz(self, method: str, expected: int) -> None:
        """The shipped durations at the capture rate the pipeline defaults to."""
        cfg = VoiceGateConfig()
        assert getattr(cfg, method)(16_000) == expected

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("pre_roll_frames", 66_150),
            ("post_roll_frames", 132_300),
            ("max_snippet_frames", 661_500),
            ("cooldown_frames", 44_100),
        ],
    )
    def test_defaults_at_44100_hz(self, method: str, expected: int) -> None:
        """44.1 kHz divides none of these durations evenly; the rounding is pinned."""
        cfg = VoiceGateConfig()
        assert getattr(cfg, method)(44_100) == expected

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("pre_roll_frames", 16_538),      # 16537.5 rounds to even
            ("post_roll_frames", 33_075),
            ("max_snippet_frames", 165_375),
            ("cooldown_frames", 11_025),
        ],
    )
    def test_defaults_at_an_awkward_11025_hz(
        self, method: str, expected: int
    ) -> None:
        """An exact .5 frame count follows round()'s banker's rounding."""
        cfg = VoiceGateConfig()
        assert getattr(cfg, method)(11_025) == expected

    @pytest.mark.parametrize(
        "method",
        ["pre_roll_frames", "post_roll_frames", "max_snippet_frames", "cooldown_frames"],
    )
    def test_frame_counts_are_ints(self, method: str) -> None:
        """Frame counts index arrays and slice bytes; they must be plain ints."""
        value = getattr(VoiceGateConfig(), method)(16_000)
        assert isinstance(value, int), f"{method} must return an int, got {type(value).__name__}"

    def test_zero_durations_convert_to_zero_frames(self) -> None:
        """No pre-roll, post-roll or cooldown configured means zero frames."""
        cfg = _cfg(pre_roll_ms=0, post_roll_ms=0, cooldown_ms=0)
        assert cfg.pre_roll_frames(16_000) == 0
        assert cfg.post_roll_frames(16_000) == 0
        assert cfg.cooldown_frames(16_000) == 0

    @pytest.mark.parametrize(
        ("max_snippet_ms", "sample_rate"),
        [
            (1, 100),        # 0.1 frames
            (1, 1),          # 0.001 frames
            (1, 999),        # 0.999 frames
            (10, 1),         # 0.01 frames
        ],
    )
    def test_max_snippet_frames_never_returns_zero(
        self, max_snippet_ms: int, sample_rate: int
    ) -> None:
        """A ceiling that rounds to zero frames is clamped to 1, never 0.

        A zero ceiling would make ``_write_snippet`` close every snippet on the
        chunk that opened it, producing an unbounded stream of empty files.
        """
        cfg = _cfg(
            pre_roll_ms=0, post_roll_ms=0, max_snippet_ms=max_snippet_ms
        )
        assert cfg.max_snippet_frames(sample_rate) == 1, (
            f"{max_snippet_ms} ms at {sample_rate} Hz rounds to 0 frames and "
            "must be clamped to 1"
        )

    def test_max_snippet_frames_is_not_clamped_when_it_is_already_large(self) -> None:
        """The clamp is a floor, not a substitution."""
        assert _cfg(max_snippet_ms=15_000).max_snippet_frames(16_000) == 240_000

    def test_the_same_config_answers_differently_per_rate(self) -> None:
        """Durations carry no rate of their own; the caller supplies it."""
        cfg = VoiceGateConfig()
        assert cfg.pre_roll_frames(8_000) == 12_000
        assert cfg.pre_roll_frames(16_000) == 24_000
        assert cfg.pre_roll_frames(32_000) == 48_000


class TestWithPhrases:
    """with_phrases() returns a new validated config and leaves the old one alone."""

    def test_returns_a_new_object(self) -> None:
        """The config is frozen, so a change is a new instance."""
        cfg = VoiceGateConfig()
        new = cfg.with_phrases(("computer",))

        assert new is not cfg
        assert isinstance(new, VoiceGateConfig)
        assert new.phrases == ("computer",)

    def test_leaves_the_original_untouched(self) -> None:
        """The caller's config must survive the call unchanged."""
        cfg = VoiceGateConfig()
        cfg.with_phrases(("computer",))

        assert cfg.phrases == DEFAULT_PHRASES
        assert cfg.normalized_phrases == ("ok google", "hey google")

    def test_preserves_every_other_field(self) -> None:
        """Only phrases change; the rest of the configuration carries over."""
        cfg = _cfg(
            enabled=True,
            model_path="/models/small",
            pre_roll_ms=500,
            post_roll_ms=750,
            max_snippet_ms=5000,
            cooldown_ms=250,
            snippet_dir="clips",
            worker_python="/opt/py/python",
            startup_timeout_s=5.0,
        )
        new = cfg.with_phrases(("computer",))

        assert new.enabled is True
        assert new.model_path == "/models/small"
        assert new.pre_roll_ms == 500
        assert new.post_roll_ms == 750
        assert new.max_snippet_ms == 5000
        assert new.cooldown_ms == 250
        assert new.snippet_dir == "clips"
        assert new.worker_python == "/opt/py/python"
        assert new.startup_timeout_s == pytest.approx(5.0)

    def test_a_list_is_coerced_to_a_tuple(self) -> None:
        """with_phrases calls tuple() itself, so a list argument is accepted here."""
        new = VoiceGateConfig().with_phrases(["computer", "jarvis"])  # type: ignore[arg-type]
        assert new.phrases == ("computer", "jarvis")
        assert isinstance(new.phrases, tuple)

    def test_the_result_is_revalidated(self) -> None:
        """An enabled gate cannot be re-pointed at phrases that mean nothing."""
        cfg = _cfg(enabled=True)
        with pytest.raises(ValueError, match=r"survives normalisation"):
            cfg.with_phrases(("", "!!!"))

    def test_a_rejected_change_leaves_the_original_usable(self) -> None:
        """A failed with_phrases must not half-apply."""
        cfg = _cfg(enabled=True)
        with pytest.raises(ValueError):
            cfg.with_phrases(("",))
        assert cfg.normalized_phrases == ("ok google", "hey google")

    def test_junk_phrases_are_allowed_while_disabled(self) -> None:
        """Validation follows enabled, so a disabled gate accepts anything."""
        new = VoiceGateConfig().with_phrases(("",))
        assert new.phrases == ("",)
        assert new.normalized_phrases == ()


class TestWithEnabled:
    """with_enabled() flips the switch, revalidating as it goes."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_returns_a_new_object_with_the_flag_set(self, enabled: bool) -> None:
        """The flag lands on a fresh instance, not on the original."""
        cfg = VoiceGateConfig()
        new = cfg.with_enabled(enabled)

        assert new is not cfg
        assert new.enabled is enabled

    def test_leaves_the_original_untouched(self) -> None:
        """Enabling a copy must not switch the caller's config on."""
        cfg = VoiceGateConfig()
        cfg.with_enabled(True)
        assert cfg.enabled is False

    def test_preserves_every_other_field(self) -> None:
        """Only enabled changes."""
        cfg = _cfg(
            phrases=("computer",),
            pre_roll_ms=500,
            post_roll_ms=750,
            max_snippet_ms=5000,
            cooldown_ms=250,
            snippet_dir="clips",
        )
        new = cfg.with_enabled(True)

        assert new.phrases == ("computer",)
        assert new.pre_roll_ms == 500
        assert new.post_roll_ms == 750
        assert new.max_snippet_ms == 5000
        assert new.cooldown_ms == 250
        assert new.snippet_dir == "clips"

    def test_enabling_a_config_with_no_usable_phrase_is_rejected(self) -> None:
        """The junk a disabled gate was allowed to hold blocks the switch-on."""
        cfg = _cfg(phrases=("", "!!!"))
        with pytest.raises(ValueError, match=r"survives normalisation"):
            cfg.with_enabled(True)

    def test_disabling_is_always_allowed(self) -> None:
        """Turning the gate off can never fail; there is nothing left to validate."""
        cfg = _cfg(enabled=True, phrases=("ok google",))
        assert cfg.with_enabled(False).enabled is False

    @pytest.mark.parametrize(("given", "expected"), [(1, True), (0, False), ("", False)])
    def test_the_flag_is_coerced_to_a_bool(self, given: Any, expected: bool) -> None:
        """with_enabled calls bool(), so the stored field is always a real bool."""
        new = VoiceGateConfig().with_enabled(given)
        assert new.enabled is expected

    def test_round_tripping_the_flag_restores_the_config(self) -> None:
        """on-then-off gives back something equal to what was started with."""
        cfg = VoiceGateConfig()
        assert cfg.with_enabled(True).with_enabled(False) == cfg


class TestImmutability:
    """The config is frozen so the GUI can swap it without a lock."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("enabled", True),
            ("phrases", ("computer",)),
            ("model_path", "/models/small"),
            ("pre_roll_ms", 100),
            ("post_roll_ms", 100),
            ("max_snippet_ms", 100),
            ("cooldown_ms", 100),
            ("snippet_dir", "clips"),
            ("worker_python", "/opt/py"),
            ("startup_timeout_s", 1.0),
        ],
    )
    def test_fields_cannot_be_assigned(self, field: str, value: Any) -> None:
        """Assignment raises and does not take effect."""
        cfg = VoiceGateConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(cfg, field, value)
        assert getattr(cfg, field) == DEFAULTS[field], (
            "the failed assignment must not take effect"
        )

    def test_unknown_attributes_are_rejected(self) -> None:
        """slots=True plus frozen=True means no ad-hoc attributes.

        The exception *type* is version-dependent on CPython 3.11; see
        ``tests/test_types.py::test_audiochunk_rejects_unknown_attribute``.
        What matters is that the write does not stick.
        """
        cfg = VoiceGateConfig()
        with pytest.raises(
            (dataclasses.FrozenInstanceError, AttributeError, TypeError)
        ):
            cfg.some_new_field = 1  # type: ignore[attr-defined]
        assert not hasattr(cfg, "some_new_field")

    def test_equal_configs_compare_equal(self) -> None:
        """Frozen dataclass equality is by field, which the GUI relies on."""
        assert VoiceGateConfig() == VoiceGateConfig()
        assert VoiceGateConfig() != _cfg(pre_roll_ms=0)

    def test_the_config_is_hashable(self) -> None:
        """Hashability is exactly why phrases has to be a tuple."""
        assert hash(VoiceGateConfig()) == hash(VoiceGateConfig())
