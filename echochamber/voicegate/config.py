"""Configuration for the wake-phrase gate.

Follows :class:`~echochamber.config.AudioConfig` exactly: a frozen dataclass
validated in ``__post_init__``, durations in milliseconds, frame counts
derived.  The GUI swaps the whole object rather than mutating a field, which
is atomic under the GIL and so needs no lock.

Durations are **not** given a sample rate of their own.  The gate always runs
at the capture rate, and a second rate here would be a second source of truth
for something the pipeline already fixes -- so ``*_frames`` are methods taking
the rate rather than properties assuming one.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass

from echochamber.config import ms_to_frames
from echochamber.voicegate.matching import normalize

__all__ = ["DEFAULT_PHRASES", "ClipMode", "VoiceGateConfig"]


class ClipMode(enum.Enum):
    """What a snippet is cut to.

    Attributes:
        PHRASE: The wake phrase itself, plus ``lead_ms`` and ``trail_ms`` of
            padding, located from the decoder's per-word timings.  This is what
            "send the audio that triggered it" means literally: the file is the
            hotword and little else, typically well under a second.  Requires a
            backend that reports timings; the gate falls back to :attr:`WINDOW`
            per snippet when they are missing or fail their sanity check.
        WINDOW: A fixed window -- ``pre_roll_ms`` before the detection and
            ``post_roll_ms`` after it.  Captures the *command* following the
            wake word, not just the trigger, and needs nothing from the decoder
            beyond the text.
    """

    PHRASE = "phrase"
    WINDOW = "window"

DEFAULT_PHRASES: tuple[str, ...] = ("ok google", "hey google")
"""Phrases the gate listens for out of the box."""


@dataclass(frozen=True, slots=True)
class VoiceGateConfig:
    """Immutable wake-phrase gate configuration.

    Attributes:
        enabled: Whether the gate runs at all.  ``False`` by default: the gate
            needs a model on disk that this repository does not ship, so
            defaulting it on would make a fresh checkout fail at start.
        phrases: Wake phrases to listen for.  Matched case- and
            punctuation-insensitively against whole words; see
            :func:`~echochamber.voicegate.matching.match_phrase`.
        model_path: Directory of the Vosk model, or ``None`` to leave the gate
            without a recogniser (it then never fires).
        clip_mode: What each snippet is cut to; see :class:`ClipMode`.  Defaults
            to :attr:`ClipMode.PHRASE`, so a snippet is the hotword rather than
            a window around it.
        lead_ms: Audio kept before the wake phrase in :attr:`ClipMode.PHRASE`.
            Small on purpose -- enough to avoid clipping the initial consonant,
            not enough to include whatever came before.
        trail_ms: Audio kept after the wake phrase in :attr:`ClipMode.PHRASE`.
        lookback_ms: How much recent audio is retained so a phrase can be cut
            out of the past.  This is **not** how much ends up in the file: it
            is the window the gate can reach back into, and it must comfortably
            exceed the decoder's reporting lag, which for a small model running
            behind a bounded queue is routinely a second or more.  Costs
            ``lookback_ms`` of 16-bit mono -- 8 s is 256 KB.
        pre_roll_ms: Audio kept from *before* the match lands in the snippet.
            This is not cosmetic: the recogniser only reports a phrase once it
            has consumed the audio containing it, so without a pre-roll the
            wake phrase itself is already in the past and the snippet would
            begin after it.  The default covers a slowly spoken phrase plus a
            little lead-in.
        post_roll_ms: Audio recorded after the match before the snippet closes.
        max_snippet_ms: Hard ceiling on one snippet's length.  A repeated match
            extends a snippet's post-roll, so without this a phrase spoken over
            and over would record indefinitely.
        cooldown_ms: Refractory period after a snippet closes, during which a
            further match is ignored.  Small models routinely emit the same
            phrase twice across consecutive results; without a cooldown that
            writes two near-identical files.
        snippet_dir: Directory snippets are written to.  Created on demand.
        worker_python: Interpreter used for the subprocess recogniser backend,
            or ``None`` for the in-process one.  On Windows ARM64 this is the
            path to an **x64** ``python.exe``; see
            :mod:`echochamber.voicegate.subprocess_recognizer` for why.
        startup_timeout_s: How long to wait for the subprocess backend to load
            its model and report readiness before giving up.

    Raises:
        ValueError: If any duration is negative, if ``pre_roll_ms`` plus
            ``post_roll_ms`` exceeds ``max_snippet_ms`` (which would make every
            snippet hit the ceiling immediately), or if ``phrases`` contains
            nothing that survives normalisation while the gate is enabled.
    """

    enabled: bool = False
    phrases: tuple[str, ...] = DEFAULT_PHRASES
    model_path: str | None = None
    clip_mode: ClipMode = ClipMode.PHRASE
    lead_ms: int = 250
    trail_ms: int = 250
    lookback_ms: int = 8000
    pre_roll_ms: int = 1500
    post_roll_ms: int = 3000
    max_snippet_ms: int = 15_000
    cooldown_ms: int = 1000
    snippet_dir: str = "snippets"
    worker_python: str | None = None
    startup_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate the configuration; see the class docstring for the rules."""
        if self.lead_ms < 0:
            raise ValueError(f"lead_ms must be >= 0, got {self.lead_ms}")
        if self.trail_ms < 0:
            raise ValueError(f"trail_ms must be >= 0, got {self.trail_ms}")
        if self.lookback_ms <= 0:
            raise ValueError(f"lookback_ms must be > 0, got {self.lookback_ms}")
        if not isinstance(self.clip_mode, ClipMode):
            raise TypeError(
                f"clip_mode must be a ClipMode, got "
                f"{type(self.clip_mode).__name__}"
            )
        if self.pre_roll_ms < 0:
            raise ValueError(f"pre_roll_ms must be >= 0, got {self.pre_roll_ms}")
        if self.post_roll_ms < 0:
            raise ValueError(f"post_roll_ms must be >= 0, got {self.post_roll_ms}")
        if self.cooldown_ms < 0:
            raise ValueError(f"cooldown_ms must be >= 0, got {self.cooldown_ms}")
        if self.max_snippet_ms <= 0:
            raise ValueError(
                f"max_snippet_ms must be > 0, got {self.max_snippet_ms}"
            )
        if self.startup_timeout_s <= 0:
            raise ValueError(
                f"startup_timeout_s must be > 0, got {self.startup_timeout_s}"
            )

        span = self.pre_roll_ms + self.post_roll_ms
        if span > self.max_snippet_ms:
            raise ValueError(
                f"pre_roll_ms + post_roll_ms ({self.pre_roll_ms} + "
                f"{self.post_roll_ms} = {span}) must be <= max_snippet_ms "
                f"({self.max_snippet_ms}); otherwise every snippet is truncated "
                f"the moment it opens"
            )

        if not isinstance(self.phrases, tuple):
            raise TypeError(
                f"phrases must be a tuple so the config stays hashable and "
                f"immutable, got {type(self.phrases).__name__}"
            )
        if self.enabled and not self.normalized_phrases:
            raise ValueError(
                "phrases must contain at least one entry that survives "
                "normalisation when the gate is enabled; got "
                f"{self.phrases!r}, which normalises to nothing"
            )
        if not self.snippet_dir:
            raise ValueError("snippet_dir must not be empty")

    @property
    def normalized_phrases(self) -> tuple[str, ...]:
        """The configured phrases, normalised, deduplicated, order preserved.

        Entries that normalise to nothing are dropped rather than kept as an
        empty string, which would otherwise match every utterance.
        """
        seen: dict[str, None] = {}
        for phrase in self.phrases:
            normalized = normalize(phrase)
            if normalized:
                seen.setdefault(normalized, None)
        return tuple(seen)

    def lead_frames(self, sample_rate: int) -> int:
        """Padding before the phrase, in frames at ``sample_rate``.

        Args:
            sample_rate: Capture sample rate in Hz.

        Returns:
            The frame count.
        """
        return ms_to_frames(self.lead_ms, sample_rate)

    def trail_frames(self, sample_rate: int) -> int:
        """Padding after the phrase, in frames at ``sample_rate``.

        Args:
            sample_rate: Capture sample rate in Hz.

        Returns:
            The frame count.
        """
        return ms_to_frames(self.trail_ms, sample_rate)

    def lookback_frames(self, sample_rate: int) -> int:
        """Retained-audio window, in frames at ``sample_rate``.

        Args:
            sample_rate: Capture sample rate in Hz.

        Returns:
            The frame count; at least enough to hold the configured pre-roll,
            since :attr:`ClipMode.WINDOW` cuts out of the same buffer and a
            lookback shorter than the pre-roll would silently shorten it.
        """
        return max(
            ms_to_frames(self.lookback_ms, sample_rate),
            self.pre_roll_frames(sample_rate),
        )

    def pre_roll_frames(self, sample_rate: int) -> int:
        """Pre-roll length in frames at ``sample_rate``.

        Args:
            sample_rate: Capture sample rate in Hz.

        Returns:
            The frame count, ``0`` when no pre-roll was configured.
        """
        return ms_to_frames(self.pre_roll_ms, sample_rate)

    def post_roll_frames(self, sample_rate: int) -> int:
        """Post-roll length in frames at ``sample_rate``.

        Args:
            sample_rate: Capture sample rate in Hz.

        Returns:
            The frame count, ``0`` when no post-roll was configured.
        """
        return ms_to_frames(self.post_roll_ms, sample_rate)

    def max_snippet_frames(self, sample_rate: int) -> int:
        """Maximum snippet length in frames at ``sample_rate``.

        Args:
            sample_rate: Capture sample rate in Hz.

        Returns:
            The frame count; always at least 1, so a snippet can never be
            configured into being empty.
        """
        return max(1, ms_to_frames(self.max_snippet_ms, sample_rate))

    def cooldown_frames(self, sample_rate: int) -> int:
        """Refractory period in frames at ``sample_rate``.

        Args:
            sample_rate: Capture sample rate in Hz.

        Returns:
            The frame count, ``0`` when no cooldown was configured.
        """
        return ms_to_frames(self.cooldown_ms, sample_rate)

    def with_phrases(self, phrases: tuple[str, ...]) -> "VoiceGateConfig":
        """Return a copy listening for ``phrases``.

        Args:
            phrases: The new wake phrases.

        Returns:
            A new validated :class:`VoiceGateConfig`.

        Raises:
            ValueError: If the result would be invalid.
        """
        return dataclasses.replace(self, phrases=tuple(phrases))

    def with_enabled(self, enabled: bool) -> "VoiceGateConfig":
        """Return a copy with the gate switched on or off.

        Args:
            enabled: Whether the gate should run.

        Returns:
            A new validated :class:`VoiceGateConfig`.

        Raises:
            ValueError: If enabling would leave the config without a usable
                phrase.
        """
        return dataclasses.replace(self, enabled=bool(enabled))
