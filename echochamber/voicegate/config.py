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
import os
from dataclasses import dataclass, field

from echochamber.config import ms_to_frames
from echochamber.voicegate.matching import normalize

__all__ = [
    "DEFAULT_PHRASES",
    "VoiceGateConfig",
    "autodetect_model_path",
    "autodetect_voice_gate_config",
    "autodetect_worker_python",
]

DEFAULT_PHRASES: tuple[str, ...] = ("ok google", "hey google")
"""Phrases the gate listens for out of the box."""

_VENV_DIR: str = ".venv-vosk"
"""Where :mod:`scripts.setup_voice_gate` builds the recogniser environment."""

_MODEL_DIR: str = "models"
"""Where :mod:`scripts.setup_voice_gate` unpacks the model."""

_MODEL_NAME: str = "vosk-model-small-en-us-0.15"
"""Must match ``scripts.setup_voice_gate.MODEL_NAME``."""

_REPO_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
"""Repository root, derived from this file's location rather than the
current working directory -- the app can be launched from anywhere, but the
venv and model :mod:`scripts.setup_voice_gate` created are always siblings of
this package, not of whatever directory the user happened to be in."""


def _venv_python(venv_dir: str) -> str:
    """Return the interpreter path inside a virtual environment.

    Mirrors ``scripts.setup_voice_gate.venv_python``; duplicated rather than
    imported because that script is a standard-library-only bootstrapper and
    must not depend on this package.
    """
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def autodetect_model_path(root: str | None = None) -> str | None:
    """Find the Vosk model :mod:`scripts.setup_voice_gate` would have unpacked.

    Args:
        root: Directory to look under; the repository root when ``None``.

    Returns:
        The model directory's absolute path if it exists, else ``None`` --
        callers treat that exactly like a user who left ``model_path`` unset.
    """
    base = _REPO_ROOT if root is None else root
    candidate = os.path.join(base, _MODEL_DIR, _MODEL_NAME)
    return candidate if os.path.isdir(candidate) else None


def autodetect_worker_python(root: str | None = None) -> str | None:
    """Find the x64 recogniser venv :mod:`scripts.setup_voice_gate` would have built.

    Only relevant to the subprocess backend; a machine where vosk installs
    directly (anywhere but Windows ARM64) has no reason to have this venv, and
    :func:`~echochamber.voicegate.backends.build_recognizer` uses the
    in-process backend when ``worker_python`` is unset.

    Args:
        root: Directory to look under; the repository root when ``None``.

    Returns:
        The venv's interpreter path if it exists, else ``None``.
    """
    base = _REPO_ROOT if root is None else root
    candidate = _venv_python(os.path.join(base, _VENV_DIR))
    return candidate if os.path.isfile(candidate) else None


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
    pre_roll_ms: int = 1500
    post_roll_ms: int = 3000
    max_snippet_ms: int = 15_000
    cooldown_ms: int = 1000
    snippet_dir: str = "snippets"
    worker_python: str | None = None
    startup_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate the configuration; see the class docstring for the rules."""
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


def autodetect_voice_gate_config(**overrides: object) -> VoiceGateConfig:
    """Build a :class:`VoiceGateConfig` with ``model_path``/``worker_python``
    filled in from what :mod:`scripts.setup_voice_gate` left on disk.

    ``VoiceGateConfig()`` itself defaults both to ``None`` -- deliberately,
    since the dataclass has no business knowing about the filesystem, and the
    test suite pins those defaults (see ``tests/test_voicegate_config.py``).
    This is the seam callers who *do* want the filesystem consulted use
    instead: the application's entry point, not the config type. Detected
    values only fill in fields the caller did not already pass in
    ``overrides`` -- an explicit ``model_path=None`` still means "no model",
    not "go detect one".

    Args:
        **overrides: Passed through to :class:`VoiceGateConfig`, taking
            precedence over anything detected.

    Returns:
        A validated :class:`VoiceGateConfig`.
    """
    if "model_path" not in overrides:
        overrides["model_path"] = autodetect_model_path()
    if "worker_python" not in overrides:
        overrides["worker_python"] = autodetect_worker_python()
    return VoiceGateConfig(**overrides)
