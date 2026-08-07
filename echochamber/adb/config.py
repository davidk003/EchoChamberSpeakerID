"""Configuration for the adb hotword trigger.

Follows :class:`~echochamber.speakerid.config.SpeakerIdConfig` and
:class:`~echochamber.voicegate.config.VoiceGateConfig` exactly: a frozen
dataclass validated in ``__post_init__``, with any filesystem/tool detection
kept in a free function so the type itself knows nothing about the machine
it happens to run on.

Unlike its siblings there is no heavy model or venv to locate here --
``adb.exe`` is a single executable, and finding it
(:func:`echochamber.adb.hotword_core.find_adb`) is cheap enough to defer to
the moment the trigger actually starts rather than to config construction.
So ``adb_path`` stays ``None`` by default and is resolved lazily by
:func:`echochamber.adb.trigger.build_adb_trigger`, exactly as
``VoiceGateConfig.model_path`` is resolved by
:func:`~echochamber.voicegate.backends.build_recognizer` rather than by the
config type.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

__all__ = ["AdbTriggerConfig", "autodetect_adb_trigger_config"]


@dataclass(frozen=True, slots=True)
class AdbTriggerConfig:
    """Immutable adb hotword trigger configuration.

    Attributes:
        enabled: Whether the trigger runs at all.  ``False`` by default: it
            shells out to ``adb`` and revokes a permission on whatever device
            is attached, which is not something a capture tool should start
            doing because a feature was merged.
        adb_path: Explicit path to ``adb.exe``, or ``None`` to autodetect it
            via :func:`echochamber.adb.hotword_core.find_adb` at build time.
            An explicit ``None`` here still means "autodetect" -- unlike the
            sibling configs' model paths, there is no meaningful "trigger
            enabled but deliberately pathless" state, since there is nothing
            else ``adb_path`` could reasonably default to.

    Raises:
        ValueError: If ``adb_path`` is set to an empty string; use ``None``
            to ask for autodetection instead.
    """

    enabled: bool = False
    adb_path: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration; see the class docstring for the rules."""
        if self.adb_path is not None and not self.adb_path:
            raise ValueError(
                "adb_path must not be an empty string; use None to autodetect"
            )

    def with_enabled(self, enabled: bool) -> "AdbTriggerConfig":
        """Return a copy with the trigger switched on or off.

        Args:
            enabled: Whether the trigger should run.

        Returns:
            A new validated :class:`AdbTriggerConfig`.
        """
        return dataclasses.replace(self, enabled=bool(enabled))

    def __repr__(self) -> str:
        """Return a debugging representation of this configuration."""
        return (
            f"{type(self).__name__}(enabled={self.enabled}, "
            f"adb_path={self.adb_path!r})"
        )


def autodetect_adb_trigger_config(**overrides: object) -> AdbTriggerConfig:
    """Build an :class:`AdbTriggerConfig`.

    ``AdbTriggerConfig()`` itself defaults ``adb_path`` to ``None`` and does
    not touch the filesystem or spawn a process -- deliberately, since the
    dataclass has no business knowing about disk state; the test suite pins
    those defaults. Unlike :func:`~echochamber.voicegate.config.autodetect_voice_gate_config`
    and :func:`~echochamber.speakerid.config.autodetect_speaker_id_config`,
    there is nothing to detect here beyond ``adb`` itself, and that detection
    is deferred to :func:`echochamber.adb.trigger.build_adb_trigger` -- this
    function exists only for symmetry with the sibling packages' entry
    points, and because a caller who does want the filesystem consulted for
    ``adb_path`` at config time still applies overrides here rather than at
    the dataclass directly.

    Args:
        **overrides: Passed through to :class:`AdbTriggerConfig`, taking
            precedence over the defaults.

    Returns:
        A validated :class:`AdbTriggerConfig`.
    """
    return AdbTriggerConfig(**overrides)
