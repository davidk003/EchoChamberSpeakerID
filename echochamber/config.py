"""Configuration for the audio ingestion pipeline.

Window geometry is expressed in milliseconds (``window_ms`` / ``hop_ms``) and
converted to integer frame counts once, here.  Overlap is *derived* from window
and hop rather than configured directly, which keeps the chunk cadence exact
and integral.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from echochamber.audio.types import DropPolicy

__all__ = ["ms_to_frames", "AudioConfig"]


def ms_to_frames(ms: float, sample_rate: int) -> int:
    """Convert a duration in milliseconds to a whole number of frames.

    Args:
        ms: Duration in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        ``round(ms * sample_rate / 1000)``, clamped so the result is never
        negative.
    """
    frames = round(ms * sample_rate / 1000.0)
    return max(0, int(frames))


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Immutable capture and windowing configuration.

    The GUI reconfigures the pipeline by swapping the whole frozen object into
    a single attribute, which is atomic under the GIL; the chunker re-reads it
    at the top of each iteration.

    Attributes:
        sample_rate: Capture sample rate in Hz.
        channels: Device channel count; audio is downmixed to mono.
        blocksize: Device callback size in frames (160 = 10 ms @ 16 kHz).
        ring_seconds: Ring buffer capacity in seconds.
        window_ms: Window length in milliseconds -- how much audio each chunk
            contains.
        hop_ms: Hop length in milliseconds -- how far the window advances.
        queue_max: Maximum depth of the bounded handoff queue.
        drop_policy: What a full queue does with a new chunk.

    Raises:
        ValueError: If any field is out of range, if the hop exceeds the
            window (this pipeline is specified for *overlapping* windows), or
            if the ring cannot hold a window plus a hop of slack.
    """

    sample_rate: int = 16_000
    channels: int = 1
    blocksize: int = 160
    ring_seconds: float = 10.0
    window_ms: int = 3000
    hop_ms: int = 1000
    queue_max: int = 8
    drop_policy: DropPolicy = DropPolicy.DROP_OLDEST

    def __post_init__(self) -> None:
        """Validate the configuration; see the class docstring for the rules."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {self.sample_rate}")
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels}")
        if self.blocksize <= 0:
            raise ValueError(f"blocksize must be > 0, got {self.blocksize}")
        if self.queue_max < 1:
            raise ValueError(f"queue_max must be >= 1, got {self.queue_max}")
        if self.ring_seconds <= 0:
            raise ValueError(f"ring_seconds must be > 0, got {self.ring_seconds}")
        if self.window_ms <= 0:
            raise ValueError(f"window_ms must be > 0, got {self.window_ms}")
        if self.hop_ms <= 0:
            raise ValueError(f"hop_ms must be > 0, got {self.hop_ms}")

        window_frames = self.window_frames
        hop_frames = self.hop_frames
        # A sub-frame window/hop rounds to zero frames, which would make the
        # pipeline emit empty chunks and make overlap_ratio divide by zero.
        # window_ms > 0 alone does not catch this at low sample rates.
        if window_frames <= 0:
            raise ValueError(
                f"window_ms ({self.window_ms}) rounds to {window_frames} frames at "
                f"sample_rate={self.sample_rate}; the window must be at least 1 frame"
            )
        if hop_frames <= 0:
            raise ValueError(
                f"hop_ms ({self.hop_ms}) rounds to {hop_frames} frames at "
                f"sample_rate={self.sample_rate}; the hop must be at least 1 frame"
            )
        if hop_frames > window_frames:
            raise ValueError(
                f"hop_frames ({hop_frames}) must be <= window_frames "
                f"({window_frames}); a hop larger than the window would skip "
                f"audio, and this pipeline is specified for overlapping windows"
            )

        ring_frames = self.ring_frames
        if ring_frames < window_frames + hop_frames:
            raise ValueError(
                f"ring_frames ({ring_frames}) must be >= window_frames + "
                f"hop_frames ({window_frames} + {hop_frames} = "
                f"{window_frames + hop_frames}); the ring must hold a window "
                f"plus a hop of slack"
            )

    @property
    def window_frames(self) -> int:
        """Window length in frames."""
        return ms_to_frames(self.window_ms, self.sample_rate)

    @property
    def hop_frames(self) -> int:
        """Hop length in frames; also the chunk cadence."""
        return ms_to_frames(self.hop_ms, self.sample_rate)

    @property
    def overlap_frames(self) -> int:
        """Frames shared by consecutive windows (``window_frames - hop_frames``)."""
        return self.window_frames - self.hop_frames

    @property
    def overlap_ratio(self) -> float:
        """Fraction of each window that overlaps the previous one, 0.0-1.0."""
        return self.overlap_frames / self.window_frames

    @property
    def ring_frames(self) -> int:
        """Ring buffer capacity in frames."""
        return round(self.ring_seconds * self.sample_rate)

    def with_window(
        self,
        *,
        window_ms: int | None = None,
        hop_ms: int | None = None,
    ) -> "AudioConfig":
        """Return a copy with new window geometry.

        Args:
            window_ms: New window length in milliseconds, or ``None`` to keep
                the current value.
            hop_ms: New hop length in milliseconds, or ``None`` to keep the
                current value.

        Returns:
            A new validated :class:`AudioConfig`.

        Raises:
            ValueError: If the resulting configuration is invalid.
        """
        return dataclasses.replace(
            self,
            window_ms=self.window_ms if window_ms is None else window_ms,
            hop_ms=self.hop_ms if hop_ms is None else hop_ms,
        )
