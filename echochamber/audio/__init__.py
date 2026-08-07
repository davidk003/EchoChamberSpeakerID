"""Audio capture primitives: value types and the SPSC ring buffer.

Re-exports the step-1 public surface so callers can write
``from echochamber.audio import RingBuffer, AudioChunk``.  This module
deliberately does not import :mod:`echochamber.config` (which imports from
here), keeping the dependency direction one-way.
"""

from __future__ import annotations

from echochamber.audio.ringbuffer import OverrunError, RingBuffer
from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats

__all__ = [
    "AudioChunk",
    "DropPolicy",
    "OverrunError",
    "RingBuffer",
    "StreamStats",
]
