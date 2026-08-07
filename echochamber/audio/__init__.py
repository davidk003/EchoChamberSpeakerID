"""Audio capture primitives: value types, the SPSC ring buffer, the chunker.

Re-exports the public surface so callers can write
``from echochamber.audio import RingBuffer, AudioChunk, WindowChunker``.  This
module deliberately does not import :mod:`echochamber.config` (which imports
from here), keeping the dependency direction one-way -- which is also why
:mod:`echochamber.audio.chunker` only imports ``AudioConfig`` for type checking.
"""

from __future__ import annotations

from echochamber.audio.chunker import POLL_INTERVAL_S, ChunkCallback, WindowChunker
from echochamber.audio.ringbuffer import OverrunError, RingBuffer
from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats

__all__ = [
    "POLL_INTERVAL_S",
    "AudioChunk",
    "ChunkCallback",
    "DropPolicy",
    "OverrunError",
    "RingBuffer",
    "StreamStats",
    "WindowChunker",
]
