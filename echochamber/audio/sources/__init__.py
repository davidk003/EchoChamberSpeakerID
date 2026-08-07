"""Audio sources: the interchangeable front ends of the pipeline.

Every source hands mono ``float32`` blocks to a callback -- in the assembled
pipeline that callback is :meth:`RingBuffer.write
<echochamber.audio.ringbuffer.RingBuffer.write>`, so a source never knows the
ring exists.  That is what lets :class:`~echochamber.audio.sources.file_source.FileSource`
substitute for live capture with nothing else in the pipeline changing.
"""

from __future__ import annotations

from echochamber.audio.sources.base import AudioCallback, AudioSource
from echochamber.audio.sources.file_source import FileSource

__all__ = [
    "AudioCallback",
    "AudioSource",
    "FileSource",
]
