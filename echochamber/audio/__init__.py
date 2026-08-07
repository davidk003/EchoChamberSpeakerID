"""Audio capture primitives: value types, ring buffer, chunker, sinks, pipeline.

Re-exports the public surface so callers can write
``from echochamber.audio import AudioPipeline, FileSource, WavRecorderSink``.
This module deliberately does not import :mod:`echochamber.config` (which
imports from here), keeping the dependency direction one-way -- which is also
why :mod:`echochamber.audio.chunker` and :mod:`echochamber.audio.pipeline` only
import ``AudioConfig`` for type checking.
"""

from __future__ import annotations

from echochamber.audio.chunker import POLL_INTERVAL_S, ChunkCallback, WindowChunker
from echochamber.audio.devices import (
    WASAPI_HOSTAPI_NAME,
    DeviceError,
    DeviceInfo,
    default_input_device,
    find_input_device,
    list_input_devices,
)
from echochamber.audio.latency import LatencySummary, LatencyTracker
from echochamber.audio.pipeline import (
    DRAIN_POLL_S,
    DRAIN_SETTLE_S,
    AudioPipeline,
    SourceFactory,
)
from echochamber.audio.ringbuffer import OverrunError, RingBuffer
from echochamber.audio.sinks import (
    CallableSink,
    ChunkSink,
    QueueSink,
    TeeSink,
    WavRecorderSink,
    new_frame_count,
)
from echochamber.audio.sources import (
    AudioCallback,
    AudioSource,
    FileSource,
    SoundDeviceSource,
)
from echochamber.audio.stub_consumer import StubInferenceSink
from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats

__all__ = [
    "DRAIN_POLL_S",
    "DRAIN_SETTLE_S",
    "POLL_INTERVAL_S",
    "WASAPI_HOSTAPI_NAME",
    "AudioCallback",
    "AudioChunk",
    "AudioPipeline",
    "AudioSource",
    "CallableSink",
    "ChunkCallback",
    "ChunkSink",
    "DeviceError",
    "DeviceInfo",
    "DropPolicy",
    "FileSource",
    "LatencySummary",
    "LatencyTracker",
    "OverrunError",
    "QueueSink",
    "RingBuffer",
    "SoundDeviceSource",
    "SourceFactory",
    "StreamStats",
    "StubInferenceSink",
    "TeeSink",
    "WavRecorderSink",
    "WindowChunker",
    "default_input_device",
    "find_input_device",
    "list_input_devices",
    "new_frame_count",
]
