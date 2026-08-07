"""Value types that cross thread boundaries in the audio pipeline.

These are deliberately plain: a frozen chunk of audio, a mutable statistics
record with a snapshot method, and the backpressure policy enum.  Nothing in
this module touches Qt, PortAudio, or any compiled extension beyond numpy.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass

import numpy as np

__all__ = ["DropPolicy", "AudioChunk", "StreamStats"]


class DropPolicy(enum.Enum):
    """What a bounded sink does when its queue is full.

    Attributes:
        DROP_OLDEST: Discard the oldest queued chunk and enqueue the new one.
            Freshness beats completeness for live classification; this is the
            default for live capture.
        BLOCK: Back-pressure the producer until space is available.  Only
            correct for file replay, where the source can be slowed down.
    """

    DROP_OLDEST = "drop_oldest"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One window of mono audio handed to a downstream sink.

    ``samples`` must be an **owned copy**, never a view into the ring buffer:
    the producer will eventually overwrite the ring region a view points at.
    See :meth:`echochamber.audio.ringbuffer.RingBuffer.read`.

    Attributes:
        samples: 1-D ``float32`` mono sample array, owned by this chunk.
        start_frame: Absolute frame index of ``samples[0]`` since stream start.
        seq: Chunk index, counting from 0.
        sample_rate: Sample rate in Hz of ``samples``.
        discontinuous: ``True`` if audio was lost (overrun) or the window
            geometry was reconfigured immediately before this chunk, so
            downstream must not assume continuity with the previous chunk.
        capture_time: ``time.perf_counter()`` reading taken when the chunker
            emitted this chunk, i.e. just after its final sample was available.
            A consumer measures its own end-to-end latency as
            ``time.perf_counter() - capture_time``.

            **Use perf_counter, not monotonic, to compare against this.** Both
            are monotonic, but on Windows ``time.monotonic()`` is
            ``GetTickCount64`` with **15.625 ms** resolution, which quantises a
            sub-millisecond handoff into alternating 0 ms and 16 ms readings --
            noise that looks exactly like real latency spikes.
            ``perf_counter`` is ``QueryPerformanceCounter``, ~100 ns.

            ``0.0`` means unset (a chunk built by hand in a test).
    """

    samples: np.ndarray
    start_frame: int
    seq: int
    sample_rate: int
    discontinuous: bool = False
    capture_time: float = 0.0

    @property
    def n_frames(self) -> int:
        """Number of frames in this chunk (``len(samples)``)."""
        return len(self.samples)

    @property
    def duration_s(self) -> float:
        """Duration of this chunk in seconds."""
        return self.n_frames / self.sample_rate

    @property
    def start_time_s(self) -> float:
        """Start time of this chunk in seconds since stream start."""
        return self.start_frame / self.sample_rate

    def age_s(self, now: float) -> float:
        """Seconds between this chunk being emitted and ``now``.

        ``now`` is supplied by the caller (a :func:`time.perf_counter` reading)
        rather than read here, so latency accounting stays testable without
        sleeping.  Returns 0.0 when :attr:`capture_time` was never set, and is
        clamped at 0.0 so a caller can never see a negative latency.

        Passing a :func:`time.monotonic` reading here mixes two unrelated clock
        origins and yields nonsense; see :attr:`capture_time`.
        """
        if not self.capture_time:
            return 0.0
        return max(0.0, now - self.capture_time)


@dataclass(slots=True)
class StreamStats:
    """Mutable counters describing the health of a running stream.

    Owned and mutated by the audio path; the GUI polls :meth:`snapshot` on a
    timer rather than receiving a signal per chunk.

    Attributes:
        frames_captured: Total frames written into the ring since start.
        chunks_emitted: Total chunks produced by the chunker.
        chunks_dropped: Chunks discarded by a bounded sink under
            :attr:`DropPolicy.DROP_OLDEST`.
        overruns: Times the ring reader fell behind the writer and lost audio.
        xruns: Device-level input overflows reported by the capture backend.
        peak_level: Most recent peak absolute sample level, 0.0-1.0.
        rms_level: Most recent RMS sample level, 0.0-1.0.
    """

    frames_captured: int = 0
    chunks_emitted: int = 0
    chunks_dropped: int = 0
    overruns: int = 0
    xruns: int = 0
    peak_level: float = 0.0
    rms_level: float = 0.0

    def snapshot(self) -> "StreamStats":
        """Return a detached copy of these stats.

        The GUI thread reads the copy, so a concurrent update from the audio
        path cannot tear the values it is displaying.
        """
        return dataclasses.replace(self)
