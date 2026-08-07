"""Chunk sinks: where a completed window goes after the chunker emits it.

The pipeline terminates at a :class:`ChunkSink`.  Keeping it a
:class:`~typing.Protocol` rather than a base class means the future ML stage,
a recorder, a level meter and a test spy are all interchangeable without any
of them knowing about the others.

Three of the sinks here are plumbing -- :class:`CallableSink` adapts a plain
function, :class:`TeeSink` fans out, :class:`QueueSink` is the bounded handoff
that decouples a slow consumer from the chunker.  The fourth,
:class:`WavRecorderSink`, is the only one that has to *understand* windowing:
chunks overlap by ``W - H`` frames, so writing them back to back would stutter
the audio.  See :func:`new_frame_count`.

Threading: ``on_chunk`` is called on the chunker thread for every sink here.
:class:`QueueSink` is the boundary -- everything downstream of its
:meth:`~QueueSink.get` runs on the consumer thread instead.
"""

from __future__ import annotations

import collections
import os
import threading
import time
import wave
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from echochamber.audio.types import AudioChunk, DropPolicy, StreamStats

__all__ = [
    "CallableSink",
    "ChunkSink",
    "QueueSink",
    "TeeSink",
    "WavRecorderSink",
    "new_frame_count",
]


@runtime_checkable
class ChunkSink(Protocol):
    """Anything that can receive chunks and be shut down.

    Implementations must tolerate :meth:`close` being called more than once,
    because :meth:`echochamber.audio.pipeline.AudioPipeline.stop` is itself
    idempotent.
    """

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Handle one completed window.  Called on the producer's thread."""
        ...

    def close(self) -> None:
        """Release whatever the sink holds.  Must be idempotent."""
        ...


class CallableSink:
    """Adapt a plain ``fn(chunk)`` callable to the :class:`ChunkSink` protocol.

    :meth:`close` is a no-op: a function owns nothing to release.
    """

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[AudioChunk], None]) -> None:
        """Wrap ``fn``.

        Args:
            fn: Callable invoked with each chunk, on the caller's thread.
        """
        self._fn: Callable[[AudioChunk], None] = fn

    @property
    def fn(self) -> Callable[[AudioChunk], None]:
        """The wrapped callable."""
        return self._fn

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Forward ``chunk`` to the wrapped callable."""
        self._fn(chunk)

    def close(self) -> None:
        """No-op; present to satisfy :class:`ChunkSink`."""

    def __repr__(self) -> str:
        """Return a debugging representation of this sink."""
        return f"{type(self).__name__}(fn={self._fn!r})"


class TeeSink:
    """Fan one chunk out to several sinks, in order.

    **One broken sink must not starve the others.**  If a sink raises, the
    remaining sinks still receive the chunk and the *first* exception is
    re-raised once all of them have been attempted.  :meth:`close` follows the
    same policy.  The alternative -- bailing out on the first failure -- would
    mean a buggy meter silently stops the recorder, which is exactly the kind
    of failure that goes unnoticed for weeks.
    """

    __slots__ = ("_sinks",)

    def __init__(self, *sinks: ChunkSink) -> None:
        """Fan out to ``sinks`` in the order given.

        Args:
            *sinks: Sinks to deliver to.  Zero sinks is legal and makes this a
                black hole, which is occasionally useful in tests.
        """
        self._sinks: tuple[ChunkSink, ...] = tuple(sinks)

    @property
    def sinks(self) -> tuple[ChunkSink, ...]:
        """The wrapped sinks, in delivery order."""
        return self._sinks

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Deliver ``chunk`` to every sink.

        Args:
            chunk: Window to deliver.

        Raises:
            BaseException: The first exception raised by any sink, re-raised
                after every sink has been given the chunk.
        """
        first: BaseException | None = None
        for sink in self._sinks:
            try:
                sink.on_chunk(chunk)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if first is None:
                    first = exc
        if first is not None:
            raise first

    def close(self) -> None:
        """Close every sink, then re-raise the first exception if any.

        Raises:
            BaseException: The first exception raised by any sink's ``close``.
        """
        first: BaseException | None = None
        for sink in self._sinks:
            try:
                sink.close()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if first is None:
                    first = exc
        if first is not None:
            raise first

    def __repr__(self) -> str:
        """Return a debugging representation of this sink."""
        return f"{type(self).__name__}({', '.join(repr(s) for s in self._sinks)})"


class QueueSink:
    """Bounded hand-off queue between the chunker and a consumer thread.

    This is the pipeline's backpressure valve.  The chunker must never be made
    to wait on the downstream consumer: a chunker that stalls stops draining
    the ring buffer, and the ring then overruns and loses audio outright.  So
    under :attr:`~echochamber.audio.types.DropPolicy.DROP_OLDEST` a full queue
    discards its *head* -- freshness beats completeness for live
    classification -- and counts the loss in
    :attr:`~echochamber.audio.types.StreamStats.chunks_dropped`.
    :attr:`~echochamber.audio.types.DropPolicy.BLOCK` back-pressures the
    producer instead and is only correct for file replay, where the producer
    can legitimately be slowed down.

    :meth:`get` returns ``None`` exactly once the sink is **closed and
    drained**; that is the end-of-stream signal the consumer thread loops on.
    Chunks arriving after :meth:`close` are discarded silently (they are not
    counted as backpressure drops) -- during shutdown the producer is stopped
    first, so this should not normally happen at all.

    Threading: one producer, one consumer, and :meth:`close` from a third
    thread are all safe.  A single :class:`threading.Condition` guards the
    deque, the closed flag, and every wakeup.
    """

    __slots__ = ("_maxsize", "_policy", "_stats", "_items", "_cond", "_closed")

    def __init__(
        self,
        maxsize: int,
        policy: DropPolicy,
        stats: StreamStats | None = None,
    ) -> None:
        """Create an empty bounded queue.

        Args:
            maxsize: Maximum number of queued chunks; must be at least 1.
            policy: What a full queue does with a new chunk.
            stats: Counter record to update; a fresh :class:`StreamStats` is
                allocated when ``None``.  Pass the pipeline's shared instance
                so the GUI sees the drop count.

        Raises:
            ValueError: If ``maxsize`` is less than 1.
        """
        maxsize = int(maxsize)
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize: int = maxsize
        self._policy: DropPolicy = policy
        self._stats: StreamStats = StreamStats() if stats is None else stats
        self._items: collections.deque[AudioChunk] = collections.deque()
        self._cond: threading.Condition = threading.Condition()
        self._closed: bool = False

    @property
    def maxsize(self) -> int:
        """Maximum number of chunks held before the policy kicks in."""
        return self._maxsize

    @property
    def policy(self) -> DropPolicy:
        """The backpressure policy in force."""
        return self._policy

    @property
    def stats(self) -> StreamStats:
        """The live stats record this sink mutates (not a snapshot)."""
        return self._stats

    @property
    def qsize(self) -> int:
        """Number of chunks currently queued."""
        return len(self._items)

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Enqueue ``chunk`` according to the drop policy.

        Under ``DROP_OLDEST`` this never blocks.  Under ``BLOCK`` it waits for
        space, but always returns if the sink is closed while waiting, so
        shutdown can never wedge the producer.

        Args:
            chunk: Window to enqueue.
        """
        with self._cond:
            if self._closed:
                return
            if len(self._items) >= self._maxsize:
                if self._policy is DropPolicy.BLOCK:
                    while len(self._items) >= self._maxsize and not self._closed:
                        self._cond.wait()
                    if self._closed:
                        return
                else:
                    self._items.popleft()
                    self._stats.chunks_dropped += 1
            self._items.append(chunk)
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> AudioChunk | None:
        """Dequeue the next chunk, waiting if the queue is empty.

        Args:
            timeout: Maximum seconds to wait, or ``None`` to wait until a
                chunk arrives or the sink is closed.

        Returns:
            The next chunk, or ``None`` if the sink is closed and drained (the
            end-of-stream signal) or ``timeout`` expired first.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._items:
                if self._closed:
                    return None
                if deadline is None:
                    self._cond.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return None
                    self._cond.wait(remaining)
            chunk = self._items.popleft()
            # Wake a producer blocked under BLOCK now that there is room.
            self._cond.notify_all()
            return chunk

    def close(self) -> None:
        """Close the sink and wake every waiter.

        Queued chunks are **kept**: a consumer keeps receiving them and only
        sees ``None`` once the queue is empty, which is what lets shutdown
        drain rather than truncate.  Idempotent, safe from any thread.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __repr__(self) -> str:
        """Return a debugging representation of the queue's state."""
        return (
            f"{type(self).__name__}(maxsize={self._maxsize}, "
            f"policy={self._policy.name}, qsize={self.qsize}, "
            f"closed={self._closed})"
        )


def new_frame_count(
    chunk_start: int, chunk_len: int, next_expected: int
) -> tuple[int, int]:
    """Return ``(n_new, gap_frames)`` for a chunk given the next expected frame.

    Pure function, deliberately separate from :class:`WavRecorderSink` so the
    de-overlapping arithmetic can be tested without touching a file.

    Args:
        chunk_start: Absolute frame index of the chunk's first sample.
        chunk_len: Number of frames in the chunk.
        next_expected: Absolute frame index the writer expects next, i.e. one
            past the last frame already written.

    Returns:
        ``n_new``: how many frames from the **end** of the chunk are new and
        must be written -- ``hop_frames`` in steady state, since consecutive
        windows share ``window - hop`` frames.  Never more than ``chunk_len``.

        ``gap_frames``: frames missing between ``next_expected`` and the
        chunk's start.  Zero normally; positive after an overrun dropped audio.
    """
    chunk_end = chunk_start + chunk_len
    if chunk_start > next_expected:
        return chunk_len, chunk_start - next_expected
    return max(0, chunk_end - next_expected), 0


class WavRecorderSink:
    """Write a continuous WAV reconstruction of the original audio.

    Chunks *overlap*.  Concatenating them verbatim would repeat ``W - H``
    frames per hop and produce a stuttering file several times longer than the
    input, so this sink writes only the frames it has not already written:
    everything for the first chunk, then the last ``hop`` frames of each
    subsequent one.  :func:`new_frame_count` does that arithmetic.

    **Gaps are not zero-filled.**  If the ring overran, the chunker resyncs and
    the next chunk starts beyond ``next_expected``; the missing audio is simply
    absent from the file.  The recording is therefore *shorter* than wall-clock
    time when overruns occurred, and :attr:`gaps` reports how many times that
    happened -- check it before treating a recording as ground truth.

    Output is mono 16-bit PCM.  The file is opened in :meth:`__init__`, so a
    bad path fails at construction rather than on the audio thread.
    """

    __slots__ = ("_path", "_sample_rate", "_wav", "_next_expected", "_frames_written", "_gaps")

    def __init__(self, path: str | os.PathLike[str], sample_rate: int) -> None:
        """Open ``path`` for writing and prepare the header.

        Args:
            path: Destination ``.wav`` path.
            sample_rate: Sample rate written into the header, in Hz.

        Raises:
            ValueError: If ``sample_rate`` is not positive.
            OSError: If the file cannot be opened.
        """
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        self._path: str | os.PathLike[str] = path
        self._sample_rate: int = sample_rate
        wav = wave.open(os.fspath(path), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        self._wav: wave.Wave_write | None = wav
        self._next_expected: int | None = None
        self._frames_written: int = 0
        self._gaps: int = 0

    @property
    def path(self) -> str | os.PathLike[str]:
        """Destination path of the recording."""
        return self._path

    @property
    def sample_rate(self) -> int:
        """Sample rate written into the WAV header, in Hz."""
        return self._sample_rate

    @property
    def frames_written(self) -> int:
        """Total frames written to the file so far."""
        return self._frames_written

    @property
    def gaps(self) -> int:
        """Number of chunks that started past the expected frame (lost audio)."""
        return self._gaps

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has finalized the file."""
        return self._wav is None

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Write the frames of ``chunk`` that are not already on disk.

        Args:
            chunk: Window to record.  Its ``sample_rate`` is not checked
                against the header; the pipeline guarantees they agree.
        """
        wav = self._wav
        if wav is None:
            return
        if self._next_expected is None:
            # First chunk defines the origin, so all of it is new.
            self._next_expected = chunk.start_frame

        n_new, gap = new_frame_count(
            chunk.start_frame, chunk.n_frames, self._next_expected
        )
        if gap > 0:
            self._gaps += 1
        if n_new > 0:
            tail = chunk.samples[chunk.n_frames - n_new :]
            wav.writeframes(_to_int16(tail).tobytes())
            self._frames_written += n_new
        self._next_expected = max(
            self._next_expected, chunk.start_frame + chunk.n_frames
        )

    def close(self) -> None:
        """Finalize the WAV file.  Idempotent."""
        wav = self._wav
        if wav is None:
            return
        self._wav = None
        wav.close()

    def __repr__(self) -> str:
        """Return a debugging representation of the recorder's state."""
        return (
            f"{type(self).__name__}(path={os.fspath(self._path)!r}, "
            f"sample_rate={self._sample_rate}, "
            f"frames_written={self._frames_written}, gaps={self._gaps}, "
            f"closed={self.closed})"
        )


def _to_int16(samples: np.ndarray) -> np.ndarray:
    """Convert mono float samples in [-1, 1] to 16-bit PCM.

    Args:
        samples: 1-D float array; values outside [-1, 1] are clipped rather
            than allowed to wrap around, which would sound like a click.

    Returns:
        A 1-D ``int16`` array of the same length.
    """
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
