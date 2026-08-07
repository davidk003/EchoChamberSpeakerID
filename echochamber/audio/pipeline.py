"""The assembled pipeline: source -> ring -> chunker -> queue -> sink.

``AudioPipeline`` is just wiring, but the wiring is where the concurrency bugs
live, so two things are worth stating plainly.

**Why there is a consumer thread at all.**  The chunker calls its callback
inline.  If that callback were the real sink -- eventually an ML model -- the
chunker would stop reading the ring for as long as inference takes, the ring
would overrun, and the pipeline would lose audio while looking perfectly
healthy.  So the chunker's callback is a bounded :class:`QueueSink` that never
waits (under ``DROP_OLDEST``), and a separate ``chunk-consumer`` thread drains
the queue into the real sink.  A slow sink then costs *dropped chunks*, which
are counted and visible, instead of *lost audio*, which is not.

**Why the shutdown order is what it is.**  Each stage is stopped only after
the stage feeding it, so nothing is still producing into something already
closed and nothing buffered is thrown away:

1. ``source.stop()`` -- no more writes into the ring.
2. ``ring.close()`` -- wakes the chunker out of ``wait_for`` immediately
   instead of leaving it to time out.
3. the chunker drains: with the ring closed it emits every remaining *full*
   window and then exits by itself; ``chunker.stop()`` bounds that.
4. ``queue_sink.close()`` -- the consumer keeps draining and only then sees
   the ``None`` end-of-stream marker.
5. join the consumer, so the sink has seen every chunk it will ever see.
6. ``sink.close()`` -- last, because until now chunks were still arriving.

Stopping in any other order either drops buffered chunks (closing the queue
before the chunker has drained) or hangs (joining the consumer before the
queue is closed).  Every step is bounded by the remaining timeout budget, so
a wedged sink or a dead source makes :meth:`AudioPipeline.stop` return
``False`` rather than block forever.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from echochamber.audio.chunker import WindowChunker
from echochamber.audio.ringbuffer import RingBuffer
from echochamber.audio.sinks import ChunkSink, QueueSink
from echochamber.audio.sources.base import AudioSource
from echochamber.audio.types import StreamStats

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    # `echochamber.config` imports `echochamber.audio.types`, so importing it
    # at runtime from inside the `echochamber.audio` package would make
    # `import echochamber.config` fail with a partially initialized module.
    # `from __future__ import annotations` keeps the annotation a string.
    from echochamber.config import AudioConfig

__all__ = ["DRAIN_POLL_S", "DRAIN_SETTLE_S", "AudioPipeline", "SourceFactory"]

SourceFactory = Callable[[Callable[[np.ndarray], None], StreamStats], AudioSource]
"""Builds a source around the audio callback and stats the pipeline supplies.

The pipeline passes :meth:`RingBuffer.write
<echochamber.audio.ringbuffer.RingBuffer.write>`, which is why a source never
needs to know that a ring buffer exists.

It also passes its :class:`StreamStats`, and the factory is expected to hand
that object to the source it builds.  The source is the only component that
ever sees raw audio, so it is the only one that can fill in
``frames_captured``, ``peak_level`` and ``rms_level``.  Injecting the stats
here rather than leaving the caller to thread the same instance into two
places removes a silent failure: a pipeline whose meters simply never move.
"""

DRAIN_POLL_S: float = 0.005
"""How often :meth:`AudioPipeline.stop` and ``wait_until_finished`` re-check."""

DRAIN_SETTLE_S: float = 0.05
"""How long the chunker must emit nothing before it counts as drained.

The chunker exposes no "am I idle?" flag, and it cannot emit again once the
source has stopped writing, so quiescence is inferred: queue empty, consumer
idle, and ``chunks_emitted`` unchanged for this long.  The chunker emits
back-to-back whenever data is available, so the only thing this interval has
to cover is scheduler jitter.  Correctness never rests on it --
:meth:`AudioPipeline.stop` drains the chunker deterministically by closing the
ring -- it only makes ``wait_until_finished`` honest on its own.
"""


class AudioPipeline:
    """Owns and sequences every moving part of the audio path.

    Construction wires the graph and allocates the ring; nothing runs until
    :meth:`start`.  The pipeline is single-use: once stopped it cannot be
    restarted, because the chunker and the source are single-use themselves.

    Threading: three threads and one queue.  The source's (device callback or
    replay thread) writes the ring, the chunker thread cuts windows into the
    queue, the ``chunk-consumer`` thread feeds ``sink``.  ``sink.on_chunk``
    therefore always runs on the consumer thread and may take as long as it
    likes without endangering capture.
    """

    __slots__ = (
        "_config",
        "_stats",
        "_ring",
        "_queue_sink",
        "_chunker",
        "_source",
        "_sink",
        "_consumer",
        "_consumer_error",
        "_delivered",
        "_base_emitted",
        "_base_dropped",
        "_started",
        "_stop_complete",
        "_stop_lock",
    )

    def __init__(
        self,
        config: "AudioConfig",
        sink: ChunkSink,
        source_factory: SourceFactory,
        stats: StreamStats | None = None,
    ) -> None:
        """Wire source -> ring -> chunker -> queue -> ``sink``.

        Args:
            config: Capture and windowing configuration.  ``ring_frames`` sizes
                the ring, ``queue_max`` / ``drop_policy`` the handoff queue.
            sink: Terminal sink, fed from the consumer thread.  May be slow.
            source_factory: Called once with the ring's ``write`` method and
                this pipeline's :class:`StreamStats`; must return an unstarted
                :class:`AudioSource`.  Pass the stats through to the source, or
                its ``frames_captured`` and level meters will go nowhere.
            stats: Shared counter record; a fresh :class:`StreamStats` is
                allocated when ``None``.  Every stage updates this one object,
                including the source via ``source_factory``, so a caller who
                supplies nothing still gets working meters.

        Raises:
            ValueError: If ``config.window_frames`` exceeds the ring capacity
                derived from ``config.ring_frames`` (``AudioConfig`` normally
                rules this out already).
        """
        self._config: "AudioConfig" = config
        self._stats: StreamStats = StreamStats() if stats is None else stats
        self._sink: ChunkSink = sink

        self._ring: RingBuffer = RingBuffer(config.ring_frames)
        self._queue_sink: QueueSink = QueueSink(
            config.queue_max, config.drop_policy, self._stats
        )
        self._chunker: WindowChunker = WindowChunker(
            self._ring, config, self._queue_sink.on_chunk, self._stats
        )
        self._source: AudioSource = source_factory(self._ring.write, self._stats)

        self._consumer: threading.Thread | None = None
        self._consumer_error: BaseException | None = None
        # Written only by the consumer thread; read by wait_until_finished.
        self._delivered: int = 0
        # Baselines, so a caller-supplied StreamStats with existing counts does
        # not make the in-flight arithmetic below nonsense.
        self._base_emitted: int = 0
        self._base_dropped: int = 0
        self._started: bool = False
        self._stop_complete: bool = False
        self._stop_lock: threading.RLock = threading.RLock()

    @property
    def config(self) -> "AudioConfig":
        """The configuration currently in force."""
        return self._config

    @property
    def stats(self) -> StreamStats:
        """The live stats record every stage mutates (not a snapshot)."""
        return self._stats

    @property
    def ring(self) -> RingBuffer:
        """The ring buffer between the source and the chunker."""
        return self._ring

    @property
    def queue_sink(self) -> QueueSink:
        """The bounded queue between the chunker and the consumer thread."""
        return self._queue_sink

    @property
    def chunker(self) -> WindowChunker:
        """The windowing thread."""
        return self._chunker

    @property
    def source(self) -> AudioSource:
        """The audio source built by the factory."""
        return self._source

    @property
    def sink(self) -> ChunkSink:
        """The terminal sink, fed from the consumer thread."""
        return self._sink

    @property
    def is_running(self) -> bool:
        """``True`` between :meth:`start` and a completed :meth:`stop`.

        A finished file source does *not* make this ``False``: the consumer is
        still alive and the pipeline is still willing to deliver chunks.  Use
        ``pipeline.source.finished`` to ask about the source specifically.
        """
        consumer = self._consumer
        return consumer is not None and consumer.is_alive()

    @property
    def error(self) -> BaseException | None:
        """First failure from the source, the chunker, or the consumer thread.

        Background threads never re-raise: each records its exception and exits,
        so this is the only way to learn that the pipeline died.  Check it after
        :meth:`stop` -- and after :meth:`wait_until_finished` returns ``False``.
        """
        for exc in (self._source.error, self._chunker.error, self._consumer_error):
            if exc is not None:
                return exc
        return None

    @property
    def consumer_error(self) -> BaseException | None:
        """Exception raised by ``sink.on_chunk`` on the consumer thread."""
        return self._consumer_error

    def start(self) -> None:
        """Start the consumer, the chunker and the source, in that order.

        Consumer first, then chunker, then source: each stage is ready before
        anything can feed it, so nothing is dropped at startup.

        Raises:
            RuntimeError: If the pipeline was already started.
            ValueError: If ``source.sample_rate`` disagrees with
                ``config.sample_rate``.  Resampling is deliberately out of
                scope (the capture backend delivers the target rate), so a
                mismatched WAV is a caller error and must be loud rather than
                silently played at the wrong speed.
        """
        if self._started:
            raise RuntimeError(
                "AudioPipeline has already been started and cannot be "
                "restarted; create a new AudioPipeline"
            )
        source_rate = self._source.sample_rate
        if source_rate != self._config.sample_rate:
            raise ValueError(
                f"source sample rate {source_rate} Hz does not match config "
                f"sample rate {self._config.sample_rate} Hz; this pipeline does "
                f"not resample"
            )

        self._started = True
        self._base_emitted = self._stats.chunks_emitted
        self._base_dropped = self._stats.chunks_dropped

        consumer = threading.Thread(
            target=self._consume, name="chunk-consumer", daemon=True
        )
        self._consumer = consumer
        consumer.start()
        self._chunker.start()
        self._source.start()

    def stop(self, timeout: float | None = 5.0) -> bool:
        """Shut every stage down in order, bounded by ``timeout`` overall.

        See the module docstring for why the order is load-bearing.  Safe to
        call without :meth:`start`, safe to call twice, and safe to call while
        the sink is slow or the source has already died -- every wait is
        bounded, so this cannot deadlock.

        Args:
            timeout: Total seconds allowed for the whole shutdown, shared
                across the stages, or ``None`` to wait as long as it takes.

        Returns:
            ``True`` if every stage finished within the budget, ``False``
            otherwise (a ``False`` leaves daemon threads running; they die with
            the process).
        """
        consumer = self._consumer
        if consumer is not None and consumer is threading.current_thread():
            # Re-entrant call from inside sink.on_chunk.  This thread can
            # neither join itself nor wait on the lock -- whoever is running
            # the real stop() is very likely joining *us* while holding it, so
            # blocking here would stall shutdown for the whole timeout.
            # Signal every stage without joining anything and report that
            # shutdown did not complete on this call.
            self._source.stop(0.0)
            self._ring.close()
            self._chunker.stop(0.0)
            self._queue_sink.close()
            return False

        with self._stop_lock:
            if self._stop_complete:
                return True

            deadline = None if timeout is None else time.monotonic() + timeout
            ok = True

            # 1. No more writes into the ring.
            ok &= self._source.stop(_remaining(deadline))

            # 2. Wake the chunker out of `wait_for` instead of making it wait
            #    out its poll interval.
            self._ring.close()

            # 3. With the ring closed the chunker emits every remaining full
            #    window and then exits on its own; `stop()` only sets a flag it
            #    checks at the *top* of the loop, so give it the chance to
            #    finish first or the tail windows would be lost.
            self._wait_while(lambda: self._chunker.is_running, deadline)
            ok &= self._chunker.stop(_remaining(deadline))

            # 4. The consumer drains what is still queued, then gets `None`.
            self._queue_sink.close()

            # 5. Join the consumer, so the sink has seen everything.
            consumer = self._consumer
            if consumer is not None:
                if consumer is threading.current_thread():
                    # Called from within sink.on_chunk: joining ourselves would
                    # deadlock.  The queue is closed, so the loop ends as soon
                    # as this callback returns.
                    ok = False
                else:
                    consumer.join(_remaining(deadline))
                    ok &= not consumer.is_alive()

            # 6. Only now can the sink be closed: until the join it was still
            #    receiving chunks.
            self._sink.close()

            self._stop_complete = ok
            return ok

    def wait_until_finished(self, timeout: float | None = None) -> bool:
        """Wait for the source to end and the pipeline to go quiet.

        This is what makes a file replay assertable::

            pipeline.start()
            pipeline.wait_until_finished()
            pipeline.stop()          # every full window has reached the sink

        Waits for ``source.finished``, then for the chunker to stop emitting
        and the queue to drain into an idle consumer.

        Args:
            timeout: Total seconds to wait, or ``None`` to wait indefinitely.

        Returns:
            ``True`` if the pipeline went quiet, ``False`` on timeout.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        remaining = _remaining(deadline)
        if not self._source.finished.wait(remaining):
            return False

        # The source is done, but the chunker may still be cutting windows out
        # of audio already in the ring, and the consumer may still be inside
        # sink.on_chunk.  Quiescence: nothing in flight, and nothing newly
        # emitted for DRAIN_SETTLE_S (see that constant for why this is a
        # settle time rather than a flag).
        last_emitted = -1
        stable_since = time.monotonic()
        while True:
            emitted = self._stats.chunks_emitted
            in_flight = (
                (emitted - self._base_emitted)
                - (self._stats.chunks_dropped - self._base_dropped)
                - self._delivered
            )
            now = time.monotonic()
            if emitted != last_emitted or in_flight > 0:
                last_emitted = emitted
                stable_since = now
            elif now - stable_since >= DRAIN_SETTLE_S:
                return True

            if deadline is not None and now >= deadline:
                return False
            time.sleep(DRAIN_POLL_S)

    def reconfigure(self, config: "AudioConfig") -> None:
        """Swap in new window geometry without restarting anything.

        Delegates to :meth:`WindowChunker.reconfigure`; the next chunk starts
        at the write head as of this call and is marked ``discontinuous``.

        The ring is **not** resized -- its capacity is fixed at construction --
        so a window larger than the existing ring is rejected and the running
        stream is left untouched.

        Args:
            config: New configuration.

        Raises:
            ValueError: If ``config.window_frames`` exceeds ``ring.capacity``.
        """
        self._chunker.reconfigure(config)
        self._config = config

    def _consume(self) -> None:
        """Consumer thread body: queue -> ``sink`` until end of stream."""
        queue_sink = self._queue_sink
        sink = self._sink
        try:
            while True:
                chunk = queue_sink.get()
                if chunk is None:
                    return
                try:
                    sink.on_chunk(chunk)
                finally:
                    # Counted even when on_chunk raised: the chunk is no longer
                    # in flight, and wait_until_finished must not hang on a
                    # sink that failed.
                    self._delivered += 1
        except BaseException as exc:  # noqa: BLE001 - reported via `error`
            # A daemon thread re-raising would only scribble on stderr; record
            # it so `error` can report it after `stop()`.
            self._consumer_error = exc

    def _wait_while(
        self, predicate: Callable[[], bool], deadline: float | None
    ) -> bool:
        """Poll until ``predicate`` is false or ``deadline`` passes.

        Args:
            predicate: Condition to wait out.
            deadline: :func:`time.monotonic` deadline, or ``None`` for no limit.

        Returns:
            ``True`` if the predicate became false in time, ``False`` on timeout.
        """
        while predicate():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(DRAIN_POLL_S)
        return True

    def __repr__(self) -> str:
        """Return a debugging representation of the pipeline's state."""
        return (
            f"{type(self).__name__}(running={self.is_running}, "
            f"window_ms={self._config.window_ms}, hop_ms={self._config.hop_ms}, "
            f"chunks_emitted={self._stats.chunks_emitted}, "
            f"chunks_dropped={self._stats.chunks_dropped})"
        )


def _remaining(deadline: float | None) -> float | None:
    """Return the seconds left before ``deadline``, never negative.

    Args:
        deadline: :func:`time.monotonic` deadline, or ``None`` for no limit.

    Returns:
        Seconds remaining, ``0.0`` if the deadline has passed, or ``None`` if
        there was no deadline.
    """
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
