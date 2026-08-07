"""The audio source interface: everything that can feed the ring buffer.

A source is the only place in the pipeline where *raw* audio is seen, so two
responsibilities live here and nowhere else:

* **Downmix.**  ``on_audio`` always receives 1-D mono ``float32``; a multi
  channel device or file is averaged down by the source itself.  Nothing
  downstream ever has to ask how many channels there were.
* **Level metering.**  :attr:`~echochamber.audio.types.StreamStats.peak_level`
  and ``rms_level`` are computed in :meth:`AudioSource._emit`, because after
  this point the audio is already in the ring and the chunker only sees
  windows.

Concrete sources own their own thread (or device callback) and follow the
lifecycle established by :class:`~echochamber.audio.chunker.WindowChunker`:
single-use, ``start()`` raises if called twice, ``stop()`` is idempotent,
bounded by a timeout, and returns whether the worker actually ended.  A worker
that dies from an exception records it in :attr:`AudioSource.error` and sets
:attr:`AudioSource.finished` rather than letting a daemon thread scribble a
traceback on stderr.
"""

from __future__ import annotations

import abc
import threading
from typing import Callable

import numpy as np

from echochamber.audio.types import StreamStats

__all__ = ["AudioCallback", "AudioSource"]

AudioCallback = Callable[[np.ndarray], None]
"""Called with each mono ``float32`` block, on the source's own thread."""


class AudioSource(abc.ABC):
    """Base class for anything producing mono ``float32`` audio blocks.

    Subclasses implement :meth:`start` and :meth:`stop` plus the
    :attr:`sample_rate` / :attr:`channels` properties, and push audio through
    :meth:`_emit` -- never through ``on_audio`` directly, or the stats stop
    updating.

    Threading contract:

    * ``on_audio`` runs on the source's thread (for a live device, a
      real-time-ish callback thread).  It must not block or allocate heavily;
      in this pipeline it is :meth:`RingBuffer.write
      <echochamber.audio.ringbuffer.RingBuffer.write>`, which does neither.
    * :meth:`start` and :meth:`stop` are called from the owning thread.
    * :attr:`finished` may be waited on from any thread.
    """

    __slots__ = ("_on_audio", "_stats", "_thread", "_error", "_finished")

    def __init__(
        self,
        on_audio: AudioCallback,
        stats: StreamStats | None = None,
    ) -> None:
        """Create a source; nothing runs until :meth:`start` is called.

        Args:
            on_audio: Callback receiving each mono ``float32`` block, in order,
                on the source's thread.
            stats: Counter record to update; a fresh :class:`StreamStats` is
                allocated when ``None``.  Pass the pipeline's shared instance
                so the GUI sees the capture counters and levels.
        """
        self._on_audio: AudioCallback = on_audio
        self._stats: StreamStats = StreamStats() if stats is None else stats
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._finished: threading.Event = threading.Event()

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of the audio this source produces, in Hz.

        Available *before* :meth:`start`, so the pipeline can reject a
        mismatch against its configuration without opening a stream.
        """

    @property
    @abc.abstractmethod
    def channels(self) -> int:
        """Channel count of the underlying source, **before** downmix.

        Informational only: ``on_audio`` always receives mono.
        """

    @property
    def stats(self) -> StreamStats:
        """The live stats record this source mutates (not a snapshot)."""
        return self._stats

    @property
    def on_audio(self) -> AudioCallback:
        """The callback this source pushes blocks to."""
        return self._on_audio

    @property
    def is_running(self) -> bool:
        """``True`` while the source's worker thread is alive.

        Sources that are driven by a device callback rather than a thread of
        their own override this.
        """
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def error(self) -> BaseException | None:
        """Exception that terminated the source, or ``None`` if it ended cleanly."""
        return self._error

    @property
    def finished(self) -> threading.Event:
        """Set once the source will produce no more audio.

        Reached at end of a non-looping file, after :meth:`stop`, or when the
        worker died -- in the last case :attr:`error` is also set.  This is the
        event :meth:`echochamber.audio.pipeline.AudioPipeline.wait_until_finished`
        waits on.
        """
        return self._finished

    @abc.abstractmethod
    def start(self) -> None:
        """Begin producing audio.

        Raises:
            RuntimeError: If the source was already started.  Sources are
                single-use; restarting would resume mid-stream against a ring
                whose frame counter has moved on.
        """

    @abc.abstractmethod
    def stop(self, timeout: float | None = 2.0) -> bool:
        """Stop producing audio and wait for the worker to finish.

        Idempotent, and a no-op returning ``True`` for a source that was never
        started.

        Args:
            timeout: Seconds to wait for the worker to end, or ``None`` to wait
                indefinitely.

        Returns:
            ``True`` if the source has ended (or never ran), ``False`` if it
            was still running when ``timeout`` expired.
        """

    def _emit(self, block: np.ndarray) -> None:
        """Update the stats from ``block``, then hand it to ``on_audio``.

        Every subclass pushes audio through here rather than calling
        ``on_audio`` itself, so the level meters and ``frames_captured`` are
        maintained in exactly one place.

        Args:
            block: 1-D mono ``float32`` block.  An empty block updates
                ``frames_captured`` by zero and reports zero levels rather
                than producing NaN from an empty mean.
        """
        stats = self._stats
        n = len(block)
        stats.frames_captured += n
        if n == 0:
            stats.peak_level = 0.0
            stats.rms_level = 0.0
        else:
            stats.peak_level = float(np.max(np.abs(block)))
            stats.rms_level = float(np.sqrt(np.mean(block**2)))
        self._on_audio(block)

    def __repr__(self) -> str:
        """Return a debugging representation of the source's state."""
        return (
            f"{type(self).__name__}(sample_rate={self.sample_rate}, "
            f"channels={self.channels}, running={self.is_running}, "
            f"finished={self._finished.is_set()})"
        )
