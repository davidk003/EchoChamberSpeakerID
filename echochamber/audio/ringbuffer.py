"""Single-producer / single-consumer ring buffer, lock-free on the hot path.

The producer is a PortAudio callback running on a real-time-ish thread: it must
never allocate, never take a lock, and never block.  :meth:`RingBuffer.write`
therefore does nothing but two numpy slice assignments, one integer store, and
an uncontended :meth:`threading.Event.set`.

The only mutable state shared across threads is the ``write_frames`` integer,
whose assignment is atomic under the GIL.  The consumer never writes it.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["OverrunError", "RingBuffer"]


class OverrunError(RuntimeError):
    """Raised when the reader asks for frames the writer has already lapped.

    The requested ``start_frame`` is older than
    :attr:`RingBuffer.oldest_frame`, so the samples no longer exist anywhere.
    The consumer should count the overrun, resynchronize its read cursor to
    :attr:`RingBuffer.oldest_frame`, and mark the next chunk ``discontinuous``.
    """


class RingBuffer:
    """Preallocated SPSC ring buffer whose reads are always contiguous.

    **Doubled-write layout.** The backing store is ``2 * capacity`` frames long
    and every block is written twice, at logical position ``p`` and at
    ``p + capacity``.  Consequently any read of ``n <= capacity`` frames
    starting at logical position ``q = start_frame % capacity`` is the plain
    contiguous slice ``buf[q : q + n]`` -- no wraparound branch and no
    stitching, at the cost of one extra memcpy of a ~10 ms block per callback.

    Threading contract:

    * :meth:`write` is called by the producer thread only.
    * :meth:`read`, :meth:`available_from` and :meth:`wait_for` are called by
      the consumer thread only.
    * :meth:`close` may be called from any thread.
    """

    __slots__ = ("_capacity", "_dtype", "_buf", "_write_frames", "_event", "_closed")

    def __init__(self, capacity_frames: int, dtype: npt.DTypeLike = np.float32) -> None:
        """Allocate a ring holding ``capacity_frames`` frames.

        Args:
            capacity_frames: Number of frames retained before the writer laps
                the reader.  Must be positive.
            dtype: Sample dtype of the backing store; ``float32`` by default.

        Raises:
            ValueError: If ``capacity_frames`` is not positive.
        """
        capacity = int(capacity_frames)
        if capacity <= 0:
            raise ValueError(f"capacity_frames must be > 0, got {capacity_frames!r}")

        self._capacity: int = capacity
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        # 2 * capacity so that every read is a contiguous slice (see class doc).
        self._buf: np.ndarray = np.zeros(2 * capacity, dtype=self._dtype)
        self._write_frames: int = 0
        self._event: threading.Event = threading.Event()
        self._closed: bool = False

    @property
    def capacity(self) -> int:
        """Number of frames the ring retains before overwriting them."""
        return self._capacity

    @property
    def dtype(self) -> np.dtype[Any]:
        """Sample dtype of the backing store."""
        return self._dtype

    @property
    def write_frames(self) -> int:
        """Total frames ever written since construction (monotonic)."""
        return self._write_frames

    @property
    def oldest_frame(self) -> int:
        """Oldest absolute frame index still resident in the ring."""
        return max(0, self._write_frames - self._capacity)

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def write(self, data: np.ndarray) -> None:
        """Append ``data`` to the ring. Producer thread only.

        Hot path: performs only slice assignments (plus a dtype cast if
        ``data`` does not already match the buffer dtype), one integer store,
        and one uncontended ``Event.set()``.  ``write_frames`` is bumped
        **after** the samples are in place, so a consumer that observes the new
        cursor is guaranteed to see the data behind it.

        Args:
            data: 1-D array of at most :attr:`capacity` frames.

        Raises:
            ValueError: If ``data`` is not 1-D, or is longer than
                :attr:`capacity`.
        """
        if data.ndim != 1:
            raise ValueError(f"data must be 1-D, got ndim={data.ndim}")

        m = data.shape[0]
        capacity = self._capacity
        if m > capacity:
            raise ValueError(
                f"write of {m} frames exceeds ring capacity {capacity}"
            )
        if m == 0:
            return

        buf = self._buf
        p = self._write_frames % capacity
        first = min(m, capacity - p)

        # Primary copy plus its mirror image `capacity` frames later.
        buf[p : p + first] = data[:first]
        buf[p + capacity : p + capacity + first] = data[:first]

        rest = m - first
        if rest:
            tail = data[first:]
            buf[0:rest] = tail
            buf[capacity : capacity + rest] = tail

        # Publish last: the data is fully in place before the cursor moves.
        self._write_frames += m
        self._event.set()

    def read(self, start_frame: int, n: int) -> np.ndarray:
        """Return ``n`` frames starting at absolute ``start_frame``.

        .. warning::
           **The returned array is a VIEW into the ring's backing store, not a
           copy.** The writer will eventually lap the ring and overwrite the
           bytes underneath it, silently corrupting whatever you are still
           holding. Callers MUST copy (``arr.copy()`` or
           ``np.array(arr, copy=True)``) before doing anything that outlives
           the next ``capacity`` frames of capture -- in particular before
           putting the data on a queue, storing it in an
           :class:`~echochamber.audio.types.AudioChunk`, or handing it to
           another thread. Only transient, immediately-consumed arithmetic may
           use the view directly.

        Consumer thread only.

        Args:
            start_frame: Absolute index of the first frame requested.
            n: Number of frames, at most :attr:`capacity`.

        Returns:
            A contiguous view of length ``n`` into the backing store.

        Raises:
            ValueError: If ``n`` is negative, exceeds :attr:`capacity`,
                ``start_frame`` is negative, or the requested range has not
                been captured yet.
            OverrunError: If ``start_frame`` is older than
                :attr:`oldest_frame`, i.e. the writer already overwrote it.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n > self._capacity:
            raise ValueError(
                f"n={n} exceeds ring capacity {self._capacity}"
            )
        if start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {start_frame}")

        write_frames = self._write_frames
        if start_frame + n > write_frames:
            raise ValueError(
                f"requested frames [{start_frame}, {start_frame + n}) not yet "
                f"captured; write_frames={write_frames}"
            )
        oldest = max(0, write_frames - self._capacity)
        if start_frame < oldest:
            raise OverrunError(
                f"start_frame={start_frame} is older than oldest_frame={oldest}; "
                f"the writer already overwrote it"
            )

        q = start_frame % self._capacity
        return self._buf[q : q + n]

    def available_from(self, start_frame: int) -> int:
        """Return how many frames are readable from ``start_frame`` onward.

        This is simply ``write_frames - start_frame``; it may be negative if
        ``start_frame`` is ahead of the write cursor, and it does not account
        for frames already lapped by the writer.
        """
        return self._write_frames - start_frame

    def wait_for(self, end_frame: int, timeout: float | None = None) -> bool:
        """Block until at least ``end_frame`` frames have been written.

        Uses a :class:`threading.Event` rather than a
        :class:`threading.Condition`: the producer is an audio callback and
        must never contend on a lock the reader holds.  The clear/re-check
        ordering below is what makes this safe against a missed wakeup -- a
        write landing between the check and the ``clear()`` is caught by the
        second check.

        Args:
            end_frame: Absolute frame count to wait for.
            timeout: Maximum seconds to wait per event wait, or ``None`` to
                wait indefinitely.

        Returns:
            ``True`` once ``write_frames >= end_frame``; ``False`` on timeout
            or if the ring was closed first.
        """
        event = self._event
        while True:
            if self._write_frames >= end_frame:
                return True
            if self._closed:
                return False
            event.clear()
            if self._write_frames >= end_frame:  # re-check AFTER clear
                return True
            if not event.wait(timeout):
                return False

    def close(self) -> None:
        """Mark the ring closed and wake every waiter.

        Waiters in :meth:`wait_for` return ``False`` unless their frame target
        is already satisfied.  Idempotent; safe from any thread.
        """
        self._closed = True
        self._event.set()

    def __repr__(self) -> str:
        """Return a debugging representation of the ring's state."""
        return (
            f"{type(self).__name__}(capacity={self._capacity}, "
            f"dtype={self._dtype.name}, write_frames={self._write_frames}, "
            f"closed={self._closed})"
        )
