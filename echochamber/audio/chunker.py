"""Windowing thread: turns the ring buffer's sample stream into chunks.

This is the piece that decides what "a chunk" means.  It reads the ring at
absolute frame positions derived from a counter -- never from a wall clock --
so chunk ``k`` covers exactly frames ``[k*H, k*H + W)`` and consecutive chunks
share exactly ``W - H`` *identical* samples no matter how badly the scheduler
jitters.  Sample-exact and drift-free is the whole point; a timer-driven
chunker would slowly slide off the audio.

Two rules earn their own emphasis because violating either produces bugs that
look like bad audio rather than bad code:

* **Chunks own their samples.**  :meth:`RingBuffer.read` hands back a view into
  the ring; the writer laps the ring and will overwrite it underneath a
  retained reference.  Every emitted chunk therefore carries ``view.copy()``.
* **Partial windows are never emitted.**  A short tail at end-of-stream is
  dropped rather than padded, so downstream can assume ``len(samples) == W``
  for the config that produced the chunk.

Live reconfiguration is lock-free: the GUI swaps a whole frozen
:class:`~echochamber.config.AudioConfig` into one attribute (atomic under the
GIL) and the loop notices by *identity* at the top of its next pass, realigns
and flags the next chunk discontinuous.  The loop reads that attribute into a
local exactly once per iteration -- reading it twice would let a swap land in
between and mix geometry from two configs into one chunk.

The realignment frame is captured by :meth:`WindowChunker.reconfigure` itself,
*not* re-read by the loop when it notices.  This matters more than it looks:
the loop may not run for up to :data:`POLL_INTERVAL_S`, and on a live stream the
write head moves the whole time.  Sampling the head inside the loop would put
the new window grid at a position nobody can predict -- it would depend purely
on thread scheduling, making the behaviour both surprising to callers and
impossible to test deterministically.  Capturing it at the call means the new
grid starts where the caller asked: at the audio that was live when they
changed the setting.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Callable

from echochamber.audio.ringbuffer import OverrunError, RingBuffer
from echochamber.audio.types import AudioChunk, StreamStats

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, see note below
    # `echochamber.config` imports `echochamber.audio.types`, so importing it
    # at runtime from inside the `echochamber.audio` package would make
    # `import echochamber.config` fail with a partially initialized module.
    # The annotation is only ever needed by type checkers, and
    # `from __future__ import annotations` keeps it a string at runtime.
    from echochamber.config import AudioConfig

__all__ = ["POLL_INTERVAL_S", "ChunkCallback", "WindowChunker"]

ChunkCallback = Callable[[AudioChunk], None]
"""Called on the chunker thread with each completed window, in order."""

POLL_INTERVAL_S: float = 0.1
"""Maximum time the loop blocks in ``wait_for`` before re-checking its flags.

The loop must never wait indefinitely: :meth:`WindowChunker.stop` and
:meth:`WindowChunker.reconfigure` are plain attribute stores that cannot
interrupt a blocked waiter, so responsiveness comes entirely from this timeout
bounding how long a silent stream can pin the thread.
"""


def _check_window_fits(config: "AudioConfig", ring: RingBuffer) -> None:
    """Reject a config whose window cannot physically be read from ``ring``.

    ``AudioConfig`` validates its own ``ring_frames``, but that is a *derived*
    figure -- nothing forces the caller to size the actual :class:`RingBuffer`
    to match it, and the two are supplied independently.
    """
    if config.window_frames > ring.capacity:
        raise ValueError(
            f"window of {config.window_frames} frames "
            f"({config.window_ms} ms @ {config.sample_rate} Hz) exceeds ring "
            f"capacity {ring.capacity}; the window can never be read. Size the "
            f"ring from config.ring_frames ({config.ring_frames})"
        )


class WindowChunker:
    """Background thread cutting overlapping windows out of a ring buffer.

    Owns one daemon thread that repeatedly waits for ``window_frames`` of audio
    to become available at its current read position, copies them out, and
    hands them to ``on_chunk`` as an :class:`AudioChunk`.

    Threading contract:

    * ``on_chunk`` runs on the chunker thread.  It must not block for long --
      the loop is single-threaded, so a slow sink delays every later chunk and
      eventually causes a ring overrun.
    * :meth:`reconfigure`, :meth:`start` and :meth:`stop` are called from the
      owning (typically GUI) thread.
    * If ``on_chunk`` raises, the thread records the exception in :attr:`error`
      and exits; the exception is never re-raised on the background thread and
      never surfaces from :meth:`stop`.
    """

    __slots__ = (
        "_ring",
        "_state",
        "_on_chunk",
        "_stats",
        "_name",
        "_thread",
        "_stop_event",
        "_error",
    )

    def __init__(
        self,
        ring: RingBuffer,
        config: "AudioConfig",
        on_chunk: ChunkCallback,
        stats: StreamStats | None = None,
        name: str = "chunker",
    ) -> None:
        """Create a chunker; no thread runs until :meth:`start` is called.

        Args:
            ring: Ring buffer to read from.  This chunker is its sole consumer.
            config: Window geometry and sample rate in force at start.
            on_chunk: Callback invoked with each completed window, in sequence
                order, on the chunker thread.
            stats: Counter record to update; a fresh :class:`StreamStats` is
                allocated when ``None``.  Pass the pipeline's shared instance
                so the GUI sees this chunker's counters.
            name: Thread name, useful in debuggers and stack dumps.

        Raises:
            ValueError: If ``config.window_frames`` exceeds ``ring.capacity``.
                Nothing otherwise ties an :class:`AudioConfig` to the particular
                ring it is used with, and the mismatch is nasty in the wild: the
                first read raises deep on the chunker thread, so the stream
                simply never produces a chunk.  Better to reject the pairing
                here, where the traceback points at the caller.
        """
        _check_window_fits(config, ring)
        self._ring: RingBuffer = ring
        # One attribute holding (frozen config, realignment frame): swapping the
        # whole tuple is atomic under the GIL, which is the entire
        # reconfiguration mechanism. The two travel together so the loop can
        # never pair a new config with a stale realignment point.
        self._state: tuple["AudioConfig", int] = (config, 0)
        self._on_chunk: ChunkCallback = on_chunk
        self._stats: StreamStats = StreamStats() if stats is None else stats
        self._name: str = name
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._error: BaseException | None = None

    @property
    def config(self) -> "AudioConfig":
        """The configuration currently in force (the next chunk may still use it)."""
        return self._state[0]

    @property
    def stats(self) -> StreamStats:
        """The live stats record this chunker mutates (not a snapshot)."""
        return self._stats

    @property
    def error(self) -> BaseException | None:
        """Exception that terminated the thread, or ``None`` if it ended cleanly."""
        return self._error

    @property
    def is_running(self) -> bool:
        """``True`` while the chunker thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def reconfigure(self, config: "AudioConfig") -> None:
        """Swap in new window geometry, effective from the next iteration.

        Detection is by *identity*, so handing back an equal-but-distinct
        config still realigns the stream -- that is deliberate: it gives the
        caller an explicit "resynchronize now" lever.  The next emitted chunk
        starts at the write head **as of this call** and is marked
        ``discontinuous``, because its first sample does not follow the
        previous chunk's audio.

        Pinning the realignment frame here rather than in the loop is what makes
        the new grid land somewhere the caller can predict; see the module
        docstring.  Any audio still buffered between the old read position and
        this point is deliberately skipped -- reconfiguring means "start fresh
        from now", not "catch up first".

        Safe to call from any thread, before or during :meth:`start`; no lock
        is taken because a single attribute store is atomic under the GIL.

        Args:
            config: New configuration to take effect on the next pass.

        Raises:
            ValueError: If the new ``window_frames`` exceeds the ring capacity.
                Checked before the swap, so a rejected reconfiguration leaves
                the running stream untouched rather than poisoning it.
        """
        _check_window_fits(config, self._ring)
        self._state = (config, self._ring.write_frames)

    def start(self) -> None:
        """Launch the chunker thread.

        Raises:
            RuntimeError: If the chunker was already started.  A chunker is
                single-use -- restarting would silently resume mid-stream with
                a stale read cursor, so a new instance is required instead.
        """
        if self._thread is not None:
            raise RuntimeError(
                f"chunker {self._name!r} has already been started and cannot "
                f"be restarted; create a new WindowChunker"
            )
        thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread = thread
        thread.start()

    def stop(self, timeout: float | None = 2.0) -> bool:
        """Signal the thread to stop and wait for it to finish.

        The loop only ever blocks for at most :data:`POLL_INTERVAL_S`, so it
        observes the stop flag within roughly 100 ms even on a silent stream.
        Idempotent, and a no-op on a chunker that was never started.

        Args:
            timeout: Seconds to wait for the thread to exit, or ``None`` to
                wait indefinitely.

        Returns:
            ``True`` if the thread has ended (or never ran), ``False`` if it
            was still alive when ``timeout`` expired.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            # Called from on_chunk: joining ourselves would deadlock.  The flag
            # is set, so the loop exits as soon as this callback returns.
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        """Thread body: emit windows until stopped, closed, or a callback raises."""
        ring = self._ring
        on_chunk = self._on_chunk
        stats = self._stats
        stop_event = self._stop_event

        next_start = 0
        seq = 0
        # The very first chunk is continuous by definition: nothing precedes it.
        pending_discontinuity = False
        last_cfg = self._state[0]

        try:
            while not stop_event.is_set():
                # Exactly one read of the state per pass; everything below uses
                # these locals so a mid-iteration swap cannot split a chunk
                # across two geometries.
                cfg, realign_from = self._state
                if cfg is not last_cfg:
                    # Jump to the head captured by reconfigure(), not the head
                    # as of right now: the old read position meant something
                    # under the old geometry, and re-sampling the head here
                    # would land the new grid at a scheduling-dependent spot.
                    next_start = realign_from
                    pending_discontinuity = True
                    last_cfg = cfg

                window_frames = cfg.window_frames
                hop_frames = cfg.hop_frames

                if not ring.wait_for(
                    next_start + window_frames, timeout=POLL_INTERVAL_S
                ):
                    if ring.closed:
                        # Closed and the target is unreachable: the remaining
                        # tail is shorter than a window, and partial windows are
                        # never emitted.
                        break
                    # Timeout: loop back to re-check the stop flag and config.
                    continue

                try:
                    samples = ring.read(next_start, window_frames).copy()
                except OverrunError:
                    # The writer lapped us; those frames are gone for good.
                    stats.overruns += 1
                    next_start = ring.oldest_frame
                    pending_discontinuity = True
                    continue

                chunk = AudioChunk(
                    samples,
                    next_start,
                    seq,
                    cfg.sample_rate,
                    pending_discontinuity,
                    # Stamped here, not in the sink: this is the moment the
                    # window became complete, so everything a consumer adds
                    # afterwards -- queueing, scheduling, inference -- shows up
                    # in its measured latency rather than being invisible.
                    # perf_counter, not monotonic: the latter is GetTickCount64
                    # on Windows (15.6 ms), too coarse to see this handoff.
                    time.perf_counter(),
                )
                pending_discontinuity = False
                seq += 1
                stats.chunks_emitted += 1
                on_chunk(chunk)
                next_start += hop_frames
        except BaseException as exc:  # noqa: BLE001 - reported via `error`
            # Nothing above us can catch this: re-raising would only print to
            # stderr from a daemon thread.  Record it and exit cleanly so the
            # owner can inspect `error` after `stop()`.
            self._error = exc

    def __repr__(self) -> str:
        """Return a debugging representation of the chunker's state."""
        cfg = self._state[0]
        return (
            f"{type(self).__name__}(name={self._name!r}, "
            f"window_frames={cfg.window_frames}, hop_frames={cfg.hop_frames}, "
            f"running={self.is_running}, chunks_emitted={self._stats.chunks_emitted})"
        )
