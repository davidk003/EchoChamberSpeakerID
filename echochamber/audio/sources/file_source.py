"""WAV replay source -- the pipeline's deterministic stand-in for a microphone.

Feeding a known signal (a sample-index ramp, say) through the real ring buffer
and the real chunker is what makes the windowing testable to the sample: chunk
``k`` must start with value ``k*H`` and overlap its predecessor by exactly
``W - H`` identical samples.  It also lets the whole GUI be exercised on a
machine with no audio hardware.

Two replay modes:

* ``realtime=True`` paces to the audio clock, which is what you want when
  demoing or measuring latency.  Pacing is computed against an **absolute**
  schedule (``start_t + frames_emitted / sample_rate``); sleeping
  ``blocksize / sample_rate`` per block instead would accumulate every
  scheduler overshoot and drift audibly within seconds.
* ``realtime=False`` replays as fast as the machine allows, which is what
  tests want.  Note this can outrun the chunker and overrun the ring on a
  long file -- that is a legitimate thing to test, not a bug, and
  :attr:`~echochamber.audio.types.StreamStats.overruns` will say so.  Pair a
  fast replay with :attr:`~echochamber.audio.types.DropPolicy.BLOCK` if you
  need every window.
"""

from __future__ import annotations

import os
import threading
import time
import wave

import numpy as np

from echochamber.audio.sources.base import AudioCallback, AudioSource
from echochamber.audio.types import StreamStats

__all__ = ["FileSource"]

_SUPPORTED_WIDTHS: tuple[int, ...] = (1, 2, 4)
"""Sample widths in bytes this source can decode: 8-, 16- and 32-bit PCM."""


def _decode(raw: bytes, sampwidth: int, channels: int) -> np.ndarray:
    """Decode interleaved PCM bytes to a mono ``float32`` array in [-1, 1].

    Args:
        raw: Frame bytes as returned by :meth:`wave.Wave_read.readframes`.
        sampwidth: Bytes per sample; 1, 2 or 4.
        channels: Interleaved channel count.

    Returns:
        1-D ``float32`` array of ``len(raw) // (sampwidth * channels)`` frames,
        averaged across channels.
    """
    if sampwidth == 1:
        # 8-bit WAV is *unsigned*, biased by 128 -- the one format that is not
        # a straight signed integer.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


class FileSource(AudioSource):
    """Replay a WAV file through the pipeline, paced or as fast as possible.

    The header is read in :meth:`__init__`, so :attr:`sample_rate` and
    :attr:`channels` are available before :meth:`start` -- the pipeline needs
    them to reject a rate mismatch without opening a stream.

    Runs on one daemon thread named ``"file-source"``.  At end of file (when
    not looping) the thread sets :attr:`finished` and exits on its own; the
    owner does not have to call :meth:`stop` first, though calling it is
    always safe.
    """

    __slots__ = (
        "_path",
        "_blocksize",
        "_realtime",
        "_loop",
        "_sample_rate",
        "_channels",
        "_sampwidth",
        "_n_frames",
        "_stop_event",
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        on_audio: AudioCallback,
        blocksize: int = 1024,
        realtime: bool = True,
        loop: bool = False,
        stats: StreamStats | None = None,
    ) -> None:
        """Open ``path`` and read its header.

        Args:
            path: WAV file to replay.
            on_audio: Callback receiving each mono ``float32`` block.
            blocksize: Frames per emitted block; the final block of a pass may
                be shorter.
            realtime: Pace the replay to the audio clock when ``True``; emit as
                fast as possible when ``False``.
            loop: Restart from the beginning at end of file instead of
                finishing.
            stats: Shared counter record; see :class:`AudioSource`.

        Raises:
            ValueError: If ``blocksize`` is not positive, if the file is not
                readable as PCM WAV, or if its sample width is not 8, 16 or 32
                bits.
            OSError: If the file cannot be opened.
        """
        super().__init__(on_audio, stats)
        blocksize = int(blocksize)
        if blocksize <= 0:
            raise ValueError(f"blocksize must be > 0, got {blocksize}")

        self._path: str | os.PathLike[str] = path
        self._blocksize: int = blocksize
        self._realtime: bool = bool(realtime)
        self._loop: bool = bool(loop)
        self._stop_event: threading.Event = threading.Event()

        fspath = os.fspath(path)
        try:
            with wave.open(fspath, "rb") as wav:
                params = wav.getparams()
        except wave.Error as exc:
            # `wave` only understands uncompressed PCM; anything else (ADPCM,
            # mu-law, a WAVE_FORMAT_EXTENSIBLE variant it dislikes) lands here.
            raise ValueError(
                f"{fspath!r} is not a readable uncompressed PCM WAV file: {exc}"
            ) from exc

        if params.comptype != "NONE":
            raise ValueError(
                f"{fspath!r} uses compression {params.comptype!r} "
                f"({params.compname!r}); only uncompressed PCM is supported"
            )
        if params.sampwidth not in _SUPPORTED_WIDTHS:
            raise ValueError(
                f"{fspath!r} has sample width {params.sampwidth * 8} bits; only "
                f"8-, 16- and 32-bit PCM are supported"
            )

        self._sample_rate: int = params.framerate
        self._channels: int = params.nchannels
        self._sampwidth: int = params.sampwidth
        self._n_frames: int = params.nframes

    @property
    def path(self) -> str | os.PathLike[str]:
        """Path of the WAV file being replayed."""
        return self._path

    @property
    def sample_rate(self) -> int:
        """Sample rate from the file header, in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Channel count from the file header, before downmix."""
        return self._channels

    @property
    def n_frames(self) -> int:
        """Frame count from the file header (one pass, ignoring ``loop``)."""
        return self._n_frames

    @property
    def blocksize(self) -> int:
        """Frames per emitted block; the last block of a pass may be shorter."""
        return self._blocksize

    @property
    def realtime(self) -> bool:
        """``True`` if replay is paced to the audio clock."""
        return self._realtime

    @property
    def loop(self) -> bool:
        """``True`` if replay restarts at end of file."""
        return self._loop

    def start(self) -> None:
        """Launch the replay thread.

        Raises:
            RuntimeError: If this source was already started.  A source is
                single-use; create a new :class:`FileSource` to replay again.
        """
        if self._thread is not None:
            raise RuntimeError(
                "FileSource has already been started and cannot be restarted; "
                "create a new FileSource"
            )
        thread = threading.Thread(target=self._run, name="file-source", daemon=True)
        self._thread = thread
        thread.start()

    def stop(self, timeout: float | None = 2.0) -> bool:
        """Signal the replay thread to stop and wait for it to finish.

        The thread checks the stop flag every block and waits on it (rather
        than sleeping) while pacing, so it reacts within one block period even
        during a real-time replay.  Idempotent; a no-op returning ``True`` if
        never started.

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
            self._finished.set()
            return True
        if thread is threading.current_thread():
            # Called from on_audio: joining ourselves would deadlock.  The flag
            # is set, so the loop exits as soon as this callback returns.
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self) -> None:
        """Thread body: decode and emit blocks until EOF, stop, or error."""
        stop_event = self._stop_event
        blocksize = self._blocksize
        sampwidth = self._sampwidth
        channels = self._channels
        sample_rate = self._sample_rate

        try:
            with wave.open(os.fspath(self._path), "rb") as wav:
                start_t = time.monotonic()
                frames_emitted = 0
                frames_this_pass = 0

                while not stop_event.is_set():
                    raw = wav.readframes(blocksize)
                    if not raw:
                        if not self._loop:
                            break
                        if frames_this_pass == 0:
                            # Empty (or unreadable) file: looping would spin
                            # this thread at 100% CPU forever.
                            break
                        wav.rewind()
                        frames_this_pass = 0
                        continue

                    block = _decode(raw, sampwidth, channels)
                    if self._realtime:
                        # Absolute schedule: any overshoot is absorbed by the
                        # next block instead of accumulating into drift.
                        delay = (
                            start_t + frames_emitted / sample_rate - time.monotonic()
                        )
                        if delay > 0.0 and stop_event.wait(delay):
                            break

                    self._emit(block)
                    frames_emitted += len(block)
                    frames_this_pass += len(block)
        except BaseException as exc:  # noqa: BLE001 - reported via `error`
            # Nothing above us can catch this: re-raising would only print to
            # stderr from a daemon thread.  Record it and exit cleanly so the
            # owner can inspect `error` after `stop()`.
            self._error = exc
        finally:
            # Set unconditionally: EOF, stop and failure all mean "no more
            # audio is coming", which is exactly what `finished` promises.
            self._finished.set()
