"""Live capture from a PortAudio input device -- the real front end.

A drop-in peer of :class:`~echochamber.audio.sources.file_source.FileSource`:
same :class:`~echochamber.audio.sources.base.AudioSource` base, same
lifecycle, same :meth:`~echochamber.audio.sources.base.AudioSource._emit` path,
so an :class:`~echochamber.audio.pipeline.AudioPipeline` swaps one for the
other and nothing else changes.  The only structural difference is that there
is no thread of our own: PortAudio owns the thread and calls us.

**WASAPI will not resample unless you ask it to.**  Measured on real hardware
(Intel mic array, 48 kHz native, PortAudio 19.7)::

    16000 Hz mono, plain        -> PortAudioError: Invalid sample rate [-9997]
    16000 Hz mono, auto_convert -> opens at 16000 Hz, real signal

So when -- and only when -- the resolved device is a WASAPI device, the stream
is opened with ``extra_settings=WasapiSettings(auto_convert=True)``.  Passing
those settings to MME, DirectSound or WDM-KS is an error, and those host APIs
convert without being asked.  Nearly every Windows capture device is 44.1 or
48 kHz native, so without this flag 16 kHz capture fails on essentially all of
them.

There is deliberately **no fallback to the device's native rate**.  The whole
pipeline derives its window geometry from ``config.sample_rate``; a source
quietly delivering 48 kHz would not merely sound wrong, it would corrupt every
``start_frame`` and every timestamp derived from one.  If the requested rate
cannot be opened, that is a :class:`~echochamber.audio.devices.DeviceError`
naming the device and its native rate.

**The callback is a real-time path.**  It allocates nothing beyond the one
mono copy numpy makes for us, takes no lock, logs nothing, and never touches
Qt.  It copies each block before emitting, because PortAudio reuses its input
buffer the moment we return and a downstream sink may hold onto what it was
given.  And it catches everything: an exception escaping into PortAudio is
undefined behaviour in a C callback, so a failure is recorded in
:attr:`~echochamber.audio.sources.base.AudioSource.error` and the stream is
torn down through PortAudio's own mechanism, ``CallbackAbort``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from echochamber.audio.devices import DeviceError, DeviceInfo, _resolve_sd, find_input_device
from echochamber.audio.sources.base import AudioCallback, AudioSource
from echochamber.audio.types import StreamStats

__all__ = ["SoundDeviceSource"]


class SoundDeviceSource(AudioSource):
    """Capture mono ``float32`` audio from a PortAudio input device.

    The device is resolved in :meth:`__init__`, so :attr:`sample_rate`,
    :attr:`channels` and :attr:`device_info` are all known before
    :meth:`start` -- exactly like ``FileSource`` reading a WAV header, and for
    the same reason: the pipeline validates the rate before anything opens.

    Single-use.  ``start()`` twice raises; create a new source instead, which
    is also what you want when the user picks a different device.
    """

    __slots__ = (
        "_sd",
        "_device_info",
        "_sample_rate",
        "_blocksize",
        "_stream",
        "_started",
    )

    def __init__(
        self,
        on_audio: AudioCallback,
        device: int | str | None = None,
        sample_rate: int = 16_000,
        blocksize: int = 160,
        stats: StreamStats | None = None,
        sd_module: Any = None,
    ) -> None:
        """Resolve the device; open nothing until :meth:`start`.

        Args:
            on_audio: Callback receiving each mono ``float32`` block, on
                PortAudio's callback thread.  In the assembled pipeline this
                is :meth:`RingBuffer.write
                <echochamber.audio.ringbuffer.RingBuffer.write>`.
            device: PortAudio index, case-insensitive name fragment, or
                ``None`` for the system default.  See
                :func:`~echochamber.audio.devices.find_input_device`.
            sample_rate: Rate to request from the device, in Hz.  Must match
                ``config.sample_rate``; this source never resamples in Python
                and never falls back to the native rate.
            blocksize: Frames per device callback (160 = 10 ms at 16 kHz).
                ``0`` lets PortAudio choose, which gives variable-sized
                blocks -- harmless here, since the ring does not care.
            stats: Shared counter record; see
                :class:`~echochamber.audio.sources.base.AudioSource`.  This
                source is also the only place ``xruns`` is ever incremented.
            sd_module: Module to use instead of the real :mod:`sounddevice`.
                The testing seam; it is threaded through device resolution and
                stream creation alike, so nothing here touches real hardware
                when it is supplied.

        Raises:
            DeviceError: If ``device`` matches no input device or is
                ambiguous.
            ValueError: If ``sample_rate`` is not positive or ``blocksize`` is
                negative.
        """
        super().__init__(on_audio, stats)

        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        blocksize = int(blocksize)
        if blocksize < 0:
            raise ValueError(f"blocksize must be >= 0, got {blocksize}")

        self._sd: Any = _resolve_sd(sd_module)
        self._device_info: DeviceInfo = find_input_device(device, sd_module=self._sd)
        self._sample_rate: int = sample_rate
        self._blocksize: int = blocksize
        self._stream: Any = None
        self._started: bool = False

    @property
    def device_info(self) -> DeviceInfo:
        """The resolved device, as of construction."""
        return self._device_info

    @property
    def sample_rate(self) -> int:
        """Rate requested from the device, in Hz -- and delivered, or the open failed."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Always 1: the stream itself is opened mono.

        WASAPI (and PortAudio for the other host APIs) downmixes a multi
        channel microphone array for us, which is cheaper and better than
        averaging in Python.  The callback still averages defensively if more
        than one channel ever arrives.
        """
        return 1

    @property
    def blocksize(self) -> int:
        """Frames per device callback; ``0`` means PortAudio chooses."""
        return self._blocksize

    @property
    def stream(self) -> Any:
        """The open ``InputStream``, or ``None`` before start / after stop."""
        return self._stream

    @property
    def is_running(self) -> bool:
        """``True`` while the PortAudio stream is active.

        Overrides the base implementation, which asks about a worker thread
        this source does not have.
        """
        stream = self._stream
        if stream is None:
            return False
        try:
            return bool(stream.active)
        except Exception:  # noqa: BLE001 - a closed or dead stream is not running
            return False

    @property
    def latency(self) -> float:
        """Input latency the stream reports, in seconds; ``0.0`` when stopped.

        PortAudio's *actual* latency for the opened stream, not the requested
        one -- which is the number the GUI's latency budget should display,
        since WASAPI shared mode routinely grants something other than what
        was asked for.
        """
        stream = self._stream
        if stream is None or not self.is_running:
            return 0.0
        try:
            latency = stream.latency
            if isinstance(latency, (tuple, list)):
                latency = latency[0] if latency else 0.0
            return float(latency)
        except Exception:  # noqa: BLE001 - informational only, never fatal
            return 0.0

    def start(self) -> None:
        """Open and start the input stream.

        Raises:
            RuntimeError: If this source was already started.  Sources are
                single-use; restarting would resume mid-stream against a ring
                whose frame counter has moved on.
            DeviceError: If PortAudio refuses to open the stream -- most often
                because the requested rate is not the device's native rate.
                The message names the device and its native rate.
        """
        if self._started:
            raise RuntimeError(
                "SoundDeviceSource has already been started and cannot be "
                "restarted; create a new SoundDeviceSource"
            )
        self._started = True

        sd = self._sd
        # Only WASAPI takes (or needs) these settings.  Handing them to MME,
        # DirectSound or WDM-KS is an error, not a no-op.
        extra_settings = (
            sd.WasapiSettings(auto_convert=True)
            if self._device_info.is_wasapi
            else None
        )

        stream = None
        try:
            stream = sd.InputStream(
                device=self._device_info.index,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._blocksize,
                callback=self._callback,
                finished_callback=self._on_stream_finished,
                extra_settings=extra_settings,
            )
            stream.start()
        except Exception as exc:  # noqa: BLE001 - re-raised as DeviceError below
            if stream is not None:
                _close_quietly(stream)
            self._stream = None
            error = self._as_device_error(exc)
            self._error = error
            # No audio is coming, so anything waiting on the pipeline must be
            # released rather than left to time out.
            self._finished.set()
            raise error from exc

        self._stream = stream

    def stop(self, timeout: float | None = 2.0) -> bool:
        """Stop and close the stream, and mark the source finished.

        Idempotent, safe on a source that was never started, and safe on one
        whose stream already died -- PortAudio errors during teardown are
        swallowed, because by then there is nothing left to salvage and the
        caller's shutdown path must not raise.

        Args:
            timeout: Accepted for interface compatibility with
                :class:`~echochamber.audio.sources.base.AudioSource`.  Closing
                a PortAudio stream is bounded by the driver, not by us, so
                there is no worker to join and no wait to bound.

        Returns:
            ``True`` once the stream is closed (or was never opened).
        """
        stream = self._stream
        self._stream = None
        if stream is not None:
            _close_quietly(stream)
        self._finished.set()
        return True

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """PortAudio callback: downmix, copy, emit.  Real-time path.

        No allocation beyond the mono copy, no lock, no logging, no Qt.  Every
        exception is captured and converted into a ``CallbackAbort``, because
        letting a Python exception unwind into a C callback is undefined
        behaviour.

        Args:
            indata: PortAudio's input buffer, shape ``(frames, channels)``.
                **Reused** after this call returns, hence the copy.
            frames: Frame count in ``indata``.
            time_info: PortAudio timestamps; unused.
            status: Callback flags.  ``input_overflow`` means the device
                dropped audio before we ever saw it -- an xrun.
        """
        try:
            if status:
                # A fake or an older binding may pass a bare truthy flag object
                # with no attribute; treat that as the overflow it is meant to
                # signal rather than silently losing the count.
                if getattr(status, "input_overflow", True):
                    self._stats.xruns += 1

            if indata.ndim == 1:
                block = indata.copy()
            elif indata.shape[1] == 1:
                # The stream is opened mono, so this is the normal path.  The
                # copy is mandatory: PortAudio reuses this buffer.
                block = indata[:, 0].copy()
            else:
                # Defensive: a host API that ignored `channels=1`.  `mean`
                # allocates a fresh array, so this is a copy already.
                block = indata.mean(axis=1, dtype=np.float32)

            self._emit(block)
        except BaseException as exc:  # noqa: BLE001 - reported via `error`
            self._error = exc
            self._finished.set()
            raise self._sd.CallbackAbort from None

    def _on_stream_finished(self) -> None:
        """PortAudio's finished callback: the stream will deliver no more audio.

        Reached on a normal stop *and* on an abort from :meth:`_callback`,
        which is what makes :attr:`finished` honest when capture dies on its
        own rather than being stopped.
        """
        self._finished.set()

    def _as_device_error(self, exc: BaseException) -> DeviceError:
        """Wrap a stream-open failure in a :class:`DeviceError` worth reading.

        The failure is nearly always the sample rate, and the two facts needed
        to understand it -- which device, and what rate it actually runs at --
        are not in PortAudio's message.

        Args:
            exc: The exception raised by ``InputStream(...)`` or ``start()``.

        Returns:
            A :class:`DeviceError`; ``exc`` itself if it already is one.
        """
        if isinstance(exc, DeviceError):
            return exc
        device = self._device_info
        hint = ""
        if not device.is_wasapi and self._sample_rate != int(device.default_samplerate):
            hint = (
                "; this host API does not convert sample rates, so try the "
                "WASAPI entry for the same device"
            )
        return DeviceError(
            f"could not open input device [{device.index}] {device.label} at "
            f"{self._sample_rate} Hz mono (device native rate "
            f"{device.default_samplerate:g} Hz): {exc}{hint}"
        )

    def __repr__(self) -> str:
        """Return a debugging representation including the device label."""
        return (
            f"{type(self).__name__}(device=[{self._device_info.index}] "
            f"{self._device_info.label}, sample_rate={self._sample_rate}, "
            f"blocksize={self._blocksize}, running={self.is_running}, "
            f"finished={self._finished.is_set()})"
        )


def _close_quietly(stream: Any) -> None:
    """Stop and close ``stream``, ignoring anything it complains about.

    Called on the teardown path and on a failed open, where an exception would
    only mask the failure that got us here.  ``stop`` rather than ``abort``:
    the blocks already handed to the driver are worth having, and the wait is
    one device period.

    Args:
        stream: A PortAudio stream, possibly already dead or half-open.
    """
    try:
        stream.stop()
    except Exception:  # noqa: BLE001 - already stopped, dead, or never started
        pass
    try:
        stream.close()
    except Exception:  # noqa: BLE001 - see above
        pass
