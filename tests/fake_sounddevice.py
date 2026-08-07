"""A fake `sounddevice` module for testing capture without hardware.

Every public entry point in ``echochamber.audio.devices`` and
``SoundDeviceSource`` accepts an ``sd_module`` argument precisely so the real
PortAudio can be swapped for this. That matters beyond convenience: the machines
that run these tests (CI, and the Windows ARM64 deployment target) do not have
this developer's microphone, and a suite that needs a specific device is a suite
that silently stops running.

The device table mirrors the shape PortAudio actually returns on Windows,
including the detail that matters most: WASAPI devices whose native rate is
48 kHz, which reject a 16 kHz request unless auto-convert is enabled.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# Host API indices, mirroring a real Windows box.
MME = 0
DIRECTSOUND = 1
WASAPI = 2
WDMKS = 3

HOSTAPIS: list[dict[str, Any]] = [
    {"name": "MME"},
    {"name": "Windows DirectSound"},
    {"name": "Windows WASAPI"},
    {"name": "Windows WDM-KS"},
]

# index -> device dict. Mirrors real hardware: 48 kHz WASAPI capture devices.
DEVICES: list[dict[str, Any]] = [
    {"name": "Microsoft Sound Mapper - Input", "hostapi": MME,
     "max_input_channels": 2, "max_output_channels": 0,
     "default_samplerate": 44100.0},
    {"name": "Speakers (Realtek)", "hostapi": MME,
     "max_input_channels": 0, "max_output_channels": 2,
     "default_samplerate": 44100.0},
    {"name": "Microphone Array (Intel Smart Sound)", "hostapi": WASAPI,
     "max_input_channels": 2, "max_output_channels": 0,
     "default_samplerate": 48000.0},
    {"name": "Virtual Mic (AudioRelay)", "hostapi": WASAPI,
     "max_input_channels": 2, "max_output_channels": 0,
     "default_samplerate": 48000.0},
    {"name": "Stereo Mix (Realtek)", "hostapi": WDMKS,
     "max_input_channels": 2, "max_output_channels": 0,
     "default_samplerate": 48000.0},
]

DEFAULT_INPUT_INDEX = 2


class PortAudioError(Exception):
    """Stand-in for sounddevice.PortAudioError."""


class CallbackAbort(Exception):
    """Stand-in for sounddevice.CallbackAbort."""


class CallbackStop(Exception):
    """Stand-in for sounddevice.CallbackStop."""


class WasapiSettings:
    """Stand-in for sounddevice.WasapiSettings."""

    def __init__(self, auto_convert: bool = False, **kwargs: Any) -> None:
        self.auto_convert = auto_convert
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"WasapiSettings(auto_convert={self.auto_convert})"


class CallbackFlags:
    """Stand-in for the `status` object handed to an input callback."""

    def __init__(self, input_overflow: bool = False) -> None:
        self.input_overflow = input_overflow
        self.input_underflow = False
        self.output_overflow = False
        self.output_underflow = False
        self.priming_output = False

    def __bool__(self) -> bool:
        return bool(self.input_overflow)

    def __str__(self) -> str:
        return "input overflow" if self.input_overflow else ""


class FakeInputStream:
    """Records how it was opened; lets a test drive the callback by hand."""

    instances: list["FakeInputStream"] = []

    # Set by a test to make the constructor blow up like PortAudio does.
    raise_on_open: Exception | None = None

    def __init__(
        self,
        device: Any = None,
        samplerate: float | None = None,
        channels: int | None = None,
        dtype: str | None = None,
        blocksize: int | None = None,
        callback: Callable[..., None] | None = None,
        extra_settings: Any = None,
        latency: Any = None,
        **kwargs: Any,
    ) -> None:
        if FakeInputStream.raise_on_open is not None:
            raise FakeInputStream.raise_on_open
        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback
        self.extra_settings = extra_settings
        self.requested_latency = latency
        self.kwargs = kwargs

        self.active = False
        self.closed = False
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0
        FakeInputStream.instances.append(self)

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self.start_calls += 1
        self.active = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.active = False

    def close(self) -> None:
        self.close_calls += 1
        self.active = False
        self.closed = True

    def __enter__(self) -> "FakeInputStream":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
        self.close()

    @property
    def latency(self) -> float:
        return 0.01

    # -- test driving ----------------------------------------------------
    def feed(self, block: np.ndarray, status: CallbackFlags | None = None) -> None:
        """Invoke the stream's callback exactly as PortAudio would."""
        assert self.callback is not None, "no callback was registered"
        arr = np.asarray(block, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.callback(arr, arr.shape[0], None, status or CallbackFlags())

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.raise_on_open = None


class _Default:
    """Stand-in for sounddevice.default."""

    def __init__(self) -> None:
        self.device = [DEFAULT_INPUT_INDEX, 1]
        self.samplerate = None
        self.channels = [1, 2]
        self.dtype = ["float32", "float32"]


class FakeSounddevice:
    """Drop-in replacement for the parts of `sounddevice` we use."""

    PortAudioError = PortAudioError
    CallbackAbort = CallbackAbort
    CallbackStop = CallbackStop
    WasapiSettings = WasapiSettings
    InputStream = FakeInputStream

    def __init__(
        self,
        devices: list[dict[str, Any]] | None = None,
        hostapis: list[dict[str, Any]] | None = None,
        default_input: int | None = DEFAULT_INPUT_INDEX,
    ) -> None:
        self._devices = [dict(d) for d in (devices if devices is not None else DEVICES)]
        self._hostapis = [dict(h) for h in (hostapis if hostapis is not None else HOSTAPIS)]
        self._default_input = default_input
        self.default = _Default()
        self.default.device = [
            -1 if default_input is None else default_input, 1
        ]
        FakeInputStream.reset()

    # -- enumeration -----------------------------------------------------
    def query_devices(self, device: Any = None, kind: str | None = None) -> Any:
        if kind == "input":
            if self._default_input is None:
                raise PortAudioError("Error querying device -1")
            return self._with_index(self._default_input)
        if device is None:
            return [self._with_index(i) for i in range(len(self._devices))]
        if isinstance(device, str):
            matches = [
                i for i, d in enumerate(self._devices)
                if device.lower() in d["name"].lower()
            ]
            if not matches:
                raise ValueError(f"no device matching {device!r}")
            return self._with_index(matches[0])
        idx = int(device)
        if idx < 0 or idx >= len(self._devices):
            raise PortAudioError(f"Error querying device {idx}")
        return self._with_index(idx)

    def query_hostapis(self, index: int | None = None) -> Any:
        if index is None:
            return [dict(h) for h in self._hostapis]
        return dict(self._hostapis[index])

    def _with_index(self, i: int) -> dict[str, Any]:
        d = dict(self._devices[i])
        d["index"] = i
        return d

    # -- misc ------------------------------------------------------------
    def check_input_settings(self, **kwargs: Any) -> None:
        return None

    def get_portaudio_version(self) -> tuple[int, str]:
        return (1247, "PortAudio V19.7.0-devel (fake)")

    def sleep(self, ms: int) -> None:
        return None
