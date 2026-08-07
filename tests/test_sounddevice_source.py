"""Tests for SoundDeviceSource, written from the step-4 contract.

Driven entirely through the injected fake `sounddevice`, so no microphone is
required. The tests invoke the PortAudio callback by hand, which is the only way
to assert the real-time path's behaviour deterministically.

The headline test is `test_wasapi_device_gets_auto_convert`. Measured on real
hardware: a WASAPI device at 48 kHz native rejects a 16 kHz request with
"Invalid sample rate [PaErrorCode -9997]" unless the stream is opened with
WasapiSettings(auto_convert=True). Almost every Windows capture device is 44.1 or
48 kHz, so without that flag live capture fails on virtually every machine --
while every headless test in this repo keeps passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from echochamber.audio.devices import DeviceError
from echochamber.audio.sources.sounddevice_source import SoundDeviceSource
from echochamber.audio.types import StreamStats
from tests.fake_sounddevice import (
    CallbackFlags,
    FakeInputStream,
    FakeSounddevice,
    PortAudioError,
    WasapiSettings,
)


@pytest.fixture
def sd() -> FakeSounddevice:
    return FakeSounddevice()


@pytest.fixture
def collector() -> list[np.ndarray]:
    return []


def make_source(sd: FakeSounddevice, collector: list, **kw) -> SoundDeviceSource:
    kw.setdefault("device", 2)          # the WASAPI mic array
    kw.setdefault("sample_rate", 16000)
    kw.setdefault("blocksize", 160)
    return SoundDeviceSource(collector.append, sd_module=sd, **kw)


def last_stream() -> FakeInputStream:
    assert FakeInputStream.instances, "no InputStream was opened"
    return FakeInputStream.instances[-1]


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_device_is_resolved_before_start(sd: FakeSounddevice, collector: list) -> None:
    """Like FileSource, the format must be known before the stream runs."""
    src = make_source(sd, collector)

    assert src.device_info.index == 2
    assert src.sample_rate == 16000
    assert src.channels == 1
    assert src.is_running is False
    assert not FakeInputStream.instances, "no stream may open until start()"


def test_device_can_be_selected_by_name(sd: FakeSounddevice, collector: list) -> None:
    src = make_source(sd, collector, device="intel smart")
    assert src.device_info.index == 2


def test_unknown_device_raises_device_error(sd: FakeSounddevice, collector: list) -> None:
    with pytest.raises(DeviceError):
        make_source(sd, collector, device="not a real device")


# --------------------------------------------------------------------------
# THE CRITICAL ONE: WASAPI auto-convert
# --------------------------------------------------------------------------

def test_wasapi_device_gets_auto_convert(sd: FakeSounddevice, collector: list) -> None:
    """Without this, 16 kHz capture fails on essentially every Windows machine."""
    src = make_source(sd, collector, device=2)   # WASAPI, 48 kHz native
    src.start()

    extra = last_stream().extra_settings
    assert isinstance(extra, WasapiSettings), (
        "a WASAPI device must be opened with WasapiSettings; PortAudio otherwise "
        "rejects any non-native sample rate with PaErrorCode -9997"
    )
    assert extra.auto_convert is True

    src.stop()


def test_non_wasapi_device_gets_no_wasapi_settings(
    sd: FakeSounddevice, collector: list
) -> None:
    """Passing WasapiSettings to MME/DirectSound/WDM-KS is an error."""
    src = make_source(sd, collector, device=0)   # MME
    src.start()

    assert last_stream().extra_settings is None
    src.stop()


def test_wdmks_device_also_gets_no_wasapi_settings(
    sd: FakeSounddevice, collector: list
) -> None:
    src = make_source(sd, collector, device=4)   # WDM-KS
    src.start()
    assert last_stream().extra_settings is None
    src.stop()


# --------------------------------------------------------------------------
# stream opening
# --------------------------------------------------------------------------

def test_stream_is_opened_with_the_requested_format(
    sd: FakeSounddevice, collector: list
) -> None:
    src = make_source(sd, collector, sample_rate=16000, blocksize=160)
    src.start()

    s = last_stream()
    assert s.samplerate == 16000
    assert s.channels == 1
    assert s.dtype == "float32"
    assert s.blocksize == 160
    assert s.device == 2
    assert callable(s.callback)

    src.stop()


def test_portaudio_failure_becomes_a_device_error_naming_the_device(
    sd: FakeSounddevice, collector: list
) -> None:
    """The raw PortAudio message alone does not tell a user what to change."""
    src = make_source(sd, collector, device=2)
    FakeInputStream.raise_on_open = PortAudioError(
        "Invalid sample rate [PaErrorCode -9997]"
    )
    try:
        with pytest.raises(DeviceError) as exc:
            src.start()
    finally:
        FakeInputStream.raise_on_open = None

    message = str(exc.value)
    assert "Microphone Array" in message, "name the device that failed"
    assert "48000" in message, "report the device's native rate"


def test_is_running_tracks_the_stream(sd: FakeSounddevice, collector: list) -> None:
    src = make_source(sd, collector)
    assert src.is_running is False
    src.start()
    assert src.is_running is True
    src.stop()
    assert src.is_running is False


def test_latency_is_reported_while_running(sd: FakeSounddevice, collector: list) -> None:
    src = make_source(sd, collector)
    assert src.latency == 0.0, "no latency before the stream exists"
    src.start()
    assert src.latency > 0.0
    src.stop()


# --------------------------------------------------------------------------
# the real-time callback
# --------------------------------------------------------------------------

def test_callback_emits_mono_float32(sd: FakeSounddevice, collector: list) -> None:
    src = make_source(sd, collector)
    src.start()

    block = np.arange(160, dtype=np.float32) / 1000.0
    last_stream().feed(block)

    assert len(collector) == 1
    out = collector[0]
    assert out.ndim == 1, "the ring expects a flat mono block"
    assert out.dtype == np.float32
    assert np.allclose(out, block)
    src.stop()


def test_callback_downmixes_multichannel_defensively(
    sd: FakeSounddevice, collector: list
) -> None:
    """We request mono, but must not corrupt audio if more channels arrive."""
    src = make_source(sd, collector)
    src.start()

    stereo = np.stack([np.full(64, 0.4, np.float32), np.full(64, 0.8, np.float32)], axis=1)
    last_stream().feed(stereo)

    assert len(collector) == 1
    assert np.allclose(collector[0], 0.6), "channels must be averaged, not truncated"
    src.stop()


def test_callback_copies_the_portaudio_buffer(
    sd: FakeSounddevice, collector: list
) -> None:
    """PortAudio reuses its buffer; a retained reference would silently corrupt."""
    src = make_source(sd, collector)
    src.start()

    block = np.full(64, 0.25, dtype=np.float32)
    last_stream().feed(block)
    block[:] = -1.0                      # PortAudio reusing its buffer

    assert np.allclose(collector[0], 0.25), (
        "the emitted block must be a copy, not a view into PortAudio's buffer"
    )
    src.stop()


def test_callback_updates_capture_stats(sd: FakeSounddevice, collector: list) -> None:
    stats = StreamStats()
    src = make_source(sd, collector, stats=stats)
    src.start()

    last_stream().feed(np.full(160, 0.5, dtype=np.float32))
    last_stream().feed(np.full(160, 0.5, dtype=np.float32))

    assert stats.frames_captured == 320
    assert stats.peak_level == pytest.approx(0.5)
    assert stats.rms_level == pytest.approx(0.5)
    src.stop()


def test_input_overflow_increments_xruns(sd: FakeSounddevice, collector: list) -> None:
    """A dropout must be counted, not swallowed -- the GUI has to surface it."""
    stats = StreamStats()
    src = make_source(sd, collector, stats=stats)
    src.start()

    last_stream().feed(np.zeros(160, np.float32), CallbackFlags(input_overflow=True))
    assert stats.xruns == 1

    last_stream().feed(np.zeros(160, np.float32), CallbackFlags(input_overflow=False))
    assert stats.xruns == 1, "a clean block must not increment xruns"
    src.stop()


def test_callback_exception_is_captured_and_aborts_the_stream(
    sd: FakeSounddevice,
) -> None:
    """An exception must never propagate into PortAudio's C callback."""
    def boom(block: np.ndarray) -> None:
        raise ValueError("downstream exploded")

    src = SoundDeviceSource(boom, device=2, sample_rate=16000, sd_module=sd)
    src.start()

    with pytest.raises(sd.CallbackAbort):
        last_stream().feed(np.zeros(160, np.float32))

    assert isinstance(src.error, ValueError)
    assert "downstream exploded" in str(src.error)
    src.stop()


# --------------------------------------------------------------------------
# lifecycle
# --------------------------------------------------------------------------

def test_stop_before_start_is_safe(sd: FakeSounddevice, collector: list) -> None:
    src = make_source(sd, collector)
    assert src.stop(timeout=1.0) is True
    assert src.finished.is_set()


def test_stop_is_idempotent(sd: FakeSounddevice, collector: list) -> None:
    src = make_source(sd, collector)
    src.start()
    assert src.stop(timeout=1.0) is True
    assert src.stop(timeout=1.0) is True


def test_stop_closes_the_stream_and_sets_finished(
    sd: FakeSounddevice, collector: list
) -> None:
    src = make_source(sd, collector)
    src.start()
    stream = last_stream()

    src.stop(timeout=1.0)

    assert stream.close_calls >= 1, "the stream must be closed, not just stopped"
    assert src.finished.is_set()
    assert src.is_running is False


def test_stop_survives_a_stream_that_already_died(
    sd: FakeSounddevice, collector: list
) -> None:
    """After a CallbackAbort the stream may already be torn down."""
    src = make_source(sd, collector)
    src.start()
    stream = last_stream()

    def explode() -> None:
        raise PortAudioError("stream already closed")

    stream.stop = explode  # type: ignore[method-assign]

    assert src.stop(timeout=1.0) is True, "stop() must not raise on a dead stream"
    assert src.finished.is_set()


def test_peak_and_rms_are_actually_different_measurements(
    sd: FakeSounddevice, collector: list
) -> None:
    """A constant signal has peak == RMS, so it cannot tell the two apart.

    Every earlier level test fed a constant block, which meant a peak meter and
    an RMS meter were indistinguishable. This uses a crest factor far from 1:
    a single full-scale sample in an otherwise silent block.
    """
    stats = StreamStats()
    src = make_source(sd, collector, stats=stats)
    src.start()

    block = np.zeros(100, dtype=np.float32)
    block[0] = 1.0
    last_stream().feed(block)

    assert stats.peak_level == pytest.approx(1.0), "peak must be the maximum"
    assert stats.rms_level == pytest.approx(0.1), "rms of one full-scale in 100 is 0.1"
    assert stats.peak_level != pytest.approx(stats.rms_level), (
        "peak and rms must be distinguishable measurements"
    )
    src.stop()
