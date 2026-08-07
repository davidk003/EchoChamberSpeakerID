"""Tests for echochamber.audio.devices, written from the step-4 contract.

Everything runs against `tests.fake_sounddevice`, never real hardware: the
deployment target (Windows ARM64) and any CI box have a completely different
device list, and a test that needs this developer's microphone is a test that
quietly stops running everywhere else.
"""

from __future__ import annotations

import pytest

from echochamber.audio.devices import (
    DeviceError,
    DeviceInfo,
    default_input_device,
    find_input_device,
    list_input_devices,
)
from tests.fake_sounddevice import DEVICES, FakeSounddevice


@pytest.fixture
def sd() -> FakeSounddevice:
    return FakeSounddevice()


# --------------------------------------------------------------------------
# enumeration
# --------------------------------------------------------------------------

def test_lists_only_devices_with_input_channels(sd: FakeSounddevice) -> None:
    infos = list_input_devices(sd)

    names = [i.name for i in infos]
    assert "Speakers (Realtek)" not in names, "output-only device must be filtered out"
    expected = sum(1 for d in DEVICES if d["max_input_channels"] > 0)
    assert len(infos) == expected


def test_devices_are_returned_in_portaudio_index_order(sd: FakeSounddevice) -> None:
    infos = list_input_devices(sd)
    assert [i.index for i in infos] == sorted(i.index for i in infos)


def test_device_info_fields_are_populated(sd: FakeSounddevice) -> None:
    info = next(i for i in list_input_devices(sd) if i.index == 2)

    assert info.name == "Microphone Array (Intel Smart Sound)"
    assert info.hostapi_name == "Windows WASAPI"
    assert info.max_input_channels == 2
    assert info.default_samplerate == 48000.0
    assert info.is_default_input is True


def test_is_wasapi_distinguishes_host_apis(sd: FakeSounddevice) -> None:
    by_index = {i.index: i for i in list_input_devices(sd)}

    assert by_index[2].is_wasapi is True, "index 2 is the WASAPI mic array"
    assert by_index[0].is_wasapi is False, "index 0 is MME"
    assert by_index[4].is_wasapi is False, "index 4 is WDM-KS"


def test_label_includes_host_api(sd: FakeSounddevice) -> None:
    """Device names repeat across host APIs, so the label must disambiguate."""
    info = next(i for i in list_input_devices(sd) if i.index == 2)
    assert "Microphone Array (Intel Smart Sound)" in info.label
    assert "Windows WASAPI" in info.label


def test_device_info_is_frozen(sd: FakeSounddevice) -> None:
    info = list_input_devices(sd)[0]
    with pytest.raises(Exception):
        info.name = "nope"  # type: ignore[misc]


def test_enumeration_survives_a_malformed_device_entry() -> None:
    """PortAudio hands back plain dicts; one bad entry must not kill the list."""
    devices = [dict(d) for d in DEVICES]
    devices.append({"name": "Broken Device"})  # missing every other key
    sd = FakeSounddevice(devices=devices)

    infos = list_input_devices(sd)  # must not raise

    assert any(i.name == "Microphone Array (Intel Smart Sound)" for i in infos), (
        "a malformed entry must not prevent the good devices being listed"
    )


# --------------------------------------------------------------------------
# default device
# --------------------------------------------------------------------------

def test_default_input_device_is_found(sd: FakeSounddevice) -> None:
    info = default_input_device(sd)
    assert info is not None
    assert info.index == 2
    assert info.is_default_input is True


def test_default_input_device_returns_none_when_there_is_none() -> None:
    """A machine with no capture hardware must not raise on startup."""
    sd = FakeSounddevice(default_input=None)
    assert default_input_device(sd) is None


def test_only_the_default_device_is_flagged(sd: FakeSounddevice) -> None:
    flagged = [i for i in list_input_devices(sd) if i.is_default_input]
    assert len(flagged) == 1 and flagged[0].index == 2


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def test_find_by_index(sd: FakeSounddevice) -> None:
    assert find_input_device(3, sd).name == "Virtual Mic (AudioRelay)"


def test_find_by_name_substring_is_case_insensitive(sd: FakeSounddevice) -> None:
    assert find_input_device("intel smart", sd).index == 2


def test_find_none_resolves_to_the_default(sd: FakeSounddevice) -> None:
    assert find_input_device(None, sd).index == 2


def test_find_unknown_index_raises_device_error(sd: FakeSounddevice) -> None:
    with pytest.raises(DeviceError):
        find_input_device(99, sd)


def test_find_unknown_name_raises_device_error_listing_devices(
    sd: FakeSounddevice,
) -> None:
    """The error has to be actionable -- the user cannot guess device names."""
    with pytest.raises(DeviceError) as exc:
        find_input_device("no such microphone", sd)

    message = str(exc.value)
    assert "no such microphone" in message
    assert "Microphone Array" in message, (
        "the error should list the devices that ARE available"
    )


def test_ambiguous_name_raises_device_error(sd: FakeSounddevice) -> None:
    """'Mic' matches two devices; silently picking one would be a trap."""
    with pytest.raises(DeviceError) as exc:
        find_input_device("mic", sd)
    assert "mic" in str(exc.value).lower()


def test_find_by_index_rejects_an_output_only_device(sd: FakeSounddevice) -> None:
    with pytest.raises(DeviceError):
        find_input_device(1, sd)  # Speakers, 0 input channels


def test_find_raises_when_no_default_exists() -> None:
    sd = FakeSounddevice(default_input=None)
    with pytest.raises(DeviceError):
        find_input_device(None, sd)


def test_returned_type_is_deviceinfo(sd: FakeSounddevice) -> None:
    assert isinstance(find_input_device(2, sd), DeviceInfo)
