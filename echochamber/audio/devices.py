"""Input device enumeration and resolution, on top of PortAudio.

This is the only module that knows how PortAudio describes hardware.  It turns
``sounddevice``'s loosely typed dictionaries into a frozen
:class:`DeviceInfo` record, and resolves the three things a caller might have
in hand -- an index, a fragment of a name, or nothing at all -- into exactly
one device.

Two properties of the design matter more than they look:

* **Every entry point takes an optional ``sd_module``.**  It defaults to the
  real :mod:`sounddevice`, but nothing here ever calls a module-level ``sd.``
  directly, so the whole module (and :class:`SoundDeviceSource
  <echochamber.audio.sources.sounddevice_source.SoundDeviceSource>` above it)
  can be exercised against a fake on a machine with no audio hardware, in CI,
  or on a build agent where opening PortAudio would fail outright.
* **Enumeration is defensive.**  PortAudio hands back plain dicts assembled by
  a host API backend; a driver that reports a malformed or incomplete entry
  must cost that one device, not the entire device list.  Likewise a machine
  with no usable default input yields ``None`` from
  :func:`default_input_device` rather than an exception -- "no microphone" is
  a state the GUI has to render, not an error.

The WASAPI distinction (:attr:`DeviceInfo.is_wasapi`) exists because a WASAPI
stream must be opened with ``auto_convert`` to accept a non-native sample rate,
and *only* a WASAPI stream may be handed ``WasapiSettings`` at all.  See
:mod:`echochamber.audio.sources.sounddevice_source`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

__all__ = [
    "DeviceError",
    "DeviceInfo",
    "WASAPI_HOSTAPI_NAME",
    "default_input_device",
    "find_input_device",
    "list_input_devices",
]

WASAPI_HOSTAPI_NAME: str = "Windows WASAPI"
"""PortAudio's name for the host API that needs ``auto_convert`` to resample."""


class DeviceError(RuntimeError):
    """A capture device could not be resolved or opened.

    Raised when a device specification matches nothing or is ambiguous, when
    the machine has no usable input at all, or when PortAudio refuses to open
    the requested stream (in which case the message names the device and its
    native sample rate).
    """


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Everything the pipeline needs to know about one input device.

    A frozen snapshot taken at enumeration time: PortAudio's device table can
    change when hardware is plugged in or unplugged, and re-enumerating is the
    only way to notice.

    Attributes:
        index: PortAudio device index, as passed to ``InputStream(device=...)``.
        name: Device name as reported by the driver.
        hostapi_index: Index of the host API this device belongs to.
        hostapi_name: Host API name, e.g. ``"Windows WASAPI"`` or ``"MME"``.
            The same physical microphone appears once per host API.
        max_input_channels: Maximum capture channels; always ``> 0`` for a
            device returned by :func:`list_input_devices`.
        default_samplerate: The device's native rate in Hz.  Requesting
            anything else from a WASAPI device requires ``auto_convert``.
        is_default_input: ``True`` if PortAudio reports this as the system
            default input device.
    """

    index: int
    name: str
    hostapi_index: int
    hostapi_name: str
    max_input_channels: int
    default_samplerate: float
    is_default_input: bool

    @property
    def is_wasapi(self) -> bool:
        """``True`` if this device is exposed through Windows WASAPI.

        The one host API that rejects a non-native sample rate unless the
        stream is opened with ``WasapiSettings(auto_convert=True)`` -- and the
        only one that may be given ``WasapiSettings`` at all.
        """
        return self.hostapi_name == WASAPI_HOSTAPI_NAME

    @property
    def label(self) -> str:
        """Human-readable ``"name (host API)"``, for menus and error messages.

        The host API is part of the label because the same microphone is
        listed once per host API and the names alone are identical.
        """
        return f"{self.name} ({self.hostapi_name})"


def list_input_devices(sd_module: Any = None) -> list[DeviceInfo]:
    """Enumerate every device with at least one input channel.

    Args:
        sd_module: Module to query instead of the real :mod:`sounddevice`.
            The testing seam; ``None`` means the real module.

    Returns:
        Devices with ``max_input_channels > 0``, in PortAudio index order.
        Output-only devices and entries too malformed to interpret are
        skipped, so this may be shorter than PortAudio's device table -- and
        may legitimately be empty on a machine with no capture hardware.

    Raises:
        DeviceError: If :mod:`sounddevice` cannot be imported.
    """
    sd = _resolve_sd(sd_module)
    entries = _query_devices(sd)
    hostapi_names = _hostapi_names(sd)
    default_index = _default_input_index(sd)

    devices: list[DeviceInfo] = []
    for position, entry in enumerate(entries):
        info = _to_device_info(entry, position, hostapi_names, default_index)
        if info is not None and info.max_input_channels > 0:
            devices.append(info)
    return devices


def default_input_device(sd_module: Any = None) -> DeviceInfo | None:
    """Return the system default input device, if there is a usable one.

    Args:
        sd_module: Module to query instead of the real :mod:`sounddevice`.

    Returns:
        The default input device, or ``None`` when PortAudio reports no
        default, reports one that is not a usable input, or the machine has no
        input devices at all.  A missing microphone is a state to display, not
        an exception to handle.

    Raises:
        DeviceError: If :mod:`sounddevice` cannot be imported.
    """
    for device in list_input_devices(sd_module):
        if device.is_default_input:
            return device
    return None


def find_input_device(spec: int | str | None, sd_module: Any = None) -> DeviceInfo:
    """Resolve a device specification to exactly one input device.

    Args:
        spec: A PortAudio device index, a case-insensitive substring of the
            device name, or ``None`` for the system default.
        sd_module: Module to query instead of the real :mod:`sounddevice`.

    Returns:
        The single matching :class:`DeviceInfo`.

    Raises:
        DeviceError: If ``spec`` matches no input device, if a name fragment
            matches more than one (the same microphone appears once per host
            API, so bare names are routinely ambiguous -- qualify with the
            host API or use the index), or if ``spec`` is ``None`` and the
            machine has no usable default input.  The message lists the
            available devices, because the caller almost always needs to see
            them to fix the spec.
        TypeError: If ``spec`` is neither an ``int``, a ``str``, nor ``None``.
    """
    devices = list_input_devices(sd_module)

    if spec is None:
        for device in devices:
            if device.is_default_input:
                return device
        raise DeviceError(
            "no default input device is available" + _available(devices)
        )

    if isinstance(spec, str):
        needle = spec.strip().lower()
        matches = [d for d in devices if needle in d.name.lower()]
        if not matches:
            raise DeviceError(
                f"no input device matches name {spec!r}" + _available(devices)
            )
        if len(matches) > 1:
            listed = ", ".join(f"[{d.index}] {d.label}" for d in matches)
            raise DeviceError(
                f"name {spec!r} is ambiguous: it matches {len(matches)} input "
                f"devices ({listed}); use a device index or a longer name"
                + _available(devices)
            )
        return matches[0]

    if isinstance(spec, int) and not isinstance(spec, bool):
        for device in devices:
            if device.index == spec:
                return device
        raise DeviceError(
            f"no input device with index {spec} (the index may name an "
            f"output-only device, or the device may have been unplugged)"
            + _available(devices)
        )

    raise TypeError(
        f"device spec must be an int index, a str name, or None, got "
        f"{type(spec).__name__}"
    )


def _resolve_sd(sd_module: Any = None) -> Any:
    """Return ``sd_module``, or import the real :mod:`sounddevice`.

    The import is deliberately lazy: importing :mod:`sounddevice` loads the
    PortAudio shared library, which is a real cost and a real failure mode on
    a headless machine.  Nothing in this package should pay it merely for
    importing a module that also happens to define value types.

    Args:
        sd_module: Caller-supplied stand-in, or ``None``.

    Returns:
        The module object to call PortAudio through.

    Raises:
        DeviceError: If the real module cannot be imported.
    """
    if sd_module is not None:
        return sd_module
    try:
        import sounddevice  # noqa: PLC0415 - deliberately deferred
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise DeviceError(
            "the 'sounddevice' package (PortAudio) could not be loaded, so no "
            f"audio device is available: {exc}"
        ) from exc
    return sounddevice


def _query_devices(sd: Any) -> list[Any]:
    """Return PortAudio's device table as a list, or empty if unavailable.

    Args:
        sd: Resolved ``sounddevice``-like module.

    Returns:
        One entry per device, in index order.  An empty list when PortAudio
        has no devices or the query failed -- "no devices" and "the host API
        blew up while listing them" are the same thing to a caller who only
        wants to populate a menu.
    """
    try:
        return list(sd.query_devices())
    except Exception:  # noqa: BLE001 - a broken driver must not kill the list
        return []


def _hostapi_names(sd: Any) -> tuple[str, ...]:
    """Return host API names indexed by host API index.

    Args:
        sd: Resolved ``sounddevice``-like module.

    Returns:
        Names in host-API-index order; empty if the query failed.  Entries
        that are not dict-like become ``""``, which then shows up as an
        unknown host API rather than aborting enumeration.
    """
    try:
        apis: Iterable[Any] = list(sd.query_hostapis())
    except Exception:  # noqa: BLE001 - see _query_devices
        return ()
    names: list[str] = []
    for api in apis:
        try:
            names.append(str(api["name"]))
        except Exception:  # noqa: BLE001 - one malformed API, not all of them
            names.append("")
    return tuple(names)


def _default_input_index(sd: Any) -> int | None:
    """Return PortAudio's default *input* device index, if any.

    ``sounddevice.default.device`` is an ``(input, output)`` pair -- but not a
    tuple: it is sounddevice's own ``_InputOutputPair``, which only quacks like
    a sequence.  So this subscripts anything that is not already an int, and
    accepts a bare int from a fake.  PortAudio uses ``-1`` for "no default".

    Args:
        sd: Resolved ``sounddevice``-like module.

    Returns:
        The default input index, or ``None`` when there is none.
    """
    try:
        default = sd.default.device
    except Exception:  # noqa: BLE001 - no default is a normal state
        return None

    if not isinstance(default, int):
        try:
            default = default[0]
        except Exception:  # noqa: BLE001 - not a pair; try it as a scalar
            pass
    try:
        index = int(default)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _to_device_info(
    entry: Any,
    position: int,
    hostapi_names: tuple[str, ...],
    default_index: int | None,
) -> DeviceInfo | None:
    """Convert one PortAudio device dict into a :class:`DeviceInfo`.

    Args:
        entry: Device mapping from ``query_devices()``.
        position: Position in the device table, used as the index when the
            entry omits one (older PortAudio bindings do).
        hostapi_names: Names by host API index.
        default_index: PortAudio's default input index, or ``None``.

    Returns:
        The converted record, or ``None`` if the entry is too malformed to
        interpret -- a device that cannot be described cannot be opened
        either, so dropping it is strictly better than failing enumeration.
    """
    try:
        name = str(entry["name"])
    except Exception:  # noqa: BLE001 - not a mapping, or nameless
        return None

    index = _as_int(_get(entry, "index"), position)
    hostapi_index = _as_int(_get(entry, "hostapi"), -1)
    max_input_channels = _as_int(_get(entry, "max_input_channels"), 0)

    try:
        default_samplerate = float(_get(entry, "default_samplerate"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        default_samplerate = 0.0

    if 0 <= hostapi_index < len(hostapi_names):
        hostapi_name = hostapi_names[hostapi_index]
    else:
        hostapi_name = ""

    return DeviceInfo(
        index=index,
        name=name,
        hostapi_index=hostapi_index,
        hostapi_name=hostapi_name,
        max_input_channels=max_input_channels,
        default_samplerate=default_samplerate,
        is_default_input=default_index is not None and index == default_index,
    )


def _get(entry: Any, key: str) -> Any:
    """Return ``entry[key]``, or ``None`` if it is missing or unindexable.

    Args:
        entry: Device mapping from PortAudio.
        key: Key to read.

    Returns:
        The value, or ``None``.
    """
    try:
        return entry[key]
    except Exception:  # noqa: BLE001 - missing key or non-mapping entry
        return None


def _as_int(value: Any, fallback: int) -> int:
    """Coerce ``value`` to ``int``, falling back when it is not numeric.

    Args:
        value: Value from a PortAudio device dict.
        fallback: Result when ``value`` cannot be converted.

    Returns:
        The integer value, or ``fallback``.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _available(devices: list[DeviceInfo]) -> str:
    """Render the available devices as a suffix for an error message.

    Args:
        devices: Devices to list.

    Returns:
        A newline-prefixed listing, or a note that there are none.
    """
    if not devices:
        return "; no input devices are available"
    listed = "\n".join(f"  [{d.index}] {d.label}" for d in devices)
    return f"\navailable input devices:\n{listed}"
