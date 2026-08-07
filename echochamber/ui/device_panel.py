"""Device picker plus the Start/Stop button.

Pure intent: the panel knows how to *show* a device list and a capture state,
and it emits what the user asked for.  It never enumerates, never opens
anything, and never decides whether starting is a good idea --
:class:`~echochamber.ui.controller.CaptureController` does all of that.

The combo box shows :attr:`~echochamber.audio.devices.DeviceInfo.label` rather
than ``name``, because the bare name is genuinely ambiguous on Windows: the
same physical microphone is listed once per host API, and MME truncates names
to 31 characters, so two entries reading ``"Microphone Array (Intel Smart"``
can be different devices with different latency behaviour.  The host API is
part of the identity here.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from echochamber.audio.devices import DeviceInfo
from echochamber.ui.controller import CaptureState

__all__ = ["DevicePanel"]


class DevicePanel(QWidget):
    """Input device selection and the capture start/stop control.

    Signals:
        device_selected: Emitted with a :class:`DeviceInfo` (or ``None``) when
            the user picks a different entry.  Not emitted by
            :meth:`set_devices`.
        refresh_requested: Emitted when the user asks to re-enumerate.
        start_requested: Emitted when the user asks to start capture.
        stop_requested: Emitted when the user asks to stop capture.
    """

    device_selected = Signal(object)  # DeviceInfo | None
    refresh_requested = Signal()
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel in its stopped, empty state.

        Args:
            parent: Parent widget, or ``None``.
        """
        super().__init__(parent)

        self._updating: bool = False
        self._has_devices: bool = False
        self._state: CaptureState = CaptureState.STOPPED

        self.device_combo: QComboBox = QComboBox(self)
        self.device_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.device_combo.setToolTip(
            "Input device. The same microphone appears once per host API; "
            "WASAPI is preferred for latency."
        )
        self.device_combo.currentIndexChanged.connect(self._on_index_changed)

        self.refresh_button: QPushButton = QPushButton("Refresh", self)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)

        self.start_stop_button: QPushButton = QPushButton("Start", self)
        self.start_stop_button.clicked.connect(self._on_start_stop_clicked)

        box = QGroupBox("Input device", self)
        row = QHBoxLayout(box)
        row.addWidget(self.device_combo, 1)
        row.addWidget(self.refresh_button)
        row.addWidget(self.start_stop_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.set_state(CaptureState.STOPPED)

    @property
    def state(self) -> CaptureState:
        """The capture state last given to :meth:`set_state`."""
        return self._state

    def current_device(self) -> DeviceInfo | None:
        """Return the device currently shown in the combo box, if any.

        Returns:
            The selected :class:`DeviceInfo`, or ``None`` when the list is
            empty or the placeholder entry is showing.
        """
        data = self.device_combo.currentData()
        return data if isinstance(data, DeviceInfo) else None

    def set_devices(
        self, devices: list[DeviceInfo], selected: DeviceInfo | None
    ) -> None:
        """Repopulate the combo box without emitting ``device_selected``.

        Repopulating a ``QComboBox`` fires ``currentIndexChanged`` several
        times as items come and go; re-emitting that as user intent would make
        a refresh look like a device change and, wired back to the controller,
        would fight the selection it just computed.

        Args:
            devices: Devices to offer, in the order they should appear.
            selected: The device to show as current, or ``None``.
        """
        self._updating = True
        try:
            self.device_combo.clear()
            self._has_devices = bool(devices)
            if not devices:
                self.device_combo.addItem("no input devices found", None)
            else:
                for device in devices:
                    self.device_combo.addItem(device.label, device)
                index = _index_of(devices, selected)
                if index >= 0:
                    self.device_combo.setCurrentIndex(index)
        finally:
            self._updating = False
        self._update_buttons()

    def set_state(self, state: CaptureState) -> None:
        """Reflect ``state`` in the button text and what stays enabled.

        The device and the sample rate are fixed when the pipeline is
        constructed, so the picker and Refresh are disabled while running
        rather than silently having no effect until the next start.

        Args:
            state: The controller's current :class:`CaptureState`.
        """
        self._state = state
        self._update_buttons()

    def _update_buttons(self) -> None:
        """Sync the button text and the enabled states to the current state."""
        running = self._state is CaptureState.RUNNING
        self.start_stop_button.setText("Stop" if running else "Start")
        # Always enabled: starting with nothing listed is still legitimate --
        # the controller falls back to the system default device.
        self.start_stop_button.setEnabled(True)
        self.refresh_button.setEnabled(not running)
        self.device_combo.setEnabled(not running and self._has_devices)

    def _on_index_changed(self, index: int) -> None:
        """Emit ``device_selected`` for a user-driven combo box change.

        Args:
            index: New combo box index; unused, the payload is the item data.
        """
        if self._updating:
            return
        self.device_selected.emit(self.current_device())

    def _on_start_stop_clicked(self) -> None:
        """Emit ``start_requested`` or ``stop_requested`` for the current state."""
        if self._state is CaptureState.RUNNING:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def __repr__(self) -> str:
        """Return a debugging representation of the panel's state."""
        device = self.current_device()
        return (
            f"{type(self).__name__}(state={self._state.value}, "
            f"devices={self.device_combo.count()}, "
            f"selected={None if device is None else device.label!r})"
        )


def _index_of(devices: list[DeviceInfo], selected: DeviceInfo | None) -> int:
    """Return the position of ``selected`` in ``devices``, or ``-1``.

    Matched on index *and* label rather than object identity: the controller
    re-enumerates, so the equal device is a different object.

    Args:
        devices: The devices in combo box order.
        selected: The device to locate, or ``None``.

    Returns:
        The zero-based position, or ``-1`` if it is not there.
    """
    if selected is None:
        return -1
    for position, device in enumerate(devices):
        if device.index == selected.index and device.label == selected.label:
            return position
    return -1
