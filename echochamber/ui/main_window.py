"""The window: assembly and wiring, and nothing else.

Every slot in this file is one line of forwarding.  Panels emit intent, the
controller acts on it and emits :class:`~echochamber.ui.controller.UiStats`,
the panels render that.  There is no computation here on purpose -- if a
behaviour is worth testing it belongs in
:mod:`echochamber.ui.controller`, which needs no window at all.

The controller is injectable (``MainWindow(controller=...)``) so the whole
window can be driven from a
:class:`~echochamber.audio.sources.file_source.FileSource` under
``QT_QPA_PLATFORM=offscreen``, with no microphone and no display.
"""

from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from echochamber.audio.devices import DeviceInfo
from echochamber.ui.controller import CaptureController, CaptureState, UiStats
from echochamber.ui.device_panel import DevicePanel
from echochamber.ui.geometry_panel import GeometryPanel
from echochamber.ui.meters import LevelMeter
from echochamber.ui.speaker_panel import SpeakerPanel
from echochamber.ui.stats_panel import StatsPanel
from echochamber.ui.voice_gate_panel import VoiceGatePanel

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Device panel, geometry panel, level meter and stats, around a controller."""

    def __init__(
        self,
        controller: CaptureController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Assemble the window and wire it to ``controller``.

        Args:
            controller: The controller to drive.  A default
                :class:`CaptureController` is created when ``None``; tests pass
                a file-backed one.  An injected controller is **not** reparented
                so its lifetime stays with whoever created it -- but it is
                still stopped by :meth:`closeEvent`, because a capture thread
                outliving its window is a leak either way.
            parent: Parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setWindowTitle("EchoChamber - audio ingestion")

        self._controller: CaptureController = (
            CaptureController(parent=self) if controller is None else controller
        )

        self.device_panel: DevicePanel = DevicePanel(self)
        self.geometry_panel: GeometryPanel = GeometryPanel(self)
        self.stats_panel: StatsPanel = StatsPanel(self)
        self.voice_gate_panel: VoiceGatePanel = VoiceGatePanel(self)
        self.speaker_panel: SpeakerPanel = SpeakerPanel(self)
        self.level_meter: LevelMeter = LevelMeter(self)

        meter_box = QGroupBox("Level", self)
        meter_layout = QVBoxLayout(meter_box)
        meter_layout.addWidget(self.level_meter)
        self.level_text: QLabel = QLabel("rms 0.000   peak 0.000", meter_box)
        meter_layout.addWidget(self.level_text)

        left = QVBoxLayout()
        left.addWidget(self.device_panel)
        left.addWidget(self.geometry_panel)
        left.addWidget(meter_box)
        left.addStretch(1)

        right = QVBoxLayout()
        right.addWidget(self.stats_panel)
        right.addWidget(self.voice_gate_panel)
        right.addWidget(self.speaker_panel)
        right.addStretch(1)

        central = QWidget(self)
        columns = QHBoxLayout(central)
        columns.addLayout(left, 3)
        columns.addLayout(right, 2)
        self.setCentralWidget(central)

        self.statusBar().showMessage("ready")

        self._connect()
        self._sync_from_controller()

    @property
    def controller(self) -> CaptureController:
        """The controller this window drives."""
        return self._controller

    def _connect(self) -> None:
        """Wire panel intent to the controller and controller state to panels."""
        self.device_panel.refresh_requested.connect(self._on_refresh_requested)
        self.device_panel.device_selected.connect(self._on_device_selected)
        self.device_panel.start_requested.connect(self._on_start_requested)
        self.device_panel.stop_requested.connect(self._on_stop_requested)
        self.geometry_panel.geometry_changed.connect(self._on_geometry_changed)
        self.voice_gate_panel.enabled_changed.connect(self._on_gate_enabled_changed)
        self.voice_gate_panel.phrases_changed.connect(self._on_gate_phrases_changed)
        self.speaker_panel.speaker_id_enabled_changed.connect(
            self._on_speaker_id_enabled_changed
        )
        self.speaker_panel.adb_trigger_enabled_changed.connect(
            self._on_adb_trigger_enabled_changed
        )

        self._controller.stats_updated.connect(self._on_stats_updated)
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.devices_changed.connect(self._on_devices_changed)
        self._controller.error_occurred.connect(self._on_error)

    def _sync_from_controller(self) -> None:
        """Render the controller's current state, then enumerate devices."""
        config = self._controller.config
        self.geometry_panel.set_geometry(config.window_ms, config.hop_ms)
        gate = self._controller.voice_gate_config
        self.voice_gate_panel.set_config(gate.enabled, gate.phrases)
        self.voice_gate_panel.set_editable(
            self._controller.state is not CaptureState.RUNNING
        )
        self.speaker_panel.set_speaker_config(self._controller.speaker_id_config)
        self.speaker_panel.set_adb_enabled(self._controller.adb_trigger_config.enabled)
        self.speaker_panel.set_editable(
            self._controller.state is not CaptureState.RUNNING
        )
        self.device_panel.set_state(self._controller.state)
        self.device_panel.set_devices(
            self._controller.devices, self._controller.selected_device
        )
        self._controller.refresh_devices()

    # -- panel intent ------------------------------------------------------

    def _on_refresh_requested(self) -> None:
        """Re-enumerate input devices."""
        self._controller.refresh_devices()

    def _on_device_selected(self, device: DeviceInfo | None) -> None:
        """Select ``device`` on the controller.

        Args:
            device: The device the user picked, or ``None`` for the system
                default.  A change made while capture is running only takes
                effect on the next start, which the status bar says.
        """
        self._controller.select_device(None if device is None else device.index)
        if self._controller.state is CaptureState.RUNNING:
            self.statusBar().showMessage(
                "device change takes effect after stop/start", 5000
            )

    def _on_start_requested(self) -> None:
        """Start capture and report the outcome in the status bar."""
        if self._controller.start():
            self.statusBar().showMessage("capturing", 3000)

    def _on_stop_requested(self) -> None:
        """Stop capture."""
        self._controller.stop()
        self.statusBar().showMessage("stopped", 3000)

    def _on_geometry_changed(self, window_ms: int, hop_ms: int) -> None:
        """Apply new geometry, clearing the panel's message on success.

        Args:
            window_ms: Window length in milliseconds.
            hop_ms: Hop length in milliseconds.
        """
        if self._controller.set_geometry(window_ms, hop_ms):
            self.geometry_panel.set_error("")
            self.statusBar().showMessage(
                f"geometry: window {window_ms} ms, hop {hop_ms} ms", 3000
            )

    def _on_gate_enabled_changed(self, enabled: bool) -> None:
        """Switch the wake-phrase gate on or off for the next run.

        Args:
            enabled: Whether the user ticked the box.  A rejected change is
                rolled back in the panel, so the checkbox never shows a state
                the controller did not accept.
        """
        if self._controller.set_voice_gate_enabled(enabled):
            self.voice_gate_panel.set_note(
                "takes effect on the next start" if enabled else ""
            )
        else:
            gate = self._controller.voice_gate_config
            self.voice_gate_panel.set_config(gate.enabled, gate.phrases)

    def _on_gate_phrases_changed(self, phrases: tuple[str, ...]) -> None:
        """Apply new wake phrases, rolling the field back if they are rejected.

        Args:
            phrases: The phrases the user typed.
        """
        if self._controller.set_voice_gate_phrases(phrases):
            self.voice_gate_panel.set_note("takes effect on the next start")
        else:
            gate = self._controller.voice_gate_config
            self.voice_gate_panel.set_config(gate.enabled, gate.phrases)

    def _on_speaker_id_enabled_changed(self, enabled: bool) -> None:
        """Switch speaker verification on or off for the next run.

        Args:
            enabled: Whether the user ticked the box.
        """
        self._controller.set_speaker_id_enabled(enabled)
        self.speaker_panel.set_note(
            "takes effect on the next start" if enabled else ""
        )

    def _on_adb_trigger_enabled_changed(self, enabled: bool) -> None:
        """Switch the ADB hotword trigger on or off for the next run.

        Args:
            enabled: Whether the user ticked the box.
        """
        self._controller.set_adb_trigger_enabled(enabled)
        self.speaker_panel.set_note(
            "takes effect on the next start" if enabled else ""
        )

    # -- controller state --------------------------------------------------

    def _on_stats_updated(self, stats: UiStats) -> None:
        """Render one poll tick.

        Args:
            stats: The snapshot emitted by the controller.
        """
        self.stats_panel.update_stats(stats)
        self.voice_gate_panel.update_stats(stats)
        self.speaker_panel.update_stats(stats)
        self.level_meter.set_levels(stats.rms_level, stats.display_peak)
        self.level_text.setText(
            f"rms {stats.rms_level:.3f}   peak {stats.display_peak:.3f}"
        )

    def _on_state_changed(self, state: CaptureState) -> None:
        """Reflect a capture state transition.

        Args:
            state: The controller's new state.
        """
        self.device_panel.set_state(state)
        # The recogniser is built once per run, so gate settings cannot be
        # edited into a running capture; see VoiceGatePanel's module docstring.
        self.voice_gate_panel.set_editable(state is not CaptureState.RUNNING)
        self.speaker_panel.set_editable(state is not CaptureState.RUNNING)

    def _on_devices_changed(self, devices: list[DeviceInfo]) -> None:
        """Repopulate the device picker.

        Args:
            devices: The freshly enumerated devices.
        """
        self.device_panel.set_devices(devices, self._controller.selected_device)

    def _on_error(self, message: str) -> None:
        """Surface a failure in the status bar and, for geometry, in the panel.

        Args:
            message: Human-readable failure text from the controller.
        """
        self.statusBar().showMessage(message, 10_000)
        if "window" in message or "hop" in message:
            self.geometry_panel.set_error(message)

    # -- lifecycle ---------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Stop capture before the window goes away.

        No thread may outlive the window: :meth:`CaptureController.stop` is
        bounded and never raises, so closing is always allowed to proceed.

        Args:
            event: Qt close event; always accepted.
        """
        try:
            self._controller.stop()
        finally:
            super().closeEvent(event)
            event.accept()

    def __repr__(self) -> str:
        """Return a debugging representation naming the controller state."""
        return f"{type(self).__name__}(state={self._controller.state.value})"
