"""Speaker enrollment and the ADB hotword trigger, in one panel.

Neither feature had a GUI before this: turning speaker verification or the
ADB trigger on meant editing :mod:`echochamber.ui.controller`'s own defaults,
and enrolling a voice meant running ``scripts/enroll_speaker.py`` from a
terminal. This panel is that missing control surface, following
:class:`~echochamber.ui.voice_gate_panel.VoiceGatePanel`'s own shape: intent
out via signals, state in via :meth:`update_stats`/``set_config``, nothing
computed here that :mod:`~echochamber.ui.controller` doesn't already own.

**Enrolling from a WAV file is the primary path, a live recording the
fallback.** A file already recorded -- on a phone, in another app, ahead of
time -- needs no microphone permission from this process, no "say something
for five seconds" awkwardness while the window has focus, and can be redone
by picking a different file instead of re-recording. The live-record button
stays, one section down, for when there is nothing to point at yet.

**Both checkboxes only matter while the wake-phrase gate is also on.**
Speaker verification and the ADB trigger both run inside
:meth:`~echochamber.ui.controller.CaptureController._build_gate`, so ticking
either with the gate off leaves the setting dormant rather than doing
nothing -- the panel says so rather than pretending they are independent.

Recording and embedding both block for real time (seconds of mic capture,
then a subprocess round trip through the QNN chain), so both run on a
background :class:`QThread` -- freezing the window for that long would look
identical to a hang.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from echochamber.speakerid.backends import build_embedder
from echochamber.speakerid.config import SpeakerIdConfig
from echochamber.speakerid.enrollment import (
    BACKEND_QNN,
    enroll,
    load_db,
    load_wav_mono,
    remove_speaker,
    save_db,
)
from echochamber.ui.controller import UiStats

__all__ = ["SpeakerPanel"]

_WARN_STYLE: str = "color: #c03030; font-weight: bold;"
"""Applied to error text -- the same shade :mod:`voice_gate_panel` uses."""

_OK_STYLE: str = "color: #2e7d32; font-weight: bold;"
"""Applied to a success message."""

_MIC_SAMPLE_RATE: int = 16_000
"""Recording rate for the live-record fallback; matches ``scripts/enroll_speaker.py``."""


class _EnrollWorker(QThread):
    """Records or reads a clip, embeds it, and writes it to the database.

    Runs entirely off the GUI thread: recording blocks for
    ``seconds`` wall-clock seconds, and starting the QNN subprocess chain the
    first time can take several more. Emits exactly one of
    :attr:`succeeded`/:attr:`failed` and then finishes.
    """

    succeeded = Signal(str)  # message
    failed = Signal(str)  # message

    def __init__(
        self,
        config: SpeakerIdConfig,
        name: str,
        wav_path: str | None,
        seconds: float,
        parent: QWidget | None = None,
    ) -> None:
        """Prepare a worker; nothing runs until :meth:`start`.

        Args:
            config: Where to embed and enroll -- ``qnn_onnx_path``,
                ``qnn_worker_python`` and ``db_path``.
            name: Speaker name to enroll under.
            wav_path: A WAV file to read, or ``None`` to record from the
                default microphone instead.
            seconds: Recording length, used only when ``wav_path`` is
                ``None``.
            parent: Parent object, or ``None``.
        """
        super().__init__(parent)
        self._config = config
        self._name = name
        self._wav_path = wav_path
        self._seconds = seconds

    def run(self) -> None:  # noqa: D102 - QThread override, see class docstring
        try:
            if self._wav_path is not None:
                samples, sample_rate = load_wav_mono(self._wav_path)
            else:
                samples, sample_rate = self._record(), _MIC_SAMPLE_RATE

            embedder = build_embedder(self._config)
            try:
                embedding = embedder.embed(samples, sample_rate)
            finally:
                embedder.close()

            db = load_db(self._config.db_path)
            enroll(db, self._name, embedding, BACKEND_QNN)
            save_db(self._config.db_path, db)
        except Exception as exc:  # noqa: BLE001 - reported to the GUI, not raised
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.succeeded.emit(f"Enrolled {self._name!r} in {self._config.db_path!r}.")

    def _record(self):
        """Record ``self._seconds`` of mono audio from the default microphone."""
        import sounddevice as sd  # noqa: PLC0415 - only needed for the live-record path

        audio = sd.rec(
            int(self._seconds * _MIC_SAMPLE_RATE),
            samplerate=_MIC_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio[:, 0]


class SpeakerPanel(QWidget):
    """Enable switches for speaker ID and the ADB trigger, plus enrollment.

    Signals:
        speaker_id_enabled_changed: Emitted with the new ``bool`` on a user
            toggle of the speaker-verification checkbox.
        adb_trigger_enabled_changed: Emitted with the new ``bool`` on a user
            toggle of the ADB-trigger checkbox.
    """

    speaker_id_enabled_changed = Signal(bool)
    adb_trigger_enabled_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel with both features off and nothing enrolled yet.

        Args:
            parent: Parent widget, or ``None``.
        """
        super().__init__(parent)

        self._updating: bool = False
        self._config: SpeakerIdConfig = SpeakerIdConfig()
        self._worker: _EnrollWorker | None = None
        self._wav_path: str | None = None

        # -- enable switches --------------------------------------------
        self.speaker_check: QCheckBox = QCheckBox(
            "Require my voice to open a snippet", self
        )
        self.speaker_check.setToolTip(
            "Only takes effect while the wake-phrase gate is also on. Takes "
            "effect on the next start."
        )
        self.adb_check: QCheckBox = QCheckBox(
            "Block \u201cOk Google\u201d for anyone else", self
        )
        self.adb_check.setToolTip(
            "Revokes the Google app's microphone permission via adb whenever "
            "an unrecognised voice says the wake phrase, and restores it "
            "when your own voice is heard. Only takes effect while both the "
            "wake-phrase gate and speaker verification above are on."
        )

        enable_box = QGroupBox("Turn on", self)
        enable_layout = QVBoxLayout(enable_box)
        enable_layout.addWidget(self.speaker_check)
        enable_layout.addWidget(self.adb_check)

        # -- enrollment: WAV file first, live recording second -----------
        self.name_edit: QLineEdit = QLineEdit(self)
        self.name_edit.setPlaceholderText("your name")

        self.wav_label: QLabel = QLabel("No file chosen.", self)
        self.wav_label.setWordWrap(True)
        self.wav_button: QPushButton = QPushButton("Choose WAV file...", self)
        self.enroll_wav_button: QPushButton = QPushButton("Enroll from file", self)
        self.enroll_wav_button.setEnabled(False)

        wav_row = QHBoxLayout()
        wav_row.addWidget(self.wav_button)
        wav_row.addWidget(self.enroll_wav_button)

        self.seconds_spin: QDoubleSpinBox = QDoubleSpinBox(self)
        self.seconds_spin.setRange(1.0, 30.0)
        self.seconds_spin.setValue(5.0)
        self.seconds_spin.setSuffix(" s")
        self.record_button: QPushButton = QPushButton("Record and enroll", self)

        record_row = QHBoxLayout()
        record_row.addWidget(self.seconds_spin)
        record_row.addWidget(self.record_button)

        self.status_label: QLabel = QLabel(self)
        self.status_label.setWordWrap(True)

        self.speaker_list: QListWidget = QListWidget(self)
        self.speaker_list.setToolTip("Everyone currently enrolled.")
        self.remove_button: QPushButton = QPushButton("Remove selected", self)

        enroll_box = QGroupBox("Enroll a voice", self)
        enroll_form = QFormLayout(enroll_box)
        enroll_form.addRow("name", self.name_edit)
        enroll_form.addRow(QLabel("From a WAV file (recommended):", self))
        enroll_form.addRow(self.wav_label)
        enroll_form.addRow(wav_row)
        enroll_form.addRow(QLabel("Or record live instead:", self))
        enroll_form.addRow(record_row)
        enroll_form.addRow(self.status_label)
        enroll_form.addRow(QLabel("Enrolled:", self))
        enroll_form.addRow(self.speaker_list)
        enroll_form.addRow(self.remove_button)

        self.note_label: QLabel = QLabel(self)
        self.note_label.setWordWrap(True)

        self._values: dict[str, QLabel] = {}
        status_box = QGroupBox("Status", self)
        status_form = QFormLayout(status_box)
        for caption, key in (
            ("speaker backend", "speaker_id_backend"),
            ("verified", "speaker_id_verified"),
            ("rejected", "speaker_id_rejected"),
            ("last speaker", "speaker_id_last_name"),
            ("adb backend", "adb_trigger_backend"),
            ("hotword", "adb_trigger_state"),
            ("blocks issued", "adb_trigger_blocks"),
        ):
            value = QLabel("-", status_box)
            value.setObjectName(f"value_{key}")
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            status_form.addRow(f"{caption}:", value)
            self._values[key] = value
        status_form.addRow("", self.note_label)

        box = QGroupBox("Speaker ID && ADB hotword trigger", self)
        outer = QVBoxLayout(box)
        outer.addWidget(enable_box)
        outer.addWidget(enroll_box)
        outer.addWidget(status_box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.speaker_check.toggled.connect(self._on_speaker_toggled)
        self.adb_check.toggled.connect(self._on_adb_toggled)
        self.wav_button.clicked.connect(self._on_choose_wav)
        self.enroll_wav_button.clicked.connect(self._on_enroll_from_wav)
        self.record_button.clicked.connect(self._on_record_and_enroll)
        self.remove_button.clicked.connect(self._on_remove_selected)

    # -- wiring from the controller -----------------------------------

    def set_speaker_config(self, config: SpeakerIdConfig) -> None:
        """Adopt ``config`` for enrollment actions and reflect its ``enabled``.

        Called with the controller's own configuration; does not emit
        either enable signal, for the same reason
        :meth:`~echochamber.ui.voice_gate_panel.VoiceGatePanel.set_config`
        doesn't.

        Args:
            config: The controller's current speaker-ID configuration.
        """
        self._config = config
        self._updating = True
        try:
            self.speaker_check.setChecked(bool(config.enabled))
        finally:
            self._updating = False
        self.refresh_enrolled()

    def set_adb_enabled(self, enabled: bool) -> None:
        """Reflect the controller's ADB-trigger ``enabled`` without emitting.

        Args:
            enabled: Whether the trigger is switched on for the next run.
        """
        self._updating = True
        try:
            self.adb_check.setChecked(bool(enabled))
        finally:
            self._updating = False

    def set_editable(self, editable: bool) -> None:
        """Enable or disable every input.

        Args:
            editable: ``False`` while capture runs; both features are built
                once per run, exactly like the wake-phrase gate itself.
        """
        for widget in (
            self.speaker_check,
            self.adb_check,
            self.name_edit,
            self.wav_button,
            self.enroll_wav_button,
            self.record_button,
            self.remove_button,
        ):
            widget.setEnabled(editable)
        # Re-apply the file-chosen gate rather than force enrollment on:
        # editable alone must not let "enroll from file" light up with no
        # file picked.
        if editable:
            self.enroll_wav_button.setEnabled(self._wav_path is not None)

    def refresh_enrolled(self) -> None:
        """Reload the enrolled-speaker list from disk."""
        self.speaker_list.clear()
        try:
            db = load_db(self._config.db_path)
        except Exception:  # noqa: BLE001 - a display refresh must not raise
            return
        for name in sorted(db):
            self.speaker_list.addItem(name)

    def update_stats(self, stats: UiStats) -> None:
        """Render one poll tick.

        Args:
            stats: The snapshot from
                :meth:`~echochamber.ui.controller.CaptureController.poll`.
        """
        rendered: dict[str, str] = {
            "speaker_id_backend": stats.speaker_id_backend,
            "speaker_id_verified": f"{stats.speaker_verified:,}",
            "speaker_id_rejected": f"{stats.speaker_rejected:,}",
            "speaker_id_last_name": stats.speaker_last_name or "\u2014",
            "adb_trigger_backend": stats.adb_trigger_backend,
            "adb_trigger_state": "blocked" if stats.adb_trigger_blocked else "open",
            "adb_trigger_blocks": f"{stats.adb_trigger_block_count:,}",
        }
        for key, text in rendered.items():
            self._values[key].setText(text)

        self._values["adb_trigger_state"].setStyleSheet(
            _WARN_STYLE if stats.adb_trigger_blocked else ""
        )

        if stats.speaker_id_error:
            self.set_note(stats.speaker_id_error, warning=True)
        elif stats.adb_trigger_error:
            self.set_note(f"adb: {stats.adb_trigger_error}", warning=True)

    def set_note(self, message: str, warning: bool = False) -> None:
        """Show (or, with an empty message, clear) the note under the status grid.

        Args:
            message: Text to show; falsy clears the label.
            warning: ``True`` to render it as a failure.
        """
        self.note_label.setText(message or "")
        self.note_label.setStyleSheet(_WARN_STYLE if warning and message else "")

    # -- user intent ----------------------------------------------------

    def _on_speaker_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self.speaker_id_enabled_changed.emit(bool(checked))

    def _on_adb_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self.adb_trigger_enabled_changed.emit(bool(checked))

    def _on_choose_wav(self) -> None:
        """Open a file picker for the enrollment clip."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a WAV file", "", "WAV files (*.wav)"
        )
        if not path:
            return
        self._wav_path = path
        self.wav_label.setText(path)
        self.enroll_wav_button.setEnabled(True)

    def _on_enroll_from_wav(self) -> None:
        """Enroll the current name from the chosen WAV file."""
        self._start_enroll(wav_path=self._wav_path, seconds=0.0)

    def _on_record_and_enroll(self) -> None:
        """Enroll the current name from a fresh live recording."""
        self._start_enroll(wav_path=None, seconds=self.seconds_spin.value())

    def _start_enroll(self, wav_path: str | None, seconds: float) -> None:
        """Validate inputs and launch a background :class:`_EnrollWorker`.

        Args:
            wav_path: A file to read, or ``None`` to record instead.
            seconds: Recording length; ignored when ``wav_path`` is given.
        """
        if self._worker is not None and self._worker.isRunning():
            return
        name = self.name_edit.text().strip()
        if not name:
            self.set_note("enter a name first", warning=True)
            return
        if not self._config.qnn_onnx_path or not self._config.qnn_worker_python:
            self.set_note(
                "speaker-ID model not set up yet -- run "
                "`python scripts/setup_speakerid_qnn.py` and "
                "`python scripts/export_speakerid_qnn.py` first",
                warning=True,
            )
            return

        self._set_busy(True)
        source = "file" if wav_path is not None else "the microphone"
        self.set_note(f"Enrolling {name!r} from {source}...")

        worker = _EnrollWorker(self._config, name, wav_path, seconds, self)
        worker.succeeded.connect(self._on_enroll_succeeded)
        worker.failed.connect(self._on_enroll_failed)
        worker.finished.connect(lambda: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _on_enroll_succeeded(self, message: str) -> None:
        self.note_label.setStyleSheet(_OK_STYLE)
        self.note_label.setText(message)
        self.refresh_enrolled()

    def _on_enroll_failed(self, message: str) -> None:
        self.set_note(message, warning=True)

    def _set_busy(self, busy: bool) -> None:
        """Disable enrollment controls while a worker is running.

        Args:
            busy: ``True`` while a recording/embedding is in flight.
        """
        self.enroll_wav_button.setEnabled(not busy and self._wav_path is not None)
        self.wav_button.setEnabled(not busy)
        self.record_button.setEnabled(not busy)
        self.name_edit.setEnabled(not busy)

    def _on_remove_selected(self) -> None:
        """Remove the selected enrolled speaker from the database."""
        item = self.speaker_list.currentItem()
        if item is None:
            self.set_note("select a speaker to remove first", warning=True)
            return
        name = item.text()
        db = load_db(self._config.db_path)
        if remove_speaker(db, name):
            save_db(self._config.db_path, db)
            self.set_note(f"Removed {name!r}.")
            self.refresh_enrolled()

    def __repr__(self) -> str:
        """Return a debugging representation of the panel's state."""
        return (
            f"{type(self).__name__}(speaker={self.speaker_check.isChecked()}, "
            f"adb={self.adb_check.isChecked()}, enrolled={self.speaker_list.count()})"
        )
