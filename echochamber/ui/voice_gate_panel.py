"""Wake-phrase gate controls and counters.

Two halves in one box, for the same reason the device panel holds both a
picker and a start button: the settings and the evidence that they are working
are read together, and separating them would mean watching one panel to find
out whether the other is doing anything.

**Settings here take effect on the next start, and the panel says so.**  The
gate's recogniser is built once per run -- for the subprocess backend, in
another process with a model already loaded -- so unlike window geometry there
is no live reconfiguration to offer.  A control that silently did nothing until
a restart would be worse than one that admits it, so the panel disables its
inputs while capture runs rather than accepting edits that go nowhere.

**``snippets`` is the number the user is actually watching.**  It is the one
that answers "did it hear me", so it gets emphasised when it moves.
``suppressed`` and ``truncated`` are shown because a gate that detects phrases
and writes no files looks identical to one that hears nothing unless the
display distinguishes them -- the same argument
:mod:`echochamber.ui.stats_panel` makes for dropped chunks.

Everything here is rendering and intent.  The values arrive already computed in
a :class:`~echochamber.ui.controller.UiStats`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from echochamber.ui.controller import UiStats

__all__ = ["PHRASE_SEPARATOR", "VoiceGatePanel", "parse_phrases", "format_phrases"]

PHRASE_SEPARATOR: str = ","
"""What separates phrases in the text field.

A comma rather than whitespace, because wake phrases contain spaces -- "ok
google" is one phrase, not two -- and a space-separated field could not express
that at all.
"""

_HIT_STYLE: str = "color: #2e7d32; font-weight: bold;"
"""Applied to the snippet counter once it is non-zero: this is the good news."""

_WARN_STYLE: str = "color: #c03030; font-weight: bold;"
"""Applied to the gate's error line, which is only ever bad news."""

_ROWS: tuple[tuple[str, str], ...] = (
    ("backend", "gate_backend"),
    ("phrases detected", "gate_detected"),
    ("snippets", "gate_snippets"),
    ("suppressed", "gate_suppressed"),
    ("truncated", "gate_truncated"),
    ("last phrase", "gate_last_phrase"),
    ("notify", "notify_state"),
    ("events sent", "notify_sent"),
)
"""``(label, key)`` in display order; ``key`` also names the value widget."""


def parse_phrases(text: str) -> tuple[str, ...]:
    """Split a comma-separated phrase list into individual phrases.

    Args:
        text: Raw field contents.

    Returns:
        The phrases, stripped, with empties dropped -- so a trailing comma or a
        double comma is tolerated rather than producing an empty phrase, which
        :class:`~echochamber.voicegate.config.VoiceGateConfig` would reject and
        which no user ever means.
    """
    return tuple(
        part.strip() for part in text.split(PHRASE_SEPARATOR) if part.strip()
    )


def format_phrases(phrases: tuple[str, ...]) -> str:
    """Render phrases for the text field.

    Args:
        phrases: The configured phrases.

    Returns:
        The phrases joined by ``", "``, which :func:`parse_phrases` reads back
        unchanged.
    """
    return f"{PHRASE_SEPARATOR} ".join(phrases)


def _notify_state_text(stats: UiStats) -> str:
    """Render the notifier's connection state in one word or two.

    Args:
        stats: The current snapshot.

    Returns:
        ``"off"`` when notifications are disabled, ``"connected"`` when the
        socket is open, ``"connecting"`` otherwise.  "connecting" rather than
        "disconnected" because the sender retries with backoff forever, so a
        down listener is a state it is working through, not a terminal one.
    """
    if not stats.notify_enabled:
        return "off"
    return "connected" if stats.notify_connected else "connecting"


def _notify_sent_text(stats: UiStats) -> str:
    """Render the events-sent counter, naming drops and backlog when non-zero.

    Args:
        stats: The current snapshot.

    Returns:
        ``"12"``, or ``"12 (2 dropped)"``, or ``"12 (3 queued)"``.  A bare count
        would let a notifier that is silently discarding every event look
        identical to one that has simply heard nothing yet.
    """
    if not stats.notify_enabled:
        return "—"
    text = f"{stats.notify_sent:,}"
    extra: list[str] = []
    if stats.notify_dropped:
        extra.append(f"{stats.notify_dropped:,} dropped")
    if stats.notify_queued:
        extra.append(f"{stats.notify_queued:,} queued")
    if extra:
        text = f"{text} ({', '.join(extra)})"
    return text


class VoiceGatePanel(QWidget):
    """Enable switch, phrase list, and the gate's counters.

    Signals:
        enabled_changed: Emitted with the new ``bool`` on a **user** toggle.
        phrases_changed: Emitted with a ``tuple[str, ...]`` when the user
            finishes editing the phrase field.
    """

    enabled_changed = Signal(bool)
    phrases_changed = Signal(object)  # tuple[str, ...]

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel with the gate off and the default phrases shown.

        Args:
            parent: Parent widget, or ``None``.
        """
        super().__init__(parent)

        self._updating: bool = False

        self.enable_check: QCheckBox = QCheckBox("Enable wake-phrase gate", self)
        self.enable_check.setToolTip(
            "Record a snippet only when a wake phrase is recognised. Takes "
            "effect on the next start."
        )

        self.phrase_edit: QLineEdit = QLineEdit(self)
        self.phrase_edit.setPlaceholderText("ok google, hey google")
        self.phrase_edit.setToolTip(
            "Comma-separated wake phrases. Matched on whole words, ignoring "
            "case and punctuation."
        )

        self.note_label: QLabel = QLabel(self)
        self.note_label.setWordWrap(True)

        self._values: dict[str, QLabel] = {}

        box = QGroupBox("Voice gate", self)
        form = QFormLayout(box)
        form.addRow(self.enable_check)
        form.addRow("phrases", self.phrase_edit)
        for caption, key in _ROWS:
            value = QLabel("-", box)
            value.setObjectName(f"value_{key}")
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            form.addRow(f"{caption}:", value)
            self._values[key] = value
        form.addRow("", self.note_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.enable_check.toggled.connect(self._on_toggled)
        self.phrase_edit.editingFinished.connect(self._on_phrases_edited)

    def value_label(self, key: str) -> QLabel:
        """Return the label rendering ``key``.

        Args:
            key: One of the :class:`UiStats` field names in the grid.

        Returns:
            The :class:`QLabel` showing that value.

        Raises:
            KeyError: If ``key`` is not displayed by this panel.
        """
        return self._values[key]

    def phrases(self) -> tuple[str, ...]:
        """Return the phrases currently shown in the field.

        Returns:
            The parsed phrases.
        """
        return parse_phrases(self.phrase_edit.text())

    def is_enabled(self) -> bool:
        """Return whether the enable box is ticked."""
        return self.enable_check.isChecked()

    def set_config(self, enabled: bool, phrases: tuple[str, ...]) -> None:
        """Display a configuration **without** emitting either signal.

        Called with the controller's own values; echoing them back as user
        intent is the Qt signal loop
        :class:`~echochamber.ui.geometry_panel.GeometryPanel` documents.

        Args:
            enabled: Whether the gate is on.
            phrases: The configured phrases.
        """
        self._updating = True
        try:
            self.enable_check.setChecked(bool(enabled))
            self.phrase_edit.setText(format_phrases(phrases))
        finally:
            self._updating = False

    def set_editable(self, editable: bool) -> None:
        """Enable or disable the inputs.

        Args:
            editable: ``False`` while capture runs, because the recogniser is
                built once per run and an edit could not take effect until the
                next one.  See the module docstring.
        """
        self.enable_check.setEnabled(editable)
        self.phrase_edit.setEnabled(editable)

    def set_note(self, message: str, warning: bool = False) -> None:
        """Show (or, with an empty message, clear) the note under the counters.

        Args:
            message: Text to show; falsy clears the label.
            warning: ``True`` to render it as a failure.
        """
        self.note_label.setText(message or "")
        self.note_label.setStyleSheet(_WARN_STYLE if warning and message else "")

    def update_stats(self, stats: UiStats) -> None:
        """Render one poll tick.

        Args:
            stats: The snapshot from
                :meth:`~echochamber.ui.controller.CaptureController.poll`.
        """
        rendered: dict[str, str] = {
            "gate_backend": stats.gate_backend,
            "gate_detected": f"{stats.gate_detected:,}",
            "gate_snippets": f"{stats.gate_snippets:,}",
            "gate_suppressed": f"{stats.gate_suppressed:,}",
            "gate_truncated": f"{stats.gate_truncated:,}",
            # An em dash rather than an empty cell: a blank next to a label
            # reads as a rendering failure, not as "nothing yet".
            "gate_last_phrase": stats.gate_last_phrase or "—",
            "notify_state": _notify_state_text(stats),
            "notify_sent": _notify_sent_text(stats),
        }
        for key, text in rendered.items():
            self._values[key].setText(text)

        self._values["gate_snippets"].setStyleSheet(
            _HIT_STYLE if stats.gate_snippets else ""
        )
        # A notifier that is on but disconnected, or dropping events, is the
        # failure this row exists to make visible: events are being generated
        # and nobody is receiving them.
        broken = stats.notify_enabled and (
            not stats.notify_connected or stats.notify_dropped
        )
        self._values["notify_state"].setStyleSheet(_WARN_STYLE if broken else "")
        self._values["notify_sent"].setStyleSheet(
            _WARN_STYLE if stats.notify_dropped else ""
        )

        if stats.gate_error:
            self.set_note(stats.gate_error, warning=True)
        elif stats.notify_error:
            self.set_note(f"notify: {stats.notify_error}", warning=True)

    def _on_toggled(self, checked: bool) -> None:
        """Emit the user's new enable state.

        Args:
            checked: The checkbox's new state.
        """
        if self._updating:
            return
        self.enabled_changed.emit(bool(checked))

    def _on_phrases_edited(self) -> None:
        """Emit the user's new phrase list once editing has finished.

        ``editingFinished`` rather than ``textChanged``: the latter fires on
        every keystroke, so ``"ok google"`` would be validated as ``"o"``,
        ``"ok"``, ``"ok "`` and so on, and each intermediate value that failed
        validation would put an error in the status bar while the user was
        still typing.
        """
        if self._updating:
            return
        self.phrases_changed.emit(self.phrases())

    def __repr__(self) -> str:
        """Return a debugging representation of the panel's state."""
        return (
            f"{type(self).__name__}(enabled={self.is_enabled()}, "
            f"phrases={len(self.phrases())})"
        )
