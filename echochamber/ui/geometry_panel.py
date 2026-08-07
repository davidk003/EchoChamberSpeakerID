"""Window / hop spin boxes and the derived overlap readout.

Overlap is *derived*, never entered: expressing geometry as window plus hop
keeps the chunk cadence exact and integral, and an overlap-percent control
would invite values that do not land on a whole number of frames.  So the
panel shows overlap and lets the user drive the two numbers it comes from.

Two rules matter here:

* ``set_geometry()`` must not re-emit ``geometry_changed``.  It is called from
  :class:`~echochamber.ui.main_window.MainWindow` with the controller's own
  values, and echoing that back as user intent is the classic Qt signal loop.
* An invalid combination (hop greater than window) is *shown*, not clamped.
  Silently correcting the user's number hides the fact that the pipeline is
  specified for overlapping windows.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

__all__ = ["GeometryPanel", "MAX_MS", "MIN_MS", "STEP_MS"]

MIN_MS: int = 10
"""Smallest window or hop the spin boxes offer, in milliseconds."""

MAX_MS: int = 30_000
"""Largest window or hop the spin boxes offer, in milliseconds."""

STEP_MS: int = 10
"""Spin box step, in milliseconds -- one device block at 16 kHz."""


class GeometryPanel(QWidget):
    """Editor for ``window_ms`` / ``hop_ms`` with a live overlap readout.

    Signals:
        geometry_changed: Emitted with ``(window_ms, hop_ms)`` on a **user**
            edit only.
    """

    geometry_changed = Signal(int, int)  # window_ms, hop_ms

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel at the pipeline's default 3000 / 1000 ms geometry.

        Args:
            parent: Parent widget, or ``None``.
        """
        super().__init__(parent)

        self._updating: bool = False

        self.window_spin: QSpinBox = _make_spin(self, 3000)
        self.window_spin.setToolTip(
            "Window length: how much audio each chunk contains. Also the "
            "dominant term in result latency."
        )
        self.hop_spin: QSpinBox = _make_spin(self, 1000)
        self.hop_spin.setToolTip(
            "Hop length: how far the window advances, and so how often a chunk "
            "arrives. Must not exceed the window."
        )

        self.overlap_label: QLabel = QLabel(self)
        self.error_label: QLabel = QLabel(self)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c03030;")

        box = QGroupBox("Window geometry", self)
        form = QFormLayout(box)
        form.addRow("window_ms", self.window_spin)
        form.addRow("hop_ms", self.hop_spin)
        form.addRow("", self.overlap_label)
        form.addRow("", self.error_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

        self.window_spin.valueChanged.connect(self._on_value_changed)
        self.hop_spin.valueChanged.connect(self._on_value_changed)
        self._refresh_derived()

    def geometry(self) -> tuple[int, int]:
        """Return the currently displayed ``(window_ms, hop_ms)``.

        Returns:
            The spin box values.
        """
        return self.window_spin.value(), self.hop_spin.value()

    def set_geometry(self, window_ms: int, hop_ms: int) -> None:
        """Display ``window_ms`` / ``hop_ms`` without emitting ``geometry_changed``.

        Args:
            window_ms: Window length in milliseconds.
            hop_ms: Hop length in milliseconds.
        """
        self._updating = True
        try:
            self.window_spin.setValue(int(window_ms))
            self.hop_spin.setValue(int(hop_ms))
        finally:
            self._updating = False
        self._refresh_derived()

    def set_overlap_text(self, text: str) -> None:
        """Set the derived-overlap label.

        Args:
            text: Text to show, e.g. ``"overlap: 2000 ms (67%)"``.
        """
        self.overlap_label.setText(text)

    def set_error(self, message: str) -> None:
        """Show (or, with an empty message, clear) the validation message.

        Args:
            message: Text to show; falsy clears the label.
        """
        self.error_label.setText(message or "")

    def _on_value_changed(self, _value: int) -> None:
        """Refresh the derived readout and emit the user's new geometry.

        Args:
            _value: The spin box's new value; unused, both are read together.
        """
        if self._updating:
            return
        self._refresh_derived()
        window_ms, hop_ms = self.geometry()
        self.geometry_changed.emit(window_ms, hop_ms)

    def _refresh_derived(self) -> None:
        """Recompute the overlap text and the local validation message."""
        window_ms, hop_ms = self.geometry()
        self.set_overlap_text(overlap_text(window_ms, hop_ms))
        if hop_ms > window_ms:
            self.set_error(
                f"hop_ms ({hop_ms}) must be <= window_ms ({window_ms}); this "
                f"pipeline is specified for overlapping windows"
            )
        else:
            self.set_error("")

    def __repr__(self) -> str:
        """Return a debugging representation of the displayed geometry."""
        window_ms, hop_ms = self.geometry()
        return f"{type(self).__name__}(window_ms={window_ms}, hop_ms={hop_ms})"


def overlap_text(window_ms: int, hop_ms: int) -> str:
    """Render the overlap derived from ``window_ms`` and ``hop_ms``.

    Args:
        window_ms: Window length in milliseconds.
        hop_ms: Hop length in milliseconds.

    Returns:
        ``"overlap: 2000 ms (67%)"`` for a valid pair, or a note that the
        combination has no overlap when the hop exceeds the window.
    """
    if window_ms <= 0 or hop_ms > window_ms:
        return "overlap: n/a (hop exceeds window)"
    overlap_ms = window_ms - hop_ms
    percent = int(round(overlap_ms * 100.0 / window_ms))
    return f"overlap: {overlap_ms} ms ({percent}%)"


def _make_spin(parent: QWidget, value: int) -> QSpinBox:
    """Create a millisecond spin box with the panel's shared range and step.

    Args:
        parent: Parent widget.
        value: Initial value in milliseconds.

    Returns:
        The configured :class:`QSpinBox`.
    """
    spin = QSpinBox(parent)
    spin.setRange(MIN_MS, MAX_MS)
    spin.setSingleStep(STEP_MS)
    spin.setSuffix(" ms")
    spin.setValue(value)
    spin.setKeyboardTracking(False)
    return spin
