"""Read-only counters for the running stream.

The one design point: **dropped chunks, overruns and xruns are emphasised the
moment they are non-zero.**  A pipeline that is quietly discarding audio while
every other number keeps climbing is the single most common failure mode in a
system like this, and it looks perfectly healthy unless the display says
otherwise.

Everything here is rendering.  The values arrive already computed in a
:class:`~echochamber.ui.controller.UiStats`.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from echochamber.ui.controller import UiStats

__all__ = ["StatsPanel"]

_FAULT_STYLE: str = "color: #c03030; font-weight: bold;"
"""Applied to a fault counter that is non-zero; cleared when it is back to 0."""

_ROWS: tuple[tuple[str, str], ...] = (
    ("state", "state"),
    ("elapsed", "elapsed"),
    ("frames captured", "frames_captured"),
    ("chunks emitted", "chunks_emitted"),
    ("chunks dropped", "chunks_dropped"),
    ("overruns", "overruns"),
    ("xruns", "xruns"),
    ("device latency", "latency_ms"),
)
"""``(label, key)`` in display order; ``key`` also names the value widget."""

_FAULT_KEYS: frozenset[str] = frozenset({"chunks_dropped", "overruns", "xruns"})
"""Counters that mean something is wrong as soon as they are non-zero."""


class StatsPanel(QWidget):
    """A read-only grid of stream counters, refreshed on every timer tick."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the grid with every counter showing its zero state.

        Args:
            parent: Parent widget, or ``None``.
        """
        super().__init__(parent)

        self._values: dict[str, QLabel] = {}

        box = QGroupBox("Stream", self)
        grid = QGridLayout(box)
        for row, (caption, key) in enumerate(_ROWS):
            name = QLabel(f"{caption}:", box)
            value = QLabel("-", box)
            value.setObjectName(f"value_{key}")
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._values[key] = value
        grid.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(box)

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

    def update_stats(self, stats: UiStats) -> None:
        """Render ``stats``, emphasising any non-zero fault counter.

        Args:
            stats: The snapshot from
                :meth:`~echochamber.ui.controller.CaptureController.poll`.
        """
        rendered: dict[str, str] = {
            "state": stats.state.value,
            "elapsed": f"{stats.elapsed_s:.1f} s",
            "frames_captured": f"{stats.frames_captured:,}",
            "chunks_emitted": f"{stats.chunks_emitted:,}",
            "chunks_dropped": f"{stats.chunks_dropped:,}",
            "overruns": f"{stats.overruns:,}",
            "xruns": f"{stats.xruns:,}",
            "latency_ms": f"{stats.latency_ms:.1f} ms",
        }
        faults: dict[str, int] = {
            "chunks_dropped": stats.chunks_dropped,
            "overruns": stats.overruns,
            "xruns": stats.xruns,
        }
        for key, text in rendered.items():
            label = self._values[key]
            label.setText(text)
            if key in _FAULT_KEYS:
                label.setStyleSheet(_FAULT_STYLE if faults[key] else "")

    def __repr__(self) -> str:
        """Return a debugging representation naming the displayed state."""
        return f"{type(self).__name__}(state={self._values['state'].text()!r})"
