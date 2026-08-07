"""Level metering: the peak-hold rule, and the widget that draws it.

:class:`PeakHold` is pure logic and imports no Qt, because it is the part with
behaviour worth testing.  It exists because
:attr:`~echochamber.audio.types.StreamStats.peak_level` is the **most recent
block only**, not a running maximum: a meter that renders it raw reads ~0
whenever the last 10 ms happened to be quiet, including at shutdown right after
a loud session.  Holding the maximum for a moment and then decaying it is what
makes a level meter readable at 30 Hz.

:class:`PeakHold.update` takes ``now`` from the caller and never reads the
clock itself.  That is not ceremony: it makes hold and decay exactly
assertable, with no sleeping in a test.
"""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["LevelMeter", "PeakHold"]


class PeakHold:
    """Peak-hold with exponential decay, in dB per second.

    The rule, in order:

    * a level **above** the current value takes it immediately and restarts
      the hold window;
    * inside the hold window the value does not move;
    * after the hold window the value decays by ``decay_db_per_s`` dB per
      second of elapsed time, and never goes below ``0.0``.

    A steady tone therefore reads steady: it decays a little past the hold
    window, and the next block -- now above the decayed value -- takes it back.

    No Qt, no clock.  ``now`` is supplied by the caller (the GUI timer passes
    :func:`time.monotonic`), which is what makes this trivially testable.
    """

    __slots__ = ("_hold_s", "_decay_db_per_s", "_value", "_hold_until", "_last_time")

    def __init__(self, hold_s: float = 1.0, decay_db_per_s: float = 20.0) -> None:
        """Configure the hold window and the decay rate.

        Args:
            hold_s: Seconds a new maximum is held before it starts to decay.
                ``0.0`` means decay immediately.
            decay_db_per_s: Decay rate in dB per second once the hold expires.
                ``0.0`` means hold forever.

        Raises:
            ValueError: If ``hold_s`` or ``decay_db_per_s`` is negative.
        """
        hold_s = float(hold_s)
        decay_db_per_s = float(decay_db_per_s)
        if hold_s < 0.0:
            raise ValueError(f"hold_s must be >= 0, got {hold_s}")
        if decay_db_per_s < 0.0:
            raise ValueError(f"decay_db_per_s must be >= 0, got {decay_db_per_s}")

        self._hold_s: float = hold_s
        self._decay_db_per_s: float = decay_db_per_s
        self._value: float = 0.0
        self._hold_until: float | None = None
        self._last_time: float | None = None

    @property
    def hold_s(self) -> float:
        """Seconds a new maximum is held before decaying."""
        return self._hold_s

    @property
    def decay_db_per_s(self) -> float:
        """Decay rate in dB per second, once the hold window has passed."""
        return self._decay_db_per_s

    @property
    def value(self) -> float:
        """Current held level, ``0.0``-``1.0``-ish (never negative)."""
        return self._value

    def update(self, level: float, now: float) -> float:
        """Fold one observed ``level`` in at time ``now`` and return the value.

        Args:
            level: Observed peak level for this tick.  Negative values (and
                ``NaN``) are treated as ``0.0``; nothing upstream produces them,
                but a meter must not be the thing that crashes.
            now: Monotonic timestamp, in seconds, supplied by the caller.  Time
                going backwards is treated as no time passing at all rather
                than as negative decay.

        Returns:
            The value after this update, identical to :attr:`value`.
        """
        level = float(level)
        # NaN fails every comparison, so test for it by identity with itself.
        if not level > 0.0 or level != level:
            level = 0.0
        now = float(now)

        if self._last_time is None:
            self._last_time = now

        if level > self._value:
            self._value = level
            self._hold_until = now + self._hold_s
            self._last_time = now
            return self._value

        hold_until = self._hold_until
        if hold_until is not None and now > hold_until and self._decay_db_per_s > 0.0:
            # Decay from wherever the last accounted-for moment was -- the end
            # of the hold window, or the previous update, whichever is later --
            # so an irregular tick rate cannot double-count or skip decay.
            since = max(hold_until, self._last_time)
            elapsed = now - since
            if elapsed > 0.0:
                self._value *= 10.0 ** (-self._decay_db_per_s * elapsed / 20.0)
                if self._value < 0.0:
                    self._value = 0.0

        self._last_time = max(self._last_time, now)
        return self._value

    def reset(self) -> None:
        """Drop back to ``0.0`` and forget the hold window.

        Called when capture starts, so a fresh run does not inherit the peak of
        the previous one.
        """
        self._value = 0.0
        self._hold_until = None
        self._last_time = None

    def __repr__(self) -> str:
        """Return a debugging representation including the current value."""
        return (
            f"{type(self).__name__}(hold_s={self._hold_s:g}, "
            f"decay_db_per_s={self._decay_db_per_s:g}, value={self._value:.4f})"
        )


class LevelMeter(QWidget):
    """A horizontal RMS bar with a peak marker.

    Paints, and nothing else: the levels it draws are handed to it by
    :class:`~echochamber.ui.main_window.MainWindow`, already peak-held by
    :class:`PeakHold` on the controller.  No policy, no clock, no state beyond
    the two numbers.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create an empty meter.

        Args:
            parent: Parent widget, or ``None``.
        """
        super().__init__(parent)
        self._rms: float = 0.0
        self._peak: float = 0.0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(18)

    @property
    def rms(self) -> float:
        """Last RMS level set, clamped to ``0.0``-``1.0``."""
        return self._rms

    @property
    def peak(self) -> float:
        """Last peak level set, clamped to ``0.0``-``1.0``."""
        return self._peak

    def set_levels(self, rms: float, peak: float) -> None:
        """Set the levels to draw and request a repaint.

        Args:
            rms: RMS level, ``0.0``-``1.0``.  Out-of-range values are clamped.
            peak: Peak level, ``0.0``-``1.0``.  Out-of-range values are clamped.
        """
        self._rms = _clamp01(rms)
        self._peak = _clamp01(peak)
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Return the preferred size of the meter."""
        return QSize(240, 18)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Return the smallest size the meter stays readable at."""
        return QSize(60, 18)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        """Draw the trough, the RMS bar and the peak marker.

        Args:
            event: Qt paint event; the whole widget is redrawn regardless.
        """
        painter = QPainter(self)
        try:
            rect = self.rect()
            painter.fillRect(rect, QColor(32, 32, 32))

            width = rect.width()
            height = rect.height()
            if width <= 0 or height <= 0:
                return

            bar_width = int(round(self._rms * width))
            if bar_width > 0:
                painter.fillRect(
                    rect.left(), rect.top(), bar_width, height, _level_colour(self._rms)
                )

            if self._peak > 0.0:
                marker_x = rect.left() + min(width - 2, int(round(self._peak * width)))
                painter.fillRect(
                    marker_x, rect.top(), 2, height, _level_colour(self._peak).lighter(140)
                )

            painter.setPen(QColor(90, 90, 90))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
        finally:
            painter.end()

    def __repr__(self) -> str:
        """Return a debugging representation including the drawn levels."""
        return f"{type(self).__name__}(rms={self._rms:.3f}, peak={self._peak:.3f})"


def _clamp01(value: float) -> float:
    """Clamp ``value`` into ``0.0``-``1.0``, mapping ``NaN`` to ``0.0``.

    Args:
        value: Any float-like level.

    Returns:
        The clamped level.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _level_colour(level: float) -> QColor:
    """Return the bar colour for ``level``: green, amber near clipping, red at it.

    Args:
        level: Level in ``0.0``-``1.0``.

    Returns:
        The colour to fill with.
    """
    if level >= 0.99:
        return QColor(220, 60, 60)
    if level >= 0.80:
        return QColor(220, 170, 50)
    return QColor(70, 180, 90)
