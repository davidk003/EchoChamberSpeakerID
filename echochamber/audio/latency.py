"""End-to-end latency measurement for the ingestion pipeline.

The architecture's latency budget is a prediction; this is how it gets checked
against reality. A chunk carries the :attr:`~echochamber.audio.types.AudioChunk.capture_time`
at which it became complete, and any consumer can report how long it took to get
there, including the queueing and scheduling that a stopwatch around the model
call would miss entirely.

Two deliberate choices:

* **Percentiles, not averages.** A mean hides the occasional 300 ms stall that a
  user actually notices. p95 and max are the numbers worth putting on screen.
* **The caller supplies ``now``.** Nothing here reads the clock, so latency
  behaviour can be tested exactly rather than by sleeping and hoping.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable

__all__ = ["LatencySummary", "LatencyTracker"]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """A snapshot of observed latencies, in milliseconds.

    Attributes:
        count: Samples in the window this summary was taken from.
        mean_ms: Arithmetic mean.
        p50_ms: Median -- the typical experience.
        p95_ms: What the slowest 1-in-20 chunks see; the number to watch.
        p99_ms: Tail.
        max_ms: Worst observation in the window.
    """

    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        """``True`` when nothing has been recorded yet."""
        return self.count == 0


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Nearest-rank rather than interpolating: every value returned is a latency
    that genuinely occurred, which is what makes "p95 was 42 ms" a true
    statement rather than an artefact of averaging two neighbours.

    Rank is ``ceil(fraction * n)``, the standard definition -- so p95 of 100
    sorted observations is the 95th, not the 96th.
    """
    if not ordered:
        return 0.0
    rank = max(1, min(len(ordered), math.ceil(fraction * len(ordered))))
    return ordered[rank - 1]


class LatencyTracker:
    """Thread-safe rolling window of latency observations.

    The consumer thread records; the GUI thread summarises. Both go through one
    lock, which is safe here because -- unlike the audio callback -- neither is
    a real-time path.

    Args:
        window: How many recent observations to keep. The default of 512 covers
            roughly 8 minutes at a 1 s hop, and bounds memory on a long run.
    """

    __slots__ = ("_samples", "_lock", "_window", "_total", "_worst_ever")

    def __init__(self, window: int = 512) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._window = window
        self._samples: Deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._total = 0
        self._worst_ever = 0.0

    @property
    def window(self) -> int:
        """Maximum observations retained."""
        return self._window

    @property
    def total_observations(self) -> int:
        """Everything ever recorded, including samples aged out of the window."""
        with self._lock:
            return self._total

    @property
    def worst_ever_ms(self) -> float:
        """Worst latency since construction, never aged out.

        A spike that scrolled out of the rolling window still happened, and is
        usually the thing worth investigating.
        """
        with self._lock:
            return self._worst_ever

    def record(self, latency_s: float) -> None:
        """Record one observation, given in **seconds**.

        Negative values are clamped to zero rather than rejected: a caller
        differencing two clocks should never poison the statistics.
        """
        value_ms = max(0.0, float(latency_s)) * 1000.0
        with self._lock:
            self._samples.append(value_ms)
            self._total += 1
            if value_ms > self._worst_ever:
                self._worst_ever = value_ms

    def record_many(self, latencies_s: Iterable[float]) -> None:
        """Record several observations in seconds."""
        for value in latencies_s:
            self.record(value)

    def summary(self) -> LatencySummary:
        """Summarise the current window. Safe to call from any thread."""
        with self._lock:
            values = list(self._samples)
        if not values:
            return LatencySummary()
        ordered = sorted(values)
        return LatencySummary(
            count=len(ordered),
            mean_ms=sum(ordered) / len(ordered),
            p50_ms=_percentile(ordered, 0.50),
            p95_ms=_percentile(ordered, 0.95),
            p99_ms=_percentile(ordered, 0.99),
            max_ms=ordered[-1],
        )

    def reset(self) -> None:
        """Forget everything, including ``worst_ever_ms``."""
        with self._lock:
            self._samples.clear()
            self._total = 0
            self._worst_ever = 0.0

    def __repr__(self) -> str:
        """Return a debugging representation naming the current percentiles."""
        s = self.summary()
        return (
            f"{type(self).__name__}(count={s.count}, p50={s.p50_ms:.1f}ms, "
            f"p95={s.p95_ms:.1f}ms, max={s.max_ms:.1f}ms)"
        )
