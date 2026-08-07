"""A stand-in for the ML inference stage.

Inference is out of scope for the ingestion project, but "the pipeline works"
is not a claim you can make without something on the far end behaving like the
real consumer. This sink does the two things that matter:

* it **measures** end-to-end latency, from the moment a window became complete
  to the moment the consumer actually got to it -- which includes queueing and
  scheduling, the parts a stopwatch around a model call would miss;
* it can **be slow on purpose**, so backpressure, drop policy and ring headroom
  are exercised before a real model with real variance is attached.

The delay is the point. A consumer that always keeps up proves nothing about
what happens when one doesn't.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

from echochamber.audio.latency import LatencySummary, LatencyTracker
from echochamber.audio.types import AudioChunk

__all__ = ["StubInferenceSink"]


class StubInferenceSink:
    """Pretends to run a model, and records how long the audio took to arrive.

    Runs on the pipeline's consumer thread. It must never touch Qt; the GUI
    reads :meth:`latency_summary` on its own timer instead.

    Args:
        delay_s: Simulated inference time per chunk. ``0.0`` keeps up with
            anything; raise it to force the bounded queue to do its job.
        tracker: Latency tracker to record into; one is created when ``None``.
        on_result: Optional callback invoked with ``(chunk, result)`` after each
            "inference", where ``result`` is whatever ``infer`` returned. Runs on
            the consumer thread, so it must not touch Qt either.
        infer: The fake model. Defaults to a cheap RMS, so the sink does real
            numeric work on real samples rather than optimising away.
    """

    __slots__ = (
        "_delay_s",
        "_tracker",
        "_on_result",
        "_infer",
        "_lock",
        "_processed",
        "_last_result",
        "_closed",
    )

    def __init__(
        self,
        delay_s: float = 0.0,
        tracker: LatencyTracker | None = None,
        on_result: Callable[[AudioChunk, float], None] | None = None,
        infer: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        if delay_s < 0.0:
            raise ValueError(f"delay_s must be >= 0, got {delay_s}")
        self._delay_s = float(delay_s)
        self._tracker = tracker if tracker is not None else LatencyTracker()
        self._on_result = on_result
        self._infer = infer if infer is not None else _rms
        self._lock = threading.Lock()
        self._processed = 0
        self._last_result: float | None = None
        self._closed = False

    @property
    def tracker(self) -> LatencyTracker:
        """The latency tracker this sink records into."""
        return self._tracker

    @property
    def delay_s(self) -> float:
        """Simulated per-chunk inference time, in seconds."""
        return self._delay_s

    @property
    def processed(self) -> int:
        """Chunks this sink has handled."""
        with self._lock:
            return self._processed

    @property
    def last_result(self) -> float | None:
        """Most recent ``infer`` result, or ``None`` before the first chunk."""
        with self._lock:
            return self._last_result

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def latency_summary(self) -> LatencySummary:
        """Current latency percentiles. Safe to call from the GUI thread."""
        return self._tracker.summary()

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Record arrival latency, run the fake model, then optionally sleep.

        Latency is measured **before** the simulated work: the question being
        answered is "how stale was this audio when we got to it", and folding
        our own processing time into that would conflate two different problems.
        """
        # perf_counter to match AudioChunk.capture_time; time.monotonic() is
        # GetTickCount64 on Windows and would quantise this to 0 or 16 ms.
        self._tracker.record(chunk.age_s(time.perf_counter()))

        result = self._infer(chunk.samples)

        with self._lock:
            self._processed += 1
            self._last_result = result

        if self._delay_s:
            time.sleep(self._delay_s)

        if self._on_result is not None:
            self._on_result(chunk, result)

    def close(self) -> None:
        """Idempotent; nothing to release, but sinks are closed uniformly."""
        self._closed = True

    def __repr__(self) -> str:
        """Return a debugging representation naming throughput and latency."""
        summary = self._tracker.summary()
        return (
            f"{type(self).__name__}(processed={self.processed}, "
            f"delay_s={self._delay_s}, p95={summary.p95_ms:.1f}ms)"
        )


def _rms(samples: np.ndarray) -> float:
    """Cheap stand-in for a model: root-mean-square of the window."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
