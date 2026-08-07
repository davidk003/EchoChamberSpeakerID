"""EchoChamber speaker-ID audio ingestion package.

Step 1 provides the foundation of the capture pipeline:

* :mod:`echochamber.config` -- :class:`~echochamber.config.AudioConfig` and the
  millisecond-to-frame helper used to derive window/hop geometry.
* :mod:`echochamber.audio.types` -- the value types that cross thread
  boundaries (:class:`~echochamber.audio.types.AudioChunk`,
  :class:`~echochamber.audio.types.StreamStats`,
  :class:`~echochamber.audio.types.DropPolicy`).
* :mod:`echochamber.audio.ringbuffer` -- the single-producer / single-consumer
  ring buffer with the doubled-write layout that makes every read a contiguous
  view.

Nothing here imports a GUI toolkit or an audio backend, so the whole module is
importable headless and on any architecture (including Windows ARM64).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__: str = "0.1.0"
