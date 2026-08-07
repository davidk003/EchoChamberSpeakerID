"""PySide6 front end for the audio ingestion pipeline.

The rule that shapes every module in this package: **Qt is never touched from
an audio thread, and the audio path never blocks on Qt.**

The pipeline's sink runs on the consumer thread, so
:class:`~echochamber.ui.controller.LatestChunkSink` records plain Python
attributes under a lock and does not emit a single Qt signal.  A ``QTimer`` on
the GUI thread polls those attributes -- and
:meth:`~echochamber.audio.types.StreamStats.snapshot` -- at 30 Hz.  At
``hop_ms = 50`` a signal per chunk would post 20 events a second into the event
loop for a display that cannot show more than ~60 Hz; polling costs one
snapshot per frame regardless of the chunk rate.

The second rule: **all logic lives in**
:mod:`echochamber.ui.controller`.  Widgets render state and emit user intent;
they compute nothing and they own no policy.  That is what makes the whole GUI
testable under ``QT_QPA_PLATFORM=offscreen`` with a
:class:`~echochamber.audio.sources.file_source.FileSource` standing in for a
microphone.
"""

from __future__ import annotations

from echochamber.ui.controller import (
    POLL_INTERVAL_MS,
    CaptureController,
    CaptureState,
    LatestChunkSink,
    UiStats,
    default_source_factory_builder,
)
from echochamber.ui.device_panel import DevicePanel
from echochamber.ui.geometry_panel import GeometryPanel
from echochamber.ui.main_window import MainWindow
from echochamber.ui.meters import LevelMeter, PeakHold
from echochamber.ui.stats_panel import StatsPanel

__all__ = [
    "POLL_INTERVAL_MS",
    "CaptureController",
    "CaptureState",
    "DevicePanel",
    "GeometryPanel",
    "LatestChunkSink",
    "LevelMeter",
    "MainWindow",
    "PeakHold",
    "StatsPanel",
    "UiStats",
    "default_source_factory_builder",
]
