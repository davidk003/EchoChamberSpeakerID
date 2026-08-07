"""All of the GUI's logic: device selection, lifecycle, and the polling tick.

Three things live here, and deliberately nothing else does.

**The sink that must not touch Qt.**  :class:`LatestChunkSink` is called on the
pipeline's consumer thread.  It emits no signal, touches no widget and calls
nothing in :mod:`echochamber.ui` that does -- it records the latest chunk's
``seq``, ``start_frame`` and a decimated preview under a lock, and the GUI
timer reads them.  A Qt signal per chunk would cross a thread boundary 20
times a second at ``hop_ms = 50`` to feed a display that refreshes at 60 Hz.

**The controller.**  :class:`CaptureController` owns the
:class:`~echochamber.audio.pipeline.AudioPipeline`, the device list, the
:class:`~echochamber.ui.meters.PeakHold` and the 30 Hz ``QTimer``.  Every
widget in this package is a renderer of the :class:`UiStats` it produces and a
source of intent it acts on; none of them computes anything.  That split is
what lets the entire GUI be driven headless from a
:class:`~echochamber.audio.sources.file_source.FileSource`.

**The injection seam.**  ``source_factory_builder`` is how a test replaces the
microphone.  The controller never constructs a
:class:`~echochamber.audio.sources.sounddevice_source.SoundDeviceSource`
directly; it asks the builder for a
:data:`~echochamber.audio.pipeline.SourceFactory`, and the default builder is
just one more function.

:meth:`CaptureController.start`, :meth:`~CaptureController.stop` and
:meth:`~CaptureController.set_geometry` never raise.  They are wired straight
to buttons and spin boxes, and an exception escaping a Qt slot in a release
build is at best a stderr traceback and at worst a dead event loop, so every
failure is reported through ``error_occurred`` and a ``bool``.
"""

from __future__ import annotations

import enum
import inspect
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal

from echochamber.audio.devices import DeviceInfo, list_input_devices
from echochamber.audio.latency import LatencyTracker
from echochamber.audio.pipeline import AudioPipeline, SourceFactory
from echochamber.audio.sources.sounddevice_source import SoundDeviceSource
from echochamber.audio.types import AudioChunk, StreamStats
from echochamber.config import AudioConfig
from echochamber.ui.meters import PeakHold

__all__ = [
    "POLL_INTERVAL_MS",
    "STOP_TIMEOUT_S",
    "CaptureController",
    "CaptureState",
    "LatestChunkSink",
    "UiStats",
    "default_source_factory_builder",
]

POLL_INTERVAL_MS: int = 33
"""GUI poll period in milliseconds -- ~30 Hz, the rate the architecture specifies."""

STOP_TIMEOUT_S: float = 2.0
"""Budget handed to :meth:`AudioPipeline.stop`.

Bounded because :meth:`CaptureController.stop` runs on the GUI thread, and a
wedged sink must cost a dropped shutdown rather than a frozen window.
"""


class CaptureState(enum.Enum):
    """What the capture pipeline is doing, as far as the GUI is concerned.

    Attributes:
        STOPPED: No pipeline, or one that has been shut down cleanly.
        RUNNING: A pipeline is started and the poll timer is ticking.
        ERROR: The last start failed, or a running pipeline died.  The GUI
            shows the message; the next successful :meth:`CaptureController.start`
            clears it.
    """

    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class UiStats:
    """Immutable snapshot handed to the widgets on each timer tick.

    Frozen because it crosses from the controller into every panel: a widget
    that could mutate it would be able to change what another widget renders.

    Attributes:
        state: Current :class:`CaptureState`.
        frames_captured: Frames written into the ring since start.
        chunks_emitted: Windows produced by the chunker.
        chunks_dropped: Windows discarded by the bounded queue.  Non-zero here
            is the failure mode the architecture calls out, so the panel
            emphasises it.
        overruns: Times the ring reader fell behind and lost audio.
        xruns: Device-level input overflows.
        peak_level: Peak of the most recent block, ``0.0``-``1.0`` -- raw, not
            held.
        rms_level: RMS of the most recent block, ``0.0``-``1.0``.
        display_peak: ``peak_level`` after hold and decay; what a meter should
            draw.  See :class:`~echochamber.ui.meters.PeakHold`.
        elapsed_s: Seconds since :meth:`CaptureController.start`, frozen at the
            value it had when capture stopped.
        latency_ms: Input latency the device reports, in milliseconds; ``0.0``
            whenever nothing is running.  This is the *device buffer* only --
            see the pipeline figures below for what a consumer actually waits.
        message: Human-readable note, normally empty; carries the failure text
            in :attr:`CaptureState.ERROR`.
        pipeline_p50_ms: Median measured latency from a window being complete to
            a consumer receiving it.  Dominated by ``window_ms`` by design, since
            a window cannot be emitted until its last sample exists.
        pipeline_p95_ms: The slowest 1-in-20.  This is the number that reveals a
            struggling consumer, which a mean would hide.
        pipeline_max_ms: Worst observation in the rolling window.
        latency_samples: How many observations the percentiles are drawn from;
            0 means the figures are not yet meaningful.
    """

    state: CaptureState
    frames_captured: int
    chunks_emitted: int
    chunks_dropped: int
    overruns: int
    xruns: int
    peak_level: float
    rms_level: float
    display_peak: float
    elapsed_s: float
    latency_ms: float
    message: str = ""
    pipeline_p50_ms: float = 0.0
    pipeline_p95_ms: float = 0.0
    pipeline_max_ms: float = 0.0
    latency_samples: int = 0


class LatestChunkSink:
    """Terminal sink that only records -- it must never touch Qt.

    ``on_chunk`` runs on the pipeline's consumer thread.  It stores the latest
    chunk's ``seq``, ``start_frame`` and a decimated waveform preview under a
    lock; the GUI timer calls :meth:`snapshot` and gets a copy.  Nothing here
    imports a widget, emits a signal, or blocks for longer than the memcpy of a
    few hundred floats -- a sink that stalls costs dropped chunks upstream.

    The preview is decimated by taking ``preview_points`` evenly spaced samples
    (endpoints included), which is cheap, allocation-bounded regardless of
    window length, and gives a stable trace to draw.
    """

    __slots__ = (
        "_preview_points",
        "_tracker",
        "_lock",
        "_seq",
        "_start_frame",
        "_preview",
        "_closed",
    )

    def __init__(
        self,
        preview_points: int = 512,
        tracker: LatencyTracker | None = None,
    ) -> None:
        """Create an empty sink.

        Args:
            preview_points: Maximum number of points in the decimated preview.
            tracker: Latency tracker to record into; one is created when
                ``None``.  Pass a shared instance to keep observations across a
                stop/start cycle.

        Raises:
            ValueError: If ``preview_points`` is not positive.
        """
        preview_points = int(preview_points)
        if preview_points <= 0:
            raise ValueError(f"preview_points must be > 0, got {preview_points}")

        self._preview_points: int = preview_points
        self._tracker: LatencyTracker = (
            LatencyTracker() if tracker is None else tracker
        )
        self._lock = threading.Lock()
        self._seq: int = -1
        self._start_frame: int = 0
        self._preview: np.ndarray = np.zeros(0, dtype=np.float32)
        self._closed: bool = False

    @property
    def preview_points(self) -> int:
        """Maximum number of points in the decimated preview."""
        return self._preview_points

    @property
    def seq(self) -> int:
        """``seq`` of the most recent chunk, or ``-1`` before the first one."""
        with self._lock:
            return self._seq

    @property
    def start_frame(self) -> int:
        """``start_frame`` of the most recent chunk, or ``0`` before the first."""
        with self._lock:
            return self._start_frame

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Record ``chunk``.  Called on the consumer thread; touches no Qt.

        Latency is measured here rather than in the GUI because this is the
        first moment the audio has actually reached a consumer -- it therefore
        includes the queueing and thread scheduling that a timer on the GUI side
        would never see.

        Args:
            chunk: The completed window handed over by the pipeline.
        """
        # perf_counter to match AudioChunk.capture_time; time.monotonic() is
        # GetTickCount64 on Windows and would quantise this to 0 or 16 ms.
        self._tracker.record(chunk.age_s(time.perf_counter()))
        preview = _decimate(chunk.samples, self._preview_points)
        with self._lock:
            self._seq = chunk.seq
            self._start_frame = chunk.start_frame
            self._preview = preview

    @property
    def tracker(self) -> LatencyTracker:
        """Rolling latency observations recorded by this sink."""
        return self._tracker

    def latency_summary(self) -> LatencySummary:
        """Latency percentiles so far.  Safe to call from the GUI thread."""
        return self._tracker.summary()

    def close(self) -> None:
        """Mark the sink closed.  Idempotent; the recorded snapshot survives.

        The last preview is deliberately kept: the GUI paints one more frame
        after the pipeline stops, and blanking here would make a completed
        capture look like a dead one.
        """
        self._closed = True

    def snapshot(self) -> tuple[int, np.ndarray]:
        """Return ``(seq, preview)`` for the GUI thread.

        Returns:
            The latest chunk's ``seq`` (``-1`` if none yet) and a **copy** of
            the decimated preview, so the caller can hold it while the consumer
            thread records the next chunk.
        """
        with self._lock:
            return self._seq, self._preview.copy()

    def __repr__(self) -> str:
        """Return a debugging representation of the recorded state."""
        seq, preview = self.snapshot()
        return (
            f"{type(self).__name__}(preview_points={self._preview_points}, "
            f"seq={seq}, preview_len={len(preview)}, closed={self._closed})"
        )


def default_source_factory_builder(
    config: AudioConfig,
    device: DeviceInfo | None = None,
    sd_module: Any = None,
) -> SourceFactory:
    """Build a factory that opens ``device`` as a live capture source.

    The default for :class:`CaptureController`; a test supplies its own builder
    returning a :class:`~echochamber.audio.sources.file_source.FileSource`
    factory instead, which is the whole reason this indirection exists.

    Args:
        config: Configuration supplying ``sample_rate`` and ``blocksize``.
        device: Device to open, or ``None`` for the system default.
        sd_module: Module to use instead of the real :mod:`sounddevice`.

    Returns:
        A :data:`~echochamber.audio.pipeline.SourceFactory`; it constructs the
        source when the pipeline calls it, so a device that refuses to open
        fails inside :meth:`CaptureController.start` where it can be reported.
    """

    def factory(on_audio: Callable[[np.ndarray], None], stats: StreamStats) -> SoundDeviceSource:
        """Construct the live source the pipeline asked for.

        Args:
            on_audio: The ring's ``write`` method.
            stats: The pipeline's shared stats record.

        Returns:
            An unstarted :class:`SoundDeviceSource`.
        """
        return SoundDeviceSource(
            on_audio,
            device=None if device is None else device.index,
            sample_rate=config.sample_rate,
            blocksize=config.blocksize,
            stats=stats,
            sd_module=sd_module,
        )

    return factory


class CaptureController(QObject):
    """Owns the pipeline, the device list and the poll timer.

    Every widget in this package talks only to this object.  It is a
    ``QObject`` so its signals are queued correctly, but it is only ever
    *called* from the GUI thread -- the audio path reaches it exclusively
    through :class:`LatestChunkSink` and
    :meth:`~echochamber.audio.types.StreamStats.snapshot`.

    Signals:
        stats_updated: Emitted with a :class:`UiStats` on every poll.
        state_changed: Emitted with a :class:`CaptureState` on every transition.
        devices_changed: Emitted with ``list[DeviceInfo]`` after a refresh.
        error_occurred: Emitted with a human-readable message on any failure.
    """

    stats_updated = Signal(object)  # UiStats
    state_changed = Signal(object)  # CaptureState
    devices_changed = Signal(object)  # list[DeviceInfo]
    error_occurred = Signal(str)

    def __init__(
        self,
        config: AudioConfig | None = None,
        source_factory_builder: Callable[..., SourceFactory] | None = None,
        sd_module: Any = None,
        parent: QObject | None = None,
    ) -> None:
        """Create a stopped controller.  Nothing is enumerated or opened yet.

        Device enumeration is not done here: importing :mod:`sounddevice` loads
        PortAudio, which is a real cost and a real failure mode, and a
        controller that cannot be constructed headless would defeat the point
        of the injection seam.  Call :meth:`refresh_devices` when ready.

        Args:
            config: Starting configuration; a default :class:`AudioConfig` when
                ``None``.
            source_factory_builder: Callable returning a
                :data:`~echochamber.audio.pipeline.SourceFactory`.  Defaults to
                :func:`default_source_factory_builder`.  It is invoked with
                whichever of the keyword arguments ``config``, ``device`` and
                ``sd_module`` its signature accepts, so a test may inject
                anything from ``lambda **kw: factory`` down to
                ``lambda: factory``.
            sd_module: Module to use instead of the real :mod:`sounddevice`,
                threaded through both enumeration and the default builder.
            parent: Qt parent, or ``None``.
        """
        super().__init__(parent)

        self._config: AudioConfig = AudioConfig() if config is None else config
        self._source_factory_builder: Callable[..., SourceFactory] = (
            default_source_factory_builder
            if source_factory_builder is None
            else source_factory_builder
        )
        self._sd_module: Any = sd_module

        self._state: CaptureState = CaptureState.STOPPED
        self._devices: list[DeviceInfo] = []
        self._selected_device: DeviceInfo | None = None
        self._pipeline: AudioPipeline | None = None
        self._sink: LatestChunkSink | None = None
        # Owned by the controller, not the sink, so the figures from a finished
        # run survive stop() -- that is precisely when you want to read them.
        self._latency_tracker: LatencyTracker = LatencyTracker()
        self._peak_hold: PeakHold = PeakHold()

        self._start_time: float = 0.0
        self._elapsed_at_stop: float = 0.0
        self._last_snapshot: StreamStats = StreamStats()
        self._latency_ms: float = 0.0
        self._message: str = ""
        self._error_reported: bool = False

        self._timer: QTimer = QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        # Precise, despite this being "only" a display: Qt's coarse timers are
        # rounded to the Windows scheduler tick (~15.6 ms), which turns a 33 ms
        # interval into ~47 ms -- measured 21 Hz instead of the 30 Hz the
        # architecture specifies, which is visible on a level meter.
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self.poll)

    # -- state -------------------------------------------------------------

    @property
    def config(self) -> AudioConfig:
        """The configuration in force; swapped wholesale by :meth:`set_geometry`."""
        return self._config

    @property
    def state(self) -> CaptureState:
        """Current :class:`CaptureState`."""
        return self._state

    @property
    def devices(self) -> list[DeviceInfo]:
        """The input devices from the last :meth:`refresh_devices` (a copy)."""
        return list(self._devices)

    @property
    def selected_device(self) -> DeviceInfo | None:
        """The device the next :meth:`start` will open, or ``None`` for default."""
        return self._selected_device

    @property
    def pipeline(self) -> AudioPipeline | None:
        """The live pipeline, or ``None`` when stopped (or after a failed start)."""
        return self._pipeline

    @property
    def sink(self) -> LatestChunkSink | None:
        """The recording sink of the live pipeline, or ``None``."""
        return self._sink

    @property
    def latency_tracker(self) -> LatencyTracker:
        """Measured end-to-end latency for the current or most recent run.

        Outlives the pipeline deliberately, so the figures are still readable
        after :meth:`stop`.  Cleared at the start of each run.
        """
        return self._latency_tracker

    @property
    def peak_hold(self) -> PeakHold:
        """The peak-hold feeding :attr:`UiStats.display_peak`."""
        return self._peak_hold

    @property
    def poll_timer(self) -> QTimer:
        """The 30 Hz timer driving :meth:`poll` while capture runs."""
        return self._timer

    # -- devices -----------------------------------------------------------

    def refresh_devices(self) -> list[DeviceInfo]:
        """Re-enumerate input devices, pick a default, and announce both.

        The default **prefers a WASAPI device over PortAudio's reported
        default**.  Measured on the dev hardware, the system default input is an
        *MME* device at ~30 ms input latency against ~22 ms for the same
        physical microphone through WASAPI -- and MME bypasses the
        ``auto_convert`` path entirely, so defaulting to it would mean the
        resampling path the pipeline depends on never gets exercised.  Among
        WASAPI devices the system default wins, then the first one; with no
        WASAPI input at all it falls back to the reported default, then the
        first input device, then ``None``.

        An existing selection that still exists is kept, so a refresh does not
        silently move the user's microphone.

        Returns:
            The devices found, possibly empty.  Enumeration failures are
            reported through ``error_occurred`` and yield an empty list: "no
            microphone" is a state the GUI renders, not an exception it
            handles.
        """
        try:
            devices = list_input_devices(self._sd_module)
        except Exception as exc:  # noqa: BLE001 - a broken host API must not kill the GUI
            devices = []
            self._fail(f"could not enumerate input devices: {exc}", set_error_state=False)

        self._devices = devices
        self._selected_device = _choose_device(devices, self._selected_device)
        self.devices_changed.emit(list(devices))
        return list(devices)

    def select_device(self, index_or_none: int | None) -> None:
        """Select the device with PortAudio index ``index_or_none``.

        Changing the device (or the sample rate) requires a stop/start: the ring
        is sized and the source is opened at pipeline construction, and both
        classes are single-use.  Calling this while running therefore records
        the choice and **leaves the running capture completely alone**; the
        caller is expected to restart for it to take effect.

        Args:
            index_or_none: A PortAudio device index from :attr:`devices`, or
                ``None`` for the system default.  An index that matches nothing
                is reported through ``error_occurred`` and leaves the selection
                unchanged -- it means the device list is stale, so re-enumerate.
        """
        if index_or_none is None:
            self._selected_device = None
            return

        for device in self._devices:
            if device.index == index_or_none:
                self._selected_device = device
                return

        self._fail(
            f"no input device with index {index_or_none} is available; "
            f"refresh the device list",
            set_error_state=False,
        )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Build and start a pipeline for the current config and device.

        Returns:
            ``True`` if capture is running (including when it already was),
            ``False`` if anything failed -- in which case the state is
            :attr:`CaptureState.ERROR` and ``error_occurred`` carries the
            reason.  Never raises: this is wired straight to a button.
        """
        if self._pipeline is not None and self._state is CaptureState.RUNNING:
            return True

        pipeline: AudioPipeline | None = None
        try:
            factory = self._build_factory()
            # A fresh run starts with fresh percentiles; the previous run's
            # figures would otherwise blend into this one's.
            self._latency_tracker.reset()
            sink = LatestChunkSink(tracker=self._latency_tracker)
            pipeline = AudioPipeline(self._config, sink, factory, StreamStats())
            pipeline.start()
        except Exception as exc:  # noqa: BLE001 - every failure is reported, not raised
            if pipeline is not None:
                try:
                    pipeline.stop(STOP_TIMEOUT_S)
                except Exception:  # noqa: BLE001 - teardown of a failed start
                    pass
            self._pipeline = None
            self._sink = None
            self._fail(_describe(exc))
            return False

        self._pipeline = pipeline
        self._sink = sink
        self._peak_hold.reset()
        self._last_snapshot = StreamStats()
        self._latency_ms = 0.0
        self._message = ""
        self._error_reported = False
        self._start_time = time.monotonic()
        self._elapsed_at_stop = 0.0
        self._set_state(CaptureState.RUNNING)
        self._timer.start()
        self.poll()
        return True

    def stop(self) -> bool:
        """Shut the pipeline down and go back to :attr:`CaptureState.STOPPED`.

        Idempotent, safe when nothing was ever started, and safe from
        ``closeEvent``.  The counters from the finished run are kept so the
        stats panel does not blank out the moment capture ends; only the
        latency, which describes an open stream, goes to zero.

        Returns:
            ``True``.  Never raises -- a pipeline that refuses to shut down
            inside :data:`STOP_TIMEOUT_S` leaves daemon threads that die with
            the process, and a frozen GUI would be strictly worse.
        """
        self._timer.stop()
        pipeline = self._pipeline
        if pipeline is not None:
            self._elapsed_at_stop = time.monotonic() - self._start_time
            try:
                self._last_snapshot = pipeline.stats.snapshot()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass
            try:
                pipeline.stop(STOP_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self.error_occurred.emit(f"error while stopping capture: {exc}")

        self._pipeline = None
        self._sink = None
        self._latency_ms = 0.0
        self._message = ""
        self._error_reported = False
        self._set_state(CaptureState.STOPPED)
        self.stats_updated.emit(self._build_stats())
        return True

    def set_geometry(self, window_ms: int, hop_ms: int) -> bool:
        """Validate and apply new window geometry, live if capture is running.

        Validation is delegated to :class:`AudioConfig`, which is the only
        place that knows the rules (hop must not exceed the window, neither may
        round to zero frames, the ring must hold a window plus a hop).

        Args:
            window_ms: New window length in milliseconds.
            hop_ms: New hop length in milliseconds.

        Returns:
            ``True`` on success.  ``False`` if the combination is invalid or the
            live reconfigure was rejected, in which case ``error_occurred``
            carries the reason and :attr:`config` is **unchanged**.  Never
            raises.
        """
        try:
            new_config = self._config.with_window(window_ms=window_ms, hop_ms=hop_ms)
        except Exception as exc:  # noqa: BLE001 - ValueError/TypeError alike
            self.error_occurred.emit(_describe(exc))
            return False

        pipeline = self._pipeline
        if pipeline is not None:
            try:
                pipeline.reconfigure(new_config)
            except Exception as exc:  # noqa: BLE001 - the running config stands
                self.error_occurred.emit(_describe(exc))
                return False

        self._config = new_config
        return True

    # -- the tick ----------------------------------------------------------

    def poll(self) -> UiStats:
        """Sample the pipeline and emit one :class:`UiStats`.

        Called by the 30 Hz timer, and directly by tests -- which is why it
        returns the snapshot as well as emitting it.  Reads only
        :meth:`StreamStats.snapshot`, the source's reported latency and
        :attr:`AudioPipeline.error`; it never blocks and never touches audio
        data.

        A pipeline that died on one of its own threads never raises anywhere a
        caller can see it, so this is where that is noticed: the state goes to
        :attr:`CaptureState.ERROR`, ``error_occurred`` fires once, and the timer
        stops.

        Returns:
            The snapshot that was emitted.
        """
        pipeline = self._pipeline
        if pipeline is not None:
            try:
                self._last_snapshot = pipeline.stats.snapshot()
                self._latency_ms = _source_latency_ms(pipeline)
            except Exception:  # noqa: BLE001 - a display tick must never raise
                pass

            error = _pipeline_error(pipeline)
            if error is not None and not self._error_reported:
                self._error_reported = True
                self._timer.stop()
                self._elapsed_at_stop = time.monotonic() - self._start_time
                self._latency_ms = 0.0
                message = f"capture failed: {_describe(error)}"
                try:
                    pipeline.stop(STOP_TIMEOUT_S)
                except Exception:  # noqa: BLE001 - already failing
                    pass
                self._set_state(CaptureState.ERROR, message)

        stats = self._build_stats()
        self.stats_updated.emit(stats)
        return stats

    # -- internals ---------------------------------------------------------

    def _build_stats(self) -> UiStats:
        """Assemble the current :class:`UiStats` from the last sampled values.

        Returns:
            The snapshot for this tick, including the peak-held display level.
        """
        raw = self._last_snapshot
        running = self._state is CaptureState.RUNNING and self._pipeline is not None
        display_peak = self._peak_hold.update(
            raw.peak_level if running else 0.0, time.monotonic()
        )
        if running:
            elapsed = time.monotonic() - self._start_time
        else:
            elapsed = self._elapsed_at_stop
        # Latency figures survive stopping on purpose: the run just finished is
        # exactly when you want to read what it achieved.
        latency = self._latency_tracker.summary()
        return UiStats(
            state=self._state,
            frames_captured=raw.frames_captured,
            chunks_emitted=raw.chunks_emitted,
            chunks_dropped=raw.chunks_dropped,
            overruns=raw.overruns,
            xruns=raw.xruns,
            peak_level=raw.peak_level,
            rms_level=raw.rms_level,
            display_peak=display_peak,
            elapsed_s=elapsed,
            latency_ms=self._latency_ms if running else 0.0,
            message=self._message,
            pipeline_p50_ms=latency.p50_ms,
            pipeline_p95_ms=latency.p95_ms,
            pipeline_max_ms=latency.max_ms,
            latency_samples=latency.count,
        )

    def _build_factory(self) -> SourceFactory:
        """Ask ``source_factory_builder`` for a factory for the current selection.

        The builder is called with whichever of ``config``, ``device`` and
        ``sd_module`` its signature will accept, so an injected
        ``lambda: file_factory`` works as well as the three-argument default.

        Returns:
            The :data:`~echochamber.audio.pipeline.SourceFactory` to build the
            source with.
        """
        builder = self._source_factory_builder
        offered: dict[str, Any] = {
            "config": self._config,
            "device": self._selected_device,
            "sd_module": self._sd_module,
        }
        return builder(**_acceptable_kwargs(builder, offered))

    def _set_state(self, state: CaptureState, message: str = "") -> None:
        """Move to ``state``, remember ``message``, and announce both.

        Args:
            state: The new state.
            message: Text for :attr:`UiStats.message`; a non-empty value is
                also emitted through ``error_occurred``.
        """
        self._message = message
        if message:
            self.error_occurred.emit(message)
        if state is not self._state:
            self._state = state
            self.state_changed.emit(state)

    def _fail(self, message: str, set_error_state: bool = True) -> None:
        """Report a failure through ``error_occurred``, optionally entering ERROR.

        Args:
            message: Human-readable failure text.
            set_error_state: ``True`` to move to :attr:`CaptureState.ERROR`.
                ``False`` for failures that leave the capture state alone, such
                as a device enumeration that came back empty.
        """
        if set_error_state:
            self._set_state(CaptureState.ERROR, message)
        else:
            self.error_occurred.emit(message)

    def __repr__(self) -> str:
        """Return a debugging representation of the controller's state."""
        device = None if self._selected_device is None else self._selected_device.label
        return (
            f"{type(self).__name__}(state={self._state.value}, device={device!r}, "
            f"window_ms={self._config.window_ms}, hop_ms={self._config.hop_ms})"
        )


def _choose_device(
    devices: list[DeviceInfo], current: DeviceInfo | None
) -> DeviceInfo | None:
    """Pick the device to select after an enumeration.

    Args:
        devices: Freshly enumerated input devices.
        current: The previous selection, if any.

    Returns:
        ``current`` re-bound to the new list if it is still present; otherwise
        a WASAPI device, then PortAudio's reported default, then the first
        input, then ``None``.

    Which WASAPI device, when there are several, is not something the
    preference rule alone settles -- and on a machine with a virtual-audio
    driver installed, "the first one" is routinely a virtual cable rather than
    a microphone.  So: a WASAPI device PortAudio already calls the default
    wins, then the WASAPI entry for the *same physical device* as the reported
    default (matched on name, allowing for MME's 31-character truncation --
    which is exactly the ~30 ms MME / ~22 ms WASAPI pair this preference exists
    to resolve), then the first WASAPI device.
    """
    if current is not None:
        for device in devices:
            if device.index == current.index and device.label == current.label:
                return device

    reported_default = next((d for d in devices if d.is_default_input), None)

    wasapi = [d for d in devices if d.is_wasapi]
    for device in wasapi:
        if device.is_default_input:
            return device
    if reported_default is not None:
        for device in wasapi:
            if _same_hardware(device.name, reported_default.name):
                return device
    if wasapi:
        return wasapi[0]
    if reported_default is not None:
        return reported_default
    return devices[0] if devices else None


def _same_hardware(a: str, b: str) -> bool:
    """Guess whether two device names describe the same physical device.

    MME truncates device names to 31 characters, so the same microphone is
    ``"Microphone Array (Intel Smart "`` under MME and the full string under
    WASAPI.  Comparing on the shorter name's length is the only handle there
    is; it is a heuristic, used solely to break a tie between candidates that
    were all acceptable anyway.

    Args:
        a: One device name.
        b: The other device name.

    Returns:
        ``True`` if one name is a case-insensitive prefix of the other and the
        shared prefix is long enough to mean something.
    """
    left = a.strip().lower()
    right = b.strip().lower()
    shared = min(len(left), len(right))
    if shared < 8:
        return False
    return left[:shared] == right[:shared]


def _decimate(samples: np.ndarray, points: int) -> np.ndarray:
    """Reduce ``samples`` to at most ``points`` evenly spaced values.

    Args:
        samples: 1-D sample array from a chunk.
        points: Maximum number of points to keep.

    Returns:
        A new ``float32`` array, never a view into ``samples``.  Endpoints are
        preserved, so the trace still starts and ends where the window does.
    """
    data = np.asarray(samples)
    n = data.size
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n <= points:
        return data.astype(np.float32, copy=True)
    indices = np.linspace(0, n - 1, points).astype(np.intp)
    return data[indices].astype(np.float32, copy=True)


def _acceptable_kwargs(fn: Callable[..., Any], offered: dict[str, Any]) -> dict[str, Any]:
    """Filter ``offered`` down to the keyword arguments ``fn`` will accept.

    The contract types ``source_factory_builder`` as ``Callable[..., SourceFactory]``
    without fixing its parameters, so the controller adapts to the callable it
    is given instead of forcing every injected builder to declare three
    parameters it may not want.

    Args:
        fn: The builder callable.
        offered: Every keyword argument the controller can supply.

    Returns:
        The subset ``fn`` accepts -- all of it if ``fn`` takes ``**kwargs``, and
        all of it as a best guess if the signature cannot be inspected.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return dict(offered)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(offered)

    keyword_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    return {
        name: value
        for name, value in offered.items()
        if name in parameters and parameters[name].kind in keyword_kinds
    }


def _pipeline_error(pipeline: AudioPipeline) -> BaseException | None:
    """Return the pipeline's first failure, if any, without ever raising.

    Args:
        pipeline: The pipeline to inspect.

    Returns:
        The recorded exception, or ``None``.
    """
    try:
        return pipeline.error
    except Exception:  # noqa: BLE001 - a display tick must never raise
        return None


def _source_latency_ms(pipeline: AudioPipeline) -> float:
    """Return the source's reported input latency in milliseconds.

    Args:
        pipeline: The running pipeline.

    Returns:
        Latency in milliseconds, or ``0.0`` for a source that does not report
        one (a :class:`~echochamber.audio.sources.file_source.FileSource` has
        no device latency to report).
    """
    try:
        latency = getattr(pipeline.source, "latency", 0.0)
        return float(latency) * 1000.0
    except Exception:  # noqa: BLE001 - informational only
        return 0.0


def _describe(exc: BaseException) -> str:
    """Render an exception as a message worth putting in front of a user.

    Args:
        exc: The exception to describe.

    Returns:
        ``"Message"``, or ``"ExceptionType"`` when the exception carries no
        message of its own -- an empty status bar explains nothing.
    """
    text = str(exc).strip()
    return text if text else type(exc).__name__
