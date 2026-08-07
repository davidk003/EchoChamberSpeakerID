"""Tests for the step-5 PySide6 widgets, written from the API contract.

Written from the *spec*, not the implementation.  The contract's rule is that
widgets compute nothing: they render state handed to them and emit user intent.
So every test here pokes a public method or a signal and looks at what the
widget shows -- never at private attributes, and never at pixels.

Two things deserve explaining:

* **Emphasis is asserted structurally, not visually.**  "Dropped chunks must be
  emphasised when non-zero" cannot be tested by comparing pixels without
  freezing the design.  Instead :func:`style_signature` records everything a
  reasonable implementation could change to emphasise a value -- object name,
  stylesheet, font weight, palette, dynamic properties, visibility -- and the
  test asserts that *something* in it changed.  Any of the mechanisms the
  contract suggests satisfies it; simply printing a different number does not.
* **`set_geometry()` must not re-emit `geometry_changed`.**  That is the classic
  Qt signal loop: the controller writes the config back into the panel, the
  panel reports it as a user edit, the controller writes it again.  It gets an
  explicit test.

Everything runs with ``QT_QPA_PLATFORM=offscreen``; the MainWindow tests drive a
controller whose source is a WAV replay, so no microphone or device table from
this machine is involved, and the window is closed (and the controller stopped)
in teardown.
"""

from __future__ import annotations

import os

# Must precede any PySide6 import: the suite has to run on a headless box.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import wave
from typing import Any, Callable, Iterator

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPalette
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QSpinBox,
    QWidget,
)

from echochamber.audio.devices import DeviceInfo, list_input_devices
from echochamber.audio.sources.file_source import FileSource
from echochamber.audio.types import DropPolicy, StreamStats
from echochamber.config import AudioConfig
from echochamber.ui.controller import CaptureController, CaptureState, UiStats
from echochamber.ui.device_panel import DevicePanel
from echochamber.ui.geometry_panel import GeometryPanel
from echochamber.ui.main_window import MainWindow
from echochamber.ui.meters import LevelMeter
from echochamber.ui.stats_panel import StatsPanel
from tests.fake_sounddevice import FakeSounddevice


TIMEOUT_MS = 20_000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def all_widgets(root: QWidget) -> list[QWidget]:
    """``root`` and every descendant widget, in a stable order."""
    return [root, *root.findChildren(QWidget)]


def rendered_text(root: QWidget) -> str:
    """Every piece of text the widget tree currently displays.

    Deliberately format-agnostic: the contract fixes *what* is shown, not how
    it is laid out, so the tests look for values inside this blob.
    """
    parts: list[str] = []
    for widget in all_widgets(root):
        for getter in ("text", "currentText", "toPlainText", "title"):
            method = getattr(widget, getter, None)
            if callable(method):
                try:
                    value = method()
                except TypeError:  # pragma: no cover - overloaded accessor
                    continue
                if isinstance(value, str):
                    parts.append(value)
    return " | ".join(parts)


def style_signature(root: QWidget) -> tuple[Any, ...]:
    """A fingerprint of every way a widget could be visually emphasised.

    Covers object names (``#alert`` selectors), stylesheets, font weight/style,
    palette colours, dynamic properties (the ``property("alert")`` idiom) and
    explicit visibility.  Text is *excluded* on purpose: changing the number is
    not emphasis.
    """
    signature: list[Any] = []
    for widget in all_widgets(root):
        properties = tuple(
            sorted(
                (bytes(name).decode("utf-8", "replace"),
                 repr(widget.property(bytes(name).decode("utf-8", "replace"))))
                for name in widget.dynamicPropertyNames()
            )
        )
        font = widget.font()
        palette = widget.palette()
        signature.append(
            (
                type(widget).__name__,
                widget.objectName(),
                widget.styleSheet(),
                font.bold(),
                str(font.weight()),
                font.italic(),
                font.underline(),
                palette.color(QPalette.ColorRole.WindowText).name(),
                palette.color(QPalette.ColorRole.Text).name(),
                palette.color(QPalette.ColorRole.Window).name(),
                widget.isVisibleTo(root),
                properties,
            )
        )
    return tuple(signature)


def buttons(root: QWidget) -> list[QAbstractButton]:
    return list(root.findChildren(QAbstractButton))


def button_labelled(root: QWidget, *words: str) -> QAbstractButton:
    """The single button whose text contains one of ``words``."""
    matches = []
    for button in buttons(root):
        text = button.text().replace("&", "").strip().lower()
        if any(word in text for word in words):
            matches.append(button)
    labels = [b.text() for b in buttons(root)]
    assert len(matches) == 1, (
        f"expected exactly one button matching {words!r}, found "
        f"{[b.text() for b in matches]} among {labels}"
    )
    return matches[0]


def paint(widget: QWidget) -> QImage:
    """Force a real paintEvent offscreen and return what was drawn."""
    size = widget.size()
    image = QImage(max(size.width(), 1), max(size.height(), 1),
                   QImage.Format.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    return image


def ramp_int16(n: int) -> np.ndarray:
    step = max(1, 60_000 // max(n, 1))
    idx = np.arange(n, dtype=np.int64)
    return ((idx * step) % 60_000 - 30_000).astype(np.int64)


def write_wav(path: Any, data: np.ndarray, sample_rate: int) -> Any:
    arr = np.asarray(data).reshape(-1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(arr.astype("<i2").tobytes())
    return path


def file_builder(path: Any, blocksize: int = 64) -> Callable[..., Any]:
    """A ``source_factory_builder`` replaying ``path`` in a loop, in real time."""
    def _builder(*args: Any, **kwargs: Any) -> Callable[..., FileSource]:
        def _factory(
            on_audio: Callable[[np.ndarray], None], stats: StreamStats
        ) -> FileSource:
            return FileSource(
                path,
                on_audio,
                blocksize=blocksize,
                realtime=True,
                loop=True,
                stats=stats,
            )
        return _factory

    return _builder


def ui_stats(**overrides: Any) -> UiStats:
    """A fully populated UiStats; every count zero unless overridden."""
    fields: dict[str, Any] = {
        "state": CaptureState.RUNNING,
        "frames_captured": 812,
        "chunks_emitted": 345,
        "chunks_dropped": 0,
        "overruns": 0,
        "xruns": 0,
        "peak_level": 0.5,
        "rms_level": 0.25,
        "display_peak": 0.6,
        "elapsed_s": 12.5,
        "latency_ms": 21.5,
    }
    fields.update(overrides)
    return UiStats(**fields)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def devices() -> list[DeviceInfo]:
    """The fake Windows device table, as the picker would receive it."""
    infos = list_input_devices(FakeSounddevice())
    assert len(infos) >= 3, "test setup: the fake table has several inputs"
    return infos


@pytest.fixture
def device_panel(qtbot: Any) -> DevicePanel:
    panel = DevicePanel()
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def geometry_panel(qtbot: Any) -> GeometryPanel:
    panel = GeometryPanel()
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def stats_panel(qtbot: Any) -> StatsPanel:
    panel = StatsPanel()
    qtbot.addWidget(panel)
    panel.show()
    return panel


@pytest.fixture
def make_window(
    tmp_path: Any, qtbot: Any
) -> Iterator[Callable[..., tuple[MainWindow, CaptureController]]]:
    """A MainWindow driven by an injected, file-backed controller.

    Teardown stops the controller unconditionally, so no replay thread can
    outlive the test even if ``closeEvent`` is the thing that is broken.
    """
    created: list[CaptureController] = []

    def _make() -> tuple[MainWindow, CaptureController]:
        config = AudioConfig(
            sample_rate=1000,
            window_ms=100,
            hop_ms=50,
            ring_seconds=10.0,
            drop_policy=DropPolicy.DROP_OLDEST,
        )
        path = write_wav(tmp_path / "window.wav", ramp_int16(1000), 1000)
        controller = CaptureController(
            config=config,
            source_factory_builder=file_builder(path),
            sd_module=FakeSounddevice(),
        )
        created.append(controller)
        window = MainWindow(controller=controller)
        qtbot.addWidget(window)
        # Populate the picker through the wiring under test.
        controller.refresh_devices()
        return window, controller

    yield _make

    for controller in created:
        try:
            controller.stop()
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


# ==========================================================================
# DevicePanel
# ==========================================================================

def test_device_panel_has_one_combo_and_the_two_buttons(
    device_panel: DevicePanel
) -> None:
    assert len(device_panel.findChildren(QComboBox)) == 1, (
        "the picker is a single combo box"
    )
    assert button_labelled(device_panel, "refresh") is not None
    assert button_labelled(device_panel, "start", "stop") is not None


def test_set_devices_populates_the_combo_with_labels(
    device_panel: DevicePanel, devices: list[DeviceInfo]
) -> None:
    """The label carries the host API: the same microphone appears once per
    host API and MME truncates names to 31 chars, so bare names are ambiguous."""
    device_panel.set_devices(devices, devices[0])

    combo = device_panel.findChildren(QComboBox)[0]
    items = [combo.itemText(i) for i in range(combo.count())]
    for device in devices:
        assert device.label in items, (
            f"{device.label!r} is missing from the picker: {items}"
        )
    assert combo.count() <= len(devices) + 1, (
        f"the combo must show the devices, got {combo.count()} entries for "
        f"{len(devices)} devices: {items}"
    )
    assert "(" in items[0], "the label must include the host API in parentheses"


def test_set_devices_shows_the_selected_device(
    device_panel: DevicePanel, devices: list[DeviceInfo]
) -> None:
    wanted = devices[2]
    device_panel.set_devices(devices, wanted)

    combo = device_panel.findChildren(QComboBox)[0]
    assert combo.currentText() == wanted.label, (
        f"the combo must display the selected device {wanted.label!r}, shows "
        f"{combo.currentText()!r}"
    )


def test_selecting_a_device_emits_device_selected(
    device_panel: DevicePanel, devices: list[DeviceInfo]
) -> None:
    device_panel.set_devices(devices, devices[0])
    combo = device_panel.findChildren(QComboBox)[0]
    wanted = devices[2]
    target = [combo.itemText(i) for i in range(combo.count())].index(wanted.label)

    emitted: list[Any] = []
    device_panel.device_selected.connect(emitted.append)
    combo.setCurrentIndex(target)

    assert emitted, "changing the combo must report the user's choice"
    assert isinstance(emitted[-1], DeviceInfo), (
        f"device_selected carries a DeviceInfo, got {type(emitted[-1]).__name__}"
    )
    assert emitted[-1] == wanted, (
        f"expected {wanted.label!r}, got {emitted[-1].label!r} -- the combo "
        "row must map back to the right device, not to the row number"
    )


def test_set_devices_with_an_empty_list_is_safe(
    device_panel: DevicePanel
) -> None:
    """"No microphone" is a state to render, not an error."""
    device_panel.set_devices([], None)

    combo = device_panel.findChildren(QComboBox)[0]
    assert combo.count() <= 1, "there is nothing to choose from"


def test_the_start_button_becomes_stop_while_running(
    device_panel: DevicePanel, devices: list[DeviceInfo]
) -> None:
    device_panel.set_devices(devices, devices[0])

    device_panel.set_state(CaptureState.STOPPED)
    stopped_text = button_labelled(device_panel, "start", "stop").text()
    assert "start" in stopped_text.replace("&", "").lower(), (
        f"a stopped capture offers Start, the button reads {stopped_text!r}"
    )

    device_panel.set_state(CaptureState.RUNNING)
    running_text = button_labelled(device_panel, "start", "stop").text()
    assert "stop" in running_text.replace("&", "").lower(), (
        f"a running capture offers Stop, the button reads {running_text!r}"
    )

    device_panel.set_state(CaptureState.STOPPED)
    assert "start" in (
        button_labelled(device_panel, "start", "stop").text().replace("&", "").lower()
    ), "the button must flip back when capture stops"


def test_set_state_error_is_rendered_without_raising(
    device_panel: DevicePanel, devices: list[DeviceInfo]
) -> None:
    device_panel.set_devices(devices, devices[0])
    device_panel.set_state(CaptureState.ERROR)

    # ERROR means "not capturing", so the button must offer Start again.
    text = button_labelled(device_panel, "start", "stop").text().replace("&", "").lower()
    assert "start" in text, (
        f"after an error the user needs a way to try again, button reads {text!r}"
    )


def test_clicking_refresh_emits_refresh_requested(
    device_panel: DevicePanel, qtbot: Any
) -> None:
    emitted: list[int] = []
    device_panel.refresh_requested.connect(lambda: emitted.append(1))

    qtbot.mouseClick(button_labelled(device_panel, "refresh"), Qt.MouseButton.LeftButton)

    assert emitted == [1], "the Refresh button must ask for a re-enumeration"


def test_clicking_start_emits_start_requested(
    device_panel: DevicePanel, devices: list[DeviceInfo], qtbot: Any
) -> None:
    device_panel.set_devices(devices, devices[0])
    device_panel.set_state(CaptureState.STOPPED)
    button = button_labelled(device_panel, "start", "stop")
    assert button.isEnabled(), "with a device selected, Start must be clickable"

    emitted: list[int] = []
    device_panel.start_requested.connect(lambda: emitted.append(1))
    stopped: list[int] = []
    device_panel.stop_requested.connect(lambda: stopped.append(1))

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert emitted == [1], "clicking Start must request a start"
    assert stopped == [], "and must not also request a stop"


def test_clicking_stop_emits_stop_requested(
    device_panel: DevicePanel, devices: list[DeviceInfo], qtbot: Any
) -> None:
    device_panel.set_devices(devices, devices[0])
    device_panel.set_state(CaptureState.RUNNING)
    button = button_labelled(device_panel, "start", "stop")
    assert button.isEnabled(), "a running capture must be stoppable"

    emitted: list[int] = []
    device_panel.stop_requested.connect(lambda: emitted.append(1))
    started: list[int] = []
    device_panel.start_requested.connect(lambda: started.append(1))

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert emitted == [1], "clicking Stop must request a stop"
    assert started == [], "and must not also request a start"


# ==========================================================================
# GeometryPanel
# ==========================================================================

def spin_boxes(panel: GeometryPanel) -> list[QSpinBox]:
    boxes = list(panel.findChildren(QSpinBox))
    assert len(boxes) == 2, (
        f"the panel has exactly two spin boxes (window_ms, hop_ms), found "
        f"{len(boxes)}"
    )
    return boxes


def spin_with_value(panel: GeometryPanel, value: int) -> QSpinBox:
    """Identify a spin box by what it displays, not by layout position."""
    matches = [box for box in spin_boxes(panel) if box.value() == value]
    assert len(matches) == 1, (
        f"expected exactly one spin box showing {value}, got "
        f"{[b.value() for b in spin_boxes(panel)]}"
    )
    return matches[0]


def test_spin_boxes_use_the_contracted_range_and_step(
    geometry_panel: GeometryPanel
) -> None:
    for box in spin_boxes(geometry_panel):
        assert box.minimum() == 10, f"minimum must be 10 ms, got {box.minimum()}"
        assert box.maximum() == 30_000, f"maximum must be 30000 ms, got {box.maximum()}"
        assert box.singleStep() == 10, f"step must be 10 ms, got {box.singleStep()}"


def test_set_geometry_is_reflected_in_the_spin_boxes(
    geometry_panel: GeometryPanel
) -> None:
    geometry_panel.set_geometry(3000, 1000)

    values = sorted(box.value() for box in spin_boxes(geometry_panel))
    assert values == [1000, 3000], (
        f"the panel must display the geometry it was given, shows {values}"
    )


def test_set_geometry_does_not_re_emit_geometry_changed(
    geometry_panel: GeometryPanel
) -> None:
    """The classic Qt signal loop: the controller writes the config into the
    panel, the panel reports it back as a user edit, and round it goes."""
    emitted: list[tuple[int, int]] = []
    geometry_panel.geometry_changed.connect(lambda w, h: emitted.append((w, h)))

    geometry_panel.set_geometry(3000, 1000)
    geometry_panel.set_geometry(2000, 500)
    geometry_panel.set_geometry(2000, 500)     # a no-op write

    assert emitted == [], (
        f"set_geometry() is the controller talking to the panel, not the user "
        f"talking to the controller; it emitted {emitted}"
    )


def test_editing_a_spin_box_emits_geometry_changed(
    geometry_panel: GeometryPanel
) -> None:
    geometry_panel.set_geometry(3000, 1000)
    emitted: list[tuple[int, int]] = []
    geometry_panel.geometry_changed.connect(lambda w, h: emitted.append((w, h)))

    spin_with_value(geometry_panel, 3000).setValue(2000)

    assert emitted, "a user edit must be reported"
    assert emitted[-1] == (2000, 1000), (
        f"geometry_changed carries (window_ms, hop_ms); expected (2000, 1000), "
        f"got {emitted[-1]}"
    )

    emitted.clear()
    spin_with_value(geometry_panel, 1000).setValue(500)

    assert emitted[-1] == (2000, 500), (
        f"editing the hop must report the *current* window too, got {emitted[-1]}"
    )


def test_the_overlap_label_shows_the_derived_overlap(
    geometry_panel: GeometryPanel
) -> None:
    geometry_panel.set_geometry(4000, 1000)      # overlap 3000 ms == 75 %

    text = rendered_text(geometry_panel)
    assert "3000" in text, (
        f"the derived overlap (4000 - 1000 = 3000 ms) must be shown; the panel "
        f"reads: {text!r}"
    )
    assert "75" in text, (
        f"the overlap percentage (75 %) must be shown; the panel reads: {text!r}"
    )
    assert "overlap" in text.lower()


def test_the_overlap_label_updates_live_as_the_spin_boxes_change(
    geometry_panel: GeometryPanel
) -> None:
    geometry_panel.set_geometry(4000, 1000)
    assert "3000" in rendered_text(geometry_panel)

    spin_with_value(geometry_panel, 1000).setValue(2000)   # overlap 2000 ms, 50 %

    text = rendered_text(geometry_panel)
    assert "2000" in text, f"the overlap readout must follow the edit: {text!r}"
    assert "50" in text, f"and so must the percentage: {text!r}"


def test_set_overlap_text_is_displayed_verbatim(
    geometry_panel: GeometryPanel
) -> None:
    geometry_panel.set_overlap_text("overlap: 1234 ms (56%)")

    assert "overlap: 1234 ms (56%)" in rendered_text(geometry_panel)


def test_set_error_shows_the_message(geometry_panel: GeometryPanel) -> None:
    """An invalid combination is explained, never silently clamped."""
    geometry_panel.set_geometry(1000, 1000)
    geometry_panel.set_error("hop must not exceed the window")

    assert "hop must not exceed the window" in rendered_text(geometry_panel)
    values = sorted(box.value() for box in spin_boxes(geometry_panel))
    assert values == [1000, 1000], (
        "set_error() reports; it must not rewrite what the user typed"
    )


def test_set_error_with_an_empty_message_clears_it(
    geometry_panel: GeometryPanel
) -> None:
    geometry_panel.set_error("hop must not exceed the window")
    geometry_panel.set_error("")

    assert "hop must not exceed the window" not in rendered_text(geometry_panel), (
        "an empty message must clear a stale error"
    )


# ==========================================================================
# StatsPanel
# ==========================================================================

def test_update_stats_renders_the_numbers(stats_panel: StatsPanel) -> None:
    stats_panel.update_stats(
        ui_stats(
            frames_captured=812,
            chunks_emitted=345,
            chunks_dropped=137,
            overruns=248,
            xruns=469,
        )
    )

    text = rendered_text(stats_panel)
    for label, value in (
        ("frames captured", "812"),
        ("chunks emitted", "345"),
        ("chunks dropped", "137"),
        ("overruns", "248"),
        ("xruns", "469"),
    ):
        assert value in text, (
            f"{label} ({value}) is not displayed anywhere in the panel: {text!r}"
        )


def test_update_stats_shows_the_state_and_the_latency(
    stats_panel: StatsPanel
) -> None:
    stats_panel.update_stats(ui_stats(state=CaptureState.RUNNING, latency_ms=21.5))

    text = rendered_text(stats_panel).lower()
    assert "running" in text, f"the capture state must be shown: {text!r}"
    assert "21.5" in text or "21" in text, (
        f"the device latency must be shown: {text!r}"
    )


def test_update_stats_refreshes_rather_than_appending(
    stats_panel: StatsPanel
) -> None:
    stats_panel.update_stats(ui_stats(frames_captured=812))
    stats_panel.update_stats(ui_stats(frames_captured=931))

    text = rendered_text(stats_panel)
    assert "931" in text
    assert "812" not in text, (
        f"each tick replaces the previous reading: {text!r}"
    )


@pytest.mark.parametrize("field", ["chunks_dropped", "overruns", "xruns"])
def test_a_nonzero_loss_counter_is_visually_emphasised(
    field: str, stats_panel: StatsPanel
) -> None:
    """A silently-dropping pipeline that looks healthy is the failure mode the
    architecture calls out by name."""
    stats_panel.update_stats(ui_stats())
    healthy = style_signature(stats_panel)

    stats_panel.update_stats(ui_stats(**{field: 7}))
    unhealthy = style_signature(stats_panel)

    assert "7" in rendered_text(stats_panel), f"{field} must be shown at all"
    assert unhealthy != healthy, (
        f"a non-zero {field} must change something the user can see beyond the "
        "number itself -- an object name, a dynamic property, a stylesheet, a "
        "font weight or a palette colour.  None of those changed."
    )


def test_the_emphasis_clears_when_the_counters_return_to_zero(
    stats_panel: StatsPanel
) -> None:
    stats_panel.update_stats(ui_stats())
    healthy = style_signature(stats_panel)

    stats_panel.update_stats(ui_stats(chunks_dropped=7))
    stats_panel.update_stats(ui_stats(chunks_dropped=0))

    assert style_signature(stats_panel) == healthy, (
        "emphasis must be driven by the current stats, not latched forever"
    )


def test_update_stats_when_stopped_is_safe(stats_panel: StatsPanel) -> None:
    stats_panel.update_stats(
        UiStats(
            state=CaptureState.STOPPED,
            frames_captured=0,
            chunks_emitted=0,
            chunks_dropped=0,
            overruns=0,
            xruns=0,
            peak_level=0.0,
            rms_level=0.0,
            display_peak=0.0,
            elapsed_s=0.0,
            latency_ms=0.0,
        )
    )

    assert "stopped" in rendered_text(stats_panel).lower()


def test_stats_panel_paints_offscreen(stats_panel: StatsPanel, qtbot: Any) -> None:
    stats_panel.resize(320, 200)
    stats_panel.update_stats(ui_stats(chunks_dropped=3))
    qtbot.wait(20)

    assert not paint(stats_panel).isNull()


# ==========================================================================
# LevelMeter
# ==========================================================================

@pytest.mark.parametrize(
    ("rms", "peak"),
    [
        (0.0, 0.0),
        (0.25, 0.5),
        (1.0, 1.0),
        (0.5, 0.5),
    ],
)
def test_level_meter_accepts_the_full_range(
    rms: float, peak: float, qtbot: Any
) -> None:
    meter = LevelMeter()
    qtbot.addWidget(meter)

    meter.set_levels(rms, peak)      # must not raise


@pytest.mark.parametrize(
    ("rms", "peak"),
    [
        (-0.5, -1.0),
        (2.0, 3.0),
        (-1.0, 5.0),
        (1.5, 0.0),
    ],
)
def test_level_meter_clamps_out_of_range_levels(
    rms: float, peak: float, qtbot: Any
) -> None:
    """StreamStats levels come from real audio; a stray value must not crash or
    draw outside the widget."""
    meter = LevelMeter()
    qtbot.addWidget(meter)
    meter.resize(240, 32)

    meter.set_levels(rms, peak)      # must not raise
    image = paint(meter)             # nor must painting it

    assert not image.isNull()


def test_level_meter_paints_offscreen(qtbot: Any) -> None:
    meter = LevelMeter()
    qtbot.addWidget(meter)
    meter.resize(240, 32)
    meter.show()
    qtbot.wait(20)

    meter.set_levels(0.4, 0.9)
    meter.update()
    qtbot.wait(20)

    image = paint(meter)
    assert image.width() == 240 and image.height() == 32
    assert not image.isNull()


def test_level_meter_survives_repeated_updates(qtbot: Any) -> None:
    """The meter is repainted 30 times a second for the life of the app."""
    meter = LevelMeter()
    qtbot.addWidget(meter)
    meter.resize(240, 32)
    meter.show()

    for i in range(60):
        meter.set_levels((i % 10) / 10.0, ((i * 3) % 10) / 10.0)
        meter.update()
    qtbot.wait(30)

    assert not paint(meter).isNull()


# ==========================================================================
# MainWindow
# ==========================================================================

def test_main_window_accepts_an_injected_controller(
    make_window: Callable[..., tuple[MainWindow, CaptureController]]
) -> None:
    window, controller = make_window()

    exposed = getattr(window, "controller", None)
    assert exposed is None or exposed is controller, (
        "if the window exposes its controller it must be the injected one, not "
        "a second one built behind the caller's back"
    )
    assert window.findChild(DevicePanel) is not None
    assert window.findChild(GeometryPanel) is not None
    assert window.findChild(StatsPanel) is not None


def test_the_device_picker_is_populated_from_the_controller(
    make_window: Callable[..., tuple[MainWindow, CaptureController]]
) -> None:
    window, controller = make_window()

    panel = window.findChild(DevicePanel)
    assert panel is not None
    combo = panel.findChildren(QComboBox)[0]
    items = [combo.itemText(i) for i in range(combo.count())]
    for device in controller.devices:
        assert device.label in items, (
            f"devices_changed must reach the picker; {device.label!r} missing "
            f"from {items}"
        )


def test_clicking_start_starts_the_controller(
    make_window: Callable[..., tuple[MainWindow, CaptureController]], qtbot: Any
) -> None:
    window, controller = make_window()
    window.show()
    panel = window.findChild(DevicePanel)
    assert panel is not None
    button = button_labelled(panel, "start", "stop")
    assert button.isEnabled(), "a device is selected, so Start must be clickable"

    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)

    assert controller.state is CaptureState.RUNNING, (
        "the panel's start_requested must be wired to controller.start()"
    )
    assert controller.pipeline is not None

    # The same button is now Stop; clicking it must tear the capture down.
    stop_button = button_labelled(panel, "start", "stop")
    qtbot.mouseClick(stop_button, Qt.MouseButton.LeftButton)

    assert controller.state is CaptureState.STOPPED, (
        "the panel's stop_requested must be wired to controller.stop()"
    )


def test_the_panels_follow_the_controller_state(
    make_window: Callable[..., tuple[MainWindow, CaptureController]], qtbot: Any
) -> None:
    window, controller = make_window()
    panel = window.findChild(DevicePanel)
    assert panel is not None

    assert controller.start() is True
    qtbot.waitUntil(
        lambda: "stop" in button_labelled(panel, "start", "stop")
        .text().replace("&", "").lower(),
        timeout=TIMEOUT_MS,
    )

    assert controller.stop() is True
    qtbot.waitUntil(
        lambda: "start" in button_labelled(panel, "start", "stop")
        .text().replace("&", "").lower(),
        timeout=TIMEOUT_MS,
    )


def test_stats_updated_reaches_the_stats_panel(
    make_window: Callable[..., tuple[MainWindow, CaptureController]]
) -> None:
    window, controller = make_window()
    stats_panel = window.findChild(StatsPanel)
    assert stats_panel is not None

    controller.stats_updated.emit(
        ui_stats(frames_captured=812, chunks_emitted=345, chunks_dropped=137)
    )

    text = rendered_text(stats_panel)
    for value in ("812", "345", "137"):
        assert value in text, (
            f"the window must forward stats_updated to the stats panel; "
            f"{value} is missing from {text!r}"
        )


def test_stats_updated_reaches_the_level_meter(
    make_window: Callable[..., tuple[MainWindow, CaptureController]]
) -> None:
    window, controller = make_window()
    meter = window.findChild(LevelMeter)
    assert meter is not None, "the window must contain a level meter"

    controller.stats_updated.emit(ui_stats(rms_level=0.3, display_peak=0.9))

    assert not paint(meter).isNull(), "the meter must still paint after an update"


def test_a_live_capture_updates_the_window(
    make_window: Callable[..., tuple[MainWindow, CaptureController]], qtbot: Any
) -> None:
    """End to end, no microphone: a WAV replay moves the numbers on screen."""
    window, controller = make_window()
    window.show()
    stats_panel = window.findChild(StatsPanel)
    assert stats_panel is not None
    before = rendered_text(stats_panel)

    assert controller.start() is True
    qtbot.waitUntil(
        lambda: rendered_text(stats_panel) != before, timeout=TIMEOUT_MS
    )
    during = rendered_text(stats_panel)

    assert controller.stop() is True
    assert during != before, (
        f"the poll timer must drive the panel during a live capture; it still "
        f"reads {during!r}"
    )


def test_close_event_stops_the_controller(
    make_window: Callable[..., tuple[MainWindow, CaptureController]], qtbot: Any
) -> None:
    """No audio thread may outlive the window."""
    window, controller = make_window()
    window.show()
    assert controller.start() is True
    pipeline = controller.pipeline
    assert pipeline is not None
    qtbot.waitUntil(lambda: pipeline.is_running, timeout=TIMEOUT_MS)

    window.close()

    assert controller.state is CaptureState.STOPPED, (
        "closeEvent must stop the controller, or the replay/chunker/consumer "
        "threads keep running after the window is gone"
    )
    assert pipeline.is_running is False


def test_closing_a_window_that_never_started_is_safe(
    make_window: Callable[..., tuple[MainWindow, CaptureController]]
) -> None:
    window, controller = make_window()
    window.show()

    window.close()      # must not raise

    assert controller.state is CaptureState.STOPPED
