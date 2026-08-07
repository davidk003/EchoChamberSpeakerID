"""The voice gate's GUI seam: backend choice, controller wiring, and the panel.

Three layers, none of which needs Vosk, a model, or a microphone:

* :mod:`echochamber.voicegate.backends` picks a recogniser from a config and is
  asserted to *return* its failures rather than raise them -- the property the
  GUI depends on to keep capturing when the gate cannot start.
* :class:`~echochamber.ui.controller.CaptureController` is driven through its
  ``recognizer_builder`` seam with a
  :class:`~echochamber.voicegate.recognizer.ScriptedRecognizer`, and through
  ``source_factory_builder`` with a
  :class:`~echochamber.audio.sources.file_source.FileSource`.  That is the whole
  pipeline -- real ring, real chunker, real bounded queue, real consumer thread
  -- with only the microphone and the decoder replaced.
* :class:`~echochamber.ui.voice_gate_panel.VoiceGatePanel` is pure rendering and
  intent, so it is tested against hand-built :class:`UiStats`.

The signal-loop test matters more than it looks: ``set_config`` is called with
the controller's own values on every sync, and a panel that echoed those back as
user intent would drive an endless round trip.
"""

from __future__ import annotations

import os
import wave

import numpy as np
import pytest

from echochamber.audio.sources.file_source import FileSource
from echochamber.audio.types import DropPolicy, StreamStats
from echochamber.config import AudioConfig
from echochamber.ui.controller import CaptureController, CaptureState, UiStats
from echochamber.ui.voice_gate_panel import (
    PHRASE_SEPARATOR,
    VoiceGatePanel,
    format_phrases,
    parse_phrases,
)
from echochamber.voicegate.backends import (
    RecognizerChoice,
    build_recognizer,
    describe_backend,
)
from echochamber.voicegate.config import VoiceGateConfig
from echochamber.voicegate.recognizer import (
    NullRecognizer,
    Recognition,
    ScriptedRecognizer,
)

pytestmark = pytest.mark.usefixtures("qapp")

SAMPLE_RATE: int = 16_000
"""Rate every fixture here runs at, matching the pipeline default."""


def _write_wav(path: str, seconds: float, sample_rate: int = SAMPLE_RATE) -> str:
    """Write a short noise WAV for :class:`FileSource` to replay.

    Args:
        path: Destination path.
        seconds: Length of the audio.
        sample_rate: Sample rate in Hz.

    Returns:
        ``path``, so this can be used inline.
    """
    rng = np.random.default_rng(0)
    samples = rng.normal(0.0, 0.1, int(seconds * sample_rate)).clip(-1.0, 1.0)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((samples * 32767.0).astype("<i2").tobytes())
    return path


def _gate_config(tmp_path, **overrides) -> VoiceGateConfig:
    """Build a gate config writing into ``tmp_path`` with fast durations.

    Args:
        tmp_path: The test's temporary directory.
        **overrides: Fields to replace.

    Returns:
        The configuration.
    """
    settings = {
        "enabled": True,
        "phrases": ("ok google",),
        "model_path": "unused-by-the-scripted-recognizer",
        "pre_roll_ms": 500,
        "post_roll_ms": 500,
        "max_snippet_ms": 5000,
        "cooldown_ms": 200,
        "snippet_dir": str(tmp_path / "snippets"),
    }
    settings.update(overrides)
    return VoiceGateConfig(**settings)


def _file_factory_builder(wav_path: str):
    """Build a ``source_factory_builder`` replaying ``wav_path`` as fast as it can.

    Args:
        wav_path: The WAV to replay.

    Returns:
        A builder accepting the controller's keyword arguments.
    """

    def builder(config=None, device=None, sd_module=None):
        def factory(on_audio, stats: StreamStats) -> FileSource:
            return FileSource(wav_path, on_audio, stats=stats, realtime=False)

        return factory

    return builder


def _scripted_builder(script, backend: str = "in-process"):
    """Build a ``recognizer_builder`` returning a scripted recogniser.

    Args:
        script: ``(byte_offset, Recognition)`` pairs.
        backend: Backend name to report.

    Returns:
        A callable matching the controller's expected signature.
    """

    def builder(config: VoiceGateConfig, sample_rate: int) -> RecognizerChoice:
        return RecognizerChoice(ScriptedRecognizer(list(script)), backend)

    return builder


def _stats(**overrides) -> UiStats:
    """Build a :class:`UiStats` with only the fields a test cares about.

    Args:
        **overrides: Fields to set.

    Returns:
        The snapshot.
    """
    settings = {
        "state": CaptureState.STOPPED,
        "frames_captured": 0,
        "chunks_emitted": 0,
        "chunks_dropped": 0,
        "overruns": 0,
        "xruns": 0,
        "peak_level": 0.0,
        "rms_level": 0.0,
        "display_peak": 0.0,
        "elapsed_s": 0.0,
        "latency_ms": 0.0,
    }
    settings.update(overrides)
    return UiStats(**settings)


class TestBuildRecognizer:
    """`build_recognizer` chooses a backend and never raises."""

    def test_disabled_gate_yields_a_null_recognizer_and_no_error(self) -> None:
        """A gate that is off is not a failure; it is simply not listening."""
        choice = build_recognizer(VoiceGateConfig(enabled=False), SAMPLE_RATE)
        assert isinstance(choice.recognizer, NullRecognizer)
        assert choice.backend == "none"
        assert choice.ok

    def test_enabled_without_a_model_reports_the_setup_script(self) -> None:
        """The error names the fix, not just the problem."""
        choice = build_recognizer(VoiceGateConfig(enabled=True), SAMPLE_RATE)
        assert not choice.ok
        assert choice.error is not None
        assert "setup_voice_gate" in choice.error

    def test_worker_python_without_a_model_is_rejected(self) -> None:
        """A worker with no model to load cannot be started."""
        config = VoiceGateConfig(
            enabled=True, worker_python="python", model_path=None
        )
        choice = build_recognizer(config, SAMPLE_RATE)
        assert not choice.ok
        assert choice.error is not None
        assert "model_path" in choice.error

    def test_a_missing_model_directory_is_returned_not_raised(self) -> None:
        """A moved model degrades the gate; it must not propagate an exception."""
        config = VoiceGateConfig(
            enabled=True, model_path="/nonexistent/model/directory"
        )
        choice = build_recognizer(config, SAMPLE_RATE)
        assert not choice.ok
        assert isinstance(choice.recognizer, NullRecognizer)

    def test_a_failed_launch_is_returned_not_raised(self, tmp_path) -> None:
        """An interpreter that does not exist is a message, not a traceback."""
        model = tmp_path / "model"
        model.mkdir()
        config = VoiceGateConfig(
            enabled=True,
            model_path=str(model),
            worker_python=str(tmp_path / "no-such-python"),
            startup_timeout_s=2.0,
        )
        choice = build_recognizer(config, SAMPLE_RATE)
        assert not choice.ok
        assert isinstance(choice.recognizer, NullRecognizer)
        assert choice.backend == "none"

    def test_describe_backend_distinguishes_off_from_broken(self) -> None:
        """'off' and 'failed to start' must not read the same in a status bar."""
        off = describe_backend(RecognizerChoice(NullRecognizer(), "none"))
        broken = describe_backend(
            RecognizerChoice(NullRecognizer(), "none", "model missing")
        )
        listening = describe_backend(
            RecognizerChoice(NullRecognizer(), "subprocess")
        )
        assert off == "voice gate off"
        assert "model missing" in broken
        assert "subprocess" in listening


class TestControllerGateWiring:
    """The controller composes the gate and surfaces its counters."""

    def test_a_disabled_gate_wires_no_sink(self, tmp_path) -> None:
        """The default costs nothing: no gate sink, no snippet directory."""
        wav = _write_wav(str(tmp_path / "in.wav"), 1.0)
        controller = CaptureController(
            config=AudioConfig(window_ms=500, hop_ms=250),
            source_factory_builder=_file_factory_builder(wav),
            voice_gate=_gate_config(tmp_path, enabled=False),
        )
        assert controller.start()
        try:
            assert controller.gate_sink is None
            assert controller.poll().gate_backend == "none"
        finally:
            controller.stop()
        assert not (tmp_path / "snippets").exists()

    def test_a_matching_phrase_produces_a_snippet_and_reaches_ui_stats(
        self, tmp_path
    ) -> None:
        """The whole path: file replay, real chunker, gate, WAV, UiStats."""
        wav = _write_wav(str(tmp_path / "in.wav"), 4.0)
        gate = _gate_config(tmp_path)
        controller = CaptureController(
            config=AudioConfig(
                window_ms=1000, hop_ms=500, drop_policy=DropPolicy.BLOCK
            ),
            source_factory_builder=_file_factory_builder(wav),
            voice_gate=gate,
            # 32000 bytes == 1 s of 16-bit mono at 16 kHz.
            recognizer_builder=_scripted_builder(
                [(32_000, Recognition("ok google turn it up", final=True))]
            ),
        )
        assert controller.start()
        try:
            assert controller.pipeline is not None
            controller.pipeline.wait_until_finished(timeout=30)
        finally:
            controller.stop()

        stats = controller.poll()
        assert stats.gate_backend == "in-process"
        assert stats.gate_detected == 1
        assert stats.gate_snippets == 1
        assert stats.gate_last_phrase == "ok google"
        assert stats.gate_error is None
        assert stats.gate_last_path is not None
        assert os.path.isfile(stats.gate_last_path)

        with wave.open(stats.gate_last_path) as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == SAMPLE_RATE
            # 500 ms of pre-roll plus 500 ms of post-roll.
            assert handle.getnframes() == pytest.approx(SAMPLE_RATE, rel=0.35)

    def test_gate_counters_survive_stop(self, tmp_path) -> None:
        """The run that just ended is exactly when you read what it caught."""
        wav = _write_wav(str(tmp_path / "in.wav"), 3.0)
        controller = CaptureController(
            config=AudioConfig(
                window_ms=1000, hop_ms=500, drop_policy=DropPolicy.BLOCK
            ),
            source_factory_builder=_file_factory_builder(wav),
            voice_gate=_gate_config(tmp_path),
            recognizer_builder=_scripted_builder(
                [(32_000, Recognition("ok google", final=True))]
            ),
        )
        assert controller.start()
        assert controller.pipeline is not None
        controller.pipeline.wait_until_finished(timeout=30)
        controller.stop()

        assert controller.state is CaptureState.STOPPED
        assert controller.gate_sink is None
        assert controller.poll().gate_snippets == 1

    def test_a_recognizer_that_will_not_start_does_not_fail_the_capture(
        self, tmp_path
    ) -> None:
        """Recording without gating beats not recording at all."""
        wav = _write_wav(str(tmp_path / "in.wav"), 1.0)
        messages: list[str] = []

        def failing_builder(config, sample_rate):
            return RecognizerChoice(NullRecognizer(), "none", "model went missing")

        controller = CaptureController(
            config=AudioConfig(window_ms=500, hop_ms=250),
            source_factory_builder=_file_factory_builder(wav),
            voice_gate=_gate_config(tmp_path),
            recognizer_builder=failing_builder,
        )
        controller.error_occurred.connect(messages.append)
        assert controller.start()
        try:
            assert controller.state is CaptureState.RUNNING
            assert controller.gate_sink is None
            assert any("model went missing" in text for text in messages)
        finally:
            controller.stop()

    def test_set_voice_gate_enabled_rejects_a_config_with_no_usable_phrase(
        self,
    ) -> None:
        """Enabling with nothing to listen for is refused, and says so."""
        messages: list[str] = []
        controller = CaptureController(
            voice_gate=VoiceGateConfig(enabled=False, phrases=("!!!",))
        )
        controller.error_occurred.connect(messages.append)

        assert controller.set_voice_gate_enabled(True) is False
        assert controller.voice_gate_config.enabled is False
        assert messages

    def test_set_voice_gate_phrases_replaces_them(self) -> None:
        """Accepted phrases are adopted for the next run."""
        controller = CaptureController(voice_gate=VoiceGateConfig(enabled=True))
        assert controller.set_voice_gate_phrases(("hey computer",))
        assert controller.voice_gate_config.phrases == ("hey computer",)

    def test_set_voice_gate_phrases_leaves_the_config_alone_when_rejected(
        self,
    ) -> None:
        """A rejected edit must not half-apply."""
        controller = CaptureController(
            voice_gate=VoiceGateConfig(enabled=True, phrases=("ok google",))
        )
        assert controller.set_voice_gate_phrases(("???",)) is False
        assert controller.voice_gate_config.phrases == ("ok google",)

    def test_set_voice_gate_rejects_none(self) -> None:
        """A None config is refused rather than replacing a working one."""
        controller = CaptureController(voice_gate=VoiceGateConfig())
        assert controller.set_voice_gate(None) is False


class TestPhraseFieldParsing:
    """The comma-separated phrase field round-trips."""

    def test_phrases_are_split_and_stripped(self) -> None:
        """Whitespace around a phrase is not part of it."""
        assert parse_phrases("ok google,  hey google ") == (
            "ok google",
            "hey google",
        )

    def test_empty_entries_are_dropped(self) -> None:
        """A trailing or doubled comma is tolerated, not turned into a phrase."""
        assert parse_phrases("ok google,,  ,") == ("ok google",)

    def test_an_empty_field_yields_no_phrases(self) -> None:
        """Nothing typed means nothing configured."""
        assert parse_phrases("   ") == ()

    def test_a_phrase_keeps_its_internal_spaces(self) -> None:
        """'ok google' is one phrase; splitting on whitespace would break it."""
        assert parse_phrases("ok google") == ("ok google",)

    def test_format_round_trips_through_parse(self) -> None:
        """What the panel displays is what it reads back."""
        phrases = ("ok google", "hey google")
        assert parse_phrases(format_phrases(phrases)) == phrases

    def test_format_uses_the_documented_separator(self) -> None:
        """The separator is a comma, as the placeholder text promises."""
        assert PHRASE_SEPARATOR in format_phrases(("a", "b"))


class TestVoiceGatePanel:
    """The panel renders stats and emits intent, and computes nothing."""

    def test_set_config_does_not_emit(self, qtbot) -> None:
        """Echoing the controller's own values back would be a signal loop."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)
        seen: list[object] = []
        panel.enabled_changed.connect(seen.append)
        panel.phrases_changed.connect(seen.append)

        panel.set_config(True, ("ok google", "hey google"))

        assert seen == []
        assert panel.is_enabled() is True
        assert panel.phrases() == ("ok google", "hey google")

    def test_a_user_toggle_emits(self, qtbot) -> None:
        """A real click is intent and must reach the controller."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)
        seen: list[bool] = []
        panel.enabled_changed.connect(seen.append)

        panel.enable_check.setChecked(True)

        assert seen == [True]

    def test_editing_phrases_emits_the_parsed_tuple(self, qtbot) -> None:
        """The panel hands over phrases, not raw text."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)
        seen: list[tuple[str, ...]] = []
        panel.phrases_changed.connect(seen.append)

        panel.phrase_edit.setText("ok google, hey google")
        panel.phrase_edit.editingFinished.emit()

        assert seen == [("ok google", "hey google")]

    def test_update_stats_renders_every_counter(self, qtbot) -> None:
        """Each value label shows its field."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)

        panel.update_stats(
            _stats(
                gate_backend="subprocess",
                gate_detected=3,
                gate_snippets=2,
                gate_suppressed=1,
                gate_truncated=4,
                gate_last_phrase="ok google",
            )
        )

        assert panel.value_label("gate_backend").text() == "subprocess"
        assert panel.value_label("gate_detected").text() == "3"
        assert panel.value_label("gate_snippets").text() == "2"
        assert panel.value_label("gate_suppressed").text() == "1"
        assert panel.value_label("gate_truncated").text() == "4"
        assert panel.value_label("gate_last_phrase").text() == "ok google"

    def test_no_phrase_yet_renders_a_dash_not_a_blank(self, qtbot) -> None:
        """An empty cell next to a label reads as a rendering failure."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)
        panel.update_stats(_stats())
        assert panel.value_label("gate_last_phrase").text() == "—"

    def test_the_snippet_counter_is_emphasised_once_it_moves(self, qtbot) -> None:
        """The count of snippets is the number the user is actually watching."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)

        panel.update_stats(_stats(gate_snippets=0))
        assert panel.value_label("gate_snippets").styleSheet() == ""

        panel.update_stats(_stats(gate_snippets=1))
        assert panel.value_label("gate_snippets").styleSheet() != ""

    def test_a_gate_error_is_shown_as_a_warning(self, qtbot) -> None:
        """A gate that is broken must not look like one that hears nothing."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)

        panel.update_stats(_stats(gate_error="recognition failed: boom"))

        assert "boom" in panel.note_label.text()
        assert panel.note_label.styleSheet() != ""

    def test_set_editable_locks_the_inputs(self, qtbot) -> None:
        """Settings cannot take effect mid-run, so they are not offered."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)

        panel.set_editable(False)
        assert not panel.enable_check.isEnabled()
        assert not panel.phrase_edit.isEnabled()

        panel.set_editable(True)
        assert panel.enable_check.isEnabled()
        assert panel.phrase_edit.isEnabled()

    def test_set_note_clears_with_an_empty_message(self, qtbot) -> None:
        """A stale note is worse than none."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)

        panel.set_note("something", warning=True)
        panel.set_note("")

        assert panel.note_label.text() == ""
        assert panel.note_label.styleSheet() == ""

    def test_repr_names_the_state(self, qtbot) -> None:
        """The repr is for debugging, so it says what the panel holds."""
        panel = VoiceGatePanel()
        qtbot.addWidget(panel)
        panel.set_config(True, ("ok google",))
        assert "enabled=True" in repr(panel)


class TestMainWindowIntegration:
    """The window forwards gate intent and renders gate stats."""

    def test_the_panel_is_wired_to_the_controller(self, qtbot) -> None:
        """A toggle in the panel reaches the controller's configuration."""
        from echochamber.ui.main_window import MainWindow

        controller = CaptureController(
            voice_gate=VoiceGateConfig(enabled=False, phrases=("ok google",)),
            sd_module=_NoDevices(),
        )
        window = MainWindow(controller=controller)
        qtbot.addWidget(window)

        window.voice_gate_panel.enable_check.setChecked(True)

        assert controller.voice_gate_config.enabled is True

    def test_a_rejected_phrase_edit_is_rolled_back_in_the_panel(
        self, qtbot
    ) -> None:
        """The field must never show a value the controller refused."""
        from echochamber.ui.main_window import MainWindow

        controller = CaptureController(
            voice_gate=VoiceGateConfig(enabled=True, phrases=("ok google",)),
            sd_module=_NoDevices(),
        )
        window = MainWindow(controller=controller)
        qtbot.addWidget(window)

        window.voice_gate_panel.phrase_edit.setText("???")
        window.voice_gate_panel.phrase_edit.editingFinished.emit()

        assert controller.voice_gate_config.phrases == ("ok google",)
        assert window.voice_gate_panel.phrases() == ("ok google",)


class _NoDevices:
    """A stand-in :mod:`sounddevice` reporting no input devices.

    The window enumerates devices on construction; without this the real
    PortAudio would be loaded, which is exactly what the injection seam exists
    to avoid.
    """

    def query_devices(self, *args, **kwargs):
        """Return no devices.

        Returns:
            An empty list.
        """
        return []

    def query_hostapis(self, *args, **kwargs):
        """Return no host APIs.

        Returns:
            An empty list.
        """
        return []

    @property
    def default(self):
        """Return a device record naming no defaults."""
        return _NoDefaults()


class _NoDefaults:
    """The ``default`` attribute of :class:`_NoDevices`."""

    device = (-1, -1)
