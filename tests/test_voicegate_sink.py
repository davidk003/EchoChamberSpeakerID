"""Tests for echochamber.voicegate.sink -- the gate driven end to end, no Vosk.

The gate is exercised exactly the way the pipeline drives it: hand-built
overlapping windows pushed straight into ``on_chunk``.  At 16 kHz with the
shipping 3000 ms / 1000 ms geometry that is ``W = 48000`` frames per chunk and
``H = 16000`` frames of hop, so chunk *k* covers absolute frames
``[k*H, k*H + W)`` and contributes ``W`` new frames for *k = 0* and ``H``
thereafter.  ``_chunk`` builds those; ``_feed`` pushes a run of them.

Recognition is supplied by
:class:`~echochamber.voicegate.recognizer.ScriptedRecognizer`, which emits a
chosen :class:`~echochamber.voicegate.recognizer.Recognition` once a chosen
number of PCM bytes has been consumed.  ``_bytes_after(k)`` converts "after
chunk *k*" into that byte offset, so a test says *when* a phrase is heard
rather than *how many bytes* it took.  **Nothing here imports vosk and nothing
needs a model on disk**; that is the property the whole voicegate package was
factored to preserve.

Two further habits worth naming:

* **Snippet sizes are asserted in exact frames, and cross-checked against the
  WAV read back off disk.**  A gate that writes the pre-roll twice, or writes
  the whole window instead of the de-overlapped tail, still produces a
  plausible-looking file -- only an exact frame count catches it.  The audio
  itself is a global ramp, so a snippet's *content* also identifies which
  absolute frames landed in it.
* **``on_chunk`` must never raise.**  Two deliberately broken collaborators (a
  recogniser that throws and a callback that throws) prove the failure is
  recorded in :attr:`~VoiceGateSink.error` and swallowed, because an exception
  on the consumer thread stops the queue draining and takes capture down.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from echochamber.audio.sinks import ChunkSink
from echochamber.audio.types import AudioChunk
from echochamber.voicegate.config import VoiceGateConfig
from echochamber.voicegate.recognizer import (
    NullRecognizer,
    Recognition,
    ScriptedRecognizer,
    float32_to_pcm16,
)
from echochamber.voicegate.sink import SnippetEvent, VoiceGateSink, VoiceGateStats
from echochamber.voicegate.speaker import VerifyResult


SR = 16_000
WINDOW = 48_000        # 3000 ms at 16 kHz
HOP = 16_000           # 1000 ms at 16 kHz
BYTES_PER_FRAME = 2

# Long enough for every chunk index and every deliberate gap used below.
_TOTAL_FRAMES = 400_000

# A strictly increasing ramp over the whole stream: a snippet's samples
# therefore say which absolute frames it was assembled from, so writing the
# wrong window -- or writing the pre-roll twice -- shows up as wrong content
# and not merely as a wrong length.
SIGNAL: np.ndarray = np.linspace(-0.9, 0.9, _TOTAL_FRAMES, dtype=np.float32)

PHRASE_TEXT = "ok google turn it up"


class Boom(Exception):
    """Sentinel raised by a deliberately broken collaborator."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _chunk(k: int, *, discontinuous: bool = False) -> AudioChunk:
    """Chunk ``k`` of the standard overlapping grid: start ``k*H``, length ``W``."""
    start = k * HOP
    return AudioChunk(
        samples=SIGNAL[start : start + WINDOW],
        start_frame=start,
        seq=k,
        sample_rate=SR,
        discontinuous=discontinuous,
    )


def _chunk_at(
    start_frame: int, seq: int = 0, n_frames: int = WINDOW
) -> AudioChunk:
    """A window at an arbitrary absolute position, for gap and stale cases."""
    return AudioChunk(
        samples=SIGNAL[start_frame : start_frame + n_frames],
        start_frame=start_frame,
        seq=seq,
        sample_rate=SR,
    )


def _feed(sink: VoiceGateSink, count: int, *, first: int = 0) -> None:
    """Push chunks ``first`` .. ``first + count - 1`` into ``sink``."""
    for k in range(first, first + count):
        sink.on_chunk(_chunk(k))


def _bytes_after(k: int) -> int:
    """PCM bytes the recogniser has consumed once chunk ``k`` has been fed.

    Chunk 0 contributes all ``W`` of its frames and every later chunk exactly
    ``H``, so this is the offset a scripted result must carry to be emitted by
    chunk ``k`` and no earlier.
    """
    return (WINDOW + k * HOP) * BYTES_PER_FRAME


def _final(text: str = PHRASE_TEXT) -> Recognition:
    """A settled recognition carrying ``text``."""
    return Recognition(text=text, final=True)


def _partial(text: str = PHRASE_TEXT) -> Recognition:
    """An unsettled recognition carrying ``text``."""
    return Recognition(text=text, final=False)


def _config(tmp_path: Path, **overrides: Any) -> VoiceGateConfig:
    """An enabled gate config writing under ``tmp_path``, with short durations.

    The defaults here are deliberately much shorter than the shipping ones so a
    snippet opens and closes within two or three hand-fed chunks.
    """
    kwargs: dict[str, Any] = dict(
        enabled=True,
        phrases=("ok google",),
        pre_roll_ms=500,          # 8000 frames
        post_roll_ms=1000,        # 16000 frames == exactly one hop
        max_snippet_ms=10_000,    # 160000 frames: effectively no ceiling
        cooldown_ms=0,
        snippet_dir=str(tmp_path / "snippets"),
    )
    kwargs.update(overrides)
    return VoiceGateConfig(**kwargs)


def _make_sink(
    config: VoiceGateConfig,
    script: list[tuple[int, Recognition]] | None = None,
    events: list[SnippetEvent] | None = None,
    **kwargs: Any,
) -> tuple[VoiceGateSink, ScriptedRecognizer]:
    """Build a gate over a ScriptedRecognizer, returning both."""
    recognizer = ScriptedRecognizer(script)
    sink = VoiceGateSink(
        config,
        SR,
        recognizer=recognizer,
        on_snippet=None if events is None else events.append,
        **kwargs,
    )
    return sink, recognizer


def _wavs(config: VoiceGateConfig) -> list[Path]:
    """Every ``.wav`` in the config's snippet directory, sorted by name."""
    directory = Path(config.snippet_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.wav"))


def _read_wav(path: Path) -> tuple[int, int, int, int, bytes]:
    """Read a WAV back as (channels, sampwidth, framerate, nframes, raw)."""
    with wave.open(str(path), "rb") as handle:
        return (
            handle.getnchannels(),
            handle.getsampwidth(),
            handle.getframerate(),
            handle.getnframes(),
            handle.readframes(handle.getnframes()),
        )


def _expected_pcm(start: int, stop: int) -> bytes:
    """The PCM the gate must have written for absolute frames ``[start, stop)``."""
    return float32_to_pcm16(SIGNAL[start:stop])


class _BoomRecognizer:
    """A recogniser whose ``accept_pcm`` always raises.

    Stands in for a decoder that faults mid-stream: the gate has to degrade to
    "not gating" rather than let the exception reach the consumer loop.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0
        self.closed = False

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        self.calls += 1
        raise Boom("the decoder faulted")

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True


class _CloseBoomRecognizer:
    """A recogniser that recognises nothing and fails to close."""

    def __init__(self) -> None:
        self.closed = False

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        return []

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        raise Boom("the decoder would not shut down")


class _ResetBoomRecognizer:
    """A recogniser whose ``reset`` raises, exercised by a discontinuity."""

    def __init__(self) -> None:
        self.resets = 0
        self.closed = False

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        return []

    def reset(self) -> None:
        self.resets += 1
        raise Boom("reset failed")

    def close(self) -> None:
        self.closed = True


def _boom_callback(event: SnippetEvent) -> None:
    """An ``on_snippet`` callback that always raises."""
    raise Boom("the callback failed")


# ==========================================================================
# construction and the inert cases
# ==========================================================================

class TestConstruction:
    """What the gate is before any audio reaches it."""

    def test_counters_start_at_zero(self, tmp_path: Path) -> None:
        """A fresh gate has processed nothing and failed at nothing."""
        sink, _ = _make_sink(_config(tmp_path))

        assert sink.frames_processed == 0
        assert sink.phrases_detected == 0
        assert sink.snippets_written == 0
        assert sink.snippets_suppressed == 0
        assert sink.snippets_truncated == 0
        assert sink.gaps == 0
        assert sink.last_phrase == ""
        assert sink.last_snippet_path is None
        assert sink.error is None
        assert sink.recording is False
        assert sink.closed is False

    def test_the_recognizer_defaults_to_a_null_one(self, tmp_path: Path) -> None:
        """A gate built without a model is inert, not broken."""
        sink = VoiceGateSink(_config(tmp_path), SR)
        assert isinstance(sink.recognizer, NullRecognizer)

    def test_a_gate_with_the_null_recognizer_never_fires(
        self, tmp_path: Path
    ) -> None:
        """No decoder means no matches, but audio still flows through."""
        config = _config(tmp_path)
        sink = VoiceGateSink(config, SR)
        _feed(sink, 4)

        assert sink.frames_processed == WINDOW + 3 * HOP
        assert sink.snippets_written == 0
        assert sink.error is None
        assert _wavs(config) == []

    def test_config_and_sample_rate_are_reported_back(
        self, tmp_path: Path
    ) -> None:
        """The gate holds the frozen config it was given, not a copy of its fields."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config)

        assert sink.config is config
        assert sink.sample_rate == SR

    @pytest.mark.parametrize("rate", [0, -1, -16_000])
    def test_non_positive_sample_rate_is_rejected(
        self, tmp_path: Path, rate: int
    ) -> None:
        """A zero rate would make every duration convert to zero frames."""
        with pytest.raises(ValueError, match=r"sample_rate must be > 0"):
            VoiceGateSink(_config(tmp_path), rate)

    def test_the_gate_satisfies_the_chunksink_protocol(
        self, tmp_path: Path
    ) -> None:
        """It has to be tee-able alongside the recorder and the meters."""
        sink, _ = _make_sink(_config(tmp_path))
        assert isinstance(sink, ChunkSink)

    def test_repr_reports_the_gate_state(self, tmp_path: Path) -> None:
        """The repr is for debugging a gate that is not firing."""
        sink, _ = _make_sink(_config(tmp_path))
        text = repr(sink)

        assert "VoiceGateSink" in text
        assert "enabled=True" in text
        assert "recording=False" in text


class TestDisabled:
    """A disabled gate is not merely quiet: it does no work at all."""

    def test_on_chunk_does_nothing(self, tmp_path: Path) -> None:
        """Not one counter moves, so a wired-in but disabled gate costs nothing."""
        config = _config(tmp_path, enabled=False)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 5)

        assert sink.frames_processed == 0, (
            "a disabled gate must return before it counts anything"
        )
        assert sink.phrases_detected == 0
        assert sink.snippets_written == 0
        assert sink.gaps == 0
        assert sink.error is None

    def test_no_pcm_reaches_the_recognizer(self, tmp_path: Path) -> None:
        """The decoder is the expensive part; a disabled gate must not run it."""
        config = _config(tmp_path, enabled=False)
        sink, recognizer = _make_sink(config, [(1, _final())])

        _feed(sink, 5)

        assert recognizer.consumed == 0, (
            f"a disabled gate fed the recogniser {recognizer.consumed} bytes"
        )
        assert recognizer.resets == 0

    def test_the_snippet_directory_is_not_created(self, tmp_path: Path) -> None:
        """A disabled gate leaves no trace on the filesystem."""
        config = _config(tmp_path, enabled=False)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 5)

        assert not os.path.exists(config.snippet_dir), (
            "a disabled gate must not create its snippet directory"
        )

    def test_a_discontinuous_chunk_is_ignored_too(self, tmp_path: Path) -> None:
        """The early return happens before the discontinuity handling."""
        config = _config(tmp_path, enabled=False)
        sink, recognizer = _make_sink(config)

        sink.on_chunk(_chunk(0, discontinuous=True))
        assert recognizer.resets == 0


# ==========================================================================
# de-overlapping
# ==========================================================================

class TestDeOverlapping:
    """The gate counts and forwards NEW frames, never the repeated overlap."""

    @pytest.mark.parametrize("count", [1, 2, 3, 5, 8])
    def test_frames_processed_is_the_new_frames_not_the_windows(
        self, tmp_path: Path, count: int
    ) -> None:
        """First chunk contributes W, every later one exactly H.

        Counting ``count * W`` instead would mean the recogniser is being fed
        the overlap again, i.e. hearing every word three times.
        """
        sink, _ = _make_sink(_config(tmp_path))
        _feed(sink, count)

        expected = WINDOW + (count - 1) * HOP
        assert sink.frames_processed == expected, (
            f"{count} overlapping windows cover {expected} frames, not "
            f"{count * WINDOW}"
        )

    @pytest.mark.parametrize("count", [1, 2, 4, 6])
    def test_the_recognizer_receives_exactly_those_frames_as_pcm(
        self, tmp_path: Path, count: int
    ) -> None:
        """Bytes fed to the decoder track frames_processed at 2 bytes a frame."""
        sink, recognizer = _make_sink(_config(tmp_path))
        _feed(sink, count)

        assert recognizer.consumed == _bytes_after(count - 1)
        assert recognizer.consumed == sink.frames_processed * BYTES_PER_FRAME

    def test_the_first_chunk_contributes_all_of_itself(
        self, tmp_path: Path
    ) -> None:
        """next_expected starts at the first chunk's start_frame, so nothing is lost."""
        sink, _ = _make_sink(_config(tmp_path))
        sink.on_chunk(_chunk(0))

        assert sink.frames_processed == WINDOW
        assert sink.gaps == 0

    def test_a_first_chunk_starting_late_is_not_a_gap(
        self, tmp_path: Path
    ) -> None:
        """A stream whose first delivered window starts at frame 5000 is not a hole."""
        sink, _ = _make_sink(_config(tmp_path))
        sink.on_chunk(_chunk_at(5000))

        assert sink.frames_processed == WINDOW
        assert sink.gaps == 0

    def test_a_stale_chunk_contributes_nothing(self, tmp_path: Path) -> None:
        """A replayed window is already covered, so it adds no frames and no gap."""
        sink, recognizer = _make_sink(_config(tmp_path))
        _feed(sink, 2)
        before_frames = sink.frames_processed
        before_bytes = recognizer.consumed

        sink.on_chunk(_chunk(0))                # replay of an old window

        assert sink.frames_processed == before_frames
        assert recognizer.consumed == before_bytes, (
            "a stale chunk must not be fed to the recogniser a second time"
        )
        assert sink.gaps == 0, "a stale chunk is not a gap"


class TestGaps:
    """gaps counts the times audio was lost upstream of the gate."""

    def test_a_chunk_past_the_expected_frame_counts_one_gap(
        self, tmp_path: Path
    ) -> None:
        """A DROP_OLDEST queue loses whole chunks; this is how the gate says so."""
        sink, _ = _make_sink(_config(tmp_path))
        sink.on_chunk(_chunk(0))                       # covers [0, 48000)
        sink.on_chunk(_chunk_at(60_000, seq=1))        # 12000-frame hole

        assert sink.gaps == 1, f"one hole means gaps == 1, got {sink.gaps}"
        assert sink.frames_processed == 2 * WINDOW, (
            "after a gap the whole chunk is new"
        )

    def test_each_discontinuity_is_counted_separately(
        self, tmp_path: Path
    ) -> None:
        """Two holes are two gaps, and a contiguous chunk in between is not one."""
        sink, _ = _make_sink(_config(tmp_path))
        sink.on_chunk(_chunk_at(0, seq=0))                     # [0, 48000)
        sink.on_chunk(_chunk_at(60_000, seq=1))                # gap
        sink.on_chunk(_chunk_at(108_000, seq=2))               # contiguous
        sink.on_chunk(_chunk_at(200_000, seq=3))               # gap

        assert sink.gaps == 2

    def test_a_contiguous_grid_has_no_gaps(self, tmp_path: Path) -> None:
        """The ordinary hop grid never reports a hole."""
        sink, _ = _make_sink(_config(tmp_path))
        _feed(sink, 8)
        assert sink.gaps == 0


# ==========================================================================
# matching -> snippets
# ==========================================================================

class TestSnippetOpening:
    """A matching final result writes exactly one snippet file."""

    def test_a_matching_final_writes_a_snippet(self, tmp_path: Path) -> None:
        """The headline test: match, file on disk, counters and event agree."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config, [(_bytes_after(0), _final())], events
        )

        _feed(sink, 2)

        files = _wavs(config)
        assert len(files) == 1, f"expected exactly one snippet, got {files}"
        assert sink.snippets_written == 1
        assert sink.phrases_detected == 1
        assert sink.snippets_suppressed == 0
        assert sink.snippets_truncated == 0
        assert sink.last_phrase == "ok google"
        assert sink.last_snippet_path == str(files[0])
        assert sink.error is None
        assert sink.recording is False, "the snippet closed when its post-roll ran out"

        assert len(events) == 1, "on_snippet must be called once per finished file"
        event = events[0]
        assert Path(event.path).exists()
        assert Path(event.path) == files[0]
        assert event.phrase == "ok google"
        assert event.text == PHRASE_TEXT
        assert event.seq == 0
        assert event.truncated is False

        channels, sampwidth, framerate, nframes, _ = _read_wav(files[0])
        assert (channels, sampwidth, framerate) == (1, 2, SR)
        assert event.frames == nframes, (
            f"the event claims {event.frames} frames but the WAV holds {nframes}"
        )
        assert event.duration_s == pytest.approx(nframes / SR)

    def test_the_snippet_is_pre_roll_plus_post_roll_exactly(
        self, tmp_path: Path
    ) -> None:
        """Length is decided by the configuration, not by the chunk cadence."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 2)

        expected = config.pre_roll_frames(SR) + config.post_roll_frames(SR)
        assert expected == 8_000 + 16_000, "test setup: 500 ms pre + 1000 ms post"
        assert events[0].frames == expected
        assert _read_wav(_wavs(config)[0])[3] == expected

    def test_the_snippet_holds_the_audio_around_the_match(
        self, tmp_path: Path
    ) -> None:
        """Content, not just length: the exact absolute frames must be in the file."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 2)

        # The match landed after chunk 0, i.e. at absolute frame 48000; the
        # pre-roll reaches 8000 frames back and the post-roll runs 16000 on.
        _, _, _, nframes, raw = _read_wav(_wavs(config)[0])
        assert nframes == 24_000
        assert raw == _expected_pcm(40_000, 64_000), (
            "the snippet must be the de-overlapped stream around the match, not "
            "a replayed window"
        )

    def test_start_frame_points_at_the_beginning_of_the_pre_roll(
        self, tmp_path: Path
    ) -> None:
        """start_frame is where the FILE begins, not where the phrase was heard."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 2)

        assert events[0].start_frame == WINDOW - config.pre_roll_frames(SR)
        assert events[0].start_frame == 40_000

    def test_the_snippet_directory_is_created_on_demand(
        self, tmp_path: Path
    ) -> None:
        """The directory appears when the first snippet does, not before."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(_bytes_after(1), _final())])

        sink.on_chunk(_chunk(0))
        assert not os.path.exists(config.snippet_dir)

        _feed(sink, 2, first=1)
        assert os.path.isdir(config.snippet_dir)

    def test_the_filename_carries_the_sequence_and_the_phrase(
        self, tmp_path: Path
    ) -> None:
        """A log line and a directory listing have to line up."""
        config = _config(tmp_path)
        sink, _ = _make_sink(
            config, [(_bytes_after(0), _final())], clock=lambda: 1_700_000_000.0
        )

        _feed(sink, 2)

        name = _wavs(config)[0].name
        assert "_0000_" in name, f"the seq must be zero-padded in {name!r}"
        assert name.endswith("_ok-google.wav"), f"unexpected snippet name {name!r}"

    def test_a_second_match_after_the_first_closes_gets_the_next_sequence(
        self, tmp_path: Path
    ) -> None:
        """Snippet counters run from 0 and never repeat within one sink."""
        config = _config(tmp_path, cooldown_ms=0)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(2), _final())],
            events,
        )

        _feed(sink, 4)

        assert [event.seq for event in events] == [0, 1]
        assert sink.snippets_written == 2
        assert len(_wavs(config)) == 2


class TestPreRoll:
    """The pre-roll is what puts the wake phrase itself into the file."""

    def test_a_pre_roll_adds_audio_from_before_the_match(
        self, tmp_path: Path
    ) -> None:
        """Quantitative: the file is exactly pre_roll_frames longer than without.

        A decoder reports a phrase only after consuming it, so a snippet that
        began at the match would start *after* the words it is named for.
        """
        with_pre = _config(tmp_path / "with", pre_roll_ms=500)
        without_pre = _config(tmp_path / "without", pre_roll_ms=0)

        sizes: dict[str, int] = {}
        for label, config in (("with", with_pre), ("without", without_pre)):
            events: list[SnippetEvent] = []
            sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)
            _feed(sink, 2)
            sizes[label] = events[0].frames

        assert sizes["without"] == 16_000, (
            "with no pre-roll the file is exactly the post-roll"
        )
        assert sizes["with"] == 24_000
        assert sizes["with"] - sizes["without"] == with_pre.pre_roll_frames(SR), (
            f"a 500 ms pre-roll must add exactly 8000 frames, added "
            f"{sizes['with'] - sizes['without']}"
        )

    @pytest.mark.parametrize(
        ("pre_roll_ms", "pre_frames"),
        [(0, 0), (250, 4_000), (500, 8_000), (1000, 16_000), (2000, 32_000)],
    )
    def test_the_pre_roll_length_follows_the_configuration(
        self, tmp_path: Path, pre_roll_ms: int, pre_frames: int
    ) -> None:
        """Every configured pre-roll lands in the file frame for frame."""
        config = _config(tmp_path, pre_roll_ms=pre_roll_ms)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 2)

        assert events[0].frames == pre_frames + config.post_roll_frames(SR)
        assert events[0].start_frame == WINDOW - pre_frames

    def test_the_pre_roll_audio_precedes_the_match(self, tmp_path: Path) -> None:
        """The file's first samples come from before the match, not after it."""
        config = _config(tmp_path, pre_roll_ms=500)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 2)

        _, _, _, _, raw = _read_wav(_wavs(config)[0])
        assert raw[:2] == _expected_pcm(40_000, 40_001), (
            "the snippet must open 8000 frames before the match at frame 48000"
        )

    def test_a_pre_roll_longer_than_the_audio_so_far_is_simply_shorter(
        self, tmp_path: Path
    ) -> None:
        """The buffer can only hold what has actually been captured."""
        config = _config(tmp_path, pre_roll_ms=4000, post_roll_ms=1000)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 2)

        assert config.pre_roll_frames(SR) == 64_000
        assert events[0].frames == WINDOW + HOP, (
            "only the 48000 frames captured so far can be in the pre-roll"
        )
        assert events[0].start_frame == 0


class TestNonFiringResults:
    """Results the gate must ignore, each for its own documented reason."""

    def test_a_partial_containing_the_phrase_does_not_fire(
        self, tmp_path: Path
    ) -> None:
        """Partials get revoked; acting on one writes a file for nothing said."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _partial())], events)

        _feed(sink, 3)

        assert sink.snippets_written == 0, (
            "a non-final result must not open a snippet"
        )
        assert sink.phrases_detected == 0
        assert events == []
        assert _wavs(config) == []
        assert not os.path.exists(config.snippet_dir)

    @pytest.mark.parametrize("text", ["", " ", "  "])
    def test_an_empty_final_does_not_fire(
        self, tmp_path: Path, text: str
    ) -> None:
        """Small models emit empty finals constantly; that is the sound of silence."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final(text))])

        _feed(sink, 3)

        assert sink.snippets_written == 0
        assert sink.phrases_detected == 0
        assert _wavs(config) == []

    @pytest.mark.parametrize(
        "text",
        [
            "turn it up",
            "look google it",          # substring, but not a whole-word run
            "okay google",
            "google ok",
            "hey google",              # not in this config's phrase list
        ],
    )
    def test_a_final_without_a_configured_phrase_does_not_fire(
        self, tmp_path: Path, text: str
    ) -> None:
        """Only the configured phrases open a snippet, matched on word boundaries."""
        config = _config(tmp_path, phrases=("ok google",))
        sink, _ = _make_sink(config, [(_bytes_after(0), _final(text))])

        _feed(sink, 3)

        assert sink.snippets_written == 0, f"{text!r} must not fire the gate"
        assert sink.phrases_detected == 0
        assert _wavs(config) == []

    def test_a_partial_followed_by_a_matching_final_fires_once(
        self, tmp_path: Path
    ) -> None:
        """The real decode order: the final is what counts, and only once."""
        config = _config(tmp_path)
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _partial()), (_bytes_after(0), _final())],
        )

        _feed(sink, 2)

        assert sink.phrases_detected == 1
        assert sink.snippets_written == 1


class TestCooldown:
    """The refractory period stops one utterance writing two files."""

    def test_a_second_match_inside_the_cooldown_is_suppressed(
        self, tmp_path: Path
    ) -> None:
        """Two close matches produce one snippet and one suppression."""
        config = _config(
            tmp_path, pre_roll_ms=0, post_roll_ms=1000, cooldown_ms=2000
        )
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(2), _final())],
            events,
        )

        _feed(sink, 4)

        assert sink.phrases_detected == 2, "both matches must be counted"
        assert sink.snippets_written == 1, (
            "the second match landed 1000 ms after the snippet closed, inside "
            "the 2000 ms cooldown"
        )
        assert sink.snippets_suppressed == 1
        assert len(events) == 1
        assert len(_wavs(config)) == 1

    def test_a_match_after_the_cooldown_opens_a_second_snippet(
        self, tmp_path: Path
    ) -> None:
        """The suppression is a refractory period, not a permanent mute."""
        config = _config(
            tmp_path, pre_roll_ms=0, post_roll_ms=1000, cooldown_ms=2000
        )
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(3), _final())],
            events,
        )

        _feed(sink, 5)

        assert sink.phrases_detected == 2
        assert sink.snippets_suppressed == 0, (
            "2000 ms of audio elapsed after the snippet closed, so the cooldown "
            "had run out"
        )
        assert sink.snippets_written == 2
        assert [event.seq for event in events] == [0, 1]

    def test_a_zero_cooldown_suppresses_nothing(self, tmp_path: Path) -> None:
        """cooldown_ms=0 is legal and means back-to-back snippets are allowed."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=1000, cooldown_ms=0)
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(2), _final())],
        )

        _feed(sink, 4)

        assert sink.snippets_suppressed == 0
        assert sink.snippets_written == 2

    def test_the_first_ever_match_is_never_suppressed(
        self, tmp_path: Path
    ) -> None:
        """No snippet has closed yet, so there is no cooldown in force."""
        config = _config(tmp_path, cooldown_ms=60_000, max_snippet_ms=120_000)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 2)

        assert sink.snippets_suppressed == 0
        assert sink.snippets_written == 1

    def test_a_suppressed_match_still_updates_last_phrase(
        self, tmp_path: Path
    ) -> None:
        """phrases_detected and last_phrase report what was heard, not what was kept."""
        config = _config(
            tmp_path,
            phrases=("ok google", "hey google"),
            pre_roll_ms=0,
            post_roll_ms=1000,
            cooldown_ms=2000,
        )
        sink, _ = _make_sink(
            config,
            [
                (_bytes_after(0), _final("ok google turn it up")),
                (_bytes_after(2), _final("hey google stop")),
            ],
        )

        _feed(sink, 4)

        assert sink.snippets_suppressed == 1
        assert sink.last_phrase == "hey google"


class TestExtension:
    """A match while a snippet is open extends it instead of opening another."""

    def test_a_repeat_match_does_not_open_a_second_file(
        self, tmp_path: Path
    ) -> None:
        """One continuous piece of speech must produce one continuous file."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=2000, cooldown_ms=0)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(1), _final())],
            events,
        )

        _feed(sink, 3)                     # chunks 0, 1, 2

        assert sink.phrases_detected == 2
        assert sink.recording is True, (
            "the second match reset the post-roll, so the snippet is still open"
        )
        assert sink.snippets_written == 0, (
            "no file has closed yet, and certainly not two"
        )
        assert events == []

        sink.on_chunk(_chunk(3))

        assert sink.snippets_written == 1, (
            "the extended snippet closes as ONE file, not two"
        )
        assert len(_wavs(config)) == 1

    def test_the_extension_pushes_the_close_one_post_roll_later(
        self, tmp_path: Path
    ) -> None:
        """Measured against a control run with a single match at the same place."""
        control_config = _config(
            tmp_path / "control", pre_roll_ms=0, post_roll_ms=2000, cooldown_ms=0
        )
        control_events: list[SnippetEvent] = []
        control, _ = _make_sink(
            control_config, [(_bytes_after(0), _final())], control_events
        )
        _feed(control, 3)

        assert control.snippets_written == 1, (
            "test setup: without an extension the snippet closes on chunk 2"
        )
        assert control_events[0].frames == 32_000

        extended_config = _config(
            tmp_path / "extended", pre_roll_ms=0, post_roll_ms=2000, cooldown_ms=0
        )
        extended_events: list[SnippetEvent] = []
        extended, _ = _make_sink(
            extended_config,
            [(_bytes_after(0), _final()), (_bytes_after(1), _final())],
            extended_events,
        )
        _feed(extended, 4)

        assert extended.snippets_written == 1
        assert extended_events[0].frames == 48_000, (
            "the extension must add exactly one more hop of audio"
        )
        assert extended_events[0].frames - control_events[0].frames == HOP

    def test_the_extension_updates_the_recorded_text(
        self, tmp_path: Path
    ) -> None:
        """The event reports the most recent utterance the snippet covered."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=2000, cooldown_ms=0)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config,
            [
                (_bytes_after(0), _final("ok google turn it up")),
                (_bytes_after(1), _final("ok google turn it down")),
            ],
            events,
        )

        _feed(sink, 4)

        assert events[0].text == "ok google turn it down"
        assert events[0].phrase == "ok google", (
            "the phrase stays the one that OPENED the snippet"
        )

    def test_an_extension_is_not_a_suppression(self, tmp_path: Path) -> None:
        """The cooldown only applies once a snippet has closed."""
        config = _config(
            tmp_path, pre_roll_ms=0, post_roll_ms=2000, cooldown_ms=60_000,
            max_snippet_ms=120_000,
        )
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(1), _final())],
        )

        _feed(sink, 4)

        assert sink.snippets_suppressed == 0
        assert sink.snippets_written == 1


class TestTruncation:
    """max_snippet_ms is a hard ceiling, not a suggestion."""

    def test_a_single_snippet_cannot_reach_the_ceiling(self, tmp_path: Path) -> None:
        """One un-extended snippet is pre-roll plus post-roll, and validation
        already bounds that below the ceiling -- so it closes normally.

        This used to truncate, but only because the writer overshot to a chunk
        boundary instead of stopping at the post-roll.  With the write trimmed,
        the ceiling is unreachable without an extension, which is what it exists
        to bound.
        """
        config = _config(
            tmp_path, pre_roll_ms=0, post_roll_ms=500, max_snippet_ms=600
        )
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 3)

        assert sink.snippets_truncated == 0
        assert events[0].truncated is False

        _, _, _, nframes, _ = _read_wav(_wavs(config)[0])
        assert nframes == config.post_roll_frames(SR), (
            "post_roll_ms must be exact, not 'at least, rounded up to a hop'"
        )

    def test_an_extended_snippet_is_cut_at_the_ceiling(self, tmp_path: Path) -> None:
        """The file stops at the ceiling and says so, in the counter and the event.

        Repeated matches push the post-roll out indefinitely, which is precisely
        the runaway ``max_snippet_ms`` is there to stop.
        """
        config = _config(
            tmp_path, pre_roll_ms=0, post_roll_ms=1500, max_snippet_ms=2000
        )
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(1), _final())],
            events,
        )

        _feed(sink, 4)

        ceiling = config.max_snippet_frames(SR)
        assert ceiling == 32_000, "test setup: 2000 ms at 16 kHz"
        assert sink.snippets_truncated == 1
        assert events[0].truncated is True

        _, _, _, nframes, _ = _read_wav(_wavs(config)[0])
        assert nframes == ceiling, (
            "the trim must be exact rather than 'within one hop of the ceiling'"
        )
        assert events[0].frames == nframes

    def test_a_generous_ceiling_truncates_nothing(self, tmp_path: Path) -> None:
        """The counter reports the ceiling specifically, not snippets in general."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 2)

        assert sink.snippets_truncated == 0
        assert events[0].truncated is False

    def test_repeated_matches_cannot_extend_past_the_ceiling(
        self, tmp_path: Path
    ) -> None:
        """The ceiling exists precisely to bound the extension mechanism."""
        config = _config(
            tmp_path, pre_roll_ms=0, post_roll_ms=1000, max_snippet_ms=2000
        )
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(
            config,
            [(_bytes_after(k), _final()) for k in range(6)],
            events,
        )

        _feed(sink, 6)
        sink.close()                       # finalize whatever is still open

        ceiling = config.max_snippet_frames(SR)
        assert len(events) >= 2, "test setup: the run must have produced snippets"
        for event in events:
            assert event.frames <= ceiling, (
                f"snippet {event.seq} ran to {event.frames} frames, past {ceiling}"
            )
        for path in _wavs(config):
            assert _read_wav(path)[3] <= ceiling


# ==========================================================================
# discontinuities
# ==========================================================================

class TestDiscontinuity:
    """A reported discontinuity resets the decoder and drops the pre-roll."""

    def test_a_discontinuous_chunk_resets_the_recognizer(
        self, tmp_path: Path
    ) -> None:
        """Decoder state describes audio that no longer joins up with what follows."""
        sink, recognizer = _make_sink(_config(tmp_path))

        sink.on_chunk(_chunk(0))
        assert recognizer.resets == 0

        sink.on_chunk(_chunk(1, discontinuous=True))
        assert recognizer.resets == 1

        sink.on_chunk(_chunk(2))
        assert recognizer.resets == 1, "a continuous chunk must not reset"

    def test_continuous_chunks_never_reset(self, tmp_path: Path) -> None:
        """Resetting on every chunk would make the decoder unable to hear a phrase."""
        sink, recognizer = _make_sink(_config(tmp_path))
        _feed(sink, 6)
        assert recognizer.resets == 0

    def test_a_discontinuity_clears_the_pre_roll(self, tmp_path: Path) -> None:
        """Audio from before lost frames must not be spliced into the snippet.

        Measured against a control run with the identical script: the only
        difference is the discontinuity flag, so the difference in snippet
        length is exactly the pre-roll that was dropped.
        """
        script = [(_bytes_after(1), _final())]

        control_config = _config(
            tmp_path / "control", pre_roll_ms=2000, post_roll_ms=1000
        )
        control_events: list[SnippetEvent] = []
        control, _ = _make_sink(control_config, list(script), control_events)
        control.on_chunk(_chunk(0))
        control.on_chunk(_chunk(1))
        control.on_chunk(_chunk(2))

        assert control_config.pre_roll_frames(SR) == 32_000
        assert control_events[0].frames == 32_000 + 16_000, (
            "test setup: the full 2000 ms pre-roll was available"
        )

        broken_config = _config(
            tmp_path / "broken", pre_roll_ms=2000, post_roll_ms=1000
        )
        broken_events: list[SnippetEvent] = []
        broken, recognizer = _make_sink(broken_config, list(script), broken_events)
        broken.on_chunk(_chunk(0))
        broken.on_chunk(_chunk(1, discontinuous=True))
        broken.on_chunk(_chunk(2))

        assert recognizer.resets == 1
        assert broken_events[0].frames == HOP + 16_000, (
            "after the pre-roll was cleared only the discontinuous chunk's own "
            "hop can precede the match"
        )
        assert broken_events[0].frames < control_events[0].frames

    def test_a_discontinuity_does_not_abandon_an_open_snippet(
        self, tmp_path: Path
    ) -> None:
        """The audio already written is real; only the buffered pre-roll is dropped."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=2000)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        sink.on_chunk(_chunk(0))
        assert sink.recording is True
        sink.on_chunk(_chunk(1, discontinuous=True))
        sink.on_chunk(_chunk(2))

        assert sink.snippets_written == 1
        assert events[0].frames == 2 * HOP


# ==========================================================================
# never raising
# ==========================================================================

class TestNeverRaises:
    """on_chunk runs on the consumer thread; an exception there is an outage."""

    def test_a_recognizer_that_raises_is_swallowed(self, tmp_path: Path) -> None:
        """A decoder fault degrades the gate to not gating, not to a dead pipeline."""
        config = _config(tmp_path)
        recognizer = _BoomRecognizer()
        sink = VoiceGateSink(config, SR, recognizer=recognizer)

        for k in range(3):
            assert sink.on_chunk(_chunk(k)) is None, (
                "on_chunk must return normally even when the recogniser throws"
            )

        assert recognizer.calls == 3, "the gate must keep trying, not give up"
        assert isinstance(sink.error, str) and sink.error, (
            "the swallowed failure must be reported in error"
        )
        assert "recognition failed" in sink.error
        assert "Boom" in sink.error
        assert sink.snippets_written == 0
        assert sink.frames_processed == WINDOW + 2 * HOP, (
            "audio must keep flowing while the gate is broken"
        )

    def test_a_snippet_callback_that_raises_is_swallowed(
        self, tmp_path: Path
    ) -> None:
        """A badly behaved callback is a latency problem, never an outage."""
        config = _config(tmp_path)
        recognizer = ScriptedRecognizer([(_bytes_after(0), _final())])
        sink = VoiceGateSink(
            config, SR, recognizer=recognizer, on_snippet=_boom_callback
        )

        for k in range(2):
            assert sink.on_chunk(_chunk(k)) is None

        assert isinstance(sink.error, str) and "the snippet callback failed" in sink.error
        assert sink.snippets_written == 1, (
            "the file was finished before the callback ran, so it still counts"
        )
        files = _wavs(config)
        assert len(files) == 1 and _read_wav(files[0])[3] == 24_000, (
            "the snippet on disk must be complete despite the callback failing"
        )

    def test_the_gate_keeps_working_after_a_callback_failure(
        self, tmp_path: Path
    ) -> None:
        """One bad callback must not permanently stop the gate."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=1000, cooldown_ms=0)
        recognizer = ScriptedRecognizer(
            [(_bytes_after(0), _final()), (_bytes_after(2), _final())]
        )
        sink = VoiceGateSink(
            config, SR, recognizer=recognizer, on_snippet=_boom_callback
        )

        _feed(sink, 4)

        assert sink.snippets_written == 2
        assert len(_wavs(config)) == 2

    def test_a_reset_that_raises_is_swallowed(self, tmp_path: Path) -> None:
        """A discontinuity must not turn a broken decoder into a crash."""
        recognizer = _ResetBoomRecognizer()
        sink = VoiceGateSink(_config(tmp_path), SR, recognizer=recognizer)

        assert sink.on_chunk(_chunk(0, discontinuous=True)) is None
        assert recognizer.resets == 1
        assert isinstance(sink.error, str)
        assert "resetting the recogniser failed" in sink.error

    def test_a_snippet_that_cannot_be_opened_is_recorded_not_raised(
        self, tmp_path: Path
    ) -> None:
        """A snippet_dir occupied by a FILE makes makedirs fail on the audio path."""
        blocked = tmp_path / "blocked"
        blocked.write_bytes(b"not a directory")
        config = _config(tmp_path, snippet_dir=str(blocked))
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        assert sink.on_chunk(_chunk(0)) is None
        assert sink.on_chunk(_chunk(1)) is None

        assert isinstance(sink.error, str) and "processing a chunk failed" in sink.error
        assert sink.snippets_written == 0
        assert sink.recording is False, "a failed open must not leave a writer behind"
        assert sink.frames_processed == WINDOW + HOP

    def test_a_recognizer_whose_close_raises_is_swallowed(
        self, tmp_path: Path
    ) -> None:
        """Shutdown must complete even when the decoder will not."""
        recognizer = _CloseBoomRecognizer()
        sink = VoiceGateSink(_config(tmp_path), SR, recognizer=recognizer)

        assert sink.close() is None
        assert recognizer.closed is True
        assert isinstance(sink.error, str)
        assert "closing the recogniser failed" in sink.error
        assert sink.closed is True

    def test_error_is_none_on_a_healthy_run(self, tmp_path: Path) -> None:
        """error is the way a broken gate is told from a quiet one."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])
        _feed(sink, 3)
        sink.close()

        assert sink.error is None


# ==========================================================================
# close()
# ==========================================================================

class TestClose:
    """close() finalizes what is open and releases the decoder, once."""

    def test_close_finalizes_an_open_snippet(self, tmp_path: Path) -> None:
        """A half-second file is more useful than no file."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=3000)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 2)
        assert sink.recording is True, "test setup: the snippet must still be open"
        assert sink.snippets_written == 0, "test setup: nothing has been announced"

        sink.close()

        assert sink.recording is False
        assert sink.snippets_written == 1
        files = _wavs(config)
        assert len(files) == 1
        _, _, _, nframes, raw = _read_wav(files[0])
        assert nframes == HOP, "the audio captured before shutdown must survive"
        assert raw, "the finalized snippet must not be empty"
        assert events and events[0].frames == nframes

    def test_a_snippet_finalized_by_close_is_not_truncated(
        self, tmp_path: Path
    ) -> None:
        """Stopping capture is not the same failure as running past the ceiling."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=3000)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())], events)

        _feed(sink, 2)
        sink.close()

        assert sink.snippets_truncated == 0, (
            "snippets_truncated reports the length ceiling specifically"
        )
        assert events[0].truncated is False

    def test_close_closes_the_recognizer(self, tmp_path: Path) -> None:
        """The decoder holds a model; shutdown has to release it."""
        sink, recognizer = _make_sink(_config(tmp_path))
        assert recognizer.closed is False

        sink.close()

        assert recognizer.closed is True
        assert sink.closed is True

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """The pipeline closes every sink, and a tee may close it again."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=3000)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 2)
        sink.close()
        sink.close()
        sink.close()

        assert sink.snippets_written == 1, (
            "a repeated close must not write or announce a second snippet"
        )
        assert len(_wavs(config)) == 1
        assert _read_wav(_wavs(config)[0])[3] == HOP, (
            "a repeated close must not truncate the finalized file"
        )

    def test_close_with_nothing_open_is_harmless(self, tmp_path: Path) -> None:
        """The common case: no snippet was in progress when capture stopped."""
        config = _config(tmp_path)
        sink, recognizer = _make_sink(config)

        _feed(sink, 3)
        sink.close()

        assert sink.snippets_written == 0
        assert recognizer.closed is True
        assert sink.error is None


class _FakeVerifier:
    """A scripted :class:`~echochamber.voicegate.speaker.SpeakerVerifier`.

    Mirrors :class:`ScriptedRecognizer`'s role for the decoder seam: a test
    decides the verdict in advance rather than driving a real embedder.
    """

    def __init__(
        self,
        matched: bool = True,
        speaker: str | None = "alice",
        score: float = 0.9,
    ) -> None:
        self.matched = matched
        self.speaker = speaker
        self.score = score
        self.calls: list[tuple[np.ndarray, int]] = []
        self.closed = False

    def verify(self, samples: np.ndarray, sample_rate: int) -> VerifyResult:
        self.calls.append((samples, sample_rate))
        return VerifyResult(matched=self.matched, speaker=self.speaker, score=self.score)

    def close(self) -> None:
        self.closed = True


class _BoomVerifier:
    """A verifier whose ``verify`` always raises, for the fail-closed path."""

    def __init__(self) -> None:
        self.calls = 0

    def verify(self, samples: np.ndarray, sample_rate: int) -> VerifyResult:
        self.calls += 1
        raise Boom("verifier died")

    def close(self) -> None:
        pass


class TestSpeakerVerification:
    """A configured verifier gates every new snippet; ``None`` lets everything through."""

    def test_no_verifier_lets_every_phrase_through_unverified(
        self, tmp_path: Path
    ) -> None:
        """The default: a gate built without speaker_verifier behaves as before."""
        config = _config(tmp_path)
        events: list = []
        detections: list = []
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final())],
            events,
            on_detected=detections.append,
        )

        _feed(sink, 2)

        assert sink.snippets_written == 1
        assert len(detections) == 1
        assert detections[0].speaker is None
        assert detections[0].speaker_score == 0.0
        stats = sink.snapshot()
        assert stats.speaker_verified == 0
        assert stats.speaker_rejected == 0

    def test_a_matching_verifier_lets_the_phrase_through(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        events: list = []
        detections: list = []
        verifier = _FakeVerifier(matched=True, speaker="alice", score=0.87)
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final())],
            events,
            on_detected=detections.append,
            speaker_verifier=verifier,
        )

        _feed(sink, 2)

        assert sink.snippets_written == 1
        assert len(events) == 1
        assert len(detections) == 1
        assert detections[0].speaker == "alice"
        assert detections[0].speaker_score == pytest.approx(0.87)
        assert len(verifier.calls) == 1
        called_samples, called_rate = verifier.calls[0]
        assert np.array_equal(called_samples, _chunk(0).samples)
        assert called_rate == SR

        stats = sink.snapshot()
        assert stats.speaker_verified == 1
        assert stats.speaker_rejected == 0
        assert stats.last_speaker == "alice"
        assert stats.last_speaker_score == pytest.approx(0.87)

    def test_a_rejecting_verifier_suppresses_the_phrase_entirely(
        self, tmp_path: Path
    ) -> None:
        """Fail closed: no snippet, no on_snippet, no on_detected -- as if unheard."""
        config = _config(tmp_path)
        events: list = []
        detections: list = []
        verifier = _FakeVerifier(matched=False, speaker=None, score=0.0)
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final())],
            events,
            on_detected=detections.append,
            speaker_verifier=verifier,
        )

        _feed(sink, 2)

        assert sink.snippets_written == 0
        assert sink.recording is False
        assert events == []
        assert detections == []
        assert _wavs(config) == []

        stats = sink.snapshot()
        assert stats.phrases_detected == 1, "the phrase was still heard by the decoder"
        assert stats.speaker_rejected == 1
        assert stats.speaker_verified == 0

    def test_a_raising_verifier_fails_closed(self, tmp_path: Path) -> None:
        """A dead embedder must suppress the phrase, not let it through."""
        config = _config(tmp_path)
        events: list = []
        detections: list = []
        verifier = _BoomVerifier()
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final())],
            events,
            on_detected=detections.append,
            speaker_verifier=verifier,
        )

        _feed(sink, 2)

        assert sink.snippets_written == 0
        assert events == []
        assert detections == []
        assert verifier.calls == 1
        assert sink.error is not None
        assert "speaker verification failed" in sink.error

        stats = sink.snapshot()
        assert stats.speaker_rejected == 1
        assert stats.speaker_verified == 0

    def test_extending_an_open_snippet_does_not_re_verify(self, tmp_path: Path) -> None:
        """One utterance is verified once; a repeated phrase just extends it."""
        config = _config(tmp_path, pre_roll_ms=0, post_roll_ms=2000, cooldown_ms=0)
        events: list = []
        detections: list = []
        verifier = _FakeVerifier(matched=True, speaker="alice", score=0.9)
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(1), _final())],
            events,
            on_detected=detections.append,
            speaker_verifier=verifier,
        )

        _feed(sink, 3)

        assert len(verifier.calls) == 1, "the second match must extend, not re-verify"
        assert len(detections) == 2
        assert detections[0].extended is False
        assert detections[1].extended is True
        assert detections[1].speaker is None, (
            "an extension carries no fresh verification of its own"
        )

    def test_rejecting_a_match_leaves_a_later_match_free_to_open_a_snippet(
        self, tmp_path: Path
    ) -> None:
        """A rejected phrase must not wedge the gate: the next attempt tries again."""
        config = _config(tmp_path, cooldown_ms=0)
        events: list = []
        verifier = _FakeVerifier(matched=False)
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(2), _final())],
            events,
            speaker_verifier=verifier,
        )

        _feed(sink, 2)
        assert sink.snippets_written == 0
        assert len(verifier.calls) == 1, "test setup: only the first match fired yet"

        verifier.matched = True
        verifier.speaker = "alice"
        sink.on_chunk(_chunk(2))

        assert len(verifier.calls) == 2
        assert sink.snippets_written == 0, "still open; post-roll has not elapsed yet"
        assert sink.recording is True

    def test_close_closes_the_speaker_verifier(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        verifier = _FakeVerifier()
        sink, _ = _make_sink(config, speaker_verifier=verifier)

        sink.close()

        assert verifier.closed is True

    def test_close_is_idempotent_with_a_verifier_configured(
        self, tmp_path: Path
    ) -> None:
        config = _config(tmp_path)
        verifier = _FakeVerifier()
        sink, _ = _make_sink(config, speaker_verifier=verifier)

        sink.close()
        sink.close()  # must not raise, must not double-close the verifier oddly

        assert verifier.closed is True


    def test_on_chunk_after_close_does_nothing(self, tmp_path: Path) -> None:
        """A late chunk during shutdown must not reopen the gate."""
        config = _config(tmp_path)
        sink, recognizer = _make_sink(config, [(_bytes_after(0), _final())])

        sink.close()
        consumed = recognizer.consumed
        _feed(sink, 3)

        assert sink.frames_processed == 0
        assert recognizer.consumed == consumed
        assert sink.snippets_written == 0

    def test_close_does_not_reset_the_counters(self, tmp_path: Path) -> None:
        """The GUI still reads the stats after the pipeline has stopped."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 2)
        frames = sink.frames_processed
        sink.close()

        assert sink.frames_processed == frames
        assert sink.phrases_detected == 1
        assert sink.snippets_written == 1


# ==========================================================================
# snapshot
# ==========================================================================

class TestSnapshot:
    """snapshot() is the GUI's coherent read of every counter at once."""

    def test_snapshot_returns_a_voicegatestats(self, tmp_path: Path) -> None:
        """The type matters: the GUI stores it and reads it later."""
        sink, _ = _make_sink(_config(tmp_path))
        assert isinstance(sink.snapshot(), VoiceGateStats)

    def test_a_fresh_gate_snapshots_all_zeroes(self, tmp_path: Path) -> None:
        """The defaults of VoiceGateStats are what an untouched gate reports."""
        sink, _ = _make_sink(_config(tmp_path))
        assert sink.snapshot() == VoiceGateStats()

    def test_every_field_agrees_with_the_matching_property(
        self, tmp_path: Path
    ) -> None:
        """Reading the properties one at a time must give the same numbers."""
        config = _config(
            tmp_path, pre_roll_ms=0, post_roll_ms=1000, cooldown_ms=2000
        )
        sink, _ = _make_sink(
            config,
            [(_bytes_after(0), _final()), (_bytes_after(2), _final())],
        )

        sink.on_chunk(_chunk(0))
        sink.on_chunk(_chunk_at(120_000, seq=1))     # forces a gap
        _feed(sink, 3, first=9)

        snap = sink.snapshot()
        for field in (
            "frames_processed",
            "phrases_detected",
            "snippets_written",
            "snippets_suppressed",
            "snippets_truncated",
            "gaps",
            "last_phrase",
            "last_snippet_path",
            "error",
        ):
            assert getattr(snap, field) == getattr(sink, field), (
                f"snapshot().{field} disagrees with the {field} property"
            )

        assert snap.gaps >= 1, "test setup: the run must have produced a gap"
        assert snap.snippets_written >= 1, "test setup: the run must have fired"

    def test_a_snapshot_is_detached_from_later_activity(
        self, tmp_path: Path
    ) -> None:
        """The GUI holds a snapshot while the consumer thread keeps counting."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(_bytes_after(2), _final())])

        sink.on_chunk(_chunk(0))
        early = sink.snapshot()
        _feed(sink, 3, first=1)

        assert early.frames_processed == WINDOW
        assert early.snippets_written == 0
        assert sink.frames_processed > early.frames_processed
        assert sink.snippets_written == 1

    def test_snapshot_reports_the_last_snippet_path(self, tmp_path: Path) -> None:
        """The path is how the GUI offers to open the file it just heard about."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(_bytes_after(0), _final())])

        _feed(sink, 2)
        snap = sink.snapshot()

        assert snap.last_phrase == "ok google"
        assert snap.last_snippet_path is not None
        assert Path(snap.last_snippet_path).exists()

    def test_snapshot_repr_names_the_counters_that_matter(
        self, tmp_path: Path
    ) -> None:
        """The repr goes into logs when a gate is behaving unexpectedly."""
        text = repr(VoiceGateStats(frames_processed=5, phrases_detected=2))
        assert "VoiceGateStats" in text
        assert "frames_processed=5" in text
        assert "phrases_detected=2" in text
