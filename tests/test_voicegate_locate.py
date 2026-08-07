"""Tests for wake-phrase *localization* -- cutting a snippet to the phrase.

The gate no longer records a fixed window around the moment a decoder happened
to finish speaking; it asks the decoder *where* the phrase was and cuts the
file to that span plus ``lead_ms`` and ``trail_ms``.  That turns a snippet from
"a few seconds that probably contain the hotword" into "the hotword", and it
makes a whole class of silent failure possible: an off-by-a-second frame
calculation still produces a plausible WAV of the wrong audio.  Everything here
exists to make that failure loud.

Testing strategy worth stating up front:

* **The audio is a ramp whose sample values encode the absolute frame index.**
  ``RAMP[i] == (i % 32000) / 32768``, so reading a written WAV's *first sample*
  back off disk says exactly which stream frame the clip begins at --
  :func:`_wav_start_frame` inverts the encoding.  Asserting a file merely exists,
  or even that it has the right length, proves nothing about *which* audio was
  cut; this does.  int16 quantisation costs a frame either way, hence
  :data:`FRAME_TOL`.
* **Nothing imports Vosk, loads a model, opens a socket or spawns a process.**
  Word timings are hand-built :class:`WordTiming` values inside hand-built
  :class:`Recognition` objects, so the decoder's contract -- "times count from
  the first sample I was ever fed" -- is expressed directly rather than
  simulated.
* **:class:`_TimedRecognizer` reports at a chosen frame count**, not at a chosen
  byte offset, because localisation is entirely about the lag between *when a
  phrase was spoken* and *when the decoder said so*.
  :class:`~echochamber.voicegate.recognizer.ScriptedRecognizer` cannot express
  "report after N frames with these timings", which is why it is not used here.
* **Chunks are non-overlapping by default** (``window == hop == 1600``), so the
  arithmetic in a failure message is readable.  The tests that care about the
  de-overlapping seam -- the discontinuity ones -- deliberately use overlapping
  windows instead, because that is the only geometry in which the anchor can be
  wrong.

Two of these started life as :func:`pytest.mark.xfail`, pinning defects this
file found: an anchor taken at a window's start rather than at the start of the
de-overlapped tail actually fed to the recogniser, which put every clip located
after a discontinuity one overlap early; and a writer that emitted whole chunks
instead of trimming to the snippet's computed end, overrunning it by up to a
hop.  Both are fixed, so both now assert plainly.
"""

from __future__ import annotations

import dataclasses
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from echochamber.audio.types import AudioChunk
from echochamber.voicegate.config import ClipMode, VoiceGateConfig
from echochamber.voicegate.matching import PhraseMatch, locate_phrase, match_phrase
from echochamber.voicegate.recognizer import (
    Recognition,
    WordTiming,
    parse_word_timings,
)
from echochamber.voicegate.sink import SnippetEvent, VoiceGateSink


SR = 16_000
HOP = 1_600                  # 100 ms: one chunk, and the default feed granularity
BYTES_PER_FRAME = 2

# The ramp repeats every RAMP_PERIOD frames, so the encoding only identifies a
# frame unambiguously inside one period.  No test feeds more than that.
RAMP_PERIOD = 32_000
RAMP: np.ndarray = (
    np.arange(RAMP_PERIOD, dtype=np.float32) % RAMP_PERIOD
) / 32768.0

# float32 -> int16 truncation, and back through a /32767 * 32768 scaling, can
# each move the recovered index by one.
FRAME_TOL = 2

PHRASE = "ok google"
TEXT = "ok google turn it up"

# The standard utterance: "ok" from 0.50 s to 0.62 s and "google" from 0.65 s
# to 0.80 s, i.e. absolute frames 8000 to 12800 -- a 4800-frame span.
PHRASE_WORDS: tuple[WordTiming, ...] = (
    WordTiming("ok", 0.50, 0.62, 0.9),
    WordTiming("google", 0.65, 0.80, 0.8),
)
PHRASE_START_FRAME = 8_000
PHRASE_END_FRAME = 12_800


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _final(
    words: tuple[WordTiming, ...] = PHRASE_WORDS, text: str = TEXT
) -> Recognition:
    """A settled recognition carrying ``text`` and its per-word timings."""
    return Recognition(text=text, final=True, words=words)


class _TimedRecognizer:
    """Emit scripted results once enough audio has been *fed*, not consumed.

    A real decoder reports a phrase some way after it was spoken, which is the
    entire reason localisation exists: the gate has to reach back into buffered
    audio rather than record from the moment it was told.  The script is keyed
    on frames fed so a test states that lag directly -- "the phrase ended at
    frame 12800 and the decoder said so at frame 16000".
    """

    __slots__ = ("_script", "_next", "_fed_frames", "_resets", "_closed")

    def __init__(self, script: list[tuple[int, Recognition]] | None = None) -> None:
        """Prepare the scripted results.

        Args:
            script: ``(frames_fed, Recognition)`` pairs.  Each result is emitted
                by the first :meth:`accept_pcm` that brings the running total of
                frames fed to at least ``frames_fed``.
        """
        self._script: list[tuple[int, Recognition]] = sorted(
            script or [], key=lambda item: item[0]
        )
        self._next: int = 0
        self._fed_frames: int = 0
        self._resets: int = 0
        self._closed: bool = False

    @property
    def fed_frames(self) -> int:
        """Total frames fed to this recogniser."""
        return self._fed_frames

    @property
    def resets(self) -> int:
        """How many times :meth:`reset` has been called."""
        return self._resets

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        """Consume ``pcm`` and emit every scripted result now due."""
        self._fed_frames += len(pcm) // BYTES_PER_FRAME
        due: list[Recognition] = []
        while self._next < len(self._script):
            at_frames, recognition = self._script[self._next]
            if at_frames > self._fed_frames:
                break
            due.append(recognition)
            self._next += 1
        return due

    def reset(self) -> None:
        """Count the reset.

        The frame total is deliberately **not** rewound: Vosk accumulates its
        word clock across ``Reset()`` (see :class:`WordTiming`), and a stub that
        rewound would hide exactly the anchor arithmetic these tests exist to
        check.
        """
        self._resets += 1

    def close(self) -> None:
        """Mark the recogniser closed.  Idempotent."""
        self._closed = True


def _config(tmp_path: Path, **overrides: Any) -> VoiceGateConfig:
    """An enabled gate cutting to the phrase, writing under ``tmp_path``.

    The durations are much shorter than the shipping ones so a whole snippet
    opens and closes inside a handful of hand-fed 100 ms chunks: 1600 frames of
    lead and trail, a 2000 ms lookback, and a ceiling nothing reaches.
    """
    kwargs: dict[str, Any] = dict(
        enabled=True,
        phrases=(PHRASE,),
        clip_mode=ClipMode.PHRASE,
        lead_ms=100,              # 1600 frames
        trail_ms=100,             # 1600 frames
        lookback_ms=2000,         # 32000 frames: nothing is evicted
        pre_roll_ms=200,          # 3200 frames, for the fallback shape
        post_roll_ms=200,         # 3200 frames
        max_snippet_ms=5000,      # 80000 frames: effectively no ceiling
        cooldown_ms=0,
        snippet_dir=str(tmp_path / "snippets"),
    )
    kwargs.update(overrides)
    return VoiceGateConfig(**kwargs)


def _chunk(
    start_frame: int,
    n_frames: int,
    seq: int,
    *,
    discontinuous: bool = False,
) -> AudioChunk:
    """One window of the ramp at an absolute position."""
    stop = start_frame + n_frames
    if stop > RAMP_PERIOD:
        raise AssertionError(
            f"test setup: frame {stop} leaves the ramp's unambiguous period "
            f"of {RAMP_PERIOD}"
        )
    return AudioChunk(
        samples=RAMP[start_frame:stop],
        start_frame=start_frame,
        seq=seq,
        sample_rate=SR,
        discontinuous=discontinuous,
    )


def _feed(
    sink: VoiceGateSink,
    count: int,
    *,
    hop: int = HOP,
    window: int = HOP,
    discontinuous_at: int | None = None,
) -> None:
    """Push ``count`` chunks of the standard grid: chunk ``k`` starts at ``k*hop``."""
    for k in range(count):
        sink.on_chunk(
            _chunk(
                k * hop, window, k, discontinuous=(k == discontinuous_at)
            )
        )


def _make_sink(
    config: VoiceGateConfig,
    script: list[tuple[int, Recognition]] | None = None,
    events: list[SnippetEvent] | None = None,
) -> tuple[VoiceGateSink, _TimedRecognizer]:
    """Build a gate over a :class:`_TimedRecognizer`, returning both."""
    recognizer = _TimedRecognizer(script)
    sink = VoiceGateSink(
        config,
        SR,
        recognizer=recognizer,
        on_snippet=None if events is None else events.append,
    )
    return sink, recognizer


def _wavs(config: VoiceGateConfig) -> list[Path]:
    """Every ``.wav`` in the config's snippet directory, sorted by name."""
    directory = Path(config.snippet_dir)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.wav"))


def _only_wav(config: VoiceGateConfig) -> Path:
    """The single snippet the run must have produced."""
    files = _wavs(config)
    assert len(files) == 1, f"expected exactly one snippet, got {files}"
    return files[0]


def _wav_frames(path: Path) -> int:
    """Frames the finalized WAV at ``path`` declares."""
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def _wav_start_frame(path: Path) -> int:
    """Recover the absolute stream frame the WAV at ``path`` begins on.

    Inverts the ramp encoding: the file's first sample identifies the frame it
    was taken from, so this says *which* audio was cut rather than how much.
    """
    with wave.open(str(path), "rb") as handle:
        if handle.getnframes() == 0:
            raise AssertionError("the snippet is empty, so it has no start frame")
        raw = handle.readframes(1)
    first = int(np.frombuffer(raw, dtype="<i2")[0])
    return int(round(first / 32767.0 * 32768.0))


def _assert_clip(path: Path, start_frame: int, frames: int) -> None:
    """Assert the WAV at ``path`` is exactly ``[start_frame, start_frame+frames)``."""
    actual_start = _wav_start_frame(path)
    assert abs(actual_start - start_frame) <= FRAME_TOL, (
        f"the clip begins at stream frame {actual_start}, not {start_frame}: "
        f"it is on the wrong audio by {actual_start - start_frame} frames"
    )
    assert _wav_frames(path) == frames, (
        f"the clip runs {_wav_frames(path)} frames, not {frames}"
    )


# ==========================================================================
# WordTiming and Recognition.words
# ==========================================================================

class TestWordTiming:
    """One word's position in the audio the recogniser was fed."""

    def test_fields_are_what_was_given(self) -> None:
        """A timing is a plain record; nothing is recomputed on construction."""
        timing = WordTiming("google", 1.25, 1.5, 0.87)

        assert timing.word == "google"
        assert timing.start == 1.25
        assert timing.end == 1.5
        assert timing.conf == 0.87

    def test_confidence_defaults_to_zero(self) -> None:
        """A backend that reports no confidence must not report a fake one."""
        assert WordTiming("ok", 0.0, 0.5).conf == 0.0

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [(0.0, 0.5, 0.5), (1.25, 1.5, 0.25), (2.0, 2.0, 0.0), (0.1, 3.1, 3.0)],
    )
    def test_duration_s_is_end_minus_start(
        self, start: float, end: float, expected: float
    ) -> None:
        """The ordinary case: a word lasts as long as its span."""
        assert WordTiming("x", start, end).duration_s == pytest.approx(expected)

    @pytest.mark.parametrize(("start", "end"), [(1.0, 0.5), (2.0, -1.0), (0.5, 0.4)])
    def test_duration_s_is_never_negative(self, start: float, end: float) -> None:
        """A backwards span clamps to zero rather than reporting negative time.

        The gate divides and multiplies these into frame counts; a negative
        duration would propagate into a snippet length below zero rather than
        into an obviously wrong file.
        """
        assert WordTiming("x", start, end).duration_s == 0.0

    def test_repr_shows_the_word_and_its_span(self) -> None:
        """The repr is for reading a timing list back in a failure message."""
        text = repr(WordTiming("google", 1.25, 1.5, 0.9))

        assert "WordTiming" in text
        assert "'google'" in text
        assert "1.250" in text
        assert "1.500" in text
        assert "s" in text

    def test_timings_are_frozen(self) -> None:
        """A timing crosses thread boundaries with a result; it is immutable."""
        timing = WordTiming("ok", 0.0, 0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            timing.start = 1.0  # type: ignore[misc]

    def test_equality_is_by_value(self) -> None:
        """parse_word_timings is asserted against constructed timings, so this
        is the comparison every one of those tests rests on."""
        assert WordTiming("ok", 0.5, 0.62, 0.9) == WordTiming("ok", 0.5, 0.62, 0.9)
        assert WordTiming("ok", 0.5, 0.62, 0.9) != WordTiming("ok", 0.5, 0.63, 0.9)


class TestRecognitionWords:
    """Recognition carries the timings, and reports the span they cover."""

    def test_words_default_to_empty(self) -> None:
        """A backend reporting no timings is the fallback path, not an error."""
        assert Recognition(text="ok google", final=True).words == ()

    def test_words_are_kept_in_order(self) -> None:
        """Order is spoken order: locate_phrase indexes into this by position."""
        recognition = _final()
        assert recognition.words == PHRASE_WORDS
        assert recognition.words[0].word == "ok"
        assert recognition.words[-1].word == "google"

    def test_span_is_the_first_start_and_the_last_end(self) -> None:
        """The span covers every word, not merely the first or the longest."""
        recognition = Recognition(
            text="ok google turn it up",
            final=True,
            words=(
                WordTiming("ok", 0.5, 0.62),
                WordTiming("google", 0.65, 0.8),
                WordTiming("turn", 0.9, 1.1),
            ),
        )
        assert recognition.span == (0.5, 1.1)

    def test_span_of_a_single_word(self) -> None:
        """First and last are the same word; the span is still that word."""
        recognition = Recognition(
            text="hey", final=True, words=(WordTiming("hey", 2.0, 2.4),)
        )
        assert recognition.span == (2.0, 2.4)

    def test_span_is_none_without_words(self) -> None:
        """``None`` rather than ``(0.0, 0.0)``: no timings is not "at the start"."""
        assert Recognition(text="ok google", final=True).span is None

    def test_a_partial_may_carry_no_words(self) -> None:
        """Partials have no settled timings, and the gate ignores them anyway."""
        assert Recognition(text="ok goo", final=False).span is None


# ==========================================================================
# parse_word_timings
# ==========================================================================

class TestParseWordTimings:
    """Vosk's ``result`` array, coerced -- and a bad entry skipped, not faked."""

    def test_builds_timings_from_the_vosk_shape(self) -> None:
        """The headline case: the exact dict shape Vosk emits per word."""
        timings = parse_word_timings(
            [
                {"word": "ok", "start": 0.5, "end": 0.62, "conf": 0.94},
                {"word": "google", "start": 0.65, "end": 0.8, "conf": 1.0},
            ]
        )

        assert timings == (
            WordTiming("ok", 0.5, 0.62, 0.94),
            WordTiming("google", 0.65, 0.8, 1.0),
        )

    def test_integers_are_accepted_as_seconds(self) -> None:
        """JSON writes ``1`` rather than ``1.0``; both are numbers."""
        timings = parse_word_timings([{"word": "hey", "start": 1, "end": 2}])
        assert timings == (WordTiming("hey", 1.0, 2.0, 0.0),)
        assert isinstance(timings[0].start, float)

    @pytest.mark.parametrize(
        "raw",
        [None, {}, "result", 5, 0.5, True, ("word",), {"word": "ok"}],
    )
    def test_a_non_list_yields_nothing(self, raw: object) -> None:
        """``result`` is absent or malformed: no timings, not an exception."""
        assert parse_word_timings(raw) == ()

    def test_an_empty_list_yields_nothing(self) -> None:
        """An utterance of silence carries an empty array, not a missing one."""
        assert parse_word_timings([]) == ()

    def test_a_list_of_non_dicts_yields_nothing(self) -> None:
        """Every entry is inspected; none of these is a word."""
        assert parse_word_timings(["ok", 1, None, ["ok", 0.0, 0.5]]) == ()

    @pytest.mark.parametrize(
        "entry",
        [
            {"start": 0.5, "end": 0.62},                  # no word
            {"word": "ok", "end": 0.62},                  # no start
            {"word": "ok", "start": 0.5},                 # no end
            {"word": "ok", "start": None, "end": 0.62},
            {"word": "ok", "start": 0.5, "end": "0.62"},
            {"word": 7, "start": 0.5, "end": 0.62},       # word is not a string
        ],
    )
    def test_an_incomplete_entry_is_skipped_not_defaulted(
        self, entry: dict[str, object]
    ) -> None:
        """A missing time is dropped, never filled in with zero.

        A fabricated ``start`` of ``0.0`` would place the wake phrase at the
        very beginning of the stream, and the gate would cut a confident,
        correctly-named snippet of entirely the wrong audio.  Losing the timing
        costs a wider clip; inventing one costs a wrong clip.
        """
        assert parse_word_timings([entry]) == ()

    def test_a_mixed_list_yields_only_the_complete_entries(self) -> None:
        """The good words survive; the unreadable ones simply are not there."""
        timings = parse_word_timings(
            [
                {"word": "ok", "start": 0.5, "end": 0.62, "conf": 0.9},
                {"word": "google", "end": 0.8},              # no start
                {"word": "turn", "start": 0.9, "end": 1.1},
                {"start": 1.2, "end": 1.3},                  # no word
            ]
        )

        assert timings == (
            WordTiming("ok", 0.5, 0.62, 0.9),
            WordTiming("turn", 0.9, 1.1, 0.0),
        ), "only the entries carrying a word, a start and an end may survive"

    def test_a_missing_confidence_becomes_zero(self) -> None:
        """Confidence IS defaulted: it is informational, and drives nothing."""
        timings = parse_word_timings([{"word": "ok", "start": 0.5, "end": 0.62}])
        assert timings == (WordTiming("ok", 0.5, 0.62, 0.0),)

    def test_an_unreadable_confidence_becomes_zero(self) -> None:
        """A bad ``conf`` is not a reason to throw away a good timing."""
        timings = parse_word_timings(
            [{"word": "ok", "start": 0.5, "end": 0.62, "conf": "high"}]
        )
        assert timings == (WordTiming("ok", 0.5, 0.62, 0.0),)

    @pytest.mark.parametrize(
        "entry",
        [
            {"word": "ok", "start": True, "end": 0.62},
            {"word": "ok", "start": 0.5, "end": True},
            {"word": "ok", "start": False, "end": 0.62},
        ],
    )
    def test_a_boolean_time_is_not_a_number(self, entry: dict[str, object]) -> None:
        """``bool`` subclasses ``int``, so ``true`` would otherwise read as 1.0.

        A time of "1.0 seconds" invented from a JSON ``true`` is precisely the
        confident-but-wrong value this parser exists to refuse.
        """
        assert parse_word_timings([entry]) == ()

    def test_a_boolean_confidence_becomes_zero(self) -> None:
        """The same rejection, but conf is defaulted rather than fatal."""
        timings = parse_word_timings(
            [{"word": "ok", "start": 0.5, "end": 0.62, "conf": True}]
        )
        assert timings == (WordTiming("ok", 0.5, 0.62, 0.0),)

    @pytest.mark.parametrize(
        ("start", "end"), [(1.0, 0.5), (2.0, 0.0), (0.62, 0.5)]
    )
    def test_an_end_below_the_start_is_clamped_up(
        self, start: float, end: float
    ) -> None:
        """A backwards word becomes a zero-length one, not a negative span."""
        timings = parse_word_timings([{"word": "ok", "start": start, "end": end}])

        assert timings == (WordTiming("ok", start, start, 0.0),)
        assert timings[0].duration_s == 0.0

    def test_the_order_of_the_array_is_preserved(self) -> None:
        """Position is meaningful: locate_phrase indexes into this by token."""
        timings = parse_word_timings(
            [
                {"word": "please", "start": 0.1, "end": 0.2},
                {"word": "ok", "start": 0.3, "end": 0.4},
                {"word": "google", "start": 0.5, "end": 0.6},
            ]
        )
        assert [timing.word for timing in timings] == ["please", "ok", "google"]


# ==========================================================================
# locate_phrase
# ==========================================================================

class TestLocatePhrase:
    """Mapping a matched phrase onto the seconds the decoder reported."""

    def test_a_normal_match_spans_the_phrase(self) -> None:
        """The span is the phrase's own words, not the whole utterance."""
        found = match_phrase(TEXT, (PHRASE,))
        assert found is not None
        words = PHRASE_WORDS + (
            WordTiming("turn", 0.9, 1.0),
            WordTiming("it", 1.0, 1.1),
            WordTiming("up", 1.1, 1.3),
        )

        assert locate_phrase(found, words) == (0.50, 0.80), (
            "the span must end with the phrase, not with the sentence"
        )

    def test_no_words_gives_up(self) -> None:
        """The fallback path: a backend that reports no timings."""
        found = match_phrase(TEXT, (PHRASE,))
        assert found is not None
        assert locate_phrase(found, ()) is None

    def test_words_that_are_not_the_phrase_give_up(self) -> None:
        """The alignment guard: the timing at the index must BE the token.

        ``token_index`` indexes the normalised tokens of the recognised text,
        while the timings come from the decoder.  If normalisation changed the
        word count -- a decoder that emits "ok" as two tokens, a filler the
        text folded away -- the index lands on a different word, and cutting on
        it would produce a snippet of some other moment entirely.
        """
        found = match_phrase("ok google", (PHRASE,))
        assert found is not None
        assert found.token_index == 0, "test setup: the match is at the start"

        misaligned = (
            WordTiming("um", 0.10, 0.20),
            WordTiming("ok", 0.50, 0.62),
            WordTiming("google", 0.65, 0.80),
        )

        assert locate_phrase(found, misaligned) is None, (
            "words[0:2] is ('um', 'ok'), not the phrase; the lookup must be "
            "refused rather than cut two words too early"
        )

    def test_a_second_word_that_does_not_match_gives_up(self) -> None:
        """Every word of the phrase is checked, not only the first."""
        found = match_phrase("ok google", (PHRASE,))
        assert found is not None
        assert (
            locate_phrase(
                found, (WordTiming("ok", 0.5, 0.6), WordTiming("chrome", 0.6, 0.8))
            )
            is None
        )

    @pytest.mark.parametrize("n_words", [0, 1])
    def test_a_word_list_shorter_than_the_phrase_gives_up(
        self, n_words: int
    ) -> None:
        """``token_index + len(phrase)`` overruns, so there is nothing to read."""
        found = match_phrase(TEXT, (PHRASE,))
        assert found is not None
        assert locate_phrase(found, PHRASE_WORDS[:n_words]) is None

    def test_an_index_past_the_end_of_the_words_gives_up(self) -> None:
        """A real overrun: the phrase matched late, the timings stopped early."""
        found = match_phrase("please ok google", (PHRASE,))
        assert found is not None
        assert found.token_index == 1

        # Only two timings, so words[1:3] runs off the end.
        assert locate_phrase(
            found, (WordTiming("please", 0.1, 0.3), WordTiming("ok", 0.5, 0.62))
        ) is None

    def test_a_match_at_a_non_zero_index_reads_the_right_words(self) -> None:
        """The index is real -- built by match_phrase -- not asserted by hand."""
        found = match_phrase("please ok google now", (PHRASE,))
        assert found is not None
        assert found.token_index == 1, "test setup: the phrase starts at token 1"

        words = (
            WordTiming("please", 0.10, 0.30),
            WordTiming("ok", 0.50, 0.62),
            WordTiming("google", 0.65, 0.80),
            WordTiming("now", 0.90, 1.05),
        )

        assert locate_phrase(found, words) == (0.50, 0.80), (
            "the span must skip 'please' and stop before 'now'"
        )

    @pytest.mark.parametrize(
        "spelling", [("OK", "GOOGLE"), ("Ok", "Google"), ("ok,", "google!")]
    )
    def test_the_decoder_spelling_is_normalised_before_comparing(
        self, spelling: tuple[str, str]
    ) -> None:
        """A decoder that capitalises or punctuates still aligns.

        The haystack was folded by :func:`normalize`; comparing the decoder's
        raw word against it would be raw-against-folded, and every capitalised
        utterance would fall back for no reason.
        """
        found = match_phrase(TEXT, (PHRASE,))
        assert found is not None
        words = (
            WordTiming(spelling[0], 0.50, 0.62),
            WordTiming(spelling[1], 0.65, 0.80),
        )
        assert locate_phrase(found, words) == (0.50, 0.80)

    def test_a_backwards_span_gives_up(self) -> None:
        """An end before the start cannot be turned into a clip length."""
        found = match_phrase("ok google", (PHRASE,))
        assert found is not None
        words = (WordTiming("ok", 1.0, 1.2), WordTiming("google", 0.2, 0.4))

        assert locate_phrase(found, words) is None

    def test_a_zero_length_span_is_still_a_span(self) -> None:
        """Equal start and end is degenerate but not backwards; the caller decides."""
        found = match_phrase("ok google", (PHRASE,))
        assert found is not None
        words = (WordTiming("ok", 0.5, 0.5), WordTiming("google", 0.5, 0.5))

        assert locate_phrase(found, words) == (0.5, 0.5)

    def test_a_phrase_with_no_words_gives_up(self) -> None:
        """A hand-built match with an empty phrase must not span everything."""
        empty = PhraseMatch(phrase="", token_index=0, text="ok google")
        assert locate_phrase(empty, PHRASE_WORDS) is None

    def test_a_negative_index_gives_up(self) -> None:
        """Defensive: a negative index would read from the end of the list."""
        bogus = PhraseMatch(phrase=PHRASE, token_index=-1, text="ok google")
        assert locate_phrase(bogus, PHRASE_WORDS) is None

    def test_the_timings_may_be_any_object_with_the_attributes(self) -> None:
        """``words`` is typed loosely to keep matching free of a circular import."""

        class Duck:
            def __init__(self, word: str, start: float, end: float) -> None:
                self.word = word
                self.start = start
                self.end = end

        found = match_phrase("ok google", (PHRASE,))
        assert found is not None
        assert locate_phrase(
            found, [Duck("ok", 0.5, 0.62), Duck("google", 0.65, 0.8)]
        ) == (0.5, 0.8)


# ==========================================================================
# ClipMode and the localisation configuration
# ==========================================================================

class TestClipMode:
    """What a snippet is cut to is a configured choice with two values."""

    def test_there_are_exactly_two_modes(self) -> None:
        """A third mode would need a third code path in _open_snippet."""
        assert set(ClipMode) == {ClipMode.PHRASE, ClipMode.WINDOW}

    @pytest.mark.parametrize(
        ("mode", "value"), [(ClipMode.PHRASE, "phrase"), (ClipMode.WINDOW, "window")]
    )
    def test_the_values_are_stable_strings(
        self, mode: ClipMode, value: str
    ) -> None:
        """The value is what a settings file would round-trip."""
        assert mode.value == value
        assert ClipMode(value) is mode

    def test_the_default_is_phrase(self) -> None:
        """The point of the feature: a snippet is the hotword, not a window."""
        assert VoiceGateConfig().clip_mode is ClipMode.PHRASE


class TestLocalizationConfig:
    """lead_ms, trail_ms and lookback_ms, and the frames they convert to."""

    def test_the_defaults_are_a_tight_clip_and_a_generous_lookback(self) -> None:
        """The lookback is what the gate can reach into, not what lands in the file."""
        config = VoiceGateConfig()

        assert config.lead_ms == 250
        assert config.trail_ms == 250
        assert config.lookback_ms == 8000
        assert config.lookback_ms > config.lead_ms + config.trail_ms, (
            "the lookback must comfortably exceed the clip it makes possible, "
            "because it also has to cover the decoder's reporting lag"
        )

    @pytest.mark.parametrize("lead_ms", [-1, -250, -10_000])
    def test_a_negative_lead_is_rejected(self, lead_ms: int) -> None:
        """Negative padding would move the clip start after the phrase."""
        with pytest.raises(ValueError, match=r"lead_ms must be >= 0"):
            VoiceGateConfig(lead_ms=lead_ms)

    @pytest.mark.parametrize("trail_ms", [-1, -250, -10_000])
    def test_a_negative_trail_is_rejected(self, trail_ms: int) -> None:
        """Same, at the other end of the clip."""
        with pytest.raises(ValueError, match=r"trail_ms must be >= 0"):
            VoiceGateConfig(trail_ms=trail_ms)

    def test_zero_lead_and_trail_are_legal(self) -> None:
        """Cutting exactly the phrase is a configuration, not an error."""
        config = VoiceGateConfig(lead_ms=0, trail_ms=0)
        assert config.lead_frames(SR) == 0
        assert config.trail_frames(SR) == 0

    @pytest.mark.parametrize("lookback_ms", [0, -1, -8000])
    def test_a_non_positive_lookback_is_rejected(self, lookback_ms: int) -> None:
        """A zero lookback retains nothing, so nothing could ever be located."""
        with pytest.raises(ValueError, match=r"lookback_ms must be > 0"):
            VoiceGateConfig(lookback_ms=lookback_ms)

    @pytest.mark.parametrize("clip_mode", ["phrase", 0, None, ClipMode])
    def test_a_clip_mode_that_is_not_a_clipmode_is_a_type_error(
        self, clip_mode: object
    ) -> None:
        """The string "phrase" is not the enum, and ``is`` comparisons would fail."""
        with pytest.raises(TypeError, match=r"clip_mode must be a ClipMode"):
            VoiceGateConfig(clip_mode=clip_mode)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("ms", "rate", "frames"),
        [(0, 16_000, 0), (250, 16_000, 4_000), (100, 16_000, 1_600),
         (250, 48_000, 12_000), (1, 16_000, 16)],
    )
    def test_lead_and_trail_convert_at_the_capture_rate(
        self, ms: int, rate: int, frames: int
    ) -> None:
        """Durations carry no rate of their own; the capture rate is passed in."""
        config = VoiceGateConfig(lead_ms=ms, trail_ms=ms)
        assert config.lead_frames(rate) == frames
        assert config.trail_frames(rate) == frames

    @pytest.mark.parametrize(
        ("lookback_ms", "rate", "frames"),
        [(8000, 16_000, 128_000), (2000, 16_000, 32_000), (8000, 48_000, 384_000)],
    )
    def test_lookback_frames_converts_at_the_capture_rate(
        self, lookback_ms: int, rate: int, frames: int
    ) -> None:
        """The retained window is sized in frames at whatever rate is captured."""
        config = VoiceGateConfig(lookback_ms=lookback_ms, pre_roll_ms=0)
        assert config.lookback_frames(rate) == frames

    def test_the_lookback_is_never_shorter_than_the_pre_roll(self) -> None:
        """WINDOW mode cuts out of the same buffer, so a short lookback would
        silently shorten every pre-roll instead of failing."""
        config = VoiceGateConfig(
            lookback_ms=100, pre_roll_ms=2000, post_roll_ms=0
        )

        assert config.lookback_ms < config.pre_roll_ms, "test setup"
        assert config.lookback_frames(SR) == config.pre_roll_frames(SR)
        assert config.lookback_frames(SR) == 32_000, (
            "the buffer must hold the whole 2000 ms pre-roll, not the 100 ms "
            "lookback that was asked for"
        )

    @pytest.mark.parametrize("pre_roll_ms", [0, 500, 1500, 8000])
    def test_lookback_frames_is_never_below_pre_roll_frames(
        self, pre_roll_ms: int
    ) -> None:
        """The invariant, across configurations either side of the crossover."""
        config = VoiceGateConfig(pre_roll_ms=pre_roll_ms, post_roll_ms=0)
        assert config.lookback_frames(SR) >= config.pre_roll_frames(SR)


# ==========================================================================
# the sink: cutting to the phrase
# ==========================================================================

class TestPhraseLocatedClip:
    """A phrase reported with timings is cut out of the buffered audio."""

    def test_the_clip_is_the_phrase_plus_the_lead_and_the_trail(
        self, tmp_path: Path
    ) -> None:
        """The headline test: the right *frames*, not merely the right length.

        The decoder reports at frame 16000, 3200 frames after the phrase ended,
        so a gate that started recording when it was told would miss the hotword
        entirely.  The ramp says which audio actually landed in the file.
        """
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(16_000, _final())])

        _feed(sink, 12)

        lead = config.lead_frames(SR)
        trail = config.trail_frames(SR)
        assert (lead, trail) == (1_600, 1_600), "test setup: 100 ms either side"

        _assert_clip(
            _only_wav(config),
            start_frame=PHRASE_START_FRAME - lead,
            frames=lead + (PHRASE_END_FRAME - PHRASE_START_FRAME) + trail,
        )

    def test_the_located_path_is_counted(self, tmp_path: Path) -> None:
        """clips_located is how an operator knows the clips are on the hotword."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(16_000, _final())])

        _feed(sink, 12)

        assert sink.clips_located == 1
        assert sink.clips_fallback == 0
        assert sink.snippets_written == 1
        assert sink.snapshot().clips_located == 1, (
            "snapshot() must agree with the property"
        )
        assert sink.snapshot().clips_fallback == 0

    def test_zero_lead_and_trail_cut_exactly_the_phrase(
        self, tmp_path: Path
    ) -> None:
        """With no padding the file IS the span the decoder reported."""
        config = _config(tmp_path, lead_ms=0, trail_ms=0)
        sink, _ = _make_sink(config, [(16_000, _final())])

        _feed(sink, 12)

        _assert_clip(
            _only_wav(config),
            start_frame=PHRASE_START_FRAME,
            frames=PHRASE_END_FRAME - PHRASE_START_FRAME,
        )

    @pytest.mark.parametrize(
        ("lead_ms", "trail_ms"), [(0, 100), (100, 0), (50, 200), (200, 50)]
    )
    def test_the_padding_follows_the_configuration(
        self, tmp_path: Path, lead_ms: int, trail_ms: int
    ) -> None:
        """Each pad is applied at its own end, and neither leaks into the other."""
        config = _config(tmp_path, lead_ms=lead_ms, trail_ms=trail_ms)
        sink, _ = _make_sink(config, [(16_000, _final())])

        _feed(sink, 12)

        lead = config.lead_frames(SR)
        trail = config.trail_frames(SR)
        _assert_clip(
            _only_wav(config),
            start_frame=PHRASE_START_FRAME - lead,
            frames=lead + (PHRASE_END_FRAME - PHRASE_START_FRAME) + trail,
        )

    def test_a_located_clip_is_shorter_and_earlier_than_the_window(
        self, tmp_path: Path
    ) -> None:
        """The whole point, measured against the mode it replaces.

        Two runs with the identical script and the identical geometry, differing
        only in ``clip_mode``: the located clip is half the length and starts
        1600 frames earlier, because the window is anchored to the moment the
        decoder spoke up and the phrase is already 3200 frames in the past by
        then.
        """
        script: list[tuple[int, Recognition]] = [(16_000, _final())]
        sizes: dict[str, tuple[int, int]] = {}
        for label, mode in (("located", ClipMode.PHRASE), ("window", ClipMode.WINDOW)):
            config = _config(
                tmp_path / label,
                clip_mode=mode,
                pre_roll_ms=500,          # 8000 frames
                post_roll_ms=500,         # 8000 frames
            )
            sink, _ = _make_sink(config, list(script))
            _feed(sink, 15)               # frames 0 .. 24000
            path = _only_wav(config)
            sizes[label] = (_wav_start_frame(path), _wav_frames(path))

        assert sizes["window"] == (pytest.approx(8_000, abs=FRAME_TOL), 16_000), (
            "test setup: the window is a 500 ms pre-roll before the report at "
            "frame 16000, then 500 ms more"
        )
        assert sizes["located"] == (pytest.approx(6_400, abs=FRAME_TOL), 8_000)
        assert sizes["located"][1] < sizes["window"][1], (
            "the located clip must be the shorter of the two"
        )
        assert sizes["located"][0] < sizes["window"][0], (
            "and must begin earlier: the window's pre-roll starts from the "
            "report, which is already past the phrase"
        )

    def test_the_event_describes_the_located_clip(self, tmp_path: Path) -> None:
        """The event's frames and start_frame must match the file on disk."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(16_000, _final())], events)

        _feed(sink, 12)

        assert len(events) == 1
        event = events[0]
        path = _only_wav(config)

        assert Path(event.path) == path
        assert event.frames == _wav_frames(path), (
            f"the event claims {event.frames} frames but the WAV holds "
            f"{_wav_frames(path)}"
        )
        assert event.start_frame == PHRASE_START_FRAME - config.lead_frames(SR)
        assert event.start_frame == 6_400
        assert event.duration_s == pytest.approx(event.frames / SR)
        assert event.phrase == PHRASE
        assert event.text == TEXT
        assert event.truncated is False

    def test_the_clip_is_taken_from_the_phrase_not_from_the_report(
        self, tmp_path: Path
    ) -> None:
        """The same phrase reported later must still produce the same clip.

        Two runs differing only in when the decoder spoke up: if the clip
        depended on the report time rather than on the timings, the files would
        differ.
        """
        prompt_config = _config(tmp_path / "prompt")
        prompt, _ = _make_sink(prompt_config, [(14_400, _final())])
        _feed(prompt, 12)

        late_config = _config(tmp_path / "late")
        late, _ = _make_sink(late_config, [(19_200, _final())])
        _feed(late, 14)

        assert _wav_frames(_only_wav(prompt_config)) == _wav_frames(
            _only_wav(late_config)
        )
        assert abs(
            _wav_start_frame(_only_wav(prompt_config))
            - _wav_start_frame(_only_wav(late_config))
        ) <= FRAME_TOL, "a later report must not move the clip"


class TestPhraseLocatedStreaming:
    """The trailing pad may not have been captured when the phrase is reported."""

    def test_the_snippet_stays_open_until_the_trail_arrives(
        self, tmp_path: Path
    ) -> None:
        """The decoder reported at frame 12800; the clip needs audio to 14400."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(12_800, _final())], events)

        _feed(sink, 8)                       # frames 0 .. 12800

        assert sink.clips_located == 1
        assert sink.recording is True, (
            "the trailing pad runs to frame 14400, which has not been captured"
        )
        assert events == [], "nothing may be announced until the file is complete"

    def test_the_following_chunks_fill_the_trailing_pad(
        self, tmp_path: Path
    ) -> None:
        """Once the audio arrives the file reaches its full lead+span+trail."""
        config = _config(tmp_path)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(12_800, _final())], events)

        _feed(sink, 9)                       # one more chunk: frames to 14400

        assert sink.recording is False
        assert len(events) == 1

        lead = config.lead_frames(SR)
        trail = config.trail_frames(SR)
        _assert_clip(
            _only_wav(config),
            start_frame=PHRASE_START_FRAME - lead,
            frames=lead + (PHRASE_END_FRAME - PHRASE_START_FRAME) + trail,
        )
        assert events[0].frames == _wav_frames(_only_wav(config))

    def test_a_clip_streamed_in_is_still_one_continuous_file(
        self, tmp_path: Path
    ) -> None:
        """Content, not length: the samples must run unbroken across the seam."""
        config = _config(tmp_path, lead_ms=0, trail_ms=100)
        sink, _ = _make_sink(config, [(12_800, _final())])

        _feed(sink, 9)

        path = _only_wav(config)
        with wave.open(str(path), "rb") as handle:
            raw = handle.readframes(handle.getnframes())
        samples = np.frombuffer(raw, dtype="<i2").astype(np.int64)
        steps = np.diff(samples)

        assert samples.size == (PHRASE_END_FRAME - PHRASE_START_FRAME) + 1_600
        assert np.all(steps == steps[0]), (
            "the ramp must increase by a constant step throughout: a jump means "
            "audio was repeated or dropped where the streamed part was joined on"
        )

    def test_the_streamed_clip_stops_at_its_computed_end(
        self, tmp_path: Path
    ) -> None:
        """The trailing pad is a length, not "until the next chunk boundary"."""
        # A 50 ms trail: the clip ends at frame 13600, which is 800 frames past
        # the report at 12800 -- half of one 1600-frame chunk.
        config = _config(tmp_path, lead_ms=0, trail_ms=50)
        sink, _ = _make_sink(config, [(12_800, _final())])

        _feed(sink, 12)

        _assert_clip(
            _only_wav(config),
            start_frame=PHRASE_START_FRAME,
            frames=(PHRASE_END_FRAME - PHRASE_START_FRAME) + 800,
        )


class TestFallbackToTheWindow:
    """Without trustworthy timings the gate cuts the old fixed window instead."""

    def test_a_result_with_no_timings_falls_back(self, tmp_path: Path) -> None:
        """A backend that reports no words is the documented fallback path."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(16_000, _final(words=()))])

        _feed(sink, 12)

        assert sink.clips_fallback == 1
        assert sink.clips_located == 0
        assert sink.snippets_written == 1

    def test_the_fallback_clip_is_the_old_pre_roll_plus_post_roll_shape(
        self, tmp_path: Path
    ) -> None:
        """Exactly the window the gate cut before localisation existed."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(16_000, _final(words=()))])

        _feed(sink, 12)

        pre = config.pre_roll_frames(SR)
        post = config.post_roll_frames(SR)
        assert (pre, post) == (3_200, 3_200), "test setup: 200 ms either side"

        _assert_clip(_only_wav(config), start_frame=16_000 - pre, frames=pre + post)

    def test_the_fallback_pre_roll_is_not_the_whole_lookback(
        self, tmp_path: Path
    ) -> None:
        """The buffer now holds seconds of audio; the window must not take it all.

        The lookback is ten times the pre-roll here, so a fallback that wrote
        ``snapshot()`` verbatim would produce a file ten times too long.
        """
        config = _config(tmp_path, lookback_ms=2000, pre_roll_ms=200)
        sink, _ = _make_sink(config, [(16_000, _final(words=()))])

        _feed(sink, 12)

        assert config.lookback_frames(SR) == 32_000, "test setup"
        assert _wav_frames(_only_wav(config)) == 3_200 + 3_200

    def test_window_mode_ignores_timings_that_are_available(
        self, tmp_path: Path
    ) -> None:
        """The mode wins over availability: WINDOW means WINDOW."""
        config = _config(tmp_path, clip_mode=ClipMode.WINDOW)
        sink, _ = _make_sink(config, [(16_000, _final())])

        _feed(sink, 12)

        assert sink.clips_located == 0, (
            "usable timings must not override the configured mode"
        )
        assert sink.clips_fallback == 1
        _assert_clip(
            _only_wav(config),
            start_frame=16_000 - config.pre_roll_frames(SR),
            frames=config.pre_roll_frames(SR) + config.post_roll_frames(SR),
        )

    def test_a_misaligned_word_list_falls_back(self, tmp_path: Path) -> None:
        """locate_phrase refusing the alignment must reach the gate as a fallback."""
        misaligned = _final(
            words=(WordTiming("um", 0.1, 0.2),) + PHRASE_WORDS
        )
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(16_000, misaligned)])

        _feed(sink, 12)

        assert sink.clips_located == 0
        assert sink.clips_fallback == 1
        assert _wav_frames(_only_wav(config)) == 6_400


class TestSanityCheckedTimings:
    """Timings that cannot be true are refused, and the gate widens instead.

    This is the guard that matters most.  Every case below would otherwise
    produce a confidently named file of the wrong moment -- the one failure mode
    that looks completely fine until somebody listens to it.  A fallback costs a
    wider clip; a trusted bad timing costs a wrong one.
    """

    def test_a_span_in_the_future_falls_back(self, tmp_path: Path) -> None:
        """The phrase claims to start after the audio the gate has consumed."""
        future = _final(
            words=(WordTiming("ok", 2.00, 2.12), WordTiming("google", 2.15, 2.30))
        )
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(16_000, future)])

        _feed(sink, 12)

        assert sink.clips_located == 0, (
            "a span starting at frame 32000 cannot be cut from a stream that "
            "has only reached frame 16000"
        )
        assert sink.clips_fallback == 1

    def test_a_span_before_the_stream_began_falls_back(
        self, tmp_path: Path
    ) -> None:
        """A large negative time is nonsense the arithmetic must not carry on with."""
        before = _final(
            words=(WordTiming("ok", -5.00, -4.90), WordTiming("google", -4.85, -4.70))
        )
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(16_000, before)])

        _feed(sink, 12)

        assert sink.clips_located == 0
        assert sink.clips_fallback == 1

    def test_a_span_older_than_the_lookback_falls_back(
        self, tmp_path: Path
    ) -> None:
        """The audio is real but no longer retained, so it cannot be cut."""
        old = _final(
            words=(WordTiming("ok", 0.10, 0.15), WordTiming("google", 0.16, 0.20))
        )
        # 200 ms of lookback: by frame 16000 only frames 12800 .. 16000 survive.
        config = _config(tmp_path, lookback_ms=200, pre_roll_ms=200)
        sink, _ = _make_sink(config, [(16_000, old)])

        _feed(sink, 12)

        assert config.lookback_frames(SR) == 3_200, "test setup"
        assert sink.clips_located == 0, (
            "frame 1600 was evicted from the lookback buffer, so cutting there "
            "would splice whatever audio now occupies those bytes"
        )
        assert sink.clips_fallback == 1

    @pytest.mark.parametrize(
        ("label", "words", "overrides"),
        [
            (
                "future",
                (WordTiming("ok", 2.00, 2.12), WordTiming("google", 2.15, 2.30)),
                {},
            ),
            (
                "before the stream",
                (WordTiming("ok", -5.0, -4.9), WordTiming("google", -4.85, -4.7)),
                {},
            ),
            (
                "older than the lookback",
                (WordTiming("ok", 0.10, 0.15), WordTiming("google", 0.16, 0.20)),
                {"lookback_ms": 200, "pre_roll_ms": 200},
            ),
        ],
    )
    def test_every_refused_span_still_produces_a_playable_file(
        self,
        tmp_path: Path,
        label: str,
        words: tuple[WordTiming, ...],
        overrides: dict[str, Any],
    ) -> None:
        """Falling back is a degradation, not a failure: a snippet is still written."""
        config = _config(tmp_path, **overrides)
        events: list[SnippetEvent] = []
        sink, _ = _make_sink(config, [(16_000, _final(words=words))], events)

        _feed(sink, 12)

        assert sink.clips_fallback == 1, f"{label} must fall back"
        assert sink.error is None, f"{label} must not be recorded as a failure"
        assert len(events) == 1

        path = _only_wav(config)
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == SR
            assert handle.getnframes() > 0, f"{label} produced an empty file"
        assert events[0].frames == _wav_frames(path)

    def test_a_zero_length_span_falls_back(self, tmp_path: Path) -> None:
        """A phrase that occupies no time cannot be a clip, padded or not."""
        config = _config(tmp_path, lead_ms=0, trail_ms=0)
        degenerate = _final(
            words=(WordTiming("ok", 0.5, 0.5), WordTiming("google", 0.5, 0.5))
        )
        sink, _ = _make_sink(config, [(16_000, degenerate)])

        _feed(sink, 12)

        assert sink.clips_located == 0
        assert sink.clips_fallback == 1

    def test_a_lead_reaching_past_the_retained_audio_falls_back(
        self, tmp_path: Path
    ) -> None:
        """The clip must be cut whole; a half-length lead is not an option."""
        config = _config(tmp_path, lead_ms=1000, trail_ms=0, pre_roll_ms=200)
        early = _final(
            words=(WordTiming("ok", 0.05, 0.10), WordTiming("google", 0.12, 0.20))
        )
        sink, _ = _make_sink(config, [(16_000, early)])

        _feed(sink, 12)

        assert config.lead_frames(SR) == 16_000, "test setup: the lead precedes 0"
        assert sink.clips_located == 0
        assert sink.clips_fallback == 1


class TestLocatedClipLimits:
    """max_snippet_ms bounds a located clip exactly as it bounds a window."""

    def test_a_long_phrase_is_capped_at_the_ceiling(self, tmp_path: Path) -> None:
        """A decoder reporting a five-second "phrase" must not write five seconds."""
        config = _config(
            tmp_path,
            lead_ms=100,
            trail_ms=100,
            pre_roll_ms=100,
            post_roll_ms=100,
            max_snippet_ms=300,          # 4800 frames
        )
        long_phrase = _final(
            words=(WordTiming("ok", 0.50, 0.70), WordTiming("google", 0.75, 1.00))
        )
        sink, _ = _make_sink(config, [(16_000, long_phrase)])

        _feed(sink, 12)

        ceiling = config.max_snippet_frames(SR)
        assert ceiling == 4_800, "test setup: 300 ms at 16 kHz"
        path = _only_wav(config)
        assert _wav_frames(path) == ceiling, (
            f"the located clip ran to {_wav_frames(path)} frames, past the "
            f"{ceiling}-frame ceiling"
        )
        assert abs(
            _wav_start_frame(path) - (PHRASE_START_FRAME - config.lead_frames(SR))
        ) <= FRAME_TOL, "the ceiling must trim the end, not move the start"

    def test_a_clip_inside_the_ceiling_is_untouched(self, tmp_path: Path) -> None:
        """The cap is a ceiling, not a length."""
        config = _config(tmp_path, max_snippet_ms=5000)
        sink, _ = _make_sink(config, [(16_000, _final())])

        _feed(sink, 12)

        assert _wav_frames(_only_wav(config)) == 8_000
        assert sink.snippets_truncated == 0


class TestLocatedClipAndDiscontinuity:
    """The anchor maps the decoder's clock onto absolute stream frames.

    Both tests below feed **overlapping** windows -- 4800-frame windows every
    1600 frames -- because that is the only geometry where the two differ: a
    chunk contributes 1600 new frames but declares itself 4800 frames long, so
    an anchor taken from ``chunk.start_frame`` sits 3200 frames before the audio
    the recogniser was actually handed.
    """

    def test_a_located_clip_is_correct_with_overlapping_windows(
        self, tmp_path: Path
    ) -> None:
        """The control: no discontinuity, so the anchor is taken on the first chunk."""
        config = _config(tmp_path, lead_ms=0, trail_ms=0)
        # "ok google" at absolute frames 9600 .. 10400, reported at frame 12800.
        words = (WordTiming("ok", 0.600, 0.620), WordTiming("google", 0.625, 0.650))
        sink, _ = _make_sink(config, [(12_800, _final(words=words))])

        _feed(sink, 10, hop=1_600, window=4_800)

        assert sink.clips_located == 1
        _assert_clip(_only_wav(config), start_frame=9_600, frames=800)

    def test_a_located_clip_is_correct_after_a_discontinuity(
        self, tmp_path: Path
    ) -> None:
        """A dropped-audio seam must not move a later clip off the hotword."""
        config = _config(tmp_path, lead_ms=0, trail_ms=0)
        words = (WordTiming("ok", 0.600, 0.620), WordTiming("google", 0.625, 0.650))
        sink, recognizer = _make_sink(config, [(12_800, _final(words=words))])

        _feed(sink, 10, hop=1_600, window=4_800, discontinuous_at=2)

        assert recognizer.resets == 1, "test setup: the seam must have been reported"
        assert sink.clips_located == 1
        _assert_clip(_only_wav(config), start_frame=9_600, frames=800)

    def test_a_discontinuity_before_any_match_does_not_break_the_fallback(
        self, tmp_path: Path
    ) -> None:
        """Whatever the anchor does, a fallback clip is still written."""
        config = _config(tmp_path)
        sink, _ = _make_sink(config, [(12_800, _final(words=()))])

        _feed(sink, 12, hop=1_600, window=4_800, discontinuous_at=2)

        assert sink.clips_fallback == 1
        assert sink.error is None
        assert _wav_frames(_only_wav(config)) > 0
