"""Tests for echochamber.voicegate.recognizer, with no Vosk and no model.

That constraint is the point rather than an inconvenience: the module promises
that a checkout without the optional ``voice-gate`` extra still imports, still
type-checks and still runs the whole gate on :class:`NullRecognizer` and
:class:`ScriptedRecognizer`.  A test file that needed ``import vosk`` would
quietly stop asserting that.  Nothing here imports :mod:`vosk`, and
:func:`load_vosk_recognizer` is exercised only on the path that fails *before*
the lazy import happens -- the missing model directory.

Two functions carry most of the weight:

* :func:`float32_to_pcm16` is the only place floats become bytes, so it is
  asserted as **exact bytes**, including the clipping behaviour.  Wrapping
  instead of clipping turns a loud passage into full-scale noise the decoder
  has no chance with, and the wrap is invisible in any test that only checks
  the length.
* :func:`parse_vosk_result` runs on the pipeline's **consumer thread**, where a
  raised exception kills the consumer and takes capture down with it.  Every
  malformed input below therefore asserts a degraded :class:`Recognition`, not
  a raise -- that is the invariant, and "malformed JSON" is not hypothetical
  when the JSON crossed a pipe from another interpreter.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pytest

from echochamber.voicegate.recognizer import (
    NullRecognizer,
    Recognition,
    Recognizer,
    ScriptedRecognizer,
    float32_to_pcm16,
    load_vosk_recognizer,
    parse_vosk_result,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def script(*items: tuple[int, str, bool]) -> list[tuple[int, Recognition]]:
    """Build a ScriptedRecognizer script from ``(offset, text, final)`` triples."""
    return [
        (offset, Recognition(text=text, final=final))
        for offset, text, final in items
    ]


def pcm(n: int) -> bytes:
    """``n`` bytes of PCM whose content is irrelevant; only the length matters."""
    return bytes(n)


def samples(*values: float) -> np.ndarray:
    """A 1-D float32 array, the shape float32_to_pcm16 is given by the chunker."""
    return np.array(values, dtype=np.float32)


def as_int16(raw: bytes) -> np.ndarray:
    """Reinterpret PCM bytes as little-endian int16, explicitly, host-independently."""
    return np.frombuffer(raw, dtype="<i2")


# ==========================================================================
# Recognition
# ==========================================================================

class TestRecognition:
    """The one value type that crosses every backend boundary."""

    def test_stores_its_fields(self) -> None:
        """text/final/confidence are kept verbatim."""
        rec = Recognition(text="ok chamber", final=True, confidence=0.875)
        assert rec.text == "ok chamber"
        assert rec.final is True
        assert rec.confidence == pytest.approx(0.875)

    def test_confidence_defaults_to_zero(self) -> None:
        """Backends that report no confidence get 0.0, not None."""
        assert Recognition(text="hi", final=False).confidence == 0.0

    def test_empty_final_text_is_legal(self) -> None:
        """Small models emit empty finals during silence; that is not an error."""
        assert Recognition(text="", final=True).text == ""

    def test_repr_says_final(self) -> None:
        """A settled result is labelled 'final' in the repr."""
        assert repr(Recognition(text="hey", final=True)) == (
            "Recognition(final, text='hey')"
        )

    def test_repr_says_partial(self) -> None:
        """An unsettled result is labelled 'partial' -- the gate never acts on it."""
        assert repr(Recognition(text="hey", final=False)) == (
            "Recognition(partial, text='hey')"
        )

    def test_repr_quotes_the_text(self) -> None:
        """!r on the text keeps whitespace and emptiness visible in a log."""
        assert repr(Recognition(text="", final=True)) == "Recognition(final, text='')"

    def test_is_frozen(self) -> None:
        """Results are shared across threads, so they are immutable."""
        rec = Recognition(text="x", final=True)
        with pytest.raises(Exception):
            rec.text = "y"  # type: ignore[misc]

    def test_equality_is_by_value(self) -> None:
        """Tests and dedup logic both compare results structurally."""
        assert Recognition("a", True, 0.5) == Recognition("a", True, 0.5)
        assert Recognition("a", True, 0.5) != Recognition("a", False, 0.5)


# ==========================================================================
# NullRecognizer
# ==========================================================================

class TestNullRecognizer:
    """The default backend: wired in everywhere, recognises nothing."""

    def test_accept_pcm_always_returns_an_empty_list(self) -> None:
        """No audio ever produces a result, so the gate never fires."""
        rec = NullRecognizer()
        assert rec.accept_pcm(b"") == []
        assert rec.accept_pcm(pcm(3200)) == []
        assert rec.accept_pcm(b"\x00\x01" * 10_000) == []

    def test_returns_a_fresh_list_each_call(self) -> None:
        """A shared mutable default would let a caller corrupt the next result."""
        rec = NullRecognizer()
        first = rec.accept_pcm(pcm(2))
        first.append(Recognition("polluted", True))
        assert rec.accept_pcm(pcm(2)) == []

    def test_reset_is_a_harmless_noop(self) -> None:
        """There is no state to discard, but the protocol still requires reset()."""
        rec = NullRecognizer()
        rec.reset()
        rec.reset()
        assert rec.accept_pcm(pcm(2)) == []

    def test_closed_starts_false(self) -> None:
        """A fresh recogniser is open."""
        assert NullRecognizer().closed is False

    def test_close_is_idempotent(self) -> None:
        """The owning sink closes idempotently, so this must too."""
        rec = NullRecognizer()
        rec.close()
        rec.close()
        rec.close()
        assert rec.closed is True

    def test_accept_pcm_still_works_after_close(self) -> None:
        """Closing degrades to 'no snippets', never to an exception."""
        rec = NullRecognizer()
        rec.close()
        assert rec.accept_pcm(pcm(160)) == []

    def test_repr_shows_closed_state(self) -> None:
        """The repr is the only state worth logging."""
        rec = NullRecognizer()
        assert repr(rec) == "NullRecognizer(closed=False)"
        rec.close()
        assert repr(rec) == "NullRecognizer(closed=True)"

    def test_satisfies_the_recognizer_protocol(self) -> None:
        """Recognizer is runtime_checkable so backends are interchangeable."""
        assert isinstance(NullRecognizer(), Recognizer)


# ==========================================================================
# ScriptedRecognizer
# ==========================================================================

class TestScriptedRecognizer:
    """A deterministic stand-in for a model, driven by cumulative byte offsets."""

    def test_emits_nothing_before_the_offset_is_reached(self) -> None:
        """A result due at 100 bytes must not appear after 99."""
        rec = ScriptedRecognizer(script((100, "hey chamber", True)))
        assert rec.accept_pcm(pcm(50)) == []
        assert rec.accept_pcm(pcm(49)) == []
        assert rec.consumed == 99

    def test_emits_on_the_call_that_reaches_the_offset(self) -> None:
        """The offset is inclusive: reaching exactly it is enough."""
        rec = ScriptedRecognizer(script((100, "hey chamber", True)))
        rec.accept_pcm(pcm(99))
        assert rec.accept_pcm(pcm(1)) == [Recognition("hey chamber", True)]

    def test_emits_at_the_right_cumulative_offsets(self) -> None:
        """Offsets are cumulative across calls, not per call."""
        rec = ScriptedRecognizer(
            script((10, "one", True), (20, "two", True), (30, "three", True))
        )
        assert rec.accept_pcm(pcm(10)) == [Recognition("one", True)]
        assert rec.accept_pcm(pcm(5)) == []
        assert rec.accept_pcm(pcm(5)) == [Recognition("two", True)]
        assert rec.accept_pcm(pcm(10)) == [Recognition("three", True)]

    def test_several_results_due_at_once_come_out_together(self) -> None:
        """One big buffer can settle several scripted results in one call."""
        rec = ScriptedRecognizer(
            script((10, "a", False), (20, "b", False), (30, "c", True))
        )
        assert rec.accept_pcm(pcm(1000)) == [
            Recognition("a", False),
            Recognition("b", False),
            Recognition("c", True),
        ]

    def test_each_result_is_emitted_exactly_once(self) -> None:
        """A scripted result must not repeat on every subsequent buffer."""
        rec = ScriptedRecognizer(script((4, "once", True)))
        assert rec.accept_pcm(pcm(4)) == [Recognition("once", True)]
        assert rec.accept_pcm(pcm(4)) == []
        assert rec.accept_pcm(pcm(400)) == []

    def test_consumed_tracks_total_bytes(self) -> None:
        """consumed is the running total, which is what offsets are measured in."""
        rec = ScriptedRecognizer()
        assert rec.consumed == 0
        rec.accept_pcm(pcm(160))
        rec.accept_pcm(pcm(320))
        rec.accept_pcm(b"")
        assert rec.consumed == 480

    def test_resets_counts_reset_calls(self) -> None:
        """resets is how a test proves a RESET frame reached the recogniser."""
        rec = ScriptedRecognizer()
        assert rec.resets == 0
        rec.reset()
        rec.reset()
        assert rec.resets == 2

    def test_reset_does_not_rewind_the_script_or_the_byte_total(self) -> None:
        """An offset means 'after this much audio', discontinuity or not."""
        rec = ScriptedRecognizer(script((10, "later", True)))
        rec.accept_pcm(pcm(9))
        rec.reset()
        assert rec.consumed == 9, "reset must not zero the byte total"
        assert rec.accept_pcm(pcm(1)) == [Recognition("later", True)]

    def test_out_of_order_script_is_sorted_on_construction(self) -> None:
        """Callers may script results in any order; emission is by offset."""
        rec = ScriptedRecognizer(
            script((30, "third", True), (10, "first", True), (20, "second", True))
        )
        assert rec.accept_pcm(pcm(30)) == [
            Recognition("first", True),
            Recognition("second", True),
            Recognition("third", True),
        ]

    def test_out_of_order_script_still_emits_incrementally(self) -> None:
        """Sorting must happen once, not be faked by draining everything at the end."""
        rec = ScriptedRecognizer(script((20, "b", True), (5, "a", True)))
        assert rec.accept_pcm(pcm(5)) == [Recognition("a", True)]
        assert rec.accept_pcm(pcm(15)) == [Recognition("b", True)]

    def test_zero_offset_results_fire_on_the_first_call(self) -> None:
        """Offset 0 is already reached, including by an empty buffer."""
        rec = ScriptedRecognizer(script((0, "immediate", True)))
        assert rec.accept_pcm(b"") == [Recognition("immediate", True)]

    @pytest.mark.parametrize("empty", [None, []])
    def test_empty_script_behaves_like_nullrecognizer(self, empty: Any) -> None:
        """No script means no results, ever -- the NullRecognizer contract."""
        rec = ScriptedRecognizer(empty)
        assert rec.accept_pcm(pcm(100_000)) == []
        assert rec.accept_pcm(b"") == []
        assert rec.consumed == 100_000

    def test_close_is_idempotent(self) -> None:
        """Matches the Recognizer contract for repeated close()."""
        rec = ScriptedRecognizer()
        assert rec.closed is False
        rec.close()
        rec.close()
        assert rec.closed is True

    def test_repr_shows_progress(self) -> None:
        """The repr names how much of the script has been emitted."""
        rec = ScriptedRecognizer(script((4, "a", True), (8, "b", True)))
        assert repr(rec) == "ScriptedRecognizer(consumed=0, emitted=0/2, closed=False)"
        rec.accept_pcm(pcm(4))
        assert repr(rec) == "ScriptedRecognizer(consumed=4, emitted=1/2, closed=False)"

    def test_satisfies_the_recognizer_protocol(self) -> None:
        """Interchangeable with the real backend, which is why it can stand in."""
        assert isinstance(ScriptedRecognizer(), Recognizer)


# ==========================================================================
# the Recognizer protocol itself
# ==========================================================================

class TestRecognizerProtocol:
    """Structural typing, checked at runtime the way ChunkSink is."""

    def test_an_object_with_all_three_methods_satisfies_it(self) -> None:
        """accept_pcm/reset/close is the whole contract."""

        class Duck:
            def accept_pcm(self, pcm: bytes) -> list[Recognition]:
                return []

            def reset(self) -> None:
                pass

            def close(self) -> None:
                pass

        assert isinstance(Duck(), Recognizer)

    def test_a_bare_object_does_not_satisfy_it(self) -> None:
        """Structural checks must still reject something unrelated."""

        class NotARecognizer:
            pass

        assert not isinstance(NotARecognizer(), Recognizer)

    def test_missing_close_does_not_satisfy_it(self) -> None:
        """close() is part of the protocol; the sink calls it on shutdown."""

        class HalfARecognizer:
            def accept_pcm(self, pcm: bytes) -> list[Recognition]:
                return []

            def reset(self) -> None:
                pass

        assert not isinstance(HalfARecognizer(), Recognizer)


# ==========================================================================
# float32_to_pcm16
# ==========================================================================

class TestFloat32ToPcm16:
    """The float -> bytes conversion, asserted as bytes rather than as floats."""

    def test_empty_array_gives_empty_bytes(self) -> None:
        """An empty chunk must not become a one-sample frame."""
        assert float32_to_pcm16(np.zeros(0, dtype=np.float32)) == b""

    def test_silence_is_all_zero_bytes(self) -> None:
        """Zero maps to zero, exactly, with no DC offset."""
        assert float32_to_pcm16(samples(0.0, 0.0, 0.0)) == b"\x00\x00" * 3

    def test_exact_bytes_for_a_known_input(self) -> None:
        """Full scale and half scale, spelled out little-endian byte by byte.

        0.5 * 32767 is 16383.5, which truncates toward zero to 16383 = 0x3FFF,
        so the expected bytes are ``ff 3f``.
        """
        assert float32_to_pcm16(samples(1.0, -1.0, 0.0, 0.5, -0.5)) == (
            b"\xff\x7f"      #  32767
            b"\x01\x80"      # -32767
            b"\x00\x00"      #      0
            b"\xff\x3f"      #  16383
            b"\x01\xc0"      # -16383
        )

    def test_is_two_bytes_per_sample(self) -> None:
        """16-bit PCM: the byte count is exactly 2n, which sizing code relies on."""
        for n in (1, 2, 7, 160, 3000, 48_000):
            raw = float32_to_pcm16(np.zeros(n, dtype=np.float32))
            assert len(raw) == 2 * n, f"{n} samples must be {2 * n} bytes"

    def test_is_little_endian_regardless_of_host(self) -> None:
        """The bytes cross a pipe to another interpreter, so '<i2' is forced.

        256 as int16 is 0x0100; little-endian puts the low byte first, so a
        native-endian implementation on a big-endian host would emit b"\\x01\\x00".
        """
        raw = float32_to_pcm16(samples(256 / 32767.0))
        assert raw == b"\x00\x01", (
            f"expected little-endian b'\\x00\\x01', got {raw!r}"
        )

    def test_clips_rather_than_wrapping_at_positive_full_scale(self) -> None:
        """+2.0 must saturate at +32767, not wrap to a large negative value.

        Wrapping is the failure that matters: 2.0 * 32767 is 65534, which as a
        16-bit two's-complement value is -2, so an unclipped conversion turns
        the loudest part of a phrase into near-silence with a sign flip.
        """
        got = as_int16(float32_to_pcm16(samples(2.0, 5.5, 100.0, 1.0)))
        assert list(got) == [32767, 32767, 32767, 32767], (
            f"positive overload must clip to full scale, got {list(got)}"
        )
        assert np.all(got > 0), "clipped positive samples must stay positive"

    def test_clips_rather_than_wrapping_at_negative_full_scale(self) -> None:
        """-2.0 must saturate near -32768, not wrap positive."""
        got = as_int16(float32_to_pcm16(samples(-2.0, -9.0, -1.0)))
        assert list(got) == [-32767, -32767, -32767], (
            f"negative overload must clip to full scale, got {list(got)}"
        )
        assert np.all(got < 0), "clipped negative samples must stay negative"

    def test_clipped_values_stay_inside_the_int16_range(self) -> None:
        """Whatever the rounding, nothing may land outside [-32768, 32767]."""
        loud = np.linspace(-50.0, 50.0, 501, dtype=np.float32)
        got = as_int16(float32_to_pcm16(loud))
        assert got.min() >= -32768 and got.max() <= 32767
        assert np.all(np.sign(got) == np.sign(np.round(np.clip(loud, -1, 1) * 32767))), (
            "clipping must preserve the sign of every sample"
        )

    def test_in_range_samples_are_monotonic(self) -> None:
        """A rising ramp must stay rising -- a wrap would show as a cliff."""
        ramp = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
        got = as_int16(float32_to_pcm16(ramp)).astype(np.int64)
        assert np.all(np.diff(got) >= 0), (
            f"conversion is not monotonic; first drop at index "
            f"{int(np.argmax(np.diff(got) < 0))}"
        )

    def test_accepts_a_plain_list_via_asarray(self) -> None:
        """np.asarray means a list works, which the GUI demo path relies on."""
        assert float32_to_pcm16(np.asarray([1.0, -1.0])) == b"\xff\x7f\x01\x80"

    def test_infinities_clip_to_full_scale(self) -> None:
        """np.clip pins +/-inf to +/-1.0, so an inf sample is loud, not garbage."""
        raw = float32_to_pcm16(np.array([np.inf, -np.inf], dtype=np.float32))
        assert list(as_int16(raw)) == [32767, -32767]


# ==========================================================================
# parse_vosk_result
# ==========================================================================

class TestParseVoskResult:
    """Runs on the consumer thread: it must degrade, never raise."""

    def test_final_reads_the_text_field(self) -> None:
        """Vosk finals arrive as {"text": ...}."""
        got = parse_vosk_result('{"text": "hey chamber"}', final=True)
        assert got == Recognition(text="hey chamber", final=True, confidence=0.0)

    def test_partial_reads_the_partial_field(self) -> None:
        """Vosk partials arrive as {"partial": ...}, under a different key."""
        got = parse_vosk_result('{"partial": "hey cham"}', final=False)
        assert got == Recognition(text="hey cham", final=False, confidence=0.0)

    def test_a_final_ignores_a_partial_field(self) -> None:
        """The key read is chosen by `final`, not by whichever key is present."""
        assert parse_vosk_result('{"partial": "wrong"}', final=True).text == ""

    def test_a_partial_ignores_a_text_field(self) -> None:
        """Symmetrically: a partial never reads the settled `text` key."""
        assert parse_vosk_result('{"text": "wrong"}', final=False).text == ""

    @pytest.mark.parametrize("final", [True, False])
    def test_empty_object_gives_empty_text(self, final: bool) -> None:
        """Vosk returns {} rather than an error when it has nothing to say."""
        assert parse_vosk_result("{}", final=final) == Recognition("", final, 0.0)

    @pytest.mark.parametrize("final", [True, False])
    def test_the_final_flag_is_passed_through_verbatim(self, final: bool) -> None:
        """The caller knows whether it called Result() or PartialResult()."""
        assert parse_vosk_result('{"text":"a","partial":"a"}', final=final).final is final

    @pytest.mark.parametrize(
        "payload",
        ["", "not json", "{", '{"text":}', "<html>error</html>", "\x00\x01"],
    )
    @pytest.mark.parametrize("final", [True, False])
    def test_malformed_json_degrades_instead_of_raising(
        self, payload: str, final: bool
    ) -> None:
        """A raise here kills the consumer thread and stops capture entirely."""
        assert parse_vosk_result(payload, final=final) == Recognition("", final, 0.0)

    @pytest.mark.parametrize("payload", ["[1,2,3]", '"a string"', "42", "null", "true"])
    def test_non_dict_json_degrades(self, payload: str) -> None:
        """Valid JSON of the wrong shape is still 'nothing recognised'."""
        assert parse_vosk_result(payload, final=True) == Recognition("", True, 0.0)

    def test_none_payload_degrades(self) -> None:
        """json.loads(None) raises TypeError, which is caught alongside ValueError."""
        assert parse_vosk_result(None, final=True) == Recognition("", True, 0.0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [123, None, ["a"], {"b": 1}, True, 1.5])
    def test_non_string_text_becomes_empty(self, value: object) -> None:
        """A version-skewed worker must not put a non-string into Recognition.text."""
        got = parse_vosk_result(json.dumps({"text": value}), final=True)
        assert got.text == "", f"text={value!r} should degrade to '', got {got.text!r}"
        assert isinstance(got.text, str)

    def test_confidence_is_the_mean_of_the_word_confidences(self) -> None:
        """Vosk reports per-word conf; the Recognition carries their average."""
        payload = json.dumps(
            {
                "text": "hey chamber",
                "result": [
                    {"word": "hey", "conf": 1.0},
                    {"word": "chamber", "conf": 0.5},
                ],
            }
        )
        got = parse_vosk_result(payload, final=True)
        assert got.text == "hey chamber"
        assert got.confidence == pytest.approx(0.75)

    def test_confidence_averages_over_three_words(self) -> None:
        """The mean is over the words that carry a conf, not a running last-wins."""
        payload = json.dumps(
            {"text": "a b c", "result": [{"conf": 0.9}, {"conf": 0.6}, {"conf": 0.3}]}
        )
        assert parse_vosk_result(payload, final=True).confidence == pytest.approx(0.6)

    def test_confidence_is_zero_when_there_is_no_result_list(self) -> None:
        """SetWords(False), or a partial, means no per-word scores at all."""
        assert parse_vosk_result('{"text":"hi"}', final=True).confidence == 0.0

    @pytest.mark.parametrize(
        "result", [[], "not a list", 5, None, {"conf": 1.0}]
    )
    def test_confidence_is_zero_for_a_malformed_result_field(
        self, result: object
    ) -> None:
        """Anything that is not a non-empty list of dicts contributes nothing."""
        payload = json.dumps({"text": "hi", "result": result})
        assert parse_vosk_result(payload, final=True).confidence == 0.0

    @pytest.mark.parametrize(
        "conf", [None, "0.9", [0.9], {"a": 1}]
    )
    def test_words_with_an_invalid_conf_are_skipped(self, conf: object) -> None:
        """A non-numeric conf is dropped rather than crashing float()."""
        payload = json.dumps({"text": "hi", "result": [{"word": "hi", "conf": conf}]})
        assert parse_vosk_result(payload, final=True).confidence == 0.0

    def test_words_missing_conf_entirely_are_skipped(self) -> None:
        """A word dict with no conf key contributes nothing to the mean."""
        payload = json.dumps(
            {"text": "a b", "result": [{"word": "a"}, {"word": "b", "conf": 0.4}]}
        )
        assert parse_vosk_result(payload, final=True).confidence == pytest.approx(0.4)

    def test_non_dict_entries_in_the_result_list_are_skipped(self) -> None:
        """A skewed worker may put anything in there; only dicts are read."""
        payload = json.dumps({"text": "a", "result": ["nope", 7, {"conf": 0.8}]})
        assert parse_vosk_result(payload, final=True).confidence == pytest.approx(0.8)

    def test_integer_confidences_are_accepted(self) -> None:
        """conf may be an int (1) rather than a float (1.0)."""
        payload = json.dumps({"text": "a", "result": [{"conf": 1}, {"conf": 0}]})
        assert parse_vosk_result(payload, final=True).confidence == pytest.approx(0.5)

    def test_returns_a_recognition_even_for_total_garbage(self) -> None:
        """The return type is unconditional; callers never get None."""
        assert isinstance(parse_vosk_result("!!!", final=False), Recognition)


# ==========================================================================
# load_vosk_recognizer -- only the pre-import failure path
# ==========================================================================

class TestLoadVoskRecognizer:
    """The success path needs Vosk and a model, so only the guard is tested here."""

    def test_missing_model_directory_raises_filenotfounderror(
        self, tmp_path: Any
    ) -> None:
        """The directory check runs before `import vosk`, so this works without it."""
        missing = tmp_path / "no-such-model"
        with pytest.raises(FileNotFoundError, match="Vosk model directory not found"):
            load_vosk_recognizer(str(missing), 16_000)

    def test_the_error_names_the_setup_script(self, tmp_path: Any) -> None:
        """A missing model is a setup problem; the message must say how to fix it."""
        missing = tmp_path / "absent"
        with pytest.raises(
            FileNotFoundError, match=re.escape("scripts/setup_voice_gate.py")
        ):
            load_vosk_recognizer(str(missing), 16_000)

    def test_the_error_names_the_path_it_looked_for(self, tmp_path: Any) -> None:
        """Naming the path is what distinguishes a typo from a missing download."""
        missing = tmp_path / "vosk-model-small-en-us"
        with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
            load_vosk_recognizer(str(missing), 16_000, ("ok chamber",))

    def test_a_file_is_not_a_model_directory(self, tmp_path: Any) -> None:
        """os.path.isdir, not os.path.exists: a downloaded .zip is not a model."""
        archive = tmp_path / "model.zip"
        archive.write_bytes(b"PK\x03\x04not really")
        with pytest.raises(FileNotFoundError, match="Vosk model directory not found"):
            load_vosk_recognizer(str(archive), 16_000)
