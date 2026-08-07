"""Tests for echochamber.voicegate.matching -- the gate's whole policy.

Everything under test here is a pure function over strings, which is the point
of the module existing at all: whether an utterance counts as a wake phrase is
decided without Vosk, without a model and without audio, so it can be pinned
down exhaustively by feeding text in and asserting what came back.  Nothing in
this file imports a recogniser.

Three invariants carry most of the weight:

* **Word boundaries, not substrings.**  ``"ok google"`` must fire on
  ``"ok google turn it up"`` and must *not* fire on ``"look google it"``, in
  which it is a genuine substring.  A single ``in`` test passes the first and
  fails the second, so that pair is asserted directly rather than left to a
  generic "matching works" smoke test.
* **The empty needle never matches.**  A phrase that normalises away to nothing
  must be skipped, because the alternative -- treating it as matching at index
  0 -- turns one bad config entry into a gate that records everything.
* **Order independence.**  ``match_phrase`` is called with a configured tuple
  whose order the operator chose arbitrarily, so every multi-match test passes
  the same phrases in both orders and asserts the identical result.
"""

from __future__ import annotations

import pytest

from echochamber.voicegate.matching import (
    PhraseMatch,
    find_tokens,
    match_phrase,
    normalize,
    tokenize,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _both_orders(text: str, phrases: list[str]) -> PhraseMatch | None:
    """Match ``phrases`` against ``text`` in both orders, asserting they agree.

    Returns the single result both orders produced, so a caller can go on to
    assert what it is without repeating the ordering check every time.
    """
    forward = match_phrase(text, phrases)
    backward = match_phrase(text, list(reversed(phrases)))
    assert forward == backward, (
        f"match_phrase must not depend on the order phrases are configured in: "
        f"{phrases!r} gave {forward!r} but {list(reversed(phrases))!r} gave "
        f"{backward!r}"
    )
    return forward


class TestNormalize:
    """normalize() folds ASR punctuation and case away without losing words."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("OK GOOGLE", "ok google"),
            ("Ok Google", "ok google"),
            ("oK gOoGlE", "ok google"),
            ("HEY", "hey"),
        ],
    )
    def test_case_is_folded_to_lowercase(self, raw: str, expected: str) -> None:
        """Case never survives normalisation, whatever case the input used."""
        assert normalize(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("ok, google", "ok google"),
            ("ok. google!", "ok google"),
            ("ok--google", "ok google"),
            ("(ok) [google]", "ok google"),
            ("ok/google", "ok google"),
            ("ok_google", "ok google"),
            ("ok:google;", "ok google"),
            ("<ok> \"google\"", "ok google"),
        ],
    )
    def test_punctuation_becomes_a_word_separator(
        self, raw: str, expected: str
    ) -> None:
        """Punctuation turns into a space, so it splits words instead of joining."""
        assert normalize(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  ok   google  ", "ok google"),
            ("ok\tgoogle", "ok google"),
            ("ok\n\ngoogle", "ok google"),
            ("\tok  \r\n google \t", "ok google"),
        ],
    )
    def test_whitespace_is_collapsed_and_stripped(
        self, raw: str, expected: str
    ) -> None:
        """Runs of whitespace collapse to one space with no leading or trailing pad."""
        assert normalize(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("what's", "what's"),
            ("What's up", "what's up"),
            ("don't stop", "don't stop"),
            ("'tis", "'tis"),
            ("rock'n'roll", "rock'n'roll"),
        ],
    )
    def test_apostrophes_survive_as_part_of_the_word(
        self, raw: str, expected: str
    ) -> None:
        """An apostrophe is a letter here: "what's" stays ONE token, not two.

        Splitting contractions would make a configured ``"what's up"``
        unmatchable against the ASR text that actually contains it.
        """
        assert normalize(raw) == expected
        assert len(tokenize(raw)) == len(expected.split())

    @pytest.mark.parametrize(
        ("accented", "plain"),
        [
            ("Café", "cafe"),
            ("café", "cafe"),
            ("naïve", "naive"),
            ("Über", "uber"),
            ("Ångström", "angstrom"),
        ],
    )
    def test_accents_fold_via_nfkd(self, accented: str, plain: str) -> None:
        """NFKD plus dropping combining marks makes "Café" and "cafe" compare equal."""
        assert normalize(accented) == plain
        assert normalize(accented) == normalize(plain)

    def test_precomposed_and_decomposed_accents_normalise_identically(self) -> None:
        """The same word typed two Unicode ways must not become two phrases."""
        precomposed = "caf\u00e9"       # e-acute as a single code point
        decomposed = "cafe\u0301"       # plain e followed by a combining acute
        assert precomposed != decomposed, "test setup: two distinct code sequences"
        assert normalize(precomposed) == normalize(decomposed) == "cafe"

    @pytest.mark.parametrize("raw", ["", " ", "   ", "\t\n", ",.!?", "-- -- --", "***"])
    def test_nothing_worth_keeping_normalises_to_the_empty_string(
        self, raw: str
    ) -> None:
        """Empty, whitespace-only and punctuation-only input all give ""."""
        assert normalize(raw) == "", f"{raw!r} must normalise to the empty string"

    def test_digits_are_kept(self) -> None:
        """Digits are alphanumeric, so "channel 4" keeps its number."""
        assert normalize("Channel 4!") == "channel 4"

    def test_normalize_is_idempotent(self) -> None:
        """Normalising already-normalised text changes nothing."""
        once = normalize("  OK, Google -- what's up?  ")
        assert normalize(once) == once


class TestTokenize:
    """tokenize() is normalize() plus a split, and returns an immutable tuple."""

    def test_returns_a_tuple(self) -> None:
        """The result is a tuple so it can be compared and hashed cheaply."""
        result = tokenize("ok google")
        assert isinstance(result, tuple), f"tokenize must return a tuple, got {type(result).__name__}"
        assert result == ("ok", "google")

    @pytest.mark.parametrize("raw", ["", "   ", "!!!", "\n"])
    def test_empty_input_gives_an_empty_tuple(self, raw: str) -> None:
        """Nothing survivable in means an empty tuple out, never ``("",)``."""
        assert tokenize(raw) == (), f"{raw!r} must tokenize to ()"

    def test_normalisation_is_applied_by_tokenize(self) -> None:
        """Callers never have to call normalize() first."""
        assert tokenize("  OK, Google!  ") == ("ok", "google")

    def test_apostrophised_word_is_one_token(self) -> None:
        """"what's up" is two tokens, not three."""
        assert tokenize("What's up") == ("what's", "up")

    def test_token_count_matches_word_count(self) -> None:
        """Every word of the normalised text becomes exactly one token."""
        text = "ok google play some music now"
        assert len(tokenize(text)) == 6


class TestFindTokens:
    """find_tokens() locates a contiguous run of whole tokens, or reports -1."""

    @pytest.mark.parametrize(
        ("haystack", "needle", "expected"),
        [
            (("a", "b", "c"), ("a",), 0),
            (("a", "b", "c"), ("b",), 1),
            (("a", "b", "c"), ("c",), 2),
            (("a", "b", "c"), ("a", "b"), 0),
            (("a", "b", "c"), ("b", "c"), 1),
            (("a", "b", "c"), ("a", "b", "c"), 0),
            (("x", "ok", "google", "y"), ("ok", "google"), 1),
        ],
    )
    def test_contiguous_run_is_found_at_the_right_index(
        self, haystack: tuple[str, ...], needle: tuple[str, ...], expected: int
    ) -> None:
        """The returned index is where the needle's FIRST token sits."""
        assert find_tokens(haystack, needle) == expected

    def test_first_occurrence_wins_when_the_needle_repeats(self) -> None:
        """A repeated phrase reports the earliest position, not the latest."""
        haystack = ("ok", "google", "ok", "google")
        assert find_tokens(haystack, ("ok", "google")) == 0

    @pytest.mark.parametrize(
        ("haystack", "needle"),
        [
            (("a", "b", "c"), ("d",)),
            (("a", "b", "c"), ("a", "c")),       # present but not contiguous
            (("a", "b", "c"), ("c", "b")),       # present but wrong order
            (("look", "google", "it"), ("ok", "google")),
            ((), ("a",)),
        ],
    )
    def test_absent_needle_returns_minus_one(
        self, haystack: tuple[str, ...], needle: tuple[str, ...]
    ) -> None:
        """A needle that is not a contiguous run reports -1, not a near miss."""
        assert find_tokens(haystack, needle) == -1

    @pytest.mark.parametrize(
        "haystack", [(), ("a",), ("a", "b"), ("ok", "google", "please")]
    )
    def test_empty_needle_never_matches(self, haystack: tuple[str, ...]) -> None:
        """An empty needle returns -1 even in an empty haystack.

        Returning 0 -- "the empty sequence occurs at the start of everything" --
        would be defensible set theory and a disaster here: a phrase that
        normalised away to nothing would then fire on every single utterance.
        """
        assert find_tokens(haystack, ()) == -1, (
            "an empty needle must never match; returning 0 would make an "
            "unusable configured phrase record everything"
        )

    @pytest.mark.parametrize(
        ("haystack", "needle"),
        [
            (("a",), ("a", "b")),
            (("a", "b"), ("a", "b", "c")),
            ((), ("a",)),
            (("ok",), ("ok", "google")),
        ],
    )
    def test_needle_longer_than_haystack_returns_minus_one(
        self, haystack: tuple[str, ...], needle: tuple[str, ...]
    ) -> None:
        """There is no room for the needle, so the answer is -1, not an error."""
        assert find_tokens(haystack, needle) == -1

    def test_partial_prefix_match_does_not_count(self) -> None:
        """A run that starts right but diverges is not a match."""
        assert find_tokens(("ok", "gordon", "google"), ("ok", "google")) == -1

    def test_accepts_lists_as_well_as_tuples(self) -> None:
        """The signature is Sequence[str]; a list must work identically."""
        assert find_tokens(["ok", "google", "now"], ["google", "now"]) == 1


class TestMatchPhrase:
    """match_phrase() picks the best configured phrase occurring in the text."""

    def test_phrase_at_the_start_of_a_longer_utterance_matches(self) -> None:
        """The headline positive case: a wake phrase followed by a command."""
        found = match_phrase("ok google turn it up", ["ok google"])
        assert found is not None
        assert found.phrase == "ok google"
        assert found.token_index == 0
        assert found.text == "ok google turn it up"

    def test_substring_inside_a_longer_word_does_not_match(self) -> None:
        """The headline negative case: "look google it" must NOT fire "ok google".

        ``"ok google"`` really is a substring of ``"look google it"``, so an
        implementation using ``in`` passes every other test in this class and
        fails this one.
        """
        assert match_phrase("look google it", ["ok google"]) is None, (
            "matching must compare whole-token runs, not substrings"
        )

    @pytest.mark.parametrize(
        "text",
        [
            "ok google turn it up",
            "OK, Google! Turn it up.",
            "hey ok google",
            "please ok google now",
            "ok google",
        ],
    )
    def test_texts_that_contain_the_phrase_as_whole_words(self, text: str) -> None:
        """Every whole-word occurrence fires, whatever surrounds or punctuates it."""
        found = match_phrase(text, ["ok google"])
        assert found is not None, f"{text!r} contains the phrase as whole words"
        assert found.phrase == "ok google"

    @pytest.mark.parametrize(
        "text",
        [
            "look google it",
            "okay google",
            "ok googled it",
            "ok the google",
            "google ok",
            "",
            "   ",
            "!!!",
        ],
    )
    def test_texts_that_do_not_contain_the_phrase(self, text: str) -> None:
        """Near misses, reorderings and empty text all return None."""
        assert match_phrase(text, ["ok google"]) is None, (
            f"{text!r} does not contain 'ok google' as a contiguous word run"
        )

    def test_the_match_is_case_and_punctuation_insensitive(self) -> None:
        """The reported phrase is the NORMALISED form, not the configured spelling."""
        found = match_phrase("Well, OK -- Google?", ["OK, Google!"])
        assert found is not None
        assert found.phrase == "ok google", (
            "PhraseMatch.phrase reports the normalised phrase so a caller never "
            "has to normalise it again"
        )
        assert found.text == "well ok google"

    def test_no_phrases_configured_matches_nothing(self) -> None:
        """An empty phrase list is not a wildcard."""
        assert match_phrase("ok google turn it up", []) is None

    def test_longest_phrase_wins(self) -> None:
        """With both configured, the more specific phrase is the one reported."""
        found = _both_orders(
            "please ok google play music now", ["ok google", "ok google play music"]
        )
        assert found is not None
        assert found.phrase == "ok google play music", (
            "reporting the shorter phrase would discard the more specific fact"
        )
        assert found.word_count == 4

    def test_longest_wins_even_when_it_occurs_later(self) -> None:
        """Word count beats position: length is the primary key, not order in text."""
        found = _both_orders(
            "hey google then ok google play music",
            ["hey google", "ok google play music"],
        )
        assert found is not None
        assert found.phrase == "ok google play music"
        assert found.token_index == 3

    def test_ties_break_to_the_earliest_position(self) -> None:
        """Two equally long matches resolve by position, so the result is stable."""
        found = _both_orders(
            "hey google and ok google", ["ok google", "hey google"]
        )
        assert found is not None
        assert found.phrase == "hey google", (
            "'hey google' starts at token 0 and 'ok google' at token 3; the "
            "earlier one wins the tie"
        )
        assert found.token_index == 0

    @pytest.mark.parametrize(
        "phrases",
        [
            ["ok google", "hey google", "ok google play music"],
            ["ok google play music", "ok google", "hey google"],
            ["hey google", "ok google play music", "ok google"],
            ["ok google", "ok google play music", "hey google"],
            ["hey google", "ok google", "ok google play music"],
            ["ok google play music", "hey google", "ok google"],
        ],
    )
    def test_result_is_independent_of_configuration_order(
        self, phrases: list[str]
    ) -> None:
        """Every permutation of the same phrase set gives the identical match."""
        found = match_phrase("hey google then ok google play music now", phrases)
        assert found is not None
        assert (found.phrase, found.token_index) == ("ok google play music", 3), (
            f"permutation {phrases!r} changed the result -- match_phrase must "
            "order candidates by (word count, position), not by iteration order"
        )

    @pytest.mark.parametrize("junk", ["", "   ", "!!!", ",", "-- --", "\t\n"])
    def test_phrases_that_normalise_to_nothing_are_skipped(self, junk: str) -> None:
        """A junk phrase is dropped, never treated as matching everything."""
        assert match_phrase("ok google turn it up", [junk]) is None, (
            f"the unusable phrase {junk!r} must not fire on arbitrary text"
        )

    def test_a_junk_phrase_does_not_shadow_a_real_one(self) -> None:
        """A skipped phrase must not become the "best" match and hide the real one."""
        found = _both_orders("ok google turn it up", ["!!!", "ok google"])
        assert found is not None
        assert found.phrase == "ok google"

    def test_only_junk_phrases_matches_nothing(self) -> None:
        """A configuration of nothing but junk gates nothing."""
        assert match_phrase("ok google turn it up", ["", "   ", "..."]) is None

    def test_empty_text_returns_none_even_with_phrases_configured(self) -> None:
        """Silence decodes to empty text; that must not fire the gate."""
        for text in ("", "   ", "???"):
            assert match_phrase(text, ["ok google"]) is None

    def test_text_is_the_full_normalised_utterance(self) -> None:
        """PhraseMatch.text keeps what was heard, not just what matched."""
        found = match_phrase("Hey!  OK, google -- play MUSIC.", ["ok google"])
        assert found is not None
        assert found.text == "hey ok google play music"
        assert found.phrase == "ok google"

    def test_accepts_a_tuple_of_phrases(self) -> None:
        """VoiceGateConfig stores phrases as a tuple; any iterable must work."""
        found = match_phrase("ok google now", ("hey google", "ok google"))
        assert found is not None
        assert found.phrase == "ok google"

    def test_accented_phrase_matches_unaccented_text(self) -> None:
        """Normalisation is applied to the phrase as well as the text."""
        found = match_phrase("hey cafe now", ["Hey Café"])
        assert found is not None
        assert found.phrase == "hey cafe"
        assert found.token_index == 0

    def test_apostrophised_phrase_matches(self) -> None:
        """"what's up" survives normalisation on both sides and still matches."""
        found = match_phrase("hey what's up there", ["what's up"])
        assert found is not None
        assert found.phrase == "what's up"
        assert found.token_index == 1


class TestPhraseMatch:
    """PhraseMatch is a frozen record with one derived property."""

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("ok", 1),
            ("ok google", 2),
            ("ok google play music", 4),
            ("what's up", 2),
        ],
    )
    def test_word_count_counts_the_phrase_words(
        self, phrase: str, expected: int
    ) -> None:
        """word_count is how many words the matched phrase has."""
        assert PhraseMatch(phrase=phrase, token_index=0, text=phrase).word_count == expected

    def test_word_count_agrees_with_the_configured_phrase(self) -> None:
        """A real match's word_count matches the phrase it was built from."""
        found = match_phrase("say ok google play music", ["ok google play music"])
        assert found is not None
        assert found.word_count == 4

    def test_repr_names_the_phrase_and_position(self) -> None:
        """The repr is for logs, so it must carry the phrase and where it was."""
        text = repr(PhraseMatch(phrase="ok google", token_index=2, text="a b ok google"))
        assert "PhraseMatch" in text
        assert "'ok google'" in text
        assert "token_index=2" in text
