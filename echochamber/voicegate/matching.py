"""Deciding whether recognised text contains a wake phrase.

Pure functions over strings, deliberately holding **no** reference to Vosk, to
a model, or to audio.  This is where the gate's actual policy lives, so it is
also the part that has to be exhaustively testable without a recogniser
present -- feed it text, assert what matched.

Three decisions are baked in here, and each of them is a judgement call rather
than an obvious truth:

**Match on a contiguous run of whole words, not a substring.**  ``"ok google"``
must fire on ``"ok google turn the volume up"`` but must *not* fire on
``"look google it"``.  A plain ``in`` test gets the second case wrong, because
``"ok google"`` really is a substring of ``"look google it"``.  Comparing
token runs makes word boundaries explicit and costs nothing at these lengths.

**Normalisation is aggressive and lossy.**  Case, punctuation and repeated
whitespace are all discarded before comparison.  Small ASR models punctuate
inconsistently -- the same utterance comes back as ``"OK, Google"`` or
``"ok google"`` depending on the decode -- so keeping any of it would make
matching depend on something the model does not promise.

**The longest phrase wins, then the earliest.**  With both ``"ok google"`` and
``"ok google play music"`` configured, text containing the latter matches the
latter: reporting the shorter one would throw away the more specific fact for
no reason.  Ties break on position so the result is deterministic regardless
of the order phrases were configured in, which is what makes the tests stable.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "PhraseMatch",
    "locate_phrase",
    "match_phrase",
    "normalize",
    "tokenize",
]


@dataclass(frozen=True, slots=True)
class PhraseMatch:
    """One configured phrase found inside a piece of recognised text.

    Attributes:
        phrase: The configured phrase that matched, in its **normalised** form
            -- what :func:`normalize` returned, not the raw configured string.
        token_index: Index of the phrase's first word within the normalised
            token sequence of the text, counting from 0.
        text: The full normalised text the match was found in, kept so a caller
            can log what was actually heard rather than only what matched.
    """

    phrase: str
    token_index: int
    text: str

    @property
    def word_count(self) -> int:
        """Number of words in the matched phrase."""
        return len(self.phrase.split())

    def __repr__(self) -> str:
        """Return a debugging representation naming the phrase and position."""
        return (
            f"{type(self).__name__}(phrase={self.phrase!r}, "
            f"token_index={self.token_index}, text={self.text!r})"
        )


def normalize(text: str) -> str:
    """Fold ``text`` to lowercase words separated by single spaces.

    Unicode is NFKD-normalised and combining marks are dropped, so ``"Café"``
    and ``"Cafe"`` compare equal; every character that is not a letter, a digit
    or an apostrophe becomes a space.  Apostrophes survive because ASR output
    contains genuine contractions (``"what's"``), and splitting those into two
    tokens would make a configured ``"what's up"`` unmatchable.

    Args:
        text: Raw recognised text, or a raw configured phrase.

    Returns:
        The normalised form: lowercase, no punctuation beyond apostrophes, no
        leading, trailing or repeated whitespace.  Possibly the empty string.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    kept: list[str] = []
    for char in decomposed:
        if unicodedata.combining(char):
            continue
        if char.isalnum() or char == "'":
            kept.append(char.lower())
        else:
            kept.append(" ")
    return " ".join("".join(kept).split())


def tokenize(text: str) -> tuple[str, ...]:
    """Split ``text`` into normalised words.

    Args:
        text: Raw text; normalisation is applied here, so callers do not need
            to call :func:`normalize` first.

    Returns:
        The words of the normalised text, empty when nothing survived.
    """
    normalized = normalize(text)
    if not normalized:
        return ()
    return tuple(normalized.split())


def find_tokens(haystack: Sequence[str], needle: Sequence[str]) -> int:
    """Return the index where ``needle`` occurs contiguously in ``haystack``.

    Args:
        haystack: Token sequence to search, already normalised.
        needle: Token sequence to find, already normalised.

    Returns:
        The index of the first token of the first occurrence, or ``-1`` when
        there is none.  An empty ``needle`` never matches: a phrase that
        normalised away to nothing must not silently fire on every utterance,
        which is what returning 0 here would do.
    """
    n_needle = len(needle)
    if n_needle == 0:
        return -1
    n_haystack = len(haystack)
    if n_needle > n_haystack:
        return -1
    first = needle[0]
    for start in range(n_haystack - n_needle + 1):
        if haystack[start] != first:
            continue
        if tuple(haystack[start : start + n_needle]) == tuple(needle):
            return start
    return -1


def locate_phrase(
    match: PhraseMatch, words: Sequence[Any]
) -> tuple[float, float] | None:
    """Return the seconds spanned by the matched phrase, if they can be trusted.

    :attr:`PhraseMatch.token_index` indexes the *normalised tokens* of the
    recognised text, while ``words`` comes from the decoder.  Those two line up
    only if normalisation did not change the word count -- which it does not for
    the small English model, whose output is already lowercase and unpunctuated.
    But "does not, in practice, for this model" is not a guarantee, and a
    mis-indexed lookup would silently cut a snippet from the wrong audio.

    So the alignment is **checked rather than assumed**: the word the timing
    claims must be the token that matched.  A mismatch returns ``None``, which
    the caller reads as "locate this the old way".

    Args:
        match: The phrase found by :func:`match_phrase`.
        words: Per-word timings from the recogniser, each exposing ``word``,
            ``start`` and ``end``.  Typed loosely to keep this module free of an
            import from :mod:`echochamber.voicegate.recognizer`, which would be
            circular.

    Returns:
        ``(start_s, end_s)`` covering every word of the phrase, or ``None`` when
        there are no timings, the index does not fit, or the words at that index
        are not the phrase.
    """
    needle = match.phrase.split()
    if not needle or not words:
        return None

    start_index = match.token_index
    end_index = start_index + len(needle)
    if start_index < 0 or end_index > len(words):
        return None

    span = list(words[start_index:end_index])
    for timing, expected in zip(span, needle):
        # Normalise the decoder's word the same way the haystack was, so the
        # comparison is like for like rather than raw-against-folded.
        if normalize(getattr(timing, "word", "")) != expected:
            return None

    start = float(getattr(span[0], "start", 0.0))
    end = float(getattr(span[-1], "end", 0.0))
    if end < start:
        return None
    return start, end


def match_phrase(text: str, phrases: Iterable[str]) -> PhraseMatch | None:
    """Find the best configured phrase occurring in ``text``.

    Args:
        text: Recognised text from the ASR backend.
        phrases: Configured wake phrases, in any order and in any case.
            Entries that normalise to nothing are skipped rather than treated
            as a wildcard.

    Returns:
        The matching :class:`PhraseMatch`, or ``None`` if no phrase occurs.
        When several match, the one with the most words wins; ties go to the
        one occurring earliest in ``text``, so the result never depends on the
        iteration order of ``phrases``.
    """
    haystack = tokenize(text)
    if not haystack:
        return None

    normalized_text = " ".join(haystack)
    best: PhraseMatch | None = None
    for phrase in phrases:
        needle = tokenize(phrase)
        index = find_tokens(haystack, needle)
        if index < 0:
            continue
        candidate = PhraseMatch(
            phrase=" ".join(needle), token_index=index, text=normalized_text
        )
        if best is None:
            best = candidate
        elif candidate.word_count > best.word_count:
            best = candidate
        elif (
            candidate.word_count == best.word_count
            and candidate.token_index < best.token_index
        ):
            best = candidate
    return best
