"""The speech-recognition seam, and the two implementations that need no model.

:class:`Recognizer` is a :class:`~typing.Protocol` for exactly the same reason
:class:`~echochamber.audio.sinks.ChunkSink` is one: it lets the real Vosk
decoder, the subprocess bridge and a scripted test double be interchangeable
without any of them knowing the others exist.

**Vosk is imported lazily, inside :func:`load_vosk_recognizer`.**  A top-level
``import vosk`` would make this module -- and therefore the GUI, and therefore
the whole test suite -- unimportable on a machine that has not installed it.
Since the deployment target may never be able to install it in-process at all
(see :mod:`echochamber.voicegate.subprocess_recognizer`), that would be a
hard dependency on something explicitly optional.

**PCM, not floats, crosses this boundary.**  Vosk's ``AcceptWaveform`` wants
16-bit little-endian mono bytes, and the subprocess backend has to serialise
whatever it is given anyway.  Fixing the wire format at ``bytes`` here means
the conversion happens once, in :func:`float32_to_pcm16`, rather than once per
backend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "NullRecognizer",
    "Recognition",
    "Recognizer",
    "ScriptedRecognizer",
    "WordTiming",
    "float32_to_pcm16",
    "load_vosk_recognizer",
    "parse_vosk_result",
    "parse_word_timings",
]


@dataclass(frozen=True, slots=True)
class WordTiming:
    """Where one recognised word sits in the audio.

    **The times are cumulative over everything fed to the recogniser**, not
    relative to the current utterance.  Vosk accumulates ``frame_offset_`` in
    its ``CleanUp()`` and never clears it in ``Reset()``, so a word two minutes
    into a stream reports ``start`` near 120.0.  That is what makes it possible
    to locate a wake phrase in the ring of buffered audio rather than guessing
    at a fixed offset -- see
    :meth:`echochamber.voicegate.sink.VoiceGateSink._locate_phrase`.

    Attributes:
        word: The word, as the decoder spelled it.
        start: Seconds from the first sample fed to the recogniser.
        end: Seconds from the first sample fed to the recogniser.
        conf: Model confidence for this word, ``0.0``-``1.0``.
    """

    word: str
    start: float
    end: float
    conf: float = 0.0

    @property
    def duration_s(self) -> float:
        """Length of this word in seconds, never negative."""
        return max(0.0, self.end - self.start)

    def __repr__(self) -> str:
        """Return a debugging representation naming the word and its span."""
        return (
            f"{type(self).__name__}({self.word!r}, "
            f"{self.start:.3f}-{self.end:.3f}s)"
        )


@dataclass(frozen=True, slots=True)
class Recognition:
    """One piece of text the recogniser has produced.

    Attributes:
        text: The recognised words.  Possibly empty -- small models emit empty
            finals constantly during silence, and callers are expected to
            ignore those rather than treat them as an error.
        final: ``True`` for a settled result, ``False`` for a partial that may
            still change.  **The gate only ever acts on finals**: a partial can
            contain a phrase that the final decode then revokes, which would
            open a snippet for something nobody said.
        confidence: Model confidence in ``0.0``-``1.0`` where the backend
            reports one, ``0.0`` otherwise.  Informational; the gate does not
            threshold on it, because the small model's confidences are not
            calibrated well enough to be worth a knob.
        words: Per-word timings when the backend reports them, in the order
            they were spoken.  **This is what lets a snippet be cut to the wake
            phrase itself** rather than to a fixed window around whenever the
            decoder happened to finish; a backend that reports nothing here
            makes the gate fall back to that window.  Empty for partials.
    """

    text: str
    final: bool
    confidence: float = 0.0
    words: tuple[WordTiming, ...] = ()

    @property
    def span(self) -> tuple[float, float] | None:
        """The seconds spanned by every word, or ``None`` without timings.

        Returns:
            ``(start, end)`` from the first and last word.
        """
        if not self.words:
            return None
        return self.words[0].start, self.words[-1].end

    def __repr__(self) -> str:
        """Return a debugging representation of the recognised text."""
        kind = "final" if self.final else "partial"
        return f"{type(self).__name__}({kind}, text={self.text!r})"


@runtime_checkable
class Recognizer(Protocol):
    """Anything that turns PCM into :class:`Recognition` results.

    Implementations must tolerate :meth:`close` being called more than once,
    because the sink that owns one is itself closed idempotently.
    """

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        """Feed 16-bit little-endian mono PCM and return whatever settled.

        Args:
            pcm: Raw sample bytes at the recogniser's configured rate.

        Returns:
            Zero or more results.  Returning an empty list is the normal case:
            most buffers do not complete an utterance.
        """
        ...

    def reset(self) -> None:
        """Discard decoder state, e.g. after a discontinuity in the audio."""
        ...

    def close(self) -> None:
        """Release whatever the recogniser holds.  Must be idempotent."""
        ...


class NullRecognizer:
    """A recogniser that never recognises anything.

    The default, and the reason a checkout with no Vosk and no model still
    runs: the gate is wired in, counts audio, and simply never fires.  Also
    what the sink falls back to if a backend fails to start, so a broken
    recogniser degrades to "no snippets" rather than to a dead pipeline.
    """

    __slots__ = ("_closed",)

    def __init__(self) -> None:
        """Create the recogniser.  Nothing is allocated."""
        self._closed: bool = False

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        """Discard ``pcm`` and return no results.

        Args:
            pcm: Ignored.

        Returns:
            An empty list, always.
        """
        return []

    def reset(self) -> None:
        """No-op; there is no state to discard."""

    def close(self) -> None:
        """Mark the recogniser closed.  Idempotent."""
        self._closed = True

    def __repr__(self) -> str:
        """Return a debugging representation of this recogniser."""
        return f"{type(self).__name__}(closed={self._closed})"


class ScriptedRecognizer:
    """Emit predetermined results after a set number of bytes.

    Ships in the package rather than in the tests because the GUI needs a way
    to demonstrate the gate end to end without a model, and because the
    subprocess worker is easier to smoke-test against something deterministic.

    Args:
        script: ``(byte_offset, Recognition)`` pairs.  Each result is emitted
            by the first :meth:`accept_pcm` call that brings the running total
            of bytes consumed to at least ``byte_offset``.  Order is
            irrelevant; the script is sorted on construction.
    """

    __slots__ = ("_script", "_consumed", "_next", "_closed", "_resets")

    def __init__(self, script: list[tuple[int, Recognition]] | None = None) -> None:
        """Prepare the scripted results.

        Args:
            script: The ``(byte_offset, Recognition)`` pairs to emit, or
                ``None`` for a recogniser that behaves like
                :class:`NullRecognizer`.
        """
        self._script: list[tuple[int, Recognition]] = sorted(
            script or [], key=lambda item: item[0]
        )
        self._consumed: int = 0
        self._next: int = 0
        self._closed: bool = False
        self._resets: int = 0

    @property
    def consumed(self) -> int:
        """Total PCM bytes fed to this recogniser."""
        return self._consumed

    @property
    def resets(self) -> int:
        """How many times :meth:`reset` has been called."""
        return self._resets

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        """Consume ``pcm`` and emit every scripted result now due.

        Args:
            pcm: Raw sample bytes; only its length matters.

        Returns:
            The scripted results whose offsets the running total has reached.
        """
        self._consumed += len(pcm)
        due: list[Recognition] = []
        while self._next < len(self._script):
            offset, recognition = self._script[self._next]
            if offset > self._consumed:
                break
            due.append(recognition)
            self._next += 1
        return due

    def reset(self) -> None:
        """Count the reset; the script and byte total are left alone.

        Deliberately not rewinding: a test that scripts a result at a byte
        offset wants that offset to mean "after this much audio", regardless of
        a discontinuity the pipeline happened to report in between.
        """
        self._resets += 1

    def close(self) -> None:
        """Mark the recogniser closed.  Idempotent."""
        self._closed = True

    def __repr__(self) -> str:
        """Return a debugging representation of the script's progress."""
        return (
            f"{type(self).__name__}(consumed={self._consumed}, "
            f"emitted={self._next}/{len(self._script)}, closed={self._closed})"
        )


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """Convert mono float samples in ``[-1, 1]`` to 16-bit little-endian PCM.

    Args:
        samples: 1-D float array.  Values outside ``[-1, 1]`` are **clipped**,
            not wrapped: wrapping turns a loud passage into a burst of noise
            that the decoder has no chance with.

    Returns:
        The little-endian ``int16`` bytes Vosk's ``AcceptWaveform`` expects.
        Little-endian is forced rather than native so the bytes mean the same
        thing after crossing a pipe to another interpreter.
    """
    data = np.asarray(samples)
    if data.size == 0:
        return b""
    scaled = np.clip(data, -1.0, 1.0) * 32767.0
    return scaled.astype("<i2").tobytes()


def parse_vosk_result(payload: str, final: bool) -> Recognition:
    """Turn one Vosk JSON result into a :class:`Recognition`.

    Vosk returns ``{"text": "..."}`` for finals and ``{"partial": "..."}`` for
    partials, and returns ``{}`` rather than an error when it has nothing to
    say.  Malformed JSON is treated as "nothing recognised" rather than raised:
    this runs on the pipeline's consumer thread, where an exception would kill
    the consumer and take the whole capture down over a decode hiccup.

    Args:
        payload: The JSON string from ``Result()`` or ``PartialResult()``.
        final: Whether ``payload`` came from ``Result()``.

    Returns:
        The parsed recognition, with empty text when nothing could be read.
    """
    try:
        parsed: Any = json.loads(payload)
    except (TypeError, ValueError):
        return Recognition(text="", final=final)
    if not isinstance(parsed, dict):
        return Recognition(text="", final=final)

    text = parsed.get("text" if final else "partial", "")
    if not isinstance(text, str):
        text = ""

    raw_words = parsed.get("result")
    timings = parse_word_timings(raw_words)
    # Confidence is averaged over the words that *reported* one, deliberately
    # independent of whether their timings parsed.  Deriving it from `timings`
    # instead would make a decoder that omits timestamps report every utterance
    # at zero confidence -- a quality signal silently turned into a packaging
    # signal.
    confidence = 0.0
    if isinstance(raw_words, list):
        scores = [
            value
            for entry in raw_words
            if isinstance(entry, dict)
            for value in (_as_float(entry.get("conf")),)
            if value is not None
        ]
        if scores:
            confidence = sum(scores) / len(scores)

    return Recognition(
        text=text, final=final, confidence=confidence, words=timings
    )


def parse_word_timings(raw: object) -> tuple[WordTiming, ...]:
    """Build :class:`WordTiming` values from Vosk's ``result`` array.

    Every field is coerced rather than trusted, and a word that cannot be read
    is **skipped rather than defaulted**.  A word with a fabricated ``start`` of
    ``0.0`` would place the wake phrase at the beginning of the stream, and the
    gate would cut a snippet from entirely the wrong audio -- a silent, wrong
    answer, which is worse than the missing-timings fallback.

    Args:
        raw: The ``result`` value from a Vosk JSON payload, if there was one.

    Returns:
        The timings, in the order given.  Empty when ``raw`` is not a list, is
        empty, or contains nothing usable.
    """
    if not isinstance(raw, list):
        return ()

    timings: list[WordTiming] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        word = entry.get("word")
        start = _as_float(entry.get("start"))
        end = _as_float(entry.get("end"))
        if not isinstance(word, str) or start is None or end is None:
            continue
        conf = _as_float(entry.get("conf"))
        timings.append(
            WordTiming(
                word=word,
                start=start,
                end=max(start, end),
                conf=0.0 if conf is None else conf,
            )
        )
    return tuple(timings)


def _as_float(value: object) -> float | None:
    """Coerce a JSON number to a float, rejecting booleans.

    Args:
        value: The raw value.

    Returns:
        The number, or ``None`` if it was not one.  ``bool`` is a subclass of
        ``int``, so a JSON ``true`` would otherwise pass as ``1.0`` -- a real
        number from a field that carried none.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def load_vosk_recognizer(
    model_path: str,
    sample_rate: int,
    phrases: tuple[str, ...] = (),
) -> "VoskRecognizer":
    """Load a Vosk model and wrap it as a :class:`Recognizer`.

    Imports :mod:`vosk` here rather than at module scope; see the module
    docstring.  Also silences Vosk's own logging, which otherwise writes Kaldi
    diagnostics straight to stderr from a background thread.

    Args:
        model_path: Directory of an unpacked Vosk model.
        sample_rate: Capture sample rate in Hz; must match the audio fed in.
        phrases: When non-empty, constrain the decoder to these phrases plus
            ``[unk]``.  For a fixed wake-phrase set this is markedly more
            accurate than open vocabulary, and cheaper -- the decoder cannot
            propose words that are not in the grammar, so ``"ok google"`` stops
            competing with every acoustically similar phrase in English.

    Returns:
        A ready :class:`VoskRecognizer`.

    Raises:
        ImportError: If the ``vosk`` package is not installed, re-raised with a
            message naming the optional extra that provides it.
        FileNotFoundError: If ``model_path`` does not exist.
    """
    import os

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Vosk model directory not found: {model_path!r}. Download the "
            f"small English model with `python scripts/setup_voice_gate.py`."
        )

    try:
        import vosk  # noqa: PLC0415 - deliberately lazy; see module docstring
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the `vosk` package is required for in-process recognition; "
            "install it with `pip install .[voice-gate]`, or point "
            "VoiceGateConfig.worker_python at an interpreter that has it"
        ) from exc

    vosk.SetLogLevel(-1)
    model = vosk.Model(model_path)
    if phrases:
        grammar = json.dumps(list(phrases) + ["[unk]"])
        recognizer = vosk.KaldiRecognizer(model, sample_rate, grammar)
    else:
        recognizer = vosk.KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)
    return VoskRecognizer(model, recognizer)


class VoskRecognizer:
    """In-process Vosk decoder.

    Built by :func:`load_vosk_recognizer`, never directly -- the constructor
    takes already-loaded Vosk objects precisely so this class itself carries no
    import of :mod:`vosk` and can be type-checked and reasoned about without
    it.

    This is what runs **inside** the subprocess worker; on a platform where
    Vosk installs natively it is also usable directly from the main process.
    """

    __slots__ = ("_model", "_recognizer", "_closed")

    def __init__(self, model: Any, recognizer: Any) -> None:
        """Wrap a loaded model and recogniser.

        Args:
            model: The ``vosk.Model``.  Held only to keep it alive: the
                recogniser does not own it, and letting it be collected
                crashes the decoder.
            recognizer: The ``vosk.KaldiRecognizer``.
        """
        self._model: Any = model
        self._recognizer: Any = recognizer
        self._closed: bool = False

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def accept_pcm(self, pcm: bytes) -> list[Recognition]:
        """Feed PCM to the decoder and return what it settled on.

        A ``True`` from ``AcceptWaveform`` means an utterance ended, so
        ``Result()`` is read; otherwise the current ``PartialResult()`` is
        returned.  Partials are reported rather than discarded so a caller can
        display them, but the gate itself acts only on finals -- see
        :attr:`Recognition.final`.

        Args:
            pcm: 16-bit little-endian mono bytes at the configured rate.

        Returns:
            One result, or none for an empty buffer or a closed recogniser.
        """
        if self._closed or not pcm:
            return []
        if self._recognizer.AcceptWaveform(pcm):
            return [parse_vosk_result(self._recognizer.Result(), final=True)]
        return [parse_vosk_result(self._recognizer.PartialResult(), final=False)]

    def final(self) -> Recognition:
        """Flush the decoder and return its last result.

        Returns:
            The final recognition, empty when the decoder is closed.
        """
        if self._closed:
            return Recognition(text="", final=True)
        return parse_vosk_result(self._recognizer.FinalResult(), final=True)

    def reset(self) -> None:
        """Discard decoder state so the next audio starts a fresh utterance."""
        if not self._closed:
            self._recognizer.Reset()

    def close(self) -> None:
        """Drop the decoder and the model.  Idempotent."""
        self._closed = True
        self._recognizer = None
        self._model = None

    def __repr__(self) -> str:
        """Return a debugging representation of this recogniser."""
        return f"{type(self).__name__}(closed={self._closed})"
