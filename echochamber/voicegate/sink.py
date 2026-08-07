"""The wake-phrase gate itself: a sink that records only when asked to.

:class:`VoiceGateSink` is a :class:`~echochamber.audio.sinks.ChunkSink`, and
that is a design decision rather than an implementation detail.  **The chunker
must not learn about recognition.**  It has one job with a hard deadline --
drain the ring and emit a window every hop -- and folding a decoder into it
would put multi-hundred-millisecond model latency on the thread whose stalling
causes the ring to overrun and lose audio outright.  As a sink behind
:class:`~echochamber.audio.sinks.QueueSink`, the gate instead runs on the
consumer thread, where it is allowed to be slow, and where the bounded queue
absorbs its jitter.  The same shape lets it be tee'd alongside a recorder or a
meter without any of them knowing about the others, and lets the whole gate be
driven in a test by handing it hand-built chunks.

**A snippet spans more than the moment of the match, in both directions.**  A
decoder reports a phrase only after consuming the audio containing it, so the
wake phrase is already in the past when the gate fires; the pre-roll buffer
supplies it (see :mod:`echochamber.voicegate.snippets`).  The post-roll keeps
recording afterwards, because a wake phrase is a *prefix* -- nobody says "ok
google" and stops, and a snippet ending at the match would capture the trigger
and discard the utterance.  Repeating the phrase extends the post-roll of the
snippet already open rather than starting a second file, so one continuous
piece of speech produces one continuous file, and
:attr:`~echochamber.voicegate.config.VoiceGateConfig.max_snippet_ms` bounds how
far that extension can go.

**Only final results are acted on.**  A partial can contain a phrase that the
final decode then revokes; gating on partials writes files for things nobody
said.  See :attr:`~echochamber.voicegate.recognizer.Recognition.final`.

**:meth:`VoiceGateSink.on_chunk` never raises.**  It runs on the consumer
thread, and an exception there kills the consumer loop, which stops the queue
being drained, which takes capture down with it -- a full-stack outage caused
by a failed ``open()`` on a snippet or a decoder that segfaulted its way into a
Python exception.  So recognition and file I/O are wrapped, the failure is
recorded in :attr:`VoiceGateSink.error`, and the gate degrades to not gating:
audio keeps flowing, the counters keep moving, no snippets appear.

**A snippet can have a hole in it, and nothing here can prevent that.**  The
gate sees only the chunks the bounded queue delivers.  Under
:attr:`~echochamber.audio.types.DropPolicy.DROP_OLDEST`, a consumer that falls
behind loses whole chunks before this sink ever sees them: the recogniser is
fed audio with a piece missing, so it may not report a phrase that was actually
spoken, and a snippet that is open across the drop is written with the gap
simply absent -- the file is *shorter* than the wall-clock span it covers, and
the splice is audible.  This is the same honesty
:class:`~echochamber.audio.sinks.WavRecorderSink` demands about its own gaps:
:attr:`VoiceGateSink.gaps` counts the chunks that arrived past the expected
frame, and a non-zero value means the snippets from that session are evidence
of what was heard, not a faithful recording of what was said.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable

from echochamber.audio.sinks import new_frame_count
from echochamber.audio.types import AudioChunk
from echochamber.voicegate.config import ClipMode, VoiceGateConfig
from echochamber.voicegate.matching import PhraseMatch, locate_phrase, match_phrase
from echochamber.voicegate.recognizer import (
    NullRecognizer,
    Recognition,
    Recognizer,
    float32_to_pcm16,
)
from echochamber.voicegate.snippets import (
    PreRollBuffer,
    SnippetWriter,
    snippet_filename,
)

__all__ = ["DetectionEvent", "SnippetEvent", "VoiceGateSink", "VoiceGateStats"]

_BYTES_PER_FRAME: int = 2
"""Bytes per frame on the wire to the recogniser: mono, 16-bit."""


@dataclass(frozen=True, slots=True)
class DetectionEvent:
    """A wake phrase, announced the moment it is recognised.

    **This fires well before the snippet exists.**  A snippet is not closed
    until its post-roll has elapsed -- three seconds, by default -- so anything
    that wants to react promptly to a wake phrase has to be told here rather
    than waiting for :class:`SnippetEvent`.  The two share a :attr:`seq`, which
    is how a listener pairs them.

    A *suppressed* match does not produce one of these.  Suppression exists
    because small models re-report the same phrase across consecutive results,
    so those are duplicates of a detection already announced, and forwarding
    them would announce one utterance several times.

    Attributes:
        phrase: The matched wake phrase, normalised.
        text: The full recognised text the phrase was found in.
        seq: Counter of the snippet this detection belongs to, from 0.  A
            detection that extends an open snippet carries that snippet's
            ``seq``, not a new one.
        start_frame: Absolute frame index of the audio the gate had consumed
            when the phrase was recognised.
        timestamp: UNIX time the detection was announced.
        extended: ``True`` when this detection landed inside an already-open
            snippet and pushed its post-roll out, rather than opening a file.
    """

    phrase: str
    text: str
    seq: int
    start_frame: int
    timestamp: float
    extended: bool = False

    def __repr__(self) -> str:
        """Return a debugging representation naming the phrase and snippet."""
        return (
            f"{type(self).__name__}(phrase={self.phrase!r}, seq={self.seq}, "
            f"extended={self.extended})"
        )


@dataclass(frozen=True, slots=True)
class SnippetEvent:
    """One finished snippet, announced after its file is closed.

    Delivered to the gate's ``on_snippet`` callback **after**
    :meth:`~echochamber.voicegate.snippets.SnippetWriter.close`, never before:
    until the header is finalized the file on disk still declares zero frames,
    so a callback that uploaded or played it would get an empty WAV.

    Attributes:
        path: Path of the written ``.wav`` file.
        phrase: The configured phrase that opened the snippet, normalised.
        text: The full recognised text the phrase was found in, kept so a log
            records what was actually heard and not only what matched.
        seq: Snippet counter for this sink, from 0.  Also embedded in the
            filename, so a log line and a directory listing line up.
        start_frame: Absolute frame index of the snippet's first sample, i.e.
            where the pre-roll began, counting from the stream's start.
        frames: Frames written to the file.
        duration_s: Length of the file in seconds.
        truncated: ``True`` if the snippet hit
            :attr:`~echochamber.voicegate.config.VoiceGateConfig.max_snippet_ms`
            and was cut, rather than closing because its post-roll elapsed.
        timestamp: UNIX time the snippet was closed -- which is ``post_roll_ms``
            or more after the detection that opened it, not the moment the
            phrase was heard.  Pair with :attr:`DetectionEvent.timestamp` on
            ``seq`` to recover both.
    """

    path: str
    phrase: str
    text: str
    seq: int
    start_frame: int
    frames: int
    duration_s: float
    truncated: bool
    timestamp: float = 0.0

    def __repr__(self) -> str:
        """Return a debugging representation naming the file and phrase."""
        return (
            f"{type(self).__name__}(seq={self.seq}, phrase={self.phrase!r}, "
            f"path={self.path!r}, duration_s={self.duration_s:.2f}, "
            f"truncated={self.truncated})"
        )


@dataclass(frozen=True, slots=True)
class VoiceGateStats:
    """A detached copy of a gate's counters.

    Frozen, and returned by :meth:`VoiceGateSink.snapshot` for exactly the
    reason :meth:`echochamber.audio.types.StreamStats.snapshot` returns a copy:
    the GUI thread polls these on a timer while the consumer thread updates
    them, and reading the live fields one at a time would let a display show
    counters from two different instants.

    Attributes:
        frames_processed: New frames the gate has consumed, after overlap
            between windows has been removed.
        phrases_detected: Matches found in final results, including matches
            that were then suppressed or folded into an open snippet.
        snippets_written: Snippet files completed.
        snippets_suppressed: Matches ignored because a snippet had just closed
            and the cooldown had not elapsed.
        snippets_truncated: Snippets cut short by the maximum-length ceiling.
        gaps: Chunks that started past the expected frame, i.e. times audio was
            lost upstream.  Non-zero means the snippets may have holes.
        clips_located: Snippets cut to the phrase using the decoder's word
            timings -- the intended path in
            :attr:`~echochamber.voicegate.config.ClipMode.PHRASE`.
        clips_fallback: Snippets that fell back to the fixed window because no
            usable timings arrived.  **Watch this one.**  A backend that never
            reports timings makes it climb in lockstep with
            ``snippets_written``, and every clip is then a wide guess rather
            than the hotword -- which looks fine until someone listens.
        last_phrase: Most recently matched phrase, empty before the first.
        last_snippet_path: Path of the most recently completed snippet, or
            ``None`` before the first.
        error: Most recent failure the gate swallowed, or ``None``.  A gate
            that is not gating explains itself here.
    """

    frames_processed: int = 0
    phrases_detected: int = 0
    snippets_written: int = 0
    snippets_suppressed: int = 0
    snippets_truncated: int = 0
    gaps: int = 0
    clips_located: int = 0
    clips_fallback: int = 0
    last_phrase: str = ""
    last_snippet_path: str | None = None
    error: str | None = None

    def __repr__(self) -> str:
        """Return a debugging representation of the counters that matter."""
        return (
            f"{type(self).__name__}(frames_processed={self.frames_processed}, "
            f"phrases_detected={self.phrases_detected}, "
            f"snippets_written={self.snippets_written}, "
            f"suppressed={self.snippets_suppressed}, "
            f"truncated={self.snippets_truncated}, gaps={self.gaps})"
        )


class VoiceGateSink:
    """Record a snippet around each recognised wake phrase, and nothing else.

    Runs on the pipeline's consumer thread.  **It must never touch Qt**: the
    GUI polls :meth:`snapshot` on its own timer instead, and ``on_snippet`` is
    invoked on the consumer thread, so a callback that wants to update a widget
    has to hop threads itself.

    Threading: :meth:`on_chunk` is single-producer by construction -- one
    consumer thread owns it -- so the audio state (pre-roll, open writer, frame
    cursor) needs no lock.  The counters do, because the GUI reads them
    concurrently, and a :class:`threading.Lock` guards those and nothing else.
    The lock is never held while the recogniser runs, while a file is written,
    or while ``on_snippet`` is called; those can all block for a long time, and
    holding a lock across them would stall the GUI thread on the audio path.
    """

    __slots__ = (
        "_config",
        "_sample_rate",
        "_recognizer",
        "_on_snippet",
        "_on_detected",
        "_clock",
        "_phrases",
        "_post_roll_frames",
        "_max_snippet_frames",
        "_cooldown_frames",
        "_clip_mode",
        "_lead_frames",
        "_trail_frames",
        "_fed_frames",
        "_anchor_fed",
        "_anchor_abs",
        "_pre_roll",
        "_next_expected",
        "_writer",
        "_next_seq",
        "_snippet_seq",
        "_snippet_phrase",
        "_snippet_text",
        "_snippet_start_frame",
        "_snippet_truncated",
        "_post_roll_left",
        "_frames_since_snippet",
        "_closed",
        "_lock",
        "_frames_processed",
        "_phrases_detected",
        "_snippets_written",
        "_snippets_suppressed",
        "_snippets_truncated",
        "_gaps",
        "_last_phrase",
        "_last_snippet_path",
        "_clips_located",
        "_clips_fallback",
        "_error",
    )

    def __init__(
        self,
        config: VoiceGateConfig,
        sample_rate: int,
        recognizer: Recognizer | None = None,
        on_snippet: Callable[[SnippetEvent], None] | None = None,
        clock: Callable[[], float] | None = None,
        on_detected: Callable[[DetectionEvent], None] | None = None,
    ) -> None:
        """Wire a gate to a recogniser.

        Args:
            config: Gate configuration.  Held as given: it is frozen, so the
                derived frame counts computed here can never go stale, and
                reconfiguring means building a new sink rather than mutating
                this one.
            sample_rate: Capture sample rate in Hz.  Must match the rate the
                recogniser was built for and the rate of the chunks fed in;
                like :class:`~echochamber.audio.sinks.WavRecorderSink`, this is
                not re-checked per chunk because the pipeline guarantees it.
            recognizer: The decoder.  Defaults to
                :class:`~echochamber.voicegate.recognizer.NullRecognizer`, so a
                gate built without a model is inert rather than broken.
            on_snippet: Called with a :class:`SnippetEvent` once each snippet
                file is closed.  Runs on the consumer thread; an exception it
                raises is recorded in :attr:`error` and otherwise ignored.
            clock: Source of UNIX timestamps for snippet filenames, defaulting
                to :func:`time.time`.  Injected so a test can assert on an
                exact filename without freezing the clock.
            on_detected: Called with a :class:`DetectionEvent` the moment a
                phrase is recognised -- before the snippet exists, and not at
                all for a suppressed duplicate.  Runs on the consumer thread, so
                it must not block; an exception it raises is recorded in
                :attr:`error` and otherwise ignored.

        Raises:
            ValueError: If ``sample_rate`` is not positive.
        """
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")

        self._config: VoiceGateConfig = config
        self._sample_rate: int = sample_rate
        self._recognizer: Recognizer = (
            NullRecognizer() if recognizer is None else recognizer
        )
        self._on_snippet: Callable[[SnippetEvent], None] | None = on_snippet
        self._on_detected: Callable[[DetectionEvent], None] | None = on_detected
        self._clock: Callable[[], float] = time.time if clock is None else clock

        # The config is frozen, so normalising the phrases once here is safe
        # and keeps that work off the per-result path.
        self._phrases: tuple[str, ...] = config.normalized_phrases
        self._post_roll_frames: int = config.post_roll_frames(sample_rate)
        self._max_snippet_frames: int = config.max_snippet_frames(sample_rate)
        self._cooldown_frames: int = config.cooldown_frames(sample_rate)
        self._clip_mode: ClipMode = config.clip_mode
        self._lead_frames: int = config.lead_frames(sample_rate)
        self._trail_frames: int = config.trail_frames(sample_rate)
        # Sized from the lookback rather than the pre-roll: locating a phrase
        # means reaching back past the decoder's reporting lag, which is far
        # longer than anything that ends up in the file.
        self._pre_roll: PreRollBuffer = PreRollBuffer(
            config.lookback_frames(sample_rate) * _BYTES_PER_FRAME
        )
        # Vosk's word times count from the first sample it was ever fed, and
        # survive Reset().  The anchor maps that clock onto absolute stream
        # frames, and is re-taken whenever audio is skipped -- after which the
        # two clocks differ by exactly the audio the recogniser never saw.
        self._fed_frames: int = 0
        self._anchor_fed: int = 0
        self._anchor_abs: int | None = None

        self._next_expected: int | None = None
        self._writer: SnippetWriter | None = None
        self._next_seq: int = 0
        self._snippet_seq: int = 0
        self._snippet_phrase: str = ""
        self._snippet_text: str = ""
        self._snippet_start_frame: int = 0
        self._snippet_truncated: bool = False
        self._post_roll_left: int = 0
        # None means "no snippet has closed yet", i.e. no cooldown in force.
        self._frames_since_snippet: int | None = None
        self._closed: bool = False

        self._lock: threading.Lock = threading.Lock()
        self._frames_processed: int = 0
        self._phrases_detected: int = 0
        self._snippets_written: int = 0
        self._snippets_suppressed: int = 0
        self._snippets_truncated: int = 0
        self._gaps: int = 0
        self._last_phrase: str = ""
        self._last_snippet_path: str | None = None
        self._clips_located: int = 0
        self._clips_fallback: int = 0
        self._error: str | None = None

    @property
    def config(self) -> VoiceGateConfig:
        """The configuration this gate was built with."""
        return self._config

    @property
    def sample_rate(self) -> int:
        """Capture sample rate the gate and its snippets assume, in Hz."""
        return self._sample_rate

    @property
    def recognizer(self) -> Recognizer:
        """The decoder in use; a ``NullRecognizer`` when none was supplied."""
        return self._recognizer

    @property
    def frames_processed(self) -> int:
        """New frames consumed, with the overlap between windows removed."""
        with self._lock:
            return self._frames_processed

    @property
    def phrases_detected(self) -> int:
        """Matches found in final results, before suppression is applied."""
        with self._lock:
            return self._phrases_detected

    @property
    def snippets_written(self) -> int:
        """Snippet files completed."""
        with self._lock:
            return self._snippets_written

    @property
    def snippets_suppressed(self) -> int:
        """Matches ignored because the cooldown after a snippet had not run out."""
        with self._lock:
            return self._snippets_suppressed

    @property
    def snippets_truncated(self) -> int:
        """Snippets cut short by the maximum-length ceiling."""
        with self._lock:
            return self._snippets_truncated

    @property
    def gaps(self) -> int:
        """Chunks that began past the expected frame, i.e. lost audio upstream."""
        with self._lock:
            return self._gaps

    @property
    def clips_located(self) -> int:
        """Snippets cut to the phrase using the decoder's word timings."""
        with self._lock:
            return self._clips_located

    @property
    def clips_fallback(self) -> int:
        """Snippets that fell back to a fixed window for want of timings."""
        with self._lock:
            return self._clips_fallback

    @property
    def last_phrase(self) -> str:
        """Most recently matched phrase; empty before the first match."""
        with self._lock:
            return self._last_phrase

    @property
    def last_snippet_path(self) -> str | None:
        """Path of the most recent completed snippet, or ``None``."""
        with self._lock:
            return self._last_snippet_path

    @property
    def error(self) -> str | None:
        """Most recent swallowed failure, or ``None`` if nothing has failed.

        A gate that has stopped producing snippets is either hearing nothing or
        broken; this is how the second case is told from the first without
        letting the failure reach the consumer thread's stack.
        """
        with self._lock:
            return self._error

    @property
    def recording(self) -> bool:
        """``True`` while a snippet file is open."""
        return self._writer is not None

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def snapshot(self) -> VoiceGateStats:
        """Return a detached copy of every counter.

        Taken under one lock acquisition, so the fields agree with each other.
        Reading the properties one by one does not give that guarantee, which
        is why the GUI should call this instead.

        Returns:
            The counters as of this moment.
        """
        with self._lock:
            return VoiceGateStats(
                frames_processed=self._frames_processed,
                phrases_detected=self._phrases_detected,
                snippets_written=self._snippets_written,
                snippets_suppressed=self._snippets_suppressed,
                snippets_truncated=self._snippets_truncated,
                gaps=self._gaps,
                clips_located=self._clips_located,
                clips_fallback=self._clips_fallback,
                last_phrase=self._last_phrase,
                last_snippet_path=self._last_snippet_path,
                error=self._error,
            )

    def on_chunk(self, chunk: AudioChunk) -> None:
        """Feed one window to the gate.

        Never raises: see the module docstring.  Any failure below is recorded
        in :attr:`error`, an in-progress snippet is abandoned, and the gate
        carries on consuming audio without gating.

        Args:
            chunk: Window from the pipeline.  Returns immediately, doing no
                work at all, when the configuration has the gate disabled --
                which is the default, and the reason a checkout with no model
                pays nothing for this sink being wired in.
        """
        if not self._config.enabled or self._closed:
            return
        try:
            self._process(chunk)
        except Exception as exc:  # noqa: BLE001 - must not kill the consumer
            self._fail("processing a chunk", exc)
            self._abandon_snippet()

    def close(self) -> None:
        """Finalize any open snippet and close the recogniser.  Idempotent.

        A snippet still open at shutdown is written out rather than discarded:
        the audio is real, and a half-second file is more useful than no file.
        It is **not** counted in :attr:`snippets_truncated`, which reports the
        length ceiling specifically -- a snippet cut short by the operator
        stopping capture is not the same failure as one that ran too long.
        """
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            try:
                self._close_snippet(truncated=False)
            except Exception as exc:  # noqa: BLE001 - shutdown must complete
                self._fail("finalizing a snippet", exc)
                self._abandon_snippet()
        try:
            self._recognizer.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must complete
            self._fail("closing the recogniser", exc)

    def _process(self, chunk: AudioChunk) -> None:
        """Run the per-chunk algorithm.  Called only from :meth:`on_chunk`.

        Args:
            chunk: Window from the pipeline.
        """
        if self._next_expected is None:
            # The first chunk defines the origin, so all of it is new.
            self._next_expected = chunk.start_frame

        n_new, gap = new_frame_count(
            chunk.start_frame, chunk.n_frames, self._next_expected
        )
        self._next_expected = max(
            self._next_expected, chunk.start_frame + chunk.n_frames
        )
        with self._lock:
            self._frames_processed += n_new
            if gap > 0:
                self._gaps += 1

        if chunk.discontinuous or gap > 0:
            # The decoder's state describes audio that no longer connects to
            # what follows, and the buffered audio would splice across the
            # hole; both are dropped rather than carried over the seam.  The
            # anchor is re-taken because from here the recogniser's clock and
            # the stream differ by exactly the frames it never received.
            self._reset_recognizer()
            self._pre_roll.clear()
            self._anchor_fed = self._fed_frames
            self._anchor_abs = chunk.start_frame

        if n_new <= 0:
            return

        if self._anchor_abs is None:
            # First audio: the recogniser's clock starts here.
            self._anchor_abs = chunk.start_frame + chunk.n_frames - n_new

        tail = chunk.samples[chunk.n_frames - n_new :]
        pcm = float32_to_pcm16(tail)
        # Appended before recognition, so the audio a phrase is found in is
        # already retained by the time the gate goes looking for it.
        self._pre_roll.append(pcm)

        if self._writer is not None:
            self._write_snippet(pcm, n_new)
        elif self._frames_since_snippet is not None:
            self._frames_since_snippet += n_new

        results = self._recognize(pcm)
        self._fed_frames += n_new
        for result in results:
            # Finals only, and empty finals are the normal sound of silence.
            if not result.final or not result.text:
                continue
            found = match_phrase(result.text, self._phrases)
            if found is not None:
                self._on_match(found, result.text, result)

    def _on_match(
        self, found: PhraseMatch, text: str, recognition: Recognition
    ) -> None:
        """Open, extend or suppress a snippet for one matched phrase.

        Args:
            found: The phrase located in ``text``.
            text: The full recognised text, stored on the resulting event.
            recognition: The result the phrase came from, carrying the per-word
                timings that let the snippet be cut to the phrase itself.
        """
        with self._lock:
            self._phrases_detected += 1
            self._last_phrase = found.phrase

        if (
            self._frames_since_snippet is not None
            and self._frames_since_snippet < self._cooldown_frames
        ):
            # Small models routinely re-report a phrase across consecutive
            # results; without this the same utterance writes two files.
            with self._lock:
                self._snippets_suppressed += 1
            return

        if self._writer is not None:
            # Extend rather than open a second file: repeated wake phrases in
            # one breath are one utterance, and splitting them mid-sentence
            # would give two files that each cut the other's audio in half.
            self._post_roll_left = self._post_roll_frames
            self._snippet_text = text
            self._emit_detected(found, text, extended=True)
            return

        self._open_snippet(found, text, recognition)
        # After the snippet is open, so the event carries the seq of the file
        # this detection actually belongs to rather than the previous one's.
        self._emit_detected(found, text, extended=False)

    def _locate_phrase(
        self, found: PhraseMatch, recognition: Recognition
    ) -> tuple[int, int] | None:
        """Map a matched phrase onto absolute stream frames, or give up.

        Vosk reports per-word times counted from the first sample it was ever
        fed (see :class:`~echochamber.voicegate.recognizer.WordTiming`), so a
        word's position in the stream is ``anchor_abs + (word_frame -
        anchor_fed)`` -- the anchor absorbing any audio the recogniser was never
        given.

        **The result is checked before it is trusted.**  A timing that lands in
        the future, before the stream began, or outside the retained audio means
        some assumption here is wrong -- a decoder whose clock does not behave as
        documented, a lookback too short for the reporting lag -- and cutting on
        it would produce a confidently mislabelled snippet of the wrong moment.
        Returning ``None`` costs a wider clip; trusting bad arithmetic costs a
        wrong one.

        Args:
            found: The matched phrase.
            recognition: The result it came from.

        Returns:
            ``(start_frame, end_frame)`` in absolute stream frames, padded by
            ``lead_ms`` and ``trail_ms``, or ``None`` to fall back to a window.
        """
        if self._clip_mode is not ClipMode.PHRASE or self._anchor_abs is None:
            return None
        span = locate_phrase(found, recognition.words)
        if span is None:
            return None

        start_s, end_s = span
        offset = self._anchor_abs - self._anchor_fed
        start = int(round(start_s * self._sample_rate)) + offset
        end = int(round(end_s * self._sample_rate)) + offset
        if end <= start:
            return None

        start -= self._lead_frames
        end += self._trail_frames

        current = self._next_expected or 0
        oldest = current - (self._pre_roll.size // _BYTES_PER_FRAME)
        # The phrase itself must be retained; the trailing pad may still be in
        # the future, and streaming fills that in.
        if start < oldest or start >= current:
            return None
        if end - start > self._max_snippet_frames:
            end = start + self._max_snippet_frames
        return start, end

    def _open_snippet(
        self, found: PhraseMatch, text: str, recognition: Recognition
    ) -> None:
        """Start a new snippet file, cut to the phrase when that is possible.

        Two paths.  With usable word timings the file begins at the phrase --
        ``lead_ms`` before its first word -- and ends ``trail_ms`` after its
        last, so the snippet *is* the hotword.  Without them it falls back to
        the fixed window: everything retained up to ``pre_roll_ms``, then
        ``post_roll_ms`` more as it arrives.

        Either way the audio already buffered is written now and the remainder
        is streamed by :meth:`_write_snippet`, so a phrase whose trailing pad
        has not been captured yet still gets it.

        Args:
            found: The phrase that opened the snippet.
            text: The full recognised text.
            recognition: The result the phrase came from.
        """
        os.makedirs(self._config.snippet_dir, exist_ok=True)
        seq = self._next_seq
        path = os.path.join(
            self._config.snippet_dir,
            snippet_filename(seq, found.phrase, self._clock()),
        )
        writer = SnippetWriter(path, self._sample_rate)

        self._next_seq = seq + 1
        self._writer = writer
        self._snippet_seq = seq
        self._snippet_phrase = found.phrase
        self._snippet_text = text
        self._snippet_truncated = False
        # A snippet is open, so the cooldown clock is not running.
        self._frames_since_snippet = None

        current = self._next_expected or 0
        located = self._locate_phrase(found, recognition)

        if located is not None:
            start_frame, end_frame = located
            available_end = min(end_frame, current)
            byte_start = self._pre_roll.appended - (current - start_frame) * _BYTES_PER_FRAME
            byte_end = self._pre_roll.appended - (current - available_end) * _BYTES_PER_FRAME
            audio = self._pre_roll.extract(byte_start, byte_end)
            if audio is not None:
                with self._lock:
                    self._clips_located += 1
                self._snippet_start_frame = start_frame
                written = writer.write(audio)
                # Whatever of the trailing pad has not been captured yet.
                self._post_roll_left = max(0, end_frame - available_end)
                if self._post_roll_left == 0:
                    self._close_snippet(truncated=False)
                else:
                    self._check_snippet_limits()
                return
            # extract() refused the range, so the phrase is not fully retained
            # after all; fall through rather than write a clipped hotword.

        with self._lock:
            self._clips_fallback += 1
        self._post_roll_left = self._post_roll_frames
        # The window path keeps only the configured pre-roll, not the whole
        # lookback the buffer now holds for locating.
        pre_roll_bytes = (
            self._config.pre_roll_frames(self._sample_rate) * _BYTES_PER_FRAME
        )
        snapshot = self._pre_roll.snapshot()
        if len(snapshot) > pre_roll_bytes:
            snapshot = snapshot[len(snapshot) - pre_roll_bytes :]
        written = writer.write(snapshot)
        self._snippet_start_frame = max(0, current - written)
        self._check_snippet_limits()

    def _write_snippet(self, pcm: bytes, frames: int) -> None:
        """Add one chunk's audio to the open snippet and apply the limits.

        Args:
            pcm: The chunk's new audio, already de-overlapped.
            frames: Number of frames ``pcm`` contains.
        """
        writer = self._writer
        if writer is None:
            return
        allowed = self._max_snippet_frames - writer.frames_written
        if allowed <= 0:
            self._close_snippet(truncated=True)
            return
        if frames > allowed:
            # Trim so the ceiling is exact rather than "within one hop of it";
            # a snippet that overran its configured maximum would make the
            # setting a suggestion.
            self._post_roll_left -= writer.write(pcm[: allowed * _BYTES_PER_FRAME])
            self._close_snippet(truncated=True)
            return
        self._post_roll_left -= writer.write(pcm)
        self._check_snippet_limits()

    def _check_snippet_limits(self) -> None:
        """Close the open snippet if its post-roll elapsed or it hit the ceiling.

        The post-roll is checked first: when both conditions fall on the same
        chunk the snippet ended for the ordinary reason, and reporting it as
        truncated would inflate a counter whose whole job is to say "audio you
        wanted is missing from this file".
        """
        writer = self._writer
        if writer is None:
            return
        if self._post_roll_left <= 0:
            self._close_snippet(truncated=False)
        elif writer.frames_written >= self._max_snippet_frames:
            self._close_snippet(truncated=True)

    def _close_snippet(self, truncated: bool) -> None:
        """Finalize the open snippet and announce it.

        Args:
            truncated: Whether the snippet was cut by the length ceiling.
        """
        writer = self._writer
        if writer is None:
            return
        self._writer = None
        self._post_roll_left = 0
        self._snippet_truncated = truncated
        # The cooldown measures time since a snippet *closed*, so it starts
        # here rather than when the match landed.
        self._frames_since_snippet = 0

        writer.close()
        path = os.fspath(writer.path)
        with self._lock:
            self._snippets_written += 1
            if truncated:
                self._snippets_truncated += 1
            self._last_snippet_path = path

        self._emit(
            SnippetEvent(
                path=path,
                phrase=self._snippet_phrase,
                text=self._snippet_text,
                seq=self._snippet_seq,
                start_frame=self._snippet_start_frame,
                frames=writer.frames_written,
                duration_s=writer.duration_s,
                truncated=truncated,
                timestamp=self._clock(),
            )
        )

    def _abandon_snippet(self) -> None:
        """Drop the open snippet after a failure, without announcing it.

        No :class:`SnippetEvent` is emitted: the callback's contract is that it
        receives a complete file, and after a write failed this one is not.
        The cooldown starts anyway, so a gate failing repeatedly does not
        thrash the filesystem once per chunk.
        """
        writer = self._writer
        if writer is None:
            return
        self._writer = None
        self._post_roll_left = 0
        self._frames_since_snippet = 0
        try:
            writer.close()
        except Exception as exc:  # noqa: BLE001 - already handling a failure
            self._fail("closing an abandoned snippet", exc)

    def _recognize(self, pcm: bytes) -> list[Recognition]:
        """Feed the recogniser, treating any failure as "heard nothing".

        Args:
            pcm: 16-bit little-endian mono bytes.

        Returns:
            The recogniser's results, or an empty list if it raised.  A decoder
            fault degrades the gate to not gating rather than propagating out
            of :meth:`on_chunk`.
        """
        try:
            return self._recognizer.accept_pcm(pcm)
        except Exception as exc:  # noqa: BLE001 - a decoder fault must not kill
            self._fail("recognition", exc)
            return []

    def _reset_recognizer(self) -> None:
        """Discard decoder state, swallowing any failure."""
        try:
            self._recognizer.reset()
        except Exception as exc:  # noqa: BLE001 - a decoder fault must not kill
            self._fail("resetting the recogniser", exc)

    def _emit_detected(
        self, found: PhraseMatch, text: str, extended: bool
    ) -> None:
        """Announce a detection, swallowing anything the callback raises.

        Args:
            found: The phrase that matched.
            text: The full recognised text.
            extended: Whether this detection extended an open snippet rather
                than opening one.
        """
        callback = self._on_detected
        if callback is None:
            return
        try:
            callback(
                DetectionEvent(
                    phrase=found.phrase,
                    text=text,
                    seq=self._snippet_seq,
                    start_frame=self._next_expected or 0,
                    timestamp=self._clock(),
                    extended=extended,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a callback must not kill
            self._fail("the detection callback", exc)

    def _emit(self, event: SnippetEvent) -> None:
        """Hand ``event`` to the callback, swallowing anything it raises.

        Args:
            event: The completed snippet.  The callback runs on the consumer
                thread, so a badly behaved one is a latency problem as well as
                a correctness one -- but never an outage.
        """
        callback = self._on_snippet
        if callback is None:
            return
        try:
            callback(event)
        except Exception as exc:  # noqa: BLE001 - a callback must not kill
            self._fail("the snippet callback", exc)

    def _fail(self, context: str, exc: BaseException) -> None:
        """Record a swallowed failure in :attr:`error`.

        Args:
            context: What was being attempted, phrased to read after "failed".
            exc: The exception that was caught.
        """
        with self._lock:
            self._error = f"{context} failed: {type(exc).__name__}: {exc}"

    def __repr__(self) -> str:
        """Return a debugging representation of the gate's state."""
        return (
            f"{type(self).__name__}(enabled={self._config.enabled}, "
            f"sample_rate={self._sample_rate}, "
            f"phrases={len(self._phrases)}, recording={self.recording}, "
            f"detected={self.phrases_detected}, "
            f"written={self.snippets_written}, closed={self._closed})"
        )
