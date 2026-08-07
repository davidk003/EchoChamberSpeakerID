"""Keeping the audio the gate will want, and writing it out when it fires.

Three pieces, none of which knows anything about recognition: a bounded buffer
of recent PCM, a one-file WAV writer, and a pure naming function.  The gate
itself lives in :mod:`echochamber.voicegate.sink`; splitting the storage out
means the buffering arithmetic and the naming rules are testable without a
recogniser, a model, or a running pipeline -- the same reason
:func:`~echochamber.audio.sinks.new_frame_count` is not a method on
:class:`~echochamber.audio.sinks.WavRecorderSink`.

**A pre-roll is not a nicety; it corrects for the recogniser's latency.**  A
decoder only reports a phrase *after* it has consumed the audio containing it,
so by the moment the gate fires, the wake phrase is already in the past.  A
snippet that began at the instant of the match would start after the phrase it
is named for -- exactly the audio a listener opens the file to hear.
:class:`PreRollBuffer` keeps the last few seconds so the file can begin before
the match instead.

**The pre-roll is a deque of chunks plus a running total, never a growing
``bytes``.**  ``buf = buf + pcm`` copies the whole buffer on every append, so
holding 1.5 s at 16 kHz and appending 1000 ms at a time is quadratic in the
buffer length -- work that lands on the pipeline's consumer thread, the one
thread that must not fall behind.  Appending an immutable chunk to a deque and
dropping whole chunks off the front is O(1) per append; only
:meth:`PreRollBuffer.snapshot` pays for a join, and that happens once per
snippet rather than once per chunk.

**:func:`snippet_filename` takes the wall-clock time as an argument and never
reads the clock.**  This repository does that consistently -- see
:meth:`echochamber.ui.meters.PeakHold.update` -- because a function that reads
its own clock can only be tested by freezing time, while one that is handed a
timestamp can simply be called with 1234567890.0 and compared to a string.
"""

from __future__ import annotations

import collections
import os
import time
import unicodedata
import wave

__all__ = ["PreRollBuffer", "SnippetWriter", "snippet_filename"]

_BYTES_PER_FRAME: int = 2
"""Bytes per frame of the format this module writes: mono, 16-bit."""

_MAX_PHRASE_CHARS: int = 40
"""Longest phrase slug allowed in a filename.

A ceiling exists because the phrase is operator-supplied text of unbounded
length, and Windows still enforces a 260-character path limit by default; a
long phrase plus a long snippet directory is a write that fails at the worst
possible moment.  Forty characters comfortably holds any plausible wake phrase.
"""


class PreRollBuffer:
    """A bounded FIFO of the most recent PCM bytes.

    Holds at most :attr:`capacity_bytes`, discarding from the front as newer
    audio arrives, so the buffer always describes the audio immediately
    preceding the last :meth:`append`.  Nothing here interprets the bytes: the
    caller decides what a frame is, and passes a capacity already converted
    from a duration.

    Not thread-safe.  The gate owns one and touches it only from the pipeline's
    consumer thread; adding a lock here would pay for synchronisation on every
    chunk to protect against a caller that does not exist.
    """

    __slots__ = ("_capacity_bytes", "_chunks", "_size")

    def __init__(self, capacity_bytes: int) -> None:
        """Create an empty buffer holding at most ``capacity_bytes``.

        Args:
            capacity_bytes: Maximum bytes retained.  ``0`` is legal and makes
                this a black hole -- which is what a configuration with no
                pre-roll asks for, so it must not be an error.

        Raises:
            ValueError: If ``capacity_bytes`` is negative.
        """
        capacity_bytes = int(capacity_bytes)
        if capacity_bytes < 0:
            raise ValueError(
                f"capacity_bytes must be >= 0, got {capacity_bytes}"
            )
        self._capacity_bytes: int = capacity_bytes
        self._chunks: collections.deque[bytes] = collections.deque()
        self._size: int = 0

    @property
    def capacity_bytes(self) -> int:
        """Maximum bytes this buffer retains."""
        return self._capacity_bytes

    @property
    def size(self) -> int:
        """Bytes currently buffered; never above :attr:`capacity_bytes`."""
        return self._size

    def append(self, pcm: bytes) -> None:
        """Add ``pcm`` to the back, evicting from the front to stay in bounds.

        An append **larger than the capacity** is not an error and is not
        rejected: only its trailing :attr:`capacity_bytes` are kept, which is
        the same "keep the newest audio" rule every other append follows.  This
        case is reached in practice whenever the pre-roll is configured shorter
        than one hop.

        Args:
            pcm: Raw sample bytes to append.  Empty is a no-op.
        """
        if self._capacity_bytes == 0 or not pcm:
            return

        if len(pcm) >= self._capacity_bytes:
            # Everything already held is older than what this append alone
            # would evict, so there is nothing to trim chunk by chunk.
            self._chunks.clear()
            tail = pcm[len(pcm) - self._capacity_bytes :]
            self._chunks.append(tail)
            self._size = len(tail)
            return

        self._chunks.append(pcm)
        self._size += len(pcm)
        excess = self._size - self._capacity_bytes
        while excess > 0:
            head = self._chunks[0]
            if len(head) <= excess:
                self._chunks.popleft()
                self._size -= len(head)
                excess -= len(head)
            else:
                # Partially consume the oldest chunk rather than dropping it
                # whole, so the buffer holds exactly the capacity and not
                # merely something under it.
                self._chunks[0] = head[excess:]
                self._size -= excess
                excess = 0

    def snapshot(self) -> bytes:
        """Return everything buffered, oldest byte first.

        Returns:
            A single ``bytes`` object; empty when nothing is buffered.  This is
            the only operation that joins the internal chunks, so it is the
            only one that costs a copy of the whole buffer -- call it once per
            snippet, not once per chunk.
        """
        if not self._chunks:
            return b""
        if len(self._chunks) == 1:
            return self._chunks[0]
        return b"".join(self._chunks)

    def clear(self) -> None:
        """Discard everything buffered.

        Called on a discontinuity: audio from before lost frames does not join
        up with audio from after them, and splicing the two into one snippet
        would produce a file with an audible cut and a misleading duration.
        """
        self._chunks.clear()
        self._size = 0

    def __repr__(self) -> str:
        """Return a debugging representation of the buffer's occupancy."""
        return (
            f"{type(self).__name__}(capacity_bytes={self._capacity_bytes}, "
            f"size={self._size}, chunks={len(self._chunks)})"
        )


class SnippetWriter:
    """Write one mono 16-bit PCM WAV file.

    Deliberately thinner than
    :class:`~echochamber.audio.sinks.WavRecorderSink`: that sink reconstructs a
    continuous stream from overlapping windows and therefore has to understand
    windowing, whereas a snippet is handed bytes that have already been
    de-overlapped by the gate.  All this class owns is the file.

    **The file is opened in :meth:`__init__`**, for the same reason the
    recorder does it: a bad path, a read-only directory or a name the
    filesystem rejects then fails at construction, on the thread that chose the
    path, instead of surfacing as a mid-recording exception on the audio path.
    """

    __slots__ = ("_path", "_sample_rate", "_wav", "_frames_written")

    def __init__(self, path: str | os.PathLike[str], sample_rate: int) -> None:
        """Open ``path`` for writing and prepare the header.

        Args:
            path: Destination ``.wav`` path.  Its parent directory must exist.
            sample_rate: Sample rate written into the header, in Hz.

        Raises:
            ValueError: If ``sample_rate`` is not positive.
            OSError: If the file cannot be opened.
        """
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        self._path: str | os.PathLike[str] = path
        self._sample_rate: int = sample_rate
        wav = wave.open(os.fspath(path), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(_BYTES_PER_FRAME)
        wav.setframerate(sample_rate)
        self._wav: wave.Wave_write | None = wav
        self._frames_written: int = 0

    @property
    def path(self) -> str | os.PathLike[str]:
        """Destination path of the snippet."""
        return self._path

    @property
    def sample_rate(self) -> int:
        """Sample rate written into the WAV header, in Hz."""
        return self._sample_rate

    @property
    def frames_written(self) -> int:
        """Total frames written to the file so far."""
        return self._frames_written

    @property
    def duration_s(self) -> float:
        """Length of the audio written so far, in seconds."""
        return self._frames_written / self._sample_rate

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has finalized the file."""
        return self._wav is None

    def write(self, pcm: bytes) -> int:
        """Append ``pcm`` to the file.

        A trailing partial frame -- an odd byte count -- is **dropped rather
        than written**.  The :mod:`wave` module would happily write it, and
        every subsequent sample in the file would then be assembled from the
        wrong pair of bytes, turning the rest of the snippet into noise.

        Args:
            pcm: Raw 16-bit little-endian mono sample bytes.

        Returns:
            The number of frames written, ``0`` for empty input or after
            :meth:`close` -- writing to a closed snippet is a no-op rather than
            an error, so a late chunk during shutdown cannot raise on the
            consumer thread.
        """
        wav = self._wav
        if wav is None or not pcm:
            return 0
        usable = len(pcm) - (len(pcm) % _BYTES_PER_FRAME)
        if usable <= 0:
            return 0
        data = pcm if usable == len(pcm) else pcm[:usable]
        wav.writeframes(data)
        frames = usable // _BYTES_PER_FRAME
        self._frames_written += frames
        return frames

    def close(self) -> None:
        """Finalize the WAV header and release the file.  Idempotent.

        **A snippet is only trustworthy once this has returned.**  How much of
        the audio is on disk before then is a question about buffering rather
        than about this class -- :mod:`wave` re-patches the frame count as it
        goes, but the underlying file object may have flushed none, some or all
        of it -- so a reader opening the file mid-recording can find anything
        from an empty file to a short one.  Only after ``close`` does the header
        agree with :attr:`frames_written`.  That is why the gate closes the
        writer before announcing a snippet to its callback.
        """
        wav = self._wav
        if wav is None:
            return
        self._wav = None
        wav.close()

    def __repr__(self) -> str:
        """Return a debugging representation of the writer's state."""
        return (
            f"{type(self).__name__}(path={os.fspath(self._path)!r}, "
            f"sample_rate={self._sample_rate}, "
            f"frames_written={self._frames_written}, closed={self.closed})"
        )


def snippet_filename(seq: int, phrase: str, when: float) -> str:
    """Build the filename for one snippet.

    The shape is ``YYYYmmdd-HHMMSS_NNNN_phrase-slug.wav``, e.g.
    ``20260807-142530_0007_ok-google.wav``.  Timestamp first so a plain
    alphabetical directory listing is also chronological; the sequence number
    second so two snippets opened inside the same second still sort in the
    order they happened, and cannot collide.

    ``when`` is rendered in **local** time, because the only consumer of a
    filename is a person browsing the snippet directory and matching files
    against when they remember speaking.  The honest cost: across a daylight
    saving fall-back, an hour's worth of names repeats earlier timestamps and
    the listing is briefly out of order.  The sequence number still
    distinguishes the files, and the WAV itself is unaffected.

    Args:
        seq: Snippet counter, rendered zero-padded to four digits.  Negative
            values are clamped to ``0``; this runs inside a sink that must
            never raise, so a nonsensical counter degrades rather than throws.
        phrase: The matched phrase, in any case and with any punctuation.
        when: UNIX timestamp for the name, as :func:`time.time` returns.
            Supplied by the caller and never read here -- see the module
            docstring.

    Returns:
        A filesystem-safe, sortable name ending in ``.wav``.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(when))
    return f"{stamp}_{max(0, int(seq)):04d}_{_slugify(phrase)}.wav"


def _slugify(phrase: str) -> str:
    """Reduce ``phrase`` to lowercase ASCII words joined by hyphens.

    Restricted to ASCII letters and digits rather than everything
    :meth:`str.isalnum` accepts.  Accented characters are folded the way
    :func:`echochamber.voicegate.matching.normalize` folds them, and anything
    left that is not ASCII is dropped: filenames are the one place where a
    non-ASCII character stops being a display question and becomes an encoding
    question, differing across filesystems, archive formats and the shells that
    will eventually be pointed at these files.

    Args:
        phrase: Raw phrase text.

    Returns:
        The slug, truncated to :data:`_MAX_PHRASE_CHARS` characters, or
        ``"phrase"`` when nothing usable survives -- an empty slug would
        produce a name ending in ``_.wav``, which reads like a bug.
    """
    decomposed = unicodedata.normalize("NFKD", phrase)
    kept: list[str] = []
    for char in decomposed:
        if unicodedata.combining(char):
            continue
        lowered = char.lower()
        kept.append(lowered if lowered.isascii() and lowered.isalnum() else "-")
    # Splitting on the separator collapses runs and strips the ends in one go.
    slug = "-".join(part for part in "".join(kept).split("-") if part)
    if len(slug) > _MAX_PHRASE_CHARS:
        slug = slug[:_MAX_PHRASE_CHARS].rstrip("-")
    return slug or "phrase"
