"""Tests for echochamber.voicegate.snippets -- buffering, writing and naming.

The three pieces here know nothing about recognition, which is the whole point
of them being a separate module: the buffering arithmetic, the WAV writing and
the naming rules can each be pinned down without a decoder, a model or a
running pipeline.  Nothing in this file imports Vosk or a recogniser.

Testing strategy worth stating up front:

* **Every WAV is read back off disk with the :mod:`wave` module**, not merely
  asserted to exist.  ``SnippetWriter`` only finalizes the header in
  ``close()``, so a file that looks right by size can still declare zero
  frames; opening it is the only assertion that catches that.
* **:func:`snippet_filename` renders local time**, so the expected timestamp is
  computed with ``time.strftime("%Y%m%d-%H%M%S", time.localtime(when))`` rather
  than hardcoded.  A hardcoded string would pass in UTC and fail everywhere
  else, which is a test bug masquerading as a code bug.
* ``PreRollBuffer`` occupancy is asserted through :attr:`~PreRollBuffer.size`
  *and* ``len(snapshot())``, because the two are maintained separately -- a
  running total and a deque -- and a bug that desynchronises them would be
  invisible to either assertion alone.
* **:attr:`~PreRollBuffer.appended` is the buffer's clock**, and the tests for
  :meth:`~PreRollBuffer.extract` are written against a *reference* copy of the
  whole appended stream rather than against remembered literals.  A range is
  therefore asserted to be the bytes that really occupied those offsets, which
  is the property the gate depends on when it cuts a wake phrase out of the
  past; comparing against a hand-copied literal would only prove the test and
  the buffer agree about what was appended.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import pytest

from echochamber.voicegate.snippets import (
    PreRollBuffer,
    SnippetWriter,
    snippet_filename,
)
from echochamber.voicegate.snippets import _MAX_PHRASE_CHARS


SR = 16_000


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _pcm(n_frames: int, value: int = 0) -> bytes:
    """Build ``n_frames`` of 16-bit little-endian mono PCM holding ``value``."""
    return (int(value) & 0xFFFF).to_bytes(2, "little") * n_frames


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


def _readable_frames(path: Path) -> int | None:
    """Frames the WAV at ``path`` declares, or ``None`` if it is not readable yet."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes()
    except Exception:
        return None


def _expected_stamp(when: float) -> str:
    """The timestamp part snippet_filename must render for ``when``.

    Computed rather than hardcoded: the function documents that it uses
    :func:`time.localtime`, so any literal would be a UTC-only assertion.
    """
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(when))


class TestPreRollBufferBasics:
    """Construction, the trivial accessors and the capacity contract."""

    def test_starts_empty(self) -> None:
        """A fresh buffer holds nothing and reports its configured capacity."""
        buf = PreRollBuffer(100)
        assert buf.capacity_bytes == 100
        assert buf.size == 0
        assert buf.snapshot() == b""

    @pytest.mark.parametrize("capacity", [-1, -2, -1000])
    def test_negative_capacity_is_rejected(self, capacity: int) -> None:
        """A negative capacity is a configuration error, not a clamp."""
        with pytest.raises(ValueError, match=r"capacity_bytes must be >= 0"):
            PreRollBuffer(capacity)

    def test_appending_nothing_is_a_no_op(self) -> None:
        """An empty append leaves the buffer untouched."""
        buf = PreRollBuffer(100)
        buf.append(b"")
        assert buf.size == 0
        assert buf.snapshot() == b""

    def test_repr_reports_the_occupancy(self) -> None:
        """The repr is for debugging a gate that is buffering the wrong amount."""
        buf = PreRollBuffer(10)
        buf.append(b"abc")
        text = repr(buf)
        assert "PreRollBuffer" in text
        assert "capacity_bytes=10" in text
        assert "size=3" in text


class TestPreRollBufferCapacity:
    """The buffer always holds the NEWEST bytes, and never more than capacity."""

    def test_appends_under_capacity_accumulate(self) -> None:
        """Nothing is evicted while there is still room."""
        buf = PreRollBuffer(10)
        buf.append(b"abc")
        buf.append(b"de")

        assert buf.size == 5
        assert buf.snapshot() == b"abcde"

    def test_snapshot_is_in_fifo_order(self) -> None:
        """Oldest byte first, so the snippet plays back in the order it arrived."""
        buf = PreRollBuffer(100)
        for part in (b"one", b"two", b"three"):
            buf.append(part)
        assert buf.snapshot() == b"onetwothree"

    def test_size_never_exceeds_capacity(self) -> None:
        """The bound holds across a long run of appends of varying sizes."""
        buf = PreRollBuffer(16)
        for n in range(1, 20):
            buf.append(bytes([n % 256]) * n)
            assert buf.size <= 16, f"size {buf.size} exceeded the 16-byte capacity"
            assert len(buf.snapshot()) == buf.size, (
                "size and the snapshot length must agree"
            )

    def test_oldest_bytes_are_evicted_first(self) -> None:
        """Overflow drops from the FRONT: the pre-roll is the most recent audio."""
        buf = PreRollBuffer(6)
        buf.append(b"aaa")
        buf.append(b"bbb")
        buf.append(b"ccc")

        assert buf.size == 6
        assert buf.snapshot() == b"bbbccc", (
            "the oldest chunk must be the one discarded, not the newest"
        )

    def test_partial_eviction_of_the_head_keeps_size_exactly_at_capacity(
        self,
    ) -> None:
        """A head chunk is trimmed, not dropped whole, so the buffer stays full.

        Dropping the whole head would leave the buffer *under* capacity, which
        silently shortens every pre-roll by up to one chunk.
        """
        buf = PreRollBuffer(10)
        buf.append(b"abcde")
        buf.append(b"fghij")
        assert buf.size == 10

        buf.append(b"klm")                       # 3 over: trim 3 off "abcde"

        assert buf.size == 10, (
            f"the head chunk must be partially consumed, leaving exactly the "
            f"capacity; got size={buf.size}"
        )
        assert buf.snapshot() == b"defghijklm"

    def test_eviction_spans_several_chunks_when_needed(self) -> None:
        """One big-ish append can evict more than one older chunk."""
        buf = PreRollBuffer(10)
        for part in (b"aa", b"bb", b"cc", b"dd", b"ee"):
            buf.append(part)
        assert buf.snapshot() == b"aabbccddee"

        buf.append(b"ffffff")                    # 6 bytes: evicts aa, bb and cc

        assert buf.size == 10
        assert buf.snapshot() == b"ddeeffffff"

    @pytest.mark.parametrize("extra", [1, 5, 50, 500])
    def test_an_append_larger_than_capacity_keeps_only_the_trailing_bytes(
        self, extra: int
    ) -> None:
        """A single oversized append keeps its LAST capacity bytes and nothing else.

        Reached in practice whenever the pre-roll is configured shorter than one
        hop, so it is a normal path rather than an error.
        """
        capacity = 8
        buf = PreRollBuffer(capacity)
        buf.append(b"OLDOLDOLD")                 # something to be thrown away

        # A repeating-but-not-uniform pattern, so keeping the leading bytes
        # instead of the trailing ones would be visible.
        payload = bytes((i * 7) % 251 for i in range(capacity + extra))
        buf.append(payload)

        assert buf.size == capacity
        assert buf.snapshot() == payload[-capacity:], (
            "an oversized append must keep its trailing bytes, not its leading ones"
        )

    def test_an_append_exactly_at_capacity_replaces_the_buffer(self) -> None:
        """Equality takes the same "keep the newest" path as an oversized append."""
        buf = PreRollBuffer(4)
        buf.append(b"old!")
        buf.append(b"new!")

        assert buf.size == 4
        assert buf.snapshot() == b"new!"

    @pytest.mark.parametrize("chunk", [b"", b"a", b"abc", b"x" * 100])
    def test_capacity_zero_is_a_black_hole(self, chunk: bytes) -> None:
        """A zero capacity is legal and swallows everything: no pre-roll configured."""
        buf = PreRollBuffer(0)
        buf.append(chunk)

        assert buf.capacity_bytes == 0
        assert buf.size == 0, "a zero-capacity buffer must never hold anything"
        assert buf.snapshot() == b""

    def test_capacity_one_keeps_the_single_newest_byte(self) -> None:
        """The degenerate non-zero capacity still keeps the newest, not the oldest."""
        buf = PreRollBuffer(1)
        buf.append(b"ab")
        buf.append(b"cd")

        assert buf.size == 1
        assert buf.snapshot() == b"d"


class TestPreRollBufferClear:
    """clear() drops everything, and the buffer is reusable afterwards."""

    def test_clear_empties_the_buffer(self) -> None:
        """Both the running total and the chunks go."""
        buf = PreRollBuffer(10)
        buf.append(b"abcde")
        buf.clear()

        assert buf.size == 0
        assert buf.snapshot() == b""

    def test_clear_keeps_the_capacity(self) -> None:
        """A discontinuity drops the audio, not the configuration."""
        buf = PreRollBuffer(10)
        buf.append(b"abcde")
        buf.clear()
        assert buf.capacity_bytes == 10

    def test_the_buffer_works_again_after_clear(self) -> None:
        """Audio from after a discontinuity buffers normally."""
        buf = PreRollBuffer(6)
        buf.append(b"aaaaaa")
        buf.clear()
        buf.append(b"bbb")

        assert buf.size == 3
        assert buf.snapshot() == b"bbb", (
            "audio from before the discontinuity must not be spliced back in"
        )

    def test_clear_on_an_empty_buffer_is_harmless(self) -> None:
        """Idempotent: clearing twice is the same as clearing once."""
        buf = PreRollBuffer(10)
        buf.clear()
        buf.clear()
        assert buf.size == 0


class TestPreRollBufferAppended:
    """appended is the stream clock: every byte ever appended, evicted or not."""

    def test_a_fresh_buffer_has_appended_nothing(self) -> None:
        """The clock starts at zero, so the first byte sits at offset 0."""
        buf = PreRollBuffer(100)
        assert buf.appended == 0
        assert buf.oldest == 0

    def test_appended_counts_every_byte(self) -> None:
        """It counts what arrived, not what is still held."""
        buf = PreRollBuffer(100)
        buf.append(b"abc")
        buf.append(b"de")

        assert buf.appended == 5
        assert buf.appended == buf.size, "nothing was evicted, so the two agree"

    def test_appended_keeps_counting_past_the_capacity(self) -> None:
        """Evicted bytes still happened; forgetting them would rewind the clock.

        A clock that stopped at the capacity would make every offset the gate
        holds mean something different once the buffer filled -- which is
        immediately, in a running pipeline.
        """
        buf = PreRollBuffer(4)
        buf.append(b"abcd")
        buf.append(b"efgh")
        buf.append(b"ij")

        assert buf.appended == 10
        assert buf.size == 4, "only the newest four bytes are retained"

    def test_an_oversized_append_counts_all_of_itself(self) -> None:
        """The bytes that were never retained were still appended."""
        buf = PreRollBuffer(4)
        buf.append(b"abcdefghij")

        assert buf.appended == 10
        assert buf.size == 4

    @pytest.mark.parametrize("chunk", [b"a", b"abc", b"x" * 100])
    def test_a_zero_capacity_buffer_still_counts(self, chunk: bytes) -> None:
        """A black hole still has a clock, so a caller's offsets stay meaningful."""
        buf = PreRollBuffer(0)
        buf.append(chunk)

        assert buf.appended == len(chunk)
        assert buf.size == 0
        assert buf.oldest == len(chunk), (
            "nothing is retained, so the oldest retained byte is one past the end"
        )

    def test_an_empty_append_does_not_move_the_clock(self) -> None:
        """A no-op append must not advance offsets nobody's bytes occupy."""
        buf = PreRollBuffer(10)
        buf.append(b"abc")
        buf.append(b"")

        assert buf.appended == 3

    @pytest.mark.parametrize("capacity", [0, 1, 4, 16, 1000])
    def test_oldest_is_appended_minus_size(self, capacity: int) -> None:
        """The invariant, held across a long run of appends of varying sizes."""
        buf = PreRollBuffer(capacity)
        for n in range(1, 20):
            buf.append(bytes([n % 256]) * n)
            assert buf.oldest == buf.appended - buf.size, (
                f"oldest={buf.oldest} disagrees with appended={buf.appended} "
                f"minus size={buf.size}"
            )
            assert buf.oldest >= 0

    def test_appended_appears_in_the_repr(self) -> None:
        """The repr is for debugging a gate that located the wrong offsets."""
        buf = PreRollBuffer(4)
        buf.append(b"abcdef")
        assert "appended=6" in repr(buf)

    def test_clear_resets_the_clock(self) -> None:
        """After a discontinuity the offsets on either side are not one stream.

        Keeping the count would let a caller holding a pre-discontinuity offset
        ask for a range that :meth:`extract` would happily satisfy out of audio
        recorded after the seam.
        """
        buf = PreRollBuffer(10)
        buf.append(b"abcde")
        buf.clear()

        assert buf.appended == 0
        assert buf.oldest == 0

    def test_the_clock_restarts_from_zero_after_a_clear(self) -> None:
        """Audio from after the seam is offset from the seam, not from the start."""
        buf = PreRollBuffer(10)
        buf.append(b"abcde")
        buf.clear()
        buf.append(b"xyz")

        assert buf.appended == 3
        assert buf.oldest == 0
        assert buf.extract(0, 3) == b"xyz"


class TestPreRollBufferExtract:
    """extract() is how the gate cuts a wake phrase out of the retained past."""

    def test_a_fully_retained_range_comes_back_verbatim(self) -> None:
        """The headline case: exact bytes for exact offsets."""
        buf = PreRollBuffer(100)
        buf.append(b"hello world")

        assert buf.extract(0, 5) == b"hello"
        assert buf.extract(6, 11) == b"world"

    def test_a_range_inside_one_chunk(self) -> None:
        """The cheap path: no join is needed, and the slice must still be right."""
        buf = PreRollBuffer(100)
        buf.append(b"abc")
        buf.append(b"defghi")

        assert buf.extract(4, 6) == b"ef"

    def test_a_range_spanning_several_chunks(self) -> None:
        """Audio arrives one chunk per hop; a phrase spans however many it takes."""
        buf = PreRollBuffer(100)
        for part in (b"ab", b"cd", b"ef", b"gh"):
            buf.append(part)

        assert buf.extract(1, 7) == b"bcdefg", (
            "the range must be stitched across the chunk boundaries it crosses"
        )

    def test_the_whole_retained_range_is_the_snapshot(self) -> None:
        """``extract(oldest, appended)`` and ``snapshot()`` describe the same bytes."""
        buf = PreRollBuffer(6)
        for part in (b"aaa", b"bbb", b"ccc"):
            buf.append(part)

        assert buf.extract(buf.oldest, buf.appended) == buf.snapshot()
        assert buf.extract(3, 9) == b"bbbccc"

    def test_a_single_byte_range(self) -> None:
        """The smallest useful range; an off-by-one here would be silent."""
        buf = PreRollBuffer(100)
        buf.append(b"abcdef")

        assert buf.extract(0, 1) == b"a"
        assert buf.extract(5, 6) == b"f"

    @pytest.mark.parametrize(("start", "end"), [(0, 0), (3, 3), (4, 2), (5, 0)])
    def test_an_empty_or_backwards_range_is_refused(
        self, start: int, end: int
    ) -> None:
        """``b""`` would read as "the phrase is here and it is silent"."""
        buf = PreRollBuffer(100)
        buf.append(b"abcdef")

        assert buf.extract(start, end) is None

    def test_a_range_that_has_been_evicted_is_refused(self) -> None:
        """The bytes are gone; the offsets now hold different audio entirely."""
        buf = PreRollBuffer(4)
        buf.append(b"abcdef")                    # keeps "cdef", oldest == 2

        assert buf.oldest == 2, "test setup"
        assert buf.extract(0, 4) is None, (
            "offsets 0 and 1 were evicted, so the range must be refused rather "
            "than served from offset 2 onwards"
        )
        assert buf.extract(1, 3) is None
        assert buf.extract(2, 6) == b"cdef", "the retained part is still readable"

    def test_a_range_that_has_not_arrived_is_refused(self) -> None:
        """The future is not buffered, however little of it is being asked for."""
        buf = PreRollBuffer(100)
        buf.append(b"abcdef")

        assert buf.extract(0, 7) is None
        assert buf.extract(6, 8) is None
        assert buf.extract(0, 6) == b"abcdef", "the boundary itself is fine"

    def test_a_partial_range_is_refused_rather_than_clipped(self) -> None:
        """Half a wake phrase sounds like a different word, so it is not returned."""
        buf = PreRollBuffer(8)
        buf.append(b"0123456789")                # keeps "23456789"

        assert buf.extract(1, 5) is None, (
            "one byte of the range was evicted; the surviving three must not be "
            "returned as though they were the whole thing"
        )

    def test_an_empty_buffer_refuses_everything(self) -> None:
        """Before any audio there is nothing to cut."""
        buf = PreRollBuffer(100)

        assert buf.extract(0, 1) is None
        assert buf.extract(0, 0) is None

    def test_a_zero_capacity_buffer_refuses_everything(self) -> None:
        """The black hole retains nothing, so no range is ever satisfiable."""
        buf = PreRollBuffer(0)
        buf.append(b"abcdef")

        assert buf.extract(0, 6) is None
        assert buf.extract(5, 6) is None

    def test_a_range_across_a_partially_consumed_head_chunk(self) -> None:
        """Eviction trims the head mid-chunk; extract must count from the trim.

        This is the case where the walk over the deque and the offsets can
        disagree: the first chunk no longer starts where it was appended, so a
        cursor that began at the chunk's original offset would return bytes
        shifted by however much was trimmed off it.
        """
        buf = PreRollBuffer(10)
        buf.append(b"abcde")
        buf.append(b"fghij")
        buf.append(b"klm")                       # trims "abc" off the head

        assert buf.size == 10, "test setup: the head was partially consumed"
        assert buf.oldest == 3
        assert buf.snapshot() == b"defghijklm"

        assert buf.extract(3, 7) == b"defg", (
            "the range starts inside the trimmed head and runs into the next "
            "chunk"
        )
        assert buf.extract(3, 5) == b"de", "wholly inside the trimmed head"
        assert buf.extract(4, 13) == b"efghijklm"
        assert buf.extract(2, 7) is None, "offset 2 was trimmed away"

    def test_a_range_after_an_oversized_append(self) -> None:
        """An append larger than the capacity replaces the deque outright."""
        buf = PreRollBuffer(4)
        buf.append(b"abc")
        buf.append(b"defghij")                   # keeps "ghij", oldest == 6

        assert buf.oldest == 6
        assert buf.extract(6, 10) == b"ghij"
        assert buf.extract(5, 10) is None

    def test_every_valid_range_matches_the_appended_stream(self) -> None:
        """Exhaustive: each retained range is compared against a reference copy.

        The reference is the concatenation of everything appended, so this
        asserts the buffer's offsets mean what the gate assumes -- a position in
        the stream -- rather than merely being self-consistent.
        """
        capacity = 12
        buf = PreRollBuffer(capacity)
        stream = bytearray()
        for n in range(1, 10):
            part = bytes((n * 31 + i) % 251 for i in range(n))
            buf.append(part)
            stream.extend(part)

            assert buf.appended == len(stream)
            for start in range(buf.oldest, buf.appended):
                for end in range(start + 1, buf.appended + 1):
                    assert buf.extract(start, end) == bytes(stream[start:end]), (
                        f"extract({start}, {end}) disagrees with the appended "
                        f"stream after {buf.appended} bytes"
                    )

    def test_extract_does_not_disturb_the_buffer(self) -> None:
        """Reading is not consuming: the same range can be cut twice."""
        buf = PreRollBuffer(10)
        buf.append(b"abcde")
        buf.append(b"fghij")

        first = buf.extract(2, 8)

        assert first == b"cdefgh"
        assert buf.extract(2, 8) == first
        assert buf.size == 10
        assert buf.appended == 10
        assert buf.snapshot() == b"abcdefghij"


class TestSnippetWriter:
    """SnippetWriter owns one file and writes whole frames only."""

    def test_the_file_is_created_by_init(self, tmp_path: Path) -> None:
        """Opening happens in __init__ so a bad path fails at the call site."""
        path = tmp_path / "opened.wav"
        writer = SnippetWriter(path, SR)
        try:
            assert path.exists()
        finally:
            writer.close()

    def test_writes_a_readable_mono_16bit_wav(self, tmp_path: Path) -> None:
        """The headline test: a real file the wave module can open and describe."""
        path = tmp_path / "snippet.wav"
        writer = SnippetWriter(path, SR)
        writer.write(_pcm(1000, 1234))
        writer.close()

        channels, sampwidth, framerate, nframes, raw = _read_wav(path)
        assert channels == 1, "snippets are mono"
        assert sampwidth == 2, "snippets are 16-bit PCM"
        assert framerate == SR
        assert nframes == 1000
        assert raw == _pcm(1000, 1234), "the samples must survive verbatim"

    @pytest.mark.parametrize("rate", [8_000, 16_000, 22_050, 44_100, 48_000])
    def test_the_header_carries_the_rate_it_was_given(
        self, tmp_path: Path, rate: int
    ) -> None:
        """The rate is written into the header, not assumed by the reader."""
        path = tmp_path / f"rate-{rate}.wav"
        writer = SnippetWriter(path, rate)
        writer.write(_pcm(10))
        writer.close()

        _, _, framerate, _, _ = _read_wav(path)
        assert framerate == rate
        assert writer.sample_rate == rate

    def test_several_writes_concatenate(self, tmp_path: Path) -> None:
        """A snippet is assembled from many chunks and must be one continuous file."""
        path = tmp_path / "many.wav"
        writer = SnippetWriter(path, SR)
        for value in (1, 2, 3):
            writer.write(_pcm(4, value))
        writer.close()

        _, _, _, nframes, raw = _read_wav(path)
        assert nframes == 12
        assert raw == _pcm(4, 1) + _pcm(4, 2) + _pcm(4, 3)

    def test_write_returns_the_frame_count(self, tmp_path: Path) -> None:
        """The return value is frames, not bytes -- the gate decrements a post-roll with it."""
        writer = SnippetWriter(tmp_path / "count.wav", SR)
        try:
            assert writer.write(_pcm(7)) == 7
            assert writer.write(_pcm(3)) == 3
        finally:
            writer.close()

    def test_frames_written_accumulates(self, tmp_path: Path) -> None:
        """frames_written is the running total across every write."""
        writer = SnippetWriter(tmp_path / "acc.wav", SR)
        try:
            assert writer.frames_written == 0
            writer.write(_pcm(100))
            assert writer.frames_written == 100
            writer.write(_pcm(50))
            assert writer.frames_written == 150
        finally:
            writer.close()

    @pytest.mark.parametrize(
        ("frames", "rate", "expected"),
        [
            (16_000, 16_000, 1.0),
            (8_000, 16_000, 0.5),
            (0, 16_000, 0.0),
            (44_100, 44_100, 1.0),
            (160, 16_000, 0.01),
        ],
    )
    def test_duration_s_is_frames_over_rate(
        self, tmp_path: Path, frames: int, rate: int, expected: float
    ) -> None:
        """duration_s is derived, so it cannot disagree with frames_written."""
        writer = SnippetWriter(tmp_path / "dur.wav", rate)
        try:
            writer.write(_pcm(frames))
            assert writer.duration_s == pytest.approx(expected)
        finally:
            writer.close()

    @pytest.mark.parametrize("n_bytes", [1, 3, 5, 101, 1001])
    def test_an_odd_byte_count_drops_the_trailing_partial_frame(
        self, tmp_path: Path, n_bytes: int
    ) -> None:
        """A half frame is discarded, not written.

        The wave module would happily write the stray byte, and every sample
        after it in the file would then be assembled from the wrong pair --
        turning the rest of the snippet into noise.
        """
        path = tmp_path / f"odd-{n_bytes}.wav"
        payload = bytes(range(256)) * (n_bytes // 256 + 1)
        payload = payload[:n_bytes]

        writer = SnippetWriter(path, SR)
        written = writer.write(payload)
        writer.close()

        assert written == n_bytes // 2, (
            f"{n_bytes} bytes is {n_bytes // 2} whole frames plus a stray byte"
        )
        assert writer.frames_written == n_bytes // 2

        _, _, _, nframes, raw = _read_wav(path)
        assert nframes == n_bytes // 2
        assert raw == payload[: (n_bytes // 2) * 2], (
            "the leading whole frames must be written unchanged"
        )

    def test_a_single_stray_byte_writes_nothing(self, tmp_path: Path) -> None:
        """One byte is less than a frame, so it contributes zero frames."""
        path = tmp_path / "one-byte.wav"
        writer = SnippetWriter(path, SR)
        assert writer.write(b"\x01") == 0
        writer.close()

        _, _, _, nframes, _ = _read_wav(path)
        assert nframes == 0

    def test_writing_empty_bytes_returns_zero(self, tmp_path: Path) -> None:
        """An empty write is a no-op, not an error."""
        writer = SnippetWriter(tmp_path / "empty-write.wav", SR)
        try:
            assert writer.write(b"") == 0
            assert writer.frames_written == 0
        finally:
            writer.close()

    def test_write_after_close_is_a_no_op_returning_zero(
        self, tmp_path: Path
    ) -> None:
        """A late chunk during shutdown must not raise on the consumer thread."""
        path = tmp_path / "late.wav"
        writer = SnippetWriter(path, SR)
        writer.write(_pcm(10))
        writer.close()

        assert writer.write(_pcm(100)) == 0, (
            "writing to a closed snippet is a no-op, not an error"
        )
        assert writer.frames_written == 10, "a late write must not move the counter"

        _, _, _, nframes, _ = _read_wav(path)
        assert nframes == 10, "the finalized file must not grow after close()"

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """The gate closes writers from more than one path; close() must tolerate it."""
        path = tmp_path / "idem.wav"
        writer = SnippetWriter(path, SR)
        writer.write(_pcm(64))

        writer.close()
        writer.close()
        writer.close()

        _, _, _, nframes, _ = _read_wav(path)
        assert nframes == 64, "a repeated close must not truncate the file"
        assert writer.frames_written == 64, "close() must not reset the counter"

    def test_closed_flag_tracks_close(self, tmp_path: Path) -> None:
        """`closed` is how the gate tells an open snippet from a finished one."""
        writer = SnippetWriter(tmp_path / "flag.wav", SR)
        assert writer.closed is False
        writer.close()
        assert writer.closed is True

    def test_an_empty_snippet_still_closes_to_a_valid_wav(
        self, tmp_path: Path
    ) -> None:
        """A snippet that never received audio is still a readable file."""
        path = tmp_path / "nothing.wav"
        SnippetWriter(path, SR).close()

        channels, sampwidth, framerate, nframes, raw = _read_wav(path)
        assert (channels, sampwidth, framerate, nframes) == (1, 2, SR, 0)
        assert raw == b""

    def test_close_finalizes_a_header_that_agrees_with_frames_written(
        self, tmp_path: Path
    ) -> None:
        """After close() the header describes exactly the frames the writer counted.

        Asserted after a *run* of writes, including one with a stray trailing
        byte: the header is patched as the file grows, so a counter that drifts
        from what was actually written only shows up once the file is finalized
        and both numbers can be compared.
        """
        path = tmp_path / "finalized.wav"
        writer = SnippetWriter(path, SR)
        writer.write(_pcm(500))
        writer.write(_pcm(250))
        writer.write(_pcm(100) + b"\x01")        # trailing partial frame
        writer.close()

        assert writer.frames_written == 850
        assert _readable_frames(path) == 850, (
            "the finalized header must declare exactly what the writer counted"
        )

    @pytest.mark.parametrize("rate", [0, -1, -16_000])
    def test_non_positive_sample_rate_is_rejected(
        self, tmp_path: Path, rate: int
    ) -> None:
        """A zero rate would make duration_s divide by zero."""
        with pytest.raises(ValueError, match=r"sample_rate must be > 0"):
            SnippetWriter(tmp_path / "bad-rate.wav", rate)

    def test_a_rejected_rate_creates_no_file(self, tmp_path: Path) -> None:
        """Validation runs before the file is opened, so nothing is left behind."""
        path = tmp_path / "never.wav"
        with pytest.raises(ValueError):
            SnippetWriter(path, 0)
        assert not path.exists()

    def test_path_property_is_what_was_given(self, tmp_path: Path) -> None:
        """The gate reports this path in its SnippetEvent, so it must round-trip."""
        path = tmp_path / "reported.wav"
        writer = SnippetWriter(path, SR)
        try:
            assert writer.path == path
        finally:
            writer.close()

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        """The gate builds its paths with os.path.join, which yields a str."""
        path = tmp_path / "as-string.wav"
        writer = SnippetWriter(str(path), SR)
        writer.write(_pcm(8))
        writer.close()

        _, _, _, nframes, _ = _read_wav(path)
        assert nframes == 8

    def test_repr_reports_path_and_progress(self, tmp_path: Path) -> None:
        """The repr is for debugging a snippet that stopped growing."""
        writer = SnippetWriter(tmp_path / "repr.wav", SR)
        try:
            writer.write(_pcm(3))
            text = repr(writer)
            assert "SnippetWriter" in text
            assert "frames_written=3" in text
            assert "closed=False" in text
        finally:
            writer.close()


class TestSnippetFilename:
    """snippet_filename is pure: it is handed the clock reading, never reads one."""

    @pytest.mark.parametrize("when", [0.0, 1_000_000_000.0, 1_770_000_000.5])
    def test_exact_name_for_a_known_timestamp(self, when: float) -> None:
        """The whole name is asserted, computed in LOCAL time as documented.

        Hardcoding the timestamp would make this test pass only in the timezone
        it was written in, so the expectation is built with the same
        ``time.localtime`` the function documents.
        """
        expected = f"{_expected_stamp(when)}_0007_ok-google.wav"
        assert snippet_filename(7, "ok google", when) == expected

    def test_the_shape_is_stamp_seq_slug(self) -> None:
        """Timestamp first so an alphabetical listing is also chronological."""
        when = 1_700_000_000.0
        name = snippet_filename(42, "Ok, Google!", when)
        stamp, seq, rest = name.split("_", 2)

        assert stamp == _expected_stamp(when)
        assert seq == "0042"
        assert rest == "ok-google.wav"

    @pytest.mark.parametrize(
        ("seq", "expected"),
        [
            (0, "0000"),
            (1, "0001"),
            (7, "0007"),
            (42, "0042"),
            (999, "0999"),
            (1000, "1000"),
            (9999, "9999"),
            (10_000, "10000"),      # wider than the pad rather than truncated
        ],
    )
    def test_sequence_is_zero_padded_to_four_digits(
        self, seq: int, expected: str
    ) -> None:
        """Padding keeps a directory listing sorted for the first 10 000 snippets."""
        name = snippet_filename(seq, "x", 1_700_000_000.0)
        assert name.split("_")[1] == expected

    @pytest.mark.parametrize("seq", [-1, -7, -9999])
    def test_negative_sequence_clamps_to_zero(self, seq: int) -> None:
        """This runs inside a sink that must never raise, so nonsense degrades."""
        name = snippet_filename(seq, "x", 1_700_000_000.0)
        assert name.split("_")[1] == "0000", (
            f"seq={seq} must clamp to 0 rather than produce a name with a minus sign"
        )
        assert "-000" not in name.split("_")[1]

    @pytest.mark.parametrize(
        ("phrase", "slug"),
        [
            ("ok google", "ok-google"),
            ("OK GOOGLE", "ok-google"),
            ("Ok, Google!", "ok-google"),
            ("  ok   google  ", "ok-google"),
            ("ok google play music", "ok-google-play-music"),
            ("hey", "hey"),
            ("channel 4", "channel-4"),
            ("what's up", "what-s-up"),
            ("Café", "cafe"),
            ("Über alles", "uber-alles"),
        ],
    )
    def test_slug_is_lowercased_and_hyphenated(
        self, phrase: str, slug: str
    ) -> None:
        """Words join with hyphens; case, spacing and punctuation are folded away."""
        name = snippet_filename(0, phrase, 1_700_000_000.0)
        assert name.endswith(f"_{slug}.wav"), (
            f"{phrase!r} must slugify to {slug!r}, got {name!r}"
        )

    @pytest.mark.parametrize(
        "phrase", ["", "   ", "!!!", "...", ",", "---", "***", "?!?", "\t\n"]
    )
    def test_a_phrase_with_nothing_usable_falls_back_to_phrase(
        self, phrase: str
    ) -> None:
        """An empty slug would produce a name ending in "_.wav", which reads like a bug."""
        name = snippet_filename(0, phrase, 1_700_000_000.0)
        assert name.endswith("_phrase.wav"), (
            f"{phrase!r} leaves no usable slug and must fall back to 'phrase', "
            f"got {name!r}"
        )

    def test_non_ascii_only_phrase_falls_back_to_phrase(self) -> None:
        """Non-ASCII is dropped from filenames entirely; nothing usable remains."""
        assert snippet_filename(0, "日本語", 1_700_000_000.0).endswith(
            "_phrase.wav"
        )

    def test_a_long_phrase_is_truncated(self) -> None:
        """Operator text is unbounded; Windows still enforces a 260-char path."""
        phrase = "b" * 100
        slug = snippet_filename(0, phrase, 1_700_000_000.0).split("_")[2][: -len(".wav")]

        assert len(slug) == _MAX_PHRASE_CHARS
        assert slug == "b" * _MAX_PHRASE_CHARS

    def test_truncation_never_leaves_a_trailing_hyphen(self) -> None:
        """Cutting exactly on a word boundary must not leave "...word-.wav"."""
        phrase = "a" * (_MAX_PHRASE_CHARS - 1) + " tail"
        slug = snippet_filename(0, phrase, 1_700_000_000.0).split("_")[2][: -len(".wav")]

        assert not slug.endswith("-"), (
            f"truncation left a dangling separator: {slug!r}"
        )
        assert slug == "a" * (_MAX_PHRASE_CHARS - 1)

    @pytest.mark.parametrize("words", [5, 12, 30])
    def test_no_slug_exceeds_the_ceiling(self, words: int) -> None:
        """Whatever the phrase, the slug stays inside the documented ceiling."""
        phrase = " ".join(["wakeword"] * words)
        slug = snippet_filename(0, phrase, 1_700_000_000.0).split("_")[2][: -len(".wav")]
        assert 0 < len(slug) <= _MAX_PHRASE_CHARS

    def test_always_ends_in_wav(self) -> None:
        """The extension is fixed; callers join it straight onto a directory."""
        for phrase in ("ok google", "", "!!!", "x" * 200):
            assert snippet_filename(0, phrase, 1_700_000_000.0).endswith(".wav")

    def test_names_from_the_same_second_are_distinguished_by_sequence(self) -> None:
        """Two snippets opened inside one second must not collide."""
        when = 1_700_000_000.0
        first = snippet_filename(0, "ok google", when)
        second = snippet_filename(1, "ok google", when)

        assert first != second
        assert first < second, "the sequence keeps same-second names in order"

    def test_later_timestamps_sort_later(self) -> None:
        """Timestamp-first naming makes an alphabetical listing chronological."""
        earlier = snippet_filename(9, "ok google", 1_700_000_000.0)
        later = snippet_filename(0, "ok google", 1_700_003_600.0)
        assert earlier < later, (
            "an hour later must sort after, even with a lower sequence number"
        )

    def test_the_name_is_usable_as_a_filename(self, tmp_path: Path) -> None:
        """The end-to-end point of slugifying: the name can actually be written."""
        name = snippet_filename(3, "OK, Google! Play/some\\music?", 1_700_000_000.0)
        path = tmp_path / name

        writer = SnippetWriter(path, SR)
        writer.write(_pcm(16))
        writer.close()

        assert path.exists()
        _, _, _, nframes, _ = _read_wav(path)
        assert nframes == 16

    def test_does_not_read_the_clock(self) -> None:
        """Handing it the same timestamp twice must give the same name."""
        when = 1_700_000_000.0
        assert snippet_filename(1, "ok google", when) == snippet_filename(
            1, "ok google", when
        )

    def test_an_integer_timestamp_is_accepted(self) -> None:
        """time.time() returns a float, but an int must not blow up either."""
        assert snippet_filename(1, "ok google", 1_700_000_000).startswith(
            _expected_stamp(1_700_000_000)
        )
