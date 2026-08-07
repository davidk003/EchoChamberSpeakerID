"""Tests for echochamber.voicegate.protocol -- the parent/worker wire format.

Nothing here needs a subprocess: the protocol is defined against ``BinaryIO``,
so every test drives it with an :class:`io.BytesIO` or with a deliberately
awkward stand-in.  Three things earn most of the assertions:

* The **header is a fixed five bytes** and both ends must agree on them
  byte-for-byte.  The header is therefore asserted as *literal bytes* built by
  hand (:func:`raw_frame`), never by calling :func:`encode_frame` on both sides
  of an equality -- a test that encodes and decodes with the same function
  passes just as happily when the layout silently changes.
* :func:`~echochamber.voicegate.protocol._read_exactly` **must loop over short
  reads**.  A pipe returns whatever is available, so a 4 KB PCM frame routinely
  arrives in pieces; a single ``read(n)`` would truncate it.  :class:`DribbleStream`
  reproduces that by handing back one byte at a time, and the large-frame test
  is the one that fails if the loop is ever removed.
* :func:`write_frame` **must flush**.  Both processes block waiting for each
  other, so an unflushed frame is a deadlock rather than a delay.
  :class:`RecordingStream` records the flush so that is a real assertion and not
  a comment.

Every failure path is asserted on its *message*, because a ``ProtocolError``
that does not say which of "desynchronised", "truncated" or "unknown kind"
happened is not much better than a hang.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

from echochamber.voicegate.protocol import (
    HEADER_SIZE,
    MAX_PAYLOAD,
    FrameKind,
    ProtocolError,
    _read_exactly,
    decode_json,
    encode_frame,
    encode_json,
    read_frame,
    write_frame,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

ALL_KINDS = list(FrameKind)


def raw_frame(kind_byte: int, length: int, payload: bytes = b"") -> bytes:
    """Build a frame by hand: 1 byte of kind, 4 big-endian bytes of length.

    Deliberately independent of :func:`encode_frame` so the two can disagree.
    ``length`` is passed separately from ``payload`` so a test can declare a
    length the payload does not actually have.
    """
    return bytes([kind_byte]) + length.to_bytes(4, "big") + payload


def stream_of(*frames: bytes) -> io.BytesIO:
    """A readable stream holding ``frames`` back to back, positioned at 0."""
    return io.BytesIO(b"".join(frames))


def read_all(stream: io.BytesIO) -> list[tuple[FrameKind, bytes]]:
    """Drain ``stream`` into a list of frames, stopping at a clean EOF."""
    frames: list[tuple[FrameKind, bytes]] = []
    while True:
        frame = read_frame(stream)
        if frame is None:
            return frames
        frames.append(frame)


class DribbleStream:
    """A stream that returns fewer bytes than asked for, on purpose.

    This is what a real pipe does under load.  ``chunk`` caps how much any one
    ``read`` hands back, so ``chunk=1`` forces ``_read_exactly`` to loop once
    per byte -- the pathological case the loop exists for.
    """

    def __init__(self, data: bytes, chunk: int = 1) -> None:
        self.data = data
        self.chunk = chunk
        self.pos = 0
        self.read_calls = 0

    def read(self, n: int = -1) -> bytes:
        self.read_calls += 1
        if n is None or n < 0:
            n = len(self.data) - self.pos
        take = min(n, self.chunk, len(self.data) - self.pos)
        out = self.data[self.pos : self.pos + take]
        self.pos += take
        return out


class RecordingStream:
    """A writable stream that records the bytes written and every flush.

    ``flush_calls`` is the point: it is how "write_frame flushes" becomes an
    assertion instead of a docstring.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.flush_calls = 0
        self.order: list[str] = []

    def write(self, data: bytes) -> int:
        self.buffer += data
        self.order.append("write")
        return len(data)

    def flush(self) -> None:
        self.flush_calls += 1
        self.order.append("flush")


# ==========================================================================
# FrameKind
# ==========================================================================

class TestFrameKind:
    """The kind byte is a compatibility contract between two interpreters."""

    def test_every_kind_has_the_documented_wire_value(self) -> None:
        """The numeric values are the wire format and may not drift."""
        assert {kind.name: int(kind) for kind in FrameKind} == {
            "AUDIO": 1,
            "RESET": 2,
            "SHUTDOWN": 3,
            "READY": 4,
            "RESULT": 5,
            "ERROR": 6,
        }

    def test_kinds_fit_in_the_single_header_byte(self) -> None:
        """A kind wider than one byte would not survive the header."""
        for kind in FrameKind:
            assert 0 <= int(kind) <= 255, f"{kind.name} does not fit in one byte"

    def test_header_size_is_five(self) -> None:
        """Both ends hard-code a 5-byte header; the constant must agree."""
        assert HEADER_SIZE == 5


# ==========================================================================
# encode_frame / read_frame
# ==========================================================================

class TestEncodeFrame:
    """encode_frame lays out the header exactly as the far side expects."""

    def test_header_is_one_kind_byte_then_four_big_endian_length_bytes(self) -> None:
        """A known frame is asserted as literal bytes, not re-encoded."""
        assert encode_frame(FrameKind.AUDIO, b"hi") == b"\x01\x00\x00\x00\x02hi"

    def test_length_prefix_is_big_endian(self) -> None:
        """258 bytes is 0x00000102 big-endian; little-endian would be 0x02010000."""
        encoded = encode_frame(FrameKind.RESULT, b"x" * 258)
        assert encoded[:HEADER_SIZE] == b"\x05\x00\x00\x01\x02", (
            f"header must be big-endian, got {encoded[:HEADER_SIZE]!r}"
        )

    def test_empty_payload_frame_is_exactly_the_header(self) -> None:
        """Control frames carry no body, so the frame is five bytes total."""
        encoded = encode_frame(FrameKind.SHUTDOWN)
        assert encoded == b"\x03\x00\x00\x00\x00"
        assert len(encoded) == HEADER_SIZE

    def test_payload_defaults_to_empty(self) -> None:
        """The payload argument is optional for control frames."""
        assert encode_frame(FrameKind.RESET) == encode_frame(FrameKind.RESET, b"")

    @pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: k.name)
    def test_encoded_length_is_header_plus_payload(self, kind: FrameKind) -> None:
        """No padding, no terminator: a frame is header + payload and nothing else."""
        payload = b"\x00\x01\x02\xff" * 7
        encoded = encode_frame(kind, payload)
        assert len(encoded) == HEADER_SIZE + len(payload)
        assert encoded[HEADER_SIZE:] == payload, "the payload must be copied verbatim"

    def test_oversized_payload_is_refused(self) -> None:
        """A payload above MAX_PAYLOAD is a bug on this side, so it raises here."""
        payload = bytes(bytearray(MAX_PAYLOAD + 1))
        with pytest.raises(ProtocolError, match=r"exceeds the \d+-byte limit"):
            encode_frame(FrameKind.AUDIO, payload)

    def test_payload_of_exactly_max_payload_is_accepted(self, monkeypatch: Any) -> None:
        """The limit is inclusive: MAX_PAYLOAD bytes is legal, one more is not.

        MAX_PAYLOAD is monkeypatched down for this test rather than allocating
        64 MB twice; the boundary arithmetic is what is under test.
        """
        import echochamber.voicegate.protocol as protocol

        monkeypatch.setattr(protocol, "MAX_PAYLOAD", 8)
        assert len(protocol.encode_frame(FrameKind.AUDIO, b"12345678")) == HEADER_SIZE + 8
        with pytest.raises(ProtocolError, match="exceeds the 8-byte limit"):
            protocol.encode_frame(FrameKind.AUDIO, b"123456789")


class TestRoundTrip:
    """Whatever encode_frame writes, read_frame must give back unchanged."""

    @pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: k.name)
    def test_round_trip_for_every_kind(self, kind: FrameKind) -> None:
        """Each FrameKind survives a full encode/read cycle with its payload."""
        payload = f"payload for {kind.name}".encode("utf-8")
        got = read_frame(stream_of(encode_frame(kind, payload)))
        assert got == (kind, payload), f"{kind.name} did not round-trip: {got!r}"

    @pytest.mark.parametrize("kind", ALL_KINDS, ids=lambda k: k.name)
    def test_round_trip_with_an_empty_payload(self, kind: FrameKind) -> None:
        """An empty payload reads back as b"", not as None or a short read."""
        got = read_frame(stream_of(encode_frame(kind, b"")))
        assert got == (kind, b""), f"{kind.name} with no payload gave {got!r}"

    def test_round_trip_preserves_arbitrary_binary(self) -> None:
        """PCM contains newlines and NULs; the framing must not care."""
        payload = (bytes(range(256)) + b"\r\n\x00") * 4
        assert b"\n" in payload and b"\r\n" in payload and b"\x00" in payload
        got = read_frame(stream_of(encode_frame(FrameKind.AUDIO, payload)))
        assert got == (FrameKind.AUDIO, payload)

    def test_several_frames_back_to_back_read_in_order(self) -> None:
        """A stream is a queue of frames; read_frame consumes exactly one each."""
        stream = io.BytesIO()
        write_frame(stream, FrameKind.READY, b'{"model":"m"}')
        write_frame(stream, FrameKind.AUDIO, b"\x01\x02\x03\x04")
        write_frame(stream, FrameKind.RESET)
        write_frame(stream, FrameKind.RESULT, b'{"text":"hello"}')
        write_frame(stream, FrameKind.SHUTDOWN)
        stream.seek(0)

        assert read_all(stream) == [
            (FrameKind.READY, b'{"model":"m"}'),
            (FrameKind.AUDIO, b"\x01\x02\x03\x04"),
            (FrameKind.RESET, b""),
            (FrameKind.RESULT, b'{"text":"hello"}'),
            (FrameKind.SHUTDOWN, b""),
        ]

    def test_reading_stops_at_the_frame_boundary(self) -> None:
        """read_frame must not over-read into the next frame."""
        stream = stream_of(
            encode_frame(FrameKind.AUDIO, b"abcd"),
            encode_frame(FrameKind.RESET),
        )
        assert read_frame(stream) == (FrameKind.AUDIO, b"abcd")
        assert stream.tell() == HEADER_SIZE + 4, (
            "the reader consumed past the end of the first frame"
        )


class TestShortReads:
    """The bug _read_exactly exists to prevent: a pipe gives partial reads."""

    def test_a_large_frame_survives_one_byte_at_a_time(self) -> None:
        """A stream dribbling 1 byte per read must still yield the whole frame.

        A single ``stream.read(n)`` would return one byte and the frame would be
        silently truncated -- which under load looks like a decoder bug, not a
        transport one.
        """
        payload = bytes(range(256)) * 20          # 5120 bytes, all byte values
        stream = DribbleStream(encode_frame(FrameKind.AUDIO, payload), chunk=1)

        got = read_frame(stream)                  # type: ignore[arg-type]

        assert got is not None
        kind, body = got
        assert kind is FrameKind.AUDIO
        assert body == payload, (
            f"got {len(body)} of {len(payload)} payload bytes -- _read_exactly "
            "is not looping over short reads"
        )
        assert stream.read_calls >= len(payload), (
            "the dribble stream should have been read once per byte"
        )

    @pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 7, 64])
    def test_frames_reassemble_at_every_chunk_size(self, chunk: int) -> None:
        """Reassembly must not depend on where the read boundaries land."""
        payload = b"".join(bytes([i % 251]) for i in range(1000))
        stream = DribbleStream(
            encode_frame(FrameKind.RESULT, payload)
            + encode_frame(FrameKind.SHUTDOWN),
            chunk=chunk,
        )
        assert read_frame(stream) == (FrameKind.RESULT, payload)  # type: ignore[arg-type]
        assert read_frame(stream) == (FrameKind.SHUTDOWN, b"")    # type: ignore[arg-type]

    def test_header_split_across_reads_is_reassembled(self) -> None:
        """Even the 5-byte header can arrive in pieces."""
        stream = DribbleStream(encode_frame(FrameKind.READY, b"ok"), chunk=2)
        assert read_frame(stream) == (FrameKind.READY, b"ok")  # type: ignore[arg-type]

    def test_read_exactly_returns_none_on_an_empty_stream(self) -> None:
        """No bytes at all is end-of-stream, which is distinct from a short read."""
        assert _read_exactly(io.BytesIO(b""), 5) is None

    def test_read_exactly_returns_what_it_got_when_the_stream_ends_early(self) -> None:
        """A partial read is returned short, so the caller can say how short."""
        assert _read_exactly(io.BytesIO(b"abc"), 10) == b"abc"


class TestReadFrameFailures:
    """Every malformed stream must fail loudly, and say which way it was bad."""

    def test_clean_eof_returns_none(self) -> None:
        """An empty stream is a peer that exited normally, not an error."""
        assert read_frame(io.BytesIO(b"")) is None

    def test_clean_eof_after_a_whole_frame_returns_none(self) -> None:
        """None is repeatable once the stream is exhausted."""
        stream = stream_of(encode_frame(FrameKind.SHUTDOWN))
        assert read_frame(stream) == (FrameKind.SHUTDOWN, b"")
        assert read_frame(stream) is None
        assert read_frame(stream) is None

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_truncated_header_raises(self, n: int) -> None:
        """A stream that dies inside the header is a broken peer, not an EOF."""
        with pytest.raises(ProtocolError, match="header"):
            read_frame(io.BytesIO(b"\x01\x00\x00\x00\x08"[:n]))

    def test_truncated_header_message_names_both_counts(self) -> None:
        """The message says how many bytes arrived out of how many needed."""
        with pytest.raises(
            ProtocolError, match=r"stream ended after 3 bytes of a 5-byte header"
        ):
            read_frame(io.BytesIO(b"\x01\x00\x00"))

    def test_truncated_payload_raises_naming_the_payload(self) -> None:
        """A frame that promised 8 bytes and delivered 3 is a torn frame."""
        with pytest.raises(
            ProtocolError,
            match=r"stream ended after 3 bytes of an? ?8-byte AUDIO payload",
        ):
            read_frame(io.BytesIO(raw_frame(int(FrameKind.AUDIO), 8, b"abc")))

    def test_payload_missing_entirely_raises(self) -> None:
        """A header with no payload behind it at all is still a torn frame."""
        with pytest.raises(ProtocolError, match="0 bytes of a 16-byte RESULT payload"):
            read_frame(io.BytesIO(raw_frame(int(FrameKind.RESULT), 16)))

    @pytest.mark.parametrize("kind_byte", [0, 7, 42, 127, 255])
    def test_unknown_kind_byte_raises(self, kind_byte: int) -> None:
        """A kind this build does not know means the stream desynchronised."""
        with pytest.raises(ProtocolError, match=f"unknown frame kind {kind_byte}"):
            read_frame(io.BytesIO(raw_frame(kind_byte, 0)))

    def test_length_above_max_payload_raises_before_allocating(self) -> None:
        """A garbage length prefix must be rejected, not turned into a 4 GB read."""
        stream = io.BytesIO(raw_frame(int(FrameKind.AUDIO), MAX_PAYLOAD + 1))
        with pytest.raises(ProtocolError, match="desynchronised"):
            read_frame(stream)

    def test_length_check_happens_before_the_kind_check(self) -> None:
        """A huge length is reported as such even when the kind is also garbage."""
        stream = io.BytesIO(raw_frame(200, 0xFFFFFFFF))
        with pytest.raises(ProtocolError, match=r"above the \d+-byte limit"):
            read_frame(stream)

    def test_length_of_exactly_max_payload_is_not_rejected_by_the_bound(self) -> None:
        """The ceiling is inclusive, so the failure is truncation, not the limit."""
        stream = io.BytesIO(raw_frame(int(FrameKind.AUDIO), MAX_PAYLOAD))
        with pytest.raises(ProtocolError, match="payload"):
            read_frame(stream)


# ==========================================================================
# write_frame
# ==========================================================================

class TestWriteFrame:
    """write_frame is encode_frame plus the flush that prevents a deadlock."""

    def test_writes_exactly_what_encode_frame_produces(self) -> None:
        """The bytes on the wire are the encoder's, unmodified."""
        stream = RecordingStream()
        write_frame(stream, FrameKind.RESULT, b'{"text":"hi"}')  # type: ignore[arg-type]
        assert bytes(stream.buffer) == encode_frame(FrameKind.RESULT, b'{"text":"hi"}')

    def test_flushes(self) -> None:
        """An unflushed frame is a deadlock: both ends block waiting to read."""
        stream = RecordingStream()
        write_frame(stream, FrameKind.AUDIO, b"\x00\x01")  # type: ignore[arg-type]
        assert stream.flush_calls == 1, (
            "write_frame must flush; a frame sitting in a buffered writer hangs "
            "the peer that is blocked reading it"
        )

    def test_flush_happens_after_the_write(self) -> None:
        """Flushing before writing would flush nothing at all."""
        stream = RecordingStream()
        write_frame(stream, FrameKind.RESET)  # type: ignore[arg-type]
        assert stream.order == ["write", "flush"]

    def test_flushes_once_per_frame(self) -> None:
        """Every frame is flushed, not just the last one before a read."""
        stream = RecordingStream()
        for kind in (FrameKind.AUDIO, FrameKind.RESET, FrameKind.SHUTDOWN):
            write_frame(stream, kind)  # type: ignore[arg-type]
        assert stream.flush_calls == 3

    def test_refuses_an_oversized_payload_without_writing_anything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The size check runs before the write, so nothing half-lands."""
        import echochamber.voicegate.protocol as protocol

        monkeypatch.setattr(protocol, "MAX_PAYLOAD", 4)
        stream = RecordingStream()
        with pytest.raises(ProtocolError, match="exceeds"):
            protocol.write_frame(stream, FrameKind.AUDIO, b"12345")  # type: ignore[arg-type]

        assert stream.buffer == b"", "a refused frame must not be partially written"
        assert stream.flush_calls == 0


# ==========================================================================
# encode_json / decode_json
# ==========================================================================

class TestJsonPayloads:
    """The control frames carry JSON objects; everything else is a bad peer."""

    @pytest.mark.parametrize(
        "obj",
        [
            {},
            {"text": "hello there", "final": True, "confidence": 0.5},
            {"model": "/models/vosk-small", "sample_rate": 16000, "phrases": []},
            {"phrases": ["ok chamber", "hey chamber"]},
            {"unicode": "éè你好"},
            {"nested": {"a": [1, 2, {"b": None}]}},
        ],
    )
    def test_round_trip(self, obj: dict[str, object]) -> None:
        """Any JSON object survives encode_json/decode_json unchanged."""
        assert decode_json(encode_json(obj)) == obj

    def test_encode_json_is_compact_utf8(self) -> None:
        """Compact separators keep the AUDIO-dominated pipe from carrying padding."""
        encoded = encode_json({"a": 1, "b": "x"})
        assert encoded == b'{"a":1,"b":"x"}'
        assert isinstance(encoded, bytes)

    def test_round_trip_through_a_frame(self) -> None:
        """The JSON helpers and the framing compose without an extra step."""
        payload = {"text": "hello", "final": True, "confidence": 0.75}
        frame = encode_frame(FrameKind.RESULT, encode_json(payload))
        got = read_frame(stream_of(frame))
        assert got is not None and got[0] is FrameKind.RESULT
        assert decode_json(got[1]) == payload

    def test_non_utf8_payload_raises(self) -> None:
        """Bytes that are not UTF-8 cannot be JSON, and must say so."""
        with pytest.raises(ProtocolError, match="not valid JSON"):
            decode_json(b'{"text":"\xff\xfe"}')

    @pytest.mark.parametrize(
        "payload",
        [b"", b"not json", b"{", b'{"a":}', b"{'a': 1}", b"nan-nonsense"],
    )
    def test_non_json_payload_raises(self, payload: bytes) -> None:
        """A malformed body is a protocol failure, not a silently empty dict."""
        with pytest.raises(ProtocolError, match="not valid JSON"):
            decode_json(payload)

    @pytest.mark.parametrize(
        "obj", [[1, 2, 3], "a string", 42, 4.5, None, True]
    )
    def test_json_that_is_not_an_object_raises(self, obj: object) -> None:
        """Every control frame is a mapping; a bare array is the wrong shape."""
        with pytest.raises(ProtocolError, match="expected a JSON object"):
            decode_json(json.dumps(obj).encode("utf-8"))

    def test_json_array_error_names_the_type_it_got(self) -> None:
        """The message names the offending type so the skew is diagnosable."""
        with pytest.raises(ProtocolError, match="expected a JSON object, got list"):
            decode_json(b'["a","b"]')

    def test_decode_json_returns_a_plain_dict(self) -> None:
        """Callers index the result directly; it must be a real mapping."""
        got = decode_json(b'{"a":1}')
        assert isinstance(got, dict)
        assert got["a"] == 1
