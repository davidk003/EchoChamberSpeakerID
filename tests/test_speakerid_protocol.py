"""Tests for echochamber.speakerid.protocol.

Same shape as tests/test_voicegate_protocol.py: this module is a duplicate of
echochamber.voicegate.protocol with its own FrameKind values (see that
module's docstring for why it is duplicated rather than shared), so its
framing correctness is tested the same way.
"""

from __future__ import annotations

import io

import pytest

from echochamber.speakerid.protocol import (
    HEADER_SIZE,
    MAX_PAYLOAD,
    FrameKind,
    ProtocolError,
    decode_json,
    encode_frame,
    encode_json,
    read_frame,
    write_frame,
)


class TestEncodeFrame:
    def test_round_trips_kind_and_payload(self) -> None:
        blob = encode_frame(FrameKind.EMBED, b"hello")
        stream = io.BytesIO(blob)
        kind, payload = read_frame(stream)
        assert kind is FrameKind.EMBED
        assert payload == b"hello"

    def test_empty_payload_round_trips(self) -> None:
        blob = encode_frame(FrameKind.SHUTDOWN)
        kind, payload = read_frame(io.BytesIO(blob))
        assert kind is FrameKind.SHUTDOWN
        assert payload == b""

    def test_header_is_five_bytes(self) -> None:
        blob = encode_frame(FrameKind.RESET)
        assert len(blob) == HEADER_SIZE

    def test_oversized_payload_raises(self) -> None:
        with pytest.raises(ProtocolError, match="exceeds"):
            encode_frame(FrameKind.EMBED, b"x" * (MAX_PAYLOAD + 1))


class TestReadFrame:
    def test_returns_none_at_clean_eof(self) -> None:
        assert read_frame(io.BytesIO(b"")) is None

    def test_truncated_header_raises(self) -> None:
        with pytest.raises(ProtocolError, match="header"):
            read_frame(io.BytesIO(b"\x01\x00"))

    def test_truncated_payload_raises(self) -> None:
        blob = encode_frame(FrameKind.RESULT, b"0123456789")
        with pytest.raises(ProtocolError, match="payload"):
            read_frame(io.BytesIO(blob[: HEADER_SIZE + 3]))

    def test_unknown_kind_byte_raises(self) -> None:
        bad = bytes([255]) + (0).to_bytes(4, "big")
        with pytest.raises(ProtocolError, match="unknown frame kind"):
            read_frame(io.BytesIO(bad))

    def test_declared_length_above_max_raises(self) -> None:
        bad = bytes([int(FrameKind.EMBED)]) + (MAX_PAYLOAD + 1).to_bytes(4, "big")
        with pytest.raises(ProtocolError, match="desynchronised"):
            read_frame(io.BytesIO(bad))

    def test_multiple_frames_in_sequence(self) -> None:
        stream = io.BytesIO(
            encode_frame(FrameKind.READY, b"a") + encode_frame(FrameKind.RESULT, b"bb")
        )
        first = read_frame(stream)
        second = read_frame(stream)
        assert first == (FrameKind.READY, b"a")
        assert second == (FrameKind.RESULT, b"bb")
        assert read_frame(stream) is None


class TestWriteFrame:
    def test_flushes_after_writing(self) -> None:
        class TrackingStream(io.BytesIO):
            def __init__(self) -> None:
                super().__init__()
                self.flush_calls = 0

            def flush(self) -> None:
                self.flush_calls += 1
                super().flush()

        stream = TrackingStream()
        write_frame(stream, FrameKind.EMBED, b"x")
        assert stream.flush_calls == 1


class TestJson:
    def test_encode_decode_round_trip(self) -> None:
        obj = {"a": 1, "b": [1, 2, 3]}
        assert decode_json(encode_json(obj)) == obj

    def test_decode_non_utf8_raises(self) -> None:
        with pytest.raises(ProtocolError, match="not valid JSON"):
            decode_json(b"\xff\xfe")

    def test_decode_non_object_raises(self) -> None:
        with pytest.raises(ProtocolError, match="expected a JSON object"):
            decode_json(b"[1, 2, 3]")


class TestFrameKind:
    def test_every_kind_has_a_distinct_value(self) -> None:
        values = [int(k) for k in FrameKind]
        assert len(values) == len(set(values))

    def test_expected_kinds_exist(self) -> None:
        assert {k.name for k in FrameKind} == {
            "EMBED",
            "RESET",
            "SHUTDOWN",
            "READY",
            "RESULT",
            "ERROR",
        }
