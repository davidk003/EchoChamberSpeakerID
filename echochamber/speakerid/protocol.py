"""The wire format between the QNN embedder chain's processes.

Length-prefixed binary frames, identical in shape to
:mod:`echochamber.voicegate.protocol` -- see that module's docstring for why
framing beats newline-delimited JSON here (the payload is raw float32 fbank
features, not text) and why stdio beats a socket. Duplicated rather than
imported so :mod:`echochamber.speakerid` and :mod:`echochamber.voicegate` stay
mutually unaware of each other; see :mod:`echochamber.speakerid`'s docstring.

Used twice, by two different process pairs:

* The main process (ARM64) <-> the driver process (x64):
  :mod:`echochamber.speakerid.qnn_subprocess` <-> ``qnn_driver_worker``.
* The driver process (x64) <-> the NPU inference process (ARM64):
  ``qnn_driver_worker`` <-> ``qnn_npu_worker``.

Both hops use the same frame kinds; the driver process is a client on one
connection and a server on the other.
"""

from __future__ import annotations

import enum
import json
import struct
from typing import BinaryIO

__all__ = [
    "HEADER_SIZE",
    "MAX_PAYLOAD",
    "FrameKind",
    "ProtocolError",
    "decode_json",
    "encode_frame",
    "encode_json",
    "read_frame",
    "write_frame",
]

HEADER_SIZE: int = 5
"""Bytes of header on every frame: one of kind, four of big-endian length."""

MAX_PAYLOAD: int = 64 * 1024 * 1024
"""Largest payload accepted, as a sanity bound rather than a real limit.

A desynchronised stream reads garbage as a length prefix; without a ceiling
that becomes a multi-gigabyte allocation far from the actual bug. Real frames
are at most a few hundred KB of float32 fbank features.
"""

_HEADER = struct.Struct(">BI")


class FrameKind(enum.IntEnum):
    """What a frame carries.

    Attributes:
        EMBED: Client to server.  Raw payload to embed -- float32 mono
            samples on the main-process/driver hop, float32 ``(T, 80)`` fbank
            features on the driver/NPU-worker hop.
        RESET: Client to server.  No payload; currently unused (the model is
            stateless per call) but kept for symmetry with
            :class:`echochamber.voicegate.protocol.FrameKind` and so a future
            stateful backend has somewhere to put it.
        SHUTDOWN: Client to server.  Exit cleanly; no payload.
        READY: Server to client.  JSON naming what loaded, sent once
            initialization (model load, QNN session creation) is done.  The
            client blocks on this.
        RESULT: Server to client.  Raw float32 embedding bytes -- ``(192,)``
            on both hops, since the driver forwards the NPU worker's output
            unchanged.
        ERROR: Server to client.  A UTF-8 message, sent instead of dying
            silently when initialization or inference fails.
    """

    EMBED = 1
    RESET = 2
    SHUTDOWN = 3
    READY = 4
    RESULT = 5
    ERROR = 6


class ProtocolError(Exception):
    """The stream carried something that is not a valid frame."""


def encode_frame(kind: FrameKind, payload: bytes = b"") -> bytes:
    """Serialise one frame.

    Args:
        kind: The frame's kind.
        payload: The frame's body; empty for control frames.

    Returns:
        Header followed by payload, ready to write.

    Raises:
        ProtocolError: If ``payload`` exceeds :data:`MAX_PAYLOAD`.
    """
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(
            f"payload of {len(payload)} bytes exceeds the {MAX_PAYLOAD}-byte limit"
        )
    return _HEADER.pack(int(kind), len(payload)) + payload


def write_frame(stream: BinaryIO, kind: FrameKind, payload: bytes = b"") -> None:
    """Write one frame and flush it.

    The flush is not optional; both ends block waiting for the other, so a
    frame sitting in a buffered writer is a deadlock rather than a delay.

    Args:
        stream: Binary stream to write to.
        kind: The frame's kind.
        payload: The frame's body.

    Raises:
        ProtocolError: If ``payload`` is too large.
        OSError: If the stream is closed or the peer has gone away.
    """
    stream.write(encode_frame(kind, payload))
    stream.flush()


def read_frame(stream: BinaryIO) -> tuple[FrameKind, bytes] | None:
    """Read one whole frame, blocking until it has arrived.

    Args:
        stream: Binary stream to read from.

    Returns:
        The frame's ``(kind, payload)``, or ``None`` at a clean end of stream.

    Raises:
        ProtocolError: If the stream ends inside a frame, if the length
            prefix exceeds :data:`MAX_PAYLOAD`, or if the kind byte is not a
            :class:`FrameKind`.
    """
    header = _read_exactly(stream, HEADER_SIZE)
    if header is None:
        return None
    if len(header) < HEADER_SIZE:
        raise ProtocolError(
            f"stream ended after {len(header)} bytes of a {HEADER_SIZE}-byte header"
        )

    raw_kind, length = _HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ProtocolError(
            f"frame declares {length} bytes, above the {MAX_PAYLOAD}-byte limit; "
            f"the stream is probably desynchronised"
        )
    try:
        kind = FrameKind(raw_kind)
    except ValueError as exc:
        raise ProtocolError(f"unknown frame kind {raw_kind}") from exc

    if length == 0:
        return kind, b""
    payload = _read_exactly(stream, length)
    if payload is None or len(payload) < length:
        got = 0 if payload is None else len(payload)
        raise ProtocolError(
            f"stream ended after {got} bytes of a {length}-byte {kind.name} payload"
        )
    return kind, payload


def encode_json(obj: object) -> bytes:
    """Serialise ``obj`` as a compact UTF-8 JSON payload."""
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def decode_json(payload: bytes) -> dict[str, object]:
    """Parse a JSON payload into a dictionary.

    Raises:
        ProtocolError: If the payload is not UTF-8, not JSON, or not an object.
    """
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _read_exactly(stream: BinaryIO, count: int) -> bytes | None:
    """Read exactly ``count`` bytes, looping over short reads.

    Args:
        stream: Binary stream to read from.
        count: Number of bytes wanted.

    Returns:
        The bytes read: exactly ``count`` on success, fewer if the stream
        ended partway, or ``None`` if it ended before any byte was read.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if not chunks:
        return None
    return b"".join(chunks)
