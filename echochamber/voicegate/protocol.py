"""The wire format between the gate and its recogniser subprocess.

Length-prefixed binary frames over the worker's stdin and stdout.  Five bytes
of header -- one byte of kind, four bytes of big-endian length -- then the
payload.  That is the whole format.

**Why framing at all, rather than newline-delimited JSON.**  Audio is the bulk
of the traffic and it is raw ``int16`` PCM: base64 would inflate it by a third
for no benefit, and a raw byte stream contains ``\\n`` constantly, so a
line-oriented protocol would need escaping anyway.  An explicit length also
means the reader never has to guess whether a short read is a message boundary
or a partial one.

**Why stdio rather than a socket.**  The worker is a child process on the same
machine with exactly one client, so a socket would add a port, a bind, a
listen and an accept -- four more things that can fail, and on Windows a
firewall prompt -- to solve a problem the pipe already solves.  Stdin/stdout
also die with the process, which is the shutdown behaviour we want anyway.

**The worker's stderr is deliberately left alone**, not folded in here.  Vosk
writes Kaldi diagnostics there, and a Python traceback from a worker that
failed to start is the single most useful artefact when this goes wrong; the
parent drains it separately and reports it.
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
that becomes a multi-gigabyte allocation and an unexplained MemoryError far
from the actual bug.  Real frames are a few kilobytes of PCM.
"""

_HEADER = struct.Struct(">BI")


class FrameKind(enum.IntEnum):
    """What a frame carries.

    Attributes:
        AUDIO: Parent to worker.  Raw 16-bit little-endian mono PCM.
        RESET: Parent to worker.  Discard decoder state; no payload.
        SHUTDOWN: Parent to worker.  Exit cleanly; no payload.
        READY: Worker to parent.  JSON naming the loaded model and rate, sent
            once the model is loaded.  The parent blocks on this, because model
            loading takes seconds and audio sent before it would be discarded.
        RESULT: Worker to parent.  JSON with ``text``, ``final`` and
            ``confidence``.
        ERROR: Worker to parent.  A UTF-8 message.  The worker sends this
            instead of dying silently when it cannot load a model.
    """

    AUDIO = 1
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

    The flush is not optional.  Both ends block waiting for the other, so a
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
        The frame's ``(kind, payload)``, or ``None`` at a clean end of stream
        -- which is how a peer that exited normally is distinguished from one
        that died mid-frame.

    Raises:
        ProtocolError: If the stream ends inside a frame, if the length prefix
            exceeds :data:`MAX_PAYLOAD`, or if the kind byte is not a
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
    """Serialise ``obj`` as a compact UTF-8 JSON payload.

    Args:
        obj: Any JSON-serialisable object.

    Returns:
        The encoded bytes.
    """
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def decode_json(payload: bytes) -> dict[str, object]:
    """Parse a JSON payload into a dictionary.

    Args:
        payload: The frame body.

    Returns:
        The decoded mapping.

    Raises:
        ProtocolError: If the payload is not UTF-8, not JSON, or not an object.
    """
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(f"payload is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError(
            f"expected a JSON object, got {type(parsed).__name__}"
        )
    return parsed


def _read_exactly(stream: BinaryIO, count: int) -> bytes | None:
    """Read exactly ``count`` bytes, looping over short reads.

    A pipe returns whatever is available, which for a 4 KB PCM frame is
    routinely less than the whole thing.  A single ``read(count)`` therefore
    silently truncates frames under load -- rarely enough in testing to look
    like a decoder bug rather than a transport one.

    Args:
        stream: Binary stream to read from.
        count: Number of bytes wanted.

    Returns:
        The bytes read: exactly ``count`` on success, fewer if the stream ended
        partway, or ``None`` if it ended before any byte was read.
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
