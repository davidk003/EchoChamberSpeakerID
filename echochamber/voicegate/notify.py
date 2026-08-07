"""Announcing wake-phrase events over a WebSocket.

**Nothing here may ever block the consumer thread.**  That is the whole design
constraint.  :meth:`WebSocketNotifier.notify` is called from inside
:meth:`~echochamber.voicegate.sink.VoiceGateSink.on_chunk`, which runs on the
pipeline's consumer thread -- the thread that drains the bounded queue.  A
socket ``send`` to an unreachable host can take a TCP timeout to fail, and for
that whole time the queue would not be drained, chunks would be discarded under
``DROP_OLDEST``, and the gate would go deaf to the audio it was discarding.  A
network problem would become an audio problem.

So the shape is the same one the pipeline already uses to separate the chunker
from a slow sink: a bounded buffer and a dedicated thread.  ``notify`` appends
to a deque and returns; a ``voicegate-notify`` thread does the connecting,
reconnecting and sending.  When the buffer is full the **oldest** event is
discarded and counted, exactly as
:class:`~echochamber.audio.sinks.QueueSink` does and for the same reason: a
notifier that cannot keep up must cost visible dropped events rather than
invisible dropped audio.

**Two events, because detection and recording finish at different times.**  A
phrase is recognised at one moment, but the snippet containing it is not
complete until ``post_roll_ms`` later -- three seconds, by default.  Sending
only on completion would make a "wake word detected" signal arrive three
seconds late; sending only on detection would never carry the audio.  So
:attr:`EventKind.DETECTED` goes out immediately with no audio, and
:attr:`EventKind.SNIPPET` follows when the file is closed.  Consumers that want
one and not the other say so in :attr:`NotifyConfig.events`.

**Reconnection is backed off, not retried in a loop.**  A server that is down
stays down for seconds or hours, and a client that reconnects as fast as it can
fail is indistinguishable from a denial-of-service.  The delay doubles from
:attr:`NotifyConfig.reconnect_initial_s` to
:attr:`NotifyConfig.reconnect_max_s` and resets on a successful send.

**The transport is a protocol, and the default sends nothing.**  Like the
recogniser, the real implementation is behind a lazy import of an optional
dependency: nothing in this module imports ``websocket`` at module scope, so a
checkout without it still imports, still runs, and still passes its tests.
:class:`RecordingTransport` is what the tests use, and it never opens a socket.
"""

from __future__ import annotations

import base64
import collections
import enum
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_URL",
    "EventKind",
    "NotifyConfig",
    "NotifyEvent",
    "NotifyStats",
    "NullTransport",
    "RecordingTransport",
    "Transport",
    "WebSocketNotifier",
    "build_transport",
]

DEFAULT_URL: str = "ws://127.0.0.1:8765"
"""Where events go when nothing else is configured: a local listener."""

_JOIN_TIMEOUT_S: float = 2.0
"""How long :meth:`WebSocketNotifier.close` waits for the sender to finish.

Bounded because ``close`` runs on the shutdown path, and a wedged socket must
cost a lost tail of events rather than a GUI that will not exit.
"""

_POLL_S: float = 0.1
"""How often the sender re-checks for work while waiting to reconnect."""


class EventKind(enum.Enum):
    """What happened.

    Attributes:
        DETECTED: A wake phrase was recognised.  Sent immediately, carries no
            audio -- the snippet does not exist yet.
        SNIPPET: A snippet file was completed and closed.  Sent
            ``post_roll_ms`` or more after the detection it belongs to, and
            carries the audio when
            :attr:`NotifyConfig.include_audio` is set.
    """

    DETECTED = "detected"
    SNIPPET = "snippet"


@dataclass(frozen=True, slots=True)
class NotifyConfig:
    """How and where wake-phrase events are announced.

    Attributes:
        enabled: Whether anything is sent at all.  ``False`` by default: this
            opens a network connection, which is not something a capture tool
            should start doing because a feature was merged.
        url: WebSocket endpoint, ``ws://`` or ``wss://``.
        events: Which :class:`EventKind` values to send.  Both by default.
        include_audio: Whether a ``SNIPPET`` event carries the WAV file's bytes,
            base64-encoded.  **On by default**, because sending the audio that
            triggered the gate is the point of the gate.  Under
            :attr:`~echochamber.voicegate.config.ClipMode.PHRASE` a clip is the
            hotword and little else -- around a second, ~32 KB, ~43 KB once
            base64'd -- so this is cheap.  Turn it off for a consumer that only
            wants to know *that* a phrase was heard, or when clips are being
            collected from the ``path`` instead.
        max_audio_bytes: Ceiling on an included snippet.  A file above it is
            sent **without** audio rather than truncated: half a WAV under a
            header claiming the full length is indistinguishable from a
            complete one, so a consumer could not tell it had been cut.
        queue_max: Events buffered before the oldest is dropped.  Small on
            purpose: these are notifications, and a backlog of stale ones is
            worth less than the newest.
        connect_timeout_s: Timeout for opening the connection.
        send_timeout_s: Timeout for one send.
        reconnect_initial_s: First reconnect delay after a failure.
        reconnect_max_s: Ceiling the backoff doubles up to.
        headers: Extra HTTP headers for the opening handshake -- an
            ``Authorization`` line, typically.

    Raises:
        ValueError: If ``enabled`` without a ``url``, if any timeout or delay is
            not positive, if ``queue_max`` is below 1, or if
            ``reconnect_initial_s`` exceeds ``reconnect_max_s``.
    """

    enabled: bool = False
    url: str = DEFAULT_URL
    events: frozenset[EventKind] = frozenset(EventKind)
    include_audio: bool = True
    max_audio_bytes: int = 4 * 1024 * 1024
    queue_max: int = 32
    connect_timeout_s: float = 5.0
    send_timeout_s: float = 5.0
    reconnect_initial_s: float = 1.0
    reconnect_max_s: float = 30.0
    headers: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate the configuration; see the class docstring for the rules."""
        if self.queue_max < 1:
            raise ValueError(f"queue_max must be >= 1, got {self.queue_max}")
        if self.max_audio_bytes < 1:
            raise ValueError(
                f"max_audio_bytes must be >= 1, got {self.max_audio_bytes}"
            )
        for name in (
            "connect_timeout_s",
            "send_timeout_s",
            "reconnect_initial_s",
            "reconnect_max_s",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.reconnect_initial_s > self.reconnect_max_s:
            raise ValueError(
                f"reconnect_initial_s ({self.reconnect_initial_s}) must be <= "
                f"reconnect_max_s ({self.reconnect_max_s})"
            )
        if not isinstance(self.events, frozenset):
            raise TypeError(
                f"events must be a frozenset so the config stays hashable, got "
                f"{type(self.events).__name__}"
            )
        if self.enabled:
            if not self.url:
                raise ValueError("url must not be empty when notifications are enabled")
            if not self.url.startswith(("ws://", "wss://")):
                raise ValueError(
                    f"url must start with ws:// or wss://, got {self.url!r}"
                )
            if not self.events:
                raise ValueError(
                    "events must name at least one EventKind when notifications "
                    "are enabled; an enabled notifier that sends nothing is "
                    "almost certainly a mistake"
                )

    def wants(self, kind: EventKind) -> bool:
        """Return whether ``kind`` should be sent.

        Args:
            kind: The event kind in question.

        Returns:
            ``True`` if enabled and ``kind`` is in :attr:`events`.
        """
        return self.enabled and kind in self.events


@dataclass(frozen=True, slots=True)
class NotifyEvent:
    """One thing worth telling a listener about.

    Attributes:
        kind: What happened.
        phrase: The matched wake phrase, normalised.
        text: The full recognised text the phrase was found in.
        seq: Snippet counter this event belongs to, from 0.  A ``DETECTED`` and
            the ``SNIPPET`` it leads to share a ``seq``, which is how a consumer
            pairs them.
        sample_rate: Capture sample rate in Hz.
        timestamp: UNIX time the event was created.
        start_frame: Absolute frame index the snippet begins at; ``0`` for a
            ``DETECTED`` event, whose snippet does not exist yet.
        path: Snippet path, or ``None`` for ``DETECTED``.
        frames: Frames in the snippet, ``0`` for ``DETECTED``.
        duration_s: Snippet length in seconds, ``0.0`` for ``DETECTED``.
        truncated: Whether the snippet hit the length ceiling.
        audio: Raw WAV bytes when the notifier was told to include them, else
            ``None``.  Held as bytes and encoded only at serialisation, so an
            event that is dropped from the queue never pays for base64.
        speaker: Name of the enrolled speaker the phrase was verified
            against, or ``None`` when speaker verification is not configured.
            A phrase that failed verification never reaches this event at
            all -- see :meth:`echochamber.voicegate.sink.VoiceGateSink._on_match`,
            which suppresses it before ``on_detected`` is ever called.
        speaker_score: Cosine similarity of ``speaker``'s match, ``0.0`` when
            ``speaker`` is ``None``.
    """

    kind: EventKind
    phrase: str
    text: str = ""
    seq: int = 0
    sample_rate: int = 0
    timestamp: float = 0.0
    start_frame: int = 0
    path: str | None = None
    frames: int = 0
    duration_s: float = 0.0
    truncated: bool = False
    audio: bytes | None = None
    speaker: str | None = None
    speaker_score: float = 0.0

    def to_payload(self) -> dict[str, Any]:
        """Render this event as the JSON object that goes on the wire.

        Returns:
            A plain dictionary.  ``DETECTED`` events omit every snippet field
            rather than sending zeros, because a consumer cannot tell a real
            ``duration_s`` of ``0.0`` from a placeholder one, and a missing key
            is unambiguous.  The same rule applies to ``speaker``: omitted
            entirely, on either event kind, when no verifier was configured.
        """
        payload: dict[str, Any] = {
            "type": self.kind.value,
            "phrase": self.phrase,
            "text": self.text,
            "seq": self.seq,
            "sample_rate": self.sample_rate,
            "timestamp": self.timestamp,
        }
        if self.speaker is not None:
            payload["speaker"] = self.speaker
            payload["speaker_score"] = round(self.speaker_score, 4)
        if self.kind is EventKind.SNIPPET:
            payload.update(
                {
                    "path": self.path,
                    "start_frame": self.start_frame,
                    "frames": self.frames,
                    "duration_s": round(self.duration_s, 4),
                    "truncated": self.truncated,
                }
            )
            if self.audio is not None:
                payload["audio"] = {
                    "encoding": "base64",
                    "format": "wav",
                    "bytes": len(self.audio),
                    "data": base64.b64encode(self.audio).decode("ascii"),
                }
        return payload

    def to_json(self) -> str:
        """Serialise this event for the wire.

        Returns:
            A compact JSON string.
        """
        return json.dumps(self.to_payload(), separators=(",", ":"))

    def __repr__(self) -> str:
        """Return a debugging representation that never dumps the audio."""
        audio = "none" if self.audio is None else f"{len(self.audio)}B"
        return (
            f"{type(self).__name__}(kind={self.kind.value}, "
            f"phrase={self.phrase!r}, seq={self.seq}, audio={audio})"
        )


@dataclass(frozen=True, slots=True)
class NotifyStats:
    """A detached copy of a notifier's counters.

    Frozen and returned by :meth:`WebSocketNotifier.snapshot` for the same
    reason :meth:`echochamber.voicegate.sink.VoiceGateSink.snapshot` is: the GUI
    polls while the sender thread updates, and reading fields one at a time
    would let a display mix two instants.  See that method for the one field
    -- :attr:`queued` -- that is sampled separately, and why.

    Attributes:
        queued: Events waiting to be sent.
        sent: Events successfully written to the socket.
        dropped: Events discarded because the buffer was full.  Non-zero means
            the listener cannot keep up, or is not there.
        failed: Sends that raised.  A send that fails is retried once via
            reconnection, so this can exceed the number of lost events.
        connects: Successful connections, including reconnections.
        connected: Whether the transport is currently open.
        error: The most recent failure, or ``None``.
    """

    queued: int = 0
    sent: int = 0
    dropped: int = 0
    failed: int = 0
    connects: int = 0
    connected: bool = False
    error: str | None = None

    def __repr__(self) -> str:
        """Return a debugging representation of the counters that matter."""
        return (
            f"{type(self).__name__}(queued={self.queued}, sent={self.sent}, "
            f"dropped={self.dropped}, failed={self.failed}, "
            f"connected={self.connected})"
        )


@runtime_checkable
class Transport(Protocol):
    """Anything that can carry a text frame to a listener.

    Deliberately narrower than a WebSocket: three methods, all synchronous,
    none of them aware of what they are carrying.  That is what lets the tests
    run the entire notifier -- queueing, backoff, drop policy, shutdown --
    without a socket.
    """

    def connect(self) -> None:
        """Open the connection, blocking until it is ready or fails.

        Raises:
            Exception: If the connection could not be established.
        """
        ...

    def send(self, message: str) -> None:
        """Write one text frame.

        Args:
            message: The payload.

        Raises:
            Exception: If the write failed.
        """
        ...

    def close(self) -> None:
        """Close the connection.  Must be idempotent and must not raise."""
        ...


class NullTransport:
    """A transport that accepts everything and sends nothing.

    The default, and what a disabled notifier holds, so the notifier's own logic
    is identical whether or not anything is configured.
    """

    __slots__ = ("_closed",)

    def __init__(self) -> None:
        """Create the transport.  Nothing is allocated."""
        self._closed: bool = False

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def connect(self) -> None:
        """Succeed without doing anything."""

    def send(self, message: str) -> None:
        """Discard ``message``.

        Args:
            message: Ignored.
        """

    def close(self) -> None:
        """Mark the transport closed.  Idempotent."""
        self._closed = True

    def __repr__(self) -> str:
        """Return a debugging representation of this transport."""
        return f"{type(self).__name__}(closed={self._closed})"


class RecordingTransport:
    """A transport that keeps what it was given, for tests and demos.

    Ships in the package rather than the tests because the GUI can use it to
    show the gate working end to end with no server running, and because a
    scripted failure is the only way to exercise the reconnect path
    deterministically.

    Args:
        fail_connects: Number of leading :meth:`connect` calls that raise.
        fail_sends: Number of leading :meth:`send` calls that raise.
    """

    __slots__ = ("_messages", "_lock", "_connects", "_closed", "_fail_connects", "_fail_sends")

    def __init__(self, fail_connects: int = 0, fail_sends: int = 0) -> None:
        """Prepare a transport that optionally fails a few times first.

        Args:
            fail_connects: How many connects raise before one succeeds.
            fail_sends: How many sends raise before one succeeds.
        """
        self._messages: list[str] = []
        self._lock: threading.Lock = threading.Lock()
        self._connects: int = 0
        self._closed: bool = False
        self._fail_connects: int = int(fail_connects)
        self._fail_sends: int = int(fail_sends)

    @property
    def messages(self) -> list[str]:
        """A copy of every message accepted so far, in order."""
        with self._lock:
            return list(self._messages)

    @property
    def connects(self) -> int:
        """How many times :meth:`connect` has been called, failures included."""
        with self._lock:
            return self._connects

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    def connect(self) -> None:
        """Count the attempt, raising for the first ``fail_connects`` calls.

        Raises:
            ConnectionError: While the scripted failure count is not exhausted.
        """
        with self._lock:
            self._connects += 1
            if self._fail_connects > 0:
                self._fail_connects -= 1
                raise ConnectionError("scripted connect failure")

    def send(self, message: str) -> None:
        """Record ``message``, raising for the first ``fail_sends`` calls.

        Args:
            message: The payload.

        Raises:
            ConnectionError: While the scripted failure count is not exhausted.
        """
        with self._lock:
            if self._fail_sends > 0:
                self._fail_sends -= 1
                raise ConnectionError("scripted send failure")
            self._messages.append(message)

    def close(self) -> None:
        """Mark the transport closed.  Idempotent, never raises."""
        self._closed = True

    def __repr__(self) -> str:
        """Return a debugging representation of what was recorded."""
        return (
            f"{type(self).__name__}(messages={len(self.messages)}, "
            f"connects={self.connects}, closed={self._closed})"
        )


class WebSocketNotifier:
    """Buffer wake-phrase events and send them from a background thread.

    :meth:`notify` is the only method the audio path calls, and it never blocks,
    never connects and never raises.  Everything else happens on the
    ``voicegate-notify`` thread.

    Single-use: once :meth:`close` has run the notifier cannot be restarted,
    which matches the pipeline's own lifecycle.
    """

    __slots__ = (
        "_config",
        "_transport",
        "_thread",
        "_cond",
        "_events",
        "_running",
        "_closed",
        "_lock",
        "_sent",
        "_dropped",
        "_failed",
        "_connects",
        "_connected",
        "_error",
    )

    def __init__(
        self, config: NotifyConfig, transport: Transport | None = None
    ) -> None:
        """Prepare a notifier; nothing is connected until :meth:`start`.

        Args:
            config: What to send and where.
            transport: The transport to use.  Defaults to
                :class:`NullTransport`, so a notifier built without one is inert
                rather than broken.  Tests inject
                :class:`RecordingTransport`; the GUI injects whatever
                :func:`build_transport` produced.
        """
        self._config: NotifyConfig = config
        self._transport: Transport = (
            NullTransport() if transport is None else transport
        )
        self._thread: threading.Thread | None = None
        self._cond: threading.Condition = threading.Condition()
        self._events: collections.deque[NotifyEvent] = collections.deque()
        self._running: bool = False
        self._closed: bool = False

        self._lock: threading.Lock = threading.Lock()
        self._sent: int = 0
        self._dropped: int = 0
        self._failed: int = 0
        self._connects: int = 0
        self._connected: bool = False
        self._error: str | None = None

    @property
    def config(self) -> NotifyConfig:
        """The configuration in force."""
        return self._config

    @property
    def transport(self) -> Transport:
        """The transport events are written to."""
        return self._transport

    @property
    def running(self) -> bool:
        """``True`` between :meth:`start` and :meth:`close`."""
        return self._running

    @property
    def closed(self) -> bool:
        """``True`` once :meth:`close` has been called."""
        return self._closed

    @property
    def queued(self) -> int:
        """Events buffered and not yet sent."""
        with self._cond:
            return len(self._events)

    def snapshot(self) -> NotifyStats:
        """Return a detached copy of every counter.

        Returns:
            The counters as of this moment.  The send counters agree with each
            other, being read under one lock; ``queued`` comes from the queue's
            own condition and so is sampled a moment earlier.  That seam is
            deliberate -- taking both locks together would order them against
            the sender thread, which holds the condition while it waits -- and
            it is harmless, because ``queued`` describes a backlog that is
            changing anyway and no consumer derives one field from another.
        """
        queued = self.queued
        with self._lock:
            return NotifyStats(
                queued=queued,
                sent=self._sent,
                dropped=self._dropped,
                failed=self._failed,
                connects=self._connects,
                connected=self._connected,
                error=self._error,
            )

    def start(self) -> None:
        """Start the sender thread.

        Connecting is left to the thread rather than done here: the endpoint may
        be unreachable, and a GUI that froze on Start because a notification
        server was down would be a much worse bug than not notifying.

        Raises:
            RuntimeError: If already started, or started after :meth:`close`.
        """
        if self._closed:
            raise RuntimeError(
                "WebSocketNotifier has been closed and cannot be restarted; "
                "create a new WebSocketNotifier"
            )
        if self._running:
            raise RuntimeError("WebSocketNotifier has already been started")
        self._running = True
        thread = threading.Thread(
            target=self._run, name="voicegate-notify", daemon=True
        )
        self._thread = thread
        thread.start()

    def notify(self, event: NotifyEvent) -> bool:
        """Queue ``event`` for sending.  Never blocks, never raises.

        Called from the pipeline's consumer thread; see the module docstring for
        why that rules out doing anything else here.

        Args:
            event: What to send.

        Returns:
            ``True`` if the event was queued, ``False`` if it was filtered out
            by :attr:`NotifyConfig.events`, or the notifier is closed.  A queued
            event that is later dropped for space still returns ``True`` -- the
            drop is reported through :attr:`NotifyStats.dropped`, because by
            then the caller is long gone.
        """
        if self._closed or not self._config.wants(event.kind):
            return False
        with self._cond:
            if len(self._events) >= self._config.queue_max:
                # Oldest first: a backlog of stale notifications is worth less
                # than the newest one, and this must not block the audio path.
                self._events.popleft()
                with self._lock:
                    self._dropped += 1
            self._events.append(event)
            self._cond.notify_all()
        return True

    def close(self) -> None:
        """Stop the sender and close the transport.  Idempotent, never raises.

        Queued events are **not** flushed.  Shutdown is bounded, the endpoint
        may be exactly what is wedged, and a notification that arrives after the
        capture it describes has ended is worth less than exiting promptly.
        """
        if self._closed:
            return
        self._closed = True
        with self._cond:
            self._running = False
            self._cond.notify_all()

        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(_JOIN_TIMEOUT_S)
        try:
            self._transport.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            self._record_error(exc)
        with self._lock:
            self._connected = False

    def _run(self) -> None:
        """Sender thread body: connect, drain, reconnect with backoff."""
        delay = self._config.reconnect_initial_s
        connected = False
        while True:
            event = self._take()
            if event is None:
                break

            if not connected:
                if not self._connect():
                    # Put the event back at the front: it has not been sent, and
                    # dropping it because the socket happened to be down would
                    # lose exactly the events a listener most wants.
                    self._requeue(event)
                    if not self._sleep(delay):
                        break
                    delay = min(delay * 2.0, self._config.reconnect_max_s)
                    continue
                connected = True
                delay = self._config.reconnect_initial_s

            if self._send(event):
                delay = self._config.reconnect_initial_s
                continue

            # A failed send means the connection is gone, whatever the socket
            # thinks.  Drop it, requeue the event and let the next iteration
            # reconnect; retrying on a half-open socket just fails again.
            connected = False
            self._disconnect()
            self._requeue(event)
            if not self._sleep(delay):
                break
            delay = min(delay * 2.0, self._config.reconnect_max_s)

    def _take(self) -> NotifyEvent | None:
        """Block until an event is available or the notifier stops.

        Returns:
            The next event, or ``None`` when the notifier is shutting down.
        """
        with self._cond:
            while self._running and not self._events:
                self._cond.wait(_POLL_S)
            if not self._running:
                return None
            return self._events.popleft()

    def _requeue(self, event: NotifyEvent) -> None:
        """Put ``event`` back at the head of the queue.

        Args:
            event: The event that could not be sent.  Discarded if the queue
                filled while it was in flight -- it is the oldest, so it is the
                one the drop policy would have chosen anyway.
        """
        with self._cond:
            if len(self._events) >= self._config.queue_max:
                with self._lock:
                    self._dropped += 1
                return
            self._events.appendleft(event)

    def _sleep(self, seconds: float) -> bool:
        """Wait out a backoff, waking early if the notifier is closing.

        Args:
            seconds: How long to wait.

        Returns:
            ``True`` if the wait completed normally, ``False`` if the notifier
            is shutting down and the sender should stop.  Waiting on the
            condition rather than sleeping is what keeps ``close`` bounded even
            with a 30 s backoff in progress.
        """
        deadline = time.monotonic() + seconds
        with self._cond:
            while self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return True
                self._cond.wait(remaining)
            return False

    def _connect(self) -> bool:
        """Try to open the transport.

        Returns:
            ``True`` on success.  A failure is counted and recorded, never
            raised: this is a daemon thread, and the owner learns through
            :meth:`snapshot`.
        """
        try:
            self._transport.connect()
        except Exception as exc:  # noqa: BLE001 - reported via snapshot()
            self._record_error(exc)
            with self._lock:
                self._connected = False
            return False
        with self._lock:
            self._connects += 1
            self._connected = True
            self._error = None
        return True

    def _send(self, event: NotifyEvent) -> bool:
        """Try to write one event.

        Args:
            event: What to send.

        Returns:
            ``True`` on success.  Serialisation failures are counted like send
            failures but are *not* retried by reconnecting -- an event that
            cannot be encoded will not encode any better on a new socket -- so
            they are swallowed here and reported as sent-and-lost.
        """
        try:
            message = event.to_json()
        except Exception as exc:  # noqa: BLE001 - a bad event must not wedge the queue
            self._record_error(exc)
            with self._lock:
                self._failed += 1
            return True

        try:
            self._transport.send(message)
        except Exception as exc:  # noqa: BLE001 - reported via snapshot()
            self._record_error(exc)
            with self._lock:
                self._failed += 1
            return False
        with self._lock:
            self._sent += 1
            self._error = None
        return True

    def _disconnect(self) -> None:
        """Close the transport after a failure, ignoring anything it raises."""
        try:
            self._transport.close()
        except Exception:  # noqa: BLE001 - already handling a failure
            pass
        with self._lock:
            self._connected = False

    def _record_error(self, exc: BaseException) -> None:
        """Store ``exc`` as the most recent failure.

        Args:
            exc: The exception that was caught.
        """
        with self._lock:
            self._error = f"{type(exc).__name__}: {exc}"

    def __repr__(self) -> str:
        """Return a debugging representation of the notifier's state."""
        stats = self.snapshot()
        return (
            f"{type(self).__name__}(url={self._config.url!r}, "
            f"running={self._running}, connected={stats.connected}, "
            f"sent={stats.sent}, dropped={stats.dropped})"
        )


def read_snippet_bytes(path: str, limit: int) -> bytes | None:
    """Read a snippet file for inclusion in an event, bounded by ``limit``.

    Args:
        path: The snippet's path.
        limit: Maximum bytes to read.  A file larger than this yields ``None``
            rather than a truncated WAV: half a file with a header claiming the
            whole length is worse than no file, because a consumer cannot tell.

    Returns:
        The file's bytes, or ``None`` if it is missing, too large, or unreadable.
        Never raises -- this runs on the consumer thread.
    """
    try:
        size = os.path.getsize(path)
        if size > limit:
            return None
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def build_transport(config: NotifyConfig) -> tuple[Transport, str | None]:
    """Build the real WebSocket transport, or explain why it is unavailable.

    ``websocket-client`` is imported here rather than at module scope, for the
    same reason vosk is: it is an optional extra, and a checkout without it must
    still import this module.  It is a **pure-Python** package -- a
    ``py3-none-any`` wheel -- so unlike vosk it carries no ARM64 packaging risk
    at all and can simply be installed on the deployment target.

    Args:
        config: The notification configuration.

    Returns:
        ``(transport, error)``.  ``error`` is ``None`` on success; otherwise the
        transport is a :class:`NullTransport` and ``error`` says what to install
        or what went wrong.  Returned rather than raised, because this is called
        when the user presses Start and a missing optional package must not stop
        a capture.
    """
    if not config.enabled:
        return NullTransport(), None
    try:
        import websocket  # noqa: PLC0415 - deliberately lazy; see the docstring
    except ImportError:
        return (
            NullTransport(),
            "the `websocket-client` package is required to send wake-phrase "
            "notifications; install it with `pip install .[notify]`",
        )
    return _WebSocketTransport(config, websocket), None


class _WebSocketTransport:
    """A :class:`Transport` over ``websocket-client``.

    Private because nothing should construct it directly: :func:`build_transport`
    is what knows whether the package is importable.  It holds the module it was
    given rather than importing one, so it carries no import of its own and can
    be exercised with a stub.
    """

    __slots__ = ("_config", "_module", "_socket")

    def __init__(self, config: NotifyConfig, module: Any) -> None:
        """Wrap the ``websocket`` module.

        Args:
            config: Supplies the URL, timeouts and headers.
            module: The imported ``websocket`` module, or a stand-in exposing
                ``create_connection``.
        """
        self._config: NotifyConfig = config
        self._module: Any = module
        self._socket: Any = None

    @property
    def connected(self) -> bool:
        """``True`` while a connection is open."""
        return self._socket is not None

    def connect(self) -> None:
        """Open the connection, replacing any existing one.

        Raises:
            Exception: Whatever ``websocket-client`` raises when it cannot
                connect -- surfaced unchanged, since the notifier's backoff is
                what decides how to react.
        """
        self.close()
        self._socket = self._module.create_connection(
            self._config.url,
            timeout=self._config.connect_timeout_s,
            header=[f"{name}: {value}" for name, value in self._config.headers],
        )
        # The connect timeout is generous because a handshake includes a DNS
        # lookup and a TLS negotiation; sends should give up sooner.
        self._socket.settimeout(self._config.send_timeout_s)

    def send(self, message: str) -> None:
        """Write one text frame.

        Args:
            message: The payload.

        Raises:
            ConnectionError: If there is no open connection.
            Exception: Whatever the socket raises on a failed write.
        """
        socket = self._socket
        if socket is None:
            raise ConnectionError("the WebSocket is not connected")
        socket.send(message)

    def close(self) -> None:
        """Close the connection if there is one.  Idempotent, never raises."""
        socket = self._socket
        self._socket = None
        if socket is None:
            return
        try:
            socket.close()
        except Exception:  # noqa: BLE001 - closing a dead socket is not news
            pass

    def __repr__(self) -> str:
        """Return a debugging representation of this transport."""
        return (
            f"{type(self).__name__}(url={self._config.url!r}, "
            f"connected={self.connected})"
        )
