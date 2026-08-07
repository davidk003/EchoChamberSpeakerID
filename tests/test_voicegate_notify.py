"""Tests for echochamber.voicegate.notify -- the whole notifier, no sockets.

**Nothing here opens a socket and nothing here imports ``websocket``.**  That is
the property the module was factored to have: :class:`Transport` is three
synchronous methods, so the queueing, the drop policy, the backoff, the
reconnection and the shutdown can all be driven by a hand-built object.
:class:`RecordingTransport` ships in the package and scripts its own failures
(``fail_connects`` / ``fail_sends``), which is what makes the reconnect path
deterministic rather than timing-dependent.  The one test that must reach the
real optional dependency reaches it only through
:func:`~unittest.mock.patch.dict` on ``sys.modules``, to force the *absent*
branch -- ``websocket-client`` is installed in this environment, so the
interesting case is the one that cannot happen here by accident.

**The sender runs on its own thread, so nothing below sleeps to synchronise.**
Every observation is made through :func:`wait_until`, which polls a predicate
against a deadline and returns a bool, and every assertion is on a *value*
rather than on how long something took.  The two exceptions are deliberate and
are about promptness, not about ordering: ``close()`` must return well inside
``_JOIN_TIMEOUT_S`` even with a 30 s backoff in flight, which is the direct
observable proof that the backoff waits on the condition variable instead of
sleeping.  ``reconnect_initial_s`` is 10 ms everywhere else so the backoff tests
cost nothing.

**Every notifier is closed by a fixture.**  ``make_notifier`` registers what it
builds and closes it on teardown, so a failing assertion in the middle of a test
still stops the ``voicegate-notify`` thread instead of leaving it parked in a
blocked ``send``.

Two structural points worth naming:

* ``to_payload`` is asserted by **key set**, not by value.  A ``DETECTED`` event
  must *omit* the snippet fields rather than send zeros, and
  ``payload["frames"] == 0`` would pass just as happily against the bug.
* The drop-oldest test blocks the transport's ``send`` on an event the test
  owns.  That pins the sender thread inside one send, which makes the queue's
  contents fully determined: exactly the overflow is dropped, and the survivors
  are the newest events.
"""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from typing import Any, Callable, Iterator
from unittest.mock import patch

import pytest

from echochamber.voicegate.notify import (
    DEFAULT_URL,
    EventKind,
    NotifyConfig,
    NotifyEvent,
    NotifyStats,
    NullTransport,
    RecordingTransport,
    Transport,
    WebSocketNotifier,
    _WebSocketTransport,
    build_transport,
)


# Generous: every wait below is event-driven, so these only bound failures.
TIMEOUT = 5.0

# A port nothing listens on, and which nothing here ever dials: every notifier
# below is handed a transport, so the URL is only ever validated and echoed.
URL = "ws://127.0.0.1:9/notify"

# Long enough that a wedged test fails instead of hanging, short enough that it
# fails inside the suite's patience.
BLOCK_TIMEOUT = 10.0

PHRASE = "ok google"
TEXT = "ok google turn it up"

# A recognisable byte pattern: if this shows up in a repr, the repr is dumping
# audio into whatever log it landed in.
AUDIO = b"RIFF" + b"\xde\xad\xbe\xef" * 8


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Poll ``pred`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def config(**overrides: Any) -> NotifyConfig:
    """An enabled config with a 10 ms backoff, so no test waits on one."""
    kwargs: dict[str, Any] = dict(
        enabled=True,
        url=URL,
        reconnect_initial_s=0.01,
        reconnect_max_s=0.05,
        connect_timeout_s=1.0,
        send_timeout_s=1.0,
    )
    kwargs.update(overrides)
    return NotifyConfig(**kwargs)


def detected(seq: int = 0, **overrides: Any) -> NotifyEvent:
    """A DETECTED event, which by construction carries no snippet and no audio."""
    kwargs: dict[str, Any] = dict(
        kind=EventKind.DETECTED,
        phrase=PHRASE,
        text=TEXT,
        seq=seq,
        sample_rate=16_000,
        timestamp=1_700_000_000.5,
        start_frame=48_000,
    )
    kwargs.update(overrides)
    return NotifyEvent(**kwargs)


def snippet(seq: int = 0, **overrides: Any) -> NotifyEvent:
    """A SNIPPET event with every snippet field populated."""
    kwargs: dict[str, Any] = dict(
        kind=EventKind.SNIPPET,
        phrase=PHRASE,
        text=TEXT,
        seq=seq,
        sample_rate=16_000,
        timestamp=1_700_000_003.25,
        start_frame=40_000,
        path="/snippets/0000_ok-google.wav",
        frames=24_000,
        duration_s=1.5,
        truncated=False,
    )
    kwargs.update(overrides)
    return NotifyEvent(**kwargs)


def seqs_of(messages: list[str]) -> list[int]:
    """The ``seq`` of every message on the wire, in arrival order."""
    return [json.loads(message)["seq"] for message in messages]


class BrokenEvent(NotifyEvent):
    """A :class:`NotifyEvent` whose serialisation always fails.

    ``NotifyEvent`` is a frozen, slotted dataclass, so neither
    :func:`unittest.mock.patch.object` nor a plain ``setattr`` can replace
    ``to_json`` on an instance -- there is nowhere to put the replacement.
    Subclassing is the one clean way in: the dataclass machinery is untouched,
    the instance is still a ``NotifyEvent`` with a real ``kind`` for
    :meth:`NotifyConfig.wants` to filter on, and only the serialisation is
    poisoned.  This stands in for the real hazard, which is a field the
    ``json`` encoder cannot handle reaching the sender thread.
    """

    def to_json(self) -> str:
        """Fail the way an unencodable payload does."""
        raise ValueError("this event cannot be serialised")


class BlockingTransport:
    """A transport whose ``send`` parks until the test releases it.

    Pinning the sender thread inside one ``send`` is what makes the drop-oldest
    test deterministic: while it is parked nothing is taken off the queue, so
    the queue's contents are exactly what the test put there.  The wait is
    bounded so a failed assertion costs a slow teardown rather than a hung
    suite.
    """

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.connects = 0
        self.closed = False

    def connect(self) -> None:
        self.connects += 1

    def send(self, message: str) -> None:
        self.entered.set()
        self.release.wait(BLOCK_TIMEOUT)
        self.messages.append(message)

    def close(self) -> None:
        self.closed = True


class FakeSocket:
    """The object ``websocket.create_connection`` returns, minus the socket."""

    def __init__(self, close_raises: BaseException | None = None) -> None:
        self.sent: list[str] = []
        self.timeouts: list[float] = []
        self.close_calls = 0
        self.close_raises = close_raises

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)

    def send(self, message: str) -> None:
        self.sent.append(message)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_raises is not None:
            raise self.close_raises


class FakeWebSocketModule:
    """A stand-in for the ``websocket`` module: one factory, fully recorded."""

    def __init__(
        self,
        sockets: list[FakeSocket] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.sockets: list[FakeSocket] = []
        self.raises = raises
        self._queued: list[FakeSocket] = list(sockets or [])

    def create_connection(self, url: str, **kwargs: Any) -> FakeSocket:
        self.calls.append((url, dict(kwargs)))
        if self.raises is not None:
            raise self.raises
        sock = self._queued.pop(0) if self._queued else FakeSocket()
        self.sockets.append(sock)
        return sock


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def make_notifier() -> Iterator[Callable[..., WebSocketNotifier]]:
    """Factory registering every notifier for guaranteed close() on teardown.

    close() is what stops the ``voicegate-notify`` thread, so a test that fails
    partway cannot leave one parked in a blocked send or a long backoff.
    """
    created: list[WebSocketNotifier] = []

    def _make(
        cfg: NotifyConfig,
        transport: Any = None,
        start: bool = True,
    ) -> WebSocketNotifier:
        notifier = WebSocketNotifier(cfg, transport)
        created.append(notifier)
        if start:
            notifier.start()
        return notifier

    yield _make

    for notifier in created:
        try:
            notifier.close()
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


# ==========================================================================
# NotifyConfig
# ==========================================================================

class TestNotifyConfigDefaults:
    """Off by default, because this one opens a network connection."""

    def test_notifications_are_disabled_by_default(self) -> None:
        """A feature that dials out must not switch itself on by being merged."""
        assert NotifyConfig().enabled is False

    def test_the_default_url_is_a_local_listener(self) -> None:
        """Nothing leaves the machine unless the user says where to."""
        assert NotifyConfig().url == DEFAULT_URL
        assert DEFAULT_URL.startswith("ws://127.0.0.1")

    def test_both_event_kinds_are_sent_by_default(self) -> None:
        """Detection and snippet answer different questions; default to both."""
        assert NotifyConfig().events == frozenset(EventKind)

    def test_audio_is_excluded_by_default(self) -> None:
        """A consumer that only wants to know *that* a phrase was heard pays nothing."""
        assert NotifyConfig().include_audio is False

    def test_a_disabled_config_is_not_validated_against_its_url(self) -> None:
        """Only an enabled config has to name somewhere reachable."""
        assert NotifyConfig(enabled=False, url="").url == ""
        assert NotifyConfig(enabled=False, url="not-a-url").enabled is False

    def test_a_disabled_config_may_send_no_kinds(self) -> None:
        """An empty events set is only a mistake when something is enabled."""
        assert NotifyConfig(enabled=False, events=frozenset()).events == frozenset()


class TestNotifyConfigValidation:
    """Every rejection path, each with the message a user would read."""

    @pytest.mark.parametrize("queue_max", [0, -1, -32])
    def test_a_queue_below_one_is_rejected(self, queue_max: int) -> None:
        """A zero-length buffer would drop every event on the way in."""
        with pytest.raises(ValueError, match=r"queue_max must be >= 1"):
            NotifyConfig(queue_max=queue_max)

    @pytest.mark.parametrize("max_audio_bytes", [0, -1, -4096])
    def test_a_max_audio_bytes_below_one_is_rejected(
        self, max_audio_bytes: int
    ) -> None:
        """A ceiling of zero means include_audio could never include anything."""
        with pytest.raises(ValueError, match=r"max_audio_bytes must be >= 1"):
            NotifyConfig(max_audio_bytes=max_audio_bytes)

    @pytest.mark.parametrize(
        "name",
        [
            "connect_timeout_s",
            "send_timeout_s",
            "reconnect_initial_s",
            "reconnect_max_s",
        ],
    )
    @pytest.mark.parametrize("value", [0.0, -1.0, -0.001])
    def test_a_non_positive_timeout_or_delay_is_rejected(
        self, name: str, value: float
    ) -> None:
        """A zero timeout is not 'no timeout', it is 'fail immediately, forever'."""
        with pytest.raises(ValueError, match=rf"{name} must be > 0"):
            NotifyConfig(**{name: value})

    def test_an_initial_delay_above_the_ceiling_is_rejected(self) -> None:
        """Backoff doubles up to the ceiling; starting above it is incoherent."""
        with pytest.raises(
            ValueError, match=r"reconnect_initial_s \(10\.0\) must be <= "
        ):
            NotifyConfig(reconnect_initial_s=10.0, reconnect_max_s=5.0)

    def test_equal_initial_and_max_delays_are_allowed(self) -> None:
        """A flat, non-doubling backoff is a legitimate choice."""
        cfg = NotifyConfig(reconnect_initial_s=2.0, reconnect_max_s=2.0)
        assert cfg.reconnect_initial_s == cfg.reconnect_max_s == 2.0

    def test_enabled_with_an_empty_url_is_rejected(self) -> None:
        """'On, but nowhere' is always a misconfiguration."""
        with pytest.raises(ValueError, match=r"url must not be empty"):
            NotifyConfig(enabled=True, url="")

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8765",
            "https://example.test/ws",
            "127.0.0.1:8765",
            "example.test",
            "/notify",
        ],
    )
    def test_enabled_with_a_non_websocket_url_is_rejected(self, url: str) -> None:
        """An http:// endpoint is the misconfiguration this catches at the door."""
        with pytest.raises(ValueError, match=r"url must start with ws:// or wss://"):
            NotifyConfig(enabled=True, url=url)

    @pytest.mark.parametrize("url", ["ws://host:1/x", "wss://host/x"])
    def test_both_websocket_schemes_are_accepted(self, url: str) -> None:
        """ws:// and wss:// are the only two the transport can dial."""
        assert NotifyConfig(enabled=True, url=url).url == url

    def test_enabled_with_no_event_kinds_is_rejected(self) -> None:
        """An enabled notifier that can never send anything is a mistake."""
        with pytest.raises(ValueError, match=r"at least one EventKind"):
            NotifyConfig(enabled=True, url=URL, events=frozenset())

    @pytest.mark.parametrize(
        "events",
        [
            {EventKind.DETECTED},
            [EventKind.DETECTED],
            (EventKind.DETECTED,),
            EventKind.DETECTED,
        ],
    )
    def test_a_non_frozenset_events_is_a_type_error(self, events: Any) -> None:
        """The config is frozen and hashable; a mutable set would break that."""
        with pytest.raises(TypeError, match=r"events must be a frozenset"):
            NotifyConfig(events=events)


class TestWants:
    """wants() is the single place the filtering decision is made."""

    @pytest.mark.parametrize("kind", list(EventKind))
    def test_a_disabled_config_wants_nothing(self, kind: EventKind) -> None:
        """Disabled beats configured: the events set is not even consulted."""
        cfg = NotifyConfig(enabled=False, events=frozenset(EventKind))
        assert cfg.wants(kind) is False

    def test_an_enabled_config_wants_the_kinds_it_names(self) -> None:
        """Consumers that only want one of the two say so here."""
        cfg = config(events=frozenset({EventKind.SNIPPET}))
        assert cfg.wants(EventKind.SNIPPET) is True
        assert cfg.wants(EventKind.DETECTED) is False

    def test_an_enabled_config_with_both_kinds_wants_both(self) -> None:
        assert all(config().wants(kind) for kind in EventKind)


# ==========================================================================
# NotifyEvent
# ==========================================================================

DETECTED_KEYS = {"type", "phrase", "text", "seq", "sample_rate", "timestamp"}
SNIPPET_ONLY_KEYS = {"path", "start_frame", "frames", "duration_s", "truncated"}


class TestPayload:
    """The wire format; a missing key is a deliberate signal, not an omission."""

    def test_a_detected_payload_carries_the_shared_fields(self) -> None:
        """Everything a listener needs to react immediately, and nothing else."""
        payload = detected(seq=7).to_payload()
        assert payload == {
            "type": "detected",
            "phrase": PHRASE,
            "text": TEXT,
            "seq": 7,
            "sample_rate": 16_000,
            "timestamp": 1_700_000_000.5,
        }

    @pytest.mark.parametrize("key", sorted(SNIPPET_ONLY_KEYS))
    def test_a_detected_payload_omits_every_snippet_key(self, key: str) -> None:
        """Absent, not zero: a consumer cannot tell a real 0.0 from a placeholder.

        Asserting ``payload[key] == 0`` would pass against exactly the bug this
        rules out, which is why the assertion is on the key's absence.
        """
        payload = detected(start_frame=48_000, frames=24_000, duration_s=1.5).to_payload()
        assert key not in payload, (
            f"a DETECTED payload must omit {key!r}; the snippet does not exist yet"
        )

    def test_a_detected_payload_omits_audio_even_when_audio_is_set(self) -> None:
        """Audio is a snippet concept; a detection never carries it."""
        assert "audio" not in detected(audio=AUDIO).to_payload()

    def test_a_snippet_payload_includes_every_snippet_key(self) -> None:
        """The snippet is what the SNIPPET event exists to describe."""
        payload = snippet(seq=3).to_payload()
        assert set(payload) == DETECTED_KEYS | SNIPPET_ONLY_KEYS
        assert payload["type"] == "snippet"
        assert payload["path"] == "/snippets/0000_ok-google.wav"
        assert payload["start_frame"] == 40_000
        assert payload["frames"] == 24_000
        assert payload["duration_s"] == 1.5
        assert payload["truncated"] is False
        assert payload["seq"] == 3

    def test_a_truncated_snippet_says_so(self) -> None:
        assert snippet(truncated=True).to_payload()["truncated"] is True

    def test_audio_is_absent_when_there_is_none(self) -> None:
        """No key at all, rather than a null: 'not included' is unambiguous."""
        assert snippet(audio=None).to_payload().keys().isdisjoint({"audio"})

    def test_audio_is_described_and_base64_encoded(self) -> None:
        """encoding/format/bytes/data, so a consumer needs no out-of-band knowledge."""
        audio = snippet(audio=AUDIO).to_payload()["audio"]
        assert audio["encoding"] == "base64"
        assert audio["format"] == "wav"
        assert audio["bytes"] == len(AUDIO)
        assert audio["data"] == base64.b64encode(AUDIO).decode("ascii")

    def test_the_base64_round_trips_to_the_original_bytes(self) -> None:
        """The headline audio assertion: what arrives is byte-for-byte the file."""
        audio = snippet(audio=AUDIO).to_payload()["audio"]
        assert base64.b64decode(audio["data"]) == AUDIO

    def test_an_empty_audio_payload_is_still_included(self) -> None:
        """b'' is not None: an empty file was read, and that is worth reporting."""
        audio = snippet(audio=b"").to_payload()["audio"]
        assert audio["bytes"] == 0
        assert base64.b64decode(audio["data"]) == b""

    @pytest.mark.parametrize(
        ("duration_s", "expected"),
        [
            (1.23456789, 1.2346),
            (0.00004999, 0.0),
            (1.5, 1.5),
            (0.0, 0.0),
            (24_000 / 16_000, 1.5),
            (1.00005, 1.0001),
        ],
    )
    def test_duration_is_rounded_to_four_places(
        self, duration_s: float, expected: float
    ) -> None:
        """A float rate division produces 17 digits nobody needs on the wire."""
        assert snippet(duration_s=duration_s).to_payload()["duration_s"] == expected


class TestToJson:
    """to_json is to_payload plus json.dumps, and must stay that way."""

    def test_json_parses_back_to_the_payload(self) -> None:
        """The serialisation adds nothing and loses nothing."""
        for event in (detected(seq=1), snippet(seq=2), snippet(audio=AUDIO)):
            assert json.loads(event.to_json()) == event.to_payload()

    def test_json_is_compact(self) -> None:
        """These go out one per frame; whitespace is pure overhead."""
        text = detected().to_json()
        assert ", " not in text and '": ' not in text

    def test_the_json_is_a_single_line(self) -> None:
        """One text frame per event; a newline would suggest otherwise."""
        assert "\n" not in snippet(audio=AUDIO).to_json()


class TestEventRepr:
    """The repr goes into logs, and logs are not where audio belongs."""

    def test_the_repr_never_contains_the_audio_bytes(self) -> None:
        """A 190 KB snippet in a log line is a bug in both size and privacy."""
        text = repr(snippet(audio=AUDIO))
        assert "RIFF" not in text, f"the repr dumped the audio: {text}"
        assert "\\xde" not in text and "\xde" not in text
        assert repr(AUDIO) not in text

    def test_the_repr_reports_the_audio_size_instead(self) -> None:
        """Enough to debug 'why is nothing being sent', not enough to leak."""
        assert f"audio={len(AUDIO)}B" in repr(snippet(audio=AUDIO))

    def test_the_repr_says_none_when_there_is_no_audio(self) -> None:
        assert "audio=none" in repr(detected())

    def test_the_repr_names_the_kind_the_phrase_and_the_seq(self) -> None:
        text = repr(snippet(seq=4))
        assert "NotifyEvent(" in text
        assert "kind=snippet" in text
        assert repr(PHRASE) in text
        assert "seq=4" in text


# ==========================================================================
# the shipped transports
# ==========================================================================

class TestTransportProtocol:
    """Transport is runtime_checkable so a stub can be validated at the seam."""

    @pytest.mark.parametrize(
        "transport", [NullTransport(), RecordingTransport(), BlockingTransport()]
    )
    def test_the_shipped_and_stub_transports_satisfy_it(
        self, transport: Any
    ) -> None:
        """connect/send/close is the whole contract."""
        assert isinstance(transport, Transport)

    def test_a_bare_object_does_not_satisfy_it(self) -> None:
        class NotATransport:
            pass

        assert not isinstance(NotATransport(), Transport)


class TestNullTransport:
    """The inert default: everything succeeds, nothing happens."""

    def test_it_accepts_everything_and_keeps_nothing(self) -> None:
        transport = NullTransport()
        transport.connect()
        transport.send("anything at all")
        assert transport.closed is False

    def test_close_is_idempotent(self) -> None:
        transport = NullTransport()
        transport.close()
        transport.close()
        assert transport.closed is True

    def test_the_repr_reports_the_closed_flag(self) -> None:
        assert "NullTransport(closed=False)" == repr(NullTransport())


class TestRecordingTransport:
    """Records in order and fails on a script -- the reconnect tests depend on it."""

    def test_it_records_messages_in_order(self) -> None:
        transport = RecordingTransport()
        for text in ("first", "second", "third"):
            transport.send(text)
        assert transport.messages == ["first", "second", "third"]

    def test_messages_is_a_copy(self) -> None:
        """A caller mutating the returned list must not corrupt the record."""
        transport = RecordingTransport()
        transport.send("kept")
        transport.messages.append("not kept")
        assert transport.messages == ["kept"]

    def test_it_counts_connects_including_the_failures(self) -> None:
        """The count is attempts, which is what a backoff test needs to see."""
        transport = RecordingTransport(fail_connects=2)
        for _ in range(2):
            with pytest.raises(ConnectionError, match="scripted connect failure"):
                transport.connect()
        transport.connect()
        assert transport.connects == 3

    def test_fail_connects_raises_exactly_the_scripted_number_of_times(self) -> None:
        transport = RecordingTransport(fail_connects=1)
        with pytest.raises(ConnectionError):
            transport.connect()
        transport.connect()          # must not raise
        transport.connect()
        assert transport.connects == 3

    def test_fail_sends_raises_then_succeeds_and_records(self) -> None:
        """A failed send records nothing; the retry after it does."""
        transport = RecordingTransport(fail_sends=2)
        for _ in range(2):
            with pytest.raises(ConnectionError, match="scripted send failure"):
                transport.send("lost")
        transport.send("kept")
        assert transport.messages == ["kept"], (
            "a send that raised must not have been recorded"
        )

    def test_zero_failures_is_the_default(self) -> None:
        transport = RecordingTransport()
        transport.connect()
        transport.send("x")
        assert transport.messages == ["x"] and transport.connects == 1

    def test_close_is_idempotent(self) -> None:
        transport = RecordingTransport()
        transport.close()
        transport.close()
        assert transport.closed is True

    def test_the_repr_summarises_what_was_recorded(self) -> None:
        transport = RecordingTransport()
        transport.connect()
        transport.send("x")
        text = repr(transport)
        assert "RecordingTransport(" in text
        assert "messages=1" in text and "connects=1" in text and "closed=False" in text


# ==========================================================================
# WebSocketNotifier -- queueing
# ==========================================================================

class TestNotifyFiltering:
    """notify() is called on the audio path: it decides fast and returns."""

    def test_a_disabled_notifier_queues_nothing(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Wiring the notifier in while disabled must cost nothing at all."""
        transport = RecordingTransport()
        notifier = make_notifier(
            NotifyConfig(enabled=False), transport, start=False
        )

        assert notifier.notify(detected()) is False
        assert notifier.notify(snippet()) is False
        assert notifier.queued == 0
        assert notifier.snapshot().dropped == 0

    def test_a_filtered_kind_returns_false_and_queues_nothing(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The events set is honoured before anything reaches the buffer."""
        notifier = make_notifier(
            config(events=frozenset({EventKind.SNIPPET})),
            RecordingTransport(),
            start=False,
        )

        assert notifier.notify(detected()) is False
        assert notifier.queued == 0

        assert notifier.notify(snippet()) is True
        assert notifier.queued == 1

    def test_notify_after_close_returns_false(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The pipeline may still be draining chunks when the notifier is shut."""
        notifier = make_notifier(config(), RecordingTransport())
        notifier.close()

        assert notifier.notify(detected()) is False
        assert notifier.queued == 0

    def test_notify_never_raises_and_never_connects(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """It runs on the consumer thread; connecting there is the whole hazard."""
        transport = RecordingTransport()
        notifier = make_notifier(config(), transport, start=False)

        for k in range(5):
            assert notifier.notify(detected(seq=k)) is True

        assert transport.connects == 0, (
            "notify() must not touch the transport; the sender thread does that"
        )
        assert notifier.queued == 5


# ==========================================================================
# WebSocketNotifier -- sending
# ==========================================================================

class TestSending:
    """The sender thread connects once and drains, in order."""

    def test_two_events_arrive_in_order_with_their_payloads(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The headline path: queue two, get two, unchanged and in order."""
        transport = RecordingTransport()
        notifier = make_notifier(config(), transport)

        assert notifier.notify(detected(seq=0)) is True
        assert notifier.notify(snippet(seq=0)) is True

        assert wait_until(lambda: notifier.snapshot().sent == 2), (
            f"only {notifier.snapshot().sent} of 2 events were sent: "
            f"{notifier.snapshot()!r}"
        )

        messages = transport.messages
        assert len(messages) == 2
        assert json.loads(messages[0]) == detected(seq=0).to_payload()
        assert json.loads(messages[1]) == snippet(seq=0).to_payload()

        stats = notifier.snapshot()
        assert stats.sent == 2
        assert stats.dropped == 0 and stats.failed == 0
        assert stats.connected is True
        assert stats.error is None

    def test_one_connection_serves_many_events(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Reconnecting per event would be a denial-of-service on the listener."""
        transport = RecordingTransport()
        notifier = make_notifier(config(queue_max=64), transport)

        for k in range(10):
            notifier.notify(detected(seq=k))

        assert wait_until(lambda: notifier.snapshot().sent == 10)
        assert seqs_of(transport.messages) == list(range(10))
        assert transport.connects == 1, (
            f"10 events must share one connection, got {transport.connects}"
        )

    def test_events_queued_before_start_are_still_sent(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The buffer exists exactly so notify() never has to wait for a thread."""
        transport = RecordingTransport()
        notifier = make_notifier(config(), transport, start=False)

        notifier.notify(detected(seq=0))
        notifier.notify(detected(seq=1))
        assert transport.messages == []

        notifier.start()
        assert wait_until(lambda: notifier.snapshot().sent == 2)
        assert seqs_of(transport.messages) == [0, 1]


class TestReconnection:
    """A socket that is down must cost a delay, never an event."""

    def test_an_event_survives_two_failed_connects(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The event is requeued, not dropped: down-time is when it matters most."""
        transport = RecordingTransport(fail_connects=2)
        notifier = make_notifier(config(), transport)

        notifier.notify(detected(seq=5))

        assert wait_until(lambda: notifier.snapshot().sent == 1), (
            f"the event was never delivered: {notifier.snapshot()!r}"
        )
        assert seqs_of(transport.messages) == [5]
        assert transport.connects >= 3, (
            f"two scripted failures plus one success is at least three connect "
            f"attempts, got {transport.connects}"
        )

        stats = notifier.snapshot()
        assert stats.dropped == 0, "a down socket must not lose the event"
        assert stats.connects == 1, "only the successful connect is counted"
        assert stats.connected is True

    def test_a_failed_send_requeues_and_delivers_exactly_once(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The key no-loss/no-duplicate property of the retry path."""
        transport = RecordingTransport(fail_sends=1)
        notifier = make_notifier(config(), transport)

        notifier.notify(snippet(seq=2))

        assert wait_until(lambda: notifier.snapshot().sent == 1)
        assert len(transport.messages) == 1, (
            f"the event must arrive exactly once, got {transport.messages}"
        )
        assert json.loads(transport.messages[0]) == snippet(seq=2).to_payload()

        stats = notifier.snapshot()
        assert stats.failed == 1, f"one send raised, so failed == 1: {stats!r}"
        assert stats.dropped == 0, "a send that raises must not lose the event"
        assert transport.connects >= 2, (
            "a failed send means the connection is gone, so the retry reconnects"
        )

    def test_the_error_is_recorded_then_cleared_by_a_success(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """`error` is how the GUI tells a quiet notifier from a broken one."""
        transport = RecordingTransport(fail_connects=1)
        notifier = make_notifier(config(), transport)

        notifier.notify(detected())
        assert wait_until(lambda: notifier.snapshot().sent == 1)
        assert notifier.snapshot().error is None, (
            "a successful send must clear the recorded failure"
        )

    def test_an_error_survives_for_inspection_while_the_socket_is_down(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """A notifier that cannot connect explains itself rather than going quiet."""
        transport = RecordingTransport(fail_connects=1_000_000)
        notifier = make_notifier(config(), transport)

        notifier.notify(detected())

        assert wait_until(lambda: notifier.snapshot().error is not None)
        assert "scripted connect failure" in str(notifier.snapshot().error)
        assert notifier.snapshot().connected is False


class TestBadEvents:
    """A single unencodable event must not wedge the queue behind it."""

    def test_an_event_whose_serialisation_raises_is_counted_and_skipped(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Counted in `failed`, never retried, and the good event behind it flows.

        Retrying it would be pointless -- an event that cannot be encoded will
        not encode any better on a new socket -- so the contract is that it is
        dropped from the queue, reported, and the sender carries straight on.
        See :class:`BrokenEvent` for why the failure is injected by subclassing.
        """
        transport = RecordingTransport()
        notifier = make_notifier(config(), transport)

        assert notifier.notify(BrokenEvent(kind=EventKind.DETECTED, phrase=PHRASE))
        assert notifier.notify(detected(seq=99))

        assert wait_until(lambda: notifier.snapshot().sent == 1), (
            f"the good event behind the broken one never arrived: "
            f"{notifier.snapshot()!r}"
        )
        assert seqs_of(transport.messages) == [99], (
            "only the encodable event may reach the wire"
        )

        stats = notifier.snapshot()
        assert stats.failed == 1, f"the unencodable event must be counted: {stats!r}"
        assert stats.queued == 0, "the bad event must not be left in the buffer"
        assert transport.connects == 1, (
            "a serialisation failure is not a connection failure and must not "
            "cause a reconnect"
        )


# ==========================================================================
# WebSocketNotifier -- back-pressure
# ==========================================================================

class TestDropOldest:
    """Freshness beats completeness, and the audio path never blocks."""

    def test_the_oldest_events_are_dropped_and_the_newest_survive(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The key back-pressure property, made deterministic by a blocked send.

        While the sender is parked inside ``send`` nothing is taken off the
        queue, so the buffer's contents are exactly what this test put there:
        a ``queue_max`` of 2 holding events 1..4 must discard 1 and 2 and keep
        3 and 4.  Event 0 was already in flight when the block began, which is
        why three messages arrive and not two.
        """
        transport = BlockingTransport()
        notifier = make_notifier(config(queue_max=2), transport)

        try:
            notifier.notify(detected(seq=0))
            assert transport.entered.wait(TIMEOUT), (
                "test setup: the sender must be parked inside send()"
            )

            for k in (1, 2, 3, 4):
                assert notifier.notify(detected(seq=k)) is True, (
                    "notify() reports queueing, not eventual delivery"
                )

            assert notifier.queued == 2, (
                f"the buffer must stay bounded at queue_max=2, got "
                f"{notifier.queued}"
            )
            assert notifier.snapshot().dropped == 2, (
                f"4 events into a 2-deep buffer means exactly 2 drops, got "
                f"{notifier.snapshot().dropped}"
            )
        finally:
            transport.release.set()

        assert wait_until(lambda: notifier.snapshot().sent == 3)
        assert seqs_of(transport.messages) == [0, 3, 4], (
            "DROP_OLDEST must discard the OLDEST queued events and keep the "
            f"newest; got {seqs_of(transport.messages)}"
        )
        assert notifier.snapshot().dropped == 2

    def test_notify_never_blocks_even_when_the_buffer_is_full(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Blocking here would stall the consumer thread and lose audio."""
        transport = BlockingTransport()
        notifier = make_notifier(config(queue_max=2), transport)

        try:
            notifier.notify(detected(seq=0))
            assert transport.entered.wait(TIMEOUT), "test setup"

            t0 = time.monotonic()
            for k in range(1, 501):
                notifier.notify(detected(seq=k))
            elapsed = time.monotonic() - t0
        finally:
            transport.release.set()

        assert elapsed < 2.0, (
            f"500 notify() calls into a full buffer took {elapsed:.2f}s -- "
            "notify() must never block the audio path"
        )
        assert notifier.snapshot().dropped == 498
        assert notifier.queued == 2

    def test_a_single_slot_buffer_keeps_only_the_newest(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """queue_max=1 is legal and degenerates to 'the latest event only'."""
        transport = BlockingTransport()
        notifier = make_notifier(config(queue_max=1), transport)

        try:
            notifier.notify(detected(seq=0))
            assert transport.entered.wait(TIMEOUT), "test setup"
            for k in (1, 2, 3):
                notifier.notify(detected(seq=k))
            assert notifier.queued == 1
            assert notifier.snapshot().dropped == 2
        finally:
            transport.release.set()

        assert wait_until(lambda: notifier.snapshot().sent == 2)
        assert seqs_of(transport.messages) == [0, 3]


# ==========================================================================
# WebSocketNotifier -- lifecycle
# ==========================================================================

class TestLifecycle:
    """Single-use, and shutdown is bounded no matter what the socket is doing."""

    def test_a_fresh_notifier_is_neither_running_nor_closed(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        notifier = make_notifier(config(), RecordingTransport(), start=False)
        assert notifier.running is False
        assert notifier.closed is False
        assert notifier.queued == 0

    def test_start_flips_running(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        notifier = make_notifier(config(), RecordingTransport())
        assert notifier.running is True

    def test_starting_twice_raises(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """A second sender thread over one buffer would interleave the frames."""
        notifier = make_notifier(config(), RecordingTransport())
        with pytest.raises(RuntimeError, match="already been started"):
            notifier.start()

    def test_starting_after_close_raises(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Single-use, matching the pipeline's own lifecycle."""
        notifier = make_notifier(config(), RecordingTransport())
        notifier.close()

        with pytest.raises(RuntimeError, match="closed"):
            notifier.start()

    def test_close_closes_the_transport_and_clears_connected(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """A leaked socket per capture is a leak across a long session."""
        transport = RecordingTransport()
        notifier = make_notifier(config(), transport)
        notifier.notify(detected())
        assert wait_until(lambda: notifier.snapshot().sent == 1)

        notifier.close()

        assert transport.closed is True
        assert notifier.closed is True
        assert notifier.running is False
        assert notifier.snapshot().connected is False

    def test_close_is_idempotent_and_never_raises(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The GUI closes it, and a failed start may have closed it already."""
        notifier = make_notifier(config(), RecordingTransport())
        assert notifier.close() is None
        assert notifier.close() is None
        assert notifier.close() is None
        assert notifier.closed is True

    def test_close_before_start_is_harmless(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """There is no thread to join, and nothing to complain about."""
        transport = RecordingTransport()
        notifier = make_notifier(config(), transport, start=False)

        notifier.close()

        assert notifier.closed is True
        assert transport.closed is True

    def test_close_returns_promptly_during_a_long_backoff(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The direct proof that the backoff waits on the condition variable.

        With a 30 s delay in flight, a sender that called ``time.sleep`` would
        make ``close()`` block for the full ``_JOIN_TIMEOUT_S`` of 2 s before
        giving up on the join.  Waiting on the condition means ``close()``'s
        ``notify_all`` wakes it at once, so this returns in milliseconds.
        """
        transport = RecordingTransport(fail_connects=1_000_000)
        notifier = make_notifier(
            config(reconnect_initial_s=30.0, reconnect_max_s=30.0), transport
        )

        notifier.notify(detected())
        assert wait_until(lambda: transport.connects >= 1), (
            "test setup: the sender must have failed a connect and be backing off"
        )

        t0 = time.monotonic()
        notifier.close()
        elapsed = time.monotonic() - t0

        assert elapsed < 1.5, (
            f"close() took {elapsed:.2f}s during a 30 s backoff -- the sender is "
            "sleeping rather than waiting on the condition variable"
        )
        assert transport.closed is True

    def test_close_stops_the_sender_thread(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """A leaked daemon thread per capture is a slow leak in a long session."""
        before = {t.name for t in threading.enumerate()}
        notifier = make_notifier(config(), RecordingTransport())
        notifier.notify(detected())
        assert wait_until(lambda: notifier.snapshot().sent == 1)

        notifier.close()

        assert wait_until(
            lambda: not {
                t.name
                for t in threading.enumerate()
                if t.name == "voicegate-notify"
            }
            - before
        ), "close() must join the voicegate-notify thread"

    def test_a_transport_whose_close_raises_does_not_break_close(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Shutdown must complete even when the socket will not cooperate."""

        class RudeTransport:
            def __init__(self) -> None:
                self.close_calls = 0

            def connect(self) -> None:
                pass

            def send(self, message: str) -> None:
                pass

            def close(self) -> None:
                self.close_calls += 1
                raise OSError("the socket refused to close")

        transport = RudeTransport()
        notifier = make_notifier(config(), transport, start=False)

        assert notifier.close() is None
        assert transport.close_calls == 1
        assert notifier.closed is True
        assert "OSError" in str(notifier.snapshot().error)


# ==========================================================================
# snapshot() and repr
# ==========================================================================

class TestSnapshot:
    """snapshot() is the GUI's coherent read while the sender thread runs."""

    def test_a_fresh_notifier_snapshots_all_defaults(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The defaults of NotifyStats are what an untouched notifier reports."""
        notifier = make_notifier(config(), RecordingTransport(), start=False)
        assert notifier.snapshot() == NotifyStats()

    def test_snapshot_returns_a_notifystats(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        notifier = make_notifier(config(), RecordingTransport(), start=False)
        assert isinstance(notifier.snapshot(), NotifyStats)

    def test_queued_agrees_with_the_property_and_the_buffer(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """queued is the backlog, and the two ways of reading it must agree."""
        notifier = make_notifier(config(queue_max=8), RecordingTransport(), start=False)

        for k in range(3):
            notifier.notify(detected(seq=k))

        assert notifier.queued == 3
        assert notifier.snapshot().queued == notifier.queued

    def test_every_counter_agrees_with_the_snapshot_after_a_real_run(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Two snapshots of an idle notifier must be identical, field for field."""
        transport = RecordingTransport(fail_sends=1)
        notifier = make_notifier(config(), transport)

        notifier.notify(detected(seq=0))
        notifier.notify(snippet(seq=0))
        assert wait_until(lambda: notifier.snapshot().sent == 2)

        first = notifier.snapshot()
        second = notifier.snapshot()
        assert first == second, "an idle notifier's counters must not drift"
        assert first.queued == notifier.queued == 0
        assert first.sent == 2 and first.failed == 1 and first.dropped == 0
        assert first.connects == transport.connects
        assert first.connected is True

    def test_a_snapshot_is_detached_from_later_activity(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """The GUI holds one while the sender thread keeps counting."""
        transport = RecordingTransport()
        notifier = make_notifier(config(queue_max=16), transport)

        notifier.notify(detected(seq=0))
        assert wait_until(lambda: notifier.snapshot().sent == 1)
        early = notifier.snapshot()

        for k in range(1, 5):
            notifier.notify(detected(seq=k))
        assert wait_until(lambda: notifier.snapshot().sent == 5)

        assert early.sent == 1, "the held snapshot must not have moved"
        assert notifier.snapshot().sent == 5

    def test_the_stats_repr_names_the_counters_that_matter(self) -> None:
        """The repr goes into logs when nothing is arriving at the listener."""
        text = repr(NotifyStats(queued=1, sent=2, dropped=3, failed=4, connected=True))
        assert "NotifyStats(" in text
        assert "queued=1" in text and "sent=2" in text
        assert "dropped=3" in text and "failed=4" in text and "connected=True" in text


class TestNotifierRepr:
    """The repr answers 'where was it pointed and did anything get through'."""

    def test_the_repr_names_the_url_and_does_not_raise(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        notifier = make_notifier(config(), RecordingTransport(), start=False)
        text = repr(notifier)

        assert "WebSocketNotifier(" in text
        assert repr(URL) in text, f"the repr must name the endpoint: {text}"
        assert "running=False" in text
        assert "sent=0" in text and "dropped=0" in text

    def test_the_repr_tracks_the_state(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        transport = RecordingTransport()
        notifier = make_notifier(config(), transport)
        notifier.notify(detected())
        assert wait_until(lambda: notifier.snapshot().sent == 1)

        text = repr(notifier)
        assert "running=True" in text and "connected=True" in text and "sent=1" in text

    def test_config_and_transport_are_reported_back(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """Held as given, not copied: the GUI reads them back off the notifier."""
        cfg = config()
        transport = RecordingTransport()
        notifier = make_notifier(cfg, transport, start=False)

        assert notifier.config is cfg
        assert notifier.transport is transport

    def test_no_transport_means_a_null_one(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """A notifier built without a transport is inert, not broken."""
        notifier = make_notifier(config(), None, start=False)
        assert isinstance(notifier.transport, NullTransport)


# ==========================================================================
# build_transport
# ==========================================================================

class TestBuildTransport:
    """The optional dependency is resolved here, and never raised about."""

    def test_a_disabled_config_gets_a_null_transport_and_no_error(self) -> None:
        """Disabled must not even attempt the import."""
        transport, error = build_transport(NotifyConfig(enabled=False))

        assert isinstance(transport, NullTransport)
        assert error is None

    def test_a_disabled_config_with_a_nonsense_url_still_works(self) -> None:
        """Nothing is dialled, so nothing about the URL matters."""
        transport, error = build_transport(NotifyConfig(enabled=False, url=""))
        assert isinstance(transport, NullTransport) and error is None

    def test_an_enabled_config_builds_a_websocket_transport_without_connecting(
        self,
    ) -> None:
        """websocket-client is installed here; building it must open nothing."""
        transport, error = build_transport(config())

        assert error is None, f"unexpected build error: {error}"
        assert isinstance(transport, _WebSocketTransport)
        assert transport.connected is False, (
            "build_transport must not open a connection; start() does that"
        )

    def test_a_missing_websocket_client_is_reported_not_raised(self) -> None:
        """A missing optional package must not stop a capture from starting.

        ``websocket-client`` *is* installed in this environment, so the absent
        branch is forced by putting ``None`` in ``sys.modules``, which is what
        makes ``import websocket`` raise ImportError.
        """
        with patch.dict(sys.modules, {"websocket": None}):
            transport, error = build_transport(config())

        assert isinstance(transport, NullTransport), (
            "without the package the notifier must be inert, not broken"
        )
        assert error is not None
        assert "websocket-client" in error
        assert "pip install .[notify]" in error, (
            f"the error must say what to install, got {error!r}"
        )


# ==========================================================================
# _WebSocketTransport -- driven with a stub module
# ==========================================================================

class TestWebSocketTransport:
    """It holds the module it was given, so a stub drives every line of it."""

    def test_connect_passes_the_url_and_the_connect_timeout(self) -> None:
        """The connect timeout is generous; it is not the send timeout."""
        module = FakeWebSocketModule()
        transport = _WebSocketTransport(
            config(connect_timeout_s=7.5, send_timeout_s=1.25), module
        )

        transport.connect()

        assert len(module.calls) == 1
        url, kwargs = module.calls[0]
        assert url == URL
        assert kwargs["timeout"] == 7.5

    def test_connect_sets_the_send_timeout_on_the_socket(self) -> None:
        """Sends must give up sooner than a handshake is allowed to take."""
        module = FakeWebSocketModule()
        transport = _WebSocketTransport(
            config(connect_timeout_s=7.5, send_timeout_s=1.25), module
        )

        transport.connect()

        assert module.sockets[0].timeouts == [1.25], (
            "settimeout must be called once, with send_timeout_s"
        )
        assert transport.connected is True

    def test_headers_are_formatted_as_name_colon_value_strings(self) -> None:
        """websocket-client wants a list of raw header lines, not a mapping."""
        module = FakeWebSocketModule()
        cfg = config(
            headers=(("Authorization", "Bearer t0ken"), ("X-Room", "studio 1"))
        )
        _WebSocketTransport(cfg, module).connect()

        _, kwargs = module.calls[0]
        assert kwargs["header"] == ["Authorization: Bearer t0ken", "X-Room: studio 1"]

    def test_no_headers_means_an_empty_list(self) -> None:
        """An empty list is what the library expects, not None."""
        module = FakeWebSocketModule()
        _WebSocketTransport(config(), module).connect()
        assert module.calls[0][1]["header"] == []

    def test_send_writes_the_message_verbatim(self) -> None:
        module = FakeWebSocketModule()
        transport = _WebSocketTransport(config(), module)
        transport.connect()

        transport.send('{"type":"detected"}')

        assert module.sockets[0].sent == ['{"type":"detected"}']

    def test_send_before_connect_raises(self) -> None:
        """A silent no-op here would look exactly like a listener that is ignoring us."""
        transport = _WebSocketTransport(config(), FakeWebSocketModule())

        with pytest.raises(ConnectionError, match="not connected"):
            transport.send("anything")

    def test_send_after_close_raises(self) -> None:
        """close() really releases the socket rather than setting a flag."""
        module = FakeWebSocketModule()
        transport = _WebSocketTransport(config(), module)
        transport.connect()
        transport.close()

        with pytest.raises(ConnectionError, match="not connected"):
            transport.send("anything")

    def test_connect_closes_the_previous_socket(self) -> None:
        """Reconnecting over a live socket would leak a file descriptor each time."""
        first, second = FakeSocket(), FakeSocket()
        module = FakeWebSocketModule([first, second])
        transport = _WebSocketTransport(config(), module)

        transport.connect()
        transport.connect()

        assert first.close_calls == 1, (
            "the socket being replaced must be closed, not abandoned"
        )
        assert second.close_calls == 0
        assert len(module.calls) == 2

    def test_close_is_idempotent(self) -> None:
        """Called on the shutdown path, and again by the notifier's own close()."""
        sock = FakeSocket()
        module = FakeWebSocketModule([sock])
        transport = _WebSocketTransport(config(), module)
        transport.connect()

        transport.close()
        transport.close()
        transport.close()

        assert sock.close_calls == 1, "only the first close may reach the socket"
        assert transport.connected is False

    def test_close_before_connect_is_harmless(self) -> None:
        transport = _WebSocketTransport(config(), FakeWebSocketModule())
        transport.close()
        assert transport.connected is False

    def test_close_swallows_a_socket_that_raises(self) -> None:
        """Closing a socket that is already dead is not news worth raising about."""
        sock = FakeSocket(close_raises=OSError("connection reset"))
        transport = _WebSocketTransport(config(), FakeWebSocketModule([sock]))
        transport.connect()

        assert transport.close() is None
        assert sock.close_calls == 1
        assert transport.connected is False

    def test_a_failing_create_connection_propagates(self) -> None:
        """The notifier's backoff is what decides how to react, so this raises."""
        boom = ConnectionRefusedError(111, "Connection refused")
        transport = _WebSocketTransport(
            config(), FakeWebSocketModule(raises=boom)
        )

        with pytest.raises(ConnectionRefusedError):
            transport.connect()
        assert transport.connected is False

    def test_it_satisfies_the_transport_protocol(self) -> None:
        assert isinstance(
            _WebSocketTransport(config(), FakeWebSocketModule()), Transport
        )

    def test_the_repr_names_the_url_and_the_connection_state(self) -> None:
        module = FakeWebSocketModule()
        transport = _WebSocketTransport(config(), module)

        assert "_WebSocketTransport(" in repr(transport)
        assert repr(URL) in repr(transport)
        assert "connected=False" in repr(transport)
        transport.connect()
        assert "connected=True" in repr(transport)

    def test_it_drives_a_whole_notifier_with_no_socket_in_sight(
        self, make_notifier: Callable[..., WebSocketNotifier]
    ) -> None:
        """End to end over the real transport class, against a stub module."""
        module = FakeWebSocketModule()
        transport = _WebSocketTransport(config(), module)
        notifier = make_notifier(config(), transport)

        notifier.notify(detected(seq=0))
        notifier.notify(snippet(seq=0))

        assert wait_until(lambda: notifier.snapshot().sent == 2)
        assert [json.loads(m)["type"] for m in module.sockets[0].sent] == [
            "detected",
            "snippet",
        ]
