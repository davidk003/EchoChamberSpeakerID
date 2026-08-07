"""Tests for echochamber.voicegate.relay -- the adapter between gate and socket.

:class:`NotifyRelay` is three lines of translation, and every one of them is a
decision worth pinning down: which fields cross over, when the snippet file is
read off disk, and what happens when reading it fails.  The file read is the
interesting part, because it happens on the pipeline's consumer thread -- the
thread whose stalling loses audio -- so it must be skipped whenever the notifier
would not send the event anyway, and it must never raise.

**Two kinds of test live here, and they use different notifiers on purpose.**

* Most tests drive the relay against :class:`RecordingNotifier`, a stub that
  applies :meth:`NotifyConfig.wants` exactly as the real notifier does and then
  keeps the :class:`NotifyEvent` object.  No thread, no transport, no waiting:
  the assertions are on the event's *fields*, which is what the relay actually
  decides.
* Two tests wire the whole thing up for real -- a :class:`VoiceGateSink` over a
  :class:`ScriptedRecognizer`, fed hand-built overlapping windows, with its
  ``on_detected``/``on_snippet`` callbacks pointed at the relay and a real
  :class:`WebSocketNotifier` behind it over a :class:`RecordingTransport`.
  Those are the tests that prove the DETECTED and the SNIPPET for one utterance
  share a ``seq``, which is the whole contract a consumer pairs them on.  They
  poll with :func:`wait_until`; nothing sleeps, and the notifier is closed by a
  fixture so a failed assertion cannot leave its thread running.

**"The file was not read" is asserted with a sentinel, not with a shrug.**  A
nonexistent path proves the relay does not crash, but not that it never
*looked*: :func:`read_snippet_bytes` swallows OSError, so a relay that read
eagerly would pass that test too.  So the tests that care replace the module's
``read_snippet_bytes`` with one that raises, and assert it was never called.
"""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest

from echochamber.audio.types import AudioChunk
from echochamber.voicegate.config import VoiceGateConfig
from echochamber.voicegate.notify import (
    EventKind,
    NotifyConfig,
    NotifyEvent,
    RecordingTransport,
    WebSocketNotifier,
    read_snippet_bytes,
)
from echochamber.voicegate.recognizer import Recognition, ScriptedRecognizer
from echochamber.voicegate.relay import NotifyRelay
from echochamber.voicegate.sink import DetectionEvent, SnippetEvent, VoiceGateSink


SR = 16_000
WINDOW = 48_000        # 3000 ms at 16 kHz
HOP = 16_000           # 1000 ms at 16 kHz
BYTES_PER_FRAME = 2

# Generous: every wait below is event-driven, so these only bound failures.
TIMEOUT = 5.0

URL = "ws://127.0.0.1:9/notify"

PHRASE = "ok google"
PHRASE_TEXT = "ok google turn it up"

_TOTAL_FRAMES = 200_000
SIGNAL: np.ndarray = np.linspace(-0.9, 0.9, _TOTAL_FRAMES, dtype=np.float32)


class Boom(Exception):
    """Sentinel raised by a read that must never happen."""


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


def notify_config(**overrides: Any) -> NotifyConfig:
    """An enabled notification config with a 10 ms backoff."""
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


def gate_config(tmp_path: Path, **overrides: Any) -> VoiceGateConfig:
    """An enabled gate writing under ``tmp_path``, with short durations."""
    kwargs: dict[str, Any] = dict(
        enabled=True,
        phrases=(PHRASE,),
        pre_roll_ms=500,          # 8000 frames
        post_roll_ms=1000,        # 16000 frames == exactly one hop
        max_snippet_ms=10_000,
        cooldown_ms=0,
        snippet_dir=str(tmp_path / "snippets"),
    )
    kwargs.update(overrides)
    return VoiceGateConfig(**kwargs)


def detection(**overrides: Any) -> DetectionEvent:
    """A DetectionEvent the way the gate builds one."""
    kwargs: dict[str, Any] = dict(
        phrase=PHRASE,
        text=PHRASE_TEXT,
        seq=3,
        start_frame=48_000,
        timestamp=1_700_000_000.5,
        extended=False,
    )
    kwargs.update(overrides)
    return DetectionEvent(**kwargs)


def snippet_event(path: str = "/snippets/0003.wav", **overrides: Any) -> SnippetEvent:
    """A SnippetEvent the way the gate builds one."""
    kwargs: dict[str, Any] = dict(
        path=path,
        phrase=PHRASE,
        text=PHRASE_TEXT,
        seq=3,
        start_frame=40_000,
        frames=24_000,
        duration_s=1.5,
        truncated=False,
        timestamp=1_700_000_003.25,
    )
    kwargs.update(overrides)
    return SnippetEvent(**kwargs)


def write_wav(path: Path, frames: int = 400) -> bytes:
    """Write a real mono 16-bit WAV and return its exact bytes off disk."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(bytes(range(256)) * (frames * 2 // 256 + 1))
    return path.read_bytes()


def chunk(k: int) -> AudioChunk:
    """Chunk ``k`` of the standard overlapping grid: start ``k*H``, length ``W``."""
    start = k * HOP
    return AudioChunk(
        samples=SIGNAL[start : start + WINDOW],
        start_frame=start,
        seq=k,
        sample_rate=SR,
    )


def bytes_after(k: int) -> int:
    """PCM bytes the recogniser has consumed once chunk ``k`` has been fed."""
    return (WINDOW + k * HOP) * BYTES_PER_FRAME


def final(text: str = PHRASE_TEXT) -> Recognition:
    """A settled recognition carrying ``text``."""
    return Recognition(text=text, final=True)


class RecordingNotifier:
    """A stand-in for :class:`WebSocketNotifier` that keeps the events.

    Applies :meth:`NotifyConfig.wants` exactly as the real notifier does, so a
    relay test can tell "the relay declined to build the event" from "the
    notifier filtered it out" -- and so a filtered event still returns ``False``
    the way the real one would.
    """

    def __init__(self, config: NotifyConfig) -> None:
        self.config = config
        self.events: list[NotifyEvent] = []

    def notify(self, event: NotifyEvent) -> bool:
        if not self.config.wants(event.kind):
            return False
        self.events.append(event)
        return True

    @property
    def kinds(self) -> list[EventKind]:
        """Just the kinds received, in order."""
        return [event.kind for event in self.events]


def make_relay(
    config: NotifyConfig | None = None, sample_rate: int = SR
) -> tuple[NotifyRelay, RecordingNotifier]:
    """A relay over a RecordingNotifier, returning both."""
    cfg = notify_config() if config is None else config
    notifier = RecordingNotifier(cfg)
    return NotifyRelay(notifier, sample_rate), notifier  # type: ignore[arg-type]


def boom_reader(path: str, limit: int) -> bytes | None:
    """A ``read_snippet_bytes`` that must never be called."""
    raise Boom(f"the relay read {path!r} when it should not have")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def make_notifier() -> Iterator[Callable[..., WebSocketNotifier]]:
    """Factory registering every real notifier for guaranteed close() on teardown."""
    created: list[WebSocketNotifier] = []

    def _make(config: NotifyConfig, transport: Any) -> WebSocketNotifier:
        notifier = WebSocketNotifier(config, transport)
        created.append(notifier)
        notifier.start()
        return notifier

    yield _make

    for notifier in created:
        try:
            notifier.close()
        except Exception:  # pragma: no cover - teardown must not mask failures
            pass


# ==========================================================================
# construction
# ==========================================================================

class TestConstruction:
    """The relay is a translator; it holds nothing it does not need."""

    def test_it_reports_its_notifier_and_sample_rate(self) -> None:
        relay, notifier = make_relay(sample_rate=48_000)
        assert relay.notifier is notifier
        assert relay.sample_rate == 48_000

    def test_the_sample_rate_is_coerced_to_int(self) -> None:
        """A float rate from a config would reach the wire as '16000.0'."""
        relay, _ = make_relay(sample_rate=16_000.0)  # type: ignore[arg-type]
        assert relay.sample_rate == 16_000
        assert isinstance(relay.sample_rate, int)

    def test_the_config_defaults_to_the_notifiers_own(self) -> None:
        """One source of truth unless a test deliberately supplies another."""
        transport = RecordingTransport()
        cfg = notify_config(include_audio=True)
        notifier = WebSocketNotifier(cfg, transport)
        try:
            relay = NotifyRelay(notifier, SR)
            relay.on_snippet(snippet_event(path=""))
            assert notifier.queued == 1, (
                "the relay must have used the notifier's own config"
            )
        finally:
            notifier.close()

    def test_an_explicit_config_overrides_the_notifiers(self) -> None:
        """Taken separately so a relay can be built against a stub."""
        transport = RecordingTransport()
        notifier = WebSocketNotifier(notify_config(), transport)
        try:
            relay = NotifyRelay(
                notifier, SR, notify_config(events=frozenset({EventKind.DETECTED}))
            )
            relay.on_snippet(snippet_event(path=""))
            assert notifier.queued == 0, (
                "the supplied config, not the notifier's, decides what is relayed"
            )
        finally:
            notifier.close()

    def test_the_repr_names_the_rate_and_the_notifier(self) -> None:
        relay, _ = make_relay()
        text = repr(relay)
        assert "NotifyRelay(" in text
        assert "sample_rate=16000" in text
        assert "RecordingNotifier" in text


# ==========================================================================
# on_detected
# ==========================================================================

class TestOnDetected:
    """A detection is forwarded immediately and carries no audio."""

    def test_every_field_crosses_over_with_the_rate_stamped_on(self) -> None:
        """The headline translation: nothing invented, nothing lost."""
        relay, notifier = make_relay(sample_rate=48_000)

        relay.on_detected(
            DetectionEvent(
                phrase="hey google",
                text="hey google stop",
                seq=7,
                start_frame=123_456,
                timestamp=1_700_000_111.25,
            )
        )

        assert len(notifier.events) == 1
        event = notifier.events[0]
        assert event.kind is EventKind.DETECTED
        assert event.phrase == "hey google"
        assert event.text == "hey google stop"
        assert event.seq == 7
        assert event.start_frame == 123_456
        assert event.timestamp == 1_700_000_111.25
        assert event.sample_rate == 48_000, (
            "the relay stamps its own capture rate; DetectionEvent has none"
        )

    def test_a_detection_never_carries_audio_or_a_path(self) -> None:
        """The snippet does not exist yet -- there is nothing to attach."""
        relay, notifier = make_relay(notify_config(include_audio=True))

        relay.on_detected(detection())

        event = notifier.events[0]
        assert event.audio is None
        assert event.path is None
        assert event.frames == 0 and event.duration_s == 0.0
        assert event.truncated is False

    def test_the_payload_omits_every_snippet_key(self) -> None:
        """What actually goes on the wire, not just what the dataclass holds."""
        relay, notifier = make_relay()
        relay.on_detected(detection())

        payload = notifier.events[0].to_payload()
        assert payload["type"] == "detected"
        for key in ("path", "frames", "duration_s", "truncated", "start_frame"):
            assert key not in payload

    def test_an_extended_detection_is_still_forwarded(self) -> None:
        """`extended` is a gate concept; a listener sees a detection either way."""
        relay, notifier = make_relay()
        relay.on_detected(detection(seq=2, extended=True))

        assert notifier.kinds == [EventKind.DETECTED]
        assert notifier.events[0].seq == 2

    def test_a_filtered_detection_reaches_the_notifier_and_is_declined(self) -> None:
        """There is nothing to read from disk, so the relay does not pre-filter."""
        relay, notifier = make_relay(
            notify_config(events=frozenset({EventKind.SNIPPET}))
        )

        assert relay.on_detected(detection()) is None
        assert notifier.events == [], "the notifier's own filter must apply"

    def test_on_detected_returns_none(self) -> None:
        """It is a gate callback; the gate ignores whatever it returns."""
        relay, _ = make_relay()
        assert relay.on_detected(detection()) is None


# ==========================================================================
# on_snippet
# ==========================================================================

class TestOnSnippet:
    """A completed snippet, with its audio only when that was asked for."""

    def test_every_snippet_field_crosses_over(self) -> None:
        relay, notifier = make_relay()

        relay.on_snippet(
            snippet_event(
                path="/snippets/0009_ok-google.wav",
                phrase="hey google",
                text="hey google stop",
                seq=9,
                start_frame=88_000,
                frames=31_500,
                duration_s=1.96875,
                truncated=True,
                timestamp=1_700_000_222.5,
            )
        )

        event = notifier.events[0]
        assert event.kind is EventKind.SNIPPET
        assert event.path == "/snippets/0009_ok-google.wav"
        assert event.phrase == "hey google"
        assert event.text == "hey google stop"
        assert event.seq == 9
        assert event.start_frame == 88_000
        assert event.frames == 31_500
        assert event.duration_s == 1.96875
        assert event.truncated is True
        assert event.timestamp == 1_700_000_222.5
        assert event.sample_rate == SR

    def test_without_include_audio_a_missing_file_is_never_even_looked_at(
        self, tmp_path: Path
    ) -> None:
        """No exception, no audio -- and the path need not exist at all."""
        relay, notifier = make_relay(notify_config(include_audio=False))
        missing = str(tmp_path / "there-is-no-such-file.wav")
        assert not os.path.exists(missing)

        assert relay.on_snippet(snippet_event(path=missing)) is None

        assert len(notifier.events) == 1
        assert notifier.events[0].audio is None
        assert notifier.events[0].path == missing, (
            "the path is still reported even when the bytes are not"
        )

    def test_without_include_audio_the_file_is_not_read(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Proved with a sentinel: a nonexistent path alone proves nothing.

        ``read_snippet_bytes`` swallows OSError, so a relay that read eagerly
        would still look innocent against a missing file.  Replacing the reader
        with one that raises is what actually pins the skip down.
        """
        monkeypatch.setattr(
            "echochamber.voicegate.relay.read_snippet_bytes", boom_reader
        )
        real = tmp_path / "real.wav"
        write_wav(real)
        relay, notifier = make_relay(notify_config(include_audio=False))

        relay.on_snippet(snippet_event(path=str(real)))

        assert len(notifier.events) == 1
        assert notifier.events[0].audio is None

    def test_with_include_audio_the_audio_is_the_files_exact_bytes(
        self, tmp_path: Path
    ) -> None:
        """Byte-for-byte: a consumer decoding this must get the same WAV."""
        path = tmp_path / "snippet.wav"
        expected = write_wav(path, frames=500)
        relay, notifier = make_relay(notify_config(include_audio=True))

        relay.on_snippet(snippet_event(path=str(path)))

        event = notifier.events[0]
        assert event.audio == expected
        assert event.audio is not None and event.audio.startswith(b"RIFF")
        assert event.to_payload()["audio"]["bytes"] == len(expected)

    def test_a_file_over_the_ceiling_is_sent_without_audio_not_truncated(
        self, tmp_path: Path
    ) -> None:
        """Half a WAV under a header claiming the full length is worse than none."""
        path = tmp_path / "big.wav"
        expected = write_wav(path, frames=500)
        assert len(expected) > 64, "test setup: the file must exceed the ceiling"

        relay, notifier = make_relay(
            notify_config(include_audio=True, max_audio_bytes=64)
        )
        relay.on_snippet(snippet_event(path=str(path)))

        assert len(notifier.events) == 1, "an oversized snippet is still announced"
        assert notifier.events[0].audio is None, (
            "an oversized snippet must carry no audio at all, never a prefix"
        )
        assert "audio" not in notifier.events[0].to_payload()

    def test_a_file_exactly_at_the_ceiling_is_included(self, tmp_path: Path) -> None:
        """The limit is a maximum, not a strict bound."""
        path = tmp_path / "exact.bin"
        path.write_bytes(b"x" * 100)
        relay, notifier = make_relay(
            notify_config(include_audio=True, max_audio_bytes=100)
        )

        relay.on_snippet(snippet_event(path=str(path)))
        assert notifier.events[0].audio == b"x" * 100

    def test_a_missing_file_is_sent_without_audio_and_without_raising(
        self, tmp_path: Path
    ) -> None:
        """A snippet that cannot be read is sent without audio, never not sent."""
        missing = str(tmp_path / "gone.wav")
        relay, notifier = make_relay(notify_config(include_audio=True))

        assert relay.on_snippet(snippet_event(path=missing)) is None

        assert len(notifier.events) == 1
        assert notifier.events[0].audio is None
        assert notifier.events[0].path == missing

    def test_an_unreadable_file_is_sent_without_audio(self, tmp_path: Path) -> None:
        """A directory where a file was expected is an OSError on open, not a crash."""
        directory = tmp_path / "a-directory.wav"
        directory.mkdir()
        relay, notifier = make_relay(notify_config(include_audio=True))

        relay.on_snippet(snippet_event(path=str(directory)))

        assert len(notifier.events) == 1
        assert notifier.events[0].audio is None

    def test_an_empty_path_is_not_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An abandoned snippet can have no path; open('') would raise."""
        monkeypatch.setattr(
            "echochamber.voicegate.relay.read_snippet_bytes", boom_reader
        )
        relay, notifier = make_relay(notify_config(include_audio=True))

        relay.on_snippet(snippet_event(path=""))

        assert len(notifier.events) == 1
        assert notifier.events[0].audio is None

    def test_a_filtered_snippet_queues_nothing_and_reads_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The whole reason the relay checks wants() itself rather than just notifying.

        Reading a snippet off disk for an event that would then be filtered out
        is pure waste on the consumer thread, which is the thread whose stalling
        costs audio.  The sentinel reader is what proves the read is skipped.
        """
        monkeypatch.setattr(
            "echochamber.voicegate.relay.read_snippet_bytes", boom_reader
        )
        path = tmp_path / "snippet.wav"
        write_wav(path)
        relay, notifier = make_relay(
            notify_config(include_audio=True, events=frozenset({EventKind.DETECTED}))
        )

        assert relay.on_snippet(snippet_event(path=str(path))) is None

        assert notifier.events == [], "a filtered snippet must not be queued"

    def test_a_disabled_config_relays_nothing_and_reads_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Disabled notifications must cost the audio path nothing at all."""
        monkeypatch.setattr(
            "echochamber.voicegate.relay.read_snippet_bytes", boom_reader
        )
        path = tmp_path / "snippet.wav"
        write_wav(path)
        relay, notifier = make_relay(
            NotifyConfig(enabled=False, include_audio=True)
        )

        relay.on_snippet(snippet_event(path=str(path)))
        relay.on_detected(detection())

        assert notifier.events == []


# ==========================================================================
# read_snippet_bytes
# ==========================================================================

class TestReadSnippetBytes:
    """Bounded, total, and never raises -- it runs on the consumer thread."""

    def test_it_reads_a_real_file_exactly(self, tmp_path: Path) -> None:
        path = tmp_path / "snippet.wav"
        expected = write_wav(path, frames=300)

        assert read_snippet_bytes(str(path), 10 * 1024 * 1024) == expected

    def test_an_empty_file_reads_as_empty_bytes_not_none(
        self, tmp_path: Path
    ) -> None:
        """b'' and None mean different things: 'nothing in it' vs 'unavailable'."""
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")

        assert read_snippet_bytes(str(path), 1024) == b""

    def test_a_missing_file_is_none(self, tmp_path: Path) -> None:
        assert read_snippet_bytes(str(tmp_path / "nope.wav"), 1024) is None

    def test_a_directory_is_none(self, tmp_path: Path) -> None:
        """The snippet_dir itself can be handed in by a confused caller."""
        directory = tmp_path / "dir"
        directory.mkdir()
        assert read_snippet_bytes(str(directory), 10 * 1024 * 1024) is None

    def test_a_file_over_the_limit_is_none(self, tmp_path: Path) -> None:
        """None, not a prefix: a truncated WAV is indistinguishable from a whole one."""
        path = tmp_path / "big.bin"
        path.write_bytes(b"y" * 1000)

        assert read_snippet_bytes(str(path), 999) is None

    def test_a_file_exactly_at_the_limit_is_read(self, tmp_path: Path) -> None:
        path = tmp_path / "exact.bin"
        path.write_bytes(b"y" * 1000)

        assert read_snippet_bytes(str(path), 1000) == b"y" * 1000

    def test_an_empty_path_is_none(self) -> None:
        """open('') raises; the caller must get None instead."""
        assert read_snippet_bytes("", 1024) is None


# ==========================================================================
# end to end: gate -> relay -> notifier -> transport
# ==========================================================================

class TestEndToEnd:
    """The shipping arrangement, with a scripted decoder and no socket."""

    def test_one_utterance_produces_a_detection_and_a_snippet_sharing_a_seq(
        self,
        tmp_path: Path,
        make_notifier: Callable[..., WebSocketNotifier],
    ) -> None:
        """The pairing contract: two events, two kinds, one seq.

        The detection goes out the moment the phrase is heard, the snippet a
        post-roll later once the file is closed; ``seq`` is the only thing that
        joins them, which is why it is asserted rather than assumed.
        """
        transport = RecordingTransport()
        notifier = make_notifier(notify_config(), transport)
        relay = NotifyRelay(notifier, SR)

        config = gate_config(tmp_path)
        sink = VoiceGateSink(
            config,
            SR,
            recognizer=ScriptedRecognizer([(bytes_after(0), final())]),
            on_snippet=relay.on_snippet,
            on_detected=relay.on_detected,
        )
        try:
            sink.on_chunk(chunk(0))
            sink.on_chunk(chunk(1))
        finally:
            sink.close()

        assert sink.snippets_written == 1, "test setup: the gate must have fired"
        assert wait_until(lambda: notifier.snapshot().sent == 2), (
            f"expected 2 events on the wire: {notifier.snapshot()!r}"
        )

        payloads = [json.loads(message) for message in transport.messages]
        assert [p["type"] for p in payloads] == ["detected", "snippet"], (
            "the detection must go out before the snippet it belongs to"
        )
        assert payloads[0]["seq"] == payloads[1]["seq"] == 0, (
            "a DETECTED and its SNIPPET must share a seq so a consumer can pair them"
        )
        assert payloads[0]["phrase"] == payloads[1]["phrase"] == PHRASE
        assert payloads[0]["text"] == payloads[1]["text"] == PHRASE_TEXT
        assert payloads[0]["sample_rate"] == payloads[1]["sample_rate"] == SR

        assert "path" not in payloads[0], "a detection has no snippet to name"
        assert Path(payloads[1]["path"]).exists()
        assert payloads[1]["frames"] == 24_000
        assert payloads[1]["duration_s"] == pytest.approx(1.5)
        assert payloads[1]["truncated"] is False
        assert "audio" not in payloads[1], "include_audio is off by default"

        assert notifier.snapshot().dropped == 0
        assert notifier.snapshot().failed == 0

    def test_include_audio_puts_the_real_snippet_on_the_wire(
        self,
        tmp_path: Path,
        make_notifier: Callable[..., WebSocketNotifier],
    ) -> None:
        """Round trip: the base64 on the wire is the file the gate just wrote."""
        import base64

        transport = RecordingTransport()
        notifier = make_notifier(notify_config(include_audio=True), transport)
        relay = NotifyRelay(notifier, SR)

        config = gate_config(tmp_path)
        sink = VoiceGateSink(
            config,
            SR,
            recognizer=ScriptedRecognizer([(bytes_after(0), final())]),
            on_snippet=relay.on_snippet,
            on_detected=relay.on_detected,
        )
        try:
            sink.on_chunk(chunk(0))
            sink.on_chunk(chunk(1))
        finally:
            sink.close()

        assert wait_until(lambda: notifier.snapshot().sent == 2)
        payload = json.loads(transport.messages[1])
        on_disk = Path(payload["path"]).read_bytes()

        assert payload["audio"]["bytes"] == len(on_disk)
        assert base64.b64decode(payload["audio"]["data"]) == on_disk, (
            "the bytes on the wire must be the snippet file, byte for byte"
        )

    def test_a_suppressed_duplicate_produces_no_second_detection(
        self, tmp_path: Path
    ) -> None:
        """Suppression exists so one utterance is announced once, not twice.

        Small models re-report the same phrase across consecutive results.  The
        gate counts both matches but only announces the first, and this is the
        relay's side of that: exactly one DETECTED reaches the notifier.
        """
        relay, notifier = make_relay()
        config = gate_config(
            tmp_path, pre_roll_ms=0, post_roll_ms=1000, cooldown_ms=2000
        )
        sink = VoiceGateSink(
            config,
            SR,
            recognizer=ScriptedRecognizer(
                [(bytes_after(0), final()), (bytes_after(2), final())]
            ),
            on_snippet=relay.on_snippet,
            on_detected=relay.on_detected,
        )
        try:
            for k in range(4):
                sink.on_chunk(chunk(k))
        finally:
            sink.close()

        assert sink.phrases_detected == 2, "test setup: both matches were heard"
        assert sink.snippets_suppressed == 1, "test setup: the second was suppressed"

        assert notifier.kinds == [EventKind.DETECTED, EventKind.SNIPPET], (
            f"a suppressed duplicate must not be announced, got {notifier.kinds}"
        )
        assert notifier.events[0].seq == notifier.events[1].seq == 0

    def test_an_extended_snippet_announces_both_detections_and_one_snippet(
        self, tmp_path: Path
    ) -> None:
        """Extension is not suppression: both phrases were really heard."""
        relay, notifier = make_relay()
        config = gate_config(
            tmp_path, pre_roll_ms=0, post_roll_ms=2000, cooldown_ms=0
        )
        sink = VoiceGateSink(
            config,
            SR,
            recognizer=ScriptedRecognizer(
                [(bytes_after(0), final()), (bytes_after(1), final())]
            ),
            on_snippet=relay.on_snippet,
            on_detected=relay.on_detected,
        )
        try:
            for k in range(4):
                sink.on_chunk(chunk(k))
        finally:
            sink.close()

        assert notifier.kinds == [
            EventKind.DETECTED,
            EventKind.DETECTED,
            EventKind.SNIPPET,
        ]
        assert {event.seq for event in notifier.events} == {0}, (
            "an extension stays inside the snippet already open, so seq is shared"
        )

    def test_a_detection_only_config_relays_no_snippets(
        self, tmp_path: Path
    ) -> None:
        """A consumer that only wants the low-latency signal gets exactly that."""
        relay, notifier = make_relay(
            notify_config(events=frozenset({EventKind.DETECTED}))
        )
        config = gate_config(tmp_path)
        sink = VoiceGateSink(
            config,
            SR,
            recognizer=ScriptedRecognizer([(bytes_after(0), final())]),
            on_snippet=relay.on_snippet,
            on_detected=relay.on_detected,
        )
        try:
            sink.on_chunk(chunk(0))
            sink.on_chunk(chunk(1))
        finally:
            sink.close()

        assert sink.snippets_written == 1, "the gate still writes the file"
        assert notifier.kinds == [EventKind.DETECTED]

    def test_the_relay_never_raises_into_the_gate(self, tmp_path: Path) -> None:
        """on_chunk must not learn about notification failures; error stays None."""
        relay, notifier = make_relay(notify_config(include_audio=True))
        config = gate_config(tmp_path)
        sink = VoiceGateSink(
            config,
            SR,
            recognizer=ScriptedRecognizer([(bytes_after(0), final())]),
            on_snippet=relay.on_snippet,
            on_detected=relay.on_detected,
        )
        try:
            assert sink.on_chunk(chunk(0)) is None
            assert sink.on_chunk(chunk(1)) is None
        finally:
            sink.close()

        assert sink.error is None, f"the relay leaked a failure into the gate: {sink.error}"
        assert len(notifier.events) == 2
