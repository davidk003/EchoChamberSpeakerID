"""Turning gate events into notifications.

A deliberately thin adapter, kept out of both modules it joins.
:mod:`echochamber.voicegate.sink` should not know that notifications exist --
it is a recorder, and gating audio has nothing to do with sockets -- and
:mod:`echochamber.voicegate.notify` should not know what an
:class:`~echochamber.audio.types.AudioChunk` is.  Putting the translation in the
GUI controller instead would have made it the one piece of this feature that
could not be tested without Qt, so it lives here, where a test is three lines.

**Reading the snippet back off disk is deliberate, and it is why this is not
just a lambda.**  The gate has the audio in memory while it records, but
handing that to the notifier would mean holding a second copy of every snippet
for as long as the send queue is backed up. The file has just been closed and
is in the page cache, so reading it is cheap, bounded by
:attr:`~echochamber.voicegate.notify.NotifyConfig.max_audio_bytes`, and it
cannot fail in a way that costs anything -- a snippet that cannot be read is
sent without audio rather than not sent.
"""

from __future__ import annotations

from echochamber.voicegate.notify import (
    EventKind,
    NotifyConfig,
    NotifyEvent,
    WebSocketNotifier,
    read_snippet_bytes,
)
from echochamber.voicegate.sink import DetectionEvent, SnippetEvent

__all__ = ["NotifyRelay"]


class NotifyRelay:
    """Adapt :class:`VoiceGateSink` callbacks to :class:`WebSocketNotifier`.

    Both methods run on the pipeline's consumer thread and neither blocks: the
    notifier's own queue is what absorbs a slow or absent listener.

    Args:
        notifier: Where events go.
        sample_rate: Capture sample rate in Hz, stamped onto every event so a
            consumer receiving raw audio knows how to play it.
        config: The notification configuration; supplies ``include_audio`` and
            ``max_audio_bytes``.  Taken separately from the notifier's own copy
            so a relay can be built against a stub in a test.
    """

    __slots__ = ("_notifier", "_sample_rate", "_config")

    def __init__(
        self,
        notifier: WebSocketNotifier,
        sample_rate: int,
        config: NotifyConfig | None = None,
    ) -> None:
        """Wire a relay between a gate and a notifier.

        Args:
            notifier: The notifier to feed.
            sample_rate: Capture sample rate in Hz.
            config: Notification configuration; the notifier's own when
                ``None``.
        """
        self._notifier: WebSocketNotifier = notifier
        self._sample_rate: int = int(sample_rate)
        self._config: NotifyConfig = (
            notifier.config if config is None else config
        )

    @property
    def notifier(self) -> WebSocketNotifier:
        """The notifier this relay feeds."""
        return self._notifier

    @property
    def sample_rate(self) -> int:
        """Sample rate stamped onto every event, in Hz."""
        return self._sample_rate

    def on_detected(self, event: DetectionEvent) -> None:
        """Forward a detection, which carries no audio.

        Args:
            event: The detection the gate announced.
        """
        self._notifier.notify(
            NotifyEvent(
                kind=EventKind.DETECTED,
                phrase=event.phrase,
                text=event.text,
                seq=event.seq,
                sample_rate=self._sample_rate,
                timestamp=event.timestamp,
                start_frame=event.start_frame,
                speaker=event.speaker,
                speaker_score=event.speaker_score,
            )
        )

    def on_snippet(self, event: SnippetEvent) -> None:
        """Forward a completed snippet, with its audio when configured.

        The file is read only when the notifier would actually send it: reading
        a snippet off disk for an event that
        :meth:`~echochamber.voicegate.notify.NotifyConfig.wants` filters out
        would be pure waste on the consumer thread.

        Args:
            event: The snippet the gate finished writing.
        """
        config = self._config
        if not config.wants(EventKind.SNIPPET):
            return

        audio: bytes | None = None
        if config.include_audio and event.path:
            audio = read_snippet_bytes(event.path, config.max_audio_bytes)

        self._notifier.notify(
            NotifyEvent(
                kind=EventKind.SNIPPET,
                phrase=event.phrase,
                text=event.text,
                seq=event.seq,
                sample_rate=self._sample_rate,
                timestamp=event.timestamp,
                start_frame=event.start_frame,
                path=event.path,
                frames=event.frames,
                duration_s=event.duration_s,
                truncated=event.truncated,
                audio=audio,
            )
        )

    def __repr__(self) -> str:
        """Return a debugging representation naming the notifier."""
        return (
            f"{type(self).__name__}(sample_rate={self._sample_rate}, "
            f"notifier={self._notifier!r})"
        )
