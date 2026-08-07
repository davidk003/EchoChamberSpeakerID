"""Wake-phrase gating: record a snippet only when a phrase is actually spoken.

The ingestion pipeline emits a window every ``hop_ms`` whether anyone is
talking or not.  This package adds the opposite policy: keep **nothing** until
a configured phrase (``"ok google"``, ``"hey google"``) is recognised, then
write exactly one snippet around it.

The design point is that recognition is *pluggable and absent by default*.
:class:`~echochamber.voicegate.recognizer.NullRecognizer` never matches, so a
checkout with no Vosk installed and no model on disk still imports, still runs
the GUI, and still passes the whole test suite -- the gate simply never fires.
Everything that decides *what counts as a match* and *which audio ends up in
the file* is pure Python over sample arrays, so it is tested without Vosk at
all; see :mod:`echochamber.voicegate.matching` and
:mod:`echochamber.voicegate.snippets`.

Why the subprocess backend exists is a packaging problem, not a design
preference -- see :mod:`echochamber.voicegate.subprocess_recognizer`.
"""

from __future__ import annotations

from echochamber.voicegate.config import VoiceGateConfig
from echochamber.voicegate.notify import (
    EventKind,
    NotifyConfig,
    NotifyEvent,
    WebSocketNotifier,
)
from echochamber.voicegate.matching import (
    PhraseMatch,
    match_phrase,
    normalize,
    tokenize,
)
from echochamber.voicegate.recognizer import (
    NullRecognizer,
    Recognition,
    Recognizer,
)

__all__ = [
    "EventKind",
    "NotifyConfig",
    "NotifyEvent",
    "NullRecognizer",
    "PhraseMatch",
    "Recognition",
    "Recognizer",
    "VoiceGateConfig",
    "WebSocketNotifier",
    "match_phrase",
    "normalize",
    "tokenize",
]
