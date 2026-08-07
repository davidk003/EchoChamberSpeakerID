"""The speaker-verification seam, mirroring :mod:`echochamber.voicegate.recognizer`.

:class:`SpeakerVerifier` is a :class:`~typing.Protocol` for the same reason
:class:`Recognizer <echochamber.voicegate.recognizer.Recognizer>` is one: it
lets the real CAM++ embedder (CPU or QNN) and a scripted test double be
interchangeable without :mod:`echochamber.voicegate.sink` importing either.
Nothing here imports :mod:`echochamber.speakerid` -- that package is wired in
only by :mod:`echochamber.ui.controller`, exactly as
:mod:`echochamber.voicegate.notify` is.

**Float32 mono samples cross this boundary, not PCM16.**  Unlike the
recogniser, which talks to Vosk's ``AcceptWaveform`` and therefore wants
16-bit PCM, CAM++'s own preprocessing (fbank extraction) works from a float
waveform in ``[-1, 1]`` -- see :func:`echochamber.speakerid.campplus.prepare_wav`.
Converting to PCM16 first would only throw bits away for no benefit to either
side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["SpeakerVerifier", "VerifyResult"]


@dataclass(frozen=True, slots=True)
class VerifyResult:
    """The outcome of comparing one clip against every enrolled speaker.

    Attributes:
        matched: ``True`` if the best-scoring enrolled speaker cleared the
            configured threshold.  ``False`` with an empty database is not a
            special case here -- it is simply the "no reference scored high
            enough" outcome, which is also what an empty database produces.
        speaker: Name of the best-scoring enrolled speaker, or ``None`` when
            the database is empty.  Set even when ``matched`` is ``False``,
            so a caller that wants to log "closest was X at 0.12" can.
        score: Cosine similarity of the best match, in ``[-1, 1]``.  ``0.0``
            when :attr:`speaker` is ``None``.
    """

    matched: bool
    speaker: str | None
    score: float = 0.0

    def __repr__(self) -> str:
        """Return a debugging representation of the verdict and its score."""
        return (
            f"{type(self).__name__}(matched={self.matched}, "
            f"speaker={self.speaker!r}, score={self.score:.3f})"
        )


@runtime_checkable
class SpeakerVerifier(Protocol):
    """Anything that can tell whether a clip was spoken by an enrolled voice."""

    def verify(self, samples: np.ndarray, sample_rate: int) -> VerifyResult:
        """Embed ``samples`` and compare it against every enrolled speaker.

        Args:
            samples: 1-D mono ``float32`` samples in ``[-1, 1]``.
            sample_rate: Sample rate of ``samples`` in Hz.

        Returns:
            The verdict.  Never raises on an ordinary "no match" -- only on a
            genuine failure (a dead worker process, a corrupt model); the
            caller treats any exception as "not verified", per
            :meth:`echochamber.voicegate.sink.VoiceGateSink._verify_speaker`.
        """
        ...

    def close(self) -> None:
        """Release whatever the verifier holds.  Must be idempotent."""
        ...
