"""Comparing a live clip's embedding against every enrolled speaker.

:class:`EnrolledSpeakerVerifier` is the thing
:mod:`echochamber.speakerid.backends` builds and hands to the gate; it
satisfies :class:`echochamber.voicegate.speaker.SpeakerVerifier` structurally,
without importing :mod:`echochamber.voicegate` -- see that module's docstring
for why the two packages stay mutually unaware of each other.

:class:`Embedder` is the seam *within* this package, even though there is
currently exactly one implementation
(:class:`~echochamber.speakerid.qnn_subprocess.QnnEmbedder`): it keeps this
module's scoring logic free of any process-management or ONNX detail, and
testable against a scripted fake with no NPU anywhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from echochamber.speakerid.enrollment import BACKEND_QNN
from echochamber.voicegate.speaker import VerifyResult

__all__ = ["Embedder", "EnrolledSpeakerVerifier"]


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns mono float samples into a speaker embedding."""

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a ``(192,)`` L2-normalized embedding for ``samples``.

        Args:
            samples: 1-D mono ``float32`` samples in ``[-1, 1]``.
            sample_rate: Sample rate of ``samples`` in Hz.

        Raises:
            Exception: On a malformed clip (too short, non-finite) or a
                backend failure.  The caller -- :class:`EnrolledSpeakerVerifier`
                -- treats any exception as "not verified".
        """
        ...

    def close(self) -> None:
        """Release whatever the embedder holds.  Must be idempotent."""
        ...


class EnrolledSpeakerVerifier:
    """Compare one clip's embedding against every enrolled reference.

    Ports the scoring in the sibling ``cam-script`` repository's
    ``enrollment.py`` (``identify_speakers``): cosine similarity -- a plain
    dot product, since every embedding is already L2-normalized -- against
    each reference, keeping the best.
    """

    __slots__ = ("_embedder", "_refs", "_threshold")

    def __init__(self, embedder: Embedder, db: dict, threshold: float) -> None:
        """Build a verifier from a loaded embedder and enrollment database.

        Args:
            embedder: Produces the embedding for a live clip.
            db: The enrollment database, as returned by
                :func:`echochamber.speakerid.enrollment.load_db` -- entries
                whose ``backend`` is not :data:`BACKEND_QNN` are excluded, so
                a database left over from a differently-shaped export is
                skipped rather than compared against incompatible vectors.
            threshold: Minimum cosine similarity to call a match.
        """
        self._embedder: Embedder = embedder
        self._refs: dict[str, np.ndarray] = {
            name: np.asarray(entry["embedding"], dtype=np.float32)
            for name, entry in db.items()
            if entry.get("backend") == BACKEND_QNN
        }
        self._threshold: float = float(threshold)

    @property
    def enrolled_count(self) -> int:
        """Number of enrolled speakers usable with this verifier."""
        return len(self._refs)

    def verify(self, samples: np.ndarray, sample_rate: int) -> VerifyResult:
        """Embed ``samples`` and compare against every usable reference.

        Args:
            samples: 1-D mono ``float32`` samples in ``[-1, 1]``.
            sample_rate: Sample rate of ``samples`` in Hz.

        Returns:
            ``VerifyResult(matched=False, speaker=None, score=0.0)`` when
            nothing is enrolled for this backend, without running the model.
            Otherwise the best-scoring speaker, matched if the score clears
            the configured threshold.
        """
        if not self._refs:
            return VerifyResult(matched=False, speaker=None, score=0.0)

        embedding = self._embedder.embed(samples, sample_rate)
        best_name: str | None = None
        best_score = float("-inf")
        for name, ref in self._refs.items():
            score = float(np.dot(embedding, ref))
            if score > best_score:
                best_name = name
                best_score = score

        return VerifyResult(
            matched=best_score >= self._threshold,
            speaker=best_name,
            score=best_score,
        )

    def close(self) -> None:
        """Release the underlying embedder.  Idempotent."""
        self._embedder.close()

    def __repr__(self) -> str:
        """Return a debugging representation of this verifier's state."""
        return (
            f"{type(self).__name__}(enrolled={len(self._refs)}, "
            f"threshold={self._threshold})"
        )
