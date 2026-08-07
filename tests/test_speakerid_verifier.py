"""Tests for echochamber.speakerid.verifier.

Driven entirely against a fake Embedder -- no subprocess, no ONNX Runtime --
the same way tests/test_voicegate_sink.py drives VoiceGateSink against a
ScriptedRecognizer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from echochamber.speakerid.enrollment import BACKEND_QNN
from echochamber.speakerid.verifier import EnrolledSpeakerVerifier
from echochamber.voicegate.speaker import VerifyResult


class FakeEmbedder:
    """Returns pre-scripted embeddings for successive calls, in order."""

    def __init__(self, embeddings: list[np.ndarray] | None = None) -> None:
        self._embeddings = list(embeddings or [])
        self.calls: list[tuple[np.ndarray, int]] = []
        self.closed = False

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls.append((samples, sample_rate))
        return self._embeddings.pop(0)

    def close(self) -> None:
        self.closed = True


class RaisingEmbedder:
    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        raise RuntimeError("embedder is dead")

    def close(self) -> None:
        pass


def _unit(vector: list[float]) -> np.ndarray:
    """L2-normalize a vector, matching what a real embedder returns."""
    arr = np.array(vector, dtype=np.float32)
    return arr / np.linalg.norm(arr)


class TestConstruction:
    def test_filters_out_entries_with_a_different_backend(self) -> None:
        db = {
            "alice": {"embedding": [1.0, 0.0], "backend": BACKEND_QNN},
            "bob": {"embedding": [0.0, 1.0], "backend": "some-other-backend"},
        }
        verifier = EnrolledSpeakerVerifier(FakeEmbedder(), db, threshold=0.5)
        assert verifier.enrolled_count == 1

    def test_empty_database_has_zero_enrolled(self) -> None:
        verifier = EnrolledSpeakerVerifier(FakeEmbedder(), {}, threshold=0.5)
        assert verifier.enrolled_count == 0


class TestVerify:
    def test_empty_database_short_circuits_without_calling_the_embedder(self) -> None:
        embedder = FakeEmbedder()
        verifier = EnrolledSpeakerVerifier(embedder, {}, threshold=0.31)
        result = verifier.verify(np.zeros(16_000, dtype=np.float32), 16_000)
        assert result == VerifyResult(matched=False, speaker=None, score=0.0)
        assert embedder.calls == []

    def test_matching_speaker_above_threshold(self) -> None:
        db = {"alice": {"embedding": _unit([1.0, 0.0]).tolist(), "backend": BACKEND_QNN}}
        embedder = FakeEmbedder([_unit([1.0, 0.0])])
        verifier = EnrolledSpeakerVerifier(embedder, db, threshold=0.31)
        result = verifier.verify(np.zeros(16_000, dtype=np.float32), 16_000)
        assert result.matched is True
        assert result.speaker == "alice"
        assert result.score == pytest.approx(1.0)

    def test_no_match_below_threshold(self) -> None:
        db = {"alice": {"embedding": _unit([1.0, 0.0]).tolist(), "backend": BACKEND_QNN}}
        embedder = FakeEmbedder([_unit([0.0, 1.0])])  # orthogonal -> score 0.0
        verifier = EnrolledSpeakerVerifier(embedder, db, threshold=0.31)
        result = verifier.verify(np.zeros(16_000, dtype=np.float32), 16_000)
        assert result.matched is False
        assert result.speaker == "alice"  # still reports the best guess
        assert result.score == pytest.approx(0.0, abs=1e-6)

    def test_picks_the_best_scoring_of_several_speakers(self) -> None:
        db = {
            "alice": {"embedding": _unit([1.0, 0.0]).tolist(), "backend": BACKEND_QNN},
            "bob": {"embedding": _unit([0.9, 0.1]).tolist(), "backend": BACKEND_QNN},
        }
        embedder = FakeEmbedder([_unit([1.0, 0.0])])
        verifier = EnrolledSpeakerVerifier(embedder, db, threshold=0.31)
        result = verifier.verify(np.zeros(16_000, dtype=np.float32), 16_000)
        assert result.speaker == "alice"
        assert result.matched is True

    def test_threshold_boundary_is_inclusive(self) -> None:
        db = {"alice": {"embedding": _unit([1.0, 0.0]).tolist(), "backend": BACKEND_QNN}}
        embedder = FakeEmbedder([_unit([1.0, 0.0])])
        verifier = EnrolledSpeakerVerifier(embedder, db, threshold=1.0)
        result = verifier.verify(np.zeros(16_000, dtype=np.float32), 16_000)
        assert result.matched is True

    def test_embedder_failure_propagates(self) -> None:
        db = {"alice": {"embedding": _unit([1.0, 0.0]).tolist(), "backend": BACKEND_QNN}}
        verifier = EnrolledSpeakerVerifier(RaisingEmbedder(), db, threshold=0.31)
        with pytest.raises(RuntimeError, match="embedder is dead"):
            verifier.verify(np.zeros(16_000, dtype=np.float32), 16_000)

    def test_forwards_samples_and_sample_rate_to_the_embedder(self) -> None:
        db = {"alice": {"embedding": _unit([1.0, 0.0]).tolist(), "backend": BACKEND_QNN}}
        embedder = FakeEmbedder([_unit([1.0, 0.0])])
        verifier = EnrolledSpeakerVerifier(embedder, db, threshold=0.31)
        samples = np.linspace(-0.1, 0.1, 8000, dtype=np.float32)
        verifier.verify(samples, 16_000)
        (call_samples, call_rate) = embedder.calls[0]
        assert call_rate == 16_000
        assert np.array_equal(call_samples, samples)


class TestClose:
    def test_closes_the_underlying_embedder(self) -> None:
        embedder = FakeEmbedder()
        verifier = EnrolledSpeakerVerifier(embedder, {}, threshold=0.31)
        verifier.close()
        assert embedder.closed is True
