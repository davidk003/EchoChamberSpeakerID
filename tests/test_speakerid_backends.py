"""Tests for echochamber.speakerid.backends.

build_verifier's own QNN-subprocess construction is monkeypatched out via
build_embedder, the same seam scripts/enroll_speaker.py uses -- these tests
care about the fallback chain and never-raise contract, not about actually
launching a subprocess (see tests/test_speakerid_qnn_subprocess.py for that).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import echochamber.speakerid.backends as backends_mod
from echochamber.speakerid.backends import build_embedder, build_verifier, describe_backend
from echochamber.speakerid.config import SpeakerIdConfig
from echochamber.speakerid.enrollment import BACKEND_QNN, save_db


class FakeEmbedder:
    def __init__(self) -> None:
        self.closed = False

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        vec = np.array([1.0, 0.0], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_embedder(monkeypatch: Any) -> FakeEmbedder:
    """Replace build_embedder with one that returns a FakeEmbedder, never a subprocess."""
    embedder = FakeEmbedder()
    monkeypatch.setattr(backends_mod, "build_embedder", lambda config: embedder)
    return embedder


class TestBuildEmbedder:
    def test_raises_when_onnx_path_unset(self) -> None:
        config = SpeakerIdConfig(qnn_worker_python="python.exe")
        with pytest.raises(ValueError, match="qnn_onnx_path"):
            build_embedder(config)

    def test_raises_when_worker_python_unset(self) -> None:
        config = SpeakerIdConfig(qnn_onnx_path="model.onnx")
        with pytest.raises(ValueError, match="qnn_worker_python"):
            build_embedder(config)


class TestBuildVerifierDisabled:
    def test_disabled_config_returns_none_backend_none_verifier(self) -> None:
        choice = build_verifier(SpeakerIdConfig(enabled=False))
        assert choice.verifier is None
        assert choice.backend == "none"
        assert choice.ok is True


class TestBuildVerifierUnconfigured:
    def test_enabled_but_unconfigured_paths_reports_an_error(self) -> None:
        choice = build_verifier(SpeakerIdConfig(enabled=True))
        assert choice.verifier is None
        assert choice.backend == "none"
        assert choice.ok is False
        assert "qnn_onnx_path" in choice.error


class TestBuildVerifierEmptyDatabase:
    def test_empty_database_returns_none_without_leaking_the_embedder(
        self, fake_embedder: FakeEmbedder, tmp_path: Any
    ) -> None:
        config = SpeakerIdConfig(
            enabled=True,
            db_path=str(tmp_path / "enrolled_speakers.json"),
            qnn_onnx_path="model.onnx",
            qnn_worker_python="python.exe",
        )
        choice = build_verifier(config)
        assert choice.verifier is None
        assert choice.backend == "none"
        assert "no speakers enrolled" in choice.error
        # A verifier that can only ever say "no match" must not hold a live
        # subprocess chain open; the embedder is closed rather than leaked.
        assert fake_embedder.closed is True


class TestBuildVerifierEnrolled:
    def test_enrolled_speaker_produces_a_working_verifier(
        self, fake_embedder: FakeEmbedder, tmp_path: Any
    ) -> None:
        db_path = str(tmp_path / "enrolled_speakers.json")
        vec = np.array([1.0, 0.0], dtype=np.float32)
        save_db(db_path, {"alice": {"embedding": (vec / np.linalg.norm(vec)).tolist(), "backend": BACKEND_QNN}})

        config = SpeakerIdConfig(
            enabled=True,
            db_path=db_path,
            qnn_onnx_path="model.onnx",
            qnn_worker_python="python.exe",
        )
        choice = build_verifier(config)
        assert choice.ok is True
        assert choice.backend == "qnn"
        assert choice.verifier is not None
        assert choice.verifier.enrolled_count == 1

        result = choice.verifier.verify(np.zeros(16_000, dtype=np.float32), 16_000)
        assert result.matched is True
        assert result.speaker == "alice"


class TestBuildVerifierStartupFailure:
    def test_embedder_start_failure_is_reported_not_raised(
        self, monkeypatch: Any, tmp_path: Any
    ) -> None:
        def failing_build_embedder(config: SpeakerIdConfig) -> None:
            raise RuntimeError("subprocess refused to start")

        monkeypatch.setattr(backends_mod, "build_embedder", failing_build_embedder)
        config = SpeakerIdConfig(
            enabled=True,
            db_path=str(tmp_path / "enrolled_speakers.json"),
            qnn_onnx_path="model.onnx",
            qnn_worker_python="python.exe",
        )
        choice = build_verifier(config)
        assert choice.verifier is None
        assert choice.backend == "none"
        assert "subprocess refused to start" in choice.error


class TestDescribeBackend:
    def test_disabled_reads_as_off(self) -> None:
        choice = build_verifier(SpeakerIdConfig(enabled=False))
        assert describe_backend(choice) == "speaker verification off"

    def test_error_is_included_in_the_description(self) -> None:
        choice = build_verifier(SpeakerIdConfig(enabled=True))
        assert "speaker verification disabled" in describe_backend(choice)

    def test_ok_backend_names_the_enrolled_count(
        self, fake_embedder: FakeEmbedder, tmp_path: Any
    ) -> None:
        db_path = str(tmp_path / "enrolled_speakers.json")
        vec = np.array([1.0, 0.0], dtype=np.float32)
        save_db(db_path, {"alice": {"embedding": (vec / np.linalg.norm(vec)).tolist(), "backend": BACKEND_QNN}})
        config = SpeakerIdConfig(
            enabled=True,
            db_path=db_path,
            qnn_onnx_path="model.onnx",
            qnn_worker_python="python.exe",
        )
        choice = build_verifier(config)
        description = describe_backend(choice)
        assert "qnn" in description
        assert "1 enrolled" in description
