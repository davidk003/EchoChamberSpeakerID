"""Tests for echochamber.speakerid.config.

Same shape as tests/test_voicegate_config.py: SpeakerIdConfig is a frozen
dataclass validated in __post_init__, so it is tested the same way --
documented defaults, every rejection path asserted on message, and the
autodetection helper proving it only fills in fields the caller left unset.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from echochamber.speakerid.config import (
    SpeakerIdConfig,
    autodetect_qnn_onnx_path,
    autodetect_qnn_worker_python,
    autodetect_speaker_id_config,
)

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "db_path": "enrolled_speakers.json",
    "threshold": 0.31,
    "qnn_onnx_path": None,
    "qnn_worker_python": None,
    "qnn_npu_python": None,
    "startup_timeout_s": 30.0,
}


class TestDefaults:
    @pytest.mark.parametrize(("field", "expected"), sorted(DEFAULTS.items()))
    def test_default_field_values(self, field: str, expected: Any) -> None:
        cfg = SpeakerIdConfig()
        assert getattr(cfg, field) == expected

    def test_enabled_defaults_to_false(self) -> None:
        """A fresh checkout has no enrollment database or exported ONNX graph."""
        assert SpeakerIdConfig().enabled is False


class TestValidation:
    def test_empty_db_path_raises(self) -> None:
        with pytest.raises(ValueError, match="db_path must not be empty"):
            SpeakerIdConfig(db_path="")

    @pytest.mark.parametrize("threshold", [-1.5, 1.5, 2.0, -2.0])
    def test_threshold_outside_range_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            SpeakerIdConfig(threshold=threshold)

    @pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0, 0.31])
    def test_threshold_at_or_inside_bounds_is_accepted(self, threshold: float) -> None:
        assert SpeakerIdConfig(threshold=threshold).threshold == threshold

    @pytest.mark.parametrize("timeout", [0.0, -1.0, -0.001])
    def test_non_positive_startup_timeout_raises(self, timeout: float) -> None:
        with pytest.raises(ValueError, match="startup_timeout_s must be > 0"):
            SpeakerIdConfig(startup_timeout_s=timeout)


class TestWithEnabled:
    def test_returns_a_new_validated_config(self) -> None:
        cfg = SpeakerIdConfig()
        enabled = cfg.with_enabled(True)
        assert enabled is not cfg
        assert enabled.enabled is True
        assert cfg.enabled is False

    def test_coerces_truthy_values(self) -> None:
        assert SpeakerIdConfig().with_enabled(1).enabled is True
        assert SpeakerIdConfig().with_enabled(0).enabled is False


class TestAutodetectQnnOnnxPath:
    def test_returns_none_when_missing(self, tmp_path: Any) -> None:
        assert autodetect_qnn_onnx_path(root=str(tmp_path)) is None

    def test_returns_path_when_present(self, tmp_path: Any) -> None:
        onnx_dir = tmp_path / "models" / "speakerid"
        onnx_dir.mkdir(parents=True)
        onnx_path = onnx_dir / "campplus_qnn.onnx"
        onnx_path.write_bytes(b"fake onnx")
        found = autodetect_qnn_onnx_path(root=str(tmp_path))
        assert found is not None
        assert os.path.isfile(found)


class TestAutodetectQnnWorkerPython:
    def test_returns_none_when_missing(self, tmp_path: Any) -> None:
        assert autodetect_qnn_worker_python(root=str(tmp_path)) is None

    def test_returns_path_when_present(self, tmp_path: Any) -> None:
        venv_dir = tmp_path / ".venv-speakerid-x64"
        scripts_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
        scripts_dir.mkdir(parents=True)
        python_name = "python.exe" if os.name == "nt" else "python"
        (scripts_dir / python_name).write_bytes(b"")
        found = autodetect_qnn_worker_python(root=str(tmp_path))
        assert found is not None
        assert os.path.isfile(found)


class TestAutodetectSpeakerIdConfig:
    def test_leaves_paths_none_when_nothing_on_disk(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.chdir(tmp_path)
        import echochamber.speakerid.config as config_mod

        monkeypatch.setattr(config_mod, "_REPO_ROOT", str(tmp_path))
        cfg = autodetect_speaker_id_config()
        assert cfg.qnn_onnx_path is None
        assert cfg.qnn_worker_python is None

    def test_explicit_override_beats_detection(self, tmp_path: Any, monkeypatch: Any) -> None:
        import echochamber.speakerid.config as config_mod

        monkeypatch.setattr(config_mod, "_REPO_ROOT", str(tmp_path))
        onnx_dir = tmp_path / "models" / "speakerid"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "campplus_qnn.onnx").write_bytes(b"fake")
        cfg = autodetect_speaker_id_config(qnn_onnx_path=None)
        # An explicit None must mean "no model", not "go detect one" -- even
        # though a real graph is sitting right where detection would find it.
        assert cfg.qnn_onnx_path is None

    def test_detects_onnx_path_left_on_disk(self, tmp_path: Any, monkeypatch: Any) -> None:
        import echochamber.speakerid.config as config_mod

        monkeypatch.setattr(config_mod, "_REPO_ROOT", str(tmp_path))
        onnx_dir = tmp_path / "models" / "speakerid"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "campplus_qnn.onnx").write_bytes(b"fake")
        cfg = autodetect_speaker_id_config()
        assert cfg.qnn_onnx_path is not None
        assert os.path.isfile(cfg.qnn_onnx_path)

    def test_overrides_pass_through_and_still_validate(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            autodetect_speaker_id_config(threshold=5.0)
