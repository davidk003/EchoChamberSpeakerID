"""Tests for echochamber.speakerid.enrollment.

Plain unit tests against the JSON database helpers -- no torch, no
subprocess, nothing beyond the filesystem and numpy.
"""

from __future__ import annotations

import json
import os
import wave
from typing import Any

import numpy as np
import pytest

from echochamber.speakerid.enrollment import (
    BACKEND_QNN,
    clear_db,
    enroll,
    load_db,
    load_wav_mono,
    remove_speaker,
    save_db,
)


def _write_wav(
    path: str,
    samples: np.ndarray,
    sample_rate: int = 16_000,
    channels: int = 1,
    sampwidth: int = 2,
) -> None:
    """Write ``samples`` (float32, ``[-1, 1]``, interleaved) as a PCM WAV file."""
    with wave.open(path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(sample_rate)
        if sampwidth == 2:
            data = (samples * 32767.0).astype("<i2")
        elif sampwidth == 1:
            data = ((samples * 127.0) + 128.0).astype(np.uint8)
        else:
            data = (samples * 2147483647.0).astype("<i4")
        wav.writeframes(data.tobytes())


class TestLoadDb:
    def test_missing_file_returns_empty_dict(self, tmp_path: Any) -> None:
        assert load_db(str(tmp_path / "missing.json")) == {}

    def test_loads_existing_json(self, tmp_path: Any) -> None:
        path = tmp_path / "db.json"
        path.write_text(json.dumps({"alice": {"embedding": [0.1, 0.2], "backend": "qnn"}}))
        db = load_db(str(path))
        assert db == {"alice": {"embedding": [0.1, 0.2], "backend": "qnn"}}


class TestSaveDb:
    def test_writes_readable_json(self, tmp_path: Any) -> None:
        path = tmp_path / "db.json"
        save_db(str(path), {"bob": {"embedding": [1.0], "backend": "qnn"}})
        assert load_db(str(path)) == {"bob": {"embedding": [1.0], "backend": "qnn"}}

    def test_round_trip_through_enroll(self, tmp_path: Any) -> None:
        path = tmp_path / "db.json"
        db: dict = {}
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        enroll(db, "carol", embedding, BACKEND_QNN)
        save_db(str(path), db)
        reloaded = load_db(str(path))
        assert reloaded["carol"]["backend"] == BACKEND_QNN
        assert reloaded["carol"]["embedding"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


class TestEnroll:
    def test_adds_a_new_entry(self) -> None:
        db: dict = {}
        enroll(db, "dave", np.array([1.0, 2.0], dtype=np.float32), BACKEND_QNN)
        assert "dave" in db
        assert db["dave"]["backend"] == BACKEND_QNN
        assert db["dave"]["embedding"] == pytest.approx([1.0, 2.0])

    def test_overwrites_an_existing_entry(self) -> None:
        db: dict = {"dave": {"embedding": [9.0], "backend": "old"}}
        enroll(db, "dave", np.array([1.0, 2.0], dtype=np.float32), BACKEND_QNN)
        assert db["dave"]["embedding"] == pytest.approx([1.0, 2.0])
        assert db["dave"]["backend"] == BACKEND_QNN

    def test_returns_the_database_for_convenience(self) -> None:
        db: dict = {}
        result = enroll(db, "eve", np.array([1.0], dtype=np.float32), BACKEND_QNN)
        assert result is db

    def test_default_backend_is_qnn(self) -> None:
        db: dict = {}
        enroll(db, "frank", np.array([1.0], dtype=np.float32))
        assert db["frank"]["backend"] == BACKEND_QNN

    def test_embedding_is_stored_as_a_plain_list(self) -> None:
        db: dict = {}
        enroll(db, "grace", np.array([1.0, 2.0], dtype=np.float32), BACKEND_QNN)
        assert isinstance(db["grace"]["embedding"], list)
        assert all(isinstance(v, float) for v in db["grace"]["embedding"])


class TestRemoveSpeaker:
    def test_removes_an_existing_entry(self) -> None:
        db: dict = {"alice": {"embedding": [1.0], "backend": "qnn"}}
        assert remove_speaker(db, "alice") is True
        assert "alice" not in db

    def test_missing_entry_returns_false(self) -> None:
        db: dict = {}
        assert remove_speaker(db, "nobody") is False
        assert db == {}


class TestClearDb:
    def test_deletes_the_file(self, tmp_path: Any) -> None:
        path = tmp_path / "db.json"
        path.write_text("{}")
        clear_db(str(path))
        assert not os.path.isfile(str(path))

    def test_missing_file_is_not_an_error(self, tmp_path: Any) -> None:
        clear_db(str(tmp_path / "missing.json"))  # must not raise


class TestLoadWavMono:
    def test_reads_mono_16bit_pcm(self, tmp_path: Any) -> None:
        path = str(tmp_path / "clip.wav")
        samples = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
        _write_wav(path, samples, sample_rate=16_000, channels=1, sampwidth=2)

        loaded, sample_rate = load_wav_mono(path)

        assert sample_rate == 16_000
        assert loaded.dtype == np.float32
        assert loaded == pytest.approx(samples, abs=1e-3)

    def test_downmixes_stereo_by_averaging(self, tmp_path: Any) -> None:
        path = str(tmp_path / "stereo.wav")
        left = np.array([1.0, -1.0], dtype=np.float32)
        right = np.array([0.0, 0.0], dtype=np.float32)
        interleaved = np.empty(4, dtype=np.float32)
        interleaved[0::2] = left
        interleaved[1::2] = right
        _write_wav(path, interleaved, channels=2, sampwidth=2)

        loaded, _ = load_wav_mono(path)

        assert loaded == pytest.approx([0.5, -0.5], abs=1e-3)

    def test_reports_the_files_own_sample_rate(self, tmp_path: Any) -> None:
        path = str(tmp_path / "clip.wav")
        _write_wav(path, np.zeros(10, dtype=np.float32), sample_rate=44_100)
        _, sample_rate = load_wav_mono(path)
        assert sample_rate == 44_100

    def test_rejects_a_missing_file(self, tmp_path: Any) -> None:
        with pytest.raises(OSError):
            load_wav_mono(str(tmp_path / "missing.wav"))
