"""Tests for echochamber.speakerid.enrollment.

Plain unit tests against the JSON database helpers -- no torch, no
subprocess, nothing beyond the filesystem and numpy.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

from echochamber.speakerid.enrollment import (
    BACKEND_QNN,
    clear_db,
    enroll,
    load_db,
    remove_speaker,
    save_db,
)


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
