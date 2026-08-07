"""Enrolled-speaker database: load/save JSON, add an entry.

Same on-disk shape as the sibling ``cam-script`` repository's
``enrollment.py`` (``{name: {"embedding": [...], "backend": "..."}}``), but
this repository keeps its own file rather than reading that one -- the two
projects' enrollments are for different purposes and drift independently.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["BACKEND_QNN", "clear_db", "enroll", "load_db", "remove_speaker", "save_db"]

BACKEND_QNN: str = "qnn"
"""The only enrollment ``backend`` tag this project writes or reads.

Kept as a named tag rather than an implicit assumption, so a database entry
from a differently-shaped export (a future change to
``T_MAX_FRAMES``/``scripts/export_speakerid_qnn.py``, say) could in principle
be told apart from one usable with the current worker --
:class:`~echochamber.speakerid.verifier.EnrolledSpeakerVerifier` filters on
this exact value. Named after the sibling ``cam-script`` repository's own
``BACKEND_CPU``/``BACKEND_NPU`` tags, which established the precedent.
"""


def load_db(path: str) -> dict:
    """Load the enrollment database, or ``{}`` if it does not exist yet.

    Args:
        path: Path to the JSON file.

    Returns:
        ``{name: {"embedding": [...], "backend": "..."}}``.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_db(path: str, db: dict) -> None:
    """Write the enrollment database as pretty-printed JSON.

    Args:
        path: Destination path.  Its parent directory must exist.
        db: The database to write.
    """
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(db, handle, indent=2)


def enroll(
    db: dict, name: str, embedding: np.ndarray, backend: str = BACKEND_QNN
) -> dict:
    """Add or overwrite ``name``'s enrollment in ``db``.

    Args:
        db: The database to update, in place.
        name: Speaker name; overwrites any existing entry for the same name.
        embedding: A ``(192,)`` L2-normalized embedding.
        backend: Which embedder produced ``embedding``.  Always
            :data:`BACKEND_QNN` in this project today; kept as a parameter
            rather than hardcoded so a database entry's provenance stays
            explicit on disk even though only one backend writes it.

    Returns:
        ``db``, for convenience.
    """
    db[name] = {"embedding": np.asarray(embedding, dtype=np.float32).tolist(), "backend": backend}
    return db


def remove_speaker(db: dict, name: str) -> bool:
    """Remove ``name`` from ``db`` if present.

    Args:
        db: The database to update, in place.
        name: Speaker to remove.

    Returns:
        ``True`` if an entry was removed, ``False`` if ``name`` was not enrolled.
    """
    return db.pop(name, None) is not None


def clear_db(path: str) -> None:
    """Delete the database file if it exists.

    Args:
        path: Path to the JSON file.
    """
    if os.path.isfile(path):
        os.remove(path)
