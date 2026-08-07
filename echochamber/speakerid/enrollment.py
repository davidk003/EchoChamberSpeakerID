"""Enrolled-speaker database: load/save JSON, add an entry.

Same on-disk shape as the sibling ``cam-script`` repository's
``enrollment.py`` (``{name: {"embedding": [...], "backend": "..."}}``), but
this repository keeps its own file rather than reading that one -- the two
projects' enrollments are for different purposes and drift independently.
"""

from __future__ import annotations

import json
import os
import wave

import numpy as np

__all__ = [
    "BACKEND_QNN",
    "clear_db",
    "enroll",
    "load_db",
    "load_wav_mono",
    "remove_speaker",
    "save_db",
]

_SUPPORTED_WIDTHS: tuple[int, ...] = (1, 2, 4)
"""Sample widths in bytes :func:`load_wav_mono` can decode: 8-, 16- and 32-bit PCM."""

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


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Read an entire WAV file as mono ``float32`` samples in ``[-1, 1]``.

    The primary way to enroll a speaker: a clip recorded once, elsewhere,
    with whatever microphone and editing tooling someone already has, rather
    than requiring them to sit in front of *this* app's own microphone
    picker. Decodes the same way
    :class:`~echochamber.audio.sources.file_source.FileSource` does -- any
    sample rate, any of the three PCM widths it understands, multi-channel
    downmixed by averaging -- so a snippet this app wrote, or a clip from
    any other recorder, both just work.

    Args:
        path: Path to an uncompressed PCM WAV file.

    Returns:
        ``(samples, sample_rate)``. ``sample_rate`` is whatever the file's
        header says; the embedder resamples internally, so nothing here
        needs to match its expected rate.

    Raises:
        ValueError: If the file is not readable PCM, or its sample width is
            not 8-, 16- or 32-bit.
        OSError: If the file cannot be opened.
    """
    try:
        with wave.open(path, "rb") as wav:
            params = wav.getparams()
            if params.comptype != "NONE":
                raise ValueError(
                    f"{path!r} uses compression {params.comptype!r} "
                    f"({params.compname!r}); only uncompressed PCM is supported"
                )
            if params.sampwidth not in _SUPPORTED_WIDTHS:
                raise ValueError(
                    f"{path!r} has sample width {params.sampwidth * 8} bits; "
                    f"only 8-, 16- and 32-bit PCM are supported"
                )
            raw = wav.readframes(params.nframes)
    except wave.Error as exc:
        raise ValueError(
            f"{path!r} is not a readable uncompressed PCM WAV file: {exc}"
        ) from exc

    if params.sampwidth == 1:
        # 8-bit WAV is *unsigned*, biased by 128 -- the one format that is
        # not a straight signed integer.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif params.sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0

    if params.nchannels > 1:
        data = data.reshape(-1, params.nchannels).mean(axis=1)
    return data.astype(np.float32), params.framerate


def clear_db(path: str) -> None:
    """Delete the database file if it exists.

    Args:
        path: Path to the JSON file.
    """
    if os.path.isfile(path):
        os.remove(path)
