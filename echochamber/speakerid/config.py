"""Configuration for speaker verification.

Follows :class:`~echochamber.voicegate.config.VoiceGateConfig` exactly: a
frozen dataclass validated in ``__post_init__``, autodetection kept in a
free function so the type itself knows nothing about the filesystem. The QNN
subprocess chain (:mod:`echochamber.speakerid.qnn_subprocess`) is the only
backend -- there is no CPU/torch fallback -- so ``qnn_onnx_path`` and
``qnn_worker_python`` are what :func:`echochamber.speakerid.backends.build_verifier`
needs to have anything to build at all, exactly as ``VoiceGateConfig.model_path``
and ``worker_python`` are for the Vosk gate.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

__all__ = [
    "SpeakerIdConfig",
    "autodetect_qnn_onnx_path",
    "autodetect_qnn_worker_python",
    "autodetect_speaker_id_config",
]

_DEFAULT_DB_NAME: str = "enrolled_speakers.json"
"""Where :mod:`scripts.enroll_speaker` writes by default."""

_ONNX_DIR: str = "models"
"""Where :mod:`scripts.export_speakerid_qnn` writes the exported graph."""

_ONNX_NAME: str = "speakerid/campplus_qnn.onnx"
"""Must match ``scripts.export_speakerid_qnn.ONNX_PATH``."""

_VENV_DIR: str = ".venv-speakerid-x64"
"""Where :mod:`scripts.setup_speakerid_qnn` builds the x64 driver environment."""

_REPO_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
"""Repository root, derived from this file's location; see
:mod:`echochamber.voicegate.config` for why this beats the working directory.
"""


def _venv_python(venv_dir: str) -> str:
    """Return the interpreter path inside a virtual environment.

    Mirrors ``scripts.setup_voice_gate.venv_python``; duplicated for the same
    reason that function is duplicated from ``scripts.setup_speakerid_qnn``
    rather than imported -- see :mod:`echochamber.voicegate.config`.
    """
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def autodetect_qnn_onnx_path(root: str | None = None) -> str | None:
    """Find the exported ONNX graph :mod:`scripts.export_speakerid_qnn` would have written.

    Args:
        root: Directory to look under; the repository root when ``None``.

    Returns:
        The graph's absolute path if it exists, else ``None``.
    """
    base = _REPO_ROOT if root is None else root
    candidate = os.path.join(base, _ONNX_DIR, _ONNX_NAME)
    return candidate if os.path.isfile(candidate) else None


def autodetect_qnn_worker_python(root: str | None = None) -> str | None:
    """Find the x64 driver venv :mod:`scripts.setup_speakerid_qnn` would have built.

    Args:
        root: Directory to look under; the repository root when ``None``.

    Returns:
        The venv's interpreter path if it exists, else ``None``.
    """
    base = _REPO_ROOT if root is None else root
    candidate = _venv_python(os.path.join(base, _VENV_DIR))
    return candidate if os.path.isfile(candidate) else None


@dataclass(frozen=True, slots=True)
class SpeakerIdConfig:
    """Immutable speaker-verification configuration.

    Attributes:
        enabled: Whether verification runs at all.  ``False`` by default: it
            needs an enrollment database and the QNN subprocess chain this
            repository does not ship pre-built, so defaulting it on would
            make a fresh checkout's gate fail closed on every phrase.
        db_path: Path to the enrollment database JSON file.
        threshold: Minimum cosine similarity to call a match; see
            :data:`echochamber.speakerid.campplus.DEFAULT_THRESHOLD` for where
            the default comes from.
        qnn_onnx_path: Path to the exported static-shape ONNX graph, or
            ``None`` to leave verification without a model (it then never
            matches).
        qnn_worker_python: Interpreter for the x64 driver process -- has
            torch, extracts fbank features, and drives the ONNX worker -- or
            ``None`` for no verifier at all. On Windows ARM64 this is an
            **x64** ``python.exe``, mirroring
            ``VoiceGateConfig.worker_python``'s role one process further out;
            see :mod:`echochamber.speakerid.qnn_subprocess`.
        qnn_npu_python: Interpreter for the native ARM64 QNN inference worker
            (``onnxruntime-qnn``, no torch), or ``None`` to let the driver
            process fall back to its own default path. Passed to the driver
            rather than launched directly by this process, mirroring
            ``CAMPPLUS_QNN_PYTHON`` in the sibling ``cam-script`` repository.
        startup_timeout_s: How long to wait for the subprocess chain to
            report readiness before giving up.

    Raises:
        ValueError: If ``threshold`` is outside ``[-1, 1]``, if
            ``startup_timeout_s`` is not positive, or if ``db_path`` is empty.
    """

    enabled: bool = False
    db_path: str = _DEFAULT_DB_NAME
    threshold: float = 0.31
    qnn_onnx_path: str | None = None
    qnn_worker_python: str | None = None
    qnn_npu_python: str | None = None
    startup_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        """Validate the configuration; see the class docstring for the rules."""
        if not self.db_path:
            raise ValueError("db_path must not be empty")
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [-1, 1] (cosine similarity), got "
                f"{self.threshold}"
            )
        if self.startup_timeout_s <= 0:
            raise ValueError(
                f"startup_timeout_s must be > 0, got {self.startup_timeout_s}"
            )

    def with_enabled(self, enabled: bool) -> "SpeakerIdConfig":
        """Return a copy with verification switched on or off.

        Args:
            enabled: Whether verification should run.

        Returns:
            A new validated :class:`SpeakerIdConfig`.
        """
        return dataclasses.replace(self, enabled=bool(enabled))

    def __repr__(self) -> str:
        """Return a debugging representation of this configuration."""
        return (
            f"{type(self).__name__}(enabled={self.enabled}, "
            f"db_path={self.db_path!r}, threshold={self.threshold})"
        )


def autodetect_speaker_id_config(**overrides: object) -> SpeakerIdConfig:
    """Build a :class:`SpeakerIdConfig` with paths filled in from what
    :mod:`scripts.setup_speakerid_qnn` and :mod:`scripts.export_speakerid_qnn`
    would have left on disk.

    ``SpeakerIdConfig()`` itself defaults every path to ``None`` or a bare
    relative name and does not touch the filesystem -- deliberately, since the
    dataclass has no business knowing about disk state; the test suite pins
    those defaults. This is the seam a caller who *does* want the filesystem
    consulted uses instead: the application's entry point, not the config
    type. Detected values only fill in fields the caller did not already pass
    in ``overrides`` -- an explicit ``qnn_onnx_path=None`` still means "no
    model", not "go detect one".

    Args:
        **overrides: Passed through to :class:`SpeakerIdConfig`, taking
            precedence over anything detected.

    Returns:
        A validated :class:`SpeakerIdConfig`.
    """
    if "db_path" not in overrides:
        candidate = os.path.join(_REPO_ROOT, _DEFAULT_DB_NAME)
        if os.path.isfile(candidate):
            overrides["db_path"] = candidate
    if "qnn_onnx_path" not in overrides:
        overrides["qnn_onnx_path"] = autodetect_qnn_onnx_path()
    if "qnn_worker_python" not in overrides:
        overrides["qnn_worker_python"] = autodetect_qnn_worker_python()
    return SpeakerIdConfig(**overrides)
