"""Choosing a speaker-verification backend from a configuration.

Mirrors :mod:`echochamber.voicegate.backends` closely: one function, a
never-raise contract, and a fallback chain where every rung is deliberate --
except there is only one rung here, since the QNN subprocess chain
(:mod:`echochamber.speakerid.qnn_subprocess`) is the only backend this
project builds. There is no CPU/torch fallback: PyTorch has no Windows
ARM64 wheel, so an in-process backend would only ever work on a dev machine,
never on the deployment target, and this project does not maintain two
speaker-verification code paths for that trade.

* ``qnn_onnx_path`` and ``qnn_worker_python`` both set -> a
  :class:`~echochamber.speakerid.verifier.EnrolledSpeakerVerifier` wrapping a
  started :class:`~echochamber.speakerid.qnn_subprocess.QnnEmbedder`.
* either unset, or the database has nothing enrolled -> ``None``.  A verifier
  with nothing to check against is inert, not broken: see
  :meth:`echochamber.speakerid.verifier.EnrolledSpeakerVerifier.verify`'s
  empty-database short-circuit, which still applies once built, but building
  a whole subprocess chain to always return "no match" is waste this
  function skips outright.

**Failures are returned, not raised.**  Launching the QNN subprocess chain
can fail for entirely ordinary reasons -- an x64 interpreter that moved, an
ONNX graph that was never exported, a machine with no Hexagon NPU at all --
and every one of them happens at the moment the user presses Start.  A raised
exception there would have to be caught by the GUI anyway, so this returns a
:class:`VerifierChoice` carrying both what it managed to build and what went
wrong, exactly like :func:`echochamber.voicegate.backends.build_recognizer`.
"""

from __future__ import annotations

from dataclasses import dataclass

from echochamber.speakerid.config import SpeakerIdConfig
from echochamber.speakerid.enrollment import load_db
from echochamber.speakerid.verifier import Embedder, EnrolledSpeakerVerifier

__all__ = ["VerifierChoice", "build_embedder", "build_verifier", "describe_backend"]


@dataclass(frozen=True, slots=True)
class VerifierChoice:
    """What :func:`build_verifier` produced, and whether it is the real thing.

    Attributes:
        verifier: The verifier to use, or ``None`` when verification is
            disabled, unconfigured, or failed to start -- unlike
            :class:`echochamber.voicegate.backends.RecognizerChoice`, there is
            no null-object stand-in here, because the gate's own fail-closed
            handling of "no verifier configured" already covers it; see
            :meth:`echochamber.voicegate.sink.VoiceGateSink._verify_speaker`.
        backend: Short name of what was built: ``"qnn"`` or ``"none"``.
        error: Why the configured backend could not be built, or ``None``
            when nothing went wrong.  Non-``None`` together with
            ``backend="none"`` means verification is configured on but
            unusable.
    """

    verifier: EnrolledSpeakerVerifier | None
    backend: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        """``True`` when the configured backend was built successfully."""
        return self.error is None

    def __repr__(self) -> str:
        """Return a debugging representation naming the backend and outcome."""
        return (
            f"{type(self).__name__}(backend={self.backend!r}, "
            f"ok={self.ok}, error={self.error!r})"
        )


def build_embedder(config: SpeakerIdConfig) -> Embedder:
    """Build and start the QNN subprocess embedder ``config`` describes.

    Split out from :func:`build_verifier` so the enrollment CLI
    (``scripts/enroll_speaker.py``) can embed a recording without also
    needing an enrollment database to compare it against.

    Args:
        config: Must have ``qnn_onnx_path`` and ``qnn_worker_python`` set.

    Returns:
        A started :class:`~echochamber.speakerid.qnn_subprocess.QnnEmbedder`.

    Raises:
        ValueError: If ``qnn_onnx_path`` or ``qnn_worker_python`` is unset.
        echochamber.speakerid.qnn_subprocess.QnnStartupError: If the
            subprocess chain could not be started.
    """
    if not config.qnn_onnx_path or not config.qnn_worker_python:
        raise ValueError(
            "qnn_onnx_path and qnn_worker_python must both be set to build "
            "the speaker-verification embedder; run "
            "`python scripts/setup_speakerid_qnn.py` and "
            "`python scripts/export_speakerid_qnn.py`"
        )
    from echochamber.speakerid.qnn_subprocess import QnnEmbedder  # noqa: PLC0415

    embedder = QnnEmbedder(
        driver_python=config.qnn_worker_python,
        onnx_path=config.qnn_onnx_path,
        npu_python=config.qnn_npu_python,
        startup_timeout_s=config.startup_timeout_s,
    )
    embedder.start()
    return embedder


def build_verifier(config: SpeakerIdConfig) -> VerifierChoice:
    """Build the speaker verifier ``config`` asks for, degrading rather than raising.

    Args:
        config: The speaker-ID configuration.

    Returns:
        A :class:`VerifierChoice`.  ``verifier`` is ``None`` unless the QNN
        chain is fully configured and started successfully.
    """
    if not config.enabled:
        return VerifierChoice(None, "none")

    if not config.qnn_onnx_path or not config.qnn_worker_python:
        return VerifierChoice(
            None,
            "none",
            "speaker verification is enabled but qnn_onnx_path/"
            "qnn_worker_python is not configured; run "
            "`python scripts/setup_speakerid_qnn.py` and "
            "`python scripts/export_speakerid_qnn.py`",
        )

    try:
        embedder = build_embedder(config)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, not raised
        return VerifierChoice(None, "none", str(exc))

    db = load_db(config.db_path)
    verifier = EnrolledSpeakerVerifier(embedder, db, config.threshold)
    if verifier.enrolled_count == 0:
        # A verifier with nothing to compare against is not a reason to keep
        # a subprocess chain running: it can only ever return "no match" for
        # every phrase, which is exactly what the gate's own None-verifier
        # path already does for free.
        verifier.close()
        return VerifierChoice(
            None,
            "none",
            f"no speakers enrolled in {config.db_path!r}; run "
            f"`python scripts/enroll_speaker.py enroll <name>`",
        )
    return VerifierChoice(verifier, "qnn")


def describe_backend(choice: VerifierChoice) -> str:
    """Render a :class:`VerifierChoice` as one line for a status bar.

    Args:
        choice: What :func:`build_verifier` returned.

    Returns:
        A short human-readable description.
    """
    if choice.error is not None:
        return f"speaker verification disabled: {choice.error}"
    if choice.backend == "none":
        return "speaker verification off"
    return f"speaker verification on ({choice.backend}, {choice.verifier.enrolled_count} enrolled)"
