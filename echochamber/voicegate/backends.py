"""Choosing a recogniser backend from a configuration.

One function, kept in its own module for an import-cycle reason rather than a
conceptual one: :mod:`echochamber.voicegate.subprocess_recognizer` imports
:mod:`echochamber.voicegate.recognizer`, so the chooser cannot live in the
latter without making the two mutually dependent.

**The choice is a fallback chain, and every rung of it is deliberate.**

* ``worker_python`` set  ->  :class:`SubprocessRecognizer`.  The Windows ARM64
  path: vosk cannot be installed into the application's own interpreter there,
  so the decoder runs in an x64 child process.
* ``model_path`` set, no ``worker_python``  ->  in-process
  :class:`~echochamber.voicegate.recognizer.VoskRecognizer`.  Simpler, one less
  process, and correct anywhere vosk actually installs.
* neither set  ->  :class:`~echochamber.voicegate.recognizer.NullRecognizer`.
  A gate with nothing to recognise with is inert, not broken.

**Failures are returned, not raised.**  Building a recogniser can fail for
entirely ordinary reasons -- a model directory that moved, an x64 interpreter
that was uninstalled, a worker that cannot import vosk -- and every one of them
happens at the moment the user presses Start.  A raised exception there would
have to be caught by the GUI anyway, so this returns a
:class:`RecognizerChoice` carrying both what it managed to build and what went
wrong.  The caller decides whether "capture works, gating does not" is
acceptable; for the GUI it is, and it says so in the status bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from echochamber.voicegate.config import VoiceGateConfig
from echochamber.voicegate.recognizer import (
    NullRecognizer,
    Recognizer,
    load_vosk_recognizer,
)

__all__ = ["RecognizerChoice", "build_recognizer", "describe_backend"]


@dataclass(frozen=True, slots=True)
class RecognizerChoice:
    """What :func:`build_recognizer` produced, and whether it is the real thing.

    Attributes:
        recognizer: The backend to use.  Never ``None`` -- a failure yields a
            :class:`~echochamber.voicegate.recognizer.NullRecognizer` so the
            caller always has something satisfying the protocol.
        backend: Short name of what was built: ``"subprocess"``, ``"in-process"``
            or ``"none"``.
        error: Why the requested backend could not be built, or ``None`` when
            nothing went wrong.  Non-``None`` together with ``backend="none"``
            means the gate is wired up but deaf.
    """

    recognizer: Recognizer
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


def build_recognizer(
    config: VoiceGateConfig, sample_rate: int
) -> RecognizerChoice:
    """Build the recogniser ``config`` asks for, degrading rather than raising.

    Args:
        config: The gate configuration.  ``worker_python`` and ``model_path``
            select the backend; see the module docstring for the chain.
        sample_rate: Capture sample rate in Hz, passed to whichever backend is
            built so its decoder matches the audio it will be fed.

    Returns:
        A :class:`RecognizerChoice`.  Its ``recognizer`` is always usable; check
        ``ok`` to find out whether it is the one that was asked for.
    """
    if not config.enabled:
        return RecognizerChoice(NullRecognizer(), "none")

    phrases = config.normalized_phrases

    if config.worker_python:
        if not config.model_path:
            return RecognizerChoice(
                NullRecognizer(),
                "none",
                "worker_python is set but model_path is not; the worker has no "
                "model to load",
            )
        # Imported here, not at module scope: it pulls in the protocol and
        # threading machinery that a configuration using the in-process backend
        # has no reason to load.
        from echochamber.voicegate.subprocess_recognizer import (  # noqa: PLC0415
            RecognizerStartupError,
            SubprocessRecognizer,
        )

        worker = SubprocessRecognizer(
            python_executable=config.worker_python,
            model_path=config.model_path,
            sample_rate=sample_rate,
            phrases=phrases,
            startup_timeout_s=config.startup_timeout_s,
        )
        try:
            worker.start()
        except (RecognizerStartupError, OSError, ValueError) as exc:
            # start() already tore the worker down, so there is no orphan to
            # clean up here -- only a message to pass on.
            return RecognizerChoice(NullRecognizer(), "none", str(exc))
        return RecognizerChoice(worker, "subprocess")

    if config.model_path:
        try:
            return RecognizerChoice(
                load_vosk_recognizer(config.model_path, sample_rate, phrases),
                "in-process",
            )
        except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
            return RecognizerChoice(NullRecognizer(), "none", str(exc))

    return RecognizerChoice(
        NullRecognizer(),
        "none",
        "the voice gate is enabled but no model_path is configured; run "
        "`python scripts/setup_voice_gate.py`",
    )


def describe_backend(choice: RecognizerChoice) -> str:
    """Render a :class:`RecognizerChoice` as one line for a status bar.

    Args:
        choice: What :func:`build_recognizer` returned.

    Returns:
        A short human-readable description.
    """
    if choice.error is not None:
        return f"voice gate disabled: {choice.error}"
    if choice.backend == "none":
        return "voice gate off"
    return f"voice gate listening ({choice.backend})"
