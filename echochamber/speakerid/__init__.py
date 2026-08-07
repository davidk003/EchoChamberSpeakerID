"""Speaker verification: is this clip an enrolled voice?

Compares a live audio clip against a small database of enrolled speaker
embeddings, using the CAM++ model exported to run on the Qualcomm Hexagon NPU
via ONNX Runtime's QNN execution provider -- the project's deployment target
is Windows on ARM64, where PyTorch has no wheel and the NPU is the only
practical place to run this model; see
:mod:`echochamber.speakerid.qnn_subprocess` for why that still needs an x64
process in between.

Like :mod:`echochamber.voicegate`, this package is *pluggable and absent by
default*: with no enrollment database and no exported ONNX graph configured,
:func:`echochamber.speakerid.backends.build_verifier` returns nothing to gate
with, and a checkout with neither still imports, still runs the GUI, and
still passes the whole test suite.

Kept mutually unaware of :mod:`echochamber.voicegate` -- neither package
imports the other -- and wired together only by
:mod:`echochamber.ui.controller`, exactly how :mod:`echochamber.voicegate` and
:mod:`echochamber.voicegate.notify` are kept apart today. The seam between
them is :mod:`echochamber.voicegate.speaker`'s ``SpeakerVerifier`` protocol,
which :class:`~echochamber.speakerid.verifier.EnrolledSpeakerVerifier`
satisfies structurally.
"""

from __future__ import annotations

from echochamber.speakerid.config import SpeakerIdConfig, autodetect_speaker_id_config
from echochamber.speakerid.enrollment import BACKEND_QNN

__all__ = [
    "BACKEND_QNN",
    "SpeakerIdConfig",
    "autodetect_speaker_id_config",
]
