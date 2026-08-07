"""CAM++ waveform preprocessing shared by the QNN driver process.

The full CAM++ *model* only ever runs on the Hexagon NPU in this project (see
:mod:`echochamber.speakerid.qnn_subprocess`); there is no CPU/torch inference
path. What still needs ``torch``/``torchaudio`` is the **feature extraction**
step ahead of it -- turning a raw waveform into the 80-dim Kaldi fbank the
exported ONNX graph expects -- because that preprocessing was traced into the
graph's *input*, not baked into the graph itself. That step runs in the x64
"driver" process (:mod:`echochamber.speakerid.qnn_driver_worker`), which is
also where this module's functions are used; the ARM64 NPU worker
(:mod:`echochamber.speakerid.qnn_npu_worker`) only runs ONNX Runtime and never
imports this module at all.

Ported unchanged from the sibling ``cam-script`` repository's ``campplus.py``,
which is not a pip-installable package and therefore cannot be a runtime
dependency.

**torch and torchaudio are imported lazily**, inside the functions that need
them, for the same reason
:func:`echochamber.voicegate.recognizer.load_vosk_recognizer` imports
:mod:`vosk` lazily: PyTorch has no ``win_arm64`` wheel, so a top-level import
here would make this module unimportable on the project's own ARM64
deployment target, where it is never actually needed (the driver process runs
under a separate x64 interpreter).
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_THRESHOLD",
    "FEAT_DIM",
    "MIN_SAMPLES",
    "SAMPLE_RATE",
    "compute_fbank",
    "load_wav_file",
    "prepare_wav",
]

FEAT_DIM = 80
SAMPLE_RATE = 16000
# modelscope pipeline default yesOrno_thr.
DEFAULT_THRESHOLD = 0.31
# StatsPool uses std(unbiased=True), so the pooled time axis needs >= 2 frames
# or the embedding comes back NaN. 720 samples = 45 ms is the empirical floor.
MIN_SAMPLES = 720
# More channels than this and the tensor is almost certainly (samples, channels).
MAX_CHANNELS = 16


def prepare_wav(samples: np.ndarray, sample_rate: int) -> Any:
    """Validate and normalize mono samples to a 16 kHz ``(1, T)`` torch tensor.

    Args:
        samples: 1-D array in ``[-1, 1]``, any float dtype.
        sample_rate: Sample rate of ``samples`` in Hz.

    Returns:
        A ``(1, T)`` float32 tensor at :data:`SAMPLE_RATE`.

    Raises:
        ValueError: If ``samples`` is not 1-D, contains non-finite values, or
            is shorter than :data:`MIN_SAMPLES` after resampling.
    """
    import torch
    import torchaudio

    data = np.asarray(samples)
    if data.ndim != 1:
        raise ValueError(f"expected a 1-D mono sample array, got shape {data.shape}")
    wav = torch.from_numpy(data.astype(np.float32, copy=True)).unsqueeze(0)
    if sample_rate != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE)
    if not torch.isfinite(wav).all():
        raise ValueError("waveform contains NaN or inf samples")
    if wav.shape[-1] < MIN_SAMPLES:
        raise ValueError(
            f"audio too short: need at least {MIN_SAMPLES} samples "
            f"({MIN_SAMPLES / SAMPLE_RATE * 1000:.0f} ms at {SAMPLE_RATE} Hz), "
            f"got {wav.shape[-1]}"
        )
    return wav


def load_wav_file(path: str) -> Any:
    """Load an audio file as a 16 kHz float waveform in ``[-1, 1]``, shape ``(1, T)``.

    Takes the first channel of multi-channel input rather than downmixing,
    matching the reference implementation.  Used by the enrollment CLI, which
    reads from files rather than the live pipeline.
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if data.shape[1] > MAX_CHANNELS and data.shape[1] > data.shape[0]:
        raise ValueError(
            f"waveform shape {tuple(data.shape)} looks transposed; expected "
            f"(samples, channels)"
        )
    return prepare_wav(data[:, 0], sr)


def compute_fbank(wav: Any) -> Any:
    """80-dim Kaldi fbank with mean normalization over time, shape ``(T, 80)``."""
    import torchaudio

    feat = torchaudio.compliance.kaldi.fbank(
        wav, num_mel_bins=FEAT_DIM, dither=0, sample_frequency=SAMPLE_RATE
    )
    return feat - feat.mean(dim=0, keepdim=True)
