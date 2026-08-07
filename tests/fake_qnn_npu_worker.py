"""A fake NPU worker for tests: speaks the same protocol, no ONNX Runtime.

Run standalone via ``python -m tests.fake_qnn_npu_worker`` (not ``-m
echochamber.speakerid.qnn_npu_worker``), so
:mod:`echochamber.speakerid.qnn_driver_worker`'s ``NpuWorkerHandle`` can be
exercised against a deterministic double with no Hexagon NPU anywhere --
mirroring how ``tests/test_voicegate_subprocess.py`` drives
``SubprocessRecognizer`` with a fake ``Popen`` rather than real Vosk.

Echoes back a deterministic ``(192,)`` float32 vector derived from the whole
input payload's bytes, so a test can assert that changing the input changes
the output without caring about the exact numbers. Seeded from the full
payload rather than a summary statistic like the mean: CAM++'s fbank features
are mean-normalized per bin over time, so two acoustically different clips of
the same length can share a near-identical *overall* mean while still
differing frame by frame -- exactly the case a summary-statistic seed would
collide on.
"""

from __future__ import annotations

import sys
import zlib

import numpy as np

from echochamber.speakerid.protocol import (
    FrameKind,
    encode_json,
    read_frame,
    write_frame,
)
from echochamber.speakerid.qnn_npu_worker import set_binary_mode


def main() -> int:
    stdin = set_binary_mode(sys.stdin.buffer)
    stdout = set_binary_mode(sys.stdout.buffer)

    write_frame(stdout, FrameKind.READY, encode_json({"fake": True}))

    while True:
        frame = read_frame(stdin)
        if frame is None:
            return 0
        kind, payload = frame
        if kind is FrameKind.EMBED:
            seed = zlib.crc32(payload)
            rng = np.random.default_rng(seed)
            emb = rng.standard_normal(192).astype(np.float32)
            write_frame(stdout, FrameKind.RESULT, emb.tobytes())
        elif kind is FrameKind.SHUTDOWN:
            return 0


if __name__ == "__main__":
    sys.exit(main())
