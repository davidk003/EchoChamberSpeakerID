"""Enroll, list, and remove speakers for the QNN speaker-verification backend.

Records a clip from the default microphone, embeds it via the QNN subprocess
chain (:mod:`echochamber.speakerid.qnn_subprocess`), and stores the result in
the enrollment database
(:mod:`echochamber.speakerid.enrollment`) -- this repository's own
``enrolled_speakers.json``, independent of the sibling ``cam-script``
repository's file of the same name.

Usage::

    python scripts/enroll_speaker.py enroll alice --seconds 5
    python scripts/enroll_speaker.py list
    python scripts/enroll_speaker.py remove alice
    python scripts/enroll_speaker.py clear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echochamber.speakerid.backends import build_embedder  # noqa: E402
from echochamber.speakerid.config import (  # noqa: E402
    SpeakerIdConfig,
    autodetect_speaker_id_config,
)
from echochamber.speakerid.enrollment import (  # noqa: E402
    BACKEND_QNN,
    clear_db,
    enroll,
    load_db,
    remove_speaker,
    save_db,
)

_SAMPLE_RATE = 16_000


def _record(seconds: float, sample_rate: int = _SAMPLE_RATE):
    """Record mono float32 audio from the default input device.

    Args:
        seconds: Clip length.
        sample_rate: Sample rate to record at.

    Returns:
        A 1-D ``float32`` numpy array.
    """
    import sounddevice as sd

    print(f"Recording {seconds:g}s from the default microphone ...")
    audio = sd.rec(
        int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    print("Done recording.")
    return audio[:, 0]


def _config_from_args(args: argparse.Namespace) -> SpeakerIdConfig:
    """Build a :class:`SpeakerIdConfig` from CLI overrides, autodetecting the rest."""
    overrides: dict[str, object] = {}
    if args.db_path is not None:
        overrides["db_path"] = args.db_path
    if args.onnx_path is not None:
        overrides["qnn_onnx_path"] = args.onnx_path
    if args.worker_python is not None:
        overrides["qnn_worker_python"] = args.worker_python
    if args.npu_python is not None:
        overrides["qnn_npu_python"] = args.npu_python
    return autodetect_speaker_id_config(**overrides)


def cmd_enroll(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    if not config.qnn_onnx_path or not config.qnn_worker_python:
        print(
            "error: qnn_onnx_path/qnn_worker_python are not configured. Run "
            "`python scripts/setup_speakerid_qnn.py` and "
            "`python scripts/export_speakerid_qnn.py` first, or pass "
            "--onnx-path/--worker-python explicitly.",
            file=sys.stderr,
        )
        return 1

    samples = _record(args.seconds)

    print("Embedding via the QNN subprocess chain (this can take a while on first run)...")
    embedder = build_embedder(config)
    try:
        embedding = embedder.embed(samples, _SAMPLE_RATE)
    finally:
        embedder.close()

    db = load_db(config.db_path)
    enroll(db, args.name, embedding, BACKEND_QNN)
    save_db(config.db_path, db)
    print(f"Enrolled {args.name!r} in {config.db_path!r}.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    db = load_db(config.db_path)
    if not db:
        print(f"No speakers enrolled in {config.db_path!r}.")
        return 0
    print(f"Speakers enrolled in {config.db_path!r}:")
    for name, entry in db.items():
        print(f"  {name} (backend={entry.get('backend')!r})")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    db = load_db(config.db_path)
    if not remove_speaker(db, args.name):
        print(f"error: {args.name!r} is not enrolled in {config.db_path!r}", file=sys.stderr)
        return 1
    save_db(config.db_path, db)
    print(f"Removed {args.name!r} from {config.db_path!r}.")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    clear_db(config.db_path)
    print(f"Cleared {config.db_path!r}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll, list, and remove speakers for the QNN speaker-verification backend."
    )
    parser.add_argument("--db-path", default=None, help="enrollment database path (default: autodetected)")
    parser.add_argument("--onnx-path", default=None, help="exported ONNX graph path (default: autodetected)")
    parser.add_argument("--worker-python", default=None, help="x64 driver interpreter (default: autodetected)")
    parser.add_argument("--npu-python", default=None, help="native ARM64 NPU-worker interpreter (default: driver's own default)")
    sub = parser.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("enroll", help="enroll a speaker from a mic recording")
    pe.add_argument("name")
    pe.add_argument("--seconds", type=float, default=5.0)
    pe.set_defaults(func=cmd_enroll)

    pl = sub.add_parser("list", help="list enrolled speakers")
    pl.set_defaults(func=cmd_list)

    pr = sub.add_parser("remove", help="remove one enrolled speaker")
    pr.add_argument("name")
    pr.set_defaults(func=cmd_remove)

    pc = sub.add_parser("clear", help="delete the entire enrollment database")
    pc.set_defaults(func=cmd_clear)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
