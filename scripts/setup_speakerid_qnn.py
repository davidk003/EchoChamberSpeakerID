"""Build the two venvs the QNN speaker-verification backend needs.

**Why two venvs.** PyTorch has no Windows ARM64 wheel, but
``onnxruntime-qnn``'s Hexagon HTP backend only loads inside a native ARM64
process -- see :mod:`echochamber.speakerid.qnn_driver_worker`'s docstring for
the full reasoning. This script builds:

* An **x64** venv (``.venv-speakerid-x64`` by default) with
  ``torch``/``torchaudio`` (feature extraction) plus ``onnxruntime`` (the
  driver process talks the same frame protocol either way, but needs *some*
  onnxruntime import path available) and ``onnx``/``onnxsim``/
  ``huggingface_hub`` (needed by ``scripts/export_speakerid_qnn.py``, the
  one-time export step run manually in this same venv).
* A native **ARM64** venv (``.venv-speakerid-arm64`` by default) with
  ``onnxruntime-qnn``/``numpy`` and no torch -- this is where the actual NPU
  inference happens.

This script is standard-library only, for the same reason
``scripts/setup_voice_gate.py`` is: it bootstraps environments before
anything is installed, and it has to run under interpreters that are *not*
the application's own.

Usage::

    python scripts/setup_speakerid_qnn.py --arm64-python "C:/Python311-arm64/python.exe"
    python scripts/setup_speakerid_qnn.py --skip-arm64      # x64 venv only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Sequence

__all__ = [
    "build_parser",
    "check_arm64_interpreter",
    "create_x64_venv",
    "create_arm64_venv",
    "main",
    "venv_python",
]

_ARM64_NAMES: frozenset[str] = frozenset({"ARM64", "AARCH64"})
"""Machine strings that mean 64-bit ARM; see scripts/setup_voice_gate.py."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        A parser accepting ``--x64-venv-dir``, ``--arm64-venv-dir``,
        ``--x64-python``, ``--arm64-python``, ``--skip-x64``,
        ``--skip-arm64`` and ``--trusted-host``.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/setup_speakerid_qnn.py",
        description=(
            "Create the two virtual environments the QNN speaker-"
            "verification backend needs: an x64 one with torch (feature "
            "extraction), and a native ARM64 one with onnxruntime-qnn (NPU "
            "inference)."
        ),
        epilog=(
            "On an ARM64 host, --x64-python must point at an x64 python.exe: "
            "torch has no ARM64 wheel, but Windows 11's Prism emulator runs "
            "that x64 process fine under emulation. --arm64-python must be a "
            "native ARM64 build, since the HTP backend only loads inside one."
        ),
    )
    parser.add_argument(
        "--x64-venv-dir",
        default=".venv-speakerid-x64",
        help="directory for the x64 driver environment (default: .venv-speakerid-x64)",
    )
    parser.add_argument(
        "--arm64-venv-dir",
        default=".venv-speakerid-arm64",
        help="directory for the ARM64 NPU-inference environment (default: .venv-speakerid-arm64)",
    )
    parser.add_argument(
        "--x64-python",
        default=sys.executable,
        help="x64 interpreter the driver venv is built from (default: the interpreter running this script)",
    )
    parser.add_argument(
        "--arm64-python",
        default=None,
        help="native ARM64 interpreter the NPU venv is built from; required unless --skip-arm64",
    )
    parser.add_argument(
        "--skip-x64",
        action="store_true",
        help="do not create the x64 driver venv",
    )
    parser.add_argument(
        "--skip-arm64",
        action="store_true",
        help="do not create the ARM64 NPU-inference venv",
    )
    parser.add_argument(
        "--trusted-host",
        action="append",
        dest="trusted_hosts",
        default=[],
        metavar="HOST",
        help=(
            "pass-through to `pip install --trusted-host`; repeat for "
            "multiple hosts. See scripts/setup_voice_gate.py for when this "
            "is needed."
        ),
    )
    return parser


def venv_python(venv_dir: str) -> str:
    """Return the interpreter path inside a virtual environment."""
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def _interpreter_machine(python: str) -> str:
    """Ask ``python`` what machine architecture it was built for.

    See ``scripts/setup_voice_gate.py.interpreter_machine`` for why asking
    the interpreter beats inspecting its path.
    """
    try:
        completed = subprocess.run(
            [python, "-c", "import platform; print(platform.machine())"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip().upper()


def check_arm64_interpreter(arm64_python: str) -> bool:
    """Warn and fail if ``arm64_python`` is not actually an ARM64 build.

    Args:
        arm64_python: The interpreter ``--arm64-python`` selected.

    Returns:
        ``True`` if it reports as ARM64, ``False`` otherwise -- in which case
        the QNN HTP backend will fail to load with an error that says nothing
        about architecture mismatches.
    """
    target = _interpreter_machine(arm64_python)
    if target in _ARM64_NAMES:
        print(f"Using {arm64_python} (reports {target}) -- good.")
        return True
    print()
    print(f"!! {arm64_python} does not report an ARM64 architecture (reports {target!r}).")
    print("!! onnxruntime-qnn's HTP backend only loads inside a *native* ARM64")
    print("!! process, so this venv would be unable to reach the Hexagon NPU.")
    print("!! Install a native ARM64 Python build, e.g.:")
    print('!!   winget install --id Python.Python.3.11 -e --architecture arm64')
    return False


def create_x64_venv(
    python: str, venv_dir: str, trusted_hosts: Sequence[str] = ()
) -> str:
    """Create the x64 driver environment and install its dependencies.

    Args:
        python: x64 interpreter to build the environment from.
        venv_dir: Directory for the environment.
        trusted_hosts: Hosts passed to ``pip install --trusted-host``.

    Returns:
        Path to the interpreter inside the new environment -- the value to
        configure as ``SpeakerIdConfig.qnn_worker_python``.

    Raises:
        RuntimeError: If creating the environment or installing dependencies
            fails.
    """
    target = venv_python(venv_dir)
    trust_args: list[str] = []
    for host in trusted_hosts:
        trust_args += ["--trusted-host", host]

    if os.path.isfile(target):
        print(f"Reusing the existing environment at {venv_dir}")
    else:
        print(f"Creating {venv_dir} from {python} ...")
        _run([python, "-m", "venv", venv_dir], "create the virtual environment")
        if not os.path.isfile(target):
            raise RuntimeError(
                f"the virtual environment at {venv_dir!r} was created but has "
                f"no interpreter at {target!r}"
            )

    print(
        "Installing torch, torchaudio, soundfile, onnxruntime, onnx, "
        "onnxsim, huggingface_hub, numpy ..."
    )
    _run([target, "-m", "pip", "install", "--upgrade", "pip", *trust_args], "upgrade pip")
    _run(
        [
            target,
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch",
            "torchaudio",
            *trust_args,
        ],
        "install torch and torchaudio",
    )
    _run(
        [
            target,
            "-m",
            "pip",
            "install",
            "soundfile",
            "onnxruntime",
            "onnx",
            "onnxsim",
            "huggingface_hub",
            "numpy>=2.3",
            *trust_args,
        ],
        "install soundfile, onnxruntime, onnx, onnxsim, huggingface_hub, and numpy",
    )
    return target


def create_arm64_venv(
    python: str, venv_dir: str, trusted_hosts: Sequence[str] = ()
) -> str:
    """Create the native ARM64 NPU-inference environment and install its dependencies.

    Args:
        python: Native ARM64 interpreter to build the environment from.
        venv_dir: Directory for the environment.
        trusted_hosts: Hosts passed to ``pip install --trusted-host``.

    Returns:
        Path to the interpreter inside the new environment -- the value to
        configure as ``SpeakerIdConfig.qnn_npu_python``.

    Raises:
        RuntimeError: If creating the environment or installing dependencies
            fails.
    """
    target = venv_python(venv_dir)
    trust_args: list[str] = []
    for host in trusted_hosts:
        trust_args += ["--trusted-host", host]

    if os.path.isfile(target):
        print(f"Reusing the existing environment at {venv_dir}")
    else:
        print(f"Creating {venv_dir} from {python} ...")
        _run([python, "-m", "venv", venv_dir], "create the virtual environment")
        if not os.path.isfile(target):
            raise RuntimeError(
                f"the virtual environment at {venv_dir!r} was created but has "
                f"no interpreter at {target!r}"
            )

    print("Installing onnxruntime-qnn and numpy (no torch needed here) ...")
    _run([target, "-m", "pip", "install", "--upgrade", "pip", *trust_args], "upgrade pip")
    _run(
        [target, "-m", "pip", "install", "onnxruntime-qnn", "numpy>=2.3", *trust_args],
        "install onnxruntime-qnn and numpy",
    )
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Run the setup steps and print what to configure.

    Args:
        argv: Argument list *without* the program name, or ``None`` to take
            ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` if a step failed, ``2`` if ``--arm64-python``
        does not report as ARM64 (nothing was attempted for that venv).
    """
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))

    x64_python_out: str | None = None
    arm64_python_out: str | None = None

    try:
        if args.skip_x64:
            print("Skipping the x64 driver environment (--skip-x64).")
        else:
            x64_python_out = create_x64_venv(
                args.x64_python, args.x64_venv_dir, args.trusted_hosts
            )

        if args.skip_arm64:
            print("Skipping the ARM64 NPU-inference environment (--skip-arm64).")
        else:
            if not args.arm64_python:
                print(
                    "\n--arm64-python is required unless --skip-arm64 is given "
                    "(no interpreter to build the ARM64 venv from).",
                    file=sys.stderr,
                )
                return 2
            if not check_arm64_interpreter(args.arm64_python):
                return 2
            arm64_python_out = create_arm64_venv(
                args.arm64_python, args.arm64_venv_dir, args.trusted_hosts
            )
    except (RuntimeError, OSError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(x64_python_out, arm64_python_out)
    return 0


def _print_summary(x64_python: str | None, arm64_python: str | None) -> None:
    """Print the values the user has to put into ``SpeakerIdConfig``."""
    print()
    print("=" * 72)
    print("Speaker-verification venvs are set up.")
    print()
    if x64_python:
        print(f"    qnn_worker_python={os.path.abspath(x64_python)!r},")
    if arm64_python:
        print(f"    qnn_npu_python={os.path.abspath(arm64_python)!r},")
    print()
    print("Next: export the ONNX graph (run in the x64 venv, needs torch):")
    if x64_python:
        print(f"    {x64_python} scripts/export_speakerid_qnn.py")
    else:
        print("    <x64 venv python> scripts/export_speakerid_qnn.py")
    print()
    print("Then enroll at least one speaker:")
    print("    python scripts/enroll_speaker.py enroll <name>")
    print("=" * 72)


def _run(command: list[str], what: str) -> None:
    """Run ``command``, raising a readable error if it fails."""
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        raise RuntimeError(f"could not {what}: {exc} (running {command[0]!r})") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"could not {what}: `{' '.join(command)}` exited with "
            f"{completed.returncode}"
        )


if __name__ == "__main__":
    sys.exit(main())
