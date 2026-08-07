"""Build the x64 venv and download the model the wake-phrase gate needs.

**Standard library only, deliberately.**  This script bootstraps an
environment, so it has to run before anything is installed -- including this
project.  It also has to run under an interpreter that is *not* the
application's: on the ARM64 target it is pointed at an x64 ``python.exe`` that
has nothing in it at all.  A single ``import requests`` here would make the
setup step depend on a setup step.

**Why any of this is necessary.**  Vosk publishes no ``win_arm64`` wheel, and
the deployment target is Windows on ARM64, so the application's own native
interpreter cannot install it.  The gate works around that by running the
decoder in a child process under a second, **x64** interpreter, which Windows
11's Prism emulator executes transparently; see
:mod:`echochamber.voicegate.subprocess_recognizer` for the full reasoning and
for why the real-time audio path stays native.  This script builds that second
environment and prints the two strings the user then puts into
``VoiceGateConfig``: ``worker_python`` and ``model_path``.

**The ARM64 trap this script exists to prevent.**  On an ARM64 machine the
obvious thing to do -- run this with the Python already on ``PATH`` -- creates
an ARM64 venv, and ``pip install vosk`` into it fails with a wheel-not-found
error that says nothing about architectures.  So the architecture of the chosen
interpreter is checked up front and the mismatch is reported as the actual
problem, with the actual fix: install the "Windows installer (64-bit)" build
from python.org and pass its ``python.exe`` to ``--python``.

The model is ``vosk-model-small-en-us-0.15``: about 40 MB, Apache-2.0, and
accurate enough for a fixed wake-phrase grammar, which is all the gate asks of
it.  It is downloaded rather than vendored because a 40 MB binary blob has no
business in a git history.

Usage::

    python scripts/setup_voice_gate.py
    python scripts/setup_voice_gate.py --python "C:/Python312-x64/python.exe"
    python scripts/setup_voice_gate.py --skip-venv     # model only
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from typing import Sequence

__all__ = [
    "MODEL_NAME",
    "MODEL_URL",
    "build_parser",
    "create_venv",
    "download_model",
    "interpreter_machine",
    "main",
    "venv_python",
]

MODEL_NAME: str = "vosk-model-small-en-us-0.15"
"""The Vosk model this script installs: small English, Apache-2.0, ~40 MB."""

MODEL_URL: str = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
"""Where that model is published."""

_ARM64_NAMES: frozenset[str] = frozenset({"ARM64", "AARCH64"})
"""Machine strings that mean 64-bit ARM.

Windows reports ``ARM64``; Linux and macOS report ``aarch64`` and ``arm64``.
All of them are normalised to upper case before being looked up here.
"""

_CHUNK: int = 64 * 1024
"""Download read size.  Large enough that the progress line is not the bottleneck."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        A parser accepting ``--venv-dir``, ``--model-dir``, ``--python``,
        ``--skip-model`` and ``--skip-venv``.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/setup_voice_gate.py",
        description=(
            "Create the recogniser venv and download the Vosk model used by "
            "the wake-phrase gate."
        ),
        epilog=(
            "On Windows ARM64, pass --python pointing at an x64 python.exe: "
            "vosk ships no ARM64 wheel, and the x64 worker runs fine under "
            "emulation."
        ),
    )
    parser.add_argument(
        "--venv-dir",
        default=".venv-vosk",
        help="directory for the recogniser virtual environment "
        "(default: .venv-vosk)",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="directory the model is unpacked into (default: models)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        dest="python",
        help=(
            "interpreter the venv is built from; on Windows ARM64 this must be "
            "an x64 python.exe (default: the interpreter running this script)"
        ),
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="do not download the model",
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="do not create the venv or install vosk",
    )
    parser.add_argument(
        "--trusted-host",
        action="append",
        dest="trusted_hosts",
        default=[],
        metavar="HOST",
        help=(
            "pass-through to `pip install --trusted-host`; repeat for "
            "multiple hosts. Needed when pip's TLS verification fails with "
            "CERTIFICATE_VERIFY_FAILED / self-signed certificate in chain -- "
            "typically a corporate TLS-inspecting proxy. Try "
            "--trusted-host pypi.org --trusted-host files.pythonhosted.org"
        ),
    )
    return parser


def venv_python(venv_dir: str) -> str:
    """Return the interpreter path inside a virtual environment.

    Args:
        venv_dir: The environment's root directory.

    Returns:
        ``<venv>/Scripts/python.exe`` on Windows, ``<venv>/bin/python``
        elsewhere.  Computed from :data:`os.name` rather than probed, so the
        path can be reported before the environment exists.
    """
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def interpreter_machine(python: str) -> str:
    """Ask ``python`` what machine architecture it was built for.

    Asking the interpreter is the only honest way to find this out.  The
    architecture that matters is the *child's*, not this script's, and on
    Windows ARM64 an x64 interpreter under emulation reports ``AMD64`` from
    inside itself -- which is exactly the answer needed, and one no amount of
    inspecting the file path would produce.

    Args:
        python: Path to the interpreter to interrogate.

    Returns:
        The upper-cased ``platform.machine()`` of that interpreter, or an empty
        string if it could not be run at all (a bad path, no execute
        permission).  Emptiness is not treated as an error here: the venv step
        will fail with a far better message than this check could give.
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


def check_architecture(python: str) -> bool:
    """Warn if ``python`` cannot possibly install Vosk on this machine.

    Args:
        python: The interpreter ``--python`` selected.

    Returns:
        ``True`` if the selection looks workable, ``False`` if this is an ARM64
        host and ``python`` is an ARM64 build -- in which case
        ``pip install vosk`` is guaranteed to fail, and saying so now is worth
        far more than the wheel-not-found error the user would otherwise get
        several minutes later.
    """
    host = platform.machine().upper()
    if host not in _ARM64_NAMES:
        return True

    print()
    print(f"Detected an ARM64 host (platform.machine() = {platform.machine()!r}).")
    print(
        "Vosk publishes no ARM64 wheel, so the recogniser must run under an\n"
        "x64 interpreter. On Windows 11 the Prism emulator runs that x64\n"
        "process transparently; only the Kaldi decode is emulated, while audio\n"
        "capture and the real-time callback stay native ARM64."
    )

    target = interpreter_machine(python)
    if target in _ARM64_NAMES:
        print()
        print(f"!! {python} is itself an ARM64 build (reports {target}).")
        print("!! `pip install vosk` into a venv made from it WILL fail.")
        print("!!")
        print("!! Install the x64 build of Python from python.org -- the")
        print('!! "Windows installer (64-bit)" download, not the ARM64 one --')
        print("!! and re-run this script pointing at it, e.g.:")
        print("!!")
        print(
            '!!   python scripts/setup_voice_gate.py --python '
            '"C:/Python312-x64/python.exe"'
        )
        return False

    if target:
        print(f"Using {python} (reports {target}) -- good.")
    return True


def create_venv(
    python: str, venv_dir: str, trusted_hosts: Sequence[str] = ()
) -> str:
    """Create ``venv_dir`` from ``python`` and install Vosk into it.

    The venv is created by *running the target interpreter*, not by calling
    :mod:`venv` in-process: the environment must inherit the target's
    architecture, and an in-process call would build one for whichever
    interpreter happens to be running this script.

    An existing environment is reused rather than rebuilt; ``pip install`` is
    still run, so this doubles as an upgrade path.

    Args:
        python: Interpreter to build the environment from.
        venv_dir: Directory for the environment.
        trusted_hosts: Hosts passed to ``pip install --trusted-host``, for
            networks where pip's TLS verification fails against
            ``pypi.org`` -- typically a corporate proxy that terminates TLS
            with its own certificate, which is not in this interpreter's
            trust store.

    Returns:
        Path to the interpreter inside the new environment -- the value to
        configure as ``VoiceGateConfig.worker_python``.

    Raises:
        RuntimeError: If creating the environment or installing Vosk fails, or
            if the environment appeared without an interpreter in it.
    """
    target = venv_python(venv_dir)
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

    trust_args: list[str] = []
    for host in trusted_hosts:
        trust_args += ["--trusted-host", host]

    print("Installing vosk and numpy (this pulls ~50 MB of wheels) ...")
    _run(
        [target, "-m", "pip", "install", "--upgrade", "pip", *trust_args],
        "upgrade pip",
    )
    # numpy is not needed to *decode* audio, but importing
    # `echochamber.voicegate.worker` pulls in the whole `echochamber` package
    # tree (voicegate.config -> echochamber.config -> echochamber.audio,
    # which imports numpy at module scope), so the worker cannot start
    # without it. This venv is always x64 -- unlike the main ARM64
    # environment -- so the ordinary win_amd64 wheel applies with no
    # architecture caveat.
    _run(
        [target, "-m", "pip", "install", "vosk", "numpy>=2.3", *trust_args],
        "install vosk and numpy",
    )
    return target


def download_model(model_dir: str, url: str = MODEL_URL) -> str:
    """Download and unpack the Vosk model into ``model_dir``.

    Skipped entirely if the unpacked directory already exists, so re-running
    this script is cheap.  The download lands in a ``.part`` file that is only
    renamed once it is complete and :func:`zipfile.is_zipfile` agrees it is an
    archive -- a truncated download that got extracted anyway would produce a
    model directory that looks present and fails at load time, days later.

    Args:
        model_dir: Directory the model is unpacked into; created if missing.
        url: Archive to fetch.

    Returns:
        Path to the unpacked model directory -- the value to configure as
        ``VoiceGateConfig.model_path``.

    Raises:
        RuntimeError: If the download fails, the archive is not a valid zip, or
            the archive did not contain the expected directory.
    """
    os.makedirs(model_dir, exist_ok=True)
    destination = os.path.join(model_dir, MODEL_NAME)
    if os.path.isdir(destination):
        print(f"Model already present at {destination}")
        return destination

    archive = os.path.join(model_dir, f"{MODEL_NAME}.zip")
    partial = f"{archive}.part"
    print(f"Downloading {url}")
    try:
        _fetch(url, partial)
        if not zipfile.is_zipfile(partial):
            raise RuntimeError(
                f"the download from {url} is not a zip archive; the server "
                f"probably returned an error page. Delete {partial!r} and "
                f"retry, or fetch the model by hand."
            )
        os.replace(partial, archive)
    except BaseException:
        # Never leave a half-written file behind: `is_zipfile` would reject it
        # next time, but a stray 12 MB .part in the repo is nobody's friend.
        _unlink(partial)
        raise

    print(f"Extracting into {model_dir} ...")
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(model_dir)
    finally:
        # The archive is 40 MB of no further use once unpacked, and keeping it
        # around invites someone to commit it.
        _unlink(archive)

    if not os.path.isdir(destination):
        raise RuntimeError(
            f"the archive unpacked but {destination!r} is missing; the model "
            f"layout may have changed upstream"
        )
    print(f"Model ready at {destination}")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    """Run the setup steps and print what to configure.

    Args:
        argv: Argument list *without* the program name, or ``None`` to take
            ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` if a step failed, ``2`` if the selected
        interpreter is an ARM64 build on an ARM64 host (nothing was attempted;
        see :func:`check_architecture`).
    """
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))

    worker_python = venv_python(args.venv_dir)
    model_path = os.path.join(args.model_dir, MODEL_NAME)

    if not args.skip_venv and not check_architecture(args.python):
        return 2

    try:
        if args.skip_venv:
            print("Skipping the virtual environment (--skip-venv).")
        else:
            worker_python = create_venv(
                args.python, args.venv_dir, args.trusted_hosts
            )

        if args.skip_model:
            print("Skipping the model download (--skip-model).")
        else:
            model_path = download_model(args.model_dir)
    except (RuntimeError, OSError) as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1

    _print_summary(os.path.abspath(worker_python), os.path.abspath(model_path))
    return 0


def _print_summary(worker_python: str, model_path: str) -> None:
    """Print the two values the user has to put into ``VoiceGateConfig``.

    Args:
        worker_python: Absolute path of the interpreter in the new venv.
        model_path: Absolute path of the unpacked model directory.
    """
    print()
    print("=" * 72)
    print("Voice gate is set up. Configure it with:")
    print()
    print("    VoiceGateConfig(")
    print("        enabled=True,")
    # `repr` already escapes backslashes, so a Windows path pastes back in
    # correctly without an `r` prefix -- which would in fact break it, by
    # keeping the doubled backslashes literal.
    print(f"        model_path={model_path!r},")
    print(f"        worker_python={worker_python!r},")
    print("    )")
    print()
    print("worker_python may be left as None on a platform where `pip install")
    print("vosk` works in the application's own interpreter; the subprocess")
    print("backend exists for Windows ARM64, where it does not.")
    print("=" * 72)


def _fetch(url: str, destination: str) -> None:
    """Download ``url`` to ``destination``, printing progress.

    Args:
        url: What to fetch.
        destination: Where to write it.

    Raises:
        RuntimeError: If the server refused the request or the transfer broke.
    """
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URL
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(destination, "wb") as handle:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    _progress(done, total)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"could not download {url}: {exc}") from exc
    finally:
        # The progress line has no newline of its own; without this the next
        # message is appended to it.
        print()
    print(f"Downloaded {done / 1e6:.1f} MB")


def _progress(done: int, total: int) -> None:
    """Redraw the one-line download progress indicator.

    Args:
        done: Bytes received so far.
        total: Expected total, or ``0`` when the server sent no
            ``Content-Length`` -- in which case only the running total is
            shown, rather than a percentage of an unknown quantity.
    """
    if total > 0:
        percent = 100.0 * done / total
        line = f"  {done / 1e6:6.1f} / {total / 1e6:.1f} MB ({percent:5.1f}%)"
    else:
        line = f"  {done / 1e6:6.1f} MB"
    sys.stdout.write(f"\r{line}")
    sys.stdout.flush()


def _run(command: list[str], what: str) -> None:
    """Run ``command``, raising a readable error if it fails.

    Args:
        command: Argument vector to execute.
        what: Short description used in the error message.

    Raises:
        RuntimeError: If the command could not be launched or exited non-zero.
    """
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        raise RuntimeError(f"could not {what}: {exc} (running {command[0]!r})") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"could not {what}: `{' '.join(command)}` exited with "
            f"{completed.returncode}"
        )


def _unlink(path: str) -> None:
    """Delete ``path`` if it exists, ignoring failures.

    Args:
        path: File to remove.
    """
    try:
        os.remove(path)
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
