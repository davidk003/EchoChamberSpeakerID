"""Tests for the echochamber.app entry point.

The import-purity check has to run in a subprocess: once a QApplication exists in
the test process, you can no longer tell whether importing created it. Nothing
else in the suite can catch this, which is why it gets its own file.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_python(source: str) -> subprocess.CompletedProcess[str]:
    """Run a snippet in a clean interpreter with the repo importable."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        cwd=str(REPO_ROOT),
    )


def test_importing_app_does_not_create_a_qapplication() -> None:
    """Importing must be a side-effect-free act.

    A module that builds a QApplication at import time cannot be imported by a
    test runner, a packaging tool, or another app that owns its own instance.
    """
    result = run_python(
        """
        import echochamber.app  # noqa: F401
        from PySide6 import QtWidgets

        instance = QtWidgets.QApplication.instance()
        print("INSTANCE:", instance)
        """
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    assert "INSTANCE: None" in result.stdout, (
        f"importing echochamber.app created a QApplication:\n{result.stdout}"
    )


def test_importing_app_does_not_open_an_audio_device() -> None:
    """PortAudio must stay unloaded until capture actually starts."""
    result = run_python(
        """
        import sys
        import echochamber.app  # noqa: F401

        print("SOUNDDEVICE_LOADED:", "sounddevice" in sys.modules)
        """
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    assert "SOUNDDEVICE_LOADED: False" in result.stdout, (
        f"importing echochamber.app loaded PortAudio:\n{result.stdout}"
    )


def test_audio_package_never_imports_pyside6() -> None:
    """The audio layer must stay usable headless, on a server, and in tests.

    Checked by import rather than by grepping, so an indirect import through a
    helper module is caught too.
    """
    result = run_python(
        """
        import sys
        import echochamber.audio
        import echochamber.audio.pipeline
        import echochamber.audio.chunker
        import echochamber.audio.sinks
        import echochamber.audio.devices
        import echochamber.audio.sources.file_source

        leaked = sorted(m for m in sys.modules if m.startswith("PySide6"))
        print("PYSIDE_MODULES:", leaked)
        """
    )
    assert result.returncode == 0, f"import failed:\n{result.stderr}"
    assert "PYSIDE_MODULES: []" in result.stdout, (
        f"echochamber.audio pulled in PySide6:\n{result.stdout}"
    )


def test_main_is_callable_and_accepts_argv() -> None:
    """main(argv) must exist with the documented shape without being run."""
    import inspect

    from echochamber.app import main

    assert callable(main)
    params = inspect.signature(main).parameters
    assert "argv" in params, "main() must accept argv"
    assert params["argv"].default is not inspect.Parameter.empty, (
        "argv must be optional so `main()` works with no arguments"
    )


def test_app_starts_and_exits_cleanly() -> None:
    """Build the window, pump the event loop, quit -- with no microphone.

    Proves the real entry point wires up without a display or a device, which is
    what a smoke test on CI or the ARM64 box would do.
    """
    result = run_python(
        """
        import sys
        from PySide6 import QtCore, QtWidgets

        from echochamber.ui.main_window import MainWindow

        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        win.show()
        QtCore.QTimer.singleShot(400, app.quit)
        app.exec()
        win.close()
        print("CLEAN_EXIT")
        """
    )
    assert result.returncode == 0, (
        f"the app could not start headless:\n{result.stdout}\n{result.stderr}"
    )
    assert "CLEAN_EXIT" in result.stdout
