"""Entry point: ``python -m echochamber.app``.

Importing this module must **not** create a :class:`QApplication`.  A
``QApplication`` is a process-wide singleton that grabs the platform plugin,
so creating one at import time would make ``import echochamber.app`` fail on a
headless box and would fight ``pytest-qt``, which owns the application object
in the test process.  Everything happens inside :func:`main`.
"""

from __future__ import annotations

import sys
from typing import Sequence

from PySide6.QtWidgets import QApplication

from echochamber.ui.main_window import MainWindow

__all__ = ["main"]

_window: MainWindow | None = None
"""The live window.

Qt does not own a top-level widget, so without a Python reference the window
would be collected the moment :func:`main` returns -- which matters only on the
path where an event loop already exists and ``main`` returns immediately.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GUI.

    Args:
        argv: Command-line arguments, ``sys.argv`` when ``None``.  Qt consumes
            its own options (``-platform``, ``-style``, ...) from here.

    Returns:
        The Qt exit code, or ``0`` when an application already exists and is
        owned by someone else (a test harness), in which case the window is
        shown but no nested event loop is started.
    """
    global _window

    args = list(sys.argv if argv is None else argv)
    existing = QApplication.instance()
    app = existing if existing is not None else QApplication(args)

    window = MainWindow()
    _window = window
    window.show()

    if existing is not None:
        # Someone else -- pytest-qt, an embedding host -- owns the event loop.
        # Starting a nested one here would hang their process.
        return 0
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
