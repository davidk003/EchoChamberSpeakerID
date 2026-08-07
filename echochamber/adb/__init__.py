"""Blocking Google Assistant's hotword over adb when speaker verification
rejects a phrase.

Pure adb logic lives in :mod:`echochamber.adb.hotword_core`, vendored from the
standalone ``EchoChambersADBTrigger`` project.  :mod:`echochamber.adb.trigger`
wraps it in a background-thread trigger, mirroring
:class:`~echochamber.voicegate.notify.WebSocketNotifier`'s threading discipline
so the audio/verification path never blocks on an adb subprocess call.

Like :mod:`echochamber.voicegate` and :mod:`echochamber.speakerid`, this
package is *pluggable and absent by default*: with no adb on the machine,
:func:`echochamber.adb.trigger.build_adb_trigger` returns nothing to trigger
with, and a checkout with no Android tooling installed still imports, still
runs the GUI, and still passes the whole test suite.
"""

from __future__ import annotations

from echochamber.adb.config import AdbTriggerConfig, autodetect_adb_trigger_config
from echochamber.adb.trigger import (
    AdbHotwordTrigger,
    AdbTriggerChoice,
    build_adb_trigger,
    describe_backend,
)

__all__ = [
    "AdbHotwordTrigger",
    "AdbTriggerChoice",
    "AdbTriggerConfig",
    "autodetect_adb_trigger_config",
    "build_adb_trigger",
    "describe_backend",
]
