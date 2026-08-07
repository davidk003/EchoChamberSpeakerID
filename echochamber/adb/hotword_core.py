"""Pure adb logic for blocking/unblocking Google Assistant's "Ok Google"
hotword, with no Qt dependency, so non-GUI callers (e.g. a voice-gated
watch script running in a different venv) can import it without pulling
in PySide6.

Vendored unchanged from the standalone ``EchoChambersADBTrigger`` project
into this repository, so :mod:`echochamber.adb.trigger` can call it
in-process rather than shelling out to a second script.

Revokes (or grants) the Google app's RECORD_AUDIO permission via
`adb shell pm revoke/grant`, which stops (or restores) its low-power
hotword DSP listening. No app install or device setup required.
"""

import glob
import os
import shutil
import subprocess
from pathlib import Path

# Safety net: if the calling process, its adb connection, or the USB/network
# link dies while the hotword is blocked, nothing left running on the PC
# could ever restore it. So every block also queues a detached job on the
# device itself (survives adb/process disconnecting, since it isn't tied to
# the host process) that re-grants the permission after this many seconds. A
# fresh block re-queues a fresh one; an unblock before it fires just makes it
# a harmless no-op later (granting an already-granted permission).
SAFETY_UNBLOCK_SECONDS = 20 * 60

# Google app hosts Assistant's "Ok Google" hotword detection; revoking its
# mic permission stops it from listening.
# Note: this does NOT work for Samsung's Bixby wake word -- its RECORD_AUDIO
# grant is SYSTEM_FIXED and `pm revoke` silently no-ops on it.
HOTWORD_PACKAGE = "com.google.android.googlequicksearchbox"
HOTWORD_PERMISSION = "android.permission.RECORD_AUDIO"


def find_adb() -> str | None:
    """Locate adb.exe even if the caller wasn't launched with a shell PATH."""
    on_path = shutil.which("adb")
    if on_path:
        return on_path

    candidates = []
    for env_var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(env_var)
        if root:
            candidates.append(Path(root) / "platform-tools" / "adb.exe")

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Android" / "Sdk" / "platform-tools" / "adb.exe")
        candidates += [
            Path(p)
            for p in glob.glob(
                str(Path(local_app_data) / "Microsoft" / "WinGet" / "Packages" / "Google.PlatformTools_*" / "platform-tools" / "adb.exe")
            )
        ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def is_hotword_granted(adb: str) -> bool | None:
    """Returns True/False for the Google app's current RECORD_AUDIO grant,
    or None if it couldn't be determined (e.g. no device / adb error)."""
    r = run([adb, "shell", "dumpsys", "package", HOTWORD_PACKAGE])
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if f"{HOTWORD_PERMISSION}: granted" in line:
            return "granted=true" in line
    return None


def set_hotword_blocked(adb: str, blocked: bool) -> bool:
    """Revokes (blocked=True) or grants (blocked=False) the hotword
    permission. Returns success."""
    action = "revoke" if blocked else "grant"
    r = run([adb, "shell", "pm", action, HOTWORD_PACKAGE, HOTWORD_PERMISSION])
    return r.returncode == 0


def schedule_safety_unblock(adb: str) -> bool:
    """Queues a detached on-device job that grants the hotword permission
    back after SAFETY_UNBLOCK_SECONDS, so a block can't outlive the caller.
    `nohup ... &` detaches the job from the adb shell session, so it keeps
    running after this command (and adb itself) disconnects."""
    remote_cmd = (
        f"nohup sh -c 'sleep {SAFETY_UNBLOCK_SECONDS}; "
        f"pm grant {HOTWORD_PACKAGE} {HOTWORD_PERMISSION}' "
        f">/dev/null 2>&1 &"
    )
    r = run([adb, "shell", remote_cmd])
    return r.returncode == 0
