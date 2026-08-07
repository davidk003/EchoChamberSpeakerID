"""Tests for echochamber.adb.trigger -- no real adb, no real device.

Mirrors ``tests/test_voicegate_notify.py``: the trigger runs its adb calls on
a dedicated background thread, so nothing here sleeps blindly to synchronise
with it. Every observation goes through :func:`wait_until`, which polls a
predicate against a deadline, exactly like that module's helper of the same
name.

``hotword_core.set_hotword_blocked`` and ``hotword_core.schedule_safety_unblock``
are monkeypatched at the point :mod:`echochamber.adb.trigger` calls them --
``echochamber.adb.trigger.hotword_core`` -- so no test ever shells out to a
real ``adb`` binary.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import pytest

from echochamber.adb import hotword_core
from echochamber.adb.config import AdbTriggerConfig
from echochamber.adb.trigger import (
    AdbHotwordTrigger,
    AdbTriggerChoice,
    build_adb_trigger,
)

TIMEOUT = 5.0
ADB_PATH = "C:/fake/adb.exe"


def wait_until(pred: Callable[[], bool], timeout: float = TIMEOUT) -> bool:
    """Poll ``pred`` until it is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


class FakeAdb:
    """Records every ``set_hotword_blocked``/``schedule_safety_unblock`` call.

    Stands in for the real subprocess calls; ``set_result`` scripts whether
    the next ``set_hotword_blocked`` call should report success.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.set_calls: list[tuple[str, bool]] = []
        self.safety_calls: list[str] = []
        self._set_result: bool = True
        self._set_side_effect: Exception | None = None

    def fail_next_set(self) -> None:
        with self.lock:
            self._set_result = False

    def raise_on_set(self, exc: Exception) -> None:
        with self.lock:
            self._set_side_effect = exc

    def set_hotword_blocked(self, adb: str, blocked: bool) -> bool:
        with self.lock:
            self.set_calls.append((adb, blocked))
            if self._set_side_effect is not None:
                exc, self._set_side_effect = self._set_side_effect, None
                raise exc
            result = self._set_result
            self._set_result = True
            return result

    def schedule_safety_unblock(self, adb: str) -> bool:
        with self.lock:
            self.safety_calls.append(adb)
            return True


@pytest.fixture()
def fake_adb(monkeypatch: pytest.MonkeyPatch) -> FakeAdb:
    """Patch the two hotword_core calls the trigger module actually uses."""
    fake = FakeAdb()
    monkeypatch.setattr(
        "echochamber.adb.trigger.hotword_core.set_hotword_blocked",
        fake.set_hotword_blocked,
    )
    monkeypatch.setattr(
        "echochamber.adb.trigger.hotword_core.schedule_safety_unblock",
        fake.schedule_safety_unblock,
    )
    return fake


@pytest.fixture()
def trigger(fake_adb: FakeAdb):
    """A trigger over the fake adb, closed automatically on teardown."""
    t = AdbHotwordTrigger(ADB_PATH)
    yield t
    t.close()


class TestNotifyVerification:
    def test_a_rejected_phrase_blocks_and_schedules_the_safety_net(
        self, trigger: AdbHotwordTrigger, fake_adb: FakeAdb
    ) -> None:
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: fake_adb.set_calls == [(ADB_PATH, True)])
        assert wait_until(lambda: fake_adb.safety_calls == [ADB_PATH])
        assert wait_until(lambda: trigger.snapshot().blocked is True)
        assert trigger.snapshot().block_count == 1

    def test_repeating_the_same_rejection_only_blocks_once(
        self, trigger: AdbHotwordTrigger, fake_adb: FakeAdb
    ) -> None:
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: len(fake_adb.set_calls) == 1)
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: trigger.snapshot().skip_count == 1)
        # give the worker a moment to prove it does NOT issue a second call
        time.sleep(0.05)
        assert fake_adb.set_calls == [(ADB_PATH, True)]

    def test_a_confirmed_speaker_unblocks_without_scheduling_the_safety_net(
        self, trigger: AdbHotwordTrigger, fake_adb: FakeAdb
    ) -> None:
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: trigger.snapshot().blocked is True)

        trigger.notify_verification(matched=True)
        assert wait_until(lambda: fake_adb.set_calls == [(ADB_PATH, True), (ADB_PATH, False)])
        assert wait_until(lambda: trigger.snapshot().blocked is False)
        # Only the one safety call from the earlier block.
        assert fake_adb.safety_calls == [ADB_PATH]

    def test_a_failed_block_records_the_error_and_does_not_flip_believed_state(
        self, trigger: AdbHotwordTrigger, fake_adb: FakeAdb
    ) -> None:
        fake_adb.fail_next_set()
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: trigger.snapshot().last_error is not None)
        stats = trigger.snapshot()
        assert stats.blocked is False
        assert stats.block_count == 0
        assert fake_adb.safety_calls == []

        # A subsequent identical request must be retried, not skipped by the
        # debounce, because the device never actually changed state.
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: fake_adb.set_calls == [(ADB_PATH, True), (ADB_PATH, True)])
        assert wait_until(lambda: trigger.snapshot().blocked is True)
        assert trigger.snapshot().skip_count == 0

    def test_a_raised_exception_is_caught_and_does_not_kill_the_thread(
        self, trigger: AdbHotwordTrigger, fake_adb: FakeAdb
    ) -> None:
        fake_adb.raise_on_set(FileNotFoundError("adb.exe not found"))
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: trigger.snapshot().last_error is not None)
        assert trigger.snapshot().blocked is False

        # The thread must still be alive and able to process a later request.
        trigger.notify_verification(matched=False)
        assert wait_until(lambda: trigger.snapshot().blocked is True)


class TestClose:
    def test_close_stops_the_thread_promptly(self, fake_adb: FakeAdb) -> None:
        t = AdbHotwordTrigger(ADB_PATH)
        t.close()
        assert wait_until(lambda: not t._thread.is_alive(), timeout=2.0)

    def test_close_is_idempotent(self, fake_adb: FakeAdb) -> None:
        t = AdbHotwordTrigger(ADB_PATH)
        t.close()
        t.close()  # must not raise


class TestBuildAdbTrigger:
    def test_disabled_returns_ok_with_no_trigger_and_touches_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            hotword_core, "find_adb", lambda: calls.append("find_adb") or ADB_PATH
        )
        monkeypatch.setattr(
            hotword_core,
            "set_hotword_blocked",
            lambda *a, **k: calls.append("set_hotword_blocked") or True,
        )
        choice = build_adb_trigger(AdbTriggerConfig(enabled=False))
        assert isinstance(choice, AdbTriggerChoice)
        assert choice.ok is True
        assert choice.trigger is None
        assert choice.backend == "none"
        assert calls == []

    def test_enabled_with_no_adb_found_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "echochamber.adb.trigger.hotword_core.find_adb", lambda: None
        )
        choice = build_adb_trigger(AdbTriggerConfig(enabled=True))
        assert choice.ok is False
        assert choice.trigger is None
        assert choice.backend == "none"
        assert choice.error is not None

    def test_enabled_with_explicit_adb_path_skips_autodetection(
        self, monkeypatch: pytest.MonkeyPatch, fake_adb: FakeAdb
    ) -> None:
        def boom() -> str | None:
            raise AssertionError("find_adb must not be called when adb_path is set")

        monkeypatch.setattr("echochamber.adb.trigger.hotword_core.find_adb", boom)
        choice = build_adb_trigger(AdbTriggerConfig(enabled=True, adb_path=ADB_PATH))
        assert choice.ok is True
        assert choice.trigger is not None
        assert choice.backend == "adb"
        choice.trigger.close()
