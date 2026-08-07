"""Blocking Google Assistant's hotword when speaker verification rejects a
phrase, and unblocking it when the enrolled speaker is confirmed.

**The threading discipline mirrors
:class:`~echochamber.voicegate.notify.WebSocketNotifier` exactly, for the same
reason.** :meth:`AdbHotwordTrigger.notify_verification` is called from the
audio consumer thread -- the same thread
:meth:`~echochamber.voicegate.notify.WebSocketNotifier.notify` is called
from -- the moment a phrase is matched or rejected. An ``adb shell pm
revoke/grant`` round trip is a subprocess call over USB or TCP; it can take
anywhere from tens of milliseconds to a full adb timeout, and none of that may
ever be spent on the thread draining the pipeline's bounded queue. So
``notify_verification`` only enqueues a desired state and returns; a dedicated
``adb-hotword-trigger`` thread does the actual ``adb`` calls, coalescing a
burst of requests down to the newest one exactly as
:meth:`WebSocketNotifier._take` drains its deque for the freshest event.

**What this buys:** a stranger who speaks the wake phrase gets rejected by
speaker verification, and the trigger revokes the Google app's
``RECORD_AUDIO`` permission (see :mod:`echochamber.adb.hotword_core`) so
"Ok Google" stops listening at all -- not just this app, the device's own
Assistant. When the enrolled owner speaks the phrase and verification
confirms it, the trigger grants the permission back.

**The starting state is "believed unblocked", and nothing touches adb until
the first verification result arrives.** This matches the real world: a
freshly granted or never-revoked Google app has its microphone permission on
by default, so assuming "unblocked" costs nothing if that assumption is
correct, and self-corrects the moment it isn't -- the first rejected phrase
still blocks, whether or not the device was actually already blocked from a
previous run. Calling adb at construction time would also mean building an
:class:`AdbHotwordTrigger` -- which :func:`build_adb_trigger` does whenever
the feature is enabled, whether or not a phrase has ever been heard -- always
pays for a subprocess round trip before there is anything to react to.

**A failed block or unblock must not update the believed state.** If it did,
a real subsequent request for the same desired state would be silently
swallowed by the debounce despite the device never having actually changed --
the one failure mode this trigger cannot allow, since it is the difference
between "the stranger's hotword is blocked" and "the code believes it is
blocked while the phone quietly disagrees."
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from echochamber.adb import hotword_core
from echochamber.adb.config import AdbTriggerConfig

__all__ = [
    "AdbHotwordTrigger",
    "AdbTriggerChoice",
    "AdbTriggerStats",
    "build_adb_trigger",
    "describe_backend",
]

_JOIN_TIMEOUT_S: float = 2.0
"""How long :meth:`AdbHotwordTrigger.close` waits for the worker thread.

Bounded for the same reason :class:`~echochamber.voicegate.notify.WebSocketNotifier`'s
``_JOIN_TIMEOUT_S`` is: shutdown must not hang on a wedged adb subprocess."""

_POLL_S: float = 0.1
"""How often the worker thread re-checks for a stop request while idle."""

_QUEUE_MAX: int = 8
"""Pending desired-states buffered before :meth:`AdbHotwordTrigger.notify_verification`
starts silently dropping.  Small on purpose: these are rare boolean events --
one per verification result -- and a backlog of stale ones is worth nothing,
since only the newest is ever acted on."""


@dataclass(frozen=True, slots=True)
class AdbTriggerStats:
    """A detached copy of an :class:`AdbHotwordTrigger`'s counters.

    Frozen and returned by :meth:`AdbHotwordTrigger.snapshot` for the same
    reason :meth:`~echochamber.voicegate.notify.WebSocketNotifier.snapshot`
    is: a GUI polling from another thread must never see two fields that
    together describe two different instants.

    Attributes:
        blocked: The trigger's current believed state of the device's
            hotword permission -- ``True`` if it last successfully revoked
            it, ``False`` if it last successfully granted it or has never
            acted.
        block_count: Successful blocks issued.
        unblock_count: Successful unblocks issued.
        skip_count: Requests that matched the already-believed state and so
            triggered no adb call at all.
        last_error: The most recent failure -- from ``adb`` itself, from
            scheduling the safety unblock, or from a raised exception -- or
            ``None`` when the last action that ran succeeded.
    """

    blocked: bool = False
    block_count: int = 0
    unblock_count: int = 0
    skip_count: int = 0
    last_error: str | None = None

    def __repr__(self) -> str:
        """Return a debugging representation of the counters that matter."""
        return (
            f"{type(self).__name__}(blocked={self.blocked}, "
            f"block_count={self.block_count}, unblock_count={self.unblock_count}, "
            f"skip_count={self.skip_count})"
        )


class AdbHotwordTrigger:
    """Blocks or unblocks the device hotword from a dedicated background thread.

    ``adb_path`` is injected rather than autodetected here -- see
    :func:`build_adb_trigger`, which is what calls
    :func:`~echochamber.adb.hotword_core.find_adb` -- so this class stays
    simple and testable: given a path (real or fake), it does exactly one
    thing, and nothing about *finding* adb needs to be exercised to test it.
    """

    __slots__ = (
        "_adb_path",
        "_queue",
        "_thread",
        "_stop_event",
        "_closed",
        "_believed_blocked",
        "_lock",
        "_blocked",
        "_block_count",
        "_unblock_count",
        "_skip_count",
        "_last_error",
    )

    def __init__(self, adb_path: str | None = None) -> None:
        """Start the worker thread; no adb call is made yet.

        Args:
            adb_path: Path to ``adb.exe``, already resolved by the caller.
                May be ``None`` in tests that never expect an adb call to
                actually run, but a real trigger built by
                :func:`build_adb_trigger` always has one.
        """
        self._adb_path: str | None = adb_path
        self._queue: "queue.Queue[bool]" = queue.Queue(maxsize=_QUEUE_MAX)
        self._stop_event: threading.Event = threading.Event()
        self._closed: bool = False

        # Only touched on the background thread; see the module docstring for
        # why this starts as "believed unblocked" and why a failed call must
        # never update it.
        self._believed_blocked: bool = False

        self._lock: threading.Lock = threading.Lock()
        self._blocked: bool = False
        self._block_count: int = 0
        self._unblock_count: int = 0
        self._skip_count: int = 0
        self._last_error: str | None = None

        self._thread: threading.Thread = threading.Thread(
            target=self._run, name="adb-hotword-trigger", daemon=True
        )
        self._thread.start()

    @property
    def adb_path(self) -> str | None:
        """The adb executable this trigger calls."""
        return self._adb_path

    def snapshot(self) -> AdbTriggerStats:
        """Return a detached copy of every counter.

        Returns:
            The counters as of this moment.
        """
        with self._lock:
            return AdbTriggerStats(
                blocked=self._blocked,
                block_count=self._block_count,
                unblock_count=self._unblock_count,
                skip_count=self._skip_count,
                last_error=self._last_error,
            )

    def notify_verification(self, matched: bool) -> None:
        """Record a verification result.  Never blocks, never raises.

        Called from the audio consumer thread the moment speaker
        verification accepts or rejects a matched phrase; see the module
        docstring for why nothing else may happen here.

        Args:
            matched: ``True`` if the enrolled speaker was confirmed (hotword
                should be unblocked), ``False`` if verification rejected the
                phrase, or no verifier is configured and it is treated as a
                stranger (hotword should be blocked).
        """
        if self._closed:
            return
        want_blocked = not matched
        try:
            self._queue.put_nowait(want_blocked)
        except queue.Full:
            # Dropping is safe, not just convenient: the queue is full of
            # requests the worker hasn't gotten to yet, and the next
            # verification event -- which will arrive soon, since these come
            # from live speech -- re-communicates whatever the current
            # desired state actually is. A dropped request can only delay
            # convergence to the right state; it can never cause the wrong
            # one, because the worker always acts on the newest request it
            # sees, never a stale one.
            pass

    def close(self) -> None:
        """Stop the worker thread.  Idempotent, never raises.

        Bounded the same way :meth:`~echochamber.voicegate.notify.WebSocketNotifier.close`
        is: shutdown must not hang because an adb call is wedged.
        """
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        thread = self._thread
        if thread is not threading.current_thread():
            thread.join(_JOIN_TIMEOUT_S)

    def _run(self) -> None:
        """Worker thread body: wait for a request, coalesce, act, repeat."""
        while not self._stop_event.is_set():
            try:
                want_blocked = self._queue.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            # Drain anything else already queued: only the freshest desired
            # state within a burst is ever worth acting on, exactly as
            # WebSocketNotifier's sender only ever sends the event at the
            # head of its deque, never a whole backlog it fell behind on.
            while True:
                try:
                    want_blocked = self._queue.get_nowait()
                except queue.Empty:
                    break
            self._process(want_blocked)

    def _process(self, want_blocked: bool) -> None:
        """Act on one (coalesced) desired state, if it is actually a change.

        Args:
            want_blocked: The most recently requested state.
        """
        if want_blocked == self._believed_blocked:
            with self._lock:
                self._skip_count += 1
            return

        try:
            ok = hotword_core.set_hotword_blocked(self._adb_path, want_blocked)
        except Exception as exc:  # noqa: BLE001 - a daemon thread must not die
            self._record_error(exc)
            return

        if not ok:
            action = "block" if want_blocked else "unblock"
            with self._lock:
                self._last_error = f"adb failed to {action} the hotword"
            # Believed state is deliberately left unchanged: the device never
            # actually moved, so a future identical request must not be
            # skipped by the debounce above.
            return

        # Safety unblock only follows a successful block; there is nothing to
        # guard against after an unblock. Its failure does not undo the block
        # that already succeeded -- it is a crash-safety net, not the primary
        # action -- but it is worth recording.
        safety_error: str | None = None
        if want_blocked:
            try:
                if not hotword_core.schedule_safety_unblock(self._adb_path):
                    safety_error = "failed to schedule the on-device safety unblock"
            except Exception as exc:  # noqa: BLE001 - must not crash the thread
                safety_error = f"{type(exc).__name__}: {exc}"

        self._believed_blocked = want_blocked
        with self._lock:
            self._blocked = want_blocked
            if want_blocked:
                self._block_count += 1
            else:
                self._unblock_count += 1
            self._last_error = safety_error

    def _record_error(self, exc: BaseException) -> None:
        """Store ``exc`` as the most recent failure.

        Args:
            exc: The exception that was caught.
        """
        with self._lock:
            self._last_error = f"{type(exc).__name__}: {exc}"

    def __repr__(self) -> str:
        """Return a debugging representation of this trigger's state."""
        stats = self.snapshot()
        return (
            f"{type(self).__name__}(blocked={stats.blocked}, "
            f"block_count={stats.block_count}, unblock_count={stats.unblock_count})"
        )


@dataclass(frozen=True, slots=True)
class AdbTriggerChoice:
    """What :func:`build_adb_trigger` produced, and whether it is the real thing.

    Mirrors :class:`~echochamber.speakerid.backends.VerifierChoice` and
    :class:`~echochamber.voicegate.backends.RecognizerChoice`, with one
    difference: ``ok`` is a stored field here rather than derived from
    ``error``, since a disabled trigger is both ``ok=True`` and has nothing
    to report.

    Attributes:
        trigger: The trigger to use, or ``None`` when the feature is
            disabled or adb could not be found.
        ok: ``True`` when the configured backend was built successfully, or
            the trigger is deliberately disabled. ``False`` only when the
            trigger was requested but could not be built.
        backend: Short name of what was built: ``"adb"`` or ``"none"``.
        error: Why the trigger could not be built, or ``None`` when nothing
            went wrong.  Non-``None`` together with ``backend="none"`` means
            the trigger is configured on but unusable.
    """

    trigger: AdbHotwordTrigger | None
    ok: bool
    backend: str
    error: str | None = None

    def __repr__(self) -> str:
        """Return a debugging representation naming the backend and outcome."""
        return (
            f"{type(self).__name__}(backend={self.backend!r}, "
            f"ok={self.ok}, error={self.error!r})"
        )


def build_adb_trigger(config: AdbTriggerConfig) -> AdbTriggerChoice:
    """Build the adb hotword trigger ``config`` asks for, degrading rather than raising.

    Mirrors :func:`~echochamber.speakerid.backends.build_verifier`: a trigger
    that refuses to start is not a reason to refuse to capture, and every
    ordinary failure -- no adb on ``PATH``, no Android SDK installed -- is
    something that only surfaces the moment the user presses Start.

    Args:
        config: The adb trigger configuration.

    Returns:
        An :class:`AdbTriggerChoice`.  ``trigger`` is ``None`` unless
        ``config.enabled`` and adb could be found.
    """
    if not config.enabled:
        return AdbTriggerChoice(trigger=None, ok=True, backend="none")

    adb_path = config.adb_path or hotword_core.find_adb()
    if not adb_path:
        return AdbTriggerChoice(
            trigger=None,
            ok=False,
            backend="none",
            error=(
                "the adb hotword trigger is enabled but adb.exe could not be "
                "found; install the Android SDK platform-tools or set "
                "adb_path explicitly"
            ),
        )

    try:
        trigger = AdbHotwordTrigger(adb_path)
    except Exception as exc:  # noqa: BLE001 - every failure is reported, not raised
        return AdbTriggerChoice(trigger=None, ok=False, backend="none", error=str(exc))
    return AdbTriggerChoice(trigger=trigger, ok=True, backend="adb")


def describe_backend(choice: AdbTriggerChoice) -> str:
    """Render an :class:`AdbTriggerChoice` as one line for a status bar.

    Args:
        choice: What :func:`build_adb_trigger` returned.

    Returns:
        A short human-readable description.
    """
    if choice.error is not None:
        return f"adb hotword trigger disabled: {choice.error}"
    if choice.backend == "none":
        return "adb hotword trigger off"
    return f"adb hotword trigger on ({choice.backend})"
