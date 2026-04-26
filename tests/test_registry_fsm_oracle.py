"""Regression tests for the worker_registry_crash FSM oracle.

Each test section corresponds to a proved invariant from
``models/worker_registry_crash.v``.  The tests exercise the extracted
``transition`` function directly and verify the five machine-checked
guarantees.

Proved invariants exercised:

  ``rescue_requires_prior_crash``    — Rescue is rejected from Absent,
    Active, and Stopped; only a Crashed slot holds a rescuable provider.

  ``no_start_while_active``          — Launch and Rescue are both rejected
    from Active; a live thread must die before its slot can be reused.

  ``crash_must_rescue``              — Launch is rejected from Crashed; a
    crashed predecessor must be rescued (not bypassed) to avoid orphaning
    the still-live provider subprocess.

  ``thread_events_only_from_active`` — ThreadDies and ThreadStops are
    rejected from every state except Active; only a live thread can produce
    lifecycle events.

  ``crash_recovery_is_total``        — Rescue from Crashed always yields
    Some Active; crash recovery is a total function from the crashed state.

Field lesson covered:

  Starting a replacement thread while the old one is still alive races on
  the fido lockfile — the old thread may not exit before the new one
  starts, and Python threads can't be cleanly killed.  The
  ``no_start_while_active`` invariant machine-checks that this class of
  bug cannot recur: Launch is always rejected from Active.
"""

from fido.rocq.worker_registry_crash import (
    Absent,
    Active,
    Crashed,
    Event,
    Launch,
    Rescue,
    State,
    Stopped,
    ThreadDies,
    ThreadStops,
    transition,
)

# ---------------------------------------------------------------------------
# Invariant: rescue_requires_prior_crash
#
# Rescue is rejected from every state except Crashed.  A provider can only
# be reclaimed from a slot that actually crashed; an absent slot has no
# predecessor, an active slot has a live thread, and a stopped slot's thread
# exited orderly with no rescuable provider.
# ---------------------------------------------------------------------------


def test_rescue_rejected_from_absent() -> None:
    """Rescue is rejected from Absent — no predecessor to rescue.

    rescue_requires_prior_crash: the absent slot has never started a thread.
    There is no provider to detach, so Rescue is refused.
    """
    result = transition(Absent(), Rescue())
    assert result is None, (
        "rescue_requires_prior_crash violated: Rescue accepted from Absent"
    )


def test_rescue_rejected_from_active() -> None:
    """Rescue is rejected from Active — a live thread is not a crashed one.

    rescue_requires_prior_crash: an active slot holds a running thread.
    Rescue is the crash-recovery path and must never fire while the slot
    is occupied by a live thread.
    """
    result = transition(Active(), Rescue())
    assert result is None, (
        "rescue_requires_prior_crash violated: Rescue accepted from Active"
    )


def test_rescue_rejected_from_stopped() -> None:
    """Rescue is rejected from Stopped — orderly stop is not a crash.

    rescue_requires_prior_crash: the stopped slot's thread exited via
    stop() with _stop=True.  Its provider was shut down orderly; there is
    nothing to rescue.  This is the machine-checked form of the invariant
    that distinguishes the crash path from the orderly-stop path.
    """
    result = transition(Stopped(), Rescue())
    assert result is None, (
        "rescue_requires_prior_crash violated: Rescue accepted from Stopped"
    )


def test_rescue_accepted_only_from_crashed() -> None:
    """Rescue is accepted only from Crashed — the sole rescue entry point.

    rescue_requires_prior_crash: Crashed is the only state where the slot's
    thread died unexpectedly (crash_error set, _stop=False), so a live
    provider subprocess may still exist.  Rescue always succeeds from
    Crashed and always yields Active.
    """
    result = transition(Crashed(), Rescue())
    assert isinstance(result, Active), (
        f"rescue_requires_prior_crash violated: Rescue from Crashed "
        f"yielded {type(result).__name__!r} instead of Active"
    )


# ---------------------------------------------------------------------------
# Invariant: no_start_while_active
#
# Launch and Rescue are both rejected from Active.  A live thread must first
# die (via ThreadDies or ThreadStops) before the slot can be reused.  This
# machine-checks the single-active-thread-per-repo rule.
# ---------------------------------------------------------------------------


def test_launch_rejected_from_active() -> None:
    """Launch is rejected from Active — cannot overlay a live thread.

    no_start_while_active: the slot is occupied by a running thread.
    Attempting a second start() via Launch is refused — the live thread
    must crash or stop before the slot becomes available.
    """
    result = transition(Active(), Launch())
    assert result is None, "no_start_while_active violated: Launch accepted from Active"


def test_rescue_rejected_from_active_by_no_start_while_active() -> None:
    """Rescue is rejected from Active — cannot overlay a live thread.

    no_start_while_active: even the rescue path is blocked while a thread
    is alive.  Both start() routes (Launch and Rescue) are refused from
    Active, enforcing that exactly one thread exists per repo at a time.
    """
    result = transition(Active(), Rescue())
    assert result is None, "no_start_while_active violated: Rescue accepted from Active"


# ---------------------------------------------------------------------------
# Invariant: crash_must_rescue
#
# Launch is rejected from Crashed.  When a predecessor crashed, start()
# must take the rescue path — a fresh Launch over a Crashed slot would
# leave the still-live provider subprocess running with no owner, and
# the session intent stored in _session_issue on the dead thread would
# be lost.
# ---------------------------------------------------------------------------


def test_launch_rejected_from_crashed() -> None:
    """Launch is rejected from Crashed — a crashed slot demands Rescue.

    crash_must_rescue: the crashed slot's provider subprocess may still be
    alive.  A fresh Launch would orphan it.  The FSM enforces that the only
    way out of Crashed is via Rescue, which explicitly calls detach_provider
    to reclaim the live subprocess.
    """
    result = transition(Crashed(), Launch())
    assert result is None, "crash_must_rescue violated: Launch accepted from Crashed"


# ---------------------------------------------------------------------------
# Invariant: thread_events_only_from_active
#
# ThreadDies and ThreadStops are rejected from every state except Active.
# Only a live thread can produce lifecycle events.  An absent slot has no
# thread to die, a crashed slot's thread is already dead, and a stopped
# slot's thread has already exited.
# ---------------------------------------------------------------------------


def test_thread_dies_rejected_from_absent() -> None:
    """ThreadDies is rejected from Absent — no thread to die."""
    result = transition(Absent(), ThreadDies())
    assert result is None, (
        "thread_events_only_from_active violated: ThreadDies accepted from Absent"
    )


def test_thread_dies_rejected_from_crashed() -> None:
    """ThreadDies is rejected from Crashed — a dead thread can't die again."""
    result = transition(Crashed(), ThreadDies())
    assert result is None, (
        "thread_events_only_from_active violated: ThreadDies accepted from Crashed"
    )


def test_thread_dies_rejected_from_stopped() -> None:
    """ThreadDies is rejected from Stopped — a stopped thread can't die unexpectedly."""
    result = transition(Stopped(), ThreadDies())
    assert result is None, (
        "thread_events_only_from_active violated: ThreadDies accepted from Stopped"
    )


def test_thread_stops_rejected_from_absent() -> None:
    """ThreadStops is rejected from Absent — no thread to stop."""
    result = transition(Absent(), ThreadStops())
    assert result is None, (
        "thread_events_only_from_active violated: ThreadStops accepted from Absent"
    )


def test_thread_stops_rejected_from_crashed() -> None:
    """ThreadStops is rejected from Crashed — a dead thread can't stop orderly."""
    result = transition(Crashed(), ThreadStops())
    assert result is None, (
        "thread_events_only_from_active violated: ThreadStops accepted from Crashed"
    )


def test_thread_stops_rejected_from_stopped() -> None:
    """ThreadStops is rejected from Stopped — a stopped thread can't stop again."""
    result = transition(Stopped(), ThreadStops())
    assert result is None, (
        "thread_events_only_from_active violated: ThreadStops accepted from Stopped"
    )


def test_thread_dies_accepted_from_active() -> None:
    """ThreadDies from Active yields Crashed — crash is detectable from Active only.

    thread_events_only_from_active: an alive thread can crash; this is the
    sole path to Crashed.  After ThreadDies, the slot holds a dead thread
    with crash_error set and a possibly rescuable provider.
    """
    result = transition(Active(), ThreadDies())
    assert isinstance(result, Crashed), (
        f"thread_events_only_from_active violated: ThreadDies from Active "
        f"yielded {type(result).__name__!r} instead of Crashed"
    )


def test_thread_stops_accepted_from_active() -> None:
    """ThreadStops from Active yields Stopped — orderly stop only from Active.

    thread_events_only_from_active: an alive thread can exit orderly; this is
    the sole path to Stopped.  After ThreadStops, the slot holds a dead
    thread with _stop=True and no rescuable provider.
    """
    result = transition(Active(), ThreadStops())
    assert isinstance(result, Stopped), (
        f"thread_events_only_from_active violated: ThreadStops from Active "
        f"yielded {type(result).__name__!r} instead of Stopped"
    )


# ---------------------------------------------------------------------------
# Invariant: crash_recovery_is_total
#
# Rescue from Crashed always yields Some Active.  Crash recovery is a total
# function from the crashed state — every detected crash has a well-defined
# recovery path.  There is no conditional branch where a crashed slot stays
# crashed after a Rescue attempt.
# ---------------------------------------------------------------------------


def test_crash_recovery_is_total() -> None:
    """Rescue from Crashed always yields Active — recovery is unconditional.

    crash_recovery_is_total: every crash the watchdog detects leads to
    registry.start() which always fires Rescue → Active.  There is no
    path where Rescue returns None from Crashed.
    """
    result = transition(Crashed(), Rescue())
    assert isinstance(result, Active), (
        f"crash_recovery_is_total violated: Rescue from Crashed "
        f"yielded {type(result).__name__!r} instead of Active"
    )


# ---------------------------------------------------------------------------
# Complete lifecycle paths
#
# End-to-end tests of the three valid multi-step paths through the FSM.
# Each succeeds at every step and terminates in Active — the operational
# state where a live thread is running.
# ---------------------------------------------------------------------------


def test_fresh_start_path() -> None:
    """Absent → Launch → Active: initial startup, no prior thread."""
    s: State = Absent()
    s = transition(s, Launch())  # type: ignore[assignment]
    assert isinstance(s, Active)


def test_crash_recovery_path() -> None:
    """Active → ThreadDies → Crashed → Rescue → Active: full crash/restart cycle.

    The complete crash/restart lifecycle succeeds at every step and returns
    the slot to Active with no stuck intermediate states.
    """
    s: State = Active()

    s = transition(s, ThreadDies())  # type: ignore[assignment]
    assert isinstance(s, Crashed)

    s = transition(s, Rescue())  # type: ignore[assignment]
    assert isinstance(s, Active)


def test_orderly_stop_and_relaunch_path() -> None:
    """Active → ThreadStops → Stopped → Launch → Active: re-enable after stop.

    An orderly-stopped slot can be re-enabled via a fresh Launch.  The
    stopped predecessor's provider is not rescued (it exited orderly).
    """
    s: State = Active()

    s = transition(s, ThreadStops())  # type: ignore[assignment]
    assert isinstance(s, Stopped)

    s = transition(s, Launch())  # type: ignore[assignment]
    assert isinstance(s, Active)


def test_repeated_crash_restart_cycles() -> None:
    """Thread can crash and restart multiple times — no FSM exhaustion.

    Each cycle goes Active → ThreadDies → Crashed → Rescue → Active.
    The FSM supports arbitrarily many crash/restart cycles with no stuck
    terminal state.
    """
    s: State = Absent()
    s = transition(s, Launch())  # type: ignore[assignment]
    assert isinstance(s, Active)

    for _ in range(3):
        s = transition(s, ThreadDies())  # type: ignore[assignment]
        assert isinstance(s, Crashed)
        s = transition(s, Rescue())  # type: ignore[assignment]
        assert isinstance(s, Active)


def test_crash_then_stop_and_relaunch() -> None:
    """Crash recovery → orderly stop → re-enable: all paths compose cleanly."""
    s: State = Absent()
    s = transition(s, Launch())  # type: ignore[assignment]
    assert isinstance(s, Active)

    # Crash and recover
    s = transition(s, ThreadDies())  # type: ignore[assignment]
    assert isinstance(s, Crashed)
    s = transition(s, Rescue())  # type: ignore[assignment]
    assert isinstance(s, Active)

    # Then orderly stop and re-enable
    s = transition(s, ThreadStops())  # type: ignore[assignment]
    assert isinstance(s, Stopped)
    s = transition(s, Launch())  # type: ignore[assignment]
    assert isinstance(s, Active)


# ---------------------------------------------------------------------------
# Exhaustive state×event matrix
#
# All transitions not already covered above — verifies the full table in
# the transition function so no arm is accidentally unreachable.
# ---------------------------------------------------------------------------


def test_launch_accepted_from_absent() -> None:
    """Launch from Absent yields Active — the normal fresh-start path."""
    assert isinstance(transition(Absent(), Launch()), Active)


def test_launch_accepted_from_stopped() -> None:
    """Launch from Stopped yields Active — re-enable after orderly stop."""
    assert isinstance(transition(Stopped(), Launch()), Active)


def test_all_events_exhaustive() -> None:
    """Every state×event pair has a deterministic transition result.

    Walks the full 4×4 matrix and asserts that the output type matches
    the transition table in ``models/worker_registry_crash.v``.  Any
    future change to the Rocq model that adds or removes a valid transition
    will be caught here.
    """
    expected: dict[tuple[type[State], type[Event]], type[State] | None] = {
        # Absent
        (Absent, Launch): Active,
        (Absent, Rescue): None,
        (Absent, ThreadDies): None,
        (Absent, ThreadStops): None,
        # Active
        (Active, Launch): None,
        (Active, Rescue): None,
        (Active, ThreadDies): Crashed,
        (Active, ThreadStops): Stopped,
        # Crashed
        (Crashed, Launch): None,
        (Crashed, Rescue): Active,
        (Crashed, ThreadDies): None,
        (Crashed, ThreadStops): None,
        # Stopped
        (Stopped, Launch): Active,
        (Stopped, Rescue): None,
        (Stopped, ThreadDies): None,
        (Stopped, ThreadStops): None,
    }
    for (state_cls, event_cls), expected_cls in expected.items():
        result = transition(state_cls(), event_cls())
        if expected_cls is None:
            assert result is None, (
                f"expected None for ({state_cls.__name__}, {event_cls.__name__}), "
                f"got {type(result).__name__}"
            )
        else:
            assert isinstance(result, expected_cls), (
                f"expected {expected_cls.__name__} for "
                f"({state_cls.__name__}, {event_cls.__name__}), "
                f"got {type(result).__name__ if result is not None else 'None'}"
            )
