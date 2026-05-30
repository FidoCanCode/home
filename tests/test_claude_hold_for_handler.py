"""Tests for ClaudeSession.hold_for_handler (#658)."""

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from fido import provider
from fido.claude import ClaudeSession
from fido.provider import SessionTalker, ThreadKind, talker_now

# ── typed subprocess fakes ────────────────────────────────────────────────────


class _WriteCallRecord:
    """Single write() invocation; ``args[0]`` is the string written."""

    def __init__(self, args: tuple[str, ...]) -> None:
        self.args = args


class _FakeStdinWrite:
    """Callable that records ``stdin.write()`` calls with a typed log."""

    def __init__(self) -> None:
        self.call_args_list: list[_WriteCallRecord] = []

    def __call__(self, data: str) -> None:
        self.call_args_list.append(_WriteCallRecord((data,)))


class _FakeStdin:
    """Fake process stdin: records writes, exposes ``closed`` flag."""

    def __init__(self) -> None:
        self.closed: bool = False
        self.write: _FakeStdinWrite = _FakeStdinWrite()

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    """Fake stdout that returns pre-loaded lines then EOF."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines: list[str] = list(lines or [])
        self._idx: int = 0

    def readline(self) -> str:
        if self._idx < len(self._lines):
            line = self._lines[self._idx]
            self._idx += 1
            return line
        return ""

    def close(self) -> None:
        pass


class _FakeStderr:
    """Fake process stderr — no-op iteration and close.

    ``ClaudeSession._start_stderr_pump`` iterates over stderr in a background
    thread; returning an empty iterator silences the pump without deadlocking.
    """

    def __iter__(self) -> Iterator[str]:
        return iter([])

    def close(self) -> None:
        pass


class _FakeProc:
    """Minimal ``subprocess.Popen`` fake for ClaudeSession tests."""

    def __init__(self, lines: list[str] | None = None) -> None:
        self.pid: int = 0
        self.returncode: int = 0
        self.stdin: _FakeStdin = _FakeStdin()
        self.stdout: _FakeStdout = _FakeStdout(lines)
        self.stderr: _FakeStderr = _FakeStderr()

    def kill(self) -> None:
        pass

    def terminate(self) -> None:
        pass

    def poll(self) -> int | None:
        return None

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


def _make_session_proc(lines: list[str]) -> _FakeProc:
    return _FakeProc(lines)


class _SpyClaudeSession(ClaudeSession):
    """ClaudeSession subclass that records ``_fire_worker_cancel`` invocations.

    Tests that need to assert whether the cancel signal fired pass an instance
    of this class instead of a bare ``ClaudeSession``.  The override skips the
    real cancel mechanism (wakeup-pipe write + ``_cancel.set()``) because these
    tests have no actual worker thread blocking inside ``iter_events`` — all
    they need is a record of the call count.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.cancel_calls: list[int] = []

    def _fire_worker_cancel(self) -> None:
        self.cancel_calls.append(1)


def _setup_session(
    tmp_path: Path,
    repo: str = "owner/repo",
    *,
    register_talker: Callable[[provider.SessionTalker], None] | None = None,
) -> ClaudeSession:
    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = _make_session_proc(['{"type":"result","result":"reply"}\n'])
    proc.pid = 55555
    return ClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=lambda *_a, **_kw: proc,  # type: ignore[arg-type]
        selector=lambda *_a, **_kw: ([proc.stdout], [], []),  # type: ignore[arg-type]
        repo_name=repo,
        model="claude-opus-4-6",
        register_talker=register_talker,
    )


def test_hold_acquires_lock_and_registers_talker(tmp_path: Path) -> None:
    from fido.rocq.transition import Free, OwnedByHandler

    session = _setup_session(tmp_path)
    provider.set_thread_kind(ThreadKind.WEBHOOK)
    try:
        with session.hold_for_handler():
            talker = provider.get_talker("owner/repo")
            assert talker is not None
            assert talker.kind == "webhook"
            assert isinstance(session._fsm_state, OwnedByHandler)
        # After exit, talker unregistered, FSM back to Free.
        assert provider.get_talker("owner/repo") is None
        assert isinstance(session._fsm_state, Free)
    finally:
        provider.set_thread_kind(None)
        session.stop()


def test_nested_with_inside_hold_does_not_double_register(
    tmp_path: Path,
) -> None:
    """Re-entering ``with session:`` inside ``hold_for_handler`` must not
    attempt a second talker registration (would raise SessionLeakError)."""
    register_calls: list[str] = []
    real_register = provider.register_talker

    def counting_register(talker: provider.SessionTalker) -> None:
        register_calls.append(talker.kind)
        real_register(talker)

    session = _setup_session(tmp_path, register_talker=counting_register)
    provider.set_thread_kind(ThreadKind.WEBHOOK)
    try:
        with session.hold_for_handler():
            assert len(register_calls) == 1  # outer entry registered once
            with session:  # nested re-entry
                assert len(register_calls) == 1  # not re-registered
            assert len(register_calls) == 1  # still registered after inner exit
        # After outer exit, unregistered.
        assert provider.get_talker("owner/repo") is None
    finally:
        provider.set_thread_kind(None)
        session.stop()


def test_hold_preempt_fires_cancel_when_worker_holds(tmp_path: Path) -> None:
    """hold_for_handler() fires _fire_worker_cancel iff
    the current lock holder is a worker and the caller is a webhook."""
    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = _make_session_proc(['{"type":"result","result":"reply"}\n'])
    proc.pid = 55555

    def fake_talker(_repo: str) -> SessionTalker:
        return SessionTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind="worker",  # type: ignore[arg-type]
            description="fake",
            subprocess_pid=55555,
            started_at=talker_now(),
        )

    session = _SpyClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=lambda *_a, **_kw: proc,  # type: ignore[arg-type]
        selector=lambda *_a, **_kw: ([proc.stdout], [], []),  # type: ignore[arg-type]
        repo_name="owner/repo",
        model="claude-opus-4-6",
        talker_resolver=fake_talker,
    )
    provider.set_thread_kind(ThreadKind.WEBHOOK)
    try:
        with session.hold_for_handler():
            pass
    finally:
        provider.set_thread_kind(None)
        session.stop()
    assert session.cancel_calls == [1]


def test_hold_preempt_no_fire_when_no_worker_holder(tmp_path: Path) -> None:
    """preempt_worker=True with no current holder — try_preempt_worker returns
    (False, None) and no cancel fires.  Exercises the ``else`` branch of the
    new preempt outcome logging in hold_for_handler (#955)."""
    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = _make_session_proc(['{"type":"result","result":"reply"}\n'])
    proc.pid = 55555

    # No holder registered — try_preempt_worker sees current_kind=None.
    session = _SpyClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=lambda *_a, **_kw: proc,  # type: ignore[arg-type]
        selector=lambda *_a, **_kw: ([proc.stdout], [], []),  # type: ignore[arg-type]
        repo_name="owner/repo",
        model="claude-opus-4-6",
        talker_resolver=lambda _repo: None,
    )
    provider.set_thread_kind(ThreadKind.WEBHOOK)
    try:
        with session.hold_for_handler():
            pass
    finally:
        provider.set_thread_kind(None)
        session.stop()
    assert session.cancel_calls == []


def test_hold_preempt_skipped_when_no_preempt_worker_flag(tmp_path: Path) -> None:
    """Default preempt_worker=False — no cancel fires even with a worker
    holder."""
    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = _make_session_proc(['{"type":"result","result":"reply"}\n'])
    proc.pid = 55555

    def fake_talker(_repo: str) -> SessionTalker:
        return SessionTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind="worker",  # type: ignore[arg-type]
            description="fake",
            subprocess_pid=55555,
            started_at=talker_now(),
        )

    session = _SpyClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=lambda *_a, **_kw: proc,  # type: ignore[arg-type]
        selector=lambda *_a, **_kw: ([proc.stdout], [], []),  # type: ignore[arg-type]
        repo_name="owner/repo",
        model="claude-opus-4-6",
        talker_resolver=fake_talker,
    )
    try:
        with session.hold_for_handler():  # preempt-always now lives in __enter__
            pass
    finally:
        session.stop()
    assert session.cancel_calls == []


def test_other_thread_blocks_while_held(tmp_path: Path) -> None:
    """A different thread trying to ``with session:`` while another thread
    is inside hold_for_handler must block until the hold exits (#658 —
    that is the whole point of holding the lock across turns)."""
    session = _setup_session(tmp_path)
    holder_entered = threading.Event()
    release_holder = threading.Event()
    other_acquired = threading.Event()
    other_finished = threading.Event()

    def holder() -> None:
        provider.set_thread_kind(ThreadKind.WEBHOOK)
        try:
            with session.hold_for_handler():
                holder_entered.set()
                release_holder.wait(timeout=5.0)
        finally:
            provider.set_thread_kind(None)

    def other() -> None:
        provider.set_thread_kind(ThreadKind.WORKER)
        try:
            with session:
                other_acquired.set()
            other_finished.set()
        finally:
            provider.set_thread_kind(None)

    t1 = threading.Thread(target=holder, daemon=True)
    t1.start()
    holder_entered.wait(timeout=2.0)
    t2 = threading.Thread(target=other, daemon=True)
    t2.start()
    # Give t2 a chance — it must NOT have acquired the lock yet.
    assert not other_acquired.wait(timeout=0.1)
    # Release the holder; t2 should now acquire.
    release_holder.set()
    t1.join(timeout=2.0)
    assert other_acquired.wait(timeout=2.0), "other thread never acquired"
    assert other_finished.wait(timeout=2.0)
    session.stop()


def test_webhook_preempts_worker_mid_turn(tmp_path: Path) -> None:
    """End-to-end: webhook calling hold_for_handler()
    wakes a worker that is blocked inside iter_events, causing the worker to
    exit its turn and release the lock so the webhook can acquire it (#955)."""
    import time

    system_file = tmp_path / "system.md"
    system_file.write_text("sys")
    proc = _FakeProc()
    proc.pid = 55555

    # Worker turn: readline blocks until cancel fires, then returns EOF.
    worker_blocked = threading.Event()
    cancel_received = threading.Event()

    def blocking_readline() -> str:
        worker_blocked.set()
        cancel_received.wait(timeout=5.0)
        return ""  # EOF — worker exits iter_events

    proc.stdout.readline = blocking_readline  # type: ignore[method-assign]

    # Selector: immediately returns stdout as ready so iter_events calls readline.
    session = ClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=lambda *_a, **_kw: proc,  # type: ignore[arg-type]
        selector=lambda *_a, **_kw: ([proc.stdout], [], []),  # type: ignore[arg-type]
        repo_name="owner/repo",
        model="claude-opus-4-6",
    )

    worker_in_turn = threading.Event()
    worker_done = threading.Event()
    webhook_acquired = threading.Event()
    webhook_done = threading.Event()

    def worker() -> None:
        provider.set_thread_kind(ThreadKind.WORKER)
        try:
            with session:
                worker_in_turn.set()
                # consume_until_result drives iter_events; readline blocks
                session.consume_until_result()
            worker_done.set()
        finally:
            provider.set_thread_kind(None)

    def webhook() -> None:
        provider.set_thread_kind(ThreadKind.WEBHOOK)
        try:
            # Wait until worker is actually blocked, then preempt.
            worker_blocked.wait(timeout=2.0)
            # Signal readline to unblock after cancel fires.
            original_fire = session._fire_worker_cancel

            def fire_and_unblock() -> None:
                original_fire()
                cancel_received.set()

            session._fire_worker_cancel = fire_and_unblock  # type: ignore[method-assign]
            t_start = time.monotonic()
            with session.hold_for_handler():
                webhook_acquired.set()
                elapsed = time.monotonic() - t_start
                assert elapsed < 2.0, (
                    f"webhook took too long to acquire: {elapsed:.2f}s"
                )
            webhook_done.set()
        finally:
            provider.set_thread_kind(None)

    t_worker = threading.Thread(target=worker, daemon=True)
    t_worker.start()
    worker_in_turn.wait(timeout=2.0)

    t_webhook = threading.Thread(target=webhook, daemon=True)
    t_webhook.start()

    assert webhook_acquired.wait(timeout=5.0), "webhook never acquired lock"
    assert webhook_done.wait(timeout=5.0)
    assert worker_done.wait(timeout=5.0)
    t_worker.join(timeout=2.0)
    t_webhook.join(timeout=2.0)
    session.stop()


def test_handler_prompt_runs_after_preempt_does_not_inherit_cancel(
    tmp_path: Path,
) -> None:
    """Post-#979: after hold_for_handler() fires the
    cancel signal, the handler's own prompt() must run to completion.  In
    the new design the prior turn's boundary is drained inside iter_events
    itself (cancel no longer breaks early), so by the time the handler
    enters the stream is clean.  The cancel signal that was set for the
    previous holder is unconditionally cleared at the start of the handler's
    iter_events call."""
    system_file = tmp_path / "system.md"
    system_file.write_text("sys")

    def fake_talker(_repo: str) -> SessionTalker:
        return SessionTalker(
            repo_name="owner/repo",
            thread_id=999_999,
            kind="worker",  # type: ignore[arg-type]
            description="fake",
            subprocess_pid=55555,
            started_at=talker_now(),
        )

    # Pipe contains exactly the handler's own response — no stale events
    # from the prior turn (those were drained inside iter_events when the
    # worker turn closed cleanly on type=result).
    #
    # The same proc fake is used for both the initial spawn and any
    # subsequent _respawn call (triggered by prompt()'s switch_tools when
    # transitioning from worker tools → READ_ONLY_ALLOWED_TOOLS).  Because
    # popen always returns the same object, _selector and _proc stay in sync
    # across the respawn so iter_events never spins on a mismatched stdout.
    proc = _make_session_proc(['{"type":"result","result":"triage-reply"}\n'])
    proc.pid = 55555
    session = ClaudeSession(
        system_file,
        work_dir=tmp_path,
        popen=lambda *_a, **_kw: proc,  # type: ignore[arg-type]
        selector=lambda *_a, **_kw: ([proc.stdout], [], []),  # type: ignore[arg-type]
        repo_name="owner/repo",
        model="claude-opus-4-6",
        talker_resolver=fake_talker,
    )
    provider.set_thread_kind(ThreadKind.WEBHOOK)
    try:
        with session.hold_for_handler():
            # _fire_worker_cancel set _cancel.  Handler's first prompt() must
            # actually send and read its own response.
            result = session.prompt("triage this please")
    finally:
        provider.set_thread_kind(None)
        session.stop()

    # The handler must have written its user message to stdin (atomicity
    # guaranteed by _stdin_lock — see #979).
    write_calls = [c.args[0] for c in proc.stdin.write.call_args_list]
    user_writes = [w for w in write_calls if "triage this please" in w]
    assert user_writes, f"handler prompt never wrote its message — writes={write_calls}"
    assert result == "triage-reply", f"handler prompt got wrong result — got {result!r}"


def test_queued_webhook_acquires_lock_before_worker_after_inner_prompt(
    tmp_path: Path,
) -> None:
    """Handler (webhook B) queuing behind an active hold_for_handler acquires
    before the worker, even after the holder makes inner prompt() calls.

    The FSM handler queue provides structural priority: workers yield to any
    queued handler regardless of when they entered _fsm_acquire_worker.
    """
    session = _setup_session(tmp_path)

    webhook_a_holding = threading.Event()
    webhook_b_queuing = threading.Event()
    order: list[str] = []
    errors: list[Exception] = []

    def webhook_a() -> None:
        provider.set_thread_kind(ThreadKind.WEBHOOK)
        try:
            with session.hold_for_handler():
                webhook_a_holding.set()
                assert webhook_b_queuing.wait(timeout=2.0), (
                    "webhook_b_queuing timed out"
                )
                # Small sleep to let webhook_b actually block in _fsm_acquire_handler
                import time as _time

                _time.sleep(0.05)
                session.prompt("inner turn from webhook A")
        except Exception as exc:
            errors.append(exc)
        finally:
            provider.set_thread_kind(None)

    def webhook_b() -> None:
        provider.set_thread_kind(ThreadKind.WEBHOOK)
        try:
            assert webhook_a_holding.wait(timeout=2.0), "webhook_a_holding timed out"
            webhook_b_queuing.set()
            # Queue behind webhook A via the FSM handler queue.
            with session:
                order.append("webhook_b")
        except Exception as exc:
            errors.append(exc)
        finally:
            provider.set_thread_kind(None)

    def worker() -> None:
        provider.set_thread_kind(ThreadKind.WORKER)
        try:
            assert webhook_b_queuing.wait(timeout=2.0), "webhook_b_queuing timed out"
            import time as _time

            _time.sleep(0.05)  # give webhook_b time to actually queue
            with session:
                order.append("worker")
        except Exception as exc:
            errors.append(exc)
        finally:
            provider.set_thread_kind(None)

    t_a = threading.Thread(target=webhook_a, daemon=True)
    t_b = threading.Thread(target=webhook_b, daemon=True)
    t_w = threading.Thread(target=worker, daemon=True)

    t_a.start()
    t_b.start()
    t_w.start()

    t_a.join(timeout=5.0)
    t_b.join(timeout=5.0)
    t_w.join(timeout=5.0)
    session.stop()

    assert not errors, f"thread errors: {errors}"
    assert order == ["webhook_b", "worker"], (
        f"webhook_b should acquire before worker — got {order}"
    )


def test_hold_reraises_leak_error_and_releases_lock(tmp_path: Path) -> None:
    """If register_talker raises SessionLeakError inside hold, the lock must
    be released before the exception propagates so we don't deadlock."""
    session = _setup_session(tmp_path)
    # Pre-register a talker for the same repo from a different thread id so
    # the hold's register_talker collides.
    provider.register_talker(
        SessionTalker(
            repo_name="owner/repo",
            thread_id=111_111,  # different tid — triggers leak
            kind=ThreadKind.WORKER,
            description="squatter",
            subprocess_pid=None,
            started_at=talker_now(),
        )
    )
    provider.set_thread_kind(ThreadKind.WEBHOOK)
    try:
        with pytest.raises(provider.SessionLeakError):
            with session.hold_for_handler():
                pass  # should not reach here
        # FSM must be Free so other threads aren't deadlocked.
        from fido.rocq.transition import Free

        assert isinstance(session._fsm_state, Free)
    finally:
        provider.set_thread_kind(None)
        provider.unregister_talker("owner/repo", 111_111)
        session.stop()


# ---------------------------------------------------------------------------
# ClaudeSession._on_force_release — kill subprocess so wedged holder escapes
# (closes #1377)
# ---------------------------------------------------------------------------


class _RecordingProc:
    """Hand-rolled subprocess.Popen-shaped fake that records ``kill()`` calls.

    Used so the test follows the project rule of hand-rolled typed fakes for
    new tests; we only need ``pid``, ``kill``, and the methods ``stop``
    touches during cleanup.
    """

    def __init__(self, pid: int = 99999) -> None:
        self.pid = pid
        self.kill_calls = 0
        self.kill_raises: BaseException | None = None
        self.returncode: int | None = None
        self.stdin: _FakeStdin = _FakeStdin()
        # ``stop()`` invokes wait/kill on shutdown — accept calls but no
        # actual subprocess work to do.
        self._wait_returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_raises is not None:
            raise self.kill_raises

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self._wait_returncode

    def poll(self) -> int | None:
        return self.returncode


def test_on_force_release_kills_subprocess_and_resets_stream(
    tmp_path: Path,
) -> None:
    """``_on_force_release`` knocks the wedged holder out of its parked IO
    by killing the subprocess; the stream FSM is also reset so the next
    acquire's first turn does not inherit stale ``Sending`` /
    ``AwaitingReply`` state from the killed subprocess."""
    from fido.claude import stream_fsm

    session = _setup_session(tmp_path)
    fake_proc = _RecordingProc(pid=12345)
    session._proc = fake_proc  # type: ignore[assignment]
    # Drive the stream FSM to AwaitingReply so we can observe the reset.
    with session._stream_lock:
        session._stream_state = stream_fsm.AwaitingReply()

    session._on_force_release(reason="watchdog test eviction")

    assert fake_proc.kill_calls == 1
    assert isinstance(session._stream_state, stream_fsm.Idle)


def test_on_force_release_swallows_already_dead_subprocess(
    tmp_path: Path,
) -> None:
    """If the subprocess has already exited (race with idle timeout or
    another recovery path), ``kill()`` raises :class:`ProcessLookupError`
    or :class:`OSError` — the eviction must continue rather than
    propagating that as a watchdog crash."""
    from fido.claude import stream_fsm

    session = _setup_session(tmp_path)
    fake_proc = _RecordingProc()
    fake_proc.kill_raises = ProcessLookupError("no such process")
    session._proc = fake_proc  # type: ignore[assignment]
    with session._stream_lock:
        session._stream_state = stream_fsm.AwaitingReply()

    # Must not raise.
    session._on_force_release(reason="dead-subprocess test")

    assert fake_proc.kill_calls == 1
    # Stream FSM is still reset even when kill failed — defensive
    # cleanup runs regardless of the kill outcome.
    assert isinstance(session._stream_state, stream_fsm.Idle)


def test_on_force_release_swallows_kill_oserror(tmp_path: Path) -> None:
    """``OSError`` from ``kill`` (e.g. EPERM in restricted contexts) is
    handled the same as ``ProcessLookupError``."""
    from fido.claude import stream_fsm

    session = _setup_session(tmp_path)
    fake_proc = _RecordingProc()
    fake_proc.kill_raises = OSError("permission denied")
    session._proc = fake_proc  # type: ignore[assignment]
    with session._stream_lock:
        session._stream_state = stream_fsm.AwaitingReply()

    session._on_force_release(reason="oserror test")

    assert fake_proc.kill_calls == 1
    assert isinstance(session._stream_state, stream_fsm.Idle)


# ---------------------------------------------------------------------------
# End-to-end smoke: wedged holder is unblocked by force_release (#1377)
# ---------------------------------------------------------------------------


class _StreamingForeverProc:
    """Hand-rolled subprocess that streams non-``result`` events forever.

    Reproduces the wedge condition from #1377: ``iter_events`` keeps
    receiving events, ``idle_timeout`` keeps getting reset, no
    ``type=result`` ever arrives, the holder is parked in
    ``consume_until_result`` indefinitely.

    ``kill()`` flips an internal EOF flag so subsequent ``readline``
    calls return ``""`` — that closes ``iter_events`` via its EOF
    branch and the holder escapes ``consume_until_result``.
    """

    def __init__(self) -> None:
        self.pid = 88888
        self.returncode: int | None = None
        self.stdin: _FakeStdin = _FakeStdin()
        self.kill_calls = 0
        self._eof = threading.Event()
        self._first_readline = threading.Event()
        # Pretend stdout is the proc itself — the selector returns this
        # object as "ready" and ``iter_events`` calls ``readline`` on it.
        self.stdout = self
        self.stderr: _FakeStderr = _FakeStderr()

    def readline(self) -> str:
        # Signal first readline so the test can synchronize on the
        # holder being inside ``iter_events`` before firing the
        # eviction.
        self._first_readline.set()
        if self._eof.is_set():
            return ""
        # Block briefly before returning the next event so the holder
        # spends real time inside the loop — but stay responsive to
        # eviction (kill flips _eof which we re-check).
        if self._eof.wait(timeout=0.05):
            return ""
        return '{"type":"system","subtype":"streaming","session_id":"x"}\n'

    def kill(self) -> None:
        self.kill_calls += 1
        self._eof.set()

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def poll(self) -> int | None:
        return self.returncode


def test_force_release_unblocks_wedged_holder_end_to_end(tmp_path: Path) -> None:
    """Full chain: a holder thread is parked in ``consume_until_result``
    on a fake subprocess that streams events forever; the watchdog
    fires ``force_release``; the kill propagates through
    ``_on_force_release``; ``iter_events`` breaks on EOF; the holder's
    ``__exit__`` runs and the ``_evicted_tids`` guard prevents the
    double-release crash; the FSM lands in ``Free`` ready for a fresh
    acquire — all within a generous 5 s budget.

    This is the missing-Rocq-IO fix proved on real concurrency, not
    just unit-by-unit: the holder thread is genuinely parked in
    ``select()`` / ``readline``, the eviction comes from a different
    thread, and the wedged thread escapes via the modeled chain.
    """
    from fido.rocq.transition import Free

    session = _setup_session(tmp_path)
    fake_proc = _StreamingForeverProc()
    session._proc = fake_proc  # type: ignore[assignment]
    # Selector: stdout always reported ready so iter_events keeps
    # calling readline on the streaming proc.
    session._selector = lambda *_a, **_kw: ([fake_proc.stdout], [], [])  # type: ignore[assignment]

    holder_done = threading.Event()
    holder_exception: list[BaseException] = []

    def holder() -> None:
        try:
            with session:
                session.consume_until_result()
        except BaseException as exc:  # noqa: BLE001 — record everything
            holder_exception.append(exc)
        finally:
            holder_done.set()

    provider.set_thread_kind(ThreadKind.WORKER)
    try:
        t = threading.Thread(target=holder, daemon=True)
        t.start()
        # Wait until the holder is genuinely inside ``iter_events`` —
        # readline has been called at least once means the loop is hot.
        assert fake_proc._first_readline.wait(timeout=5.0), (
            "holder never entered iter_events readline"
        )
        # FSM must show the holder owning the lock at this point.
        from fido.rocq.transition import OwnedByWorker

        assert isinstance(session._fsm_state, OwnedByWorker)

        # Watchdog (running on this main thread) evicts the wedged holder.
        evicted = session.force_release(reason="end-to-end smoke")
        assert evicted is True

        # The holder must escape consume_until_result and finish its
        # ``with`` block within a generous budget.
        assert holder_done.wait(timeout=5.0), (
            "wedged holder never escaped after force_release"
        )

        # No exception leaked from the holder thread — the
        # ``_evicted_tids`` guard prevented the double-release crash.
        assert holder_exception == [], (
            f"holder thread raised unexpectedly: {holder_exception!r}"
        )

        # ``_on_force_release`` actually killed the subprocess.
        assert fake_proc.kill_calls == 1
        # Final FSM state is Free — ready for the next acquire with no
        # stale holder slot.
        assert isinstance(session._fsm_state, Free)
    finally:
        provider.set_thread_kind(None)
        # Don't call session.stop — the proc is already in EOF state and
        # ``stop`` would call wait/kill on the initial default proc.
        # Letting the daemon thread reap on process exit is sufficient.
