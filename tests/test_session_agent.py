import datetime
from pathlib import Path

import pytest
from frozendict import frozendict

from fido.appstate import (
    _ZERO_GITHUB_LIMITS,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    ProviderSnapshot,
    zero_repo_state,
)
from fido.atomic import AtomicUpdater, create_atomic
from fido.provider import (
    ProviderModel,
    TurnSessionMode,
)
from fido.session_agent import SessionBackedAgent

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


# ---------------------------------------------------------------------------
# Call-recording stub (replaces MagicMock for method-level stubbing)
# ---------------------------------------------------------------------------


class _FakeCallRecorder:
    """Callable that records calls and supports configurable return/side-effect.

    side_effect may be:
    - a list  → consumed sequentially; exception items raised, others returned
    - a BaseException instance → raised on every call
    - a callable → invoked and its return value returned
    - None (default) → return_value is returned
    """

    def __init__(self, return_value: object = None) -> None:
        self.return_value: object = return_value
        self._side_effect: object = None
        self._side_effect_idx: int = 0
        self._calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    @property
    def side_effect(self) -> object:
        return self._side_effect

    @side_effect.setter
    def side_effect(self, value: object) -> None:
        self._side_effect = value
        self._side_effect_idx = 0

    def __call__(self, *args: object, **kwargs: object) -> object:
        self._calls.append((args, kwargs))
        se = self._side_effect
        if se is not None:
            if isinstance(se, list):
                item = se[self._side_effect_idx]
                self._side_effect_idx += 1
                if isinstance(item, BaseException):
                    raise item
                return item
            if isinstance(se, BaseException):
                raise se
            if callable(se):
                return se(*args, **kwargs)
        return self.return_value

    @property
    def call_args_list(self) -> list[tuple[tuple[object, ...], dict[str, object]]]:
        return list(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def called(self) -> bool:
        return bool(self._calls)

    def assert_not_called(self) -> None:
        assert not self._calls, (
            f"Expected not called but got {len(self._calls)} call(s)"
        )

    def assert_called_once(self) -> None:
        assert len(self._calls) == 1, f"Expected 1 call but got {len(self._calls)}"

    def assert_called_once_with(self, *args: object, **kwargs: object) -> None:
        assert len(self._calls) == 1, f"Expected 1 call but got {len(self._calls)}"
        actual_args, actual_kwargs = self._calls[0]
        assert actual_args == args and actual_kwargs == kwargs, (
            f"Expected call({args!r}, {kwargs!r}) "
            f"but got call({actual_args!r}, {actual_kwargs!r})"
        )


class _FakeSession:
    """Typed fake for session objects used by SessionBackedAgent tests."""

    def __init__(
        self,
        *,
        owner: str = "",
        pid: int = 0,
        session_id: str | None = None,
        dropped_session_count: int = 0,
        sent_count: int = 0,
        received_count: int = 0,
        is_alive_return: bool = True,
    ) -> None:
        self.owner: str = owner
        self.pid: int = pid
        self.session_id: str | None = session_id
        self.dropped_session_count: int = dropped_session_count
        self.sent_count: int = sent_count
        self.received_count: int = received_count
        self.last_turn_cancelled: bool = False
        self.prompt: _FakeCallRecorder = _FakeCallRecorder()
        self.recover: _FakeCallRecorder = _FakeCallRecorder()
        self.switch_model: _FakeCallRecorder = _FakeCallRecorder()
        self.is_alive: _FakeCallRecorder = _FakeCallRecorder(
            return_value=is_alive_return
        )


def _make_fido_state_with_repo(repo_name: str) -> FidoState:
    """Return a minimal :class:`FidoState` with one pre-initialised repo entry."""
    return FidoState(
        repos=frozendict({repo_name: zero_repo_state(repo_name)}),
        github_limits=_ZERO_GITHUB_LIMITS,
        process_started_at=_EPOCH,
    )


class _FakeAgent(SessionBackedAgent):
    voice_model = ProviderModel("voice")
    brief_model = ProviderModel("brief")

    def __init__(
        self,
        *,
        session_fn: object = lambda: None,
        session_system_file: Path | None = None,
        work_dir: Path | str | None = None,
        repo_name: str | None = None,
        session: object = None,
        session_factory: object = None,
        state_updater: AtomicUpdater[FidoState] | None = None,
    ) -> None:
        self._session_factory = (
            _FakeCallRecorder() if session_factory is None else session_factory
        )
        super().__init__(
            session_fn=session_fn,
            session_system_file=session_system_file,
            work_dir=work_dir,
            repo_name=repo_name,
            session=session,
            state_updater=state_updater,
        )

    def _spawn_owned_session(
        self, model: ProviderModel, *, session_id: str = None
    ) -> object:
        self._last_session_id = session_id
        return self._session_factory(model)


class TestSessionBackedAgent:
    def test_base_spawn_owned_session_raises(self) -> None:
        agent = SessionBackedAgent(
            session_fn=lambda: None,
            session_system_file=None,
            work_dir=None,
            repo_name=None,
            session=None,
        )
        with pytest.raises(NotImplementedError):
            agent._spawn_owned_session(ProviderModel("model"))

    def test_session_properties_attach_and_detach(self) -> None:
        session = _FakeSession(
            owner="worker",
            pid=123,
            session_id="sess-1",
            dropped_session_count=2,
            sent_count=10,
            received_count=8,
            is_alive_return=True,
        )
        agent = _FakeAgent(session=session)
        assert agent.session is session
        assert agent.session_owner == "worker"
        assert agent.session_alive is True
        assert agent.session_pid == 123
        assert agent.session_id == "sess-1"
        assert agent.session_dropped_count == 2
        assert agent.session_sent_count == 10
        assert agent.session_received_count == 8
        assert agent.detach_session() is session
        assert agent.session is None

    def test_session_id_none_branches(self) -> None:
        assert _FakeAgent().session_id is None
        assert _FakeAgent(session=object()).session_id is None
        assert (
            _FakeAgent(session=type("S", (), {"session_id": 123})()).session_id is None
        )
        assert _FakeAgent().session_dropped_count == 0
        assert _FakeAgent().session_sent_count == 0
        assert _FakeAgent().session_received_count == 0

    def test_ensure_session_requires_factory_inputs(self) -> None:
        with pytest.raises(
            ValueError,
            match="_FakeAgent.ensure_session requires session_system_file and work_dir",
        ):
            _FakeAgent().ensure_session()

    def test_ensure_session_requires_model_when_creating_session(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(
            ValueError,
            match="_FakeAgent.ensure_session requires model when creating a session",
        ):
            _FakeAgent(
                session_system_file=tmp_path / "persona.md",
                work_dir=tmp_path,
            ).ensure_session()

    def test_ensure_session_spawns_owned_and_switches_existing(
        self, tmp_path: Path
    ) -> None:
        spawned = object()
        factory = _FakeCallRecorder(return_value=spawned)
        agent = _FakeAgent(
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
            session_factory=factory,
        )
        agent.ensure_session(agent.voice_model)
        factory.assert_called_once_with(agent.voice_model)
        attached = _FakeSession()
        agent = _FakeAgent(session=attached)
        agent.ensure_session(agent.brief_model)
        attached.switch_model.assert_called_once_with(agent.brief_model)

    def test_recover_session_returns_false_without_session(self) -> None:
        assert _FakeAgent().recover_session() is False

    def test_recover_session_delegates_to_attached_session(self) -> None:
        session = _FakeSession()
        agent = _FakeAgent(session=session)
        assert agent.recover_session() is True
        session.recover.assert_called_once_with()

    def test_resolve_turn_prefers_live_session_over_owned_spawn(
        self, tmp_path: Path
    ) -> None:
        resolved = _FakeSession()
        resolved.prompt.return_value = "ok"
        factory = _FakeCallRecorder()
        agent = _FakeAgent(
            session_fn=lambda: resolved,
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
            session_factory=factory,
        )
        assert agent.generate_reply("hi") == "ok"
        factory.assert_not_called()

    def test_run_turn_requires_model_when_spawning_owned_session(
        self, tmp_path: Path
    ) -> None:
        agent = _FakeAgent(
            session_system_file=tmp_path / "persona.md",
            work_dir=tmp_path,
        )
        with pytest.raises(
            ValueError,
            match="_FakeAgent.run_turn requires model when creating a session",
        ):
            agent.run_turn("hi")

    def test_resolve_turn_raises_resolver_error_or_generic_error(self) -> None:
        agent = _FakeAgent(
            session_fn=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with pytest.raises(RuntimeError, match="boom"):
            agent.generate_reply("hi")
        with pytest.raises(
            RuntimeError, match="_FakeAgent.run_turn could not resolve a session"
        ):
            _FakeAgent(session_fn=lambda: None).generate_reply("hi")

    def test_fresh_session_mode_requires_resettable_session(self) -> None:
        agent = _FakeAgent(session=object())
        with pytest.raises(
            ValueError,
            match="_FakeAgent.run_turn session_mode=fresh requires resettable session",
        ):
            agent.run_turn(
                "hi",
                model=agent.voice_model,
                session_mode=TurnSessionMode.FRESH,
            )

    def test_shared_helper_methods_and_json_parsing(self) -> None:
        session = _FakeSession(session_id="sess-1")
        session.prompt.side_effect = [
            "reply text",
            "branch-name\nextra",
            "status text",
            '{"emoji":"rocket"}',
            "status with session",
            "resumed status",
        ]
        agent = _FakeAgent(session=session)
        assert agent.generate_reply("reply") == "reply text"
        assert agent.generate_branch_name("branch") == "branch-name"
        assert agent.generate_status("status", "system") == "status text"
        assert agent.generate_status_emoji("emoji", "system") == "rocket"
        assert agent.generate_status_with_session("status", "system") == (
            "status with session",
            "sess-1",
        )
        assert agent.resume_status("sess-1", "resume") == "resumed status"

    def test_status_emoji_returns_empty_on_empty_or_bad_json(self) -> None:
        session = _FakeSession()
        session.prompt.side_effect = ["", "not json"]
        agent = _FakeAgent(session=session)
        assert agent.generate_status_emoji("emoji", "system") == ""
        assert agent.generate_status_emoji("emoji", "system") == ""

    def test_resume_status_requires_matching_live_session(self) -> None:
        session = _FakeSession(session_id="other")
        agent = _FakeAgent(session=session)
        with pytest.raises(
            RuntimeError,
            match="_FakeAgent resume helpers require the matching live session",
        ):
            agent.resume_status("sess-1", "resume")

    def test_resume_status_uses_resolved_session_when_not_attached(self) -> None:
        session = _FakeSession(session_id="sess-1")
        session.prompt.return_value = "resumed status"
        agent = _FakeAgent(session_fn=lambda: session)
        assert agent.resume_status("sess-1", "resume") == "resumed status"

    def test_prompt_with_recovery_recovers_after_dead_prompt_failure(self) -> None:
        session = _FakeSession(is_alive_return=False)
        session.prompt.side_effect = [BrokenPipeError("boom"), "done"]
        session.last_turn_cancelled = False
        agent = _FakeAgent(session=session)
        assert agent.run_turn("hi", model=agent.voice_model) == "done"
        session.recover.assert_called_once_with()

    def test_prompt_with_recovery_returns_empty_on_cancel_induced_failure(
        self,
    ) -> None:
        """Regression for #1792.

        When the prompt fails because a peer thread fired a cancel
        during the in-flight turn (claude exits -2 from the interrupt
        signal), ``_prompt_with_recovery`` MUST NOT retry the *prompt*
        on a respawned session — that would silently consume the
        cancel intent and the caller would observe a normal return,
        blocking the worker loop from yielding to
        ``handle_queued_comments``.  Return empty so the caller's
        ``last_turn_cancelled`` check fires.

        Recovery of the dead session DOES happen before returning
        (codex P1 follow-up on PR #1793) — otherwise a subsequent
        prompt would hit the same dead-session + sticky-cancel state
        and loop.

        Locked in by the FSM oracle from
        ``models/cancel_survives_respawn.v``: any path that observes
        ``CancelFire`` cannot return ``RetNormal``."""
        session = _FakeSession(is_alive_return=False)
        # ``cancel_observed`` on the exception is the authoritative
        # channel — real :meth:`PromptSession.prompt` impls capture
        # the cancel bit INSIDE the lock at the moment of failure and
        # attach it to the raised exception (codex P1 family on PR
        # #1793).  The recovery loop reads it by value instead of
        # racing with the next acquirer for ``last_turn_cancelled``.
        boom = BrokenPipeError("claude exited with code -2")
        boom.cancel_observed = True  # type: ignore[attr-defined]
        session.prompt.side_effect = boom
        session.last_turn_cancelled = True
        agent = _FakeAgent(session=session)
        assert agent.run_turn("hi", model=agent.voice_model) == ""
        # Prompt MUST NOT have been retried — that's the bug.
        assert session.prompt.call_count == 1
        # Session MUST have been recovered so the next caller has a
        # live session waiting — defense in depth for the
        # ``retry_on_preempt=True`` callers and the next worker turn.
        session.recover.assert_called_once_with()

    def test_prompt_with_recovery_raises_after_second_dead_empty_result(self) -> None:
        session = _FakeSession()
        session.prompt.side_effect = ["", ""]
        session.last_turn_cancelled = False
        session.is_alive.side_effect = [False, False]
        agent = _FakeAgent(session=session)
        with pytest.raises(RuntimeError, match="session died during prompt"):
            agent.run_turn("hi", model=agent.voice_model)
        session.recover.assert_called_once_with()

    def test_generate_reply_recovers_after_dead_prompt_failure(self) -> None:
        # Regression: _run_shared_turn used to call session.prompt directly
        # without recovery, so a BrokenPipe from a stale subprocess killed
        # the worker thread and left the persistent ClaudeSession FSM stuck
        # in Sending forever.
        session = _FakeSession(is_alive_return=False)
        session.prompt.side_effect = [BrokenPipeError("boom"), "ok"]
        agent = _FakeAgent(session=session)
        assert agent.generate_reply("hi") == "ok"
        session.recover.assert_called_once_with()

    def test_run_turn_retries_after_preempt(self) -> None:
        session = _FakeSession()
        session.last_turn_cancelled = False
        prompts = iter(["partial", "done"])

        def prompt(*args: object, **kwargs: object) -> str:
            result = next(prompts)
            session.last_turn_cancelled = result == "partial"
            return result

        session.prompt.side_effect = prompt
        agent = _FakeAgent(session=session)
        assert agent.run_turn("hi", retry_on_preempt=True) == "done"
        assert session.prompt.call_count == 2

    def test_state_updater_defaults_to_none(self) -> None:
        agent = _FakeAgent()
        assert agent.state_updater is None

    def test_state_updater_stores_injected_updater(self) -> None:
        _, fake = create_atomic(_make_fido_state_with_repo("test/repo"))
        agent = _FakeAgent(state_updater=fake)
        assert agent.state_updater is fake

    # ── publish_metrics (SnapshotPublisher) ─────────────────────────────────

    def test_publish_metrics_noop_without_updater(self) -> None:
        """publish_metrics does nothing when state_updater is None."""
        agent = _FakeAgent(repo_name="test/repo")
        # No state_updater → must not raise.
        agent.publish_metrics(
            owner="worker",
            alive=True,
            pid=42,
            dropped_count=0,
            sent_count=3,
            received_count=1,
        )

    def test_publish_metrics_noop_without_repo_name(self) -> None:
        """publish_metrics does nothing when repo_name is None."""
        _, updater = create_atomic(
            FidoState(
                repos=frozendict(),
                github_limits=_ZERO_GITHUB_LIMITS,
                process_started_at=_EPOCH,
            )
        )
        agent = _FakeAgent(state_updater=updater)
        # repo_name is None → must not raise (no lens path to navigate).
        agent.publish_metrics(
            owner=None,
            alive=True,
            pid=None,
            dropped_count=0,
            sent_count=5,
            received_count=0,
        )

    def test_publish_metrics_writes_snapshot_fields(self) -> None:
        """publish_metrics installs a ProviderSnapshot with correct fields."""
        repo_name = "owner/myrepo"
        reader, updater = create_atomic(_make_fido_state_with_repo(repo_name))
        agent = _FakeAgent(
            repo_name=repo_name,
            state_updater=updater,
        )
        agent.publish_metrics(
            owner="worker-thread",
            alive=True,
            pid=99,
            dropped_count=1,
            sent_count=7,
            received_count=4,
        )
        provider = reader.get().repos[repo_name].provider
        assert provider is not None
        assert isinstance(provider, ProviderSnapshot)
        assert provider.session_owner == "worker-thread"
        assert provider.session_alive is True
        assert provider.session_pid == 99
        assert provider.session_dropped_count == 1
        assert provider.session_sent_count == 7
        assert provider.session_received_count == 4

    def test_publish_metrics_reflects_updated_sent_count(self) -> None:
        """A second publish with a new count reflects the new value."""
        repo_name = "owner/myrepo"
        reader, updater = create_atomic(_make_fido_state_with_repo(repo_name))
        agent = _FakeAgent(
            repo_name=repo_name,
            state_updater=updater,
        )
        agent.publish_metrics(
            owner=None,
            alive=True,
            pid=None,
            dropped_count=0,
            sent_count=0,
            received_count=0,
        )
        assert reader.get().repos[repo_name].provider is not None
        assert reader.get().repos[repo_name].provider.session_sent_count == 0  # type: ignore[union-attr]

        agent.publish_metrics(
            owner=None,
            alive=True,
            pid=None,
            dropped_count=0,
            sent_count=3,
            received_count=0,
        )
        assert reader.get().repos[repo_name].provider.session_sent_count == 3  # type: ignore[union-attr]

    def test_publish_metrics_reflects_updated_received_count(self) -> None:
        """A second publish with a new received count reflects the new value."""
        repo_name = "owner/myrepo"
        reader, updater = create_atomic(_make_fido_state_with_repo(repo_name))
        agent = _FakeAgent(
            repo_name=repo_name,
            state_updater=updater,
        )
        agent.publish_metrics(
            owner=None,
            alive=True,
            pid=None,
            dropped_count=0,
            sent_count=0,
            received_count=0,
        )
        assert reader.get().repos[repo_name].provider is not None
        assert reader.get().repos[repo_name].provider.session_received_count == 0  # type: ignore[union-attr]

        agent.publish_metrics(
            owner=None,
            alive=True,
            pid=None,
            dropped_count=0,
            sent_count=0,
            received_count=5,
        )
        assert reader.get().repos[repo_name].provider.session_received_count == 5  # type: ignore[union-attr]
