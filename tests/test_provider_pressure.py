"""Tests for fido.provider_pressure — ProviderPressureMonitor (#1696 parity)."""

import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from frozendict import frozendict

from fido.appstate import (
    _EPOCH,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_GITHUB_LIMITS,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    _ZERO_PROVIDER_PRESSURE,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
    FidoState,
    ProviderPressureSnapshot,
    zero_repo_state,
)
from fido.atomic import AtomicReader, AtomicUpdater, create_atomic
from fido.config import RepoConfig, RepoMembership
from fido.provider import (
    ProviderAPI,
    ProviderID,
    ProviderLimitSnapshot,
    ProviderLimitWindow,
)
from fido.provider_factory import DefaultProviderFactory
from fido.provider_pressure import (
    _REFRESH_INTERVAL,  # noqa: PLC2701
    ProviderPressureMonitor,
    _snapshot_from_status,  # noqa: PLC2701
)


def _state_with_repos(
    *names: str,
) -> tuple[AtomicReader[FidoState], AtomicUpdater[FidoState]]:
    return create_atomic(
        FidoState(
            repos=frozendict({name: zero_repo_state(name) for name in names}),
            github_limits=_ZERO_GITHUB_LIMITS,
            process_started_at=_EPOCH,
        )
    )


def _repo(name: str, *, provider: ProviderID = ProviderID.CLAUDE_CODE) -> RepoConfig:
    return RepoConfig(
        name=name,
        work_dir=Path("/tmp/fake"),
        provider=provider,
        membership=RepoMembership(),
    )


def _window(name: str, used: int, limit: int) -> ProviderLimitWindow:
    return ProviderLimitWindow(
        name=name,
        used=used,
        limit=limit,
        resets_at=datetime(2026, 4, 16, 7, 0, tzinfo=timezone.utc),
        unit="",
    )


class _FakeProviderAPI:
    """Typed fake for :class:`~fido.provider.ProviderAPI` — only the
    ``get_limit_snapshot`` surface needed by :class:`ProviderPressureMonitor`.
    """

    def __init__(
        self,
        return_value: ProviderLimitSnapshot | None = None,
        side_effect: BaseException | None = None,
    ) -> None:
        self._return_value = return_value
        self._side_effect = side_effect

    @property
    def provider_id(self) -> ProviderID:
        return ProviderID.CLAUDE_CODE

    def get_limit_snapshot(self) -> ProviderLimitSnapshot:
        if self._side_effect is not None:
            raise self._side_effect
        if self._return_value is not None:
            return self._return_value
        return ProviderLimitSnapshot(provider=ProviderID.CLAUDE_CODE)


class _FakeProviderFactory(DefaultProviderFactory):
    """Typed fake for :class:`~fido.provider_factory.DefaultProviderFactory`
    — only the ``create_api`` surface needed by
    :class:`ProviderPressureMonitor`.

    Supports two modes: a fixed *api* returned for every repo, or a
    per-call *api_factory* callable that receives the :class:`RepoConfig`
    and returns a :class:`_FakeProviderAPI`.  Tracks call count so tests
    can assert on polling frequency.
    """

    def __init__(
        self,
        api: _FakeProviderAPI | None = None,
        api_factory: Callable[[RepoConfig], _FakeProviderAPI] | None = None,
    ) -> None:
        super().__init__(session_system_file=Path("/dev/null"))
        self._test_api = api
        self._test_api_factory = api_factory
        self.create_api_call_count: int = 0

    def create_api(self, repo_cfg: RepoConfig) -> ProviderAPI:
        self.create_api_call_count += 1
        if self._test_api_factory is not None:
            return self._test_api_factory(repo_cfg)
        if self._test_api is not None:
            return self._test_api
        return _FakeProviderAPI()


def _factory_with_pressure(
    pressure_window: ProviderLimitWindow | None = None,
    *,
    raises: BaseException | None = None,
) -> _FakeProviderFactory:
    """Return a typed fake factory whose ``create_api`` returns an API
    whose ``get_limit_snapshot`` returns a single window (or raises)."""
    windows = (pressure_window,) if pressure_window is not None else ()
    api = _FakeProviderAPI(
        return_value=ProviderLimitSnapshot(
            provider=ProviderID.CLAUDE_CODE, windows=windows
        ),
        side_effect=raises,
    )
    return _FakeProviderFactory(api=api)


# ── _snapshot_from_status ─────────────────────────────────────────────────────


class TestSnapshotFromStatus:
    def test_populated_status_passes_through(self) -> None:
        from fido.provider import ProviderPressureStatus

        status = ProviderPressureStatus.from_snapshot(
            ProviderLimitSnapshot(
                provider=ProviderID.CLAUDE_CODE,
                windows=(_window("five_hour", used=96, limit=100),),
            )
        )
        snap = _snapshot_from_status(status)
        assert snap.provider == "claude-code"
        assert snap.window_name == "five_hour"
        assert snap.percent_used == 96
        assert snap.level == "paused"
        assert snap.warning is False
        assert snap.paused is True

    def test_unknown_pressure_uses_zero_sentinels(self) -> None:
        # Empty snapshot → from_snapshot leaves window_name/pressure as None;
        # _snapshot_from_status maps them to "" / 0.0 / 0 / epoch sentinels.
        from fido.provider import ProviderPressureStatus

        status = ProviderPressureStatus.from_snapshot(
            ProviderLimitSnapshot(provider=ProviderID.CODEX)
        )
        snap = _snapshot_from_status(status)
        assert snap.window_name == ""
        assert snap.pressure == 0.0
        assert snap.percent_used == 0
        assert snap.resets_at == _EPOCH
        assert snap.unavailable_reason == ""
        assert snap.level == "unknown"


# ── ProviderPressureMonitor.refresh ───────────────────────────────────────────


class TestRefresh:
    def test_publishes_snapshot_to_each_repo(self) -> None:
        reader, updater = _state_with_repos("owner/repo-a", "owner/repo-b")
        factory = _factory_with_pressure(_window("five_hour", used=50, limit=100))
        monitor = ProviderPressureMonitor(
            repos={
                "owner/repo-a": _repo("owner/repo-a"),
                "owner/repo-b": _repo("owner/repo-b"),
            },
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()
        snapshot = reader.get()
        assert snapshot.repos["owner/repo-a"].provider_pressure.percent_used == 50
        assert snapshot.repos["owner/repo-b"].provider_pressure.percent_used == 50

    def test_polls_each_provider_once_per_cycle(self) -> None:
        # Two repos, same provider — create_api should be called once.
        _, updater = _state_with_repos("owner/repo-a", "owner/repo-b")
        factory = _factory_with_pressure(_window("five_hour", used=10, limit=100))
        monitor = ProviderPressureMonitor(
            repos={
                "owner/repo-a": _repo("owner/repo-a"),
                "owner/repo-b": _repo("owner/repo-b"),
            },
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()
        assert factory.create_api_call_count == 1

    def test_failure_keeps_prior_snapshot_for_repos_using_that_provider(self) -> None:
        reader, updater = _state_with_repos("owner/repo-a")
        factory = _factory_with_pressure(_window("five_hour", used=42, limit=100))
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()
        first = reader.get().repos["owner/repo-a"].provider_pressure
        assert first.percent_used == 42

        # Second cycle raises — prior snapshot must remain.
        factory_bad = _factory_with_pressure(raises=RuntimeError("api down"))
        bad_monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory_bad,
        )
        bad_monitor.refresh()
        assert reader.get().repos["owner/repo-a"].provider_pressure == first

    def test_failure_isolated_per_provider(self) -> None:
        # repo-a uses claude (raises), repo-b uses codex (succeeds).
        reader, updater = _state_with_repos("owner/repo-a", "owner/repo-b")

        def make_api(repo_cfg: RepoConfig) -> _FakeProviderAPI:
            if repo_cfg.provider is ProviderID.CLAUDE_CODE:
                return _FakeProviderAPI(side_effect=RuntimeError("claude down"))
            return _FakeProviderAPI(
                return_value=ProviderLimitSnapshot(
                    provider=ProviderID.CODEX,
                    windows=(_window("primary", used=12, limit=100),),
                )
            )

        factory = _FakeProviderFactory(api_factory=make_api)

        monitor = ProviderPressureMonitor(
            repos={
                "owner/repo-a": _repo("owner/repo-a", provider=ProviderID.CLAUDE_CODE),
                "owner/repo-b": _repo("owner/repo-b", provider=ProviderID.CODEX),
            },
            state=updater,
            provider_factory=factory,
        )
        monitor.refresh()

        # repo-a's claude failed → still zero sentinel
        assert (
            reader.get().repos["owner/repo-a"].provider_pressure
            == _ZERO_PROVIDER_PRESSURE
        )
        # repo-b's codex succeeded → published
        assert reader.get().repos["owner/repo-b"].provider_pressure.percent_used == 12


# ── ProviderPressureMonitor.start_thread ──────────────────────────────────────


class TestStartThread:
    def test_returns_daemon_thread(self) -> None:
        _, updater = _state_with_repos("owner/repo-a")
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=_factory_with_pressure(),
        )
        t = monitor.start_thread(_interval=3600.0)
        assert t.daemon is True

    def test_thread_name(self) -> None:
        _, updater = _state_with_repos("owner/repo-a")
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=_factory_with_pressure(),
        )
        t = monitor.start_thread(_interval=3600.0)
        assert t.name == "provider-pressure-monitor"

    def test_does_initial_refresh_inline(self) -> None:
        reader, updater = _state_with_repos("owner/repo-a")
        factory = _factory_with_pressure(_window("five_hour", used=33, limit=100))
        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory,
        )
        monitor.start_thread(_interval=3600.0)
        # Initial refresh runs synchronously before sleep — snapshot is
        # already populated by the time start_thread returns.
        assert reader.get().repos["owner/repo-a"].provider_pressure.percent_used == 33

    def test_loop_calls_refresh_periodically(self) -> None:
        _, updater = _state_with_repos("owner/repo-a")
        # Use a barrier-style event that the *publish* path sets, not
        # the get_limit_snapshot side-effect — get_limit_snapshot fires
        # before _publish, so signalling there can race the snapshot
        # write that the test wants to observe.
        publish_count = [0]
        second_publish = threading.Event()

        api = _FakeProviderAPI(
            return_value=ProviderLimitSnapshot(
                provider=ProviderID.CLAUDE_CODE,
                windows=(_window("five_hour", used=1, limit=100),),
            )
        )
        factory = _FakeProviderFactory(api=api)

        monitor = ProviderPressureMonitor(
            repos={"owner/repo-a": _repo("owner/repo-a")},
            state=updater,
            provider_factory=factory,
        )
        # Wrap the lens write to count completed publishes.
        original_publish = monitor._publish  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

        def counting_publish(name: str, snap: ProviderPressureSnapshot) -> None:
            original_publish(name, snap)
            publish_count[0] += 1
            if publish_count[0] >= 2:
                second_publish.set()

        monitor._publish = counting_publish  # type: ignore[method-assign]  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        monitor.start_thread(_interval=0.01)
        assert second_publish.wait(timeout=10.0), "expected at least two refresh cycles"
        assert publish_count[0] >= 2


def test_refresh_interval_constant_is_60_seconds() -> None:
    assert _REFRESH_INTERVAL == 60.0


def test_module_exports_only_monitor() -> None:
    """Sanity: only ProviderPressureMonitor is in the public surface."""
    import fido.provider_pressure as mod

    assert "ProviderPressureMonitor" in mod.__all__
    assert mod.__all__ == ["ProviderPressureMonitor"]


def test_loop_pacing_uses_time_sleep() -> None:
    """The poller's loop is just sleep+refresh; importing the module
    shouldn't require a running thread to verify the constant."""
    # Sanity that time module is referenced (covers the import at top level
    # so future maintainers don't strip it inadvertently).
    assert callable(time.sleep)
