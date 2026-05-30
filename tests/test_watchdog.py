"""Tests for fido.watchdog — Watchdog class and run() entry point."""

import time
from collections.abc import Callable
from pathlib import Path

import pytest
import requests

from fido.config import RepoConfig as _RepoConfig
from fido.provider import ProviderID
from fido.watchdog import (
    _RECONCILE_INTERVAL,  # noqa: PLC2701
    _STALE_THRESHOLD,  # noqa: PLC2701
    IssueReconcileWatchdog,
    Watchdog,
    run,
)


class RepoConfig(_RepoConfig):
    def __init__(
        self,
        *args: object,
        provider: ProviderID = ProviderID.CLAUDE_CODE,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, provider=provider, **kwargs)


def _repo(name: str = "owner/repo") -> RepoConfig:
    return RepoConfig(name=name, work_dir=Path("/tmp/repo"))


class _FakeWatchdogRegistry:
    """Typed fake for WorkerRegistry — only the Watchdog/IssueReconcileWatchdog surface."""

    def __init__(self) -> None:
        self.is_alive_return: bool = True
        self.is_alive_side_effect: Callable[[str], bool] | None = None
        self.is_alive_calls: list[str] = []
        self.get_thread_crash_error_return: str | None = None
        self.record_crash_calls: list[tuple[str, str]] = []
        self.record_crash_side_effect: Callable[[str, str], None] | None = None
        self.start_calls: list[_RepoConfig] = []
        self.start_side_effect: Callable[[_RepoConfig], None] | None = None
        self.stop_and_join_call_count: int = 0
        self.get_issue_cache_return: _FakeIssueCache | None = None
        self.get_issue_cache_calls: list[str] = []

    def is_alive(self, repo_name: str) -> bool:
        self.is_alive_calls.append(repo_name)
        if self.is_alive_side_effect is not None:
            return self.is_alive_side_effect(repo_name)
        return self.is_alive_return

    def get_thread_crash_error(self, repo_name: str) -> str | None:
        return self.get_thread_crash_error_return

    def record_crash(self, repo_name: str, error: str) -> None:
        self.record_crash_calls.append((repo_name, error))
        if self.record_crash_side_effect is not None:
            self.record_crash_side_effect(repo_name, error)

    def start(self, repo_cfg: _RepoConfig) -> None:
        self.start_calls.append(repo_cfg)
        if self.start_side_effect is not None:
            self.start_side_effect(repo_cfg)

    def stop_and_join(self, *args: object, **kwargs: object) -> None:
        self.stop_and_join_call_count += 1

    def get_issue_cache(self, repo_name: str) -> "_FakeIssueCache":
        self.get_issue_cache_calls.append(repo_name)
        assert self.get_issue_cache_return is not None, (
            "set get_issue_cache_return first"
        )
        return self.get_issue_cache_return


class _FakeIssueCache:
    """Typed fake for IssueCache — reconcile surface for IssueReconcileWatchdog."""

    def __init__(self, *, is_loaded: bool = True) -> None:
        self.is_loaded = is_loaded
        self.reconcile_with_inventory_calls: list[tuple[list, dict]] = []
        self.reconcile_with_inventory_return: int = 0

    def reconcile_with_inventory(
        self, issues: list, *, snapshot_started_at: object
    ) -> int:
        self.reconcile_with_inventory_calls.append(
            (issues, {"snapshot_started_at": snapshot_started_at})
        )
        return self.reconcile_with_inventory_return


class _FakeGitHubForWatchdog:
    """Typed fake for GitHub — only the find_all_open_issues surface."""

    def __init__(self) -> None:
        self.find_all_open_issues_return: list[dict[str, object]] = []
        self.find_all_open_issues_calls: list[tuple[str, str]] = []
        self.find_all_open_issues_side_effect: (
            Callable[[str, str], list[dict[str, object]]] | BaseException | None
        ) = None

    def find_all_open_issues(self, owner: str, name: str) -> list[dict[str, object]]:
        self.find_all_open_issues_calls.append((owner, name))
        if isinstance(self.find_all_open_issues_side_effect, BaseException):
            raise self.find_all_open_issues_side_effect
        if callable(self.find_all_open_issues_side_effect):
            return self.find_all_open_issues_side_effect(owner, name)
        return self.find_all_open_issues_return


def _make(
    repos: dict[str, RepoConfig] | None = None,
) -> tuple[Watchdog, _FakeWatchdogRegistry]:
    if repos is None:
        repos = {"owner/repo": _repo()}
    registry = _FakeWatchdogRegistry()
    return Watchdog(registry, repos), registry  # type: ignore[arg-type]


# ── Watchdog.run ───────────────────────────────────────────────────────────────


class TestWatchdogRun:
    def test_returns_zero(self) -> None:
        w, registry = _make()
        registry.is_alive_return = True
        assert w.run() == 0

    def test_does_nothing_when_thread_alive(self) -> None:
        """A live thread is never restarted, even if it looks stale.  Stale
        threads are claude's problem — claude has its own idle timeout."""
        w, registry = _make()
        registry.is_alive_return = True
        w.run()
        assert registry.start_calls == []
        assert registry.stop_and_join_call_count == 0

    def test_restarts_dead_thread(self) -> None:
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive_return = False
        registry.get_thread_crash_error_return = None
        w.run()
        assert registry.start_calls == [repo_cfg]

    def test_checks_is_alive_with_repo_name(self) -> None:
        repo_cfg = _repo("myorg/myrepo")
        w, registry = _make({"myorg/myrepo": repo_cfg})
        registry.is_alive_return = True
        w.run()
        assert registry.is_alive_calls == ["myorg/myrepo"]

    def test_restarts_only_dead_threads_across_multiple_repos(self) -> None:
        alive_cfg = _repo("org/alive")
        dead_cfg = _repo("org/dead")
        repos = {"org/alive": alive_cfg, "org/dead": dead_cfg}
        w, registry = _make(repos)

        def is_alive(name: str) -> bool:
            return name == "org/alive"

        registry.is_alive_side_effect = is_alive
        registry.get_thread_crash_error_return = None
        w.run()
        assert registry.start_calls == [dead_cfg]

    def test_restarts_multiple_dead_threads(self) -> None:
        repo_a = _repo("org/a")
        repo_b = _repo("org/b")
        repos = {"org/a": repo_a, "org/b": repo_b}
        w, registry = _make(repos)
        registry.is_alive_return = False
        registry.get_thread_crash_error_return = None
        w.run()
        assert len(registry.start_calls) == 2
        assert repo_a in registry.start_calls
        assert repo_b in registry.start_calls

    def test_records_crash_before_restart_when_crash_error_set(self) -> None:
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive_return = False
        registry.get_thread_crash_error_return = "RuntimeError: boom"
        w.run()
        assert registry.record_crash_calls == [("owner/repo", "RuntimeError: boom")]
        assert registry.start_calls == [repo_cfg]

    def test_record_crash_called_before_start(self) -> None:
        call_order: list[str] = []
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive_return = False
        registry.get_thread_crash_error_return = "ValueError: oops"
        registry.record_crash_side_effect = lambda *_: call_order.append("record_crash")
        registry.start_side_effect = lambda *_: call_order.append("start")
        w.run()
        assert call_order == ["record_crash", "start"]

    def test_does_not_record_crash_when_crash_error_is_none(self) -> None:
        repo_cfg = _repo()
        w, registry = _make({"owner/repo": repo_cfg})
        registry.is_alive_return = False
        registry.get_thread_crash_error_return = None
        w.run()
        assert registry.record_crash_calls == []
        assert registry.start_calls == [repo_cfg]

    def test_no_repos_is_no_op(self) -> None:
        w, registry = _make({})
        w.run()
        assert registry.is_alive_calls == []
        assert registry.start_calls == []

    def test_is_stale_never_called_for_restart(self) -> None:
        """Stale detection is not a restart trigger.  is_stale may be called
        elsewhere (e.g. /status endpoint) but never by the watchdog itself.

        Verified implicitly: is_stale is not defined on the fake, so any call
        would raise AttributeError and cause this test to fail."""
        w, registry = _make()
        registry.is_alive_return = True
        w.run()  # no AttributeError → is_stale was never accessed

    def test_does_not_stop_and_join_alive_thread(self) -> None:
        w, registry = _make()
        registry.is_alive_return = True
        w.run()
        assert registry.stop_and_join_call_count == 0


# ── display-only constants ────────────────────────────────────────────────────


class TestConstants:
    def test_stale_threshold_is_display_only(self) -> None:
        """_STALE_THRESHOLD exists for /status endpoint display.  It is not
        consumed by the Watchdog class itself — documented via this test."""
        assert _STALE_THRESHOLD > 0


# ── Watchdog.start_thread ─────────────────────────────────────────────────────


def _registry(*, alive: bool = True) -> _FakeWatchdogRegistry:
    reg = _FakeWatchdogRegistry()
    reg.is_alive_return = alive
    return reg


class TestStartThread:
    def _repos(self, tmp_path: Path) -> dict:
        return {"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)}

    def test_returns_daemon_thread(self, tmp_path: Path) -> None:
        t = Watchdog(_registry(), self._repos(tmp_path)).start_thread(  # type: ignore[arg-type]
            _interval=60.0
        )
        assert t.daemon

    def test_thread_name_is_watchdog(self, tmp_path: Path) -> None:
        t = Watchdog(_registry(), self._repos(tmp_path)).start_thread(  # type: ignore[arg-type]
            _interval=60.0
        )
        assert t.name == "watchdog"

    def test_thread_is_alive(self, tmp_path: Path) -> None:
        t = Watchdog(_registry(), self._repos(tmp_path)).start_thread(  # type: ignore[arg-type]
            _interval=60.0
        )
        assert t.is_alive()

    def test_calls_run_periodically(self, tmp_path: Path) -> None:
        reg = _registry()
        Watchdog(reg, self._repos(tmp_path)).start_thread(  # type: ignore[arg-type]
            _interval=0.01
        )
        time.sleep(0.1)
        assert reg.is_alive_calls

    def test_restarts_dead_worker(self, tmp_path: Path) -> None:
        repo_cfg = RepoConfig(name="owner/repo", work_dir=tmp_path)
        reg = _registry(alive=False)
        Watchdog(reg, {"owner/repo": repo_cfg}).start_thread(  # type: ignore[arg-type]
            _interval=0.01
        )
        time.sleep(0.1)
        assert repo_cfg in reg.start_calls


# ── module-level run() ─────────────────────────────────────────────────────────


class TestModuleLevelRun:
    def test_delegates_to_watchdog(self) -> None:
        repo_cfg = _repo()
        repos = {"owner/repo": repo_cfg}
        reg = _registry()
        result = run(reg, repos)  # type: ignore[arg-type]
        assert result == 0
        assert reg.is_alive_calls == ["owner/repo"]

    def test_returns_zero(self) -> None:
        assert run(_registry(), {"owner/repo": _repo()}) == 0  # type: ignore[arg-type]


# ── IssueReconcileWatchdog (closes #812) ───────────────────────────────────────────


def _reconcile(
    repos: dict[str, RepoConfig] | None = None,
    *,
    cache_loaded: bool = True,
) -> tuple[
    IssueReconcileWatchdog,
    _FakeWatchdogRegistry,
    _FakeIssueCache,
    _FakeGitHubForWatchdog,
]:
    if repos is None:
        repos = {"owner/repo": _repo()}
    registry = _FakeWatchdogRegistry()
    cache = _FakeIssueCache(is_loaded=cache_loaded)
    registry.get_issue_cache_return = cache
    gh = _FakeGitHubForWatchdog()
    return (
        IssueReconcileWatchdog(registry, repos, gh),  # type: ignore[arg-type]
        registry,
        cache,
        gh,
    )


class TestIssueReconcileWatchdogRun:
    def test_returns_zero(self) -> None:
        rw, _reg, _cache, gh = _reconcile(cache_loaded=False)
        assert rw.run() == 0
        assert gh.find_all_open_issues_calls == []

    def test_skips_repo_when_cache_not_loaded(self) -> None:
        rw, _reg, cache, gh = _reconcile(cache_loaded=False)
        rw.run()
        assert gh.find_all_open_issues_calls == []
        assert cache.reconcile_with_inventory_calls == []

    def test_reconciles_loaded_cache(self) -> None:
        rw, _reg, cache, gh = _reconcile()
        gh.find_all_open_issues_return = [{"number": 1}]
        cache.reconcile_with_inventory_return = 0
        rw.run()
        assert gh.find_all_open_issues_calls == [("owner", "repo")]
        assert len(cache.reconcile_with_inventory_calls) == 1
        issues, kwargs = cache.reconcile_with_inventory_calls[0]
        assert issues == [{"number": 1}]
        assert "snapshot_started_at" in kwargs

    def test_continues_to_next_repo_when_inventory_call_raises(self) -> None:
        """Transient GitHub/network errors are swallowed — the next hourly
        tick will retry.  The second repo still runs."""
        repo_a = _repo("org/a")
        repo_b = _repo("org/b")
        rw, _reg, cache, gh = _reconcile({"org/a": repo_a, "org/b": repo_b})

        def side_effect(owner: str, _name: str) -> list[dict[str, object]]:
            if owner == "org" and _name == "a":
                raise requests.RequestException("rate limited")
            return [{"number": 9}]

        gh.find_all_open_issues_side_effect = side_effect
        rw.run()
        # b succeeded even though a raised
        assert len(cache.reconcile_with_inventory_calls) == 1

    def test_logic_bug_propagates_from_reconcile(self) -> None:
        """Logic bugs (e.g. KeyError) are NOT swallowed — they propagate so
        the watchdog thread crashes loudly rather than silently corrupting."""
        rw, _reg, _cache, gh = _reconcile()
        gh.find_all_open_issues_side_effect = KeyError("unexpected_key")
        with pytest.raises(KeyError):
            rw.run()

    def test_handles_multiple_repos_independently(self) -> None:
        repo_a = _repo("org/a")
        repo_b = _repo("org/b")
        rw, _reg, cache, gh = _reconcile({"org/a": repo_a, "org/b": repo_b})
        gh.find_all_open_issues_return = []
        cache.reconcile_with_inventory_return = 0
        rw.run()
        assert len(gh.find_all_open_issues_calls) == 2


class TestIssueReconcileWatchdogStartThread:
    def test_returns_daemon_thread(self) -> None:
        rw, _reg, _cache, _gh = _reconcile(cache_loaded=False)
        t = rw.start_thread(_interval=60.0)
        assert t.daemon

    def test_thread_name_is_issue_reconcile_watchdog(self) -> None:
        rw, _reg, _cache, _gh = _reconcile(cache_loaded=False)
        t = rw.start_thread(_interval=60.0)
        assert t.name == "issue-reconcile-watchdog"

    def test_thread_is_alive(self) -> None:
        rw, _reg, _cache, _gh = _reconcile(cache_loaded=False)
        t = rw.start_thread(_interval=60.0)
        assert t.is_alive()

    def test_calls_run_periodically(self) -> None:
        rw, registry, _cache, _gh = _reconcile(cache_loaded=False)
        rw.start_thread(_interval=0.01)
        time.sleep(0.1)
        assert registry.get_issue_cache_calls


class TestReconcileInterval:
    def test_default_interval_is_one_hour(self) -> None:
        assert _RECONCILE_INTERVAL == 3600.0
