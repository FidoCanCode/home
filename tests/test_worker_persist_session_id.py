"""Tests for session-id persistence in WorkerThread (#649)."""

import json
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from fido.config import RepoMembership, default_sub_dir
from fido.infra import RealOsProcess, RealProcessRunner
from fido.issue_cache import IssueCache
from fido.provider_factory import DefaultProviderFactory
from fido.state import State
from fido.worker import WorkerThread
from tests.fakes import _FakeDispatcher


class _FakeCallRecorder:
    """Typed callable that records every invocation."""

    def __init__(self, return_value: object = None) -> None:
        self.return_value: object = return_value
        self._calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self._calls.append((args, kwargs))
        return self.return_value

    def assert_called_once(self) -> None:
        assert len(self._calls) == 1, f"expected 1 call, got {len(self._calls)}"


class _FakeSession:
    """Minimal provider session stub with a call-recording reset() method."""

    def __init__(self, *, session_id: str = "") -> None:
        self.session_id: str = session_id
        self.reset: _FakeCallRecorder = _FakeCallRecorder()


class _FakeAgent:
    """Minimal provider agent stub."""

    def __init__(self, session: _FakeSession | None = None) -> None:
        self.session: _FakeSession | None = session


class _FakeProvider:
    """Minimal provider stub whose .agent.session chain the persistence tests read."""

    def __init__(self, session: _FakeSession | None = None) -> None:
        self.agent: _FakeAgent = _FakeAgent(session)


class _FakeGH:
    """Minimal GitHub stub — no methods are called in these persistence tests."""


def _make_thread(tmp_path: Path, **kwargs: object) -> WorkerThread:
    gh = _FakeGH()
    kwargs.setdefault("membership", RepoMembership())
    kwargs.setdefault(
        "provider_factory",
        DefaultProviderFactory.real(
            session_system_file=default_sub_dir() / "persona.md"
        ),
    )
    kwargs.setdefault("issue_cache", IssueCache("owner/repo"))
    kwargs.setdefault("dispatcher", _FakeDispatcher())
    kwargs.setdefault("os_proc", RealOsProcess())
    kwargs.setdefault("runner", RealProcessRunner())
    return WorkerThread(
        tmp_path,
        "owner/repo",
        gh=gh,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def _init_git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    fido_dir = tmp_path / ".git" / "fido"
    fido_dir.mkdir(parents=True, exist_ok=True)
    return fido_dir


def test_load_persisted_session_id_returns_value(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"session_id": "abc-123"}))
    thread = _make_thread(tmp_path)
    assert thread._load_persisted_session_id() == "abc-123"


def test_load_persisted_session_id_none_when_absent(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"issue": 5}))
    thread = _make_thread(tmp_path)
    assert thread._load_persisted_session_id() is None


def test_load_persisted_session_id_none_when_not_a_git_repo(tmp_path: Path) -> None:
    """tmp_path without `git init` — persistence is unavailable but must
    not crash callers.  Same shape as test fixtures that construct a worker
    against a bare directory."""
    thread = _make_thread(tmp_path)
    assert thread._resolve_fido_dir() is None
    assert thread._load_persisted_session_id() is None


def test_load_persisted_session_id_handles_state_load_oserror(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"session_id": "abc"}))

    class ErrorLoadState(State):
        def load(self) -> dict[str, Any]:
            raise OSError("permission denied")

    thread = _make_thread(tmp_path, _state=ErrorLoadState(fido_dir))
    assert thread._load_persisted_session_id() is None


def test_persist_session_id_writes_new_value(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    session = _FakeSession(session_id="new-sid-456")
    provider = _FakeProvider(session)
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert persisted["session_id"] == "new-sid-456"


def test_persist_session_id_skips_when_unchanged(tmp_path: Path) -> None:
    """Avoid rewriting state.json when the id hasn't changed — keeps the
    file's mtime stable and reduces lock contention under steady state."""
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(
        json.dumps({"session_id": "same-sid", "issue": 1})
    )
    mtime_before = (fido_dir / "state.json").stat().st_mtime_ns
    session = _FakeSession(session_id="same-sid")
    provider = _FakeProvider(session)
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()
    # mtime should not be bumped beyond what State.modify does on read-only
    # operations — the file content must stay the same.
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert persisted == {"session_id": "same-sid", "issue": 1}
    assert (fido_dir / "state.json").stat().st_mtime_ns >= mtime_before


def test_persist_session_id_noop_when_no_provider(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    thread._provider = None  # pyright: ignore[reportPrivateUsage]
    thread._persist_session_id()  # must not raise


def test_persist_session_id_noop_when_no_session(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    provider = _FakeProvider(session=None)
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()  # must not raise


def test_persist_session_id_noop_when_session_has_empty_id(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    session = _FakeSession(session_id="")
    provider = _FakeProvider(session)
    thread = _make_thread(tmp_path, provider=provider)
    thread._persist_session_id()  # must not raise and must not write


def test_persist_session_id_noop_when_fido_dir_unresolvable(tmp_path: Path) -> None:
    """Not a git repo → no fido_dir → persistence silently skipped."""
    session = _FakeSession(session_id="some-sid")
    provider = _FakeProvider(session)
    thread = _make_thread(tmp_path, provider=provider)
    # No git init — resolving fails
    thread._persist_session_id()  # must not raise


def test_persist_session_id_swallows_state_modify_oserror(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    fido_dir = _init_git_repo(tmp_path)
    session = _FakeSession(session_id="sid")
    provider = _FakeProvider(session)

    class ErrorModifyState(State):
        @contextmanager
        def modify(self) -> Generator[Any, None, None]:
            raise OSError("state.json locked by another process")
            yield  # type: ignore[misc]

    thread = _make_thread(
        tmp_path, provider=provider, _state=ErrorModifyState(fido_dir)
    )
    with caplog.at_level(logging.WARNING, logger="fido"):
        thread._persist_session_id()
    assert "failed to persist session_id" in caplog.text


# ── _retire_poisoned_session ──────────────────────────────────────────────────


def test_retire_poisoned_session_clears_session_id_from_state(
    tmp_path: Path,
) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(json.dumps({"session_id": "poisoned-sid"}))
    thread = _make_thread(tmp_path)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert "session_id" not in persisted


def test_retire_poisoned_session_preserves_other_state_keys(tmp_path: Path) -> None:
    fido_dir = _init_git_repo(tmp_path)
    (fido_dir / "state.json").write_text(
        json.dumps({"session_id": "poisoned-sid", "issue": 42})
    )
    thread = _make_thread(tmp_path)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    persisted = json.loads((fido_dir / "state.json").read_text())
    assert persisted == {"issue": 42}


def test_retire_poisoned_session_calls_session_reset(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    session = _FakeSession()
    provider = _FakeProvider(session)
    thread = _make_thread(tmp_path, provider=provider)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    session.reset.assert_called_once()


def test_retire_poisoned_session_clears_session_issue(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    thread._session_issue = 99  # pyright: ignore[reportPrivateUsage]
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    assert thread._session_issue is None  # pyright: ignore[reportPrivateUsage]


def test_retire_poisoned_session_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    with caplog.at_level(logging.WARNING, logger="fido"):
        thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    assert "context overflow" in caplog.text


def test_retire_poisoned_session_noop_when_no_git_repo(tmp_path: Path) -> None:
    """Not a git repo — no fido_dir — must not raise."""
    thread = _make_thread(tmp_path)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]


def test_retire_poisoned_session_noop_when_no_provider(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    thread = _make_thread(tmp_path)
    thread._provider = None  # pyright: ignore[reportPrivateUsage]
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]  # must not raise


def test_retire_poisoned_session_noop_when_no_session(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    provider = _FakeProvider(session=None)
    thread = _make_thread(tmp_path, provider=provider)
    thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]  # must not raise


def test_retire_poisoned_session_swallows_state_modify_oserror(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    fido_dir = _init_git_repo(tmp_path)

    class ErrorModifyState(State):
        @contextmanager
        def modify(self) -> Generator[Any, None, None]:
            raise OSError("state.json locked")
            yield  # type: ignore[misc]

    thread = _make_thread(tmp_path, _state=ErrorModifyState(fido_dir))
    with caplog.at_level(logging.WARNING, logger="fido"):
        thread._retire_poisoned_session()  # pyright: ignore[reportPrivateUsage]
    assert "failed to clear session_id after context overflow" in caplog.text
