from pathlib import Path

from fido.github import GitHub
from fido.sync_tasks_cli import main
from fido.tasks import AutoCompleter, GitDirResolver


class _FakeGitHub(GitHub):
    """GitHub subclass that skips token fetch — needed only as a no-op
    instantiable class for sync-tasks-CLI tests where the ``gh`` object
    is passed to ``sync_tasks_fn`` but never called."""

    def __init__(
        self, token: str | None = None, session: object = None, **_kwargs: object
    ) -> None:
        # Do not call super().__init__() — that reads the token from a file
        # or environment variable, which is unavailable in unit tests.
        pass


class _FakeSyncTasks:
    """Typed fake for SyncTasksFn — records calls for assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, object]] = []

    def __call__(
        self,
        work_dir: Path,
        gh: GitHub,
        *,
        git_dir_resolver: GitDirResolver,
        auto_completer: AutoCompleter,
    ) -> None:
        self.calls.append((work_dir, gh))


def test_main_syncs_explicit_work_dir(tmp_path: Path) -> None:
    fake_sync = _FakeSyncTasks()

    main([str(tmp_path)], github_factory=lambda: _FakeGitHub(), sync_tasks_fn=fake_sync)

    assert len(fake_sync.calls) == 1
    assert fake_sync.calls[0][0] == tmp_path


def test_main_defaults_to_cwd() -> None:
    fake_sync = _FakeSyncTasks()

    main([], github_factory=lambda: _FakeGitHub(), sync_tasks_fn=fake_sync)

    assert len(fake_sync.calls) == 1
    assert fake_sync.calls[0][0] == Path.cwd()
