from pathlib import Path
from typing import Any

from fido.github import GitHub
from fido.sync_tasks_cli import main


class _FakeGitHub(GitHub):
    """GitHub subclass that skips token fetch — needed only as a no-op
    instantiable class for sync-tasks-CLI tests where the ``gh`` object
    is passed to ``_sync_tasks`` but never called."""

    def __init__(self) -> None:
        # Do not call super().__init__() — that reads the token from a file
        # or environment variable, which is unavailable in unit tests.
        pass


def test_main_syncs_explicit_work_dir(tmp_path: Path) -> None:
    sync_calls: list[tuple[Any, Any]] = []

    def fake_sync(work_dir: Path, gh: object) -> None:
        sync_calls.append((work_dir, gh))

    main([str(tmp_path)], _GitHub=_FakeGitHub, _sync_tasks=fake_sync)

    assert len(sync_calls) == 1
    assert sync_calls[0][0] == tmp_path


def test_main_defaults_to_cwd() -> None:
    sync_calls: list[tuple[Any, Any]] = []

    def fake_sync(work_dir: Path, gh: object) -> None:
        sync_calls.append((work_dir, gh))

    main([], _GitHub=_FakeGitHub, _sync_tasks=fake_sync)

    assert len(sync_calls) == 1
    assert sync_calls[0][0] == Path.cwd()
