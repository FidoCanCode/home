"""CLI wrapper for syncing repo tasks to GitHub."""

import sys
from pathlib import Path
from typing import Protocol

from fido.github import (
    GitHub,
    _gh_token,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)
from fido.infra import RealClock, RealProcessRunner
from fido.tasks import (
    AutoCompleter,
    GitDirResolver,
    RealGitDirResolver,
    _auto_complete_ask_tasks,  # noqa: PLC2701  # pyright: ignore[reportPrivateUsage]
)


class SyncTasksFn(Protocol):
    """Typed collaborator: sync tasks.json → PR body work queue."""

    def __call__(
        self,
        work_dir: Path,
        gh: GitHub,
        *,
        git_dir_resolver: GitDirResolver,
        auto_completer: AutoCompleter,
    ) -> None: ...


def main(
    argv: list[str] | None = None,
    *,
    _GitHub: type[GitHub] = GitHub,
    sync_tasks_fn: SyncTasksFn | None = None,
) -> None:
    if sync_tasks_fn is None:
        from fido.tasks import sync_tasks as sync_tasks_fn  # pragma: no cover

    args = sys.argv[1:] if argv is None else argv
    work_dir = Path(args[0]) if args else Path.cwd()
    runner = RealProcessRunner()
    gh = _GitHub(
        runner=runner,
        clock=RealClock(),
        token_fetcher=lambda: _gh_token(runner=runner),
    )
    sync_tasks_fn(
        work_dir,
        gh,
        git_dir_resolver=RealGitDirResolver(runner),
        auto_completer=_auto_complete_ask_tasks,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
