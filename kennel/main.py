"""Top-level kennel entry point — dispatches to 'serve', 'task', or 'worker'."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv

    # TODO: remove this compat shim once shell scripts are fully removed
    if args and args[0] == "task":
        from kennel.cli import main as task_main

        task_main(args[1:])
    elif args and args[0] == "worker":
        from kennel.worker import run

        work_dir = Path(args[1]) if len(args) > 1 else Path.cwd()
        sys.exit(run(work_dir))
    elif args and args[0] == "watchdog":
        from kennel.watchdog import run

        work_dir = Path(args[1]) if len(args) > 1 else Path.cwd()
        sys.exit(run(work_dir))
    else:
        from kennel.server import run as server_run

        server_run()
