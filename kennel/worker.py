"""Fido worker — runs one iteration of the work loop for a single repo."""

from __future__ import annotations

import fcntl
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

log = logging.getLogger("kennel")


class LockHeld(Exception):
    """Raised when the fido lock is already held by another process."""


@dataclass
class WorkerContext:
    work_dir: Path
    git_dir: Path
    fido_dir: Path
    lock_fd: IO[str]


def resolve_git_dir(work_dir: Path) -> Path:
    """Return the absolute .git directory for the repo at work_dir."""
    result = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def acquire_lock(fido_dir: Path) -> IO[str]:
    """Acquire the fido lock file exclusively (non-blocking).

    Returns the open file object (must stay open to hold the lock).
    Raises LockHeld if another fido is already running.
    """
    fido_dir.mkdir(parents=True, exist_ok=True)
    lock_path = fido_dir / "lock"
    fd = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.close()
        raise LockHeld("another fido is running")
    return fd


def create_context(work_dir: Path) -> WorkerContext:
    """Build a WorkerContext for work_dir, acquiring the fido lock.

    Raises LockHeld if the lock is already held.
    """
    git_dir = resolve_git_dir(work_dir)
    fido_dir = git_dir / "fido"
    lock_fd = acquire_lock(fido_dir)
    return WorkerContext(
        work_dir=work_dir,
        git_dir=git_dir,
        fido_dir=fido_dir,
        lock_fd=lock_fd,
    )


def run(work_dir: Path) -> int:
    """Run one iteration of the worker loop.

    Returns:
        0 — no more work (all done or idle)
        2 — lock held / transient failure (retry later)
    """
    try:
        ctx = create_context(work_dir)
    except LockHeld:
        log.warning("another fido is running — exiting")
        return 2
    log.info("worker started for %s (git_dir=%s)", work_dir, ctx.git_dir)
    # TODO: main loop implemented in subsequent tasks
    return 0
