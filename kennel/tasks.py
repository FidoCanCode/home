"""Shared task file operations with flock-based locking."""
from __future__ import annotations

import fcntl
import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("kennel")


def _task_file(work_dir: Path) -> Path:
    return work_dir / ".git" / "fido" / "tasks.json"


def _locked(path: Path, write: bool = False):
    """Context manager: flock the task file."""
    class Lock:
        def __init__(self):
            self.fd = None
        def __enter__(self):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
            self.fd = open(path, "r+")
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            return self
        def __exit__(self, *_):
            if self.fd:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                self.fd.close()
        def read(self) -> list[dict[str, Any]]:
            self.fd.seek(0)
            text = self.fd.read().strip()
            if not text:
                return []
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                log.warning("corrupt tasks.json — resetting")
                return []
        def write(self, tasks: list[dict[str, Any]]) -> None:
            self.fd.seek(0)
            self.fd.truncate()
            json.dump(tasks, self.fd, indent=2)
            self.fd.flush()
    return Lock()


def add_task(work_dir: Path, title: str, description: str = "", status: str = "pending") -> dict[str, Any]:
    """Add a task to the shared task file. Returns the new task."""
    task = {
        "id": str(int(time.time() * 1000)),
        "title": title,
        "description": description,
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = _task_file(work_dir)
    with _locked(path, write=True) as lock:
        tasks = lock.read()
        tasks.append(task)
        lock.write(tasks)
    log.info("task added: %s", title[:80])
    return task


def update_task(work_dir: Path, task_id: str, status: str) -> bool:
    """Update a task's status. Returns True if found."""
    path = _task_file(work_dir)
    with _locked(path, write=True) as lock:
        tasks = lock.read()
        for t in tasks:
            if t["id"] == task_id:
                t["status"] = status
                lock.write(tasks)
                log.info("task %s → %s", task_id, status)
                return True
    return False


def list_tasks(work_dir: Path) -> list[dict[str, Any]]:
    """Read all tasks."""
    path = _task_file(work_dir)
    with _locked(path) as lock:
        return lock.read()


def remove_task(work_dir: Path, task_id: str) -> bool:
    """Remove a task. Returns True if found."""
    path = _task_file(work_dir)
    with _locked(path, write=True) as lock:
        tasks = lock.read()
        new_tasks = [t for t in tasks if t["id"] != task_id]
        if len(new_tasks) < len(tasks):
            lock.write(new_tasks)
            return True
    return False
