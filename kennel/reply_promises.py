"""Durable per-comment reply promises stored as empty files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReplyPromise:
    """One owed reply keyed only by comment type and comment id."""

    comment_type: str
    comment_id: int
    path: Path


def _promise_dir(fido_dir: Path) -> Path:
    return fido_dir / "reply-promises"


def _promise_path(fido_dir: Path, comment_type: str, comment_id: int) -> Path:
    return _promise_dir(fido_dir) / f"{comment_type}-{comment_id}"


def add_reply_promise(fido_dir: Path, comment_type: str, comment_id: int) -> Path:
    """Create the durable promise file if it does not already exist."""
    path = _promise_path(fido_dir, comment_type, comment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def remove_reply_promise(fido_dir: Path, comment_type: str, comment_id: int) -> None:
    """Delete the durable promise file if it exists."""
    _promise_path(fido_dir, comment_type, comment_id).unlink(missing_ok=True)


def list_reply_promises(fido_dir: Path) -> list[ReplyPromise]:
    """Return promises in filesystem timestamp order."""
    result: list[ReplyPromise] = []
    promise_dir = _promise_dir(fido_dir)
    if not promise_dir.exists():
        return result
    for path in promise_dir.iterdir():
        if not path.is_file():
            continue
        comment_type, sep, raw_id = path.name.partition("-")
        if sep != "-" or comment_type not in {"issues", "pulls"}:
            continue
        try:
            comment_id = int(raw_id)
        except ValueError:
            continue
        result.append(
            ReplyPromise(
                comment_type=comment_type,
                comment_id=comment_id,
                path=path,
            )
        )
    result.sort(key=lambda item: item.path.stat().st_mtime_ns)
    return result
