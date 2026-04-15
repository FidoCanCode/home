from __future__ import annotations

import os
from pathlib import Path

from kennel.reply_promises import (
    ReplyPromise,
    add_reply_promise,
    list_reply_promises,
    remove_reply_promise,
)


def test_add_reply_promise_creates_empty_file(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    path = add_reply_promise(fido_dir, "pulls", 123)
    assert path == fido_dir / "reply-promises" / "pulls-123"
    assert path.exists()
    assert path.read_text() == ""


def test_add_reply_promise_is_idempotent(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    first = add_reply_promise(fido_dir, "issues", 55)
    first.write_text("")
    second = add_reply_promise(fido_dir, "issues", 55)
    assert first == second
    assert list_reply_promises(fido_dir) == [
        ReplyPromise(comment_type="issues", comment_id=55, path=first)
    ]


def test_remove_reply_promise_deletes_file(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    path = add_reply_promise(fido_dir, "pulls", 321)
    remove_reply_promise(fido_dir, "pulls", 321)
    assert not path.exists()


def test_remove_reply_promise_missing_file_is_noop(tmp_path: Path) -> None:
    remove_reply_promise(tmp_path / "fido", "pulls", 999)


def test_list_reply_promises_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    assert list_reply_promises(tmp_path / "fido") == []


def test_list_reply_promises_sorts_by_mtime(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    first = add_reply_promise(fido_dir, "pulls", 1)
    second = add_reply_promise(fido_dir, "issues", 2)
    os.utime(first, ns=(1, 1))
    os.utime(second, ns=(2, 2))
    assert list_reply_promises(fido_dir) == [
        ReplyPromise(comment_type="pulls", comment_id=1, path=first),
        ReplyPromise(comment_type="issues", comment_id=2, path=second),
    ]


def test_list_reply_promises_ignores_non_files_and_bad_names(tmp_path: Path) -> None:
    fido_dir = tmp_path / "fido"
    promise_dir = fido_dir / "reply-promises"
    promise_dir.mkdir(parents=True)
    (promise_dir / "pulls-42").touch()
    (promise_dir / "junk").touch()
    (promise_dir / "pulls-nope").touch()
    (promise_dir / "comments-7").touch()
    (promise_dir / "subdir").mkdir()
    assert list_reply_promises(fido_dir) == [
        ReplyPromise(comment_type="pulls", comment_id=42, path=promise_dir / "pulls-42")
    ]
