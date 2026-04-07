"""Tests for kennel.worker — WorkerContext, lock acquisition, git context."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kennel.worker import (
    LockHeld,
    WorkerContext,
    acquire_lock,
    create_context,
    resolve_git_dir,
    run,
)


class TestResolveGitDir:
    def test_returns_path(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/some/repo/.git\n")
            result = resolve_git_dir(tmp_path)
        assert result == Path("/some/repo/.git")

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="  /a/b/.git  \n")
            result = resolve_git_dir(tmp_path)
        assert result == Path("/a/b/.git")

    def test_calls_correct_command(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/a/.git")
            resolve_git_dir(tmp_path)
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--absolute-git-dir"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_propagates_called_process_error(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(128, "git")
            with pytest.raises(subprocess.CalledProcessError):
                resolve_git_dir(tmp_path)


class TestAcquireLock:
    def test_returns_open_fd(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = acquire_lock(fido_dir)
        assert not fd.closed
        fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "nested" / "fido"
        fd = acquire_lock(fido_dir)
        assert fido_dir.is_dir()
        fd.close()

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd = acquire_lock(fido_dir)
        assert (fido_dir / "lock").exists()
        fd.close()

    def test_raises_lock_held_when_already_locked(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld):
                acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_lock_held_message(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        try:
            with pytest.raises(LockHeld, match="another fido"):
                acquire_lock(fido_dir)
        finally:
            fd1.close()

    def test_reacquirable_after_release(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / "fido"
        fd1 = acquire_lock(fido_dir)
        fd1.close()
        fd2 = acquire_lock(fido_dir)
        assert not fd2.closed
        fd2.close()


class TestCreateContext:
    def test_returns_worker_context(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        with patch("kennel.worker.resolve_git_dir", return_value=git_dir):
            ctx = create_context(tmp_path)
        assert isinstance(ctx, WorkerContext)
        assert ctx.work_dir == tmp_path
        assert ctx.git_dir == git_dir
        assert ctx.fido_dir == git_dir / "fido"
        ctx.lock_fd.close()

    def test_creates_fido_dir(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        with patch("kennel.worker.resolve_git_dir", return_value=git_dir):
            ctx = create_context(tmp_path)
        assert ctx.fido_dir.is_dir()
        ctx.lock_fd.close()

    def test_propagates_lock_held(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        fido_dir = git_dir / "fido"
        with patch("kennel.worker.resolve_git_dir", return_value=git_dir):
            fd1 = acquire_lock(fido_dir)
            try:
                with pytest.raises(LockHeld):
                    create_context(tmp_path)
            finally:
                fd1.close()


class TestRun:
    def test_returns_2_when_lock_held(self, tmp_path: Path) -> None:
        with patch("kennel.worker.create_context", side_effect=LockHeld("held")):
            assert run(tmp_path) == 2

    def test_returns_0_on_success(self, tmp_path: Path) -> None:
        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        with patch("kennel.worker.create_context", return_value=mock_ctx):
            assert run(tmp_path) == 0

    def test_logs_warning_on_lock_held(self, tmp_path: Path, caplog) -> None:
        import logging

        with patch("kennel.worker.create_context", side_effect=LockHeld("held")):
            with caplog.at_level(logging.WARNING, logger="kennel"):
                run(tmp_path)
        assert "another fido" in caplog.text

    def test_logs_info_on_success(self, tmp_path: Path, caplog) -> None:
        import logging

        mock_ctx = MagicMock(spec=WorkerContext)
        mock_ctx.git_dir = tmp_path / ".git"
        with patch("kennel.worker.create_context", return_value=mock_ctx):
            with caplog.at_level(logging.INFO, logger="kennel"):
                run(tmp_path)
        assert "worker started" in caplog.text
