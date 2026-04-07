"""Tests for kennel.watchdog — Watchdog class and run() entry point."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from kennel.watchdog import Watchdog, run

# ── resolve_git_dir ────────────────────────────────────────────────────────────


class TestResolveGitDir:
    def test_returns_path(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/repo/.git\n")
            result = Watchdog(tmp_path).resolve_git_dir()
        assert result == Path("/repo/.git")

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="  /a/b/.git  \n")
            result = Watchdog(tmp_path).resolve_git_dir()
        assert result == Path("/a/b/.git")

    def test_calls_correct_command(self, tmp_path: Path) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="/a/.git")
            Watchdog(tmp_path).resolve_git_dir()
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
                Watchdog(tmp_path).resolve_git_dir()


# ── is_lock_free ───────────────────────────────────────────────────────────────


class TestIsLockFree:
    def test_returns_true_when_file_missing(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "fido" / "lock"
        assert Watchdog(tmp_path).is_lock_free(lock_path) is True

    def test_returns_true_when_lock_acquirable(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "lock"
        lock_path.touch()
        assert Watchdog(tmp_path).is_lock_free(lock_path) is True

    def test_returns_false_when_lock_held(self, tmp_path: Path) -> None:

        lock_path = tmp_path / "lock"
        lock_path.touch()
        with patch("kennel.watchdog.fcntl.flock") as mock_flock:
            mock_flock.side_effect = BlockingIOError()
            result = Watchdog(tmp_path).is_lock_free(lock_path)
        assert result is False


# ── is_stale ───────────────────────────────────────────────────────────────────


class TestIsStale:
    def test_returns_false_when_file_missing(self, tmp_path: Path) -> None:
        log_path = tmp_path / "fido.log"
        assert Watchdog(tmp_path).is_stale(log_path) is False

    def test_returns_false_when_file_is_fresh(self, tmp_path: Path) -> None:
        log_path = tmp_path / "fido.log"
        log_path.touch()
        # mtime is "now" — well within 10 minutes
        with patch("time.time", return_value=log_path.stat().st_mtime + 60):
            result = Watchdog(tmp_path).is_stale(log_path)
        assert result is False

    def test_returns_true_when_file_is_stale(self, tmp_path: Path) -> None:
        log_path = tmp_path / "fido.log"
        log_path.touch()
        # pretend "now" is 11 minutes after the file's mtime
        with patch("time.time", return_value=log_path.stat().st_mtime + 11 * 60):
            result = Watchdog(tmp_path).is_stale(log_path)
        assert result is True

    def test_boundary_exactly_ten_minutes_is_not_stale(self, tmp_path: Path) -> None:
        log_path = tmp_path / "fido.log"
        log_path.touch()
        with patch("time.time", return_value=log_path.stat().st_mtime + 10 * 60):
            result = Watchdog(tmp_path).is_stale(log_path)
        assert result is False


# ── get_lock_pids ──────────────────────────────────────────────────────────────


class TestGetLockPids:
    def _run(self, tmp_path: Path, stdout: str) -> list[int]:
        lock_path = tmp_path / "lock"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=stdout)
            result = Watchdog(tmp_path).get_lock_pids(lock_path)
        mock_run.assert_called_once_with(
            ["lsof", str(lock_path)],
            capture_output=True,
            text=True,
        )
        return result

    def test_parses_single_pid(self, tmp_path: Path) -> None:
        stdout = (
            "COMMAND  PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"
            "python  1234 user    3uW  REG    8,1     0 12345 /path/lock\n"
        )
        assert self._run(tmp_path, stdout) == [1234]

    def test_parses_multiple_pids_deduplicated(self, tmp_path: Path) -> None:
        stdout = (
            "COMMAND  PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"
            "python  1234 user    3uW  REG    8,1     0 12345 /path/lock\n"
            "python  5678 user    4uW  REG    8,1     0 12345 /path/lock\n"
            "python  1234 user    5uW  REG    8,1     0 12345 /path/lock\n"
        )
        assert self._run(tmp_path, stdout) == [1234, 5678]

    def test_returns_empty_when_no_processes(self, tmp_path: Path) -> None:
        assert self._run(tmp_path, "") == []

    def test_skips_non_integer_pid(self, tmp_path: Path) -> None:
        # Header row + a data row whose PID field is not an integer
        stdout = (
            "COMMAND  PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"
            "python   N/A user    3uW  REG    8,1     0 12345 /path/lock\n"
        )
        assert self._run(tmp_path, stdout) == []


# ── kill_pids ──────────────────────────────────────────────────────────────────


class TestKillPids:
    def test_kills_each_pid(self, tmp_path: Path) -> None:
        with patch("os.kill") as mock_kill:
            Watchdog(tmp_path).kill_pids([1, 2, 3])
        assert mock_kill.call_args_list == [call(1, 9), call(2, 9), call(3, 9)]

    def test_empty_list_is_no_op(self, tmp_path: Path) -> None:
        with patch("os.kill") as mock_kill:
            Watchdog(tmp_path).kill_pids([])
        mock_kill.assert_not_called()

    def test_ignores_process_lookup_error(self, tmp_path: Path) -> None:
        with patch("os.kill") as mock_kill:
            mock_kill.side_effect = ProcessLookupError()
            # Should not raise
            Watchdog(tmp_path).kill_pids([999])


# ── restart_worker ─────────────────────────────────────────────────────────────


class TestRestartWorker:
    def test_launches_worker_in_background(self, tmp_path: Path) -> None:
        log_path = tmp_path / "log" / "fido.log"
        work_dir = tmp_path / "repo"
        with patch("subprocess.Popen") as mock_popen:
            Watchdog(work_dir).restart_worker(log_path)
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert args[0] == ["uv", "run", "kennel", "worker", str(work_dir)]
        assert kwargs["start_new_session"] is True

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        log_path = tmp_path / "deep" / "nested" / "fido.log"
        with patch("subprocess.Popen"):
            Watchdog(tmp_path).restart_worker(log_path)
        assert log_path.parent.is_dir()

    def test_appends_to_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "fido.log"
        log_path.write_text("existing\n")
        with patch("subprocess.Popen") as mock_popen:
            Watchdog(tmp_path).restart_worker(log_path)
        _, kwargs = mock_popen.call_args
        # stdout and stderr both point to the same open file descriptor
        assert kwargs["stdout"] is kwargs["stderr"]


# ── run (Watchdog.run) ─────────────────────────────────────────────────────────


class TestWatchdogRun:
    def _make(self, tmp_path: Path) -> Watchdog:
        w = Watchdog(tmp_path)
        git_dir = tmp_path / ".git"
        w.resolve_git_dir = MagicMock(return_value=git_dir)
        return w

    def test_returns_zero_when_lock_free(self, tmp_path: Path) -> None:
        w = self._make(tmp_path)
        w.is_lock_free = MagicMock(return_value=True)
        w.is_stale = MagicMock()
        w.kill_pids = MagicMock()
        w.restart_worker = MagicMock()

        assert w.run() == 0
        w.is_stale.assert_not_called()
        w.kill_pids.assert_not_called()
        w.restart_worker.assert_not_called()

    def test_returns_zero_when_running_but_fresh(self, tmp_path: Path) -> None:
        w = self._make(tmp_path)
        w.is_lock_free = MagicMock(return_value=False)
        w.is_stale = MagicMock(return_value=False)
        w.kill_pids = MagicMock()
        w.restart_worker = MagicMock()

        assert w.run() == 0
        w.kill_pids.assert_not_called()
        w.restart_worker.assert_not_called()

    def test_kills_and_restarts_when_stale(self, tmp_path: Path) -> None:
        w = self._make(tmp_path)
        w.is_lock_free = MagicMock(return_value=False)
        w.is_stale = MagicMock(return_value=True)
        w.get_lock_pids = MagicMock(return_value=[42, 43])
        w.kill_pids = MagicMock()
        w.restart_worker = MagicMock()

        with patch("time.sleep") as mock_sleep:
            result = w.run()

        assert result == 0
        w.kill_pids.assert_called_once_with([42, 43])
        mock_sleep.assert_called_once_with(2.0)
        w.restart_worker.assert_called_once()

    def test_lock_path_is_git_dir_fido_lock(self, tmp_path: Path) -> None:
        """is_lock_free receives the correct lock path."""
        w = self._make(tmp_path)
        captured: list[Path] = []

        def capture_lock(lock_path: Path) -> bool:
            captured.append(lock_path)
            return True

        w.is_lock_free = capture_lock
        w.run()

        assert captured == [tmp_path / ".git" / "fido" / "lock"]

    def test_log_path_uses_home(self, tmp_path: Path) -> None:
        """is_stale receives a path under $HOME/log."""
        w = self._make(tmp_path)
        w.is_lock_free = MagicMock(return_value=False)
        captured: list[Path] = []

        def capture_log(log_path: Path) -> bool:
            captured.append(log_path)
            return False

        w.is_stale = capture_log
        w.run()

        assert captured == [Path.home() / "log" / "fido.log"]


# ── module-level run() ─────────────────────────────────────────────────────────


class TestModuleLevelRun:
    def test_delegates_to_watchdog(self, tmp_path: Path) -> None:
        with patch("kennel.watchdog.Watchdog") as MockWatchdog:
            mock_instance = MockWatchdog.return_value
            mock_instance.run.return_value = 0
            result = run(tmp_path)
        MockWatchdog.assert_called_once_with(tmp_path)
        mock_instance.run.assert_called_once_with()
        assert result == 0
