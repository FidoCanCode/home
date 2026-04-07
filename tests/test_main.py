"""Tests for kennel.main — top-level entry point dispatcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kennel.main import main


class TestMain:
    def test_task_subcommand_dispatches_to_cli(self, tmp_path) -> None:
        """'kennel task ...' should delegate to the task CLI."""
        git_dir = tmp_path / ".git" / "fido"
        git_dir.mkdir(parents=True)

        with patch("kennel.tasks.add_task") as mock_add:
            main(["task", str(tmp_path), "add", "my task"])

        mock_add.assert_called_once()

    def test_no_args_dispatches_to_server(self) -> None:
        """No 'task' prefix should invoke the server."""
        with patch("kennel.server.run") as mock_run:
            main([])

        mock_run.assert_called_once()

    def test_server_args_dispatches_to_server(self) -> None:
        """Server args (no 'task' prefix) should invoke the server."""
        with patch("kennel.server.run") as mock_run:
            main(["--port", "9000"])

        mock_run.assert_called_once()

    def test_argv_none_uses_sys_argv(self) -> None:
        """When argv is None, sys.argv[1:] is used."""
        with (
            patch("sys.argv", ["kennel"]),
            patch("kennel.server.run") as mock_run,
        ):
            main()

        mock_run.assert_called_once()

    def test_argv_none_task_uses_sys_argv(self, tmp_path) -> None:
        """When argv is None and sys.argv has 'task', dispatches to CLI."""
        git_dir = tmp_path / ".git" / "fido"
        git_dir.mkdir(parents=True)

        with (
            patch("sys.argv", ["kennel", "task", str(tmp_path), "list"]),
            patch("kennel.tasks.list_tasks", return_value=[]),
        ):
            main()  # should not raise

    def test_worker_subcommand_dispatches_to_worker(self, tmp_path) -> None:
        """'kennel worker <work_dir>' should call worker.run and sys.exit."""
        with patch("kennel.worker.run", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                main(["worker", str(tmp_path)])
        mock_run.assert_called_once_with(Path(str(tmp_path)))
        assert exc_info.value.code == 0

    def test_worker_subcommand_exits_with_run_code(self, tmp_path) -> None:
        """worker exit code is passed through sys.exit."""
        with patch("kennel.worker.run", return_value=2):
            with pytest.raises(SystemExit) as exc_info:
                main(["worker", str(tmp_path)])
        assert exc_info.value.code == 2

    def test_worker_subcommand_defaults_to_cwd(self) -> None:
        """'kennel worker' with no path defaults to Path.cwd()."""
        with patch("kennel.worker.run", return_value=0) as mock_run:
            with pytest.raises(SystemExit):
                main(["worker"])
        mock_run.assert_called_once_with(Path.cwd())

    def test_argv_none_worker_uses_sys_argv(self, tmp_path) -> None:
        """When argv is None and sys.argv has 'worker', dispatches to worker."""
        with (
            patch("sys.argv", ["kennel", "worker", str(tmp_path)]),
            patch("kennel.worker.run", return_value=0),
            pytest.raises(SystemExit),
        ):
            main()

    def test_watchdog_subcommand_dispatches_to_watchdog(self, tmp_path) -> None:
        """'kennel watchdog <work_dir>' should call watchdog.run and sys.exit."""
        with patch("kennel.watchdog.run", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                main(["watchdog", str(tmp_path)])
        mock_run.assert_called_once_with(Path(str(tmp_path)))
        assert exc_info.value.code == 0

    def test_watchdog_subcommand_exits_with_run_code(self, tmp_path) -> None:
        """watchdog exit code is passed through sys.exit."""
        with patch("kennel.watchdog.run", return_value=2):
            with pytest.raises(SystemExit) as exc_info:
                main(["watchdog", str(tmp_path)])
        assert exc_info.value.code == 2

    def test_watchdog_subcommand_defaults_to_cwd(self) -> None:
        """'kennel watchdog' with no path defaults to Path.cwd()."""
        with patch("kennel.watchdog.run", return_value=0) as mock_run:
            with pytest.raises(SystemExit):
                main(["watchdog"])
        mock_run.assert_called_once_with(Path.cwd())

    def test_argv_none_watchdog_uses_sys_argv(self, tmp_path) -> None:
        """When argv is None and sys.argv has 'watchdog', dispatches to watchdog."""
        with (
            patch("sys.argv", ["kennel", "watchdog", str(tmp_path)]),
            patch("kennel.watchdog.run", return_value=0),
            pytest.raises(SystemExit),
        ):
            main()
