"""Tests for the `uv run tests` entrypoint."""

from __future__ import annotations

from unittest.mock import patch

from kennel.tests_main import main


def test_main_delegates_to_pytest_with_repo_defaults() -> None:
    with (
        patch("sys.argv", ["tests", "-q"]),
        patch("pytest.main", return_value=0) as mock_pytest_main,
    ):
        result = main()

    assert result == 0
    mock_pytest_main.assert_called_once_with(
        [
            "--cov",
            "--cov-report=term-missing",
            "--cov-fail-under=100",
            "-q",
        ]
    )
