from __future__ import annotations

from unittest.mock import patch


def test_main_calls_run() -> None:
    with patch("kennel.server.run") as mock_run:
        import kennel.__main__  # noqa: F401 — importing executes run()
    mock_run.assert_called_once()
