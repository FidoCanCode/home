"""Tests for fido.main — top-level server entry point."""

from fido.main import main


class TestMain:
    def test_no_args_dispatches_to_server(self) -> None:
        calls: list[None] = []
        main([], _run=lambda: calls.append(None))
        assert calls == [None]

    def test_server_args_dispatches_to_server(self) -> None:
        calls: list[None] = []
        main(["--port", "9000"], _run=lambda: calls.append(None))
        assert calls == [None]

    def test_argv_none_uses_server_path(self) -> None:
        calls: list[None] = []
        main(_run=lambda: calls.append(None))
        assert calls == [None]
