from __future__ import annotations

from pathlib import Path

import pytest

from kennel.config import Config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "KENNEL_SECRET",
        "KENNEL_WORK_DIR",
        "KENNEL_PROJECT",
        "KENNEL_PORT",
        "KENNEL_WORK_SCRIPT",
        "KENNEL_ALLOWED_BOTS",
        "KENNEL_LOG_LEVEL",
    ):
        monkeypatch.delenv(k, raising=False)


def _set_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KENNEL_SECRET", "test-secret")
    monkeypatch.setenv("KENNEL_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("KENNEL_PROJECT", "test-project")


class TestFromEnv:
    def test_required_vars(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _set_required(monkeypatch, tmp_path)
        cfg = Config.from_env()
        assert cfg.secret == b"test-secret"
        assert cfg.work_dir == tmp_path
        assert cfg.project == "test-project"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_required(monkeypatch, tmp_path)
        cfg = Config.from_env()
        assert cfg.port == 9000
        assert cfg.log_level == "INFO"
        assert "copilot[bot]" in cfg.allowed_bots

    def test_custom_port(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_required(monkeypatch, tmp_path)
        monkeypatch.setenv("KENNEL_PORT", "8080")
        cfg = Config.from_env()
        assert cfg.port == 8080

    def test_custom_bots(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_required(monkeypatch, tmp_path)
        monkeypatch.setenv("KENNEL_ALLOWED_BOTS", "bot1[bot],bot2[bot]")
        cfg = Config.from_env()
        assert cfg.allowed_bots == frozenset({"bot1[bot]", "bot2[bot]"})

    def test_missing_secret(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("KENNEL_WORK_DIR", str(tmp_path))
        monkeypatch.setenv("KENNEL_PROJECT", "test")
        with pytest.raises(SystemExit):
            Config.from_env()

    def test_missing_work_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KENNEL_SECRET", "s")
        monkeypatch.setenv("KENNEL_PROJECT", "p")
        with pytest.raises(SystemExit):
            Config.from_env()

    def test_missing_project(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("KENNEL_SECRET", "s")
        monkeypatch.setenv("KENNEL_WORK_DIR", str(tmp_path))
        with pytest.raises(SystemExit):
            Config.from_env()
