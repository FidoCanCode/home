from pathlib import Path

from fido.provider import ProviderID
from fido.status import _repos_from_pid


class TestReposFromPid:
    def test_parses_repo_provider_from_cmdline(self, tmp_path: Path) -> None:
        pid = 1234
        cmdline = f"uv\x00run\x00fido\x00owner/repo:{tmp_path}:copilot-cli\x00".encode()
        repos = _repos_from_pid(
            pid,
            _read_bytes=lambda p: cmdline if p == Path(f"/proc/{pid}/cmdline") else b"",
        )
        assert len(repos) == 1
        assert repos[0].name == "owner/repo"
        assert repos[0].provider == ProviderID.COPILOT_CLI

    def test_skips_invalid_provider_in_cmdline(self, tmp_path: Path) -> None:
        pid = 1234
        cmdline = (
            f"uv\x00run\x00fido\x00owner/repo:{tmp_path}:bad-provider\x00".encode()
        )
        assert (
            _repos_from_pid(
                pid,
                _read_bytes=lambda p: (
                    cmdline if p == Path(f"/proc/{pid}/cmdline") else b""
                ),
            )
            == []
        )

    def test_skips_repo_without_provider_in_cmdline(self, tmp_path: Path) -> None:
        pid = 1234
        cmdline = f"uv\x00run\x00fido\x00owner/repo:{tmp_path}\x00".encode()
        assert (
            _repos_from_pid(
                pid,
                _read_bytes=lambda p: (
                    cmdline if p == Path(f"/proc/{pid}/cmdline") else b""
                ),
            )
            == []
        )
