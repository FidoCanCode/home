import json
from pathlib import Path

import pytest

from fido import rocq_generated_pyright


class ExecCalled(Exception):
    def __init__(self, file: str, args: list[str]) -> None:
        self.file = file
        self.argv = args


class _FakeOsProcess:
    """Minimal :class:`~fido.infra.OsProcess` fake for pyright-exec tests.

    Only :meth:`execvp` is exercised; the other methods are not implemented.
    Raises *ExecCalled* (a sentinel exception) so tests can assert on the
    arguments without the process actually being replaced.
    """

    def execvp(self, file: str, args: list[str]) -> None:
        raise ExecCalled(file, args)


class _NoOpOsProcess:
    """OsProcess fake whose execvp silently returns (simulates exec failure)."""

    def execvp(self, file: str, args: list[str]) -> None:
        return None


def test_writes_pyright_config_and_execs_pyright(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated_dir = tmp_path / "generated"
    checks_dir = tmp_path / "checks"
    generated_dir.mkdir()
    checks_dir.mkdir()
    (generated_dir / "generated_module.py").write_text("value: int = 1\n")
    (checks_dir / "pyright_generated_check.py").write_text(
        "from generated_module import value\ncheck: int = value\n"
    )

    monkeypatch.chdir(tmp_path)

    with pytest.raises(ExecCalled) as exc_info:
        rocq_generated_pyright.main(
            [str(generated_dir), "--checks-dir", str(checks_dir)],
            os_proc=_FakeOsProcess(),
        )

    copied_check = generated_dir / "pyright_generated_check.py"
    config_path = generated_dir / "pyrightconfig.json"
    assert copied_check.read_text() == (
        "from generated_module import value\ncheck: int = value\n"
    )
    assert exc_info.value.file == "pyright"
    assert exc_info.value.argv == ["pyright", "-p", str(config_path)]
    assert json.loads(config_path.read_text()) == {
        "include": ["pyright_generated_check.py"],
        "executionEnvironments": [{"root": ".", "extraPaths": ["."]}],
        "reportUnusedImport": False,
        "reportUnusedVariable": False,
        "reportUnknownLambdaType": False,
        "reportRedeclaration": False,
    }


def test_raises_if_exec_returns(tmp_path: Path) -> None:
    generated_dir = tmp_path / "generated"
    checks_dir = tmp_path / "checks"
    generated_dir.mkdir()
    checks_dir.mkdir()

    with pytest.raises(RuntimeError, match="pyright exec failed"):
        rocq_generated_pyright.main(
            [str(generated_dir), "--checks-dir", str(checks_dir)],
            os_proc=_NoOpOsProcess(),
        )
