#!/usr/bin/env python3
"""Guard: reject new monkeypatch.setattr usage outside exemptions.

monkeypatch.setattr is patching under another name — it reaches into module or
object internals from outside, bypassing constructor-DI the same way
unittest.mock.patch does.

New sites must use constructor-DI instead: accept collaborators via __init__
and pass fakes at construction time.  See CLAUDE.md "OO + constructor-DI
architecture" and #1773 for the migration epic.
"""

import re
import sys
from pathlib import Path

# Temporary exemptions — existing monkeypatch.setattr sites pending
# constructor-DI migration under #1773.  Do not add new files here; fix the
# root cause instead.
_EXEMPTIONS: frozenset[str] = frozenset(
    {
        "tests/test_cli.py",
        "tests/test_claude_hold_for_handler.py",
        "tests/test_copilot_hold_for_handler.py",
        "tests/test_rocq_generated_pyright.py",
        "tests/test_rocq_lsp.py",
        "tests/test_rocq_pymap.py",
        "tests/test_rocq_repl.py",
        "tests/test_rocq_traceback.py",
        "tests/test_server.py",
        "tests/test_status_provider.py",
        "tests/test_worker.py",
        "tests/test_worker_persist_session_id.py",
    }
)

_PATTERN: re.Pattern[str] = re.compile(r"\.setattr\s*\(")

_GUIDANCE = (
    "\nmonkeypatch.setattr bypasses constructor-DI the same way\n"
    "unittest.mock.patch does — it reaches into module or object internals\n"
    "from outside, creating invisible coupling.\n"
    "\n"
    "Fix: accept collaborators via __init__ and pass fakes at construction\n"
    "time.  See CLAUDE.md 'OO + constructor-DI architecture' and #1773.\n"
)


def check(
    root: Path,
    exemptions: frozenset[str] | None = None,
) -> int:
    """Scan ``root/tests`` for monkeypatch.setattr usage outside exemptions.

    Returns 0 if clean, 1 if violations found.  Violations are written to
    stderr along with a pointer to constructor-DI.
    """
    if exemptions is None:
        exemptions = _EXEMPTIONS
    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return 0
    violations: list[str] = []
    for py_file in sorted(tests_dir.rglob("*.py")):
        rel = py_file.relative_to(root).as_posix()
        if rel in exemptions:
            continue
        for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
            if _PATTERN.search(line):
                violations.append(
                    f"{rel}:{lineno}: monkeypatch.setattr reach-through\n"
                )
    if violations:
        sys.stderr.write("".join(violations))
        sys.stderr.write(_GUIDANCE)
        sys.stderr.write(f"\n{len(violations)} violation(s) found.\n")
        return 1
    return 0


def main() -> int:
    """Entry point: scan from the repo root and return the check result."""
    root = Path(__file__).resolve().parents[1]
    return check(root)


if __name__ == "__main__":
    raise SystemExit(main())
