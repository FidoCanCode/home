"""Project test runner entrypoint."""

from __future__ import annotations

import sys


def main() -> int:
    import pytest

    args = [
        "--cov",
        "--cov-report=term-missing",
        "--cov-fail-under=100",
        *sys.argv[1:],
    ]
    return pytest.main(args)
