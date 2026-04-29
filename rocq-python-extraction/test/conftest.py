import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DEFAULT = REPO_ROOT / "_build" / "default"


class RenderedSourceAssert(Protocol):
    def __call__(
        self,
        source: str,
        expected: str,
        forbidden: tuple[str, ...] = (),
    ) -> None: ...


if not BUILD_DEFAULT.is_dir():
    raise RuntimeError(
        "Rocq pytest artifacts are missing; run tests through the Docker-backed "
        "./fido tests command"
    )

sys.path.insert(0, str(BUILD_DEFAULT))


@pytest.fixture
def build_default() -> Iterator[Path]:
    yield BUILD_DEFAULT


@pytest.fixture
def assert_rendered_source() -> RenderedSourceAssert:
    def assert_source(
        source: str,
        expected: str,
        forbidden: tuple[str, ...] = (),
    ) -> None:
        assert expected in source
        for snippet in forbidden:
            assert snippet not in source

    return assert_source
