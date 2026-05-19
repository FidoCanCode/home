"""Top-level fido server entry point."""

from collections.abc import Callable


def main(
    argv: list[str] | None = None,
    *,
    _run: Callable[[], None] | None = None,
) -> None:
    del argv
    if _run is None:
        from fido.server import run as _run  # pragma: no cover
    _run()


if __name__ == "__main__":  # pragma: no cover
    main()
