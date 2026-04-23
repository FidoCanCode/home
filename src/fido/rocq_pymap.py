"""Reader for Rocq extraction source maps."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

_FIELDNAMES = (
    "version",
    "python_file",
    "python_start_line",
    "python_start_col",
    "python_end_line",
    "python_end_col",
    "source_file",
    "source_start_line",
    "source_start_col",
    "source_end_line",
    "source_end_col",
    "kind",
    "symbol",
)


class PyMapError(ValueError):
    pass


class UnsupportedPyMapVersion(PyMapError):
    pass


@dataclass(frozen=True)
class PyMapEntry:
    python_file: str
    python_start_line: int
    python_start_col: int
    python_end_line: int
    python_end_col: int
    source_file: str
    source_start_line: int
    source_start_col: int
    source_end_line: int
    source_end_col: int
    kind: str
    symbol: str

    @classmethod
    def from_row(cls, row: dict[str, str | None], path: Path) -> Self:
        version = row.get("version")
        if version != "1":
            raise UnsupportedPyMapVersion(f"unsupported source map version in {path}")
        try:
            return cls(
                python_file=_field(row, "python_file"),
                python_start_line=int(_field(row, "python_start_line")),
                python_start_col=int(_field(row, "python_start_col")),
                python_end_line=int(_field(row, "python_end_line")),
                python_end_col=int(_field(row, "python_end_col")),
                source_file=_field(row, "source_file"),
                source_start_line=int(_field(row, "source_start_line")),
                source_start_col=int(_field(row, "source_start_col")),
                source_end_line=int(_field(row, "source_end_line")),
                source_end_col=int(_field(row, "source_end_col")),
                kind=_field(row, "kind"),
                symbol=_field(row, "symbol"),
            )
        except ValueError as exc:
            raise PyMapError(f"bad source map row in {path}: {exc}") from exc


@dataclass(frozen=True)
class PyMap:
    entries: tuple[PyMapEntry, ...]

    @classmethod
    def load(cls, path: Path) -> Self:
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != _FIELDNAMES:
                    raise PyMapError(f"bad source map header in {path}")
                return cls(tuple(PyMapEntry.from_row(row, path) for row in reader))
        except csv.Error as exc:
            raise PyMapError(f"bad source map CSV in {path}: {exc}") from exc


def _field(row: dict[str, str | None], name: str) -> str:
    value = row.get(name)
    if value is None:
        raise PyMapError(f"missing {name}")
    return value
