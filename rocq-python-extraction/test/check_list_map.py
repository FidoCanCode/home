# ruff: noqa: E402
import os
import sys
from pathlib import Path

_d = os.path.dirname(os.path.abspath(__file__))
while not (
    os.path.basename(_d) == "default"
    and os.path.basename(os.path.dirname(_d)) == "_build"
):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)

from list_map import list_map

assert list_map(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert list_map(str, [1, 2, 3]) == ["1", "2", "3"]

module_path = Path(_d) / "list_map.py"
source = module_path.read_text()
assert "TypeVar" in source, "list_map.py must declare TypeVars"
assert "def list_map" in source, "list_map.py must define list_map"
assert "-> list[" in source, "list_map.py must annotate the return type"

print("Phase 8 list_map polymorphism round-trip: OK")
