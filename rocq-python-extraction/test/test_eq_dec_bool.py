# ruff: noqa: E402
from pathlib import Path

from test_support import add_build_default_to_syspath, run_as_script

build_default = add_build_default_to_syspath()

from eq_dec_bool import Left, Right, eq_dec_bool


def test_eq_dec_bool_round_trip() -> None:
    assert eq_dec_bool(Left()) is True
    assert eq_dec_bool(Right()) is False

    source = (Path(build_default) / "eq_dec_bool.py").read_text()
    assert "return _impossible()" not in source
    assert "return __" not in source


if __name__ == "__main__":
    run_as_script(test_eq_dec_bool_round_trip, "prop/set decidable match: OK")
