# ruff: noqa: E402
from pathlib import Path

from test_support import add_build_default_to_syspath

build_default = add_build_default_to_syspath()

from option_chain import option_chain


def test_option_chain_uses_none_short_circuit() -> None:
    assert option_chain(None) is None
    assert option_chain(4) == 5

    source = (Path(build_default) / "option_chain.py").read_text()
    assert "None if __option_value is None else" in source
