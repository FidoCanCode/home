# ruff: noqa: E402
from pathlib import Path

import pytest
from test_support import add_build_default_to_syspath, run_as_script

build_default = add_build_default_to_syspath()

from proof_pair_zero import _Impossible, proof_pair_zero


def test_proof_pair_zero_uses_impossible_witness() -> None:
    with pytest.raises(_Impossible):
        proof_pair_zero(0)

    source = (Path(build_default) / "proof_pair_zero.py").read_text()
    assert "_impossible()" in source


if __name__ == "__main__":
    run_as_script(
        test_proof_pair_zero_uses_impossible_witness,
        "prop/set impossible witness: OK",
    )
