from itertools import islice
from typing import cast

from coinductives import (
    CNode,
    Cotree,
    Stream,
    coforce,
    coprefix_eq,
    coprefix_hash,
    repeat_tree,
    tree_root_of_repeat,
    zeros,
    zeros_pair,
)


def test_coinductive_round_trip() -> None:
    assert list(islice(zeros, 6)) == [0, 0, 0, 0, 0, 0]

    pair = cast(tuple[Stream[int], Stream[int]], zeros_pair)
    left = pair[0]
    right = pair[1]
    assert coprefix_eq(8, left, right)
    assert coprefix_hash(8, left) == coprefix_hash(8, right)

    step = coforce(repeat_tree)
    assert isinstance(step, CNode)
    assert isinstance(repeat_tree, Cotree)
    assert step.arg0 == 0

    assert tree_root_of_repeat == 0
