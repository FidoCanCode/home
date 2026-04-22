from itertools import islice
from typing import cast

from repeat_tree import CNode, Cotree, coforce, repeat_tree
from tree_root_of_repeat import tree_root_of_repeat
from zeros import Stream, zeros
from zeros_pair import coprefix_eq, coprefix_hash, zeros_pair

zero_prefix = list(islice(zeros, 4))
assert all(x == 0 for x in zero_prefix)

pair = cast(tuple[Stream[int], Stream[int]], zeros_pair)
assert coprefix_eq(4, pair[0], pair[1])
assert isinstance(coprefix_hash(4, pair[0]), int)
assert isinstance(coforce(repeat_tree), CNode)
assert isinstance(repeat_tree, Cotree)
assert isinstance(tree_root_of_repeat, int)
