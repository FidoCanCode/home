"""Recursive deep-freeze for JSON-shaped Python objects (#1748).

Caches that hold raw GitHub objects need to guarantee callers
can't mutate the cached state under their feet — but projecting
into a narrow typed dataclass throws away half the surface area
GitHub gives us.  The honest middle ground is to keep the dict
GitHub returned and make it deeply immutable.

:func:`deep_freeze` walks a JSON-shaped value and rebuilds it as
:class:`frozendict.frozendict` for mappings and ``tuple`` for
sequences.  Strings, numbers, bools, and ``None`` are already
immutable so they pass through untouched.

``frozendict`` (already a project dependency) is a real
immutable mapping — mutation methods raise rather than write
back — and it works seamlessly with ``isinstance(..., Mapping)``
checks consumers already use.
"""

from collections.abc import Mapping
from typing import Any

from frozendict import frozendict

# JSON-shape, covering BOTH directions:
#
#  * incoming (what ``json.loads`` produces) — ``dict`` for objects,
#    ``list`` for arrays
#  * outgoing (what :func:`deep_freeze` returns) — ``frozendict``
#    for objects, ``tuple`` for arrays
#
# ``Mapping[str, …]`` accepts both ``dict`` and ``frozendict``
# uniformly; the ``list | tuple`` arms cover the array variants
# without pulling in ``Sequence`` (which would also match ``str`` /
# ``bytes`` and break the recursion).  PEP 695 ``type`` statement
# makes the recursion lazy so the self-reference just works.
type JsonLike = (
    Mapping[str, JsonLike]
    | list[JsonLike]
    | tuple[JsonLike, ...]
    | str
    | int
    | float
    | bool
    | None
)


def deep_freeze(obj: JsonLike) -> JsonLike:
    """Recursively rebuild ``obj`` as a deeply-immutable JSON shape.

    Mappings become ``frozendict`` over a fresh dict of frozen
    values; ``list`` / ``tuple`` arrays become tuples of frozen
    values; everything else (``str``, numbers, bools, ``None``) is
    already immutable and passes through.

    Idempotent — already-frozen inputs round-trip without copying
    further down.

    For top-level use against a known-object payload (a GitHub API
    response, say) prefer :func:`freeze_object` — its return type
    is narrowed to ``Mapping[str, Any]`` which is what callers
    actually want to consume.
    """
    if isinstance(obj, Mapping):
        return frozendict({k: deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        # Match list/tuple explicitly rather than ``Sequence`` — the
        # latter would also catch ``str`` and turn "hi" into
        # ('h', 'i').  GitHub JSON has no other sequence shapes.
        return tuple(deep_freeze(x) for x in obj)
    return obj


def freeze_object(obj: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deep-freeze a JSON *object* (top-level mapping).

    Narrower-typed wrapper around :func:`deep_freeze` for the
    common case where the caller knows the input is a JSON object.
    The return type is preserved as ``Mapping[str, Any]`` so callers
    don't need to narrow a ``JsonLike`` union before consuming it.
    """
    return frozendict({k: deep_freeze(v) for k, v in obj.items()})
