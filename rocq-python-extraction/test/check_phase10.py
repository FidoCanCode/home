# ruff: noqa: E402
import os
import sys

_d = os.path.dirname(os.path.abspath(__file__))
while not (
    os.path.basename(_d) == "default"
    and os.path.basename(os.path.dirname(_d)) == "_build"
):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)

import Phase10Mod

phase10 = Phase10Mod.Phase10Mod

assert phase10.NatLookup.run == 0
assert phase10.SuccLookup.run == 2
assert phase10.NatLookup is phase10.NatLookupAgain
assert phase10.FreshLookupA.run == 0
assert phase10.FreshLookupB.run == 0
assert phase10.FreshLookupA is not phase10.FreshLookupB

print("Phase 10 module/functor round-trip: OK")
