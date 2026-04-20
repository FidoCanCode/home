Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Extract Inductive nat => "int"
  [ "0" "(lambda x: x + 1)" ]
  "(lambda fO, fS, n: fO() if n == 0 else fS(n - 1))".

Module Phase10Mod.
  Module Type MapSig.
    Parameter missing : nat.
    Parameter lookup : nat -> nat.
  End MapSig.

  Module NatMap <: MapSig.
    Definition missing : nat := 0.
    Definition lookup (n : nat) : nat := n.
  End NatMap.

  Module SuccMap <: MapSig.
    Definition missing : nat := 1.
    Definition lookup (n : nat) : nat := S n.
  End SuccMap.

  Module MakeLookup (X : MapSig).
    Definition run : nat := X.lookup X.missing.
  End MakeLookup.

  Module NatLookup := MakeLookup NatMap.
  Module NatLookupAgain := MakeLookup NatMap.
  Module SuccLookup := MakeLookup SuccMap.

  Module FreshLookupAFunctor (X : MapSig).
    Definition run : nat := X.lookup X.missing.
  End FreshLookupAFunctor.

  Module FreshLookupBFunctor (X : MapSig).
    Definition run : nat := X.lookup X.missing.
  End FreshLookupBFunctor.

  Module FreshLookupA := FreshLookupAFunctor NatMap.
  Module FreshLookupB := FreshLookupBFunctor NatMap.
End Phase10Mod.

Python Module Extraction Phase10Mod.
