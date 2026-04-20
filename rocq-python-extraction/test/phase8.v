(** Phase 8 acceptance tests: rank-1 polymorphism with typed Python output.

    Acceptance: a polymorphic list-map function extracts with real TypeVars
    that pyright verifies. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

Open Scope list_scope.

Extract Inductive list => "list"
  [ "[]" "(lambda h, t: [h] + t)" ]
  "(lambda fnil, fcons, xs: fnil() if xs == [] else fcons(xs[0], xs[1:]))".

Fixpoint list_map {A B : Set} (f : A -> B) (xs : list A) : list B :=
  match xs with
  | nil => nil
  | x :: tl => f x :: list_map f tl
  end.

Python Extraction list_map.
