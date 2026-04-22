(** Keyed coordination-state vocabulary.

    This model exercises backend-owned finite-map and finite-set extraction for
    the identifier shapes Fido uses most often: numeric GitHub IDs and string
    repo/provider/task IDs. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From Stdlib Require Import
  FSets.FMapPositive
  MSets.MSetPositive
  Lists.List
  Numbers.BinNums
  Strings.String.

Open Scope positive_scope.
Open Scope string_scope.
Import ListNotations.

(** [add_claim] records that a numeric thread/claim id is currently owned.
    The caller supplies the current set; no claim ids are known at compile
    time. *)
Definition add_claim (thread : positive) (claims : PositiveSet.t) : PositiveSet.t :=
  PositiveSet.add thread claims.

(** [remove_claim] clears a numeric thread/claim id from a caller-provided
    claim set.  This covers the finite-set remove operation in the model
    surface Fido will use for coordination state. *)
Definition remove_claim (thread : positive) (claims : PositiveSet.t) : PositiveSet.t :=
  PositiveSet.remove thread claims.

(** [has_claim] tests membership in the runtime claim set for one positive id. *)
Definition has_claim (claims : PositiveSet.t) (thread : positive) : bool :=
  PositiveSet.mem thread claims.

(** [assign_issue] associates a runtime GitHub issue id with a runtime owner
    string in the caller-provided issue-owner map.  PR task/checklist state is
    deliberately separate from this GitHub issue coordination index. *)
Definition assign_issue (issue : positive) (owner : String.string)
    (owners : PositiveMap.t String.string) : PositiveMap.t String.string :=
  PositiveMap.add issue owner owners.

(** [unassign_issue] removes a runtime GitHub issue id from the caller-provided
    issue-owner map. *)
Definition unassign_issue (issue : positive)
    (owners : PositiveMap.t String.string) : PositiveMap.t String.string :=
  PositiveMap.remove issue owners.

(** [issue_owner] looks up the owner string for a runtime GitHub issue id. *)
Definition issue_owner (owners : PositiveMap.t String.string)
    (issue : positive) : option String.string :=
  PositiveMap.find issue owners.

(** [repo_entry] is one CLI-provided repo tuple: owner/repo, path on disk, and
    provider.  The repo collection is supplied at runtime, so this type is
    only the shape of one entry, not a compile-time repo list. *)
Definition repo_entry : Type :=
  (String.string * (String.string * String.string))%type.

(** [repo_provider] projects the provider field from one runtime repo entry. *)
Definition repo_provider (repo : repo_entry) : String.string :=
  match repo with
  | (_, (_, provider)) => provider
  end.

(** [repo_providers] projects providers from the runtime repo entries without
    baking any repo or provider names into generated code.  It traverses the
    input list as given; callers should not treat that as a semantic order. *)
Fixpoint repo_providers (repos : list repo_entry) : list String.string :=
  match repos with
  | [] => []
  | repo :: rest => repo_provider repo :: repo_providers rest
  end.

(** [repo_count] counts the runtime repo list provided by the CLI. *)
Definition repo_count (repos : list repo_entry) : nat :=
  List.length repos.

Python File Extraction coord_index
  "add_claim remove_claim has_claim assign_issue unassign_issue issue_owner repo_providers repo_count".
