(** Recovery-loop FSM: cancel intent survives a session respawn.

    Closes the gap exposed by #1792.  Models the coordination contract
    inside ``session_agent._prompt_with_recovery`` for the cross-boundary
    case where a peer thread fires a cancel between the worker's
    ``session.prompt(...)`` call and its return, *and* the subprocess
    exits before producing a ``TurnReturn`` event so the recovery loop
    respawns and retries.

    The bug ``_prompt_with_recovery`` had at the time #1792 was filed:
    on subprocess crash, the recovery path silently created a fresh
    session in [NoCancel] state and re-issued the original prompt.
    The retried turn ran to completion normally, the caller observed
    [Returned RetNormal], and the cancel signal was silently dropped —
    so ``execute_task`` never saw ``last_turn_cancelled == True`` and
    the worker never yielded to ``handle_queued_comments``.  Queued PR
    comments sat in the durable queue indefinitely behind in-flight
    work.

    This FSM locks down two invariants:

    1. [recover_preserves_cancel_intent] — ``Recover`` is a transition
       on the cancel-intent bit's *value*, not on the session identity:
       respawn must carry the intent across the boundary.

    2. [return_reports_cancel_iff_intent] — the only legal ``Return``
       from [Pending Cancelled] is [Returned RetCancelled].
       Conversely the only legal ``Return`` from [Pending NoCancel] is
       [Returned RetNormal].  No path from [Pending Cancelled] reaches
       [Returned RetNormal] for any sequence of events.

    Events:

    [Prompt]          — caller enters [_prompt_with_recovery]; the FSM
                        leaves [Initial] and enters [Pending NoCancel].
    [CancelFire]      — a peer thread sets the cancel event during the
                        current cycle.  Sets the intent bit.  Idempotent
                        from [Pending Cancelled].
    [SubprocessExit]  — the underlying subprocess died before producing
                        a result.  The intent bit is preserved.
    [Recover]         — ``_prompt_with_recovery`` respawned the session
                        and re-issued the prompt.  The intent bit is
                        preserved.
    [Return]          — ``_prompt_with_recovery`` returned to the
                        caller.  ``Returned RetCancelled`` iff the
                        intent bit was set; ``Returned RetNormal``
                        otherwise.

    The Python runtime oracle that ships with this model wraps every
    ``_prompt_with_recovery`` call: each loop iteration fires an event
    on the FSM and the oracle crashes if Python diverges from the
    proof.  See ``src/fido/session_agent.py`` for the integration. *)

From FidoModels Require Import preamble.

(** * State *)

Inductive CancelIntent : Type :=
| NoCancel  : CancelIntent
| Cancelled : CancelIntent.

Inductive ReturnedKind : Type :=
| RetNormal    : ReturnedKind
| RetCancelled : ReturnedKind.

Inductive State : Type :=
| Initial  : State
| Pending  : CancelIntent -> State
| Returned : ReturnedKind -> State.

Definition initial_state : State := Initial.
Definition pending_fresh : State := Pending NoCancel.
Definition pending_cancelled : State := Pending Cancelled.
Definition returned_normal : State := Returned RetNormal.
Definition returned_cancelled : State := Returned RetCancelled.

(** * Event *)

Inductive Event : Type :=
| Prompt         : Event
| CancelFire     : Event
| SubprocessExit : Event
| Recover        : Event
| Return         : Event.

(** * Transition function *)

Definition transition (current : State) (event : Event) : option State :=
  match current, event with
  (* Entry: Prompt from Initial enters Pending in the [NoCancel] phase. *)
  | Initial, Prompt =>
      Some (Pending NoCancel)

  (* CancelFire sets the intent bit.  Idempotent from [Cancelled]. *)
  | Pending NoCancel, CancelFire =>
      Some (Pending Cancelled)
  | Pending Cancelled, CancelFire =>
      Some (Pending Cancelled)

  (* SubprocessExit is observation, not state change — the recovery
     loop will respond with [Recover] next.  Intent preserved. *)
  | Pending ci, SubprocessExit =>
      Some (Pending ci)

  (* Recover: respawn happened; intent MUST persist across the boundary.
     This is the load-bearing transition for the #1792 invariant. *)
  | Pending ci, Recover =>
      Some (Pending ci)

  (* Return: report kind matching the current intent. *)
  | Pending NoCancel, Return =>
      Some (Returned RetNormal)
  | Pending Cancelled, Return =>
      Some (Returned RetCancelled)

  (* Every other (state, event) pair is illegal: Prompt outside Initial,
     events on a Returned state, etc.  Python oracle crashes if the
     runtime drives one of these. *)
  | _, _ => None
  end.

Python File Extraction cancel_survives_respawn
  "initial_state pending_fresh pending_cancelled returned_normal returned_cancelled transition".

(** * Proved invariants *)

(** Recover from a [Cancelled] intent state cannot reset the intent.
    This is the precise statement of the bug #1792 captures: the
    pre-fix Python re-spawned the session and effectively transitioned
    [Pending Cancelled] → [Pending NoCancel], which this transition
    function refuses. *)
Lemma recover_preserves_cancel_intent :
  transition pending_cancelled Recover = Some pending_cancelled /\
  transition pending_fresh Recover = Some pending_fresh.
Proof.
  split; reflexivity.
Qed.

(** No event from [Pending Cancelled] yields [Pending NoCancel].
    This is the structural reason the bug cannot recur: the FSM has
    no transition that strips the cancel bit. *)
Lemma cancel_intent_never_resets :
  (forall e, transition pending_cancelled e <> Some pending_fresh) /\
  (forall e, transition pending_cancelled e <> Some returned_normal).
Proof.
  split; intros e; destruct e; simpl; discriminate.
Qed.

(** Return is intent-faithful: the kind of [Returned] state matches
    the cancel bit at return time. *)
Lemma return_reports_cancel_iff_intent :
  transition pending_cancelled Return = Some returned_cancelled /\
  transition pending_fresh Return = Some returned_normal.
Proof.
  split; reflexivity.
Qed.

(** SubprocessExit is observation-only.  It cannot change the intent
    bit; only [CancelFire] sets it and only [Return] consumes it. *)
Lemma subprocess_exit_preserves_state :
  transition pending_fresh SubprocessExit = Some pending_fresh /\
  transition pending_cancelled SubprocessExit = Some pending_cancelled.
Proof.
  split; reflexivity.
Qed.

(** CancelFire is idempotent — duplicate cancel events from racing
    peer threads do not destabilise the FSM. *)
Lemma cancel_fire_idempotent :
  transition pending_cancelled CancelFire = Some pending_cancelled.
Proof.
  reflexivity.
Qed.

(** [Prompt] is only valid from [Initial] — re-entering the recovery
    loop on an existing [Pending] or [Returned] state is illegal.
    This blocks the "nested recovery loop" failure mode. *)
Lemma prompt_only_from_initial :
  transition pending_fresh Prompt = None /\
  transition pending_cancelled Prompt = None /\
  transition returned_normal Prompt = None /\
  transition returned_cancelled Prompt = None.
Proof.
  repeat split; reflexivity.
Qed.

(** Returned states are terminal — no further events are accepted.
    The caller of [_prompt_with_recovery] must construct a fresh FSM
    instance for the next prompt. *)
Lemma returned_is_terminal :
  (forall e, transition returned_normal e = None) /\
  (forall e, transition returned_cancelled e = None).
Proof.
  split; intros e; destruct e; reflexivity.
Qed.

(** Full happy-path trace: prompt → cancel → subprocess crashes →
    respawn → return, with the cancel intent surviving end-to-end and
    the return observable matching. *)
Lemma cancel_survives_respawn_path :
  transition initial_state Prompt = Some pending_fresh /\
  transition pending_fresh CancelFire = Some pending_cancelled /\
  transition pending_cancelled SubprocessExit = Some pending_cancelled /\
  transition pending_cancelled Recover = Some pending_cancelled /\
  transition pending_cancelled Return = Some returned_cancelled.
Proof.
  repeat split; reflexivity.
Qed.

(** No-cancel happy path: prompt → subprocess crashes → recover →
    return normal.  Recovery is allowed when nothing was cancelled —
    transient subprocess failures must still be recoverable. *)
Lemma no_cancel_recover_path :
  transition initial_state Prompt = Some pending_fresh /\
  transition pending_fresh SubprocessExit = Some pending_fresh /\
  transition pending_fresh Recover = Some pending_fresh /\
  transition pending_fresh Return = Some returned_normal.
Proof.
  repeat split; reflexivity.
Qed.

(** Cancel-after-recover: even when the cancel fires *between* the
    respawn and the next return, the intent is still observed. *)
Lemma cancel_after_recover_path :
  transition initial_state Prompt = Some pending_fresh /\
  transition pending_fresh SubprocessExit = Some pending_fresh /\
  transition pending_fresh Recover = Some pending_fresh /\
  transition pending_fresh CancelFire = Some pending_cancelled /\
  transition pending_cancelled Return = Some returned_cancelled.
Proof.
  repeat split; reflexivity.
Qed.
