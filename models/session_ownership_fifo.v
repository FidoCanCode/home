(** Session-ownership FIFO: full queue-based ownership with worker deferral.

    Extends the binary session lock of [session_lock] into a per-session
    FIFO queue.  Non-worker contenders ([Handler], [CronSweep]) queue and
    drain in arrival order.  The background worker uses a separate
    [Deferred] slot and runs only when the queue is empty and the slot
    is set.

    Worker resumption is itself preemptible: any [Enqueue] while the worker
    is [WorkerActive] implicitly re-defers the worker and appends the new
    contender to the queue tail.  There is no explicit [Preempt] event —
    preemption is structural, not a named transition.

    This is the first cross-[.v] import in the repo: [Release] carries a
    [ReviewReplyOutcome] from [replied_comment_claims], letting the FIFO
    state machine observe the outcome without owning reply logic.

    D4 lands as specification ahead of implementation.  Today's webhook
    path blocks the HTTP thread for the full turn duration
    ([server.py:776-782]); the instant-exit [Enqueue]/[Dequeue] split is a
    future alignment tracked in D5 #743.  Six theorems over [transition]
    are proved in a later commit in this PR. *)

Declare ML Module "rocq-python-extraction".
Declare ML Module "rocq-runtime.plugins.extraction".

From FidoModels Require Import replied_comment_claims.

From Stdlib Require Import
  Lists.List.

Import ListNotations.

(* Prevent sort-polymorphism so nullary-constructor extraction is clean.
   See the note in [rocq-python-extraction/test/datatypes.v] for context. *)
Unset Universe Polymorphism.

(* Remap [option] so [Some x] erases to [x] and [None] stays [None].
   This makes [transition] return the new [FifoState] directly on success
   and [None] on rejection — a natural Python return type. *)
Extract Inductive option => ""
  [ "" "None" ]
  "(lambda fSome, fNone, x: fNone() if x is None else fSome(x))".

(** * Contender

    Non-worker session-holder kinds.  [Handler] covers webhook-handler and
    CI-fix turns (CI is folded into Handler; no rank ordering inside the
    queue).  [CronSweep] covers the periodic idle-drain sweep.  The
    background worker never enters the FIFO queue — it is managed through
    the [fifo_worker_deferred] slot. *)
Inductive Contender : Type :=
| Handler   : Contender
| CronSweep : Contender.

(** * ActiveSlot

    Who currently holds the session.
    [Idle]             — nobody holds; the queue may be non-empty (pending
                         a [Dequeue]) or the worker may be deferred.
    [HolderActive c]   — the FIFO contender [c] holds the session.
    [WorkerActive]     — the background worker holds the session. *)
Inductive ActiveSlot : Type :=
| Idle                         : ActiveSlot
| HolderActive (c : Contender) : ActiveSlot
| WorkerActive                 : ActiveSlot.

(** * FifoState

    Full per-session FIFO ownership state.

    [fifo_queue]           — ordered list of pending contenders; head is
                             next to be activated by [Dequeue].
    [fifo_active_slot]     — who currently holds the session.
    [fifo_worker_deferred] — [true] when the worker is parked and waiting
                             to resume after the queue drains. *)
Record FifoState : Type := {
  fifo_queue           : list Contender;
  fifo_active_slot     : ActiveSlot;
  fifo_worker_deferred : bool
}.

(** * Event

    Five events cover all ownership transitions.  No explicit [Preempt]
    event exists — worker preemption is implicit in [Enqueue] when the
    worker holds the session.

    [Enqueue c]        — a contender arrives; total, always succeeds.  If
                         the worker is [WorkerActive], it is implicitly
                         re-deferred and [c] appends to the queue tail.
    [Dequeue]          — activate the queue head; only valid when [Idle]
                         and the queue is non-empty.
    [WorkerDefer]      — the worker explicitly parks into the deferred
                         slot; only valid when [WorkerActive].
    [WorkerResume]     — the worker re-activates; only valid when [Idle],
                         the queue is empty, and the deferred slot is set.
    [Release outcome]  — the current holder relinquishes; valid from
                         [HolderActive] or [WorkerActive]; rejected from
                         [Idle]. *)
Inductive Event : Type :=
| Enqueue     (c : Contender)                : Event
| Dequeue                                    : Event
| WorkerDefer                                : Event
| WorkerResume                               : Event
| Release     (outcome : ReviewReplyOutcome) : Event.

(** * Transition function

    [transition s event] returns [Some s'] when [event] is valid in [s],
    or [None] when it is rejected.

    [Enqueue] is the only total event — it always returns [Some _] — and
    is the formal statement that webhooks exit instantly: [Enqueue] never
    blocks on the current holder state. *)
Definition transition (s : FifoState) (event : Event) : option FifoState :=
  match event with

  | Enqueue c =>
      (* Total.  Worker preemption is implicit: if the worker is active,
         re-defer it and append the new contender. *)
      match fifo_active_slot s with
      | WorkerActive =>
          Some {| fifo_queue           := fifo_queue s ++ [c];
                  fifo_active_slot     := Idle;
                  fifo_worker_deferred := true |}
      | _ =>
          Some {| fifo_queue           := fifo_queue s ++ [c];
                  fifo_active_slot     := fifo_active_slot s;
                  fifo_worker_deferred := fifo_worker_deferred s |}
      end

  | Dequeue =>
      (* Activate head.  Rejected if the slot is occupied or queue empty. *)
      match fifo_active_slot s, fifo_queue s with
      | Idle, c :: rest =>
          Some {| fifo_queue           := rest;
                  fifo_active_slot     := HolderActive c;
                  fifo_worker_deferred := fifo_worker_deferred s |}
      | _, _ => None
      end

  | WorkerDefer =>
      (* Explicit park.  Only valid when the worker is active. *)
      match fifo_active_slot s with
      | WorkerActive =>
          Some {| fifo_queue           := fifo_queue s;
                  fifo_active_slot     := Idle;
                  fifo_worker_deferred := true |}
      | _ => None
      end

  | WorkerResume =>
      (* Re-activate the deferred worker.  Only valid when queue is empty
         and the deferred slot is set. *)
      match fifo_active_slot s, fifo_queue s, fifo_worker_deferred s with
      | Idle, [], true =>
          Some {| fifo_queue           := [];
                  fifo_active_slot     := WorkerActive;
                  fifo_worker_deferred := false |}
      | _, _, _ => None
      end

  | Release _ =>
      (* Holder relinquishes.  Valid from [HolderActive] or [WorkerActive];
         rejected from [Idle]. *)
      match fifo_active_slot s with
      | Idle => None
      | _    =>
          Some {| fifo_queue           := fifo_queue s;
                  fifo_active_slot     := Idle;
                  fifo_worker_deferred := fifo_worker_deferred s |}
      end

  end.

Python File Extraction session_ownership_fifo "transition".
