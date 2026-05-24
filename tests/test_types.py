"""Tests for shared type definitions in :mod:`fido.types`.

Currently focused on :class:`IntentVerdict` (#1798 / INV-D) — the
per-intent verdict shape Opus emits as part of a rescope batch.  The
verdict drives the INV-E (#1803) reply-back classifier downstream.
"""

import dataclasses
from collections.abc import Iterator
from typing import Any

import pytest

from fido.types import IntentVerdict

_ = (Iterator, Any)  # keep imports for type hints inside test bodies


class TestIntentVerdictShape:
    def test_minimal_construction_with_outcome_honored(self) -> None:
        """A fully-honored intent: id + outcome + narrative.

        HOL-3 (#1897): narrative is required on every outcome — without
        it the per-task narrative chain (#1894 HOL-8) loses history.
        """
        v = IntentVerdict(
            intent_comment_id=42, outcome="honored", narrative="As requested."
        )
        assert v.intent_comment_id == 42
        assert v.outcome == "honored"
        assert v.ops == ()
        assert v.affected_task_ids == ()
        assert v.by_intent_comment_id is None
        assert v.narrative == "As requested."

    def test_construction_with_all_fields(self) -> None:
        v = IntentVerdict(
            intent_comment_id=100,
            outcome="reshaped",
            ops=({"op": "rewrite", "id": "T1", "title": "new"},),
            affected_task_ids=("T1",),
            by_intent_comment_id=None,
            narrative="Folded the ask into the existing parser task.",
        )
        assert v.outcome == "reshaped"
        assert v.ops == ({"op": "rewrite", "id": "T1", "title": "new"},)
        assert v.affected_task_ids == ("T1",)
        assert v.narrative is not None

    def test_supersedence_pointer_within_batch(self) -> None:
        # "red" → "no, green" — red's verdict points at green's intent
        # comment id.  by_intent_comment_id is set; ops may be empty
        # (full supersedence) or partial (component supersedence).
        v = IntentVerdict(
            intent_comment_id=10,
            outcome="superseded",
            by_intent_comment_id=11,
            narrative="Color request was overridden by a later comment.",
        )
        assert v.by_intent_comment_id == 11
        assert v.outcome == "superseded"

    def test_partial_supersedence_keeps_ops(self) -> None:
        # "paint and make it red" → "no, green": the paint op survives
        # (still honored), the color component is superseded.  Ops AND
        # by_intent_comment_id coexist.
        v = IntentVerdict(
            intent_comment_id=10,
            outcome="superseded",
            ops=({"op": "new", "title": "Paint the surface", "type": "spec"},),
            affected_task_ids=("T-paint",),
            by_intent_comment_id=11,
            narrative="Paint kept; color component superseded by later comment.",
        )
        assert v.ops != ()
        assert v.by_intent_comment_id == 11

    def test_jointly_honored_intents_share_task_id(self) -> None:
        # Canonical 3+1 reviewer pattern: three comments asking the
        # same fix + a fourth "just fix all of these" all attribute
        # to ONE consolidated task.  Each verdict can list the same
        # task id in affected_task_ids.
        verdicts = tuple(
            IntentVerdict(
                intent_comment_id=cid,
                outcome="honored",
                ops=(),
                affected_task_ids=("T-fix-all",),
                narrative="Jointly honored by the consolidated task.",
            )
            for cid in (1, 2, 3, 4)
        )
        for v in verdicts:
            assert "T-fix-all" in v.affected_task_ids

    def test_no_op_outcome(self) -> None:
        # Acknowledgement / "request already covered" verdict.  No ops,
        # no affected tasks — but narrative IS required (HOL-3 / #1897)
        # because no_op is the requestor's "your ask was dropped"
        # signal; without a reason the user has no idea what happened.
        v = IntentVerdict(
            intent_comment_id=999,
            outcome="no_op",
            narrative="Already covered by an in-flight task.",
        )
        assert v.outcome == "no_op"
        assert v.ops == ()
        assert v.affected_task_ids == ()
        assert v.narrative == "Already covered by an in-flight task."

    def test_frozen(self) -> None:
        # Verdict shape is frozen — once Opus emits it, the runtime
        # must not mutate.  Catches accidental in-place edits.
        v = IntentVerdict(intent_comment_id=1, outcome="honored", narrative="x")
        try:
            v.outcome = "reshaped"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("IntentVerdict must be frozen")

    def test_ops_is_tuple_not_list(self) -> None:
        # codex P1 on #1802: deep immutability — ops as a list lets
        # callers ``verdict.ops.append(...)`` past the frozen guard.
        # Tuple makes the in-place mutation a TypeError at boundary.
        v = IntentVerdict(intent_comment_id=1, outcome="honored", narrative="x")
        assert isinstance(v.ops, tuple)
        assert isinstance(v.affected_task_ids, tuple)

    def test_ops_list_input_coerced_to_tuple(self) -> None:
        # codex P1 on #1802 (round 2): the tuple annotation isn't
        # enforced at runtime.  Callers can pass a plain list; the
        # post-init coerces.  Mutating the original list after
        # construction MUST NOT bleed into the verdict.
        original_ops = [{"op": "rewrite", "id": "T1"}]
        v = IntentVerdict(
            intent_comment_id=1,
            outcome="reshaped",
            ops=original_ops,  # type: ignore[arg-type]
            affected_task_ids=["T1"],  # type: ignore[arg-type]
            narrative="x",
        )
        original_ops.append({"op": "remove", "id": "T2"})
        assert isinstance(v.ops, tuple)
        assert len(v.ops) == 1
        assert isinstance(v.affected_task_ids, tuple)

    def test_op_payload_dicts_are_frozen(self) -> None:
        # codex P2 on #1802: even with tuple ops, each ``dict`` was
        # still mutable.  Deep-freeze coerces to frozendict so
        # ``v.ops[0]["op"] = "x"`` raises at the boundary.
        v = IntentVerdict(
            intent_comment_id=1,
            outcome="reshaped",
            ops=({"op": "rewrite", "id": "T1"},),
            affected_task_ids=("T1",),
            narrative="x",
        )
        from collections.abc import Mapping

        assert isinstance(v.ops[0], Mapping)
        # frozendict raises TypeError on item assignment.
        with pytest.raises((TypeError, AttributeError)):
            v.ops[0]["op"] = "remove"  # type: ignore[index]

    def test_default_collections_are_singleton_empty_tuples(self) -> None:
        # Tuples are immutable so two verdicts sharing the default
        # empty tuple is safe (no foot-gun like a shared default
        # ``[]`` would have been).
        a = IntentVerdict(intent_comment_id=1, outcome="honored", narrative="x")
        b = IntentVerdict(intent_comment_id=2, outcome="honored", narrative="x")
        assert a.ops == b.ops == ()
        assert a.affected_task_ids == b.affected_task_ids == ()


class TestIntentVerdictValidation:
    def test_self_supersedence_rejected(self) -> None:
        # codex P2 on #1802: by_intent_comment_id pointing at the
        # verdict's own intent is meaningless and indicates a
        # malformed batch.  Fail at construction.
        with pytest.raises(ValueError, match="must not reference"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                by_intent_comment_id=10,
                narrative="self",
            )

    def test_superseded_outcome_requires_by_intent_comment_id(self) -> None:
        # codex P2 on #1802: superseded outcome without a pointer is
        # nonsense — INV-E can't tell self-vs-cross-author without it.
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                narrative="something",
            )

    def test_reshaped_outcome_requires_narrative(self) -> None:
        # reshaped means material change → reply-back fires → reply
        # needs prose.  Empty narrative is a contract violation.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="reshaped",
                ops=({"op": "rewrite", "id": "T1", "title": "x"},),
                affected_task_ids=("T1",),
            )

    def test_superseded_outcome_requires_narrative(self) -> None:
        # Same reasoning as the reshaped case.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                by_intent_comment_id=11,
                narrative="",
            )

    def test_superseded_whitespace_only_narrative_rejected(self) -> None:
        # Narrative must be substantively non-empty.  "   " is empty
        # for human-facing reply purposes.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(
                intent_comment_id=10,
                outcome="superseded",
                by_intent_comment_id=11,
                narrative="   ",
            )

    def test_honored_outcome_requires_narrative(self) -> None:
        # HOL-3 / #1897: narrative is required on EVERY outcome (not
        # just reshaped/superseded).  The per-task narrative chain
        # (#1894 HOL-8) accumulates one entry per verdict; without a
        # honored-narrative, future rescopes can't reason about WHY a
        # task got the work it has.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(intent_comment_id=10, outcome="honored")

    def test_no_op_outcome_requires_narrative(self) -> None:
        # HOL-3 / #1897: no_op without narrative is the PR #1890 bug
        # — Rob's intent silently dropped.  The narrative IS the
        # reply-back reason ("why am I skipping this?").
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(intent_comment_id=10, outcome="no_op")

    def test_honored_whitespace_only_narrative_rejected(self) -> None:
        # Same substantively-non-empty rule as reshaped/superseded.
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(intent_comment_id=10, outcome="honored", narrative="  ")

    def test_no_op_whitespace_only_narrative_rejected(self) -> None:
        with pytest.raises(ValueError, match="narrative"):
            IntentVerdict(intent_comment_id=10, outcome="no_op", narrative="\t\n")

    def test_by_intent_comment_id_rejected_on_honored(self) -> None:
        # codex P2 round 2 on #1802: only ``superseded`` carries a
        # supersedence pointer.  ``honored`` + ``by_intent_comment_id``
        # is contradictory metadata; reject at boundary.
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                by_intent_comment_id=2,
            )

    def test_by_intent_comment_id_rejected_on_reshaped(self) -> None:
        # ``reshaped`` is Opus's reframing of a single intent — no
        # other intent is involved, so the pointer is contradictory.
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="reshaped",
                by_intent_comment_id=2,
                narrative="x",
            )

    def test_by_intent_comment_id_rejected_on_no_op(self) -> None:
        with pytest.raises(ValueError, match="by_intent_comment_id"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="no_op",
                by_intent_comment_id=2,
            )


class TestIntentVerdictTypeChecks:
    def test_outcome_typo_rejected(self) -> None:
        # codex P2 round 3 on #1802: Literal annotation isn't
        # enforced at runtime — a parser typo like "supersede" must
        # crash at the boundary.
        with pytest.raises(ValueError, match="outcome must be one of"):
            IntentVerdict(intent_comment_id=1, outcome="supersede")  # type: ignore[arg-type]

    def test_intent_comment_id_must_be_int(self) -> None:
        with pytest.raises(TypeError, match="intent_comment_id must be int"):
            IntentVerdict(intent_comment_id="123", outcome="honored")  # type: ignore[arg-type]

    def test_intent_comment_id_bool_rejected(self) -> None:
        # ``bool`` is an ``int`` subclass in Python; reject it
        # explicitly so ``True`` / ``False`` can't slip through as
        # comment ids.
        with pytest.raises(TypeError, match="intent_comment_id must be int"):
            IntentVerdict(intent_comment_id=True, outcome="honored")  # type: ignore[arg-type]

    def test_by_intent_comment_id_must_be_int_or_none(self) -> None:
        with pytest.raises(TypeError, match="by_intent_comment_id must be int"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="superseded",
                by_intent_comment_id="2",  # type: ignore[arg-type]
                narrative="x",
            )

    def test_ops_single_mapping_input_rejected(self) -> None:
        # codex P2 round 3 on #1802: a bare dict ``ops={"op":"keep"}``
        # would iterate as its keys and produce a tuple of strings
        # via ``deep_freeze(list(...))``.  Reject before coercion.
        with pytest.raises(TypeError, match="sequence of op mappings"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                ops={"op": "keep", "id": "T1"},  # type: ignore[arg-type]
            )

    def test_ops_entry_must_be_mapping(self) -> None:
        with pytest.raises(TypeError, match="ops entries must be Mapping"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                ops=("not a dict",),  # type: ignore[arg-type]
            )

    def test_affected_task_ids_entry_must_be_str(self) -> None:
        with pytest.raises(TypeError, match="affected_task_ids entries must be str"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                affected_task_ids=(42,),  # type: ignore[arg-type]
            )

    def test_affected_task_ids_rejects_none_entry(self) -> None:
        # Don't silently coerce ``None`` to ``"None"`` — that hides
        # parser bugs.
        with pytest.raises(TypeError, match="affected_task_ids entries must be str"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                affected_task_ids=(None,),  # type: ignore[arg-type]
            )

    def test_narrative_must_be_str_or_none(self) -> None:
        # codex P2 round 4 on #1802: a non-str narrative would hit
        # ``.strip()`` for reshaped/superseded and AttributeError
        # instead of a clean boundary error.
        with pytest.raises(TypeError, match="narrative must be str or None"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                narrative=42,  # type: ignore[arg-type]
            )

    def test_affected_task_ids_rejects_bare_str(self) -> None:
        # codex P2 round 4 on #1802: ``affected_task_ids="T1"`` iterates
        # as ('T', '1') and would silently be mis-stored.  Reject.
        with pytest.raises(TypeError, match="not a bare str"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                affected_task_ids="T1",  # type: ignore[arg-type]
            )

    def test_affected_task_ids_rejects_set(self) -> None:
        # codex P2 round 4 on #1802: docstring contract says ordered.
        # Set iteration order is nondeterministic — would produce
        # non-reproducible verdicts.
        with pytest.raises(TypeError, match="must be ordered"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                affected_task_ids={"T1", "T2"},  # type: ignore[arg-type]
            )

    def test_affected_task_ids_rejects_frozenset(self) -> None:
        with pytest.raises(TypeError, match="must be ordered"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="honored",
                affected_task_ids=frozenset({"T1"}),  # type: ignore[arg-type]
            )

    def test_ops_generator_input_preserves_entries(self) -> None:
        # codex P2 round 4 on #1802: a generator iterates once.  If
        # the validation loop and the freeze pass each iterate, the
        # second pass sees an exhausted iterator and silently stores
        # an empty tuple.  Materialize once up-front.
        def ops_gen() -> "Iterator[dict[str, Any]]":
            yield {"op": "rewrite", "id": "T1"}
            yield {"op": "remove", "id": "T2"}

        v = IntentVerdict(
            intent_comment_id=1,
            outcome="reshaped",
            ops=ops_gen(),  # type: ignore[arg-type]
            affected_task_ids=("T1",),
            narrative="x",
        )
        assert len(v.ops) == 2, "generator entries must not be silently dropped"

    def test_affected_task_ids_generator_input_preserves_entries(self) -> None:
        def ids_gen() -> "Iterator[str]":
            yield "T1"
            yield "T2"

        v = IntentVerdict(
            intent_comment_id=1,
            outcome="honored",
            affected_task_ids=ids_gen(),  # type: ignore[arg-type]
            narrative="x",
        )
        assert v.affected_task_ids == ("T1", "T2")

    def test_no_op_rejects_non_empty_ops(self) -> None:
        # codex P2 round 5 on #1802: ``no_op`` means "produced no task
        # changes".  A no_op verdict carrying ops is contradictory and
        # would mislead INV-E (skip reply-back) while still claiming
        # task attribution.
        with pytest.raises(ValueError, match="no_op.*must have empty ops"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="no_op",
                ops=({"op": "rewrite", "id": "T1"},),
                narrative="x",
            )

    def test_no_op_rejects_non_empty_affected_task_ids(self) -> None:
        with pytest.raises(ValueError, match="no_op.*must have empty"):
            IntentVerdict(
                intent_comment_id=1,
                outcome="no_op",
                affected_task_ids=("T1",),
                narrative="x",
            )


# ── TaskStatus.SKIPPED (HOL-5 / #1899) ───────────────────────────────────────


class TestTaskStatusSkipped:
    """HOL-5 (#1899): SKIPPED enum value for no_op marker tasks.

    Pinned at the type layer; the projections into the per-oracle
    TaskStatus types are exercised by tests/test_tasks.py and
    tests/test_worker.py — there's no extra wiring to test here other
    than membership + value.
    """

    def test_skipped_member_present(self) -> None:
        from fido.types import TaskStatus

        assert hasattr(TaskStatus, "SKIPPED")

    def test_skipped_string_value(self) -> None:
        from fido.types import TaskStatus

        # Wire-format value is the lowercase string we write into
        # tasks.json — same convention as the other members.
        assert str(TaskStatus.SKIPPED) == "skipped"

    def test_skipped_distinct_from_completed_and_blocked(self) -> None:
        # The whole point of SKIPPED is to be distinguishable from
        # COMPLETED for UI/lineage purposes, and from BLOCKED because
        # SKIPPED is terminal (never picked, never unblockable) while
        # BLOCKED is pending-but-paused.
        from fido.types import TaskStatus

        assert TaskStatus.SKIPPED != TaskStatus.COMPLETED
        assert TaskStatus.SKIPPED != TaskStatus.BLOCKED
        assert TaskStatus.SKIPPED != TaskStatus.PENDING
        assert TaskStatus.SKIPPED != TaskStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# HOL-23 / #1917 — verdict_is_material predicate
# ---------------------------------------------------------------------------


class TestVerdictIsMaterial:
    """HOL-23 / #1917: predicate gates per-thread reply-back emission.

    Test matrix covers each outcome × match/divergence combination
    per the issue's acceptance criteria.
    """

    def _intent(self, change_request: str = "Fix the bug") -> object:
        from fido.types import RescopeIntent

        return RescopeIntent(
            change_request=change_request,
            comment_id=42,
            timestamp="2026-05-24T10:00:00Z",
        )

    def _verdict(self, outcome: str) -> object:
        from fido.types import IntentVerdict

        return IntentVerdict(
            intent_comment_id=42,
            outcome=outcome,
            narrative="x",
        )

    def test_reshaped_is_material(self) -> None:
        from fido.types import verdict_is_material

        assert verdict_is_material(self._intent(), self._verdict("reshaped"))

    def test_superseded_is_material(self) -> None:
        from fido.types import IntentVerdict, verdict_is_material

        verdict = IntentVerdict(
            intent_comment_id=42,
            outcome="superseded",
            by_intent_comment_id=99,
            narrative="x",
        )
        assert verdict_is_material(self._intent(), verdict)

    def test_no_op_is_material(self) -> None:
        from fido.types import verdict_is_material

        assert verdict_is_material(self._intent(), self._verdict("no_op"))

    def test_honored_without_descriptions_is_not_material(self) -> None:
        from fido.types import verdict_is_material

        assert not verdict_is_material(self._intent(), self._verdict("honored"))

    def test_honored_matching_task_description_is_not_material(self) -> None:
        from fido.types import verdict_is_material

        assert not verdict_is_material(
            self._intent("Fix the bug"),
            self._verdict("honored"),
            task_descriptions=("Fix the bug in the retry path.",),
        )

    def test_honored_diverging_task_description_is_material(self) -> None:
        from fido.types import verdict_is_material

        # Task description doesn't contain the change_request text —
        # "you said X, I queued Y" pattern → material.
        assert verdict_is_material(
            self._intent("Fix the bug"),
            self._verdict("honored"),
            task_descriptions=("Refactor the entire retry subsystem.",),
        )

    def test_honored_case_insensitive_match(self) -> None:
        from fido.types import verdict_is_material

        assert not verdict_is_material(
            self._intent("FIX THE BUG"),
            self._verdict("honored"),
            task_descriptions=("fix the bug pls.",),
        )

    def test_honored_with_multiple_tasks_any_match_is_not_material(self) -> None:
        from fido.types import verdict_is_material

        # If ANY task description covers the request verbatim, the
        # intent is honoured-as-asked → not material.
        assert not verdict_is_material(
            self._intent("Fix the bug"),
            self._verdict("honored"),
            task_descriptions=(
                "Unrelated description.",
                "Fix the bug as part of the broader refactor.",
            ),
        )

    def test_honored_with_empty_change_request_is_not_material(self) -> None:
        from fido.types import verdict_is_material

        # Empty change_request can't be a basis for divergence —
        # callers shouldn't construct intents this way but the
        # predicate must not divide-by-zero.
        assert not verdict_is_material(
            self._intent(""),
            self._verdict("honored"),
            task_descriptions=("anything",),
        )


class TestIntentThreadTerminal:
    """HOL-27 / #1921: an intent is terminal when every task in its
    contributing_intents has reached a terminal status."""

    def _task(
        self,
        tid: str,
        status: str,
        contributing_intents: tuple[int, ...] = (),
    ) -> dict[str, object]:
        return {
            "id": tid,
            "status": status,
            "contributing_intents": list(contributing_intents),
        }

    def test_all_completed_is_terminal(self) -> None:
        from fido.types import intent_thread_terminal

        tasks = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "completed", (101,)),
        ]
        assert intent_thread_terminal(101, tasks)

    def test_all_skipped_is_terminal(self) -> None:
        from fido.types import intent_thread_terminal

        tasks = [self._task("t1", "skipped", (101,))]
        assert intent_thread_terminal(101, tasks)

    def test_mixed_completed_and_skipped_is_terminal(self) -> None:
        from fido.types import intent_thread_terminal

        tasks = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "skipped", (101,)),
        ]
        assert intent_thread_terminal(101, tasks)

    def test_any_pending_is_not_terminal(self) -> None:
        from fido.types import intent_thread_terminal

        tasks = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "pending", (101,)),
        ]
        assert not intent_thread_terminal(101, tasks)

    def test_any_in_progress_is_not_terminal(self) -> None:
        from fido.types import intent_thread_terminal

        tasks = [self._task("t1", "in_progress", (101,))]
        assert not intent_thread_terminal(101, tasks)

    def test_any_blocked_is_not_terminal(self) -> None:
        from fido.types import intent_thread_terminal

        tasks = [self._task("t1", "blocked", (101,))]
        assert not intent_thread_terminal(101, tasks)

    def test_no_contributing_tasks_is_not_terminal(self) -> None:
        from fido.types import intent_thread_terminal

        # Vacuous case: intent 101 never landed on any task.  HOL-24's
        # no_op path already notified; this predicate must NOT also
        # fire (otherwise HOL-28 double-replies).
        tasks = [self._task("t1", "completed", (202,))]
        assert not intent_thread_terminal(101, tasks)

    def test_tolerates_taskstatus_enum_instance(self) -> None:
        from fido.types import TaskStatus, intent_thread_terminal

        tasks = [
            {
                "id": "t1",
                "status": TaskStatus.COMPLETED,
                "contributing_intents": [101],
            }
        ]
        assert intent_thread_terminal(101, tasks)

    def test_ignores_tasks_not_contributing(self) -> None:
        from fido.types import intent_thread_terminal

        # t2 is pending but doesn't list intent 101 — must not block
        # 101's terminal status.
        tasks = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "pending", (202,)),
        ]
        assert intent_thread_terminal(101, tasks)


class TestNewlyTerminalIntentThreads:
    """HOL-27: identifies intent threads that transitioned from
    non-terminal to terminal across a tasks.json before/after snapshot."""

    def _task(
        self,
        tid: str,
        status: str,
        contributing_intents: tuple[int, ...] = (),
    ) -> dict[str, object]:
        return {
            "id": tid,
            "status": status,
            "contributing_intents": list(contributing_intents),
        }

    def test_one_intent_transitioned(self) -> None:
        from fido.types import newly_terminal_intent_threads

        prev = [self._task("t1", "in_progress", (101,))]
        new = [self._task("t1", "completed", (101,))]
        assert newly_terminal_intent_threads(prev, new) == (101,)

    def test_intent_already_terminal_excluded(self) -> None:
        from fido.types import newly_terminal_intent_threads

        # Intent 101 was already all-terminal before the transition,
        # don't re-emit a reply for it.
        prev = [self._task("t1", "completed", (101,))]
        new = [self._task("t1", "completed", (101,))]
        assert newly_terminal_intent_threads(prev, new) == ()

    def test_intent_still_pending_excluded(self) -> None:
        from fido.types import newly_terminal_intent_threads

        prev = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "pending", (101,)),
        ]
        new = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "in_progress", (101,)),
        ]
        assert newly_terminal_intent_threads(prev, new) == ()

    def test_multiple_intents_transitioned_in_insertion_order(self) -> None:
        from fido.types import newly_terminal_intent_threads

        prev = [
            self._task("t1", "pending", (101,)),
            self._task("t2", "pending", (202,)),
        ]
        new = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "completed", (202,)),
        ]
        # 101 first because t1 lists it first.
        assert newly_terminal_intent_threads(prev, new) == (101, 202)

    def test_de_duplicated(self) -> None:
        from fido.types import newly_terminal_intent_threads

        prev = [
            self._task("t1", "pending", (101,)),
            self._task("t2", "pending", (101,)),
        ]
        new = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "completed", (101,)),
        ]
        # 101 appears in both tasks but emitted once.
        assert newly_terminal_intent_threads(prev, new) == (101,)

    def test_partial_thread_transition_excluded(self) -> None:
        from fido.types import newly_terminal_intent_threads

        # Intent 101 contributes to t1 (now done) AND t2 (still pending)
        # — thread is NOT yet terminal.
        prev = [
            self._task("t1", "pending", (101,)),
            self._task("t2", "pending", (101,)),
        ]
        new = [
            self._task("t1", "completed", (101,)),
            self._task("t2", "pending", (101,)),
        ]
        assert newly_terminal_intent_threads(prev, new) == ()
