"""Tests for fido.critics — Layer 2 critic verdict types, parsers, runners."""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from fido.critics import (
    InsightDedupVerdict,
    ReplyProseVerdict,
    TaskCompletionVerdict,
    TaskCreationProposedSplit,
    TaskCreationVerdict,
    _parse_insight_dedup_verdict,
    _parse_proposed_splits,
    _parse_reply_prose_verdict,
    _parse_task_completion_verdict,
    _parse_task_creation_verdict,
    run_insight_dedup_critic,
    run_reply_prose_critic,
    run_task_completion_critic,
    run_task_creation_critic,
)

# ---------------------------------------------------------------------------
# Hand-rolled fakes (per project rule: no MagicMock in new test files)
# ---------------------------------------------------------------------------


@dataclass
class _FakeAgentCall:
    args: tuple
    kwargs: dict


@dataclass
class _FakeAgent:
    """Hand-rolled ProviderAgent stand-in for critic tests.

    ``run_turn_responses`` is consumed in order.  When it's exhausted the
    next call raises ``IndexError`` — tests rely on that to catch
    excess critic invocations.

    ``run_turn_exception``, when set, is raised in place of returning a
    response; tests use it to exercise the fail-open transport-error path.
    """

    run_turn_responses: list[str] = field(default_factory=list)
    run_turn_exception: BaseException | None = None
    calls: list[_FakeAgentCall] = field(default_factory=list)

    def run_turn(self, *args: object, **kwargs: object) -> str:
        self.calls.append(_FakeAgentCall(args=args, kwargs=kwargs))
        if self.run_turn_exception is not None:
            raise self.run_turn_exception
        return self.run_turn_responses.pop(0)


@dataclass
class _FakePrompts:
    """Hand-rolled Prompts stand-in — only the methods exercised here."""

    task_creation_critic_prompt_value: str = "critic-prompt"
    calls: list[tuple] = field(default_factory=list)

    def task_creation_critic_prompt(
        self,
        proposed_task: dict[str, Any],
        current_queue: list[dict[str, Any]],
    ) -> str:
        self.calls.append(("task_creation_critic_prompt", proposed_task))
        return self.task_creation_critic_prompt_value


# ---------------------------------------------------------------------------
# TaskCreationVerdict — constructor invariants
# ---------------------------------------------------------------------------


class TestTaskCreationVerdictDefaults:
    """Default verdict is "distinct + single, pass through" so a no-critic
    code path or a fail-open result leaves the proposed task unchanged."""

    def test_default_passes_through(self) -> None:
        v = TaskCreationVerdict()
        assert v.relationship == "distinct"
        assert v.scope == "single"
        assert v.drops_proposal is False
        assert v.fans_out is False

    def test_distinct_with_single_scope_is_pass_through(self) -> None:
        v = TaskCreationVerdict(relationship="distinct", scope="single")
        assert not v.drops_proposal
        assert not v.fans_out


class TestTaskCreationVerdictConstructorInvariants:
    """Constructor enforces the structural rules so callers can rely on
    field shapes without runtime checks."""

    def test_duplicate_of_requires_duplicate_of_id(self) -> None:
        with pytest.raises(ValueError, match="duplicate_of_id is required"):
            TaskCreationVerdict(relationship="duplicate_of")

    def test_supersedes_requires_supersedes_id(self) -> None:
        with pytest.raises(ValueError, match="supersedes_id is required"):
            TaskCreationVerdict(relationship="supersedes")

    def test_multi_requires_proposed_splits(self) -> None:
        with pytest.raises(ValueError, match="proposed_splits is required"):
            TaskCreationVerdict(scope="multi")

    def test_duplicate_of_with_id_constructs(self) -> None:
        v = TaskCreationVerdict(relationship="duplicate_of", duplicate_of_id="t-1")
        assert v.duplicate_of_id == "t-1"
        assert v.drops_proposal

    def test_supersedes_with_id_constructs(self) -> None:
        v = TaskCreationVerdict(relationship="supersedes", supersedes_id="t-2")
        assert v.supersedes_id == "t-2"
        assert v.drops_proposal

    def test_multi_with_splits_constructs(self) -> None:
        split = TaskCreationProposedSplit(title="A", description="d", invariant="inv-A")
        v = TaskCreationVerdict(scope="multi", proposed_splits=(split,))
        assert v.fans_out
        assert v.proposed_splits == (split,)


# ---------------------------------------------------------------------------
# _parse_proposed_splits
# ---------------------------------------------------------------------------


class TestParseProposedSplits:
    def test_none_returns_empty(self) -> None:
        assert _parse_proposed_splits(None) == ()

    def test_empty_list_returns_empty(self) -> None:
        assert _parse_proposed_splits([]) == ()

    def test_valid_list_parses(self) -> None:
        raw = [
            {"title": "A", "description": "d-A", "invariant": "inv-A"},
            {"title": "B", "description": "d-B", "invariant": "inv-B"},
        ]
        result = _parse_proposed_splits(raw)
        assert result is not None
        assert len(result) == 2
        assert result[0].title == "A"
        assert result[0].invariant == "inv-A"
        assert result[1].title == "B"

    def test_non_list_returns_none(self) -> None:
        assert _parse_proposed_splits("not a list") is None

    def test_non_dict_entry_returns_none(self) -> None:
        # Put the non-dict FIRST so the validity check is the one that
        # rejects it (a leading dict missing ``invariant`` would
        # short-circuit first via the missing-field path).
        assert _parse_proposed_splits(["bad", {"title": "A"}]) is None

    def test_missing_title_returns_none(self) -> None:
        assert (
            _parse_proposed_splits([{"description": "d", "invariant": "inv"}]) is None
        )

    def test_blank_title_returns_none(self) -> None:
        assert (
            _parse_proposed_splits(
                [{"title": "  ", "description": "d", "invariant": "inv"}]
            )
            is None
        )

    def test_missing_invariant_returns_none(self) -> None:
        """HOL-12 contract: every proposed split must name its own
        invariant.  Without one, the split is meaningless — the whole
        point is to fan multi-invariant proposals into invariant-sized
        children."""
        assert _parse_proposed_splits([{"title": "A", "description": "d"}]) is None

    def test_blank_invariant_returns_none(self) -> None:
        assert (
            _parse_proposed_splits(
                [{"title": "A", "description": "d", "invariant": " "}]
            )
            is None
        )

    def test_non_string_description_returns_none(self) -> None:
        """Description must be a string (empty allowed); ``null`` or
        non-string values reject the split."""
        assert (
            _parse_proposed_splits(
                [{"title": "A", "description": None, "invariant": "inv"}]
            )
            is None
        )

    def test_non_string_invariant_returns_none(self) -> None:
        assert (
            _parse_proposed_splits(
                [{"title": "A", "description": "d", "invariant": 42}]
            )
            is None
        )

    def test_strips_whitespace_on_title_and_invariant(self) -> None:
        result = _parse_proposed_splits(
            [{"title": "  A  ", "description": "d", "invariant": "  inv  "}]
        )
        assert result is not None
        assert result[0].title == "A"
        assert result[0].invariant == "inv"


# ---------------------------------------------------------------------------
# _parse_task_creation_verdict
# ---------------------------------------------------------------------------


class TestParseTaskCreationVerdict:
    def test_parses_distinct_single(self) -> None:
        obj: dict[str, Any] = {
            "relationship": "distinct",
            "scope": "single",
            "proposed_splits": [],
            "rationale": "Genuinely new work.",
        }
        v = _parse_task_creation_verdict(obj)
        assert v is not None
        assert v.relationship == "distinct"
        assert v.scope == "single"
        assert v.rationale == "Genuinely new work."

    def test_parses_duplicate_of(self) -> None:
        obj: dict[str, Any] = {
            "relationship": "duplicate_of",
            "duplicate_of_id": "existing-1",
            "scope": "single",
            "proposed_splits": [],
            "rationale": "Covered by existing-1.",
        }
        v = _parse_task_creation_verdict(obj)
        assert v is not None
        assert v.relationship == "duplicate_of"
        assert v.duplicate_of_id == "existing-1"

    def test_parses_supersedes(self) -> None:
        obj: dict[str, Any] = {
            "relationship": "supersedes",
            "supersedes_id": "old-task",
            "scope": "single",
            "proposed_splits": [],
            "rationale": "Replaces old-task.",
        }
        v = _parse_task_creation_verdict(obj)
        assert v is not None
        assert v.relationship == "supersedes"
        assert v.supersedes_id == "old-task"

    def test_parses_multi_with_splits(self) -> None:
        obj: dict[str, Any] = {
            "relationship": "distinct",
            "scope": "multi",
            "proposed_splits": [
                {"title": "A", "description": "d", "invariant": "inv-A"},
                {"title": "B", "description": "", "invariant": "inv-B"},
            ],
            "rationale": "Spans 2 invariants.",
        }
        v = _parse_task_creation_verdict(obj)
        assert v is not None
        assert v.scope == "multi"
        assert len(v.proposed_splits) == 2

    def test_rejects_duplicate_of_with_multi_scope(self) -> None:
        """Rob review on PR #1932: ``duplicate_of`` + ``scope=multi``
        is contradictory — the apply path drops before fanning out, so
        the malformed envelope would silently delete legitimate new
        work.  Reject at parse time (fail-open to default verdict)."""
        assert (
            _parse_task_creation_verdict(
                {
                    "relationship": "duplicate_of",
                    "duplicate_of_id": "t-1",
                    "scope": "multi",
                    "proposed_splits": [
                        {
                            "title": "A",
                            "description": "",
                            "invariant": "inv-A",
                        },
                    ],
                }
            )
            is None
        )

    def test_rejects_supersedes_with_multi_scope(self) -> None:
        """Same defense on the supersedes axis."""
        assert (
            _parse_task_creation_verdict(
                {
                    "relationship": "supersedes",
                    "supersedes_id": "t-1",
                    "scope": "multi",
                    "proposed_splits": [
                        {
                            "title": "A",
                            "description": "",
                            "invariant": "inv-A",
                        },
                    ],
                }
            )
            is None
        )

    def test_unknown_relationship_returns_none(self) -> None:
        assert (
            _parse_task_creation_verdict(
                {"relationship": "wat", "scope": "single", "proposed_splits": []}
            )
            is None
        )

    def test_unknown_scope_returns_none(self) -> None:
        assert (
            _parse_task_creation_verdict(
                {"relationship": "distinct", "scope": "wat", "proposed_splits": []}
            )
            is None
        )

    def test_duplicate_of_without_id_returns_none(self) -> None:
        assert (
            _parse_task_creation_verdict(
                {
                    "relationship": "duplicate_of",
                    "scope": "single",
                    "proposed_splits": [],
                }
            )
            is None
        )

    def test_supersedes_without_id_returns_none(self) -> None:
        assert (
            _parse_task_creation_verdict(
                {
                    "relationship": "supersedes",
                    "scope": "single",
                    "proposed_splits": [],
                }
            )
            is None
        )

    def test_multi_without_splits_returns_none(self) -> None:
        assert (
            _parse_task_creation_verdict(
                {
                    "relationship": "distinct",
                    "scope": "multi",
                    "proposed_splits": [],
                }
            )
            is None
        )

    def test_malformed_splits_returns_none(self) -> None:
        assert (
            _parse_task_creation_verdict(
                {
                    "relationship": "distinct",
                    "scope": "multi",
                    "proposed_splits": "not a list",
                }
            )
            is None
        )


# ---------------------------------------------------------------------------
# run_task_creation_critic — end-to-end with hand-rolled fakes
# ---------------------------------------------------------------------------


class TestRunTaskCreationCritic:
    def _proposed(self) -> dict[str, Any]:
        return {
            "title": "Add retry logic",
            "description": "Retry on transient failures.",
            "invariant": "transient failures retry up to 3 times",
        }

    def _queue(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "t-1",
                "title": "Other work",
                "description": "Unrelated.",
            }
        ]

    def test_pass_through_on_distinct_single(self) -> None:
        raw = json.dumps(
            {
                "relationship": "distinct",
                "scope": "single",
                "proposed_splits": [],
                "rationale": "Genuinely new.",
            }
        )
        agent = _FakeAgent(run_turn_responses=[raw])
        prompts = _FakePrompts()

        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=agent,
            prompts=prompts,
            critic_system_prompt="followup",
        )

        assert verdict.relationship == "distinct"
        assert not verdict.drops_proposal
        assert not verdict.fans_out

    def test_returns_duplicate_of_verdict(self) -> None:
        raw = json.dumps(
            {
                "relationship": "duplicate_of",
                "duplicate_of_id": "t-1",
                "scope": "single",
                "proposed_splits": [],
                "rationale": "Already covered.",
            }
        )
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakePrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.drops_proposal
        assert verdict.duplicate_of_id == "t-1"

    def test_returns_supersedes_verdict(self) -> None:
        raw = json.dumps(
            {
                "relationship": "supersedes",
                "supersedes_id": "t-1",
                "scope": "single",
                "proposed_splits": [],
                "rationale": "Replaces t-1.",
            }
        )
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakePrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.drops_proposal
        assert verdict.supersedes_id == "t-1"

    def test_returns_multi_verdict_with_splits(self) -> None:
        raw = json.dumps(
            {
                "relationship": "distinct",
                "scope": "multi",
                "proposed_splits": [
                    {
                        "title": "Phase 1",
                        "description": "First half.",
                        "invariant": "first half invariant",
                    },
                    {
                        "title": "Phase 2",
                        "description": "Second half.",
                        "invariant": "second half invariant",
                    },
                ],
                "rationale": "Spans two invariants.",
            }
        )
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakePrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.fans_out
        assert len(verdict.proposed_splits) == 2
        assert verdict.proposed_splits[0].invariant == "first half invariant"

    def test_transport_error_fails_open(self) -> None:
        agent = _FakeAgent(run_turn_exception=RuntimeError("transport"))
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=agent,
            prompts=_FakePrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.relationship == "distinct"
        assert verdict.scope == "single"

    def test_unparseable_json_fails_open(self) -> None:
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=["not json"]),
            prompts=_FakePrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.relationship == "distinct"

    def test_scans_past_leading_unrelated_json(self) -> None:
        """codex r3293359040 on PR #1932: a response that leads with an
        unrelated JSON object before the real verdict envelope must
        still pick up the verdict — not silently fail open through
        only-check-the-first-object logic."""
        raw = (
            '{} {"relationship": "duplicate_of", "duplicate_of_id": "t-1", '
            '"scope": "single", "proposed_splits": [], '
            '"rationale": "covered"}'
        )
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakePrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.relationship == "duplicate_of"
        assert verdict.duplicate_of_id == "t-1"

    def test_malformed_verdict_fails_open(self) -> None:
        raw = json.dumps({"relationship": "wat", "scope": "single"})
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakePrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.relationship == "distinct"

    def test_uses_critic_system_prompt(self) -> None:
        agent = _FakeAgent(
            run_turn_responses=[
                '{"relationship": "distinct", "scope": "single", "proposed_splits": []}'
            ]
        )
        run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=agent,
            prompts=_FakePrompts(),
            critic_system_prompt="FOLLOWUP",
        )
        assert len(agent.calls) == 1
        assert agent.calls[0].kwargs["system_prompt"] == "FOLLOWUP"

    def test_context_overflow_propagates(self) -> None:
        from fido.provider import ContextOverflowError

        agent = _FakeAgent(run_turn_exception=ContextOverflowError("overflow"))
        with pytest.raises(ContextOverflowError):
            run_task_creation_critic(
                self._proposed(),
                self._queue(),
                agent=agent,
                prompts=_FakePrompts(),
                critic_system_prompt="followup",
            )

    def test_session_leak_propagates(self) -> None:
        from fido.provider import SessionLeakError

        agent = _FakeAgent(run_turn_exception=SessionLeakError("leak"))
        with pytest.raises(SessionLeakError):
            run_task_creation_critic(
                self._proposed(),
                self._queue(),
                agent=agent,
                prompts=_FakePrompts(),
                critic_system_prompt="followup",
            )


# ---------------------------------------------------------------------------
# TaskCompletionVerdict — constructor invariants
# ---------------------------------------------------------------------------


class TestTaskCompletionVerdictDefaults:
    """Default verdict is "passed, no gap" so a no-critic code path or
    fail-open result leaves the just-landed commit unchanged."""

    def test_default_passes(self) -> None:
        v = TaskCompletionVerdict()
        assert v.passed is True
        assert v.gap == ""
        assert v.rationale == ""

    def test_pass_with_rationale_constructs(self) -> None:
        v = TaskCompletionVerdict(passed=True, rationale="diff covers invariant")
        assert v.passed
        assert v.rationale == "diff covers invariant"


class TestTaskCompletionVerdictConstructorInvariants:
    def test_fail_requires_gap(self) -> None:
        with pytest.raises(ValueError, match="gap is required"):
            TaskCompletionVerdict(passed=False)

    def test_fail_rejects_blank_gap(self) -> None:
        with pytest.raises(ValueError, match="gap is required"):
            TaskCompletionVerdict(passed=False, gap=" \t ")

    def test_fail_with_gap_constructs(self) -> None:
        v = TaskCompletionVerdict(
            passed=False, gap="diff touched scope-creep file", rationale="long"
        )
        assert not v.passed
        assert v.gap == "diff touched scope-creep file"


# ---------------------------------------------------------------------------
# _parse_task_completion_verdict
# ---------------------------------------------------------------------------


class TestParseTaskCompletionVerdict:
    def test_parses_passed_true(self) -> None:
        v = _parse_task_completion_verdict({"passed": True, "rationale": "looks right"})
        assert v is not None
        assert v.passed
        assert v.rationale == "looks right"

    def test_parses_passed_false_with_gap(self) -> None:
        v = _parse_task_completion_verdict(
            {
                "passed": False,
                "gap": "diff doesn't touch the invariant",
                "rationale": "longer",
            }
        )
        assert v is not None
        assert not v.passed
        assert v.gap == "diff doesn't touch the invariant"
        assert v.rationale == "longer"

    def test_missing_passed_returns_none(self) -> None:
        assert _parse_task_completion_verdict({"rationale": "x"}) is None

    def test_non_bool_passed_returns_none(self) -> None:
        assert _parse_task_completion_verdict({"passed": "yes"}) is None

    def test_passed_false_without_gap_returns_none(self) -> None:
        """A ``false`` verdict with no gap can't drive a meaningful
        retry — caller treats as fail-open to avoid useless soft-resets."""
        assert (
            _parse_task_completion_verdict({"passed": False, "rationale": "x"}) is None
        )

    def test_passed_false_with_blank_gap_returns_none(self) -> None:
        assert (
            _parse_task_completion_verdict(
                {"passed": False, "gap": " ", "rationale": "x"}
            )
            is None
        )


# ---------------------------------------------------------------------------
# run_task_completion_critic
# ---------------------------------------------------------------------------


@dataclass
class _FakeCompletionPrompts:
    """Hand-rolled Prompts stand-in for the completion critic."""

    prompt_value: str = "completion-critic-prompt"
    calls: list[tuple] = field(default_factory=list)

    def task_completion_critic_prompt(
        self,
        task_invariant: str,
        task_description: str,
        diff: str,
    ) -> str:
        self.calls.append(("task_completion_critic_prompt", task_invariant, diff))
        return self.prompt_value


class TestRunTaskCompletionCritic:
    def test_pass_through_on_passed_true(self) -> None:
        agent = _FakeAgent(
            run_turn_responses=[
                '{"passed": true, "rationale": "diff establishes the invariant"}'
            ]
        )
        verdict = run_task_completion_critic(
            task_invariant="rescope ops are typed values",
            task_description="Replace raw dicts in the rescope reducer.",
            diff="diff --git a/x b/x\n+typed\n",
            agent=agent,
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.passed
        assert verdict.rationale == "diff establishes the invariant"

    def test_returns_fail_verdict_with_gap(self) -> None:
        raw = json.dumps(
            {
                "passed": False,
                "gap": "diff also touches unrelated file foo.py",
                "rationale": "scope-creep into formatting",
            }
        )
        verdict = run_task_completion_critic(
            task_invariant="rescope ops are typed values",
            task_description="Replace raw dicts in the rescope reducer.",
            diff="diff --git a/foo b/foo\n+unrelated\n",
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="followup",
        )
        assert not verdict.passed
        assert "foo.py" in verdict.gap

    def test_empty_diff_with_legitimate_fail_verdict(self) -> None:
        """An empty diff should typically draw a ``passed=false`` verdict
        from a well-behaved critic (PR #1858 pattern).  The runner just
        carries that verdict through — the prompt does the "empty diff
        fails establishes axis" reasoning."""
        raw = json.dumps(
            {
                "passed": False,
                "gap": "no code change for the named invariant",
                "rationale": "empty diff",
            }
        )
        verdict = run_task_completion_critic(
            task_invariant="any invariant",
            task_description="",
            diff="",
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="followup",
        )
        assert not verdict.passed

    def test_transport_error_fails_open(self) -> None:
        agent = _FakeAgent(run_turn_exception=RuntimeError("transport"))
        verdict = run_task_completion_critic(
            task_invariant="x",
            task_description="",
            diff="diff",
            agent=agent,
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="followup",
        )
        # Fail-open default — pass through, don't block the commit.
        assert verdict.passed

    def test_unparseable_json_fails_open(self) -> None:
        verdict = run_task_completion_critic(
            task_invariant="x",
            task_description="",
            diff="diff",
            agent=_FakeAgent(run_turn_responses=["not json at all"]),
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.passed

    def test_malformed_verdict_fails_open(self) -> None:
        verdict = run_task_completion_critic(
            task_invariant="x",
            task_description="",
            diff="diff",
            agent=_FakeAgent(run_turn_responses=['{"verdict": "wat"}']),
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="followup",
        )
        assert verdict.passed

    def test_scans_past_leading_unrelated_json(self) -> None:
        """Same defense as the other critics: a leading unrelated object
        before the verdict envelope must not mask the verdict."""
        raw = '{} {"passed": false, "gap": "scope-creep"}'
        verdict = run_task_completion_critic(
            task_invariant="x",
            task_description="",
            diff="diff",
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="followup",
        )
        assert not verdict.passed
        assert verdict.gap == "scope-creep"

    def test_uses_critic_system_prompt(self) -> None:
        agent = _FakeAgent(run_turn_responses=['{"passed": true}'])
        run_task_completion_critic(
            task_invariant="x",
            task_description="",
            diff="diff",
            agent=agent,
            prompts=_FakeCompletionPrompts(),
            critic_system_prompt="FOLLOWUP",
        )
        assert agent.calls[0].kwargs["system_prompt"] == "FOLLOWUP"

    def test_context_overflow_propagates(self) -> None:
        from fido.provider import ContextOverflowError

        agent = _FakeAgent(run_turn_exception=ContextOverflowError("overflow"))
        with pytest.raises(ContextOverflowError):
            run_task_completion_critic(
                task_invariant="x",
                task_description="",
                diff="diff",
                agent=agent,
                prompts=_FakeCompletionPrompts(),
                critic_system_prompt="followup",
            )

    def test_session_leak_propagates(self) -> None:
        from fido.provider import SessionLeakError

        agent = _FakeAgent(run_turn_exception=SessionLeakError("leak"))
        with pytest.raises(SessionLeakError):
            run_task_completion_critic(
                task_invariant="x",
                task_description="",
                diff="diff",
                agent=agent,
                prompts=_FakeCompletionPrompts(),
                critic_system_prompt="followup",
            )


# ---------------------------------------------------------------------------
# HOL-18 / #1912 — reply-prose claim-grounding critic
# ---------------------------------------------------------------------------


class TestReplyProseVerdictDefaults:
    def test_default_passes(self) -> None:
        v = ReplyProseVerdict()
        assert v.passed is True
        assert v.gap == ""
        assert v.rationale == ""

    def test_pass_with_rationale_constructs(self) -> None:
        v = ReplyProseVerdict(passed=True, rationale="claims all map")
        assert v.passed


class TestReplyProseVerdictConstructorInvariants:
    def test_fail_requires_gap(self) -> None:
        with pytest.raises(ValueError, match="gap is required"):
            ReplyProseVerdict(passed=False)

    def test_fail_rejects_blank_gap(self) -> None:
        with pytest.raises(ValueError, match="gap is required"):
            ReplyProseVerdict(passed=False, gap=" \t \n")

    def test_fail_with_gap_constructs(self) -> None:
        v = ReplyProseVerdict(
            passed=False,
            gap='prose claims "commit deadbee" but no such SHA',
        )
        assert not v.passed


class TestParseReplyProseVerdict:
    def test_parses_passed_true(self) -> None:
        v = _parse_reply_prose_verdict(
            {"passed": True, "rationale": "no falsifiable specifics"}
        )
        assert v is not None
        assert v.passed
        assert v.rationale == "no falsifiable specifics"

    def test_parses_passed_false_with_gap(self) -> None:
        v = _parse_reply_prose_verdict(
            {
                "passed": False,
                "gap": 'prose claims "commit deadbee" not in repo',
                "rationale": "checked git log",
            }
        )
        assert v is not None
        assert not v.passed
        assert "deadbee" in v.gap

    def test_missing_passed_returns_none(self) -> None:
        assert _parse_reply_prose_verdict({"rationale": "x"}) is None

    def test_non_bool_passed_returns_none(self) -> None:
        assert _parse_reply_prose_verdict({"passed": "yes"}) is None

    def test_fail_without_gap_returns_none(self) -> None:
        """No-gap fail can't drive a meaningful retry — caller fails open."""
        assert _parse_reply_prose_verdict({"passed": False}) is None


@dataclass
class _FakeReplyProsePrompts:
    """Hand-rolled Prompts stand-in for the reply-prose critic."""

    prompt_value: str = "reply-prose-critic-prompt"
    calls: list[tuple] = field(default_factory=list)

    def reply_prose_claim_grounding_prompt(
        self,
        reply_text: str,
        structured_state: dict[str, Any],
    ) -> str:
        self.calls.append(("reply_prose", reply_text, structured_state))
        return self.prompt_value


class TestRunReplyProseCritic:
    """End-to-end: critic catches PR #1858's "the work is in commit ABC"
    pattern + #1855's "I filed an issue" without-URL pattern."""

    def _state(self) -> dict[str, Any]:
        return {
            "recent_commit_shas": ["abc1234567"],
            "open_issue_numbers": [101, 102],
            "filed_insights": [],
            "referenced_files": ["src/fido/tasks.py"],
        }

    def test_pass_through_on_grounded_prose(self) -> None:
        raw = json.dumps(
            {"passed": True, "rationale": "all SHAs/numbers map to ground truth"}
        )
        verdict = run_reply_prose_critic(
            reply_text="Pushed the fix in commit abc1234567 — see #101.",
            structured_state=self._state(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.passed

    def test_catches_nonexistent_commit_claim(self) -> None:
        """PR #1858 regression — prose claims a SHA that's not in the repo."""
        raw = json.dumps(
            {
                "passed": False,
                "gap": 'prose claims "commit deadbeef" not in recent_commit_shas',
                "rationale": "Only abc1234567 was committed this turn.",
            }
        )
        verdict = run_reply_prose_critic(
            reply_text="Fixed it in commit deadbeef.",
            structured_state=self._state(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.passed
        assert "deadbeef" in verdict.gap

    def test_catches_nonexistent_issue_claim(self) -> None:
        """#1855 regression — prose claims a filed issue with no URL anchor."""
        raw = json.dumps(
            {
                "passed": False,
                "gap": 'prose says "filed issue #999" but #999 not in open_issue_numbers',
                "rationale": "Insight filing presumably failed silently.",
            }
        )
        verdict = run_reply_prose_critic(
            reply_text="I filed issue #999 to track this.",
            structured_state=self._state(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.passed
        assert "999" in verdict.gap

    def test_transport_error_fails_open(self) -> None:
        agent = _FakeAgent(run_turn_exception=RuntimeError("transport"))
        verdict = run_reply_prose_critic(
            reply_text="x",
            structured_state={},
            agent=agent,
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.passed

    def test_unparseable_json_fails_open(self) -> None:
        verdict = run_reply_prose_critic(
            reply_text="x",
            structured_state={},
            agent=_FakeAgent(run_turn_responses=["not json at all"]),
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.passed

    def test_malformed_verdict_fails_open(self) -> None:
        verdict = run_reply_prose_critic(
            reply_text="x",
            structured_state={},
            agent=_FakeAgent(run_turn_responses=['{"verdict": "wat"}']),
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.passed

    def test_fail_without_gap_alone_fails_open(self) -> None:
        """A response that's ONLY ``{"passed": false}`` (no gap, no
        following valid object) must fail open — the loop scans the
        whole list, finds nothing usable, and returns the default
        pass-through verdict.  Covers the saw_malformed_fail log path."""
        verdict = run_reply_prose_critic(
            reply_text="x",
            structured_state={},
            agent=_FakeAgent(run_turn_responses=['{"passed": false}']),
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.passed

    def test_scans_past_malformed_early_envelope(self) -> None:
        """Same defense as the intent-coverage critic (codex r3293424368):
        ``{"passed": false}`` early must not mask a later real verdict."""
        raw = (
            '{"passed": false} '
            '{"passed": false, "gap": "real complaint", "rationale": "x"}'
        )
        verdict = run_reply_prose_critic(
            reply_text="x",
            structured_state={},
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.passed
        assert verdict.gap == "real complaint"

    def test_uses_critic_system_prompt(self) -> None:
        agent = _FakeAgent(run_turn_responses=['{"passed": true}'])
        run_reply_prose_critic(
            reply_text="x",
            structured_state={},
            agent=agent,
            prompts=_FakeReplyProsePrompts(),
            critic_system_prompt="CRITIC-SYSTEM",
        )
        assert agent.calls[0].kwargs["system_prompt"] == "CRITIC-SYSTEM"

    def test_context_overflow_propagates(self) -> None:
        from fido.provider import ContextOverflowError

        agent = _FakeAgent(run_turn_exception=ContextOverflowError("overflow"))
        with pytest.raises(ContextOverflowError):
            run_reply_prose_critic(
                reply_text="x",
                structured_state={},
                agent=agent,
                prompts=_FakeReplyProsePrompts(),
                critic_system_prompt="critic-sys",
            )

    def test_session_leak_propagates(self) -> None:
        from fido.provider import SessionLeakError

        agent = _FakeAgent(run_turn_exception=SessionLeakError("leak"))
        with pytest.raises(SessionLeakError):
            run_reply_prose_critic(
                reply_text="x",
                structured_state={},
                agent=agent,
                prompts=_FakeReplyProsePrompts(),
                critic_system_prompt="critic-sys",
            )


# ---------------------------------------------------------------------------
# InsightDedupVerdict — HOL-19 / #1913
# ---------------------------------------------------------------------------


class TestInsightDedupVerdictDefaults:
    """Default verdict is "not a duplicate, no rationale" so a no-critic
    code path or a fail-open result lets the insight file unchanged."""

    def test_defaults_to_not_duplicate(self) -> None:
        v = InsightDedupVerdict()
        assert v.is_duplicate is False
        assert v.duplicate_url == ""
        assert v.rationale == ""


class TestInsightDedupVerdictConstructorInvariants:
    """is_duplicate=True requires duplicate_url — without it the filer
    has nothing to log and the verdict can't be acted on meaningfully."""

    def test_duplicate_without_url_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate_url is required"):
            InsightDedupVerdict(is_duplicate=True, duplicate_url="")

    def test_duplicate_with_whitespace_only_url_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate_url is required"):
            InsightDedupVerdict(is_duplicate=True, duplicate_url="   ")

    def test_duplicate_with_url_accepted(self) -> None:
        v = InsightDedupVerdict(
            is_duplicate=True,
            duplicate_url="https://github.com/FidoCanCode/home/issues/9",
            rationale="same lesson as #9",
        )
        assert v.is_duplicate
        assert v.duplicate_url.endswith("/9")


class TestParseInsightDedupVerdict:
    """Parser round-trips well-formed envelopes and returns None on every
    malformed shape so callers fail open."""

    def test_parses_clean_pass(self) -> None:
        verdict = _parse_insight_dedup_verdict(
            {"is_duplicate": False, "rationale": "distinct"}
        )
        assert verdict is not None
        assert verdict.is_duplicate is False
        assert verdict.rationale == "distinct"

    def test_parses_clean_duplicate(self) -> None:
        verdict = _parse_insight_dedup_verdict(
            {
                "is_duplicate": True,
                "duplicate_url": "https://github.com/FidoCanCode/home/issues/42",
                "rationale": "covers same invariant",
            }
        )
        assert verdict is not None
        assert verdict.is_duplicate
        assert verdict.duplicate_url == (
            "https://github.com/FidoCanCode/home/issues/42"
        )

    def test_malformed_url_treated_as_malformed_verdict(self) -> None:
        """Codex on PR #1932: a duplicate verdict with a malformed
        ``duplicate_url`` (not a GitHub /issues/N URL on the insight
        repo) used to be accepted, causing the filer to skip the
        insight while the marker-writer silently dropped the
        durability record — the insight was lost entirely.  Parser
        now returns None for any URL that doesn't target the insight
        repo's issue path, so the runner fails open and the insight
        files normally.

        Cross-repo URLs count as malformed too (codex follow-up): the
        critic only sees insights from ``_INSIGHT_REPO``, so a URL
        pointing anywhere else is hallucination."""
        for malformed in (
            "not a real URL at all",
            "https://example.com/FidoCanCode/home/issues/42",  # wrong host
            "https://github.com/FidoCanCode/home/pull/42",  # PR not issue
            "https://github.com/FidoCanCode/home/discussions/42",  # discussion
            "https://github.com/FidoCanCode/home/issues/notanumber",
            # Cross-repo: corpus only contains _INSIGHT_REPO insights
            # so any other repo is hallucination.
            "https://github.com/some-other-org/repo/issues/9",
            "https://github.com/rhencke/confusio/issues/5",
        ):
            assert (
                _parse_insight_dedup_verdict(
                    {"is_duplicate": True, "duplicate_url": malformed}
                )
                is None
            ), f"expected malformed URL to fail parse: {malformed!r}"

    def test_case_insensitive_github_url_accepted(self) -> None:
        """The shape check is case-insensitive to match GitHub's
        case-insensitive owner/repo names; the strict per-repo
        equality lives downstream at the marker-write boundary."""
        for url in (
            "https://github.com/fidocancode/home/issues/42",
            "https://GITHUB.COM/FidoCanCode/home/issues/42",
        ):
            verdict = _parse_insight_dedup_verdict(
                {"is_duplicate": True, "duplicate_url": url}
            )
            assert verdict is not None, f"expected accepted: {url!r}"
            assert verdict.is_duplicate

    def test_missing_is_duplicate_returns_none(self) -> None:
        assert _parse_insight_dedup_verdict({}) is None

    def test_non_bool_is_duplicate_returns_none(self) -> None:
        assert _parse_insight_dedup_verdict({"is_duplicate": "yes"}) is None

    def test_duplicate_without_url_returns_none(self) -> None:
        assert _parse_insight_dedup_verdict({"is_duplicate": True}) is None

    def test_duplicate_with_whitespace_url_returns_none(self) -> None:
        assert (
            _parse_insight_dedup_verdict({"is_duplicate": True, "duplicate_url": "  "})
            is None
        )

    def test_rationale_non_string_treated_as_empty(self) -> None:
        verdict = _parse_insight_dedup_verdict({"is_duplicate": False, "rationale": 42})
        assert verdict is not None
        assert verdict.rationale == ""


@dataclass
class _FakeInsightDedupPrompts:
    """Hand-rolled Prompts stand-in for the insight-dedup critic."""

    prompt_value: str = "insight-dedup-critic-prompt"
    calls: list[tuple] = field(default_factory=list)

    def insight_dedup_critic_prompt(
        self,
        proposed_insight: dict[str, str],
        recent_insights: list[dict[str, str]],
    ) -> str:
        self.calls.append(("insight_dedup", proposed_insight, recent_insights))
        return self.prompt_value


class TestRunInsightDedupCritic:
    """End-to-end: HOL-19 critic catches cross-comment near-duplicates
    that the per-comment idempotency marker can't see."""

    def _proposed(self) -> dict[str, str]:
        return {
            "title": "Mocking the network masks contract drift",
            "hook": "Tests passed in CI but the migration failed in prod.",
            "why": "Mock/prod divergence is a recurring class of bug.",
        }

    def _recent(self) -> list[dict[str, str]]:
        return [
            {
                "title": "Some other insight",
                "body": "Unrelated story.",
                "url": "https://github.com/FidoCanCode/home/issues/100",
            }
        ]

    def test_pass_through_on_distinct_insight(self) -> None:
        raw = json.dumps({"is_duplicate": False, "rationale": "distinct lesson"})
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.is_duplicate

    def test_catches_cross_comment_near_duplicate(self) -> None:
        """HOL-19 acceptance: same core lesson filed against a different
        comment is detected as near-duplicate.  The verdict's
        ``duplicate_url`` must point at an entry actually shown in
        the corpus (Codex on PR #1932 — out-of-corpus URLs are
        hallucination, fail open)."""
        # The corpus _recent() returns ends at /issues/100 — use that
        # URL so the runner's corpus-membership check sees a match.
        raw = json.dumps(
            {
                "is_duplicate": True,
                "duplicate_url": "https://github.com/FidoCanCode/home/issues/100",
                "rationale": "same mock/prod divergence lesson as #100",
            }
        )
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.is_duplicate
        assert verdict.duplicate_url.endswith("/100")

    def test_transport_error_fails_open(self) -> None:
        agent = _FakeAgent(run_turn_exception=RuntimeError("transport"))
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=agent,
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.is_duplicate

    def test_unparseable_json_fails_open(self) -> None:
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=_FakeAgent(run_turn_responses=["not json"]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.is_duplicate

    def test_malformed_verdict_fails_open(self) -> None:
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=_FakeAgent(run_turn_responses=['{"verdict": "wat"}']),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.is_duplicate

    def test_duplicate_without_url_alone_fails_open(self) -> None:
        """``{"is_duplicate": true}`` with no URL must NOT short-circuit
        the scan; the loop finishes, finds no envelope, and returns the
        default pass-through verdict.  Covers the saw_malformed_dup log."""
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=_FakeAgent(run_turn_responses=['{"is_duplicate": true}']),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.is_duplicate

    def test_scans_past_malformed_early_envelope(self) -> None:
        """Mirrors the reply-prose scan: an early ``is_duplicate=true``
        without a URL must not mask a later well-formed verdict.  The
        second verdict's URL must be in the ``_recent`` corpus or the
        runner's membership check would fail it open too (Codex on
        PR #1932)."""
        raw = (
            '{"is_duplicate": true} '
            '{"is_duplicate": true, '
            '"duplicate_url": "https://github.com/FidoCanCode/home/issues/100", '
            '"rationale": "real dup"}'
        )
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.is_duplicate
        assert verdict.duplicate_url == (
            "https://github.com/FidoCanCode/home/issues/100"
        )

    def test_duplicate_url_not_in_corpus_fails_open(self) -> None:
        """Codex on PR #1932: the prompt contract says the duplicate
        URL must be one of the ``recent_insights`` shown to the
        critic.  A repo-shaped URL that ISN'T in the corpus is
        hallucination — the critic invented a number that maps to an
        existing-but-unrelated FidoCanCode/home issue, which would
        otherwise make the filer skip filing and record a marker on
        the wrong issue.  Runner must check membership in the
        corpus before honouring the duplicate verdict; on miss, fail
        open so the insight files normally."""
        recent = [
            {
                "title": "Previously filed",
                "body": "",
                "url": "https://github.com/FidoCanCode/home/issues/100",
            }
        ]
        # Critic hallucinates a different (but repo-valid) issue
        # number that was never shown in the corpus.
        raw = json.dumps(
            {
                "is_duplicate": True,
                "duplicate_url": "https://github.com/FidoCanCode/home/issues/999",
                "rationale": "claims dup of #999 which was never shown",
            }
        )
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=recent,
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        # Failed open — out-of-corpus URL is hallucination.
        assert not verdict.is_duplicate

    def test_duplicate_url_in_corpus_case_insensitive(self) -> None:
        """The corpus check normalises both sides so a verdict that
        capitalises the URL differently than the corpus entry still
        matches.  GitHub URLs are case-insensitive on scheme/host/
        owner/repo, so requiring strict equality would reject valid
        signals from a critic that paraphrased the URL."""
        recent = [
            {
                "title": "p",
                "body": "",
                "url": "https://github.com/FidoCanCode/home/issues/42",
            }
        ]
        raw = json.dumps(
            {
                "is_duplicate": True,
                "duplicate_url": "HTTPS://GITHUB.COM/fidocancode/home/issues/42/",
                "rationale": "case-variant",
            }
        )
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=recent,
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert verdict.is_duplicate

    def test_duplicate_url_with_anchor_or_query_matches_bare_corpus_url(
        self,
    ) -> None:
        """Codex on PR #1932: a verdict URL with an anchor (e.g.
        ``#issuecomment-1``) or query (``?foo=bar``) refers to the
        SAME issue as the bare URL.  Normalisation must collapse
        these so the membership check sees them as matching.
        Without it, a critic that paraphrased the URL with an
        anchor would be treated as out-of-corpus and fail open,
        filing the duplicate insight."""
        recent = [
            {
                "title": "Existing",
                "body": "",
                "url": "https://github.com/FidoCanCode/home/issues/100",
            }
        ]
        for variant in (
            "https://github.com/FidoCanCode/home/issues/100#issuecomment-1",
            "https://github.com/FidoCanCode/home/issues/100?notification_referrer_id=x",
            "https://github.com/FidoCanCode/home/issues/100/",
            "https://github.com/FidoCanCode/home/issues/100#",
        ):
            raw = json.dumps(
                {
                    "is_duplicate": True,
                    "duplicate_url": variant,
                    "rationale": "x",
                }
            )
            verdict = run_insight_dedup_critic(
                proposed_insight=self._proposed(),
                recent_insights=recent,
                agent=_FakeAgent(run_turn_responses=[raw]),
                prompts=_FakeInsightDedupPrompts(),
                critic_system_prompt="critic-sys",
            )
            assert verdict.is_duplicate, (
                f"expected variant to match corpus: {variant!r}"
            )

    def test_duplicate_against_empty_corpus_fails_open(self) -> None:
        """Empty corpus = "any duplicate verdict is hallucination" by
        the prompt contract.  The fail-open posture lets the insight
        file even when the critic mistakenly returns a duplicate
        verdict (e.g., recent-insights search degraded to empty and
        the critic still claimed a dup against some imagined URL)."""
        raw = json.dumps(
            {
                "is_duplicate": True,
                "duplicate_url": "https://github.com/FidoCanCode/home/issues/1",
                "rationale": "imagined",
            }
        )
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=[],
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.is_duplicate

    def test_uses_critic_system_prompt(self) -> None:
        agent = _FakeAgent(run_turn_responses=['{"is_duplicate": false}'])
        run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=self._recent(),
            agent=agent,
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="CRITIC-SYSTEM",
        )
        assert agent.calls[0].kwargs["system_prompt"] == "CRITIC-SYSTEM"

    def test_empty_recent_list_passes(self) -> None:
        """The first insight ever filed has no recent peers to compare
        against — the critic correctly returns pass."""
        raw = json.dumps({"is_duplicate": False, "rationale": "no peers to compare"})
        verdict = run_insight_dedup_critic(
            proposed_insight=self._proposed(),
            recent_insights=[],
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakeInsightDedupPrompts(),
            critic_system_prompt="critic-sys",
        )
        assert not verdict.is_duplicate

    def test_context_overflow_propagates(self) -> None:
        from fido.provider import ContextOverflowError

        agent = _FakeAgent(run_turn_exception=ContextOverflowError("overflow"))
        with pytest.raises(ContextOverflowError):
            run_insight_dedup_critic(
                proposed_insight=self._proposed(),
                recent_insights=self._recent(),
                agent=agent,
                prompts=_FakeInsightDedupPrompts(),
                critic_system_prompt="critic-sys",
            )

    def test_session_leak_propagates(self) -> None:
        from fido.provider import SessionLeakError

        agent = _FakeAgent(run_turn_exception=SessionLeakError("leak"))
        with pytest.raises(SessionLeakError):
            run_insight_dedup_critic(
                proposed_insight=self._proposed(),
                recent_insights=self._recent(),
                agent=agent,
                prompts=_FakeInsightDedupPrompts(),
                critic_system_prompt="critic-sys",
            )
