"""Tests for fido.critics — Layer 2 critic verdict types, parsers, runners."""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from fido.critics import (
    TaskCreationProposedSplit,
    TaskCreationVerdict,
    _parse_proposed_splits,
    _parse_task_creation_verdict,
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
            followup_system_prompt="followup",
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
            followup_system_prompt="followup",
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
            followup_system_prompt="followup",
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
            followup_system_prompt="followup",
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
            followup_system_prompt="followup",
        )
        assert verdict.relationship == "distinct"
        assert verdict.scope == "single"

    def test_unparseable_json_fails_open(self) -> None:
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=["not json"]),
            prompts=_FakePrompts(),
            followup_system_prompt="followup",
        )
        assert verdict.relationship == "distinct"

    def test_malformed_verdict_fails_open(self) -> None:
        raw = json.dumps({"relationship": "wat", "scope": "single"})
        verdict = run_task_creation_critic(
            self._proposed(),
            self._queue(),
            agent=_FakeAgent(run_turn_responses=[raw]),
            prompts=_FakePrompts(),
            followup_system_prompt="followup",
        )
        assert verdict.relationship == "distinct"

    def test_uses_followup_system_prompt(self) -> None:
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
            followup_system_prompt="FOLLOWUP",
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
                followup_system_prompt="followup",
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
                followup_system_prompt="followup",
            )
