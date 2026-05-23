"""Tests for fido.synthesis_call — synthesis LLM call and JSON parsing."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from fido.synthesis import Insight
from fido.synthesis_call import (
    MAX_RETRIES,
    CriticExhaustedError,
    CriticVerdict,
    SynthesisExhaustedError,
    _parse_comment_response,
    call_failure_explanation,
    call_synthesis,
    critic_loop,
    extract_json_objects,
)
from fido.types import ActiveIssue, ActivePR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(
    reasoning: str = "thinking",
    reply_text: str = "My reply.",
    emoji: str | None = None,
    change_request: str | None = None,
    insights: list[dict[str, str]] | None = None,
) -> str:
    obj: dict[str, Any] = {
        "reasoning": reasoning,
        "reply_text": reply_text,
    }
    if emoji is not None:
        obj["emoji"] = emoji
    if change_request is not None:
        obj["change_request"] = change_request
    if insights is not None:
        obj["insights"] = insights
    return json.dumps(obj)


def _make_agent(return_value: str | list[str]) -> MagicMock:
    """Return a mock agent whose run_turn returns *return_value*.

    If *return_value* is a list, successive calls return successive elements.
    """
    agent = MagicMock()
    if isinstance(return_value, list):
        agent.run_turn.side_effect = return_value
    else:
        agent.run_turn.return_value = return_value
    return agent


# HOL-15 / #1909: stubs for the intent-coverage critic turn.  Threaded
# into ``side_effect`` sequences between synthesis and verify turns so
# the call ordering remains synthesis → critic → verify[ → derive].
_CRITIC_PASS = '{"passed": true}'


def _critic_fail(gap: str = "missing X") -> str:
    """Stub for a failing critic verdict — drives a synthesis retry."""
    return json.dumps({"passed": False, "gap": gap})


def _make_prompts(
    system: str = "sys",
    user: str = "user",
    followup_system: str = "followup-sys",
    critic_system: str = "critic-sys",
) -> MagicMock:
    prompts = MagicMock()
    prompts.synthesis_system_prompt.return_value = system
    prompts.synthesis_followup_system_prompt.return_value = followup_system
    prompts.critic_system_prompt.return_value = critic_system
    prompts.synthesis_prompt.return_value = user
    return prompts


# ---------------------------------------------------------------------------
# extract_json_objects
# ---------------------------------------------------------------------------


class TestExtractJsonObjects:
    def test_returns_parsed_dict_for_clean_json(self) -> None:
        result = extract_json_objects('{"a": 1}')
        assert result == [{"a": 1}]

    def test_returns_empty_for_no_braces(self) -> None:
        result = extract_json_objects("not json at all")
        assert result == []

    def test_finds_object_when_preamble_present(self) -> None:
        result = extract_json_objects('Here is the JSON: {"a": 1} done.')
        assert result == [{"a": 1}]

    def test_skips_invalid_brace_and_continues(self) -> None:
        result = extract_json_objects('{ not json, then {"a": 1}')
        assert result == [{"a": 1}]

    def test_returns_all_objects_in_order(self) -> None:
        result = extract_json_objects('{"a": 1} then {"b": 2}')
        assert result == [{"a": 1}, {"b": 2}]

    def test_handles_nested_objects(self) -> None:
        result = extract_json_objects('{"outer": {"inner": 42}}')
        assert result == [{"outer": {"inner": 42}}]

    def test_skips_non_dict_json_values(self) -> None:
        # A JSON array starting with [ has no {, and a bare number has no {.
        # A JSON string has no {. Confirm arrays are skipped.
        result = extract_json_objects("[1, 2, 3]")
        assert result == []

    def test_handles_leading_and_trailing_whitespace(self) -> None:
        result = extract_json_objects('  {"a": 1}  ')
        assert result == [{"a": 1}]

    def test_returns_empty_for_empty_string(self) -> None:
        result = extract_json_objects("")
        assert result == []


# ---------------------------------------------------------------------------
# _parse_comment_response
# ---------------------------------------------------------------------------


class TestParseCommentResponse:
    def test_valid_minimal(self) -> None:
        raw = _make_raw()
        r = _parse_comment_response(raw)
        assert r.reply_text == "My reply."
        assert r.reasoning == "thinking"
        assert r.emoji is None
        assert r.change_request is None
        assert r.insights == []

    def test_valid_with_emoji(self) -> None:
        raw = _make_raw(emoji="rocket")
        r = _parse_comment_response(raw)
        assert r.emoji == "rocket"

    def test_valid_with_change_request(self) -> None:
        raw = _make_raw(change_request="Add more tests")
        r = _parse_comment_response(raw)
        assert r.change_request == "Add more tests"

    def test_valid_with_both(self) -> None:
        raw = _make_raw(emoji="heart", change_request="Reorder the tasks")
        r = _parse_comment_response(raw)
        assert r.emoji == "heart"
        assert r.change_request == "Reorder the tasks"

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_comment_response("not json")

    def test_missing_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r"})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_empty_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": ""})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_whitespace_reply_text_raises(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "   "})
        with pytest.raises(ValueError, match="reply_text"):
            _parse_comment_response(raw)

    def test_json_array_raises(self) -> None:
        with pytest.raises(ValueError):
            _parse_comment_response("[1, 2, 3]")

    def test_json_wrapped_in_preamble(self) -> None:
        inner = _make_raw()
        raw = f"Here's the JSON:\n{inner}\nDone."
        r = _parse_comment_response(raw)
        assert r.reply_text == "My reply."

    def test_invalid_emoji_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": "thinking"})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_empty_emoji_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": ""})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_null_emoji_stays_none(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": None})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_whitespace_change_request_dropped(self) -> None:
        raw = json.dumps(
            {"reasoning": "r", "reply_text": "OK.", "change_request": "   "}
        )
        r = _parse_comment_response(raw)
        assert r.change_request is None

    def test_null_change_request_stays_none(self) -> None:
        raw = json.dumps(
            {"reasoning": "r", "reply_text": "OK.", "change_request": None}
        )
        r = _parse_comment_response(raw)
        assert r.change_request is None

    def test_missing_reasoning_defaults_to_empty(self) -> None:
        raw = json.dumps({"reply_text": "Reply."})
        r = _parse_comment_response(raw)
        assert r.reasoning == ""

    def test_non_string_emoji_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "emoji": 42})
        r = _parse_comment_response(raw)
        assert r.emoji is None

    def test_non_string_change_request_dropped(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "change_request": 42})
        r = _parse_comment_response(raw)
        assert r.change_request is None

    def test_valid_insights_parsed(self) -> None:
        raw = _make_raw(
            insights=[{"title": "Good catch", "hook": "Rob prefers X.", "why": "Y."}]
        )
        r = _parse_comment_response(raw)
        assert len(r.insights) == 1
        assert r.insights[0] == Insight(
            title="Good catch", hook="Rob prefers X.", why="Y."
        )

    def test_multiple_insights_parsed(self) -> None:
        raw = _make_raw(
            insights=[
                {"title": "A", "hook": "H1", "why": "W1"},
                {"title": "B", "hook": "H2", "why": "W2"},
            ]
        )
        r = _parse_comment_response(raw)
        assert len(r.insights) == 2
        assert r.insights[0].title == "A"
        assert r.insights[1].title == "B"

    def test_missing_insights_defaults_to_empty(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK."})
        r = _parse_comment_response(raw)
        assert r.insights == []

    def test_null_insights_defaults_to_empty(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "insights": None})
        r = _parse_comment_response(raw)
        assert r.insights == []

    def test_non_list_insights_defaults_to_empty(self) -> None:
        raw = json.dumps({"reasoning": "r", "reply_text": "OK.", "insights": "bad"})
        r = _parse_comment_response(raw)
        assert r.insights == []

    def test_malformed_insight_entry_dropped(self) -> None:
        raw = _make_raw(
            insights=[
                {"title": "", "hook": "H", "why": "W"},  # empty title
                {"title": "Good one", "hook": "H", "why": "W"},
            ]
        )
        r = _parse_comment_response(raw)
        assert len(r.insights) == 1
        assert r.insights[0].title == "Good one"

    def test_non_dict_insight_entry_dropped(self) -> None:
        raw = json.dumps(
            {"reasoning": "r", "reply_text": "OK.", "insights": ["not a dict"]}
        )
        r = _parse_comment_response(raw)
        assert r.insights == []


# ---------------------------------------------------------------------------
# call_synthesis
# ---------------------------------------------------------------------------


class TestCallSynthesis:
    def test_success_on_first_attempt(self) -> None:
        raw = _make_raw(reply_text="Great feedback!")
        agent = _make_agent(raw)
        prompts = _make_prompts()

        result = call_synthesis(
            "Please fix this", is_bot=False, agent=agent, prompts=prompts
        )

        assert result.reply_text == "Great feedback!"
        # 1 synthesis call + 1 critic call (HOL-15: always fires after a
        # parse; here it gets back the same raw JSON which has no "passed"
        # key so it fails open) + 1 verify call (change_request is None
        # so the legacy #1218 verify still fires)
        assert agent.run_turn.call_count == 3

    def test_passes_system_prompt_to_agent(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts(system="my-system-prompt")

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # Check the first call (synthesis); the verify turn is the second call.
        _, kwargs = agent.run_turn.call_args_list[0]
        assert kwargs["system_prompt"] == "my-system-prompt"

    def test_passes_user_prompt_to_agent(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts(user="my-user-prompt")

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # Check the first call (synthesis); the verify turn is the second call.
        args, _ = agent.run_turn.call_args_list[0]
        assert args[0] == "my-user-prompt"

    def test_retry_on_parse_failure_then_success(self) -> None:
        raw_bad = "not json"
        raw_good = _make_raw(reply_text="Fixed!")
        # 1 bad synthesis + 1 good synthesis + 1 critic (passes) + 1 verify
        agent = _make_agent([raw_bad, raw_good, _CRITIC_PASS, "Yes"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Fixed!"
        assert agent.run_turn.call_count == 4

    def test_retry_appends_suffix_to_prompt(self) -> None:
        raw_bad = "not json"
        raw_good = _make_raw()
        # 1 bad synthesis + 1 good synthesis + 1 verify
        agent = _make_agent([raw_bad, raw_good, "Yes"])
        prompts = _make_prompts(user="base-prompt")

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        first_call_prompt = agent.run_turn.call_args_list[0][0][0]
        second_call_prompt = agent.run_turn.call_args_list[1][0][0]
        assert first_call_prompt == "base-prompt"
        assert second_call_prompt != "base-prompt"
        assert second_call_prompt.startswith("base-prompt")

    def test_exhausts_all_retries_and_raises(self) -> None:
        agent = _make_agent(["bad"] * MAX_RETRIES)
        prompts = _make_prompts()

        with pytest.raises(SynthesisExhaustedError, match="exhausted"):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert agent.run_turn.call_count == MAX_RETRIES

    def test_exhaustion_error_mentions_constraint_b(self) -> None:
        agent = _make_agent(["bad"] * MAX_RETRIES)
        prompts = _make_prompts()

        with pytest.raises(SynthesisExhaustedError, match="Constraint B"):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

    def test_passes_issue_to_system_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()
        issue = ActiveIssue(number=7, title="Fix crash", body="It crashes.")

        call_synthesis(
            "comment", is_bot=False, agent=agent, prompts=prompts, issue=issue
        )

        prompts.synthesis_system_prompt.assert_called_once_with(issue=issue, pr=None)

    def test_passes_pr_to_system_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()
        issue = ActiveIssue(number=7, title="T", body="")
        pr = ActivePR(
            number=42, title="My PR", url="https://github.com/a/b/pull/42", body=""
        )

        call_synthesis(
            "comment", is_bot=False, agent=agent, prompts=prompts, issue=issue, pr=pr
        )

        prompts.synthesis_system_prompt.assert_called_once_with(issue=issue, pr=pr)

    def test_passes_is_bot_to_synthesis_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=True, agent=agent, prompts=prompts)

        call_kwargs = prompts.synthesis_prompt.call_args
        assert call_kwargs[1]["is_bot"] is True

    def test_passes_context_to_synthesis_prompt(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()
        ctx = {"pr_title": "My PR"}

        call_synthesis(
            "comment", is_bot=False, context=ctx, agent=agent, prompts=prompts
        )

        call_kwargs = prompts.synthesis_prompt.call_args
        assert call_kwargs[1]["context"] == ctx

    def test_provider_error_propagates_immediately(self) -> None:
        agent = MagicMock()
        agent.run_turn.side_effect = RuntimeError("provider down")
        prompts = _make_prompts()

        with pytest.raises(RuntimeError, match="provider down"):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # No retries — provider errors propagate immediately
        assert agent.run_turn.call_count == 1

    def test_success_with_emoji_and_change_request(self) -> None:
        raw = _make_raw(
            reply_text="Here's what I think.",
            emoji="heart",
            change_request="Drop the parser refactor",
        )
        agent = _make_agent(raw)
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.emoji == "heart"
        assert result.change_request == "Drop the parser refactor"

    def test_default_context_is_none(self) -> None:
        agent = _make_agent(_make_raw())
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        call_kwargs = prompts.synthesis_prompt.call_args
        assert call_kwargs[1].get("context") is None

    def test_max_retries_constant_is_three(self) -> None:
        assert MAX_RETRIES == 3


# ---------------------------------------------------------------------------
# call_failure_explanation
# ---------------------------------------------------------------------------


def _make_failure_prompts(failure_text: str = "fallback prompt: str") -> MagicMock:
    prompts = MagicMock()
    prompts.synthesis_failure_explanation_prompt.return_value = failure_text
    return prompts


class TestCallFailureExplanation:
    def test_success_on_first_attempt(self) -> None:
        agent = _make_agent("Sorry, I couldn't reply — please rephrase.")
        prompts = _make_failure_prompts("explain failure")

        result = call_failure_explanation(
            "please fix this", agent=agent, prompts=prompts
        )

        assert result.reply_text == "Sorry, I couldn't reply — please rephrase."
        assert result.emoji is None
        assert result.change_request is None
        assert result.insights == []
        # Prompt builder was given the original comment so the LLM can
        # reference it in the explanation.
        prompts.synthesis_failure_explanation_prompt.assert_called_once_with(
            "please fix this"
        )
        assert agent.run_turn.call_count == 1

    def test_strips_whitespace_from_reply(self) -> None:
        agent = _make_agent("   Plain reply.\n\n")
        prompts = _make_failure_prompts()

        result = call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert result.reply_text == "Plain reply."

    def test_retries_with_nudge_on_empty_response(self) -> None:
        # First attempt empty, second succeeds — exercises the retry-with-nudge
        # loop AND the ``if attempt > 0`` success-after-retry log branch.
        agent = _make_agent(["", "Eventually a real reply."])
        prompts = _make_failure_prompts("base")

        result = call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert result.reply_text == "Eventually a real reply."
        assert agent.run_turn.call_count == 2
        # The retry attempt's prompt has the nudge suffix appended.
        first_call_arg = agent.run_turn.call_args_list[0][0][0]
        second_call_arg = agent.run_turn.call_args_list[1][0][0]
        assert first_call_arg == "base"
        assert second_call_arg.startswith("base")
        assert second_call_arg != first_call_arg  # nudge was appended

    def test_raises_synthesis_exhausted_when_all_attempts_empty(self) -> None:
        agent = _make_agent([""] * MAX_RETRIES)
        prompts = _make_failure_prompts()

        with pytest.raises(SynthesisExhaustedError, match="failure-explanation"):
            call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert agent.run_turn.call_count == MAX_RETRIES

    def test_handles_none_response_as_empty(self) -> None:
        # Defensive: agent.run_turn returning None should be treated as empty,
        # not crash on .strip().
        agent = _make_agent([None, "real reply"])  # type: ignore[list-item]
        prompts = _make_failure_prompts()

        result = call_failure_explanation("comment", agent=agent, prompts=prompts)

        assert result.reply_text == "real reply"

    def test_uses_retry_on_preempt(self) -> None:
        """Failure-explanation retry loop must also pass
        ``retry_on_preempt=True`` (#1687) — otherwise three consecutive
        cancellations would arrive here as empty strings, exhaust the
        retry budget, and raise ``SynthesisExhaustedError`` even though
        the model never produced a real failure."""
        agent = _make_agent("Sorry, please try again.")
        prompts = _make_failure_prompts()

        call_failure_explanation("comment", agent=agent, prompts=prompts)

        _, kwargs = agent.run_turn.call_args_list[0]
        assert kwargs.get("retry_on_preempt") is True


# ---------------------------------------------------------------------------
# call_synthesis — LLM verification turn
# ---------------------------------------------------------------------------


class TestCallSynthesisVerificationTurn:
    """Tests for the LLM verification turn wired into call_synthesis (fixes #1218).

    After a successful synthesis parse with ``change_request=None``, a
    brief yes/no turn asks the model whether it recorded every request.
    A "No" answer triggers a follow-up derive turn that populates
    ``change_request`` and promotes the response to ACT.
    """

    def test_verify_yes_no_promotion(self) -> None:
        """When verify says Yes, change_request stays None."""
        raw = _make_raw(reply_text="Looks fine as-is.", change_request=None)
        # HOL-15: critic turn between synthesis and verify.
        agent = _make_agent([raw, _CRITIC_PASS, "Yes"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request is None
        # synthesis + critic + verify
        assert agent.run_turn.call_count == 3

    def test_verify_no_derives_change_request(self) -> None:
        """When verify says No, the derive turn populates change_request."""
        raw = _make_raw(reply_text="This looks fine.", change_request=None)
        agent = _make_agent([raw, _CRITIC_PASS, "No", "Update the test coverage"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request == "Update the test coverage"
        assert result.reply_text == "This looks fine."
        # synthesis + critic + verify + derive
        assert agent.run_turn.call_count == 4

    def test_verify_no_preserves_reply_text(self) -> None:
        """Promotion via verify must not alter reply_text."""
        raw = _make_raw(reply_text="Understood.", change_request=None)
        agent = _make_agent([raw, _CRITIC_PASS, "No", "Add missing tests"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Understood."

    def test_verify_skipped_when_change_request_set(self) -> None:
        """When change_request is already populated, verify turn is never
        called.  HOL-15 critic still fires (it gates EVERY response, not
        just change_request=None ones)."""
        raw = _make_raw(reply_text="Got it.", change_request="Fix the tests")
        agent = _make_agent(raw)
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request == "Fix the tests"
        # synthesis + critic — no verify (change_request already set)
        assert agent.run_turn.call_count == 2

    def test_verify_no_case_insensitive(self) -> None:
        """'NO', 'No.', 'no' etc. all trigger promotion."""
        raw = _make_raw(reply_text="Sure.", change_request=None)
        agent = _make_agent([raw, _CRITIC_PASS, "NO.", "Handle the edge case"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request == "Handle the edge case"

    def test_verify_no_with_trailing_text_triggers_promotion(self) -> None:
        """'No, I missed X' (starts with No) also triggers promotion."""
        raw = _make_raw(reply_text="Sure.", change_request=None)
        agent = _make_agent(
            [raw, _CRITIC_PASS, "No, I did not record it.", "Fix the linting"]
        )
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request == "Fix the linting"

    def test_verify_no_skips_promotion_when_derive_empty(self) -> None:
        """When the derive turn returns empty, no promotion — original returned."""
        raw = _make_raw(reply_text="This looks fine.", change_request=None)
        agent = _make_agent([raw, _CRITIC_PASS, "No", ""])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request is None
        # synthesis + critic + verify + derive (even though derive is empty)
        assert agent.run_turn.call_count == 4

    def test_verify_no_preserves_emoji_on_promotion(self) -> None:
        """Promotion via verify preserves the original emoji."""
        raw = _make_raw(reply_text="Got it.", emoji="rocket", change_request=None)
        agent = _make_agent([raw, _CRITIC_PASS, "No", "Add the missing test"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request == "Add the missing test"
        assert result.emoji == "rocket"

    def test_verify_no_preserves_insights_on_promotion(self) -> None:
        """Promotion via verify preserves the original insights list."""
        insight_data = [{"title": "T", "hook": "H.", "why": "W."}]
        raw = _make_raw(
            reply_text="Got it.", change_request=None, insights=insight_data
        )
        agent = _make_agent([raw, _CRITIC_PASS, "No", "Add the missing test"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request == "Add the missing test"
        assert len(result.insights) == 1
        assert result.insights[0].title == "T"

    def test_verify_no_preserves_reasoning_on_promotion(self) -> None:
        """Promotion via verify preserves the original reasoning."""
        raw = _make_raw(
            reasoning="my private chain-of-thought",
            reply_text="Looks good.",
            change_request=None,
        )
        agent = _make_agent([raw, "No", "Fix the thing"])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reasoning == "my private chain-of-thought"

    def test_synthesis_turn_uses_retry_on_preempt(self) -> None:
        """Main synthesis turn passes ``retry_on_preempt=True`` so a
        cancelled/preempted turn is transparently re-tried by the agent
        instead of arriving here as an empty string and burning a
        parse-failure retry slot — three consecutive preemptions used
        to trip the failure-explanation fallback even though the model
        never got a chance to answer (#1687)."""
        raw = _make_raw(reply_text="Looks fine.", change_request=None)
        agent = _make_agent([raw, "Yes"])
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # First call is the synthesis turn itself.
        _, kwargs = agent.run_turn.call_args_list[0]
        assert kwargs.get("retry_on_preempt") is True

    def test_verify_turn_uses_retry_on_preempt(self) -> None:
        """Verification turn (#1644 self-verification guard) is also
        on the synthesis retry path — same preempt-eaten-as-empty bug
        applies if not protected (#1687)."""
        raw = _make_raw(reply_text="Looks fine.", change_request=None)
        agent = _make_agent([raw, "Yes"])
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        _, kwargs = agent.run_turn.call_args_list[1]
        assert kwargs.get("retry_on_preempt") is True

    def test_derive_turn_uses_retry_on_preempt(self) -> None:
        """Derive turn (after a "No" verify) — same protection (#1687)."""
        raw = _make_raw(reply_text="Looks fine.", change_request=None)
        agent = _make_agent([raw, "No", "Add missing tests"])
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        _, kwargs = agent.run_turn.call_args_list[2]
        assert kwargs.get("retry_on_preempt") is True

    def test_verify_turn_uses_read_only_allowed_tools(self) -> None:
        """Verification turns must pass READ_ONLY_ALLOWED_TOOLS, not None."""
        from fido.provider import READ_ONLY_ALLOWED_TOOLS

        raw = _make_raw(reply_text="Looks fine.", change_request=None)
        agent = _make_agent([raw, "Yes"])
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # The second call is the verification turn.
        _, kwargs = agent.run_turn.call_args_list[1]
        assert kwargs.get("allowed_tools") == READ_ONLY_ALLOWED_TOOLS

    def test_derive_turn_uses_read_only_allowed_tools(self) -> None:
        """Derive turn (after No) must pass READ_ONLY_ALLOWED_TOOLS, not None."""
        from fido.provider import READ_ONLY_ALLOWED_TOOLS

        raw = _make_raw(reply_text="Looks fine.", change_request=None)
        agent = _make_agent([raw, "No", "Add missing tests"])
        prompts = _make_prompts()

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # Third call is the derive turn.
        _, kwargs = agent.run_turn.call_args_list[2]
        assert kwargs.get("allowed_tools") == READ_ONLY_ALLOWED_TOOLS

    def test_verify_turn_carries_followup_system_prompt(self) -> None:
        """#1850: the verify turn must pass the *follow-up* synthesis
        system prompt, not the main one.  The worker's persistent
        ClaudeSession has no anchor across turns — a bare Yes/No prompt
        against task framing is read as task continuation, and the agent
        goes off running unrelated tools (observed on PR #1842, 84-second
        turn before the reply landed).  The follow-up variant strips the
        JSON-only directive so ``startswith("no")`` actually works
        (codex P1).

        Call order is now synth → critic → verify after HOL-15; verify
        sits at index 2."""
        raw = _make_raw(reply_text="Looks fine.", change_request=None)
        agent = _make_agent([raw, _CRITIC_PASS, "Yes"])
        prompts = _make_prompts(
            system="SYNTHESIS_JSON_ONLY",
            followup_system="SYNTHESIS_FOLLOWUP",
            critic_system="SYNTHESIS_CRITIC",
        )

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # Idx 0: main synthesis with the JSON-only system prompt.
        _, synth_kwargs = agent.run_turn.call_args_list[0]
        assert synth_kwargs.get("system_prompt") == "SYNTHESIS_JSON_ONLY"
        # Idx 1: HOL-15 critic with its own JSON-capable prompt.
        _, critic_kwargs = agent.run_turn.call_args_list[1]
        assert critic_kwargs.get("system_prompt") == "SYNTHESIS_CRITIC"
        # Idx 2: legacy verify with the plain-text follow-up prompt.
        _, verify_kwargs = agent.run_turn.call_args_list[2]
        assert verify_kwargs.get("system_prompt") == "SYNTHESIS_FOLLOWUP"

    def test_derive_turn_carries_followup_system_prompt(self) -> None:
        """#1850: the derive turn (after a No verify) must also carry the
        follow-up synthesis system prompt — same plain-text framing.

        Call order is synth → critic → verify (No) → derive; derive
        sits at index 3 after HOL-15."""
        raw = _make_raw(reply_text="Looks fine.", change_request=None)
        agent = _make_agent([raw, _CRITIC_PASS, "No", "Add missing tests"])
        prompts = _make_prompts(
            system="SYNTHESIS_JSON_ONLY",
            followup_system="SYNTHESIS_FOLLOWUP",
            critic_system="SYNTHESIS_CRITIC",
        )

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        _, derive_kwargs = agent.run_turn.call_args_list[3]
        assert derive_kwargs.get("system_prompt") == "SYNTHESIS_FOLLOWUP"

    def test_verify_turn_exception_returns_original_response(self) -> None:
        """A transport error in the verify turn must not discard the synthesis result."""
        raw = _make_raw(reply_text="Original reply.", change_request=None)
        agent = _make_agent([raw])
        agent.run_turn.side_effect = [
            raw,
            _CRITIC_PASS,
            RuntimeError("transport error"),
        ]
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Original reply."
        assert result.change_request is None
        # synthesis + critic + verify (which raises and is swallowed)
        assert agent.run_turn.call_count == 3

    def test_derive_turn_exception_returns_original_response(self) -> None:
        """A transport error in the derive turn must not discard the synthesis result."""
        raw = _make_raw(reply_text="Original reply.", change_request=None)
        agent = _make_agent([raw])
        agent.run_turn.side_effect = [
            raw,
            "No",
            RuntimeError("derive transport error"),
        ]
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Original reply."
        assert result.change_request is None
        assert agent.run_turn.call_count == 3

    def test_context_overflow_error_propagates_from_verify_turn(self) -> None:
        """ContextOverflowError in verify must propagate — not be swallowed."""
        from fido.provider import ContextOverflowError

        raw = _make_raw(reply_text="Reply.", change_request=None)
        agent = _make_agent([raw])
        agent.run_turn.side_effect = [raw, ContextOverflowError("overflow")]
        prompts = _make_prompts()

        with pytest.raises(ContextOverflowError):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

    def test_session_leak_error_propagates_from_verify_turn(self) -> None:
        """SessionLeakError in verify must propagate — not be swallowed."""
        from fido.provider import SessionLeakError

        raw = _make_raw(reply_text="Reply.", change_request=None)
        agent = _make_agent([raw])
        agent.run_turn.side_effect = [raw, SessionLeakError("leak")]
        prompts = _make_prompts()

        with pytest.raises(SessionLeakError):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

    def test_context_overflow_error_propagates_from_derive_turn(self) -> None:
        """ContextOverflowError in derive must propagate — not be swallowed."""
        from fido.provider import ContextOverflowError

        raw = _make_raw(reply_text="Reply.", change_request=None)
        agent = _make_agent([raw])
        agent.run_turn.side_effect = [raw, "No", ContextOverflowError("overflow")]
        prompts = _make_prompts()

        with pytest.raises(ContextOverflowError):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

    def test_session_leak_error_propagates_from_derive_turn(self) -> None:
        """SessionLeakError in derive must propagate — not be swallowed."""
        from fido.provider import SessionLeakError

        raw = _make_raw(reply_text="Reply.", change_request=None)
        agent = _make_agent([raw])
        agent.run_turn.side_effect = [raw, "No", SessionLeakError("leak")]
        prompts = _make_prompts()

        with pytest.raises(SessionLeakError):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)


# ---------------------------------------------------------------------------
# HOL-15 / #1909 — intent-coverage critic at triage
# ---------------------------------------------------------------------------


class TestIntentCoverageCritic:
    """End-to-end behaviour of the HOL-15 intent-coverage gate wired
    into ``call_synthesis``.  The gate fires for every parsed response
    and can demand a synthesis retry with a specific gap."""

    def test_passing_critic_does_not_change_count(self) -> None:
        """A clean ``{"passed": true}`` verdict short-circuits the
        critic — no retry, same synthesis result returned."""
        raw = _make_raw(reply_text="Done.", change_request="Fix the test")
        agent = _make_agent([raw, _CRITIC_PASS])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Done."
        # synthesis + critic — no verify (change_request set), no retry
        assert agent.run_turn.call_count == 2

    def test_failing_critic_drives_retry_with_gap_in_nudge(self) -> None:
        """When the critic fails, the next synthesis attempt's user
        prompt MUST include the gap text so Opus addresses the specific
        complaint — generic JSON-strict nudges don't help when the
        problem is coverage, not parse shape.  This is the #1862 fix
        path: the first response promised X but didn't queue it; the
        critic catches the mismatch and the retry has the specific
        gap to address."""
        raw_v1 = _make_raw(reply_text="I'll add tests.", change_request=None)
        raw_v2 = _make_raw(
            reply_text="Tests added.", change_request="Add the missing tests"
        )
        agent = _make_agent(
            [
                raw_v1,
                _critic_fail("prose promises tests but no change_request queued"),
                raw_v2,
                _CRITIC_PASS,
            ]
        )
        prompts = _make_prompts(user="USER-PROMPT")

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.change_request == "Add the missing tests"
        # synthesis-v1 + critic-fail + synthesis-v2 + critic-pass = 4
        assert agent.run_turn.call_count == 4
        # The second synthesis turn's user prompt must carry the gap text.
        retry_args, _ = agent.run_turn.call_args_list[2]
        assert "prose promises tests but no change_request queued" in retry_args[0]
        assert "critic" in retry_args[0].lower()

    def test_critic_exhaustion_raises_synthesis_exhausted_error(self) -> None:
        """When MAX_RETRIES critic-fail/synthesis cycles all fail, the
        outer call raises ``SynthesisExhaustedError`` so the executor
        can route to the failure-explanation fallback (Constraint B:
        never silently default to empty)."""
        raw = _make_raw(reply_text="Promise.", change_request=None)
        # Every attempt: synth → critic-fail.  3 attempts × 2 turns = 6 calls.
        side_effects = []
        for _ in range(MAX_RETRIES):
            side_effects.extend([raw, _critic_fail("still wrong")])
        agent = _make_agent(side_effects)
        prompts = _make_prompts()

        with pytest.raises(SynthesisExhaustedError, match="intent-coverage critic"):
            call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)
        assert agent.run_turn.call_count == 2 * MAX_RETRIES

    def test_critic_transport_failure_fails_open(self) -> None:
        """A transport error in the critic turn must not block a valid
        synthesis response — the critic is an additional gate, not the
        only one.  Critic exception → fail open → response shipped."""
        raw = _make_raw(reply_text="Done.", change_request="Fix it")
        agent = _make_agent([raw])
        agent.run_turn.side_effect = [
            raw,
            RuntimeError("critic transport error"),
        ]
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Done."
        assert agent.run_turn.call_count == 2

    def test_critic_malformed_response_fails_open(self) -> None:
        """A critic response that doesn't parse as a verdict envelope
        (no ``passed`` field, or non-bool ``passed``) must fail open
        rather than blocking a valid synthesis response."""
        raw = _make_raw(reply_text="Done.", change_request="Fix it")
        agent = _make_agent(
            [
                raw,
                '{"verdict": "yes"}',  # not the right schema
            ]
        )
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Done."

    def test_critic_failure_without_gap_fails_open(self) -> None:
        """A ``{"passed": false}`` verdict with no ``gap`` field can't
        drive a meaningful retry — there's no specific complaint to
        nudge with.  Treat as fail-open to avoid useless retries."""
        raw = _make_raw(reply_text="Done.", change_request="Fix it")
        agent = _make_agent([raw, '{"passed": false}'])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Done."

    def test_critic_scans_past_leading_unrelated_json(self) -> None:
        """codex r3293359040 on PR #1932: a response that leads with an
        unrelated JSON object before the real verdict envelope must
        still pick up the verdict.  Before the fix, only the first
        decoded object was checked — a leading ``{}`` would mask a
        following ``{"passed": false, "gap": "..."}`` and the
        intent-coverage failure would silently ship."""
        raw_v1 = _make_raw(reply_text="Promise.", change_request=None)
        raw_v2 = _make_raw(reply_text="Tests added.", change_request="Add tests")
        critic_with_leading_noise = '{} {"passed": false, "gap": "missing tests"}'
        agent = _make_agent([raw_v1, critic_with_leading_noise, raw_v2, _CRITIC_PASS])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # Critic verdict was picked up — synthesis retried (4 turns
        # total: v1 + critic-fail + v2 + critic-pass).
        assert result.change_request == "Add tests"
        assert agent.run_turn.call_count == 4

    def test_reply_prose_critic_fires_when_grounding_state_provided(
        self,
    ) -> None:
        """HOL-18 / #1912: when the caller passes
        ``claim_grounding_state``, the reply-prose critic runs after
        intent-coverage and gates the response on grounded claims.
        Order in the loop: synthesis → intent-coverage critic →
        reply-prose critic → verify."""
        raw_v1 = _make_raw(
            reply_text="Fixed in commit deadbeef.",
            change_request="Fix it",
        )
        raw_v2 = _make_raw(
            reply_text="Fixed in commit abc1234.",
            change_request="Fix it",
        )
        prose_fail = json.dumps(
            {
                "passed": False,
                "gap": "commit deadbeef not in recent_commit_shas",
                "rationale": "checked git log",
            }
        )
        prose_pass = json.dumps({"passed": True, "rationale": "grounded"})
        # synth-v1 + intent-pass + prose-fail + synth-v2 + intent-pass +
        # prose-pass = 6 turns (no verify since change_request is set).
        agent = _make_agent(
            [raw_v1, _CRITIC_PASS, prose_fail, raw_v2, _CRITIC_PASS, prose_pass]
        )
        prompts = _make_prompts()
        # Also stub the reply-prose prompt method on the MagicMock
        # so the critic's prompt-build doesn't blow up.
        prompts.reply_prose_claim_grounding_prompt.return_value = "reply-prose-prompt"

        result = call_synthesis(
            "comment",
            is_bot=False,
            agent=agent,
            prompts=prompts,
            claim_grounding_state={"recent_commit_shas": ["abc1234"]},
        )

        # Real grounded reply shipped.
        assert "abc1234" in result.reply_text
        assert agent.run_turn.call_count == 6

    def test_reply_prose_critic_skipped_when_grounding_state_empty(
        self,
    ) -> None:
        """When no ground-truth state is available, the prose critic
        must not fire — the legacy single-critic flow ships unchanged.
        Verifies the default-None / empty-dict back-compat."""
        raw = _make_raw(reply_text="Done.", change_request="Fix it")
        agent = _make_agent([raw, _CRITIC_PASS])
        prompts = _make_prompts()

        call_synthesis(
            "comment",
            is_bot=False,
            agent=agent,
            prompts=prompts,
            claim_grounding_state=None,
        )

        # Only synth + intent-coverage critic ran (no prose critic, no
        # verify since change_request is set).
        assert agent.run_turn.call_count == 2

    def test_critic_recovers_real_fail_after_malformed_early_envelope(
        self,
    ) -> None:
        """codex r3293424368 on PR #1932: if the critic emits
        ``{"passed": false}`` with no gap followed by a real
        ``{"passed": false, "gap": "..."}``, we must use the real
        verdict instead of failing open on the first malformed one.
        Otherwise an early malformed envelope masks a legitimate
        intent-coverage failure and lets it ship."""
        raw_v1 = _make_raw(reply_text="Promise.", change_request=None)
        raw_v2 = _make_raw(reply_text="Tests added.", change_request="Add tests")
        critic_response = '{"passed": false} {"passed": false, "gap": "missing tests"}'
        agent = _make_agent([raw_v1, critic_response, raw_v2, _CRITIC_PASS])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # Real verdict was picked up — synthesis retried.
        assert result.change_request == "Add tests"
        assert agent.run_turn.call_count == 4

    def test_critic_scans_past_leading_unrelated_json_pass_case(self) -> None:
        """Symmetric to the fail-case test: a leading unrelated object
        followed by ``{"passed": true}`` must still resolve as a pass
        (not fail-open through ignorance)."""
        raw = _make_raw(reply_text="Done.", change_request="Fix it")
        agent = _make_agent([raw, '{} {"passed": true}'])
        prompts = _make_prompts()

        result = call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        assert result.reply_text == "Done."
        # synthesis + critic only — no retry, no verify (change_request set).
        assert agent.run_turn.call_count == 2

    def test_critic_uses_critic_system_prompt(self) -> None:
        """Critic turn must run with the JSON-capable critic system
        prompt, NOT the follow-up plain-text prompt — the latter tells
        the model "no JSON", which silently disables the gate (codex
        r3293399801 on PR #1932)."""
        raw = _make_raw(reply_text="Done.", change_request="Fix it")
        agent = _make_agent([raw, _CRITIC_PASS])
        prompts = _make_prompts(
            system="MAIN-SYS",
            followup_system="FOLLOWUP-SYS",
            critic_system="CRITIC-SYS",
        )

        call_synthesis("comment", is_bot=False, agent=agent, prompts=prompts)

        # First call: main synthesis with the main system prompt.
        _, synth_kw = agent.run_turn.call_args_list[0]
        assert synth_kw["system_prompt"] == "MAIN-SYS"
        # Second call: critic with the JSON-capable critic prompt.
        _, critic_kw = agent.run_turn.call_args_list[1]
        assert critic_kw["system_prompt"] == "CRITIC-SYS"


# ---------------------------------------------------------------------------
# HOL-14 / #1908 — critic_loop helper
# ---------------------------------------------------------------------------


class TestCriticVerdict:
    """Constructor invariant: a failing verdict must carry a non-empty
    gap.  The gap is the nudge the next generate() attempt needs to
    address — losing it defeats the loop."""

    def test_pass_requires_no_gap(self) -> None:
        v = CriticVerdict(passed=True)
        assert v.passed
        assert v.gap == ""

    def test_pass_with_gap_allowed(self) -> None:
        v = CriticVerdict(passed=True, gap="optional context")
        assert v.passed
        assert v.gap == "optional context"

    def test_fail_requires_non_empty_gap(self) -> None:
        with pytest.raises(ValueError, match="gap is required"):
            CriticVerdict(passed=False)

    def test_fail_rejects_empty_gap(self) -> None:
        with pytest.raises(ValueError, match="gap is required"):
            CriticVerdict(passed=False, gap="")

    def test_fail_rejects_whitespace_gap(self) -> None:
        with pytest.raises(ValueError, match="gap is required"):
            CriticVerdict(passed=False, gap="   \n\t  ")


class TestCriticLoop:
    """Behavioural contract of :func:`critic_loop` — the shape HOL-15..HOL-19
    depend on.  Tests use hand-rolled lambdas / fakes per project rule."""

    def test_returns_first_passing_candidate(self) -> None:
        attempts: list[tuple[int, str]] = []

        def generate(attempt: int, gap: str) -> str:
            attempts.append((attempt, gap))
            return "candidate"

        def verify(_c: str) -> CriticVerdict:
            return CriticVerdict(passed=True)

        result = critic_loop(generate, verify, label="t")
        assert result == "candidate"
        assert attempts == [(0, "")]

    def test_loops_with_gap_until_passing(self) -> None:
        gap_seen: list[str] = []

        def generate(attempt: int, gap: str) -> str:
            gap_seen.append(gap)
            return f"v{attempt}"

        def verify(candidate: str) -> CriticVerdict:
            if candidate == "v0":
                return CriticVerdict(passed=False, gap="missing X")
            if candidate == "v1":
                return CriticVerdict(passed=False, gap="missing Y")
            return CriticVerdict(passed=True)

        result = critic_loop(generate, verify, label="t", max_attempts=5)
        assert result == "v2"
        assert gap_seen == ["", "missing X", "missing Y"]

    def test_raises_critic_exhausted_after_max_attempts(self) -> None:
        def generate(attempt: int, gap: str) -> str:
            return f"v{attempt}"

        def verify(c: str) -> CriticVerdict:
            return CriticVerdict(passed=False, gap=f"never-passes ({c})")

        with pytest.raises(CriticExhaustedError) as excinfo:
            critic_loop(generate, verify, label="my-emission", max_attempts=3)
        err = excinfo.value
        assert err.label == "my-emission"
        # Every gap is preserved so HOL-20/21 can attach full retry
        # history to the auto-filed bug.
        assert err.gaps == [
            "never-passes (v0)",
            "never-passes (v1)",
            "never-passes (v2)",
        ]

    def test_generate_exception_propagates(self) -> None:
        """The loop is for verification gaps, not generation errors.
        Generation exceptions (transport, parsing, context overflow)
        should bubble out untouched so the caller decides whether to
        retry or surface."""

        class _Boom(RuntimeError):
            pass

        def generate(attempt: int, gap: str) -> str:
            raise _Boom("transport")

        def verify(_c: str) -> CriticVerdict:
            return CriticVerdict(passed=True)

        with pytest.raises(_Boom, match="transport"):
            critic_loop(generate, verify, label="t")

    def test_default_max_attempts_uses_synthesis_constant(self) -> None:
        attempts: list[int] = []

        def generate(attempt: int, gap: str) -> str:
            attempts.append(attempt)
            return "x"

        def verify(_c: str) -> CriticVerdict:
            return CriticVerdict(passed=False, gap="nope")

        with pytest.raises(CriticExhaustedError):
            critic_loop(generate, verify, label="t")
        assert len(attempts) == MAX_RETRIES

    def test_label_in_exhausted_message(self) -> None:
        """HOL-20/21 routes by label, so the label has to make it onto
        the error path verbatim."""

        def generate(attempt: int, gap: str) -> str:
            return "x"

        def verify(_c: str) -> CriticVerdict:
            return CriticVerdict(passed=False, gap="g")

        with pytest.raises(CriticExhaustedError, match="my-special-label"):
            critic_loop(generate, verify, label="my-special-label", max_attempts=1)
