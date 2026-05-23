"""Synthesis LLM call — unified comment-handling turn.

Wraps the two-prompt synthesis exchange into a single typed call that
returns a :class:`~fido.synthesis.CommentResponse`.  Retries on
malformed JSON up to :data:`MAX_RETRIES` times with a stricter
instruction appended, then raises :class:`SynthesisExhaustedError`
(fail-closed per Constraint B: reply text is always required, never
defaulted).

After a successful parse, a brief verification turn asks the model
whether it recorded every request into ``change_request``.  A "No"
answer triggers a follow-up turn that derives the omitted
``change_request`` text and promotes the response to ACT, ensuring
prose promises always correspond to queued tasks (fixes #1218).
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from fido.prompts import Prompts
from fido.provider import (
    READ_ONLY_ALLOWED_TOOLS,
    ContextOverflowError,
    ProviderAgent,
    SessionLeakError,
)
from fido.synthesis import (
    VALID_REACTIONS,
    CommentResponse,
    Insight,
)
from fido.types import ActiveIssue, ActivePR

log = logging.getLogger(__name__)

#: Maximum number of synthesis LLM attempts before raising.
MAX_RETRIES: int = 3


# ---------------------------------------------------------------------------
# Critic loop helper (HOL-14 / #1908)
# ---------------------------------------------------------------------------
#
# The same shape — generate → verify → loop on gap → exhaust — repeats at
# every LLM emission point in the holistic-gate architecture (#1894 / Layer 2):
# intent registration, new-task creation, task completion, reply prose,
# insight filing.  HOL-15..HOL-19 each instantiate this with their own
# narrow verify question.  HOL-14 owns the helper; downstream leaves only
# supply ``generate`` and ``verify``.

_T = TypeVar("_T")


@dataclass(frozen=True)
class CriticVerdict:
    """One verdict from a critic over a proposed LLM emission.

    ``passed`` is the dispatch axis: ``True`` short-circuits the loop and
    returns the proposal; ``False`` records ``gap`` as a hint for the next
    ``generate`` attempt.

    ``gap`` is a one-line plain-English description of why the proposal
    failed — fed back into ``generate`` as the retry nudge so the next
    proposal can address the specific complaint.  Empty when ``passed``;
    required when ``not passed``.
    """

    passed: bool
    gap: str = ""

    def __post_init__(self) -> None:
        if not self.passed and not self.gap.strip():
            raise ValueError(
                "CriticVerdict.gap is required when passed=False — "
                "the gap is what the next generate() attempt needs to address"
            )


class CriticExhaustedError(Exception):
    """All :func:`critic_loop` attempts failed verification.

    Carries the chain of gaps so the caller can route the exhaustion to the
    appropriate sentinel (HOL-20/21: bug-and-block via the existing
    ``stuck-on-task`` BLOCKED path).
    """

    def __init__(self, label: str, gaps: list[str]) -> None:
        self.label = label
        self.gaps = list(gaps)
        msg = (
            f"critic_loop({label!r}) exhausted {len(gaps)} attempts — "
            f"final gap: {gaps[-1]!r}"
        )
        super().__init__(msg)


def critic_loop(
    generate: Callable[[int, str], _T],
    verify: Callable[[_T], CriticVerdict],
    *,
    label: str,
    max_attempts: int = MAX_RETRIES,
) -> _T:
    """Run the generate → verify → loop pattern over an LLM emission.

    ``generate(attempt, gap_so_far)`` produces a candidate of type ``_T``.
    On the first attempt ``gap_so_far`` is the empty string.  On later
    attempts it is the gap from the previous failed verdict, which the
    generator weaves into its prompt nudge so the next candidate addresses
    the specific complaint.

    ``verify(candidate)`` returns a :class:`CriticVerdict`.  ``passed=True``
    immediately returns the candidate; ``passed=False`` records the gap and
    loops.

    ``label`` is a short identifier (e.g. ``"intent-coverage"``,
    ``"task-completion"``) used in the exhaustion error and log lines so
    HOL-20/21 can route by emission point.

    Raises :class:`CriticExhaustedError` when ``max_attempts`` candidates
    all fail verification.  The error carries every gap so the bug-and-block
    routing can attach the full retry history to the auto-filed issue.

    ``generate`` is allowed to raise — its exception propagates out
    untouched.  Only verification gaps drive the loop.
    """
    gaps: list[str] = []
    gap_so_far = ""
    for attempt in range(max_attempts):
        candidate = generate(attempt, gap_so_far)
        verdict = verify(candidate)
        if verdict.passed:
            if attempt > 0:
                log.info(
                    "critic_loop[%s]: passed on attempt %d/%d",
                    label,
                    attempt + 1,
                    max_attempts,
                )
            return candidate
        gaps.append(verdict.gap)
        gap_so_far = verdict.gap
        log.warning(
            "critic_loop[%s]: attempt %d/%d failed — gap: %s",
            label,
            attempt + 1,
            max_attempts,
            verdict.gap,
        )
    raise CriticExhaustedError(label, gaps)


_RETRY_SUFFIX = (
    "\n\n---\n"
    "Your previous response was not valid JSON matching the required schema.  "
    "Respond with ONLY a JSON object — no preamble, no trailing text, no markdown "
    "code fences.  The reply_text field must be a non-empty string."
)


# ---------------------------------------------------------------------------
# LLM verification turn (fixes #1218)
# ---------------------------------------------------------------------------
#
# After the synthesis LLM produces a reply, a brief yes/no turn asks whether
# it recorded every request into ``change_request``.  If the answer is "No",
# a second short turn derives the missing description so we can promote the
# response to ACT and ensure a task is always queued behind any prose promise.

_VERIFY_CHANGE_REQUEST_PROMPT: str = (
    "Did you record every request from the previous reply into ``change_request`` "
    "(if any)?  Reply only with the single word Yes or No."
)

_DERIVE_CHANGE_REQUEST_PROMPT: str = (
    "You answered No.  In one concise sentence, state the request(s) you omitted from "
    "``change_request`` — no preamble, no trailing text."
)


def _check_and_promote(
    response: CommentResponse,
    agent: ProviderAgent,
    followup_system_prompt: str,
) -> CommentResponse:
    """Run a verification turn to detect unrecorded change requests.

    When ``response.change_request`` is already set, returns *response*
    unchanged — no extra turn is needed.

    Otherwise, asks the agent whether every request was recorded into
    ``change_request``.  If the agent answers "No" (case-insensitive,
    with or without trailing punctuation), a follow-up turn derives a
    concise ``change_request`` string and returns a new
    :class:`~fido.synthesis.CommentResponse` with that field populated.

    If the agent answers "Yes" (or anything other than "No"), or if the
    derive turn returns an empty string, *response* is returned unchanged.
    Any exception raised by either turn (transport errors, timeouts, etc.)
    is caught, logged, and treated as a non-fatal guard failure — *response*
    is returned unchanged so a valid synthesis result is never discarded.

    This enforces the invariant: prose promises must correspond to queued
    tasks (fixes #1218).

    *followup_system_prompt* (#1850) anchors both turns in synthesis-reply
    mode.  Without it, the bare Yes/No prompt arrives in the worker's
    persistent session against whatever task framing the prior turn left
    behind — the agent reads it as task continuation and goes off running
    unrelated tools.  It must be the *follow-up* variant
    (:meth:`fido.prompts.Prompts.synthesis_followup_system_prompt`), not
    the main synthesis system prompt — the latter demands a JSON-only
    reply which would break the ``startswith("no")`` check (codex P1
    on #1851).
    """
    if response.change_request is not None:
        return response

    try:
        verify_raw = agent.run_turn(
            _VERIFY_CHANGE_REQUEST_PROMPT,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
            system_prompt=followup_system_prompt,
            retry_on_preempt=True,
        )
        if not (verify_raw or "").strip().lower().startswith("no"):
            return response

        log.warning(
            "synthesis guard: model indicated unrecorded request — "
            "deriving change_request via follow-up turn"
        )
        derived_raw = agent.run_turn(
            _DERIVE_CHANGE_REQUEST_PROMPT,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
            system_prompt=followup_system_prompt,
            retry_on_preempt=True,
        )
        derived = (derived_raw or "").strip()
        if not derived:
            log.warning(
                "synthesis guard: follow-up turn returned empty — skipping promotion"
            )
            return response

        return CommentResponse(
            reasoning=response.reasoning,
            reply_text=response.reply_text,
            emoji=response.emoji,
            change_request=derived,
            insights=response.insights,
        )
    except ContextOverflowError, SessionLeakError:
        raise
    except Exception as exc:
        log.warning(
            "synthesis guard: verification turn failed (%s) — "
            "returning original response unchanged",
            exc,
        )
        return response


class SynthesisExhaustedError(Exception):
    """All synthesis retries exhausted without a valid :class:`~fido.synthesis.CommentResponse`.

    Signals a Constraint B violation: the synthesis call never defaults or
    returns an empty reply — it either succeeds or fails loudly.
    """


# ---------------------------------------------------------------------------
# Intent-coverage critic (HOL-15 / #1909)
# ---------------------------------------------------------------------------
#
# After a synthesis response parses, this critic asks the model to verify
# that the registered intents (``change_request``) + reply prose faithfully
# cover the original comment.  Catches:
#
#   - missing : prose promises work the change_request doesn't capture
#               (#1862's shape — closes that bug).
#   - invented: prose claims to do something the comment didn't ask for.
#   - mismatched: change_request scope doesn't match the comment's ask.
#
# The bare Yes/No verification turn (``_check_and_promote`` / #1218) only
# catches "missing change_request"; this broader critic covers all three
# axes.  It runs ADDITIVELY today — both gates fire per attempt.  A future
# leaf removes ``_check_and_promote`` once the critic has earned its keep.

_CRITIC_RETRY_SUFFIX_TEMPLATE = (
    "\n\n---\n"
    "Your previous response failed the intent-coverage critic with this "
    "gap:\n\n"
    "  {gap}\n\n"
    "Rewrite the synthesis response to address the gap.  Same JSON schema "
    "as before."
)


def _run_intent_coverage_critic(
    response: CommentResponse,
    comment_body: str,
    agent: ProviderAgent,
    prompts: Prompts,
    critic_system_prompt: str,
) -> CriticVerdict:
    """Ask the model whether ``response`` faithfully covers ``comment_body``.

    Returns ``CriticVerdict(passed=True)`` on a clean pass, or
    ``CriticVerdict(passed=False, gap=...)`` when the critic spots a
    coverage problem.  Parse failures or unrecognised JSON from the critic
    fail OPEN (return ``passed=True``) — the critic is an additional gate,
    not the only one, so a flaky critic turn must not block a valid
    response from shipping.  ``ContextOverflowError`` / ``SessionLeakError``
    still propagate per project convention.

    ``critic_system_prompt`` must be the JSON-capable variant
    (:meth:`fido.prompts.Prompts.critic_system_prompt`) — passing the
    plain-text follow-up prompt makes the model emit non-envelope
    responses that ``extract_json_objects`` can't see, silently failing
    open (codex r3293399801 on PR #1932).
    """
    prompt = prompts.intent_coverage_critic_prompt(
        comment_body=comment_body,
        reply_text=response.reply_text,
        change_request=response.change_request,
    )
    try:
        raw = agent.run_turn(
            prompt,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
            system_prompt=critic_system_prompt,
            retry_on_preempt=True,
        )
    except ContextOverflowError, SessionLeakError:
        raise
    except Exception as exc:
        log.warning(
            "intent-coverage critic transport failure (%s) — failing open",
            exc,
        )
        return CriticVerdict(passed=True)

    objs = extract_json_objects(raw or "")
    if not objs:
        log.warning("intent-coverage critic returned no parseable JSON — failing open")
        return CriticVerdict(passed=True)
    # codex r3293359040 on PR #1932: scan ALL extracted JSON objects for
    # the first one that matches the verdict envelope.  A response that
    # leads with an unrelated ``{}`` followed by a real verdict would
    # otherwise be treated as malformed and fail open — silently letting
    # an intent-coverage failure ship.
    # codex r3293424368 on PR #1932: a malformed early envelope
    # (e.g. ``{"passed": false}`` without a gap) must NOT short-circuit
    # the scan — a later object may carry the real verdict with its
    # gap.  Track the first malformed-fail we saw; only return its
    # fail-open verdict after the whole scan finds no usable envelope.
    saw_malformed_fail = False
    for obj in objs:
        if obj.get("passed") is True:
            return CriticVerdict(passed=True)
        if obj.get("passed") is False:
            gap = obj.get("gap")
            if isinstance(gap, str) and gap.strip():
                return CriticVerdict(passed=False, gap=gap.strip())
            # Critic claimed fail but gave no gap — remember and keep
            # scanning in case a later object has the real verdict.
            saw_malformed_fail = True
    if saw_malformed_fail:
        log.warning("intent-coverage critic failed without a gap — failing open")
    else:
        log.warning(
            "intent-coverage critic returned no envelope-shaped JSON in %d "
            "objects — failing open",
            len(objs),
        )
    return CriticVerdict(passed=True)


def extract_json_objects(raw: str) -> list[dict[str, Any]]:
    """Return all JSON objects found in *raw* using a consume loop.

    Scans *raw* for ``{``, attempts :meth:`json.JSONDecoder.raw_decode`
    from that position, advances past the decoded span on success, or
    skips the character and continues on failure.  Returns a list of
    every successfully decoded dict in order of appearance.

    This is more robust than a ``first-{-to-last-}`` span heuristic: it
    handles preamble prose, trailing explanation text, stray braces, and
    nested objects correctly because the decoder itself determines the
    exact end of each JSON document.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    pos = 0
    while pos < len(raw):
        brace = raw.find("{", pos)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(raw, brace)
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        pos = end
    return objects


def _parse_comment_response(raw: str) -> CommentResponse:
    """Parse *raw* model output into a :class:`~fido.synthesis.CommentResponse`.

    Extracts all JSON objects from *raw* via :func:`extract_json_objects`
    and returns the first that validates as a ``CommentResponse``.
    Raises :exc:`ValueError` if none does; the caller
    (:func:`call_synthesis`) catches this and retries.
    """
    last_error: Exception = ValueError("no JSON objects found in model output")
    for obj in extract_json_objects(raw):
        reasoning = obj.get("reasoning", "")
        reply_text = obj.get("reply_text", "")

        if not isinstance(reply_text, str) or not reply_text.strip():
            last_error = ValueError(
                f"reply_text absent or empty (Constraint B): {reply_text!r}"
            )
            continue

        # Parse optional emoji — invalid shortcodes are warned and dropped.
        emoji_raw = obj.get("emoji")
        emoji: str | None = None
        if isinstance(emoji_raw, str) and emoji_raw:
            if emoji_raw in VALID_REACTIONS:
                emoji = emoji_raw
            else:
                log.warning(
                    "synthesis: invalid reaction shortcode %r — dropping", emoji_raw
                )

        # Parse optional change_request — must be a non-empty string or null.
        change_request_raw = obj.get("change_request")
        change_request: str | None = None
        if isinstance(change_request_raw, str) and change_request_raw.strip():
            change_request = change_request_raw

        # Parse optional insights list — each entry must have title, hook, why.
        insights: list[Insight] = []
        insights_raw = obj.get("insights")
        if isinstance(insights_raw, list):
            for entry in insights_raw:
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title", "")
                hook = entry.get("hook", "")
                why = entry.get("why", "")
                if (
                    isinstance(title, str)
                    and title.strip()
                    and isinstance(hook, str)
                    and hook.strip()
                    and isinstance(why, str)
                    and why.strip()
                ):
                    insights.append(Insight(title=title, hook=hook, why=why))
                else:
                    log.warning("synthesis: dropping malformed insight entry %r", entry)

        try:
            return CommentResponse(
                reasoning=str(reasoning),
                reply_text=reply_text,
                emoji=emoji,
                change_request=change_request,
                insights=insights,
            )
        except ValueError as exc:  # pragma: no cover - defensive
            # All ``CommentResponse`` invariants (non-empty reply_text,
            # valid emoji shortcode, non-empty change_request when set) are
            # pre-checked above, so construction here cannot fail through
            # the normal parser path.  Kept as a defensive catch in case
            # the dataclass adds new validation that the parser hasn't
            # learned about yet.
            last_error = exc
            continue

    raise ValueError(
        f"failed to parse CommentResponse from model output: {last_error!r}"
        f"\nraw output: {raw!r}"
    )


def call_synthesis(
    comment_body: str,
    *,
    is_bot: bool,
    context: dict[str, Any] | None = None,
    issue: ActiveIssue | None = None,
    pr: ActivePR | None = None,
    agent: ProviderAgent,
    prompts: Prompts,
) -> CommentResponse:
    """Run the unified comment-handling synthesis turn.

    Makes up to :data:`MAX_RETRIES` LLM calls.  The first attempt uses
    the base prompt; each subsequent attempt appends a stricter JSON-output
    instruction.  After a successful parse, a brief verification turn checks
    whether every request was recorded into ``change_request``; a "No"
    answer triggers a follow-up derive turn and promotes the response to ACT.
    Raises :class:`SynthesisExhaustedError` if all synthesis attempts fail
    (Constraint B: reply text is always required, never silently defaulted).

    Parameters
    ----------
    comment_body:
        The text of the PR comment to respond to.
    is_bot:
        Whether the comment came from an automated tool (adjusts voice
        guidance in the prompt).
    context:
        Optional triage context dict (e.g. ``{"pr_title": ...}``) passed
        to the prompt builder.
    issue:
        Active issue, injected into the system prompt for ground-truth
        context.
    pr:
        Active PR, injected into the system prompt alongside *issue*.
    agent:
        LLM agent with a ``run_turn(content, *, system_prompt)`` method.
    prompts:
        Prompt builder (``Prompts`` instance).
    """
    system_prompt = prompts.synthesis_system_prompt(issue=issue, pr=pr)
    followup_system_prompt = prompts.synthesis_followup_system_prompt(
        issue=issue, pr=pr
    )
    # HOL-15 critic needs a JSON-capable system prompt — the followup
    # prompt above explicitly forbids JSON (it's for plain-text Yes/No
    # turns), so wiring the critic through it makes the model emit
    # non-envelope responses that ``extract_json_objects`` can't see,
    # silently failing open (codex r3293399801 on PR #1932).
    critic_system_prompt = prompts.critic_system_prompt(issue=issue, pr=pr)
    base_user_prompt = prompts.synthesis_prompt(
        comment_body, is_bot=is_bot, context=context
    )

    last_error: Exception | None = None
    last_critic_gap: str | None = None
    for attempt in range(MAX_RETRIES):
        # Per-attempt prompt suffix.  Critic gap takes priority over the
        # generic JSON-strictness nudge — the critic gap is more specific
        # and addresses a successful-parse-but-bad-coverage problem.
        if last_critic_gap is not None:
            user_prompt = base_user_prompt + _CRITIC_RETRY_SUFFIX_TEMPLATE.format(
                gap=last_critic_gap
            )
        elif attempt > 0:
            user_prompt = base_user_prompt + _RETRY_SUFFIX
        else:
            user_prompt = base_user_prompt
        # ``retry_on_preempt=True`` (#1687): a preempted (cancelled)
        # turn returns an empty string from the drain.  Without this,
        # ``_parse_comment_response`` below would raise "no JSON
        # objects found", logged as "parse failure" — and three
        # preemptions in a row would burn the retry budget and trip
        # the failure-explanation fallback even though the model never
        # got a chance to answer.  ``session_agent`` re-runs cancelled
        # turns transparently, so the parse-failure branch only fires
        # for genuine model misbehavior.
        raw = agent.run_turn(
            user_prompt,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
            system_prompt=system_prompt,
            retry_on_preempt=True,
        )
        log.debug(
            "synthesis attempt %d/%d raw output: %r", attempt + 1, MAX_RETRIES, raw
        )

        try:
            response = _parse_comment_response(raw)
        except ValueError as exc:
            last_error = exc
            last_critic_gap = None
            log.warning(
                "synthesis attempt %d/%d parse failure — %s",
                attempt + 1,
                MAX_RETRIES,
                exc,
            )
            continue

        # HOL-15 / #1909: intent-coverage critic gate.  Fires for every
        # parsed response and can demand a retry with a specific gap.
        # Fails open on transport/parse problems (see
        # ``_run_intent_coverage_critic`` docstring) — the gate is
        # additional, not the only one, so a flaky critic must not
        # block a valid response from shipping.
        verdict = _run_intent_coverage_critic(
            response,
            comment_body=comment_body,
            agent=agent,
            prompts=prompts,
            critic_system_prompt=critic_system_prompt,
        )
        if not verdict.passed:
            last_error = ValueError(f"intent-coverage critic: {verdict.gap}")
            last_critic_gap = verdict.gap
            log.warning(
                "synthesis attempt %d/%d critic gap — %s",
                attempt + 1,
                MAX_RETRIES,
                verdict.gap,
            )
            continue

        if attempt > 0:
            log.info("synthesis: succeeded on attempt %d/%d", attempt + 1, MAX_RETRIES)
        # Legacy ``_check_and_promote`` (#1218) still fires as the
        # "missing change_request" backstop.  Removed in a follow-up
        # once the HOL-15 critic has earned its keep across enough
        # production turns to retire the bare Yes/No fallback.
        return _check_and_promote(
            response, agent, followup_system_prompt=followup_system_prompt
        )

    raise SynthesisExhaustedError(
        f"synthesis exhausted {MAX_RETRIES} retries without a valid CommentResponse "
        f"(Constraint B violation) — last error: {last_error}"
    )


_FAILURE_EXPLANATION_RETRY_SUFFIX = (
    "\n\n---\n"
    "Your previous response was empty.  Output the reply text now — plain prose only, "
    "no JSON, no markdown fences, no preamble.  At least one full sentence."
)


def call_failure_explanation(
    comment_body: str,
    *,
    agent: ProviderAgent,
    prompts: Prompts,
) -> CommentResponse:
    """Generate a fallback reply when :func:`call_synthesis` exhausted retries.

    Asks the LLM, via the same retry-with-nudge loop as :func:`call_synthesis`,
    to write a short reply acknowledging the failure and asking the commenter
    to rephrase.  Returns a :class:`CommentResponse` with only ``reply_text``
    populated — no emoji, no change_request, no insights — so the executor's
    success-path effects (post reply, clear eyes) handle it identically to a
    normal synthesis result.

    Raises :exc:`SynthesisExhaustedError` if the fallback also exhausts
    retries.  Caller is responsible for any further cleanup.
    """
    user_prompt = prompts.synthesis_failure_explanation_prompt(comment_body)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        suffix = _FAILURE_EXPLANATION_RETRY_SUFFIX if attempt > 0 else ""
        raw = agent.run_turn(
            user_prompt + suffix,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
            retry_on_preempt=True,
        )
        log.debug(
            "failure-explanation attempt %d/%d raw output: %r",
            attempt + 1,
            MAX_RETRIES,
            raw,
        )
        text = (raw or "").strip()
        if not text:
            last_error = ValueError("empty reply from failure-explanation turn")
            log.warning(
                "failure-explanation attempt %d/%d returned empty text — retrying",
                attempt + 1,
                MAX_RETRIES,
            )
            continue
        if attempt > 0:
            log.info(
                "failure-explanation: succeeded on attempt %d/%d",
                attempt + 1,
                MAX_RETRIES,
            )
        return CommentResponse(
            reasoning=(
                "(synthesis exhausted retries; this reply was generated by the "
                "fallback failure-explanation turn)"
            ),
            reply_text=text,
        )

    raise SynthesisExhaustedError(
        f"failure-explanation exhausted {MAX_RETRIES} retries without a usable reply "
        f"— last error: {last_error}"
    )
