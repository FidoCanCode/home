"""Layer 2 critics for the holistic-gate architecture (#1894).

One module per critic axis would proliferate; this module owns the
per-emission verdict types, JSON parsers, and runner helpers that
HOL-16..HOL-19 (and follow-ups) instantiate.  The generic
``critic_loop`` plumbing lives in ``fido.synthesis_call``; this module
holds the per-axis prompt callers + verdict shapes.

Each runner shares the same fail-open posture as
``_run_intent_coverage_critic`` (HOL-15): transport errors, malformed
verdicts, or unexpected response shapes log a warning and return a
pass-through verdict, so a flaky critic must not block valid LLM
output from shipping.
"""

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from fido.prompts import Prompts
from fido.provider import (
    READ_ONLY_ALLOWED_TOOLS,
    ContextOverflowError,
    ProviderAgent,
    SessionLeakError,
)
from fido.synthesis_call import extract_json_objects

log = logging.getLogger(__name__)

__all__ = [
    "InsightDedupTransportCounter",
    "InsightDedupVerdict",
    "ReplyProseVerdict",
    "TaskCompletionVerdict",
    "TaskCreationProposedSplit",
    "TaskCreationVerdict",
    "run_insight_dedup_critic",
    "run_reply_prose_critic",
    "run_task_completion_critic",
    "run_task_creation_critic",
]


# ---------------------------------------------------------------------------
# Task-creation critic (HOL-16 / #1910)
# ---------------------------------------------------------------------------
#
# Runs once per proposed ``new`` op in a rescope batch.  The critic gates
# the new task along two axes:
#
#   - Relationship to the existing queue:
#       * distinct       — genuinely new work
#       * duplicate_of   — covered by an existing task (drop the new task)
#       * supersedes     — replaces an existing task (drop the new task;
#                          a follow-up leaf will fold this into a rewrite
#                          op so the existing task is updated in place)
#
#   - Scope (HOL-12 contract):
#       * single — one statable invariant
#       * multi  — spans multiple invariants; fan out into proposed_splits
#
# Wiring in ``fido.tasks._apply_task_creation_critics`` walks
# ``ordered_items`` BEFORE materialisation and mutates the item list per
# verdict — distinct+single passes through, duplicate/supersedes drops
# the item, multi replaces the item with one new-op per split.


@dataclass(frozen=True)
class TaskCreationProposedSplit:
    """One child task in a ``multi``-scope verdict's ``proposed_splits``."""

    title: str
    description: str
    invariant: str


@dataclass(frozen=True)
class TaskCreationVerdict:
    """Critic verdict for one proposed ``new`` op.

    Default (no critic wired, or critic failed open) is "distinct + single,
    no rationale" so the proposed task passes through unchanged.

    ``relationship``:
      * ``"distinct"`` — keep the new task as-is.
      * ``"duplicate_of"`` — drop the new task; ``duplicate_of_id`` names
        the existing task whose scope already covers it.
      * ``"supersedes"`` — drop the new task; ``supersedes_id`` names the
        existing task it would replace.  A follow-up leaf turns this into
        a rewrite op so the existing task is updated in place.

    ``scope``:
      * ``"single"`` — one statable invariant, pass through.
      * ``"multi"`` — fan out into ``proposed_splits`` (each carrying its
        own ``title``/``description``/``invariant``).

    ``rationale`` is the critic's one-line explanation, suitable for log
    lines and (eventually) reply-back narrative.
    """

    relationship: Literal["distinct", "duplicate_of", "supersedes"] = "distinct"
    duplicate_of_id: str | None = None
    supersedes_id: str | None = None
    scope: Literal["single", "multi"] = "single"
    proposed_splits: tuple[TaskCreationProposedSplit, ...] = ()
    rationale: str = ""

    @property
    def drops_proposal(self) -> bool:
        """True when the verdict says drop the proposed new task."""
        return self.relationship in ("duplicate_of", "supersedes")

    @property
    def fans_out(self) -> bool:
        """True when the verdict says fan out into multiple children."""
        return self.scope == "multi" and bool(self.proposed_splits)

    def __post_init__(self) -> None:
        if self.relationship == "duplicate_of" and not self.duplicate_of_id:
            raise ValueError(
                "TaskCreationVerdict.duplicate_of_id is required when "
                "relationship='duplicate_of'"
            )
        if self.relationship == "supersedes" and not self.supersedes_id:
            raise ValueError(
                "TaskCreationVerdict.supersedes_id is required when "
                "relationship='supersedes'"
            )
        if self.scope == "multi" and not self.proposed_splits:
            raise ValueError(
                "TaskCreationVerdict.proposed_splits is required when scope='multi'"
            )


def _parse_proposed_splits(
    raw: object,
) -> tuple[TaskCreationProposedSplit, ...] | None:
    """Parse the ``proposed_splits`` array from a critic response.

    Returns ``None`` when the array is missing, malformed, or any entry
    fails validation.  An empty list returns an empty tuple (caller
    treats as no fan-out).
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        return None
    out: list[TaskCreationProposedSplit] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        title = item.get("title")
        description = item.get("description")
        invariant = item.get("invariant")
        if not isinstance(title, str) or not title.strip():
            return None
        if not isinstance(description, str):
            return None
        if not isinstance(invariant, str) or not invariant.strip():
            return None
        out.append(
            TaskCreationProposedSplit(
                title=title.strip(),
                description=description,
                invariant=invariant.strip(),
            )
        )
    return tuple(out)


def _parse_task_creation_verdict(
    obj: dict[str, Any],
) -> TaskCreationVerdict | None:
    """Parse a verdict envelope dict into :class:`TaskCreationVerdict`.

    Returns ``None`` when the envelope is malformed — caller treats as
    fail-open (default verdict).
    """
    relationship = obj.get("relationship")
    if relationship not in ("distinct", "duplicate_of", "supersedes"):
        return None
    scope = obj.get("scope")
    if scope not in ("single", "multi"):
        return None
    # Rob review on PR #1932: ``duplicate_of`` / ``supersedes`` mean
    # "drop this proposal" — combining either with ``scope="multi"``
    # is contradictory (the apply path handles drops BEFORE fan-out
    # so a malformed envelope of that shape would silently delete
    # legitimate new work).  Reject the contradictory combo at parse
    # time; caller treats the None return as fail-open.
    if relationship in ("duplicate_of", "supersedes") and scope == "multi":
        return None

    duplicate_of_id: str | None = None
    if relationship == "duplicate_of":
        raw_id = obj.get("duplicate_of_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return None
        duplicate_of_id = raw_id.strip()

    supersedes_id: str | None = None
    if relationship == "supersedes":
        raw_id = obj.get("supersedes_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return None
        supersedes_id = raw_id.strip()

    proposed_splits = _parse_proposed_splits(obj.get("proposed_splits"))
    if proposed_splits is None:
        return None
    if scope == "multi" and not proposed_splits:
        return None

    rationale_raw = obj.get("rationale", "")
    rationale = rationale_raw.strip() if isinstance(rationale_raw, str) else ""

    return TaskCreationVerdict(
        relationship=relationship,
        duplicate_of_id=duplicate_of_id,
        supersedes_id=supersedes_id,
        scope=scope,
        proposed_splits=proposed_splits,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Task-completion critic (HOL-17 / #1911)
# ---------------------------------------------------------------------------
#
# Runs after the worker emits ``commit-task-complete`` and the harness has
# staged + committed.  The critic verifies the committed diff against the
# task's named invariant along TWO axes:
#
#   - establishes: the diff actually makes the invariant true (catches
#                  PR #1858's "13 tasks marked complete without any code
#                  change" pattern — empty diff, claimed done).
#   - only:        the diff does ONLY the named work; no scope-creep
#                  refactors, dependency bumps, or unrelated cleanups.
#
# On ``passed=false`` the worker handler resets the just-landed commit
# (``git reset --soft HEAD~1`` — changes remain staged), marks the task
# back to ``in_progress`` with the gap appended to the description, and
# the worker re-picks on the next cycle to address the specific complaint.


@dataclass(frozen=True)
class TaskCompletionVerdict:
    """Critic verdict for one ``commit-task-complete`` op.

    Default (no critic wired, or critic failed open) is "passed, no
    rationale" so the just-landed commit ships unchanged — preserves
    the legacy no-critic behaviour.

    ``passed``: ``True`` → push and mark complete (current behaviour);
    ``False`` → soft-reset the commit, append ``gap`` to the task
    description, mark task in_progress, worker re-picks next cycle.

    ``gap`` is the one-line complaint suitable for appending to the
    task description so the next worker turn sees it as guidance.
    Required when ``passed=False``.
    """

    passed: bool = True
    gap: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.passed and not self.gap.strip():
            raise ValueError(
                "TaskCompletionVerdict.gap is required when passed=False — "
                "the gap is what the next worker turn needs to address"
            )


def _parse_task_completion_verdict(
    obj: dict[str, Any],
) -> TaskCompletionVerdict | None:
    """Parse a verdict envelope dict into :class:`TaskCompletionVerdict`.

    Returns ``None`` when the envelope is malformed — caller treats as
    fail-open (default verdict: ``passed=True``).
    """
    passed = obj.get("passed")
    if not isinstance(passed, bool):
        return None
    rationale_raw = obj.get("rationale", "")
    rationale = rationale_raw.strip() if isinstance(rationale_raw, str) else ""
    if passed:
        return TaskCompletionVerdict(passed=True, rationale=rationale)
    gap = obj.get("gap")
    if not isinstance(gap, str) or not gap.strip():
        return None
    return TaskCompletionVerdict(passed=False, gap=gap.strip(), rationale=rationale)


def run_task_completion_critic(
    task_invariant: str,
    task_description: str,
    diff: str,
    *,
    agent: ProviderAgent,
    prompts: Prompts,
    critic_system_prompt: str,
) -> TaskCompletionVerdict:
    """Ask Opus to verdict the committed diff against the task invariant.

    Returns the parsed :class:`TaskCompletionVerdict`.  Fail-open on
    transport errors, malformed responses, or unparseable verdicts —
    the default verdict (``passed=True``) preserves the legacy
    no-critic behaviour, so a flaky critic must not block valid task
    completion from shipping.  ``ContextOverflowError`` /
    ``SessionLeakError`` still propagate per project convention.

    When the task has no declared invariant (legacy tasks pre-HOL-11),
    the critic still runs but the prompt renders ``(no invariant
    declared)`` so Opus reasons about scope-creep without an anchor.
    Empty-diff complete still fails the establishes axis even without
    an invariant (a completed task that didn't change anything is the
    PR #1858 pattern regardless of whether HOL-12 named the work).
    """
    prompt = prompts.task_completion_critic_prompt(
        task_invariant=task_invariant,
        task_description=task_description,
        diff=diff,
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
            "task-completion critic transport failure (%s) — failing open",
            exc,
        )
        return TaskCompletionVerdict()

    objs = extract_json_objects(raw or "")
    if not objs:
        log.warning("task-completion critic returned no parseable JSON — failing open")
        return TaskCompletionVerdict()
    for obj in objs:
        verdict = _parse_task_completion_verdict(obj)
        if verdict is not None:
            return verdict
    log.warning(
        "task-completion critic returned no envelope-shaped JSON in %d "
        "objects — failing open",
        len(objs),
    )
    return TaskCompletionVerdict()


def run_task_creation_critic(
    proposed_task: dict[str, Any],
    current_queue: list[dict[str, Any]],
    *,
    agent: ProviderAgent,
    prompts: Prompts,
    critic_system_prompt: str,
) -> TaskCreationVerdict:
    """Ask Opus to verdict the proposed ``new`` task against the queue.

    Returns the parsed :class:`TaskCreationVerdict`.  Fail-open on
    transport errors, malformed responses, or unparseable verdicts —
    the default verdict (``distinct`` + ``single``) passes the proposed
    task through unchanged, matching the legacy no-critic behaviour.
    ``ContextOverflowError`` / ``SessionLeakError`` still propagate per
    project convention.

    ``critic_system_prompt`` must be the JSON-capable variant
    (:meth:`fido.prompts.Prompts.critic_system_prompt`) — codex
    r3293399806 on PR #1932 flagged that wiring the follow-up text-only
    prompt here silently disabled the gate.
    """
    prompt = prompts.task_creation_critic_prompt(
        proposed_task=proposed_task,
        current_queue=current_queue,
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
            "task-creation critic transport failure (%s) — failing open",
            exc,
        )
        return TaskCreationVerdict()

    objs = extract_json_objects(raw or "")
    if not objs:
        log.warning("task-creation critic returned no parseable JSON — failing open")
        return TaskCreationVerdict()
    # codex r3293359040 on PR #1932: scan all extracted JSON objects for
    # the first one that parses as the verdict envelope.  A response that
    # leads with an unrelated ``{}`` followed by a real verdict would
    # otherwise be treated as malformed and fail open silently.
    for obj in objs:
        verdict = _parse_task_creation_verdict(obj)
        if verdict is not None:
            return verdict
    log.warning(
        "task-creation critic returned no envelope-shaped JSON in %d "
        "objects — failing open",
        len(objs),
    )
    return TaskCreationVerdict()


# ---------------------------------------------------------------------------
# Reply-prose claim-grounding critic (HOL-18 / #1912)
# ---------------------------------------------------------------------------
#
# Fires at every reply prose emission (triage, material-divergence, terminal
# aggregate).  Verifies that every specific claim in the prose (commit SHAs,
# issue/PR numbers, file paths, "I filed", "the work is in") maps to ground
# truth the caller has gathered.  Catches PR #1858's "work is in a recent
# commit" lies and closes #1855 at the prose-verification layer.


@dataclass(frozen=True)
class ReplyProseVerdict:
    """Critic verdict for one reply prose emission.

    Default (no critic wired, or critic failed open) is "passed, no
    rationale" so the prose ships unchanged — matches the legacy
    no-critic behaviour.

    ``passed=True`` → ship the prose.  ``passed=False`` → regenerate
    with ``gap`` as the nudge so the next prose addresses the specific
    unverified claim.
    """

    passed: bool = True
    gap: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.passed and not self.gap.strip():
            raise ValueError(
                "ReplyProseVerdict.gap is required when passed=False — "
                "the gap names the unverified claim the next prose attempt "
                "must address"
            )


def _parse_reply_prose_verdict(
    obj: dict[str, Any],
) -> ReplyProseVerdict | None:
    """Parse a verdict envelope dict into :class:`ReplyProseVerdict`.

    Returns ``None`` when the envelope is malformed — caller treats as
    fail-open (default verdict: ``passed=True``).
    """
    passed = obj.get("passed")
    if not isinstance(passed, bool):
        return None
    rationale_raw = obj.get("rationale", "")
    rationale = rationale_raw.strip() if isinstance(rationale_raw, str) else ""
    if passed:
        return ReplyProseVerdict(passed=True, rationale=rationale)
    gap = obj.get("gap")
    if not isinstance(gap, str) or not gap.strip():
        return None
    return ReplyProseVerdict(passed=False, gap=gap.strip(), rationale=rationale)


def run_reply_prose_critic(
    reply_text: str,
    structured_state: dict[str, Any],
    *,
    agent: ProviderAgent,
    prompts: Prompts,
    critic_system_prompt: str,
) -> ReplyProseVerdict:
    """Ask Opus whether ``reply_text``'s specific claims are grounded.

    Returns the parsed :class:`ReplyProseVerdict`.  Fail-open on
    transport errors, malformed responses, or unparseable verdicts —
    the default verdict (``passed=True``) preserves the legacy
    no-critic behaviour.  ``ContextOverflowError`` /
    ``SessionLeakError`` still propagate per project convention.

    Like the sibling critics, scans every extracted JSON object and
    accepts the first that parses as the verdict envelope; a malformed
    early envelope (``{"passed": false}`` without a gap) does NOT
    short-circuit the scan.
    """
    prompt = prompts.reply_prose_claim_grounding_prompt(
        reply_text=reply_text,
        structured_state=structured_state,
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
            "reply-prose critic transport failure (%s) — failing open",
            exc,
        )
        return ReplyProseVerdict()

    objs = extract_json_objects(raw or "")
    if not objs:
        log.warning("reply-prose critic returned no parseable JSON — failing open")
        return ReplyProseVerdict()
    saw_malformed_fail = False
    for obj in objs:
        verdict = _parse_reply_prose_verdict(obj)
        if verdict is not None:
            return verdict
        if obj.get("passed") is False:
            saw_malformed_fail = True
    if saw_malformed_fail:
        log.warning("reply-prose critic claimed fail without a gap — failing open")
    else:
        log.warning(
            "reply-prose critic returned no envelope-shaped JSON in %d "
            "objects — failing open",
            len(objs),
        )
    return ReplyProseVerdict()


# ---------------------------------------------------------------------------
# Insight-filing critic (HOL-19 / #1913)
# ---------------------------------------------------------------------------
#
# Runs once per proposed insight at filing time, AFTER the cheap
# per-comment idempotency marker check in
# ``_GitHubInsightFiler.file_insight`` has already cleared the
# same-comment-replay path.  The critic catches what the marker can't:
# cross-comment near-duplicates — an insight whose core claim was
# already filed against a different comment.  Returns a verdict the
# caller uses to skip filing (with a log link to the existing
# duplicate) or proceed.


@dataclass(frozen=True)
class InsightDedupVerdict:
    """Critic verdict for one proposed insight filing.

    Default (no critic wired, or critic failed open) is "not a
    duplicate, no rationale" so the insight files unchanged — matches
    the legacy no-critic behaviour.

    ``is_duplicate=True`` requires ``duplicate_url`` so the filer can
    log a pointer to the existing insight; the validator in
    :meth:`__post_init__` rejects malformed verdicts at construction
    time rather than at the call site.
    """

    is_duplicate: bool = False
    duplicate_url: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.is_duplicate and not self.duplicate_url.strip():
            raise ValueError(
                "InsightDedupVerdict.duplicate_url is required when "
                "is_duplicate=True — the URL points at the existing "
                "insight so the filer can log a pointer rather than "
                "silently dropping the proposed insight"
            )


# Repo that owns the corpus of insights ``_recent_insights_for_critic``
# feeds in.  A duplicate_url pointing anywhere else is hallucination —
# the critic was only ever shown insights from this repo, so any
# claim of duplication against another repo is malformed output and
# the verdict must fail open (let the insight file normally).
# Shared with ``fido.events`` so the filer's marker-write boundary
# stays in lockstep with this validator.
INSIGHT_REPO = "FidoCanCode/home"


# Codex on PR #1932: validate the URL at parse time so a malformed
# ``duplicate_url`` is treated as a malformed verdict (parser
# returns None → runner fails open → insight files normally) rather
# than as a valid "skip this filing" verdict that the filer would
# honor while the marker-writer silently dropped the durability
# record.  The check requires both the shape AND the target repo —
# splitting them into "shape here, repo downstream" let a
# cross-repo URL silently lose the insight too (the filer skipped
# while the marker-writer rejected the cross-repo target).
# Case-insensitive to match GitHub's case-insensitive owner/repo
# names.
_DEDUP_URL_RE = re.compile(
    r"^https://github\.com/" + re.escape(INSIGHT_REPO) + r"/issues/\d+(?:[#/?].*)?$",
    re.IGNORECASE,
)


def _parse_insight_dedup_verdict(
    obj: dict[str, Any],
) -> InsightDedupVerdict | None:
    """Parse a verdict envelope dict into :class:`InsightDedupVerdict`.

    Returns ``None`` when the envelope is malformed — caller treats as
    fail-open (default verdict: ``is_duplicate=False``), which lets
    the insight file normally.  A duplicate verdict with no URL OR a
    URL that doesn't match the GitHub issue shape both count as
    malformed; without the URL-shape check, the filer would skip
    the filing while the marker-writer silently dropped the
    durability record, losing the insight entirely.
    """
    is_duplicate = obj.get("is_duplicate")
    if not isinstance(is_duplicate, bool):
        return None
    rationale_raw = obj.get("rationale", "")
    rationale = rationale_raw.strip() if isinstance(rationale_raw, str) else ""
    if not is_duplicate:
        return InsightDedupVerdict(is_duplicate=False, rationale=rationale)
    url = obj.get("duplicate_url")
    if not isinstance(url, str) or not url.strip():
        return None
    if not _DEDUP_URL_RE.match(url.strip()):
        return None
    return InsightDedupVerdict(
        is_duplicate=True,
        duplicate_url=url.strip(),
        rationale=rationale,
    )


class InsightDedupTransportCounter:
    """HOL-19 follow-up / #1935: process-local consecutive-transport-
    failure counter for ``run_insight_dedup_critic``.

    Each ``run_insight_dedup_critic`` call updates the counter:
    transport failures (the ``except Exception`` arm — RuntimeError,
    network blips, model unavailable) bump it; a verdict-shape-valid
    response resets it.  Parse failures don't reset (no clean signal
    that the critic infrastructure is healthy) and don't bump (might
    just be one bad model turn).

    When the counter crosses ``threshold`` consecutive failures, the
    injected ``escalator`` callback is invoked with the failure count.
    Production wiring escalates via ``file_stuck_on_critic_bug`` with
    ``emission_point="insight-dedup-transport"``.

    Thread-safe under free-threaded Python: a lock guards every read-
    modify cycle so two simultaneous critic calls can't both observe
    "below threshold" and both miss an escalation.

    Process-local persistence per the issue body ("a flaky day would
    still trigger the bug within a session") — survives across critic
    calls in one Fido process, resets on restart.
    """

    def __init__(
        self,
        *,
        threshold: int = 3,
        escalator: Callable[[int], None] | None = None,
    ) -> None:
        self._threshold = threshold
        self._escalator = escalator
        self._consecutive_failures = 0
        self._already_escalated = False
        self._lock = threading.Lock()

    def record_transport_failure(self) -> None:
        escalator: Callable[[int], None] | None = None
        count = 0
        with self._lock:
            self._consecutive_failures += 1
            if (
                self._consecutive_failures >= self._threshold
                and not self._already_escalated
                and self._escalator is not None
            ):
                self._already_escalated = True
                escalator = self._escalator
                count = self._consecutive_failures
        # Fire the escalator OUTSIDE the lock — it may touch GitHub
        # and we don't want a slow API call holding the counter lock.
        # ``_already_escalated`` set under the lock prevents duplicate
        # fires from concurrent at-threshold transitions.
        if escalator is not None:
            escalator(count)

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._already_escalated = False


def run_insight_dedup_critic(
    proposed_insight: dict[str, str],
    recent_insights: list[dict[str, str]],
    *,
    agent: ProviderAgent,
    prompts: Prompts,
    critic_system_prompt: str,
    transport_counter: InsightDedupTransportCounter | None = None,
) -> InsightDedupVerdict:
    """Ask Opus whether ``proposed_insight`` near-duplicates any of
    ``recent_insights``.

    Returns the parsed :class:`InsightDedupVerdict`.  Fail-open on
    transport errors, malformed responses, or unparseable verdicts —
    the default verdict (``is_duplicate=False``) preserves the legacy
    no-critic behaviour so a flaky critic cannot block legitimate
    insight filing.  ``ContextOverflowError`` / ``SessionLeakError``
    still propagate per project convention.

    Like the sibling critics, scans every extracted JSON object and
    accepts the first that parses as the verdict envelope; a malformed
    early envelope (``{"is_duplicate": true}`` without a URL) does NOT
    short-circuit the scan.
    """
    prompt = prompts.insight_dedup_critic_prompt(
        proposed_insight=proposed_insight,
        recent_insights=recent_insights,
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
            "insight-dedup critic transport failure (%s) — failing open",
            exc,
        )
        # HOL-19 follow-up / #1935: count transport failures so a
        # cross-comment pattern (the critic infra itself is broken,
        # not "this insight is duplicate") can escalate to a bug.
        if transport_counter is not None:
            transport_counter.record_transport_failure()
        return InsightDedupVerdict()

    objs = extract_json_objects(raw or "")
    if not objs:
        log.warning("insight-dedup critic returned no parseable JSON — failing open")
        return InsightDedupVerdict()
    # Codex on PR #1932: the prompt contract is "duplicate_url MUST
    # be one of the recent_insights shown to the critic" — a URL
    # that's repo-shaped but ISN'T in the corpus is hallucination,
    # not a legitimate dedup signal.  Normalize and pre-compute the
    # allowed set once so the per-verdict check is cheap; an empty
    # corpus (recent_insights==[]) reduces this to "every duplicate
    # verdict is hallucination", which matches the prompt's "any
    # insight is non-duplicate by default" rule on empty input.
    allowed_urls = {
        _normalize_insight_url(str(entry.get("url", ""))) for entry in recent_insights
    }
    allowed_urls.discard("")
    saw_malformed_dup = False
    for obj in objs:
        verdict = _parse_insight_dedup_verdict(obj)
        if verdict is not None:
            if verdict.is_duplicate and (
                _normalize_insight_url(verdict.duplicate_url) not in allowed_urls
            ):
                log.warning(
                    "insight-dedup critic returned duplicate_url %r not in "
                    "corpus of %d recent insights — failing open (likely "
                    "hallucinated)",
                    verdict.duplicate_url,
                    len(allowed_urls),
                )
                continue
            # HOL-19 follow-up / #1935: a verdict-shape-valid response
            # means the critic infra is healthy — reset the
            # transport-failure counter.  Parse failures don't reset
            # (no clean health signal).
            if transport_counter is not None:
                transport_counter.record_success()
            return verdict
        if obj.get("is_duplicate") is True:
            saw_malformed_dup = True
    if saw_malformed_dup:
        log.warning(
            "insight-dedup critic claimed duplicate without a URL — failing open"
        )
    else:
        log.warning(
            "insight-dedup critic returned no envelope-shaped JSON in %d "
            "objects — failing open",
            len(objs),
        )
    return InsightDedupVerdict()


def _normalize_insight_url(url: str) -> str:
    """Normalize a GitHub insight URL to its ISSUE IDENTITY for
    corpus-membership comparison.

    Two URLs that refer to the same issue must produce the same
    normalised form, even when one carries an anchor / query string
    / trailing slash that the other doesn't.  Codex on PR #1932:
    without that collapse, a verdict like
    ``.../issues/100#issuecomment-1`` was treated as out-of-corpus
    against ``.../issues/100`` — failing open and letting the
    duplicate insight file.

    Uses :func:`urllib.parse.urlparse` to split off query+fragment
    (cleaner than slicing on ``#``/``?`` manually; handles edge
    cases like multiple ``?`` and percent-encoding correctly) and
    keeps only ``scheme://netloc/path``.  Lowercase + trailing-slash
    strip cover the remaining axes (GitHub is case-insensitive on
    scheme/host/owner/repo).
    """
    parsed = urlparse(url.strip())
    canonical = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return canonical.lower().rstrip("/")
