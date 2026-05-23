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
from dataclasses import dataclass
from typing import Any, Literal

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
    "TaskCompletionVerdict",
    "TaskCreationProposedSplit",
    "TaskCreationVerdict",
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
    followup_system_prompt: str,
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
            system_prompt=followup_system_prompt,
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
    followup_system_prompt: str,
) -> TaskCreationVerdict:
    """Ask Opus to verdict the proposed ``new`` task against the queue.

    Returns the parsed :class:`TaskCreationVerdict`.  Fail-open on
    transport errors, malformed responses, or unparseable verdicts —
    the default verdict (``distinct`` + ``single``) passes the proposed
    task through unchanged, matching the legacy no-critic behaviour.
    ``ContextOverflowError`` / ``SessionLeakError`` still propagate per
    project convention.
    """
    prompt = prompts.task_creation_critic_prompt(
        proposed_task=proposed_task,
        current_queue=current_queue,
    )
    try:
        raw = agent.run_turn(
            prompt,
            allowed_tools=READ_ONLY_ALLOWED_TOOLS,
            system_prompt=followup_system_prompt,
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
