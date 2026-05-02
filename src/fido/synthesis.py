"""Synthesis output types for the unified comment-handling turn.

The action vocabulary is constrained and enumerated (Constraint A from
#1230): Fido may only perform operations from this closed set when
responding to a PR comment.  No verb outside this list exists; no
free-form action escape hatch.

Reply text is a required top-level field (Constraint B): every synthesis
response must include a non-empty reply.  The invariant is enforced by
the type rather than buried as an optional list element — a
``CommentResponse`` without prose simply cannot be constructed.
"""

from dataclasses import dataclass

from fido.types import TaskType

# Valid GitHub reaction shortcodes.
VALID_REACTIONS: frozenset[str] = frozenset(
    {"+1", "-1", "laugh", "confused", "heart", "hooray", "rocket", "eyes"}
)


def validate_reaction(emoji: str) -> str:
    """Return *emoji* unchanged, or raise ``ValueError`` if not a valid GitHub reaction."""
    if emoji not in VALID_REACTIONS:
        raise ValueError(
            f"Invalid reaction {emoji!r}. Valid reactions: {sorted(VALID_REACTIONS)}"
        )
    return emoji


# ---------------------------------------------------------------------------
# Individual action types (Constraint A — closed vocabulary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddReaction:
    """Add an emoji reaction to the triggering comment.

    *emoji* must be one of the GitHub reaction shortcodes in
    :data:`VALID_REACTIONS`.
    """

    emoji: str


@dataclass(frozen=True)
class CreateTask:
    """Create a new work-queue task.

    *task_type* defaults to ``THREAD`` since synthesis responses handle
    PR comment threads.  Use ``SPEC`` when the synthesis identifies broader
    planned work not tied to a single comment lineage.
    """

    title: str
    task_type: TaskType = TaskType.THREAD
    description: str = ""


@dataclass(frozen=True)
class CompleteTask:
    """Mark an existing work-queue task completed by ID."""

    task_id: str


@dataclass(frozen=True)
class ModifyTask:
    """Update an existing task's title, description, or both.

    At least one of *new_title* or *new_description* must be provided.
    """

    task_id: str
    new_title: str | None = None
    new_description: str | None = None

    def __post_init__(self) -> None:
        if self.new_title is None and self.new_description is None:
            raise ValueError(
                "ModifyTask requires at least one of new_title or new_description"
            )


@dataclass(frozen=True)
class Preempt:
    """Signal whether the current in-progress worker task should be preempted.

    When *preempt* is ``True``, the handler requests that the worker abort
    its current task and re-evaluate its queue immediately after actions
    are applied.
    """

    preempt: bool


@dataclass(frozen=True)
class NoOp:
    """Explicitly take no additional action beyond posting the required reply."""


# The closed vocabulary of additional effects Fido may produce from a single
# synthesis call.  Constraint A: no operations outside this set exist.
SynthesisAction = AddReaction | CreateTask | CompleteTask | ModifyTask | Preempt | NoOp


# ---------------------------------------------------------------------------
# Synthesis response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommentResponse:
    """Structured output from the unified comment-handling synthesis LLM call.

    Attributes
    ----------
    reasoning:
        Private chain-of-thought — logged for traceability, never posted
        to GitHub.
    reply_text:
        The reply to post to the PR comment thread.  Always required and
        always non-empty (Constraint B).  Always freshly synthesised from
        the actual comment context — never a template, never a canned
        phrase, never absent.
    actions:
        Ordered sequence of additional effects from the closed action
        vocabulary.  Executed in order after the reply is posted.
    """

    reasoning: str
    reply_text: str
    actions: tuple[SynthesisAction, ...]

    def __post_init__(self) -> None:
        if not self.reply_text.strip():
            raise ValueError(
                "CommentResponse.reply_text must be non-empty (Constraint B: "
                "reply prose is always required and always freshly synthesised)"
            )
