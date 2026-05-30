"""Synthesis executor — dispatches a CommentResponse to GitHub effects.

Given a :class:`~fido.synthesis.CommentResponse` from the synthesis LLM
call, the executor posts the reply, manages the emoji reaction lifecycle
(eyes → final emoji), triggers rescope when a ``change_request`` is
present, and returns the :class:`ReviewReplyOutcome` for the Rocq
oracle.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from fido.rocq.replied_comment_claims import ReviewReplyOutcome
from fido.synthesis import CommentResponse, Insight, outcome_for_response
from fido.types import RescopeIntent

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommentTarget:
    """Identifies the comment being replied to and reacted on.

    Attributes
    ----------
    repo:
        Full repo slug (e.g. ``"owner/repo"``).
    pr:
        PR number.
    comment_id:
        ID of the triggering comment.
    comment_type:
        GitHub comment namespace: ``"pulls"`` for review comments,
        ``"issues"`` for top-level PR/issue comments.
    """

    repo: str
    pr: int
    comment_id: int
    comment_type: str
    author: str = ""
    """GitHub login of the commenter (per #1801 / INV-C).  Forwarded
    into the ``RescopeIntent`` so the rescope prompt can render
    per-author attribution.  Empty when the caller didn't populate it
    (legacy / synthetic test fixtures)."""


class ReplyPoster(Protocol):
    """Posts a reply to a comment and returns the API response."""

    def reply_to_review_comment(
        self,
        repo: str,
        pr: int | str,
        body: str,
        in_reply_to: int | str,
    ) -> dict[str, Any]: ...

    def comment_issue(
        self,
        repo: str,
        number: int | str,
        body: str,
    ) -> dict[str, Any]: ...

    def add_reaction(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
        content: str,
    ) -> None: ...

    def list_reactions(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
    ) -> list[dict[str, Any]]: ...

    def delete_reaction(
        self,
        repo: str,
        comment_type: str,
        comment_id: int | str,
        reaction_id: int | str,
    ) -> None: ...


class RescopeTrigger(Protocol):
    """Triggers a background rescope from a comment synthesis response.

    Receives a :class:`~fido.types.RescopeIntent` carrying the plain-English
    change request alongside the originating comment identity and timestamp
    so the rescope machinery can track which comment triggered each intent
    and reply back on material outcomes.
    """

    def trigger_rescope(self, intent: RescopeIntent) -> None: ...


class InsightFiler(Protocol):
    """Files a synthesis insight as a durable GitHub issue."""

    def file_insight(self, insight: Insight, target: CommentTarget) -> None: ...


class SynthesisExecutor:
    """Dispatches a :class:`CommentResponse` to its side effects.

    Effects, in order:

    1. **Post reply** — always (Constraint B guarantees non-empty text).
    2. **Add emoji reaction** — if ``response.emoji`` is set, add it to
       the triggering comment.
    3. **Trigger rescope** — if ``response.change_request`` is set, hand
       it to the rescope trigger (which preempts the current task and
       passes the intent to the rescope machinery).
    4. **File insights** — if ``response.insights`` is non-empty and an
       :class:`InsightFiler` was injected, file each insight as a GitHub
       issue.
    5. **Return outcome** — maps the response to a
       :class:`ReviewReplyOutcome` for the Rocq oracle.

    Dependencies are injected through the constructor per the
    constructor-DI convention.
    """

    def __init__(
        self,
        gh: ReplyPoster,
        rescope: RescopeTrigger | None = None,
        insight_filer: InsightFiler | None = None,
        *,
        fido_logins: frozenset[str] = frozenset(),
    ) -> None:
        self._gh = gh
        self._rescope = rescope
        self._insight_filer = insight_filer
        self._fido_logins = fido_logins

    def file_insights_pre_reply(
        self,
        response: CommentResponse,
        target: CommentTarget,
    ) -> None:
        """HOL-26 / #1920: file insights BEFORE the reply is posted, so
        the durable GitHub issues exist before any user-visible reply
        can claim they do.

        Public hook used by production reply paths (``Dispatcher``'s
        ``reply_to_review_comment`` / ``reply_to_issue_comment``) which
        own their own post step and must call this before posting.  The
        matching ``execute_effects_only`` call must then pass
        ``insights_already_filed=True`` so the post-reply effects pass
        does not double-file.

        The convenience entry :meth:`execute` chains this + post +
        post-reply effects internally; production callers that thread
        the outbox protocol through their own post can't use that
        chain and split it explicitly via this hook.
        """
        self._file_insights(response, target)

    def execute(
        self,
        response: CommentResponse,
        target: CommentTarget,
    ) -> ReviewReplyOutcome:
        """Post the reply and run all post-reply effects in one call.

        HOL-26 / #1920: files insights FIRST so the reply post sequences
        AFTER their durability is established.  The full sequencing-layer
        fix (synthesis prompt sees ``insights_with_urls[]`` so the LLM
        can weave URLs into prose by construction) requires splitting
        synthesis into two LLM calls; that's a substantial restructure
        out of scope here.  The minimal viable fix in this method
        addresses the ordering contract — insights exist as durable
        GitHub issues before any user-visible reply claims they do.
        HOL-18's reply-prose claim-grounding critic remains the
        verification-layer backstop for URL-in-prose.

        Production callers in ``Dispatcher`` use
        :meth:`file_insights_pre_reply` + their own outbox-routed post
        + :meth:`execute_effects_only` (with
        ``insights_already_filed=True``) instead of this entry point,
        because they need to thread the outbox protocol through the
        post step.
        """
        self.file_insights_pre_reply(response, target)
        self._post_reply(response.reply_text, target)
        return self.execute_effects_only(response, target, insights_already_filed=True)

    def execute_effects_only(
        self,
        response: CommentResponse,
        target: CommentTarget,
        *,
        insights_already_filed: bool = False,
    ) -> ReviewReplyOutcome:
        """Execute post-reply effects for *response* against *target*.

        This method handles the emoji reaction lifecycle and rescope trigger,
        but does **not** post the reply itself.  Use this when the caller owns
        the reply-posting step (e.g. to thread the outbox protocol through
        it), and only needs the executor to manage the remaining side effects.

        Effects, in order:

        1. **Remove eyes reaction** — deletes any ``eyes`` reaction on the
           triggering comment that was posted by the authenticated user
           (best-effort; errors are logged but do not propagate).
        2. **Add emoji reaction** — if ``response.emoji`` is set, adds it to
           the triggering comment after removing eyes.
        3. **Trigger rescope** — if ``response.change_request`` is set, hands
           it to the rescope trigger.
        4. **File insights** — if ``response.insights`` is non-empty and an
           :class:`InsightFiler` was injected, files each insight as a GitHub
           issue.
        5. **Return outcome** — maps the response to a
           :class:`ReviewReplyOutcome` for the Rocq oracle.
        """
        # 1. Remove eyes reaction (best-effort)
        self.remove_eyes_reaction(target)

        # 2. Add emoji reaction if present
        if response.emoji is not None:
            log.info(
                "adding %s reaction to comment %s",
                response.emoji,
                target.comment_id,
            )
            self._gh.add_reaction(
                target.repo,
                target.comment_type,
                target.comment_id,
                response.emoji,
            )

        # 3. Trigger rescope if change_request present
        self._maybe_trigger_rescope(response, target)

        # 4. File insights if any (HOL-26: may have already been filed
        #    pre-reply by execute() to sequence durability before prose).
        if not insights_already_filed:
            self._file_insights(response, target)

        # 5. Return outcome for Rocq oracle
        return outcome_for_response(response)

    def _post_reply(self, body: str, target: CommentTarget) -> None:
        """Post the reply to the correct endpoint based on comment type."""
        if target.comment_type == "pulls":
            log.info(
                "posting review reply to %s PR #%d comment %d",
                target.repo,
                target.pr,
                target.comment_id,
            )
            self._gh.reply_to_review_comment(
                target.repo, target.pr, body, target.comment_id
            )
        else:
            log.info(
                "posting issue comment to %s #%d",
                target.repo,
                target.pr,
            )
            self._gh.comment_issue(target.repo, target.pr, body)

    def _maybe_trigger_rescope(
        self, response: CommentResponse, target: CommentTarget
    ) -> None:
        """Build a :class:`RescopeIntent` and fire the rescope trigger if configured.

        No-op when *response.change_request* is ``None``.  Logs a warning if a
        change_request is present but no rescope trigger was injected.
        """
        if response.change_request is None:
            return
        if self._rescope is not None:
            intent = RescopeIntent(
                change_request=response.change_request,
                comment_id=target.comment_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                comment_type=target.comment_type,
                author=target.author,
                repo=target.repo,
                pr_number=target.pr,
            )
            log.info(
                "triggering rescope for comment %d: %s",
                intent.comment_id,
                intent.change_request[:80],
            )
            self._rescope.trigger_rescope(intent)
        else:
            log.warning(
                "change_request present but no rescope trigger configured — "
                "skipping rescope for: %s",
                response.change_request[:80],
            )

    def _file_insights(
        self,
        response: CommentResponse,
        target: CommentTarget,
    ) -> None:
        """File each insight in *response* through the configured InsightFiler.

        No-op when *response.insights* is empty or no :class:`InsightFiler`
        was injected.
        """
        if not response.insights or self._insight_filer is None:
            return
        for insight in response.insights:
            log.info(
                "filing insight %r for comment %d",
                insight.title,
                target.comment_id,
            )
            self._insight_filer.file_insight(insight, target)

    def remove_eyes_reaction(self, target: CommentTarget) -> None:
        """Remove Fido's ``eyes`` reaction from *target* (best-effort).

        Lists all reactions on the comment and deletes any with
        ``content == "eyes"`` whose author matches ``fido_logins``.
        Other users' eyes reactions are left untouched.  Errors are
        logged but never propagated — a failed reaction cleanup must not
        abort an otherwise-successful reply.
        """
        try:
            reactions = self._gh.list_reactions(
                target.repo,
                target.comment_type,
                target.comment_id,
            )
            for reaction in reactions:
                if reaction.get("content") != "eyes":
                    continue
                # Only delete Fido's own eyes reactions — other users' eyes
                # belong to them and must not be removed.
                login = reaction.get("user", {}).get("login", "")
                if self._fido_logins and login.lower() not in self._fido_logins:
                    continue
                reaction_id = reaction.get("id")
                if reaction_id is not None:
                    log.info(
                        "removing eyes reaction %s from comment %s",
                        reaction_id,
                        target.comment_id,
                    )
                    self._gh.delete_reaction(
                        target.repo,
                        target.comment_type,
                        target.comment_id,
                        reaction_id,
                    )
        except Exception:
            # Best-effort UI cleanup — the eyes reaction is a cosmetic
            # signal, not authoritative state.  GitHub's reaction API can
            # raise requests.RequestException, KeyError on a malformed
            # response, or various provider-side errors; catching broadly
            # here is the right policy exception to CLAUDE.md's fail-closed
            # rule, which targets authoritative runner paths.
            log.exception(
                "failed to remove eyes reaction from comment %s — continuing",
                target.comment_id,
            )
