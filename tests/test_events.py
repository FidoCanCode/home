from pathlib import Path
from typing import Never
from unittest.mock import MagicMock

import pytest

from fido.claude import ClaudeClient
from fido.config import Config, RepoMembership
from fido.config import RepoConfig as _RepoConfig
from fido.events import (
    _INSIGHT_LABEL,
    _INSIGHT_REPO,
    Action,
    Dispatcher,
    WebhookIngressOracle,
    _BackgroundRescopeTrigger,
    _build_issue_comment_action,
    _configured_agent,
    _existing_reply_artifact,
    _get_commit_summary,
    _GitHubInsightFiler,
    _insight_source_link,
    _is_allowed,
    _load_active_context_for_rescope,
    _posted_comment_id,
    _record_reply_artifact,
    _reply_promise_ids,
    _rewrite_pr_description,
    _task_snapshot,
    build_review_comment_action,
    launch_worker,
    needs_more_context,
    reply_to_review,
    thread_lineage_comment_ids,
)
from fido.prompts import Prompts
from fido.provider import ProviderID, ThreadKind
from fido.rocq import replied_comment_claims as oracle
from fido.rocq import task_queue_rescope
from fido.state import State
from fido.store import FidoStore, ReplyPromiseRecord
from fido.synthesis import CommentResponse, Insight
from fido.synthesis_call import SynthesisCriticExhaustedError, SynthesisExhaustedError
from fido.synthesis_executor import CommentTarget
from fido.tasks import Tasks
from fido.types import ActiveIssue, ActivePR, IntentVerdict, RescopeIntent
from fido.worker import ActivityReporter
from tests.fakes import _FakeDispatcher


def _synthesis_response(
    reply_text: str = "I'll look into that.",
    emoji: str | None = None,
    change_request: str | None = None,
) -> CommentResponse:
    """Build a CommentResponse for use in test patches."""
    return CommentResponse(
        reasoning="thinking",
        reply_text=reply_text,
        emoji=emoji,
        change_request=change_request,
    )


class RepoConfig(_RepoConfig):
    def __init__(
        self,
        *args: object,
        provider: ProviderID = ProviderID.CLAUDE_CODE,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, provider=provider, **kwargs)


def _config(tmp_path: Path) -> Config:
    return Config(
        port=9000,
        secret=b"test",
        repos={},
        allowed_bots=frozenset({"copilot[bot]"}),
        log_level="WARNING",
        sub_dir=tmp_path / "sub",
    )


def _repo_cfg(tmp_path: Path) -> RepoConfig:
    from fido.config import RepoMembership

    return RepoConfig(
        name="owner/repo",
        work_dir=tmp_path,
        membership=RepoMembership(collaborators=frozenset({"owner"})),
    )


def _payload(repo_owner: str = "owner") -> dict:
    return {
        "repository": {
            "full_name": f"{repo_owner}/repo",
            "owner": {"login": repo_owner},
        },
    }


def _client(return_value: str = "", *, side_effect: object = None) -> MagicMock:
    """Build a mock ClaudeClient with run_turn configured."""
    client = MagicMock(spec=ClaudeClient)
    client.voice_model = "claude-opus-4-6"
    client.work_model = "claude-sonnet-4-6"
    client.brief_model = "claude-haiku-4-5"
    if side_effect is not None:
        client.run_turn.side_effect = side_effect
    else:
        client.run_turn.return_value = return_value
    return client


def _make_mock_gh() -> MagicMock:
    """Return a MagicMock pre-configured with the gh return shapes the
    reply pipeline depends on.

    The reply-outbox protocol asserts ``_posted_comment_id(posted) is not
    None`` when promise_ids are queued, so ``reply_to_review_comment`` and
    ``comment_issue`` must return ``{"id": int}`` rather than the
    auto-mocked sub-MagicMock.  ``fetch_comment_thread`` defaults to ``[]``
    so the truthiness guard at events.py:1547 doesn't IndexError on
    auto-mocked iterables.  Tests that need a different shape override
    after construction.
    """
    gh = MagicMock()
    gh.reply_to_review_comment.return_value = {"id": 90_001}
    gh.comment_issue.return_value = {"id": 90_002}
    gh.fetch_comment_thread.return_value = []
    # ``create_issue`` returns a string URL — production passes it
    # through ``uuid.uuid5`` (events.py:403), which would TypeError on
    # an auto-mocked sub-MagicMock.
    gh.create_issue.return_value = "https://github.com/owner/repo/issues/0"
    # ``get_repo_info`` returns the repo slug; production sites pass it
    # as a string into ``CommentTarget.repo`` (and downstream into the
    # ``rescope_intent_outbox`` SQLite row), which would TypeError on an
    # auto-mocked sub-MagicMock.
    gh.get_repo_info.return_value = "owner/repo"
    return gh


def _oracle_owner(owner: str) -> object:
    match owner:
        case "webhook":
            return oracle.OwnerWebhook()
        case "worker":
            return oracle.OwnerWorker()
        case "recovery":
            return oracle.OwnerRecovery()


def _promise_state_name(state: object) -> str:
    if isinstance(state, str):
        return {
            "prepared": "PromisePrepared",
            "posted": "PromisePosted",
            "acked": "PromiseAcked",
            "failed": "PromiseFailed",
            "in_progress": "ClaimInProgress",
            "completed": "ClaimCompleted",
            "retryable_failed": "ClaimRetryableFailed",
        }[state]
    return type(state).__name__


class TestNeedsMoreContext:
    def test_haiku_yes_returns_true(self) -> None:
        assert needs_more_context("same", agent=_client("YES"))

    def test_haiku_yes_with_explanation_returns_true(self) -> None:
        assert needs_more_context("^", agent=_client("YES, this is vague"))

    def test_haiku_no_returns_false(self) -> None:
        assert not needs_more_context(
            "This is a detailed review comment.",
            agent=_client("NO"),
        )

    def test_haiku_no_with_explanation_returns_false(self) -> None:
        assert not needs_more_context(
            "Please rename this variable to be more descriptive.",
            agent=_client("NO, it's clear"),
        )

    def test_subprocess_exception_returns_false(self) -> None:
        assert not needs_more_context("ditto", agent=_client(""))

    def test_timeout_returns_false(self) -> None:
        assert not needs_more_context("same", agent=_client(""))

    def test_empty_output_returns_false(self) -> None:
        assert not needs_more_context("here too", agent=_client(""))

    def test_uses_haiku_model(self) -> None:
        client = _client("YES")
        needs_more_context("same", agent=client)
        assert client.run_turn.call_args.kwargs["model"] == "claude-haiku-4-5"

    def test_requires_agent(self) -> None:
        with pytest.raises(ValueError, match="needs_more_context requires agent"):
            needs_more_context("some comment")

    def test_configured_agent_uses_provider_factory(self, tmp_path: Path) -> None:
        from fido.provider_factory import DefaultProviderFactory

        cfg = _config(tmp_path)
        cfg.repos["owner/repo"] = RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            provider=ProviderID.COPILOT_CLI,
        )
        sentinel = MagicMock()
        factory = MagicMock(spec=DefaultProviderFactory)
        factory.create_agent.return_value = sentinel
        assert (
            _configured_agent(cfg, cfg.repos["owner/repo"], _factory=factory)
            is sentinel
        )


class TestRecoverReplyPromises:
    def _prepare_promise(
        self, tmp_path: Path, comment_type: str, comment_id: int
    ) -> ReplyPromiseRecord:
        promise = FidoStore(tmp_path).prepare_reply(
            owner="recovery",
            comment_type=comment_type,
            anchor_comment_id=comment_id,
        )
        assert promise is not None
        return promise

    def _assert_recovery_matches_oracle(
        self,
        tmp_path: Path,
        promise: ReplyPromiseRecord,
        observation: object,
        *,
        covered_comment_ids: tuple[int, ...] | None = None,
    ) -> None:
        comments = list(
            covered_comment_ids
            if covered_comment_ids is not None
            else promise.covered_comment_ids[1:]
        )
        prepared = oracle.prepare_claims(
            _oracle_owner("recovery"),
            1,
            promise.anchor_comment_id,
            comments,
            {},
            {},
        )
        assert prepared is not None
        claims, promises = prepared
        claims, promises = oracle.recover_promise(1, observation, claims, promises)

        persisted = FidoStore(tmp_path).promise(promise.promise_id)
        assert persisted is not None
        assert _promise_state_name(persisted.state) == _promise_state_name(
            promises[1].promise_state
        )
        for comment_id in promise.covered_comment_ids:
            assert (
                FidoStore(tmp_path).claim_state(comment_id)
                == {
                    "ClaimInProgress": "in_progress",
                    "ClaimCompleted": "completed",
                    "ClaimRetryableFailed": "retryable_failed",
                }[_promise_state_name(claims[comment_id].claim_state)]
            )

    def test_returns_false_when_no_promises(self, tmp_path: Path) -> None:
        gh = MagicMock()
        assert not Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).recover_reply_promises(
            tmp_path / ".git" / "fido",
            7,
            registry=MagicMock(spec=ActivityReporter),
        )

    def test_recovers_issue_comment_promise(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "user": {"login": "owner"},
        }
        fake_registry = MagicMock(spec=ActivityReporter)
        spy_tasks = Tasks(tmp_path)
        fake_registry.tasks_for.return_value = spy_tasks

        result = Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                change_request="task one"
            ),
            sync_fn=lambda *a, **kw: None,
            thread_start_fn=lambda t: None,
            reorder_coalesce_state={},
            get_commit_summary_fn=lambda wd: "summary",
        ).recover_reply_promises(fido_dir, 7, registry=fake_registry)
        assert result is True
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "acked"
        task_list = spy_tasks.list()
        assert len(task_list) == 1
        assert task_list[0]["title"] == "task one"
        # The thread dict carries lineage metadata
        # (``lineage_comment_ids`` + ``lineage_key``) so concurrent
        # comments in the same conversation can be deduped against this
        # task without re-fetching the thread.
        assert task_list[0]["thread"] == {
            "repo": "owner/repo",
            "pr": 7,
            "comment_id": 302,
            "url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "author": "owner",
            "comment_type": "issues",
            "lineage_key": "issues:owner/repo:7",
            "lineage_comment_ids": [302],
        }

    def test_recovers_stale_issue_marker_without_reposting(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="webhook", comment_type="issues", anchor_comment_id=303
        )
        assert promise is not None
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comments.return_value = [
            {
                "id": 1,
                "body": f"done\n\n<!-- fido:reply-promise:{promise.promise_id} -->",
            }
        ]

        assert Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).recover_reply_promises(
            fido_dir,
            7,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert store.promise(promise.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.SeenPromiseMarker(),
        )

    def test_recovers_stale_pull_marker_without_reposting(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="webhook", comment_type="pulls", anchor_comment_id=305
        )
        assert promise is not None
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.fetch_comment_thread.return_value = [
            {
                "id": 1,
                "body": f"done\n\n<!-- fido:reply-promise:{promise.promise_id} -->",
            }
        ]

        assert Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).recover_reply_promises(
            fido_dir,
            7,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert store.promise(promise.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.SeenPromiseMarker(),
        )

    def test_deleted_comment_promise_is_removed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "pulls", 205)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = None
        assert not Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).recover_reply_promises(
            fido_dir,
            7,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.AnchorDeleted(),
        )

    def test_deleted_issue_comment_promise_is_removed(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = None
        assert not Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).recover_reply_promises(
            fido_dir,
            7,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.AnchorDeleted(),
        )

    def test_other_pr_promise_is_left_for_later(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "pulls", 205)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = {
            "id": 205,
            "body": "please fix",
            "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/8",
            "html_url": "https://github.com/owner/repo/pull/8#discussion_r205",
            "user": {"login": "owner"},
        }
        assert not Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).recover_reply_promises(
            fido_dir,
            7,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "prepared"
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [205]
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.WrongPullRequest(),
        )

    def test_other_pr_issue_promise_is_left_for_later(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/8",
            "html_url": "https://github.com/owner/repo/pull/8#issuecomment-302",
            "user": {"login": "owner"},
        }
        assert not Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).recover_reply_promises(
            fido_dir,
            7,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "prepared"
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [302]
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.WrongPullRequest(),
        )

    def test_issue_comment_without_pr_url_raises(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "issue_url": "https://api.github.com/repos/owner/repo/not-an-issue",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "user": {"login": "owner"},
        }

        with pytest.raises(ValueError, match="invalid GitHub API URL"):
            Dispatcher(
                _config(tmp_path), _repo_cfg(tmp_path), gh
            ).recover_reply_promises(
                fido_dir,
                7,
                registry=MagicMock(spec=ActivityReporter),
            )
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [302]

    def test_issue_recovery_marks_failed_when_reply_raises(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "user": {"login": "owner"},
        }

        def _failing_synthesis(*args: object, **kwargs: object) -> Never:
            raise RuntimeError("reply failed")

        with pytest.raises(RuntimeError, match="reply failed"):
            Dispatcher(
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                call_synthesis_fn=_failing_synthesis,
            ).recover_reply_promises(
                fido_dir,
                7,
                registry=MagicMock(spec=ActivityReporter),
            )
        assert FidoStore(tmp_path).claim_state(302) == "retryable_failed"
        assert FidoStore(tmp_path).recoverable_promises()[0].state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.ReplayFailed(),
        )

    def test_pull_comment_without_pr_url_raises(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "pulls", 205)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = {
            "id": 205,
            "body": "please fix",
            "pull_request_url": "https://api.github.com/repos/owner/repo/not-a-pr",
            "html_url": "https://github.com/owner/repo/pull/7#discussion_r205",
            "user": {"login": "owner"},
        }

        with pytest.raises(ValueError, match="invalid GitHub API URL"):
            Dispatcher(
                _config(tmp_path), _repo_cfg(tmp_path), gh
            ).recover_reply_promises(
                fido_dir,
                7,
                registry=MagicMock(spec=ActivityReporter),
            )
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [205]

    def test_pull_recovery_marks_failed_when_reply_raises(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "pulls", 205)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_pull_comment.return_value = {
            "id": 205,
            "body": "please fix",
            "path": "foo.py",
            "line": 1,
            "diff_hunk": "@@ @@",
            "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
            "html_url": "https://github.com/owner/repo/pull/7#discussion_r205",
            "user": {"login": "owner"},
        }

        def _failing_synthesis(*args: object, **kwargs: object) -> Never:
            raise RuntimeError("reply failed")

        with pytest.raises(RuntimeError, match="reply failed"):
            Dispatcher(
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                call_synthesis_fn=_failing_synthesis,
            ).recover_reply_promises(
                fido_dir,
                7,
                registry=MagicMock(spec=ActivityReporter),
            )
        assert FidoStore(tmp_path).claim_state(205) == "retryable_failed"
        assert FidoStore(tmp_path).recoverable_promises()[0].state == "failed"
        self._assert_recovery_matches_oracle(
            tmp_path,
            promise,
            oracle.ReplayFailed(),
        )

    def test_defer_recovery_skips_task_creation(self, tmp_path: Path) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "user": {"login": "owner"},
        }
        # Synthesis returning no change_request yields ANSWER category,
        # which does not trigger task creation — equivalent to old DEFER.
        result = Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(),
            sync_fn=lambda *a, **kw: None,
            thread_start_fn=lambda t: None,
            reorder_coalesce_state={},
        ).recover_reply_promises(fido_dir, 7, registry=MagicMock(spec=ActivityReporter))
        assert result is True
        assert FidoStore(tmp_path).promise(promise.promise_id).state == "acked"

    def test_issue_recovery_commits_tasks_before_promise_ack(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        promise = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_issue_comment.return_value = {
            "id": 302,
            "body": "please fix",
            "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
            "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
            "user": {"login": "owner"},
        }

        class _FailingTasks:
            def add(self, **kwargs: object) -> Never:
                raise RuntimeError("task add failed")

        fake_registry = MagicMock(spec=ActivityReporter)
        fake_registry.tasks_for.return_value = _FailingTasks()

        with pytest.raises(RuntimeError, match="task add failed"):
            Dispatcher(
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                    change_request="task one"
                ),
                sync_fn=lambda *a, **kw: None,
                thread_start_fn=lambda t: None,
                reorder_coalesce_state={},
                get_commit_summary_fn=lambda wd: "summary",
            ).recover_reply_promises(fido_dir, 7, registry=fake_registry)
        # Task creation failure must not ack the promise — the worker can retry.
        # The reply was sent before tasks are created, so state is "posted"
        # (reply sent, flow incomplete) rather than "acked" (fully done) or
        # "prepared" (nothing done yet).
        assert FidoStore(tmp_path).promise(promise.promise_id).state != "acked"

    def test_coalesces_review_comment_promises_in_same_thread(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "pulls", 101)
        second = self._prepare_promise(tmp_path, "pulls", 102)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment
        synthesis_bodies: list[str] = []
        fake_registry = MagicMock(spec=ActivityReporter)
        fake_registry.tasks_for.return_value = Tasks(tmp_path)

        def capture_synthesis(comment: str, *args: object, **kwargs: object) -> object:
            synthesis_bodies.append(comment)
            return _synthesis_response(change_request="task a")

        result = Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=capture_synthesis,
            sync_fn=lambda *a, **kw: None,
            thread_start_fn=lambda t: None,
            reorder_coalesce_state={},
            get_commit_summary_fn=lambda wd: "summary",
        ).recover_reply_promises(fido_dir, 7, registry=fake_registry)
        assert result is True
        assert synthesis_bodies[0] == "first\n\n---\n\nsecond"
        assert len(Tasks(tmp_path).list()) == 1
        store = FidoStore(tmp_path)
        assert store.promise(first.promise_id).state == "acked"
        assert store.promise(second.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            first,
            oracle.ReplayPosted(),
        )
        self._assert_recovery_matches_oracle(
            tmp_path,
            second,
            oracle.ReplayPosted(),
        )

    def test_coalesces_issue_comment_promises_in_same_pr_lane(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "issues", 301)
        second = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_issue_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                301: {
                    "id": 301,
                    "body": "first",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-301",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
                302: {
                    "id": 302,
                    "body": "second",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_issue_comment.side_effect = get_issue_comment
        synthesis_bodies: list[str] = []
        fake_registry = MagicMock(spec=ActivityReporter)
        fake_registry.tasks_for.return_value = Tasks(tmp_path)

        def capture_synthesis(comment: str, *args: object, **kwargs: object) -> object:
            synthesis_bodies.append(comment)
            return _synthesis_response(change_request="task a")

        result = Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=capture_synthesis,
            sync_fn=lambda *a, **kw: None,
            thread_start_fn=lambda t: None,
            reorder_coalesce_state={},
            get_commit_summary_fn=lambda wd: "summary",
        ).recover_reply_promises(fido_dir, 7, registry=fake_registry)
        assert result is True
        assert synthesis_bodies[0] == "first\n\n---\n\nsecond"
        assert len(Tasks(tmp_path).list()) == 1
        store = FidoStore(tmp_path)
        assert store.promise(first.promise_id).state == "acked"
        assert store.promise(second.promise_id).state == "acked"
        self._assert_recovery_matches_oracle(
            tmp_path,
            first,
            oracle.ReplayPosted(),
        )
        self._assert_recovery_matches_oracle(
            tmp_path,
            second,
            oracle.ReplayPosted(),
        )

    def test_issue_recovery_replay_records_one_artifact_for_group(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "issues", 301)
        second = self._prepare_promise(tmp_path, "issues", 302)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.get_repo_info.return_value = "owner/repo"
        gh.comment_issue.return_value = {"id": 9001}

        def get_issue_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                301: {
                    "id": 301,
                    "body": "first",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-301",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
                302: {
                    "id": 302,
                    "body": "second",
                    "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
                    "issue_url": "https://api.github.com/repos/owner/repo/issues/7",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_issue_comment.side_effect = get_issue_comment

        assert Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "One combined reply."
            ),
            sync_fn=lambda *a, **kw: None,
            thread_start_fn=lambda t: None,
            reorder_coalesce_state={},
        ).recover_reply_promises(
            fido_dir,
            7,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        store = FidoStore(tmp_path)
        first_artifact = store.artifact_for_promise(first.promise_id)
        second_artifact = store.artifact_for_promise(second.promise_id)
        assert first_artifact is not None
        assert second_artifact is not None
        assert first_artifact == second_artifact
        assert first_artifact.artifact_comment_id == 9001
        assert first_artifact.lane_key == "issues:owner/repo:7"
        assert first_artifact.promise_ids == tuple(
            sorted((first.promise_id, second.promise_id))
        )

    def test_review_recovery_clears_group_promises_before_task_creation(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "pulls", 101)
        second = self._prepare_promise(tmp_path, "pulls", 102)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment

        class _FailingTasks:
            def add(self, **kwargs: object) -> Never:
                # The reply is sent before task creation, so state is "posted"
                # (not "prepared"). Verify promises are not yet "acked" so the
                # worker can retry.
                store = FidoStore(tmp_path)
                assert store.promise(first.promise_id).state != "acked"
                assert store.promise(second.promise_id).state != "acked"
                raise RuntimeError("task add failed")

        fake_registry = MagicMock(spec=ActivityReporter)
        fake_registry.tasks_for.return_value = _FailingTasks()

        with pytest.raises(RuntimeError, match="task add failed"):
            Dispatcher(
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                    change_request="task a"
                ),
                sync_fn=lambda *a, **kw: None,
                thread_start_fn=lambda t: None,
                reorder_coalesce_state={},
                get_commit_summary_fn=lambda wd: "summary",
            ).recover_reply_promises(fido_dir, 7, registry=fake_registry)
        store = FidoStore(tmp_path)
        assert store.promise(first.promise_id).state != "acked"
        assert store.promise(second.promise_id).state != "acked"

    def test_review_recovery_replay_records_one_artifact_for_group(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        first = self._prepare_promise(tmp_path, "pulls", 101)
        second = self._prepare_promise(tmp_path, "pulls", 102)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}
        gh.reply_to_review_comment.return_value = {"id": 9101}

        comments = {
            101: {
                "id": 101,
                "body": "first",
                "path": "foo.py",
                "line": 1,
                "diff_hunk": "@@ @@",
                "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                "user": {"login": "owner"},
            },
            102: {
                "id": 102,
                "body": "second",
                "path": "foo.py",
                "line": 2,
                "diff_hunk": "@@ @@",
                "in_reply_to_id": 101,
                "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                "user": {"login": "owner"},
            },
        }

        gh.get_pull_comment.side_effect = lambda _repo, comment_id: comments[comment_id]
        gh.fetch_comment_thread.side_effect = lambda _repo, _pr, _comment_id: [
            {
                "id": 101,
                "body": "first",
                "author": "owner",
            },
            {
                "id": 102,
                "body": "second",
                "author": "owner",
            },
        ]

        assert Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "One combined review reply."
            ),
            sync_fn=lambda *a, **kw: None,
            thread_start_fn=lambda t: None,
            reorder_coalesce_state={},
        ).recover_reply_promises(
            fido_dir,
            7,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        store = FidoStore(tmp_path)
        first_artifact = store.artifact_for_promise(first.promise_id)
        second_artifact = store.artifact_for_promise(second.promise_id)
        assert first_artifact is not None
        assert second_artifact is not None
        assert first_artifact == second_artifact
        assert first_artifact.artifact_comment_id == 9101
        assert first_artifact.lane_key == "pulls:owner/repo:7:thread:101"
        assert first_artifact.promise_ids == tuple(
            sorted((first.promise_id, second.promise_id))
        )

    def test_recovery_raises_on_invalid_candidate_in_later_group(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "pulls", 101)
        self._prepare_promise(tmp_path, "pulls", 102)
        self._prepare_promise(tmp_path, "pulls", 201)
        self._prepare_promise(tmp_path, "pulls", 999)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
                201: {
                    "id": 201,
                    "body": "third",
                    "path": "bar.py",
                    "line": 3,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r201",
                    "user": {"login": "owner"},
                },
                999: {
                    "id": 999,
                    "body": "ignored",
                    "path": "zap.py",
                    "line": 9,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/not-a-pr",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r999",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment
        synthesis_calls: list[object] = []

        def capture_synthesis(*args: object, **kwargs: object) -> object:
            synthesis_calls.append(args)
            return _synthesis_response()

        with pytest.raises(ValueError, match="invalid GitHub API URL"):
            Dispatcher(
                _config(tmp_path),
                _repo_cfg(tmp_path),
                gh,
                call_synthesis_fn=capture_synthesis,
            ).recover_reply_promises(
                fido_dir,
                7,
                registry=MagicMock(spec=ActivityReporter),
            )
        assert not synthesis_calls
        assert [
            p.anchor_comment_id for p in FidoStore(tmp_path).recoverable_promises()
        ] == [101, 102, 201, 999]

    def test_recovery_skips_handled_candidates_when_processing_later_groups(
        self, tmp_path: Path
    ) -> None:
        fido_dir = tmp_path / ".git" / "fido"
        self._prepare_promise(tmp_path, "pulls", 101)
        self._prepare_promise(tmp_path, "pulls", 102)
        self._prepare_promise(tmp_path, "pulls", 201)
        gh = _make_mock_gh()
        gh.view_issue.return_value = {"title": "My PR", "body": "body"}

        def get_pull_comment(_repo: str, comment_id: int) -> dict[str, object]:
            comments = {
                101: {
                    "id": 101,
                    "body": "first",
                    "path": "foo.py",
                    "line": 1,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r101",
                    "user": {"login": "owner"},
                },
                102: {
                    "id": 102,
                    "body": "second",
                    "path": "foo.py",
                    "line": 2,
                    "diff_hunk": "@@ @@",
                    "in_reply_to_id": 101,
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                    "user": {"login": "owner"},
                },
                201: {
                    "id": 201,
                    "body": "third",
                    "path": "bar.py",
                    "line": 3,
                    "diff_hunk": "@@ @@",
                    "pull_request_url": "https://api.github.com/repos/owner/repo/pulls/7",
                    "html_url": "https://github.com/owner/repo/pull/7#discussion_r201",
                    "user": {"login": "owner"},
                },
            }
            return comments[comment_id]

        gh.get_pull_comment.side_effect = get_pull_comment
        synthesis_calls: list[object] = []

        def capture_synthesis(*args: object, **kwargs: object) -> object:
            synthesis_calls.append(args)
            return _synthesis_response()

        result = Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=capture_synthesis,
        ).recover_reply_promises(
            fido_dir,
            7,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert result is True
        assert len(synthesis_calls) == 2
        assert FidoStore(tmp_path).recoverable_promises() == []


class TestReplayPendingRescopeIntents:
    """Codex P1 (twelfth round) on PR #1938: startup replay drains any
    pending rescope intents orphaned by a crash between the visible
    ACT reply and ``_on_done``.  GitHub will not redeliver a
    successful webhook, so this is the only path that closes the
    lost-rescope window.
    """

    def test_no_pending_returns_zero(self, tmp_path: Path) -> None:
        gh = _make_mock_gh()
        replayed = Dispatcher(
            _config(tmp_path), _repo_cfg(tmp_path), gh
        ).replay_pending_rescope_intents(
            MagicMock(spec=ActivityReporter),
            agent=MagicMock(),
            prompts=Prompts("p"),
        )
        assert replayed == 0

    def test_pending_intent_is_redispatched(self, tmp_path: Path) -> None:
        # Seed a pending row directly via the store: this is the
        # state that a crash-between-claim-and-_on_done leaves
        # behind.
        store = FidoStore(tmp_path)
        assert store.claim_rescope_intent(
            intent_comment_id=909,
            change_request="please refactor the parser",
            intent_timestamp="2024-01-15T10:00:00+00:00",
            author="owner",
            comment_type="issues",
            repo="owner/repo",
            pr_number=7,
        )
        gh = _make_mock_gh()
        thread_starts: list[object] = []
        coalesce_state: dict[str, object] = {}

        replayed = Dispatcher(
            _config(tmp_path),
            _repo_cfg(tmp_path),
            gh,
            thread_start_fn=thread_starts.append,
            reorder_coalesce_state=coalesce_state,
        ).replay_pending_rescope_intents(
            MagicMock(spec=ActivityReporter),
            agent=MagicMock(),
            prompts=Prompts("p"),
        )

        assert replayed == 1
        # reorder_tasks_background was called for the pending row —
        # the coalesce state has an entry for this work_dir and the
        # thread-start fn was invoked (without actually starting a
        # daemon thread).
        assert len(thread_starts) == 1
        assert str(tmp_path) in coalesce_state
        # _on_done has not fired (thread never ran), so the outbox
        # row is still pending — exactly what the next restart
        # would replay again until the rescope completes.
        pending = store.pending_rescope_intents()
        assert [intent.comment_id for intent in pending] == [909]


class TestIsAllowed:
    def _repo_cfg(
        self, tmp_path: Path, collaborators: frozenset[str] = frozenset({"owner"})
    ) -> RepoConfig:
        from fido.config import RepoMembership

        return RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            membership=RepoMembership(collaborators=collaborators),
        )

    def test_collaborator_allowed(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path)
        assert _is_allowed("owner", rc, cfg)

    def test_any_collaborator_allowed(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(
            tmp_path, collaborators=frozenset({"alice", "bob", "rhencke"})
        )
        assert _is_allowed("rhencke", rc, cfg)

    def test_bot_allowed_even_without_collab(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path, collaborators=frozenset())
        assert _is_allowed("copilot[bot]", rc, cfg)

    def test_random_user_denied(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path)
        assert not _is_allowed("rando", rc, cfg)

    def test_empty_collaborators_denies_all_humans(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        rc = self._repo_cfg(tmp_path, collaborators=frozenset())
        assert not _is_allowed("anyone", rc, cfg)


class TestReplyPromiseHelpers:
    def test_reply_promise_ids_deduplicates_context_values(self) -> None:
        assert _reply_promise_ids(
            {
                "reply_promise_id": "one",
                "reply_promise_ids": ["one", "two", "", None],
            }
        ) == ("one", "two")

    def test_reply_promise_ids_handles_missing_context(self) -> None:
        assert _reply_promise_ids(None) == ()

    def test_review_comment_action_carries_lineage(self) -> None:
        action = build_review_comment_action(
            "owner/repo",
            7,
            "PR title",
            "PR body",
            {
                "id": 102,
                "in_reply_to_id": 101,
                "body": "follow up",
                "html_url": "https://github.com/owner/repo/pull/7#discussion_r102",
                "path": "x.py",
                "line": 5,
                "diff_hunk": "@@",
                "user": {"login": "owner"},
            },
        )

        assert action.reply_to is not None
        assert action.reply_to["lineage_key"] == "pulls:owner/repo:7:thread:101"
        assert action.reply_to["lineage_comment_ids"] == [101, 102]
        assert thread_lineage_comment_ids(action.reply_to) == (101, 102)

    def test_issue_comment_action_carries_pr_lineage(self) -> None:
        action = _build_issue_comment_action(
            "owner/repo",
            7,
            "PR title",
            "PR body",
            {
                "id": 302,
                "body": "please fix",
                "html_url": "https://github.com/owner/repo/pull/7#issuecomment-302",
                "user": {"login": "owner"},
            },
        )

        assert action.thread is not None
        assert action.thread["lineage_key"] == "issues:owner/repo:7"
        assert action.thread["lineage_comment_ids"] == [302]
        assert thread_lineage_comment_ids(action.thread) == (302,)

    def test_posted_comment_id_extracts_int_only(self) -> None:
        assert _posted_comment_id({"id": 7}) == 7
        assert _posted_comment_id({"id": "7"}) is None
        assert _posted_comment_id(None) is None

    def test_record_reply_artifact_persists_and_marks_posted(
        self, tmp_path: Path
    ) -> None:
        repo_cfg = _repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=700
        )
        assert promise is not None
        store.claim_reply_outbox_effect(
            promise_id=promise.promise_id,
            delivery_id="delivery-700",
            origin_id=700,
        )

        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=9007,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(promise.promise_id,),
        )

        assert store.promise(promise.promise_id).state == "posted"
        artifact = store.artifact_for_promise(promise.promise_id)
        assert artifact is not None
        assert artifact.artifact_comment_id == 9007

    def test_record_reply_artifact_ignores_missing_comment_id(
        self, tmp_path: Path
    ) -> None:
        repo_cfg = _repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=701
        )
        assert promise is not None

        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=None,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(promise.promise_id,),
        )

        assert store.promise(promise.promise_id).state == "prepared"
        assert store.artifact_for_promise(promise.promise_id) is None

    def test_existing_reply_artifact_requires_every_promise(
        self, tmp_path: Path
    ) -> None:
        repo_cfg = _repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        first = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=702
        )
        second = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=703
        )
        assert first is not None
        assert second is not None

        assert _existing_reply_artifact(repo_cfg, ()) is None
        assert _existing_reply_artifact(repo_cfg, (first.promise_id,)) is None
        store.claim_reply_outbox_effect(
            promise_id=first.promise_id,
            delivery_id="delivery-702",
            origin_id=702,
        )
        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=9702,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(first.promise_id,),
        )

        assert (
            _existing_reply_artifact(repo_cfg, (first.promise_id, second.promise_id))
            is None
        )
        assert _existing_reply_artifact(repo_cfg, (first.promise_id,)) == 9702

    def test_existing_reply_artifact_rejects_split_artifacts(
        self, tmp_path: Path
    ) -> None:
        repo_cfg = _repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        first = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=704
        )
        second = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=705
        )
        assert first is not None
        assert second is not None
        store.claim_reply_outbox_effect(
            promise_id=first.promise_id,
            delivery_id="delivery-704",
            origin_id=704,
        )
        store.claim_reply_outbox_effect(
            promise_id=second.promise_id,
            delivery_id="delivery-705",
            origin_id=705,
        )
        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=9704,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(first.promise_id,),
        )
        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=9705,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(second.promise_id,),
        )

        assert (
            _existing_reply_artifact(repo_cfg, (first.promise_id, second.promise_id))
            is None
        )


class TestDispatchPing:
    def test_returns_none(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "ping", {"hook_id": 123, **_payload()}
        )
        assert result is None


class TestDispatchIssuesAssigned:
    def test_returns_action(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "assigned",
            "assignee": {"login": "fido"},
            "issue": {"number": 1, "title": "test issue"},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "issues", payload
        )
        assert result is not None
        assert "#1" in result.prompt

    def test_zero_number_returns_none(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "assigned",
            "assignee": {"login": "fido"},
            "issue": {"number": 0, "title": "test"},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "issues", payload
        )
        assert result is None


class TestDispatchReviewComment:
    def test_owner_comment(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 123,
                "body": "fix this",
                "user": {"login": "owner"},
                "html_url": "https://example.com",
                "path": "test.py",
                "line": 10,
                "diff_hunk": "@@ -1,3 +1,3 @@",
            },
            "pull_request": {"number": 5, "title": "pr title", "body": "pr body"},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review_comment", payload
        )
        assert result is not None
        assert result.reply_to is None
        assert result.comment_body is None
        assert result.preempts_worker is True
        assert result.thread is not None
        assert result.thread["comment_id"] == 123
        assert result.thread["url"] == "https://example.com"
        assert result.thread["comment_type"] == "pulls"

    def test_reply_to_includes_author(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 124,
                "body": "nit",
                "user": {"login": "owner"},
                "html_url": "https://example.com/comment",
                "path": "test.py",
                "line": 1,
                "diff_hunk": "@@ -1 +1 @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review_comment", payload
        )
        assert result is not None
        assert result.thread is not None
        assert result.thread["author"] == "owner"
        assert result.thread["comment_type"] == "pulls"

    def test_review_comment_webhook_enqueues_fifo_record(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 125,
                "body": "please keep this durable",
                "created_at": "2026-04-30T12:00:00Z",
                "user": {"login": "owner"},
                "html_url": "https://example.com/comment",
                "path": "test.py",
                "line": 1,
                "diff_hunk": "@@ -1 +1 @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }

        result = Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "pull_request_review_comment",
            payload,
            delivery_id="delivery-review-125",
        )

        assert result is not None
        records = FidoStore(tmp_path).pending_pr_comments(repo="owner/repo")
        assert len(records) == 1
        record = records[0]
        assert record.delivery_id == "delivery-review-125"
        assert record.pr_number == 5
        assert record.comment_type == "pulls"
        assert record.comment_id == 125
        assert record.author == "owner"
        assert record.body == "please keep this durable"
        assert record.github_created_at == "2026-04-30T12:00:00Z"
        assert record.payload_json

    def test_review_comment_webhook_deduplicates_fifo_record(
        self, tmp_path: Path
    ) -> None:
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 126,
                "body": "only once",
                "created_at": "2026-04-30T12:00:00Z",
                "user": {"login": "owner"},
                "html_url": "https://example.com/comment",
                "path": "test.py",
                "line": 1,
                "diff_hunk": "@@ -1 +1 @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }

        Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "pull_request_review_comment",
            payload,
            delivery_id="delivery-review-126-a",
        )
        Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "pull_request_review_comment",
            payload,
            delivery_id="delivery-review-126-b",
        )

        records = FidoStore(tmp_path).pending_pr_comments(repo="owner/repo")
        assert [record.comment_id for record in records] == [126]

    def test_review_comment_edit_updates_fifo_record(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 127,
                "body": "original",
                "created_at": "2026-04-30T12:00:00Z",
                "user": {"login": "owner"},
                "html_url": "https://example.com/comment",
                "path": "test.py",
                "line": 1,
                "diff_hunk": "@@ -1 +1 @@",
            },
            "pull_request": {"number": 5, "title": "My PR", "body": ""},
        }

        Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "pull_request_review_comment",
            payload,
            delivery_id="delivery-review-127-a",
        )
        edited_payload = {
            **payload,
            "action": "edited",
            "comment": {
                **payload["comment"],
                "body": "edited before drain",
                "updated_at": "2026-04-30T12:05:00Z",
            },
        }
        result = Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "pull_request_review_comment",
            edited_payload,
            delivery_id="delivery-review-127-b",
        )

        assert result is not None
        records = FidoStore(tmp_path).pending_pr_comments(repo="owner/repo")
        assert len(records) == 1
        assert records[0].delivery_id == "delivery-review-127-b"
        assert records[0].body == "edited before drain"
        assert records[0].github_created_at == "2026-04-30T12:00:00Z"

    def test_self_comment_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "done", "user": {"login": "FidoCanCode"}},
            "pull_request": {"number": 5},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review_comment", payload
        )
        assert result is None

    def test_unallowed_user_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "rando"}},
            "pull_request": {"number": 5},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review_comment", payload
        )
        assert result is None


class TestDispatchCheckRun:
    def test_failure(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "completed",
            "check_run": {
                "conclusion": "failure",
                "name": "test",
                "pull_requests": [{"number": 3}],
            },
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "check_run", payload
        )
        assert result is not None
        assert "CI failure" in result.prompt
        assert result.preempts_worker is True

    def test_success_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "completed",
            "check_run": {"conclusion": "success", "name": "lint", "pull_requests": []},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "check_run", payload
        )
        assert result is None


class TestDispatchPullRequest:
    def test_merged(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        FidoStore(tmp_path).enqueue_pr_comment(
            delivery_id="delivery-queued",
            repo="owner/repo",
            pr_number=7,
            comment_type="issues",
            comment_id=900,
            author="owner",
            is_bot=False,
            body="queued",
            github_created_at="2026-04-30T12:00:00Z",
        )
        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 7, "merged": True},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request", payload
        )
        assert result is not None
        assert "merged" in result.prompt
        assert FidoStore(tmp_path).pending_pr_comments(repo="owner/repo") == []

    def test_closed_not_merged(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        FidoStore(tmp_path).enqueue_pr_comment(
            delivery_id="delivery-queued",
            repo="owner/repo",
            pr_number=7,
            comment_type="issues",
            comment_id=900,
            author="owner",
            is_bot=False,
            body="queued",
            github_created_at="2026-04-30T12:00:00Z",
        )
        payload = {
            **_payload(),
            "action": "closed",
            "pull_request": {"number": 7, "merged": False},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request", payload
        )
        assert result is None
        assert FidoStore(tmp_path).pending_pr_comments(repo="owner/repo") == []


class TestDispatchIssueComment:
    def test_pr_comment(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 456,
                "body": "looks good",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/10#issuecomment-456",
            },
            "issue": {
                "number": 10,
                "title": "test pr",
                "body": "desc",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "issue_comment", payload
        )
        assert result is not None
        assert result.comment_body is None
        assert result.preempts_worker is True
        assert result.thread is not None
        assert (
            result.thread["url"]
            == "https://github.com/owner/repo/pull/10#issuecomment-456"
        )
        assert result.thread["author"] == "owner"
        assert result.thread["comment_type"] == "issues"

    def test_pr_issue_comment_webhook_enqueues_fifo_record(
        self, tmp_path: Path
    ) -> None:
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 457,
                "body": "top-level durability",
                "created_at": "2026-04-30T12:01:00Z",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/10#issuecomment-457",
            },
            "issue": {
                "number": 10,
                "title": "test pr",
                "body": "desc",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }

        result = Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "issue_comment",
            payload,
            delivery_id="delivery-issue-457",
        )

        assert result is not None
        records = FidoStore(tmp_path).pending_pr_comments(repo="owner/repo")
        assert len(records) == 1
        record = records[0]
        assert record.delivery_id == "delivery-issue-457"
        assert record.pr_number == 10
        assert record.comment_type == "issues"
        assert record.comment_id == 457
        assert record.author == "owner"
        assert record.body == "top-level durability"
        assert record.github_created_at == "2026-04-30T12:01:00Z"

    def test_pr_issue_comment_edit_updates_fifo_record(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 458,
                "body": "original",
                "created_at": "2026-04-30T12:01:00Z",
                "user": {"login": "owner"},
                "html_url": "https://github.com/owner/repo/pull/10#issuecomment-458",
            },
            "issue": {
                "number": 10,
                "title": "test pr",
                "body": "desc",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }

        Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "issue_comment", payload, delivery_id="delivery-a"
        )
        edited_payload = {
            **payload,
            "action": "edited",
            "comment": {
                **payload["comment"],
                "body": "edited before drain",
                "updated_at": "2026-04-30T12:05:00Z",
            },
        }
        result = Dispatcher(cfg, repo_cfg, MagicMock()).dispatch(
            "issue_comment", edited_payload, delivery_id="delivery-b"
        )

        assert result is not None
        records = FidoStore(tmp_path).pending_pr_comments(repo="owner/repo")
        assert len(records) == 1
        assert records[0].delivery_id == "delivery-b"
        assert records[0].body == "edited before drain"
        assert records[0].github_created_at == "2026-04-30T12:01:00Z"

    def test_non_pr_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "owner"}},
            "issue": {"number": 10, "title": "issue"},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "issue_comment", payload
        )
        assert result is None


class TestDispatchUnknown:
    def test_unknown_event(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "unknown_event",
            {**_payload(), "action": "whatever"},
        )
        assert result is None


class TestReplyToComment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": self._repo_cfg(tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def _mock_gh(self) -> MagicMock:
        """Return a MagicMock pre-configured with the gh return shapes
        ``reply_to_comment`` actually depends on.

        The reply-outbox protocol asserts ``_posted_comment_id(posted) is not None``
        when promise_ids are queued, so ``reply_to_review_comment`` must return
        ``{"id": int}`` rather than the auto-mocked sub-MagicMock.  Tests that
        want a different shape override after construction.
        """
        gh = _make_mock_gh()
        gh.reply_to_review_comment.return_value = {"id": 90_001}
        gh.comment_issue.return_value = {"id": 90_002}
        gh.fetch_comment_thread.return_value = []
        return gh

    def test_no_reply_to_returns_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(prompt="do stuff")
        gh = _make_mock_gh()
        cat, titles = Dispatcher(cfg, self._repo_cfg(tmp_path), gh).reply_to_comment(
            action,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"

    def test_no_comment_body_returns_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="something",
            reply_to={"repo": "a/b", "pr": 1, "comment_id": 5},
        )
        gh = _make_mock_gh()
        cat, titles = Dispatcher(cfg, self._repo_cfg(tmp_path), gh).reply_to_comment(
            action,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"

    def _mock_gh(self) -> MagicMock:
        """Return a MagicMock GitHub client with reply methods configured."""
        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = []
        mock_gh.reply_to_review_comment.return_value = {"id": 999}
        mock_gh.comment_issue.return_value = {"id": 999}
        return mock_gh

    def test_full_flow_act(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 10},
            comment_body="please add logging",
            is_bot=False,
            context={
                "pr_title": "My PR",
                "file": "foo.py",
                "line": 5,
                "diff_hunk": "@@ @@",
            },
        )

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                reply_text="I will add logging.",
                change_request="Add logging to the request handler",
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert "logging" in titles[0].lower()

    def test_claims_review_reply_outbox_before_posting(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="pulls", anchor_comment_id=10
        )
        assert promise is not None
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 10},
            comment_body="please add logging",
            is_bot=False,
            context={
                "delivery_id": "github-delivery-10",
                "reply_promise_id": promise.promise_id,
            },
        )

        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = []

        def reply_to_review_comment(
            repo: str, pr: int, body: str, comment_id: int
        ) -> object:
            effect = store.reply_outbox_effect(promise.promise_id)
            assert effect is not None
            assert effect.delivery_id == "github-delivery-10"
            assert effect.origin_id == 10
            assert effect.state == "claimed"
            assert effect.external_id is None
            return {"id": 9010}

        mock_gh.reply_to_review_comment.side_effect = reply_to_review_comment

        Dispatcher(
            cfg,
            repo_cfg,
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response("Yep."),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        effect = store.reply_outbox_effect(promise.promise_id)
        assert effect is not None
        assert effect.state == "delivered"
        assert effect.external_id == 9010

    def test_rejects_review_reply_when_outbox_already_claimed(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="pulls", anchor_comment_id=10
        )
        assert promise is not None
        store.claim_reply_outbox_effect(
            promise_id=promise.promise_id,
            delivery_id="github-delivery-10",
            origin_id=10,
        )
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 10},
            comment_body="please add logging",
            is_bot=False,
            context={
                "delivery_id": "github-delivery-10",
                "reply_promise_id": promise.promise_id,
            },
        )

        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = []

        with pytest.raises(RuntimeError, match="already claimed"):
            Dispatcher(
                cfg,
                repo_cfg,
                mock_gh,
                call_synthesis_fn=lambda *a, **kw: _synthesis_response("Yep."),
            ).reply_to_comment(
                action,
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
            )

        mock_gh.reply_to_review_comment.assert_not_called()

    def test_full_flow_ask(self, tmp_path: Path) -> None:
        """Synthesis path: no change_request → ANSWER (replaces old ASK category)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 11},
            comment_body="can you clarify?",
            is_bot=False,
        )

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "What specifically?"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"

    def test_synthesis_path_does_not_resolve_review_thread(
        self, tmp_path: Path
    ) -> None:
        """Synthesis path never resolves review threads."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 11},
            comment_body="please defer",
            is_bot=False,
        )
        gh = _make_mock_gh()
        gh.fetch_comment_thread.return_value = [
            {"id": 11, "author": "owner", "body": "please defer"}
        ]
        gh.reply_to_review_comment.return_value = {"id": 88}

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response("Handled."),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        gh.resolve_thread.assert_not_called()

    def test_apply_reply_result_skips_non_task_issue_categories(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)

        Dispatcher(cfg, repo_cfg, MagicMock())._apply_reply_result(
            "ASK",
            ["ignored"],
            thread=None,
            registry=None,
        )
        # ASK category must not create any tasks.
        from fido.tasks import Tasks

        assert Tasks(tmp_path).list() == []

    def test_apply_reply_result_preserves_triggering_comment_link(
        self, tmp_path: Path
    ) -> None:
        """Task metadata links to the comment that requested work."""
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        thread = {
            "repo": "owner/repo",
            "pr": 1,
            "comment_id": 102,
            "url": "https://github.com/owner/repo/pull/1#discussion_r102",
            "author": "rhencke",
            "comment_type": "pulls",
        }
        from fido.tasks import Tasks

        spy_tasks = Tasks(tmp_path)
        fake_registry = MagicMock(spec=ActivityReporter)
        fake_registry.tasks_for.return_value = spy_tasks

        Dispatcher(
            cfg,
            repo_cfg,
            MagicMock(),
            sync_fn=lambda *a, **kw: None,
            thread_start_fn=lambda t: None,
            reorder_coalesce_state={},
            get_commit_summary_fn=lambda wd: "summary",
        )._apply_reply_result(
            "ACT",
            ["Remove redundant empty-list concatenation"],
            thread=thread,
            registry=fake_registry,
        )
        tasks = spy_tasks.list()
        assert len(tasks) == 1
        assert tasks[0]["thread"]["comment_id"] == 102
        assert (
            tasks[0]["thread"]["url"]
            == "https://github.com/owner/repo/pull/1#discussion_r102"
        )

    def test_full_flow_answer(self, tmp_path: Path) -> None:
        """Synthesis path: no change_request → ANSWER."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 12},
            comment_body="why did you do this?",
            is_bot=False,
        )

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "I did this because..."
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"
        assert titles == []

    def test_full_flow_do(self, tmp_path: Path) -> None:
        """Synthesis path: change_request present → ACT (replaces old DO for bots)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 13},
            comment_body="cache the results for performance",
            is_bot=True,
        )

        mock_gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "On it!",
                change_request="Cache results for performance",
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert titles == ["Cache results for performance"]

    def test_full_flow_defer(self, tmp_path: Path) -> None:
        """Synthesis path: change_request with scope description → ACT (replaces old DEFER)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 14},
            comment_body="refactor everything",
            is_bot=False,
        )

        mock_gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "That's noted for a future PR.",
                change_request="Refactor everything in a separate PR",
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert "refactor" in titles[0].lower()
        # Synthesis path never calls create_issue
        mock_gh.create_issue.assert_not_called()

    def test_full_flow_dump(self, tmp_path: Path) -> None:
        """Synthesis path: no change_request → ANSWER (replaces old DUMP)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 14},
            comment_body="use a different language",
            is_bot=False,
        )

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Not applicable here."
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"

    def test_empty_reply_body_raises(self, tmp_path: Path) -> None:
        """Synthesis exhausted error propagates fail-closed."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 15},
            comment_body="do something",
            is_bot=False,
        )

        def _raise_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisExhaustedError("exhausted")

        with pytest.raises(SynthesisExhaustedError):
            gh = MagicMock()
            Dispatcher(
                cfg,
                self._repo_cfg(tmp_path),
                gh,
                call_synthesis_fn=_raise_exhausted,
            ).reply_to_comment(
                action,
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
            )

    def test_claim_race_returns_act_with_no_titles(self, tmp_path: Path) -> None:
        """Second call with same comment_id is blocked by SQLite claim.

        Must return empty titles so the server creates no phantom tasks —
        the process that owns the claim will handle reply and task creation.
        """
        cfg = self._cfg(tmp_path)
        cid = 999
        assert FidoStore(tmp_path).prepare_reply(
            owner="webhook", comment_type="pulls", anchor_comment_id=cid
        )
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": cid},
            comment_body="competing update",
            is_bot=False,
        )
        gh = _make_mock_gh()
        cat, titles = Dispatcher(cfg, self._repo_cfg(tmp_path), gh).reply_to_comment(
            action,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert titles == []

    def test_no_comment_id_skips_lock(self, tmp_path: Path) -> None:
        """When comment_id is None, lock is skipped; synthesis still called."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": None},
            comment_body="some comment",
            is_bot=False,
        )

        gh = MagicMock()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "ok", change_request="Do it"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"

    def test_act_title_comes_from_synthesis_change_request(
        self, tmp_path: Path
    ) -> None:
        """ACT task title is the change_request from the synthesis response."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 77},
            comment_body="add tests and update docs",
            is_bot=False,
        )

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "On it!",
                change_request="Add tests and update docs",
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert titles == ["Add tests and update docs"]

    def test_synthesis_reply_body_is_posted(self, tmp_path: Path) -> None:
        """The reply text from synthesis is what gets posted to GitHub."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 42},
            comment_body="please fix the parser",
            is_bot=False,
        )

        mock_gh = self._mock_gh()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 42, "author": "rhencke", "body": "please fix the parser"},
        ]

        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "I'll fix the parser right away.",
                change_request="Fix the parser",
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert titles == ["Fix the parser"]
        reply_args = mock_gh.reply_to_review_comment.call_args.args
        assert reply_args[:2] == ("owner/repo", 1)
        assert "I'll fix the parser right away." in reply_args[2]
        assert "fido:reply-promise:" in reply_args[2]
        # No prior Fido reply in thread — a new reply is posted
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_posts_new_reply_when_human_comments_after_fido(
        self, tmp_path: Path
    ) -> None:
        """When a human posts after Fido's reply, post a new reply rather than editing."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 300},
            comment_body="We need sub issues in priority order.",
            is_bot=False,
        )

        mock_gh = self._mock_gh()
        # Thread: root → fido reply → NEW human comment (fido must not edit)
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 300, "author": "rhencke", "body": "Add orderBy"},
            {"id": 301, "author": "fidocancode", "body": "Got it!"},
            {
                "id": 302,
                "author": "rhencke",
                "body": "We need sub issues in priority order.",
            },
        ]
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "On it!", change_request="Reorder sub issues by priority"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        # Human spoke last — must post a fresh reply, never edit the old one
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_answer_reply_posts_synthesis_text(self, tmp_path: Path) -> None:
        """ANSWER replies post synthesis reply_text directly."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 44},
            comment_body="Sure, sounds good",
            is_bot=False,
        )

        mock_gh = self._mock_gh()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 200, "author": "reviewer", "body": "What do you think?"},
            {"id": 201, "author": "fidocancode", "body": "Sure, sounds good"},
        ]
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Could you clarify?"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"
        # Posted replies are immutable; answer replies also post a new artifact.
        reply_args = mock_gh.reply_to_review_comment.call_args.args
        assert reply_args[:2] == ("owner/repo", 1)
        assert "Could you clarify?" in reply_args[2]
        assert "fido:reply-promise:" in reply_args[2]
        mock_gh.edit_review_comment.assert_not_called()

    def test_call_synthesis_receives_issue_and_pr_context(self, tmp_path: Path) -> None:
        """reply_to_comment passes active issue/PR context to call_synthesis."""
        cfg = self._cfg(tmp_path)
        # Set up state.json so _load_active_context_for_rescope finds active issue.
        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        State(fido_dir).save({"issue": 7})
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 10},
            comment_body="please add logging",
            is_bot=False,
        )

        mock_gh = self._mock_gh()
        mock_gh.view_issue.return_value = {"title": "Fix crash", "body": "It crashes."}

        captured_calls: list[dict] = []

        def capture_synthesis(*args: object, **kwargs: object) -> object:
            captured_calls.append(kwargs)
            return _synthesis_response("I will add logging.")

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=capture_synthesis,
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert captured_calls
        call_kwargs = captured_calls[0]
        assert call_kwargs.get("issue") is not None
        assert call_kwargs["issue"].title == "Fix crash"

    def test_no_active_context_when_no_state(self, tmp_path: Path) -> None:
        """reply_to_comment passes None issue/pr to call_synthesis when no state.json."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 10},
            comment_body="please add logging",
            is_bot=False,
        )

        mock_gh = self._mock_gh()

        captured_calls: list[dict] = []

        def capture_synthesis(*args: object, **kwargs: object) -> object:
            captured_calls.append(kwargs)
            return _synthesis_response("I will add logging.")

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=capture_synthesis,
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert captured_calls
        call_kwargs = captured_calls[0]
        assert call_kwargs.get("issue") is None


class TestReplyToReview:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_is_a_no_op_for_inline_comments(self, tmp_path: Path) -> None:
        """Inline comments are now exclusively handled by the per-comment
        webhook (``pull_request_review_comment``).  ``reply_to_review`` no
        longer iterates the inline comments — closes #518 (double-reply on
        review submission)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="review",
            review_comments={"repo": "owner/repo", "pr": 5, "review_id": 777},
        )
        mock_gh = MagicMock()
        reply_to_review(action, cfg, self._repo_cfg(tmp_path), mock_gh, agent=_client())
        # Doesn't fetch, doesn't post — no GitHub side effects at all.
        mock_gh.get_review_comments.assert_not_called()
        mock_gh.reply_to_review_comment.assert_not_called()

    def test_no_op_with_no_review_comments(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = Action(prompt="review", review_comments=None)
        # should return without error
        reply_to_review(action, cfg, self._repo_cfg(tmp_path), _make_mock_gh())


class TestReplyToCommentSynthesisFallback:
    """Cover the synthesis-exhausted → call_failure_explanation paths in
    both reply_to_comment and reply_to_issue_comment, and the eyes-add
    failure best-effort branch in reply_to_issue_comment."""

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": self._repo_cfg(tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def _mock_gh(self) -> MagicMock:
        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.comment_issue.return_value = {"id": 9999}
        mock_gh.reply_to_review_comment.return_value = {"id": 9999}
        # Bug-marker search defaults to "no existing bug" so HOL-20
        # routes proceed to create_issue (codex on PR #1932: the
        # bug-filing idempotency search would otherwise hit the
        # MagicMock truthy default and short-circuit the create).
        mock_gh.search_issues.return_value = []
        return mock_gh

    def test_review_comment_falls_back_when_synthesis_exhausted(
        self, tmp_path: Path
    ) -> None:
        from fido.synthesis_call import SynthesisExhaustedError

        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 555},
            comment_body="please rephrase",
            is_bot=False,
        )
        mock_gh = self._mock_gh()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 555, "author": "rhencke", "body": "please rephrase"},
        ]

        fallback = _synthesis_response(
            "I tried to respond but my structured-output turn failed."
        )
        fallback_calls: list[tuple[object, ...]] = []

        def fake_fallback(*args: object, **kwargs: object) -> object:
            fallback_calls.append(args)
            return fallback

        def _raise_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisExhaustedError("retries exhausted")

        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=_raise_exhausted,
            call_failure_explanation_fn=fake_fallback,
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"
        assert titles == []
        assert len(fallback_calls) == 1
        # Fallback reply was posted to GitHub.
        assert mock_gh.reply_to_review_comment.called
        reply_body = mock_gh.reply_to_review_comment.call_args.args[2]
        assert "structured-output turn failed" in reply_body

    def test_review_comment_clears_eyes_when_fallback_also_exhausts(
        self, tmp_path: Path
    ) -> None:
        from fido.synthesis_call import SynthesisExhaustedError

        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 556},
            comment_body="please rephrase",
            is_bot=False,
        )
        mock_gh = self._mock_gh()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 556, "author": "rhencke", "body": "please rephrase"},
        ]

        remove_eyes_calls: list[object] = []

        class _FakeExecutor:
            def remove_eyes_reaction(self, target: object) -> None:
                remove_eyes_calls.append(target)

            def execute_effects_only(self, *args: object, **kwargs: object) -> object:
                raise AssertionError("execute_effects_only should not be called")

        def _raise_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisExhaustedError("retries exhausted")

        def _raise_fallback_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisExhaustedError("fallback also exhausted")

        with pytest.raises(SynthesisExhaustedError, match="fallback also"):
            Dispatcher(
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                call_synthesis_fn=_raise_exhausted,
                call_failure_explanation_fn=_raise_fallback_exhausted,
            ).reply_to_comment(
                action,
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
                _executor=_FakeExecutor(),  # type: ignore[arg-type]
            )
        assert len(remove_eyes_calls) == 1

    def test_review_comment_critic_exhausted_routes_to_blocked(
        self, tmp_path: Path
    ) -> None:
        """HOL-20 / #1914: when ``call_synthesis`` raises the critic
        subclass, ``reply_to_comment`` skips the "please rephrase"
        fallback and routes through ``_route_critic_exhausted_blocked``
        — BLOCKED comment + auto-filed bug, NO fallback reply."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 777},
            comment_body="something the critic hates",
            is_bot=False,
        )
        mock_gh = self._mock_gh()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 777, "author": "rhencke", "body": "x"},
        ]
        mock_gh.create_issue.return_value = (
            "https://github.com/FidoCanCode/home/issues/1"
        )

        fallback_calls: list[object] = []

        def fake_fallback(*args: object, **kwargs: object) -> object:
            fallback_calls.append(args)
            return _synthesis_response("should not be posted")

        def _raise_critic_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisCriticExhaustedError(
                "intent-coverage",
                ["missing X", "still missing X", "STILL missing X"],
                ["v1 preview", "v2 preview", "v3 preview"],
            )

        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=_raise_critic_exhausted,
            call_failure_explanation_fn=fake_fallback,
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        # BLOCKED category, no fallback fired, no review reply posted.
        assert cat == "BLOCKED"
        assert titles == []
        assert fallback_calls == []
        assert not mock_gh.reply_to_review_comment.called
        # BLOCKED comment posted on the source PR (issue-level comment).
        assert mock_gh.comment_issue.called
        blocked_args = mock_gh.comment_issue.call_args.args
        assert blocked_args[0] == "owner/repo"
        assert blocked_args[1] == 1
        assert "BLOCKED" in blocked_args[2]
        # Bug auto-filed against FidoCanCode/home.
        bug_args = mock_gh.create_issue.call_args.args
        assert bug_args[0] == "FidoCanCode/home"
        assert "intent-coverage" in bug_args[1]
        # Codex on PR #1932: the direct_promise the dispatcher created
        # MUST be acked, not left dangling.  A dangling promise keeps
        # the comment permanently claimed — every webhook redelivery
        # would short-circuit before reaching the synthesis call,
        # silently swallowing the work.  ``recoverable_promises``
        # returns claimed-but-not-acked promises; for a cleanly
        # routed BLOCKED comment, the list must be empty.
        assert FidoStore(tmp_path).recoverable_promises() == []

    def test_review_comment_critic_exhausted_post_failure_propagates(
        self, tmp_path: Path
    ) -> None:
        """Codex on PR #1932: when the BLOCKED comment post fails, the
        critic exception must propagate so queued callers
        (``queue_reply_tasks`` and the direct webhook caller in
        server.py) hit their error branch and mark the promises
        FAILED rather than acking them silently.  A failed post with
        a silent ack would leave no visible signal AND no recoverable
        promise."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 778},
            comment_body="something the critic hates",
            is_bot=False,
        )
        mock_gh = self._mock_gh()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 778, "author": "rhencke", "body": "x"},
        ]
        mock_gh.create_issue.return_value = (
            "https://github.com/FidoCanCode/home/issues/2"
        )
        # BLOCKED comment post fails — helper returns False, dispatcher
        # must re-raise so the queued caller marks the promise failed.
        mock_gh.comment_issue.side_effect = RuntimeError("GitHub 503")

        def _raise_critic_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisCriticExhaustedError(
                "intent-coverage", ["g1", "g2", "g3"], ["v1", "v2", "v3"]
            )

        with pytest.raises(SynthesisCriticExhaustedError):
            Dispatcher(
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                call_synthesis_fn=_raise_critic_exhausted,
                call_failure_explanation_fn=lambda *a, **kw: _synthesis_response(
                    "should-not-fire"
                ),
            ).reply_to_comment(
                action,
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
            )
        # Promise must NOT be acked — it stays in the recoverable set
        # so the queued/recovery caller (or the server's catch) can
        # mark it failed and retry later.
        assert FidoStore(tmp_path).recoverable_promises() != []

    def test_issue_comment_critic_exhausted_post_failure_propagates(
        self, tmp_path: Path
    ) -> None:
        """Issue-comment sibling of
        ``test_review_comment_critic_exhausted_post_failure_propagates``:
        when the BLOCKED post fails, the critic exception must
        propagate so the queued/recovery caller marks the promise
        failed instead of acking it silently."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\ntrigger",
            comment_body="trigger",
            is_bot=False,
            context={"pr_title": "My PR", "comment_id": 889},
        )
        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.create_issue.return_value = (
            "https://github.com/FidoCanCode/home/issues/3"
        )
        mock_gh.comment_issue.side_effect = RuntimeError("GitHub 503")

        def _raise_critic_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisCriticExhaustedError(
                "reply-prose", ["g1", "g2"], ["v1", "v2"]
            )

        with pytest.raises(SynthesisCriticExhaustedError):
            Dispatcher(
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                call_synthesis_fn=_raise_critic_exhausted,
                call_failure_explanation_fn=lambda *a, **kw: _synthesis_response(
                    "should-not-fire"
                ),
            ).reply_to_issue_comment(
                action,
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
            )

    def test_issue_comment_critic_exhausted_routes_to_blocked(
        self, tmp_path: Path
    ) -> None:
        """HOL-20 sibling for the top-level issue-comment path."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\ntrigger",
            comment_body="trigger",
            is_bot=False,
            context={"pr_title": "My PR", "comment_id": 888},
        )
        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.comment_issue.return_value = {"id": 8889}
        mock_gh.search_issues.return_value = []  # HOL-20 bug-marker search
        mock_gh.create_issue.return_value = (
            "https://github.com/FidoCanCode/home/issues/2"
        )

        def _raise_critic_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisCriticExhaustedError(
                "reply-prose",
                ["bad SHA", "still bad SHA"],
                ["v1", "v2"],
            )

        cat, _titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=_raise_critic_exhausted,
            call_failure_explanation_fn=lambda *a, **kw: _synthesis_response(
                "should-not-fire"
            ),
        ).reply_to_issue_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "BLOCKED"
        # comment_issue fires for the BLOCKED notice (no fallback post).
        # The HOL-20 route also calls create_issue for the auto-filed bug.
        assert mock_gh.comment_issue.called
        assert mock_gh.create_issue.called
        bug_args = mock_gh.create_issue.call_args.args
        assert "reply-prose" in bug_args[1]
        # Promise must be acked (see review-comment sibling test).
        assert FidoStore(tmp_path).recoverable_promises() == []

    def test_issue_comment_falls_back_when_synthesis_exhausted(
        self, tmp_path: Path
    ) -> None:
        from fido.synthesis_call import SynthesisExhaustedError

        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nplease fix",
            comment_body="please fix",
            is_bot=False,
            context={"pr_title": "My PR", "comment_id": 700},
        )

        fallback = _synthesis_response("Sorry — please rephrase your comment.")
        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.comment_issue.return_value = {"id": 8888}

        fallback_calls: list[tuple[object, ...]] = []

        def fake_fallback(*args: object, **kwargs: object) -> object:
            fallback_calls.append(args)
            return fallback

        def _raise_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisExhaustedError("retries exhausted")

        cat, _titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=_raise_exhausted,
            call_failure_explanation_fn=fake_fallback,
        ).reply_to_issue_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"
        assert len(fallback_calls) == 1
        # Fallback was posted as a top-level issue comment.
        assert mock_gh.comment_issue.called
        body = mock_gh.comment_issue.call_args.args[2]
        assert "please rephrase" in body


class TestReplyToIssueComment:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": self._repo_cfg(tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def _action(
        self, comment: object = "please fix", is_bot: bool = False, cid: int = 42
    ) -> object:
        return Action(
            prompt="PR top-level comment on #7 by owner:\n\nplease fix",
            comment_body=comment,
            is_bot=is_bot,
            context={"pr_title": "My PR", "comment_id": cid},
        )

    def _mock_gh(self) -> MagicMock:
        """Return a MagicMock GitHub client with reply methods configured."""
        mock_gh = MagicMock()
        mock_gh.get_repo_info.return_value = "owner/repo"
        mock_gh.comment_issue.return_value = {"id": 9999}
        return mock_gh

    def test_act_reply(self, tmp_path: Path) -> None:
        """Synthesis path: change_request present → ACT."""
        cfg = self._cfg(tmp_path)

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "I'll fix that.", change_request="Fix the bug"
            ),
        ).reply_to_issue_comment(
            self._action(),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert titles == ["Fix the bug"]

    def test_ask_reply(self, tmp_path: Path) -> None:
        """Synthesis path: no change_request → ANSWER (replaces old ASK)."""
        cfg = self._cfg(tmp_path)

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response("What do you mean?"),
        ).reply_to_issue_comment(
            self._action("unclear"),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"

    def test_answer_reply(self, tmp_path: Path) -> None:
        """Synthesis path: no change_request → ANSWER."""
        cfg = self._cfg(tmp_path)

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response("Yes, because..."),
        ).reply_to_issue_comment(
            self._action("why?"),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"

    def test_claims_issue_reply_outbox_before_posting(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=42
        )
        assert promise is not None
        action = self._action("why?")
        action.context = {
            **(action.context or {}),
            "delivery_id": "github-delivery-42",
            "reply_promise_id": promise.promise_id,
        }

        mock_gh = self._mock_gh()

        def comment_issue(repo: str, number: int, body: str) -> object:
            effect = store.reply_outbox_effect(promise.promise_id)
            assert effect is not None
            assert effect.delivery_id == "github-delivery-42"
            assert effect.origin_id == 42
            assert effect.state == "claimed"
            assert effect.external_id is None
            return {"id": 9042}

        mock_gh.comment_issue.side_effect = comment_issue

        Dispatcher(
            cfg,
            repo_cfg,
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response("Yes, because..."),
        ).reply_to_issue_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        effect = store.reply_outbox_effect(promise.promise_id)
        assert effect is not None
        assert effect.state == "delivered"
        assert effect.external_id == 9042

    def test_dump_reply(self, tmp_path: Path) -> None:
        """Synthesis path: no change_request → ANSWER (replaces old DUMP)."""
        cfg = self._cfg(tmp_path)

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "That won't work here."
            ),
        ).reply_to_issue_comment(
            self._action("do it differently"),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"

    def test_defer_reply(self, tmp_path: Path) -> None:
        """Synthesis path: change_request with scope description → ACT (replaces old DEFER)."""
        cfg = self._cfg(tmp_path)

        mock_gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Out of scope for now.",
                change_request="Big refactor in separate PR",
            ),
        ).reply_to_issue_comment(
            self._action("big refactor"),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        assert "refactor" in titles[0].lower()
        # Synthesis path never calls create_issue
        mock_gh.create_issue.assert_not_called()

    def test_skips_issue_reply_when_artifact_already_recorded(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="issues", anchor_comment_id=44
        )
        assert promise is not None
        store.claim_reply_outbox_effect(
            promise_id=promise.promise_id,
            delivery_id="delivery-44",
            origin_id=44,
        )
        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=9044,
            comment_type="issues",
            lane_key="issues:owner/repo:7",
            promise_ids=(promise.promise_id,),
        )

        mock_gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            repo_cfg,
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response("Yep."),
        ).reply_to_issue_comment(
            Action(
                prompt="PR top-level comment on #7 by owner:\n\nplease fix",
                comment_body="please fix",
                is_bot=False,
                context={
                    "pr_title": "My PR",
                    "comment_id": 44,
                    "reply_promise_id": promise.promise_id,
                },
            ),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        assert cat == "ANSWER"
        assert titles == []
        mock_gh.comment_issue.assert_not_called()

    def test_empty_reply_body_raises(self, tmp_path: Path) -> None:
        """Synthesis exhausted error propagates fail-closed."""
        cfg = self._cfg(tmp_path)

        def _raise_exhausted(*args: object, **kwargs: object) -> Never:
            raise SynthesisExhaustedError("exhausted")

        with pytest.raises(SynthesisExhaustedError):
            gh = self._mock_gh()
            Dispatcher(
                cfg,
                self._repo_cfg(tmp_path),
                gh,
                call_synthesis_fn=_raise_exhausted,
            ).reply_to_issue_comment(
                self._action(),
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
            )

    def test_post_exception_propagates(self, tmp_path: Path) -> None:
        """comment_issue failure propagates so callers fail closed."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nplease fix",
            comment_body="please fix",
            is_bot=False,
            context={"pr_title": "My PR"},
        )

        mock_gh = self._mock_gh()
        mock_gh.comment_issue.side_effect = Exception("gh fail")
        with pytest.raises(Exception, match="gh fail"):
            Dispatcher(
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                call_synthesis_fn=lambda *a, **kw: _synthesis_response("ok"),
            ).reply_to_issue_comment(
                action,
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
            )

    def test_no_comment_id_skips_react(self, tmp_path: Path) -> None:
        """When comment_id is absent, no reaction is attempted."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nhi",
            comment_body="hi",
            is_bot=False,
            context={"pr_title": "My PR"},  # no comment_id
        )

        mock_gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "ok", change_request="Do it"
            ),
        ).reply_to_issue_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        mock_gh.add_reaction.assert_not_called()

    def test_defaults_to_repo_configured_agent(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        action = self._action()

        create_agent_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class _FakeFactory:
            def create_agent(self, *args: object, **kwargs: object) -> object:
                create_agent_calls.append((args, kwargs))
                return _client()

        gh = self._mock_gh()
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            provider_factory=_FakeFactory(),  # type: ignore[arg-type]
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "ok", change_request="Do it"
            ),
        ).reply_to_issue_comment(
            action,
            registry=MagicMock(spec=ActivityReporter),
        )
        assert len(create_agent_calls) == 1
        args, kwargs = create_agent_calls[0]
        assert args == (self._repo_cfg(tmp_path),)
        assert kwargs == {"work_dir": tmp_path, "repo_name": "owner/repo"}
        assert cat == "ACT"

    def test_includes_conversation_context_in_synthesis(self, tmp_path: Path) -> None:
        """Conversation history is fetched and passed to call_synthesis as context."""
        cfg = self._cfg(tmp_path)
        action = self._action()
        mock_gh = self._mock_gh()
        mock_gh.get_issue_comments.return_value = [
            {"user": {"login": "alice"}, "body": "first comment"},
            {"user": {"login": "bob"}, "body": "second comment"},
            {"user": {"login": "owner"}, "body": "please fix"},
        ]

        captured_calls: list[dict] = []

        def capture_synthesis(*args: object, **kwargs: object) -> object:
            captured_calls.append(kwargs)
            return _synthesis_response("ok", change_request="Do it")

        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=capture_synthesis,
        ).reply_to_issue_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"
        # INV-6 routes the conversation fetch through the per-(repo, item)
        # CommentCache rather than calling gh.get_issue_comments directly.
        mock_gh.get_issue_comments.assert_called_once()
        # Verify conversation context was built and passed to call_synthesis
        assert captured_calls
        ctx = captured_calls[0].get("context") or {}
        assert "conversation" in ctx
        assert "alice: first comment" in ctx["conversation"]
        assert "bob: second comment" in ctx["conversation"]

    def test_conversation_context_fetch_failure_logs_and_continues(
        self, tmp_path: Path
    ) -> None:
        """Conversation fetch failure logs a warning and proceeds without context."""
        cfg = self._cfg(tmp_path)
        action = self._action()
        mock_gh = self._mock_gh()
        mock_gh.get_issue_comments.side_effect = RuntimeError("API down")

        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "ok", change_request="Do it"
            ),
        ).reply_to_issue_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ACT"

    def test_writes_durable_claim_after_reply(self, tmp_path: Path) -> None:
        """After posting a reply, the comment id is completed in SQLite."""
        cfg = self._cfg(tmp_path)
        mock_gh = self._mock_gh()

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Yes, here is why..."
            ),
        ).reply_to_issue_comment(
            self._action(cid=4275080243),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert FidoStore(tmp_path).claim_state(4275080243) == "completed"

    def test_claimed_issue_comment_returns_no_titles(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        promise = FidoStore(tmp_path).prepare_reply(
            owner="webhook", comment_type="issues", anchor_comment_id=4275080244
        )
        assert promise is not None

        gh = _make_mock_gh()
        category, titles = Dispatcher(
            cfg, self._repo_cfg(tmp_path), gh
        ).reply_to_issue_comment(
            self._action(cid=4275080244),
            agent=_client("unused"),
            registry=MagicMock(spec=ActivityReporter),
        )

        assert category == "ACT"
        assert titles == []

    def test_no_comment_id_skips_claim_write(self, tmp_path: Path) -> None:
        """When comment_id is absent, no claim file is created (no-op)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="PR top-level comment on #7 by owner:\n\nhi",
            comment_body="hi",
            is_bot=False,
            context={"pr_title": "My PR"},  # no comment_id
        )

        gh = self._mock_gh()
        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "ok", change_request="Do it"
            ),
        ).reply_to_issue_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        claim_dir = tmp_path / ".git" / "fido" / "comments"
        assert not claim_dir.exists() or not list(claim_dir.iterdir()), (
            "no claim files should be written when comment_id is absent"
        )

    def test_active_context_injected_into_synthesis(self, tmp_path: Path) -> None:
        """reply_to_issue_comment passes active-issue context to call_synthesis."""
        cfg = self._cfg(tmp_path)
        # Set up state.json so _load_active_context_for_rescope finds active issue.
        fido_dir = tmp_path / ".git" / "fido"
        fido_dir.mkdir(parents=True)
        State(fido_dir).save({"issue": 7})

        mock_gh = self._mock_gh()
        mock_gh.view_issue.return_value = {"title": "Fix crash", "body": "It crashes."}

        captured_calls: list[dict] = []

        def capture_synthesis(*args: object, **kwargs: object) -> object:
            captured_calls.append(kwargs)
            return _synthesis_response("I'll fix that.", change_request="Fix crash")

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=capture_synthesis,
        ).reply_to_issue_comment(
            self._action(),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert captured_calls
        call_kwargs = captured_calls[0]
        assert call_kwargs.get("issue") is not None
        assert call_kwargs["issue"].title == "Fix crash"

    def test_no_active_context_when_no_state(self, tmp_path: Path) -> None:
        """reply_to_issue_comment passes None issue to call_synthesis when no state.json."""
        cfg = self._cfg(tmp_path)

        captured_calls: list[dict] = []

        def capture_synthesis(*args: object, **kwargs: object) -> object:
            captured_calls.append(kwargs)
            return _synthesis_response("I'll fix that.")

        mock_gh = self._mock_gh()

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=capture_synthesis,
        ).reply_to_issue_comment(
            self._action(),
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert captured_calls
        call_kwargs = captured_calls[0]
        assert call_kwargs.get("issue") is None


class TestGetCommitSummary:
    def test_returns_git_log_output(self, tmp_path: Path) -> None:
        import subprocess as sp

        fake_result = sp.CompletedProcess(
            args=[], returncode=0, stdout="abc123 add thing\n", stderr=""
        )
        run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_run(*args: object, **kwargs: object) -> object:
            run_calls.append((args, kwargs))
            return fake_result

        result = _get_commit_summary(tmp_path, _run=fake_run)
        assert result == "abc123 add thing"
        assert len(run_calls) == 1
        args, kwargs = run_calls[0]
        assert args == (["git", "log", "--oneline", "-20"],)
        assert kwargs == {
            "cwd": tmp_path,
            "capture_output": True,
            "text": True,
            "timeout": 10,
            "check": True,
        }

    def test_raises_on_file_not_found(self, tmp_path: Path) -> None:
        def _raise(*args: object, **kwargs: object) -> Never:
            raise FileNotFoundError

        with pytest.raises(FileNotFoundError):
            _get_commit_summary(tmp_path, _run=_raise)

    def test_raises_on_timeout(self, tmp_path: Path) -> None:
        import subprocess as sp

        def _raise(*args: object, **kwargs: object) -> Never:
            raise sp.TimeoutExpired(cmd="git", timeout=10)

        with pytest.raises(sp.TimeoutExpired):
            _get_commit_summary(tmp_path, _run=_raise)

    def test_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        import subprocess as sp

        def _raise(*args: object, **kwargs: object) -> Never:
            raise sp.CalledProcessError(128, ["git"])

        with pytest.raises(sp.CalledProcessError):
            _get_commit_summary(tmp_path, _run=_raise)

    def test_raises_on_oserror(self, tmp_path: Path) -> None:
        def _raise(*args: object, **kwargs: object) -> Never:
            raise OSError("permission denied")

        with pytest.raises(OSError):
            _get_commit_summary(tmp_path, _run=_raise)


class _FakeRescopeRegistry:
    """Hand-rolled fake of the registry slice used by `_reorder_tasks_background`.

    Records calls in order in ``self.calls`` so tests can assert on
    sequencing — notably the #1280 ordering invariant: the inbox release
    must run before the late-cleanup steps that could fail.

    No MagicMock — see Rob's no-magicmock feedback (#1280 epic).
    """

    def __init__(
        self,
        *,
        raise_on_set_rescoping_true: bool = False,
        raise_on_set_rescoping_false: bool = False,
    ) -> None:
        self.calls: list[tuple] = []
        self._raise_on_true = raise_on_set_rescoping_true
        self._raise_on_false = raise_on_set_rescoping_false

    def set_rescoping(self, repo_name: str, active: bool) -> None:
        self.calls.append(("set_rescoping", repo_name, active))
        if active and self._raise_on_true:
            raise RuntimeError("rescoping flag broken (true)")
        if not active and self._raise_on_false:
            raise RuntimeError("rescoping flag broken (false)")

    def exit_untriaged(self, repo_name: str) -> None:
        self.calls.append(("exit_untriaged", repo_name))

    def abort_task(self, repo_name: str, *, task_id: str | None = None) -> None:
        self.calls.append(("abort_task", repo_name, task_id))

    def tasks_for(self, repo_name: str) -> object:
        # Sentinel — these tests inject a fake reorder fn so the Tasks
        # value is never dereferenced.  Don't record this call: tests
        # in this class assert exact .calls sequencing for inbox/
        # rescoping ordering (#1280) and tasks_for is incidental.
        _ = repo_name
        return object()


class TestReorderTasksBackground:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def _dispatcher(self, tmp_path: Path, gh: object, **kwargs: object) -> Dispatcher:
        return Dispatcher(self._cfg(tmp_path), self._repo_cfg(tmp_path), gh, **kwargs)  # type: ignore[arg-type]

    def _run_thread(self, started: list) -> None:
        """Run the captured thread's target synchronously."""
        started[0]._target()

    def _capture_reorder_calls(self) -> tuple[list, callable]:
        """Return (calls_list, mock_reorder_fn) that records (work_dir, cs, kwargs)."""
        calls: list = []

        def mock_reorder(work_dir: Path, commit_summary: str, **kwargs: object) -> None:
            calls.append((work_dir, commit_summary, kwargs))

        return calls, mock_reorder

    def test_starts_daemon_thread(self, tmp_path: Path) -> None:
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "some commits",
            registry=MagicMock(spec=ActivityReporter),
        )
        assert len(started) == 1
        t = started[0]
        assert t.daemon is True

    def test_thread_name_includes_dir_name(self, tmp_path: Path) -> None:
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=MagicMock(spec=ActivityReporter),
        )
        assert tmp_path.name in started[0].name

    def test_thread_calls_reorder_with_registry_tasks_and_commit_summary(
        self, tmp_path: Path
    ) -> None:
        """Routes the rescope write through ``registry.tasks_for(name)`` so
        the publishing-aware Tasks's on_mutate hook fires (#1696)."""
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()
        registry = MagicMock(spec=ActivityReporter)
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "feat: add parser",
            registry=registry,
        )
        self._run_thread(started)
        assert len(calls) == 1
        assert calls[0][0] is registry.tasks_for.return_value
        registry.tasks_for.assert_called_with("owner/repo")
        assert calls[0][1] == "feat: add parser"

    def test_on_inprogress_affected_aborts_worker_via_registry(
        self, tmp_path: Path
    ) -> None:
        started: list = []
        registry = MagicMock()
        calls, mock_reorder = self._capture_reorder_calls()
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=registry,
        )
        self._run_thread(started)
        on_inprogress_affected = calls[0][2]["_on_inprogress_affected"]
        on_inprogress_affected("t-current")
        registry.abort_task.assert_called_once_with("owner/repo", task_id="t-current")

    def test_releases_untriaged_hold_after_reorder_finishes(
        self, tmp_path: Path
    ) -> None:
        started: list = []
        registry = MagicMock()
        _, mock_reorder = self._capture_reorder_calls()
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=registry,
            _release_untriaged_on_finish=True,
        )

        self._run_thread(started)

        registry.exit_untriaged.assert_called_once_with("owner/repo")

    def test_releases_untriaged_hold_when_thread_start_fails(
        self, tmp_path: Path
    ) -> None:
        registry = MagicMock()

        def fail_start(_thread: object) -> Never:
            raise RuntimeError("cannot start")

        with pytest.raises(RuntimeError, match="cannot start"):
            self._dispatcher(
                tmp_path,
                MagicMock(),
                thread_start_fn=fail_start,
                reorder_fn=MagicMock(),
                reorder_coalesce_state={},
            ).reorder_tasks_background(
                "commits",
                registry=registry,
                _release_untriaged_on_finish=True,
            )

        registry.set_rescoping.assert_called_once_with("owner/repo", False)
        registry.exit_untriaged.assert_called_once_with("owner/repo")

    def test_release_runs_before_set_rescoping_in_finally(self, tmp_path: Path) -> None:
        """Release must fire BEFORE set_rescoping so a failure in the latter
        cannot leave the inbox stuck (#1280).
        """
        started: list = []
        registry = _FakeRescopeRegistry()
        _, mock_reorder = self._capture_reorder_calls()

        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=registry,
            _release_untriaged_on_finish=True,
        )
        self._run_thread(started)

        # set_rescoping(True) at thread start, then exit_untriaged in finally,
        # then set_rescoping(False) — exit must precede the False call.
        assert registry.calls == [
            ("set_rescoping", "owner/repo", True),
            ("exit_untriaged", "owner/repo"),
            ("set_rescoping", "owner/repo", False),
        ]

    def test_release_runs_even_when_prelude_set_rescoping_raises(
        self, tmp_path: Path
    ) -> None:
        """If `set_rescoping(True)` in the BG prelude raises, the inbox hold
        must still be released. The prelude is now inside the try/finally
        for exactly this reason — without that move, any prelude failure
        would skip the release and leak the count (#1280).

        In production the BG thread is a daemon, so the propagated exception
        is just printed and the thread dies. Here we run it synchronously,
        so we catch the propagated exception and inspect the calls list.
        """
        started: list = []
        registry = _FakeRescopeRegistry(raise_on_set_rescoping_true=True)
        _, mock_reorder = self._capture_reorder_calls()

        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=registry,
            _release_untriaged_on_finish=True,
        )
        with pytest.raises(RuntimeError, match="rescoping flag broken"):
            self._run_thread(started)

        # The prelude raised, so reorder never ran — but the finally must
        # still have released the inbox hold added synchronously.
        assert ("exit_untriaged", "owner/repo") in registry.calls

    def test_release_survives_set_rescoping_raising_on_thread_start_failure(
        self, tmp_path: Path
    ) -> None:
        """In the spawn-failure path, exit_untriaged must fire before
        set_rescoping(False) so a raise in the latter cannot swallow the
        release (#1280).
        """
        registry = _FakeRescopeRegistry(raise_on_set_rescoping_false=True)

        def fail_start(_thread: object) -> Never:
            raise RuntimeError("cannot start")

        with pytest.raises(RuntimeError):
            self._dispatcher(
                tmp_path,
                MagicMock(),
                thread_start_fn=fail_start,
                reorder_fn=MagicMock(),
                reorder_coalesce_state={},
            ).reorder_tasks_background(
                "commits",
                registry=registry,
                _release_untriaged_on_finish=True,
            )

        # exit_untriaged must come before the failing set_rescoping(False).
        assert ("exit_untriaged", "owner/repo") in registry.calls
        exit_idx = registry.calls.index(("exit_untriaged", "owner/repo"))
        set_false_idx = registry.calls.index(("set_rescoping", "owner/repo", False))
        assert exit_idx < set_false_idx

    def test_on_done_kwarg_calls_rewrite_fn(self, tmp_path: Path) -> None:
        started: list = []
        rewrite_calls: list = []
        sync_calls: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        def mock_rewrite(*a: object, **kw: object) -> None:
            rewrite_calls.append((a, kw))

        def mock_sync(*a: object, **kw: object) -> None:
            sync_calls.append((a, kw))

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            rewrite_fn=mock_rewrite,
            reorder_fn=mock_reorder,
            sync_fn=mock_sync,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        on_done()
        assert len(sync_calls) == 1
        assert len(rewrite_calls) == 1
        args, kwargs = rewrite_calls[0]
        assert args[0] == tmp_path

    def test_on_done_passes_agent_to_rewrite_fn(self, tmp_path: Path) -> None:
        started: list = []
        rewrite_calls: list = []
        fake_client = MagicMock()
        calls, mock_reorder = self._capture_reorder_calls()

        def mock_rewrite(*a: object, **kw: object) -> None:
            rewrite_calls.append(kw)

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            rewrite_fn=mock_rewrite,
            sync_fn=lambda *a, **kw: None,
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=MagicMock(spec=ActivityReporter),
            agent=fake_client,
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        on_done()
        assert rewrite_calls[0].get("agent") is fake_client

    def test_on_done_syncs_before_rewrite(self, tmp_path: Path) -> None:
        started: list = []
        order: list[str] = []
        calls, mock_reorder = self._capture_reorder_calls()

        def mock_sync(*a: object, **kw: object) -> None:
            order.append("sync")

        def mock_rewrite(*a: object, **kw: object) -> None:
            order.append("rewrite")

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            rewrite_fn=mock_rewrite,
            reorder_fn=mock_reorder,
            sync_fn=mock_sync,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "commits",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        on_done()
        assert order == ["sync", "rewrite"]

    def test_on_done_calls_sync_fn(self, tmp_path: Path) -> None:
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()
        sync_calls: list[tuple[object, ...]] = []

        def fake_sync(*args: object, **kwargs: object) -> None:
            sync_calls.append(args)

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            rewrite_fn=lambda *a, **kw: None,
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
            sync_fn=fake_sync,
        ).reorder_tasks_background(
            "commits",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        on_done = calls[0][2]["_on_done"]
        on_done()
        assert len(sync_calls) == 1

    def test_coalesces_when_already_running(self, tmp_path: Path) -> None:
        """Second call while first is running marks pending, does not spawn thread."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()

        # First call — marks running, spawns thread
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs1",
            registry=MagicMock(spec=ActivityReporter),
        )
        assert len(started) == 1
        assert state[str(tmp_path)]["running"] is True

        # Second call while thread has not run yet — should coalesce, not spawn
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs2",
            registry=MagicMock(spec=ActivityReporter),
        )
        assert len(started) == 1  # no second thread spawned
        assert state[str(tmp_path)]["pending"] is not None
        assert state[str(tmp_path)]["pending"][0] == "cs2"

    def test_coalesced_call_reruns_after_first_completes(self, tmp_path: Path) -> None:
        """Thread loops once for the pending coalesced call, then stops."""
        state: dict = {}
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs1",
            registry=MagicMock(spec=ActivityReporter),
        )
        # Simulate a second trigger arriving before the thread runs
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs2",
            registry=MagicMock(spec=ActivityReporter),
        )
        # Run the single thread — should execute reorder twice (cs1 then cs2)
        self._run_thread(started)
        assert len(calls) == 2
        assert calls[0][1] == "cs1"
        assert calls[1][1] == "cs2"
        assert state[str(tmp_path)]["running"] is False
        assert state[str(tmp_path)]["pending"] is None

    def test_only_last_pending_call_is_preserved(self, tmp_path: Path) -> None:
        """Multiple coalesced callers: only the last pending commit_summary is used."""
        state: dict = {}
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        for cs in ("cs1", "cs2", "cs3", "cs4"):
            gh = MagicMock()
            self._dispatcher(
                tmp_path,
                gh,
                thread_start_fn=lambda t: started.append(t),
                reorder_fn=mock_reorder,
                reorder_coalesce_state=state,
            ).reorder_tasks_background(
                cs,
                registry=MagicMock(spec=ActivityReporter),
            )
        # Only one thread spawned; pending holds cs4 (the latest)
        assert len(started) == 1
        assert state[str(tmp_path)]["pending"][0] == "cs4"
        self._run_thread(started)
        # Ran cs1 (first call) then cs4 (latest pending); cs2 and cs3 dropped
        assert len(calls) == 2
        assert calls[0][1] == "cs1"
        assert calls[1][1] == "cs4"

    def test_intents_accumulate_across_coalesced_calls(self, tmp_path: Path) -> None:
        """Intents from concurrent callers are all preserved, not overwritten."""
        state: dict = {}
        started: list = []
        calls, mock_reorder = self._capture_reorder_calls()

        intent1 = RescopeIntent("Add logging", 10, "2024-01-15T10:00:00+00:00")
        intent2 = RescopeIntent("Refactor tests", 20, "2024-01-15T10:01:00+00:00")
        intent3 = RescopeIntent("Fix typing", 30, "2024-01-15T10:02:00+00:00")

        # First call — starts thread with intent1
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs1",
            registry=MagicMock(spec=ActivityReporter),
            intents=[intent1],
        )
        # Second call — coalesces, adds intent2
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs2",
            registry=MagicMock(spec=ActivityReporter),
            intents=[intent2],
        )
        # Third call — coalesces, adds intent3
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs3",
            registry=MagicMock(spec=ActivityReporter),
            intents=[intent3],
        )
        # Only one thread spawned; pending holds all three intents
        assert len(started) == 1
        pending = state[str(tmp_path)]["pending"]
        assert pending[2] == [intent2, intent3]  # intent1 went with first call

        # Run thread — cs1+intent1 then cs3+[intent2,intent3]
        self._run_thread(started)
        assert len(calls) == 2
        # First call: intent1 only
        assert calls[0][2].get("intents") == [intent1]
        # Second call: intent2 and intent3 accumulated
        assert calls[1][2].get("intents") == [intent2, intent3]

    def test_running_flag_cleared_after_no_pending(self, tmp_path: Path) -> None:
        """After a normal run with no pending call, running is set to False."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        assert state[str(tmp_path)]["running"] is False

    def test_second_call_after_first_completes_spawns_new_thread(
        self, tmp_path: Path
    ) -> None:
        """Once the first thread finishes, a subsequent call spawns a fresh thread."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs1",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)  # first thread completes

        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs2",
            registry=MagicMock(spec=ActivityReporter),
        )
        assert len(started) == 2  # new thread spawned

    def test_different_work_dirs_do_not_interfere(self, tmp_path: Path) -> None:
        """Coalescing is per work_dir; different dirs get independent threads."""
        state: dict = {}
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"

        gh = MagicMock()
        Dispatcher(
            self._cfg(tmp_path),
            RepoConfig(name="owner/repo", work_dir=dir_a),
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        Dispatcher(
            self._cfg(tmp_path),
            RepoConfig(name="owner/repo", work_dir=dir_b),
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state=state,
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        assert len(started) == 2  # each dir gets its own thread

    def test_sets_rescoping_true_before_reorder(self, tmp_path: Path) -> None:
        """set_rescoping(True) is called on the registry before reorder runs."""
        started: list = []
        rescoping_calls: list = []
        _, mock_reorder = self._capture_reorder_calls()
        registry = MagicMock()
        registry.set_rescoping.side_effect = lambda repo, active: (
            rescoping_calls.append(active)
        )
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=registry,
        )
        self._run_thread(started)
        assert rescoping_calls[0] is True

    def test_clears_rescoping_false_after_reorder(self, tmp_path: Path) -> None:
        """set_rescoping(False) is called on the registry after the loop finishes."""
        started: list = []
        _, mock_reorder = self._capture_reorder_calls()
        registry = MagicMock()
        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=registry,
        )
        self._run_thread(started)
        # Last call must clear the flag
        last_call = registry.set_rescoping.call_args_list[-1]
        assert last_call[0] == ("owner/repo", False)

    def test_clears_rescoping_on_reorder_exception(self, tmp_path: Path) -> None:
        """set_rescoping(False) is called even when reorder raises."""
        started: list = []
        registry = MagicMock()

        def boom(work_dir: Path, commit_summary: str, **kwargs: object) -> Never:
            raise RuntimeError("reorder exploded")

        self._dispatcher(
            tmp_path,
            MagicMock(),
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=boom,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=registry,
        )
        import pytest as _pytest

        with _pytest.raises(RuntimeError, match="reorder exploded"):
            self._run_thread(started)
        last_call = registry.set_rescoping.call_args_list[-1]
        assert last_call[0] == ("owner/repo", False)

    def test_sets_thread_local_repo_name_during_reorder(self, tmp_path: Path) -> None:
        """Thread-local repo_name is set to repo_cfg.name when reorder runs."""
        from fido.provider import current_repo

        started: list = []
        seen: list = []

        def mock_reorder(work_dir: Path, commit_summary: str, **kwargs: object) -> None:
            seen.append(current_repo())

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        assert seen == ["owner/repo"]

    def test_clears_thread_local_repo_name_after_reorder(self, tmp_path: Path) -> None:
        """Thread-local repo_name is cleared in the finally block after reorder."""
        from fido.provider import current_repo, set_thread_repo

        started: list = []
        _, mock_reorder = self._capture_reorder_calls()

        set_thread_repo("owner/repo")  # pre-set to confirm it gets cleared
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        assert current_repo() is None

    def test_clears_thread_local_repo_name_on_reorder_exception(
        self, tmp_path: Path
    ) -> None:
        """Thread-local repo_name is cleared even when reorder raises."""
        from fido.provider import current_repo

        started: list = []

        def boom(work_dir: Path, commit_summary: str, **kwargs: object) -> Never:
            raise RuntimeError("reorder exploded")

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=boom,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        with pytest.raises(RuntimeError, match="reorder exploded"):
            self._run_thread(started)
        assert current_repo() is None

    def test_sets_thread_kind_background_during_reorder(self, tmp_path: Path) -> None:
        """Thread kind is set to 'background' while the reorder loop runs (#1711).

        Was 'webhook' originally (#955) — that protected the rescope thread
        from being identified as the worker by OTHER webhooks, but it also
        caused the rescope thread itself to preempt the worker on every
        iteration, livelocking long worker turns.  The third kind
        'background' preserves the #955 protection (kind != 'worker' still
        means OTHER webhooks won't preempt this thread) without granting
        preemption rights to the rescope thread itself."""
        from fido.provider import current_thread_kind

        started: list = []
        seen: list = []

        def mock_reorder(work_dir: Path, commit_summary: str, **kwargs: object) -> None:
            seen.append(current_thread_kind())

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        assert seen == ["background"]

    def test_clears_thread_kind_after_reorder(self, tmp_path: Path) -> None:
        """Thread kind is cleared in the finally block after the reorder loop."""
        from fido.provider import current_thread_kind, set_thread_kind

        started: list = []
        _, mock_reorder = self._capture_reorder_calls()

        set_thread_kind(
            ThreadKind.WEBHOOK
        )  # pre-set to confirm the finally block clears it
        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=mock_reorder,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        self._run_thread(started)
        # run_loop must clear kind in its finally block so the caller's
        # thread-local state is not polluted.
        assert current_thread_kind() == "worker"  # default when not set

    def test_clears_thread_kind_on_reorder_exception(self, tmp_path: Path) -> None:
        """Thread kind is cleared even when reorder raises."""
        from fido.provider import current_thread_kind

        started: list = []

        def boom(work_dir: Path, commit_summary: str, **kwargs: object) -> Never:
            raise RuntimeError("reorder exploded")

        gh = MagicMock()
        self._dispatcher(
            tmp_path,
            gh,
            thread_start_fn=lambda t: started.append(t),
            reorder_fn=boom,
            reorder_coalesce_state={},
        ).reorder_tasks_background(
            "cs",
            registry=MagicMock(spec=ActivityReporter),
        )
        with pytest.raises(RuntimeError, match="reorder exploded"):
            self._run_thread(started)
        assert current_thread_kind() == "worker"  # default when not set


class TestBuildOnRescopeApply:
    """INV-F (#1804): the post-apply notifier delegates the reply-back
    decision entirely to ``oracle.reply_back_intents_for``; this layer
    is just data-shape adapter glue and the call into
    ``_notify_intent_outcome`` per yielded entry."""

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _intent(
        self,
        comment_id: int,
        ts: str = "2024-01-15T10:00:00+00:00",
        author: str = "",
    ) -> RescopeIntent:
        return RescopeIntent(
            change_request=f"req {comment_id}",
            comment_id=comment_id,
            timestamp=ts,
            author=author,
        )

    def _pr(self) -> ActivePR:
        return ActivePR(
            number=42,
            title="t",
            url="https://github.com/owner/repo/pull/42",
            body="",
        )

    def test_no_pr_ctx_skips_silently(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[self._intent(101)],
            pr_ctx=None,
            agent=_client("nope"),
            prompts=Prompts("p"),
        )
        cb([], [], {}, [])
        gh.reply_to_review_comment.assert_not_called()

    def test_cross_author_rewrite_notifies_displaced_commenter(
        self, tmp_path: Path
    ) -> None:
        # alice contributed to task t1; bob (later, different author)
        # rewrote it.  Oracle says alice gets NotifyChanged; this layer
        # fires the GitHub reply for her.
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        alice = self._intent(101, "2024-01-15T10:00:00+00:00", author="alice")
        bob = self._intent(202, "2024-01-15T10:01:00+00:00", author="bob")
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[alice, bob],
            pr_ctx=self._pr(),
            agent=_client("Reply text"),
            prompts=Prompts("p"),
        )
        result = [
            {
                "id": "t1",
                "title": "Renamed by bob",
                "contributing_intents": [101, 202],
            }
        ]
        op_inputs = [
            task_queue_rescope.OpInput(
                oi_op=task_queue_rescope.RewriteTask(1, "Renamed by bob", ""),
                oi_intents=[202],
            )
        ]
        cb(result, op_inputs, {"t1": 1}, [])
        # Alice (101) gets notified; bob is the actor, not displaced.
        comment_ids = [c.args[3] for c in gh.reply_to_review_comment.call_args_list]
        assert comment_ids == [101]

    def test_same_author_followup_is_silent(self, tmp_path: Path) -> None:
        # alice's later intent rewrites her own earlier-contributing task
        # → silent (she already knows).
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        alice1 = self._intent(101, "2024-01-15T10:00:00+00:00", author="alice")
        alice2 = self._intent(102, "2024-01-15T10:01:00+00:00", author="alice")
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[alice1, alice2],
            pr_ctx=self._pr(),
            agent=_client("nope"),
            prompts=Prompts("p"),
        )
        result = [
            {
                "id": "t1",
                "title": "Renamed",
                "contributing_intents": [101, 102],
            }
        ]
        op_inputs = [
            task_queue_rescope.OpInput(
                oi_op=task_queue_rescope.RewriteTask(1, "Renamed", ""),
                oi_intents=[102],
            )
        ]
        cb(result, op_inputs, {"t1": 1}, [])
        gh.reply_to_review_comment.assert_not_called()

    def test_cross_author_merge_does_not_notify(self, tmp_path: Path) -> None:
        # Even cross-author merge — alice's task gets merged into bob's
        # — doesn't notify, because the work still gets done.
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        alice = self._intent(101, "2024-01-15T10:00:00+00:00", author="alice")
        bob = self._intent(202, "2024-01-15T10:01:00+00:00", author="bob")
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[alice, bob],
            pr_ctx=self._pr(),
            agent=_client("nope"),
            prompts=Prompts("p"),
        )
        result = [
            {"id": "t1", "title": "merged", "contributing_intents": [101, 202]},
            {"id": "t2", "title": "old alice", "contributing_intents": [101]},
        ]
        op_inputs = [
            task_queue_rescope.OpInput(
                oi_op=task_queue_rescope.MergeTasks(1, [2], "merged", ""),
                oi_intents=[202],
            ),
            task_queue_rescope.OpInput(
                oi_op=task_queue_rescope.CompleteTask(2),
                oi_intents=[202],
            ),
        ]
        cb(result, op_inputs, {"t1": 1, "t2": 2}, [])
        gh.reply_to_review_comment.assert_not_called()

    def test_no_op_verdict_notifies_via_hol24_path(self, tmp_path: Path) -> None:
        # HOL-24 / #1918: the oracle doesn't fire for no_op verdicts
        # (nothing reorganized), so the verdict-based pass picks them
        # up and notifies the requester their ask was silently dropped.
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        alice = self._intent(101, author="alice")
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[alice],
            pr_ctx=self._pr(),
            agent=_client("Reply text"),
            prompts=Prompts("p"),
        )
        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="no_op",
            narrative="dropped because the requested change conflicts with the spec",
        )
        cb([], [], {}, [verdict])
        comment_ids = [c.args[3] for c in gh.reply_to_review_comment.call_args_list]
        assert comment_ids == [101]

    def test_honored_non_divergent_verdict_silent(self, tmp_path: Path) -> None:
        # HOL-24: an honored verdict whose task description contains
        # the change_request verbatim is NOT material — triage reply
        # was enough, no second voice reply.
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        alice = self._intent(101, author="alice")
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[alice],
            pr_ctx=self._pr(),
            agent=_client("nope"),
            prompts=Prompts("p"),
        )
        result = [
            {
                "id": "t1",
                "title": "t",
                "description": "implement req 101 exactly as written",
                "contributing_intents": [101],
            }
        ]
        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="honored",
            narrative="queued as asked",
            affected_task_ids=("t1",),
        )
        cb(result, [], {"t1": 1}, [verdict])
        gh.reply_to_review_comment.assert_not_called()

    def test_hol25_no_op_framing_carries_narrative(self) -> None:
        # HOL-25 / #1919: no_op verdict framing names the outcome
        # explicitly AND includes the verdict's narrative so Opus has
        # the rescope's reasoning to weave into the reply.
        from fido.events import _hol25_verdict_framing

        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="no_op",
            narrative="duplicates work already queued as #42",
        )
        framing = _hol25_verdict_framing(verdict)
        assert "NOT queued" in framing
        assert "duplicates work already queued as #42" in framing
        assert "in flight" in framing  # negative-claim guard

    def test_hol25_honored_framing_carries_narrative(self) -> None:
        from fido.events import _hol25_verdict_framing

        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="honored",
            narrative="scoped to the parser only, not the whole pipeline",
        )
        framing = _hol25_verdict_framing(verdict)
        assert "queued" in framing
        assert "scoped to the parser only" in framing
        assert "verbatim" in framing  # negative-claim guard

    def test_hol25_reshaped_framing_carries_narrative(self) -> None:
        from fido.events import _hol25_verdict_framing

        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="reshaped",
            narrative="merged with adjacent typo fix as one commit",
        )
        framing = _hol25_verdict_framing(verdict)
        assert "RESHAPED" in framing
        assert "merged with adjacent typo fix as one commit" in framing

    def test_hol25_superseded_framing_renders_by_intent_link(self) -> None:
        from fido.events import _hol25_verdict_framing

        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="superseded",
            narrative="overridden by a later directive",
            by_intent_comment_id=202,
        )
        framing = _hol25_verdict_framing(verdict)
        assert "SUPERSEDED" in framing
        assert "#202" in framing
        assert "overridden by a later directive" in framing

    def test_honored_verdict_falls_back_to_contributing_intents(
        self, tmp_path: Path
    ) -> None:
        # Codex P2 on PR #1938: an honored verdict that omits
        # ``affected_task_ids`` would skip the divergence check (no
        # task descriptions to compare against → not material → silent
        # drop), even when the resulting task text diverges.  Fall
        # back to every task whose contributing_intents lists this
        # intent's comment_id.
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        alice = self._intent(101, author="alice")
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[alice],
            pr_ctx=self._pr(),
            agent=_client("Reply text"),
            prompts=Prompts("p"),
        )
        result = [
            {
                "id": "t1",
                "title": "Renamed",
                "description": "completely different from the ask",
                "contributing_intents": [101],
            }
        ]
        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="honored",
            narrative="queued as asked",
            affected_task_ids=(),  # omitted
        )
        cb(result, [], {"t1": 1}, [verdict])
        # Divergence detected via the fallback — reply fires.
        comment_ids = [c.args[3] for c in gh.reply_to_review_comment.call_args_list]
        assert comment_ids == [101]

    def test_verdict_dedupes_against_oracle_path(self, tmp_path: Path) -> None:
        # HOL-24: if the oracle already notified intent 101 (cross-author
        # rewrite), a material verdict for the same intent_comment_id
        # must NOT double-notify.
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        alice = self._intent(101, "2024-01-15T10:00:00+00:00", author="alice")
        bob = self._intent(202, "2024-01-15T10:01:00+00:00", author="bob")
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[alice, bob],
            pr_ctx=self._pr(),
            agent=_client("Reply text"),
            prompts=Prompts("p"),
        )
        result = [
            {
                "id": "t1",
                "title": "Renamed by bob",
                "description": "",
                "contributing_intents": [101, 202],
            }
        ]
        op_inputs = [
            task_queue_rescope.OpInput(
                oi_op=task_queue_rescope.RewriteTask(1, "Renamed by bob", ""),
                oi_intents=[202],
            )
        ]
        verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="reshaped",
            narrative="rewritten",
            affected_task_ids=("t1",),
        )
        cb(result, op_inputs, {"t1": 1}, [verdict])
        comment_ids = [c.args[3] for c in gh.reply_to_review_comment.call_args_list]
        assert comment_ids == [101]  # once, not twice


class TestHol34NoOpSilentDropRegression:
    """HOL-34 / #1928: end-to-end regression for PR #1890's
    no_op-silent-drop pattern.

    Original failure (pre-epic): a rescope verdict of ``no_op`` for a
    user's change request silently dropped the intent — no SKIPPED
    marker, no reply-back, the requester never knew.  Once the HOL
    chain landed (HOL-3 enforces narrative, HOL-6 creates SKIPPED
    marker, HOL-23/24/25 emit per-thread reply with narrative), this
    pattern can't recur.

    This test pins the reply-back leg of the closure: a no_op verdict
    in the rescope batch MUST produce a notification to the
    originating commenter carrying the verdict's narrative.  HOL-6 +
    SKIPPED marker is tested in test_tasks.py around the reducer
    transition; this test is the events-side regression that closes
    the whole story.
    """

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _intent(self, comment_id: int) -> RescopeIntent:
        return RescopeIntent(
            change_request="Please add retry logic to the cache lookup",
            comment_id=comment_id,
            timestamp="2024-01-15T10:00:00+00:00",
            author="alice",
        )

    def _pr(self) -> ActivePR:
        return ActivePR(
            number=42,
            title="t",
            url="https://github.com/owner/repo/pull/42",
            body="",
        )

    def test_pr_1890_no_op_drop_now_replies_with_narrative(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        gh.reply_to_review_comment.return_value = {"id": 999}
        intent = self._intent(101)
        cb = Dispatcher(cfg, cfg.repos["owner/repo"], gh)._build_on_rescope_apply(
            intents=[intent],
            pr_ctx=self._pr(),
            agent=_client("Heard you — couldn't queue this; here's why."),
            prompts=Prompts("p"),
        )

        # Rescope decided the request didn't fit the spec — no_op
        # verdict with a narrative.  PRE-EPIC: silently dropped.
        # POST-EPIC: HOL-24 sees the no_op via ``verdict_is_material``
        # and emits a reply through ``_notify_intent_outcome``
        # (HOL-25 framing carries the narrative).
        no_op_verdict = IntentVerdict(
            intent_comment_id=101,
            outcome="no_op",
            narrative=(
                "Retry logic is intentionally out-of-scope for the cache "
                "layer per ARCH-7; routing this to the resilience epic "
                "instead"
            ),
        )

        cb([], [], {}, [no_op_verdict])

        # The reply fires — the failure pattern is closed.
        gh.reply_to_review_comment.assert_called_once()
        # And it targets the originating commenter (alice's comment id).
        assert gh.reply_to_review_comment.call_args.args[3] == 101


class TestHol28TerminalThreadFraming:
    """HOL-28 / #1922: per-thread terminal aggregate framing names
    landed commits + skipped tasks + carries the requester's ask."""

    def _intent(self, comment_id: int = 101) -> RescopeIntent:
        return RescopeIntent(
            change_request="please rename the parser",
            comment_id=comment_id,
            timestamp="2024-01-15T10:00:00+00:00",
        )

    def test_lists_landed_commits(self) -> None:
        from fido.events import _hol28_terminal_thread_framing

        completed = [{"id": "t1", "title": "rename parser module"}]
        framing = _hol28_terminal_thread_framing(self._intent(), completed, [])
        assert "LANDED" in framing
        assert "t1: rename parser module" in framing

    def test_lists_skipped_tasks_with_reason(self) -> None:
        from fido.events import _hol28_terminal_thread_framing

        skipped = [
            {
                "id": "t2",
                "title": "rename internal helper",
                "description": "redundant with t1; skipping",
            }
        ]
        framing = _hol28_terminal_thread_framing(self._intent(), [], skipped)
        assert "SKIPPED" in framing
        assert "t2: rename internal helper — redundant with t1; skipping" in framing

    def test_carries_intent_change_request(self) -> None:
        from fido.events import _hol28_terminal_thread_framing

        framing = _hol28_terminal_thread_framing(self._intent(), [], [])
        assert "please rename the parser" in framing
        assert "comment id: 101" in framing

    def test_skipped_with_no_reason_uses_fallback(self) -> None:
        from fido.events import _hol28_terminal_thread_framing

        skipped = [{"id": "t1", "title": "x", "description": ""}]
        framing = _hol28_terminal_thread_framing(self._intent(), [], skipped)
        assert "(no reason recorded)" in framing

    def test_untitled_task_uses_fallback(self) -> None:
        from fido.events import _hol28_terminal_thread_framing

        completed = [{"id": "t1", "title": ""}]
        framing = _hol28_terminal_thread_framing(self._intent(), completed, [])
        assert "t1: (untitled)" in framing


class TestSafeFileInsightsPreReply:
    """HOL-26 / #1920 (codex P1 sixth round on PR #1938): the
    production reply paths wrap ``file_insights_pre_reply`` so a
    transient insight-filer failure does NOT drop the user-visible
    reply."""

    def test_swallows_exception(self) -> None:
        from fido.events import _safe_file_insights_pre_reply

        class _Boom:
            def file_insights_pre_reply(self, _r: object, _t: object) -> None:
                raise RuntimeError("GitHub down")

        # Must not raise.
        _safe_file_insights_pre_reply(
            _Boom(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            where="test",
        )

    def test_passes_response_and_target_on_success(self) -> None:
        from fido.events import _safe_file_insights_pre_reply

        seen: list[tuple[object, object]] = []

        class _Spy:
            def file_insights_pre_reply(self, r: object, t: object) -> None:
                seen.append((r, t))

        resp = object()
        target = object()
        _safe_file_insights_pre_reply(
            _Spy(),  # type: ignore[arg-type]
            resp,  # type: ignore[arg-type]
            target,  # type: ignore[arg-type]
            where="test",
        )
        assert seen == [(resp, target)]


class TestNotifyTerminalTaskThread:
    """HOL-28 wire: ``Dispatcher.notify_terminal_task_thread`` posts
    the aggregate reply at *anchor_intent*'s comment for the
    just-terminal task.  Caller-side transition check moved to the
    worker per codex P1 (fifth round) on PR #1938 — the dispatcher
    method is now a focused post-with-framing helper."""

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _anchor(self, comment_id: int = 999) -> RescopeIntent:
        return RescopeIntent(
            change_request="(task-anchor reconstruction)",
            comment_id=comment_id,
            timestamp="",
            comment_type="pulls",
        )

    def _pulls_task(
        self, tid: str, status: str, anchor_cid: int = 999, **extra: object
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "id": tid,
            "status": status,
            "thread": {"comment_type": "pulls", "comment_id": anchor_cid},
        }
        row.update(extra)
        return row

    def test_completed_task_fires_reply_at_anchor(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        new_tasks = [self._pulls_task("t1", "completed", title="did it")]
        Dispatcher(cfg, cfg.repos["owner/repo"], gh).notify_terminal_task_thread(
            new_tasks,
            self._anchor(999),
            pr=42,
            agent=_client("Done."),
            prompts=Prompts("p"),
        )
        gh.reply_to_review_comment.assert_called_once()
        assert gh.reply_to_review_comment.call_args.args[3] == 999

    def test_skipped_task_fires_reply(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        new_tasks = [
            self._pulls_task(
                "t1",
                "skipped",
                title="y",
                description="no longer needed",
            )
        ]
        Dispatcher(cfg, cfg.repos["owner/repo"], gh).notify_terminal_task_thread(
            new_tasks,
            self._anchor(),
            pr=42,
            agent=_client("Done."),
            prompts=Prompts("p"),
        )
        gh.reply_to_review_comment.assert_called_once()

    def test_non_terminal_anchored_task_skips_dispatch(self, tmp_path: Path) -> None:
        # Even if SOME anchored tasks are terminal, a single sibling
        # still pending blocks the closing summary.
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        new_tasks = [
            self._pulls_task("t1", "completed"),
            self._pulls_task("t2", "pending"),
        ]
        Dispatcher(cfg, cfg.repos["owner/repo"], gh).notify_terminal_task_thread(
            new_tasks,
            self._anchor(),
            pr=42,
            agent=_client("nope"),
            prompts=Prompts("p"),
        )
        gh.reply_to_review_comment.assert_not_called()

    def test_aggregate_includes_all_sibling_tasks(self, tmp_path: Path) -> None:
        # Codex P2 (seventh round) on PR #1938: when the closing
        # summary fires for a multi-task thread, it must mention ALL
        # sibling terminal tasks, not just one.
        cfg = self._cfg(tmp_path)
        captured: list[str] = []

        class _CapturingAgent:
            voice_model = "claude-opus-4-7"

            def run_turn(self, prompt: str, **_kw: object) -> str:
                captured.append(prompt)
                return "Done."

        gh = MagicMock()
        new_tasks = [
            self._pulls_task("t1", "completed", title="first commit"),
            self._pulls_task("t2", "completed", title="second commit"),
            self._pulls_task("t3", "skipped", title="dropped", description="why"),
        ]
        Dispatcher(cfg, cfg.repos["owner/repo"], gh).notify_terminal_task_thread(
            new_tasks,
            self._anchor(),
            pr=42,
            agent=_CapturingAgent(),  # type: ignore[arg-type]
            prompts=Prompts("p"),
        )
        assert captured, "agent.run_turn was not called"
        framing = captured[0]
        assert "t1: first commit" in framing
        assert "t2: second commit" in framing
        assert "t3: dropped — why" in framing

    def test_anchor_empty_change_request_not_leaked(self, tmp_path: Path) -> None:
        # Codex P2 (seventh round) on PR #1938: the worker constructs
        # the anchor intent with ``change_request=""`` to avoid
        # leaking a placeholder string into the reply.  The framing
        # must NOT include the change_request line when the field is
        # empty.
        cfg = self._cfg(tmp_path)
        captured: list[str] = []

        class _CapturingAgent:
            voice_model = "claude-opus-4-7"

            def run_turn(self, prompt: str, **_kw: object) -> str:
                captured.append(prompt)
                return "Done."

        gh = MagicMock()
        anchor = RescopeIntent(
            change_request="",
            comment_id=999,
            timestamp="",
            comment_type="pulls",
        )
        Dispatcher(cfg, cfg.repos["owner/repo"], gh).notify_terminal_task_thread(
            [self._pulls_task("t1", "completed", title="x")],
            anchor,
            pr=42,
            agent=_CapturingAgent(),  # type: ignore[arg-type]
            prompts=Prompts("p"),
        )
        framing = captured[0]
        assert "change request" not in framing.lower()
        # And the placeholder string from older versions must not appear.
        assert "task-anchor reconstruction" not in framing

    def test_post_failure_logged_not_raised(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        gh.reply_to_review_comment.side_effect = RuntimeError("api down")
        Dispatcher(cfg, cfg.repos["owner/repo"], gh).notify_terminal_task_thread(
            [self._pulls_task("t1", "completed", title="x")],
            self._anchor(),
            pr=42,
            agent=_client("Done."),
            prompts=Prompts("p"),
        )

    def test_no_anchored_tasks_skips_dispatch(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        gh = MagicMock()
        # Anchor is comment 999 but no tasks live at that anchor.
        Dispatcher(cfg, cfg.repos["owner/repo"], gh).notify_terminal_task_thread(
            [self._pulls_task("t1", "completed", anchor_cid=12345)],
            self._anchor(999),
            pr=42,
            agent=_client("nope"),
            prompts=Prompts("p"),
        )
        gh.reply_to_review_comment.assert_not_called()


class TestNotifyIntentOutcome:
    """INV-F (#1804): ``_notify_intent_outcome`` posts a per-intent
    reply with framing keyed on the oracle's ``NotifyKind``."""

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _intent(
        self, comment_id: int = 101, comment_type: str = "pulls"
    ) -> RescopeIntent:
        return RescopeIntent(
            change_request="please rename the parser",
            comment_id=comment_id,
            timestamp="2024-01-15T10:00:00+00:00",
            comment_type=comment_type,
        )

    def _dispatcher(self, tmp_path: Path, gh: object) -> Dispatcher:
        cfg = self._cfg(tmp_path)
        return Dispatcher(cfg, cfg.repos["owner/repo"], gh)

    def _kwargs(self, agent: object) -> dict[str, object]:
        return {
            "pr": 42,
            "batch_intents": [self._intent()],
            "affected_task_ids": ["t1"],
            "result": [],
            "agent": agent,
            "prompts": Prompts("p"),
        }

    def test_notify_changed_posts_review_thread_reply(self, tmp_path: Path) -> None:
        gh = MagicMock()
        intent = self._intent(comment_id=999)
        self._dispatcher(tmp_path, gh)._notify_intent_outcome(
            intent,
            task_queue_rescope.NotifyChanged(),
            **self._kwargs(_client("Replanned, not done.")),
        )
        gh.reply_to_review_comment.assert_called_once_with(
            "owner/repo", 42, "Replanned, not done.", 999
        )

    def test_notify_dropped_uses_drop_framing(self, tmp_path: Path) -> None:
        captured: list[str] = []

        def fake(prompt: str, model: object, **kwargs: object) -> str:
            captured.append(prompt)
            return "ok"

        self._dispatcher(tmp_path, MagicMock())._notify_intent_outcome(
            self._intent(),
            task_queue_rescope.NotifyDropped(),
            **self._kwargs(_client(side_effect=fake)),
        )
        assert any("DROPPED" in p for p in captured)
        assert any("Do NOT say their work is done" in p for p in captured)

    def test_notify_changed_uses_replan_framing(self, tmp_path: Path) -> None:
        captured: list[str] = []

        def fake(prompt: str, model: object, **kwargs: object) -> str:
            captured.append(prompt)
            return "ok"

        self._dispatcher(tmp_path, MagicMock())._notify_intent_outcome(
            self._intent(),
            task_queue_rescope.NotifyChanged(),
            **self._kwargs(_client(side_effect=fake)),
        )
        assert any("REPLANNED" in p for p in captured)
        assert any("Do NOT say the work is done" in p for p in captured)

    def test_issue_comment_intent_skips(self, tmp_path: Path) -> None:
        gh = MagicMock()
        intent = self._intent(comment_id=42, comment_type="issues")
        self._dispatcher(tmp_path, gh)._notify_intent_outcome(
            intent,
            task_queue_rescope.NotifyChanged(),
            **{**self._kwargs(_client("nope")), "batch_intents": [intent]},
        )
        gh.reply_to_review_comment.assert_not_called()

    def test_post_failure_does_not_raise(self, tmp_path: Path) -> None:
        gh = MagicMock()
        gh.reply_to_review_comment.side_effect = RuntimeError("network")
        self._dispatcher(tmp_path, gh)._notify_intent_outcome(
            self._intent(),
            task_queue_rescope.NotifyChanged(),
            **self._kwargs(_client("ok")),
        )

    def test_co_contributing_intent_outside_batch_silently_skipped(
        self, tmp_path: Path
    ) -> None:
        # The result task lists a contributing_intent id that isn't in
        # batch_intents (e.g. a prior-batch intent that's no longer
        # being notified about).  The notifier silently skips it.
        captured: list[str] = []

        def fake(prompt: str, model: object, **kwargs: object) -> str:
            captured.append(prompt)
            return "ok"

        intent = self._intent(101)
        result = [{"id": "t1", "title": "T", "contributing_intents": [101, 999]}]
        self._dispatcher(tmp_path, MagicMock())._notify_intent_outcome(
            intent,
            task_queue_rescope.NotifyChanged(),
            **{
                **self._kwargs(_client(side_effect=fake)),
                "batch_intents": [intent],  # 999 deliberately absent
                "result": result,
            },
        )
        joined = "\n".join(captured)
        assert "comment #999" not in joined

    def test_default_agent_and_prompts_constructed_when_none(
        self, tmp_path: Path
    ) -> None:
        intent = self._intent()
        kwargs = self._kwargs(_client("Auto reply"))
        del kwargs["agent"]
        del kwargs["prompts"]

        create_agent_calls: list[tuple[object, ...]] = []

        class _FakeFactory:
            def create_agent(self, *args: object, **kwargs: object) -> object:
                create_agent_calls.append(args)
                return _client("Auto reply")

        cfg = self._cfg(tmp_path)
        Dispatcher(
            cfg,
            cfg.repos["owner/repo"],
            MagicMock(),
            provider_factory=_FakeFactory(),  # type: ignore[arg-type]
        )._notify_intent_outcome(
            intent,
            task_queue_rescope.NotifyChanged(),
            **kwargs,
        )
        assert len(create_agent_calls) == 1


class TestBuildOpInputsAuthors:
    """Coverage for the empty-author branch of ``_intent_author_ids``
    (unauthored intents get unique sentinel ids so they never
    accidentally compare equal in the oracle's later-cross-author
    check)."""

    def test_unauthored_intents_get_unique_ids(self, tmp_path: Path) -> None:
        from fido.events import _intent_author_ids

        intents = [
            RescopeIntent(
                change_request="a",
                comment_id=10,
                timestamp="2024-01-15T10:00:00+00:00",
                author="",
            ),
            RescopeIntent(
                change_request="b",
                comment_id=20,
                timestamp="2024-01-15T10:01:00+00:00",
                author="",
            ),
        ]
        out = _intent_author_ids(intents)
        # Distinct sentinel ids — unauthored intents must never
        # accidentally compare equal-author.
        assert out[10] != out[20]


class TestBackfillMissedPrComments:
    """Replay of issue_comment webhooks missed during fido downtime (fix #794).

    Only top-level PR comments are in scope — inline review comments and review
    threads are already scanned each iteration by ``Worker.handle_threads``.
    """

    def _cfg(
        self, tmp_path: Path, allowed_bots: frozenset[str] = frozenset()
    ) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=allowed_bots,
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(
        self, tmp_path: Path, collaborators: frozenset[str] = frozenset({"rhencke"})
    ) -> RepoConfig:
        return RepoConfig(
            name="owner/repo",
            work_dir=tmp_path,
            membership=RepoMembership(collaborators=collaborators),
        )

    def _comment(
        self,
        comment_id: int,
        user: str = "rhencke",
        body: str = "hello",
    ) -> dict:
        return {
            "id": comment_id,
            "user": {"login": user},
            "body": body,
            "html_url": f"https://github.com/owner/repo/pull/1#issuecomment-{comment_id}",
        }

    def _gh_with_pr(self, comments: list[dict]) -> MagicMock:
        """Default GH mock for backfill tests: get_issue_comments returns
        the given list, get_pr returns a stub PR for Action context."""
        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = comments
        mock_gh.get_pr.return_value = {"title": "PR title", "body": "PR body"}
        mock_gh.is_thread_resolved_for_comment.return_value = False
        return mock_gh

    def _registry(self) -> MagicMock:
        return MagicMock()

    def test_routes_allowed_collaborator_comment_through_synthesis(
        self, tmp_path: Path
    ) -> None:
        # #1814 / INV-B slice 1: backfill calls reply_to_issue_comment
        # (the live synthesis path), not events.create_task.  The
        # Action's thread metadata carries the comment id + author.
        mock_gh = self._gh_with_pr([self._comment(100)])
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        reply_calls: list[tuple[object, ...]] = []

        def fake_reply(*args: object, **kwargs: object) -> tuple[str, list[str]]:
            reply_calls.append(args)
            return ("ACT", [])

        count = Dispatcher(
            cfg, repo_cfg, mock_gh, backfill_reply_fn=fake_reply
        ).backfill_missed_pr_comments(
            1, gh_user="fidocancode", registry=self._registry()
        )
        assert count == 1
        assert len(reply_calls) == 1
        action = reply_calls[0][0]
        assert action.thread["comment_id"] == 100
        assert action.thread["comment_type"] == "issues"
        assert action.thread["author"] == "rhencke"

    def test_does_not_call_legacy_create_task(self, tmp_path: Path) -> None:
        # #1814 / INV-B slice 1 anti-regression: events.create_task is
        # never invoked from backfill in the new path (still exists
        # as a function; slice 2 removes it once we confirm zero
        # callers).
        # Backfill never calls create_task — it delegates entirely to reply_fn.
        # Verify the call completes without raising (if create_task were invoked
        # by backfill, the real function would blow up without the right args).
        mock_gh = self._gh_with_pr([self._comment(100)])
        Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=lambda *a, **kw: ("ACT", []),
        ).backfill_missed_pr_comments(
            1,
            gh_user="fidocancode",
            registry=self._registry(),
        )

    def test_skips_fido_own_comments(self, tmp_path: Path) -> None:
        mock_gh = self._gh_with_pr(
            [self._comment(100, user="fidocancode", body="my own reply")]
        )

        def _should_not_be_called_reply(*args: object, **kwargs: object) -> Never:
            raise AssertionError("reply_to_issue_comment should not be called")

        Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=_should_not_be_called_reply,
        ).backfill_missed_pr_comments(
            1,
            gh_user="FidoCanCode",
            registry=self._registry(),
        )

    def test_skips_by_gh_user_case_insensitive(self, tmp_path: Path) -> None:
        mock_gh = self._gh_with_pr([self._comment(100, user="Alice", body="mine")])

        def _should_not_be_called_reply(*args: object, **kwargs: object) -> Never:
            raise AssertionError("reply_to_issue_comment should not be called")

        Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=_should_not_be_called_reply,
        ).backfill_missed_pr_comments(
            1,
            gh_user="alice",
            registry=self._registry(),
        )

    def test_skips_fido_literal_name_even_if_gh_user_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Defense in depth: even if ``gh_user`` is misconfigured, comments
        from the literal fido account must never trigger a backfill task."""
        mock_gh = self._gh_with_pr(
            [self._comment(100, user="fido-can-code", body="my reply")]
        )

        def _should_not_be_called_reply(*args: object, **kwargs: object) -> Never:
            raise AssertionError("reply_to_issue_comment should not be called")

        Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=_should_not_be_called_reply,
        ).backfill_missed_pr_comments(
            1,
            gh_user="mis-configured-bot",
            registry=self._registry(),
        )

    def test_skips_non_allowed_users(self, tmp_path: Path) -> None:
        mock_gh = self._gh_with_pr([self._comment(100, user="random-stranger")])

        def _should_not_be_called_reply(*args: object, **kwargs: object) -> Never:
            raise AssertionError("reply_to_issue_comment should not be called")

        Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path, collaborators=frozenset({"rhencke"})),
            mock_gh,
            backfill_reply_fn=_should_not_be_called_reply,
        ).backfill_missed_pr_comments(
            1,
            gh_user="fidocancode",
            registry=self._registry(),
        )

    def test_allows_configured_bots(self, tmp_path: Path) -> None:
        mock_gh = self._gh_with_pr(
            [self._comment(100, user="dependabot[bot]", body="bump dep")]
        )
        reply_calls: list[tuple[object, ...]] = []

        def fake_reply(*args: object, **kwargs: object) -> tuple[str, list[str]]:
            reply_calls.append(args)
            return ("ACT", [])

        Dispatcher(
            self._cfg(tmp_path, allowed_bots=frozenset({"dependabot[bot]"})),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=fake_reply,
        ).backfill_missed_pr_comments(
            1, gh_user="fidocancode", registry=self._registry()
        )
        assert len(reply_calls) == 1
        action = reply_calls[0][0]
        assert "bot" in action.thread["author"]

    def test_prompt_marks_bot_vs_human(self, tmp_path: Path) -> None:
        mock_gh = self._gh_with_pr(
            [
                self._comment(100, user="rhencke", body="human msg"),
                self._comment(101, user="bot[bot]", body="bot msg"),
            ]
        )
        reply_calls: list[tuple[object, ...]] = []

        def fake_reply(*args: object, **kwargs: object) -> tuple[str, list[str]]:
            reply_calls.append(args)
            return ("ACT", [])

        Dispatcher(
            self._cfg(tmp_path, allowed_bots=frozenset({"bot[bot]"})),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=fake_reply,
        ).backfill_missed_pr_comments(
            1, gh_user="fidocancode", registry=self._registry()
        )
        actions = [call[0] for call in reply_calls]
        # Action.is_bot reflects the per-comment user marker; the
        # prompt format remains stable for synthesis classification.
        assert any(not a.is_bot for a in actions)
        assert any(a.is_bot for a in actions)

    def test_skips_empty_login_and_missing_id(self, tmp_path: Path) -> None:
        mock_gh = self._gh_with_pr(
            [
                {"id": 1, "user": {"login": ""}, "body": "x"},
                {"id": None, "user": {"login": "rhencke"}, "body": "x"},
                {"id": 2, "user": None, "body": "x"},
            ]
        )

        def _should_not_be_called_reply(*args: object, **kwargs: object) -> Never:
            raise AssertionError("reply_to_issue_comment should not be called")

        Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=_should_not_be_called_reply,
        ).backfill_missed_pr_comments(
            1,
            gh_user="fidocancode",
            registry=self._registry(),
        )

    def test_empty_comment_list_is_noop(self, tmp_path: Path) -> None:
        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = []

        def _should_not_be_called_reply(*args: object, **kwargs: object) -> Never:
            raise AssertionError("reply_to_issue_comment should not be called")

        count = Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=_should_not_be_called_reply,
        ).backfill_missed_pr_comments(
            1,
            gh_user="fidocancode",
            registry=self._registry(),
        )
        assert count == 0
        # Also: with zero comments, no need to fetch PR metadata.
        mock_gh.get_pr.assert_not_called()

    def test_skips_already_claimed_comments(self, tmp_path: Path) -> None:
        """Comments with a durable SQLite claim are not re-queued on restart.

        reply_to_issue_comment completes comment ids in SQLite after posting;
        backfill must honour that durable claim and skip re-queueing.
        """
        mock_gh = self._gh_with_pr(
            [
                self._comment(100, body="already answered"),
                self._comment(200, body="not yet handled"),
            ]
        )
        promise = FidoStore(tmp_path).prepare_reply(
            owner="webhook", comment_type="issues", anchor_comment_id=100
        )
        assert promise is not None
        FidoStore(tmp_path).ack_promise(promise.promise_id)

        reply_calls: list[tuple[object, ...]] = []

        def fake_reply(*args: object, **kwargs: object) -> tuple[str, list[str]]:
            reply_calls.append(args)
            return ("ACT", [])

        Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=fake_reply,
        ).backfill_missed_pr_comments(
            1, gh_user="fidocancode", registry=self._registry()
        )

        # Only comment 200 (unclaimed) should be routed through synthesis.
        assert len(reply_calls) == 1
        action = reply_calls[0][0]
        assert action.thread["comment_id"] == 200

    def test_synthesis_failure_per_comment_does_not_break_loop(
        self, tmp_path: Path
    ) -> None:
        # #1814: per-comment exceptions are logged and the loop
        # continues — one malformed comment shouldn't poison the
        # whole backfill.
        mock_gh = self._gh_with_pr(
            [
                self._comment(100, body="first"),
                self._comment(200, body="second"),
            ]
        )
        call_count = 0

        def fake_reply(*args: object, **kwargs: object) -> tuple[str, list[str]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("synth failed")
            return ("ACT", [])

        count = Dispatcher(
            self._cfg(tmp_path),
            self._repo_cfg(tmp_path),
            mock_gh,
            backfill_reply_fn=fake_reply,
        ).backfill_missed_pr_comments(
            1, gh_user="fidocancode", registry=self._registry()
        )
        assert count == 2
        assert call_count == 2


class TestLaunchSync:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_calls_sync_tasks_background(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        mock_gh = MagicMock()
        sync_calls: list[tuple[object, ...]] = []

        def fake_sync(*args: object, **kwargs: object) -> None:
            sync_calls.append(args)

        Dispatcher(
            cfg, self._repo_cfg(tmp_path), mock_gh, sync_fn=fake_sync
        ).launch_sync()
        assert len(sync_calls) == 1
        assert sync_calls[0] == (tmp_path, mock_gh)

    def test_does_not_raise(self, tmp_path: Path) -> None:
        cfg = self._cfg(tmp_path)
        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            _make_mock_gh(),
            sync_fn=lambda *a, **kw: None,
        ).launch_sync()  # should not raise


class TestLaunchWorker:
    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_wakes_registry_for_repo(self, tmp_path: Path) -> None:
        registry = MagicMock()
        launch_worker(self._repo_cfg(tmp_path), registry)
        registry.wake.assert_called_once_with("owner/repo")


class TestDispatchPullRequestReview:
    def test_submitted_with_review_id(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {
                "id": 55,
                "state": "changes_requested",
                "user": {"login": "owner"},
            },
            "pull_request": {"number": 3},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review", payload
        )
        assert result is not None
        assert result.review_comments is not None
        assert result.review_comments["review_id"] == 55

    def test_submitted_without_review_id(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"state": "approved", "user": {"login": "owner"}},
            "pull_request": {"number": 3},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review", payload
        )
        assert result is not None
        assert result.review_comments is None

    def test_submitted_no_number_returns_none(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"id": 1, "state": "approved", "user": {"login": "owner"}},
            "pull_request": {},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review", payload
        )
        assert result is None

    def test_not_allowed_user_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"id": 1, "state": "approved", "user": {"login": "stranger"}},
            "pull_request": {"number": 3},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review", payload
        )
        assert result is None

    def test_commented_review_collapsed_by_oracle(self, tmp_path: Path) -> None:
        """Oracle collapses state=commented reviews — inline comments are handled
        individually by pull_request_review_comment events, so the review-level
        event must not also dispatch."""
        cfg = _config(tmp_path)
        oracle = WebhookIngressOracle()
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"id": 77, "state": "commented", "user": {"login": "owner"}},
            "pull_request": {"number": 4},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review",
            payload,
            delivery_id="delivery-commented-1",
            oracle=oracle,
        )
        assert result is None, "commented review should be collapsed (None)"

    def test_approved_review_not_collapsed_by_oracle(self, tmp_path: Path) -> None:
        """Oracle must NOT collapse approved reviews — the worker must wake
        immediately rather than waiting for the next poll cycle (~60 s lag)."""
        cfg = _config(tmp_path)
        oracle = WebhookIngressOracle()
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"id": 78, "state": "approved", "user": {"login": "owner"}},
            "pull_request": {"number": 5},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review",
            payload,
            delivery_id="delivery-approved-1",
            oracle=oracle,
        )
        assert result is not None, "approved review must not be collapsed"

    def test_changes_requested_not_collapsed_by_oracle(self, tmp_path: Path) -> None:
        """changes_requested is a decisive review state — must dispatch so the
        worker wakes up without waiting for the next poll cycle."""
        cfg = _config(tmp_path)
        oracle = WebhookIngressOracle()
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {
                "id": 79,
                "state": "changes_requested",
                "user": {"login": "owner"},
            },
            "pull_request": {"number": 6},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review",
            payload,
            delivery_id="delivery-changes-requested-1",
            oracle=oracle,
        )
        assert result is not None, "changes_requested review must not be collapsed"

    def test_dismissed_not_collapsed_by_oracle(self, tmp_path: Path) -> None:
        """dismissed is a decisive review state — must dispatch so the worker
        wakes up without waiting for the next poll cycle."""
        cfg = _config(tmp_path)
        oracle = WebhookIngressOracle()
        payload = {
            **_payload(),
            "action": "submitted",
            "review": {"id": 80, "state": "dismissed", "user": {"login": "owner"}},
            "pull_request": {"number": 7},
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review",
            payload,
            delivery_id="delivery-dismissed-1",
            oracle=oracle,
        )
        assert result is not None, "dismissed review must not be collapsed"


class TestDispatchCheckRunNoPrs:
    def test_no_prs(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "completed",
            "check_run": {
                "conclusion": "failure",
                "name": "test",
                "pull_requests": [],
            },
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "check_run", payload
        )
        assert result is not None
        assert "unknown PR" in result.prompt


class TestDispatchIssueCommentSelf:
    def test_self_comment_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "fido-can-code"}},
            "issue": {
                "number": 10,
                "title": "t",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "issue_comment", payload
        )
        assert result is None

    def test_unallowed_user_ignored(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {"id": 1, "body": "hi", "user": {"login": "stranger"}},
            "issue": {
                "number": 10,
                "title": "t",
                "pull_request": {"url": "https://api.github.com/..."},
            },
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "issue_comment", payload
        )
        assert result is None


class TestDispatchReviewCommentNoNumber:
    def test_no_number_after_self_check(self, tmp_path: Path) -> None:
        """Non-self user, but pr has no number → returns None (line 81)."""
        cfg = _config(tmp_path)
        payload = {
            **_payload(),
            "action": "created",
            "comment": {
                "id": 1,
                "body": "hi",
                "user": {"login": "owner"},  # allowed, not self
                "html_url": "https://example.com",
            },
            "pull_request": {},  # no number
        }
        result = Dispatcher(cfg, _repo_cfg(tmp_path), MagicMock()).dispatch(
            "pull_request_review_comment", payload
        )
        assert result is None


class _FakeRescopeTriggerStore:
    """Codex P1 ninth/tenth rounds on PR #1938: hand-rolled stand-in
    for the rescope-intent outbox used in
    ``_BackgroundRescopeTrigger`` tests."""

    def __init__(self, *, claims_succeed: bool) -> None:
        self._claims_succeed = claims_succeed
        self.claimed_ids: list[int] = []
        self.applied_ids: list[int] = []
        self.released_ids: list[int] = []

    def claim_rescope_intent(
        self,
        *,
        intent_comment_id: int,
        change_request: str,
        intent_timestamp: str,
        author: str,
        comment_type: str,
        repo: str,
        pr_number: int,
    ) -> bool:
        self.claimed_ids.append(intent_comment_id)
        return self._claims_succeed

    def mark_rescope_intent_applied(self, intent_comment_id: int) -> None:
        self.applied_ids.append(intent_comment_id)

    def release_rescope_intent_claim(self, intent_comment_id: int) -> None:
        self.released_ids.append(intent_comment_id)


class TestBackgroundRescopeTrigger:
    """_BackgroundRescopeTrigger delegates to Dispatcher.reorder_tasks_background."""

    def _make_intent(
        self,
        change_request: str = "Add logging",
        comment_id: int = 42,
        timestamp: str = "2024-01-15T10:00:00+00:00",
    ) -> RescopeIntent:
        return RescopeIntent(
            change_request=change_request,
            comment_id=comment_id,
            timestamp=timestamp,
        )

    def test_trigger_rescope_calls_reorder_tasks_background(self) -> None:
        fake_dispatcher = _FakeDispatcher()
        fake_registry = MagicMock(spec=ActivityReporter)
        intent = self._make_intent("Add logging to the handler")

        trigger = _BackgroundRescopeTrigger(
            fake_registry,
            agent=MagicMock(),
            prompts=Prompts("p"),
            dispatcher=fake_dispatcher,
            store=_FakeRescopeTriggerStore(claims_succeed=True),
        )
        trigger.trigger_rescope(intent)

        assert len(fake_dispatcher.reorder_tasks_background_calls) == 1
        args, _ = fake_dispatcher.reorder_tasks_background_calls[0]
        assert args[0] == "Add logging to the handler"

    def test_trigger_rescope_passes_intent_as_intents_list(self) -> None:
        """_BackgroundRescopeTrigger wraps the intent in a single-item list."""
        fake_dispatcher = _FakeDispatcher()
        intent = self._make_intent("Refactor tests", comment_id=77)

        trigger = _BackgroundRescopeTrigger(
            MagicMock(spec=ActivityReporter),
            agent=MagicMock(),
            prompts=Prompts("p"),
            dispatcher=fake_dispatcher,
            store=_FakeRescopeTriggerStore(claims_succeed=True),
        )
        trigger.trigger_rescope(intent)

        _, kwargs = fake_dispatcher.reorder_tasks_background_calls[0]
        assert kwargs["intents"] == [intent]

    def test_trigger_rescope_passes_registry_and_agent(self) -> None:
        """_BackgroundRescopeTrigger forwards registry, agent, and prompts."""
        fake_dispatcher = _FakeDispatcher()
        fake_registry = MagicMock(spec=ActivityReporter)
        fake_agent = MagicMock()
        fake_prompts = Prompts("p")

        trigger = _BackgroundRescopeTrigger(
            fake_registry,
            agent=fake_agent,
            prompts=fake_prompts,
            dispatcher=fake_dispatcher,
            store=_FakeRescopeTriggerStore(claims_succeed=True),
        )
        trigger.trigger_rescope(self._make_intent("Refactor the parser"))

        _, kwargs = fake_dispatcher.reorder_tasks_background_calls[0]
        assert kwargs["agent"] is fake_agent
        assert kwargs["prompts"] is fake_prompts

    def test_trigger_rescope_skips_when_store_claims_fail(self) -> None:
        # Codex P1 (ninth round) on PR #1938: when the durable
        # ``claim_rescope_trigger`` returns False, the rescope was
        # already triggered (this Fido process or another after a
        # restart) — skip the dispatch.  Prevents replay of the
        # post-reply effects from re-firing rescope/preempt.
        fake_dispatcher = _FakeDispatcher()
        store = _FakeRescopeTriggerStore(claims_succeed=False)
        trigger = _BackgroundRescopeTrigger(
            MagicMock(spec=ActivityReporter),
            agent=MagicMock(),
            prompts=Prompts("p"),
            dispatcher=fake_dispatcher,
            store=store,
        )
        trigger.trigger_rescope(self._make_intent(comment_id=42))
        assert store.claimed_ids == [42]
        # No dispatch — duplicate rescope was suppressed.
        assert fake_dispatcher.reorder_tasks_background_calls == []

    def test_trigger_rescope_marks_applied_via_on_done(self) -> None:
        # Codex P1 (tenth round) on PR #1938: the durable
        # ``applied`` state advances only after the rescope's
        # ``_on_done`` callback fires (i.e. tasks.json has been
        # synced and the PR description rewritten).  The trigger
        # wires ``_on_done`` to ``mark_rescope_intent_applied`` so
        # the applied marker matches "work durably landed."
        fake_dispatcher = _FakeDispatcher()
        store = _FakeRescopeTriggerStore(claims_succeed=True)
        trigger = _BackgroundRescopeTrigger(
            MagicMock(spec=ActivityReporter),
            agent=MagicMock(),
            prompts=Prompts("p"),
            dispatcher=fake_dispatcher,
            store=store,
        )
        trigger.trigger_rescope(self._make_intent(comment_id=77))
        # The trigger should have passed an _on_done callable.
        assert len(fake_dispatcher.reorder_tasks_background_calls) == 1
        _, kwargs = fake_dispatcher.reorder_tasks_background_calls[0]
        on_done = kwargs["_on_done"]
        # Applied marker fires only AFTER the on_done callback runs.
        assert store.applied_ids == []
        on_done()
        assert store.applied_ids == [77]

    def test_trigger_rescope_releases_claim_on_synchronous_failure(self) -> None:
        # Codex P1 (tenth round) on PR #1938: a synchronous failure
        # between claim and dispatch must release the pending claim
        # so the next trigger fires.  Without release, the next
        # attempt sees the durable claim, skips dispatch, and the
        # rescope never runs.
        class _BoomDispatcher:
            def reorder_tasks_background(self, *_args: object, **_kw: object) -> None:
                raise RuntimeError("thread-start failed")

        store = _FakeRescopeTriggerStore(claims_succeed=True)
        trigger = _BackgroundRescopeTrigger(
            MagicMock(spec=ActivityReporter),
            agent=MagicMock(),
            prompts=Prompts("p"),
            dispatcher=_BoomDispatcher(),  # type: ignore[arg-type]
            store=store,
        )
        with pytest.raises(RuntimeError, match="thread-start failed"):
            trigger.trigger_rescope(self._make_intent(comment_id=88))
        assert store.claimed_ids == [88]
        assert store.released_ids == [88]
        assert store.applied_ids == []

    def test_trigger_rescope_claims_before_dispatch(self) -> None:
        # Claim happens before the dispatch so a crash mid-dispatch
        # doesn't leave the claim un-set (replay would re-fire).
        fake_dispatcher = _FakeDispatcher()
        store = _FakeRescopeTriggerStore(claims_succeed=True)
        trigger = _BackgroundRescopeTrigger(
            MagicMock(spec=ActivityReporter),
            agent=MagicMock(),
            prompts=Prompts("p"),
            dispatcher=fake_dispatcher,
            store=store,
        )
        trigger.trigger_rescope(self._make_intent(comment_id=99))
        assert store.claimed_ids == [99]
        assert len(fake_dispatcher.reorder_tasks_background_calls) == 1


class TestReplyToCommentElseBranch:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_synthesis_reply_returns_answer_without_change_request(
        self, tmp_path: Path
    ) -> None:
        """Synthesis path: no change_request → ANSWER."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 50},
            comment_body="do something",
            is_bot=False,
        )
        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = []
        mock_gh.reply_to_review_comment.return_value = {"id": 999}
        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "I'll look into this."
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )
        assert cat == "ANSWER"
        assert titles == []

    def test_gh_post_exception_propagates(self, tmp_path: Path) -> None:
        """Exception in reply_to_review_comment propagates so callers fail closed."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 51},
            comment_body="please fix this",
            is_bot=False,
        )

        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = []
        mock_gh.reply_to_review_comment.side_effect = RuntimeError("network down")
        with pytest.raises(RuntimeError, match="network down"):
            Dispatcher(
                cfg,
                self._repo_cfg(tmp_path),
                mock_gh,
                call_synthesis_fn=lambda *a, **kw: _synthesis_response("I'll fix it."),
            ).reply_to_comment(
                action,
                agent=_client(),
                registry=MagicMock(spec=ActivityReporter),
            )

    def test_skips_review_reply_when_artifact_already_recorded(
        self, tmp_path: Path
    ) -> None:
        cfg = self._cfg(tmp_path)
        repo_cfg = self._repo_cfg(tmp_path)
        store = FidoStore(tmp_path)
        promise = store.prepare_reply(
            owner="worker", comment_type="pulls", anchor_comment_id=52
        )
        assert promise is not None
        store.claim_reply_outbox_effect(
            promise_id=promise.promise_id,
            delivery_id="delivery-52",
            origin_id=52,
        )
        _record_reply_artifact(
            repo_cfg,
            artifact_comment_id=9052,
            comment_type="pulls",
            lane_key="pulls:owner/repo:1:thread:52",
            promise_ids=(promise.promise_id,),
        )
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 52},
            comment_body="please fix this",
            is_bot=False,
            context={"reply_promise_id": promise.promise_id},
        )

        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 52, "author": "owner", "body": "please fix this"}
        ]
        cat, titles = Dispatcher(
            cfg,
            repo_cfg,
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "I'll fix it.", change_request="Fix it"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        assert cat == "ACT"
        assert titles == ["Fix it"]
        mock_gh.reply_to_review_comment.assert_not_called()


# ── reply_to_comment: thread re-fetch before posting ─────────────────────────


class TestReplyToCommentThreadRefetch:
    """The thread is re-fetched from GitHub right before posting so the
    edit-vs-post decision uses current state, not the stale snapshot from
    before triage.  Closes #438."""

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path) -> RepoConfig:
        return RepoConfig(name="owner/repo", work_dir=tmp_path)

    def test_fetch_comment_thread_called_twice(self, tmp_path: Path) -> None:
        """fetch_comment_thread is called once for context (before synthesis) and
        once right before posting (after synthesis)."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 500},
            comment_body="please refactor this",
            is_bot=False,
        )

        mock_gh = MagicMock()
        mock_gh.fetch_comment_thread.return_value = [
            {"id": 500, "author": "reviewer", "body": "please refactor this"}
        ]
        mock_gh.reply_to_review_comment.return_value = {"id": 999}

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "On it!", change_request="Refactor this module"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Must be called exactly twice: initial context fetch + pre-post re-fetch
        assert mock_gh.fetch_comment_thread.call_count == 2
        mock_gh.fetch_comment_thread.assert_called_with("owner/repo", 1, 500)

    def test_refetch_result_used_for_edit_vs_post(self, tmp_path: Path) -> None:
        """The edit-vs-post decision uses re-fetched thread state, not the
        stale initial snapshot.  When the initial fetch shows Fido as last
        speaker (→ would edit) but the re-fetch reveals a human replied since
        (→ should post fresh), the re-fetch data wins: a new reply is posted.
        Note: the Fido reply ID is in both fetches, so the concurrent-skip
        guard is not triggered."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 501},
            comment_body="add type hints",
            is_bot=False,
        )

        mock_gh = MagicMock()
        mock_gh.reply_to_review_comment.return_value = {"id": 999}
        call_count = 0

        def fetch_side_effect(repo: str, pr: int, cid: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial fetch: Fido was last speaker (stale snapshot)
                return [
                    {"id": 501, "author": "reviewer", "body": "add type hints"},
                    {"id": 502, "author": "fidocancode", "body": "Got it!"},
                ]
            else:
                # Re-fetch: human commented after the initial fetch
                return [
                    {"id": 501, "author": "reviewer", "body": "add type hints"},
                    {"id": 502, "author": "fidocancode", "body": "Got it!"},
                    {"id": 503, "author": "reviewer", "body": "also type the return"},
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response("Will do!"),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Re-fetch shows human is last → post new reply, not edit
        # (Fido ID 502 existed in initial, so concurrent-skip is NOT triggered)
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_refetch_human_comment_added_during_synthesis_triggers_new_post(
        self, tmp_path: Path
    ) -> None:
        """If a human comments AFTER the initial fetch but BEFORE the re-fetch,
        the fresh data shows them as last speaker, so a new reply is posted
        rather than editing the old Fido reply."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 503},
            comment_body="please add tests",
            is_bot=False,
        )

        mock_gh = MagicMock()
        mock_gh.reply_to_review_comment.return_value = {"id": 999}
        call_count = 0

        def fetch_side_effect(repo: str, pr: int, cid: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial fetch: Fido is last speaker
                return [
                    {"id": 503, "author": "reviewer", "body": "please add tests"},
                    {"id": 504, "author": "fidocancode", "body": "On it!"},
                ]
            else:
                # Re-fetch: human replied after the initial fetch
                return [
                    {"id": 503, "author": "reviewer", "body": "please add tests"},
                    {"id": 504, "author": "fidocancode", "body": "On it!"},
                    {
                        "id": 505,
                        "author": "reviewer",
                        "body": "also add integration tests",
                    },
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Adding tests now!", change_request="Add tests"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Fresh data shows human is last speaker → post new reply, never edit
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_refetch_returns_empty_falls_back_to_initial(self, tmp_path: Path) -> None:
        """If the re-fetch returns empty/None (e.g. race with deletion), the
        stale initial snapshot is kept and posting proceeds normally."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 506},
            comment_body="fix the import",
            is_bot=False,
        )

        mock_gh = MagicMock()
        mock_gh.reply_to_review_comment.return_value = {"id": 999}
        call_count = 0

        def fetch_side_effect(repo: str, pr: int, cid: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 506, "author": "reviewer", "body": "fix the import"}]
            else:
                return None  # re-fetch returned nothing

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Fixed!", change_request="Fix the import"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Falls back to initial snapshot (no Fido reply) → posts new reply
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_skips_post_when_concurrent_fido_reply_detected(
        self, tmp_path: Path
    ) -> None:
        """If a new Fido reply appears in the re-fetch that wasn't in the
        initial snapshot, a concurrent handler already replied — skip to
        avoid duplicates.  Closes #438."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 507},
            comment_body="please add docstrings",
            is_bot=False,
        )

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo: str, pr: int, cid: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial fetch: no Fido reply yet
                return [
                    {"id": 507, "author": "reviewer", "body": "please add docstrings"}
                ]
            else:
                # Re-fetch: concurrent handler posted a Fido reply to THIS
                # comment (in_reply_to_id == 507) during synthesis.
                return [
                    {"id": 507, "author": "reviewer", "body": "please add docstrings"},
                    {
                        "id": 508,
                        "author": "fidocancode",
                        "body": "On it!",
                        "in_reply_to_id": 507,
                    },
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        cat, titles = Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Woof, on it!", change_request="Add docstrings"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Concurrent handler already replied — neither post nor edit is called
        mock_gh.reply_to_review_comment.assert_not_called()
        mock_gh.edit_review_comment.assert_not_called()
        # Synthesis result is still returned so the caller can queue tasks
        assert cat == "ACT"
        assert titles == ["Add docstrings"]

    def test_no_skip_when_concurrent_reply_is_to_sibling_comment(
        self, tmp_path: Path
    ) -> None:
        """A Fido reply that appeared during synthesis but targets a *different*
        comment (sibling in the same review) must NOT trip the skip — that
        was the #1004 silent-drop bug.  Closes #1004."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 507},
            comment_body="please add docstrings",
            is_bot=False,
        )

        mock_gh = MagicMock()
        mock_gh.reply_to_review_comment.return_value = {"id": 999}
        call_count = 0

        def fetch_side_effect(repo: str, pr: int, cid: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [
                    {"id": 507, "author": "reviewer", "body": "please add docstrings"}
                ]
            # Re-fetch: a sibling comment (id 600) got a fido reply (id 601)
            # while we were in synthesis — that's NOT a reply to OUR comment.
            return [
                {"id": 507, "author": "reviewer", "body": "please add docstrings"},
                {
                    "id": 601,
                    "author": "fidocancode",
                    "body": "Sibling reply",
                    "in_reply_to_id": 600,
                },
            ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Woof, on it!", change_request="Add docstrings"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Sibling-comment reply must NOT skip our post — we still reply.
        mock_gh.reply_to_review_comment.assert_called_once()

    def test_no_skip_when_fido_reply_was_already_in_initial_fetch(
        self, tmp_path: Path
    ) -> None:
        """A Fido reply that existed in the initial fetch is not treated as
        a concurrent duplicate — post proceeds normally."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 509},
            comment_body="looks good to me",
            is_bot=False,
        )

        mock_gh = MagicMock()
        mock_gh.reply_to_review_comment.return_value = {"id": 999}

        def fetch_side_effect(repo: str, pr: int, cid: int) -> object:
            # Both fetches return the same Fido reply — it was already there
            return [
                {"id": 509, "author": "reviewer", "body": "looks good"},
                {"id": 510, "author": "fidocancode", "body": "On it!"},
            ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Thanks for the feedback!"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Posted replies are immutable; Fido posts a new reply instead.
        mock_gh.reply_to_review_comment.assert_called_once()
        mock_gh.edit_review_comment.assert_not_called()

    def test_skips_post_fido_can_code_login_also_detected(self, tmp_path: Path) -> None:
        """The 'fido-can-code' login is also recognised as a Fido reply
        when checking for concurrent duplicates."""
        cfg = self._cfg(tmp_path)
        action = Action(
            prompt="comment",
            reply_to={"repo": "owner/repo", "pr": 1, "comment_id": 511},
            comment_body="fix the typo",
            is_bot=False,
        )

        mock_gh = MagicMock()
        call_count = 0

        def fetch_side_effect(repo: str, pr: int, cid: int) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [{"id": 511, "author": "reviewer", "body": "fix the typo"}]
            else:
                # Concurrent reply from the fido-can-code login variant,
                # threaded under THIS comment.
                return [
                    {"id": 511, "author": "reviewer", "body": "fix the typo"},
                    {
                        "id": 512,
                        "author": "fido-can-code",
                        "body": "Fixed!",
                        "in_reply_to_id": 511,
                    },
                ]

        mock_gh.fetch_comment_thread.side_effect = fetch_side_effect

        Dispatcher(
            cfg,
            self._repo_cfg(tmp_path),
            mock_gh,
            call_synthesis_fn=lambda *a, **kw: _synthesis_response(
                "Fixed the typo!", change_request="Fix the typo"
            ),
        ).reply_to_comment(
            action,
            agent=_client(),
            registry=MagicMock(spec=ActivityReporter),
        )

        # Concurrent Fido reply detected (via fido-can-code) — skip
        mock_gh.reply_to_review_comment.assert_not_called()
        mock_gh.edit_review_comment.assert_not_called()


# ── _rewrite_pr_description ───────────────────────────────────────────────────


class TestRewritePrDescription:
    @pytest.fixture(autouse=True)
    def _init_git(self, tmp_path: Path) -> None:
        """Initialize a git repo so pr_body_lock can resolve .git."""
        import subprocess as sp

        sp.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)

    def _pr_body(self, desc: str = "Does something useful.\n\nFixes #42.") -> str:
        return (
            f"{desc}\n\n---\n\n## Work queue\n\n"
            "<!-- WORK_QUEUE_START -->\n- [ ] do a thing\n<!-- WORK_QUEUE_END -->"
        )

    def _mock_gh(self, body: str | None = None) -> MagicMock:
        gh = MagicMock()
        gh.get_repo_info.return_value = "owner/repo"
        gh.get_user.return_value = "fido"
        gh.find_pr.return_value = {"number": 99, "state": "OPEN"}
        gh.get_pr_body.return_value = body if body is not None else self._pr_body()
        return gh

    def _mock_state(self, issue: int | None = 42) -> MagicMock:
        state = MagicMock()
        state.load.return_value = {"issue": issue} if issue else {}
        return state

    def _mock_tasks(self, task_list: list | None = None) -> MagicMock:
        tasks = MagicMock()
        tasks.list.return_value = task_list if task_list is not None else []
        return tasks

    def test_skips_when_no_issue_in_state(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client(),
            _state=self._mock_state(issue=None),
        )
        mock_gh.edit_pr_body.assert_not_called()

    def test_raises_on_get_repo_exception(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.get_repo_info.side_effect = RuntimeError("network error")
        with pytest.raises(RuntimeError, match="network error"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(),
                _state=self._mock_state(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_skips_when_no_open_pr(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.find_pr.return_value = None
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client(),
            _state=self._mock_state(),
        )
        mock_gh.edit_pr_body.assert_not_called()

    def test_skips_when_pr_not_open(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.find_pr.return_value = {"number": 99, "state": "MERGED"}
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client(),
            _state=self._mock_state(),
        )
        mock_gh.edit_pr_body.assert_not_called()

    def test_raises_on_get_pr_body_exception(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.get_pr_body.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="API error"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_self_heals_when_no_divider_in_body(self, tmp_path: Path) -> None:
        """#1335 regression: divider-less bodies must self-heal, not raise."""
        mock_gh = self._mock_gh(
            body="No divider here. <!-- WORK_QUEUE_START -->x<!-- WORK_QUEUE_END -->"
        )
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
        )
        body = mock_gh.edit_pr_body.call_args[0][2]
        assert "\n\n---\n\n" in body
        assert "<!-- WORK_QUEUE_START -->" in body

    def test_raises_when_opus_returns_empty(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        with pytest.raises(ValueError, match="run_turn returned empty"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client(""),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )
        mock_gh.edit_pr_body.assert_not_called()

    def test_updates_pr_body_with_new_description(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New description.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
        )
        mock_gh.edit_pr_body.assert_called_once()
        new_body = mock_gh.edit_pr_body.call_args[0][2]
        assert "New description." in new_body

    def test_preserves_work_queue_section(self, tmp_path: Path) -> None:
        """work queue start/end markers are preserved even with an empty task list."""
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>Updated description.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(
                [
                    {
                        "id": "1",
                        "title": "do a thing",
                        "type": "spec",
                        "status": "pending",
                    }
                ]
            ),
        )
        new_body = mock_gh.edit_pr_body.call_args[0][2]
        assert "<!-- WORK_QUEUE_START -->" in new_body
        assert "<!-- WORK_QUEUE_END -->" in new_body

    def test_description_replaces_only_before_divider(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>Fresh desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
        )
        new_body = mock_gh.edit_pr_body.call_args[0][2]
        assert "Does something useful." not in new_body
        assert "Fresh desc." in new_body
        assert "## Work queue" in new_body

    def test_raises_on_edit_pr_body_exception(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        mock_gh.edit_pr_body.side_effect = RuntimeError("write failed")
        with pytest.raises(RuntimeError, match="write failed"):
            _rewrite_pr_description(
                tmp_path,
                mock_gh,
                agent=_client("<body>New desc.\n\nFixes #42.</body>"),
                _state=self._mock_state(),
                _tasks=self._mock_tasks(),
            )

    def test_defaults_to_none_agent(self, tmp_path: Path) -> None:
        mock_gh = self._mock_gh()
        write_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def capture_write(*args: object, **kwargs: object) -> None:
            write_calls.append((args, kwargs))

        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
            _write_fn=capture_write,
        )
        assert len(write_calls) == 1
        assert write_calls[0][1].get("agent") is None

    def test_defaults_to_state(self, tmp_path: Path) -> None:
        # Without an explicit _state, _rewrite_pr_description constructs a
        # real State from tmp_path/.git/fido. Since no state.json exists there,
        # State.load() returns {}, issue is None, and the function returns early.
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client(),
            _tasks=self._mock_tasks(),
        )
        mock_gh.edit_pr_body.assert_not_called()

    def test_does_not_retry_when_task_list_unchanged(self, tmp_path: Path) -> None:
        """When task list is stable, description is written exactly once."""
        task = {"id": "t1", "status": "pending", "title": "Do a thing"}
        tasks = MagicMock()
        tasks.list.return_value = [task]  # same list every call
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
        )
        mock_gh.edit_pr_body.assert_called_once()

    def test_retries_when_task_list_changes_during_opus(self, tmp_path: Path) -> None:
        """If task list changes while Opus runs, the description is rewritten."""
        task_before = {"id": "t1", "status": "pending", "title": "Do a thing"}
        task_after = {"id": "t2", "status": "pending", "title": "New task"}
        tasks = MagicMock()
        # list() called: before attempt 1, after attempt 1, before attempt 2, after attempt 2
        tasks.list.side_effect = [
            [task_before],  # snapshot before attempt 1
            [task_after],  # snapshot after attempt 1 (changed → retry)
            [task_after],  # snapshot before attempt 2
            [task_after],  # snapshot after attempt 2 (stable → done)
        ]
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
        )
        assert mock_gh.edit_pr_body.call_count == 2

    def test_stops_after_max_retries(self, tmp_path: Path) -> None:
        """Never retries more than _max_retries times even if task list keeps changing."""
        tasks = MagicMock()
        call_count = [0]

        def ever_changing() -> object:
            n = call_count[0]
            call_count[0] += 1
            return [{"id": str(n), "status": "pending", "title": f"task {n}"}]

        tasks.list.side_effect = ever_changing
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
            _max_retries=3,
        )
        assert mock_gh.edit_pr_body.call_count == 3

    def test_no_divider_self_heals(self, tmp_path: Path) -> None:
        """#1335 regression: a body with no --- divider self-heals."""
        mock_gh = self._mock_gh(body="No divider here. Nothing.")
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=self._mock_tasks(),
        )
        body = mock_gh.edit_pr_body.call_args[0][2]
        assert "\n\n---\n\n" in body
        assert "<!-- WORK_QUEUE_START -->" in body

    def test_refetches_pr_body_on_retry(self, tmp_path: Path) -> None:
        """PR body is re-fetched on each attempt so work-queue stays current."""
        task_before = {"id": "t1", "status": "pending", "title": "Before"}
        task_after = {"id": "t2", "status": "pending", "title": "After"}
        tasks = MagicMock()
        tasks.list.side_effect = [
            [task_before],
            [task_after],  # changed → retry
            [task_after],
            [task_after],  # stable
        ]
        mock_gh = self._mock_gh()
        _rewrite_pr_description(
            tmp_path,
            mock_gh,
            agent=_client("<body>New desc.\n\nFixes #42.</body>"),
            _state=self._mock_state(),
            _tasks=tasks,
        )
        assert mock_gh.get_pr_body.call_count == 2  # once per attempt


# ── _task_snapshot ────────────────────────────────────────────────────────────


class TestTaskSnapshot:
    def test_returns_id_status_title_tuples(self) -> None:
        tasks = [
            {"id": "a", "status": "pending", "title": "Do A"},
            {"id": "b", "status": "completed", "title": "Done B"},
        ]
        assert _task_snapshot(tasks) == [
            ("a", "pending", "Do A"),
            ("b", "completed", "Done B"),
        ]

    def test_empty_list(self) -> None:
        assert _task_snapshot([]) == []

    def test_order_is_preserved(self) -> None:
        tasks = [
            {"id": "z", "status": "pending", "title": "Z"},
            {"id": "a", "status": "pending", "title": "A"},
        ]
        result = _task_snapshot(tasks)
        assert result[0][0] == "z"
        assert result[1][0] == "a"


# ── _load_active_context_for_rescope ─────────────────────────────────────────


class TestLoadActiveContextForRescope:
    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_returns_none_none_when_no_state_file(self, tmp_path: Path) -> None:
        gh = MagicMock()
        issue, pr = _load_active_context_for_rescope(tmp_path, "owner/repo", gh)
        assert issue is None
        assert pr is None
        gh.view_issue.assert_not_called()

    def test_returns_none_none_when_no_issue_in_state(self, tmp_path: Path) -> None:
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"pr_number": 5})
        gh = MagicMock()
        issue, pr = _load_active_context_for_rescope(tmp_path, "owner/repo", gh)
        assert issue is None
        assert pr is None
        gh.view_issue.assert_not_called()

    def test_returns_active_issue_when_issue_in_state(self, tmp_path: Path) -> None:
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 7})
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "Fix crash", "body": "It crashes."}
        issue, _ = _load_active_context_for_rescope(tmp_path, "owner/repo", gh)
        assert isinstance(issue, ActiveIssue)
        assert issue.number == 7
        assert issue.title == "Fix crash"
        assert issue.body == "It crashes."
        gh.view_issue.assert_called_once_with("owner/repo", 7)

    def test_returns_none_pr_when_no_pr_in_state(self, tmp_path: Path) -> None:
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 7})
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "Fix crash", "body": ""}
        _, pr = _load_active_context_for_rescope(tmp_path, "owner/repo", gh)
        assert pr is None
        gh.get_pr.assert_not_called()

    def test_returns_active_pr_when_pr_in_state(self, tmp_path: Path) -> None:
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 7, "pr_number": 42})
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "Fix crash", "body": ""}
        gh.get_pr.return_value = {"title": "Fix crash (closes #7)", "body": "PR body."}
        _, pr = _load_active_context_for_rescope(tmp_path, "owner/repo", gh)
        assert isinstance(pr, ActivePR)
        assert pr.number == 42
        assert pr.title == "Fix crash (closes #7)"
        assert pr.body == "PR body."
        assert pr.url == "https://github.com/owner/repo/pull/42"
        gh.get_pr.assert_called_once_with("owner/repo", 42)


# ── _make_reorder_kwargs active-context injection ────────────────────────────


class TestMakeReorderKwargsActiveContext:
    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def _repo_cfg(self, tmp_path: Path, name: str = "owner/repo") -> _RepoConfig:
        return _RepoConfig(
            name=name,
            work_dir=tmp_path,
            provider=ProviderID.CLAUDE_CODE,
        )

    def _fido_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / ".git" / "fido"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_no_issue_key_omitted_when_no_state(self, tmp_path: Path) -> None:
        gh = MagicMock()
        kwargs = Dispatcher(
            self._cfg(tmp_path), self._repo_cfg(tmp_path), gh
        )._make_reorder_kwargs(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert "issue" not in kwargs
        assert "pr" not in kwargs

    def test_issue_included_when_state_has_issue(self, tmp_path: Path) -> None:
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 5})
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "Do the thing", "body": "Details."}
        kwargs = Dispatcher(
            self._cfg(tmp_path), self._repo_cfg(tmp_path), gh
        )._make_reorder_kwargs(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert "issue" in kwargs
        issue = kwargs["issue"]
        assert isinstance(issue, ActiveIssue)
        assert issue.number == 5
        assert issue.title == "Do the thing"
        assert issue.body == "Details."

    def test_pr_included_when_state_has_pr_number(self, tmp_path: Path) -> None:
        fido_dir = self._fido_dir(tmp_path)
        State(fido_dir).save({"issue": 5, "pr_number": 99})
        gh = MagicMock()
        gh.view_issue.return_value = {"title": "t", "body": ""}
        gh.get_pr.return_value = {"title": "Fix it (closes #5)", "body": ""}
        kwargs = Dispatcher(
            self._cfg(tmp_path), self._repo_cfg(tmp_path), gh
        )._make_reorder_kwargs(MagicMock(), MagicMock(), MagicMock(), MagicMock())
        assert "pr" in kwargs
        pr = kwargs["pr"]
        assert isinstance(pr, ActivePR)
        assert pr.number == 99
        assert pr.url == "https://github.com/owner/repo/pull/99"


class TestMakeReorderKwargsAfterApply:
    """Codex P1 tenth round on PR #1938: ``_make_reorder_kwargs``
    chains a caller-supplied ``after_apply`` callback after the
    built-in sync/rewrite on_done sequence.  The rescope-intent
    outbox uses this to advance ``state='applied'`` ONLY after the
    durable apply completes."""

    def _cfg(self, tmp_path: Path) -> Config:
        return Config(
            port=9000,
            secret=b"test",
            repos={"owner/repo": RepoConfig(name="owner/repo", work_dir=tmp_path)},
            allowed_bots=frozenset(),
            log_level="WARNING",
            sub_dir=tmp_path / "sub",
        )

    def test_after_apply_fires_when_on_done_invoked(self, tmp_path: Path) -> None:
        calls: list[str] = []
        gh = MagicMock()

        def rewrite_fn(*_a: object, **_kw: object) -> None:
            calls.append("rewrite")

        def sync_fn(_w: Path, _g: object) -> None:
            calls.append("sync")

        def after_apply() -> None:
            calls.append("after_apply")

        kwargs = Dispatcher(
            self._cfg(tmp_path), self._cfg(tmp_path).repos["owner/repo"], gh
        )._make_reorder_kwargs(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            rewrite_fn,
            sync_fn,
            after_apply=after_apply,
        )
        kwargs["_on_done"]()
        # after_apply runs LAST — after sync and rewrite, so the
        # applied-marker only advances on successful durable apply.
        assert calls == ["sync", "rewrite", "after_apply"]


# ---------------------------------------------------------------------------
# _GitHubInsightFiler
# ---------------------------------------------------------------------------


class TestGitHubInsightFiler:
    """_GitHubInsightFiler files Insight observations as GitHub issues."""

    def _make_insight(
        self,
        title: str = "Interesting observation",
        hook: str = "Hook sentence.",
        why: str = "Why it matters.",
    ) -> Insight:
        return Insight(title=title, hook=hook, why=why)

    def _make_target(
        self,
        repo: str = "owner/repo",
        pr: int = 42,
        comment_id: int = 100,
        comment_type: str = "pulls",
    ) -> CommentTarget:
        return CommentTarget(
            repo=repo, pr=pr, comment_id=comment_id, comment_type=comment_type
        )

    def test_creates_issue_when_none_exists(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/99"
        filer = _GitHubInsightFiler(gh)
        insight = self._make_insight()
        target = self._make_target()

        filer.file_insight(insight, target)

        gh.create_issue.assert_called_once()

    def test_skips_creation_when_issue_already_exists(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = [
            {"html_url": f"https://github.com/{_INSIGHT_REPO}/issues/5"}
        ]
        filer = _GitHubInsightFiler(gh)

        filer.file_insight(self._make_insight(), self._make_target())

        gh.create_issue.assert_not_called()

    def test_issue_title_uses_insight_title(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/1"
        filer = _GitHubInsightFiler(gh)

        filer.file_insight(
            self._make_insight(title="My Observation"), self._make_target()
        )

        title = gh.create_issue.call_args.args[1]
        assert title == "Insight: My Observation"

    def test_issue_filed_against_insight_repo(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/2"
        filer = _GitHubInsightFiler(gh)

        filer.file_insight(self._make_insight(), self._make_target())

        repo = gh.create_issue.call_args.args[0]
        assert repo == _INSIGHT_REPO

    def test_issue_has_insight_label(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/3"
        filer = _GitHubInsightFiler(gh)

        filer.file_insight(self._make_insight(), self._make_target())

        labels = (
            gh.create_issue.call_args.kwargs.get("labels")
            or gh.create_issue.call_args.args[3]
        )
        assert _INSIGHT_LABEL in labels

    def test_body_contains_hook_and_why(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/4"
        filer = _GitHubInsightFiler(gh)
        insight = self._make_insight(hook="The hook.", why="The why.")

        filer.file_insight(insight, self._make_target())

        body = gh.create_issue.call_args.args[2]
        assert "The hook." in body
        assert "The why." in body

    def test_body_contains_idempotency_marker(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/5"
        filer = _GitHubInsightFiler(gh)
        target = self._make_target(comment_id=777)

        filer.file_insight(self._make_insight(), target)

        body = gh.create_issue.call_args.args[2]
        assert "<!-- insight-source: 777 -->" in body

    def test_search_uses_idempotency_marker(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/6"
        filer = _GitHubInsightFiler(gh)
        target = self._make_target(comment_id=42)

        filer.file_insight(self._make_insight(), target)

        query = gh.search_issues.call_args.args[1]
        assert "insight-source: 42" in query

    def test_source_link_pulls_format(self) -> None:
        target = CommentTarget(
            repo="org/proj", pr=10, comment_id=555, comment_type="pulls"
        )
        link = _insight_source_link(target)
        assert link == "https://github.com/org/proj/pull/10#discussion_r555"

    def test_source_link_issues_format(self) -> None:
        target = CommentTarget(
            repo="org/proj", pr=10, comment_id=555, comment_type="issues"
        )
        link = _insight_source_link(target)
        assert link == "https://github.com/org/proj/issues/10#issuecomment-555"

    def test_body_contains_source_link(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/7"
        filer = _GitHubInsightFiler(gh)
        target = self._make_target(
            repo="org/proj", pr=10, comment_id=555, comment_type="pulls"
        )

        filer.file_insight(self._make_insight(), target)

        body = gh.create_issue.call_args.args[2]
        assert "https://github.com/org/proj/pull/10#discussion_r555" in body

    # ── HOL-19 / #1913: cross-comment near-duplicate critic ──────────

    def _make_critic_filer(
        self,
        gh: object,
        agent_response: str,
    ) -> _GitHubInsightFiler:
        """Build a critic-enabled filer wired with a hand-rolled agent
        + prompts pair.  Each canned response is one JSON envelope the
        critic will see."""
        from dataclasses import dataclass, field

        @dataclass
        class _Agent:
            response: str
            calls: list[object] = field(default_factory=list)

            def run_turn(self, *args: object, **kwargs: object) -> str:
                self.calls.append((args, kwargs))
                return self.response

        @dataclass
        class _Prompts:
            def insight_dedup_critic_prompt(
                self,
                proposed_insight: dict[str, str],
                recent_insights: list[dict[str, str]],
            ) -> str:
                return "critic-prompt"

        return _GitHubInsightFiler(
            gh,
            agent=_Agent(response=agent_response),  # type: ignore[arg-type]
            prompts=_Prompts(),  # type: ignore[arg-type]
            critic_system_prompt="critic-sys",
        )

    def test_critic_disabled_when_collaborators_omitted(self) -> None:
        """Legacy two-arg construction (no agent) preserves the pre-HOL-19
        behaviour exactly — marker-only dedup, no critic call, no
        ``search_issues`` for recent insights."""
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/8"
        filer = _GitHubInsightFiler(gh)

        filer.file_insight(self._make_insight(), self._make_target())

        # Only the marker query — no second search_issues for the
        # recent-insights critic input.
        assert gh.search_issues.call_count == 1
        gh.create_issue.assert_called_once()

    def test_critic_distinct_proceeds_to_file(self) -> None:
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/9"
        filer = self._make_critic_filer(
            gh, agent_response='{"is_duplicate": false, "rationale": "distinct"}'
        )

        filer.file_insight(self._make_insight(), self._make_target())

        gh.create_issue.assert_called_once()

    def test_critic_duplicate_skips_filing(self) -> None:
        """HOL-19 acceptance: a near-duplicate insight filed against a
        different comment is NOT filed again."""
        gh = MagicMock()
        # First search (marker check): empty.  Second search (recent
        # insights): returns one prior insight.
        gh.search_issues.side_effect = [
            [],
            [
                {
                    "title": "Previously filed",
                    "body": "Same core lesson.",
                    "html_url": "https://github.com/FidoCanCode/home/issues/42",
                }
            ],
        ]
        filer = self._make_critic_filer(
            gh,
            agent_response=(
                '{"is_duplicate": true, '
                '"duplicate_url": "https://github.com/FidoCanCode/home/issues/42", '
                '"rationale": "covers same lesson"}'
            ),
        )

        filer.file_insight(self._make_insight(), self._make_target())

        gh.create_issue.assert_not_called()

    def test_critic_failure_proceeds_to_file(self) -> None:
        """Fail-open: a transport / parse failure must not block legitimate
        insight filing.  The marker check still prevents exact replays."""
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/10"
        filer = self._make_critic_filer(gh, agent_response="not parseable JSON at all")

        filer.file_insight(self._make_insight(), self._make_target())

        gh.create_issue.assert_called_once()

    def test_marker_short_circuits_before_critic_runs(self) -> None:
        """The per-comment idempotency marker is the cheap path — when it
        hits, the (expensive) critic must NOT fire and recent insights
        must NOT be fetched.  Pins the order so a future refactor can't
        accidentally invert it."""
        gh = MagicMock()
        gh.search_issues.return_value = [
            {"html_url": f"https://github.com/{_INSIGHT_REPO}/issues/5"}
        ]
        filer = self._make_critic_filer(gh, agent_response='{"is_duplicate": false}')

        filer.file_insight(self._make_insight(), self._make_target())

        # Only the marker search ran; no recent-insights search.
        assert gh.search_issues.call_count == 1
        gh.create_issue.assert_not_called()

    def test_critic_duplicate_records_skip_marker_on_dup_issue(self) -> None:
        """Codex on PR #1932: when the dedup critic skips, the new
        comment_id must still get a durable marker — otherwise a
        replay + fail-open critic could file the duplicate later.
        Marker is recorded as a comment on the duplicate-of issue."""
        gh = MagicMock()
        gh.search_issues.side_effect = [
            [],  # marker lookup empty
            [  # recent insights for the critic
                {
                    "title": "Previously filed",
                    "body": "Same lesson.",
                    "html_url": "https://github.com/FidoCanCode/home/issues/42",
                }
            ],
        ]
        filer = self._make_critic_filer(
            gh,
            agent_response=(
                '{"is_duplicate": true, '
                '"duplicate_url": "https://github.com/FidoCanCode/home/issues/42", '
                '"rationale": "same lesson"}'
            ),
        )

        filer.file_insight(self._make_insight(), self._make_target(comment_id=777))

        # No NEW issue filed.
        gh.create_issue.assert_not_called()
        # Marker comment posted on the duplicate-of issue.
        gh.comment_issue.assert_called_once()
        repo_arg, number_arg, body_arg = gh.comment_issue.call_args.args
        assert repo_arg == _INSIGHT_REPO
        assert number_arg == 42
        assert "<!-- insight-source: 777 -->" in body_arg

    def test_critic_duplicate_marker_uses_in_body_comments_search(self) -> None:
        """The marker search must include comments — otherwise the
        marker we just wrote on the duplicate-of issue (as a comment,
        not in the body) wouldn't be found on a future replay."""
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/12"
        filer = _GitHubInsightFiler(gh)

        filer.file_insight(self._make_insight(), self._make_target())

        query = gh.search_issues.call_args.args[1]
        # Query MUST include ``in:body,comments`` so the dedup-skip
        # markers (posted as comments) are searchable on replay.
        assert "in:body,comments" in query

    def test_critic_duplicate_cross_repo_url_files_anyway(self) -> None:
        """Codex on PR #1932 (follow-up): the corpus the critic sees
        only contains ``_INSIGHT_REPO`` insights, so a cross-repo
        ``duplicate_url`` is hallucination — parser rejects it,
        runner fails open, insight files normally.  The previous
        two-layer "shape here, repo downstream" design let cross-
        repo URLs silently lose insights (filer skipped while the
        marker-writer also skipped on the wrong repo)."""
        gh = MagicMock()
        gh.search_issues.return_value = []  # parser rejects → no marker lookup
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/99"
        filer = self._make_critic_filer(
            gh,
            agent_response=(
                '{"is_duplicate": true, '
                '"duplicate_url": '
                '"https://github.com/some-other-org/repo/issues/9", '
                '"rationale": "x"}'
            ),
        )

        filer.file_insight(self._make_insight(), self._make_target())

        # Insight files normally — the cross-repo URL is hallucination
        # and the runner failed open with is_duplicate=False.
        gh.create_issue.assert_called_once()
        gh.comment_issue.assert_not_called()

    def test_critic_duplicate_with_unparseable_url_files_anyway(self) -> None:
        """Codex on PR #1932: a duplicate verdict with a malformed
        ``duplicate_url`` (not a GitHub issue URL) is now treated as
        a malformed verdict at parse time — the runner fails open
        with ``is_duplicate=False`` and the insight files normally.
        Previously the filer would skip the filing while the
        marker-writer silently dropped the durability record,
        losing the insight entirely."""
        gh = MagicMock()
        gh.search_issues.side_effect = [
            [],
            [
                {
                    "title": "x",
                    "body": "y",
                    "html_url": "https://github.com/FidoCanCode/home/issues/1",
                }
            ],
        ]
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/99"
        filer = self._make_critic_filer(
            gh,
            agent_response=(
                '{"is_duplicate": true, '
                '"duplicate_url": "not a real URL at all", '
                '"rationale": "x"}'
            ),
        )

        filer.file_insight(self._make_insight(), self._make_target())

        # Filing PROCEEDED (malformed verdict → fail open) — no
        # silently-lost insight.  No marker comment because the
        # filer didn't skip.
        gh.create_issue.assert_called_once()
        gh.comment_issue.assert_not_called()

    def test_critic_duplicate_marker_write_failure_falls_open(self) -> None:
        """Codex on PR #1932: the marker write is best-effort
        bookkeeping at an external-system boundary.  A transient
        GitHub failure (rate limit, network blip, target issue
        closed/transferred between the critic's lookup and our
        comment) must NOT crash dispatch — the critic's primary
        decision (skip filing the duplicate) is already honoured by
        the early return, so the worst case is losing the
        replay-durability marker, not corrupting state."""
        gh = MagicMock()
        gh.search_issues.side_effect = [
            [],
            [
                {
                    "title": "x",
                    "body": "y",
                    "html_url": "https://github.com/FidoCanCode/home/issues/42",
                }
            ],
        ]
        # Marker write blows up — must be caught and logged, not raised.
        gh.comment_issue.side_effect = RuntimeError("GitHub 503")
        filer = self._make_critic_filer(
            gh,
            agent_response=(
                '{"is_duplicate": true, '
                '"duplicate_url": "https://github.com/FidoCanCode/home/issues/42", '
                '"rationale": "dup"}'
            ),
        )

        # Should NOT raise.  Filing still skipped (critic decision
        # honoured), marker write attempted-and-failed-quietly.
        filer.file_insight(self._make_insight(), self._make_target())

        gh.create_issue.assert_not_called()
        gh.comment_issue.assert_called_once()

    def test_critic_distinct_does_not_post_marker_comment(self) -> None:
        """When the critic passes the insight is filed normally — no
        marker comment on any prior issue."""
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/13"
        filer = self._make_critic_filer(
            gh, agent_response='{"is_duplicate": false, "rationale": "distinct"}'
        )

        filer.file_insight(self._make_insight(), self._make_target())

        gh.create_issue.assert_called_once()
        gh.comment_issue.assert_not_called()

    def test_parse_insight_issue_number_accepts_insight_repo_url(self) -> None:
        """The URL extractor accepts the canonical
        ``https://github.com/FidoCanCode/home/issues/{N}`` shape and
        ignores trailing anchors."""
        from fido.events import _parse_insight_issue_number

        assert (
            _parse_insight_issue_number("https://github.com/FidoCanCode/home/issues/42")
            == 42
        )
        assert (
            _parse_insight_issue_number(
                "https://github.com/FidoCanCode/home/issues/1234#issuecomment-555"
            )
            == 1234
        )

    def test_parse_insight_issue_number_accepts_case_variants(self) -> None:
        """Codex on PR #1932: GitHub owner/repo names are
        case-insensitive, so a critic URL like
        ``fidocancode/home`` (lowercase) must be treated as the
        same repo as ``FidoCanCode/home``.  Strict equality dropped
        the durable skip-marker write and let a later fail-open
        replay file the duplicate insight after all."""
        from fido.events import _parse_insight_issue_number

        assert (
            _parse_insight_issue_number("https://github.com/fidocancode/home/issues/42")
            == 42
        )
        assert (
            _parse_insight_issue_number("https://github.com/FIDOCANCODE/HOME/issues/7")
            == 7
        )

    def test_parse_insight_issue_number_rejects_other_repo_url(self) -> None:
        """Codex on PR #1932: a critic that returned a valid issue URL
        from a DIFFERENT repo (``rhencke/confusio/issues/5``)
        previously got accepted, so the replay marker landed on issue
        #5 of FidoCanCode/home — almost certainly the wrong issue.
        The parser must reject URLs targeting any repo besides
        ``_INSIGHT_REPO``."""
        from fido.events import _parse_insight_issue_number

        assert (
            _parse_insight_issue_number("https://github.com/rhencke/confusio/issues/5")
            is None
        )
        assert (
            _parse_insight_issue_number(
                "https://github.com/foo/bar/issues/1234#issuecomment-555"
            )
            is None
        )

    def test_parse_insight_issue_number_rejects_non_issue_url(self) -> None:
        """PR URLs, non-GitHub hosts, and plain garbage all return None
        so the caller skips the marker step instead of crashing."""
        from fido.events import _parse_insight_issue_number

        assert (
            _parse_insight_issue_number("https://github.com/FidoCanCode/home/pull/42")
            is None
        )
        assert _parse_insight_issue_number("not a URL") is None
        assert _parse_insight_issue_number("") is None
        # Non-github hosts are rejected even when the path looks right.
        assert (
            _parse_insight_issue_number(
                "https://example.com/FidoCanCode/home/issues/42"
            )
            is None
        )

    def test_recent_insights_query_sorts_by_created_desc(self) -> None:
        """Codex on PR #1932: GitHub issue-search defaults to
        best-match ranking, so without an explicit
        ``sort:created-desc`` the first
        :data:`_INSIGHT_RECENT_LIMIT` results may not be the most
        recent ones — feeding stale comparison examples to the
        dedup critic and raising the false-"distinct" rate.  Pin
        the qualifier so a future refactor can't accidentally drop
        it."""
        gh = MagicMock()
        gh.search_issues.side_effect = [
            [],  # marker lookup
            [],  # recent-insights lookup — content doesn't matter
        ]
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/17"
        filer = self._make_critic_filer(gh, agent_response='{"is_duplicate": false}')

        filer.file_insight(self._make_insight(), self._make_target())

        # Second call is the recent-insights search.
        recent_query = gh.search_issues.call_args_list[1].args[1]
        assert "sort:created-desc" in recent_query

    def test_recent_insights_search_failure_falls_open(self) -> None:
        """Codex on PR #1932: the recent-corpus search runs OUTSIDE
        any fail-open guard in ``run_insight_dedup_critic`` (the
        runner only wraps the agent call).  A transient
        ``gh.search_issues`` failure here used to propagate out of
        ``file_insight`` and abort the entire dispatch.  The fix
        wraps the search itself so a flaky GitHub degrades to "no
        peers" and the critic's "passes by default on empty corpus"
        rule lets the insight file normally."""
        gh = MagicMock()
        # First call (marker lookup): no existing marker.
        # Second call (recent-insights search): fails.
        gh.search_issues.side_effect = [
            [],
            RuntimeError("GitHub 503"),
        ]
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/16"
        filer = self._make_critic_filer(
            gh, agent_response='{"is_duplicate": false, "rationale": "distinct"}'
        )

        # Should NOT raise — the search failure must not abort dispatch.
        filer.file_insight(self._make_insight(), self._make_target())

        # Insight still files normally.
        gh.create_issue.assert_called_once()

    def test_recent_insights_capped(self) -> None:
        """Caller-supplied search may return more than the cap — the
        filer trims to :data:`_INSIGHT_RECENT_LIMIT` before passing to
        the critic so the prompt stays bounded."""
        from fido.events import _INSIGHT_RECENT_LIMIT

        gh = MagicMock()
        # Return way more than the cap — the filer should pass only
        # the leading slice to the critic.
        oversized = [
            {
                "title": f"Insight {i}",
                "body": "b",
                "html_url": f"https://x/issues/{i}",
            }
            for i in range(_INSIGHT_RECENT_LIMIT * 3)
        ]
        gh.search_issues.side_effect = [[], oversized]
        gh.create_issue.return_value = f"https://github.com/{_INSIGHT_REPO}/issues/11"

        captured: dict[str, object] = {}

        from dataclasses import dataclass, field

        @dataclass
        class _Agent:
            calls: list[object] = field(default_factory=list)

            def run_turn(self, *args: object, **kwargs: object) -> str:
                self.calls.append((args, kwargs))
                return '{"is_duplicate": false}'

        @dataclass
        class _Prompts:
            def insight_dedup_critic_prompt(
                self,
                proposed_insight: dict[str, str],
                recent_insights: list[dict[str, str]],
            ) -> str:
                captured["recent_count"] = len(recent_insights)
                return "p"

        filer = _GitHubInsightFiler(
            gh,
            agent=_Agent(),  # type: ignore[arg-type]
            prompts=_Prompts(),  # type: ignore[arg-type]
            critic_system_prompt="critic-sys",
        )

        filer.file_insight(self._make_insight(), self._make_target())

        assert captured["recent_count"] == _INSIGHT_RECENT_LIMIT


class TestFileStuckOnCriticBug:
    """HOL-21 / #1915: shared bug-file helper for ALL critic
    exhaustion routes (HOL-15..HOL-19).  Pulled out of
    ``_route_critic_exhausted_blocked`` so per-leaf retry budgets
    for HOL-16/HOL-17/HOL-19 can call it without duplicating the
    idempotency-marker + body templating logic."""

    def _ctx(
        self,
        *,
        emission_point: str = "task-completion",
        target_kind: str = "task",
        target_id: str = "t-9",
        attempts_preview: tuple[str, ...] = (),
    ) -> object:
        from fido.events import StuckOnCriticContext

        return StuckOnCriticContext(
            emission_point=emission_point,
            source_repo="owner/repo",
            source_pr=42,
            target_kind=target_kind,
            target_id=target_id,
            source_link="https://github.com/owner/repo/pull/42",
            gaps=("g1", "g2", "g3"),
            attempts_preview=attempts_preview,
        )

    def test_files_new_bug_with_idempotency_marker(self) -> None:
        from fido.events import file_stuck_on_critic_bug

        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/1000"

        url = file_stuck_on_critic_bug(self._ctx(), gh=gh)
        assert url == "https://github.com/FidoCanCode/home/issues/1000"
        body = gh.create_issue.call_args.args[2]
        assert "<!-- critic-exhaustion: task-completion:" in body
        assert "owner/repo#42:task:t-9 -->" in body
        for gap in ("g1", "g2", "g3"):
            assert gap in body
        title = gh.create_issue.call_args.args[1]
        assert "task-completion critic exhausted" in title
        # Default empty attempts_preview → sentinel line in body.
        assert "(no per-attempt previews recorded)" in body

    def test_reuses_existing_bug_on_marker_hit(self) -> None:
        from fido.events import file_stuck_on_critic_bug

        gh = MagicMock()
        gh.search_issues.return_value = [
            {"html_url": "https://github.com/FidoCanCode/home/issues/2000"}
        ]

        url = file_stuck_on_critic_bug(self._ctx(), gh=gh)
        assert url == "https://github.com/FidoCanCode/home/issues/2000"
        gh.create_issue.assert_not_called()

    def test_search_failure_falls_through_to_create(self) -> None:
        from fido.events import file_stuck_on_critic_bug

        gh = MagicMock()
        gh.search_issues.side_effect = RuntimeError("503")
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/3000"

        url = file_stuck_on_critic_bug(self._ctx(), gh=gh)
        assert url == "https://github.com/FidoCanCode/home/issues/3000"

    def test_create_failure_returns_none(self) -> None:
        from fido.events import file_stuck_on_critic_bug

        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.side_effect = RuntimeError("500")

        url = file_stuck_on_critic_bug(self._ctx(), gh=gh)
        assert url is None

    def test_emission_point_part_of_idempotency_key(self) -> None:
        """Two different critics exhausting on the same target file
        SEPARATE bugs because the emission_point is in the marker."""
        from fido.events import file_stuck_on_critic_bug

        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = "https://x/issues/1"

        file_stuck_on_critic_bug(self._ctx(emission_point="task-completion"), gh=gh)
        intent_body = gh.create_issue.call_args.args[2]
        gh.reset_mock()
        gh.search_issues.return_value = []

        file_stuck_on_critic_bug(self._ctx(emission_point="task-creation"), gh=gh)
        creation_body = gh.create_issue.call_args.args[2]

        assert "task-completion:" in intent_body
        assert "task-creation:" in creation_body
        assert "task-completion:" not in creation_body

    def test_attempts_preview_when_provided(self) -> None:
        """When the caller has per-attempt previews (HOL-15/18 path
        always does), they're rendered in the bug body."""
        from fido.events import file_stuck_on_critic_bug

        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = "https://x/issues/1"

        file_stuck_on_critic_bug(
            self._ctx(attempts_preview=("preview-A", "preview-B")), gh=gh
        )
        body = gh.create_issue.call_args.args[2]
        assert "preview-A" in body
        assert "preview-B" in body
        assert "(no per-attempt previews recorded)" not in body


class TestRouteCriticExhaustedBlocked:
    """HOL-20 / #1914: critic-exhaustion route — BLOCKED PR comment +
    auto-filed bug carrying the full retry trace."""

    def _make_exc(
        self,
        label: str = "intent-coverage",
        gaps: list[str] | None = None,
        attempts: list[str] | None = None,
    ) -> SynthesisCriticExhaustedError:
        return SynthesisCriticExhaustedError(
            label,
            gaps or ["missing tests", "still missing tests", "STILL missing tests"],
            attempts or ["v1 preview", "v2 preview", "v3 preview"],
        )

    def _make_target(self) -> CommentTarget:
        return CommentTarget(
            repo="owner/repo", pr=42, comment_id=999, comment_type="pulls"
        )

    def _make_executor(self) -> MagicMock:
        executor = MagicMock()
        executor.remove_eyes_reaction = MagicMock()
        return executor

    def test_happy_path_files_bug_then_posts_blocked_then_removes_eyes(
        self,
    ) -> None:
        """Codex on PR #1932: file bug FIRST so the BLOCKED comment
        can cite the actual URL; THEN post the BLOCKED comment
        (durability boundary — embed the promise marker so a crash
        after this point lets recovery dedup); THEN remove eyes
        (pure UX, best-effort).  Returns True on success so the
        dispatcher acks the promise."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.search_issues.return_value = []  # no existing bug — proceed to create
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/9"
        executor = self._make_executor()
        target = self._make_target()
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            target,
            gh=gh,
            executor=executor,
            promise_ids=["promise-uuid-xyz"],
        )
        assert posted is True

        # Bug filed FIRST.
        bug_call = gh.create_issue.call_args
        assert bug_call.args[0] == "FidoCanCode/home"
        assert "intent-coverage" in bug_call.args[1]
        assert "owner/repo" in bug_call.args[1]
        assert "999" in bug_call.args[1]
        assert bug_call.kwargs.get("labels") == ["Bug"]
        # Body carries the full retry trace.
        bug_body = bug_call.args[2]
        for gap in exc.critic_gaps:
            assert gap in bug_body
        for preview in exc.synthesis_attempts:
            assert preview in bug_body
        # BLOCKED comment posted on source PR with the bug URL.
        comment_call = gh.comment_issue.call_args
        assert comment_call.args[0] == "owner/repo"
        assert comment_call.args[1] == 42
        blocked = comment_call.args[2]
        assert "BLOCKED" in blocked
        assert "intent-coverage" in blocked
        # The cited bug URL is the actual create_issue return value.
        assert "https://github.com/FidoCanCode/home/issues/9" in blocked
        # Recovery marker embedded so a crash after post can dedup.
        assert "promise-uuid-xyz" in blocked
        # Eyes removed via executor (best-effort).
        executor.remove_eyes_reaction.assert_called_once_with(target)

    def test_bug_filing_failure_softens_wording_but_still_posts(self) -> None:
        """Codex on PR #1932: when ``create_issue`` fails, the BLOCKED
        comment must NOT claim ``Auto-filed against …`` (that would
        be a false durable-state claim on the PR).  Soften the
        wording to ``Auto-filing … FAILED`` and continue posting —
        the human notice still matters even when the bug log
        didn't land."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.search_issues.return_value = []  # no existing bug — proceed to create
        gh.create_issue.side_effect = RuntimeError("create 500")
        executor = self._make_executor()
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["p1"],
        )
        # Comment landed → True (ack the promise).
        assert posted is True
        gh.comment_issue.assert_called_once()
        body = gh.comment_issue.call_args.args[2]
        # No URL claimed; failure noted explicitly.
        assert "FAILED" in body
        assert "Auto-filed against `FidoCanCode/home` for follow-up:" not in body

    def test_comment_failure_returns_false_so_caller_skips_ack(self) -> None:
        """Codex on PR #1932: if the BLOCKED comment post fails, the
        helper returns False so the caller does NOT ack the
        direct_promise.  The unacked promise lets recovery re-fire
        on the next webhook delivery, eventually getting the user
        signal posted.  Returning True here would silently swallow
        the failure — the user would see nothing AND the promise
        would be marked handled."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.search_issues.return_value = []  # no existing bug — proceed to create
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/11"
        gh.comment_issue.side_effect = RuntimeError("comment 429")
        executor = self._make_executor()
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["p2"],
        )
        assert posted is False
        # Bug WAS filed (preceded the post); only the comment failed.
        gh.create_issue.assert_called_once()
        gh.comment_issue.assert_called_once()
        # Eyes NOT touched — short-circuit on post failure.
        executor.remove_eyes_reaction.assert_not_called()

    def test_eyes_failure_still_returns_true(self) -> None:
        """Eyes removal is pure UX cleanup AFTER the comment post.
        A failure here means the user-visible BLOCKED comment
        already landed, so the helper still returns True (caller
        acks the promise — the work is durably done)."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/12"
        executor = self._make_executor()
        executor.remove_eyes_reaction.side_effect = RuntimeError("eyes 503")
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["p3"],
        )
        assert posted is True

    def test_blocked_comment_includes_final_gap(self) -> None:
        """The BLOCKED comment quotes the final critic gap so the
        human reading the PR sees exactly what the critic complained
        about, not just "blocked"."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/13"
        executor = self._make_executor()
        exc = self._make_exc(
            gaps=["g1", "g2", "the very specific final complaint"],
        )

        _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["p4"],
        )

        body = gh.comment_issue.call_args.args[2]
        assert "the very specific final complaint" in body

    def test_empty_promise_ids_omits_marker(self) -> None:
        """When the caller passes ``promise_ids=[]`` (no claim — e.g.
        a webhook with no comment_id), the BLOCKED comment still
        lands but carries no marker.  Recovery has nothing to dedup
        against on this path, but the comment is still informative."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/14"
        executor = self._make_executor()
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=[],
        )
        assert posted is True
        body = gh.comment_issue.call_args.args[2]
        # No reply-promise marker pattern present.
        assert "fido:reply-promise:" not in body

    def test_bug_filing_idempotent_across_retries(self) -> None:
        """Codex on PR #1932: when the BLOCKED comment post fails and
        the route is re-fired by recovery, the FIRST act each retry
        does is file a bug — without idempotency, transient failures
        spam ``FidoCanCode/home`` with duplicate Bug issues that
        obscure real incidents.

        Idempotency uses a marker keyed on
        ``(critic_label, source_repo, source_pr, source_comment_id)``.
        On a retry, the search finds the existing bug and reuses
        its URL instead of creating a second one."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        # First call: marker search returns existing bug → skip create.
        gh.search_issues.return_value = [
            {"html_url": "https://github.com/FidoCanCode/home/issues/9000"}
        ]
        executor = self._make_executor()
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["p1"],
        )
        assert posted is True
        # No new bug filed — the search hit short-circuited the create.
        gh.create_issue.assert_not_called()
        # Posted BLOCKED comment cites the EXISTING bug URL.
        body = gh.comment_issue.call_args.args[2]
        assert "https://github.com/FidoCanCode/home/issues/9000" in body

    def test_bug_search_failure_falls_through_to_create(self) -> None:
        """Belt-and-braces: a search failure (rate limit, etc.) must
        NOT block the bug filing — degrade to "may create a
        duplicate" rather than skipping the bug entirely."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.search_issues.side_effect = RuntimeError("search 503")
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/9001"
        executor = self._make_executor()
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["p1"],
        )
        assert posted is True
        # Create did fire (no existing bug found because search blew up).
        gh.create_issue.assert_called_once()

    def test_bug_marker_distinguishes_critic_label(self) -> None:
        """Two different critics exhausting on the SAME comment file
        SEPARATE bugs because the marker key includes the critic
        label (different root causes warrant separate tickets)."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/9002"
        executor = self._make_executor()
        exc = self._make_exc(label="reply-prose")

        _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["p1"],
        )
        bug_body = gh.create_issue.call_args.args[2]
        # The marker carries the label so the next intent-coverage
        # exhaustion on the same comment doesn't dedup against this
        # reply-prose bug.
        assert "<!-- critic-exhaustion: reply-prose:" in bug_body
        search_query = gh.search_issues.call_args.args[1]
        assert "reply-prose:" in search_query

    def test_multiple_promise_ids_embedded_for_grouped_paths(self) -> None:
        """Codex on PR #1932: queued / recovery paths can carry
        multiple promise ids in ``context["reply_promise_ids"]`` for
        grouped comments.  ALL of them must map back to the same
        posted BLOCKED comment so ``recover_from_bodies`` can dedup
        on any redelivery (whichever promise the next webhook
        delivery surfaces, the same posted comment is found)."""
        from fido.events import _route_critic_exhausted_blocked

        gh = MagicMock()
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/15"
        executor = self._make_executor()
        exc = self._make_exc()

        posted = _route_critic_exhausted_blocked(
            exc,
            self._make_target(),
            gh=gh,
            executor=executor,
            promise_ids=["promise-A", "promise-B", "promise-C"],
        )
        assert posted is True
        body = gh.comment_issue.call_args.args[2]
        # All three markers embedded so ``recover_from_bodies`` can
        # find a match for any of the three promise ids.
        assert "promise-A" in body
        assert "promise-B" in body
        assert "promise-C" in body


class TestDispatcher:
    """Unit tests for the :class:`~fido.events.Dispatcher` collaborator."""

    def test_dispatch_routes_ping_to_none(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        d = Dispatcher(cfg, repo_cfg, MagicMock())
        result = d.dispatch("ping", {"hook_id": 1})
        assert result is None

    def test_dispatch_returns_action_for_issue_assigned(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        payload = {
            "action": "assigned",
            "assignee": {"login": "rhencke"},
            "issue": {"number": 7, "title": "Do the thing"},
        }
        d = Dispatcher(cfg, repo_cfg, MagicMock())
        result = d.dispatch("issues", payload)
        assert isinstance(result, Action)
        assert "7" in result.prompt

    def test_backfill_returns_count(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        mock_gh = MagicMock()
        mock_gh.get_issue_comments.return_value = []
        repo_cfg = _repo_cfg(tmp_path)
        d = Dispatcher(cfg, repo_cfg, mock_gh)
        count = d.backfill_missed_pr_comments(42, gh_user="fido", registry=MagicMock())
        mock_gh.get_issue_comments.assert_called_once_with(repo_cfg.name, 42)
        assert count == 0

    def test_task_creation_drop_escalator_files_bug(self, tmp_path: Path) -> None:
        # HOL-16 follow-up / #1934: when the process-local per-intent
        # drop counter crosses the threshold, the Dispatcher-bound
        # escalator files a bug via ``file_stuck_on_critic_bug`` with
        # ``emission_point="task-creation"`` and the comment id as
        # ``target_id``.
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/9998"
        d = Dispatcher(cfg, repo_cfg, gh)
        d._escalate_task_creation_drop_streak(intent_comment_id=12345, count=3)
        gh.create_issue.assert_called_once()
        args = gh.create_issue.call_args.args
        assert args[0] == "FidoCanCode/home"
        assert "task-creation" in args[2]
        assert "12345" in args[2]

    def test_insight_dedup_transport_escalator_files_bug(self, tmp_path: Path) -> None:
        # HOL-19 follow-up / #1935: when the process-local
        # transport-failure counter crosses the threshold, the
        # Dispatcher-bound escalator files a bug via
        # ``file_stuck_on_critic_bug``.
        cfg = _config(tmp_path)
        repo_cfg = _repo_cfg(tmp_path)
        gh = MagicMock()
        gh.search_issues.return_value = []
        gh.create_issue.return_value = "https://github.com/FidoCanCode/home/issues/9999"
        d = Dispatcher(cfg, repo_cfg, gh)
        d._escalate_insight_dedup_transport_failure(3)
        gh.create_issue.assert_called_once()
        args = gh.create_issue.call_args.args
        assert args[0] == "FidoCanCode/home"
        assert "insight-dedup-transport" in args[2]

    def test_launch_sync_calls_background(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path)
        mock_gh = MagicMock()
        repo_cfg = _repo_cfg(tmp_path)
        sync_calls: list[tuple[object, ...]] = []

        def fake_sync(*args: object, **kwargs: object) -> None:
            sync_calls.append(args)

        d = Dispatcher(cfg, repo_cfg, mock_gh, sync_fn=fake_sync)
        d.launch_sync()
        assert len(sync_calls) == 1
        assert sync_calls[0] == (repo_cfg.work_dir, mock_gh)


# ─── HOL-18 / #1912 — _gather_claim_grounding_state ────────────────────────


class TestGatherClaimGroundingState:
    """Helper that collects ground-truth references for the reply-prose
    critic.  Current shape: ``recent_commit_shas`` from ``git log -20``."""

    def test_real_git_repo_returns_recent_shas(self, tmp_path: Path) -> None:
        """In a real git repo with commits, every recent SHA shows up
        in ``recent_commit_shas`` — that's what gates PR #1858's
        empty-commit lies."""
        import subprocess as _subprocess

        from fido.events import _gather_claim_grounding_state

        # Bootstrap a tiny repo with 3 commits.
        for cmd in (
            ["git", "init", "-q"],
            ["git", "config", "user.email", "fido@example.com"],
            ["git", "config", "user.name", "Fido"],
            ["git", "commit", "--allow-empty", "-q", "-m", "first"],
            ["git", "commit", "--allow-empty", "-q", "-m", "second"],
            ["git", "commit", "--allow-empty", "-q", "-m", "third"],
        ):
            _subprocess.run(cmd, cwd=tmp_path, check=True)

        state = _gather_claim_grounding_state(tmp_path)
        assert "recent_commit_shas" in state
        # 3 commits, both full (40-char) AND short (7-char) forms per
        # codex on PR #1932 — replies routinely cite either form, so
        # the grounding list must contain both so the critic can match.
        shas = state["recent_commit_shas"]
        assert len(shas) == 6
        full_shas = [s for s in shas if len(s) == 40]
        short_shas = [s for s in shas if len(s) == 7]
        assert len(full_shas) == 3
        assert len(short_shas) == 3
        for sha in shas:
            assert all(c in "0123456789abcdef" for c in sha)
        # Each short SHA must be a prefix of one of the full SHAs —
        # that's the property the critic relies on when prose cites
        # the short form.
        for short in short_shas:
            assert any(full.startswith(short) for full in full_shas)

    def test_non_git_directory_returns_empty(self, tmp_path: Path) -> None:
        """A non-git work_dir must return ``{}`` so the caller (and the
        critic) treat it as "no ground truth available" and skip the
        critic — better than crashing."""
        from fido.events import _gather_claim_grounding_state

        state = _gather_claim_grounding_state(tmp_path)
        assert state == {}

    def test_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        """Even an entirely missing path fails open to empty state."""
        from fido.events import _gather_claim_grounding_state

        state = _gather_claim_grounding_state(tmp_path / "does-not-exist")
        assert state == {}

    def test_arbitrary_os_error_returns_empty(self, tmp_path: Path) -> None:
        """Rob review on PR #1932: the original narrow ``(CalledProcess
        Error, FileNotFoundError, NotADirectoryError)`` catch let other
        OSError subclasses (PermissionError, etc.) crash dispatch.  The
        broadened ``(CalledProcessError, OSError)`` catch must swallow
        every OSError subclass and degrade to empty state."""
        import subprocess as _subprocess

        from fido.events import _gather_claim_grounding_state

        original_run = _subprocess.run

        def boom(*args: object, **kwargs: object) -> object:
            raise PermissionError("simulated lock-down")

        _subprocess.run = boom  # type: ignore[assignment]
        try:
            state = _gather_claim_grounding_state(tmp_path)
        finally:
            _subprocess.run = original_run
        assert state == {}
