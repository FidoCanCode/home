from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from kennel.github import (
    _gh,
    add_pr_reviewer,
    add_reaction,
    close_issue,
    comment_issue,
    create_pr,
    edit_pr_body,
    find_issues,
    find_pr,
    get_default_branch,
    get_issue_comments,
    get_pr,
    get_repo_info,
    get_review_comments,
    get_review_threads,
    get_reviews,
    get_run_log,
    get_user,
    pr_checks,
    pr_merge,
    pr_ready,
    reply_to_review_comment,
    resolve_thread,
    set_user_status,
    view_issue,
)


def _completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
    )


class TestGhHelper:
    def test_calls_subprocess_run(self) -> None:
        with patch("subprocess.run", return_value=_completed("out")) as mock:
            result = _gh("api", "user", cwd="/tmp", timeout=5)
        mock.assert_called_once_with(
            ["gh", "api", "user"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd="/tmp",
        )
        assert result.stdout == "out"

    def test_defaults(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            _gh("version")
        _, kwargs = mock.call_args
        assert kwargs["timeout"] == 30
        assert kwargs["cwd"] is None


class TestGetRepoInfo:
    def test_returns_stripped_name(self) -> None:
        with patch("subprocess.run", return_value=_completed("owner/repo\n")):
            assert get_repo_info() == "owner/repo"

    def test_passes_cwd(self) -> None:
        with patch("subprocess.run", return_value=_completed("o/r")) as mock:
            get_repo_info(cwd="/some/path")
        assert mock.call_args.kwargs["cwd"] == "/some/path"


class TestGetUser:
    def test_returns_login(self) -> None:
        with patch("subprocess.run", return_value=_completed("fido\n")):
            assert get_user() == "fido"


class TestGetDefaultBranch:
    def test_returns_branch(self) -> None:
        with patch("subprocess.run", return_value=_completed("main\n")):
            assert get_default_branch() == "main"

    def test_passes_cwd(self) -> None:
        with patch("subprocess.run", return_value=_completed("main")) as mock:
            get_default_branch(cwd=Path("/repo"))
        assert mock.call_args.kwargs["cwd"] == Path("/repo")


class TestSetUserStatus:
    def test_busy_true(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            set_user_status("coding", "🐶", busy=True)
        cmd = mock.call_args.args[0]
        assert "busy=true" in cmd

    def test_busy_false(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            set_user_status("napping", "💤", busy=False)
        cmd = mock.call_args.args[0]
        assert "busy=false" in cmd

    def test_passes_msg_and_emoji(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            set_user_status("working", "🚀")
        cmd = mock.call_args.args[0]
        assert "msg=working" in cmd
        assert "emoji=🚀" in cmd


class TestFindIssues:
    def test_returns_nodes(self) -> None:
        nodes = [{"number": 1, "title": "Fix it", "subIssues": {"nodes": []}}]
        payload = {"data": {"repository": {"issues": {"nodes": nodes}}}}
        with patch("subprocess.run", return_value=_completed(json.dumps(payload))):
            result = find_issues("owner", "repo", "fido")
        assert result == nodes

    def test_passes_variables(self) -> None:
        payload = {"data": {"repository": {"issues": {"nodes": []}}}}
        with patch(
            "subprocess.run", return_value=_completed(json.dumps(payload))
        ) as mock:
            find_issues("myowner", "myrepo", "mylogin")
        cmd = mock.call_args.args[0]
        assert "-F" in cmd
        assert "owner=myowner" in cmd
        assert "repo=myrepo" in cmd
        assert "login=mylogin" in cmd


class TestViewIssue:
    def test_returns_parsed_json(self) -> None:
        issue = {"state": "OPEN", "title": "Bug", "body": "desc"}
        with patch("subprocess.run", return_value=_completed(json.dumps(issue))):
            assert view_issue("o/r", 5) == issue

    def test_converts_number_to_str(self) -> None:
        issue = {"state": "OPEN", "title": "T", "body": ""}
        with patch(
            "subprocess.run", return_value=_completed(json.dumps(issue))
        ) as mock:
            view_issue("o/r", 42)
        cmd = mock.call_args.args[0]
        assert "42" in cmd


class TestCloseIssue:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            close_issue("o/r", 3)
        cmd = mock.call_args.args[0]
        assert cmd == ["gh", "issue", "close", "3", "--repo", "o/r"]


class TestCommentIssue:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            comment_issue("o/r", 7, "hello")
        cmd = mock.call_args.args[0]
        assert cmd == [
            "gh",
            "issue",
            "comment",
            "7",
            "--repo",
            "o/r",
            "--body",
            "hello",
        ]


class TestGetIssueComments:
    def test_returns_list(self) -> None:
        comments = [{"id": 1, "body": "hi"}]
        with patch("subprocess.run", return_value=_completed(json.dumps(comments))):
            assert get_issue_comments("o/r", 1) == comments

    def test_uses_api_endpoint(self) -> None:
        with patch("subprocess.run", return_value=_completed("[]")) as mock:
            get_issue_comments("o/r", 9)
        cmd = mock.call_args.args[0]
        assert "repos/o/r/issues/9/comments" in cmd


class TestFindPr:
    def test_returns_matching_pr(self) -> None:
        prs = [
            {
                "number": 1,
                "headRefName": "feat",
                "state": "OPEN",
                "author": {"login": "fido"},
            },
            {
                "number": 2,
                "headRefName": "other",
                "state": "OPEN",
                "author": {"login": "other"},
            },
        ]
        with patch("subprocess.run", return_value=_completed(json.dumps(prs))):
            result = find_pr("o/r", 5, "fido")
        assert result == prs[0]

    def test_returns_none_if_not_found(self) -> None:
        prs = [
            {
                "number": 1,
                "headRefName": "feat",
                "state": "OPEN",
                "author": {"login": "other"},
            }
        ]
        with patch("subprocess.run", return_value=_completed(json.dumps(prs))):
            assert find_pr("o/r", 5, "fido") is None

    def test_returns_none_on_empty(self) -> None:
        with patch("subprocess.run", return_value=_completed("[]")):
            assert find_pr("o/r", 1, "fido") is None


class TestCreatePr:
    def test_returns_url(self) -> None:
        with patch(
            "subprocess.run",
            return_value=_completed("https://github.com/o/r/pull/10\n"),
        ):
            assert (
                create_pr("o/r", "title", "body", "main", "feat")
                == "https://github.com/o/r/pull/10"
            )

    def test_passes_args(self) -> None:
        with patch("subprocess.run", return_value=_completed("url")) as mock:
            create_pr("o/r", "T", "B", "main", "branch")
        cmd = mock.call_args.args[0]
        assert "--draft" in cmd
        assert "--title" in cmd
        assert "T" in cmd
        assert "--base" in cmd
        assert "main" in cmd
        assert "--head" in cmd
        assert "branch" in cmd


class TestEditPrBody:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            edit_pr_body("o/r", 10, "new body")
        cmd = mock.call_args.args[0]
        assert cmd == ["gh", "pr", "edit", "10", "--repo", "o/r", "--body", "new body"]


class TestAddPrReviewer:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            add_pr_reviewer("o/r", 10, "rhencke")
        cmd = mock.call_args.args[0]
        assert cmd == [
            "gh",
            "pr",
            "edit",
            "10",
            "--repo",
            "o/r",
            "--add-reviewer",
            "rhencke",
        ]


class TestPrChecks:
    def test_returns_list(self) -> None:
        checks = [{"name": "ci", "state": "SUCCESS", "link": "http://..."}]
        with patch("subprocess.run", return_value=_completed(json.dumps(checks))):
            assert pr_checks("o/r", 10) == checks


class TestPrReady:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_ready("o/r", 10)
        cmd = mock.call_args.args[0]
        assert cmd == ["gh", "pr", "ready", "10", "--repo", "o/r"]


class TestPrMerge:
    def test_squash_default(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_merge("o/r", 10)
        cmd = mock.call_args.args[0]
        assert "--squash" in cmd
        assert "--auto" not in cmd

    def test_auto_flag(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_merge("o/r", 10, auto=True)
        cmd = mock.call_args.args[0]
        assert "--auto" in cmd

    def test_no_squash(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            pr_merge("o/r", 10, squash=False)
        cmd = mock.call_args.args[0]
        assert "--squash" not in cmd


class TestGetPr:
    def test_returns_dict(self) -> None:
        data = {
            "reviews": [],
            "isDraft": True,
            "mergeStateStatus": "CLEAN",
            "body": "",
            "commits": [],
        }
        with patch("subprocess.run", return_value=_completed(json.dumps(data))):
            assert get_pr("o/r", 10) == data

    def test_requests_all_fields(self) -> None:
        data = {
            "reviews": [],
            "isDraft": False,
            "mergeStateStatus": "CLEAN",
            "body": "",
            "commits": [],
        }
        with patch("subprocess.run", return_value=_completed(json.dumps(data))) as mock:
            get_pr("o/r", 10)
        cmd = mock.call_args.args[0]
        assert "reviews,isDraft,mergeStateStatus,body,commits" in cmd


class TestGetReviews:
    def test_returns_dict(self) -> None:
        data = {"reviews": [], "isDraft": False}
        with patch("subprocess.run", return_value=_completed(json.dumps(data))):
            assert get_reviews("o/r", 10) == data


class TestGetReviewComments:
    def test_returns_ids(self) -> None:
        with patch("subprocess.run", return_value=_completed("101\n102\n103\n")):
            assert get_review_comments("o/r", 10, 99) == [101, 102, 103]

    def test_empty_output(self) -> None:
        with patch("subprocess.run", return_value=_completed("")):
            assert get_review_comments("o/r", 10, 99) == []

    def test_uses_correct_endpoint(self) -> None:
        with patch("subprocess.run", return_value=_completed("1")) as mock:
            get_review_comments("o/r", 10, 99)
        cmd = mock.call_args.args[0]
        assert "repos/o/r/pulls/10/reviews/99/comments" in cmd


class TestReplyToReviewComment:
    def test_calls_gh(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            reply_to_review_comment("o/r", 10, "lgtm", 55)
        cmd = mock.call_args.args[0]
        assert "repos/o/r/pulls/10/comments" in cmd
        assert "-X" in cmd
        assert "POST" in cmd
        assert "body=lgtm" in cmd
        assert "in_reply_to=55" in cmd


class TestAddReaction:
    def test_calls_gh_pulls(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            add_reaction("o/r", "pulls", 42, "rocket")
        cmd = mock.call_args.args[0]
        assert "repos/o/r/pulls/comments/42/reactions" in cmd
        assert "content=rocket" in cmd

    def test_calls_gh_issues(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            add_reaction("o/r", "issues", 7, "+1")
        cmd = mock.call_args.args[0]
        assert "repos/o/r/issues/comments/7/reactions" in cmd


class TestGetReviewThreads:
    def test_returns_parsed_json(self) -> None:
        data = {
            "data": {"repository": {"pullRequest": {"reviewThreads": {"nodes": []}}}}
        }
        with patch("subprocess.run", return_value=_completed(json.dumps(data))):
            assert get_review_threads("owner", "repo", 10) == data

    def test_passes_variables(self) -> None:
        data = {"data": {}}
        with patch("subprocess.run", return_value=_completed(json.dumps(data))) as mock:
            get_review_threads("myowner", "myrepo", 10)
        cmd = mock.call_args.args[0]
        assert "owner=myowner" in cmd
        assert "repo=myrepo" in cmd
        assert "pr=10" in cmd


class TestResolveThread:
    def test_calls_graphql(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            resolve_thread("T_kwDOABC123")
        cmd = mock.call_args.args[0]
        assert "graphql" in cmd
        assert "id=T_kwDOABC123" in cmd
        assert any("resolveReviewThread" in a for a in cmd)


class TestGetRunLog:
    def test_returns_stdout(self) -> None:
        with patch("subprocess.run", return_value=_completed("log output\n")):
            assert get_run_log(12345) == "log output\n"

    def test_uses_timeout_60(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            get_run_log("99")
        assert mock.call_args.kwargs["timeout"] == 60

    def test_converts_run_id(self) -> None:
        with patch("subprocess.run", return_value=_completed()) as mock:
            get_run_log(42)
        cmd = mock.call_args.args[0]
        assert "42" in cmd
