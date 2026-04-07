"""GitHub CLI wrappers — all gh subprocess calls in one place."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _gh(
    *args: str,
    cwd: Path | str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


# ── Repo / user context ───────────────────────────────────────────────────────


def get_repo_info(cwd: Path | str | None = None) -> str:
    """Return 'owner/repo' for the repo at cwd."""
    result = _gh(
        "repo",
        "view",
        "--json",
        "nameWithOwner",
        "--jq",
        ".nameWithOwner",
        cwd=cwd,
    )
    return result.stdout.strip()


def get_user() -> str:
    """Return the authenticated GitHub username."""
    result = _gh("api", "user", "--jq", ".login")
    return result.stdout.strip()


def get_default_branch(cwd: Path | str | None = None) -> str:
    """Return the default branch name for the repo at cwd."""
    result = _gh(
        "repo",
        "view",
        "--json",
        "defaultBranchRef",
        "--jq",
        ".defaultBranchRef.name",
        cwd=cwd,
    )
    return result.stdout.strip()


def set_user_status(msg: str, emoji: str, busy: bool = True) -> None:
    """Set the authenticated user's GitHub status."""
    query = (
        "mutation($msg:String!,$emoji:String!,$busy:Boolean!)"
        "{ changeUserStatus(input: {message: $msg, emoji: $emoji,"
        " limitedAvailability: $busy})"
        "{ status { message emoji indicatesLimitedAvailability } } }"
    )
    _gh(
        "api",
        "graphql",
        "-F",
        f"msg={msg}",
        "-F",
        f"emoji={emoji}",
        "-F",
        f"busy={'true' if busy else 'false'}",
        "-f",
        f"query={query}",
    )


# ── Issues ────────────────────────────────────────────────────────────────────


def find_issues(owner: str, repo: str, login: str) -> list[dict[str, Any]]:
    """Return open issues assigned to login (oldest first) with sub-issue states."""
    query = (
        "query($owner:String!,$repo:String!,$login:String!){"
        "repository(owner:$owner,name:$repo){"
        "issues(first:50,states:[OPEN],filterBy:{assignee:$login},"
        "orderBy:{field:CREATED_AT,direction:ASC}){"
        "nodes{number title subIssues(first:10){nodes{state}}}}}}"
    )
    result = _gh(
        "api",
        "graphql",
        "-F",
        f"owner={owner}",
        "-F",
        f"repo={repo}",
        "-F",
        f"login={login}",
        "-f",
        f"query={query}",
    )
    data = json.loads(result.stdout)
    return data["data"]["repository"]["issues"]["nodes"]


def view_issue(repo: str, number: int | str) -> dict[str, Any]:
    """Return issue data (state, title, body)."""
    result = _gh(
        "issue",
        "view",
        str(number),
        "--repo",
        repo,
        "--json",
        "state,title,body",
    )
    return json.loads(result.stdout)


def close_issue(repo: str, number: int | str) -> None:
    """Close an issue."""
    _gh("issue", "close", str(number), "--repo", repo)


def comment_issue(repo: str, number: int | str, body: str) -> None:
    """Post a comment on an issue."""
    _gh("issue", "comment", str(number), "--repo", repo, "--body", body)


def get_issue_comments(repo: str, number: int | str) -> list[dict[str, Any]]:
    """Return all comments on an issue."""
    result = _gh("api", f"repos/{repo}/issues/{number}/comments")
    return json.loads(result.stdout)


# ── Pull requests ─────────────────────────────────────────────────────────────


def find_pr(repo: str, issue_number: int | str, user: str) -> dict[str, Any] | None:
    """Find the most recent PR linked to issue_number authored by user, or None."""
    result = _gh(
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "all",
        "--json",
        "number,headRefName,state,author",
        "--search",
        f"#{issue_number} in:body",
    )
    prs = json.loads(result.stdout)
    user_prs = [p for p in prs if p.get("author", {}).get("login") == user]
    return user_prs[0] if user_prs else None


def create_pr(
    repo: str,
    title: str,
    body: str,
    base: str,
    head: str,
) -> str:
    """Create a draft PR and return its URL."""
    result = _gh(
        "pr",
        "create",
        "--draft",
        "--title",
        title,
        "--body",
        body,
        "--base",
        base,
        "--head",
        head,
        "--repo",
        repo,
    )
    return result.stdout.strip()


def edit_pr_body(repo: str, pr: int | str, body: str) -> None:
    """Edit a PR's body."""
    _gh("pr", "edit", str(pr), "--repo", repo, "--body", body)


def add_pr_reviewer(repo: str, pr: int | str, reviewer: str) -> None:
    """Add a reviewer to a PR."""
    _gh("pr", "edit", str(pr), "--repo", repo, "--add-reviewer", reviewer)


def pr_checks(repo: str, pr: int | str) -> list[dict[str, Any]]:
    """Return check statuses for a PR."""
    result = _gh(
        "pr",
        "checks",
        str(pr),
        "--repo",
        repo,
        "--json",
        "name,state,link",
    )
    return json.loads(result.stdout)


def pr_ready(repo: str, pr: int | str) -> None:
    """Mark a PR ready for review."""
    _gh("pr", "ready", str(pr), "--repo", repo)


def pr_merge(repo: str, pr: int | str, squash: bool = True, auto: bool = False) -> None:
    """Merge a PR."""
    args = ["pr", "merge", str(pr), "--repo", repo]
    if squash:
        args.append("--squash")
    if auto:
        args.append("--auto")
    _gh(*args)


def get_pr(repo: str, pr: int | str) -> dict[str, Any]:
    """Return PR data (reviews, isDraft, mergeStateStatus, body, commits)."""
    result = _gh(
        "pr",
        "view",
        str(pr),
        "--repo",
        repo,
        "--json",
        "reviews,isDraft,mergeStateStatus,body,commits",
    )
    return json.loads(result.stdout)


def get_reviews(repo: str, pr: int | str) -> dict[str, Any]:
    """Return reviews and isDraft for a PR."""
    result = _gh(
        "pr",
        "view",
        str(pr),
        "--repo",
        repo,
        "--json",
        "reviews,isDraft",
    )
    return json.loads(result.stdout)


def get_review_comments(repo: str, pr: int | str, review_id: int | str) -> list[int]:
    """Return list of comment IDs from a review."""
    result = _gh(
        "api",
        f"repos/{repo}/pulls/{pr}/reviews/{review_id}/comments",
        "--jq",
        ".[].id",
    )
    return [
        int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip()
    ]


def reply_to_review_comment(
    repo: str,
    pr: int | str,
    body: str,
    in_reply_to: int | str,
) -> None:
    """Post a reply to an inline review comment."""
    _gh(
        "api",
        f"repos/{repo}/pulls/{pr}/comments",
        "-X",
        "POST",
        "-f",
        f"body={body}",
        "-F",
        f"in_reply_to={in_reply_to}",
    )


def add_reaction(
    repo: str,
    comment_type: str,
    comment_id: int | str,
    content: str,
) -> None:
    """Add a reaction to a comment. comment_type: 'pulls' or 'issues'."""
    endpoint = f"repos/{repo}/{comment_type}/comments/{comment_id}/reactions"
    _gh("api", endpoint, "-X", "POST", "-f", f"content={content}")


# ── Review threads ────────────────────────────────────────────────────────────


def get_review_threads(owner: str, repo: str, pr: int | str) -> dict[str, Any]:
    """Return full review-thread data for a PR (GraphQL)."""
    query = (
        "query($owner:String!,$repo:String!,$pr:Int!){"
        "repository(owner:$owner,name:$repo){"
        "pullRequest(number:$pr){"
        "reviewThreads(first:100){"
        "nodes{id isResolved"
        " comments(first:50){nodes{author{login} body url createdAt databaseId}}"
        "}}}}}"
    )
    result = _gh(
        "api",
        "graphql",
        "-F",
        f"owner={owner}",
        "-F",
        f"repo={repo}",
        "-F",
        f"pr={pr}",
        "-f",
        f"query={query}",
    )
    return json.loads(result.stdout)


def resolve_thread(thread_id: str) -> None:
    """Resolve a review thread via GraphQL mutation."""
    query = (
        "mutation($id:ID!)"
        "{resolveReviewThread(input:{threadId:$id}){thread{isResolved}}}"
    )
    _gh(
        "api",
        "graphql",
        "-F",
        f"id={thread_id}",
        "-f",
        f"query={query}",
    )


# ── CI runs ───────────────────────────────────────────────────────────────────


def get_run_log(run_id: str | int) -> str:
    """Return the failed log output for a CI run."""
    result = _gh("run", "view", str(run_id), "--log-failed", timeout=60)
    return result.stdout
