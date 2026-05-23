"""Mechanical claim-grounding validators for reply prose (HOL-4 / #1898).

When Fido's reply prose mentions a commit SHA, an issue/PR number, or a
GitHub URL, the referenced entity must actually exist.  PR #1858's
recurring "the work is in a recent commit" lies — the prose claimed a
SHA that wasn't in git — fall under this shape.

This module is the **mechanical** validator: regex-extracts the claims
out of prose, calls a cheap existence check on each, and returns a list
of errors.  It does NOT decide what to do with the errors — the caller
(synthesis flow, reply-emission path) decides whether to fail closed,
retry, or surface the errors back to the LLM.

The richer LLM critic (HOL-18 / #1912) layers on top: it catches
semantic claim drift the regex can't see (e.g. "I refactored module X"
when the diff didn't touch X).  Together they form Layer 1 + Layer 2
of #1894's critic-gated architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# ── claim shapes ────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CommitClaim:
    """One ``commit <sha>`` reference extracted from reply prose.

    ``sha`` is the literal token matched (7-40 hex chars).  ``offset``
    is the byte offset of the SHA within the prose, for error
    reporting.
    """

    sha: str
    offset: int


@dataclass(frozen=True, slots=True)
class IssueOrPrNumberClaim:
    """One ``#<number>`` reference extracted from reply prose.

    Issues and PRs share GitHub's numeric namespace per repo, so the
    existence check uses the issues endpoint (which returns PRs too).
    """

    number: int
    offset: int


@dataclass(frozen=True, slots=True)
class GitHubUrlClaim:
    """One full ``https://github.com/...`` reference extracted from prose.

    Covers ``commit/<sha>``, ``issues/<n>``, and ``pull/<n>`` URLs.
    The ``kind`` field tells the validator which existence check to
    run.
    """

    repo: str  # "owner/name"
    kind: str  # "commit" | "issue" | "pr"
    ref: str  # SHA for commit, str(number) for issue/pr
    offset: int


@dataclass(frozen=True, slots=True)
class ClaimError:
    """One validation failure: the prose claim doesn't resolve.

    ``message`` is human-readable (suitable for nudging the LLM or for
    surfacing to a reviewer); ``offset`` is the offset of the failing
    claim within the original prose, for highlighting.
    """

    message: str
    offset: int


# ── existence-checker protocol ──────────────────────────────────────────────


class ClaimChecker(Protocol):
    """Existence-check collaborator for claim validators.

    Production wires this to a thin :class:`~fido.github.GitHub`-backed
    adapter; tests pass hand-rolled fakes per the project's
    no-MagicMock rule.  Each method returns ``True`` iff the referenced
    entity exists in the given repo.
    """

    def commit_exists(self, repo: str, sha: str) -> bool: ...

    def issue_or_pr_exists(self, repo: str, number: int) -> bool: ...


# ── regex extractors ────────────────────────────────────────────────────────

# ``commit a1b2c3d`` / ``commit a1b2c3d4e5f6...`` — 7 to 40 hex chars
# after the word ``commit`` (case-insensitive), bounded by word break.
# Bare backticks are tolerated around the SHA (the markdown form Fido
# writes most of the time): ``commit `a1b2c3d` `` matches the SHA.
_COMMIT_SHA_PATTERN = re.compile(
    r"\bcommit\s+`?([0-9a-f]{7,40})`?\b",
    re.IGNORECASE,
)

# ``#123`` — numeric issue / PR reference.  Must be preceded by either
# the start of the string or whitespace / punctuation so we don't match
# ``foo#123`` (a colour code, anchor fragment, etc.).
_ISSUE_PR_NUMBER_PATTERN = re.compile(
    r"(?:^|(?<=[\s(\[]))#([1-9][0-9]*)\b",
)

# ``https://github.com/owner/repo/commit/<sha>`` etc.  ``kind`` group
# distinguishes commit vs issues vs pull.  Trailing slash + fragment +
# query are tolerated; the captured ref is the bare SHA / number.
_GITHUB_URL_PATTERN = re.compile(
    r"https://github\.com/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)/"
    r"(commit|issues|pull)/"
    r"([0-9a-f]{7,40}|[1-9][0-9]*)\b",
    re.IGNORECASE,
)


def extract_commit_claims(text: str) -> list[CommitClaim]:
    """Return every ``commit <sha>`` claim found in *text*."""
    return [
        CommitClaim(sha=m.group(1).lower(), offset=m.start(1))
        for m in _COMMIT_SHA_PATTERN.finditer(text)
    ]


def extract_issue_or_pr_number_claims(text: str) -> list[IssueOrPrNumberClaim]:
    """Return every ``#<number>`` claim found in *text*.

    Numbers that are obviously not issue / PR refs — addresses inside
    URLs like ``#discussion_r123`` or ``#issuecomment-456`` — are
    naturally excluded by the leading-word-break requirement (the URL
    fragment introducer ``#`` isn't preceded by whitespace / brackets).
    """
    return [
        IssueOrPrNumberClaim(number=int(m.group(1)), offset=m.start(1))
        for m in _ISSUE_PR_NUMBER_PATTERN.finditer(text)
    ]


def extract_github_url_claims(text: str) -> list[GitHubUrlClaim]:
    """Return every ``https://github.com/...`` claim found in *text*."""
    out: list[GitHubUrlClaim] = []
    for m in _GITHUB_URL_PATTERN.finditer(text):
        kind_token = m.group(2).lower()
        kind = (
            "pr"
            if kind_token == "pull"
            else ("commit" if kind_token == "commit" else "issue")
        )
        ref = m.group(3).lower() if kind == "commit" else m.group(3)
        out.append(
            GitHubUrlClaim(repo=m.group(1), kind=kind, ref=ref, offset=m.start())
        )
    return out


# ── validation ──────────────────────────────────────────────────────────────


def validate_reply_prose_claims(
    text: str,
    *,
    repo: str,
    checker: ClaimChecker,
) -> list[ClaimError]:
    """Validate every claim in *text* against *checker*.

    Walks the three claim families:

    1. ``commit <sha>`` references (interpreted against *repo*).
    2. ``#<number>`` references (interpreted against *repo*).
    3. Full ``https://github.com/...`` URLs (interpreted against the
       URL's own ``owner/name``; *repo* is ignored for these).

    Returns the list of failures.  Empty list means every claim
    resolved — the prose is grounded.  Callers decide whether to fail
    closed (block posting) or retry (nudge the LLM with the errors).
    """
    errors: list[ClaimError] = []

    for commit in extract_commit_claims(text):
        if not checker.commit_exists(repo, commit.sha):
            errors.append(
                ClaimError(
                    message=(
                        f"reply prose claims commit {commit.sha} in {repo}, "
                        "but the commit does not exist"
                    ),
                    offset=commit.offset,
                )
            )

    for ref in extract_issue_or_pr_number_claims(text):
        if not checker.issue_or_pr_exists(repo, ref.number):
            errors.append(
                ClaimError(
                    message=(
                        f"reply prose claims issue/PR #{ref.number} in {repo}, "
                        "but the issue/PR does not exist"
                    ),
                    offset=ref.offset,
                )
            )

    for url in extract_github_url_claims(text):
        exists = (
            checker.commit_exists(url.repo, url.ref)
            if url.kind == "commit"
            else checker.issue_or_pr_exists(url.repo, int(url.ref))
        )
        if not exists:
            errors.append(
                ClaimError(
                    message=(
                        f"reply prose claims {url.kind} {url.ref} in {url.repo}, "
                        "but the entity does not exist"
                    ),
                    offset=url.offset,
                )
            )

    return errors
