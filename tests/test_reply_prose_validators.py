"""Tests for the mechanical reply-prose claim validators (HOL-4 / #1898)."""

from dataclasses import dataclass, field

from fido.reply_prose_validators import (
    ClaimError,
    CommitClaim,
    GitHubUrlClaim,
    IssueOrPrNumberClaim,
    extract_commit_claims,
    extract_github_url_claims,
    extract_issue_or_pr_number_claims,
    validate_reply_prose_claims,
)

# ── hand-rolled ClaimChecker fakes (no MagicMock per project memory) ──────────
#
# ``ClaimChecker`` in production is a :class:`typing.Protocol`; this fake
# implements the same method signatures via duck typing rather than
# subclassing (Protocols don't make sense as runtime base classes).


@dataclass
class _StaticChecker:
    """Checker that returns True for the configured (repo, ref) pairs.

    Two sets: ``commits`` keyed by ``(repo, sha)`` and ``issues_prs``
    keyed by ``(repo, number)``.  Any lookup not in either set returns
    False.
    """

    commits: set[tuple[str, str]] = field(default_factory=set)
    issues_prs: set[tuple[str, int]] = field(default_factory=set)
    calls: list[tuple[str, str, str | int]] = field(default_factory=list)

    def commit_exists(self, repo: str, sha: str) -> bool:
        self.calls.append(("commit", repo, sha))
        return (repo, sha) in self.commits

    def issue_or_pr_exists(self, repo: str, number: int) -> bool:
        self.calls.append(("issue_or_pr", repo, number))
        return (repo, number) in self.issues_prs


# ── extract_commit_claims ─────────────────────────────────────────────────────


class TestExtractCommitClaims:
    def test_empty_text_returns_empty(self) -> None:
        assert extract_commit_claims("") == []

    def test_no_commit_word_returns_empty(self) -> None:
        # "abc1234" alone is not a claim — must follow the word ``commit``.
        assert extract_commit_claims("see abc1234") == []

    def test_short_sha_seven_chars(self) -> None:
        result = extract_commit_claims("Done in commit abc1234.")
        assert result == [CommitClaim(sha="abc1234", offset=len("Done in commit "))]

    def test_long_sha_forty_chars(self) -> None:
        sha = "0" * 40
        result = extract_commit_claims(f"see commit {sha} for details")
        assert result == [CommitClaim(sha=sha, offset=len("see commit "))]

    def test_sha_in_backticks_extracted(self) -> None:
        # Fido's voice usually wraps refs in backticks.
        result = extract_commit_claims("Landed in commit `a1b2c3d`.")
        assert result == [CommitClaim(sha="a1b2c3d", offset=len("Landed in commit `"))]

    def test_multiple_commit_claims(self) -> None:
        text = "First commit abc1234, then commit def5678."
        result = extract_commit_claims(text)
        assert [c.sha for c in result] == ["abc1234", "def5678"]

    def test_case_insensitive_commit_word(self) -> None:
        result = extract_commit_claims("See Commit abc1234.")
        assert [c.sha for c in result] == ["abc1234"]

    def test_sha_lowercased(self) -> None:
        # SHAs in the wild can be upper or mixed case; normalise.
        result = extract_commit_claims("see commit ABC1234")
        assert [c.sha for c in result] == ["abc1234"]

    def test_too_short_sha_not_matched(self) -> None:
        # < 7 chars — not a valid SHA.
        assert extract_commit_claims("commit abc123") == []

    def test_too_long_sha_not_matched(self) -> None:
        # > 40 chars — not a valid SHA.
        assert extract_commit_claims("commit " + "0" * 41) == []


# ── extract_issue_or_pr_number_claims ─────────────────────────────────────────


class TestExtractIssueOrPrNumberClaims:
    def test_empty_text_returns_empty(self) -> None:
        assert extract_issue_or_pr_number_claims("") == []

    def test_basic_number_claim(self) -> None:
        result = extract_issue_or_pr_number_claims("see #1858 for context")
        assert result == [IssueOrPrNumberClaim(number=1858, offset=len("see #"))]

    def test_multiple_number_claims(self) -> None:
        result = extract_issue_or_pr_number_claims("filed #1894 closes #1855")
        assert [c.number for c in result] == [1894, 1855]

    def test_number_at_start_of_string(self) -> None:
        # No preceding whitespace, but start-of-string is a valid boundary.
        result = extract_issue_or_pr_number_claims("#1 done")
        assert [c.number for c in result] == [1]

    def test_number_after_open_paren(self) -> None:
        result = extract_issue_or_pr_number_claims("queue (#1234) is open")
        assert [c.number for c in result] == [1234]

    def test_number_after_open_bracket(self) -> None:
        result = extract_issue_or_pr_number_claims("[#42]")
        assert [c.number for c in result] == [42]

    def test_url_fragment_not_matched(self) -> None:
        # ``foo#discussion_r123`` is a URL fragment, not an issue ref.
        # The leading-whitespace requirement excludes it.
        result = extract_issue_or_pr_number_claims("see http://x/y#discussion_r123")
        assert result == []

    def test_anchored_to_identifier_not_matched(self) -> None:
        # ``foo#123`` glued to an identifier — not what we mean.
        result = extract_issue_or_pr_number_claims("color foo#abc")
        assert result == []

    def test_zero_prefixed_not_matched(self) -> None:
        # ``#01234`` is not a real issue id.
        result = extract_issue_or_pr_number_claims("see #01234")
        assert result == []

    def test_zero_alone_not_matched(self) -> None:
        # ``#0`` is not a valid GitHub issue id.
        result = extract_issue_or_pr_number_claims("see #0")
        assert result == []


# ── extract_github_url_claims ─────────────────────────────────────────────────


class TestExtractGitHubUrlClaims:
    def test_empty_text_returns_empty(self) -> None:
        assert extract_github_url_claims("") == []

    def test_commit_url_extracted(self) -> None:
        text = "see https://github.com/FidoCanCode/home/commit/abc1234 for the fix"
        result = extract_github_url_claims(text)
        assert result == [
            GitHubUrlClaim(
                repo="FidoCanCode/home", kind="commit", ref="abc1234", offset=4
            )
        ]

    def test_issue_url_extracted(self) -> None:
        text = "filed https://github.com/FidoCanCode/home/issues/1894"
        result = extract_github_url_claims(text)
        assert result == [
            GitHubUrlClaim(repo="FidoCanCode/home", kind="issue", ref="1894", offset=6)
        ]

    def test_pull_url_normalized_to_pr(self) -> None:
        # GitHub URLs use ``/pull/`` but we normalise to ``"pr"`` kind.
        text = "see https://github.com/FidoCanCode/home/pull/1929"
        result = extract_github_url_claims(text)
        assert [(c.kind, c.ref) for c in result] == [("pr", "1929")]

    def test_multiple_urls_in_one_text(self) -> None:
        text = (
            "compare https://github.com/FidoCanCode/home/commit/abc1234 "
            "with https://github.com/FidoCanCode/home/pull/1929"
        )
        result = extract_github_url_claims(text)
        assert [(c.kind, c.ref) for c in result] == [
            ("commit", "abc1234"),
            ("pr", "1929"),
        ]

    def test_owner_repo_with_dots_and_dashes(self) -> None:
        text = "https://github.com/some-owner/repo.name/issues/1"
        result = extract_github_url_claims(text)
        assert result == [
            GitHubUrlClaim(repo="some-owner/repo.name", kind="issue", ref="1", offset=0)
        ]

    def test_commit_sha_lowercased(self) -> None:
        text = "https://github.com/o/r/commit/ABC1234"
        result = extract_github_url_claims(text)
        assert [c.ref for c in result] == ["abc1234"]


# ── validate_reply_prose_claims (integration) ──────────────────────────────────


class TestValidateReplyProseClaims:
    def test_empty_text_no_errors(self) -> None:
        checker = _StaticChecker()
        assert validate_reply_prose_claims("", repo="o/r", checker=checker) == []
        # No work done either.
        assert checker.calls == []

    def test_existing_commit_no_error(self) -> None:
        checker = _StaticChecker(commits={("o/r", "abc1234")})
        result = validate_reply_prose_claims(
            "Done in commit abc1234.", repo="o/r", checker=checker
        )
        assert result == []
        assert checker.calls == [("commit", "o/r", "abc1234")]

    def test_missing_commit_yields_error(self) -> None:
        checker = _StaticChecker()  # no commits configured
        result = validate_reply_prose_claims(
            "Done in commit deadbeef.", repo="o/r", checker=checker
        )
        assert len(result) == 1
        assert isinstance(result[0], ClaimError)
        assert "commit deadbeef" in result[0].message
        assert "o/r" in result[0].message

    def test_existing_issue_pr_no_error(self) -> None:
        checker = _StaticChecker(issues_prs={("o/r", 1858)})
        result = validate_reply_prose_claims("see #1858", repo="o/r", checker=checker)
        assert result == []

    def test_missing_issue_yields_error(self) -> None:
        checker = _StaticChecker()
        result = validate_reply_prose_claims("see #9999", repo="o/r", checker=checker)
        assert len(result) == 1
        assert "#9999" in result[0].message

    def test_github_url_validated_against_url_owner_not_repo_arg(self) -> None:
        # URL carries its OWN owner/name — repo= arg is ignored for URLs.
        # An existing commit in the URL's own repo passes even if it
        # doesn't exist in the repo= arg.
        checker = _StaticChecker(commits={("other/repo", "abc1234")})
        result = validate_reply_prose_claims(
            "see https://github.com/other/repo/commit/abc1234",
            repo="o/r",
            checker=checker,
        )
        assert result == []

    def test_multiple_failures_each_reported(self) -> None:
        checker = _StaticChecker()  # nothing exists
        text = (
            "Done in commit abc1234, see also #9999 and "
            "https://github.com/o/r/pull/12345 for context"
        )
        result = validate_reply_prose_claims(text, repo="o/r", checker=checker)
        assert len(result) == 3
        # Errors are returned in extraction order: commit, then #N, then URL.
        assert "commit abc1234" in result[0].message
        assert "#9999" in result[1].message
        assert "pull" in result[2].message or "pr" in result[2].message

    def test_mixed_existing_and_missing_only_missing_reported(self) -> None:
        checker = _StaticChecker(
            commits={("o/r", "abc1234")},
            issues_prs={("o/r", 1858)},
        )
        text = "fixed in commit abc1234 (related to #1858, follow-up #9999)"
        result = validate_reply_prose_claims(text, repo="o/r", checker=checker)
        assert len(result) == 1
        assert "#9999" in result[0].message

    def test_offset_points_at_failing_claim(self) -> None:
        checker = _StaticChecker()
        text = "abcd commit deadbeef wxyz"
        result = validate_reply_prose_claims(text, repo="o/r", checker=checker)
        assert len(result) == 1
        # Offset is the start of the SHA token itself.
        assert text[result[0].offset : result[0].offset + 8] == "deadbeef"

    def test_real_world_prose_no_false_positives(self) -> None:
        # A typical Fido reply: prose plus a fido marker, no claim refs.
        checker = _StaticChecker()
        text = (
            "Got it — I'll add the typed Protocol collaborator and remove "
            "the callable slot.\n\n<!-- fido:reply-promise:abc -->"
        )
        result = validate_reply_prose_claims(text, repo="o/r", checker=checker)
        assert result == []
