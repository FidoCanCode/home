"""Tests for the per-(repo, item) CommentCache (#1748, #1754).

INV-1 scope: shape + per-(repo, item) keying + ``apply_event``.
Hydration via ``load_inventory`` and list getters come in #1756
and are tested there.
"""

from datetime import datetime, timezone
from typing import Any

from frozendict import frozendict

from fido.comment_cache import (
    KIND_ISSUES,
    KIND_PULLS,
    KIND_REVIEWS,
    CommentCache,
)


class _FakeGH:
    """Hand-rolled GitHub fake — no MagicMock per testing convention.

    INV-1 doesn't exercise any GH methods (the cache only fills via
    apply_event), so this is a placeholder that records nothing.
    Hydration tests (#1756) will give it teeth.
    """


def _comment_payload(
    *,
    item: int,
    comment_id: int,
    body: str = "hi",
    author: str = "alice",
    updated_at: str = "2024-01-15T10:00:00Z",
    in_reply_to_id: int | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Raw GitHub comment payload (top-level or review-thread share shape).

    GitHub bumps ``updated_at`` on every edit; freshness check
    relies on this.
    """
    comment: dict[str, Any] = {
        "id": comment_id,
        "body": body,
        "user": {"login": author},
        "created_at": updated_at,
        "updated_at": updated_at,
        "html_url": "https://example/c",
    }
    if in_reply_to_id is not None:
        comment["in_reply_to_id"] = in_reply_to_id
    if path is not None:
        comment["path"] = path
    return comment


def _review_payload(
    *,
    review_id: int,
    state: str = "COMMENTED",
    body: str = "lgtm",
    submitted_at: str = "2024-01-15T10:00:00Z",
) -> dict[str, Any]:
    """Raw GitHub review-submission payload."""
    return {
        "id": review_id,
        "state": state,
        "body": body,
        "user": {"login": "alice"},
        "submitted_at": submitted_at,
    }


class TestConstruction:
    def test_bound_to_one_repo_item_pair(self) -> None:
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        m = cache.metrics()
        assert m.item == 7
        assert m.entries_cached == 0
        # Repo name is folded into the log-friendly identifier.
        assert m.repo_name == "owner/repo#7"

    def test_two_items_in_same_repo_are_independent(self) -> None:
        c7 = CommentCache("owner/repo", _FakeGH(), 7)
        c8 = CommentCache("owner/repo", _FakeGH(), 8)
        c7.apply_event(
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 7},
                "comment": _comment_payload(item=7, comment_id=42),
            },
        )
        # c7 has the entry, c8 is empty.
        assert c7.get(KIND_ISSUES, 42) is not None
        assert c8.get(KIND_ISSUES, 42) is None


class TestApplyIssueComment:
    def _cache(self) -> CommentCache:
        # WebhookCache queues events pre-inventory; mark loaded so
        # INV-1 tests exercise the steady-state path directly.
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        cache.load_inventory([], datetime(2024, 1, 1, tzinfo=timezone.utc))
        return cache

    def test_created_upserts(self) -> None:
        cache = self._cache()
        cache.apply_event(
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 7},
                "comment": _comment_payload(item=7, comment_id=42, body="fresh"),
            },
        )
        got = cache.get(KIND_ISSUES, 42)
        assert got is not None
        assert got["body"] == "fresh"
        assert isinstance(got, frozendict)

    def test_edited_overwrites(self) -> None:
        cache = self._cache()
        cache.apply_event(
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 7},
                "comment": _comment_payload(
                    item=7,
                    comment_id=42,
                    body="original",
                    updated_at="2024-01-15T10:00:00Z",
                ),
            },
        )
        cache.apply_event(
            "issue_comment",
            {
                "action": "edited",
                "issue": {"number": 7},
                "comment": _comment_payload(
                    item=7,
                    comment_id=42,
                    body="edited",
                    updated_at="2024-02-01T10:00:00Z",
                ),
            },
        )
        got = cache.get(KIND_ISSUES, 42)
        assert got is not None and got["body"] == "edited"

    def test_deleted_evicts(self) -> None:
        cache = self._cache()
        cache.apply_event(
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 7},
                "comment": _comment_payload(item=7, comment_id=42),
            },
        )
        assert cache.metrics().entries_cached == 1
        cache.apply_event(
            "issue_comment",
            {
                "action": "deleted",
                "issue": {"number": 7},
                "comment": _comment_payload(
                    item=7,
                    comment_id=42,
                    updated_at="2024-02-01T10:00:00Z",
                ),
            },
        )
        assert cache.metrics().entries_cached == 0
        assert cache.get(KIND_ISSUES, 42) is None


class TestApplyPullRequestReviewComment:
    def _cache(self) -> CommentCache:
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        cache.load_inventory([], datetime(2024, 1, 1, tzinfo=timezone.utc))
        return cache

    def test_routed_to_pulls_kind(self) -> None:
        cache = self._cache()
        cache.apply_event(
            "pull_request_review_comment",
            {
                "action": "created",
                "pull_request": {"number": 7},
                "comment": _comment_payload(
                    item=7,
                    comment_id=100,
                    body="review-thread",
                    path="src/foo.py",
                ),
            },
        )
        got = cache.get(KIND_PULLS, 100)
        assert got is not None and got["body"] == "review-thread"
        # Same id under KIND_ISSUES does NOT collide.
        assert cache.get(KIND_ISSUES, 100) is None


class TestApplyPullRequestReview:
    def _cache(self) -> CommentCache:
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        cache.load_inventory([], datetime(2024, 1, 1, tzinfo=timezone.utc))
        return cache

    def test_routed_to_reviews_kind(self) -> None:
        cache = self._cache()
        cache.apply_event(
            "pull_request_review",
            {
                "action": "submitted",
                "pull_request": {"number": 7},
                "review": _review_payload(
                    review_id=1000, state="APPROVED", body="lgtm"
                ),
            },
        )
        got = cache.get(KIND_REVIEWS, 1000)
        assert got is not None
        assert got["state"] == "APPROVED"
        assert got["body"] == "lgtm"

    def test_review_id_and_comment_id_coexist_under_different_kinds(self) -> None:
        cache = self._cache()
        cache.apply_event(
            "pull_request_review_comment",
            {
                "action": "created",
                "pull_request": {"number": 7},
                "comment": _comment_payload(item=7, comment_id=50, body="comment"),
            },
        )
        cache.apply_event(
            "pull_request_review",
            {
                "action": "submitted",
                "pull_request": {"number": 7},
                "review": _review_payload(review_id=50, body="review"),
            },
        )
        assert cache.metrics().entries_cached == 2
        c = cache.get(KIND_PULLS, 50)
        r = cache.get(KIND_REVIEWS, 50)
        assert c is not None and c["body"] == "comment"
        assert r is not None and r["body"] == "review"


class TestStalenessCheck:
    def _cache(self) -> CommentCache:
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        cache.load_inventory([], datetime(2024, 1, 1, tzinfo=timezone.utc))
        return cache

    def test_older_event_dropped_after_newer(self) -> None:
        # Out-of-order delivery: an older edit arrives after a newer
        # one.  WebhookCache's last_applied_at check drops it.
        cache = self._cache()
        cache.apply_event(
            "issue_comment",
            {
                "action": "edited",
                "issue": {"number": 7},
                "comment": _comment_payload(
                    item=7,
                    comment_id=42,
                    body="newer",
                    updated_at="2024-02-01T10:00:00Z",
                ),
            },
        )
        cache.apply_event(
            "issue_comment",
            {
                "action": "edited",
                "issue": {"number": 7},
                "comment": _comment_payload(
                    item=7,
                    comment_id=42,
                    body="older",
                    updated_at="2024-01-15T10:00:00Z",
                ),
            },
        )
        got = cache.get(KIND_ISSUES, 42)
        assert got is not None and got["body"] == "newer"
        assert cache.metrics().events_dropped_stale == 1


class TestLoadInventory:
    """Inventory hydration shape (#1756 wires the bootstrap; this verifies
    the parse contract that consumers will compose against)."""

    def test_loads_three_kinds_via_explicit_kind_tag(self) -> None:
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        snapshot_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        inventory = [
            {"_kind": KIND_ISSUES, **_comment_payload(item=7, comment_id=10)},
            {
                "_kind": KIND_PULLS,
                **_comment_payload(
                    item=7, comment_id=100, path="src/foo.py", body="line"
                ),
            },
            {"_kind": KIND_REVIEWS, **_review_payload(review_id=1000, body="lgtm")},
        ]
        cache.load_inventory(inventory, snapshot_ts)
        assert cache.metrics().entries_cached == 3
        assert cache.get(KIND_ISSUES, 10) is not None
        assert cache.get(KIND_PULLS, 100) is not None
        rev = cache.get(KIND_REVIEWS, 1000)
        assert rev is not None and rev["body"] == "lgtm"

    def test_reconcile_evicts_missing_and_uses_nodes_equal(self) -> None:
        # Reconcile diffs the cache against a fresh snapshot: ids absent
        # are removed; ids that diverge from the snapshot are replaced.
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        snapshot_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        cache.load_inventory(
            [
                {"_kind": KIND_ISSUES, **_comment_payload(item=7, comment_id=10)},
                {"_kind": KIND_ISSUES, **_comment_payload(item=7, comment_id=11)},
            ],
            snapshot_ts,
        )
        # Reconcile with a fresh snapshot that drops 11 and changes 10.
        cache.reconcile_with_inventory(
            [
                {
                    "_kind": KIND_ISSUES,
                    **_comment_payload(item=7, comment_id=10, body="edited"),
                },
            ],
            datetime(2024, 7, 1, tzinfo=timezone.utc),
        )
        assert cache.metrics().entries_cached == 1
        ten = cache.get(KIND_ISSUES, 10)
        assert ten is not None and ten["body"] == "edited"
        assert cache.get(KIND_ISSUES, 11) is None


class TestIgnoredEvents:
    def _cache(self) -> CommentCache:
        cache = CommentCache("owner/repo", _FakeGH(), 7)
        cache.load_inventory([], datetime(2024, 1, 1, tzinfo=timezone.utc))
        return cache

    def test_unrelated_event_type_ignored(self) -> None:
        cache = self._cache()
        cache.apply_event("push", {"ref": "refs/heads/main"})
        assert cache.metrics().entries_cached == 0
        assert cache.metrics().events_applied == 0

    def test_event_for_other_item_ignored(self) -> None:
        cache = self._cache()
        # Bound to item 7, event targets item 99.
        cache.apply_event(
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 99},
                "comment": _comment_payload(item=99, comment_id=42),
            },
        )
        assert cache.metrics().entries_cached == 0
        assert cache.metrics().events_applied == 0

    def test_malformed_payload_ignored(self) -> None:
        cache = self._cache()
        # No issue/pull_request key.
        cache.apply_event("issue_comment", {"action": "created"})
        # parent is not a dict.
        cache.apply_event("issue_comment", {"action": "created", "issue": "not-a-dict"})
        # No number on parent.
        cache.apply_event("issue_comment", {"action": "created", "issue": {}})
        # number not int.
        cache.apply_event(
            "issue_comment", {"action": "created", "issue": {"number": "x"}}
        )
        # No comment payload.
        cache.apply_event(
            "issue_comment", {"action": "created", "issue": {"number": 7}}
        )
        # comment not a dict.
        cache.apply_event(
            "issue_comment",
            {"action": "created", "issue": {"number": 7}, "comment": "nope"},
        )
        # comment id missing.
        cache.apply_event(
            "issue_comment",
            {"action": "created", "issue": {"number": 7}, "comment": {}},
        )
        # comment id not int.
        cache.apply_event(
            "issue_comment",
            {
                "action": "created",
                "issue": {"number": 7},
                "comment": {"id": "x"},
            },
        )
        assert cache.metrics().events_applied == 0
