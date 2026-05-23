"""Unit tests for fido/prompts.py — prompt-building functions and Prompts class."""

from typing import Any

from fido.prompts import (
    TRIAGE_CLAUSE,
    Prompts,
    render_active_context,
    triage_context_block,
)
from fido.types import (
    ActiveIssue,
    ActivePR,
    ClosedPR,
    ClosedSubIssue,
    RescopeIntent,
    TaskSnapshot,
)

# ── triage_context_block ──────────────────────────────────────────────────────


class TestTriageContextBlock:
    def test_empty_context(self) -> None:
        assert triage_context_block(None) == ""
        assert triage_context_block({}) == ""

    def test_pr_title_only(self) -> None:
        result = triage_context_block({"pr_title": "Fix bug"})
        assert "PR: Fix bug" in result

    def test_file_only(self) -> None:
        result = triage_context_block({"file": "src/foo.py"})
        assert "File: src/foo.py" in result

    def test_diff_hunk_only(self) -> None:
        result = triage_context_block({"diff_hunk": "@@ -1,2 +1,3 @@"})
        assert "Diff:" in result
        assert "@@ -1,2 +1,3 @@" in result

    def test_pr_body(self) -> None:
        result = triage_context_block({"pr_body": "Adds caching to the parser."})
        assert "PR description:" in result
        assert "Adds caching to the parser." in result

    def test_empty_pr_body_omitted(self) -> None:
        result = triage_context_block({"pr_body": ""})
        assert "PR description:" not in result

    def test_all_fields(self) -> None:
        result = triage_context_block(
            {
                "pr_title": "Refactor",
                "pr_body": "Refactors the parser.",
                "file": "app.py",
                "diff_hunk": "- old\n+ new",
            }
        )
        assert "PR: Refactor" in result
        assert "PR description:" in result
        assert "Refactors the parser." in result
        assert "File: app.py" in result
        assert "Diff:" in result
        assert "- old\n+ new" in result

    def test_ignores_unknown_keys(self) -> None:
        result = triage_context_block({"unknown_key": "value", "pr_title": "hi"})
        assert "unknown_key" not in result
        assert "PR: hi" in result

    def test_sibling_threads_rendered(self) -> None:
        result = triage_context_block(
            {
                "sibling_threads": [
                    {
                        "path": "src/foo.py",
                        "line": 10,
                        "comments": [
                            {"author": "alice", "body": "why is this here?"},
                            {"author": "fido", "body": "good catch!"},
                        ],
                    }
                ]
            }
        )
        assert "Sibling threads:" in result
        assert "src/foo.py:10" in result
        assert "alice: why is this here?" in result
        assert "fido: good catch!" in result

    def test_sibling_threads_no_line(self) -> None:
        result = triage_context_block(
            {
                "sibling_threads": [
                    {
                        "path": "README.md",
                        "line": None,
                        "comments": [{"author": "bob", "body": "typo"}],
                    }
                ]
            }
        )
        assert "README.md" in result
        assert "bob: typo" in result

    def test_sibling_threads_multiple(self) -> None:
        result = triage_context_block(
            {
                "sibling_threads": [
                    {
                        "path": "a.py",
                        "line": 1,
                        "comments": [{"author": "x", "body": "first"}],
                    },
                    {
                        "path": "b.py",
                        "line": 2,
                        "comments": [{"author": "y", "body": "second"}],
                    },
                ]
            }
        )
        assert "a.py:1" in result
        assert "b.py:2" in result
        assert "x: first" in result
        assert "y: second" in result

    def test_empty_sibling_threads_omitted(self) -> None:
        result = triage_context_block({"sibling_threads": []})
        assert "Sibling threads:" not in result

    def test_comment_thread_rendered(self) -> None:
        result = triage_context_block(
            {
                "comment_thread": [
                    {"author": "alice", "body": "fix this please"},
                    {"author": "fido", "body": "done in latest commit"},
                ]
            }
        )
        assert "Comment thread:" in result
        assert "alice: fix this please" in result
        assert "fido: done in latest commit" in result

    def test_empty_comment_thread_omitted(self) -> None:
        result = triage_context_block({"comment_thread": []})
        assert "Comment thread:" not in result

    def test_conversation_rendered(self) -> None:
        result = triage_context_block(
            {"conversation": "\n\nFull conversation:\nalice: hi"}
        )
        assert "Full conversation:" in result
        assert "alice: hi" in result


# ── Prompts.status_system_prompt ─────────────────────────────────────────────


class TestStatusSystemPrompt:
    def test_returns_string(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert isinstance(result, str)

    def test_mentions_json(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "JSON" in result

    def test_mentions_status_and_emoji_fields(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert '"status"' in result
        assert '"emoji"' in result

    def test_mentions_80_chars(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "80" in result

    def test_mentions_fido(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "Fido" in result

    def test_instructs_busy_priority(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "busy" in result

    def test_instructs_idle_napping(self) -> None:
        result = Prompts("persona").status_system_prompt()
        assert "idle" in result or "napping" in result.lower()


# ── Prompts class ─────────────────────────────────────────────────────────────


class TestPromptsReplySystemPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").reply_system_prompt()
        assert "I am Fido." in result

    def test_prohibits_preamble_phrases(self) -> None:
        result = Prompts("persona").reply_system_prompt()
        assert "Here's" in result or "preamble" in result

    def test_output_only_instruction(self) -> None:
        result = Prompts("persona").reply_system_prompt()
        assert "ONLY" in result

    def test_no_meta_commentary(self) -> None:
        result = Prompts("persona").reply_system_prompt()
        assert "meta-commentary" in result or "Here's the reply" in result

    def test_active_context_included_when_issue_provided(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="It crashes.")
        result = Prompts("persona").reply_system_prompt(issue=issue)
        assert "## Active issue" in result
        assert "Fix crash" in result
        assert "It crashes." in result

    def test_no_active_context_when_issue_is_none(self) -> None:
        result = Prompts("persona").reply_system_prompt()
        assert "## Active issue" not in result

    def test_active_context_includes_pr_when_provided(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="")
        pr = ActivePR(
            number=42,
            title="Fix crash PR",
            url="https://github.com/a/b/pull/42",
            body="",
        )
        result = Prompts("persona").reply_system_prompt(issue=issue, pr=pr)
        assert "## Active PR" in result
        assert "Fix crash PR" in result

    def test_active_context_no_pr_section_when_pr_is_none(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="")
        result = Prompts("persona").reply_system_prompt(issue=issue, pr=None)
        assert "## Active PR" not in result


class TestPromptsPersonaWrap:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").persona_wrap("Write a reply.")
        assert "I am Fido." in result

    def test_includes_instruction(self) -> None:
        result = Prompts("persona").persona_wrap("do the thing")
        assert "do the thing" in result

    def test_includes_output_constraint(self) -> None:
        result = Prompts("persona").persona_wrap("instruction")
        assert "Output only the comment text" in result
        assert "no quotes" in result

    def test_empty_persona(self) -> None:
        result = Prompts("").persona_wrap("instruct")
        assert "instruct" in result
        assert "Output only" in result


class TestPromptsStatusPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").status_prompt(
            [("owner/repo", "writing tests", True)]
        )
        assert "I am Fido." in result

    def test_includes_what(self) -> None:
        result = Prompts("persona").status_prompt(
            [("owner/repo", "reviewing PRs", True)]
        )
        assert "reviewing PRs" in result

    def test_includes_repo_name(self) -> None:
        result = Prompts("persona").status_prompt(
            [("FidoCanCode/home", "fixing a bug", True)]
        )
        assert "FidoCanCode/home" in result

    def test_busy_worker_labeled(self) -> None:
        result = Prompts("persona").status_prompt(
            [("owner/repo", "working hard", True)]
        )
        assert "busy" in result

    def test_idle_worker_labeled(self) -> None:
        result = Prompts("persona").status_prompt([("owner/repo", "napping", False)])
        assert "idle" in result

    def test_multiple_repos_all_present(self) -> None:
        result = Prompts("persona").status_prompt(
            [
                ("a/busy", "Writing code", True),
                ("b/idle", "Napping", False),
            ]
        )
        assert "a/busy" in result
        assert "Writing code" in result
        assert "b/idle" in result
        assert "Napping" in result

    def test_empty_activities(self) -> None:
        result = Prompts("persona").status_prompt([])
        assert isinstance(result, str)
        assert "No active workers" in result


class TestPromptsPickupCommentPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").pickup_comment_prompt("Fix the thing")
        assert "I am Fido." in result

    def test_includes_issue_title(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("Refactor auth module")
        assert "Refactor auth module" in result

    def test_includes_plain_text(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("Add caching")
        assert "Picking up issue: Add caching" in result

    def test_instructs_fido_character(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("title")
        assert "Fido" in result

    def test_requests_short_output(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("title")
        assert "1-2 sentences" in result

    def test_output_constraint_present(self) -> None:
        result = Prompts("persona").pickup_comment_prompt("title")
        assert "Output only the comment text" in result

    def test_empty_persona(self) -> None:
        result = Prompts("").pickup_comment_prompt("Some issue")
        assert "Picking up issue: Some issue" in result

    def test_returns_string(self) -> None:
        assert isinstance(Prompts("persona").pickup_comment_prompt("title"), str)


class TestPromptsPickupRetryCommentPrompt:
    """Fresh-retry pickup-ack prompt (fix for FidoCanCode/home#802)."""

    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").pickup_retry_comment_prompt("t", [10])
        assert "I am Fido." in result

    def test_includes_issue_title_and_prs(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("Migrate Gitea", [215, 210])
        assert "Migrate Gitea" in result
        assert "#215" in result
        assert "#210" in result

    def test_instructs_acknowledgement(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [7])
        assert "starting genuinely fresh" in result
        assert "closed PR" in result

    def test_requests_fresh_start_commitment(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [1])
        assert "not reusing anything" in result

    def test_output_constraint_present(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [1])
        assert "Output only the comment text" in result

    def test_single_pr_formatted(self) -> None:
        result = Prompts("p").pickup_retry_comment_prompt("t", [42])
        assert "#42" in result

    def test_returns_string(self) -> None:
        assert isinstance(Prompts("p").pickup_retry_comment_prompt("t", [1]), str)


# ── Prompts.rescope_prompt ───────────────────────────────────────────────────


class TestRescopePrompt:
    def _task(
        self,
        title: str,
        task_id: str = "1",
        task_type: str = "spec",
        status: str = "pending",
        description: str = "",
    ) -> dict:
        return {
            "id": task_id,
            "title": title,
            "type": task_type,
            "status": status,
            "description": description,
        }

    def test_includes_pending_tasks_json(self) -> None:
        tasks = [self._task("Add feature", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "Add feature" in result
        assert '"id": "1"' in result

    def test_excludes_completed_from_pending_json(self) -> None:
        tasks = [
            self._task("Done task", task_id="1", status="completed"),
            self._task("Todo task", task_id="2"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        # Completed appears in the completed block, not the pending JSON
        assert '"id": "2"' in result
        assert '"id": "1"' not in result.split("Pending tasks")[1]

    def test_completed_titles_listed_in_completed_block(self) -> None:
        tasks = [
            self._task("Already done", task_id="1", status="completed"),
            self._task("Still pending", task_id="2"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "Already done" in result.split("Pending tasks")[0]

    def test_no_completed_tasks_shows_none(self) -> None:
        tasks = [self._task("Only pending", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "(none)" in result.split("Pending tasks")[0]

    def test_commit_summary_included(self) -> None:
        tasks = [self._task("Add tests", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "feat: add parser method")
        assert "feat: add parser method" in result

    def test_narrative_chain_rendered_when_present(self) -> None:
        # HOL-8 / #1902: per-task narrative_chain lands in the rescope
        # prompt so Opus sees the history of verdicts that touched the
        # task across prior rescopes.
        task = self._task("Carry-over", task_id="T1")
        task["narrative_chain"] = [
            {
                "outcome": "reshaped",
                "narrative": "Split into prereq + feature.",
                "intent_comment_id": 42,
                "ts": "2026-05-23T10:00:00Z",
            }
        ]
        result = Prompts("").rescope_prompt([task], "")
        # The chain JSON is part of the pending-task block.
        assert '"narrative_chain"' in result
        assert "Split into prereq + feature." in result
        assert '"intent_comment_id": 42' in result

    def test_narrative_chain_omitted_when_empty(self) -> None:
        # No chain → no field in the rendered JSON.  Keeps the prompt
        # compact for fresh tasks that haven't gone through a rescope.
        tasks = [self._task("Fresh", task_id="T1")]
        result = Prompts("").rescope_prompt(tasks, "")
        # Pending JSON block must NOT contain the field name.
        assert '"narrative_chain"' not in result

    def test_empty_commit_summary_shows_none(self) -> None:
        tasks = [self._task("Add tests", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "(none)" in result

    def test_ci_tasks_must_come_first_rule_stated(self) -> None:
        tasks = [self._task("Fix CI", task_id="1", task_type="ci")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "ci" in result.lower()
        assert "first" in result

    def test_json_output_format_instructed(self) -> None:
        tasks = [self._task("Do something", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert '{"operations": [...]}' in result

    def test_preserve_ids_rule_stated(self) -> None:
        tasks = [self._task("Task A", task_id="abc-123")]
        result = Prompts("").rescope_prompt(tasks, "")
        # New schema requires every existing id to appear in exactly one
        # operation — the preservation rule is "id must be in the
        # snapshot" and "no inventing ids".
        assert "must be in the pending snapshot" in result
        assert "no inventing ids" in result

    def test_remove_covered_tasks_rule_stated(self) -> None:
        tasks = [self._task("Task A", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "commit covering this")
        assert "commit" in result.lower() or "covered" in result.lower()

    def test_rewrite_spec_on_thread_conflict_rule_stated(self) -> None:
        tasks = [
            self._task("Old spec title", task_id="1", task_type="spec"),
            self._task("New comment task", task_id="2", task_type="thread"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "thread" in result.lower() or "rewrite" in result.lower()

    def test_no_other_text_instruction_present(self) -> None:
        tasks = [self._task("X", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "No other text" in result

    def test_in_progress_tasks_included_in_pending(self) -> None:
        tasks = [
            self._task("Running task", task_id="1", status="in_progress"),
        ]
        result = Prompts("").rescope_prompt(tasks, "")
        assert '"id": "1"' in result

    def test_description_included_in_task_json(self) -> None:
        tasks = [self._task("X", task_id="1", description="important details")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "important details" in result

    def test_empty_task_list(self) -> None:
        result = Prompts("").rescope_prompt([], "")
        assert isinstance(result, str)
        assert "(none)" in result  # both completed and commit summary

    def test_active_context_prefix_included_when_issue_provided(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        issue = ActiveIssue(number=42, title="Fix crash", body="It crashes on startup.")
        result = Prompts("").rescope_prompt(tasks, "", issue=issue)
        assert "## Active issue" in result
        assert "#42: Fix crash" in result
        assert "It crashes on startup." in result

    def test_no_active_context_prefix_when_issue_is_none(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "", issue=None)
        assert "## Active issue" not in result

    def test_active_context_includes_pr_when_provided(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        issue = ActiveIssue(number=1, title="T", body="")
        pr = ActivePR(
            number=7,
            title="Fix T (closes #1)",
            url="https://github.com/o/r/pull/7",
            body="",
        )
        result = Prompts("").rescope_prompt(tasks, "", issue=issue, pr=pr)
        assert "## Active PR" in result
        assert "PR #7" in result

    def test_active_context_no_pr_section_when_pr_is_none(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        issue = ActiveIssue(number=1, title="T", body="")
        result = Prompts("").rescope_prompt(tasks, "", issue=issue, pr=None)
        assert "## Active issue" in result
        assert "## Active PR" not in result

    def test_intents_block_included_when_intents_provided(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent(
                change_request="Add logging to the parser",
                comment_id=123,
                timestamp="2024-01-15T10:00:00+00:00",
            )
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        assert "Pending change requests from PR comments" in result
        assert "comment #123" in result
        assert "Add logging to the parser" in result
        assert "2024-01-15T10:00:00+00:00" in result

    def test_intents_block_absent_when_no_intents(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "Pending change requests" not in result

    def test_multiple_intents_all_included(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent("Add logging", 111, "2024-01-15T10:00:00+00:00"),
            RescopeIntent("Refactor tests", 222, "2024-01-15T10:01:00+00:00"),
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        assert "comment #111" in result
        assert "comment #222" in result
        assert "Add logging" in result
        assert "Refactor tests" in result

    def test_intents_rendered_in_timestamp_order(self) -> None:
        # #1720: out-of-order arrival → timestamp-ordered prompt.  The
        # rescope prompt sorts by timestamp before rendering so the
        # "newer overrides older on conflict" rule has a stable
        # referent that doesn't depend on webhook delivery order or
        # rescope batching reshuffles.
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent("Newest", 333, "2024-03-01T12:00:00+00:00"),
            RescopeIntent("Oldest", 111, "2024-01-15T10:00:00+00:00"),
            RescopeIntent("Middle", 222, "2024-02-15T11:00:00+00:00"),
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        oldest_pos = result.index("Oldest")
        middle_pos = result.index("Middle")
        newest_pos = result.index("Newest")
        assert oldest_pos < middle_pos < newest_pos, (
            "intents must render oldest-first regardless of arrival order"
        )

    def test_newer_overrides_older_conflict_rule_stated(self) -> None:
        # #1720: the conflict-resolution rule must be present so Opus
        # knows how to interpret two intents pointing at the same task.
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent("First take", 111, "2024-01-15T10:00:00+00:00"),
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        assert "newer one" in result and "overrides" in result
        # And the inverse: unrelated intents are preserved (not absorbed
        # by a "newer wins" reading).
        assert "don't conflict" in result

    def test_intents_block_renders_author_when_present(self) -> None:
        # #1801 / INV-C: per-intent author identity is part of the
        # input contract.  Opus needs it to apply the self-supersedence
        # rule (#1803 / INV-E) — suppress reply-back when an intent
        # was overridden by another from the same author.
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent(
                change_request="Add logging",
                comment_id=111,
                timestamp="2024-01-15T10:00:00+00:00",
                author="alice",
            ),
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        assert "by @alice" in result

    def test_intents_block_renders_unknown_author_marker_when_blank(self) -> None:
        # Backward-compat: pre-#1801 intents have author="".  Render a
        # ``?`` marker so Opus can tell "unknown author" from "real
        # author" rather than seeing a silently-empty ``by @``.
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent("Add logging", 111, "2024-01-15T10:00:00+00:00"),
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        assert "by @?" in result

    def test_intents_block_states_self_supersedence_rule(self) -> None:
        # #1801 / INV-C: prompt must spell out that self-supersedence
        # is a self-correction (no reply-back) while cross-author
        # supersedence warrants follow-up.  This is the contract the
        # downstream INV-E executor relies on Opus understanding.
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent("X", 111, "2024-01-15T10:00:00+00:00", author="alice"),
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        assert "Self-correction" in result
        # "*same* author" — Markdown emphasis on "same" inside the
        # prompt body — so we match the keyword fragments separately.
        assert "same" in result and "author" in result
        assert "Cross-author" in result

    def test_intents_block_header_announces_ordering(self) -> None:
        # The header line tells Opus the list is chronological so the
        # ordering itself is a signal, not just visual sugar.
        tasks = [self._task("Do thing", task_id="1")]
        intents = [
            RescopeIntent("X", 111, "2024-01-15T10:00:00+00:00"),
        ]
        result = Prompts("").rescope_prompt(tasks, "", intents=intents)
        assert "timestamp order" in result
        assert "oldest first" in result

    def test_explicit_operations_framing_present(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        # New schema replaces "synthesize a task list" with "reply with
        # a typed list of operations" (#1719).  Each pending id may
        # appear in at most one operation; omission means keep (#1721).
        # Codex on PR #1744 caught the contradiction between the
        # original "MUST appear in exactly one operation" framing and
        # rule 7's "omitted ids are kept" — assert the resolved wording
        # so the contradiction can't silently come back.
        assert "operations" in result.lower()
        assert "at most one operation" in result
        assert "kept unchanged" in result

    def test_every_op_kind_documented(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        # The schema reference must list every operation Opus can emit
        # so it has the full vocabulary inline.
        for op_name in (
            "keep",
            "rewrite",
            "rewrite_anchor",
            "remove",
            "merge",
            "split",
            "new",
        ):
            assert f'"op": "{op_name}"' in result, f"missing op {op_name}"

    def test_new_op_documented_for_brand_new_tasks(self) -> None:
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        # Brand-new tasks come from "new" ops, no null-id mechanism
        # in the new schema.
        assert '{"op": "new"' in result

    def test_new_op_schema_advertises_invariant_field(self) -> None:
        """HOL-12 / #1906: the ``new`` op schema must show
        ``"invariant": "..."`` so Opus emits one per task."""
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        # Look at the new-op JSON sample only (not the constraint
        # paragraph, which mentions "invariant" as well).
        new_op_block = result.split('{"op": "new"')[1].split("\n")[0]
        assert '"invariant"' in new_op_block

    def test_split_children_schema_advertises_invariant_field(self) -> None:
        """HOL-12 / #1906: every ``split`` child is a fresh task too,
        so the schema example must include an ``invariant`` field on
        each child."""
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        split_block = result.split('{"op": "split"')[1].split("\n")[0]
        assert '"invariant"' in split_block

    def test_one_invariant_per_task_rule_stated(self) -> None:
        """HOL-12 / #1906: the rule itself — ``new``/``split`` children
        carry exactly one invariant — must appear in plain English so
        Opus reasons about it, not just the schema field."""
        tasks = [self._task("Do thing", task_id="1")]
        result = Prompts("").rescope_prompt(tasks, "")
        assert "One invariant per task" in result
        assert "HOL-12" in result
        # The split-when-too-big guidance has to be explicit — otherwise
        # Opus will emit one giant ``new`` op with a hand-wavy
        # invariant rather than fan it into multiple ops.
        assert "split it into multiple" in result


# ── Prompts.rescope_prompt_verdicts (#1810 / INV-D wiring leaf) ──────────────


class TestRescopePromptVerdicts:
    @staticmethod
    def _task(title: str = "Do thing", task_id: str = "1") -> dict[str, Any]:
        return {
            "id": task_id,
            "title": title,
            "type": "spec",
            "status": "pending",
            "description": "",
        }

    @staticmethod
    def _intent(
        cid: int,
        text: str = "Add logging",
        *,
        timestamp: str = "2024-01-15T10:00:00+00:00",
        author: str = "alice",
    ) -> RescopeIntent:
        return RescopeIntent(
            change_request=text,
            comment_id=cid,
            timestamp=timestamp,
            author=author,
        )

    def test_asks_for_verdicts_envelope(self) -> None:
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        # Envelope keyword + explicit JSON shape directive.
        assert '"verdicts"' in result
        assert '{"verdicts": [...]}' in result

    def test_narrative_chain_rendered_when_present(self) -> None:
        # HOL-8 / #1902: per-task narrative_chain renders in the
        # verdict-envelope prompt too — the verdict-shape prompt is
        # the modern path, so the chain has to land here.
        task = self._task("Carry-over", task_id="T1")
        task["narrative_chain"] = [
            {
                "outcome": "no_op",
                "narrative": "Already covered by an in-flight task.",
                "intent_comment_id": 99,
                "ts": "2026-05-23T11:00:00Z",
            }
        ]
        result = Prompts("").rescope_prompt_verdicts(
            [task], "", intents=[self._intent(1)]
        )
        assert '"narrative_chain"' in result
        assert "Already covered by an in-flight task." in result
        assert '"intent_comment_id": 99' in result

    def test_narrative_chain_omitted_when_empty(self) -> None:
        # No chain → no field — keeps the prompt compact for fresh
        # tasks that haven't seen a rescope yet.
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        assert '"narrative_chain"' not in result

    def test_includes_each_intent_with_author(self) -> None:
        intents = [
            self._intent(111, "Add logging", author="alice"),
            self._intent(
                222,
                "Refactor tests",
                timestamp="2024-01-15T10:01:00+00:00",
                author="bob",
            ),
        ]
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=intents
        )
        assert "comment #111" in result
        assert "by @alice" in result
        assert "comment #222" in result
        assert "by @bob" in result

    def test_intents_rendered_chronologically(self) -> None:
        intents = [
            self._intent(333, "Newest", timestamp="2024-03-01T12:00:00+00:00"),
            self._intent(111, "Oldest", timestamp="2024-01-15T10:00:00+00:00"),
            self._intent(222, "Middle", timestamp="2024-02-15T11:00:00+00:00"),
        ]
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=intents
        )
        oldest_pos = result.index("Oldest")
        middle_pos = result.index("Middle")
        newest_pos = result.index("Newest")
        assert oldest_pos < middle_pos < newest_pos

    def test_lists_intent_ids_in_schema_hint(self) -> None:
        # The schema sample includes the actual batch's intent ids in
        # the "must be one of" hint so Opus can't reference a
        # made-up id.
        intents = [self._intent(111), self._intent(222)]
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=intents
        )
        assert "must be one of: 111, 222" in result or "111, 222" in result

    def test_outcome_enum_documented(self) -> None:
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        for name in ("honored", "reshaped", "superseded", "no_op"):
            assert f'"{name}"' in result

    def test_outcome_semantics_distinguish_reply_back(self) -> None:
        # Prompt teaches Opus when reply-back fires so it can frame
        # narrative appropriately.
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        # honored explicitly does NOT warrant reply-back
        assert "no follow-up" in result
        # HOL-3 / #1897: narrative is REQUIRED on every outcome —
        # honored / reshaped / superseded / no_op all carry one.
        assert "Narrative REQUIRED" in result
        # The no_op narrative IS the reply-back reason (PR #1890 fix).
        assert "PR #1890" in result

    def test_supersedence_constraints_stated(self) -> None:
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        # by_intent_comment_id only on superseded
        assert "only set when ``outcome == 'superseded'``" in result
        # acyclic
        assert "acyclic" in result

    def test_joint_attribution_allowed(self) -> None:
        # The 3+1 reviewer-pattern case — multiple verdicts can share
        # an affected task id when all are honored together.
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        assert "affected_task_ids" in result
        assert "may overlap" in result

    def test_no_op_constraint_stated(self) -> None:
        # Slice-1 boundary contract: no_op verdicts have empty ops + ids.
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        assert "no_op" in result
        # Phrased as MUST be empty.
        assert "MUST be empty" in result

    def test_includes_op_schema_recap(self) -> None:
        # Same op vocabulary as rescope_prompt, restated inline so
        # Opus has full reference without recall.
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        for op_name in (
            "keep",
            "rewrite",
            "rewrite_anchor",
            "remove",
            "merge",
            "split",
            "new",
        ):
            assert f'"op": "{op_name}"' in result

    def test_new_op_schema_advertises_invariant_field(self) -> None:
        """HOL-12 / #1906: ``new`` op schema in the verdict envelope
        must advertise ``invariant`` so Opus emits one per task."""
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        new_op_block = result.split('{"op": "new"')[1].split("\n")[0]
        assert '"invariant"' in new_op_block

    def test_split_children_schema_advertises_invariant_field(self) -> None:
        """HOL-12 / #1906: every ``split`` child carries one invariant."""
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        split_block = result.split('{"op": "split"')[1].split("\n")[0]
        assert '"invariant"' in split_block

    def test_one_invariant_per_task_rule_stated(self) -> None:
        """HOL-12 / #1906: the prose rule must be explicit (so Opus
        reasons about scope, not just fills in a slot)."""
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)]
        )
        assert "One invariant per task" in result
        assert "HOL-12" in result
        assert "split it into multiple" in result

    def test_current_task_list_included(self) -> None:
        tasks = [
            self._task("Do A", task_id="T1"),
            self._task("Do B", task_id="T2"),
        ]
        result = Prompts("").rescope_prompt_verdicts(
            tasks, "", intents=[self._intent(1)]
        )
        assert "Do A" in result
        assert "Do B" in result

    def test_active_context_rendered_when_issue_provided(self) -> None:
        # Coverage: the ``issue is not None`` branch that pulls in
        # render_active_context.  Matches rescope_prompt's behavior.
        issue = ActiveIssue(number=7, title="Fix crash", body="It crashes.")
        result = Prompts("").rescope_prompt_verdicts(
            [self._task()], "", intents=[self._intent(1)], issue=issue
        )
        assert "## Active issue" in result
        assert "Fix crash" in result


class TestRescopeVerdictsParseNudge:
    def test_includes_errors(self) -> None:
        result = Prompts("").rescope_verdicts_parse_nudge(
            ["verdicts[0]: missing required field 'outcome'"],
            attempts_remaining=2,
        )
        assert "missing required field 'outcome'" in result

    def test_includes_attempt_budget(self) -> None:
        result = Prompts("").rescope_verdicts_parse_nudge(["err"], attempts_remaining=2)
        assert "2 attempt(s) remaining" in result

    def test_final_attempt_message(self) -> None:
        result = Prompts("").rescope_verdicts_parse_nudge(["err"], attempts_remaining=0)
        assert "final attempt" in result
        assert "dropped" in result

    def test_verdict_schema_recap(self) -> None:
        result = Prompts("").rescope_verdicts_parse_nudge(["err"], attempts_remaining=1)
        assert "intent_comment_id" in result
        assert "by_intent_comment_id" in result
        assert "outcome" in result

    def test_asks_for_verdicts_envelope_on_retry(self) -> None:
        result = Prompts("").rescope_verdicts_parse_nudge(["err"], attempts_remaining=1)
        assert '{"verdicts": [...]}' in result


# ── Prompts.rescope_duplicate_nudge ──────────────────────────────────────────


class TestRescopeDuplicateNudge:
    def test_includes_duplicate_titles(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(
            ["Same name"], attempts_remaining=0
        )
        assert "Same name" in result

    def test_includes_multiple_duplicate_titles(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(
            ["Title A", "Title B"], attempts_remaining=0
        )
        assert "Title A" in result
        assert "Title B" in result

    def test_asks_for_unique_titles(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert "unique" in result.lower()

    def test_includes_json_format_instruction(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert '{"operations": [...]}' in result

    def test_no_other_text_instruction_present(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert "No other text" in result

    def test_final_attempt_message_when_zero_remaining(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=0)
        assert "final attempt" in result.lower()

    def test_remaining_count_when_nonzero(self) -> None:
        result = Prompts("").rescope_duplicate_nudge(["Dup"], attempts_remaining=2)
        assert "2" in result


# ── Prompts.stores_persona ────────────────────────────────────────────────────


class TestPromptsStoresPersona:
    def test_persona_stored(self) -> None:
        p = Prompts("my persona")
        assert p.persona == "my persona"

    def test_different_personas_independent(self) -> None:
        p1 = Prompts("persona A")
        p2 = Prompts("persona B")
        activities = [("owner/repo", "working", True)]
        assert "persona A" in p1.status_prompt(activities)
        assert "persona B" in p2.status_prompt(activities)
        assert "persona A" not in p2.status_prompt(activities)


# ── Prompts.rewrite_description_prompt ───────────────────────────────────────


class TestRewriteDescriptionPrompt:
    def _task(
        self,
        title: str,
        task_id: str = "1",
        status: str = "pending",
        description: str = "",
    ) -> dict:
        return {
            "id": task_id,
            "title": title,
            "status": status,
            "description": description,
        }

    def _body(self, desc: str = "Does something useful.\n\nFixes #5.") -> str:
        return (
            f"{desc}\n\n---\n\n## Work queue\n\n"
            "<!-- WORK_QUEUE_START -->\n- [ ] do a thing\n<!-- WORK_QUEUE_END -->"
        )

    def test_includes_current_description(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body("Implements the feature.\n\nFixes #7."),
            [self._task("New task")],
        )
        assert "Implements the feature." in result

    def test_excludes_work_queue_section_from_context(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(), [self._task("A task")]
        )
        assert "WORK_QUEUE_START" not in result
        assert "do a thing" not in result

    def test_includes_pending_tasks(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [self._task("Add caching layer")],
        )
        assert "Add caching layer" in result

    def test_excludes_completed_tasks(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [
                self._task("Done already", status="completed"),
                self._task("Still pending"),
            ],
        )
        assert "Done already" not in result

    def test_empty_pending_shows_none(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [self._task("Finished", status="completed")],
        )
        assert "(none)" in result

    def test_task_description_included(self) -> None:
        result = Prompts("").rewrite_description_prompt(
            self._body(),
            [self._task("Cache results", description="use Redis")],
        )
        assert "use Redis" in result

    def test_fixes_line_preservation_rule_stated(self) -> None:
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert "Fixes #N" in result or "Fixes #" in result
        assert "preserve" in result.lower() or "exactly" in result.lower()

    def test_no_work_queue_content_rule_stated(self) -> None:
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert "work queue" in result.lower()

    def test_body_tag_contract_stated(self) -> None:
        """Prompt must instruct Opus to wrap output in <body> tags so we can
        reliably strip preamble and trailing chatter."""
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert "<body>" in result
        assert "</body>" in result

    def test_extracts_description_at_divider(self) -> None:
        body = "My description.\n\nFixes #3.\n\n---\n\nStuff below divider."
        result = Prompts("").rewrite_description_prompt(body, [])
        assert "My description." in result
        assert "Stuff below divider." not in result

    def test_fallback_to_wq_marker_when_no_divider(self) -> None:
        body = (
            "Short desc.\n<!-- WORK_QUEUE_START -->"
            "\n- [ ] task\n<!-- WORK_QUEUE_END -->"
        )
        result = Prompts("").rewrite_description_prompt(body, [])
        assert "Short desc." in result
        assert "WORK_QUEUE_START" not in result

    def test_fallback_to_full_body_when_no_markers(self) -> None:
        body = "Plain description with no markers."
        result = Prompts("").rewrite_description_prompt(body, [])
        assert "Plain description with no markers." in result

    def test_empty_task_list(self) -> None:
        result = Prompts("").rewrite_description_prompt(self._body(), [])
        assert isinstance(result, str)
        assert "(none)" in result


# ── render_active_context ─────────────────────────────────────────────────────


class TestRenderActiveContext:
    """Tests for the render_active_context() module-level renderer."""

    def _issue(
        self,
        number: int = 42,
        title: str = "Fix the parser",
        body: str = "It crashes on empty input.",
    ) -> ActiveIssue:
        return ActiveIssue(number=number, title=title, body=body)

    def _pr(
        self,
        number: int = 10,
        title: str = "Fix parser crash",
        url: str = "https://github.com/owner/repo/pull/10",
        body: str = "Implements the fix.",
    ) -> ActivePR:
        return ActivePR(number=number, title=title, url=url, body=body)

    def _task(
        self,
        title: str = "Add tests",
        status: str = "pending",
        task_type: str = "spec",
        description: str = "",
    ) -> TaskSnapshot:
        return TaskSnapshot(
            title=title,
            status=status,
            type=task_type,
            description=description,
        )

    def _closed_pr(
        self,
        number: int = 5,
        title: str = "Prior attempt",
        body: str = "Old description.",
        close_reason: str = "scope creep",
    ) -> ClosedPR:
        return ClosedPR(
            number=number, title=title, body=body, close_reason=close_reason
        )

    # ── Active issue block ────────────────────────────────────────────────────

    def test_active_issue_number_and_title(self) -> None:
        result = render_active_context(self._issue(42, "Fix crash"), None, [], None, [])
        assert "## Active issue" in result
        assert "#42: Fix crash" in result

    def test_active_issue_body_included(self) -> None:
        result = render_active_context(
            self._issue(body="It panics on nil."), None, [], None, []
        )
        assert "It panics on nil." in result

    def test_active_issue_empty_body_omitted(self) -> None:
        result = render_active_context(
            ActiveIssue(number=1, title="T", body=""), None, [], None, []
        )
        # Header still present, but no blank-line separator before missing body
        assert "## Active issue" in result
        lines = result.split("\n")
        assert not any(
            line.strip() == "" and i > 0 and lines[i - 1] == ""
            for i, line in enumerate(lines)
            if i > 0
        )

    # ── Active PR block ───────────────────────────────────────────────────────

    def test_active_pr_number_and_title(self) -> None:
        result = render_active_context(
            self._issue(), self._pr(10, "Fix crash"), [], None, []
        )
        assert "## Active PR" in result
        assert "PR #10: Fix crash" in result

    def test_active_pr_url_included(self) -> None:
        result = render_active_context(
            self._issue(),
            self._pr(url="https://github.com/x/y/pull/10"),
            [],
            None,
            [],
        )
        assert "https://github.com/x/y/pull/10" in result

    def test_active_pr_body_included(self) -> None:
        result = render_active_context(
            self._issue(), self._pr(body="Adds the cache layer."), [], None, []
        )
        assert "Adds the cache layer." in result

    def test_active_pr_empty_body_omitted(self) -> None:
        pr = ActivePR(number=1, title="T", url="https://example.com", body="")
        result = render_active_context(self._issue(), pr, [], None, [])
        # URL still there; no spurious blank body
        assert "https://example.com" in result

    def test_active_pr_absent_when_none(self) -> None:
        result = render_active_context(self._issue(), None, [], None, [])
        assert "## Active PR" not in result

    # ── Prior attempts block ──────────────────────────────────────────────────

    def test_prior_attempts_header_present(self) -> None:
        result = render_active_context(
            self._issue(), None, [], None, [self._closed_pr()]
        )
        assert "## Prior attempts" in result

    def test_prior_attempt_number_and_title(self) -> None:
        result = render_active_context(
            self._issue(), None, [], None, [self._closed_pr(5, "Old attempt")]
        )
        assert "### PR #5: Old attempt" in result

    def test_prior_attempt_close_reason(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [self._closed_pr(close_reason="out of scope")],
        )
        assert "Close reason: out of scope" in result

    def test_prior_attempt_body_included(self) -> None:
        result = render_active_context(
            self._issue(), None, [], None, [self._closed_pr(body="Had a bug.")]
        )
        assert "Had a bug." in result

    def test_prior_attempts_absent_when_empty(self) -> None:
        result = render_active_context(self._issue(), None, [], None, [])
        assert "## Prior attempts" not in result

    def test_multiple_prior_attempts(self) -> None:
        attempts = [
            self._closed_pr(1, "First try"),
            self._closed_pr(2, "Second try"),
        ]
        result = render_active_context(self._issue(), None, [], None, attempts)
        assert "### PR #1: First try" in result
        assert "### PR #2: Second try" in result

    def test_prior_attempt_empty_close_reason_omitted(self) -> None:
        pr = ClosedPR(number=3, title="Old", body="body", close_reason="")
        result = render_active_context(self._issue(), None, [], None, [pr])
        assert "Close reason:" not in result

    # ── Tasks block ───────────────────────────────────────────────────────────

    def test_tasks_header_present(self) -> None:
        result = render_active_context(self._issue(), None, [self._task()], None, [])
        assert "## Tasks" in result

    def test_task_title_included(self) -> None:
        result = render_active_context(
            self._issue(), None, [self._task("Add caching")], None, []
        )
        assert "Add caching" in result

    def test_pending_task_marker(self) -> None:
        result = render_active_context(
            self._issue(), None, [self._task(status="pending")], None, []
        )
        assert "[ ]" in result

    def test_completed_task_marker(self) -> None:
        result = render_active_context(
            self._issue(), None, [self._task(status="completed")], None, []
        )
        assert "[x]" in result

    def test_in_progress_task_marker(self) -> None:
        result = render_active_context(
            self._issue(), None, [self._task(status="in_progress")], None, []
        )
        assert "[~]" in result

    def test_blocked_task_marker(self) -> None:
        result = render_active_context(
            self._issue(), None, [self._task(status="blocked")], None, []
        )
        assert "[!]" in result

    def test_task_type_included(self) -> None:
        result = render_active_context(
            self._issue(), None, [self._task(task_type="ci")], None, []
        )
        assert "[ci]" in result

    def test_tasks_absent_when_empty(self) -> None:
        result = render_active_context(self._issue(), None, [], None, [])
        assert "## Tasks" not in result

    def test_multiple_tasks_all_present(self) -> None:
        tasks = [
            self._task("Add tests", status="completed", task_type="spec"),
            self._task("Fix lint", status="in_progress", task_type="ci"),
            self._task("Update docs", status="pending", task_type="spec"),
        ]
        result = render_active_context(self._issue(), None, tasks, None, [])
        assert "Add tests" in result
        assert "Fix lint" in result
        assert "Update docs" in result

    # ── Right now block ───────────────────────────────────────────────────────

    def test_right_now_header_present(self) -> None:
        result = render_active_context(
            self._issue(), None, [], self._task("Write the test"), []
        )
        assert "## Right now" in result

    def test_right_now_title_included(self) -> None:
        result = render_active_context(
            self._issue(), None, [], self._task("Write the test"), []
        )
        assert "Write the test" in result

    def test_right_now_description_included(self) -> None:
        task = self._task(description="Use pytest fixtures.")
        result = render_active_context(self._issue(), None, [], task, [])
        assert "Use pytest fixtures." in result

    def test_right_now_empty_description_omitted(self) -> None:
        task = self._task(description="")
        result = render_active_context(self._issue(), None, [], task, [])
        assert "## Right now" in result
        # No trailing blank after header when no description
        after_header = result.split("## Right now")[1]
        assert after_header.strip() != ""

    def test_right_now_absent_when_none(self) -> None:
        result = render_active_context(self._issue(), None, [], None, [])
        assert "## Right now" not in result

    # ── Cache-stability: stable prefix is byte-identical across task changes ──

    def test_stable_prefix_unchanged_after_task_add(self) -> None:
        """Active issue + PR + prior attempts must be byte-identical before/after
        a task is added to the list — so the provider's prompt cache stays warm."""
        issue = self._issue()
        pr = self._pr()
        attempts = [self._closed_pr()]
        current = self._task("Do something")

        before = render_active_context(
            issue, pr, [self._task("Task A")], current, attempts
        )
        after = render_active_context(
            issue,
            pr,
            [self._task("Task A"), self._task("Task B")],
            current,
            attempts,
        )

        # Extract the stable prefix (everything before ## Tasks)
        assert before.split("## Tasks")[0] == after.split("## Tasks")[0]

    def test_stable_prefix_unchanged_after_task_complete(self) -> None:
        issue = self._issue()
        pr = self._pr()
        attempts = [self._closed_pr()]
        current = self._task("Remaining")

        before = render_active_context(
            issue,
            pr,
            [self._task("Done", status="pending"), self._task("Remaining")],
            current,
            attempts,
        )
        after = render_active_context(
            issue,
            pr,
            [self._task("Done", status="completed"), self._task("Remaining")],
            current,
            attempts,
        )

        assert before.split("## Tasks")[0] == after.split("## Tasks")[0]

    def test_tasks_section_changes_after_task_complete(self) -> None:
        issue = self._issue()
        before = render_active_context(
            issue, None, [self._task("A", status="pending")], None, []
        )
        after = render_active_context(
            issue, None, [self._task("A", status="completed")], None, []
        )
        assert "[ ]" in before
        assert "[x]" in after

    def test_right_now_changes_after_task_switch(self) -> None:
        issue = self._issue()
        before = render_active_context(issue, None, [], self._task("Old task"), [])
        after = render_active_context(issue, None, [], self._task("New task"), [])
        assert "Old task" in before
        assert "New task" in after
        assert "Old task" not in after

    # ── Closed sub-issues block ─────────────────────────────────────────────

    def _sub_issue(
        self,
        number: int = 100,
        title: str = "Sub-task A",
        body: str = "Handle the A case.",
        close_state: str = "merged",
        pr_number: int | None = 200,
        pr_repo: str | None = None,
        pr_body: str = "Implements sub-task A.",
        state_reason: str | None = None,
    ) -> ClosedSubIssue:
        return ClosedSubIssue(
            number=number,
            title=title,
            body=body,
            close_state=close_state,
            state_reason=state_reason,
            pr_number=pr_number,
            pr_repo=pr_repo,
            pr_body=pr_body,
        )

    def test_closed_sub_issues_header_present(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue()],
        )
        assert "## Closed sub-issues" in result

    def test_closed_sub_issues_absent_when_none(self) -> None:
        result = render_active_context(self._issue(), None, [], None, [])
        assert "## Closed sub-issues" not in result

    def test_closed_sub_issues_absent_when_empty(self) -> None:
        result = render_active_context(
            self._issue(), None, [], None, [], closed_sub_issues=[]
        )
        assert "## Closed sub-issues" not in result

    def test_closed_sub_issue_number_title_and_state(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(42, "Fix widget", close_state="merged")],
        )
        assert "### #42: Fix widget (merged)" in result

    def test_closed_sub_issue_linked_pr(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(pr_number=77)],
        )
        assert "Linked PR: #77" in result

    def test_closed_sub_issue_linked_pr_same_repo(self) -> None:
        """When pr_repo matches parent_repo, renders as bare #N (no owner/repo prefix)."""
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(pr_number=77, pr_repo="owner/proj")],
            parent_repo="owner/proj",
        )
        assert "Linked PR: #77" in result
        assert "owner/proj#77" not in result

    def test_closed_sub_issue_linked_pr_cross_repo(self) -> None:
        """When pr_repo differs from parent_repo, renders as owner/repo#N."""
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(pr_number=77, pr_repo="other/repo")],
            parent_repo="owner/proj",
        )
        assert "Linked PR: other/repo#77" in result

    def test_closed_sub_issue_linked_pr_cross_repo_no_parent(self) -> None:
        """When parent_repo is None (not provided), pr_repo is shown if set."""
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(pr_number=77, pr_repo="other/repo")],
        )
        assert "Linked PR: other/repo#77" in result

    def test_closed_sub_issue_no_linked_pr(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[
                self._sub_issue(close_state="closed_no_pr", pr_number=None)
            ],
        )
        assert "Linked PR:" not in result

    def test_closed_sub_issue_body_included(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(body="Details here.")],
        )
        assert "Details here." in result

    def test_closed_sub_issue_empty_body_omitted(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(body="")],
        )
        # PR body still present, but no spurious blank for missing issue body
        lines = result.split("## Closed sub-issues")[1]
        assert "\n\n\n" not in lines  # no triple newline from missing body

    def test_closed_sub_issue_pr_body_included(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(pr_body="Added the handler.")],
        )
        assert "PR description: Added the handler." in result

    def test_closed_sub_issue_empty_pr_body_omitted(self) -> None:
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[self._sub_issue(pr_body="")],
        )
        assert "PR description:" not in result

    def test_multiple_closed_sub_issues(self) -> None:
        subs = [
            self._sub_issue(10, "First sub"),
            self._sub_issue(20, "Second sub"),
        ]
        result = render_active_context(
            self._issue(), None, [], None, [], closed_sub_issues=subs
        )
        assert "### #10: First sub" in result
        assert "### #20: Second sub" in result

    def test_closed_sub_issue_close_states(self) -> None:
        subs = [
            self._sub_issue(1, "Merged one", close_state="merged"),
            self._sub_issue(2, "Closed unmerged", close_state="closed_unmerged"),
            self._sub_issue(
                3, "No PR", close_state="closed_no_pr", pr_number=None, pr_body=""
            ),
        ]
        result = render_active_context(
            self._issue(), None, [], None, [], closed_sub_issues=subs
        )
        assert "(merged)" in result
        assert "(closed_unmerged)" in result
        assert "(closed_no_pr)" in result

    def test_closed_sub_issue_state_reason_shown_for_no_pr(self) -> None:
        """state_reason is appended to close_state for closed_no_pr sub-issues."""
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[
                self._sub_issue(
                    3,
                    "Deferred task",
                    close_state="closed_no_pr",
                    pr_number=None,
                    pr_body="",
                    state_reason="not_planned",
                )
            ],
        )
        assert "(closed_no_pr (not_planned))" in result

    def test_closed_sub_issue_state_reason_omitted_when_none(self) -> None:
        """When state_reason is None the label is just the close_state."""
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[
                self._sub_issue(
                    4,
                    "Finished task",
                    close_state="closed_no_pr",
                    pr_number=None,
                    pr_body="",
                    state_reason=None,
                )
            ],
        )
        assert "(closed_no_pr)" in result
        assert "(closed_no_pr (None))" not in result

    def test_closed_sub_issue_state_reason_not_shown_for_merged(self) -> None:
        """state_reason is not appended when close_state is merged (has a PR)."""
        result = render_active_context(
            self._issue(),
            None,
            [],
            None,
            [],
            closed_sub_issues=[
                self._sub_issue(
                    5,
                    "Done via PR",
                    close_state="merged",
                    pr_number=50,
                    pr_body="",
                    state_reason="completed",
                )
            ],
        )
        assert "(merged)" in result
        assert "(merged (completed))" not in result

    def test_closed_sub_issues_in_stable_prefix(self) -> None:
        """Closed sub-issues are stable context — they don't change during a session."""
        subs = [self._sub_issue()]
        before = render_active_context(
            self._issue(),
            None,
            [self._task("Task A")],
            None,
            [],
            closed_sub_issues=subs,
        )
        after = render_active_context(
            self._issue(),
            None,
            [self._task("Task A"), self._task("Task B")],
            None,
            [],
            closed_sub_issues=subs,
        )
        assert before.split("## Tasks")[0] == after.split("## Tasks")[0]

    # ── Section order ─────────────────────────────────────────────────────────

    def test_section_order_stable_prefix_before_dynamic(self) -> None:
        """Stable prefix (issue, PR, attempts, sub-issues) must appear before Tasks / Right now."""
        result = render_active_context(
            self._issue(),
            self._pr(),
            [self._task()],
            self._task("current"),
            [self._closed_pr()],
            closed_sub_issues=[self._sub_issue()],
        )
        issue_pos = result.index("## Active issue")
        pr_pos = result.index("## Active PR")
        attempts_pos = result.index("## Prior attempts")
        subs_pos = result.index("## Closed sub-issues")
        tasks_pos = result.index("## Tasks")
        now_pos = result.index("## Right now")

        assert issue_pos < pr_pos < attempts_pos < subs_pos < tasks_pos < now_pos

    # ── Returns a string ──────────────────────────────────────────────────────

    def test_returns_string(self) -> None:
        result = render_active_context(self._issue(), None, [], None, [])
        assert isinstance(result, str)

    def test_minimal_call_issue_only(self) -> None:
        """Minimal valid call: only the required issue argument is meaningful."""
        result = render_active_context(
            ActiveIssue(number=1, title="T", body=""), None, [], None, []
        )
        assert "## Active issue" in result
        assert "#1: T" in result


# ── Prompts.synthesis_system_prompt ──────────────────────────────────────────


class TestSynthesisSystemPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").synthesis_system_prompt()
        assert "I am Fido." in result

    def test_includes_triage_clause(self) -> None:
        result = Prompts("persona").synthesis_system_prompt()
        assert TRIAGE_CLAUSE in result

    def test_json_output_instruction(self) -> None:
        result = Prompts("persona").synthesis_system_prompt()
        assert "JSON" in result
        assert "ONLY" in result

    def test_no_active_context_when_issue_is_none(self) -> None:
        result = Prompts("persona").synthesis_system_prompt()
        assert "## Active issue" not in result

    def test_active_context_included_when_issue_provided(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="It crashes.")
        result = Prompts("persona").synthesis_system_prompt(issue=issue)
        assert "## Active issue" in result
        assert "Fix crash" in result
        assert "It crashes." in result

    def test_active_context_includes_pr_when_provided(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="")
        pr = ActivePR(
            number=42,
            title="Fix crash PR",
            url="https://github.com/a/b/pull/42",
            body="",
        )
        result = Prompts("persona").synthesis_system_prompt(issue=issue, pr=pr)
        assert "## Active PR" in result
        assert "Fix crash PR" in result

    def test_active_context_no_pr_section_when_pr_is_none(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="")
        result = Prompts("persona").synthesis_system_prompt(issue=issue, pr=None)
        assert "## Active PR" not in result

    def test_empty_persona(self) -> None:
        result = Prompts("").synthesis_system_prompt()
        assert "JSON" in result


# ── Prompts.synthesis_followup_system_prompt ─────────────────────────────────


class TestSynthesisFollowupSystemPrompt:
    def test_includes_persona(self) -> None:
        result = Prompts("I am Fido.").synthesis_followup_system_prompt()
        assert "I am Fido." in result

    def test_drops_json_only_directive(self) -> None:
        """#1850 codex P1: the follow-up turn asks for a plain Yes/No
        answer.  Reusing the JSON-only directive from the main synthesis
        prompt would push the model toward wrapping its answer in JSON,
        and ``startswith("no")`` would then fail on a ``{...`` reply."""
        result = Prompts("persona").synthesis_followup_system_prompt()
        # The main synthesis prompt says "Output ONLY the JSON object";
        # the follow-up must NOT carry that directive.
        assert "Output ONLY the JSON" not in result
        assert "structured JSON response" not in result
        # And it must explicitly steer the model away from JSON output.
        assert "no JSON" in result

    def test_active_context_included_when_issue_given(self) -> None:
        """Same active-context anchoring as the main synthesis prompt —
        the follow-up still needs the active issue/PR to reason about
        what was queued."""
        issue = ActiveIssue(number=7, title="Fix crash", body="some body")
        result = Prompts("persona").synthesis_followup_system_prompt(issue=issue)
        assert "## Active issue" in result
        assert "#7: Fix crash" in result

    def test_active_context_no_pr_section_when_pr_is_none(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="")
        result = Prompts("persona").synthesis_followup_system_prompt(
            issue=issue, pr=None
        )
        assert "## Active PR" not in result

    def test_active_context_includes_pr_when_given(self) -> None:
        issue = ActiveIssue(number=7, title="Fix crash", body="")
        pr = ActivePR(
            number=42, title="Fix the crash", body="", url="https://example/42"
        )
        result = Prompts("persona").synthesis_followup_system_prompt(issue=issue, pr=pr)
        assert "## Active PR" in result
        assert "PR #42: Fix the crash" in result


# ── Prompts.synthesis_prompt ─────────────────────────────────────────────────


class TestSynthesisPrompt:
    def test_includes_comment(self) -> None:
        result = Prompts("").synthesis_prompt("please fix the bug", is_bot=False)
        assert "please fix the bug" in result

    def test_includes_json_schema(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "reasoning" in result
        assert "reply_text" in result
        assert "emoji" in result
        assert "change_request" in result
        assert "insights" in result

    def test_no_actions_list_in_schema(self) -> None:
        # Flat schema — no actions array, no action type objects.
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert '"actions"' not in result
        assert "add_reaction" not in result
        assert "rescope_intent" not in result
        assert "no_op" not in result

    def test_includes_valid_emoji_shortcodes(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "rocket" in result
        assert "heart" in result
        assert "eyes" in result

    def test_reply_text_required_constraint(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "REQUIRED" in result
        assert "non-empty" in result

    def test_bot_note_present_when_is_bot_true(self) -> None:
        result = Prompts("").synthesis_prompt("suggestion", is_bot=True)
        assert "automated tool" in result

    def test_bot_note_absent_when_is_bot_false(self) -> None:
        result = Prompts("").synthesis_prompt("suggestion", is_bot=False)
        assert "automated tool" not in result

    def test_includes_context_when_provided(self) -> None:
        result = Prompts("").synthesis_prompt(
            "comment", is_bot=False, context={"pr_title": "My PR"}
        )
        assert "PR: My PR" in result

    def test_no_context_still_works(self) -> None:
        result = Prompts("").synthesis_prompt("hello", is_bot=False, context=None)
        assert "hello" in result

    def test_voice_guidelines_present(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "Take a position" in result
        assert "Disagree" in result

    def test_disagree_defers_after_one_pushback(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "already pushed back" in result
        assert "defer" in result

    def test_change_request_description(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "collaborator" in result
        assert "scope or tasks" in result

    def test_json_only_instruction(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "ONLY the JSON" in result

    def test_insights_schema_includes_subfields(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "title" in result
        assert "hook" in result
        assert "why" in result

    def test_insights_instructs_when_to_populate(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "worth pausing over" in result

    def test_insights_empty_array_when_nothing_stood_out(self) -> None:
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "Empty array" in result or "empty array" in result or "Empty" in result

    def test_no_promise_without_change_request_constraint(self) -> None:
        # The prompt must forbid future-tense commitments when change_request
        # is null, so the LLM doesn't produce "I'll fix this" with no task.
        result = Prompts("").synthesis_prompt("comment", is_bot=False)
        assert "change_request" in result
        assert "future" in result.lower() or "I'll" in result or "I will" in result


# ─── HOL-15 / #1909 — intent-coverage critic prompt ────────────────────────


class TestIntentCoverageCriticPrompt:
    """Layer 2 critic prompt that gates the synthesis turn.  Wraps the
    comment + reply + change_request into a question Opus answers as
    ``{"passed": true}`` or ``{"passed": false, "gap": "..."}``."""

    def test_includes_comment_and_reply(self) -> None:
        result = Prompts("").intent_coverage_critic_prompt(
            comment_body="Please fix the test.",
            reply_text="I'll add the missing test case.",
            change_request="Add the missing test case",
        )
        assert "Please fix the test." in result
        assert "I'll add the missing test case." in result
        assert "Add the missing test case" in result

    def test_renders_none_change_request_as_label(self) -> None:
        """When no work is queued, the prompt must render a clear
        sentinel rather than the bare string ``None`` so Opus can
        reason about the no-queue case."""
        result = Prompts("").intent_coverage_critic_prompt(
            comment_body="just thinking out loud",
            reply_text="Got it.",
            change_request=None,
        )
        assert "(none — no work queued)" in result
        assert "None" not in result.split("REGISTERED CHANGE REQUEST")[1].split(
            "QUESTION"
        )[0].replace("(none — no work queued)", "")

    def test_empty_change_request_treated_as_none(self) -> None:
        """Empty string and whitespace-only change_request render the
        same as None — they carry no actionable scope."""
        result = Prompts("").intent_coverage_critic_prompt(
            comment_body="x",
            reply_text="y",
            change_request="   \t",
        )
        assert "(none — no work queued)" in result

    def test_question_distinguishes_three_failure_axes(self) -> None:
        """The pass rule must enumerate ``missing``/``invented``/
        ``mismatched`` so Opus knows which axes to check.  The bare
        Yes/No (#1218) only caught missing — HOL-15 is the broader
        gate."""
        result = Prompts("").intent_coverage_critic_prompt(
            comment_body="x", reply_text="y", change_request=None
        )
        assert "missing" in result
        assert "invented" in result
        assert "mismatched" in result

    def test_response_schema_is_passed_gap_envelope(self) -> None:
        """Output schema must be the fixed ``{"passed": bool}`` /
        ``{"passed": false, "gap": "..."}`` envelope so the Python
        parser has one shape to handle."""
        result = Prompts("").intent_coverage_critic_prompt(
            comment_body="x", reply_text="y", change_request=None
        )
        assert '"passed": true' in result
        assert '"passed": false' in result
        assert '"gap"' in result
        # JSON-only directive — no preamble/postamble lets parser pick
        # the verdict cleanly from the response.
        assert "No other text" in result


# ─── HOL-16 / #1910 — task-creation critic prompt ──────────────────────────


class TestTaskCreationCriticPrompt:
    """Layer 2 critic that gates each proposed ``new`` op against the
    queue along two axes: relationship (distinct/duplicate_of/supersedes)
    and scope (single/multi)."""

    def _proposed(self) -> dict[str, Any]:
        return {
            "title": "Add retry logic",
            "description": "Retry on transient failures.",
            "invariant": "transient failures retry up to 3 times",
        }

    def _queue(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "t-1",
                "title": "Add CI for the new module",
                "description": "Wire workflow.",
                "invariant": "ci runs on every PR",
            },
            {
                "id": "t-2",
                "title": "Document the retry policy",
                "description": "",
                "invariant": "",
            },
        ]

    def test_includes_proposed_task_fields(self) -> None:
        result = Prompts("").task_creation_critic_prompt(
            self._proposed(), self._queue()
        )
        assert "Add retry logic" in result
        assert "Retry on transient failures." in result
        assert "transient failures retry up to 3 times" in result

    def test_includes_each_queue_entry_with_id(self) -> None:
        result = Prompts("").task_creation_critic_prompt(
            self._proposed(), self._queue()
        )
        assert "id=t-1" in result
        assert "Add CI for the new module" in result
        assert "id=t-2" in result
        assert "Document the retry policy" in result

    def test_renders_queue_invariants_when_present(self) -> None:
        """When a queue entry has an invariant, render it — Opus
        needs the invariant to compare scope for duplicate/supersede
        verdicts."""
        result = Prompts("").task_creation_critic_prompt(
            self._proposed(), self._queue()
        )
        assert "ci runs on every PR" in result

    def test_empty_queue_renders_explicit_sentinel(self) -> None:
        """The empty-queue case is distinct from "many tasks" — render
        a clear marker so Opus doesn't mistake silence for context."""
        result = Prompts("").task_creation_critic_prompt(self._proposed(), [])
        assert "queue is empty" in result

    def test_relationship_axis_documented(self) -> None:
        """Opus must see all three relationship values explained, not
        just listed in the schema."""
        result = Prompts("").task_creation_critic_prompt(
            self._proposed(), self._queue()
        )
        assert '"distinct"' in result
        assert '"duplicate_of"' in result
        assert '"supersedes"' in result
        # The follow-up actions on duplicate/supersedes (drop/replace)
        # need to be discoverable from the prompt.
        assert "duplicate_of_id" in result
        assert "supersedes_id" in result

    def test_scope_axis_documented_with_hol12_link(self) -> None:
        """The ``scope`` axis is the HOL-12 invariant carrier — call
        out the link so Opus reasons about single-invariant scoping."""
        result = Prompts("").task_creation_critic_prompt(
            self._proposed(), self._queue()
        )
        assert '"single"' in result
        assert '"multi"' in result
        assert "HOL-12" in result
        assert "proposed_splits" in result

    def test_response_schema_is_verdict_envelope(self) -> None:
        """Output schema must include every field downstream parsing
        reads — relationship/scope/duplicate_of_id/supersedes_id/
        proposed_splits/rationale."""
        result = Prompts("").task_creation_critic_prompt(
            self._proposed(), self._queue()
        )
        assert '"relationship":' in result
        assert '"scope":' in result
        assert '"duplicate_of_id":' in result
        assert '"supersedes_id":' in result
        assert '"proposed_splits":' in result
        assert '"rationale":' in result
        assert "No other text" in result


# ─── HOL-17 / #1911 — task-completion critic prompt ────────────────────────


class TestTaskCompletionCriticPrompt:
    """Layer 2 critic that gates ``commit-task-complete`` against the
    just-landed diff and the task's named invariant."""

    def test_includes_invariant_description_diff(self) -> None:
        result = Prompts("").task_completion_critic_prompt(
            task_invariant="transient failures retry up to 3 times",
            task_description="Wire retry on the network layer.",
            diff="diff --git a/x b/x\n+changed\n",
        )
        assert "transient failures retry up to 3 times" in result
        assert "Wire retry on the network layer." in result
        assert "+changed" in result

    def test_empty_invariant_renders_sentinel(self) -> None:
        """Tasks predating HOL-11 don't have an invariant; render a
        clear marker so Opus knows to reason about scope-creep without
        an anchor (but the establishes-axis can still catch empty
        diffs)."""
        result = Prompts("").task_completion_critic_prompt(
            task_invariant="",
            task_description="Legacy task body.",
            diff="diff --git a/x b/x\n+ok\n",
        )
        assert "no invariant declared" in result

    def test_empty_diff_renders_sentinel(self) -> None:
        """PR #1858's empty-commit pattern: the prompt must render
        ``(empty diff)`` distinct from "missing field" so Opus reads it
        as a clear flag for the establishes axis."""
        result = Prompts("").task_completion_critic_prompt(
            task_invariant="x",
            task_description="",
            diff="",
        )
        assert "(empty diff)" in result

    def test_both_axes_documented(self) -> None:
        """The pass rule must enumerate ``establishes`` and ``only``
        so Opus reasons about both."""
        result = Prompts("").task_completion_critic_prompt(
            task_invariant="x",
            task_description="",
            diff="d",
        )
        assert "establishes" in result
        assert "only" in result
        # Pattern reference: PR #1858 is the empty-commit prior art
        # this critic exists to catch.
        assert "PR #1858" in result

    def test_response_schema_is_verdict_envelope(self) -> None:
        result = Prompts("").task_completion_critic_prompt(
            task_invariant="x",
            task_description="",
            diff="d",
        )
        assert '"passed": true' in result
        assert '"passed": false' in result
        assert '"gap":' in result
        assert '"rationale":' in result
        assert "No other text" in result
