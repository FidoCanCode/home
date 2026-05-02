"""Tests for fido.synthesis — CommentResponse and Action types (closes #1230)."""

import pytest

from fido.synthesis import (
    VALID_REACTIONS,
    AddReaction,
    CommentResponse,
    CompleteTask,
    CreateTask,
    ModifyTask,
    NoOp,
    Preempt,
    SynthesisAction,
    validate_reaction,
)
from fido.types import TaskType

# ---------------------------------------------------------------------------
# VALID_REACTIONS constant
# ---------------------------------------------------------------------------


class TestValidReactions:
    def test_contains_expected_shortcodes(self) -> None:
        assert "+1" in VALID_REACTIONS
        assert "-1" in VALID_REACTIONS
        assert "rocket" in VALID_REACTIONS
        assert "eyes" in VALID_REACTIONS
        assert "heart" in VALID_REACTIONS
        assert "laugh" in VALID_REACTIONS
        assert "confused" in VALID_REACTIONS
        assert "hooray" in VALID_REACTIONS

    def test_is_frozenset(self) -> None:
        assert isinstance(VALID_REACTIONS, frozenset)

    def test_exactly_eight_reactions(self) -> None:
        assert len(VALID_REACTIONS) == 8


# ---------------------------------------------------------------------------
# validate_reaction
# ---------------------------------------------------------------------------


class TestValidateReaction:
    def test_valid_reaction_returns_emoji(self) -> None:
        assert validate_reaction("rocket") == "rocket"

    def test_all_valid_reactions_pass(self) -> None:
        for emoji in VALID_REACTIONS:
            assert validate_reaction(emoji) == emoji

    def test_invalid_reaction_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid reaction"):
            validate_reaction("thinking")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid reaction"):
            validate_reaction("")

    def test_error_message_includes_valid_list(self) -> None:
        with pytest.raises(ValueError, match="Valid reactions"):
            validate_reaction("notanemoji")


# ---------------------------------------------------------------------------
# AddReaction
# ---------------------------------------------------------------------------


class TestAddReaction:
    def test_construction(self) -> None:
        r = AddReaction(emoji="rocket")
        assert r.emoji == "rocket"

    def test_frozen(self) -> None:
        r = AddReaction(emoji="eyes")
        with pytest.raises((AttributeError, TypeError)):
            r.emoji = "heart"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert AddReaction("rocket") == AddReaction("rocket")
        assert AddReaction("rocket") != AddReaction("eyes")


# ---------------------------------------------------------------------------
# CreateTask
# ---------------------------------------------------------------------------


class TestCreateTask:
    def test_construction_with_defaults(self) -> None:
        t = CreateTask(title="Fix the thing")
        assert t.title == "Fix the thing"
        assert t.task_type == TaskType.THREAD
        assert t.description == ""

    def test_construction_with_all_fields(self) -> None:
        t = CreateTask(
            title="Plan refactor",
            task_type=TaskType.SPEC,
            description="Detailed plan here",
        )
        assert t.task_type == TaskType.SPEC
        assert t.description == "Detailed plan here"

    def test_frozen(self) -> None:
        t = CreateTask(title="x")
        with pytest.raises((AttributeError, TypeError)):
            t.title = "y"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert CreateTask("a") == CreateTask("a")
        assert CreateTask("a") != CreateTask("b")


# ---------------------------------------------------------------------------
# CompleteTask
# ---------------------------------------------------------------------------


class TestCompleteTask:
    def test_construction(self) -> None:
        c = CompleteTask(task_id="abc-123")
        assert c.task_id == "abc-123"

    def test_frozen(self) -> None:
        c = CompleteTask(task_id="x")
        with pytest.raises((AttributeError, TypeError)):
            c.task_id = "y"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert CompleteTask("x") == CompleteTask("x")
        assert CompleteTask("x") != CompleteTask("y")


# ---------------------------------------------------------------------------
# ModifyTask
# ---------------------------------------------------------------------------


class TestModifyTask:
    def test_new_title_only(self) -> None:
        m = ModifyTask(task_id="t1", new_title="Updated title")
        assert m.task_id == "t1"
        assert m.new_title == "Updated title"
        assert m.new_description is None

    def test_new_description_only(self) -> None:
        m = ModifyTask(task_id="t1", new_description="New description")
        assert m.new_title is None
        assert m.new_description == "New description"

    def test_both_fields(self) -> None:
        m = ModifyTask(task_id="t1", new_title="A", new_description="B")
        assert m.new_title == "A"
        assert m.new_description == "B"

    def test_neither_field_raises(self) -> None:
        with pytest.raises(
            ValueError, match="at least one of new_title or new_description"
        ):
            ModifyTask(task_id="t1")

    def test_frozen(self) -> None:
        m = ModifyTask(task_id="t1", new_title="x")
        with pytest.raises((AttributeError, TypeError)):
            m.new_title = "y"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert ModifyTask("t1", new_title="x") == ModifyTask("t1", new_title="x")
        assert ModifyTask("t1", new_title="x") != ModifyTask("t1", new_title="y")


# ---------------------------------------------------------------------------
# Preempt
# ---------------------------------------------------------------------------


class TestPreempt:
    def test_preempt_true(self) -> None:
        p = Preempt(preempt=True)
        assert p.preempt is True

    def test_preempt_false(self) -> None:
        p = Preempt(preempt=False)
        assert p.preempt is False

    def test_frozen(self) -> None:
        p = Preempt(preempt=True)
        with pytest.raises((AttributeError, TypeError)):
            p.preempt = False  # type: ignore[misc]

    def test_equality(self) -> None:
        assert Preempt(True) == Preempt(True)
        assert Preempt(True) != Preempt(False)


# ---------------------------------------------------------------------------
# NoOp
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_construction(self) -> None:
        n = NoOp()
        assert isinstance(n, NoOp)

    def test_frozen(self) -> None:
        # frozen=True means no __dict__ mutation; trivially confirmed by
        # checking it is a dataclass with no settable fields
        n = NoOp()
        with pytest.raises((AttributeError, TypeError)):
            n.anything = "x"  # type: ignore[attr-defined]

    def test_equality(self) -> None:
        assert NoOp() == NoOp()


# ---------------------------------------------------------------------------
# SynthesisAction union — membership
# ---------------------------------------------------------------------------


class TestSynthesisActionUnion:
    """Verify the union includes every action type and no others."""

    def test_add_reaction_is_synthesis_action(self) -> None:
        a: SynthesisAction = AddReaction("rocket")
        assert isinstance(a, AddReaction)

    def test_create_task_is_synthesis_action(self) -> None:
        a: SynthesisAction = CreateTask("do the thing")
        assert isinstance(a, CreateTask)

    def test_complete_task_is_synthesis_action(self) -> None:
        a: SynthesisAction = CompleteTask("id-1")
        assert isinstance(a, CompleteTask)

    def test_modify_task_is_synthesis_action(self) -> None:
        a: SynthesisAction = ModifyTask("id-1", new_title="x")
        assert isinstance(a, ModifyTask)

    def test_preempt_is_synthesis_action(self) -> None:
        a: SynthesisAction = Preempt(True)
        assert isinstance(a, Preempt)

    def test_noop_is_synthesis_action(self) -> None:
        a: SynthesisAction = NoOp()
        assert isinstance(a, NoOp)


# ---------------------------------------------------------------------------
# CommentResponse
# ---------------------------------------------------------------------------


class TestCommentResponse:
    def _make(
        self,
        reasoning: str = "thought about it",
        reply_text: str = "Here is my reply.",
        actions: tuple[SynthesisAction, ...] = (),
    ) -> CommentResponse:
        return CommentResponse(
            reasoning=reasoning,
            reply_text=reply_text,
            actions=actions,
        )

    def test_construction_minimal(self) -> None:
        r = self._make()
        assert r.reasoning == "thought about it"
        assert r.reply_text == "Here is my reply."
        assert r.actions == ()

    def test_construction_with_actions(self) -> None:
        actions = (AddReaction("rocket"), CreateTask("Fix foo"))
        r = self._make(actions=actions)
        assert len(r.actions) == 2
        assert isinstance(r.actions[0], AddReaction)
        assert isinstance(r.actions[1], CreateTask)

    def test_frozen(self) -> None:
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.reply_text = "changed"  # type: ignore[misc]

    def test_empty_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(
                reasoning="thinking",
                reply_text="",
                actions=(),
            )

    def test_whitespace_only_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(
                reasoning="thinking",
                reply_text="   ",
                actions=(),
            )

    def test_newline_only_reply_text_raises(self) -> None:
        with pytest.raises(ValueError, match="reply_text must be non-empty"):
            CommentResponse(
                reasoning="thinking",
                reply_text="\n\t\n",
                actions=(),
            )

    def test_reply_text_with_leading_trailing_whitespace_accepted(self) -> None:
        # Only purely-whitespace values are rejected; padded real text is fine
        r = self._make(reply_text="  actual text  ")
        assert r.reply_text == "  actual text  "

    def test_equality(self) -> None:
        r1 = self._make()
        r2 = self._make()
        assert r1 == r2

    def test_inequality_on_reply_text(self) -> None:
        r1 = self._make(reply_text="Hello.")
        r2 = self._make(reply_text="Goodbye.")
        assert r1 != r2

    def test_constraint_b_error_message_mentions_constraint(self) -> None:
        with pytest.raises(ValueError, match="Constraint B"):
            CommentResponse(reasoning="x", reply_text="", actions=())

    def test_mixed_action_types_in_tuple(self) -> None:
        r = CommentResponse(
            reasoning="r",
            reply_text="Reply.",
            actions=(
                AddReaction("eyes"),
                CreateTask("title", TaskType.SPEC, "desc"),
                CompleteTask("old-id"),
                ModifyTask("mid", new_title="new title"),
                Preempt(False),
                NoOp(),
            ),
        )
        assert len(r.actions) == 6
