"""Tests for fido.cli — add/complete/list subcommands."""

import json
import logging
from pathlib import Path

import pytest

from fido.cli import Cmd, build_parser, main
from fido.infra import RealProcessRunner
from fido.tasks import Tasks
from fido.types import TaskType

# ── helpers ───────────────────────────────────────────────────────────────────


def _task_file(tmp_path: Path) -> Path:
    git_dir = tmp_path / ".git" / "fido"
    git_dir.mkdir(parents=True)
    return git_dir / "tasks.json"


class _FakeCallRecorder:
    """Typed callable that records every invocation."""

    def __init__(self, return_value: object = None) -> None:
        self.return_value: object = return_value
        self._side_effect: object = None
        self._calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    @property
    def side_effect(self) -> object:
        return self._side_effect

    @side_effect.setter
    def side_effect(self, value: object) -> None:
        self._side_effect = value

    def __call__(self, *args: object, **kwargs: object) -> object:
        self._calls.append((args, kwargs))
        se = self._side_effect
        if se is not None:
            if isinstance(se, BaseException):
                raise se
            if callable(se):
                return se(*args, **kwargs)
        return self.return_value

    def assert_not_called(self) -> None:
        assert not self._calls, f"expected no calls, got {len(self._calls)}"

    def assert_called_once_with(self, *args: object, **kwargs: object) -> None:
        assert len(self._calls) == 1, f"expected 1 call, got {len(self._calls)}"
        actual_args, actual_kwargs = self._calls[0]
        assert actual_args == args and actual_kwargs == kwargs, (
            f"expected {args!r} / {kwargs!r}, got {actual_args!r} / {actual_kwargs!r}"
        )


class _FakeGitHub:
    """Minimal GitHub fake for CLI tests.

    Only exposes the methods that ``Cmd.complete`` / ``complete_with_resolve``
    actually calls: ``get_user``, ``get_review_threads``, and
    ``resolve_thread``.  ``get_pull_comments`` is included because some tests
    set it up (legacy of an earlier implementation path) even though the
    current oracle path does not call it.

    Accepts the same constructor signature as :class:`~fido.github.GitHub`
    so it can be passed as ``_GitHub=_FakeGitHub`` to :func:`~fido.cli.main`.
    """

    def __init__(
        self,
        token: str | None = None,
        session: object = None,
        *,
        runner: object = None,
        clock: object = None,
        token_fetcher: object = None,
    ) -> None:
        self.get_user: _FakeCallRecorder = _FakeCallRecorder(return_value="")
        self.get_pull_comments: _FakeCallRecorder = _FakeCallRecorder(return_value=[])
        self.get_review_threads: _FakeCallRecorder = _FakeCallRecorder(return_value=[])
        self.resolve_thread: _FakeCallRecorder = _FakeCallRecorder()


def _cmd(github: "_FakeGitHub | object") -> Cmd:
    """Construct a ``Cmd`` with the given github fake and a noop runner."""
    return Cmd(github=github, runner=RealProcessRunner())  # type: ignore[arg-type]


class _FakeArgs:
    """Minimal argparse.Namespace substitute."""

    def __init__(self, command: str) -> None:
        self.command = command


class _FakeParser:
    """Minimal argparse.ArgumentParser substitute.

    Returns a fixed :class:`_FakeArgs` from :meth:`parse_args` so tests can
    drive the ``match args.command`` branch without going through the real
    parser.
    """

    def __init__(self, fake_args: _FakeArgs) -> None:
        self._fake_args = fake_args

    def parse_args(self, argv: object) -> _FakeArgs:
        return self._fake_args


# ── build_parser ──────────────────────────────────────────────────────────────


class TestBuildParser:
    def test_add_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "spec", "my task"])
        assert args.command == "add"
        assert args.task_type == TaskType.SPEC
        assert args.title == "my task"
        assert args.description == ""

    def test_add_with_description(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "ci", "title", "desc"])
        assert args.task_type == TaskType.CI
        assert args.description == "desc"

    def test_add_with_comment_id(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                str(tmp_path),
                "add",
                "thread",
                "my task",
                "--comment-id",
                "42",
                "--repo",
                "a/b",
                "--pr",
                "7",
            ]
        )
        assert args.task_type == TaskType.THREAD
        assert args.comment_id == 42
        assert args.repo == "a/b"
        assert args.pr == 7

    def test_add_without_comment_id_defaults_none(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "add", "spec", "my task"])
        assert args.comment_id is None
        assert args.repo is None
        assert args.pr is None

    def test_complete_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "complete", "task-id-123"])
        assert args.command == "complete"
        assert args.task_id == "task-id-123"

    def test_list_subcommand(self, tmp_path: Path) -> None:
        parser = build_parser()
        args = parser.parse_args([str(tmp_path), "list"])
        assert args.command == "list"

    def test_no_command_exits(self, tmp_path: Path) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([str(tmp_path)])


# ── Cmd.add ───────────────────────────────────────────────────────────────────


class TestCmdAdd:
    def test_adds_task(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).add(  # type: ignore[arg-type]
            tmp_path, TaskType.SPEC, "my task", "some description"
        )
        capsys.readouterr()  # consume add output

        tasks = Tasks(tmp_path).list()
        assert len(tasks) == 1
        assert tasks[0]["title"] == "my task"
        assert tasks[0]["description"] == "some description"
        assert tasks[0]["type"] == "spec"

    def test_adds_task_no_description(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).add(tmp_path, TaskType.CI, "bare task", "")  # type: ignore[arg-type]
        capsys.readouterr()

        tasks = Tasks(tmp_path).list()
        assert tasks[0]["description"] == ""
        assert tasks[0]["type"] == "ci"

    def test_adds_task_with_comment_id_builds_thread(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).add(  # type: ignore[arg-type]
            tmp_path, TaskType.THREAD, "threaded", "", comment_id=42, repo="a/b", pr=7
        )
        capsys.readouterr()

        tasks = Tasks(tmp_path).list()
        assert tasks[0]["thread"] == {"comment_id": 42, "repo": "a/b", "pr": 7}

    def test_adds_task_comment_id_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """comment_id without repo/pr still sets a thread for dedup purposes."""
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).add(  # type: ignore[arg-type]
            tmp_path, TaskType.THREAD, "threaded", "", comment_id=99
        )
        capsys.readouterr()

        tasks = Tasks(tmp_path).list()
        assert tasks[0]["thread"] == {"comment_id": 99}

    def test_add_deduplicates_by_comment_id(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).add(  # type: ignore[arg-type]
            tmp_path,
            TaskType.THREAD,
            "first title",
            "",
            comment_id=42,
            repo="a/b",
            pr=7,
        )
        capsys.readouterr()
        _cmd(_FakeGitHub()).add(  # type: ignore[arg-type]
            tmp_path,
            TaskType.THREAD,
            "different title",
            "",
            comment_id=42,
            repo="a/b",
            pr=7,
        )
        capsys.readouterr()

        tasks = Tasks(tmp_path).list()
        assert len(tasks) == 1
        assert tasks[0]["title"] == "first title"

    def test_add_prints_task_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).add(tmp_path, TaskType.SPEC, "my task", "")  # type: ignore[arg-type]
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["title"] == "my task"
        assert "id" in data


# ── Cmd.complete ──────────────────────────────────────────────────────────────


class TestCmdComplete:
    def test_completes_task_no_thread(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        cmd = _cmd(_FakeGitHub())  # type: ignore[arg-type]
        task = cmd.add(tmp_path, TaskType.SPEC, "task to finish", "")
        capsys.readouterr()
        cmd.complete(tmp_path, task["id"])

        assert Tasks(tmp_path).list()[0]["status"] == "completed"

    def test_completes_task_with_thread_resolves(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _task_file(tmp_path)

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(tmp_path).add(
            title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = _FakeGitHub()
        mock_github.get_user.return_value = "fido-bot"
        # The originating commenter is the repo owner ("a") so the
        # auto-resolve oracle classifies them as CommentByActionable.
        # Resolution requires at least one actionable comment in the
        # chain that Fido has answered (#1663).
        mock_github.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "a"},
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 99,
                "in_reply_to_id": 42,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]
        # The auto-resolve oracle reads thread.comments.nodes (not the
        # flat get_pull_comments list) to decide who was last in the
        # thread.  Include fido's reply (id=99) so the decision is
        # ResolveReviewThread (we posted last).
        mock_github.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "a"}},
                        {"databaseId": 99, "author": {"login": "fido-bot"}},
                    ],
                },
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            _cmd(mock_github).complete(tmp_path, task["id"])  # type: ignore[arg-type]

        mock_github.resolve_thread.assert_called_once_with("thread_node_abc")
        assert "thread resolved: thread_node_abc" in caplog.text

    def test_completes_task_with_thread_skips_if_not_last(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _task_file(tmp_path)

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(tmp_path).add(
            title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = _FakeGitHub()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "id": 99,
                "in_reply_to_id": 42,
                "user": {"login": "copilot[bot]"},
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]
        # Bot reviewer was last in the thread.  ``[bot]``-suffixed authors
        # are classified as ``CommentByBot`` by the auto-resolve oracle,
        # which excludes them from being "last fido comment", so the
        # oracle decides DON'T resolve.  (Plain "reviewer" without bot
        # suffix or collaborator membership is classified as
        # ``CommentIgnored`` and skipped, which would let fido be
        # considered last — that's a genuine CLI gap when collaborator
        # metadata isn't available, tested elsewhere.)
        mock_github.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": False,
                "comments": {
                    "nodes": [
                        {"databaseId": 42, "author": {"login": "fido-bot"}},
                        {"databaseId": 99, "author": {"login": "copilot[bot]"}},
                    ],
                },
            }
        ]

        with caplog.at_level(logging.INFO, logger="fido"):
            _cmd(mock_github).complete(tmp_path, task["id"])  # type: ignore[arg-type]

        mock_github.resolve_thread.assert_not_called()
        assert "not resolving" in caplog.text

    def test_completes_task_with_thread_no_matching_comments(
        self, tmp_path: Path
    ) -> None:
        _task_file(tmp_path)

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(tmp_path).add(
            title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = _FakeGitHub()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = []

        _cmd(mock_github).complete(tmp_path, task["id"])  # type: ignore[arg-type]

        mock_github.resolve_thread.assert_not_called()

    def test_completes_task_with_thread_exception_silenced(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _task_file(tmp_path)

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(tmp_path).add(
            title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = _FakeGitHub()
        mock_github.get_user.side_effect = RuntimeError("network error")

        # Should not raise; exception is swallowed and logged
        with caplog.at_level(logging.WARNING, logger="fido"):
            _cmd(mock_github).complete(tmp_path, task["id"])  # type: ignore[arg-type]
        assert "thread resolution skipped" in caplog.text

    def test_completes_task_with_thread_already_resolved(self, tmp_path: Path) -> None:
        _task_file(tmp_path)

        thread = {"repo": "a/b", "pr": 1, "comment_id": 42}
        task = Tasks(tmp_path).add(
            title="threaded task", task_type=TaskType.THREAD, thread=thread
        )

        mock_github = _FakeGitHub()
        mock_github.get_user.return_value = "fido-bot"
        mock_github.get_pull_comments.return_value = [
            {
                "id": 42,
                "in_reply_to_id": None,
                "user": {"login": "fido-bot"},
                "created_at": "2024-01-01T00:00:00Z",
            },
        ]
        mock_github.get_review_threads.return_value = [
            {
                "id": "thread_node_abc",
                "isResolved": True,
                "comments": {"nodes": [{"databaseId": 42}]},
            }
        ]

        _cmd(mock_github).complete(tmp_path, task["id"])  # type: ignore[arg-type]

        mock_github.resolve_thread.assert_not_called()

    def test_thread_missing_fields_skips(self, tmp_path: Path) -> None:
        """Thread dict with missing fields should silently skip resolution."""
        _task_file(tmp_path)

        # thread missing 'pr' and 'comment_id'
        task = Tasks(tmp_path).add(
            title="task", task_type=TaskType.THREAD, thread={"repo": "a/b"}
        )

        mock_github = _FakeGitHub()
        _cmd(mock_github).complete(tmp_path, task["id"])  # type: ignore[arg-type]

        mock_github.resolve_thread.assert_not_called()

    def test_complete_nonexistent_id_no_error(self, tmp_path: Path) -> None:
        """Completing a non-existent task ID should not raise."""
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).complete(tmp_path, "nonexistent-id")  # type: ignore[arg-type]


# ── Cmd.list ──────────────────────────────────────────────────────────────────


class TestCmdList:
    def test_prints_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        cmd = _cmd(_FakeGitHub())  # type: ignore[arg-type]
        cmd.add(tmp_path, TaskType.SPEC, "alpha", "")
        capsys.readouterr()
        cmd.add(tmp_path, TaskType.SPEC, "beta", "desc")
        capsys.readouterr()
        cmd.list(tmp_path)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 2
        assert data[0]["title"] == "alpha"
        assert data[1]["title"] == "beta"

    def test_empty_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        _cmd(_FakeGitHub()).list(tmp_path)  # type: ignore[arg-type]
        out = capsys.readouterr().out
        assert json.loads(out) == []


# ── main (integration) ────────────────────────────────────────────────────────


class TestMain:
    def test_add_via_main(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "spec", "task title"], _GitHub=_FakeGitHub)  # type: ignore[arg-type]
        capsys.readouterr()

        tasks = Tasks(tmp_path).list()
        assert tasks[0]["title"] == "task title"

    def test_add_via_main_with_comment_id(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        main(
            [
                str(tmp_path),
                "add",
                "thread",
                "task title",
                "--comment-id",
                "55",
                "--repo",
                "r/r",
                "--pr",
                "3",
            ],
            _GitHub=_FakeGitHub,  # type: ignore[arg-type]
        )
        capsys.readouterr()

        tasks = Tasks(tmp_path).list()
        assert tasks[0]["thread"] == {"comment_id": 55, "repo": "r/r", "pr": 3}

    def test_complete_via_main(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "spec", "finish me"], _GitHub=_FakeGitHub)  # type: ignore[arg-type]
        out = capsys.readouterr().out
        task_id = json.loads(out)["id"]
        main([str(tmp_path), "complete", task_id], _GitHub=_FakeGitHub)  # type: ignore[arg-type]

        assert Tasks(tmp_path).list()[0]["status"] == "completed"

    def test_list_via_main(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _task_file(tmp_path)
        main([str(tmp_path), "add", "spec", "one"], _GitHub=_FakeGitHub)  # type: ignore[arg-type]
        capsys.readouterr()
        main([str(tmp_path), "list"], _GitHub=_FakeGitHub)  # type: ignore[arg-type]
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data[0]["title"] == "one"

    def test_no_args_exits(self) -> None:
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command_raises(self, tmp_path: Path) -> None:
        """Fallback case in match statement raises AssertionError."""
        fake_args = _FakeArgs("bogus")
        fake_parser = _FakeParser(fake_args)

        with pytest.raises(AssertionError, match="unreachable"):
            main([], _GitHub=_FakeGitHub, _build_parser=lambda: fake_parser)  # type: ignore[arg-type]
