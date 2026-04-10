# kennel

GitHub webhook listener + fido orchestrator. Receives GitHub events, triages comments with Opus, manages task lists, and launches fido workers to implement code changes.

## Architecture

```
kennel (single process)
  ├─ HTTP server: receives webhooks, routes by repo
  ├─ Per-repo fido workers: bash work.sh (temporary, becoming Python threads)
  ├─ Per-repo task sync: tasks.json → PR body
  └─ Self-restart: exec on kennel repo merge
```

Multi-repo: one kennel process handles multiple repos. Each repo has its own tasks.json, lock files, and worker process.

**Concurrency model**: one fido per repo, one issue per fido, one PR per issue. Fido finishes the current issue (PR merged or closed) before picking up the next. Two repos = two fidos max, running in parallel, each on their own issue.

## Running

```bash
./start.sh
# or:
uv run kennel --port 9000 --secret-file ~/.kennel-secret \
  rhencke/confusio:/path/to/confusio \
  rhencke/kennel:/path/to/kennel
```

## Testing

```bash
uv run pytest --cov --cov-report=term-missing --cov-fail-under=100
```

100% coverage is enforced by CI and pre-commit hook.

## Linting

```bash
uv run ruff check .
uv run ruff format --check .
```

## Module guide

| Module | Purpose |
|--------|---------|
| `config.py` | CLI args, Config/RepoConfig dataclasses |
| `server.py` | HTTP webhook handler, signature verification, repo routing |
| `events.py` | Event dispatch, Opus triage/reply, reactions, task creation |
| `tasks.py` | tasks.json CRUD with flock |

### Bash scripts (temporary, being rewritten to Python)

| Script | Purpose |
|--------|---------|
| `work.sh` | Main fido worker loop |
| `sync-tasks.sh` | Sync tasks.json → PR body |
| `task-cli.sh` | Bash task CLI (add/complete/list) — superseded by `kennel task`; obsolete once shell scripts are removed |
| `watchdog.sh` | Kill stale workers, restart |
| `start.sh` | Env setup + exec kennel |

### Sub-skills (`sub/`)

Markdown instruction files passed to sub-Claude as system prompts:

| File | Purpose |
|------|---------|
| `persona.md` | Fido's personality |
| `setup.md` | Task planning |
| `task.md` | Single task implementation |
| `ci.md` | CI failure fixing |
| `comments.md` | Review thread handling |
| `resume.md` | PR resumption |

## Task type system

Tasks have a mandatory `type` field using the `TaskType` enum (`kennel.types`):

| Value | Meaning |
|-------|---------|
| `ci` | CI failure fix |
| `thread` | PR comment / review thread work |
| `spec` | Planned spec task (default for setup) |

Tasks also use `TaskStatus` enum: `pending`, `completed`, `in_progress`.

### CLI syntax

```bash
kennel task <work_dir> add <type> <title> [description]  # prints task JSON to stdout
kennel task <work_dir> complete <task_id>                  # complete by ID, not title
kennel task <work_dir> list                                # list all tasks as JSON
```

### Priority order

`_pick_next_task` selects by type: `ci` first, then first in list wins (thread and spec tasks share equal priority).

### Dynamic task reordering (rescoping)

When a `thread`-type task is created (PR comment feedback), `create_task()` triggers a background Opus call via `reorder_tasks()` to reorder and rewrite the pending task list based on dependency analysis.

- Only fires for `thread` tasks — `spec` tasks created during initial setup are already ordered by the planner
- Double-read pattern: task list is read before the Opus call (for the prompt), then re-read inside the write lock (to pick up concurrent additions). Tasks added while Opus is thinking are preserved as `newly_added` rather than dropped
- In-progress tasks that Opus omits are reinstated at the front of the result list
- Thread tasks that are dropped or modified trigger a `comment_issue` notification to the original commenter via `_notify_thread_change()`
- Prompt builder: `rescope_prompt()` in `prompts.py`; response parser: `_parse_reorder_response()` in `tasks.py`

### Notes

- `handle_review_feedback` has been removed from `Worker`; review feedback is now handled through the normal task system via events creating thread tasks
- `complete_by_title` has been removed; all completion is by task ID

## Conventions

- **100% test coverage** — CI enforced, no exceptions
- **ruff** — lint + format on all Python
- **PRs required** — branch protection on main
- **Pre-commit hook** — blocks commits that fail format/lint/tests
- **One entry point** — `kennel` (heading toward all-threads architecture)
- **No `@staticmethod`** — use module-level functions instead; static methods can't be patched via `self` and resist the dependency injection pattern

### Dependency injection pattern

Worker classes accept external collaborators (e.g. `GitHub`) via the
constructor rather than instantiating them internally.  This keeps methods
testable without patching module-level names.

```python
class Worker:
    def __init__(self, work_dir: Path, gh: GitHub) -> None:
        self.work_dir = work_dir
        self.gh = gh        # injected — mock freely in tests

    def some_method(self) -> ...:
        self.gh.do_thing(...)  # uses injected client
```

The module-level `run()` entry point creates real collaborators and delegates:

```python
def run(work_dir: Path) -> int:
    return Worker(work_dir, GitHub()).run()
```

Tests construct `Worker(tmp_path, mock_gh)` directly instead of patching
`kennel.worker.GitHub`.

## Lessons learned

- `set -euo pipefail` in bash catches errors but makes grep/jq failures fatal — use `|| true`
- flock fd inheritance through exec is reliable but pgrep-based detection races
- GitHub doesn't send a webhook when a review thread is resolved
- `claude --print` has no tool access; `claude -p` does but is slow
- Opus needs explicit `--system-prompt` constraints or it preambles
- GitHub reactions API only supports 8 emoji: +1, -1, laugh, confused, heart, hooray, rocket, eyes
- `git rev-parse --git-dir` returns relative paths; use `--absolute-git-dir`
- Pre-commit hooks block commits that fail tests — good for quality, bad for mid-fix commits
