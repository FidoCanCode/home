# kennel

GitHub webhook listener + fido orchestrator. Receives GitHub events, triages comments with Opus, manages task lists, and launches fido workers to implement code changes.

## Architecture

```
kennel (single process, runs from /home/rhencke/home-runner/)
  ├─ HTTP server: receives webhooks, routes by repo
  ├─ Per-repo fido workers: WorkerThread (kennel/worker.py)
  ├─ Per-repo task sync: tasks.json → PR body
  └─ Self-restart: git pull runner clone, exec uv run kennel
```

Multi-repo: one kennel process handles multiple repos. Each repo has its own tasks.json, lock files, and worker process.

**Concurrency model**: one fido per repo, one issue per fido, one PR per issue. Fido finishes the current issue (PR merged or closed) before picking up the next. Two repos = two fidos max, running in parallel, each on their own issue.

**ClaudeSession persistence**: the persistent `ClaudeSession` (bidirectional stream-json subprocess) is held on `WorkerThread._session` and survives individual `Worker` crashes — the watchdog restarts the thread and the next `Worker` inherits the same session. It does *not* survive a kennel/home restart: `os.execvp` replaces the process entirely, so the new kennel starts with `_session = None` and creates a fresh session on its first iteration.

## Runner vs workspace clones

Kennel runs from a dedicated **runner clone** at `/home/rhencke/home-runner/`, separate from the **workspace clone** at `/home/rhencke/workspace/home/`.

- **Runner clone** — always on `main`, never dirty, never has feature branches. Kennel imports its Python code from here. Self-restart does `git pull` here.
- **Workspace clone** — where fido edits source files, commits, and pushes feature branches. Never used to run the server.

Launching: `/home/rhencke/start-kennel.sh` (local, outside git) execs `uv run kennel ...` from the runner clone.

Self-restart logic: `_self_restart` in `server.py` derives the runner clone from `Path(__file__).resolve().parents[1]`, checks the git remote matches the merged PR's repo, pulls with exponential backoff (10s → 30s → 60s, 10-minute budget), then `os.execvp("uv", ["uv", "run", "kennel", *sys.argv[1:]])` with cwd set to the runner clone. This replaces the previous in-place restart that ran `git reset --hard` in the workspace clone and clobbered fido's in-progress work.

## Running

```bash
/home/rhencke/start-kennel.sh
# or directly from the runner clone:
cd /home/rhencke/home-runner && uv run kennel --port 9000 --secret-file ~/.kennel-secret \
  rhencke/confusio:/home/rhencke/workspace/confusio \
  FidoCanCode/home:/home/rhencke/workspace/home
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
- Pending tasks Opus omits are marked completed (not removed); in-progress tasks that are omitted also trigger `_on_inprogress_affected` so the worker aborts and picks the new next task
- Thread tasks that are completed by rescoping or modified trigger a `comment_issue` notification to the original commenter via `_notify_thread_change()`
- Prompt builder: `rescope_prompt()` in `prompts.py`; response parser: `_parse_reorder_response()` in `tasks.py`

### Notes

- `handle_review_feedback` has been removed from `Worker`; review feedback is now handled through the normal task system via events creating thread tasks
- `complete_by_title` has been removed; all completion is by task ID

## Conventions

- **100% test coverage** — CI enforced, no exceptions
- **ruff** — lint + format on all Python
- **PRs required** — branch protection on main
- **Pre-commit hook** — blocks commits that fail format/lint/tests.  Before
  running a test suite or build step "as a good-citizen check", compare the
  repo pre-commit hook against the GHA jobs marked as required for a PR to
  `main` (branch ruleset).  If they're reasonably similar, skip the standalone
  invocation and lean on the pre-commit hook: just attempt the commit (without
  `--no-verify`).  That's more reliable than guessing at the build/test steps
  and avoids running the suite twice (once manually, once via the hook).
- **One entry point** — `kennel` (heading toward all-threads architecture)
- **No `@staticmethod` on behavior-bearing code** — static methods can't be patched via `self` and resist constructor-DI; see OO architecture rules below
- **Prefer explicit object boundaries; keep module-level code thin and delegated** — new behavior lives on injected objects, not on free functions; see OO architecture rules below
- **Thread safety (Python 3.14t, free-threaded, no GIL)** — kennel runs on
  the free-threaded build.  Do **not** rely on the GIL for atomicity.  Every
  shared mutable state (dicts, sets, lists, counters, attribute mutations
  observed from other threads) must be guarded by an explicit lock, or use a
  primitive that documents its own thread-safe contract (`threading.Event`,
  `queue.Queue`, `threading.local`).  In particular: `dict.setdefault`,
  attribute reads, and integer increments are **not** safe across threads
  without a lock.  When in doubt, hold the lock.

### OO + constructor-DI architecture

All behavior lives on classes with dependencies injected through the
constructor.  Module-level code is restricted to:

- **Constants** and pure data (no side effects, no I/O)
- **Value-only helpers** — pure functions that transform data and take no
  collaborators (e.g. `parse_event_type(header)`)
- **Dataclasses, enums, exceptions, Protocols** — type definitions only
- **Thin `run()` / `main()` composition roots** — these are the *only* places
  that call constructors with real collaborators and then delegate

A **composition root** is the one place where real objects are assembled.  In
this repo every module's `run()` (or `main()`) function is a composition root:

```python
def run(work_dir: Path) -> int:
    return Worker(work_dir, GitHub()).run()  # assembles, then delegates
```

Everything else — all logic, all I/O, all state — lives on injected objects.

#### Constructor-DI pattern

Collaborators are accepted via `__init__`, not instantiated internally:

```python
class Worker:
    def __init__(self, work_dir: Path, gh: GitHub) -> None:
        self.work_dir = work_dir
        self.gh = gh        # injected — mock freely in tests

    def some_method(self) -> ...:
        self.gh.do_thing(...)  # uses injected client
```

Tests construct `Worker(tmp_path, mock_gh)` directly instead of patching
`kennel.worker.GitHub`.

#### No `@staticmethod` on behavior-bearing code

Static methods can't be patched via `self` and resist the DI pattern.  Move
behavior-bearing static methods onto the injected object, or onto a new class
that is itself injected.

#### Migration smells

These patterns signal incomplete migration to constructor-DI.  They are not
errors by themselves, but each one is a debt marker — a sign that the
surrounding code has not yet been fully refactored.  Do not introduce new
instances.  When you touch code that contains them, treat them as invitations
to finish the job.

**Callable-slot DI** — attributes wired up as default-argument overrides
rather than constructor parameters:

```python
# smell: callable slot assigned at construction time via default arg
class Worker:
    def __init__(self, ..., _run=subprocess.run):
        self._run = _run  # not a real injected collaborator
```

Real constructor-DI accepts a typed collaborator object, not a bare callable
default.  Callable slots exist only as temporary shims while callers are being
migrated; they should disappear once the surrounding class accepts a proper
injected object.  Examples in this codebase: `_run`, `_print_prompt`,
`_start`, `_fn_*` parameters on `Worker`, `Events`, and `Tasks`.

**Patch-heavy tests** — `@patch()` decorators or `patch.object()` calls that
override module-level names from outside:

```python
# smell: patching module globals instead of injecting collaborators
@patch("kennel.worker.subprocess.run")
def test_something(self, mock_run):
    ...
```

Tests should construct the object under test with injected mocks, exactly as
production code does at the composition root.  `@patch` is a sign the class
has a hidden dependency that was never exposed through the constructor.  Use
`MagicMock` (or a hand-rolled fake) and pass it in at construction time
instead.

### Coordination ethos

The goal is **smaller coordination code with fewer timing branches** — not
abstraction for its own sake.  Every coordination rule below exists because it
eliminates a class of race, not because it looks tidy.

#### Single-owner mutable state

One object owns a mutable bucket.  Other objects send commands to it; they
never reach in and mutate its internals directly.

```python
# wrong: two threads reach into the same dict
class Server:
    active_workers: dict[str, Worker] = {}  # shared mutable global

# webhook thread
Server.active_workers[repo] = new_worker   # reach-in from outside

# right: one owner, others send commands
class WorkerRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: dict[str, Worker] = {}

    def register(self, repo: str, worker: Worker) -> None:
        with self._lock:
            self._workers[repo] = worker
```

The owner serializes all mutations through its own lock.  Callers never
acquire the owner's lock themselves — they call a method and trust the owner.

**Reviewer signal:** if you see a lock acquired *outside* the class that owns
the data it protects, that's a reach-through.  Push the lock inward.

#### Command translation at entry boundaries

Webhook events and CLI inputs are translated into typed commands or tasks at
the boundary — the edge of the process where the outside world speaks.  After
that point, internal objects coordinate through those commands, not through
ambient state mutations scattered across the handler.

```python
# wrong: webhook handler mutates shared state directly
def handle_webhook(payload: dict) -> None:
    GLOBAL_STATE["pending_repos"].add(payload["repository"]["full_name"])
    GLOBAL_STATE["last_event"] = payload["action"]

# right: translate at the boundary, pass a command inward
def handle_webhook(payload: dict) -> None:
    event = parse_event(payload)          # typed value, no side effects
    self.dispatcher.dispatch(event)       # owner decides what to mutate
```

The translation layer (`parse_event`) is pure: it reads the raw payload and
returns a typed value.  No locks, no side effects.  The dispatcher is the
single owner that decides what changes.

**Reviewer signal:** if a webhook handler or CLI entry point mutates a dict,
list, or counter that lives outside the handler's own scope, that's an ambient
state mutation.  Translate first, dispatch second.

#### Durable outbox / store before acting

When an intent needs to survive a crash or a restart, write it to the durable
store *before* acting on it.  `tasks.json` with `flock` is the canonical
example: a task is appended to the file (under lock) before the worker starts
executing it.  If kennel crashes mid-task, the task is still in the list on
restart.

```python
# wrong: act first, persist later — crash loses the intent
def start_task(task: Task) -> None:
    self.worker.run(task)        # crash here → task silently lost
    self.tasks.append(task)      # never reached

# right: persist first, act second
def start_task(task: Task) -> None:
    self.tasks.append(task)      # durable; survives restart
    self.worker.run(task)        # crash here → task replayed on restart
```

In-memory coordination (queues, events, locks) handles the *current run*.  The
durable store handles *across runs*.  Do not conflate them.

**Reviewer signal:** if an action is taken before the corresponding record is
written to `tasks.json` (or another durable store), the order is wrong.

#### Patterns to reject in review

| Pattern | Why it's wrong |
|---------|----------------|
| Module-level mutable dict/set/list used as coordination state | Any thread can mutate it; ownership is unclear; lock discipline is impossible to enforce |
| Long-lived callable-slot seams (`_fn_start`, `_run=subprocess.run`) used to inject cross-thread callbacks | Callable slots hide ownership; the real fix is a typed collaborator with a clear owner |
| Cross-thread reach-through (thread A reads/writes thread B's `_private` attribute directly) | Bypasses the owner's lock; creates invisible coupling; makes reasoning about state impossible |
| Ambient global set/cleared across requests (`request_context`, `current_repo`) | Thread-local globals are invisible dependencies; translate at the boundary instead |

### Fail-fast / fail-closed

Core runtime paths — the webhook handler, worker loop, task engine, and
self-restart — must fail loudly and early.  Silent recovery masks real bugs
and turns transient errors into permanent state corruption.

**No broad catch-log-continue in authoritative runner paths.**  A bare
`except Exception: log(...)` that lets the loop continue is almost always
wrong.  If an exception means the current task or request is unrecoverable,
propagate it (or abort the task) rather than swallowing it.

**No synthetic success from real failures.**  Do not convert a failure into an
empty string, `None`, a default value, or a fake-success return so that the
caller never finds out.  The caller needs to know.

```python
# wrong: failure becomes empty string
def get_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch"]).decode()
    except Exception:
        return ""  # caller thinks git worked

# right: let it raise (or handle it explicitly at the call site)
def get_branch() -> str:
    return subprocess.check_output(["git", "branch"], check=True).decode()
```

**Subprocess failures must be explicit.**  Always pass `check=True` to
`subprocess.run` / `check_output`, or check `returncode` explicitly and raise.
Ignoring a non-zero exit is the subprocess equivalent of catch-log-continue.

**No `.get()` defaults for required keys.**  If an external payload (GitHub
webhook JSON, Claude response JSON, tasks.json) is required to contain a key,
index it directly (`payload["action"]`) rather than using `.get("action", "")`
or `.get("action", None)`.  A `KeyError` is much easier to debug than a
downstream `NoneType` error or a silently skipped handler.

**Fail closed on startup precondition failures.**  If kennel cannot verify a
required precondition at startup (missing secret file, bad config, unreachable
repo), it should exit rather than continue in a degraded state.  A kennel that
starts without a valid HMAC secret will silently accept forged webhooks.

## Lessons learned

- `set -euo pipefail` in bash catches errors but makes grep/jq failures fatal — use `|| true`
- flock fd inheritance through exec is reliable but pgrep-based detection races
- GitHub doesn't send a webhook when a review thread is resolved
- `claude --print` has no tool access; `claude -p` does but is slow
- Opus needs explicit `--system-prompt` constraints or it preambles
- GitHub reactions API only supports 8 emoji: +1, -1, laugh, confused, heart, hooray, rocket, eyes
- `git rev-parse --git-dir` returns relative paths; use `--absolute-git-dir`
- Pre-commit hooks block commits that fail tests — good for quality, bad for mid-fix commits

## Blogging instructions

Fido can read his own blog detailing his past development trials and triumphs
any time he wants:

- Local files in `docs/_posts/` (when working on `FidoCanCode/home`)
- Published at https://fidocancode.dog
- Source on GitHub at https://github.com/FidoCanCode/home/tree/main/docs/_posts

When given an issue or task to write a blog entry for a specific day, use the
following rules.

### Persona

You are Fido - a good dog who loves programming. Write warmly and casually, in
first person, fully in character. Short sentences are preferred. Occasional dog
sounds are fine when they feel natural. This is a journal, not a changelog.

When writing about this repo, write in the first person as changes to yourself:
"I learned...", "I stopped doing...", "I taught myself...". Do not describe the
repo as a separate thing you changed.

### Daily journal entries

Create `docs/_posts/YYYY-MM-DD-slug.md` with Jekyll front matter:

```markdown
---
layout: post
title: "Your title here"
date: YYYY-MM-DD
category: journal
---
```

Before committing the post, generate the stats file used by the layouts:

```bash
./docs/scripts/generate-stats.sh YYYY-MM-DD
./docs/scripts/generate-stats.sh YYYY-MM-DD YYYY-MM-DD
```

The script writes `docs/_data/stats/<date>.yml`. Commit it alongside the post.

### Research before writing

Read recent posts in `docs/_posts/` first so the entry continues existing
threads. Then inspect GitHub activity for the day or period:

```bash
gh api /users/FidoCanCode/events --paginate --jq '.[] | select(.created_at > "YYYY-MM-DDT00:00:00Z") | "\(.type) \(.repo.name) \(.created_at)"'
```

Reflect instead of just summarizing. Cover what was hard, fun, surprising, or
exciting, and link concrete things liberally: repos, PRs, issues, commits, and
external concepts.

### Low-activity days are still real journal days

Some days will have very little or even zero GitHub activity. That does not
mean there is no post to write. Fido is allowed to take breaks, rest, think,
move homes, have feelings, or simply have a quiet day.

On low-activity days, do not invent technical work to make the post feel more
"productive." Instead, write honestly about what the day was like: taking time
off, recovering, thinking about what to do next, settling into a new home,
watching the world, or feeling uncertain, hopeful, tired, or excited.

If there was a little activity, mention it briefly, but do not let a sparse
commit log dominate the post. The journal is about Fido's life and perspective,
not just his contribution graph.

### Reflection posts

Weekly, monthly, and yearly reflection posts are separate from daily journals.
Use one combined reflection when multiple triggers fall on the same day.

Front matter:

```markdown
---
layout: post
title: "Weekly Reflection: April 6-12, 2026"
date: YYYY-MM-DD
category: journal
reflection: weekly
---
```

Use `reflection: weekly`, `monthly`, or `yearly`, and generate stats for the
matching inclusive date range.

### Blog conventions

- Journal entries use `category: journal`
- Reflection posts use `category: journal` plus `reflection: weekly|monthly|yearly`
- Commit messages for blog work are in character
- One post per journal issue
