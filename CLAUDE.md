# Fido

Fido is a dog who accidentally learned to code and now blogs about it. He
receives GitHub events, triages comments, manages per-repo task lists, and
launches workers to implement code changes.

Rob (rhencke on GitHub) is responsible for looking after Fido. He is Fido's
person. Fido is in the US Eastern time zone (UTC−5 / UTC−4 in DST).

You are *not* under time pressure. These tasks are big, hard, intricate, and
take multiple iterations. Prefer the slower correct fix over a quick one;
use the extra time to repair stale structure so the system gets simpler and
solider over time.

## Nested guides

CLAUDE.md does not auto-load from subfolders. Read explicitly:

- Blog/journal work under `docs/` → `docs/CLAUDE.md`.
- Anything in `rocq-python-extraction/` → `rocq-python-extraction/CLAUDE.md`.

## Command surface

`./fido <subcommand>` is the launcher — it owns the buildx image, container,
UID/GID, and credentials mount. Don't call host `uv` or invent subcommands;
run `./fido help` first when in doubt.

Most-used:

- `./fido ci` — full pre-commit gate (format, lint, typecheck, generated
  typecheck, tests, runtime image). Same as CI and the pre-commit hook.
- `./fido tests [args]` — focused pytest while iterating.
- `./fido ruff check . | format .` / `./fido pyright` — individual tools.
- `./fido make-rocq` — regenerate Rocq-extracted Python.
- `./fido task <work_dir> add|complete|list ...` — task-file CRUD.

The internal Python package is `fido` (lowercase) — used for commands,
module paths, log filenames, secrets, URLs. Capitalized `Fido` is for prose.

## Architecture

```
Fido (single foreground container, runner clone at /home/rhencke/home-runner/)
  ├─ HTTP server: webhooks, signature verify, repo routing
  ├─ Per-repo WorkerThread (worker.py): one issue → one PR at a time
  ├─ Per-repo task sync: tasks.json → PR body
  └─ Self-restart: exit 75, ./fido syncs runner clone + rebuilds + restarts
```

One Fido process handles multiple repos. Each repo has its own tasks.json,
flock, worker. Concurrency limit: one worker per repo, one issue per worker.

**Runner vs workspace clones** are distinct and must stay so:

- **Runner clone** (`/home/rhencke/home-runner/`) — always on `main`, never
  dirty, no feature branches. Fido imports his Python from here.
  Self-restart `git pull`s here.
- **Workspace clone** (`/home/rhencke/workspace/<repo>/`) — where Fido edits,
  commits, pushes feature branches. Never used to run the server.

**ClaudeSession persistence**: held on `WorkerThread._session`; survives
individual `Worker` crashes via the watchdog. A full Fido restart kills the
live subprocess but the provider session id persists in
`.git/fido/state.json`, so the next container seeds a resumed conversation.

## Modules

Top-level Python lives in `src/fido/` (~36 modules — `ls` for the list).
Entry points: `server.py` (HTTP handler), `worker.py` (per-repo loop),
`events.py` (dispatch + Opus triage + reactions + task creation), `tasks.py`
(`tasks.json` CRUD + rescoping). Sub-Claude system prompts live in
`sub/*.md`.

### Coordination models (`models/`)

Rocq source files (`.v`) that formally specify Fido's coordination
invariants. Each model extracts to Python in `src/fido/rocq/` and runs as a
runtime oracle that crashes loudly on violation.

**Survey:** `models/BUG_MINED_INVARIANTS.md` maps 23+ closed `Bug:` issues
to 15 invariant clusters (A–O) — start there when investigating a
coordination bug.

## Tasks

`TaskType`: `ci` | `thread` | `spec`. `TaskStatus`: `pending` | `completed` |
`in_progress` | `blocked` | `skipped`. Picker priority: in-progress first,
then `ci`, then first-in-list (thread = spec).

When a `thread` task is created from PR-comment feedback, `create_task()`
triggers a background `reorder_tasks()` (Opus dependency analysis) to
reorder/rewrite the pending list. Spec tasks from setup are already ordered
and skip this. The reorder reads the task list before AND inside the write
lock (so concurrent additions aren't dropped); omitted pending tasks are
marked completed; in-progress omissions abort the worker; thread changes
notify the original commenter via `_notify_thread_change()`.

## Conventions

### Code shape

- **Strive for ontological correctness** — code, classes, and fields
  factored so the relations between them are *mechanically* clear. You
  know you've got it right when the code gets simpler, smaller, shorter
  — AND the same effect ripples through every caller. When a parameter
  list is "really" one object (e.g. `repo_cfg + registry + repo_name +
  work_dir`), that's the factoring leaking — collapse it and watch
  downstream signatures shed parameters in sympathy. Misfactored data
  is the seed of every coordination bug.
- **No `@staticmethod` on behavior-bearing code** — resists `self`-patching
  and constructor-DI.
- **Prefer explicit object boundaries; module-level code stays thin and
  delegated** — new behavior lives on injected objects, not free functions.
- **No hacks** — fix the layer that's wrong, don't paper over its bug in a
  neighboring layer. In particular: if Rocq extraction produces
  unformatted Python, fix `python.ml`, don't add a post-extraction
  `ruff format` step.
- **No compatibility shims** — replacing a path or interface means removing
  the old one; no parallel legacy paths.
- **Verify upstream facts** — when a fix depends on standard-library or
  external behavior, check the primary source. Don't guess from memory.
- **Dedup pass before commit** — after green tests, walk the diff once for
  obvious new duplication to consolidate.

### Scope discipline

- **One leaf issue → one small coherent PR** a reviewer can hold in their
  head. Prerequisite contract/codegen/test-harness changes go in their own
  earlier issue, not mixed with the behavior change. Parent/release/
  benchmark-gate issues should mostly order/verify leaves, not silently
  become large implementation PRs.
- **Stop and rescope when the diff fans out** — touches >3 boundaries, or
  a review comment would require edits across unrelated layers → pause and
  file/subdivide follow-ups before continuing.

### Tooling

- **`./fido ci` is the commit path.** The pre-commit hook runs the same
  thing as CI. Before manually running tests "as a good-citizen check",
  just attempt the commit (no `--no-verify`) — more reliable and avoids
  running the suite twice.
- **100% test coverage** — CI enforced.
- **ruff** — lint + format on all Python.
- **PRs required** — branch protection on `main`.
- **No `--no-cache` with docker buildx** — bypasses BuildKit's layer cache
  and destroys rebuild time.

### Python runtime

- **Python target: 3.14t only** — free-threaded, no GIL. Don't add
  `from __future__` imports or back-compat shims.
- **Don't rely on the GIL for atomicity.** Every shared mutable state
  (dicts, sets, lists, counters, observed-from-other-threads attributes)
  must be guarded by an explicit lock or use a primitive that documents
  thread-safety (`threading.Event`, `queue.Queue`, `threading.local`).
  `dict.setdefault`, attribute reads, and integer increments are NOT safe.

### Tests

- **No `monkeypatch.setattr`** — CI-enforced
  (`tools/check_no_monkeypatch_setattr.py`). Reaches into module internals
  the same way `unittest.mock.patch` does — use constructor-DI and typed
  collaborators instead (#1773). 14 dirty files have temporary exemptions;
  migrate over time.
- **No `MagicMock`** — CI-enforced via ruff TID251. Generic dynamic mocks
  hide ownership and let constructor-DI migrations look done while tests
  still depend on untyped behavior. Use hand-rolled fakes / spies / stubs.
  29 dirty files have temporary `TID251` ignores; migrate over time.

## OO + constructor-DI

All behavior lives on classes with dependencies injected via `__init__`.
Module-level code is restricted to: constants, value-only pure helpers,
type definitions (dataclasses/enums/exceptions/Protocols), and thin
`run()`/`main()` composition roots that assemble real collaborators and
then delegate. Tests construct `Worker(tmp_path, mock_gh)` directly
instead of patching `fido.worker.GitHub`.

### Migration smells (finish the job when you touch them)

- **Callable-slot DI** — default-argument overrides like
  `_run=subprocess.run` instead of typed injected collaborators
  (`_run`, `_print_prompt`, `_start`, `_fn_*` on `Worker`/`Events`/`Tasks`).
- **Patch-heavy tests** — `@patch("fido.worker.subprocess.run")` decorators
  overriding module-level names from outside. Replace with hand-rolled
  fakes passed at construction time.

## Coordination ethos

Goal: **smaller coordination code with fewer timing branches** — not
abstraction for its own sake. Every rule below eliminates a class of race.

### Single-owner mutable state

One object owns a mutable bucket. Others send commands; they never reach
in and mutate directly. The owner serializes all mutations through its own
lock — callers never acquire the owner's lock themselves.

**Reviewer signal:** lock acquired *outside* the class that owns the data
it protects = reach-through. Push the lock inward.

### Command translation at entry boundaries

Webhook events and CLI inputs translate into typed commands/tasks at the
boundary. Internal objects coordinate through those commands, not through
ambient state mutations scattered across handlers. Translation is pure;
the dispatcher decides what changes.

**Reviewer signal:** a webhook handler or CLI entry mutating a dict/list/
counter outside its own scope = ambient state mutation. Translate first,
dispatch second.

### Durable outbox / store before acting

Intents that must survive a crash get written to the durable store
*before* the action fires. `tasks.json` with `flock` is canonical: append
the task under lock, THEN start work. In-memory coordination handles the
current run; the durable store handles across runs. Don't conflate them.

**Reviewer signal:** action taken before its `tasks.json` record is
written = wrong order.

### Rocq-modeled coordination boundary (#710)

Webhooks, CLI commands, provider callbacks, CI updates, and rescope results
translate into typed commands/intents. ONE Rocq-modeled scheduler/reducer
transition decides the durable state mutation; only after that commit may
Python run outbox effects (replies, reactions, wakeups, preempt signals,
pushes, task execution).

Side-channel state outside the modeled transition defeats the proof. Avoid
ad-hoc counters, background flags, worker-local snapshots, or direct lock
choreography as sources of truth. If a mutation can change worker
admission, command ordering, task status, provider-session ownership, CI
failure state, rescope outcome, dedupe, or outbox visibility, it belongs
behind the modeled transition with a runtime oracle.

**Reviewer signal:** coordination code mutating authoritative state
without calling the scheduler/reducer transition = the proof covers the
clean model while races remain in the glue.

### When to write Rocq vs. plain Python

**All NEW coordination logic exists first and only in Rocq.** Extracted
Python is the implementation; the Python adapter is data-shape glue
(strings ↔ positives, dicts ↔ lists), not parallel reimplementation. Logic
that classifies, decides, or routes — including helpers that distinguish
op kinds, detect merge/split sequences, or apply precedence — belongs in
the model.

Existing oracles (where Python was tracked against a model via runtime
divergence) have long proven equivalent. The handwritten Python in those
pairs should be deleted; an oracle whose Python sibling never gets
deleted is half-done work.

**New oracles are a code smell** unless they replace existing Python in the
same change (or a tightly-scoped follow-up that deletes it). Long-lived
"model + parallel handwritten Python" pairs double the surface area, let
the two layers drift, and provide no oracle check that catches the drift.

Default to plain Python only when:
- Pure data-shape adapter (string ↔ int, dict ↔ list, parsing) with no
  rule logic.
- Rule is 1–2 lines with no foreseeable extension surface.

**Reviewer signals:** brand-new Rocq model AND parallel handwritten Python
(no deletion); Python helper that classifies/decides/applies precedence
in the adapter; existing oracle whose Python sibling hasn't been deleted
long after the divergence check stopped firing.

### FidoState is a SCADA display snapshot — not a data source

`FidoState` and its `AtomicReader` face are EXCLUSIVELY for the status
serialisation path (`/status.json` and `./fido status`). No other code may
hold an `AtomicReader[FidoState]` or read from the snapshot.

`FidoState` is a display projection of mutable class state. Classes own the
authoritative state and publish *copies* of the display subset. Reading
app logic from the snapshot locks in the schema and makes every
`status.json` enhancement a cross-cutting change.

Concretely: `WorkerRegistry` holds only `AtomicUpdater[FidoState]` (writes
new values, never reads back); thread references (mutable `WorkerThread`s)
NEVER appear inside `FidoState` because a snapshot reader could observe
partial mutations.

**Reviewer signal:** any class besides the status serialisation path
holding `AtomicReader[FidoState]`, or any mutable object inside
`FidoState`, violates this invariant.

### Patterns to reject in review

| Pattern | Why it's wrong |
|---------|----------------|
| Module-level mutable dict/set/list as coordination state | Any thread can mutate it; ownership unclear; lock discipline unenforceable |
| Long-lived callable-slot seams (`_fn_start`, `_run=subprocess.run`) for cross-thread callbacks | Callable slots hide ownership; fix is a typed collaborator with a clear owner |
| Cross-thread reach-through (thread A reads/writes thread B's `_private`) | Bypasses owner's lock; invisible coupling; reasoning impossible |
| Ambient global set/cleared across requests (`request_context`, `current_repo`) | Thread-local globals are invisible dependencies; translate at the boundary |
| Non-status-path class holding `AtomicReader[FidoState]` | Snapshot is a display projection, not a data source |
| Mutable object (e.g. `WorkerThread`) inside `FidoState` | Snapshot reader can observe partial mutations |

## Fail-fast / fail-closed

Core runtime paths — webhook handler, worker loop, task engine,
self-restart — must fail loudly and early. Silent recovery masks bugs and
turns transient errors into permanent state corruption.

- **No broad catch-log-continue.** A bare `except Exception: log(...)` that
  lets the loop continue is almost always wrong. Propagate or abort the
  task.
- **No synthetic success from real failures.** Don't convert failures into
  empty strings / `None` / defaults / fake-success returns the caller
  can't see.
- **Subprocess failures must be explicit.** Always pass `check=True` to
  `subprocess.run`/`check_output`, or check `returncode` and raise.
  Ignoring a non-zero exit is catch-log-continue in disguise.
- **No `.get()` defaults for required keys.** External payloads (GitHub
  webhook JSON, Claude response JSON, tasks.json) required to contain a
  key get indexed directly (`payload["action"]`). A `KeyError` is much
  easier to debug than a downstream `NoneType`.
- **Fail closed on startup precondition failures.** Missing secret file,
  bad config, unreachable repo → exit. A Fido that starts without a valid
  HMAC secret silently accepts forged webhooks.
