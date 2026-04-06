Implement one task from the work queue. All context (PR, repo, branch, task title) is in the Context section above.

## Steps

### 1. Look up the full task
Call TaskList. Find the task whose title matches the one given in the Context section. Read its full description — it may contain a thread node_id, first_db_id, and other implementation notes.

Pick by priority if multiple are available:
1. Fix CI: tasks
2. PR comment: tasks
3. All other tasks

Skip any ASK: tasks — those are waiting on human input.

### 2. Implement
1. TaskUpdate → in_progress
2. Read CLAUDE.md for conventions, test commands, commit discipline.
3. Implement the change.
4. Verify (CLAUDE.md test command; default: `make test`). If it fails, keep in_progress and exit — do not move on.
5. Commit with a descriptive message and push:
   ```bash
   git commit -m "<descriptive message>"
   git push
   ```
6. TaskUpdate → completed

### 3. If title starts with "PR comment:" (or is a link starting with "[PR comment:")
The task title is a markdown link with a hidden HTML comment carrying the IDs:
`[PR comment: <summary>](<comment_url>) <!-- thread:<node_id> first_db_id:<first_db_id> -->`
Parse `node_id`, `first_db_id`, and the commenter login from the HTML comment / task description.

**Before implementing**, fetch the current thread to read all comments — the reviewer may have added clarifications after the task was queued:
```bash
gh api repos/<owner>/<repo>/pulls/<pr>/comments \
  --jq '[.[] | select(.in_reply_to_id == <first_db_id> or .id == <first_db_id>)] | sort_by(.created_at) | .[] | "\(.user.login): \(.body)"'
```
Implement based on the full thread, not just the task title. If the latest comment changes or narrows the requirement, honour it.

**Committer attribution** — the commenter is the committer; Fido is the author. Look up and apply before committing:
```bash
_COMMENTER_NAME=$(gh api /users/<commenter_login> --jq '.name // .login')
_COMMENTER_EMAIL=$(gh api /users/<commenter_login> --jq '.email // empty')
: "${_COMMENTER_EMAIL:=<commenter_login>@users.noreply.github.com}"
GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" \
GIT_COMMITTER_NAME="$_COMMENTER_NAME" GIT_COMMITTER_EMAIL="$_COMMENTER_EMAIL" \
  git commit -m "<descriptive message>"
```

Post directly with `in_reply_to` — do not check for pending reviews first. A pending review by the PR author does not block `pulls/{pr}/comments`; just post and it will succeed.
```bash
# Draft the reply in plain English, then rewrite in character via Opus:
# If change was made:
_PLAIN="Done — <one-line plain-English summary of what was changed>"
# If infeasible / no-op:
_PLAIN="Investigated — <brief plain-English explanation>"

_PERSONA=$(cat ~/.claude/skills/fido/sub/persona.md)
_BODY=$(printf '%s\n\nRewrite the following GitHub PR comment in character as Fido. Keep it brief. Output only the comment text, no quotes, no explanation.\n\n%s' "$_PERSONA" "$_PLAIN" \
  | claude --model claude-opus-4-6 --print 2>/dev/null | head -10)
: "${_BODY:=$_PLAIN}"
gh api repos/<owner>/<repo>/pulls/<pr>/comments \
  -X POST -F body="$_BODY" \
  -F in_reply_to=<first_db_id>

gh api graphql -F threadId=<node_id> \
  -f query='mutation($threadId:ID!){resolveReviewThread(input:{threadId:$threadId}){thread{id}}}'
```

### 4. Sync the work queue
Fetch PR body, update the block between the HTML comment markers, write it back:
- Pending tasks: `- [ ] Title` list; the **first** item must have `**→ next**` appended — no exceptions
- Completed tasks: `<details><summary>Completed (N)</summary>` block; omit if none
- `PR comment:` titles rendered as links to thread URLs

```bash
gh pr edit <pr> --repo <repo> --body "..."
```

## Done when
Task completed, committed and pushed, thread resolved (if applicable), work queue synced.

**Stop immediately after completing this one task. Do not read the work queue. Do not start the next task. Your job is exactly one task per invocation.**

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft. That is the user's decision.
- **Never** continue to another task after completing the current one. One task per invocation, period.
- **Never** rebase, amend, or force-push. New commits only.
- **Never** call any `/reviews` endpoint (read or write). Use only `pulls/{pr}/comments` with `in_reply_to=<first_db_id>` for thread replies — no pre-checks, no fallbacks.
