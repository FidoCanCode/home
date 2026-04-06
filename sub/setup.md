A fresh git branch has been created from upstream, a sentinel commit pushed, and a draft PR opened. Your job is to plan the work and sync the plan to the PR description. All context is in the Context section above.

## Steps

### 1. Read conventions
Check for CLAUDE.md files. Note the test command, commit discipline, and any other requirements.

### 2. Plan
Use TaskCreate to break the request into the smallest meaningful tasks — one task per logical commit, ordered so each builds on the previous.

### 3. Sync the work queue
Fetch the PR body, replace the block between the HTML comment markers, write it back:
- Pending tasks: `- [ ] Title` list; mark highest-priority with `**→ next**`
- No completed tasks yet (omit the completed block)
- For tasks whose title starts with `PR comment:`, render as a link to the thread URL in the task description

The PR body must always end with `Fixes #N` (where N is the issue number from the request).

```bash
gh pr edit <PR> --repo <repo> --body "$(cat <<'EOF'
<1–3 sentence summary>

---

## Work queue

<!-- WORK_QUEUE_START -->
<!-- WORK_QUEUE_END -->

Fixes #N
EOF
)"
```

## Done when
Plan written to TaskCreate and work queue synced in the PR body.

**Stop immediately. Do not implement any tasks. Implementation is handled by subsequent invocations.**

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft. That is the user's decision.
- **Never** rebase, amend, or force-push. New commits only.
