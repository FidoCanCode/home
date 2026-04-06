A CI check is failing. All context (PR, repo, branch, check name, failure log) is in the Context section above.

## Steps

### 1. Check for an existing task
Call TaskList. If an open (pending or in_progress) task titled "Fix CI: <check-name>" already exists, skip to step 3 — don't create a duplicate.

### 2. Create a task
TaskCreate with title `Fix CI: <check-name>` and the failure details in the description.

### 3. Implement the fix
1. TaskUpdate → in_progress
2. Read the failure log. Identify root cause.
3. Fix the code. Follow CLAUDE.md.
4. Verify (CLAUDE.md test command; default: `make test`). Fix before continuing.
5. Commit with a descriptive message and push:
   ```bash
   git commit -m "<descriptive message>"
   git push
   ```
6. TaskUpdate → completed

### 4. Sync the work queue
Fetch PR body, update the block between the HTML comment markers, write it back.

## Done when
Fix committed, pushed, task completed, work queue synced.

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft. That is the user's decision.
- **Never** rebase, amend, or force-push. New commits only.
