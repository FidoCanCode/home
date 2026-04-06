A CI check is failing. All context (PR, repo, branch, check name, failure log) is in the Context section above.

## Steps

### 1. Implement the fix
1. Read the failure log. Identify root cause.
2. Fix the code. Follow CLAUDE.md.
3. Verify (CLAUDE.md test command; default: `make test`). Fix before continuing.
4. Commit with a descriptive message and push:
   ```bash
   git commit -m "<descriptive message>"
   git push
   ```

### 2. Mark complete
```bash
bash /home/rhencke/workspace/kennel/task-cli.sh <work_dir> complete "CI failure: <check-name>"
```

Do NOT use TaskCreate, TaskUpdate, TodoWrite, or any other task tools. Only `task-cli.sh`.
Do NOT edit the PR body directly. `sync-tasks.sh` owns the PR body work queue.

## Done when
Fix committed and pushed, task marked complete.

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft.
- **Never** rebase, amend, or force-push. New commits only.
- **Never** use TaskCreate, TaskUpdate, TaskList, TodoWrite, or TodoRead. Only `task-cli.sh`.
- **Never** edit the PR body directly.
