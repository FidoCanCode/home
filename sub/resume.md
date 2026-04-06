An existing PR is being continued. Your only job in this step is to restore the task list.

## Steps

1. Fetch the PR body: `gh pr view <pr> --repo <repo> --json body --jq .body`
2. Parse the work queue block (between `WORK_QUEUE_START` and `WORK_QUEUE_END`).
3. Call `TodoWrite` to restore all tasks with their current status:
   - `- [ ]` lines → `todo`
   - `- [x]` lines → `completed`
4. Output a single line: `READY`

## Done when
TodoWrite has been called and you have output `READY`.

**Stop immediately. Do NOT implement any tasks. Do NOT run tests. Do NOT read code files. The bash script determines what to do next.**
