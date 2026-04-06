Unresolved PR review threads need responses. All context (PR, repo, thread JSON) is in the Context section above.

The thread JSON contains only threads that are unresolved and don't already have a final reply — no further filtering needed.

## First: follow up on open ASK tasks

Call TaskList. For each task titled "ASK: ...":
- The task description contains the thread node_id.
- Check the thread's last comment (it's in the JSON above if still unresolved, or re-fetch if needed).
- If the human has replied after the last "" message:
  - **Still unclear** → post another follow-up, update the task description. Leave the task open.
    ```bash
    gh api repos/<owner>/<repo>/pulls/<pr>/comments \
      -X POST -F body="<follow-up question>" \
      -F in_reply_to=<first_db_id>
    ```
  - **Clear enough to act** → create/update/remove tasks as needed. Post a reply. Mark the ASK task completed.
    ```bash
    gh api repos/<owner>/<repo>/pulls/<pr>/comments \
      -X POST -F body="Got it — <brief summary of what will be done>" \
      -F in_reply_to=<first_db_id>
    ```
- If the human has NOT replied since the last "" message: skip.

## Then: process remaining unresolved threads

For each thread in the JSON that does not already have an open task and has not just been handled above:

**Bot threads** (`is_bot: true`):
- **DO** — worth implementing:
  TaskCreate `PR comment: <short summary>` with thread body, URL, node_id, first_db_id in description.
- **DEFER** — useful but out of scope:
  ```bash
  gh issue create --repo <repo> --title "<suggestion>" --body "<context + thread URL>"
  gh api repos/<owner>/<repo>/pulls/<pr>/comments \
    -X POST -F body="Deferring to <issue URL>" -F in_reply_to=<first_db_id>
  gh api graphql -F threadId=<id> \
    -f query='mutation($threadId:ID!){resolveReviewThread(input:{threadId:$threadId}){thread{id}}}'
  ```
- **DUMP** — not applicable:
  ```bash
  gh api repos/<owner>/<repo>/pulls/<pr>/comments \
    -X POST -F body="Declining: <reason>" -F in_reply_to=<first_db_id>
  gh api graphql -F threadId=<id> \
    -f query='mutation($threadId:ID!){resolveReviewThread(input:{threadId:$threadId}){thread{id}}}'
  ```

**Human threads** (`is_bot: false`):
- **ACT** — you know what to do:
  TaskCreate with this exact title format — the link text and hidden comment are how task.md finds the thread to close:
  `[PR comment: <short summary>](<comment_url>) <!-- thread:<node_id> first_db_id:<first_db_id> -->`
  Post acknowledgement reply.
  **Do NOT implement here** — only queue the work. Implementation happens in the next iteration via task.md.
  ```bash
  gh api repos/<owner>/<repo>/pulls/<pr>/comments \
    -X POST -F body="On it — <brief summary>" -F in_reply_to=<first_db_id>
  ```
- **ASK** — unclear what to do:
  ```bash
  gh api repos/<owner>/<repo>/pulls/<pr>/comments \
    -X POST -F body="<focused question>" -F in_reply_to=<first_db_id>
  ```
  TaskCreate `ASK: <short summary>` with thread node_id, human comment, and your question in description.
- **ANSWER** — a question, not a code change request:
  ```bash
  gh api repos/<owner>/<repo>/pulls/<pr>/comments \
    -X POST -F body="<direct answer>" -F in_reply_to=<first_db_id>
  ```
  Do NOT resolve. Do NOT create a task.

## Finally: sync the work queue
If any tasks were created or modified, fetch the PR body, update the block between the HTML comment markers, and write it back:
- Pending tasks: `- [ ] Title` list; the **first** item must have `**→ next**` appended — no exceptions
- Completed tasks: `<details><summary>Completed (N)</summary>` block; omit if none
- `PR comment:` titles rendered as links to thread URLs

## Done when
Every thread in the JSON has been responded to or has an open task.

## Constraints
- **Never** mark the PR as ready for review (`gh pr ready`). It must stay draft. That is the user's decision.
- **Never** call any `/reviews` endpoint (read or write). Use only `pulls/{pr}/comments` with `in_reply_to=<first_db_id>` for thread replies — no pre-checks, no fallbacks.
