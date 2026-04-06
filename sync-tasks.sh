#!/usr/bin/env bash
# sync-tasks.sh — sync Claude Code task list → PR body work queue
# Triggered by: PostToolUse hook, kennel webhook, cron
# Protected by flock to prevent concurrent runs
set -euo pipefail

WORK_DIR="${1:-$PWD}"
cd "$WORK_DIR"

mkdir -p "$HOME/log"
log() {
  local msg
  msg="$(printf '[%s] sync: %s' "$(date '+%H:%M:%S')" "$*")"
  printf '%s\n' "$msg"
  printf '%s\n' "$msg" >> "$HOME/log/sync-tasks.log"
}

# ── Lock ──────────────────────────────────────────────────────────────────
SYNC_LOCK="$(git rev-parse --absolute-git-dir)/fido/sync.lock"
mkdir -p "$(dirname "$SYNC_LOCK")"
exec 8>"$SYNC_LOCK"
flock -n 8 || { log "another sync running — skipping"; exit 0; }

# ── Find current PR ──────────────────────────────────────────────────────
REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
STATE_FILE="$(git rev-parse --git-dir)/fido/state.json"

if [[ ! -f "$STATE_FILE" ]]; then
  log "no state file — nothing to sync"
  exit 0
fi

CURRENT_ISSUE=$(jq -r '.issue // empty' "$STATE_FILE")
if [[ -z "$CURRENT_ISSUE" ]]; then
  log "no current issue — nothing to sync"
  exit 0
fi

GH_USER=$(gh api user --jq .login)
_PR_JSON=$(gh pr list --repo "$REPO" --state open --json number,headRefName,author \
  --search "#$CURRENT_ISSUE in:body" 2>/dev/null \
  | jq -r --arg user "$GH_USER" '[.[] | select(.author.login == $user)] | .[0] // empty')
PR=$(printf '%s' "$_PR_JSON" | jq -r '.number // empty')

if [[ -z "$PR" ]]; then
  log "no open PR for issue #$CURRENT_ISSUE — nothing to sync"
  exit 0
fi

log "syncing task list → PR #$PR"

# ── Read task list via claude ─────────────────────────────────────────────
PROJECT="${CLAUDE_CODE_TASK_LIST_ID:-confusio}"
TASK_OUTPUT=$(CLAUDE_CODE_TASK_LIST_ID="$PROJECT" claude -p "List all tasks with their status (pending/in_progress/completed). Output ONLY lines in this exact format, one per task, no other text:
- [STATUS] TITLE

Where STATUS is one of: pending, in_progress, completed
And TITLE is the full task title exactly as stored." 2>/dev/null || true)

if [[ -z "$TASK_OUTPUT" ]]; then
  log "no tasks returned — skipping"
  exit 0
fi

# ── Format as work queue markdown ─────────────────────────────────────────
PENDING=""
COMPLETED=""
FIRST_PENDING=true

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  # Parse: - [status] title
  if [[ "$line" =~ ^-\ \[(pending|in_progress)\]\ (.+)$ ]]; then
    title="${BASH_REMATCH[2]}"
    if $FIRST_PENDING; then
      PENDING+="- [ ] $title **→ next**"$'\n'
      FIRST_PENDING=false
    else
      PENDING+="- [ ] $title"$'\n'
    fi
  elif [[ "$line" =~ ^-\ \[completed\]\ (.+)$ ]]; then
    title="${BASH_REMATCH[1]}"
    COMPLETED+="- [x] $title"$'\n'
  fi
done <<< "$TASK_OUTPUT"

# Build work queue block
QUEUE=""
if [[ -n "$PENDING" ]]; then
  QUEUE+="$PENDING"
fi
if [[ -n "$COMPLETED" ]]; then
  QUEUE+=$'\n'"<details><summary>Completed ($(echo "$COMPLETED" | grep -c '^- '))</summary>"$'\n\n'
  QUEUE+="$COMPLETED"
  QUEUE+="</details>"$'\n'
fi

# ── Update PR body ────────────────────────────────────────────────────────
CURRENT_BODY=$(gh pr view "$PR" --repo "$REPO" --json body --jq .body)

if ! echo "$CURRENT_BODY" | grep -q "WORK_QUEUE_START"; then
  log "PR #$PR has no work queue markers — skipping"
  exit 0
fi

# Replace content between markers
NEW_BODY=$(echo "$CURRENT_BODY" | python3 -c "
import sys
body = sys.stdin.read()
queue = sys.argv[1]
start = body.find('<!-- WORK_QUEUE_START -->')
end = body.find('<!-- WORK_QUEUE_END -->')
if start == -1 or end == -1:
    print(body, end='')
    sys.exit()
start += len('<!-- WORK_QUEUE_START -->')
new = body[:start] + '\n' + queue + body[end:]
print(new, end='')
" "$QUEUE")

gh pr edit "$PR" --repo "$REPO" --body "$NEW_BODY"
log "PR #$PR work queue synced"
