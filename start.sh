#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export KENNEL_SECRET=$(cat ~/.kennel-secret)
export KENNEL_WORK_DIR=/home/rhencke/workspace/confusio
export KENNEL_PROJECT=confusio

exec uv run kennel
