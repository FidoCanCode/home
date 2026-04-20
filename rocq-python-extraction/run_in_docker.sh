#!/bin/sh
set -eu

KEEP_WORKSPACE=0
OUT_DIR=
TARGETS_FILE=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --keep-workspace)
      KEEP_WORKSPACE=1
      shift
      ;;
    --output-dir)
      OUT_DIR=$2
      shift 2
      ;;
    --targets-file)
      TARGETS_FILE=$2
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [ -n "$OUT_DIR" ] && [ -z "$TARGETS_FILE" ]; then
  echo "--output-dir requires --targets-file" >&2
  exit 2
fi

if [ -n "$TARGETS_FILE" ] && [ -z "$OUT_DIR" ]; then
  echo "--targets-file requires --output-dir" >&2
  exit 2
fi

if [ -n "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
fi

if [ "$#" -lt 2 ]; then
  echo "usage: $0 [--keep-workspace] [--output-dir DIR --targets-file FILE] <workdir-under-/tmp/work> <command> [args...]" >&2
  exit 2
fi

WORKDIR=$1
shift

docker_args="--rm -v $PWD:/src:ro"
if [ -n "$OUT_DIR" ]; then
  docker_args="$docker_args -v $OUT_DIR:/out -v $TARGETS_FILE:/targets.txt:ro"
fi

docker run $docker_args rocq-python-extraction:ci \
  bash -euo pipefail -c '
    cp -r /src /tmp/work
    chmod -R u+w /tmp/work
    if [ "$1" = 0 ]; then
      rm -f /tmp/work/dune-workspace
    fi
    cd "/tmp/work/$2"
    shift 2
    "$@"
    if [ -f /targets.txt ]; then
      while IFS= read -r name; do
        cp "_build/default/$name" /out/
      done < /targets.txt
    fi
  ' bash "$KEEP_WORKSPACE" "$WORKDIR" "$@"
