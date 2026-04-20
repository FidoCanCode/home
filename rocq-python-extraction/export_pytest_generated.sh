#!/bin/sh
set -eu

OUT_DIR=$PWD/_build/default

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

./rocq-python-extraction/run_in_docker.sh \
  --output-dir "$OUT_DIR" \
  --targets-file "$PWD/rocq-python-extraction/test/generated_pytest_targets.txt" \
  rocq-python-extraction \
  opam exec -- dune build test/python.vo
