#!/bin/sh
# Single verification entry point: `make verify` locally, default CMD in Docker.
# Fails on the first broken step. No network, no GPU, no third-party packages.
set -eu
PY="${PY:-python3}"
cd "$(dirname "$0")/.."

echo "== interpreter"
"$PY" --version

echo "== unit tests"
"$PY" -m unittest discover -s tests -t .

echo "== QA detector on golden (fails only on critical findings)"
"$PY" -I qa/qa_golden.py tests/fixtures/golden_v0.1/episode

echo "== stress, 5 seeds of the generator, verifier V and oracle (full run: make stress N=300)"
"$PY" scripts/stress.py --n 5

echo "verify: OK"
