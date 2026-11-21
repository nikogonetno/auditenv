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

echo "== stress, one seed per config of configs/stress_v3.json: generator, verifier V, oracle, same bytes under PYTHONHASHSEED 0 and 1 (full run: make stress N=300)"
"$PY" scripts/stress.py --n 11

echo "verify: OK"
