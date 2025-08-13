#!/usr/bin/env bash
# Run the same checks as CI: lint, format, types, tests.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff check"
ruff check .

echo "==> ruff format --check"
ruff format --check .

echo "==> mypy"
mypy tessera

echo "==> pytest"
pytest "$@"
