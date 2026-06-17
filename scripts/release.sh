#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

branch="$(git branch --show-current)"
test -n "$branch"
git diff --quiet
git diff --cached --quiet

uv run hftbt check
rm -rf dist
uv build
uv run twine upload --skip-existing dist/*
git push origin "$branch"
