#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

branch="$(git branch --show-current)"
test -n "$branch"
git diff --quiet
git diff --cached --quiet

bump="${1:-patch}"
version="$(
  uv run python - "$bump" <<'PY'
from pathlib import Path
import re
import sys

bump = sys.argv[1]
path = Path("pyproject.toml")
text = path.read_text()
m = re.search(r'^(version = ")(\d+)\.(\d+)\.(\d+)(")$', text, re.M)
if not m:
    raise SystemExit("version not found")
major, minor, patch = map(int, m.group(2, 3, 4))
if bump == "major":
    major, minor, patch = major + 1, 0, 0
elif bump == "minor":
    minor, patch = minor + 1, 0
elif bump == "patch":
    patch += 1
else:
    raise SystemExit("usage: scripts/release.sh [patch|minor|major]")
version = f"{major}.{minor}.{patch}"
path.write_text(text[:m.start()] + f'{m.group(1)}{version}{m.group(5)}' + text[m.end():])
print(version)
PY
)"

uv run hftbt check
rm -rf dist
uv build
uv run twine upload --skip-existing dist/*
git add pyproject.toml uv.lock
git commit -m "Release v$version"
git push origin "$branch"
