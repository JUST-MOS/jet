#!/usr/bin/bash
# tools/git-commit.sh - gate, commit and push with a single command.
#
# Usage:
#   tools/git-commit.sh "<message>" [path1 path2 ...] [--amend] [--no-push] [--yes]
#
#   <message>  required; the commit message
#   paths      optional; commit only these files/dirs (default: everything)
#   --amend    amend the previous commit instead of creating a new one
#   --no-push  commit but do not push
#   --yes      skip the interactive confirmation
#
# The script refuses to commit directly on main and never publishes data:
# freshly built *.npz bundles only produce a warning, because they belong on a
# GitHub Release (tools/publish_data.py), not in the git history.
set -euo pipefail

# --- helpers -----------------------------------------------------------------
die() { echo "error: $*" >&2; exit 1; }

# --- parse arguments ----------------------------------------------------------
AMEND=0
NO_PUSH=0
YES=0
MESSAGE=""
PATHS=()

for arg in "$@"; do
    case "$arg" in
        --amend)  AMEND=1 ;;
        --no-push) NO_PUSH=1 ;;
        --yes)    YES=1 ;;
        -*)       die "unknown flag: $arg" ;;
        *)
            if [ -z "$MESSAGE" ]; then
                MESSAGE="$arg"
            else
                PATHS+=("$arg")
            fi
            ;;
    esac
done

[ -n "$MESSAGE" ] || die "usage: tools/git-commit.sh <message> [paths...] [--amend|--no-push|--yes]"

# --- refuse direct commits on main --------------------------------------------
BRANCH=$(git branch --show-current)
[ "$BRANCH" = "main" ] && die "refusing to commit directly on main; work on a feature branch"

# --- gates: format, lint, tests ------------------------------------------------
command -v ruff >/dev/null 2>&1 || die "ruff not found (pip install ruff)"
echo "==> ruff format --check"
ruff format --check . || die "format check failed; run 'ruff format .'"
echo "==> ruff check"
ruff check . || die "lint check failed; run 'ruff check . --fix'"

echo "==> tests"
if command -v pytest >/dev/null 2>&1; then
    pytest -q || die "pytest failed"
else
    python -m unittest discover -s tests || die "unittest failed"
fi

# --- warn about data changes (never stage or publish them) ---------------------
python - "$(git log -1 --format=%ct 2>/dev/null || echo 0)" <<'PY'
import os
import sys

cutoff = int(sys.argv[1])
for root in ("jet/data", "weights", os.environ.get("JET_DATA_DIR", "")):
    if not root or not os.path.isdir(root):
        continue
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name.endswith(".npz") and os.path.getmtime(path) > cutoff:
            print(
                f"WARN: {path} changed since the last commit; publish it via "
                "a GitHub Release, then run tools/publish_data.py (never commit it)"
            )
PY

# --- stage ----------------------------------------------------------------------
if [ "${#PATHS[@]}" -gt 0 ]; then
    git add -- "${PATHS[@]}"
else
    git add -A
fi

if git diff --cached --quiet; then
    echo "nothing staged; nothing to commit"
    exit 0
fi

# --- confirm ---------------------------------------------------------------------
if [ "$YES" -eq 0 ]; then
    echo "==> staged changes:"
    git status --short
    read -r -p "commit and push? [y/N] " ans
    [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "aborted"; exit 0; }
fi

# --- commit ----------------------------------------------------------------------
if [ "$AMEND" -eq 1 ]; then
    git commit --amend -m "$MESSAGE"
else
    git commit -m "$MESSAGE"
fi

# --- push + PR link ----------------------------------------------------------------
if [ "$NO_PUSH" -eq 0 ]; then
    git push -u origin "$BRANCH"
    repo=$(git remote get-url origin 2>/dev/null | sed -E 's#^.*github\.com[/:]##; s#\.git$##' || true)
    if [ -n "$repo" ]; then
        echo "PR: https://github.com/$repo/pull/new/$BRANCH"
    fi
fi


