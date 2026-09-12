#!/usr/bin/env bash
#
# gravity-user.sh -- make `jet` importable for a member of the group.
#
#   bash gravity-user.sh      set up, then print what to put in a new shell                                                             
#   source gravity-user.sh    set up and put the interpreter on PATH here
#
# What it does, and why each part is needed:
#
#   * jet is pip-installed into the caller's own ~/.local, on top of the shared
#     conda environment. That environment is readable but not writable by its
#     group, so nothing is installed into it; the scientific stack (numpy,
#     scipy, scikit-learn, torch, h5py, emcee) is inherited from there and pip
#     leaves it alone because those requirements are already satisfied.
#   * jet comes from the dev branch, so re-running follows new commits.
#   * Only an interpreter of the same major.minor version sees
#     ~/.local/lib/pythonX.Y, so the shared environment's bin/ goes on PATH:
#     that is what makes `python` resolve to the interpreter that can actually
#     import jet. Pass JET_NO_PATH=1 to leave PATH alone.
#   * The bundled weights (a few megabytes of .npz, not tracked by git) live in
#     the maintainer's checkout. $JET_DATA_DIR points at them -- both in this
#     shell and inside the registered Jupyter kernel, which is why the path is
#     written into the kernelspec rather than left to shell inheritance. The
#     kernelspec names the interpreter by absolute path, so it works whichever
#     jupyter starts it.
#
# Re-running is cheap and safe. Knobs:
#   JET_REINSTALL=1     re-download jet even if it already imports
#   JET_NO_KERNEL=1     skip the Jupyter kernel registration
#   JET_NO_PATH=1       leave PATH alone
#   JET_SHARED_ENV=...  use a different conda environment as the base
#   JET_SHARED_DATA=... point at a different copy of the bundled weights

SELF="${BASH_SOURCE[0]}"

SHARED_ENV="${JET_SHARED_ENV:-/home/chenzhao/.conda/envs/jet}"
SHARED_DATA="${JET_SHARED_DATA:-/home/chenzhao/worksoftware/jet/jet/data}"
REPO="${JET_REPO:-git+https://github.com/czymh/jet.git@dev}"
KERNEL_NAME="${JET_KERNEL_NAME:-jet}"
PY="$SHARED_ENV/bin/python"

say() { printf '==> %s\n' "$*"; }

gravity_setup() {
    command -v git >/dev/null 2>&1 || { say "git is missing; pip cannot fetch the repository"; return 1; }
    [ -x "$PY" ] || { say "no python at $PY (set JET_SHARED_ENV)"; return 1; }
    [ -r "$SHARED_DATA" ] || { say "cannot read $SHARED_DATA (set JET_SHARED_DATA)"; return 1; }

    say "interpreter     : $PY"
    say "bundled weights : $SHARED_DATA"
    say "installing into : $("$PY" -c 'import site; print(site.getusersitepackages())')"

    if [ "${JET_REINSTALL:-0}" = "1" ] || ! "$PY" -c 'import jet' >/dev/null 2>&1; then
        say "installing jet from $REPO"
        "$PY" -m pip install --user --quiet --upgrade "$REPO" || return 1
    else
        say "jet already imports (JET_REINSTALL=1 to refresh)"
    fi

    if [ "${JET_NO_KERNEL:-0}" != "1" ]; then
        if ! "$PY" -c 'import ipykernel' >/dev/null 2>&1; then
            say "installing ipykernel"
            "$PY" -m pip install --user --quiet ipykernel || return 1
        fi
        say "registering the Jupyter kernel '$KERNEL_NAME'"
        "$PY" -m ipykernel install --user \
            --name "$KERNEL_NAME" --display-name "jet (dev)" >/dev/null || return 1
        # ipykernel offers no way to set environment variables, and the kernel
        # is started by whichever jupyter the user happens to run, so the data
        # path is baked into the kernelspec instead of inherited from a shell.
        "$PY" -c '
import json, sys
from pathlib import Path
name, data = sys.argv[1], sys.argv[2]
path = Path.home() / ".local/share/jupyter/kernels" / name / "kernel.json"
spec = json.loads(path.read_text())
spec.setdefault("env", {})["JET_DATA_DIR"] = data
path.write_text(json.dumps(spec, indent=1) + "\n")
' "$KERNEL_NAME" "$SHARED_DATA" || return 1
    fi

    export JET_DATA_DIR="$SHARED_DATA"
    if [ "${JET_NO_PATH:-0}" != "1" ]; then
        case ":$PATH:" in
            *":$SHARED_ENV/bin:"*) ;;
            *) PATH="$SHARED_ENV/bin:$PATH"; export PATH ;;
        esac
    fi
    
    "$PY" -c 'import jet; print("==> jet", jet.__version__)' || return 1
    return 0
}

if [ "$SELF" = "$0" ]; then
    set -euo pipefail
    gravity_setup "$@" || exit 1
    cat <<EOF

Done. In a new shell, put the interpreter on PATH first:

    export PATH=$SHARED_ENV/bin:\$PATH
    export JET_DATA_DIR=$SHARED_DATA

Or get both at once by sourcing this script instead of running it:

    source $SELF

In Jupyter, choose the "$KERNEL_NAME" kernel -- it carries the data path
itself, so it works whichever jupyter starts it.
EOF
else
    gravity_setup "$@"
fi