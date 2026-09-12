#!/usr/bin/env python
"""
Upload the bundled weights to the GitHub Release of the current version.

The weights live in ``jet/data/`` and are not tracked by git. Once a release
has been cut (a merge to ``main`` with a ``bump:*`` label creates the tag and
the release), attach the current ``.npz`` files to it so that
``Emulator.load`` can fetch them automatically on first use::

    python tools/publish_data.py              # upload jet/data/*.npz to the VERSION tag
    python tools/publish_data.py --tag v0.1.0 # a specific tag
    python tools/publish_data.py --dry-run    # list what would be uploaded

Requires the ``gh`` CLI, authenticated against the repository (``gh auth
login``). Uploads are idempotent: an asset with the same name is overwritten.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = "czymh/jet"


def _version() -> str:
    """Read the version from the top-level VERSION file."""
    return (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()


def _data_files() -> list[Path]:
    """Return the weight files under jet/data/, sorted by name."""
    data = Path(__file__).resolve().parent.parent / "jet" / "data"
    return sorted(p for p in data.glob("*.npz"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="release tag (default: the VERSION file)")
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"github repository (default: {DEFAULT_REPO})"
    )
    parser.add_argument("--dry-run", action="store_true", help="list files without uploading")
    args = parser.parse_args()

    tag = args.tag or _version()
    files = _data_files()
    if not files:
        print("no .npz files under jet/data/ -- run tools/build_*.py first", file=sys.stderr)
        return 1

    print(f"release {args.repo}@{tag}")
    for path in files:
        print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0

    try:
        subprocess.run(
            ["gh", "release", "view", tag, "--repo", args.repo],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        print(
            "gh CLI not found: install it (https://cli.github.com), then run `gh auth login` once.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError:
        print(
            f"release {tag!r} does not exist yet. Cut it by merging to main with a "
            "bump:* label, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    subprocess.run(
        [
            "gh",
            "release",
            "upload",
            tag,
            *[str(p) for p in files],
            "--repo",
            args.repo,
            "--clobber",
        ],
        check=True,
    )
    print(f"uploaded {len(files)} file(s) to {args.repo}@{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
