#!/usr/bin/env python
"""
Upload the bundled weights to the GitHub Release of the current version.

The weights live in ``jet/data/`` and are not tracked by git. Once a release
has been cut (a merge to ``main`` with a ``bump:*`` label creates the tag and
the release), attach the current ``.npz`` files to it so that
``Emulator.load`` can fetch them automatically on first use::

    export GITHUB_TOKEN=ghp_...            # fine-grained token, Contents: read & write
    python tools/publish_data.py              # upload jet/data/*.npz to the VERSION tag
    python tools/publish_data.py --tag v0.1.0 # a specific tag
    python tools/publish_data.py --dry-run    # list what would be uploaded

No external CLI is needed: the upload goes through the GitHub REST API with
standard-library ``urllib``. An asset with the same name is replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_REPO = "czymh/jet"

_API = "https://api.github.com/repos"
_UPLOADS = "https://uploads.github.com/repos"


class _APIError(RuntimeError):
    """A GitHub API call failed; carries the HTTP status code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _version() -> str:
    """Read the version from the top-level VERSION file."""
    return (Path(__file__).resolve().parent.parent / "VERSION").read_text().strip()


def _data_files() -> list[Path]:
    """Return the weight files under jet/data/, sorted by name."""
    data = Path(__file__).resolve().parent.parent / "jet" / "data"
    return sorted(p for p in data.glob("*.npz"))


def _token() -> str:
    """Return a GitHub token from the environment, or explain how to get one."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("JET_DATA_TOKEN")
    if not token:
        raise SystemExit(
            "no GitHub token found: set GITHUB_TOKEN (or JET_DATA_TOKEN) to a "
            "fine-grained token with 'Contents: read and write' on the repository."
        )
    return token


def _request(
    method: str,
    url: str,
    token: str,
    *,
    data: bytes | None = None,
    accept: str = "application/vnd.github+json",
    content_type: str | None = None,
) -> dict:
    """One GitHub REST call; returns the decoded JSON (empty for 204s)."""
    headers = {
        "Accept": accept,
        "Authorization": f"Bearer {token}",
        "User-Agent": "jet",
    }
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise _APIError(
            exc.code,
            f"GitHub API {method} {url} failed with {exc.code} {exc.reason}: {detail[:300]}",
        ) from None
    return json.loads(body) if body else {}


def _release(tag: str, repo: str, token: str) -> dict:
    """Fetch the release for ``tag``, with a helpful message when it is absent."""
    url = f"{_API}/{repo}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    try:
        return _request("GET", url, token)
    except _APIError as exc:
        if exc.code == 404:
            raise SystemExit(
                f"release {tag!r} does not exist yet. Cut it by merging to main "
                "with a bump:* label, then re-run this script."
            ) from None
        raise


def _replace_asset(release: dict, path: Path, repo: str, token: str) -> None:
    """Upload ``path`` to the release, deleting a same-named asset first."""
    release_id = release["id"]
    name = path.name

    existing = [a for a in release.get("assets", []) if a["name"] == name]
    if existing:
        delete_url = f"{_API}/{repo}/releases/assets/{existing[0]['id']}"
        _request("DELETE", delete_url, token)

    upload_url = (
        f"{_UPLOADS}/{repo}/releases/{release_id}/assets"
        f"?name={urllib.parse.quote(name)}"
    )
    with open(path, "rb") as handle:
        payload = handle.read()
    _request(
        "POST", upload_url, token, data=payload, content_type="application/octet-stream"
    )


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

    token = _token()
    release = _release(tag, args.repo, token)
    for path in files:
        _replace_asset(release, path, args.repo, token)
        print(f"  uploaded {path.name}")
    print(f"uploaded {len(files)} file(s) to {args.repo}@{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
