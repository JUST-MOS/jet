# AGENTS.md

Guidelines for AI coding assistants working in this repository. `jet` is a
standard-layout (flat) Python package with ACM-style version management
(bump2version + `VERSION` file + CI workflows).

## Repository layout (standard flat layout)

- `jet/` — the Python package at the repository root (NOT `py/jet/`, NOT `src/`). Do not move it.
  - `__init__.py` — exposes `__version__` via `importlib.metadata.version("jet")`; do not read the VERSION file from here
- `tests/` — tests at the repository root, organized by module (`tests/test_<module>.py`).
- `VERSION` — single source of truth for the version; one line (`X.Y.Z` or `X.Y.Z-dev.N`). **Never edit by hand.**
- `.bumpversion.cfg` — bump2version config; `current_version` must always match `VERSION`
- `pyproject.toml` — metadata + tool config; `[tool.setuptools.dynamic] version = {file = "VERSION"}`
  and `[tool.pytest.ini_options] pythonpath = ["."]` are intentional
- `.github/workflows/` — CI (format/lint/test + version bump workflows)
- `README.md`, `LICENSE` (BSD-3-Clause recommended), `.gitignore`

## Naming

- Package name `jet`: short, lowercase, **no hyphens** (it is exported as an env var; shells reject hyphens).
- Tests: files `test_*.py`, classes `TestXxx(unittest.TestCase)`, methods `test_*`.
- Private helpers/parsers use a leading underscore (`_parse_arguments`); CLI entry point is always `main()` returning an int exit code.

## Tests (unittest style)

- Write tests with standard-library `unittest.TestCase` (CI runs them via pytest; do not switch to bare pytest functions).
- Every test class implements `setUp`/`tearDown` (class-level `setUpClass`/`tearDownClass` when needed).
- Mock external dependencies with `unittest.mock.patch` (e.g. `@patch('sys.argv', ...)`, `@patch('builtins.print')`).
- Import the module under test as a top-level package import: `from jet.<module> import ...`.
- Run: `pip install -e .[test]` then `pytest`. Single test: `pytest tests/test_main.py`.

## Coding style

- numpy-style docstrings (module, function, `Returns`); follow the lint config in `pyproject.toml` before pushing.
- No large data files in the repo. Weights are distributed via GitHub Release and
  auto-fetched on first use (`resolve_weights_path`, disabled by `JET_NO_AUTO_FETCH`);
  run `tools/publish_data.py` after retraining.

## README

- `README.md` (Markdown, auto-rendered on GitHub).
- Structure: status badges (CI / coverage / docs) at top, then Introduction, install command
  (`pip install git+https://github.com/czymh/jet.git@<tag>`), quickstart, directory overview, contributing/license.

## Version management (bump2version, ACM-style)

- Version lives in the top-level `VERSION` file (one line, e.g. `0.1.0-dev.3`).
  Release format `X.Y.Z`; dev format `X.Y.Z-dev.N`.
- **Never edit `VERSION` or `.bumpversion.cfg` by hand.** `bump2version` owns both —
  it rewrites the version, commits, and tags atomically. Run it explicitly only when asked.
- Bump types: `patch`/`minor`/`major` for releases (auto tag created), `dev` for dev increments (no tag).
- CI automates all bumps (see `.github/workflows/`):
  - Merge to `dev` touching `jet/**` → auto `bump2version dev` (no tag).
  - Merge to `main` → the PR **must** carry exactly one of `bump:patch` / `bump:minor` / `bump:major`
    labels (enforced by `check-version-label.yml`); the bump creates a tag + GitHub release, then `main`
    is merged back into `dev` to keep versions in sync.
- On a version conflict when merging: set `current_version` (and `VERSION`) to the target branch's current
  value and let CI bump after merge — never bump manually to "fix" the conflict.
