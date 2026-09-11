# jet

[![CI](https://github.com/<org>/jet/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/jet/actions)
[![Coverage](https://coveralls.io/repos/github/<org>/jet/badge.svg)](https://coveralls.io/github/<org>/jet)
[![Documentation](https://readthedocs.org/projects/jet/badge/?version=latest)](https://jet.readthedocs.io/en/latest/)

**jet** — **JUST Emulator Toolkit**: a cosmological emulator pipeline for the
[Jiao Tong University Spectroscopic Telescope](https://just.sjtu.edu.cn) (JUST)


> [!WARNING]
> `jet` is a research project in constant evolution. Features may be experimental
> or under development. Check for updates regularly and use with caution.

## Pipeline

`jet` is organized as five modular, independently runnable stages:

1. **Simulations & mocks** — dark-matter boxes → HOD galaxy mocks (box)
2. **Statistics** — halo mass function, halo-matter cross-correlation, galaxy-galaxy lensing, galaxy clustering, etc.
3. **Emulator** — Gaussian Process Regression or neural-network emulator 
4. **Inference** — nested sampling (e.g. PocoMC) with data + emulator covariance,
   plus greedy-Fisher statistic selection

## Installation

Install from source (editable, recommended for development):

```bash
git clone git@github.com:czymh/jet.git
cd jet
pip install -e .[test]
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/czymh/jet.git@<tag>
```

## Quickstart

```python
>>> import jet
>>> jet.__version__   # reads the version from package metadata (VERSION file)
'0.1.0'
```

*Detailed examples will be added under `examples/` as the pipeline matures.*

## Repository layout

```
jet/
├── jet/                  # the Python package (standard flat layout)
│   └── __init__.py
├── tests/                # pytest/unittest suite
├── examples/             # notebooks & usage examples (to come)
├── VERSION               # single source of truth for the version (bump2version-owned)
├── .bumpversion.cfg      # version bump configuration
├── pyproject.toml
└── .github/workflows/    # CI + automated version bumping
```

## Version management

`jet` follows semantic versioning with an ACM-style automation:

- `X.Y.Z` for releases, `X.Y.Z-dev.N` during development.
- The `VERSION` file is the single source of truth and is managed exclusively by
  [`bump2version`](https://github.com/c4urself/bump2version) — do not edit it by hand.
- Merging to `dev` auto-bumps the dev version (no tag); merging to `main` with a
  `bump:patch` / `bump:minor` / `bump:major` label creates a tag and a GitHub release,
  and `main` is merged back into `dev`.

## Contributing

branching model: `main` for stable releases, `dev` for ongoing development.

- `feature/<feature>` — new features, merged into `dev`
- `hotfix/<fix>` — urgent bug fixes, merged into `main` and `dev`
- `docs/**` — documentation improvements

Every PR to `dev` must pass CI (lint + type check + pytest); every PR to `main`
must carry exactly one version-bump label. See `AGENTS.md` for the full ruleset
for AI-assisted coding.

## License

`jet` is free software licensed under the BSD-3-Clause license. See `LICENSE`.

