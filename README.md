# jet

**jet** — **JUST Emulator Toolkit**: a cosmological emulator pipeline for the
[Jiao Tong University Spectroscopic Telescope](https://just.sjtu.edu.cn) (JUST)

> [!WARNING]
> `jet` is a research project in constant evolution. Features may be experimental
> or under development. Check for updates regularly and use with caution.

## Pipeline

`jet` is organized as four modular stages:

| Stage | Responsibility | Status |
| --- | --- | --- |
| `jet.catalog` | halo catalogs → HOD galaxy catalogs | interface only |
| `jet.estimator` | galaxy/halo samples → measured statistics | interface only |
| `jet.emulator` | parameters → statistics; train, save, load, predict | **implemented** |
| `jet.inference` | statistics + covariance → parameter posterior | interface only |

`jet` does not run N-body simulations or halo finders. It takes halo catalogs
produced upstream and works downwards. The shared vocabulary is
[`jet.spec`](jet/spec.py): a `ParameterSpec` names and bounds the input axes, a
`DataVectorSpec` describes the output bins, and every trained model records the
hashes of the specs it was fitted against so a mismatch is caught on load rather
than showing up as a quietly wrong prediction.

## Installation

Core installs with NumPy and SciPy only, which is enough to **load a trained
model and predict**. Training needs an extra:

```bash
pip install jet              # numpy + scipy
pip install jet[gp]          # + scikit-learn, for Gaussian-process training
pip install jet[nn]          # + PyTorch, for neural-network training
pip install jet[inf]         # + emcee, for inference
pip install jet[all]         # everything
```

> [!NOTE]
> On a cluster, install PyTorch the way the site expects (module or conda) before
> installing `jet`. `pip install jet[all]` will otherwise replace an
> CUDA-matched PyTorch with the generic PyPI build.

From source, for development:

```bash
git clone git@github.com:czymh/jet.git
cd jet
pip install -e ".[test,gp]"
pytest
```

Or straight from GitHub:

```bash
pip install git+https://github.com/czymh/jet.git@<tag>
```

## Quickstart

```python
import numpy as np
from jet.spec import ParameterSpec, Param, DataVectorSpec
from jet.emulator import Emulator

x_spec = ParameterSpec(
    [
        Param("Omegab", bounds=(0.04, 0.06), block="cosmo"),
        Param("ns", bounds=(0.92, 1.00), block="cosmo"),
        Param("logMcut", bounds=(12.0, 13.8), block="hod"),
    ]
)
y_spec = DataVectorSpec("wp", n_bins=15, log=True)

# X: (N, 3) raw parameters, y: (N, 15) raw statistics. Arbitrary points, not a grid.
emulator = Emulator(x_spec, y_spec, backend="gp", n_pca=8)
emulator.fit(X, y)

mean, std = emulator.predict(X_new)
emulator.save("wp.gp.npz")
```

Loading needs neither scikit-learn nor PyTorch:

```python
from jet.emulator import Emulator

emulator = Emulator.load("wp.gp.npz")  # resolved against $JET_DATA_DIR
emulator.verify(x_spec=x_spec)  # optional, catches a column reordering
```

## Derived parameters

A parameter that is a function of the others is not a new axis, and `jet` does
not treat it as one. Declaring it on the spec adds no column — it lets the same
spec be *addressed* along either parameter, converting at the boundary:

```python
from jet.derived import Sigma8FromAs

x_spec = ParameterSpec(
    [
        Param("Omegab", ...),
        Param("Omegam", ...),
        Param("H0", ...),
        Param("ns", ...),
        Param("As", bounds=(1.7e-9, 2.5e-9)),
        Param("w0", ...),
        Param("wa", ...),
        Param("mnu", ...),
    ],
    derived=[Sigma8FromAs()],  # sigma8 <-> As
)

emulator = Emulator(x_spec, y_spec).fit(X, y)  # trained on As, as the table has it
emulator.predict(X_new)  # X_new is in As

frame = emulator.x_spec.frame("sigma8")
emulator.predict(frame.to_base(X_sigma8))  # driven by sigma8 instead
```

`X` keeps its `(n_samples, x_spec.dim)` shape in either frame, and the two
agree to the accuracy of the conversion. `Sigma8FromAs` evaluates `sigma8`
through `jet`'s bundled linear-theory `P(k)` emulator and inverts with the same
fixed point `csstemu` uses; see [Derived parameters](jet/derived.py) and
[the bundled data](jet/data/README.md).

The conversion is applied by **you**, not by the model: an emulator reads the
axes of its spec and nothing else. That keeps a parameter array to a single
reading at the point it is consumed, and it means a bundle is bound to the axes
alone — declaring or removing a derived parameter never invalidates one. A
loaded model still carries its declarations, so `emulator.x_spec.derived_names`
tells you which frames are available.

> [!NOTE]
> The bundled weights are not part of the distribution. A wheel or a fresh
> clone has no `jet/data/pklin.gp.npz`, so `Sigma8FromAs` raises
> `FileNotFoundError` explaining how to obtain it. Generate it with
> `python tools/build_pklin_bundle.py`, or point `$JET_DATA_DIR` at a copy.
> Everything else works without it — `Omegam` needs no data at all.

## Halo mass function

`HMFEmulator` predicts the cumulative halo abundance `n(>= M)` and its
derivative `dn/dlnM`, in the `RockstarM200m` mass definition, over the same
eight cosmological parameters:

```python
from jet.emulator.hmf import HMFEmulator

hmf = HMFEmulator.load()
n = hmf.number_density(theta, z=[0.0, 1.0, 3.0], M=[1e12, 1e13, 1e14])  # (1, 3, 3)
d = hmf.dndlnM(theta, z=0.5, M=1e13)
```

`theta` is an `(n_samples, 8)` array — or a single row — in `theta_spec()`
order, and the result is `(n_samples, n_z, n_M)` in `(h/Mpc)^3`. Multiply by a
volume in `(Mpc/h)^3` for an expected halo count. Any redshift or mass inside
the emulator's range may be asked for; both are interpolated.

Two things distinguish this model from the power spectrum:

- **It emulates a ratio, not the mass function.** Recovering the physical
  quantity means multiplying by an analytic Castro23 baseline, which
  `jet.emulator.hmf` computes from the bundled `P(k)` emulator and
  [`jet.cosmology`](jet/cosmology.py). That baseline is why a prediction costs
  about a third of a second rather than a millisecond.
- **Its data vector is not a rectangle.** The reference trained on a different
  mass range at each redshift, so the twelve blocks have different lengths.
  `jet.emulator.hmf.mass_slices()` and `data_slices()` give the ranges.

See [`examples/hmf.ipynb`](examples/hmf.ipynb) for a runnable walkthrough of the
whole interface.

> [!NOTE]
> `jet/data/hmf_rockstar_m200m.gp.npz` is not tracked either. Generate it with
> `python tools/build_hmf_bundle.py`. The baseline's physics works without it;
> only the emulated part raises.

## Weight files

Trained models are tens of megabytes and are retrained often, so they are not
kept in the repository. `Emulator.load` accepts either an absolute path or a
bare name, in which case it is looked up under `$JET_DATA_DIR`:

```bash
export JET_DATA_DIR=/share/users/<you>/jet-weights
```

### Fetching from GitHub Release

Since v0.1.0 the weights are also attached to the GitHub Release of each
version, so a plain `pip install` needs no environment configuration: the first
`Emulator.load("<name>.npz")` fetches the file from the release into the user
cache (`$XDG_CACHE_HOME/jet/data`, defaulting to `~/.cache/jet/data`), and
every later call reads the cache. The fetch lives in
`jet.emulator.resolve_weights_path`; disable it with `JET_NO_AUTO_FETCH`, pin a
release with `JET_RELEASE_TAG`, and point a private repository's token via
`GITHUB_TOKEN` / `JET_DATA_TOKEN`.

Publish the current weights with:

```bash
python tools/publish_data.py          # uploads jet/data/*.npz to the VERSION tag
```

## Repository layout

```
jet/
├── jet/                    # the Python package (standard flat layout)
│   ├── spec.py             # ParameterSpec / DataVectorSpec, shared by all stages
│   ├── derived.py          # derived parameters: sigma8 <- As, h <- H0
│   ├── cosmology.py        # background expansion the ported physics needs
│   ├── emulator/           # transforms, backends, bundle format, metrics
│   │   ├── pklin.py        # reader for the bundled linear-theory P(k) emulator
│   │   └── hmf.py          # halo mass function: the emulator and its baseline
│   ├── data/               # bundled weights (not tracked by git; see its README)
│   ├── catalog/            # design stage
│   ├── estimator/          # design stage
│   └── inference/          # design stage
├── tools/                  # one-off scripts that regenerate bundled data
│   ├── build_pklin_bundle.py
│   └── build_hmf_bundle.py
├── tests/                  # unittest suite, run via pytest
├── examples/               # usage notebooks
│   └── hmf.ipynb           #   halo mass function quick start
├── environment.yml         # shared development environment (conda + pip)
├── VERSION                 # single source of truth for the version (bump2version-owned)
├── .bumpversion.cfg        # version bump configuration
├── pyproject.toml
└── .github/workflows/      # automated version bumping
```

## Development environment

For the group, one shared conda environment is enough; `jet` itself is always
installed with pip on top of it:

```bash
module load anaconda/anaconda-mamba
mamba env create -f environment.yml      # or: conda activate /path/to/existing/env
conda activate jet
pip install -e ".[test,gp]"
```

If you are the user of Gravity cluster, you can use 
```bash
source gravity-user.sh
```
to load my conda env and setup the path of trained data file for a quick usage.

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

Every PR to `main` must carry exactly one version-bump label. See `AGENTS.md` for
the full ruleset for AI-assisted coding.

> [!NOTE]
> There is currently no lint/test CI workflow: `.github/workflows/` only automates
> version bumps. Lint and tests are configured in `pyproject.toml` and are run
> locally (`ruff check`, `ruff format --check`, `pytest`).

## License

`jet` is free software licensed under the BSD-3-Clause license. See `LICENSE`.
