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
pip install jet[class]       # + CLASS, for total-matter sigma8 targets
pip install jet[camb]        # + CAMB, for sigma8 and out-of-range fallbacks
pip install jet[boltzmann]   # + both Boltzmann solvers
pip install jet[all]         # everything
```

> [!NOTE]
> On a cluster, install PyTorch the way the site expects (module or conda) before
> installing `jet`. `pip install jet[all]` will otherwise replace an
> CUDA-matched PyTorch with the generic PyPI build.

From source, for development:

```bash
git clone git@github.com:JUST-MOS/jet.git
cd jet
pip install -e ".[test,gp]"
pytest
```

Or straight from GitHub:

```bash
pip install git+https://github.com/JUST-MOS/jet.git@<tag>
```

## Cosmological parameters

Every bundled emulator (P(k), HMF, ...) is trained on the same flat
**w0waCDM** cosmology with eight free parameters. The column order of any
`theta` array is fixed and shared across models:

| # | name | bounds | meaning |
|---|------|--------|---------|
| 0 | `Omegab` | (0.04, 0.06) | baryon density |
| 1 | `Omegam` | (0.24, 0.40) | total matter density (not include neutrino) |
| 2 | `H0` | (60.0, 80.0) | Hubble constant [km/s/Mpc] |
| 3 | `ns` | (0.92, 1.00) | scalar spectral index |
| 4 | `As` | (1.7e-9, 2.5e-9) | primordial amplitude (not A_s×10⁹) |
| 5 | `w0` | (-1.3, -0.7) | dark-energy equation of state today |
| 6 | `wa` | (-0.5, 0.5) | running of the equation of state |
| 7 | `mnu` | (0.0, 0.3) | sum of neutrino masses [eV] |

Don't memorise the table; read it from the spec, which is the single source
of truth for names and bounds:

```python
from jet.emulator.hmf import theta_spec

spec = theta_spec()
print(spec.names)  # the 8 names, in column order
print(spec.bounds_array())  # (8, 2) lower/upper bounds per column
```

## Quickstart

> [!NOTE]
> The trained weights for the emulators below are not shipped with the
> package. They are fetched automatically on first use: simply load the
> corresponding emulator on a machine with internet access (e.g. a login
> node), and the bundle is downloaded from the project's GitHub Release
> and cached locally. Alternatively, download the released `.npz` files
> manually and point `JET_DATA_DIR` at a directory containing them.

### Halo mass function

`HMFEmulator` predicts the cumulative halo abundance `n(>= M)` and its
derivative `dn/dlnM`, in the `RockstarM200m` mass definition, over the same
eight cosmological parameters:

```python
from jet.emulator.hmf import HMFEmulator

## Omegab, Omegacb, H0, ns, As, w0, wa, mnu
theta = [0.049, 0.31, 67.66, 0.9665, 2.1e-9, -1.0, 0.0, 0.06]
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

### Transformation between sigma8 to As

> [!NOTE]
> The emulator only accepts the trainned parameter set. If
> you want use sigma8 instead of As, use the following script
> to transfer sigma8 to As and input `theta_as` into the hmf emulator.

```python
import numpy as np
from jet.cosmology import cosmo_parameter_spec
from jet.derived import Sigma8FromAs
from jet.spec import ParameterSpec

# The spec the bundled emulators use, plus sigma8 as an alternative name for As.
spec = ParameterSpec(list(cosmo_parameter_spec()), derived=[Sigma8FromAs()])
frame = spec.frame("sigma8")

# columns: Omegab, Omegam, H0, ns, sigma8, w0, wa, mnu
# Note Omegam only contains the cold dark matter and baryon contributions, not neutrinos.
theta_sigma8 = np.array([[0.049, 0.31, 67.66, 0.9665, 0.81, -1.0, 0.0, 0.06]])

theta_as = frame.to_base(theta_sigma8)
theta_as[0, 4]  # array(2.1025e-09)
print(theta_as)
theta = frame.from_base(theta_as)  # back to 0.81
```

`sigma8` here is the **total-matter** value, including massive neutrinos. The
inversion uses the total-matter spectrum composed in `jet.emulator.pkmm`; the
halo mass function itself still consumes `P_cb`.

### Halo mass function with baryonic feedback

`BCMHFEmulator` predicts the cumulative abundance with the baryon correction
model of Arico et al., over four **feedback parameters and no cosmological
ones**:

```python
from jet.emulator.hmf_bcm import BCMHFEmulator, PARAMETER_DEFAULTS, mass_grid

## logMc, thej, mu, delta
theta_default = [15, 1.5, 3.5, 4.5]
emu = BCMHFEmulator.load()
n = emu.cumulative(theta_default[None, :])  # (1, 46) in (h/Mpc)^3
d = emu.dndlgM(theta_default[None, :])  # dn/dlog10(M)
M = mass_grid()  # the 46 masses, in Msun/h — the grid travels in the bundle
```

`theta` is an `(n_samples, 4)` array — or a single row — in `theta_spec()` order:
`logMc`, `thej`, `mu`, `delta`.

Three things set it apart from the model above:

- **It takes no cosmological parameters.** This is the one bundled model whose
  input is not a cosmology; its four parameters are declared under
  `block="baryon"`, so they can be concatenated with a cosmological spec when a
  joint model is eventually needed. That is what `ParameterSpec.subset()` is for.
- **It is one cosmology at one redshift.** The weights encode `c0000` at
  `z = 0.5` and there is no axis along which either can vary. The bundle records
  both as provenance; neither is extrapolated over.
- **The mass grid is fixed and rectangular.** No interpolation, no baseline, no
  per-redshift blocks — the reference is a plain regression of `log10 n(>=M)`
  onto four numbers, and the module is correspondingly small.

See [`examples/hmf_bcm_example.py`](examples/hmf_bcm_example.py) for a runnable
walkthrough.

### Halo-matter correlation function

`XiHMEmulator` predicts the cross-correlation between a halo sample and the
matter field, over the same eight cosmological parameters. The sample can be
selected either by cumulative number density or by mass:

```python
from jet.emulator.xihm import XiHMEmulator

## Omegab, Omegacb, H0, ns, As, w0, wa, mnu
theta = [0.049, 0.31, 67.66, 0.9665, 2.1e-9, -1.0, 0.0, 0.06]
emu = XiHMEmulator.load()
z, r = [0.0, 0.5, 1.5], np.logspace(-1.0, 1.5, 12)

xihm = emu.xihm_lgnbar_threshold(theta, z=z, r=r, lgden=[-5.0, -4.0])  # (n, 2, 3, 12)
xihm = emu.xihm_mass(theta, z=z, r=r, M=[1e13, 1e14])  # (n, 2, 3, 12)

# The three pieces, if you want them separately
box = emu.bias_ratio(theta)  # the raw emulated quantity, (n, 2664)
b = emu.brhm(theta, z=z, lgden=[-5.0], r=r)  # the bias ratio on your own axes
xi_mm = emu.ximm(theta, z=z, r=r)  # linear matter correlation function
bias = emu.bias_lgnbar_threshold(theta, z=z, lgden=[-5.0, -4.0])  # the halo bias
```

`theta` is an `(n_samples, 8)` array — or a single row — in `theta_spec()` order.
`lgden` is `log10` of the cumulative number density in `(h/Mpc)^3`, so a *larger*
value is a *lower* mass threshold and a weaker bias; `r` is in `h^-1 Mpc` and
`M` in `Msun/h`.

Four things are worth knowing before using it:

- **The emulated quantity is a ratio, not the correlation function.** The
  Gaussian process predicts `B_hm = xi_hm / xi_mm`; `XiHMEmulator` multiplies
  back in a matter correlation function it computes by Hankel-transforming the
  bundled linear `P(k)` (`jet.fourier`). At large separations it blends onto a
  linear-bias baseline, `b * P_lin` transformed — so this is where the halo
  *bias* gets used, and `jet.emulator.bhm` is what supplies it.
- **`z`, `lgden` and `r` are interpolation axes, not emulator axes.** The
  Gaussian process sees the eight cosmological parameters only; the other three
  are handled afterwards by linear interpolation of the box, linear in
  `log10(r)`. Asking for a point outside the box on these three axes warns and
  extrapolates rather than raising; set
  `XiHMEmulator(warn_on_extrapolation=False)` to silence them. The eight
  cosmological parameters are different: an out-of-range `theta` always raises
  `ValueError`.
- **Density threshold and mass threshold are different samples.** Holding
  `n(>=M)` fixed and varying `M` is not the same selection as the reverse, so
  `xihm_mass` converts by a finite difference in `log10(n)`, weighted by the
  density — the same one per cent step the reference uses.
- **The result is real-space and matter-only.** No redshift-space distortions,
  no HOD, no survey selection.

Neither entry point needs the bias bundle for its emulated term; the baseline is
what needs it.

See [`examples/xihm_example.py`](examples/xihm_example.py) for a runnable
walkthrough.

## Out-of-range parameters

Every bundled cosmological emulator treats `Param.bounds` as a **validity**
range, not merely the extent of the training sample. By default, a `theta` row
outside those bounds raises `ValueError` and lists every offending parameter,
rather than silently extrapolating. The same check is available directly as
`ParameterSpec.require_within_bounds`.

The strict check covers the eight cosmological columns of `theta`. The
independent interpolation axes (`z`, `lgden`, `r`) keep their existing warning
behaviour.

Where an analytic fallback exists, it can be enabled explicitly:

```python
hmf = HMFEmulator.load(fallback_to_camb=True)
n = hmf.number_density(theta, z=z, M=M)
```

- `HMFEmulator(..., fallback_to_camb=True)` evaluates out-of-range rows with a
  linear CAMB CDM+baryon `P(k)` and the pure Castro23 baseline (the emulated
  ratio is set to one). In-range rows in the same batch still use the emulator.
  A `UserWarning` names the rows and parameters that were replaced.
- `Sigma8FromAs(..., fallback_to_camb=True)` uses CAMB total-matter `sigma8`
  (`delta_tot`) for out-of-range rows. `to_base` checks the input and the final
  solution; the iteration itself may explore intermediate amplitudes outside
  the box.
- `spectral_equivalence_as` defaults to `sigma8_type="emulator"` and
  `fallback_to_camb=True`. Both the target and the auxiliary sigma8 use the
  total-matter emulator; each side that leaves the box is replaced by CAMB,
  with a `UserWarning`. Set `fallback_to_camb=False` to make either side raise
  `ValueError` instead. `sigma8_type="class"` or `"camb"` uses that Boltzmann
  backend for **both** sides, so there is no emulator box and
  `fallback_to_camb` is ignored.

`BhmEmulator`, `XiHMEmulator` and `BCMHFEmulator` never fall back: their
emulated quantity has no analytic counterpart that preserves its meaning, so
an out-of-range input is always an error.

## Spectral equivalence

[`jet.equivalence`](jet/equivalence.py) ports the CSST emulator's
spectral-equivalence method. For a flat `w0waCDM` cosmology it constructs an
auxiliary model at each redshift whose comoving distance to the last-scattering
surface (`z* = 1100`) matches the original model, and then rescales `As` so that
the **total-matter** `sigma8(z)` (including massive neutrinos) matches as well:

```python
import numpy as np
from jet.equivalence import spectral_equivalence_as, spectral_equivalence_de

# Omegab, Omegam, H0, ns, As, w0, wa, mnu
theta = np.array([0.05, 0.31, 68.0, 0.965, 2.1e-9, -1.1, 0.2, 0.06])
z = np.array([0.5, 1.0, 2.0])

w_const = spectral_equivalence_de(theta, z, wcdm=True)
w0, wa, err = spectral_equivalence_de(theta, z, return_error=True)
As_eq = spectral_equivalence_as(theta, z, w0, wa)
```

`Omegam` is baryons plus cold dark matter, excluding massive neutrinos, and
`As` is the physical amplitude. Both functions accept a single eight-column
row or a batch of rows and always keep the sample axis, returning `(1, n_z)` or
`(n, n_z)`.

`sigma8_type` controls the **entire** sigma8 calculation, not just the target:

- `"emulator"` (default) uses the bundled total-matter emulator for both the
  target and the auxiliary model. Each side outside the emulator box falls back
  to CAMB when `fallback_to_camb=True` (the default); set it to `False` to raise
  instead. The fallback is per side.
- `"class"` or `"camb"` uses that Boltzmann backend for both sides; no emulator
  bounds are involved and `fallback_to_camb` is ignored.

The emulator/CAMB comparison assumes the reference's `Nncdm = 1` massive
neutrino convention (one massive species for `mnu > 0`, massless otherwise).
See [`examples/spectral_equivalence_example.py`](examples/spectral_equivalence_example.py)
for a runnable walkthrough.

## Repository layout

```
jet/
├── jet/                    # the Python package (standard flat layout)
│   ├── spec.py             # ParameterSpec / DataVectorSpec, shared by all stages
│   ├── derived.py          # derived parameters: sigma8 <- As, h <- H0
│   ├── cosmology.py        # background expansion the ported physics needs
│   ├── equivalence.py      # spectral equivalence: distance and sigma8 matching
│   ├── fourier.py          # FFTLog Hankel transforms (P(k) <-> xi(r))
│   ├── emulator/           # transforms, backends, bundle format, metrics
│   │   ├── pklin.py        # reader for the bundled linear-theory P(k) emulator
│   │   ├── hmf.py          # halo mass function: the emulator and its baseline
│   │   ├── hmf_bcm.py      # the same with baryonic feedback: no cosmology at all
│   │   ├── bhm.py          # halo bias: the emulator, the baseline, the inversion
│   │   └── xihm.py         # halo-matter correlation function, composed from those
│   ├── data/               # bundled weights (not tracked by git; see its README)
│   ├── catalog/            # design stage
│   ├── estimator/          # design stage
│   └── inference/          # design stage
├── tools/                  # one-off scripts that regenerate bundled data
│   ├── build_pklin_bundle.py
│   ├── build_hmf_bundle.py
│   ├── build_hmf_bcm_bundle.py
│   └── build_xihm_bundle.py
├── tests/                  # unittest suite, run via pytest
├── examples/               # usage notebooks
│   ├── hmf.ipynb           #   halo mass function quick start
│   ├── hmf_bcm_example.py  #   baryonic mass function quick start
│   ├── xihm_example.py     #   halo-matter bias ratio quick start
│   └── equivalence_example.py  # distance and sigma8 matching quick start
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
conda env create -f environment.yml      # or: conda activate /path/to/existing/env
conda activate jet
pip install -e ".[test,gp]"
```

If you are the user of Gravity cluster, you can use 
```bash
source gravity-user.sh
```
to load my conda env and setup the path of trained data file for a quick usage.

## For development

This section targets advanced users who intend to train their own emulators
instead of consuming the pretrained bundles. **If you just want to run an
existing emulator, skip this entire section.**

### Train your own emulator in jet

> [!WARNING]
> The snippet below is **illustrative pseudo-code**, not a runnable script.
> If you actually want to train your own emulator, read the module docstrings
> of `jet.emulator` and the reference builder scripts under `tools/` carefully
> first. Getting the data-vector layout, the transform chain, or the bundle
> format wrong does not raise an error — it silently produces an emulator
> that looks fine but predicts garbage.

```python
import numpy as np
from jet.spec import ParameterSpec, Param, DataVectorSpec
from jet.emulator import Emulator

# 1. Declare your specs. The column order of X below must match x_spec
#    exactly; the transform chain (e.g. log) is derived from y_spec.log,
#    so getting either wrong is a silent failure.
x_spec = ParameterSpec([...])  # <- your input parameters
y_spec = DataVectorSpec("my_stat", n_bins=..., log=True)  # <- your data vector

# 2. Your own training set: (n_samples, n_params) and (n_samples, n_bins).
X, y = load_my_training_data()  # <- you must provide this

# 3. Build, fit, save. The saved .npz is later picked up by Emulator.load().
emu = Emulator(x_spec, y_spec, backend="gp", n_pca=8)
emu.fit(X, y)
emu.save("my_emulator.npz")
```

Loading needs neither scikit-learn nor PyTorch:

```python
from jet.emulator import Emulator

emulator = Emulator.load("wp.gp.npz")  # resolved against $JET_DATA_DIR
emulator.verify(x_spec=x_spec)  # optional, catches a column reordering
```

### Derived parameters

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

### Weight files

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
