# Bundled data

Everything in this directory except this file is **excluded from version
control**. It is data, not code: a few megabytes of arrays that would otherwise
be re-downloaded or regenerated into every clone, and that `jet` does not
originate.

## `pklin.gp.npz`

A Gaussian-process emulator for the linear-theory cold-dark-matter + baryon
power spectrum, packaged in jet's own bundle format. It is what backs the
`sigma8` derived parameter (see `jet.derived.Sigma8FromAs`), because jet does
not compute power spectra itself.

It predicts `P(k, z)` in `(Mpc/h)^3` on a flattened grid of 12 redshifts
(`jet.emulator.pklin.Z_GRID`, ascending, `z = 0` first) by 1000 wavenumbers.
The wavenumber grid travels inside the same file, as an `extra` array, so
there is nothing to keep in step by hand. Read `jet/emulator/pklin.py` for the
layout and the accessors.

The halo-mass-function bundle reads its mass grid, its per-redshift block
boundaries and the Castro23 coefficients from its `extra` arrays in the same
way, so it needs no constants beyond the ones it declares itself.

## `hmf_rockstar_m200m.gp.npz`

A Gaussian-process emulator for the halo mass function in the `RockstarM200m`
mass definition, also in jet's own bundle format, backing
`jet.emulator.hmf.HMFEmulator`.

Unlike the power-spectrum bundle this one does **not** hold the physical
quantity. It holds the *ratio* of the mass function to an analytic Castro23
baseline, which is what the reference chose to emulate — the ratio is a smooth
order-unity function, while the mass function itself spans ten orders of
magnitude. `HMFEmulator` multiplies the baseline back in, computing it from the
same power-spectrum bundle above. See `jet/emulator/hmf.py` for the data-vector
layout, which is twelve variable-length per-redshift mass blocks rather than a
rectangle.

It is small, about 65 kB, because it is ten Gaussian processes over ten PCA
components rather than one process per output bin.

## `xihm_brhm_rockstar_m200m.gp.npz`

A Gaussian-process emulator for the halo-matter effective bias ratio
`B_hm = xi_hm / xi_mm`, backing `jet.emulator.xihm.XiHMEmulator`. Same story as
the mass function: the ratio, not the observable, because the ratio is a smooth
function of cosmology while the correlation function inherits all of the
cosmological dependence of the matter power spectrum.

Its box is rectangular — 12 redshifts by 6 cumulative number-density thresholds
by 37 separations — and all three axes travel in the bundle as `extra` arrays,
so a weights file is self-describing. `jet/emulator/xihm.py` keeps the same
values as module constants so that the plumbing can be exercised without the
weights; the two are asserted equal where both are present.

Every component was trained with a Matern kernel of order `nu = 3/2`, which is
the reason `jet.emulator.backends.gp` carries a `"matern32"` key.

## `hmf_bcm_m200m.gp.npz`

The cumulative halo abundance with baryonic feedback, backing
`jet.emulator.hmf_bcm.BCMHFEmulator`. It is the one bundle whose input is **not
a cosmology**: four Arico et al. feedback parameters, declared under
`block="baryon"`, and nothing else.

The weights encode `c0000` at `z = 0.5` and there is no axis along which either
can vary, so unlike every other bundle this one describes a single point in
cosmology and time. The output is a plain 46-point rectangle on a fixed mass
grid, which travels inside the bundle — no baseline, no per-redshift blocks, no
interpolation.

The one thing worth knowing before touching the builder: the reference stores a
fitted `WhiteKernel` noise level per component alongside a `ConstantKernel * Matern`,
and the stored dual coefficients show that the noise term is not in the arithmetic —
they reproduce the plain kernel with the shared `1e-10` floor, not that kernel plus
the stored level. So the bundle carries `1e-10` and records the levels in the
manifest as provenance.

## `bhm_rockstar_m200m.gp.npz`

The halo bias divided by its Castro23 baseline, on the same 12 by 6 axes,
backing `jet.emulator.bhm.BhmEmulator`. It exists because the correlation
function's large-separation baseline is `b * P_lin`, and the bias in it is an
emulated quantity rather than an analytic one.

It is the smallest of the four, about 20 kB, for the same reason the mass
function is small: ten processes over ten PCA components, not one per bin. The
bias is `1.0` to `18` over the emulated range, so it *could* have been emulated
directly; the ratio is emulated instead so that the thing being fitted is
smooth and order one.

`jet.emulator.bhm` also carries the two steps that turn a threshold into a mass
and back. They are not regressions and they need no weights of their own beyond
this file and the mass function above.

### Regenerating

```bash
python tools/build_pklin_bundle.py                 # default source path
python tools/build_pklin_bundle.py --source <dir>  # another checkout

python tools/build_hmf_bundle.py                   # default source path
python tools/build_hmf_bundle.py --source <dir>    # another checkout

python tools/build_xihm_bundle.py                  # default source path
python tools/build_xihm_bundle.py --source <dir>   # another checkout

python tools/build_hmf_bcm_bundle.py               # default source path
python tools/build_hmf_bcm_bundle.py --source <dir>  # another checkout
```

All four scripts read the reference arrays, repackage them, and then **verify
the result against the reference implementation** if that implementation is
importable — refusing the build when the two disagree by more than floating-point
round-off. `build_hmf_bundle.py` checks the regression and the baseline
separately; `build_xihm_bundle.py` checks each layer of the correlation function
separately, because the failures are independent and a single number would not
say which one broke. Run them after changing anything about the transform
chains, the parameter bounds, or the ported physics.

`build_hmf_bcm_bundle.py` is the exception to the "compare against the stored
coefficients" habit, and the reason is worth carrying over to any future port.
Those coefficients are the regression's *targets*, and a Gaussian process with a
non-zero floor does not return its targets: the reference disagrees with its own
stored table by `1.7e-7`, because the `10**` at the end of its chain amplifies a
score-space residual of `1e-9`. This bundle disagrees with them by `1.7e-7` as
well — the same number — which is why that check carries a loose tolerance and
does the real work against the reference's *process* instead.

`build_xihm_bundle.py` builds two files and checks nine things, and the split is
worth understanding before relaxing any of its tolerances:

| check | measured | catches |
|---|---|---|
| Brhm box at the 129 training cosmologies | `3.2e-9` | the transform chain and layout — an unpermuted redshift slab costs `3.6` |
| Brhm box *between* the training cosmologies | `1.9e-10` | the kernel and hyperparameters — Matern 5/2 instead of 3/2 costs `1.7e-2` here but only `1.1e-5` at the training points |
| three-axis interpolation | `1.8e-10` | the coordinate order and `log10(r)` vs `r` |
| `xi_mm` | `2.3e-9` | the FFTLog wiring, the power-spectrum resampling and the `log10(r)` spline |
| `xi_hm`, density threshold | `2.3e-9` | the composition: ratio × `xi_mm`, the bias baseline, the blend |
| `xi_hm`, mass threshold | `4.2e-8` | the mass perturbation, the density-weighted difference and its axis reversal |
| Bhm box, at and between the training points | `1.7e-10` / `7.1e-13` | the same two things for the bias model, which is far better conditioned |
| mass from `lgden` | `3.6e-9` | the inversion of the mass function |
| bias at density and at mass thresholds | `1.2e-9` / `3.6e-9` | the baseline, and the finite difference between the two selections |

The middle row is the non-obvious one. At a training point these Gaussian
processes are exact interpolants, so they reproduce the reference's stored
coefficients whatever kernel they were built with — which means a check confined
to the training set cannot tell the kernels apart at all. It has to probe
between the points.

The `1e-9`-ish floor is not a jet-versus-reference difference: that model's
training covariance has a condition number of about `5e9`, so a two-ulp
difference in the covariance comes out as a `3e-9` difference in the box.
Replacing jet's Gram-trick distance with the direct difference changes the
covariance by the same two ulp, so it does not help.

### Where they are read from

`jet.emulator.data_dir()` resolves to `$JET_DATA_DIR` when that variable is set,
and to this directory otherwise. The override is there so that a shared
installation can point at one copy on a cluster filesystem instead of every user
keeping their own.

### Sharing

There is no distribution mechanism yet: the files live wherever the package was
built, and the `$JET_DATA_DIR` override is the only way to point elsewhere. A
content-addressed store — so that a bundle records the hash of the data it was
built from and cannot silently be paired with a different revision — is the
intended replacement.
