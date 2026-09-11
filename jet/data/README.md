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

### Regenerating

```bash
python tools/build_pklin_bundle.py                 # default source path
python tools/build_pklin_bundle.py --source <dir>  # another checkout

python tools/build_hmf_bundle.py                   # default source path
python tools/build_hmf_bundle.py --source <dir>    # another checkout
```

Both scripts read the reference arrays, repackage them, and then **verify the
result against the reference implementation** if that implementation is
importable — refusing the build when the two disagree by more than floating-point
round-off. `build_hmf_bundle.py` checks the regression and the baseline
separately, because the two fail independently. Run them after changing anything
about the transform chains, the parameter bounds, or the ported physics.

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
