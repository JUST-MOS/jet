#!/usr/bin/env python
"""
Repack the reference halo bias and halo-matter correlation emulators.

Run this against a checkout of the reference implementation to produce
``jet/data/xihm_brhm_rockstar_m200m.gp.npz`` and ``jet/data/bhm_rockstar_m200m.gp.npz``::

    python tools/build_xihm_bundle.py \\
        --source /path/to/csstemu/CEmulator/data \\
        --output-dir jet/data

What is being repacked
----------------------
Two Gaussian processes, both over a ratio rather than the physical quantity:

* ``Brhm`` is the effective bias ratio :math:`\\xi_{hm}/\\xi_{mm}` on a box of
  12 redshifts, 6 cumulative number-density thresholds and 37 separations.
  :mod:`jet.emulator.xihm` multiplies a matter correlation function back in.
* ``Bhm`` is the halo bias :math:`b(\\geq M)` divided by its Castro23 baseline,
  on the same 12 by 6 axes. :mod:`jet.emulator.bhm` puts the baseline back.

Neither is refitted here. The reference's hyperparameters are fixed and never
optimised, so its processes are copyable into a jet bundle directly.

Three conversions happen on the way in, identically for both models:

* the 129 training cosmologies store ``A_s * 1e9``; jet uses physical ``As``;
* the reference's data vector runs from ``z = 3`` down to ``z = 0``, jet's from
  ``z = 0`` up. Both boxes lead with the redshift axis, so this is a block
  permutation of the slabs; it leaves the PCA scores untouched and needs no
  refitting, but the PCA basis and the standardisers have to be permuted to
  match;
* the reference's per-process state is split into the arrays
  :class:`~jet.emulator.backends.GPBackend` stores, with the dual coefficients
  re-solved here from the same hyperparameters.

How the result is checked
-------------------------
Each model is checked at three levels, and the split matters:

1. the box at all 129 training cosmologies, against the coefficients the
   reference stored;
2. the box at cosmologies *between* the training points, against the
   reference's own Gaussian process;
3. for ``Brhm``, the three-axis interpolation; for ``Bhm``, the two conversions
   that turn a bias correction into a bias -- the density-to-mass inversion and
   the mass-threshold finite difference.

At a training point these processes are exact interpolants, so they reproduce
the stored coefficients whatever kernel they were built with: check 1 catches an
unpermuted redshift slab (a factor of ``3.6``) but is nearly blind to the
kernel. Check 2 is where the kernel is pinned -- it separates Matern ``3/2``
from ``5/2`` by seven orders of magnitude. Check 3 is where the conventions that
are not the regression get pinned. A single end-to-end number would report that
something broke without saying which.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jet.emulator import Emulator, bhm, xihm  # noqa: E402
from jet.emulator.backends.gp import GPBackend, _factorise  # noqa: E402
from jet.emulator.bundle import save_bundle  # noqa: E402
from jet.emulator.transforms import PCA, BoundsNorm, StandardScaler  # noqa: E402

#: Default location of the reference implementation's data directory.
DEFAULT_SOURCE = Path("/home/chenzhao/csst/simulation/csstemu/CEmulator/data")

#: Default destination directory.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "jet" / "data"

#: File names of the two reference bundles inside the source directory.
REFERENCE_BRHM_FILE = "BrhmNumBin_RockstarM200m.npz"
REFERENCE_BHM_FILE = "bhmNumBin_RockstarM200m.npz"

#: File names the two jet bundles are written to.
BRHM_BUNDLE_NAME = "xihm_brhm_rockstar_m200m.gp.npz"
BHM_BUNDLE_NAME = "bhm_rockstar_m200m.gp.npz"

#: Training cosmologies, which the reference stores separately.
REFERENCE_COSMOLOGIES = "cosmologies_8d_train_n129_Sobol.npy"

#: The reference trains on the first 129 rows of its Sobol design.
N_TRAIN = 129

#: The reference stores the primordial amplitude scaled by this factor.
REFERENCE_AS_SCALE = 1e9

#: Column of ``As`` in the parameter vector, in both implementations' order.
AS_COLUMN = 4

#: Noise floor the reference adds to the training covariance.
ALPHA = 1e-10

#: The kernel every component of both models was trained with.
KERNEL = "matern32"

#: Mass definition the weights belong to.
MASS_DEFINITION = "RockstarM200m"

#: Worst relative disagreement tolerated on a box, at the training points.
#:
#: Exact equality is not available, and the reason is worth stating because it
#: is not the usual "different order of operations". The training covariance of
#: these models has a condition number of order ``1e9`` -- the length scales are
#: large compared with a parameter box of side 1, so every training point is
#: almost perfectly correlated with every other and the matrix is close to rank
#: one. A two-ulp difference in the covariance, which is what two
#: arithmetically equivalent but differently ordered evaluations cost, is
#: amplified by that condition number into the last few digits of the dual
#: coefficients and from there into the box.
GRID_TOLERANCE = 1e-6

#: Worst relative disagreement tolerated on a box *between* the training
#: points. This is the sharp check: measured, swapping Matern ``3/2`` for
#: ``5/2`` costs about ``1e-5`` at the design and ``1.7e-2`` between it.
OFF_TRAINING_TOLERANCE = 1e-6

#: Worst relative disagreement tolerated on anything downstream of the box --
#: the interpolation axes, the bias baselines, the density-to-mass inversion.
#: These are smooth functions of a box that already matches, so they agree much
#: more tightly than the regression does, and a mistake in them is ``O(1)``.
DOWNSTREAM_TOLERANCE = 1e-6


def _z_first_permutation(shape: tuple[int, ...]) -> np.ndarray:
    """Return the column permutation taking a reference layout to ascending z.

    The reference flattens both boxes redshift-first with ``z = 3`` leading;
    jet's data vectors lead with ``z = 0``. Reversing the slabs is a permutation
    of the columns, which is the primitive quantity the bundles store -- deriving
    the ascending layout by slicing instead would let the two disagree silently.
    """
    return np.arange(int(np.prod(shape))).reshape(shape)[::-1].ravel()


def _load_reference(source: Path, filename: str) -> dict:
    """Read one reference bundle, converting the cosmology table to jet's units."""
    bundle = np.load(source / filename, allow_pickle=True)
    cosmologies = np.load(source / REFERENCE_COSMOLOGIES)[:N_TRAIN].copy()
    cosmologies[:, AS_COLUMN] = cosmologies[:, AS_COLUMN] / REFERENCE_AS_SCALE
    return {
        "Bcoeff": np.asarray(bundle["Bcoeff"], dtype=float),
        "pca_data": np.asarray(bundle["pca_data"], dtype=float),
        "pcaSS_data": np.asarray(bundle["pcaSS_data"], dtype=float),
        "gprinfo": bundle["gprinfo"],
        "cosmologies": cosmologies,
    }


def _input_chain() -> list:
    """Return the input transform chain.

    Bounds normalisation and nothing else: the reference applies ``NormCosmo``,
    which is the same affine map onto ``[0, 1]``, and leaves the result alone
    (its ``NormBeforeGP`` is false, so the standardiser branch that fills the
    same code path in its other models is dead here). Adding a standardiser
    would leave the chain well defined and the model wrong.
    """
    limits = np.array([param.bounds for param in xihm.theta_spec()])
    return [BoundsNorm(limits[:, 0], limits[:, 1])]


def _output_chain(reference: dict, permutation: np.ndarray) -> list:
    """Return the output transform chain: standardise, PCA, standardise again.

    The reference standardised the raw ratio box before running the PCA but kept
    only the standardiser, not the box, so its state is installed directly
    rather than re-derived. ``pca_data`` row 0 is the component mean and the
    rest is the basis itself.

    The trailing standardiser is the reference's ``normalize_y=True``: it
    standardises the PCA scores, one column at a time, and the Gaussian process
    sees those. jet keeps it in the chain rather than inside the backend so that
    the fitted state stays explicit and does not depend on scikit-learn's
    internal attribute names.
    """
    components = reference["pca_data"][1:][:, permutation]
    mean = reference["pca_data"][0][permutation]
    scores = reference["Bcoeff"]

    # Explained variance is not stored by the reference and is not needed for a
    # prediction, but leaving it as zeros would make the property lie. It is
    # recoverable exactly: the scores are the training box's coordinates on an
    # orthonormal basis, so each component's share is its score variance over
    # the total variance of the reconstructed box.
    reconstructed = scores @ components + mean
    total = float(reconstructed.var(axis=0).sum())
    explained = scores.var(axis=0) / total if total > 0.0 else np.zeros(components.shape[0])

    return [
        StandardScaler.from_state(
            {
                "mean": reference["pcaSS_data"][0][permutation],
                "scale": reference["pcaSS_data"][1][permutation],
            }
        ),
        PCA.from_state(
            {
                "n_components": components.shape[0],
                "mean": mean,
                "components": components,
                "explained_variance_ratio": explained,
            }
        ),
        StandardScaler().fit(scores),
    ]


def _apply(chain: list, data: np.ndarray) -> np.ndarray:
    """Run ``data`` through a transform chain, in order."""
    for step in chain:
        data = step.transform(data)
    return data


def _backend_state(reference: dict, x_chain: list, y_chain: list) -> dict[str, np.ndarray]:
    """Rebuild the reference's Gaussian processes as a jet backend state.

    By the time the transforms are in place this is mostly a copy. Only the
    dual coefficients are re-solved here, so that the bundle's own prediction
    path is the one that produced them.
    """
    gprinfo = reference["gprinfo"]
    X_train = _apply(x_chain, reference["cosmologies"])
    # Only the chain's last step applies here. The stored coefficients already
    # *are* the PCA scores, so the two steps that act on the raw ratio box --
    # which the reference did not keep -- have nothing left to do.
    scores = y_chain[-1].transform(reference["Bcoeff"])

    n_out = scores.shape[1]
    length_scale = np.empty((n_out, X_train.shape[1]), dtype=float)
    constant_value = np.empty(n_out, dtype=float)
    alpha_coef = np.empty((n_out, X_train.shape[0]), dtype=float)
    noise = np.empty(n_out, dtype=float)
    y_mean = np.empty(n_out, dtype=float)
    y_std = np.empty(n_out, dtype=float)

    for column in range(n_out):
        info = gprinfo[column]
        constant_value[column] = float(info["k1__constant_value"])
        length_scale[column] = np.atleast_1d(info["k2__length_scale"])
        if float(info["k2__nu"]) != 1.5:
            raise RuntimeError(
                f"output column {column} was trained with nu = {float(info['k2__nu'])}, "
                f"which is not the {KERNEL} kernel this builder stores"
            )

        y_mean[column] = scores[:, column].mean()
        y_std[column] = scores[:, column].std()
        scaled = (scores[:, column] - y_mean[column]) / y_std[column]

        factor, used_alpha = _factorise(
            X_train,
            constant_value[column],
            length_scale[column],
            ALPHA,
            KERNEL,
            warn=f"output column {column}",
        )
        if not np.isclose(used_alpha, ALPHA):
            raise RuntimeError(
                f"output column {column} needed a noise floor of {used_alpha:g} instead "
                f"of {ALPHA:g}; the reference factorised at {ALPHA:g}, so this bundle "
                "could not reproduce its predictions"
            )
        noise[column] = used_alpha
        alpha_coef[column] = cho_solve((factor, True), scaled, check_finite=False)

    return {
        "X_train": X_train,
        "length_scale": length_scale,
        "constant_value": constant_value,
        "noise": noise,
        "y_mean": y_mean,
        "y_std": y_std,
        "alpha_coef": alpha_coef,
        "kernel": np.asarray(KERNEL),
    }


def _pack(
    reference: dict,
    output: Path,
    shape: tuple[int, ...],
    x_spec,
    y_spec,
    extra: dict,
    extra_arrays: dict,
) -> Path:
    """Assemble one bundle from a reference state, refusing an inconsistent permutation."""
    permutation = _z_first_permutation(shape)
    if permutation.size != reference["pca_data"].shape[1]:
        raise RuntimeError(
            f"the box shape {shape} does not match the reference's "
            f"{reference['pca_data'].shape[1]} columns"
        )
    if not np.array_equal(np.sort(permutation), np.arange(permutation.size)):
        # A repeated or dropped column would leave the bundle the right shape
        # and quietly wrong.
        raise RuntimeError("the redshift permutation is not a permutation of the data vector")

    x_chain = _input_chain()
    y_chain = _output_chain(reference, permutation)
    backend = GPBackend.from_state(_backend_state(reference, x_chain, y_chain))

    return save_bundle(
        output,
        x_spec=x_spec,
        y_spec=y_spec,
        x_chain=x_chain,
        y_chain=y_chain,
        backend=backend,
        extra=extra,
        extra_arrays=extra_arrays,
    )


# ----------------------------------------------------------------------
# Brhm: the correlation-function box
# ----------------------------------------------------------------------
def _probe_cosmologies() -> np.ndarray:
    """Return interior parameter points that are not in the training set.

    This is the whole point of the off-training check. At a training point these
    Gaussian processes are exact interpolants -- their noise floor is ``1e-10``
    against a covariance of order ``1e3`` -- so they reproduce the stored
    coefficients whatever kernel they were built with. The points come from a
    fixed seed, so the reported numbers are reproducible.
    """
    rng = np.random.default_rng(20260913)
    interior = rng.uniform(0.02, 0.98, size=(6, len(xihm.theta_spec().params)))
    limits = np.array([param.bounds for param in xihm.theta_spec().params])
    return interior * (limits[:, 1] - limits[:, 0]) + limits[:, 0]


def _state_only_box(reference: dict, permutation: np.ndarray) -> np.ndarray:
    """Rebuild the reference's box from its stored coefficients.

    This is exact: at a training cosmology the reference's processes reproduce
    their stored scores to machine precision, so pushing those scores back
    through its own standardisers and PCA basis states what the reference holds
    rather than approximating it. It also needs no reference *class*, and so is
    the check that still runs when the implementation cannot be imported.
    """
    box = reference["Bcoeff"] @ reference["pca_data"][1:] + reference["pca_data"][0]
    box = box * reference["pcaSS_data"][1] + reference["pcaSS_data"][0]
    return box[:, permutation]


def _check(name: str, error: float, tolerance: float, message: str) -> None:
    """Report one measured disagreement and refuse the build if it is too large."""
    print(f"  {name:<24s}: worst relative difference {error:.3e}")
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f"{message} (measured {error:.3e}, tolerance {tolerance:.0e})")


def build_brhm(source: Path, output: Path, skip_verify: bool = False) -> Path:
    """Write the correlation-function bundle, verifying it against the reference."""
    reference = _load_reference(source, REFERENCE_BRHM_FILE)
    path = _pack(
        reference,
        output,
        (len(xihm.Z_GRID), len(xihm.LGDEN_GRID), xihm.R_MID.size),
        xihm.theta_spec(),
        xihm.data_vector_spec(),
        extra={
            "source": str(source / REFERENCE_BRHM_FILE),
            "mass_definition": MASS_DEFINITION,
            "n_train": N_TRAIN,
            "n_pca": int(reference["pca_data"].shape[0] - 1),
            "note": (
                "Effective bias ratio xi_hm / xi_mm on a fixed box, not a "
                "correlation function. See jet.emulator.xihm for the layout."
            ),
        },
        extra_arrays={
            xihm.ZGRID_KEY: np.asarray(xihm.Z_GRID),
            xihm.LGDEN_KEY: np.asarray(xihm.LGDEN_GRID),
            xihm.RMID_KEY: np.asarray(xihm.R_MID),
        },
    )

    if not skip_verify:
        permutation = _z_first_permutation(
            (len(xihm.Z_GRID), len(xihm.LGDEN_GRID), xihm.R_MID.size)
        )
        _verify_box_on_training(reference, path, permutation)
        _verify_box_off_training(reference, path, source)
        _verify_brhm_interpolation(reference, path, source)
        _verify_xihm_physics(path, source)
    return path


def _verify_box_on_training(reference: dict, path: Path, permutation: np.ndarray) -> None:
    """Check the box at the design points, against the reference's stored state."""

    predicted = np.asarray(
        Emulator.load(path).predict(reference["cosmologies"], return_std=False),
        dtype=float,
    )
    expected = _state_only_box(reference, permutation)
    _check(
        "box, training points",
        float(np.max(np.abs(predicted - expected) / np.abs(expected))),
        GRID_TOLERANCE,
        f"the rebuilt box disagrees with the reference state; refusing to write {path.name}",
    )


def _verify_box_off_training(reference: dict, path: Path, source: Path) -> None:
    """Check the box between the design points, against the reference's processes.

    This is the check that pins the kernel and the hyperparameters, and the only
    one that needs the reference to be importable. The comparison is made on the
    reshaped box with the twelve slabs reversed, in the shape the layout formula
    describes rather than through the column permutation, so a wrong permutation
    is caught here too.
    """
    handle = _import_reference(source)
    if handle is None:
        print("  box, off training       : reference not importable; skipped")
        return

    probe = handle.brhm_probe(reference)
    emulator = Emulator.load(path)

    worst = 0.0
    for row in _probe_cosmologies():
        probe.set_cosmology(row)
        expected = xihm.brhm_grid(probe.get_box())[0][::-1]
        got = xihm.brhm_grid(np.asarray(emulator.predict(row[None, :], return_std=False)))[0]
        worst = max(worst, float(np.max(np.abs(got - expected) / np.abs(expected))))

    _check(
        "box, off training",
        worst,
        OFF_TRAINING_TOLERANCE,
        "the box disagrees with the reference's own Gaussian process away from the "
        "training set; the kernel or the hyperparameters are not the reference's",
    )


def _verify_brhm_interpolation(reference: dict, path: Path, source: Path) -> None:
    """Check the three axes, against the reference's ``get_Brhm``.

    The box is ``(z, lgden, r)`` flattened one way in the reference and the
    other way in jet, and the interpolation is in ``log10(r)`` rather than in
    ``r``; getting any of those wrong leaves the box correct and the
    interpolated values crossed, which neither check above can see.
    """
    handle = _import_reference(source)
    if handle is None:
        print("  interpolation           : reference not importable; skipped")
        return

    emulator = xihm.XiHMEmulator.load(path)
    probe = handle.brhm_probe(reference)

    # Coordinates strictly inside the box, so that this check is about the
    # interpolation order and not about the extrapolation policy.
    z_probe = np.array([0.05, 0.4, 0.9, 1.4, 1.9, 2.7])
    lgden_probe = np.array([-4.75, -4.1, -3.4, -2.9, -2.6])
    r_probe = np.logspace(-1.5, 1.4, 11)

    worst = 0.0
    for row in _probe_cosmologies():
        probe.set_cosmology(row)
        expected = probe.get_Brhm(r=r_probe, z=z_probe, lgden=lgden_probe)
        got = emulator.brhm(row[None, :], z=z_probe, lgden=lgden_probe, r=r_probe)[0]
        if got.shape != expected.shape:
            raise RuntimeError(
                f"shape mismatch against the reference: jet {got.shape}, reference {expected.shape}"
            )
        worst = max(worst, float(np.max(np.abs(got - expected) / np.abs(expected))))

    _check(
        "interpolation",
        worst,
        DOWNSTREAM_TOLERANCE,
        "the interpolated box disagrees with the reference's get_Brhm",
    )


def _verify_xihm_physics(path: Path, source: Path) -> None:
    r"""Check :math:`\xi_{hm}` end to end against the reference's own entry points.

    This is the only check that exercises every layer at once -- the linear
    power spectrum, the FFTLog transform, the matter correlation function, the
    bias baseline, the blend and the emulated ratio. It is deliberately kept
    *after* the per-layer checks for exactly that reason: everything below it
    has already been compared on its own, so a failure here without a failure
    there points at the composition rather than the parts.

    It also needs the full reference entry point, so it is the slowest and the
    most dependent on the reference being importable.
    """
    handle = _import_reference(source)
    if handle is None:
        print("  xi_hm end to end        : reference not importable; skipped")
        return

    emulator = xihm.XiHMEmulator.load(path)
    probe = handle.xihm_probe()

    z_probe = np.array([0.0, 0.5, 1.5, 3.0])
    r_probe = np.logspace(-1.5, 1.2, 12)
    lgden_probe = np.array([-4.8, -4.2, -3.6, -3.0])
    mass_probe = np.logspace(12.0, 14.0, 5)

    worst_ximm = 0.0
    worst_density = 0.0
    worst_mass = 0.0
    for row in _probe_cosmologies()[:2]:
        probe.set_cosmology(row)
        theta = row[None, :]

        worst_ximm = max(
            worst_ximm,
            _worst_relative(
                emulator.ximm(theta, z=z_probe, r=r_probe)[0],
                probe.get_ximmlinear(z=z_probe, r=r_probe),
            ),
        )
        worst_density = max(
            worst_density,
            _worst_relative(
                emulator.xihm_lgnbar_threshold(theta, z=z_probe, r=r_probe, lgden=lgden_probe)[0],
                probe.get_xihm_lgnbar_threshold(z=z_probe, r=r_probe, lgden=lgden_probe),
            ),
        )
        worst_mass = max(
            worst_mass,
            _worst_relative(
                emulator.xihm_mass(theta, z=z_probe, r=r_probe, M=mass_probe)[0],
                probe.get_xihm_mass(z=z_probe, r=r_probe, M=mass_probe),
            ),
        )

    _check(
        "xi_mm",
        worst_ximm,
        DOWNSTREAM_TOLERANCE,
        "the matter correlation function disagrees with the reference",
    )
    _check(
        "xi_hm, n-threshold",
        worst_density,
        DOWNSTREAM_TOLERANCE,
        "xi_hm at a fixed density threshold disagrees with the reference",
    )
    _check(
        "xi_hm, mass threshold",
        worst_mass,
        DOWNSTREAM_TOLERANCE,
        "xi_hm at a fixed mass threshold disagrees with the reference",
    )


# ----------------------------------------------------------------------
# Bhm: the bias box and the conversions on top of it
# ----------------------------------------------------------------------
def build_bhm(source: Path, output: Path, skip_verify: bool = False) -> Path:
    """Write the halo-bias bundle, verifying it against the reference."""
    reference = _load_reference(source, REFERENCE_BHM_FILE)
    path = _pack(
        reference,
        output,
        (len(bhm.Z_GRID), len(bhm.LGDEN_GRID)),
        bhm.theta_spec(),
        bhm.data_vector_spec(),
        extra={
            "source": str(source / REFERENCE_BHM_FILE),
            "mass_definition": MASS_DEFINITION,
            "n_train": N_TRAIN,
            "n_pca": int(reference["pca_data"].shape[0] - 1),
            "note": (
                "Halo bias divided by its Castro23 baseline, not a bias. "
                "See jet.emulator.bhm for the layout."
            ),
        },
        extra_arrays={
            xihm.ZGRID_KEY: np.asarray(bhm.Z_GRID),
            xihm.LGDEN_KEY: np.asarray(bhm.LGDEN_GRID),
        },
    )

    if not skip_verify:
        permutation = _z_first_permutation((len(bhm.Z_GRID), len(bhm.LGDEN_GRID)))
        _verify_box_on_training(reference, path, permutation)
        _verify_bhm_off_training(reference, path, source)
        _verify_bhm_conversions(reference, path, source)
    return path


def _verify_bhm_off_training(reference: dict, path: Path, source: Path) -> None:
    """Check the bias box between the design points, against the reference."""
    handle = _import_reference(source)
    if handle is None:
        print("  box, off training       : reference not importable; skipped")
        return

    probe = handle.hmf_probe()
    emulator = Emulator.load(path)

    worst = 0.0
    for row in _probe_cosmologies():
        probe.set_cosmology(row)
        expected = bhm.bias_grid(probe.get_bias_box())[0][::-1]
        got = bhm.bias_grid(np.asarray(emulator.predict(row[None, :], return_std=False)))[0]
        worst = max(worst, float(np.max(np.abs(got - expected) / np.abs(expected))))

    _check(
        "box, off training",
        worst,
        OFF_TRAINING_TOLERANCE,
        "the bias box disagrees with the reference's own Gaussian process away from the "
        "training set; the kernel or the hyperparameters are not the reference's",
    )


def _verify_bhm_conversions(reference: dict, path: Path, source: Path) -> None:
    """Check the density-to-mass inversion and the two bias conversions.

    These are the parts of this model that are not a regression, and they fail
    independently of it: a wrong inversion step moves every mass, a wrong
    finite-difference weight moves only the mass-threshold bias.
    """
    handle = _import_reference(source)
    if handle is None:
        print("  conversions             : reference not importable; skipped")
        return

    emulator = bhm.BhmEmulator.load(path)
    probe = handle.hmf_probe()

    z_probe = np.array([0.0, 0.5, 1.5, 3.0])
    lgden_probe = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5])
    mass_probe = np.logspace(11.5, 14.5, 8)

    worst_mass = 0.0
    worst_density = 0.0
    worst_mass_bias = 0.0
    for row in _probe_cosmologies()[:4]:
        probe.set_cosmology(row)

        expected_mass = probe.get_mass_from_lgden(z=z_probe, lgden=lgden_probe)
        got_mass = emulator.mass_from_lgden(row[None, :], z=z_probe, lgden=lgden_probe)[0]
        worst_mass = max(worst_mass, _worst_relative(got_mass, expected_mass))

        expected_density = probe.get_bias_lgnbar_threshold(z=z_probe, lgden=lgden_probe)
        got_density = emulator.bias_lgnbar_threshold(row[None, :], z=z_probe, lgden=lgden_probe)[0]
        worst_density = max(worst_density, _worst_relative(got_density, expected_density))

        with warnings.catch_warnings():
            # Bins with no halos are the reference's own documented behaviour,
            # not a failure of the port; both implementations warn and return
            # zero there.
            warnings.simplefilter("ignore", RuntimeWarning)
            expected_mass_bias = probe.get_bias_mass(z=z_probe, M=mass_probe)
            got_mass_bias = emulator.bias_mass(row[None, :], z=z_probe, M=mass_probe)[0]
        worst_mass_bias = max(worst_mass_bias, _worst_relative(got_mass_bias, expected_mass_bias))

    _check(
        "mass from lgden",
        worst_mass,
        DOWNSTREAM_TOLERANCE,
        "the density-to-mass inversion disagrees with the reference",
    )
    _check(
        "bias, n-threshold",
        worst_density,
        DOWNSTREAM_TOLERANCE,
        "the density-threshold bias disagrees with the reference",
    )
    _check(
        "bias, mass threshold",
        worst_mass_bias,
        DOWNSTREAM_TOLERANCE,
        "the mass-threshold bias disagrees with the reference",
    )


def _worst_relative(got: np.ndarray, expected: np.ndarray) -> float:
    """Worst relative difference over the entries the reference does not zero out."""
    usable = np.abs(expected) > 0.0
    if not np.any(usable):
        return 0.0
    return float(np.max(np.abs(got[usable] - expected[usable]) / np.abs(expected[usable])))


# ----------------------------------------------------------------------
# The reference implementation, if it is importable
# ----------------------------------------------------------------------
class _ReferenceProbes:
    """Factories for the two reference objects the checks drive.

    The reference's model classes take no cosmology of their own; their owner
    injects a normalised vector before every call. Only that one attribute (and,
    for the bias, the owner's helper methods) is needed here, so the root
    ``Xihm_CEmulator`` -- which builds every power spectrum, mass function and
    correlation function in the package -- is avoided for the correlation box
    and used only where its helpers are genuinely required.
    """

    def __init__(self, brhm_cls, norm_cosmo, param_names, param_limits, hmf_cls, xihm_cls) -> None:
        self._brhm_cls = brhm_cls
        self._norm_cosmo = norm_cosmo
        self._param_names = param_names
        self._param_limits = param_limits
        self._hmf_cls = hmf_cls
        self._xihm_cls = xihm_cls

    def _normalise(self, row: np.ndarray) -> np.ndarray:
        raw = row[None, :].copy()
        raw[:, AS_COLUMN] = raw[:, AS_COLUMN] * REFERENCE_AS_SCALE
        return self._norm_cosmo(raw, self._param_names, self._param_limits)[0]

    def brhm_probe(self, reference: dict):
        """Return a driver for the reference's correlation-function box."""

        class _BrhmProbe:
            def __init__(probe_self) -> None:
                probe_self._gp = self._brhm_cls(verbose=False)

            def set_cosmology(probe_self, row: np.ndarray) -> None:
                probe_self._gp.ncosmo = self._normalise(row)
                # The reference never invalidates this cache when the cosmology
                # changes, so reusing one object across cosmologies silently
                # returns the first one's answer. jet's own emulator has no such
                # state to reset. Harmless once the reference is fixed, and
                # needed until then.
                probe_self._gp._Brhm_interp = None

            def get_box(probe_self) -> np.ndarray:
                return probe_self._gp.get_data()

            def get_Brhm(probe_self, r, z, lgden) -> np.ndarray:
                return probe_self._gp.get_Brhm(r=r, z=z, lgden=lgden)

        return _BrhmProbe()

    def hmf_probe(self):
        """Return a driver for the reference's bias model and its conversions.

        This one needs the owner class: the density-to-mass inversion and the
        mass-threshold finite difference live on it, not on the box model.
        """
        emulator = self._hmf_cls(verbose=False)

        class _HmfProbe:
            def set_cosmology(probe_self, row: np.ndarray) -> None:
                Omegab, Omegam, H0, ns, As, w0, wa, mnu = row
                emulator.set_cosmos(
                    Omegab=Omegab,
                    Omegac=Omegam - Omegab,
                    H0=H0,
                    As=As,
                    ns=ns,
                    w=w0,
                    wa=wa,
                    mnu=mnu,
                    checkbound=False,
                )

            def get_bias_box(probe_self) -> np.ndarray:
                return emulator.bhmRockstarM200m.get_data()

            def get_mass_from_lgden(probe_self, z, lgden) -> np.ndarray:
                return emulator.get_mass_from_lgden(z=z, lgden=lgden, massdef=MASS_DEFINITION)

            def get_bias_lgnbar_threshold(probe_self, z, lgden) -> np.ndarray:
                return emulator.get_bias_lgnbar_threshold(z=z, lgden=lgden, massdef=MASS_DEFINITION)

            def get_bias_mass(probe_self, z, M) -> np.ndarray:
                return emulator.get_bias_mass(z=z, M=M, massdef=MASS_DEFINITION)

        return _HmfProbe()

    def xihm_probe(self):
        """Return a driver for the reference's full correlation-function entry points.

        The heaviest of the three: this class builds every emulator in the
        package. It is needed here because the correlation function is composed
        from all of them, and only its own methods know how the reference
        composes them.
        """
        emulator = self._xihm_cls(verbose=False)

        class _XihmProbe:
            def set_cosmology(probe_self, row: np.ndarray) -> None:
                Omegab, Omegam, H0, ns, As, w0, wa, mnu = row
                emulator.set_cosmos(
                    Omegab=Omegab,
                    Omegac=Omegam - Omegab,
                    H0=H0,
                    As=As,
                    ns=ns,
                    w=w0,
                    wa=wa,
                    mnu=mnu,
                    checkbound=False,
                )
                # The same stale-cache workaround as in `brhm_probe`, and it
                # matters more here: this object serves three checks in a row.
                emulator.BrhmRockstarM200m._Brhm_interp = None

            def get_ximmlinear(probe_self, z, r) -> np.ndarray:
                return emulator.get_ximmlinear(z=z, r=r, Pcb=True)

            def get_xihm_lgnbar_threshold(probe_self, z, r, lgden) -> np.ndarray:
                return emulator.get_xihm_lgnbar_threshold(
                    z=z, r=r, lgden=lgden, massdef=MASS_DEFINITION, Pcb=True
                )

            def get_xihm_mass(probe_self, z, r, M) -> np.ndarray:
                return emulator.get_xihm_mass(z=z, r=r, M=M, massdef=MASS_DEFINITION, Pcb=True)

        return _XihmProbe()


def _import_reference(source: Path):
    """Return the reference probe factories, or ``None`` if it cannot be imported.

    ``source`` is the reference's data directory; its package root is two levels
    up.
    """
    sys.path.insert(0, str(source.parent.parent))
    try:
        from CEmulator.Emulator import (  # type: ignore[import-not-found]
            HMF_CEmulator,
            Xihm_CEmulator,
        )
        from CEmulator.emulator.Xihm import (  # type: ignore[import-not-found]
            BrhmRockstarM200m_gp,
        )
        from CEmulator.utils import (  # type: ignore[import-not-found]
            NormCosmo,
            param_limits,
            param_names,
        )
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(f"  (reference implementation not importable: {exc})")
        return None

    warnings.filterwarnings("ignore")
    return _ReferenceProbes(
        BrhmRockstarM200m_gp,
        NormCosmo,
        param_names,
        param_limits,
        HMF_CEmulator,
        Xihm_CEmulator,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="directory holding the reference bundles (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="destination directory (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="write the bundles without checking them against the reference",
    )
    args = parser.parse_args(argv)

    print(f"building from {args.source}")
    for label, builder, name in (
        ("Brhm (xi_hm / xi_mm)", build_brhm, BRHM_BUNDLE_NAME),
        ("Bhm (halo bias)", build_bhm, BHM_BUNDLE_NAME),
    ):
        print(f"\n{label}")
        path = builder(args.source, args.output_dir / name, skip_verify=args.skip_verify)
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
