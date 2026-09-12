#!/usr/bin/env python
"""
Repack the reference halo mass function emulator into a jet bundle.

Run this against a checkout of the reference implementation to produce
``jet/data/hmf_rockstar_m200m.gp.npz``::

    python tools/build_hmf_bundle.py \\
        --source /path/to/csstemu/CEmulator/data \\
        --output jet/data/hmf_rockstar_m200m.gp.npz

What is being repacked
----------------------
The reference's file holds a Gaussian process over the *ratio* of the mass
function to a Castro23 baseline, not the mass function itself, so the bundle
holds the same ratio model and :mod:`jet.emulator.hmf` multiplies the baseline
back in at prediction time. The reference's classes carry fixed, never-optimised
hyperparameters, which makes them copyable into a jet bundle without refitting
anything.

Three conversions happen on the way in:

* the 129 training cosmologies store ``A_s * 1e9``; jet uses physical ``As``;
* the reference's data vector runs from ``z = 3`` down to ``z = 0``, jet's from
  ``z = 0`` up. Permuting the blocks leaves the PCA scores untouched and so
  needs no refitting, but the PCA basis and the standardisers have to be
  permuted to match;
* the reference's per-process state is split into the arrays
  :class:`~jet.emulator.backends.GPBackend` stores, with the dual coefficients
  re-solved here from the same hyperparameters.

The script checks the result -- against the reference's own Gaussian process and
against its baseline separately -- and refuses to write a bundle that does not
reproduce both. The two halves of the prediction fail independently, so checking
only the end-to-end number would not tell you which half broke.
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

from jet.emulator.backends.gp import GPBackend, _factorise  # noqa: E402
from jet.emulator.bundle import save_bundle  # noqa: E402
from jet.emulator.hmf import (  # noqa: E402
    BUNDLE_NAME,
    CASTRO23_COEFFICIENTS,
    CASTRO23_KEY,
    FINE_CENTERS,
    MASS_DEFINITION,
    MASS_EDGES,
    MASS_EDGES_KEY,
    MASS_SLICES_KEY,
    N_MASS_BINS,
    ZGRID_KEY,
    castro23_dndlnM,
    data_vector_spec,
    theta_spec,
)
from jet.emulator.transforms import PCA, BoundsNorm, StandardScaler  # noqa: E402

#: Default location of the reference implementation's data directory.
DEFAULT_SOURCE = Path("/home/chenzhao/csst/simulation/csstemu/CEmulator/data")

#: File names inside the source directory.
REFERENCE_BUNDLE = "cumhmf_rockstar_M200m.npz"
REFERENCE_COSMOLOGIES = "cosmologies_8d_train_n129_Sobol.npy"

#: The reference trains on the first 129 rows of its Sobol design.
N_TRAIN = 129

#: The reference stores the primordial amplitude scaled by this factor.
REFERENCE_AS_SCALE = 1e9

#: Column of ``As`` in the parameter vector, in both implementations' order.
AS_COLUMN = 4

#: Noise floor the reference adds to the training covariance.
ALPHA = 1e-10

#: The reference's per-redshift upper mass index into its 60-bin grid, at
#: redshifts 3.0, 2.5, ..., 0.0. The lower index is 10 for every redshift
#: (``M > 1e11 Msun/h``). Copied here so the builder can reconstruct the
#: data-vector slices without importing the reference's classes.
REFERENCE_MASS_INDEX_MIN = 10
REFERENCE_MASS_INDEX_MAX = np.array([36, 38, 40, 41, 43, 45, 46, 48, 50, 52, 53, 53])

#: Redshifts of the data vector, ascending.
REDSHIFTS = np.array([0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])

#: Worst relative disagreement tolerated between this bundle and the reference's
#: own Gaussian process. The two paths do the same arithmetic in a different
#: order, so exact equality is not available; a structural mistake -- a
#: mis-ordered chain, an unpermuted column -- shows up at ``1e-3`` and above.
RATIO_TOLERANCE = 1e-8

#: Worst disagreement tolerated in the baseline, in dex. The baseline is a
#: closed-form function evaluated through two different power-spectrum
#: emulators, so it agrees far more tightly than the regression half.
BASELINE_TOLERANCE = 1e-6


def _mass_lengths_ascending() -> np.ndarray:
    """Block lengths per redshift, in ascending redshift order."""
    return (REFERENCE_MASS_INDEX_MAX - REFERENCE_MASS_INDEX_MIN)[::-1]


def _mass_slices() -> np.ndarray:
    """Return the ``(N_Z, 2)`` *mass-bin* index range of each redshift's block.

    These index :data:`jet.emulator.hmf.MASS_EDGES`, not the data vector: the
    two ranges happen to start together but diverge, because only the mass bins
    between them are stored. Ascending redshift order.

    Returns
    -------
    ndarray of shape (N_Z, 2)
        ``(start, stop)`` bin indices, ``start`` inclusive and ``stop``
        exclusive.
    """
    # Every redshift's block starts at the same lower mass; only the upper end
    # moves. Cumulating the lengths would give data-vector offsets, which is a
    # different (though related) index space.
    starts = np.full(REFERENCE_MASS_INDEX_MAX.size, REFERENCE_MASS_INDEX_MIN)
    return np.column_stack([starts, REFERENCE_MASS_INDEX_MAX[::-1]]).astype(int)


def _permutation() -> np.ndarray:
    """Return the column permutation taking the reference layout to ascending z.

    The reference concatenates its redshift blocks from ``z = 3`` downwards; the
    bundle wants them from ``z = 0`` upwards. Permuting whole blocks leaves the
    PCA scores unchanged -- the basis is permuted to match -- so this costs
    nothing but has to actually be applied, and an identity permutation here
    would leave the bundle internally consistent and outwardly wrong.
    """
    lengths = REFERENCE_MASS_INDEX_MAX - REFERENCE_MASS_INDEX_MIN
    starts = np.cumsum(lengths) - lengths
    return np.concatenate(
        [np.arange(starts[i], starts[i] + lengths[i]) for i in reversed(range(lengths.size))]
    )


def _load_reference(source: Path) -> dict:
    """Read the reference's arrays and its training cosmology design."""
    bundle = np.load(source / REFERENCE_BUNDLE, allow_pickle=True)
    cosmologies = np.load(source / REFERENCE_COSMOLOGIES)[:N_TRAIN].copy()
    cosmologies[:, AS_COLUMN] /= REFERENCE_AS_SCALE
    return {
        "Bcoeff": bundle["Bcoeff"],
        "pca_data": bundle["pca_data"],
        "pcaSS_data": bundle["pcaSS_data"],
        "gprinfo": bundle["gprinfo"],
        "cosmologies": cosmologies,
    }


def _fitted_scaler(data: np.ndarray) -> StandardScaler:
    """Fit jet's standardiser on ``data``.

    jet's transform divides by ``numpy.std`` with the default ``ddof=0``, which
    is what the reference's own standardiser does, so the two states are
    interchangeable to the last bit.
    """
    return StandardScaler().fit(data)


def _input_chain(reference: dict) -> list:
    """Return the input transform chain: bounds normalisation, then standardisation.

    The standardiser is fitted on the *normalised* parameters, not the raw ones,
    which is the order the reference applies them in. Fitting it on the raw
    values would leave the chain well defined and completely wrong, so it is
    worth being explicit about.
    """
    limits = np.array([param.bounds for param in theta_spec()])
    bounds = BoundsNorm(limits[:, 0], limits[:, 1])
    return [bounds, _fitted_scaler(bounds.transform(reference["cosmologies"]))]


def _output_chain(reference: dict, permutation: np.ndarray) -> list:
    """Return the output transform chain: standardise, PCA, standardise again.

    The reference standardised the raw ratio vectors before running the PCA, but
    only kept the standardiser, not the vectors. Those two arrays are enough:
    the transform is fitted by construction, so its state can be installed
    directly rather than re-derived.
    """
    components = reference["pca_data"][1:][:, permutation]
    mean = reference["pca_data"][0][permutation]

    # Explained variance is not stored by the reference and is not needed for a
    # prediction, but leaving it as zeros would make the property lie. It is
    # recoverable exactly: the scores are the training data's coordinates on an
    # orthonormal basis, so each component's share is its score variance over
    # the total variance of the reconstructed data.
    reconstructed = reference["Bcoeff"] @ components + mean
    total = float(reconstructed.var(axis=0).sum())
    explained = (
        reference["Bcoeff"].var(axis=0) / total if total > 0.0 else np.zeros(components.shape[0])
    )

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
        _fitted_scaler(reference["Bcoeff"]),
    ]


def _apply(chain: list, data: np.ndarray) -> np.ndarray:
    """Run ``data`` through a transform chain, in order."""
    for step in chain:
        data = step.transform(data)
    return data


def _backend_state(reference: dict, x_chain: list, y_chain: list) -> dict[str, np.ndarray]:
    """Rebuild the reference's Gaussian processes as a jet backend state.

    By the time the transforms are in place this is mostly a copy. Two things
    are not: the target scaling each process applies internally (the reference's
    ``normalize_y``, which jet folds into the backend rather than delegating to
    scikit-learn), and the dual coefficients, which are re-solved here so that
    the bundle's prediction path is the one that produced them.
    """
    gprinfo = reference["gprinfo"]
    X_train = _apply(x_chain, reference["cosmologies"])
    # Only the chain's last step applies here. The stored coefficients already
    # *are* the PCA scores, so the two steps that act on the raw ratio vectors
    # -- which the reference did not keep -- have nothing left to do.
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

        y_mean[column] = scores[:, column].mean()
        y_std[column] = scores[:, column].std()
        scaled = (scores[:, column] - y_mean[column]) / y_std[column]

        factor, used_alpha = _factorise(
            X_train,
            constant_value[column],
            length_scale[column],
            ALPHA,
            "matern52",
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
        "kernel": np.asarray("matern52"),
    }


def build(source: Path, output: Path, skip_verify: bool = False) -> Path:
    """Write the bundle, verifying it against the reference first."""
    reference = _load_reference(source)
    permutation = _permutation()
    if not np.array_equal(np.sort(permutation), np.arange(permutation.size)):
        # A repeated or dropped column would leave the bundle the right shape
        # and quietly wrong.
        raise RuntimeError("the block permutation is not a permutation of the data vector")

    x_chain = _input_chain(reference)
    y_chain = _output_chain(reference, permutation)
    backend = GPBackend.from_state(_backend_state(reference, x_chain, y_chain))

    path = save_bundle(
        output,
        x_spec=theta_spec(),
        y_spec=data_vector_spec(),
        x_chain=x_chain,
        y_chain=y_chain,
        backend=backend,
        extra={
            "source": str(source / REFERENCE_BUNDLE),
            "mass_definition": MASS_DEFINITION,
            "n_train": N_TRAIN,
            "n_mass_bins": N_MASS_BINS,
            "note": (
                "Ratio to the Castro23/T08 baseline, not a mass function. "
                "See jet.emulator.hmf for the layout and the baseline."
            ),
        },
        extra_arrays={
            ZGRID_KEY: REDSHIFTS,
            MASS_EDGES_KEY: MASS_EDGES,
            MASS_SLICES_KEY: _mass_slices(),
            CASTRO23_KEY: CASTRO23_COEFFICIENTS,
        },
    )

    if not skip_verify:
        _verify_ratio(reference, path, permutation)
        _verify_baseline(reference["cosmologies"][:3], source)
        _verify_end_to_end(reference["cosmologies"][:6], path, source)
    return path


def _verify_ratio(reference: dict, path: Path, permutation: np.ndarray) -> None:
    """Compare the bundle's ratio with the reference's own Gaussian process.

    At a training point the reference's process reproduces its stored scores to
    machine precision -- it uses a negligible noise floor -- so reconstructing
    the ratio from those scores is an exact statement of what the reference
    predicts, not an approximation of it.
    """
    from jet.emulator.hmf import load_hmf_emulator

    emulator = load_hmf_emulator()
    predicted = np.asarray(emulator.predict(reference["cosmologies"], return_std=False))

    expected = reference["Bcoeff"] @ reference["pca_data"][1:] + reference["pca_data"][0]
    expected = expected * reference["pcaSS_data"][1] + reference["pcaSS_data"][0]
    expected = expected[:, permutation]

    error = float(np.max(np.abs(predicted - expected) / np.abs(expected)))
    print(f"  ratio vs reference GP   : worst relative difference {error:.3e}")
    if not np.isfinite(error) or error > RATIO_TOLERANCE:
        raise RuntimeError(
            f"the rebuilt ratio disagrees with the reference by {error:.3e}, above the "
            f"{RATIO_TOLERANCE:.0e} tolerance; refusing to write {path.name}"
        )


def _verify_baseline(cosmologies: np.ndarray, source: Path) -> None:
    """Compare jet's Castro23 baseline with the reference's, if importable."""
    reference = _import_reference(source)
    if reference is None:
        print("  baseline                : reference not importable; skipped")
        return

    from CEmulator.utils import zlists  # type: ignore[import-not-found]

    worst = 0.0
    for row in cosmologies:
        Omegab, Omegam, H0, ns, As, w0, wa, mnu = row
        reference.set_cosmos(
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
        expected = reference.get_dndlnM_Castro23(
            z=np.asarray(zlists),
            M=FINE_CENTERS,
            Pcb=True,
            revisted=True,
            massdef=MASS_DEFINITION,
        )
        got = castro23_dndlnM(row[None, :], z=np.asarray(zlists), M=FINE_CENTERS)[0]
        worst = max(worst, float(np.max(np.abs(got / expected - 1.0))))

    print(f"  baseline vs reference   : worst relative difference {worst:.3e}")
    if worst > BASELINE_TOLERANCE:
        raise RuntimeError(
            f"the ported Castro23 baseline disagrees with the reference by {worst:.3e}"
        )


def _verify_end_to_end(cosmologies: np.ndarray, path: Path, source: Path) -> None:
    """Compare jet's mass function with the reference's own, if importable."""
    reference = _import_reference(source)
    if reference is None:
        print("  mass function           : reference not importable; skipped")
        return

    from jet.emulator.hmf import HMFEmulator

    hmf = HMFEmulator.load(path)
    z_probe = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    m_probe = np.logspace(11.0, 14.5, 12)

    worst_cumulative = 0.0
    worst_differential = 0.0
    for row in cosmologies:
        Omegab, Omegam, H0, ns, As, w0, wa, mnu = row
        reference.set_cosmos(
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
        # Ascending redshift and mass: the reference evaluates its bivariate
        # spline with `grid=True`, which requires both axes to increase.
        # No sample axis: the reference evaluates one cosmology at a time.
        expected_n = reference.get_Nhalo(z=z_probe, M=m_probe, massdef=MASS_DEFINITION)
        expected_d = reference.get_dndlnM(z=z_probe, M=m_probe, massdef=MASS_DEFINITION)

        got_n = hmf.number_density(row[None, :], z=z_probe, M=m_probe)[0]
        got_d = hmf.dndlnM(row[None, :], z=z_probe, M=m_probe)[0]

        worst_cumulative = max(worst_cumulative, _worst_dex(got_n, expected_n))
        worst_differential = max(worst_differential, _worst_dex(got_d, expected_d))

    print(f"  n(>=M) vs reference     : worst dex difference      {worst_cumulative:.3e}")
    print(f"  dn/dlnM vs reference    : worst dex difference      {worst_differential:.3e}")
    if max(worst_cumulative, worst_differential) > BASELINE_TOLERANCE:
        raise RuntimeError(
            "the mass function disagrees with the reference by more than "
            f"{BASELINE_TOLERANCE:.0e} dex"
        )


def _import_reference(source: Path):
    """Import the reference's mass function class, or return ``None``."""
    sys.path.insert(0, str(source.parent.parent))
    try:
        from CEmulator.Emulator import HMF_CEmulator  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(f"  (reference implementation not importable: {exc})")
        return None
    warnings.filterwarnings("ignore")
    return HMF_CEmulator(verbose=False)


def _worst_dex(got: np.ndarray, expected: np.ndarray) -> float:
    """Worst difference in dex between two mass functions, ignoring the tails.

    Both implementations clamp the far tail towards zero through the same
    additive offset, and the ratio of two clamped values there is meaningless.
    Points below the floor are dropped rather than compared.
    """
    floor = 1e-12
    usable = (got > floor) & (expected > floor)
    if not np.any(usable):
        return 0.0
    return float(np.max(np.abs(np.log10(got[usable]) - np.log10(expected[usable]))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"directory holding {REFERENCE_BUNDLE} (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "jet" / "data" / BUNDLE_NAME,
        help="destination .npz (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="write without checking against the reference; for debugging only",
    )
    args = parser.parse_args(argv)

    if not (args.source / REFERENCE_BUNDLE).exists():
        parser.error(f"{args.source / REFERENCE_BUNDLE} does not exist; pass --source")

    print(f"Repacking {args.source / REFERENCE_BUNDLE}")
    path = build(args.source, args.output, skip_verify=args.skip_verify)
    print(f"\nWrote {path} ({path.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
