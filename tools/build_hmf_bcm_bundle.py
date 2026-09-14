#!/usr/bin/env python
"""
Repack the reference baryonic halo mass function emulator into a jet bundle.

Run this against a checkout of the reference implementation to produce
``jet/data/hmf_bcm_m200m.gp.npz``::

    python tools/build_hmf_bcm_bundle.py \\
        --source /path/to/CEmulatorG/data \\
        --output jet/data/hmf_bcm_m200m.gp.npz

What is being repacked
----------------------
``cumhmf_M200m_bcm.npz`` holds a Gaussian process over
:math:`\\log_{10} n(\\geq M)` as a function of four baryonic feedback
parameters -- and of nothing else. There is no cosmological axis and no
redshift axis: the training data is one cosmology at one redshift, and the
model cannot be asked about any other. That is unusual for this package and it
is why the spec declares ``block="baryon"`` rather than a cosmological block.

The output is a rectangle: a single uniform mass grid, no baseline to multiply
back in, no per-redshift blocks to slice. The reference's own class is a
hundred lines and most of it is loading.

The kernel, and the stored noise level
--------------------------------------
The reference stores four things per process -- a ``ConstantKernel`` amplitude,
a Matern length scale, ``nu = 2.5``, and a ``WhiteKernel`` noise level -- and
builds ``ConstantKernel * Matern + WhiteKernel``. A ``WhiteKernel`` is how a
Gaussian process says its training points carry noise; it belongs in the
training covariance and is meant to drop out of the cross-covariance at
prediction time.

On these weights the stored dual coefficients are what settles the question, and
they match ``ConstantKernel * Matern`` with the negligible floor every other
bundled model uses -- not that kernel plus the stored noise level. So the bundle
carries ``alpha = 1e-10``, and the levels ride along in the manifest as
provenance: reproducing the reference's numbers is the job, and recording what
was left out is what makes a future divergence legible instead of mysterious.

What is checked
---------------
Two comparisons against the reference's own Gaussian process -- at the 128
training points and at points *between* them -- and one reference-free fallback
against the coefficients the reference stored.

The stored coefficients are worth a word, because at first sight they look like
the cleanest possible check and they are not. They are the regression's
*targets*, and a Gaussian process with a non-zero floor does not return its
targets: ``K (K + alpha I)^-1 y`` sits ``alpha`` short of ``y``, which for a
score-space residual of order ``1e-9`` is invisible until the transform chain
exponentiates and it becomes ``1.7e-7`` in the abundance, where the smallest
bin is ``8e-4``. Measured, the **reference itself** disagrees with its own
stored coefficients by ``1.68e-7``, and this bundle disagrees with them by
``1.68e-7`` as well -- the same number, which is the point. So that comparison
is kept, with a tolerance that admits what it cannot resolve, and the real
check is against the reference's process rather than its table.

This is the mirror image of the lesson in ``build_xihm_bundle.py``. There the
process interpolates exactly and the stored state is a faithful proxy for the
prediction; here it nearly does, and "nearly" is not enough once ``10**`` has
had its say.
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
from jet.emulator.hmf_bcm import (  # noqa: E402
    ALPHA,
    BUNDLE_NAME,
    MASS_GRID_KEY,
    data_vector_spec,
    theta_spec,
)
from jet.emulator.transforms import PCA, BoundsNorm, Log10, StandardScaler  # noqa: E402

#: Default location of the reference implementation's data directory.
DEFAULT_SOURCE = Path("/home/chenzhao/lensing+baryon/CEmulatorG/data")

#: The reference bundle and the file holding the training design.
REFERENCE_BUNDLE = "cumhmf_M200m_bcm.npz"

#: The design file. Note this is the **2048** variant: the reference's ``utils``
#: loads ``X1`` from it and ``X2`` from it too. The 1024 variant carries an
#: identical ``X1``, so switching to it would not change the numbers today --
#: which is exactly why it is worth naming, rather than discovering later that
#: a "harmless" swap changed nothing until it did.
REFERENCE_PARAMS = "param_b128h2048_baseHOD_nbar-3.5_-2.5.npz"

#: Design points the reference trained on.
N_TRAIN = 128

#: Kernel every component was trained with. The reference's ``Matern`` carries
#: ``nu = 2.5``; see the module docstring for why no ``WhiteKernel`` term
#: appears here.
KERNEL = "matern52"

#: Cone the reference's mass grid is cut from, before the lower-edge slice.
MASS_EDGES_FULL = np.logspace(10.0, 16.0, 61)

#: Bins dropped from the low-mass end. The reference slices the 61 edges of a
#: ten-decade grid down to ``[14:]``, which starts the model at
#: :math:`10^{11.4} M_\odot/h`.
MASS_EDGE_OFFSET = 14

#: Worst relative disagreement tolerated against the reference's own Gaussian
#: process, at the training points and between them.
#:
#: Both implementations solve the same linear system, so this is round-off and
#: nothing else; a structural mistake -- a dropped transform, a transposed
#: chain, a wrong length scale -- is an ``O(1)`` error.
#:
#: Measured at ``5.3e-9`` at the design points, ``3.2e-8`` between them and
#: ``4.9e-8`` on the differential. Those are about a hundred times larger than
#: the previous revision of the reference's array gave (``2.7e-10`` and
#: ``1.8e-9``), so the margin here is a factor of two rather than a decade: the
#: awkward quantity is the reference's, not this bundle's, and it moves when the
#: reference is retrained. Worth widening to ``1e-6`` -- still four orders below
#: anything structural -- if the next retraining pushes it further.
REFERENCE_TOLERANCE = 1e-7

#: Worst relative disagreement tolerated against the coefficients the reference
#: *stored*.
#:
#: This is the fallback for a checkout that cannot import the reference, and the
#: tolerance is loose because the comparison cannot be tight: those coefficients
#: are the regression's targets, not its predictions, and the gap between the
#: two is what the ``10**`` in the chain magnifies. Measured, the reference
#: disagrees with its own stored table by ``1.7e-7``; a dropped transform is
#: still an ``O(1)`` error, so the check keeps its teeth.
STORED_STATE_TOLERANCE = 1e-5


def _mass_grid() -> np.ndarray:
    """Return the lower edge of each mass bin, in :math:`M_\\odot/h`.

    The reference keeps ``m_edges = logspace(10, 16, 61)[14:]`` (47 values) and
    uses ``m_edges[:-1]`` as the 46 bin lowers, with ``m_edges[1:]`` as their
    uppers. The data vector is 46 long, which is what fixes the reading.
    """
    edges = MASS_EDGES_FULL[MASS_EDGE_OFFSET:]
    grid = edges[:-1]
    if grid.size != data_vector_spec().n_bins:
        raise RuntimeError(
            f"the reference's grid gives {grid.size} bins but the spec declares "
            f"{data_vector_spec().n_bins}"
        )
    # A single width is used for every bin, so a non-uniform grid would make
    # `dndlgM` quietly wrong rather than loudly broken.
    widths = np.diff(np.log10(grid))
    if not np.allclose(widths, widths[0], rtol=1e-12, atol=0.0):
        raise RuntimeError("the reference's mass grid is not uniform in log10 M")
    return grid


def _load_reference(source: Path) -> dict:
    """Read the reference's arrays and its training design."""
    bundle = np.load(source / REFERENCE_BUNDLE, allow_pickle=True)
    design = np.load(source / REFERENCE_PARAMS)
    parameters = np.asarray(design["X1"], dtype=float)
    if parameters.shape != (N_TRAIN, theta_spec().dim):
        raise RuntimeError(
            f"the training design is {parameters.shape}, expected {(N_TRAIN, theta_spec().dim)}"
        )
    return {
        "Bcoeff": np.asarray(bundle["Bcoeff"], dtype=float),
        "pca_data": np.asarray(bundle["pca_data"], dtype=float),
        "pcaSS_data": np.asarray(bundle["pcaSS_data"], dtype=float),
        "gprinfo": bundle["gprinfo"],
        "nvec": int(bundle["nvec"]),
        "parameters": parameters,
    }


def _fitted_scaler(data: np.ndarray) -> StandardScaler:
    """Fit jet's standardiser on ``data``.

    jet's transform divides by ``numpy.std`` with the default ``ddof=0``, which
    is what the reference's own standardiser does, so the two states are
    interchangeable to the last bit.
    """
    return StandardScaler().fit(data)


def _input_chain(reference: dict) -> list:
    """Return the input chain: bounds normalisation, then standardisation.

    The standardiser is fitted on the *normalised* parameters, not the raw
    ones. Fitting it on the raw values would leave the chain well defined and
    completely wrong, and the parameter ranges here differ by four orders of
    magnitude, so the mistake would not be subtle in the numbers either.
    """
    limits = np.array([param.bounds for param in theta_spec()])
    bounds = BoundsNorm(limits[:, 0], limits[:, 1])
    return [bounds, _fitted_scaler(bounds.transform(reference["parameters"]))]


def _output_chain(reference: dict) -> list:
    r"""Return the output chain: log, standardise, PCA, standardise.

    The reference stores the standardiser and the PCA basis but not the
    pre-PCA box, so those two are installed directly rather than re-derived.
    ``pca_data`` row 0 is the component mean and the rest is the basis.

    The leading :class:`~jet.emulator.transforms.Log10` is not optional
    book-keeping: the reference emulates :math:`\log_{10}` of the abundance and
    exponentiates at the end, so a chain without it would predict the
    logarithm and call it a density.
    """
    components = reference["pca_data"][1:]
    mean = reference["pca_data"][0]
    scores = reference["Bcoeff"]
    if components.shape[0] != reference["nvec"]:
        raise RuntimeError(
            f"the reference stores {reference['nvec']} components but {components.shape[0]} "
            "rows of basis; the two disagree about the model"
        )

    # Explained variance is not stored and is not needed for a prediction, but
    # leaving it as zeros would make the property lie. It is recoverable
    # exactly: the scores are the training box's coordinates on an orthonormal
    # basis, so each component's share is its score variance over the total
    # variance of the reconstructed box.
    reconstructed = scores @ components + mean
    total = float(reconstructed.var(axis=0).sum())
    explained = scores.var(axis=0) / total if total > 0.0 else np.zeros(components.shape[0])

    return [
        Log10(),
        StandardScaler.from_state(
            {"mean": reference["pcaSS_data"][0], "scale": reference["pcaSS_data"][1]}
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

    The one judgement here is the noise floor. The reference stores a fitted
    ``WhiteKernel`` level per component and then never uses it, so the floor
    this bundle carries is :data:`ALPHA` -- the same negligible value every
    other bundled model uses -- and the stored levels are recorded in the
    manifest instead. See the module docstring.
    """
    gprinfo = reference["gprinfo"]
    X_train = _apply(x_chain, reference["parameters"])
    # Only the chain's last step applies here: the stored coefficients already
    # *are* the PCA scores, so the steps that act on the abundance box -- which
    # the reference did not keep -- have nothing left to do.
    scores = y_chain[-1].transform(reference["Bcoeff"])

    n_out = scores.shape[1]
    length_scale = np.empty((n_out, X_train.shape[1]), dtype=float)
    constant_value = np.empty(n_out, dtype=float)
    alpha_coef = np.empty((n_out, X_train.shape[0]), dtype=float)
    noise = np.empty(n_out, dtype=float)
    y_mean = np.empty(n_out, dtype=float)
    y_std = np.empty(n_out, dtype=float)
    discarded = np.empty(n_out, dtype=float)

    for column in range(n_out):
        info = gprinfo[column]
        constant_value[column] = float(info["k1__k1__constant_value"])
        length_scale[column] = np.atleast_1d(info["k1__k2__length_scale"])
        if float(info["k1__k2__nu"]) != 2.5:
            raise RuntimeError(
                f"output column {column} was trained with nu = {float(info['k1__k2__nu'])}, "
                f"which is not the {KERNEL} kernel this builder stores"
            )
        # Read only so the manifest can say what was left out; it is not used.
        discarded[column] = float(info["k2__noise_level"])

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
        "discarded_noise": discarded,
    }


def build(source: Path, output: Path, skip_verify: bool = False) -> Path:
    """Write the bundle, verifying it against the reference first."""
    reference = _load_reference(source)
    x_chain = _input_chain(reference)
    y_chain = _output_chain(reference)
    state = _backend_state(reference, x_chain, y_chain)
    backend = GPBackend.from_state(state)

    path = save_bundle(
        output,
        x_spec=theta_spec(),
        y_spec=data_vector_spec(),
        x_chain=x_chain,
        y_chain=y_chain,
        backend=backend,
        extra={
            "source": str(source / REFERENCE_BUNDLE),
            "n_train": N_TRAIN,
            "n_pca": int(reference["nvec"]),
            "redshift": 0.5,
            "cosmology": "c0000",
            "discarded_noise_level": [
                float(value) for value in np.atleast_1d(state["discarded_noise"])
            ],
            "note": (
                "Cumulative abundance with baryonic feedback, at one fixed cosmology "
                "and redshift and with no axis for either. The WhiteKernel noise the "
                "reference stores is discarded by the reference itself -- see the "
                "builder's docstring -- so it is not used here either."
            ),
        },
        extra_arrays={MASS_GRID_KEY: _mass_grid()},
    )

    if not skip_verify:
        _verify_against_stored_state(reference, path)
        _verify_against_reference(reference, path, source)
    return path


def _reference_box(reference: dict) -> np.ndarray:
    r"""Rebuild the reference's prediction at its training points, in physical units.

    Pushes the stored coefficients back through the reference's own standardiser
    and PCA basis and exponentiates, which is exactly what its ``get_data``
    does once the Gaussian process has returned them unchanged -- and it does
    return them unchanged, because the process is an interpolant.
    """
    scores = reference["Bcoeff"]
    standardised_log10 = scores @ reference["pca_data"][1:] + reference["pca_data"][0]
    log10_box = standardised_log10 * reference["pcaSS_data"][1] + reference["pcaSS_data"][0]
    return np.power(10.0, log10_box)


def _verify_against_stored_state(reference: dict, path: Path) -> None:
    """Check the bundle against the coefficients the reference stored.

    The reference-free fallback. It cannot be tight -- see the module docstring
    and :data:`STORED_STATE_TOLERANCE` -- but it still catches anything
    structural, because a dropped transform or a transposed chain moves the
    abundance by a factor rather than by a few digits.
    """
    from jet.emulator.emulator import Emulator

    predicted = np.asarray(
        Emulator.load(path).predict(reference["parameters"], return_std=False), dtype=float
    )
    expected = _reference_box(reference)

    error = float(np.max(np.abs(predicted - expected) / np.abs(expected)))
    print(f"  vs the stored state     : worst relative difference {error:.3e}")
    if not np.isfinite(error) or error > STORED_STATE_TOLERANCE:
        raise RuntimeError(
            f"the rebuilt abundance disagrees with the reference state by {error:.3e}, "
            f"above the {STORED_STATE_TOLERANCE:.0e} tolerance; refusing to write "
            f"{path.name}"
        )


def _verify_against_reference(reference: dict, path: Path, source: Path) -> None:
    """Check the bundle against the reference's own process, on and off the design.

    Two reported numbers because they fail for different reasons. Agreement at
    the design points means the chains and the stored state were carried across
    correctly. Agreement *between* them means the kernel and its hyperparameters
    are the reference's -- a wrong length scale or a wrong Matern order is
    nearly invisible at points the process was fitted to and obvious between
    them.
    """
    handle = _import_reference(source)
    if handle is None:
        print("  vs the reference        : reference not importable; skipped")
        return

    from jet.emulator.emulator import Emulator
    from jet.emulator.hmf_bcm import BCMHFEmulator

    probe = handle()
    emulator = Emulator.load(path)

    # Interior points of the training box, drawn on a fixed seed so the reported
    # number is reproducible.
    rng = np.random.default_rng(20260913)
    limits = np.array([param.bounds for param in theta_spec()])
    interior = rng.uniform(0.05, 0.95, size=(6, theta_spec().dim))
    between = interior * (limits[:, 1] - limits[:, 0]) + limits[:, 0]

    worst_training = _worst_against(probe, emulator, reference["parameters"])
    worst_between = _worst_against(probe, emulator, between)

    print(f"  vs the reference, on    : worst relative difference {worst_training:.3e}")
    print(f"  vs the reference, between: worst relative difference {worst_between:.3e}")
    if max(worst_training, worst_between) > REFERENCE_TOLERANCE:
        raise RuntimeError(
            f"the abundance disagrees with the reference's own process by "
            f"{max(worst_training, worst_between):.3e}, above the "
            f"{REFERENCE_TOLERANCE:.0e} tolerance; the kernel or its hyperparameters "
            "are not the reference's"
        )

    _verify_differential(probe, BCMHFEmulator.load(path), between)


def _worst_against(probe, emulator, points: np.ndarray) -> float:
    """Worst relative difference between the bundle and the reference's process."""
    worst = 0.0
    for row in points:
        expected = probe.get_data(row)
        got = np.asarray(emulator.predict(row[None, :], return_std=False))[0]
        worst = max(worst, float(np.max(np.abs(got - expected) / np.abs(expected))))
    return worst


def _verify_differential(probe, emulator, points: np.ndarray) -> None:
    """Check ``dndlgM`` against the reference's differencing convention.

    The differential abundance is a plain difference of the cumulative one, so
    this does not test any physics -- it tests the convention: which way the
    array is reversed, what the last bin differences against, and which bin
    width is divided out. All three are silent when wrong.
    """
    from jet.emulator.hmf_bcm import bin_width

    worst = 0.0
    for row in points:
        cumulative = probe.get_data(row)
        # Transcribed from the reference's `get_dndlgm`, which wraps a 1-D
        # cumulative array in exactly this expression:
        # `np.diff(cumhmf[::-1], axis=0, prepend=0)[::-1] / lgM * 1e-9`.
        expected = np.diff(cumulative[::-1], axis=0, prepend=0.0)[::-1] / bin_width() * 1e-9
        got = emulator.dndlgM(row[None, :])[0]
        worst = max(worst, float(np.max(np.abs(got - expected) / np.abs(expected))))

    print(f"  dndlgM                  : worst relative difference {worst:.3e}")
    if worst > REFERENCE_TOLERANCE:
        raise RuntimeError(
            f"dndlgM disagrees with the reference's differencing by {worst:.3e}; the "
            "reversal, the last bin or the bin width is not the reference's"
        )


def _import_reference(source: Path):
    """Return a driver for the reference's mass function, or ``None``.

    The reference class takes no parameters of its own; its owner injects a
    normalised vector into ``X1norm`` before every call. Only that one attribute
    is needed here, so the ``HODEmulator`` -- which also builds the projected
    correlation function and the excess surface density, from two dozen
    megabytes of weights -- is avoided.
    """
    sys.path.insert(0, str(source.parent.parent))
    try:
        from CEmulatorG.emulator.HMF import (  # type: ignore[import-not-found]
            HMFRockstarM200m_bcm_gp,
        )
        from CEmulatorG.utils import (  # type: ignore[import-not-found]
            NormCosmo,
            bary_limits,
            bary_names,
        )
    except ImportError as exc:  # pragma: no cover - depends on the environment
        print(f"  (reference implementation not importable: {exc})")
        return None

    warnings.filterwarnings("ignore")
    emulator = HMFRockstarM200m_bcm_gp(verbose=False)
    limits = {name: bary_limits[name] for name in bary_names}

    class _Probe:
        def get_data(self, row: np.ndarray) -> np.ndarray:
            emulator.X1norm = NormCosmo(row[None, :], bary_names, limits)
            return emulator.get_data()

    return _Probe


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
        help="write the bundle without checking it against the reference",
    )
    args = parser.parse_args(argv)

    print(f"building {args.output.name} from {args.source}")
    path = build(args.source, args.output, skip_verify=args.skip_verify)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
