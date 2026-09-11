#!/usr/bin/env python3
"""
Build the bundled linear-theory P(k) emulator from a reference's raw arrays.

``jet`` does not train this emulator and does not ship a power-spectrum code. It
reads a set of arrays published by a reference Gaussian-process emulator and
repackages them into jet's own bundle format, so that the runtime path
(:mod:`jet.emulator.pklin`) needs nothing but NumPy and stays
``allow_pickle=False``.

Run it once, from the repository root::

    python tools/build_pklin_bundle.py

The outputs land in ``jet/data``, which is excluded from version control: the
weights are a few megabytes, are not jet's to license, and would otherwise be
retrained into every clone. See ``jet/data/README.md``.

What the transformation does
----------------------------
The reference stores four things: the training cosmologies, the principal
components of ``log10 P(k)`` on its grid, the Gaussian-process hyperparameters
per component, and the principal-component coefficients. Assembling those into
a jet bundle is a matter of expressing each stage as a jet transform:

===============  ==========================================================
reference stage  jet equivalent
===============  ==========================================================
limits -> [0,1]  ``BoundsNorm(bounds)``
mean/std         ``StandardScaler``
``log10 P(k)``   ``DataVectorSpec(log=True)`` -> ``Log10``
PCA projection   ``PCA``
GP per component ``GPBackend`` with fixed hyperparameters
===============  ==========================================================

Two deliberate departures from the reference layout:

* the primordial amplitude is stored in physical units (``As``, not
  ``As * 1e9``), so the parameter names match the rest of jet;
* the data vector is re-ordered to redshift-ascending, so that row 0 is
  ``z = 0`` and no consumer has to remember that the file is upside down.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jet.emulator.backends.gp import GPBackend, _factorise  # noqa: E402
from jet.emulator.bundle import save_bundle  # noqa: E402
from jet.emulator.pklin import (  # noqa: E402
    BUNDLE_NAME,
    KGRID_KEY,
    N_K,
    N_Z,
    Z_GRID,
    data_dir,
    data_vector_spec,
    theta_spec,
)
from jet.emulator.transforms import (  # noqa: E402
    PCA,
    BoundsNorm,
    Log10,
    StandardScaler,
)

#: Where the reference arrays live. Overridable with ``--source``; the default
#: is the checkout this package was built against.
DEFAULT_SOURCE = Path("/home/chenzhao/worksoftware/csstemu-jax/CEmulator/data")

#: Redshifts of the reference grid, in the order its data vector stores them.
REFERENCE_Z = (3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.8, 0.5, 0.25, 0.1, 0.0)

#: Column of the primordial amplitude in the reference's cosmology table.
REFERENCE_AS_COLUMN = 4

#: The reference stores this value instead of A_s. Its limits are quoted in the
#: same units, so scaling the column as well as the bounds is a no-op for the
#: normalisation -- it only makes the jet-side names mean what they say.
REFERENCE_AS_SCALE = 1e-9

ALPHA = 1e-10

#: The reference does not record how much variance its components capture, and
#: it cannot be recovered without the training spectra. The field is descriptive
#: only -- nothing in the prediction path reads it -- so it is filled with NaN
#: rather than a plausible-looking number.
UNKNOWN_VARIANCE = np.nan


def _reorder_to_ascending_redshift(pca_data: np.ndarray) -> np.ndarray:
    """Permute the columns of a reference PCA table into redshift order.

    The reference flattens its grid redshift-by-redshift starting from the
    highest redshift; jet's data vector is documented as starting from z = 0.
    Both the PCA mean and its components live in that flat layout, so both have
    to be permuted consistently.
    """
    permutation = np.arange(N_Z * N_K).reshape(N_Z, N_K)[::-1].ravel()
    return np.asarray(pca_data)[:, permutation]


def build(source: Path, destination: Path, verbose: bool = True) -> Path:
    """Read the reference arrays and write the jet bundle.

    Parameters
    ----------
    source : pathlib.Path
        Directory holding the reference's ``.npz``/``.npy`` files.
    destination : pathlib.Path
        jet's data directory; created if missing.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    pathlib.Path
        The bundle that was written.
    """
    table = source / "cosmologies_8d_train_n513_Sobol.npy"
    weights = source / "lgpkLin.npz"
    kfile = source / "karr_kmax100.npy"
    for path in (table, weights, kfile):
        if not path.exists():
            raise FileNotFoundError(f"reference file not found: {path}")

    spec = theta_spec()
    bounds = spec.bounds_array()

    # ---------------------------------------------------------------- inputs
    raw = np.load(table).astype(float)
    if raw.shape[1] != spec.dim:
        raise ValueError(f"expected {spec.dim} training columns, got {raw.shape[1]}")
    # Reference order is [Omegab, Omegam, H0, ns, A, w, wa, mnu] with A = As*1e9,
    # which is the same order as theta_spec() apart from the amplitude units.
    raw[:, REFERENCE_AS_COLUMN] *= REFERENCE_AS_SCALE
    if raw[:, REFERENCE_AS_COLUMN].max() > bounds[REFERENCE_AS_COLUMN, 1]:
        raise ValueError("the converted amplitude column escaped its declared bounds")

    normed = (raw - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    x_mean, x_scale = normed.mean(axis=0), normed.std(axis=0)
    x_train = (normed - x_mean) / x_scale

    if verbose:
        print(f"inputs : {x_train.shape[0]} training cosmologies, {x_train.shape[1]} parameters")

    # --------------------------------------------------------------- outputs
    with np.load(weights, allow_pickle=True) as archive:
        pca_data = np.asarray(archive["pca_data"], dtype=float)
        coefficients = np.asarray(archive["Bcoeff"], dtype=float)
        gprinfo = archive["gprinfo"]

    pca_data = _reorder_to_ascending_redshift(pca_data)
    n_components = coefficients.shape[1]
    if pca_data.shape[0] != n_components + 1:
        raise ValueError(
            f"expected {n_components} components plus a mean row, got {pca_data.shape[0]} rows"
        )
    if pca_data.shape[1] != N_Z * N_K:
        raise ValueError(f"expected {N_Z * N_K} grid points, got {pca_data.shape[1]}")

    c_mean, c_scale = coefficients.mean(axis=0), coefficients.std(axis=0)
    scaled_coefficients = (coefficients - c_mean) / c_scale

    if verbose:
        print(f"outputs: {n_components} principal components over {N_Z} redshifts x {N_K} k")

    # -------------------------------------------------------------- backend
    length_scale = np.array(
        [np.atleast_1d(gprinfo[i]["k2__length_scale"]) for i in range(n_components)],
        dtype=float,
    )
    constant_value = np.array(
        [float(gprinfo[i]["k1__constant_value"]) for i in range(n_components)], dtype=float
    )

    n_samples = x_train.shape[0]
    alpha_coef = np.empty((n_components, n_samples), dtype=float)
    noise = np.empty(n_components, dtype=float)
    for column in range(n_components):
        factor, used = _factorise(
            x_train, constant_value[column], length_scale[column], ALPHA, warn=None
        )
        noise[column] = used
        alpha_coef[column] = cho_solve((factor, True), scaled_coefficients[:, column])
    if np.any(noise > ALPHA):
        raise RuntimeError(
            "a covariance matrix needed a raised noise floor; the reference numbers "
            "would not be reproduced exactly, so this build is refused"
        )

    backend = GPBackend.from_state(
        {
            "X_train": x_train,
            "length_scale": length_scale,
            "constant_value": constant_value,
            "noise": noise,
            "y_mean": scaled_coefficients.mean(axis=0),
            "y_std": scaled_coefficients.std(axis=0),
            "alpha_coef": alpha_coef,
            "alpha_requested": np.asarray(ALPHA),
        }
    )

    # ----------------------------------------------------------------- chains
    x_chain = [
        BoundsNorm(bounds[:, 0], bounds[:, 1]),
        StandardScaler.from_state({"mean": x_mean, "scale": x_scale}),
    ]
    y_chain = [
        Log10(),
        PCA.from_state(
            {
                "mean": pca_data[0],
                "components": pca_data[1:],
                "explained_variance_ratio": np.full(n_components, UNKNOWN_VARIANCE),
                "n_components": np.asarray(n_components),
            }
        ),
        StandardScaler.from_state({"mean": c_mean, "scale": c_scale}),
    ]

    # ------------------------------------------------------------------ write
    destination.mkdir(parents=True, exist_ok=True)
    written = save_bundle(
        destination / BUNDLE_NAME,
        x_spec=spec,
        y_spec=data_vector_spec(),
        x_chain=x_chain,
        y_chain=y_chain,
        backend=backend,
        extra={
            "description": "linear-theory cold-dark-matter + baryon P(k)",
            "source": str(source),
            "layout": "value[j * n_k + i] = P(k_i, z_j), z and k ascending",
            "redshifts": list(Z_GRID),
            "n_wavenumbers": N_K,
            "note": (
                "packaged by tools/build_pklin_bundle.py from a reference emulator's "
                "arrays; not trained by jet"
            ),
        },
        # The wavenumber grid travels inside the bundle: a sibling file would be
        # one more thing to copy, and the two could silently disagree.
        extra_arrays={KGRID_KEY: np.load(kfile)},
    )

    if verbose:
        print(f"wrote  : {written} ({written.stat().st_size / 1e6:.1f} MB)")
    return written


def _reference_sigma8(source: Path, thetas: np.ndarray) -> np.ndarray | None:
    """Compute sigma8 with the reference implementation, if it is importable.

    Used only to check the build. ``csstemu`` is not a jet dependency, so a
    missing or unimportable package means the check is skipped rather than
    failed.
    """
    sys.path.insert(0, str(source.parent.parent))
    try:
        from CEmulator.emulator.PkLin import PkcbLin_gp
        from CEmulator.utils import NormCosmo, param_limits, param_names
    except ImportError:
        return None

    emulator = PkcbLin_gp()
    k = np.logspace(-4.99, 1.99, 1000)
    x = k * 8.0
    window = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3

    out = []
    for theta in thetas:
        emulator.ncosmo = NormCosmo(theta[None, :], param_names, param_limits)
        pk = emulator.get_pkcbLin(z=0, k=k)[0]
        out.append(np.sqrt(np.trapezoid(pk * window**2 * k**3 / (2 * np.pi**2), np.log(k))))
    return np.array(out)


def _verify(destination: Path, source: Path, verbose: bool = True) -> int:
    """Reload the bundle and compare sigma8 against the reference.

    Returns
    -------
    int
        Number of samples compared, or 0 when the reference is unavailable.
    """
    from jet.emulator import pklin

    pklin.load_pklin_emulator.cache_clear()
    pklin.k_grid.cache_clear()
    pklin._extra_arrays.cache_clear()
    emulator = pklin.load_pklin_emulator()

    # A few points spread over the training box, quoted in the *reference's*
    # units (so they can be handed straight to its own classes) and rescaled to
    # jet's physical amplitude for ours.
    base = np.array([0.049, 0.31, 67.66, 0.9665, 2.105, -1.0, 0.0, 0.06])
    reference_theta = np.array(
        [
            base,
            [0.055, 0.31, 67.66, 0.9665, 2.105, -1.0, 0.0, 0.06],
            [0.049, 0.28, 72.0, 0.975, 2.40, -0.9, 0.2, 0.10],
        ]
    )
    jet_theta = reference_theta.copy()
    jet_theta[:, REFERENCE_AS_COLUMN] *= REFERENCE_AS_SCALE

    mine = pklin.sigma8_of_pk(emulator.predict(jet_theta, return_std=False))
    reference = _reference_sigma8(source, reference_theta)
    if reference is None:
        if verbose:
            print("verify : reference implementation not importable; skipped")
        return 0

    worst = float(np.max(np.abs(mine / reference - 1.0)))
    if verbose:
        for got, want in zip(mine, reference, strict=True):
            print(f"verify : sigma8 = {got:.12f}  reference = {want:.12f}")
        print(f"verify : worst relative difference {worst:.3e}")
    # The two evaluations are mathematically identical but not bit-identical:
    # the distances come from a Gram matrix rather than scipy's cdist, and the
    # dual coefficients are recomputed here rather than reused. The covariance
    # has a condition number around 1e12, so a few hundred times double
    # precision is the expected spread. A structural mistake -- a wrong kernel,
    # a mis-ordered grid, a missing normalisation -- shows up as 1e-3 or worse,
    # so this bound still separates the two cases cleanly.
    if worst > 1e-8:
        raise AssertionError(
            f"rebuilt emulator disagrees with the reference by {worst:.3e}, which is "
            "far more than floating-point round-off allows for"
        )
    return reference_theta.shape[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=data_dir())
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    build(arguments.source, arguments.destination, verbose=not arguments.quiet)
    _verify(arguments.destination, arguments.source, verbose=not arguments.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
