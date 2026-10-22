#!/usr/bin/env python3
"""Build the bundled ``P_nn / P_cb`` ratio emulator.

The total-matter linear spectrum in :mod:`jet.emulator.pkmm` is composed from
the existing :math:`P_{cb}` bundle and a second GP for the neutrino-to-cb
ratio. This script repackages the reference's published PCA/GP arrays into
jet's bundle format, exactly as ``build_pklin_bundle.py`` does for :math:`P_{cb}`.

Run it from the repository root::

    python tools/build_pknn_ratio_bundle.py

The output lands in ``jet/data`` and is not tracked by git. The reference's
ratio model drops the one training cosmology with ``mnu = 0`` (row ``c0001``),
so the input table is filtered in the same way.
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
    KGRID_KEY,
    N_K,
    N_Z,
    Z_GRID,
    ZGRID_KEY,
    data_dir,
)
from jet.emulator.pkmm import (  # noqa: E402
    BUNDLE_NAME,
    ratio_data_vector_spec,
    ratio_theta_spec,
)
from jet.emulator.transforms import (  # noqa: E402
    PCA,
    BoundsNorm,
    Log10,
    StandardScaler,
)

#: Where the reference arrays live. Overridable with ``--source``.
DEFAULT_SOURCE = Path("/home/chenzhao/worksoftware/csstemu-jax/CEmulator/data")

#: The ratio GP removes this row (the only ``mnu = 0`` training cosmology).
DROPPED_ROW = 1

#: Redshifts of the reference grid, in the order its data vector stores them.
REFERENCE_Z = (3.0, 2.5, 2.0, 1.75, 1.5, 1.25, 1.0, 0.8, 0.5, 0.25, 0.1, 0.0)

#: Column of the primordial amplitude in the reference's cosmology table.
REFERENCE_AS_COLUMN = 4

#: The reference stores ``As * 1e9``.
REFERENCE_AS_SCALE = 1e-9

ALPHA = 1e-10

#: Descriptive only; the reference does not record explained variance.
UNKNOWN_VARIANCE = np.nan


def _reorder_to_ascending_redshift(pca_data: np.ndarray) -> np.ndarray:
    """Permute a reference PCA table from descending to ascending redshift."""
    permutation = np.arange(N_Z * N_K).reshape(N_Z, N_K)[::-1].ravel()
    return np.asarray(pca_data)[:, permutation]


def build(source: Path, destination: Path, verbose: bool = True) -> Path:
    """Read the reference ratio arrays and write the jet bundle."""
    table = source / "cosmologies_8d_train_n513_Sobol.npy"
    weights = source / "lgpkLin_nn_cb.npz"
    kfile = source / "karr_kmax100.npy"
    for path in (table, weights, kfile):
        if not path.exists():
            raise FileNotFoundError(f"reference file not found: {path}")

    spec = ratio_theta_spec()
    bounds = spec.bounds_array()

    # ---------------------------------------------------------------- inputs
    raw = np.load(table).astype(float)
    if raw.shape[1] != spec.dim:
        raise ValueError(f"expected {spec.dim} training columns, got {raw.shape[1]}")
    raw = np.delete(raw, DROPPED_ROW, axis=0)
    raw[:, REFERENCE_AS_COLUMN] *= REFERENCE_AS_SCALE

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

    if coefficients.shape[0] != x_train.shape[0]:
        raise ValueError(
            f"coefficient table has {coefficients.shape[0]} rows but the filtered "
            f"cosmology table has {x_train.shape[0]}"
        )

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
            x_train,
            constant_value[column],
            length_scale[column],
            ALPHA,
            "rbf",
            warn=None,
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
        y_spec=ratio_data_vector_spec(),
        x_chain=x_chain,
        y_chain=y_chain,
        backend=backend,
        extra={
            "description": "linear-theory P_nn / P_cb ratio",
            "source": str(source),
            "layout": "value[j * n_k + i] = ratio(k_i, z_j), z and k ascending",
            "redshifts": list(Z_GRID),
            "n_wavenumbers": N_K,
            "note": (
                "packaged by tools/build_pknn_ratio_bundle.py from a reference "
                "emulator's arrays; not trained by jet"
            ),
        },
        extra_arrays={KGRID_KEY: np.load(kfile), ZGRID_KEY: np.asarray(Z_GRID)},
    )
    if verbose:
        print(f"wrote  : {written} ({written.stat().st_size / 1e6:.1f} MB)")
    return written


def _reference_sigma8(source: Path, thetas: np.ndarray) -> np.ndarray | None:
    """Return total-matter sigma8 from the reference implementation, if importable."""
    sys.path.insert(0, str(source.parent.parent))
    try:
        from CEmulator.Emulator import CBaseEmulator
    except ImportError:
        return None

    emulator = CBaseEmulator(verbose=False)
    out = []
    for theta in thetas:
        Omegab, Omegam, H0, ns, As, w0, wa, mnu = theta
        emulator.set_cosmos(
            Omegab=Omegab,
            Omegac=Omegam - Omegab,
            H0=H0,
            # ``thetas`` are in the reference's A*1e9 units; set_cosmos wants
            # the physical amplitude.
            As=As * REFERENCE_AS_SCALE,
            ns=ns,
            w=w0,
            wa=wa,
            mnu=mnu,
            checkbound=False,
        )
        out.append(float(emulator.get_sigma8(type="Emulator")))
    return np.array(out)


def _verify(destination: Path, source: Path, verbose: bool = True) -> int:
    """Reload the ratio bundle and compare total sigma8 against the reference."""
    from jet.emulator import pkmm

    pkmm.load_ratio_emulator.cache_clear()
    pkmm.load_pklin_emulator.cache_clear()

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

    mine = pkmm.sigma8_total(jet_theta)
    reference = _reference_sigma8(source, reference_theta)
    if reference is None:
        if verbose:
            print("verify : reference implementation not importable; skipped")
        return 0

    worst = float(np.max(np.abs(mine / reference - 1.0)))
    if verbose:
        for got, want in zip(mine, reference, strict=True):
            print(f"verify : sigma8_total = {got:.12f}  reference = {want:.12f}")
        print(f"verify : worst relative difference {worst:.3e}")
    if worst > 1e-6:
        raise AssertionError(
            f"rebuilt total-matter emulator disagrees with the reference by "
            f"{worst:.3e}, which is far more than the two GP reconstructions allow"
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
