"""
The bundled linear-theory :math:`P(k)` emulator behind the ``sigma8`` conversion.

:class:`jet.derived.Sigma8FromAs` needs to know :math:`\\sigma_8` as a function
of cosmology, and jet does not compute power spectra. It borrows a small
Gaussian-process emulator instead, which is read from ``jet/data`` and wrapped as
an ordinary :class:`~jet.emulator.Emulator` so that it reuses the same
transforms, backend and bundle format as everything else.

The emulator predicts :math:`\\log_{10} P_{\\mathrm{cb,lin}}(k, z)` on a fixed
grid: 12 redshifts from the reference file, and 1000 wavenumbers in
:math:`h/\\mathrm{Mpc}`. Its data vector is that grid flattened, redshift first::

    value[j * n_k + i]  ==  P(k_i, z_j)        0 <= j < N_Z, 0 <= i < N_K

with :data:`Z_GRID` and :func:`k_grid` giving the two axes in ascending order.
Line 0 is therefore :math:`z = 0`, which is the row :math:`\\sigma_8` is defined
from.

.. warning::
    The weights under ``jet/data`` are large and are *not* tracked by git; see
    the note in ``jet/data/README.md``. They are produced by
    ``tools/build_pklin_bundle.py`` from a reference emulator's published
    arrays, so the module below is a reader, not a trainer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..cosmology import cosmo_parameter_spec
from ..spec import DataVectorSpec, ParameterSpec
from .emulator import Emulator, data_dir, resolve_weights_path

__all__ = [
    "BUNDLE_NAME",
    "KGRID_KEY",
    "ZGRID_KEY",
    "N_K",
    "N_Z",
    "N_BINS",
    "Z_GRID",
    "INTEGRATION_K",
    "data_dir",
    "k_grid",
    "z_grid",
    "theta_spec",
    "data_vector_spec",
    "load_pklin_emulator",
    "pk_grid",
    "log_pk_at",
    "sigma_of_log_pk",
    "sigma_of_pk",
    "sigma8_of_pk",
]

#: Name of the ``.npz`` bundle under the data directory.
BUNDLE_NAME = "pklin.gp.npz"

#: Bundle key holding the wavenumber grid of the data vector. It travels inside
#: the bundle rather than as a sibling file, so that the two cannot drift apart.
KGRID_KEY = "k"

#: Bundle key holding the redshift grid, written by the builder since v0.2.0.
#: Older bundles carry only ``k``; :func:`z_grid` then falls back to
#: :data:`Z_GRID`.
ZGRID_KEY = "z"

#: Redshifts of the grid, ascending. Row 0 is ``z = 0``.
Z_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

N_Z = len(Z_GRID)

#: Wavenumbers in the grid, i.e. ``k_grid().size``.
N_K = 1000

#: Entries in the emulator's data vector.
N_BINS = N_Z * N_K

#: Wavenumbers the top-hat window integral is evaluated on. This is the grid the
#: reference implementation integrates over; it is deliberately *not* the
#: emulator's own ``k_grid()``, so that a :math:`\\sigma_8` computed here can be
#: compared against the reference number directly.
INTEGRATION_K = np.logspace(-4.99, 1.99, 1000)

#: Bounding box of the reference spline in ``(z, log10 k)``. Points outside are
#: clamped rather than extrapolated, matching the reference.
_SPLINE_BBOX = [0.0, 3.0, -6.0, 6.0]


def _data_path(name: str) -> Path:
    """Resolve one bundled data file, auto-fetching from GitHub Release if needed."""
    try:
        return resolve_weights_path(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"the bundled data file {name!r} is missing from {data_dir()}. It is not "
            "tracked by git; regenerate it with tools/build_pklin_bundle.py, or set "
            "JET_DATA_DIR to a directory that already holds it."
        ) from exc


@lru_cache(maxsize=1)
def _extra_arrays() -> dict[str, np.ndarray]:
    """Return the bundle's auxiliary arrays, read once per process."""
    from .bundle import load_bundle

    return dict(load_bundle(_data_path(BUNDLE_NAME))["extra_arrays"])


@lru_cache(maxsize=1)
def k_grid() -> np.ndarray:
    """Return the wavenumbers of the data vector, ascending, in ``h/Mpc``."""
    extras = _extra_arrays()
    if KGRID_KEY not in extras:
        raise KeyError(
            f"the bundled emulator has no {KGRID_KEY!r} grid; it was written without "
            "one and cannot be interpreted. Rebuild it with tools/build_pklin_bundle.py"
        )
    return extras[KGRID_KEY]


@lru_cache(maxsize=1)
def z_grid() -> np.ndarray:
    """Return the redshifts of the data vector, ascending (``z = 0`` first).

    Read from the bundle when present; bundles written before the redshift
    grid was stored carry only ``k``, in which case the module-level
    :data:`Z_GRID` constant is used.
    """
    extras = _extra_arrays()
    if ZGRID_KEY in extras:
        return extras[ZGRID_KEY]
    return np.asarray(Z_GRID)


def theta_spec() -> ParameterSpec:
    """Return the input specification of the bundled emulator.

    The bounds are the region the emulator was trained on. ``As`` is the
    physical primordial amplitude; the reference file stores it multiplied by
    ``1e9``, which the builder script converts.

    Returns
    -------
    ParameterSpec
        Eight cosmological parameters, in the emulator's column order.
    """
    return cosmo_parameter_spec()


def data_vector_spec() -> DataVectorSpec:
    """Return the output specification of the bundled emulator."""
    return DataVectorSpec(name="pkcb_lin", n_bins=N_BINS, log=True)


@lru_cache(maxsize=1)
def load_pklin_emulator() -> Emulator:
    """Load the bundled :math:`P(k)` emulator.

    The result is cached, because the conversion calls it once per iteration of
    the :math:`\\sigma_8` inversion and the file is a few megabytes.

    Returns
    -------
    Emulator
        Predicts :math:`P(k, z)` in :math:`(\\mathrm{Mpc}/h)^3` on the flattened
        :data:`Z_GRID` x :func:`k_grid` layout.

    Raises
    ------
    FileNotFoundError
        If the data files are absent.
    """
    # Extrapolation warnings are suppressed on purpose: the sigma8 inversion
    # probes intermediate As values, some of which legitimately leave the box,
    # and one warning per iteration would drown the caller's own. The caller
    # sees a warning about *its* parameters from the outer emulator instead.
    emulator = Emulator.load(_data_path(BUNDLE_NAME))
    emulator.warn_on_extrapolation = False
    return emulator


def pk_grid(Pk: np.ndarray) -> np.ndarray:
    """Reshape a data vector into the ``(n_samples, N_Z, N_K)`` grid."""
    Pk = np.atleast_2d(np.asarray(Pk, dtype=float))
    if Pk.shape[1] != N_BINS:
        raise ValueError(
            f"expected data vectors of length {N_BINS} ({N_Z} redshifts x {N_K} "
            f"wavenumbers), got {Pk.shape[1]}"
        )
    return Pk.reshape(Pk.shape[0], N_Z, N_K)


def log_pk_at(log10_Pk: np.ndarray, z: float = 0.0) -> np.ndarray:
    """Interpolate :math:`\\log_{10} P(k)` onto :func:`k_grid` at one redshift.

    The redshift direction is a cubic spline and the wavenumber direction is
    linear in :math:`\\log_{10} k`, matching the reference implementation.

    Parameters
    ----------
    log10_Pk : ndarray of shape (n_samples, N_BINS)
        Data vectors in :math:`\\log_{10}` of the physical power spectrum, i.e.
        what the emulator's transform chain consumes.
    z : float, optional
        Redshift to evaluate at.

    Returns
    -------
    ndarray of shape (n_samples, N_K)
        :math:`\\log_{10} P(k)` on :func:`k_grid`, still in log space.
    """
    from scipy.interpolate import RectBivariateSpline

    grid = pk_grid(log10_Pk)
    k = k_grid()
    log_k = np.log10(k)
    out = np.empty((grid.shape[0], k.size), dtype=float)

    for row in range(grid.shape[0]):
        spline = RectBivariateSpline(z_grid(), log_k, grid[row], kx=3, ky=1, bbox=_SPLINE_BBOX, s=0)
        out[row] = spline(z, log_k)[0]

    return out


def sigma_of_log_pk(log10_Pk: np.ndarray, R: float | np.ndarray) -> np.ndarray:
    r"""Top-hat variance from :math:`\log_{10} P(k)` already on :func:`k_grid`.

    Computes

    .. math::
        \sigma^2(R) = \int P(k)\, W^2(kR)\, \frac{k^3}{2\pi^2}\, d\ln k,
        \qquad W(x) = \frac{3(\sin x - x\cos x)}{x^3}

    on :data:`INTEGRATION_K`, with :math:`P(k)` interpolated from
    :func:`k_grid`. The window is the Fourier transform of a real-space top
    hat, so :math:`\sigma_R` is the RMS matter fluctuation in spheres of radius
    ``R``.

    Split out from :func:`sigma_of_pk` because the halo mass function needs
    :math:`\sigma_R` for hundreds of radii at each of twelve redshifts that
    come from a single emulator call: doing the redshift spline once and the
    window integral many times is the natural order, and it keeps the one place
    where the integration measure is decided in one place.

    Parameters
    ----------
    log10_Pk : ndarray of shape (n_z, N_K)
        :math:`\log_{10} P(k)` on :func:`k_grid`, one row per redshift.
    R : float or array-like
        Smoothing scale(s) in :math:`\mathrm{Mpc}/h`.

    Returns
    -------
    ndarray of shape (n_z, n_R)
        :math:`\sigma_R`, dimensionless. The radius axis is squeezed when ``R``
        is a scalar, so the result is then ``(n_z,)``.
    """
    log10_Pk = np.atleast_2d(np.asarray(log10_Pk, dtype=float))
    if log10_Pk.shape[1] != N_K:
        raise ValueError(
            f"expected {N_K} wavenumbers (the emulator's k grid), got {log10_Pk.shape[1]}"
        )

    # A linear interpolation in log10(k) from the emulator's own grid onto the
    # integration grid, matching the reference's single bivariate spline -- with
    # ky=1 the k direction is piecewise linear either way.
    log10_k_source = np.log10(k_grid())
    log10_k = np.log10(INTEGRATION_K)

    # The integral is over d ln k, not d log10 k. The two differ by a constant
    # factor, so getting it wrong does not show up as a small offset -- it
    # rescales sigma by exactly 1/sqrt(ln 10), about 34 per cent. The
    # interpolation itself may be done in either variable, since a constant
    # factor on the abscissa leaves a linear interpolant unchanged.
    ln_k = np.log(INTEGRATION_K)

    scalar_R = np.ndim(R) == 0
    R = np.atleast_1d(np.asarray(R, dtype=float))
    if np.any(R <= 0.0):
        raise ValueError(f"R must be positive, got {R.min()}")

    x = INTEGRATION_K[None, :] * R[:, None]
    window = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    measure = INTEGRATION_K**3 / (2.0 * np.pi**2)

    out = np.empty((log10_Pk.shape[0], R.size), dtype=float)
    for row in range(log10_Pk.shape[0]):
        log_pk = np.interp(log10_k, log10_k_source, log10_Pk[row])
        integrand = np.power(10.0, log_pk)[None, :] * window**2 * measure[None, :]
        out[row] = np.sqrt(np.trapezoid(integrand, ln_k, axis=1))

    return out[:, 0] if scalar_R else out


def sigma_of_pk(Pk: np.ndarray, R: float | np.ndarray, z: float = 0.0) -> np.ndarray:
    r"""Top-hat variance from physical power spectra at one redshift.

    Applies the redshift spline of :func:`log_pk_at` and hands the result to
    :func:`sigma_of_log_pk`; see that function for the integral itself.

    Parameters
    ----------
    Pk : ndarray of shape (n_samples, N_BINS)
        Power spectra in :math:`(\mathrm{Mpc}/h)^3`, as returned by the
        emulator.
    R : float or array-like
        Smoothing scale(s) in :math:`\mathrm{Mpc}/h`.
    z : float, optional
        Redshift.

    Returns
    -------
    ndarray of shape (n_samples, n_R)
        :math:`\sigma_R(z)`, dimensionless. The radius axis is squeezed when
        ``R`` is a scalar.
    """
    Pk = np.atleast_2d(np.asarray(Pk, dtype=float))
    if np.any(Pk <= 0.0):
        raise ValueError("P(k) must be strictly positive everywhere to be log-interpolated")
    return sigma_of_log_pk(log_pk_at(np.log10(Pk), z=z), R)


def sigma8_of_pk(Pk: np.ndarray, R: float = 8.0, z: float = 0.0) -> np.ndarray:
    r"""Integrate power spectra against a top-hat window to get :math:`\sigma_R`.

    Computes

    .. math::
        \\sigma^2(R) = \\int P(k)\\, W^2(kR)\\, \\frac{k^3}{2\\pi^2}\\, d\\ln k,
        \\qquad W(x) = \\frac{3(\\sin x - x\\cos x)}{x^3}

    on :data:`INTEGRATION_K`, with :math:`P(k)` interpolated from the emulator's
    grid. The window is the Fourier transform of a real-space top hat, so
    :math:`\\sigma_8` is the RMS matter fluctuation in spheres of
    :math:`8\\,\\mathrm{Mpc}/h`.

    Parameters
    ----------
    Pk : ndarray of shape (n_samples, N_BINS)
        Power spectra in :math:`(\\mathrm{Mpc}/h)^3`, as returned by the
        emulator.
    R : float, optional
        Smoothing scale in :math:`\\mathrm{Mpc}/h`.
    z : float, optional
        Redshift.

    Returns
    -------
    ndarray of shape (n_samples,)
        :math:`\\sigma_R(z)`, dimensionless.

    Notes
    -----
    A thin wrapper over :func:`sigma_of_pk` that fixes the radius axis to the
    single value :math:`\sigma_8` is defined at. Kept separate because the
    :math:`\sigma_8` golden value is what the derived-parameter machinery is
    regression-tested against, and a conversion should not have to index into a
    two-dimensional result to get it.
    """
    values = sigma_of_pk(Pk, R=R, z=z)
    if values.ndim != 1:
        raise AssertionError("a scalar R must give a one-dimensional result")
    return values
