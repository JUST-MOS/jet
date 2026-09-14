r"""
The bundled halo-matter correlation function emulator.

Predicts :math:`\xi_{hm}(r \mid z, \bar{n})`, the cross-correlation between a
halo sample selected by a cumulative number-density threshold and the matter
field, over 12 redshifts, 6 density thresholds and 37 separations.

Structure
---------
The bundled Gaussian process does *not* predict :math:`\xi_{hm}`. It predicts
the **effective bias ratio**

.. math::
    B_{hm}(r \mid z, \lg\bar{n}) = \frac{\xi_{hm}(r)}{\xi_{mm}(r)}

on a fixed box, and the physical correlation function is formed by multiplying
by a matter correlation function the caller supplies. The reason is that the
ratio is a far smoother function of cosmology than either factor: the
cosmological dependence of :math:`\xi_{mm}` is already carried by the linear
power spectrum, and leaving it inside the regression would make the emulator
relearn it at every :math:`r`. The reference implementation does the same, and
:meth:`XiHMEmulator.xihm_lgnbar_threshold` is where the multiplication, the
large-scale linear-bias baseline and the blend between them happen.

Layout
------
The data vector is the box flattened in C order, redshift outermost and
separation innermost::

    value[(i_z * N_LGDEN + i_den) * N_R + i_r]  ==  B_hm(r_i, z_j, lgden_k)

with :func:`z_grid`, :func:`lgden_grid` and :func:`r_mid` giving the three axes
in ascending order. Line 0 is :math:`z = 0`; the reference stores the redshifts
the other way round, and the builder reverses that axis, which leaves the PCA
scores untouched.

Two of the three axes are *interpolation* axes rather than emulator axes. The
Gaussian process sees only the eight cosmological parameters; ``z``, ``lgden``
and ``r`` are evaluated afterwards by linear interpolation of the predicted box
(linear in :math:`\log_{10} r`), exactly as the reference does it. That keeps
the box, and therefore the weights, independent of the sample being emulated.

.. warning::
    The weights under ``jet/data`` are not tracked by git; see the note in
    ``jet/data/README.md``. They are rebuilt from the reference's published
    arrays by ``tools/build_xihm_bundle.py``, so this module is a reader and a
    wrapper, not a trainer.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..cosmology import cosmo_parameter_spec
from ..spec import DataVectorSpec, ParameterSpec
from .bhm import LGDEN_GRID, LGDEN_KEY, N_LGDEN, Z_GRID, ZGRID_KEY, warn_outside
from .emulator import Emulator, data_dir, resolve_weights_path

__all__ = [
    "BUNDLE_NAME",
    "ZGRID_KEY",
    "LGDEN_KEY",
    "RMID_KEY",
    "N_Z",
    "N_LGDEN",
    "N_R",
    "N_BINS",
    "Z_GRID",
    "LGDEN_GRID",
    "R_EDGES",
    "R_MID",
    "data_dir",
    "z_grid",
    "lgden_grid",
    "r_mid",
    "R_SWITCH",
    "XIMM_K",
    "XIMM_KFFT",
    "TREE_K",
    "theta_spec",
    "data_vector_spec",
    "load_brhm_emulator",
    "bias_ratio",
    "brhm_grid",
    "brhm_interpolate",
    "ximm_linear",
    "xihm_lgnbar_threshold",
    "xihm_mass",
    "XiHMEmulator",
]

#: Name of the ``.npz`` bundle under the data directory.
BUNDLE_NAME = "xihm_brhm_rockstar_m200m.gp.npz"

#: Bundle key holding the separation grid.
RMID_KEY = "r_mid"

#: Edges of the separation bins in :math:`h^{-1}\mathrm{Mpc}`: 30 log-spaced
#: bins from 0.01 to 10, then 5 Mpc steps up to 50.
R_EDGES = np.concatenate([np.logspace(-2, 1, 31)[:-1], np.arange(10, 50, 5)])

#: Midpoints of :data:`R_EDGES`, the separations the box is defined on.
#:
#: The reference applies an ``r_mid < 50`` filter here. It is always all-true
#: (the largest midpoint is 42.5) and is kept only so that the grid can be
#: recognised as the same one; extending :data:`R_EDGES` upwards would silently
#: drop bins in the reference and raises here instead.
R_MID = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])

N_Z = len(Z_GRID)
N_R = R_MID.size

#: Entries in the emulator's data vector.
N_BINS = N_Z * N_LGDEN * N_R

#: Separation at which the emulated correlation function hands over to the
#: linear-bias baseline, in :math:`h^{-1}\mathrm{Mpc}`.
R_SWITCH = 40.0

#: Wavenumbers the linear power spectrum is sampled at before the transform to
#: :math:`\xi_{mm}`. The reference's 512 points, not the emulator's own grid:
#: the resampling is part of the number being reproduced.
XIMM_K = np.logspace(-4.99, 1.99, 512)

#: Wavenumbers the FFTLog transform to :math:`\xi_{mm}` actually runs on.
XIMM_KFFT = np.logspace(-5, 5, 1024)

#: Wavenumbers the baseline's transform runs on. Wider than
#: :data:`XIMM_KFFT` at the low end and narrower at the high end, again matching
#: the reference; the two transforms therefore do not share a truncation error.
TREE_K = np.logspace(-5, 3, 1024)


def _check_grid_consistency() -> None:
    """Fail loudly if the module-level grid constants disagree with each other.

    :data:`R_EDGES` is written out rather than read from the bundle, so that the
    binning the reference used is visible here; these assertions are what keeps
    that transcription honest. The redshift and threshold axes are not checked
    here: they belong to :mod:`jet.emulator.bhm` and are imported, so they
    cannot drift.
    """
    if N_BINS != N_Z * N_LGDEN * N_R:
        raise AssertionError("module grid constants disagree")
    if not np.all(np.diff(R_MID) > 0.0):
        raise AssertionError("R_MID must be strictly increasing")
    if not np.isclose(R_MID[-1], 42.5):
        raise AssertionError(f"R_MID no longer ends at 42.5, it ends at {R_MID[-1]}")


_check_grid_consistency()


def _data_path(name: str) -> Path:
    """Resolve one bundled data file, auto-fetching from GitHub Release if needed."""
    try:
        return resolve_weights_path(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"the bundled data file {name!r} is missing from {data_dir()}. It is not "
            "tracked by git; regenerate it with tools/build_xihm_bundle.py, or set "
            "JET_DATA_DIR to a directory that already holds it."
        ) from exc


@lru_cache(maxsize=1)
def _extra_arrays() -> dict[str, np.ndarray]:
    """Return the bundle's auxiliary arrays, read once per process.

    An absent bundle is not an error here. The three axes are module constants
    -- they describe the model, not the fit -- and the bundle only restates them
    so that a weights file is self-describing. Callers that want the *values*
    go through :func:`load_brhm_emulator`, which does raise.
    """
    from .bundle import load_bundle

    try:
        return dict(load_bundle(_data_path(BUNDLE_NAME))["extra_arrays"])
    except FileNotFoundError:
        return {}


def _grid_from_bundle(key: str, fallback: np.ndarray) -> np.ndarray:
    """Return one axis, preferring the bundle's copy over the module constant."""
    extras = _extra_arrays()
    if key in extras:
        return extras[key]
    return np.asarray(fallback)


@lru_cache(maxsize=1)
def z_grid() -> np.ndarray:
    """Return the redshifts of the box, ascending (``z = 0`` first)."""
    return _grid_from_bundle(ZGRID_KEY, np.asarray(Z_GRID))


@lru_cache(maxsize=1)
def lgden_grid() -> np.ndarray:
    """Return the number-density thresholds, ascending."""
    return _grid_from_bundle(LGDEN_KEY, np.asarray(LGDEN_GRID))


@lru_cache(maxsize=1)
def r_mid() -> np.ndarray:
    """Return the separations of the box in :math:`h^{-1}\\mathrm{Mpc}`, ascending."""
    return _grid_from_bundle(RMID_KEY, R_MID)


def theta_spec() -> ParameterSpec:
    """Return the input specification of the bundled emulator.

    The eight cosmological parameters are the ones every bundled model shares;
    see :func:`jet.cosmology.cosmo_parameter_spec`. ``As`` is the physical
    primordial amplitude, while the reference's files store it multiplied by
    ``1e9`` -- the builder converts.
    """
    return cosmo_parameter_spec()


def data_vector_spec() -> DataVectorSpec:
    """Return the output specification of the bundled emulator.

    ``log`` is false on purpose: the emulated quantity is a ratio, not a
    positive-definite density, and the reference's own chain applies no
    logarithm to it.
    """
    return DataVectorSpec(name="brhm_rockstar_m200m", n_bins=N_BINS, log=False)


@lru_cache(maxsize=1)
def load_brhm_emulator() -> Emulator:
    """Load the bundled bias-ratio emulator.

    The result is cached; the file is read once per process and the caller may
    keep a reference to it.

    Returns
    -------
    Emulator
        Predicts :math:`B_{hm}(r, z, \\lg\\bar{n})` on the flattened layout
        described in the module docstring.

    Raises
    ------
    FileNotFoundError
        If the data files are absent.
    """
    # Extrapolation warnings are suppressed on purpose. Every physical entry
    # point probes the box at a *displaced* density threshold -- the mass
    # threshold is converted into one by a finite difference -- and those probes
    # deliberately leave the box at the edges. One warning per probe would drown
    # the caller's own. The wrapper checks the caller's own arguments instead.
    emulator = Emulator.load(_data_path(BUNDLE_NAME))
    emulator.warn_on_extrapolation = False
    return emulator


def bias_ratio(theta: np.ndarray) -> np.ndarray:
    r"""Return the bias ratio of the bundled model on the box.

    The module-level counterpart of :meth:`XiHMEmulator.bias_ratio`, for callers
    that want the box without constructing a wrapper. The emulator itself is
    loaded once per process.

    Parameters
    ----------
    theta : ndarray of shape (n, 8)
        Cosmological parameters.

    Returns
    -------
    ndarray of shape (n, N_BINS)
        The flat bias-ratio data vector.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    return np.asarray(load_brhm_emulator().predict(theta, return_std=False))


def brhm_grid(values: np.ndarray) -> np.ndarray:
    """Reshape data vectors into the ``(n_samples, N_Z, N_LGDEN, N_R)`` box."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    if values.shape[1] != N_BINS:
        raise ValueError(
            f"expected data vectors of length {N_BINS} ({N_Z} redshifts x {N_LGDEN} "
            f"thresholds x {N_R} separations), got {values.shape[1]}"
        )
    return values.reshape(values.shape[0], N_Z, N_LGDEN, N_R)


def brhm_interpolate(
    values: np.ndarray,
    z: float | np.ndarray,
    lgden: float | np.ndarray,
    r: float | np.ndarray,
) -> np.ndarray:
    r"""Evaluate predicted boxes at arbitrary ``(z, lgden, r)``.

    Linear in each axis, and linear in :math:`\log_{10} r` rather than in
    :math:`r`, matching the reference. Points outside the box are extrapolated
    with a :class:`RuntimeWarning`; the caller decides whether that is
    acceptable, because for :math:`r` above the box the answer is blended away
    by :meth:`XiHMEmulator.xihm_lgnbar_threshold` anyway.

    Parameters
    ----------
    values : ndarray of shape (n_samples, N_BINS)
        Predicted bias ratios on the box.
    z, lgden, r : float or array-like
        Redshift, number-density threshold and separation. ``r`` must be
        strictly positive, since the interpolation is in its logarithm.

    Returns
    -------
    ndarray of shape (n_samples, n_lgden, n_z, n_r)
        The reference's axis order.
    """
    grid = brhm_grid(values)

    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    lgden_in = np.atleast_1d(np.asarray(lgden, dtype=float))
    r_in = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(r_in <= 0.0):
        raise ValueError(f"r must be strictly positive, got {r_in.min()}")

    z_axis = z_grid()
    lgden_axis = lgden_grid()
    r_axis = r_mid()
    warn_outside("z", z_in, float(z_axis[0]), float(z_axis[-1]))
    warn_outside("lgden", lgden_in, float(lgden_axis[0]), float(lgden_axis[-1]))
    warn_outside("r", r_in, float(r_axis[0]), float(r_axis[-1]))

    n_den, n_z, n_r = lgden_in.size, z_in.size, r_in.size
    out = np.empty((grid.shape[0], n_den, n_z, n_r), dtype=float)

    # The query grid repeats for every sample, so it is built once and only the
    # values are swapped out per sample: interpolating a (n_den, n_z, n_r)
    # point cloud is the expensive part, not constructing the interpolator.
    lg_mesh, z_mesh, lr_mesh = np.meshgrid(lgden_in, z_in, np.log10(r_in), indexing="ij")
    points = np.stack([z_mesh, lg_mesh, lr_mesh], axis=-1).reshape(-1, 3)

    for row in range(grid.shape[0]):
        interpolator = RegularGridInterpolator(
            (z_axis, lgden_axis, np.log10(r_axis)),
            grid[row],
            method="linear",
            bounds_error=False,
            fill_value=None,  # linear extrapolation, as the reference does
        )
        out[row] = interpolator(points).reshape(n_den, n_z, n_r)

    return out


# ----------------------------------------------------------------------
# The physics the ratio is multiplied into
# ----------------------------------------------------------------------
def _require_positive_r(r: np.ndarray) -> np.ndarray:
    """Return ``r`` as an array, raising if it is not strictly positive."""
    r_in = np.atleast_1d(np.asarray(r, dtype=float))
    if np.any(r_in <= 0.0):
        raise ValueError(f"r must be strictly positive, got {r_in.min()}")
    return r_in


def ximm_linear(theta: np.ndarray, z: float | np.ndarray, r: float | np.ndarray) -> np.ndarray:
    r"""Return the linear matter correlation function :math:`\xi_{mm}(r, z)`.

    The Hankel transform of the bundled linear :math:`P_{cb}(k)`, on the
    reference's own two grids: the emulator is sampled at 512 wavenumbers, the
    result is splined logarithmically onto 1024, transformed by FFTLog, and the
    correlation function is then read off with a linear spline in
    :math:`\log_{10} r`. Every one of those steps is reproduced because the
    values are compared point by point -- the grid sizes in particular, which
    set FFTLog's ringing and would otherwise move the answer at the fourth
    digit.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters.
    z, r : float or array-like
        Redshifts and separations in :math:`h^{-1}\mathrm{Mpc}`.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_r)
        :math:`\xi_{mm}`, dimensionless.
    """
    from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

    from ..fourier import p2xi
    from .pklin import load_pklin_emulator, log_pk_at_k

    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    r_in = _require_positive_r(r)

    emulator = load_pklin_emulator()
    log10_pk = np.log10(np.asarray(emulator.predict(theta, return_std=False), dtype=float))
    sampled = np.power(10.0, log_pk_at_k(log10_pk, XIMM_K, z_in))

    log10_k = np.log10(XIMM_K)
    log10_kfft = np.log10(XIMM_KFFT)
    out = np.empty((theta.shape[0], z_in.size, r_in.size), dtype=float)
    for row in range(theta.shape[0]):
        for index in range(z_in.size):
            # Cubic in log k, extrapolated past both ends of the sampled range:
            # the transform needs a wide, clean, periodic grid and the emulator
            # only covers seven decades of it.
            spline = interp1d(
                log10_k, np.log10(sampled[row, index]), kind="cubic", fill_value="extrapolate"
            )
            power = np.power(10.0, spline(log10_kfft))
            radii, xi = p2xi(XIMM_KFFT, power, l=0)
            out[row, index] = InterpolatedUnivariateSpline(np.log10(radii), xi.real, k=1, ext=0)(
                np.log10(r_in)
            )

    return out


def _tree_baseline(theta: np.ndarray, z: np.ndarray, r: np.ndarray, bias: np.ndarray) -> np.ndarray:
    r"""Return the linear-bias baseline for a set of halo samples.

    A halo sample biased by :math:`b` traces the matter field linearly at large
    separations, so :math:`\xi_{hm} = b\,\xi_{mm}` there. This computes it the
    long way round -- :math:`b \times P_{\mathrm{lin}}`, Hankel-transformed --
    rather than as :math:`b \times \xi_{mm}`, because that is what the reference
    does and because the two differ by more than round-off: the transform here
    runs on a different k grid, with power-law extrapolation, so its answer
    carries a different truncation error.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters.
    z, r : ndarray
        Redshifts and separations.
    bias : ndarray of shape (n_samples, n_bias, n_z)
        Biases, one per halo sample, redshift and cosmology.

    Returns
    -------
    ndarray of shape (n_samples, n_bias, n_z, n_r)
        The baseline correlation function.
    """
    from scipy.interpolate import UnivariateSpline

    from ..fourier import p2xi
    from .pklin import load_pklin_emulator, log_pk_at_k

    emulator = load_pklin_emulator()
    log10_pk = np.log10(np.asarray(emulator.predict(theta, return_std=False), dtype=float))
    # k reaches 1e3 here, an order of magnitude past the emulator's own grid;
    # `log_pk_at_k` extrapolates in log k, which is what the reference's spline
    # bounding box does too. The transform's own power-law padding then
    # continues it to the ends of the FFT grid.
    linear = np.power(10.0, log_pk_at_k(log10_pk, TREE_K, z))

    out = np.empty((theta.shape[0], bias.shape[1], z.size, r.size), dtype=float)
    for row in range(theta.shape[0]):
        for index in range(bias.shape[1]):
            for iz in range(z.size):
                power = linear[row, iz] * bias[row, index, iz]
                radii, xi = p2xi(TREE_K, power, l=0, ext=3)
                out[row, index, iz] = UnivariateSpline(radii, xi.real, k=1, s=0, ext=0)(r)
    return out


def _blend(direct: np.ndarray, tree: np.ndarray, r: np.ndarray) -> np.ndarray:
    r"""Blend the emulated correlation function onto the linear-bias baseline.

    .. math::
        w(r) = e^{-(r/40)^4}

    weights the emulated term at small separations and the baseline at large
    ones. The fourth power is what makes the switch-over sharp: the weight is
    ``0.37`` at :math:`40\,h^{-1}\mathrm{Mpc}`, ``0.28`` at the edge of the
    emulated box, ``6e-3`` at 60, ``1e-7`` at 80 and ``1e-17`` at 100.

    Those numbers are why asking for separations beyond the box's
    :math:`42.5` is not as dangerous as it looks: the weight on the
    extrapolated part falls by four orders of magnitude over the decade past
    it, so the region where the extrapolation genuinely matters is roughly
    :math:`42` to :math:`80`. Past that the answer is the baseline and the
    emulator contributes nothing.
    """
    weight = np.exp(-((r / R_SWITCH) ** 4))
    shape = (1,) * (direct.ndim - 1) + (r.size,)
    return direct * weight.reshape(shape) + tree * (1.0 - weight.reshape(shape))


def xihm_lgnbar_threshold(
    theta: np.ndarray,
    z: float | np.ndarray,
    r: float | np.ndarray,
    lgden: float | np.ndarray,
    hmf_emulator=None,
) -> np.ndarray:
    r"""Return :math:`\xi_{hm}` for a cumulative number-density threshold.

    The sample is selected by :math:`\bar{n}(\geq M)`, so the bias that enters
    the baseline is the density-threshold bias.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters.
    z, r : float or array-like
        Redshifts and separations in :math:`h^{-1}\mathrm{Mpc}`.
    lgden : float or array-like
        Cumulative number-density thresholds in
        :math:`\log_{10}(h^3\,\mathrm{Mpc}^{-3})`.

    Returns
    -------
    ndarray of shape (n_samples, n_lgden, n_z, n_r)
        :math:`\xi_{hm}`. The axis order is the reference's, with the cosmology
        prepended.
    """
    from .bhm import bias_lgnbar_threshold

    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    r_in = _require_positive_r(r)
    lgden_in = np.atleast_1d(np.asarray(lgden, dtype=float))

    ximm = ximm_linear(theta, z=z_in, r=r_in)
    ratio = brhm_interpolate(bias_ratio(theta), z=z_in, lgden=lgden_in, r=r_in)
    direct = ratio * ximm[:, None, :, :]

    bias = bias_lgnbar_threshold(theta, z=z_in, lgden=lgden_in, hmf_emulator=hmf_emulator)
    tree = _tree_baseline(theta, z_in, r_in, np.transpose(bias, (0, 2, 1)))

    return _blend(direct, tree, r_in)


def xihm_mass(
    theta: np.ndarray,
    z: float | np.ndarray,
    r: float | np.ndarray,
    M: np.ndarray,
    hmf_emulator=None,
) -> np.ndarray:
    r"""Return :math:`\xi_{hm}` for a fixed mass threshold.

    A mass threshold is not a density threshold, so the emulated ratio cannot be
    evaluated once and reused. The reference converts by perturbing the mass by
    one per cent each way, forming the *density-threshold* correlation functions
    at those two densities, and taking the density-weighted difference --
    the finite-difference form of holding :math:`\bar{n}` fixed instead of
    :math:`M`. Reproduced with the same one per cent, including the reversal of
    the mass axis that the reference's array layout requires.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters.
    z, r : float or array-like
        Redshifts and separations in :math:`h^{-1}\mathrm{Mpc}`.
    M : array-like
        Halo masses in :math:`M_\odot/h`.

    Returns
    -------
    ndarray of shape (n_samples, n_M, n_z, n_r)
        :math:`\xi_{hm}`. Bins whose number density underflows to zero are
        returned as zero.
    """
    from .bhm import MASS_PERTURBATION, bias_mass
    from .hmf import HMFEmulator

    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    r_in = _require_positive_r(r)
    M = np.atleast_1d(np.asarray(M, dtype=float))

    hmf = HMFEmulator.load() if hmf_emulator is None else hmf_emulator
    ximm = ximm_linear(theta, z=z_in, r=r_in)
    masses_up = M * (1.0 + MASS_PERTURBATION)
    masses_down = M * (1.0 - MASS_PERTURBATION)

    out = np.empty((theta.shape[0], M.size, z_in.size, r_in.size), dtype=float)
    for row in range(theta.shape[0]):
        # Reversed, so that the density read off the mass axis increases with
        # index; see jet.emulator.bhm.bias_mass for why that matters. The result
        # is reversed back at the end.
        density_up = np.asarray(hmf.number_density(theta[row : row + 1], z=z_in, M=masses_up))[0][
            :, ::-1
        ]
        density_down = np.asarray(hmf.number_density(theta[row : row + 1], z=z_in, M=masses_down))[
            0
        ][:, ::-1]

        reversed_block = np.zeros((M.size, z_in.size, r_in.size), dtype=float)
        for index in range(z_in.size):
            usable = (density_up[index] > 0.0) & (density_down[index] > 0.0)
            if not np.any(usable):
                continue
            density_upper = density_up[index][usable]
            density_lower = density_down[index][usable]
            # One Brhm prediction covers both perturbed thresholds.
            upper = (
                brhm_interpolate(
                    bias_ratio(theta[row : row + 1]),
                    z=z_in[index],
                    lgden=np.log10(density_upper),
                    r=r_in,
                )[0, :, 0, :]
                * ximm[row, index][None, :]
            )
            lower = (
                brhm_interpolate(
                    bias_ratio(theta[row : row + 1]),
                    z=z_in[index],
                    lgden=np.log10(density_lower),
                    r=r_in,
                )[0, :, 0, :]
                * ximm[row, index][None, :]
            )
            reversed_block[usable, index] = (
                upper * density_upper[:, None] - lower * density_lower[:, None]
            ) / (density_upper - density_lower)[:, None]

        direct = reversed_block[::-1]

        bias = bias_mass(theta[row : row + 1], z=z_in, M=M, hmf_emulator=hmf)
        tree = _tree_baseline(theta[row : row + 1], z_in, r_in, np.transpose(bias, (0, 2, 1)))[0]
        out[row] = _blend(direct, tree, r_in)

    return out


class XiHMEmulator:
    r"""The bundled halo-matter correlation function emulator.

    Wraps the generic :class:`~jet.emulator.Emulator` and adds the physics that
    turns its output -- a bias ratio -- into :math:`\xi_{hm}`.

    Parameters
    ----------
    emulator : Emulator, optional
        The fitted bias-ratio model. Defaults to the bundled one.
    warn_on_extrapolation : bool, optional
        Whether to warn when ``z``, ``lgden`` or ``r`` leaves the emulated box.
        Warnings about the *cosmological* parameters are always issued by the
        inner emulator and are not controlled here.
    """

    def __init__(
        self, emulator: Emulator | None = None, warn_on_extrapolation: bool = True
    ) -> None:
        self._emulator = load_brhm_emulator() if emulator is None else emulator
        self.warn_on_extrapolation = bool(warn_on_extrapolation)

    @classmethod
    def load(cls, path: str | Path | None = None) -> XiHMEmulator:
        """Load a bias-ratio model from ``path``, or the bundled one."""
        if path is None:
            return cls()
        emulator = Emulator.load(path)
        # The model's own box is the only one the wrapper knows about, so the
        # caller's arguments are checked against it rather than silenced.
        emulator.warn_on_extrapolation = False
        return cls(emulator)

    @property
    def x_spec(self) -> ParameterSpec:
        """Input specification of the wrapped model."""
        return self._emulator.x_spec

    @property
    def y_spec(self) -> DataVectorSpec:
        """Output specification of the wrapped model."""
        return self._emulator.y_spec

    @property
    def emulator(self) -> Emulator:
        """The underlying generic emulator, for callers that want the box."""
        return self._emulator

    def bias_ratio(self, theta: np.ndarray) -> np.ndarray:
        r"""Return :math:`B_{hm}` on the box.

        Parameters
        ----------
        theta : ndarray of shape (n, 8)
            Cosmological parameters in :func:`theta_spec` order.

        Returns
        -------
        ndarray of shape (n, N_BINS)
            The flat bias-ratio data vector.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.asarray(self._emulator.predict(theta, return_std=False))

    def ximm(self, theta: np.ndarray, z: float | np.ndarray, r: float | np.ndarray) -> np.ndarray:
        r"""Return the linear matter correlation function :math:`\xi_{mm}`.

        See :func:`ximm_linear`. Needs no bundled bias weights, only the linear
        power spectrum.
        """
        return self._run(ximm_linear, theta, z=z, r=r)

    def _run(self, function, *args, **kwargs):
        """Call a module function, honouring this object's warning preference.

        The interface has three interpolation axes and every layer of it warns
        about them, so asking a caller to silence each layer separately would be
        a poor deal. ``warn_on_extrapolation`` therefore covers the whole stack.
        """
        if self.warn_on_extrapolation:
            return function(*args, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return function(*args, **kwargs)

    def xihm_lgnbar_threshold(self, theta: np.ndarray, z, r, lgden) -> np.ndarray:
        r"""Return :math:`\xi_{hm}` for a cumulative number-density threshold.

        See :func:`xihm_lgnbar_threshold`. Works without the bias bundle too,
        since the baseline needs a bias but the direct term does not -- though
        the blend between them needs both.
        """
        return self._run(xihm_lgnbar_threshold, theta, z=z, r=r, lgden=lgden)

    def xihm_mass(self, theta: np.ndarray, z, r, M: np.ndarray) -> np.ndarray:
        r"""Return :math:`\xi_{hm}` for a fixed mass threshold.

        See :func:`xihm_mass`.
        """
        return self._run(xihm_mass, theta, z=z, r=r, M=M)

    def bias_lgnbar_threshold(self, theta: np.ndarray, z, lgden) -> np.ndarray:
        r"""Return the halo bias at a fixed number-density threshold.

        See :func:`jet.emulator.bhm.bias_lgnbar_threshold`.
        """
        from .bhm import bias_lgnbar_threshold as _impl

        return _impl(theta, z=z, lgden=lgden)

    def bias_mass(self, theta: np.ndarray, z, M: np.ndarray) -> np.ndarray:
        r"""Return the halo bias at a fixed mass threshold.

        See :func:`jet.emulator.bhm.bias_mass`.
        """
        from .bhm import bias_mass as _impl

        return _impl(theta, z=z, M=M)

    def brhm(
        self,
        theta: np.ndarray,
        z: float | np.ndarray,
        lgden: float | np.ndarray,
        r: float | np.ndarray,
    ) -> np.ndarray:
        r"""Return :math:`B_{hm}(r \mid z, \lg\bar{n})` at arbitrary coordinates.

        Parameters
        ----------
        theta : ndarray of shape (n, 8)
            Cosmological parameters.
        z, lgden, r : float or array-like
            Redshift, number-density threshold and separation in
            :math:`h^{-1}\mathrm{Mpc}`.

        Returns
        -------
        ndarray of shape (n, n_lgden, n_z, n_r)
            The reference's axis order.
        """
        if not self.warn_on_extrapolation:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return brhm_interpolate(self.bias_ratio(theta), z, lgden, r)
        return brhm_interpolate(self.bias_ratio(theta), z, lgden, r)

    def __repr__(self) -> str:
        return f"XiHMEmulator(fitted={self._emulator.is_fitted}, n_bins={N_BINS})"
