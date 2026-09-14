r"""
The bundled halo bias emulator, and the density-to-mass conversion it needs.

This is the small model underneath the halo-matter correlation function. It
predicts the **effective bias ratio**

.. math::
    b_{hm}(z, \lg\bar{n}) = \frac{b(\geq M)}{b_{\mathrm{Castro23}}(\geq M)},
    \qquad M = M(\lg\bar{n})

where the mass is the one whose cumulative number density matches the requested
threshold, so the emulated quantity is a correction to an analytic baseline
rather than a bias itself. The reference does it that way for the same reason it
emulates the mass function as a ratio: :math:`b(\geq M)` runs from one to tens
over the mass range, while the correction is a smooth order-unity function.

Two conversion steps stand between a caller and that ratio, and both are here
rather than in the caller:

- :func:`mass_from_lgden` inverts the cumulative mass function to turn a number
  density into a mass, using jet's own bundled
  :class:`~jet.emulator.hmf.HMFEmulator`;
- :func:`bias_lgnbar_threshold` and :func:`bias_mass` put the baseline back and
  convert between the two ways of selecting a halo sample. A threshold in
  density and a threshold in mass are not interchangeable -- holding
  :math:`\bar{n}` fixed and varying mass gives a different sample than the
  reverse -- so the mass-threshold version is obtained by a finite difference in
  :math:`\lg\bar{n}`, weighted by the density, exactly as the reference does it.

Layout
------
The data vector is the box flattened in C order, redshift outermost::

    value[i_z * N_LGDEN + i_den]  ==  b_hm(z_i, lgden_k)

with :func:`z_grid` and :func:`lgden_grid` giving both axes in ascending order.
Line 0 is :math:`z = 0`. The reference stores the redshifts the other way round;
the builder reverses that axis, which leaves the PCA scores untouched.

:data:`Z_GRID` and :data:`LGDEN_GRID` also describe the halo selection of the
correlation-function box in :mod:`jet.emulator.xihm`, which imports them from
here. They are the same two axes in the reference -- one pair of lists serves
every model built on that Sobol design -- so keeping one copy is what stops the
two emulators from disagreeing about what a redshift is.

.. warning::
    The weights under ``jet/data`` are not tracked by git; see the note in
    ``jet/data/README.md``. They are rebuilt from the reference's published
    arrays by ``tools/build_xihm_bundle.py``.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

from ..cosmology import cosmo_parameter_spec
from ..spec import DataVectorSpec, ParameterSpec
from .emulator import Emulator, data_dir, resolve_weights_path
from .hmf import HMFEmulator, castro23_bias

__all__ = [
    "BUNDLE_NAME",
    "ZGRID_KEY",
    "LGDEN_KEY",
    "N_Z",
    "N_LGDEN",
    "N_BINS",
    "Z_GRID",
    "LGDEN_GRID",
    "MASS_EDGES",
    "MASS_PERTURBATION",
    "LGDEN_INVERSION_GRID",
    "data_dir",
    "z_grid",
    "lgden_grid",
    "theta_spec",
    "data_vector_spec",
    "load_bhm_emulator",
    "bias_grid",
    "bias_ratio_at",
    "mass_from_lgden",
    "bias_lgnbar_threshold",
    "bias_mass",
    "BhmEmulator",
]

#: Name of the ``.npz`` bundle under the data directory.
BUNDLE_NAME = "bhm_rockstar_m200m.gp.npz"

#: Bundle key holding the redshift grid.
ZGRID_KEY = "z"

#: Bundle key holding the number-density threshold grid.
LGDEN_KEY = "lgden"

#: Redshifts of the box, ascending. Row 0 is ``z = 0``.
Z_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

#: Cumulative number-density thresholds in
#: :math:`\log_{10}(h^3\,\mathrm{Mpc}^{-3})`, ascending.
LGDEN_GRID: tuple[float, ...] = (-5.0, -4.5, -4.0, -3.5, -3.0, -2.5)

N_Z = len(Z_GRID)
N_LGDEN = len(LGDEN_GRID)

#: Entries in the emulator's data vector.
N_BINS = N_Z * N_LGDEN

#: Masses the cumulative mass function is tabulated on before being inverted.
MASS_EDGES = np.logspace(11.0, 16.0, 51)

#: Number-density grid the inversion is tabulated on, and the range of it.
#:
#: The reference samples ``log10(n)`` uniformly between these two limits and
#: then interpolates the resulting ``log10(M)``; the limits are wider than any
#: threshold the emulator's box covers, so that the corner of the requested
#: region does not fall off the end of the spline.
LGDEN_INVERSION_MIN = -8.0
LGDEN_INVERSION_MAX = -1.0
LGDEN_INVERSION_POINTS = 100
LGDEN_INVERSION_GRID = np.linspace(LGDEN_INVERSION_MIN, LGDEN_INVERSION_MAX, LGDEN_INVERSION_POINTS)

#: Bounding box of the inversion spline in ``(z, log10 n)``.
_INVERSION_BBOX = [Z_GRID[0], Z_GRID[-1], -20.0, -1.0]

#: Relative offset by which a mass threshold is perturbed to convert it into a
#: density threshold. The reference uses one per cent; it is a finite-difference
#: step, not a physical quantity.
MASS_PERTURBATION = 0.01


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

    An absent bundle is not an error: both axes are module constants, and the
    bundle only restates them so a weights file is self-describing. Callers
    that want the *values* go through :func:`load_bhm_emulator`.
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


def theta_spec() -> ParameterSpec:
    """Return the input specification of the bundled emulator."""
    return cosmo_parameter_spec()


def data_vector_spec() -> DataVectorSpec:
    """Return the output specification of the bundled emulator.

    ``log`` is false: the emulated quantity is a positive bias correction of
    order one, and the reference applies no logarithm to it.
    """
    return DataVectorSpec(name="bhm_rockstar_m200m", n_bins=N_BINS, log=False)


@lru_cache(maxsize=1)
def load_bhm_emulator() -> Emulator:
    """Load the bundled bias-ratio emulator.

    Returns
    -------
    Emulator
        Predicts the bias correction on the flattened layout described in the
        module docstring.

    Raises
    ------
    FileNotFoundError
        If the data file is absent.
    """
    emulator = Emulator.load(_data_path(BUNDLE_NAME))
    # Suppressed for the same reason as in the other bundled readers: the
    # mass-threshold conversions below probe deliberately displaced densities
    # near the edges of the box, and warning about each would drown the
    # caller's own warning about the cosmology.
    emulator.warn_on_extrapolation = False
    return emulator


@lru_cache(maxsize=1)
def _default_hmf() -> HMFEmulator:
    """Return the bundled mass function, for the density-to-mass inversion."""
    return HMFEmulator.load()


def warn_outside(name: str, values: np.ndarray, low: float, high: float) -> None:
    """Warn once if any entry of ``values`` leaves ``[low, high]``.

    Shared with :mod:`jet.emulator.xihm`, which has the same three interpolation
    axes one of which -- the separation -- is *expected* to be asked for outside
    its range, because the correlation function is blended onto a different
    baseline out there.
    """
    if values.size == 0:
        return
    if np.any(values < low) or np.any(values > high):
        worst = float(np.min(values)) if np.any(values < low) else float(np.max(values))
        warnings.warn(
            f"{name}={worst:g} is outside the emulated range [{low:g}, {high:g}]; the "
            "grid is being extrapolated linearly and the result is not trustworthy there.",
            RuntimeWarning,
            stacklevel=3,
        )


def _evaluate_spline(spline, z: np.ndarray, lgden: np.ndarray) -> np.ndarray:
    r"""Evaluate a bivariate spline over the outer product ``z x lgden``.

    ``RectBivariateSpline`` evaluates on a grid and therefore insists that both
    query arrays increase. Callers do not always present them that way -- the
    mass-threshold conversion reads densities off a mass grid, which runs the
    other direction -- so the axes are sorted for the evaluation and permuted
    back afterwards. The value at a point does not depend on the order the
    points arrive in, so this changes nothing numerically.
    """
    z_order = np.argsort(z)
    lgden_order = np.argsort(lgden)
    evaluated = spline(z[z_order], lgden[lgden_order])
    out = np.empty_like(evaluated)
    out[np.ix_(z_order, lgden_order)] = evaluated
    return out


def bias_grid(ratio: np.ndarray) -> np.ndarray:
    """Reshape data vectors into the ``(n_samples, N_Z, N_LGDEN)`` box."""
    ratio = np.atleast_2d(np.asarray(ratio, dtype=float))
    if ratio.shape[1] != N_BINS:
        raise ValueError(
            f"expected data vectors of length {N_BINS} ({N_Z} redshifts x {N_LGDEN} "
            f"thresholds), got {ratio.shape[1]}"
        )
    return ratio.reshape(ratio.shape[0], N_Z, N_LGDEN)


def bias_ratio_at(
    ratio: np.ndarray, z: float | np.ndarray, lgden: float | np.ndarray, warn: bool = True
) -> np.ndarray:
    r"""Evaluate predicted bias corrections at arbitrary ``(z, lgden)``.

    Linear in both axes, matching the reference. There is no ``r`` axis here --
    this model has no scale dependence; the correlation function's shape comes
    entirely from :mod:`jet.emulator.xihm`.

    Parameters
    ----------
    ratio : ndarray of shape (n_samples, N_BINS)
        Predicted bias corrections on the box.
    z, lgden : float or array-like
        Redshift and number-density threshold.
    warn : bool, optional
        Whether to warn about points outside the box. Turned off by callers
        that derived the thresholds from something else -- the mass-threshold
        conversion does -- and would otherwise warn about a quantity the caller
        never named.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_lgden)
        The reference's axis order.
    """
    grid = bias_grid(ratio)
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    lgden_in = np.atleast_1d(np.asarray(lgden, dtype=float))

    z_axis, lgden_axis = z_grid(), lgden_grid()
    if warn:
        warn_outside("z", z_in, float(z_axis[0]), float(z_axis[-1]))
        warn_outside("lgden", lgden_in, float(lgden_axis[0]), float(lgden_axis[-1]))

    out = np.empty((grid.shape[0], z_in.size, lgden_in.size), dtype=float)
    for row in range(grid.shape[0]):
        spline = RectBivariateSpline(z_axis, lgden_axis, grid[row], kx=1, ky=1)
        out[row] = _evaluate_spline(spline, z_in, lgden_in)
    return out


def mass_from_lgden(
    theta: np.ndarray,
    z: float | np.ndarray,
    lgden: float | np.ndarray,
    hmf_emulator: HMFEmulator | None = None,
) -> np.ndarray:
    r"""Return the halo mass whose abundance matches a density threshold.

    Inverts :math:`n(\geq M) = \bar{n}` on a fixed mass grid, once per redshift,
    and then interpolates the resulting :math:`\log_{10}M` bilinearly in
    :math:`(z, \lg\bar{n})`. The mass grid is :data:`MASS_EDGES`, and the
    inversion is a cubic spline of :math:`\log_{10}M` against
    :math:`\log_{10}\bar{n}` -- the reference's prescription, reproduced
    because the values are compared point by point.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters.
    z, lgden : float or array-like
        Redshift and number-density threshold in
        :math:`\log_{10}(h^3\,\mathrm{Mpc}^{-3})`.
    hmf_emulator : HMFEmulator, optional
        Mass function to invert. Defaults to the bundled one.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_lgden)
        Halo mass in :math:`M_\odot/h`.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    lgden_in = np.atleast_1d(np.asarray(lgden, dtype=float))
    hmf = _default_hmf() if hmf_emulator is None else hmf_emulator

    # The inversion is tabulated on the emulator's own twelve redshifts, because
    # the mass function is only known there and re-splining it per requested
    # redshift would put a second interpolation inside a spline that is itself
    # the thing being inverted.
    z_axis = z_grid()
    cumulative = np.asarray(hmf.number_density(theta, z=z_axis, M=MASS_EDGES), dtype=float)

    out = np.empty((theta.shape[0], z_in.size, lgden_in.size), dtype=float)
    for row in range(theta.shape[0]):
        log_m = np.zeros((z_axis.size, LGDEN_INVERSION_GRID.size), dtype=float)
        for index in range(z_axis.size):
            column = cumulative[row, index]
            usable = column > 0.0
            if usable.sum() < 2:
                continue
            # ``n(>=M)`` falls with mass, so both arrays are written in
            # descending order here; `interp1d` sorts them internally, and the
            # result is the one the reference gets from the same call.
            interpolator = interp1d(
                np.log10(column[usable]),
                np.log10(MASS_EDGES[usable]),
                kind="cubic",
                fill_value="extrapolate",
            )
            log_m[index] = interpolator(LGDEN_INVERSION_GRID)

        spline = RectBivariateSpline(
            z_axis, LGDEN_INVERSION_GRID, log_m, kx=1, ky=1, s=0, bbox=_INVERSION_BBOX
        )
        out[row] = np.power(10.0, _evaluate_spline(spline, z_in, lgden_in))

    return out


def _baseline_bias(theta_row: np.ndarray, z_value, masses: np.ndarray) -> np.ndarray:
    r"""Return the Castro23 bias at each of ``masses``, which must ascend.

    The bias is a numerical derivative of the multiplicity function in mass, so
    the answer depends on which masses are supplied -- there is no derivative at
    a single point. With two or more the difference is taken on the caller's own
    grid, which is what makes the values comparable with the reference's. With
    one, a local stencil is built around it: a better answer to "what is the
    bias of halos above this threshold", but a different finite difference, so
    the two branches do not agree exactly and are deliberately separate.
    """
    if masses.size >= 2:
        return castro23_bias(theta_row, z=z_value, M=masses)[0, 0]
    centre = float(masses[0])
    stencil = centre * np.array([1.0 - MASS_PERTURBATION, 1.0, 1.0 + MASS_PERTURBATION])
    return castro23_bias(theta_row, z=z_value, M=stencil)[0, 0, 1:2]


def bias_lgnbar_threshold(
    theta: np.ndarray,
    z: float | np.ndarray,
    lgden: float | np.ndarray,
    hmf_emulator: HMFEmulator | None = None,
) -> np.ndarray:
    r"""Return the halo bias at a fixed cumulative number-density threshold.

    Computes :math:`b(\geq M)\,b_{hm}` with :math:`M = M(\lg\bar{n})`: the
    analytic Castro23 bias evaluated at the mass the threshold corresponds to,
    corrected by the emulated ratio.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters.
    z, lgden : float or array-like
        Redshift and density threshold.
    hmf_emulator : HMFEmulator, optional
        Mass function used by the inversion. Defaults to the bundled one.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_lgden)
        The bias, dimensionless.

    Notes
    -----
    The baseline is a numerical derivative of the multiplicity function in mass,
    so its value at a given threshold depends on how far apart the thresholds
    are. With two or more the derivative is taken on the caller's own grid, which
    is what makes the values comparable with the reference's point by point.
    With one, a local one-per-cent stencil is built around the mass instead --
    a better answer to "what is the bias of halos above this threshold", but a
    different finite difference, so the two paths do not agree exactly.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    lgden_in = np.atleast_1d(np.asarray(lgden, dtype=float))

    mass = mass_from_lgden(theta, z=z_in, lgden=lgden_in, hmf_emulator=hmf_emulator)
    ratio = np.asarray(load_bhm_emulator().predict(theta, return_std=False))

    out = np.empty((theta.shape[0], z_in.size, lgden_in.size), dtype=float)
    for row in range(theta.shape[0]):
        baseline = np.empty((z_in.size, lgden_in.size), dtype=float)
        for index in range(z_in.size):
            # A larger threshold is a lower mass, so the mass column runs
            # *down* the threshold axis. `_baseline_bias` differentiates a mass
            # grid and therefore needs it ascending, so the column is reversed
            # in and reversed back out. The finite difference is symmetric in
            # the other direction, but the reversal is kept so that the values
            # match the reference's to the last bit.
            column = mass[row, index][::-1]
            baseline[index] = _baseline_bias(theta[row : row + 1], z_in[index : index + 1], column)[
                ::-1
            ]

        out[row] = bias_ratio_at(ratio[row : row + 1], z=z_in, lgden=lgden_in)[0] * baseline

    return out


def bias_mass(
    theta: np.ndarray,
    z: float | np.ndarray,
    M: np.ndarray,
    hmf_emulator: HMFEmulator | None = None,
) -> np.ndarray:
    r"""Return the halo bias at a fixed mass threshold.

    A mass threshold is not a density threshold, and the bias is not a function
    of the mass alone, so this cannot be read off the density-threshold result.
    The reference converts by perturbing the mass by one per cent in each
    direction, reading off the two number densities and the two biases, and
    taking the density-weighted difference -- which is the finite-difference
    form of "hold :math:`\bar{n}` fixed instead of :math:`M`". Reproduced here
    with the same one per cent.

    Mass bins whose number density underflows to zero are returned as zero and
    warned about, because a weighted average of two zeros is not defined.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters.
    z : float or array-like
        Redshifts.
    M : array-like
        Halo masses in :math:`M_\odot/h`.
    hmf_emulator : HMFEmulator, optional
        Mass function used for the perturbation. Defaults to the bundled one.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_M)
        The bias, dimensionless. Zero wherever the mass bin had no halos.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z_in = np.atleast_1d(np.asarray(z, dtype=float))
    M = np.atleast_1d(np.asarray(M, dtype=float))
    hmf = _default_hmf() if hmf_emulator is None else hmf_emulator
    ratio = np.asarray(load_bhm_emulator().predict(theta, return_std=False))

    masses_up = M * (1.0 + MASS_PERTURBATION)
    masses_down = M * (1.0 - MASS_PERTURBATION)

    out = np.empty((theta.shape[0], z_in.size, M.size), dtype=float)
    for row in range(theta.shape[0]):
        # Everything below is done with the mass axis reversed, so that the
        # number density read off it increases with index. The reference does
        # the same `[:, ::-1]`, and it matters twice over: the bias lookups need
        # an increasing threshold axis, and an increasing density is the
        # physically meaningful order for the finite difference. The result is
        # reversed back at the end.
        density_up = np.asarray(hmf.number_density(theta[row : row + 1], z=z_in, M=masses_up))[0][
            :, ::-1
        ]
        density_down = np.asarray(hmf.number_density(theta[row : row + 1], z=z_in, M=masses_down))[
            0
        ][:, ::-1]

        # The two halves of each bias are taken from different places on
        # purpose, and it is worth being explicit because the obvious
        # simplification is wrong. The emulated ratio is looked up at the
        # *density* the perturbed mass corresponds to, while the Castro23
        # baseline is evaluated at the perturbed mass itself -- not at the mass
        # a density-to-mass inversion would return for that threshold. The two
        # differ by the inversion's own interpolation error, which is small but
        # not zero, and the finite difference below divides by two per cent, so
        # that error is amplified a hundredfold. The reference does it this way
        # and the values are compared point by point.
        baseline_up = np.stack(
            [
                _baseline_bias(theta[row : row + 1], z_in[index : index + 1], masses_up)
                for index in range(z_in.size)
            ]
        )[:, ::-1]
        baseline_down = np.stack(
            [
                _baseline_bias(theta[row : row + 1], z_in[index : index + 1], masses_down)
                for index in range(z_in.size)
            ]
        )[:, ::-1]

        reversed_row = np.zeros((z_in.size, M.size), dtype=float)
        for index in range(z_in.size):
            usable = (density_up[index] > 0.0) & (density_down[index] > 0.0)
            if not np.all(usable):
                unreachable = M[::-1][~usable]
                warnings.warn(
                    f"M={unreachable[0]:.2e} has zero number density at "
                    f"z={z_in[index]:.2f}; the bias is returned as zero for that bin.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if not np.any(usable):
                continue
            density_upper = density_up[index][usable]
            density_lower = density_down[index][usable]
            # `warn=False` because these thresholds were derived from the masses
            # the caller passed rather than named by them. The range is reported
            # once below, in terms of masses, where it is actionable.
            upper = (
                bias_ratio_at(
                    ratio[row : row + 1],
                    z=z_in[index],
                    lgden=np.log10(density_upper),
                    warn=False,
                )[0, 0]
                * baseline_up[index][usable]
            )
            lower = (
                bias_ratio_at(
                    ratio[row : row + 1],
                    z=z_in[index],
                    lgden=np.log10(density_lower),
                    warn=False,
                )[0, 0]
                * baseline_down[index][usable]
            )
            reversed_row[index, usable] = (upper * density_upper - lower * density_lower) / (
                density_upper - density_lower
            )

        out[row] = reversed_row[:, ::-1]

        # One warning per cosmology, in terms of what the caller asked for. The
        # masses whose implied thresholds leave the bias box are those far from
        # the middle of the emulated range -- too low to be rare, or too high to
        # have been trained on -- and their bias is extrapolated.
        thresholds = np.concatenate(
            [
                np.log10(density_up[density_up > 0.0].ravel()),
                np.log10(density_down[density_down > 0.0].ravel()),
            ]
        )
        lgden_axis = lgden_grid()
        warn_outside("lgden implied by M", thresholds, float(lgden_axis[0]), float(lgden_axis[-1]))

    return out


class BhmEmulator:
    """The bundled halo bias emulator.

    Wraps the generic :class:`~jet.emulator.Emulator` over the bias correction
    together with the conversions that turn it into a physical bias.

    Parameters
    ----------
    emulator : Emulator, optional
        The fitted bias-correction model. Defaults to the bundled one.
    hmf_emulator : HMFEmulator, optional
        Mass function used for the density-to-mass inversion. Defaults to the
        bundled one.
    """

    def __init__(
        self, emulator: Emulator | None = None, hmf_emulator: HMFEmulator | None = None
    ) -> None:
        self._emulator = load_bhm_emulator() if emulator is None else emulator
        self._hmf = hmf_emulator

    @classmethod
    def load(cls, path: str | Path | None = None) -> BhmEmulator:
        """Load a bias-correction model from ``path``, or the bundled one."""
        if path is None:
            return cls()
        emulator = Emulator.load(path)
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

    def ratio(self, theta: np.ndarray) -> np.ndarray:
        """Return the bias correction on the box, shape ``(n, N_BINS)``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.asarray(self._emulator.predict(theta, return_std=False))

    def ratio_at(self, theta: np.ndarray, z, lgden) -> np.ndarray:
        """Return the bias correction at arbitrary ``(z, lgden)``."""
        return bias_ratio_at(self.ratio(theta), z=z, lgden=lgden)

    def mass_from_lgden(self, theta: np.ndarray, z, lgden) -> np.ndarray:
        """Return the mass matching a cumulative number-density threshold."""
        return mass_from_lgden(theta, z=z, lgden=lgden, hmf_emulator=self._hmf)

    def bias_lgnbar_threshold(self, theta: np.ndarray, z, lgden) -> np.ndarray:
        """Return the bias at a fixed cumulative number-density threshold."""
        return bias_lgnbar_threshold(theta, z=z, lgden=lgden, hmf_emulator=self._hmf)

    def bias_mass(self, theta: np.ndarray, z, M) -> np.ndarray:
        """Return the bias at a fixed mass threshold."""
        return bias_mass(theta, z=z, M=M, hmf_emulator=self._hmf)

    def __repr__(self) -> str:
        return f"BhmEmulator(fitted={self._emulator.is_fitted}, n_bins={N_BINS})"
