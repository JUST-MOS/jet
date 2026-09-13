"""
The bundled halo mass function emulator, for the ``RockstarM200m`` mass definition.

The reference implementation does not emulate the mass function directly. It
emulates the *ratio* of the mass function to a Castro23 (Tinker08-shaped)
baseline, because that ratio is a smooth, order-unity function of cosmology
while the mass function itself spans ten orders of magnitude. Recovering the
physical quantity is then a multiplication by an analytic baseline:

.. math::

    n(\\geq M, z) = \\underbrace{r(\\theta, z, M)}_{\\text{Gaussian process}}
                    \\;\\times\\;
                    \\underbrace{n_{\\mathrm{Castro23}}(\\theta, z, M)}_{\\text{closed form}}

Both halves are reproduced here: the Gaussian process from the reference's own
published weights, read out of a jet bundle, and the baseline as code (see
:mod:`jet.cosmology` for the expansion history it needs, and
:func:`castro23_cumulative` for the multiplicity function).

Data vector layout
------------------
The ratio is not a rectangular grid. The reference stores, for each of twelve
redshifts, the ratio over the mass range it was trained on -- masses are logged
between :math:`10^{10}` and :math:`10^{16} M_\\odot/h` in sixty bins, and the
upper end of the trained range falls with redshift, from bin 53 at ``z = 0`` to
bin 36 at ``z = 3``. The data vector is those twelve variable-length blocks
concatenated, 425 numbers in all::

    value[start_z + i]  ==  r(z, M_i)      for  start_z <= i < stop_z

with ``z`` ascending, so block 0 is ``z = 0``. :func:`data_slices` gives the
twelve ``(start, stop)`` column pairs, :func:`mass_slices` the mass bin ranges
they cover, and :func:`mass_edges` the bin edges those index. This is the
reference's layout with the redshift blocks reversed into ascending order;
reversing blocks leaves the PCA scores untouched, so the stored basis is the
reference's own.

.. warning::
    The weights under ``jet/data`` are large and are *not* tracked by git; see
    the note in ``jet/data/README.md``. Unlike the power-spectrum bundle, this
    one is written by :mod:`tools.build_hmf_bundle`, which needs the reference
    implementation to generate its inputs.

Notes
-----
Two numerical details of the reference are reproduced deliberately, because
departing from them would break point-by-point agreement:

``addnum``
    The cumulative function is interpolated in :math:`\\log_{10}(n + a)` rather
    than :math:`\\log_{10} n`, with ``a`` chosen per prediction as
    ``1 - min(n)`` so that the argument stays positive where the mass function
    underflows to zero. The same ``a`` is subtracted again afterwards. It is
    algebraically nearly a no-op and numerically it keeps the spline away from
    ``log10(0)``.

The double interpolation
    Core masses are first resampled per redshift with a linear spline in
    :math:`\\log_{10} M` onto a 140-point grid, multiplied by the baseline
    there, and only then turned into a bivariate spline over
    :math:`(z, \\log_{10} M)` with a linear redshift direction and a cubic mass
    direction. The two grids are offset by half a bin, which is a real
    inconsistency in the reference; it is kept, because correcting it would
    move every number this module is verified against.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline, interp1d
from scipy.special import gamma

from ..cosmology import Cosmology
from ..spec import DataVectorSpec, ParameterSpec
from .emulator import Emulator, data_dir, resolve_weights_path
from .pklin import log_pk_at, sigma_of_log_pk

__all__ = [
    "BUNDLE_NAME",
    "MASS_DEFINITION",
    "N_Z",
    "N_BINS",
    "N_FINE",
    "FINE_CENTERS",
    "ZGRID_KEY",
    "MASS_EDGES_KEY",
    "MASS_SLICES_KEY",
    "CASTRO23_KEY",
    "data_dir",
    "z_grid",
    "mass_edges",
    "mass_slices",
    "data_slices",
    "castro23_coefficients",
    "theta_spec",
    "data_vector_spec",
    "load_hmf_emulator",
    "castro23_dndlnM",
    "castro23_cumulative",
    "castro23_multiplicity",
    "castro23_bias",
    "HMFEmulator",
]

#: Name of the ``.npz`` bundle under the data directory.
BUNDLE_NAME = "hmf_rockstar_m200m.gp.npz"

#: Halo mass definition the bundled model was trained on. Masses are
#: :math:`M_{200\\mathrm{m}}` as found by Rockstar, in :math:`M_\\odot/h`.
MASS_DEFINITION = "RockstarM200m"

#: Number of redshifts in the emulator's grid.
N_Z = 12

#: Entries in the emulator's data vector: the twelve mass blocks concatenated.
N_BINS = 425

#: Number of bins in the reference's mass grid, i.e. ``mass_edges().size - 1``.
N_MASS_BINS = 60

#: Mass edges of the ratio vector, in :math:`M_\\odot/h`. Fixed by the
#: reference's training set; the bundle stores a copy so that a reader never has
#: to assume it.
MASS_EDGES = np.logspace(10.0, 16.0, N_MASS_BINS + 1)

#: Edges of the fine grid the ratio is resampled onto before the bivariate
#: spline is built, and the number of intervals between them.
_FINE_EDGES = np.logspace(10.0, 17.0, 141)
N_FINE = _FINE_EDGES.size - 1

#: Geometric centres of :data:`_FINE_EDGES`, i.e. the masses the baseline is
#: evaluated at. Note the reference pairs these with the *lower* edges when it
#: multiplies the two together; see the module docstring.
FINE_CENTERS = 10.0 ** ((np.log10(_FINE_EDGES[1:]) + np.log10(_FINE_EDGES[:-1])) / 2.0)

#: Log10 spacing of :data:`_FINE_EDGES`.
_FINE_DLOG10M = np.log10(_FINE_EDGES[1]) - np.log10(_FINE_EDGES[0])

#: Step of the logarithmic derivative ``d ln sigma / d ln R`` the Castro23
#: multiplicity uses, and the step of the numerical mass derivative.
_DLNR = 0.01
_DLOG10M_STEP = 1e-3

#: Castro23 coefficients for :data:`MASS_DEFINITION`, in the reference's order:
#: ``a1, a2, az, p1, p2, q1, q2, qz``. These are the only numbers that change
#: between mass definitions -- the rest of the baseline is shared physics.
CASTRO23_COEFFICIENTS = np.array(
    [
        0.77283085,
        0.3686431,
        -0.03214047,
        -0.46080809,
        -0.53419608,
        0.37336379,
        -0.32322067,
        -0.19204407,
    ]
)

#: Bundle keys for the arrays that describe the grid and the baseline.
ZGRID_KEY = "z"
MASS_EDGES_KEY = "mass_edges"
MASS_SLICES_KEY = "mass_slices"
CASTRO23_KEY = "castro23"


def _data_path(name: str = BUNDLE_NAME) -> Path:
    """Resolve one bundled data file, auto-fetching from GitHub Release if needed."""
    try:
        return resolve_weights_path(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"the bundled data file {name!r} is missing from {data_dir()}. It is not "
            "tracked by git; regenerate it with tools/build_hmf_bundle.py, or set "
            "JET_DATA_DIR to a directory that already holds it."
        ) from exc


@lru_cache(maxsize=1)
def _extra_arrays() -> dict[str, np.ndarray]:
    """Return the bundle's auxiliary arrays, read once per process."""
    from .bundle import load_bundle

    return dict(load_bundle(_data_path())["extra_arrays"])


def z_grid() -> np.ndarray:
    """Return the redshifts of the data vector, ascending, in ``z = 0`` first.

    Returns
    -------
    ndarray of shape (N_Z,)
    """
    return _extra_arrays()[ZGRID_KEY]


def mass_edges() -> np.ndarray:
    """Return the mass edges the ratio vector is defined on, in :math:`M_\\odot/h`.

    Returns
    -------
    ndarray of shape (N_MASS_BINS + 1,)
        Ascending. Each redshift's block of the data vector spans a contiguous
        sub-range of these, given by :func:`mass_slices`.
    """
    return _extra_arrays()[MASS_EDGES_KEY]


def mass_slices() -> np.ndarray:
    """Return each redshift's mass range as ``(start, stop)`` *bin* indices.

    These index :func:`mass_edges`, not the data vector. The two ranges start
    together -- both begin at the first stored mass -- but diverge immediately,
    because a redshift's block holds only the bins between its own limits.

    Returns
    -------
    ndarray of shape (N_Z, 2)
        Row ``j`` gives the half-open bin range of :func:`z_grid` entry ``j``.
    """
    return _extra_arrays()[MASS_SLICES_KEY].astype(int)


def data_slices() -> np.ndarray:
    """Return each redshift's block as ``(start, stop)`` *data-vector* indices.

    Derived from :func:`mass_slices` by accumulating the block lengths, so the
    two can never disagree. Row ``j`` gives the half-open column range of
    :func:`z_grid` entry ``j``.

    Returns
    -------
    ndarray of shape (N_Z, 2)
    """
    lengths = np.diff(mass_slices(), axis=1)[:, 0]
    stops = np.cumsum(lengths)
    return np.column_stack([stops - lengths, stops]).astype(int)


def castro23_coefficients() -> np.ndarray:
    """Return the Castro23 coefficients stored in the bundle, shape ``(8,)``."""
    return _extra_arrays()[CASTRO23_KEY]


def theta_spec() -> ParameterSpec:
    """Return the input specification of the bundled emulator.

    The same eight cosmological parameters as the power-spectrum bundle,
    declared once in :func:`jet.cosmology.cosmo_parameter_spec`.

    Returns
    -------
    ParameterSpec
        Eight parameters, in the emulator's column order.
    """
    from ..cosmology import cosmo_parameter_spec

    return cosmo_parameter_spec()


def data_vector_spec() -> DataVectorSpec:
    """Return the output specification of the bundled emulator.

    ``log`` is false on purpose: the emulated quantity is the ratio to the
    baseline, not a mass function, and it is neither positive-definite in
    general nor sensibly logged.
    """
    return DataVectorSpec(name="hmf_ratio_rockstar_m200m", n_bins=N_BINS, log=False)


@lru_cache(maxsize=1)
def load_hmf_emulator() -> Emulator:
    """Load the bundled Gaussian-process emulator of the ratio.

    The result is cached, because a single halo-mass-function evaluation goes
    through it and the baseline's power-spectrum evaluation immediately after.

    Returns
    -------
    Emulator
        Predicts the 425-entry ratio data vector from the eight cosmological
        parameters. See the module docstring for the index layout.

    Raises
    ------
    FileNotFoundError
        If the bundle is absent.
    """
    emulator = Emulator.load(_data_path())
    # The baseline probes arbitrary cosmologies and radii, and the caller's own
    # parameters are checked by the outer object; one warning per internal call
    # would drown that out.
    emulator.warn_on_extrapolation = False
    return emulator


# ----------------------------------------------------------------------
# The Castro23 baseline
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def _column_indices() -> dict[str, int]:
    """Return the column index of each input parameter, by name.

    The baseline picks the densities, the equation of state and the neutrino
    mass out of a parameter row by name rather than by position, so that
    reordering the spec cannot silently feed it the wrong numbers.
    """
    return {name: index for index, name in enumerate(theta_spec().names)}


def _lagrangian_radius(M: np.ndarray, cosmo: Cosmology) -> np.ndarray:
    r"""Return the Lagrangian radius of a halo of mass ``M``, in :math:`M_\odot/h`.

    Uses the baryon-plus-cold-dark-matter density, matching the reference's
    ``Pcb=True`` convention.
    """
    rho_m = cosmo.rho_crit * cosmo.Omegam
    return (3.0 * M / (4.0 * np.pi * rho_m)) ** (1.0 / 3.0)


def _delta_c(z: float, cosmo: Cosmology) -> float:
    r"""Return the linear collapse threshold :math:`\delta_c(z)`.

    The reference's redshift-dependent fitting form, which carries the weak
    :math:`\Omega_m(z)` dependence of the spherical-collapse value.
    """
    omega = float(cosmo.omega_cb(z)[0])
    return float(3.0 / 20.0 * (12.0 * np.pi) ** (2.0 / 3.0) * (1.0 + 0.0123 * np.log10(omega)))


def castro23_dndlnM(
    theta: np.ndarray,
    z: np.ndarray | None = None,
    M: np.ndarray | None = None,
    coefficients: np.ndarray | None = None,
    pk_emulator: Emulator | None = None,
) -> np.ndarray:
    r"""Evaluate the Castro23 multiplicity function, :math:`dn/d\ln M`.

    This is the analytic baseline the emulator's ratio is defined against. It
    is :math:`n(M) = \nu f(\nu)\, \rho_m / M \, |d\ln\sigma/d\ln M|` with the
    Castro23 form of :math:`\nu f(\nu)`, evaluated with the cold-dark-matter-
    plus-baryon power spectrum and the density parameter that excludes massive
    neutrinos.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters, columns in :func:`theta_spec` order.
    z : ndarray, optional
        Redshifts. Defaults to :func:`z_grid`.
    M : ndarray, optional
        Halo masses in :math:`M_\odot/h`. Defaults to :data:`FINE_CENTERS`.
    coefficients : ndarray, optional
        The eight Castro23 coefficients. Defaults to the bundled set.
    pk_emulator : Emulator, optional
        Power-spectrum emulator used for :math:`\sigma_{cb}`. Defaults to the
        bundled one.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_M)
        :math:`dn/d\ln M` in :math:`(h/\mathrm{Mpc})^3`.
    """
    nu, multiplicity, prefactor = _castro23_terms(theta, z, M, coefficients, pk_emulator)
    del nu
    return multiplicity * prefactor


def _castro23_terms(
    theta: np.ndarray,
    z: np.ndarray | None,
    M: np.ndarray | None,
    coefficients: np.ndarray | None,
    pk_emulator: Emulator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the three pieces every Castro23 quantity is built from.

    The mass function is :math:`\nu f(\nu) \times \rho_m/M\,|d\ln\sigma/d\ln R|`,
    and the linear bias is a derivative of the first factor alone with respect
    to :math:`\ln\nu`. Returning them separately keeps one implementation of the
    eight coefficients in the package: the alternative -- a second module
    recomputing :math:`\nu f(\nu)` for the bias -- is a way for the two to
    disagree, and the coefficients are the only thing that changes per mass
    definition.

    Returns
    -------
    nu, multiplicity, prefactor : ndarray of shape (n_samples, n_z, n_M)
        :math:`\nu = \delta_c/\sigma`, :math:`\nu f(\nu)`, and
        :math:`\rho_m/M\,(-d\ln\sigma/d\ln R/3)`.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    coefficients = (
        castro23_coefficients() if coefficients is None else np.asarray(coefficients, dtype=float)
    )
    z = z_grid() if z is None else np.atleast_1d(np.asarray(z, dtype=float))
    M = FINE_CENTERS if M is None else np.atleast_1d(np.asarray(M, dtype=float))
    if theta.shape[1] != 8:
        raise ValueError(f"theta must have 8 columns, got {theta.shape[1]}")

    columns = _column_indices()
    a1, a2, az, p1, p2, q1, q2, qz = coefficients
    shape = (theta.shape[0], z.size, M.size)
    nu_out = np.empty(shape, dtype=float)
    multiplicity_out = np.empty(shape, dtype=float)
    prefactor_out = np.empty(shape, dtype=float)

    for row, parameters in enumerate(theta):
        cosmo = Cosmology(
            Omegab=parameters[columns["Omegab"]],
            Omegam=parameters[columns["Omegam"]],
            H0=parameters[columns["H0"]],
            w0=parameters[columns["w0"]],
            wa=parameters[columns["wa"]],
            mnu=parameters[columns["mnu"]],
        )
        # One power-spectrum call gives every redshift the emulator knows about.
        # Note this is the *cold dark matter plus baryon* spectrum, which is
        # what the baseline is defined against -- not a total-matter one.
        power = np.log10(_power_spectrum(parameters, pk_emulator))
        radius = _lagrangian_radius(M, cosmo)
        rho_m = cosmo.rho_crit * cosmo.Omegam

        for index, redshift in enumerate(z):
            # The requested redshift need not be one of the twelve the emulator
            # stores, so the spectrum is re-splined per redshift rather than
            # indexed. Getting this wrong is silent: the twelve stored
            # redshifts are a superset of any reasonable request, so a
            # positional index produces a plausible number for the wrong
            # cosmology.
            log10_pk = log_pk_at(power, z=float(redshift))
            # d ln sigma / d ln R by a central difference in radius, exactly as
            # the reference does it -- the radius is scaled, not the mass.
            sigma_lo = sigma_of_log_pk(log10_pk, radius * np.exp(-_DLNR / 2.0))
            sigma_hi = sigma_of_log_pk(log10_pk, radius * np.exp(+_DLNR / 2.0))
            dln_sigma_dln_r = np.log(sigma_hi / sigma_lo)[0] / _DLNR
            sigma = sigma_of_log_pk(log10_pk, radius)[0]

            omega_cb = float(cosmo.omega_cb(redshift)[0])
            nu = _delta_c(redshift, cosmo) / sigma

            a_r = a1 + a2 * (dln_sigma_dln_r + 0.6125) ** 2
            q_r = q1 + q2 * (dln_sigma_dln_r + 0.5)
            a = a_r * omega_cb**az
            q = q_r * omega_cb**qz
            p = p1 + p2 * (dln_sigma_dln_r + 0.5)

            norm = 1.0 / (
                (2.0 ** (-0.5 - p + q / 2.0) / np.sqrt(np.pi))
                * (2.0**p * gamma(0.5 * q) + gamma(-p + 0.5 * q))
            )
            multiplicity = (
                norm
                * np.sqrt(2.0 * a * nu**2 / np.pi)
                * np.exp(-0.5 * a * nu**2)
                * (1.0 + 1.0 / (a * nu**2) ** p)
                * (np.sqrt(a) * nu) ** (q - 1.0)
            )
            nu_out[row, index] = nu
            multiplicity_out[row, index] = multiplicity
            prefactor_out[row, index] = rho_m / M * (-dln_sigma_dln_r / 3.0)

    return nu_out, multiplicity_out, prefactor_out


def castro23_multiplicity(
    theta: np.ndarray,
    z: np.ndarray | None = None,
    M: np.ndarray | None = None,
    coefficients: np.ndarray | None = None,
    pk_emulator: Emulator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`\nu` and :math:`\nu f(\nu)`.

    The two factors :func:`castro23_bias` differentiates. Exposed because a
    caller with a different prescription for the :math:`\rho_m/M\,|d\ln\sigma/d\ln R|`
    prefactor still wants the same multiplicity function rather than a second
    transcription of Castro23's coefficients.

    Parameters
    ----------
    theta, z, M, coefficients, pk_emulator
        As in :func:`castro23_dndlnM`.

    Returns
    -------
    nu, multiplicity : tuple of ndarray of shape (n_samples, n_z, n_M)
        :math:`\nu = \delta_c/\sigma` and the multiplicity :math:`\nu f(\nu)`.
    """
    nu, multiplicity, _ = _castro23_terms(theta, z, M, coefficients, pk_emulator)
    return nu, multiplicity


def castro23_bias(
    theta: np.ndarray,
    z: np.ndarray | None = None,
    M: np.ndarray | None = None,
    coefficients: np.ndarray | None = None,
    pk_emulator: Emulator | None = None,
) -> np.ndarray:
    r"""Evaluate the Castro23 linear halo bias.

    Computes

    .. math::
        b(M) = 1 - \frac{1}{\delta_c} \frac{d\ln \nu f(\nu)}{d\ln \nu}

    which is the large-scale limit of the peak-background split, evaluated by
    numerical differentiation of the multiplicity function the mass function
    uses. The derivative is taken with :func:`numpy.gradient` on a mass grid the
    caller supplies, so the grid must be strictly increasing and fine enough for
    a finite difference to be meaningful.

    Parameters
    ----------
    theta : ndarray of shape (n_samples, 8)
        Cosmological parameters, columns in :func:`theta_spec` order.
    z : ndarray, optional
        Redshifts. Defaults to :func:`z_grid`.
    M : ndarray, optional
        Halo masses in :math:`M_\odot/h`, strictly increasing. Defaults to
        :data:`FINE_CENTERS`.
    coefficients : ndarray, optional
        The eight Castro23 coefficients. Defaults to the bundled set.
    pk_emulator : Emulator, optional
        Power-spectrum emulator used for :math:`\sigma_{cb}`.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_M)
        The bias, dimensionless.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    z = z_grid() if z is None else np.atleast_1d(np.asarray(z, dtype=float))
    M = FINE_CENTERS if M is None else np.atleast_1d(np.asarray(M, dtype=float))
    if M.size < 2:
        raise ValueError("a numerical derivative needs at least two masses")
    if not np.all(np.diff(M) > 0.0):
        raise ValueError("M must be strictly increasing for the derivative to mean anything")

    # The reference pads the mass grid by 5 per cent at each end before
    # differentiating and drops the two padded columns afterwards, so that the
    # one-sided difference at the ends of the requested grid does not leak into
    # the answer. Reproduced here because the values are compared point by
    # point.
    padded = np.insert(M, [0, M.size], [0.95 * M.min(), 1.05 * M.max()])
    nu, multiplicity = castro23_multiplicity(
        theta, z=z, M=padded, coefficients=coefficients, pk_emulator=pk_emulator
    )

    out = np.empty((theta.shape[0], z.size, M.size), dtype=float)
    for row, parameters in enumerate(theta):
        columns = _column_indices()
        cosmo = Cosmology(
            Omegab=parameters[columns["Omegab"]],
            Omegam=parameters[columns["Omegam"]],
            H0=parameters[columns["H0"]],
            w0=parameters[columns["w0"]],
            wa=parameters[columns["wa"]],
            mnu=parameters[columns["mnu"]],
        )
        for index, redshift in enumerate(z):
            delta_c = _delta_c(redshift, cosmo)
            slope = np.gradient(np.log(multiplicity[row, index]), np.log(nu[row, index]))
            out[row, index] = (1.0 - slope / delta_c)[1:-1]

    return out


def castro23_cumulative(
    theta: np.ndarray,
    z: np.ndarray | None = None,
    M: np.ndarray | None = None,
    coefficients: np.ndarray | None = None,
    pk_emulator: Emulator | None = None,
) -> np.ndarray:
    r"""Integrate :func:`castro23_dndlnM` into :math:`n(\geq M)`, the baseline.

    Parameters
    ----------
    theta, z, M, coefficients, pk_emulator
        As for :func:`castro23_dndlnM`.

    Returns
    -------
    ndarray of shape (n_samples, n_z, n_M)
        Cumulative number density in :math:`(\mathrm{Gpc}/h)^{-3}`, i.e. the
        reference's internal units rather than the physical ones: the
        multiplication by :math:`10^9` happens here so that the result can be
        compared with the reference's own baseline array.
    """
    dndlnm = castro23_dndlnM(theta, z=z, M=M, coefficients=coefficients, pk_emulator=pk_emulator)
    dln_m = _FINE_DLOG10M * np.log(10.0)
    # Summing dn/dlnM over the log-spaced grid from the top down gives n(>= M).
    return np.cumsum(dndlnm[:, :, ::-1] * dln_m * 1e9, axis=2)[:, :, ::-1]


def _power_spectrum(theta_row: np.ndarray, pk_emulator: Emulator | None) -> np.ndarray:
    r"""Return one cosmology's :math:`P_{cb,\mathrm{lin}}(k, z)` as a flat data vector."""
    if pk_emulator is None:
        from .pklin import load_pklin_emulator

        pk_emulator = load_pklin_emulator()
    return np.asarray(pk_emulator.predict(theta_row[None, :], return_std=False))[0]


# ----------------------------------------------------------------------
# The emulator
# ----------------------------------------------------------------------
class HMFEmulator:
    """The bundled RockstarM200m halo mass function emulator.

    Wraps the Gaussian process (a plain :class:`~jet.emulator.Emulator` over the
    ratio) together with the analytic Castro23 baseline and the interpolation
    back onto physical masses. Construct one with :meth:`load`.

    Parameters
    ----------
    emulator : Emulator
        The ratio emulator.
    pk_emulator : Emulator, optional
        Power-spectrum emulator for the baseline's :math:`\\sigma_{cb}`.
        Defaults to the bundled one, loaded on first use.

    Notes
    -----
    A prediction costs about half a second per cosmology. Most of it is the
    baseline: twelve redshifts times four hundred radii, each an integral over a
    thousand wavenumbers. The Gaussian process itself is negligible by
    comparison, which is why the ratio is what was emulated in the first place.
    """

    def __init__(self, emulator: Emulator, pk_emulator: Emulator | None = None) -> None:
        self.emulator = emulator
        self._pk_emulator = pk_emulator

    @classmethod
    def load(cls, path: str | Path | None = None) -> HMFEmulator:
        """Load the bundled emulator, or one from an explicit path.

        Parameters
        ----------
        path : str or path-like, optional
            A bundle written by :mod:`tools.build_hmf_bundle`. Defaults to the
            bundled :data:`BUNDLE_NAME`, resolved against :func:`data_dir`.

        Returns
        -------
        HMFEmulator
        """
        emulator = load_hmf_emulator() if path is None else Emulator.load(path)
        emulator.warn_on_extrapolation = False
        return cls(emulator)

    # -- the pieces -----------------------------------------------------
    @property
    def x_spec(self) -> ParameterSpec:
        """Input specification of the underlying model."""
        return self.emulator.x_spec

    @property
    def y_spec(self) -> DataVectorSpec:
        """Output specification of the underlying model."""
        return self.emulator.y_spec

    def ratio(self, theta: np.ndarray) -> np.ndarray:
        """Return the emulated ratio to the Castro23 baseline.

        Parameters
        ----------
        theta : ndarray of shape (n_samples, 8)
            Cosmological parameters.

        Returns
        -------
        ndarray of shape (n_samples, N_BINS)
            The raw data vector. See the module docstring for its layout.
        """
        return np.asarray(self.emulator.predict(theta, return_std=False))

    def number_density(
        self, theta: np.ndarray, z: np.ndarray | None = None, M: np.ndarray | None = None
    ) -> np.ndarray:
        r"""Return the cumulative halo number density :math:`n(\geq M)`.

        Parameters
        ----------
        theta : ndarray of shape (n_samples, 8)
            Cosmological parameters, columns in :func:`theta_spec` order.
        z : float or array-like, optional
            Redshifts. Defaults to :func:`z_grid`. Any values inside the
            emulator's range may be asked for; they are interpolated.
        M : float or array-like, optional
            Halo masses in :math:`M_\odot/h`. Defaults to :data:`FINE_CENTERS`.

        Returns
        -------
        ndarray of shape (n_samples, n_z, n_M)
            Number density in :math:`(h/\mathrm{Mpc})^3`. Multiply by a volume
            in :math:`(\mathrm{Mpc}/h)^3` to get an expected halo count.
        """
        z, M = self._resolve_axes(z, M)
        return self._per_cosmology(theta, z, M, self._cumulative_spline)

    def dndlnM(
        self, theta: np.ndarray, z: np.ndarray | None = None, M: np.ndarray | None = None
    ) -> np.ndarray:
        r"""Return the differential halo mass function :math:`dn/d\ln M`.

        Obtained by differentiating the interpolated cumulative function
        numerically, as the reference does, rather than by emulating the
        differential function directly: it is the cumulative one the training
        data holds.

        Parameters
        ----------
        theta : ndarray of shape (n_samples, 8)
            Cosmological parameters.
        z, M
            As for :meth:`number_density`.

        Returns
        -------
        ndarray of shape (n_samples, n_z, n_M)
            :math:`dn/d\ln M` in :math:`(h/\mathrm{Mpc})^3`.
        """
        z, M = self._resolve_axes(z, M)
        return self._per_cosmology(theta, z, M, self._differential_spline)

    # -- internals ------------------------------------------------------
    def _per_cosmology(self, theta, z: np.ndarray, M: np.ndarray, build) -> np.ndarray:
        """Evaluate one variant of the interpolation for every row of ``theta``.

        The reference builds one spline per cosmology and so does this: the
        offset folded into the spline's values is a property of the prediction,
        not of the model, and cannot be shared between rows.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        expected = theta_spec().dim
        if theta.shape[1] != expected:
            raise ValueError(f"theta must have {expected} columns, got {theta.shape[1]}")

        out = np.empty((theta.shape[0], z.size, M.size), dtype=float)
        for row in range(theta.shape[0]):
            spline, addnum = build(theta[row])
            out[row] = self._evaluate(spline, z, M) - addnum
        return out * 1e-9

    @staticmethod
    def _resolve_axes(z, M) -> tuple[np.ndarray, np.ndarray]:
        """Broadcast the redshift and mass arguments into 1-D arrays."""
        z = z_grid() if z is None else np.atleast_1d(np.asarray(z, dtype=float))
        M = FINE_CENTERS if M is None else np.atleast_1d(np.asarray(M, dtype=float))
        if np.any(M <= 0.0):
            raise ValueError(f"halo masses must be positive, got {M.min()}")
        return z, M

    def _baseline(self, theta_row: np.ndarray) -> np.ndarray:
        r"""Return one cosmology's Castro23 baseline on the fine grid, ``(N_Z, N_FINE)``."""
        return castro23_cumulative(
            theta_row[None, :], z=z_grid(), M=FINE_CENTERS, pk_emulator=self._pk_emulator
        )[0]

    def _ratio_on_fine_grid(self, theta_row: np.ndarray) -> np.ndarray:
        r"""Resample one cosmology's ratio blocks onto the fine mass grid.

        Returns
        -------
        ndarray of shape (N_Z, N_FINE)
            The ratio evaluated at the fine grid's *lower* edges, which is what
            the reference multiplies the baseline at -- even though the baseline
            itself sits at the centres, half a bin away. See the module
            docstring.
        """
        ratio = self.ratio(theta_row[None, :])[0]
        edges = mass_edges()
        bins = mass_slices()
        blocks = data_slices()
        target = np.log10(_FINE_EDGES[:-1])
        out = np.empty((N_Z, N_FINE), dtype=float)

        for index in range(N_Z):
            mass_start, mass_stop = bins[index]
            start, stop = blocks[index]
            x = np.log10(edges[mass_start:mass_stop])
            y = ratio[start:stop]
            # The reference drops non-positive entries before fitting, so a
            # block that dips through zero simply loses those knots and the
            # spline extrapolates across the gap.
            usable = y > 0.0
            spline = UnivariateSpline(x[usable], np.log10(y[usable]), k=1, s=0, ext=0)
            out[index] = 10.0 ** spline(target)
        return out

    def _cumulative_spline(self, theta_row: np.ndarray) -> tuple[RectBivariateSpline, float]:
        """Build the ``(z, log10 M)`` spline of the cumulative function.

        Returns
        -------
        spline : RectBivariateSpline
            Maps ``(z, log10 M)`` to :math:`\\log_{10}(n + a)`.
        addnum : float
            The offset ``a`` that must be subtracted from ``10**spline``.
            Returned with the spline rather than stored on the instance. The
            reference keeps it as mutable state and shortcuts out of a cache
            hit without recomputing it, so a cached spline can be paired with
            whatever offset the previous call left behind. For this model both
            offsets come out at exactly ``1.0``, so nothing is wrong today --
            but the coupling is a trap for whoever adds a second mass
            definition, and returning the pair costs nothing.
        """
        physical = self._ratio_on_fine_grid(theta_row) * self._baseline(theta_row)
        addnum = float(1.0 - np.min(physical))
        spline = RectBivariateSpline(
            z_grid(),
            np.log10(_FINE_EDGES[:-1]),
            np.log10(physical + addnum),
            kx=1,
            ky=3,
        )
        return spline, addnum

    def _differential_spline(self, theta_row: np.ndarray) -> tuple[RectBivariateSpline, float]:
        """Build the ``(z, log10 M)`` spline of the differential function.

        The cumulative function is first differentiated on the fine grid by a
        central difference in :math:`\\log_{10} M` with a step much smaller than
        the grid spacing, which is what makes the result smooth enough for a
        spline in the first place.

        Returns
        -------
        spline, addnum
            As for :meth:`_cumulative_spline`.
        """
        physical = self._ratio_on_fine_grid(theta_row) * self._baseline(theta_row)
        addnum = float(1.0 - np.min(physical))
        log_m = np.log10(_FINE_EDGES[:-1])
        centres = np.log10(FINE_CENTERS)
        half = _DLOG10M_STEP / 2.0
        dln_m = _DLOG10M_STEP * np.log(10.0)

        # The cubic here is in log10 M over the fine grid; the reference uses a
        # *cubic* interpolant for this step and a linear one for the cumulative
        # path, which is not an inconsistency but a consequence of needing a
        # smooth derivative.
        differential = np.empty_like(physical)
        for index in range(N_Z):
            function = interp1d(
                log_m,
                np.log10(physical[index] + addnum),
                kind="cubic",
                fill_value="extrapolate",
            )
            # The two addnum terms cancel algebraically; they are left in the
            # reference and dropped here for clarity.
            differential[index] = (
                10.0 ** function(centres - half) - 10.0 ** function(centres + half)
            ) / dln_m

        new_addnum = float(1.0 - np.min(differential))
        spline = RectBivariateSpline(
            z_grid(),
            centres,
            np.log10(differential + new_addnum),
            kx=1,
            ky=1,
        )
        return spline, new_addnum

    @staticmethod
    def _evaluate(spline: RectBivariateSpline, z: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Evaluate a ``(z, log10 M)`` spline on a grid of redshifts and masses.

        The reference evaluates with ``grid=True``, which requires both axes to
        be strictly increasing and returns their outer product. Passing
        ``grid=False`` on a broadcast pair gives the same numbers -- it is the
        same tensor-product spline, evaluated at the same points -- without
        forcing the caller to sort. The two agree to the last bit; see the
        test suite.
        """
        values = spline(z[:, None], np.log10(M)[None, :], grid=False)
        return np.power(10.0, values)
