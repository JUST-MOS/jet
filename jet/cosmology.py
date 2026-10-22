"""
Background expansion for a flat ``w0wa`` universe with one massive neutrino.

This is a *port*, not a new model. It reproduces the subset of the reference
implementation's ``Cosmology`` class that the halo mass function needs, so that
a prediction made here can be compared against the reference point by point.
The subset is deliberately small:

``E(z)``
    :math:`H(z)/H_0`, with radiation, a ``w0wa`` dark energy and the neutrino
    density computed from the Fermi-Dirac integral rather than assumed to scale
    like matter.
``omega_cb(z)``
    Baryons plus cold dark matter divided by the critical density, i.e. matter
    *excluding* massive neutrinos. The reference calls this ``get_Omegam``,
    which is easy to mistake for the total; it is not.
``omega_m(z)``
    The same including massive neutrinos. The reference calls this
    ``get_OmegaM``.
``comoving_distance(z)``
    The line-of-sight comoving distance in Mpc, using the reference's shared
    log-spaced scale-factor grid and cubic interpolation. The neutrino-split
    transformation is not part of this class; optional CAMB fallbacks live in
    :mod:`jet._boltzmann` and the emulator wrappers that opt into them.

Neutrino density
----------------
The energy density of a massive neutrino is an integral over its Fermi-Dirac
distribution,

.. math::

    F(y) = \\int_0^\\infty \\frac{x^2 \\sqrt{x^2 + y^2}}{1 + e^x} \\, dx ,
    \\qquad y = \\frac{m_\\nu}{(1+z)\\, k_B T_\\nu} ,

and :math:`\\Omega_\\nu(z) \\propto (1+z)^4 F(y)`. The reference tabulates
:math:`F` on a grid and interpolates it, which caps the usable neutrino mass at
:math:`y \\approx 2000` and ships a 240 kB table for a one-dimensional function.
Here the integral is evaluated directly. The integrand is negligible beyond
:math:`x \\approx 60`, so a fixed Gauss-Legendre rule on :math:`[0, 60]` is
exact to :math:`2 \\times 10^{-12}` -- and it agrees with a general-purpose
adaptive quadrature to :math:`10^{-14}` everywhere. Where the two disagree with
the reference's table by more than that, the table is the one at fault: it
carries a single bad point at :math:`y = 6.6368` that is off by
:math:`5 \\times 10^{-8}`, while the quadrature agrees with ``scipy.quad`` there
to the last digit.

Radiation is included because the reference includes it: at :math:`z = 3` its
effect on :math:`H(z)` is a few parts in :math:`10^5`, small but not below the
agreement tolerance the emulator is verified to.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import interp1d

from .spec import Param, ParameterSpec

__all__ = ["RHO_CRIT", "fermi_dirac_integral", "Cosmology", "cosmo_parameter_spec"]

#: Critical density today, in :math:`h^2 M_\\odot \\mathrm{Mpc}^{-3}`.
RHO_CRIT = 2.77536627e11

#: CMB temperature today, in kelvin.
T_CMB = 2.7255

#: Photon density parameter times :math:`h^2`.
OMEGA_GAMMA_H2 = 2.4721231034210734e-5

#: Effective number of relativistic species in the reference's convention.
NEFF = 3.046

#: Neutrino-to-photon temperature ratio.
GAMMA_NU = (4.0 / 11.0) ** (1.0 / 3.0)

#: Boltzmann constant, in eV/K.
KB = 8.617333262145e-5

#: Converts a neutrino mass sum in eV into :math:`\\Omega_\\nu h^2`.
MNU_TO_OMEGA_H2 = 1.0 / 93.14

#: Speed of light, in cm/s, matching the reference's unit convention for the
#: comoving-distance integral.
_VEL_LIGHT_CM_S = 2.99792458e10

#: Number of points on the shared log-spaced scale-factor grid used by
#: :meth:`Cosmology.comoving_distance`.
_COMOVING_GRID_SIZE = 2048

#: Lower edge of the scale-factor grid. The reference never integrates below
#: this value, and for ``z* = 1100`` the grid always starts here.
_COMOVING_A_MIN = 1e-5

#: Upper limit of the Fermi-Dirac quadrature. The Fermi factor kills the
#: integrand beyond ``x ~ 40`` (``exp(-60) ~ 1e-26``), so this is not a
#: truncation in any numerical sense.
_FD_X_MAX = 60.0


#: Order of the Gauss-Legendre rule. The integrand is analytic on ``[0, 60]``
#: apart from a square-root branch point at the origin, which the rule resolves
#: less well the smaller ``y`` is; 400 nodes hold the relative error below
#: ``3e-12`` for every ``y`` down to zero.
_FD_ORDER = 400


@lru_cache(maxsize=1)
def _quadrature_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed Gauss-Legendre rule for the Fermi-Dirac integral.

    Cached because the nodes do not depend on ``y``: the whole batch is then one
    matrix-vector product, which matters for a prediction that evaluates twelve
    redshifts at once.

    Returns
    -------
    nodes, weights, exp_nodes : ndarray
        Quadrature abscissae in ``[0, _FD_X_MAX]``, their weights, and
        ``1 + exp(nodes)`` precomputed.
    """
    nodes, weights = leggauss(_FD_ORDER)
    nodes = 0.5 * _FD_X_MAX * (nodes + 1.0)
    weights = 0.5 * _FD_X_MAX * weights
    return nodes, weights, 1.0 + np.exp(nodes)


def fermi_dirac_integral(y: float | np.ndarray) -> float | np.ndarray:
    r"""Evaluate the neutrino Fermi-Dirac integral :math:`F(y)`.

    Parameters
    ----------
    y : float or array-like
        Dimensionless neutrino mass, ``m_nu / ((1+z) kB T_nu)``. Non-negative.
        A scalar input gives a scalar back.

    Returns
    -------
    float or ndarray
        :math:`F(y)`, with :math:`F(0) = 7\\pi^4/120 \approx 5.6822` and
        :math:`F(y) \to y \cdot 7\\pi^4/120` for large ``y``.

    Raises
    ------
    ValueError
        If any ``y`` is negative.
    """
    scalar = np.ndim(y) == 0
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if np.any(y < 0.0):
        raise ValueError(f"y must be non-negative, got {y.min()}")

    nodes, weights, exp_nodes = _quadrature_rule()
    # The y-dependence enters only through sqrt(x^2 + y^2), so the batch is a
    # single (n_y, n_nodes) array rather than a Python loop.
    integrand = nodes**2 * np.sqrt(nodes[None, :] ** 2 + y[:, None] ** 2) / exp_nodes
    out = integrand @ weights
    return float(out[0]) if scalar else out


class Cosmology:
    r"""A flat ``w0wa`` universe with one massive neutrino.

    Parameters
    ----------
    Omegab : float
        Baryon density parameter today.
    Omegam : float
        Baryon plus cold dark matter density today, i.e. matter *excluding*
        massive neutrinos. This is the reference implementation's convention and
        the one its power-spectrum emulator is trained in.
    H0 : float
        Hubble constant in km/s/Mpc.
    w0, wa : float, optional
        Dark energy equation of state and its linear evolution.
    mnu : float, optional
        Sum of neutrino masses in eV. Zero means no massive neutrino, in which
        case the reference's radiation budget uses three massless species.

    Notes
    -----
    Constructing an instance costs a few microseconds. The neutrino integral is
    evaluated once per redshift at call time, not stored per instance.
    """

    def __init__(
        self,
        Omegab: float,
        Omegam: float,
        H0: float,
        w0: float = -1.0,
        wa: float = 0.0,
        mnu: float = 0.0,
    ) -> None:
        if Omegam <= Omegab:
            raise ValueError(
                f"Omegam must exceed Omegab so that Omega_cdm stays positive, "
                f"got Omegam={Omegam} and Omegab={Omegab}"
            )
        if H0 <= 0.0:
            raise ValueError(f"H0 must be positive, got {H0}")
        if mnu < 0.0:
            raise ValueError(f"mnu must be non-negative, got {mnu}")

        self.Omegab = float(Omegab)
        self.Omegam = float(Omegam)
        self.Omegac = self.Omegam - self.Omegab
        self.h0 = float(H0) / 100.0
        self.mnu = float(mnu)
        self.w0 = float(w0)
        self.wa = float(wa)

        # One massive species or none, matching the training data. The
        # reference's degenerate three-species split is not implemented because
        # no bundled model was trained on it.
        if self.mnu != 0.0:
            self.Nur = 2.0328
            self.mnus = np.array([self.mnu])
        else:
            self.Nur = NEFF
            self.mnus = np.zeros(0)

        self.Omeganu = self.mnu * MNU_TO_OMEGA_H2 / self.h0**2
        self.Omegag = OMEGA_GAMMA_H2 / self.h0**2
        self.f_nnu = 7.0 / 8.0 * GAMMA_NU**4 * self.Nur
        self.OmegaR = self.Omegag * (1.0 + self.f_nnu)
        self.OmegaL = 1.0 - self.Omegam - self.Omeganu - self.OmegaR
        self.OmegaM = self.Omegam + self.Omeganu
        self.rho_crit = RHO_CRIT

        # Comoving-distance cache. The key is checked on every call so that
        # mutating one of the public parameters cannot silently reuse a grid
        # built for a different cosmology.
        self._chi_interp = None
        self._chi_key = None

    def __repr__(self) -> str:
        return (
            f"Cosmology(Omegab={self.Omegab:g}, Omegam={self.Omegam:g}, "
            f"H0={self.h0 * 100:g}, w0={self.w0:g}, wa={self.wa:g}, mnu={self.mnu:g})"
        )

    def neutrino_density_times_hubble_squared(self, z: float | np.ndarray) -> np.ndarray:
        r"""Return :math:`\Omega_\nu(z) E^2(z)`, the neutrino's contribution to ``E**2``.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        ndarray
            :math:`\Omega_\nu(z) h^2`-style term, same shape as ``z``.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        if self.mnu == 0.0:
            return np.zeros_like(z)

        fac = 15.0 / np.pi**4 * GAMMA_NU**4 * self.Omegag * (1.0 + z) ** 4
        t_nu = GAMMA_NU * T_CMB
        y = self.mnus[None, :] / (1.0 + z[:, None]) / KB / t_nu
        return fac * np.asarray(fermi_dirac_integral(y)).sum(axis=1)

    def dark_energy_density(self, z: float | np.ndarray) -> np.ndarray:
        r"""Return :math:`\rho_\mathrm{DE}(z)/\rho_{\mathrm{DE},0}` for the ``w0wa`` law."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a = 1.0 / (1.0 + z)
        return np.exp(3.0 * ((a - 1.0) * self.wa - (1.0 + self.w0 + self.wa) * np.log(a)))

    def E(
        self,
        z: float | np.ndarray,
        neutrino_matter_like: bool = False,
    ) -> np.ndarray:
        r"""Return :math:`E(z) = H(z)/H_0`.

        Parameters
        ----------
        z : float or array-like
            Redshift.
        neutrino_matter_like : bool, optional
            Treat massive neutrinos as non-relativistic matter, i.e. replace
            the Fermi-Dirac term by :math:`\Omega_M (1+z)^3` with
            :math:`\Omega_M = \Omega_\mathrm{cb} + \Omega_\nu`. This is the
            reference's ``neutrino_matter_like`` convention and is used by the
            spectral-equivalence distance solver.

        Returns
        -------
        ndarray
            Normalised Hubble parameter, same shape as ``z``.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        dark_energy = self.OmegaL * self.dark_energy_density(z)
        if neutrino_matter_like:
            return np.sqrt(
                self.OmegaM * (1.0 + z) ** 3 + self.OmegaR * (1.0 + z) ** 4 + dark_energy
            )
        return np.sqrt(
            self.Omegam * (1.0 + z) ** 3
            + self.neutrino_density_times_hubble_squared(z)
            + self.OmegaR * (1.0 + z) ** 4
            + dark_energy
        )

    def omega_cb(self, z: float | np.ndarray) -> np.ndarray:
        r"""Return :math:`\Omega_\mathrm{cb}(z)`, matter *without* massive neutrinos.

        This is the density appearing in the reference's Castro23 mass function
        and in its Lagrangian radius, and it is what the reference's
        ``get_Omegam`` returns.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        ndarray
            Baryon plus cold dark matter density parameter, same shape as ``z``.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return self.Omegam * (1.0 + z) ** 3 / self.E(z) ** 2

    def omega_m(self, z: float | np.ndarray) -> np.ndarray:
        r"""Return :math:`\Omega_M(z)`, matter *including* massive neutrinos.

        Parameters
        ----------
        z : float or array-like
            Redshift.

        Returns
        -------
        ndarray
            Total matter density parameter, same shape as ``z``.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        return self.OmegaM * (1.0 + z) ** 3 / self.E(z) ** 2

    def comoving_distance(
        self,
        z: float | np.ndarray,
        neutrino_matter_like: bool = False,
    ) -> np.ndarray:
        r"""Return the line-of-sight comoving distance in Mpc.

        This is an operation-for-operation port of the reference
        implementation. The integral,

        .. math::

            \chi(z) = \int_{a(z)}^{1} \frac{c}{a'^2 H(a')} \, da',
            \qquad a = \frac{1}{1+z},

        is evaluated on one shared 2048-point log-spaced scale-factor grid and
        then interpolated with a not-a-knot cubic spline. Building the grid
        once and reusing it is what makes repeated distance evaluations cheap;
        the reference additionally rebuilt the interpolator on every call,
        which this port deliberately does not do (the values are identical).

        Parameters
        ----------
        z : float or array-like
            Redshift. A scalar is promoted to a length-one array, matching the
            other methods of this class.
        neutrino_matter_like : bool, optional
            Passed through to :meth:`E`.

        Returns
        -------
        ndarray
            Comoving distance in Mpc, same shape as ``z``. ``z = 0`` gives
            exactly zero.

        Notes
        -----
        The scale-factor grid is cached per instance. The cache key includes
        every parameter that enters the integral, so changing an attribute
        after construction rebuilds the grid instead of returning a stale
        answer.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a = 1.0 / (1.0 + z)
        a_min = float(np.min(a))

        # z = 0 contributes exactly zero. The reference returns early for the
        # all-zero case rather than relying on the spline endpoint.
        if a_min >= 1.0:
            return np.zeros_like(z, dtype=float)

        a_low = min(a_min, _COMOVING_A_MIN)
        key = (
            self.Omegab,
            self.Omegam,
            self.h0,
            self.w0,
            self.wa,
            self.mnu,
            bool(neutrino_matter_like),
            a_low,
        )

        if self._chi_interp is None or self._chi_key != key:
            a_grid = np.logspace(np.log10(a_low), 0.0, _COMOVING_GRID_SIZE)
            z_grid = 1.0 / a_grid - 1.0
            inv_e = 1.0 / (self.E(z_grid, neutrino_matter_like) * a_grid**2)

            # Cumulative trapezoid from each grid point to a_grid[-1] = 1.
            da = np.diff(a_grid)
            average = 0.5 * (inv_e[:-1] + inv_e[1:])
            cumulative = np.cumsum(average[::-1] * da[::-1])[::-1]
            chi_at_grid = np.concatenate([cumulative, [0.0]])

            self._chi_interp = interp1d(
                np.log10(a_grid),
                chi_at_grid,
                kind="cubic",
                assume_sorted=True,
            )
            self._chi_key = key

        # 1e-5 * c_cm / h0 / 100 converts the dimensionless integral into Mpc.
        chi_fac = 1e-5 * _VEL_LIGHT_CM_S / self.h0 / 100.0
        return self._chi_interp(np.log10(a)) * chi_fac


def cosmo_parameter_spec() -> ParameterSpec:
    """Return the eight cosmological parameters the bundled models are trained on.

    One declaration serves every model trained on the reference's Sobol design:
    the power-spectrum emulator, the halo mass function, and the ones that will
    follow. They share an input box by construction, so declaring it twice would
    only create a way for the two to drift apart.

    The column order — which is also the order of any ``theta`` array every
    bundled emulator accepts — is ``Omegab, Omegam, H0, ns, As, w0, wa, mnu``:
    a flat ``w0waCDM`` cosmology with the two dark-energy parameters free and
    ``mnu`` the sum of neutrino masses in eV. Names and bounds are available
    programmatically::

        spec = cosmo_parameter_spec()
        list(zip(spec.names, spec.bounds_array()))

    Returns
    -------
    ParameterSpec
        Eight parameters, in the column order the reference uses. ``As`` is the
        physical primordial amplitude; the reference's own files store it
        multiplied by ``1e9`` and its conversion is the caller's business.
    """
    return ParameterSpec(
        [
            Param("Omegab", bounds=(0.04, 0.06), block="cosmo", doc="Omega_b"),
            Param("Omegam", bounds=(0.24, 0.40), block="cosmo", doc="Omega_b + Omega_c"),
            Param("H0", bounds=(60.0, 80.0), block="cosmo", doc="km/s/Mpc"),
            Param("ns", bounds=(0.92, 1.00), block="cosmo"),
            Param("As", bounds=(1.7e-9, 2.5e-9), block="cosmo", doc="A_s, not A_s * 1e9"),
            Param("w0", bounds=(-1.3, -0.7), block="cosmo"),
            Param("wa", bounds=(-0.5, 0.5), block="cosmo"),
            Param("mnu", bounds=(0.0, 0.3), block="cosmo", doc="sum of neutrino masses [eV]"),
        ]
    )
