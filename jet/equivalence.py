"""Cosmological spectral equivalence.

For a flat ``w0waCDM`` cosmology, the *spectral equivalence* method constructs
an auxiliary model at each redshift whose distance to the last-scattering
surface and whose :math:`\\sigma_{8,\\mathrm{cb}}(z)` match the original model.
The auxiliary model is either a ``wCDM`` model (one free equation-of-state
parameter) or a ``w0waCDM`` model (two free parameters), and its amplitude is
rescaled so that the clustering amplitude agrees.

This module is a port of ``CEmulator/SpectralEquivalence.py`` from the CSST
emulator. The numerical conventions are deliberately the reference ones:

* the distance is the comoving distance from ``z`` to ``zstar = 1100``;
* the ``wCDM`` root is found with SciPy's secant method (or the reference's
  legacy finite-difference Newton variant);
* the ``w0waCDM`` solution is found with :func:`scipy.optimize.minimize` on
  the bounded box ``w0 in [-1.3, -0.7]``, ``wa in [-0.5, 0.5]``;
* the amplitude is rescaled as ``As_eq = As * (sigma8_target / sigma8_eq)**2``.

Two behaviours of the reference are intentionally preserved:

1. The ``w0waCDM`` problem is underdetermined: one distance constraint with
   two parameters leaves a one-dimensional solution curve, so the returned
   ``(w0, wa)`` depends on the optimizer path. Do not expect a different
   solver to return the same point on that curve.
2. In the normal amplitude path the target :math:`\\sigma_8` comes from CLASS
   or CAMB while the auxiliary model's :math:`\\sigma_8` comes from the
   emulator. When the auxiliary parameters leave the emulator box, the
   reference falls back to CAMB for the auxiliary value as well. This mixed
   behaviour is part of the reference output and is reproduced here.

Parameters are passed as an eight-column ``theta`` array in the order of
:func:`jet.cosmology.cosmo_parameter_spec`::

    Omegab, Omegam, H0, ns, As, w0, wa, mnu

``Omegam`` is the baryon-plus-CDM density, *excluding* massive neutrinos, and
``As`` is the physical amplitude (not ``As * 1e9``). A single sample may be
passed as a one-dimensional array; outputs always keep the sample axis, so a
single sample returns shape ``(1, n_z)``.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, newton

from .cosmology import (
    _COMOVING_A_MIN,
    _COMOVING_GRID_SIZE,
    _VEL_LIGHT_CM_S,
    Cosmology,
)
from .emulator.emulator import Emulator
from .emulator.pklin import sigma8_of_pk
from .emulator.pklin import theta_spec as pklin_theta_spec

__all__ = ["spectral_equivalence_de", "spectral_equivalence_as"]

#: Column order of the ``theta`` arrays accepted by this module.
PARAMETER_NAMES = ("Omegab", "Omegam", "H0", "ns", "As", "w0", "wa", "mnu")

#: Default redshift of the last-scattering surface.
ZSTAR = 1100.0

#: Bounds of the auxiliary ``w0waCDM`` solution, matching the reference.
W0_BOUNDS = (-1.3, -0.7)
WA_BOUNDS = (-0.5, 0.5)

#: Smoothing scale of the amplitude matching, in Mpc/h.
SIGMA8_R = 8.0

#: Maximum wavenumber passed to CLASS and CAMB, in 1/Mpc.
_BOLTZMANN_KMAX = 10.0

#: CMB temperature used by the reference's Boltzmann runs, in kelvin.
_T_CMB = 2.7255

#: Effective number of relativistic species used by the reference.
_NEFF = 3.046

#: Change in the relativistic species count per massive neutrino.
_NEFF_PER_NCDM = 1.0132


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_theta(theta: np.ndarray) -> np.ndarray:
    """Return ``theta`` as a finite ``(n, 8)`` float array."""
    values = np.asarray(theta, dtype=float)
    if values.ndim == 1:
        if values.size != len(PARAMETER_NAMES):
            raise ValueError(
                f"theta must have {len(PARAMETER_NAMES)} columns "
                f"({', '.join(PARAMETER_NAMES)}), got shape {values.shape}"
            )
        values = values[None, :]
    elif values.ndim == 2:
        if values.shape[1] != len(PARAMETER_NAMES):
            raise ValueError(
                f"theta must have {len(PARAMETER_NAMES)} columns "
                f"({', '.join(PARAMETER_NAMES)}), got shape {values.shape}"
            )
    else:
        raise ValueError(f"theta must be one- or two-dimensional, got shape {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("theta must contain at least one sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must be finite")
    return values


def _validate_z(z: np.ndarray) -> np.ndarray:
    """Return ``z`` as a finite, non-negative one-dimensional array."""
    values = np.atleast_1d(np.asarray(z, dtype=float))
    if values.ndim != 1:
        raise ValueError(f"z must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("z must contain at least one redshift")
    if not np.all(np.isfinite(values)):
        raise ValueError("z must be finite")
    if np.any(values < 0.0):
        raise ValueError(f"z must be non-negative, got {values.min()}")
    return values


def _make_cosmology(row: np.ndarray, w0: float, wa: float) -> Cosmology:
    """Build a :class:`Cosmology` from a ``theta`` row and auxiliary DE values."""
    return Cosmology(
        Omegab=row[0],
        Omegam=row[1],
        H0=row[2],
        w0=w0,
        wa=wa,
        mnu=row[7],
    )


class _DistanceGrid:
    """Shared distance grid for all residual evaluations of one cosmology.

    The reference rebuilds a :class:`Cosmology`, its 2048-point scale-factor
    grid and the Fermi-Dirac neutrino term for every residual evaluation. Only
    the dark-energy factor depends on the trial ``(w0, wa)``, so those pieces
    are computed once here. The arithmetic used to evaluate a trial model is
    otherwise identical to :meth:`Cosmology.comoving_distance`.
    """

    def __init__(
        self,
        row: np.ndarray,
        z_queries: np.ndarray,
        zstar: float,
        neutrino_matter_like: bool,
    ) -> None:
        self.z_queries = np.concatenate(
            [np.array([float(zstar)]), np.asarray(z_queries, dtype=float)]
        )
        a_queries = 1.0 / (1.0 + self.z_queries)
        a_low = min(float(a_queries.min()), _COMOVING_A_MIN)
        self.trivial = a_low >= 1.0
        if self.trivial:
            return

        self.a_grid = np.logspace(np.log10(a_low), 0.0, _COMOVING_GRID_SIZE)
        self.log_a_grid = np.log10(self.a_grid)
        self.log_a_queries = np.log10(a_queries)

        z_grid = 1.0 / self.a_grid - 1.0
        self.a_inv = 1.0 / (1.0 + z_grid)
        self.log_a_inv = np.log(self.a_inv)
        self.a2 = self.a_grid**2
        self.da = np.diff(self.a_grid)

        # One Cosmology instance supplies the parameter-derived constants and
        # the neutrino term; it is never asked for a distance.
        self.cosmo = _make_cosmology(row, w0=float(row[5]), wa=float(row[6]))
        self.OmegaL = self.cosmo.OmegaL
        if neutrino_matter_like:
            base = self.cosmo.OmegaM * (1.0 + z_grid) ** 3 + self.cosmo.OmegaR * (1.0 + z_grid) ** 4
        else:
            base = (
                self.cosmo.Omegam * (1.0 + z_grid) ** 3
                + self.cosmo.neutrino_density_times_hubble_squared(z_grid)
                + self.cosmo.OmegaR * (1.0 + z_grid) ** 4
            )
        self.base = base
        self.chi_fac = 1e-5 * _VEL_LIGHT_CM_S / self.cosmo.h0 / 100.0

    def chi(self, w0: float, wa: float) -> np.ndarray:
        """Return the distance at ``[zstar, *z_queries]`` for one trial model."""
        if self.trivial:
            return np.zeros_like(self.z_queries)

        dark_energy = np.exp(3.0 * ((self.a_inv - 1.0) * wa - (1.0 + w0 + wa) * self.log_a_inv))
        inv_e = 1.0 / (np.sqrt(self.base + self.OmegaL * dark_energy) * self.a2)

        average = 0.5 * (inv_e[:-1] + inv_e[1:])
        cumulative = np.cumsum(average[::-1] * self.da[::-1])[::-1]
        chi_at_grid = np.concatenate([cumulative, [0.0]])

        # One spline serves zstar and every requested redshift, matching the
        # reference's values without rebuilding the interpolator per call.
        spline = interp1d(
            self.log_a_grid,
            chi_at_grid,
            kind="cubic",
            assume_sorted=True,
        )
        out = spline(self.log_a_queries) * self.chi_fac
        out[self.z_queries == 0.0] = 0.0
        return out


# ---------------------------------------------------------------------------
# Distance residual and root finders
# ---------------------------------------------------------------------------
def _root_find_newton(f, x0, tol=1e-10, max_iter=100, h=1e-5):
    """SciPy secant root finder with the reference's safeguards."""
    try:
        root = newton(f, x0, fprime=None, tol=tol, maxiter=max_iter)
    except (RuntimeError, ValueError, OverflowError):
        return np.nan, np.nan
    if not np.isfinite(root) or abs(root) > 10.0:
        return np.nan, np.nan
    return float(root), float(f(root))


def finite_diff_newton(f, x0, tol=1e-10, max_iter=100, h=1e-5):
    """Legacy finite-difference Newton solver.

    This reproduces the reference implementation operation for operation,
    including its adaptive step and its heuristic update when the numerical
    derivative vanishes. It is kept because callers may select it explicitly;
    the default is :func:`_root_find_newton`.
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, fx
        # Adaptive step size.
        h = 0.5 * h * (tol / abs(fx)) ** 0.5
        h_adaptive = max(h, 1e-8)
        df = (f(x + h_adaptive) - f(x - h_adaptive)) / (2.0 * h_adaptive)
        if abs(df) < 1e-32:
            # When the derivative is very small, use the reference's stable
            # fallback update.
            x_new = x * (1.0 + fx) ** 10
        else:
            x_new = x - fx / df
            if abs(x_new - x) < tol:
                return x_new, fx
        if abs(x_new) > 10.0:
            return np.nan, np.nan
        x = x_new
    return np.nan, np.nan


def _get_err_w(w: float, grid: _DistanceGrid, index: int, chi_target: float) -> float:
    """Relative distance error of a constant-``w`` auxiliary model."""
    chi = grid.chi(w, 0.0)
    delta = chi[0] - chi[index + 1]
    return float(abs(1.0 - delta / chi_target))


def _get_err_w0wa(
    w0: float,
    wa: float,
    grid: _DistanceGrid,
    index: int,
    chi_target: float,
) -> float:
    """Relative distance error of a ``w0waCDM`` auxiliary model."""
    chi = grid.chi(w0, wa)
    delta = chi[0] - chi[index + 1]
    return float(abs(1.0 - delta / chi_target))


# ---------------------------------------------------------------------------
# Amplitude matching: sigma8 backends
# ---------------------------------------------------------------------------
def _normalise_sigma8_type(sigma8_type: str) -> str:
    value = str(sigma8_type).lower()
    if value not in {"class", "camb", "emulator"}:
        raise ValueError(f"sigma8_type must be 'class', 'camb' or 'emulator', got {sigma8_type!r}")
    return value


def _class_params(row: np.ndarray, z: np.ndarray) -> dict:
    """CLASS parameters matching ``CEmulator.utils.useCLASS``."""
    Omegab, Omegam, H0, ns, As, w0, wa, mnu = row
    h0 = H0 / 100.0
    params = {
        "output": "mPk",
        "P_k_max_1/Mpc": _BOLTZMANN_KMAX / h0,
        "A_s": float(As),
        "n_s": float(ns),
        "h": float(h0),
        "Omega_Lambda": 0.0,
        "wa_fld": float(wa),
        "w0_fld": float(w0),
        "Omega_b": float(Omegab),
        "Omega_cdm": float(Omegam - Omegab),
        # CLASS expects a comma-separated string, formatted exactly as the
        # reference's get_cosmo_class does.
        "z_pk": ", ".join(f"{zz:.4f}" for zz in np.atleast_1d(z)),
        "T_cmb": _T_CMB,
        "gauge": "synchronous",
    }
    if mnu == 0.0:
        params["N_ur"] = _NEFF
    else:
        params["N_ur"] = _NEFF - _NEFF_PER_NCDM
        params["N_ncdm"] = 1
        params["m_ncdm"] = f"{float(mnu):f}"
    return params


def _sigma8_class(row: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Target :math:`\\sigma_{8,\\mathrm{cb}}(z)` from one CLASS run."""
    try:
        from classy import Class
    except ImportError:
        raise ImportError(
            "sigma8_type='class' requires the 'classy' package; "
            "install it with 'pip install classy' or use sigma8_type='emulator'"
        ) from None

    z = np.atleast_1d(np.asarray(z, dtype=float))
    cosmo = Class()
    cosmo.set(_class_params(row, z))
    cosmo.compute()

    h0 = row[2] / 100.0
    radius = SIGMA8_R / h0
    if row[7] == 0.0:
        return np.array([float(cosmo.sigma(radius, zz)) for zz in z])
    return np.array([float(cosmo.sigma_cb(radius, zz)) for zz in z])


def _sigma8_camb(row: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Target :math:`\\sigma_{8,\\mathrm{cb}}(z)` from one CAMB run."""
    try:
        import camb
    except ImportError:
        raise ImportError(
            "sigma8_type='camb' or an out-of-bounds amplitude fallback requires "
            "the 'camb' package; install it with 'pip install camb'"
        ) from None

    z = np.atleast_1d(np.asarray(z, dtype=float))
    Omegab, Omegam, H0, ns, As, w0, wa, mnu = row
    h0 = H0 / 100.0
    params = camb.CAMBparams()
    params.set_cosmology(
        H0=float(h0 * 100.0),
        omch2=float((Omegam - Omegab) * h0 * h0),
        ombh2=float(Omegab * h0 * h0),
        num_massive_neutrinos=0 if mnu == 0.0 else 1,
        standard_neutrino_neff=_NEFF,
        mnu=float(mnu),
    )
    params.InitPower.set_params(ns=float(ns), As=float(As))
    params.set_dark_energy(w=float(w0), wa=float(wa), dark_energy_model="ppf")
    params.set_matter_power(kmax=_BOLTZMANN_KMAX, redshifts=z)
    results = camb.get_results(params)

    # CAMB may reorder or extend the redshift list; look each requested value
    # up explicitly rather than assuming its position.
    transfer_z = np.asarray(results.transfer_redshifts, dtype=float)
    values = []
    for zz in z:
        matches = np.flatnonzero(np.isclose(transfer_z, zz))
        if matches.size == 0:
            raise RuntimeError(f"CAMB did not return sigma8 at z={zz}")
        sigma = results.get_sigmaR(
            R=SIGMA8_R,
            z_indices=int(matches[0]),
            hubble_units=True,
            var1="delta_nonu",
            var2="delta_nonu",
        )
        values.append(float(np.asarray(sigma).reshape(-1)[0]))
    return np.array(values)


def _sigma8_from_pk(Pk: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate :func:`sigma8_of_pk` row by row with per-row redshifts."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    Pk = np.atleast_2d(np.asarray(Pk, dtype=float))
    if Pk.shape[0] != z.size:
        raise ValueError(f"Pk has {Pk.shape[0]} rows but z has {z.size} entries")
    out = np.empty(z.size, dtype=float)
    for zz in np.unique(z):
        mask = z == zz
        out[mask] = sigma8_of_pk(Pk[mask], R=SIGMA8_R, z=float(zz))
    return out


def _sigma8_emulator(
    rows: np.ndarray,
    z: np.ndarray,
    emulator: Emulator,
) -> np.ndarray:
    """Evaluate the emulator :math:`\\sigma_{8,\\mathrm{cb}}(z)` for rows."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if rows.shape[0] == 1 and z.size > 1:
        # One cosmology evaluated at several redshifts: predict once.
        Pk = emulator.predict(rows, return_std=False)
        Pk = np.repeat(Pk, z.size, axis=0)
    else:
        Pk = emulator.predict(rows, return_std=False)
    return _sigma8_from_pk(Pk, z)


@lru_cache(maxsize=64)
def _target_sigma8_cached(
    sigma8_type: str,
    row_tuple: tuple[float, ...],
    z_tuple: tuple[float, ...],
) -> np.ndarray:
    """Cached CLASS/CAMB target sigma8 for repeated identical calls."""
    row = np.asarray(row_tuple, dtype=float)
    z = np.asarray(z_tuple, dtype=float)
    if sigma8_type == "class":
        return _sigma8_class(row, z)
    return _sigma8_camb(row, z)


def _target_sigma8(
    row: np.ndarray,
    z: np.ndarray,
    sigma8_type: str,
    emulator: Emulator | None,
) -> np.ndarray:
    """Target sigma8 of the original model at every ``z``."""
    if sigma8_type == "emulator":
        return _sigma8_emulator(row[None, :], z, emulator)
    return _target_sigma8_cached(
        sigma8_type,
        tuple(float(value) for value in row),
        tuple(float(value) for value in np.atleast_1d(z)),
    )


def _resolved_emulator(emulator: Emulator | None) -> Emulator:
    """Return the caller's emulator or load the bundled one."""
    if emulator is not None:
        if not isinstance(emulator, Emulator):
            raise TypeError(
                f"emulator must be a jet.emulator.Emulator, got {type(emulator).__name__}"
            )
        return emulator
    from .emulator.pklin import load_pklin_emulator

    return load_pklin_emulator()


def _outside_bounds(rows: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Boolean mask of rows outside an array of ``(dim, 2)`` parameter bounds."""
    return np.any((rows < bounds[:, 0]) | (rows > bounds[:, 1]), axis=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def spectral_equivalence_de(
    theta,
    z,
    *,
    wcdm: bool = False,
    return_error: bool = False,
    method: str = "newton",
    w0wa_method: str = "L-BFGS-B",
    zstar: float = ZSTAR,
    neutrino_matter_like: bool = False,
):
    r"""Find the auxiliary DE model matching the distance to the LSS.

    Parameters
    ----------
    theta : array-like of shape (8,) or (n, 8)
        Cosmological parameters in :data:`PARAMETER_NAMES` order. ``Omegam``
        excludes massive neutrinos and ``As`` is the physical amplitude.
    z : array-like of shape (n_z,)
        Redshifts at which the auxiliary model is constructed.
    wcdm : bool, optional
        If true, solve for a constant-``w`` auxiliary model. Otherwise solve
        for the two ``w0waCDM`` parameters.
    return_error : bool, optional
        Also return the relative distance error
        ``|1 - chi_aux(z, zstar) / chi_original(z, zstar)|``.
    method : {"newton", "finite_diff_newton"}, optional
        Root finder used by the ``wCDM`` branch. ``"newton"`` is SciPy's
        secant method; ``"finite_diff_newton"`` is the reference's legacy
        finite-difference Newton solver.
    w0wa_method : str, optional
        ``method`` argument passed to :func:`scipy.optimize.minimize` for the
        ``w0waCDM`` branch.
    zstar : float, optional
        Redshift of the last-scattering surface.
    neutrino_matter_like : bool, optional
        Treat massive neutrinos as matter in the distance integral. Passed
        through to :meth:`jet.cosmology.Cosmology.comoving_distance`.

    Returns
    -------
    w_const : ndarray of shape (n, n_z)
        Auxiliary constant ``w``. Returned when ``wcdm=True``.
    w0, wa : ndarray of shape (n, n_z)
        Auxiliary ``w0waCDM`` parameters. Returned when ``wcdm=False``.
    err : ndarray of shape (n, n_z)
        Relative distance error, appended when ``return_error=True``.

    Raises
    ------
    ValueError
        If an input shape is invalid or ``method`` is unknown.
    """
    theta_values = _validate_theta(theta)
    z_values = _validate_z(z)
    zstar = float(zstar)

    n_samples, n_z = theta_values.shape[0], z_values.size

    # One grid per sample serves the target model and every trial model.
    chi_target = np.empty((n_samples, n_z), dtype=float)
    grids: list[_DistanceGrid] = []
    for index, row in enumerate(theta_values):
        grid = _DistanceGrid(row, z_values, zstar, neutrino_matter_like)
        chi_original = grid.chi(float(row[5]), float(row[6]))
        chi_target[index] = chi_original[0] - chi_original[1:]
        grids.append(grid)

    if wcdm:
        if method == "newton":
            root_func = _root_find_newton
        elif method == "finite_diff_newton":
            root_func = finite_diff_newton
        else:
            raise ValueError(f"method must be 'newton' or 'finite_diff_newton', got {method!r}")

        w_const = np.empty((n_samples, n_z), dtype=float)
        err = np.empty((n_samples, n_z), dtype=float)
        for index, row in enumerate(theta_values):
            grid = grids[index]
            for j in range(n_z):
                func = lambda value, _j=j, _grid=grid, _chi=chi_target[index, j]: (  # noqa: E731
                    _get_err_w(value, _grid, _j, _chi)
                )
                w_const[index, j], err[index, j] = root_func(
                    func,
                    float(row[5]),
                    max_iter=100,
                    tol=1e-7,
                )
        if not np.all(np.isfinite(w_const)):
            warnings.warn(
                "spectral_equivalence_de: the wCDM root finder did not "
                "converge at one or more redshifts; returning NaN",
                RuntimeWarning,
                stacklevel=2,
            )
        if return_error:
            return w_const, err
        return w_const

    w0 = np.empty((n_samples, n_z), dtype=float)
    wa = np.empty((n_samples, n_z), dtype=float)
    err = np.empty((n_samples, n_z), dtype=float)
    for index, row in enumerate(theta_values):
        w0_initial = float(np.clip(row[5], *W0_BOUNDS))
        wa_initial = float(np.clip(row[6], *WA_BOUNDS))
        grid = grids[index]
        for j, zz in enumerate(z_values):
            func = lambda values, _j=j, _grid=grid, _chi=chi_target[index, j]: (  # noqa: E731
                _get_err_w0wa(values[0], values[1], _grid, _j, _chi)
            )
            result = minimize(
                func,
                [w0_initial, wa_initial],
                method=w0wa_method,
                bounds=[W0_BOUNDS, WA_BOUNDS],
            )
            # L-BFGS-B reports ABNORMAL when the starting point is already an
            # optimum, which is common here and harmless. Warn only when the
            # residual is actually large.
            if not result.success and result.fun > 1e-8:
                warnings.warn(
                    "spectral_equivalence_de: the w0waCDM minimiser did not "
                    f"report success at z={zz} ({result.message})",
                    RuntimeWarning,
                    stacklevel=2,
                )
            w0[index, j], wa[index, j] = result.x
            err[index, j] = result.fun
    if return_error:
        return w0, wa, err
    return w0, wa


def spectral_equivalence_as(
    theta,
    z,
    w0_eq,
    wa_eq=None,
    *,
    sigma8_type: str = "class",
    emulator: Emulator | None = None,
) -> np.ndarray:
    r"""Rescale ``As`` so the auxiliary model matches the target sigma8.

    The amplitude is scaled as

    .. math::

        A_{s,\mathrm{eq}}(z) = A_s
        \left(\frac{\sigma_{8,\mathrm{cb}}^{\mathrm{target}}(z)}
        {\sigma_{8,\mathrm{cb}}^{\mathrm{eq}}(z)}\right)^2,

    which relies on :math:`\sigma_8 \propto \sqrt{A_s}` in the linear regime.

    Parameters
    ----------
    theta : array-like of shape (8,) or (n, 8)
        Original cosmological parameters, in :data:`PARAMETER_NAMES` order.
    z : array-like of shape (n_z,)
        Redshifts.
    w0_eq, wa_eq : array-like
        Auxiliary dark-energy parameters. For a single input sample a
        one-dimensional array of length ``n_z`` is accepted; for ``n > 1`` the
        shape must be ``(n, n_z)``. ``wa_eq=None`` means zero.
    sigma8_type : {"class", "camb", "emulator"}, optional
        Backend for the **target** sigma8. For ``"class"`` and ``"camb"`` the
        auxiliary sigma8 comes from the emulator when the auxiliary parameters
        lie inside its training box, and from CAMB when they do not. This
        mixed behaviour is the reference's and is preserved deliberately.
        ``"emulator"`` uses the emulator for both the target and the auxiliary
        value.
    emulator : jet.emulator.Emulator, optional
        Emulator used for the auxiliary sigma8. ``None`` loads the bundled
        linear P(k) emulator.

    Returns
    -------
    ndarray of shape (n, n_z)
        Equivalent physical ``As`` values.
    """
    theta_values = _validate_theta(theta)
    z_values = _validate_z(z)
    sigma8_type = _normalise_sigma8_type(sigma8_type)

    # The bounds live in the shared cosmological spec, so the emulator bundle
    # is only needed when an auxiliary sigma8 actually comes from it.
    if sigma8_type == "emulator":
        emulator = _resolved_emulator(emulator)
        bounds = emulator.x_spec.bounds_array()
    elif emulator is not None:
        bounds = emulator.x_spec.bounds_array()
    else:
        bounds = pklin_theta_spec().bounds_array()

    n_samples, n_z = theta_values.shape[0], z_values.size

    w0_eq_values = np.asarray(w0_eq, dtype=float)
    if w0_eq_values.ndim == 1:
        if n_samples != 1:
            raise ValueError(
                "w0_eq must have shape (n, n_z) when theta contains more than "
                f"one sample, got shape {w0_eq_values.shape}"
            )
        if w0_eq_values.size != n_z:
            raise ValueError(f"w0_eq must have length n_z={n_z}, got {w0_eq_values.size}")
        w0_eq_values = w0_eq_values[None, :]
    elif w0_eq_values.shape != (n_samples, n_z):
        raise ValueError(f"w0_eq must have shape ({n_samples}, {n_z}), got {w0_eq_values.shape}")
    if not np.all(np.isfinite(w0_eq_values)):
        raise ValueError("w0_eq must be finite")

    if wa_eq is None:
        wa_eq_values = np.zeros_like(w0_eq_values)
    else:
        wa_eq_values = np.asarray(wa_eq, dtype=float)
        if wa_eq_values.ndim == 1:
            if n_samples != 1 or wa_eq_values.size != n_z:
                raise ValueError(
                    f"wa_eq must have shape (1, n_z) or (n, n_z), got {wa_eq_values.shape}"
                )
            wa_eq_values = wa_eq_values[None, :]
        elif wa_eq_values.shape != (n_samples, n_z):
            raise ValueError(
                f"wa_eq must have shape ({n_samples}, {n_z}), got {wa_eq_values.shape}"
            )
        if not np.all(np.isfinite(wa_eq_values)):
            raise ValueError("wa_eq must be finite")

    # Target sigma8 of the original model.
    sigma8_target = np.empty((n_samples, n_z), dtype=float)
    for index, row in enumerate(theta_values):
        sigma8_target[index] = _target_sigma8(row, z_values, sigma8_type, emulator)

    # Auxiliary theta, flattened row-major as (sample, redshift).
    theta_eq = np.empty((n_samples, n_z, len(PARAMETER_NAMES)), dtype=float)
    theta_eq[:, :, 0] = theta_values[:, 0, None]
    theta_eq[:, :, 1] = theta_values[:, 1, None]
    theta_eq[:, :, 2] = theta_values[:, 2, None]
    theta_eq[:, :, 3] = theta_values[:, 3, None]
    theta_eq[:, :, 4] = theta_values[:, 4, None]
    theta_eq[:, :, 5] = w0_eq_values
    theta_eq[:, :, 6] = wa_eq_values
    theta_eq[:, :, 7] = theta_values[:, 7, None]
    theta_eq_flat = theta_eq.reshape(n_samples * n_z, len(PARAMETER_NAMES))
    z_flat = np.tile(z_values, n_samples)

    sigma8_aux = np.empty(n_samples * n_z, dtype=float)

    # The reference uses the emulator whenever the auxiliary model is inside
    # its design box. Outside the box it falls back to CAMB (the reference's
    # set_cosmos default), not to sigma8_type. The explicit "emulator" mode is
    # a jet extension and always uses the emulator.
    if sigma8_type == "emulator":
        emulator_mask = np.ones(n_samples * n_z, dtype=bool)
    else:
        emulator_mask = ~_outside_bounds(theta_eq_flat, bounds)

    if np.any(emulator_mask):
        if emulator is None:
            emulator = _resolved_emulator(None)
        sigma8_aux[emulator_mask] = _sigma8_emulator(
            theta_eq_flat[emulator_mask],
            z_flat[emulator_mask],
            emulator,
        )

    if np.any(~emulator_mask):
        warnings.warn(
            "spectral_equivalence_as: auxiliary parameters leave the "
            "emulator's training box; falling back to CAMB for sigma8",
            UserWarning,
            stacklevel=2,
        )
        for flat_index in np.flatnonzero(~emulator_mask):
            sigma8_aux[flat_index] = _sigma8_camb(
                theta_eq_flat[flat_index],
                np.array([z_flat[flat_index]]),
            )[0]

    sigma8_aux = sigma8_aux.reshape(n_samples, n_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.power(sigma8_target / sigma8_aux, 2)
    return theta_values[:, 4, None] * amplitude
