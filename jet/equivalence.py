"""Cosmological spectral equivalence.

For a flat ``w0waCDM`` cosmology, the *spectral equivalence* method constructs
an auxiliary model at each redshift whose distance to the last-scattering
surface and whose total-matter :math:`\\sigma_8(z)` match the original model.
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

Two behaviours are important to keep in mind:

1. The ``w0waCDM`` problem is underdetermined: one distance constraint with
   two parameters leaves a one-dimensional solution curve, so the returned
   ``(w0, wa)`` depends on the optimizer path. Do not expect a different
   solver to return the same point on that curve.
2. ``sigma8_type`` controls the **entire** amplitude calculation, not just the
   target. ``"emulator"`` (the default) uses the total-matter emulator for
   both the target and the auxiliary model, with a per-side CAMB fallback when
   that side leaves the emulator box (``fallback_to_camb=True`` by default).
   ``"class"`` and ``"camb"`` use that Boltzmann backend for both sides. There
   is no mixed target/auxiliary backend default.

The emulator and CAMB comparisons assume the reference's ``Nncdm = 1``
massive-neutrino convention: one massive species when ``mnu > 0`` and massless
neutrinos otherwise. The degenerate three-species split is not supported here.

Parameters are passed as an eight-column ``theta`` array in the order of
:func:`jet.cosmology.cosmo_parameter_spec`::

    Omegab, Omegam, H0, ns, As, w0, wa, mnu

``Omegam`` is the baryon-plus-CDM density, *excluding* massive neutrinos, and
``As`` is the physical amplitude (not ``As * 1e9``). A single sample may be
passed as a one-dimensional array; outputs always keep the sample axis, so a
single sample returns shape ``(1, n_z)``.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, newton

from ._boltzmann import camb_sigma8
from .cosmology import (
    _COMOVING_A_MIN,
    _COMOVING_GRID_SIZE,
    _VEL_LIGHT_CM_S,
    Cosmology,
    cosmo_parameter_spec,
)
from .emulator.emulator import Emulator
from .emulator.pklin import Z_GRID, sigma8_of_pk
from .emulator.pklin import theta_spec as pklin_theta_spec

__all__ = [
    "EquivalentCosmology",
    "spectral_equivalence_de",
    "spectral_equivalence_as",
]

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

#: Logger for expected, deliberately silenced CAMB fallbacks.
_LOGGER = logging.getLogger(__name__)

#: Redshifts the bundled emulators are trained on.
_DEFAULT_Z = np.asarray(Z_GRID, dtype=float)

#: Warnings emitted when the internal sigma8 calculation falls back to CAMB.
#: The high-level facade silences exactly these; other warnings still escape.
_FALLBACK_WARNING = r"(Sigma8FromAs|spectral_equivalence_as).*using CAMB sigma8"

#: Relative distance mismatch above which the facade reports a warning.
_DISTANCE_ERROR_WARNING = 1e-6


@contextmanager
def _quiet_expected_fallbacks(enabled: bool):
    """Silence the CAMB-fallback ``UserWarning`` emitted by the sigma8 helpers.

    The high-level facade exists precisely to handle out-of-box ``w0/wa``:
    using CAMB for the target sigma8 is an expected part of that flow, not
    something the caller should have to filter. Unexpected warnings, including
    solver ``RuntimeWarning``s, still propagate.
    """
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_FALLBACK_WARNING, category=UserWarning)
        _LOGGER.debug("EquivalentCosmology: suppressing expected CAMB sigma8 fallback warnings")
        yield


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
    """Target total-matter :math:`\\sigma_R(z)` from one CLASS run."""
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
    return np.array([float(cosmo.sigma(radius, zz)) for zz in z])


def _sigma8_camb(row: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Target total-matter :math:`\\sigma_R(z)` from one CAMB run."""
    return camb_sigma8(row, R=SIGMA8_R, z=z, kmax=_BOLTZMANN_KMAX)


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
    provider,
) -> np.ndarray:
    """Evaluate the total-matter emulator :math:`\\sigma_R(z)` for rows."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if rows.shape[0] == 1 and z.size > 1:
        # One cosmology evaluated at several redshifts: predict once.
        Pk = provider.predict(rows, return_std=False)
        Pk = np.repeat(Pk, z.size, axis=0)
    else:
        Pk = provider.predict(rows, return_std=False)
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
    provider,
) -> np.ndarray:
    """Target sigma8 of the original model at every ``z``."""
    if sigma8_type == "emulator":
        return _sigma8_emulator(row[None, :], z, provider)
    return _target_sigma8_cached(
        sigma8_type,
        tuple(float(value) for value in row),
        tuple(float(value) for value in np.atleast_1d(z)),
    )


def _resolved_emulator(emulator: Emulator | None) -> Emulator:
    """Return the caller's Pcb emulator or load the bundled one."""
    if emulator is not None:
        if not isinstance(emulator, Emulator):
            raise TypeError(
                f"emulator must be a jet.emulator.Emulator, got {type(emulator).__name__}"
            )
        return emulator
    from .emulator.pklin import load_pklin_emulator

    return load_pklin_emulator()


def _resolved_ratio(ratio_emulator: Emulator | None) -> Emulator:
    """Return the caller's Pnn/Pcb ratio emulator or the bundled one."""
    if ratio_emulator is not None:
        if not isinstance(ratio_emulator, Emulator):
            raise TypeError(
                "ratio_emulator must be a jet.emulator.Emulator, got "
                f"{type(ratio_emulator).__name__}"
            )
        return ratio_emulator
    from .emulator.pkmm import load_ratio_emulator

    return load_ratio_emulator()


def _resolved_pkmm(
    emulator: Emulator | None,
    ratio_emulator: Emulator | None,
):
    """Return the total-matter P(k) provider."""
    from .emulator.pkmm import PkmmProvider, load_pkmm_provider

    if emulator is None and ratio_emulator is None:
        return load_pkmm_provider()
    return PkmmProvider(
        _resolved_emulator(emulator),
        _resolved_ratio(ratio_emulator),
    )


def _outside_bounds(rows: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Boolean mask of rows outside an array of ``(dim, 2)`` parameter bounds."""
    return np.any((rows < bounds[:, 0]) | (rows > bounds[:, 1]), axis=1)


def _warn_auxiliary_fallback(
    theta: np.ndarray,
    mask: np.ndarray,
    spec,
    z_flat: np.ndarray,
    n_z: int,
) -> None:
    """Warn that CAMB is supplying the auxiliary sigma8 for masked rows."""
    violations = spec.check_within_bounds(theta[mask], slack=0.0)
    original_rows = np.flatnonzero(mask)
    details = []
    for param_name, row, value in violations:
        flat_index = int(original_rows[row])
        low, high = spec.params[spec.index(param_name)].bounds  # type: ignore[misc]
        details.append(
            f"sample {flat_index // n_z}, z={z_flat[flat_index]:g}: "
            f"{param_name}={value:.6g} outside [{low:.6g}, {high:.6g}]"
        )
    warnings.warn(
        "spectral_equivalence_as: auxiliary parameters leave the emulator's "
        "training box; using CAMB sigma8:\n  " + "\n  ".join(details),
        UserWarning,
        stacklevel=3,
    )


def _emulator_spec(emulator: Emulator | None):
    """Return the bounds spec of the Pcb emulator without loading it."""
    if emulator is not None:
        return emulator.x_spec
    return pklin_theta_spec()


def _warn_target_fallback(theta: np.ndarray, mask: np.ndarray, spec) -> None:
    """Warn that CAMB is supplying the target sigma8 for masked samples."""
    violations = spec.check_within_bounds(theta[mask], slack=0.0)
    original_rows = np.flatnonzero(mask)
    details = []
    for param_name, row, value in violations:
        sample = int(original_rows[row])
        low, high = spec.params[spec.index(param_name)].bounds  # type: ignore[misc]
        details.append(f"sample {sample}: {param_name}={value:.6g} outside [{low:.6g}, {high:.6g}]")
    warnings.warn(
        "spectral_equivalence_as: target parameters leave the emulator's "
        "training box; using CAMB sigma8:\n  " + "\n  ".join(details),
        UserWarning,
        stacklevel=3,
    )


def _sigma8_class_camb_rows(
    rows: np.ndarray,
    z_rows: np.ndarray,
    sigma8_type: str,
) -> np.ndarray:
    """Evaluate CLASS/CAMB sigma8 for one ``(row, z)`` pair per call.

    The auxiliary model changes with redshift, so each flat row is a distinct
    cosmology and gets its own Boltzmann run. This is the expensive path; the
    emulator mode reuses one provider for the whole batch.
    """
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    z_rows = np.atleast_1d(np.asarray(z_rows, dtype=float))
    if rows.shape[0] != z_rows.size:
        raise ValueError(f"rows has {rows.shape[0]} entries but z has {z_rows.size}")
    function = _sigma8_class if sigma8_type == "class" else _sigma8_camb
    out = np.empty(z_rows.size, dtype=float)
    for index in range(z_rows.size):
        out[index] = float(function(rows[index], np.array([z_rows[index]]))[0])
    return out


def _sigma8_emulator_target(
    theta: np.ndarray,
    z: np.ndarray,
    fallback_to_camb: bool,
    emulator: Emulator | None,
    ratio_emulator: Emulator | None,
) -> np.ndarray:
    """Target total-matter sigma8 for every sample, with per-side fallback."""
    spec = _emulator_spec(emulator)
    outside = _outside_bounds(theta, spec.bounds_array())
    if np.any(outside) and not fallback_to_camb:
        spec.require_within_bounds(theta, name="target theta")

    out = np.empty((theta.shape[0], z.size), dtype=float)
    inside = ~outside
    if np.any(inside):
        provider = _resolved_pkmm(emulator, ratio_emulator)
        for index in np.flatnonzero(inside):
            out[index] = _sigma8_emulator(theta[index : index + 1], z, provider)
    if np.any(outside):
        _warn_target_fallback(theta, outside, spec)
        for index in np.flatnonzero(outside):
            out[index] = _sigma8_camb(theta[index], z)
    return out


def _sigma8_emulator_aux(
    theta: np.ndarray,
    z_flat: np.ndarray,
    fallback_to_camb: bool,
    emulator: Emulator | None,
    ratio_emulator: Emulator | None,
    n_z: int,
) -> np.ndarray:
    """Auxiliary total-matter sigma8 per flat row, with per-side fallback."""
    spec = _emulator_spec(emulator)
    outside = _outside_bounds(theta, spec.bounds_array())
    if np.any(outside) and not fallback_to_camb:
        spec.require_within_bounds(theta, name="auxiliary theta")

    out = np.empty(theta.shape[0], dtype=float)
    inside = ~outside
    if np.any(inside):
        provider = _resolved_pkmm(emulator, ratio_emulator)
        out[inside] = _sigma8_emulator(theta[inside], z_flat[inside], provider)
    if np.any(outside):
        _warn_auxiliary_fallback(theta, outside, spec, z_flat, n_z)
        for index in np.flatnonzero(outside):
            out[index] = _sigma8_camb(theta[index], np.array([z_flat[index]]))[0]
    return out


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
    sigma8_type: str = "emulator",
    fallback_to_camb: bool = True,
    emulator: Emulator | None = None,
    ratio_emulator: Emulator | None = None,
) -> np.ndarray:
    r"""Rescale ``As`` so the auxiliary model matches the target sigma8.

    The amplitude is scaled as

    .. math::

        A_{s,\mathrm{eq}}(z) = A_s
        \left(\frac{\sigma_8^{\mathrm{target}}(z)}
        {\sigma_8^{\mathrm{eq}}(z)}\right)^2,

    using the **total-matter** :math:`\sigma_8` (including massive neutrinos),
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
    sigma8_type : {"emulator", "class", "camb"}, optional
        Backend for the **entire** sigma8 calculation, i.e. both the target and
        the auxiliary model:

        ``"emulator"`` (default)
            Both sides use the bundled total-matter emulator. A side outside
            the emulator's parameter bounds falls back to CAMB when
            ``fallback_to_camb=True``; otherwise it raises ``ValueError``.
            The fallback is per side, so a target outside the box does not
            force an in-box auxiliary to CAMB (and vice versa).
        ``"class"``
            Both sides use CLASS's total-matter sigma8. No emulator bounds are
            involved and ``fallback_to_camb`` is ignored.
        ``"camb"``
            Both sides use CAMB's total-matter sigma8 (``delta_tot``). No
            emulator bounds are involved and ``fallback_to_camb`` is ignored.

        This replaces the previous mixed semantics where ``sigma8_type`` only
        selected the target backend.
    fallback_to_camb : bool, optional
        Whether an emulator-side sigma8 outside the box may use CAMB. Only
        affects ``sigma8_type="emulator"``. Defaults to true.
    emulator : jet.emulator.Emulator, optional
        :math:`P_{cb}(k)` emulator, the first factor of the total-matter
        composition. Only used by ``sigma8_type="emulator"``. ``None`` loads
        the bundled one.
    ratio_emulator : jet.emulator.Emulator, optional
        :math:`P_{nn}/P_{cb}` ratio emulator, the second factor. Only used by
        ``sigma8_type="emulator"``. ``None`` loads the bundled one.

    Notes
    -----
    The emulator path assumes the reference's ``Nncdm = 1`` massive-neutrino
    convention (one massive species when ``mnu > 0``, massless otherwise). The
    CLASS and CAMB helpers use the same convention. The degenerate
    three-species split is not supported by this comparison.

    Returns
    -------
    ndarray of shape (n, n_z)
        Equivalent physical ``As`` values.
    """
    theta_values = _validate_theta(theta)
    z_values = _validate_z(z)
    sigma8_type = _normalise_sigma8_type(sigma8_type)

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
    if sigma8_type == "emulator":
        sigma8_target = _sigma8_emulator_target(
            theta_values,
            z_values,
            fallback_to_camb,
            emulator,
            ratio_emulator,
        )
    else:
        sigma8_target = np.empty((n_samples, n_z), dtype=float)
        for index, row in enumerate(theta_values):
            sigma8_target[index] = _target_sigma8(row, z_values, sigma8_type, None)

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

    if sigma8_type == "emulator":
        sigma8_aux = _sigma8_emulator_aux(
            theta_eq_flat,
            z_flat,
            fallback_to_camb,
            emulator,
            ratio_emulator,
            n_z,
        )
    else:
        sigma8_aux = _sigma8_class_camb_rows(theta_eq_flat, z_flat, sigma8_type)

    sigma8_aux = sigma8_aux.reshape(n_samples, n_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        amplitude = np.power(sigma8_target / sigma8_aux, 2)
    return theta_values[:, 4, None] * amplitude


class EquivalentCosmology:
    r"""A single cosmology evaluated through its spectral-equivalent model.

    This is the high-level entry point for the common case: an input cosmology
    whose dark-energy parameters ``w0, wa`` lie outside the bundled emulators'
    box. It converts :math:`\sigma_8` to :math:`A_s` when requested, constructs
    the per-redshift auxiliary ``w0waCDM`` model with
    :func:`spectral_equivalence_de` and :func:`spectral_equivalence_as`, and
    evaluates the bundled halo statistics on that auxiliary model.

    Parameters
    ----------
    theta : array-like of shape (8,) or (1, 8)
        Cosmological parameters in :data:`PARAMETER_NAMES` order. One instance
        represents one cosmology; for a batch, create one instance per row.
    use_sigma8 : bool, optional
        If true (the default), the fifth column is :math:`\sigma_8` at
        ``z = 0`` and is converted to :math:`A_s` internally. If false, the
        fifth column is already the physical :math:`A_s` (not ``As * 1e9``).
    fallback_to_camb : bool, optional
        Policy for the **internal** sigma8 calculation. The default ``True``
        lets :class:`~jet.derived.Sigma8FromAs` and
        :func:`spectral_equivalence_as` use CAMB when the target or auxiliary
        model leaves the emulator box. This is what makes an out-of-box
        ``w0/wa`` input usable. If false, those helpers raise
        :class:`ValueError` instead. Downstream :meth:`number_density`,
        :meth:`bias_mass` and :meth:`xihm_mass` are always strict: they never
        fall back to CAMB.
    sigma8_type : {"emulator", "class", "camb"}, optional
        Backend passed to :func:`spectral_equivalence_as`. ``"emulator"``
        (the default) uses the total-matter emulator with the per-side CAMB
        fallback controlled by ``fallback_to_camb``. ``"class"`` and
        ``"camb"`` use that Boltzmann code for both the target and the
        auxiliary sigma8, so ``fallback_to_camb`` is ignored.
    zstar : float, optional
        Redshift of the last-scattering surface used by the distance match.
    neutrino_matter_like : bool, optional
        Treat massive neutrinos as matter in the distance integral; see
        :meth:`jet.cosmology.Cosmology.comoving_distance`.

    Notes
    -----
    The input may be out of the emulator box only in ``w0/wa``. The other five
    parameters and the input amplitude must be inside the box; spectral
    equivalence cannot repair them, so those cases raise :class:`ValueError`
    immediately.

    The auxiliary model is constructed at each requested redshift. Its
    ``w0/wa`` are inside the box by construction, but its ``A_s`` can leave the
    box when the requested amplitude is near the edge of what the emulator can
    produce. :meth:`theta` still returns that auxiliary theta so it can be used
    with ``HMFEmulator.load(fallback_to_camb=True)``; the direct statistics
    methods raise a :class:`ValueError` in that case, because BHM and XiHM have
    no fallback.

    Expected CAMB-fallback ``UserWarning``s from the internal sigma8 helpers
    are suppressed. Solver ``RuntimeWarning``s and the downstream emulators'
    own warnings are not.

    Examples
    --------
    >>> import numpy as np
    >>> cosmo = EquivalentCosmology(
    ...     [0.055, 0.35140742, 63.6, 0.964, 0.781, -0.42, -1.75, 0.06]
    ... )
    >>> theta_eq = cosmo.theta_at(0.0)
    >>> theta_eq.shape
    (8,)
    >>> n = cosmo.number_density(z=[0.0, 0.5, 1.0], M=np.array([1e13, 1e14]))
    >>> n.shape
    (3, 2)
    """

    def __init__(
        self,
        theta,
        *,
        use_sigma8: bool = True,
        fallback_to_camb: bool = True,
        sigma8_type: str = "emulator",
        zstar: float = ZSTAR,
        neutrino_matter_like: bool = False,
    ) -> None:
        values = np.asarray(theta, dtype=float)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 1 or values.size != len(PARAMETER_NAMES):
            raise ValueError(
                "theta must have shape (8,) or (1, 8) in the order "
                f"{', '.join(PARAMETER_NAMES)}; got {np.shape(theta)}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("theta must be finite")

        self.use_sigma8 = bool(use_sigma8)
        self.fallback_to_camb = bool(fallback_to_camb)
        self.sigma8_type = _normalise_sigma8_type(sigma8_type)
        self.zstar = float(zstar)
        self.neutrino_matter_like = bool(neutrino_matter_like)

        self._input = values.copy()
        self._validate_non_de(self._input)
        if self.use_sigma8 and self._input[4] <= 0.0:
            raise ValueError(f"sigma8 must be positive, got {self._input[4]!r}")
        self._theta_as = self._to_as(self._input)
        self._validate_amplitude(self._theta_as)
        self._needs_equivalence = not (
            W0_BOUNDS[0] <= self._theta_as[5] <= W0_BOUNDS[1]
            and WA_BOUNDS[0] <= self._theta_as[6] <= WA_BOUNDS[1]
        )
        self._aux_cache: dict[float, np.ndarray] = {}
        self._hmf_emulator = None
        self._bhm_emulator = None
        self._xihm_emulator = None

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_non_de(theta: np.ndarray) -> None:
        """Reject parameters spectral equivalence cannot map back into the box."""
        spec = cosmo_parameter_spec()
        violations = [
            (name, value)
            for name, _row, value in spec.check_within_bounds(theta[None, :])
            if name not in {"As", "w0", "wa"}
        ]
        if not violations:
            return
        details = []
        for name, value in violations:
            low, high = spec.params[spec.index(name)].bounds  # type: ignore[misc]
            details.append(f"{name}={value:.6g} outside [{low:.6g}, {high:.6g}]")
        raise ValueError(
            "spectral equivalence can only map w0/wa and the amplitude; "
            "these input parameters are outside the emulator box: " + ", ".join(details)
        )

    @staticmethod
    def _validate_amplitude(theta: np.ndarray) -> None:
        """Require the input (already As-based) amplitude to stay in the box."""
        spec = cosmo_parameter_spec()
        low, high = spec.params[spec.index("As")].bounds  # type: ignore[misc]
        As = float(theta[4])
        if not np.isfinite(As) or As <= 0.0:
            raise ValueError(f"As must be finite and positive, got {As!r}")
        if not low <= As <= high:
            raise ValueError(
                f"the input amplitude corresponds to As={As:.6g}, outside the "
                f"emulator box [{low:.6g}, {high:.6g}]; spectral equivalence can "
                "only remap w0/wa and cannot bring the amplitude into range"
            )

    def _to_as(self, theta: np.ndarray) -> np.ndarray:
        """Return ``theta`` with the fifth column expressed as physical As."""
        if not self.use_sigma8:
            return theta.copy()

        from .derived import Sigma8FromAs
        from .spec import ParameterSpec

        conversion = Sigma8FromAs(fallback_to_camb=self.fallback_to_camb)
        spec = ParameterSpec(list(cosmo_parameter_spec()), derived=[conversion])
        frame = spec.frame("sigma8")
        try:
            with _quiet_expected_fallbacks(self.fallback_to_camb):
                return frame.to_base(theta[None, :])[0]
        except ValueError as exc:
            raise ValueError(
                f"could not convert sigma8 to As: {exc}\n"
                "Set fallback_to_camb=True to evaluate an out-of-box target "
                "sigma8 with CAMB."
            ) from None

    # ------------------------------------------------------------------
    # Per-redshift auxiliary model
    # ------------------------------------------------------------------
    @staticmethod
    def _z_values(z) -> np.ndarray:
        """Return validated redshifts, defaulting to the emulator grid."""
        if z is None:
            return _DEFAULT_Z.copy()
        values = np.atleast_1d(np.asarray(z, dtype=float))
        if values.ndim != 1:
            raise ValueError(f"z must be a scalar or one-dimensional, got shape {values.shape}")
        if values.size == 0:
            raise ValueError("z must contain at least one redshift")
        if not np.all(np.isfinite(values)):
            raise ValueError("z must be finite")
        low, high = float(_DEFAULT_Z[0]), float(_DEFAULT_Z[-1])
        if np.any(values < low) or np.any(values > high):
            raise ValueError(
                f"z must lie within the emulator grid [{low:g}, {high:g}], "
                f"got [{values.min():g}, {values.max():g}]"
            )
        return values

    def _compute_rows(self, z_values: np.ndarray) -> None:
        """Compute and cache the auxiliary theta at every missing redshift."""
        if not self._needs_equivalence:
            for zz in z_values:
                self._aux_cache[float(zz)] = self._theta_as.copy()
            return

        with _quiet_expected_fallbacks(self.fallback_to_camb):
            w0, wa, err = spectral_equivalence_de(
                self._theta_as,
                z_values,
                wcdm=False,
                return_error=True,
                zstar=self.zstar,
                neutrino_matter_like=self.neutrino_matter_like,
            )
            As_eq = spectral_equivalence_as(
                self._theta_as,
                z_values,
                w0,
                wa,
                sigma8_type=self.sigma8_type,
                fallback_to_camb=self.fallback_to_camb,
            )

        for index, zz in enumerate(z_values):
            row = self._theta_as.copy()
            row[4] = float(As_eq[0, index])
            row[5] = float(w0[0, index])
            row[6] = float(wa[0, index])
            if not np.all(np.isfinite(row)):
                raise RuntimeError(
                    f"spectral equivalence produced a non-finite auxiliary theta at z={zz:g}"
                )
            if row[4] <= 0.0:
                raise RuntimeError(
                    f"spectral equivalence produced a non-positive As_eq={row[4]!r} at z={zz:g}"
                )
            if not (W0_BOUNDS[0] <= row[5] <= W0_BOUNDS[1]) or not (
                WA_BOUNDS[0] <= row[6] <= WA_BOUNDS[1]
            ):
                raise RuntimeError(
                    "spectral equivalence produced auxiliary w0/wa outside the "
                    f"emulator box at z={zz:g}: w0={row[5]:.6g}, wa={row[6]:.6g}"
                )

            error = float(err[0, index])
            if not np.isfinite(error):
                raise RuntimeError(
                    f"spectral equivalence returned a non-finite distance error at z={zz:g}"
                )
            if error > _DISTANCE_ERROR_WARNING:
                warnings.warn(
                    f"spectral equivalence distance mismatch at z={zz:g}: "
                    f"relative error {error:.3g}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            self._aux_cache[float(zz)] = row

    def _theta_rows(self, z_values: np.ndarray) -> np.ndarray:
        """Return one auxiliary theta row per requested redshift."""
        missing = [float(zz) for zz in np.unique(z_values) if float(zz) not in self._aux_cache]
        if missing:
            self._compute_rows(np.asarray(missing, dtype=float))
        return np.stack([self._aux_cache[float(zz)] for zz in z_values], axis=0)

    def _auxiliary_rows(self, z_values: np.ndarray, method: str) -> np.ndarray:
        """Return auxiliary rows, refusing an out-of-box downstream call."""
        rows = self._theta_rows(z_values)
        spec = cosmo_parameter_spec()
        violations = spec.check_within_bounds(rows)
        if not violations:
            return rows

        details = []
        for name, row, value in violations:
            low, high = spec.params[spec.index(name)].bounds  # type: ignore[misc]
            details.append(
                f"z={z_values[row]:g}: {name}={value:.6g} outside [{low:.6g}, {high:.6g}]"
            )
        raise ValueError(
            f"{method}: auxiliary theta is outside the emulator box:\n  "
            + "\n  ".join(details)
            + "\nUse .theta(z) to inspect it; HMF can be evaluated with "
            "HMFEmulator.load(fallback_to_camb=True), while BHM and XiHM have "
            "no CAMB fallback."
        )

    # ------------------------------------------------------------------
    # Public theta accessors
    # ------------------------------------------------------------------
    @property
    def input_theta(self) -> np.ndarray:
        """The theta exactly as supplied, in its original amplitude base."""
        return self._input.copy()

    @property
    def theta_as(self) -> np.ndarray:
        """The input theta with the fifth column expressed as physical As."""
        return self._theta_as.copy()

    @property
    def needs_equivalence(self) -> bool:
        """Whether the input ``w0/wa`` lie outside the emulator box."""
        return self._needs_equivalence

    def theta(self, z=None) -> np.ndarray:
        """Return the auxiliary theta at each redshift, shape ``(n_z, 8)``.

        The fifth column is physical ``As``. The auxiliary ``w0/wa`` are inside
        the box; ``As`` may be outside it when the target amplitude is near the
        edge of what the emulator can produce.
        """
        return self._theta_rows(self._z_values(z))

    def theta_at(self, z: float) -> np.ndarray:
        """Return the auxiliary theta at one redshift, shape ``(8,)``."""
        z_values = self._z_values(z)
        if z_values.size != 1:
            raise ValueError(f"theta_at expects a scalar z, got {z_values.size} values")
        return self._theta_rows(z_values)[0]

    # ------------------------------------------------------------------
    # Lazy emulators
    # ------------------------------------------------------------------
    def _get_hmf(self):
        if self._hmf_emulator is None:
            from .emulator.hmf import HMFEmulator

            self._hmf_emulator = HMFEmulator.load()
        return self._hmf_emulator

    def _get_bhm(self):
        if self._bhm_emulator is None:
            from .emulator.bhm import BhmEmulator

            self._bhm_emulator = BhmEmulator.load()
        return self._bhm_emulator

    def _get_xihm(self):
        if self._xihm_emulator is None:
            from .emulator.xihm import XiHMEmulator

            self._xihm_emulator = XiHMEmulator.load()
        return self._xihm_emulator

    # ------------------------------------------------------------------
    # Direct statistics
    # ------------------------------------------------------------------
    def number_density(self, z=None, M=None) -> np.ndarray:
        r"""Cumulative halo number density, shape ``(n_z, n_M)``.

        Units are :math:`(h/\mathrm{Mpc})^3`. ``M`` defaults to the emulator's
        fine mass grid. The auxiliary theta is required to be entirely inside
        the emulator box.
        """
        z_values = self._z_values(z)
        rows = self._auxiliary_rows(z_values, "number_density")
        hmf = self._get_hmf()
        parts = [
            hmf.number_density(row[None, :], z=[float(zz)], M=M)[0, 0]
            for row, zz in zip(rows, z_values, strict=True)
        ]
        return np.stack(parts, axis=0)

    def dndlnM(self, z=None, M=None) -> np.ndarray:
        r"""Differential halo mass function, shape ``(n_z, n_M)``.

        Units are :math:`(h/\mathrm{Mpc})^3` per :math:`d\ln M`.
        """
        z_values = self._z_values(z)
        rows = self._auxiliary_rows(z_values, "dndlnM")
        hmf = self._get_hmf()
        parts = [
            hmf.dndlnM(row[None, :], z=[float(zz)], M=M)[0, 0]
            for row, zz in zip(rows, z_values, strict=True)
        ]
        return np.stack(parts, axis=0)

    def bias_mass(self, z=None, M=None) -> np.ndarray:
        """Halo bias at a fixed mass threshold, shape ``(n_z, n_M)``."""
        if M is None:
            raise ValueError("bias_mass requires a mass threshold M")
        z_values = self._z_values(z)
        rows = self._auxiliary_rows(z_values, "bias_mass")
        bhm = self._get_bhm()
        parts = [
            bhm.bias_mass(row[None, :], z=[float(zz)], M=M)[0, 0]
            for row, zz in zip(rows, z_values, strict=True)
        ]
        return np.stack(parts, axis=0)

    def bias_lgnbar_threshold(self, z=None, lgden=None) -> np.ndarray:
        """Halo bias at a fixed density threshold, shape ``(n_z, n_lgden)``."""
        if lgden is None:
            raise ValueError("bias_lgnbar_threshold requires lgden")
        z_values = self._z_values(z)
        rows = self._auxiliary_rows(z_values, "bias_lgnbar_threshold")
        bhm = self._get_bhm()
        parts = [
            bhm.bias_lgnbar_threshold(row[None, :], z=[float(zz)], lgden=lgden)[0, 0]
            for row, zz in zip(rows, z_values, strict=True)
        ]
        return np.stack(parts, axis=0)

    def xihm_mass(self, z=None, r=None, M=None) -> np.ndarray:
        """Halo-matter correlation at a mass threshold, shape ``(n_z, n_M, n_r)``."""
        if r is None:
            raise ValueError("xihm_mass requires separations r")
        if M is None:
            raise ValueError("xihm_mass requires a mass threshold M")
        z_values = self._z_values(z)
        rows = self._auxiliary_rows(z_values, "xihm_mass")
        xihm = self._get_xihm()
        parts = []
        for row, zz in zip(rows, z_values, strict=True):
            value = xihm.xihm_mass(row[None, :], z=[float(zz)], r=r, M=M)
            # Native shape is (sample, n_M, n_z, n_r); keep z first.
            parts.append(value[0, :, 0, :])
        return np.stack(parts, axis=0)

    def xihm_lgnbar_threshold(self, z=None, r=None, lgden=None) -> np.ndarray:
        """Halo-matter correlation at a density threshold, ``(n_z, n_lgden, n_r)``."""
        if r is None:
            raise ValueError("xihm_lgnbar_threshold requires separations r")
        if lgden is None:
            raise ValueError("xihm_lgnbar_threshold requires lgden")
        z_values = self._z_values(z)
        rows = self._auxiliary_rows(z_values, "xihm_lgnbar_threshold")
        xihm = self._get_xihm()
        parts = []
        for row, zz in zip(rows, z_values, strict=True):
            value = xihm.xihm_lgnbar_threshold(row[None, :], z=[float(zz)], r=r, lgden=lgden)
            # Native shape is (sample, n_lgden, n_z, n_r); keep z first.
            parts.append(value[0, :, 0, :])
        return np.stack(parts, axis=0)

    def ximm_linear(self, z=None, r=None) -> np.ndarray:
        r"""Linear CDM+baryon correlation function, shape ``(n_z, n_r)``.

        This uses the same :math:`P_{cb}` definition as
        :func:`jet.emulator.xihm.ximm_linear`; it is not the total-matter
        correlation function including massive neutrinos. The auxiliary theta
        is evaluated at each redshift.
        """
        if r is None:
            raise ValueError("ximm_linear requires separations r")
        z_values = self._z_values(z)
        rows = self._auxiliary_rows(z_values, "ximm_linear")
        xihm = self._get_xihm()
        parts = [
            xihm.ximm_linear(row[None, :], z=[float(zz)], r=r)[0, 0]
            for row, zz in zip(rows, z_values, strict=True)
        ]
        return np.stack(parts, axis=0)

    def __repr__(self) -> str:
        return (
            f"EquivalentCosmology(use_sigma8={self.use_sigma8}, "
            f"w0={self._theta_as[5]:.6g}, wa={self._theta_as[6]:.6g}, "
            f"needs_equivalence={self._needs_equivalence})"
        )
