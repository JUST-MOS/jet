r"""Optional CAMB helpers used by out-of-range fallbacks.

The core package never imports :mod:`camb`; every function here imports it
lazily and raises an actionable :class:`ImportError` when it is missing.

The two spectra are deliberately different:

* :func:`camb_matter_power` supplies the **CDM+baryon** spectrum
  (``delta_nonu``) used by the halo mass function's Castro23 fallback;
* :func:`camb_sigma8` supplies the **total-matter** spectrum (``delta_tot``),
  because :math:`\sigma_8` is defined including massive neutrinos.

Both use ``k`` in :math:`h/\mathrm{Mpc}` and ``P(k)`` in
:math:`(\mathrm{Mpc}/h)^3`, with the same massive-neutrino and ``w0wa``
treatment as the reference.
"""

from __future__ import annotations

import numpy as np

__all__ = ["camb_matter_power", "camb_sigma8", "CambPkProvider"]

#: Maximum wavenumber used for sigma8 evaluations, in 1/Mpc.
_SIGMA8_KMAX = 10.0

#: Maximum wavenumber used for halo-mass-function fallbacks, in 1/Mpc. The
#: Castro23 baseline integrates the top-hat window up to the emulator's k grid
#: (~98 h/Mpc); 100/Mpc covers that for every h in the training box.
_HMF_KMAX = 100.0

#: Effective number of relativistic species used by the reference.
_NEFF = 3.046


def _import_camb():
    """Import :mod:`camb`, or explain how to install it."""
    try:
        import camb
    except ImportError:
        raise ImportError(
            "fallback_to_camb=True requires the 'camb' package; install it "
            "with 'pip install jet[camb]' or 'pip install camb'"
        ) from None
    return camb


def _camb_params(theta_row: np.ndarray, kmax: float, redshifts: np.ndarray):
    """Build the CAMB parameters shared by the fallback helpers."""
    camb = _import_camb()
    Omegab, Omegam, H0, ns, As, w0, wa, mnu = (float(value) for value in theta_row)
    h0 = H0 / 100.0

    params = camb.CAMBparams()
    params.set_cosmology(
        H0=H0,
        omch2=(Omegam - Omegab) * h0 * h0,
        ombh2=Omegab * h0 * h0,
        num_massive_neutrinos=0 if mnu == 0.0 else 1,
        standard_neutrino_neff=_NEFF,
        mnu=mnu,
    )
    params.InitPower.set_params(ns=ns, As=As)
    params.set_dark_energy(w=w0, wa=wa, dark_energy_model="ppf")
    params.set_matter_power(kmax=kmax, redshifts=np.atleast_1d(redshifts))
    # Only the matter transfer is needed. Disabling the CMB spectra also skips
    # CAMB's lensing sanity check, which rejects the large A_s values that a
    # high-sigma8 inversion can visit before it converges.
    params.WantCls = False
    return params


def _transfer_indices(results, z: np.ndarray) -> list[int]:
    """Return the CAMB storage index of every requested redshift."""
    transfer_z = np.asarray(results.transfer_redshifts, dtype=float)
    indices = []
    for zz in np.atleast_1d(z):
        matches = np.flatnonzero(np.isclose(transfer_z, zz))
        if matches.size == 0:
            raise RuntimeError(f"CAMB did not return results at z={zz}")
        indices.append(int(matches[0]))
    return indices


def camb_sigma8(
    theta_row: np.ndarray,
    R: float = 8.0,
    z: float | np.ndarray = 0.0,
    kmax: float = _SIGMA8_KMAX,
) -> np.ndarray:
    r"""Return the total-matter :math:`\sigma_R(z)` from a linear CAMB run.

    This uses ``delta_tot``: :math:`\sigma_8` is conventionally defined from
    the total matter spectrum, including massive neutrinos. The halo mass
    function's CAMB fallback is separate and keeps ``delta_nonu``.

    Parameters
    ----------
    theta_row : ndarray of shape (8,)
        Cosmological parameters in ``jet`` order.
    R : float, optional
        Smoothing radius in :math:`\mathrm{Mpc}/h`.
    z : float or array-like, optional
        Redshift(s).
    kmax : float, optional
        Maximum wavenumber in :math:`1/\mathrm{Mpc}`.

    Returns
    -------
    ndarray
        :math:`\sigma_{R,\mathrm{cb}}` at every requested redshift.
    """
    camb = _import_camb()
    z_values = np.atleast_1d(np.asarray(z, dtype=float))
    params = _camb_params(theta_row, kmax, z_values)
    results = camb.get_results(params)
    indices = _transfer_indices(results, z_values)

    values = []
    for index in indices:
        sigma = results.get_sigmaR(
            R=R,
            z_indices=index,
            hubble_units=True,
            var1="delta_tot",
            var2="delta_tot",
        )
        values.append(float(np.asarray(sigma).reshape(-1)[0]))
    return np.array(values)


def camb_matter_power(
    theta_row: np.ndarray,
    k: np.ndarray,
    z: np.ndarray,
    kmax: float = _HMF_KMAX,
) -> np.ndarray:
    r"""Return linear CDM+baryon :math:`P(k,z)` from CAMB.

    Parameters
    ----------
    theta_row : ndarray of shape (8,)
        Cosmological parameters in ``jet`` order.
    k : ndarray
        Wavenumbers in :math:`h/\mathrm{Mpc}`.
    z : ndarray
        Redshifts.
    kmax : float, optional
        Maximum wavenumber in :math:`1/\mathrm{Mpc}`.

    Returns
    -------
    ndarray of shape (n_z, n_k)
        :math:`P(k,z)` in :math:`(\mathrm{Mpc}/h)^3`.
    """
    camb = _import_camb()
    k_values = np.atleast_1d(np.asarray(k, dtype=float))
    z_values = np.atleast_1d(np.asarray(z, dtype=float))
    if np.any(k_values <= 0.0):
        raise ValueError(f"k must be positive, got {k_values.min()}")

    params = _camb_params(theta_row, kmax, z_values)
    results = camb.get_results(params)

    # Sample CAMB's own linear spectrum over the requested k range and
    # interpolate in log-log. The returned kh is in h/Mpc and PK in
    # (Mpc/h)^3, matching the emulator's convention.
    minkh = float(k_values.min())
    maxkh = float(k_values.max())
    kh, z_out, pk = results.get_matter_power_spectrum(
        minkh=minkh,
        maxkh=maxkh,
        npoints=max(1000, k_values.size),
        var1="delta_nonu",
        var2="delta_nonu",
    )
    kh = np.asarray(kh, dtype=float)
    z_out = np.asarray(z_out, dtype=float)
    pk = np.atleast_2d(np.asarray(pk, dtype=float))

    out = np.empty((z_values.size, k_values.size), dtype=float)
    log_k = np.log(k_values)
    for index, redshift in enumerate(z_values):
        matches = np.flatnonzero(np.isclose(z_out, redshift))
        if matches.size == 0:
            raise RuntimeError(f"CAMB did not return P(k) at z={redshift}")
        row = pk[int(matches[0])]
        out[index] = np.exp(np.interp(log_k, np.log(kh), np.log(row)))
    return out


class CambPkProvider:
    """A CDM+baryon P(k) provider for the halo-mass-function fallback.

    ``castro23_*`` and the other analytic baselines only need an object with a
    ``predict(X, return_std=False)`` method returning the flattened linear
    spectrum on the emulator's own ``12 x 1000`` grid. This provider supplies
    that layout from CAMB using ``delta_nonu``, so the rest of the baseline
    code is unchanged. Sigma8 fallbacks use ``delta_tot`` instead; see
    :func:`camb_sigma8`.
    """

    def __init__(self, kmax: float = _HMF_KMAX) -> None:
        self.kmax = float(kmax)

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Return CAMB ``P(k, z)`` for every row of ``X`` on the emulator grid."""
        from .emulator.pklin import k_grid, z_grid

        values = np.atleast_2d(np.asarray(X, dtype=float))
        k = k_grid()
        z = np.asarray(z_grid(), dtype=float)
        out = np.empty((values.shape[0], z.size * k.size), dtype=float)
        for row, theta_row in enumerate(values):
            spectrum = camb_matter_power(theta_row, k, z, kmax=self.kmax)
            out[row] = spectrum.reshape(-1)
        return out
