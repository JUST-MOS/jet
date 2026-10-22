r"""Total-matter linear power spectrum from :math:`P_{cb}` and neutrinos.

The bundled :mod:`jet.emulator.pklin` model predicts the cold-dark-matter plus
baryon spectrum :math:`P_{cb}`. That is the correct input for the halo mass
function, but :math:`\sigma_8` is conventionally defined from the **total**
matter spectrum, including massive neutrinos.

This module composes the two without training a third emulator. It uses the
reference's linear-theory construction,

.. math::

    P_{mm} = P_{cb}
    \left(f_{cb} + f_\nu \sqrt{P_{nn}/P_{cb}}\right)^2,
    \qquad f_{cb} = \frac{\Omega_{cb}}{\Omega_M},
    \qquad f_\nu = \frac{\Omega_\nu}{\Omega_M},

with :math:`P_{nn}/P_{cb}` read from a second bundled GP (:data:`BUNDLE_NAME`).
For ``mnu = 0`` the ratio is not needed and :math:`P_{mm} = P_{cb}` exactly.

The halo mass function and the bias/correlation-function models continue to use
:math:`P_{cb}`; only ``Sigma8FromAs`` and ``spectral_equivalence_as`` consume
the total-matter spectrum.

Both the Pcb emulator and the ratio emulator assume the reference's
``Nncdm = 1`` massive-neutrino convention: one massive species when ``mnu > 0``,
massless neutrinos otherwise. The ratio bundle has no ``mnu = 0`` training row,
and ``mnu = 0`` is handled analytically by returning Pcb. The degenerate
three-species split is not supported by this composition.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..spec import DataVectorSpec
from .emulator import Emulator, resolve_weights_path
from .pklin import (
    N_BINS,
    data_dir,
    load_pklin_emulator,
    sigma8_of_pk,
    theta_spec,
)

__all__ = [
    "BUNDLE_NAME",
    "ratio_data_vector_spec",
    "ratio_theta_spec",
    "load_ratio_emulator",
    "ratio_at",
    "total_pk",
    "PkmmProvider",
    "load_pkmm_provider",
    "sigma8_total",
]

#: Name of the Pnn/Pcb ratio bundle under the data directory.
BUNDLE_NAME = "pknn_ratio.gp.npz"

#: Converts a neutrino mass sum in eV into :math:`\Omega_\nu h^2`.
MNU_TO_OMEGA_H2 = 1.0 / 93.14


def _data_path(name: str):
    """Resolve one bundled data file, with rebuild instructions on failure."""
    try:
        return resolve_weights_path(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"the bundled data file {name!r} is missing from {data_dir()}. It is not "
            "tracked by git; regenerate it with tools/build_pknn_ratio_bundle.py, or "
            "set JET_DATA_DIR to a directory that already holds it."
        ) from exc


def ratio_theta_spec():
    """Input specification of the ratio emulator, shared with the P(k) model."""
    return theta_spec()


def ratio_data_vector_spec() -> DataVectorSpec:
    """Output specification: ``log10(P_nn / P_cb)`` on the P(k) grid."""
    return DataVectorSpec(name="pknn_cb_ratio", n_bins=N_BINS, log=True)


@lru_cache(maxsize=1)
def load_ratio_emulator() -> Emulator:
    """Load the bundled :math:`P_{nn}/P_{cb}` emulator."""
    emulator = Emulator.load(_data_path(BUNDLE_NAME))
    emulator.warn_on_extrapolation = False
    return emulator


def ratio_at(theta: np.ndarray) -> np.ndarray:
    r"""Return :math:`P_{nn}/P_{cb}` on the flat ``12 x 1000`` grid."""
    values = np.atleast_2d(np.asarray(theta, dtype=float))
    return np.asarray(load_ratio_emulator().predict(values, return_std=False))


def total_pk(
    theta: np.ndarray,
    pcb: np.ndarray | None = None,
    ratio: np.ndarray | None = None,
) -> np.ndarray:
    r"""Compose the total-matter linear spectrum.

    Parameters
    ----------
    theta : ndarray of shape (n, 8)
        Cosmological parameters in ``theta_spec`` order. ``Omegam`` is the
        CDM+baryon density and ``mnu`` the neutrino mass sum in eV.
    pcb : ndarray of shape (n, N_BINS), optional
        CDM+baryon spectrum. Defaults to the bundled P(k) emulator.
    ratio : ndarray of shape (n_mnu, N_BINS), optional
        :math:`P_{nn}/P_{cb}` for the rows with ``mnu > 0``. Defaults to the
        bundled ratio emulator.

    Returns
    -------
    ndarray of shape (n, N_BINS)
        Total-matter linear spectrum in :math:`(\mathrm{Mpc}/h)^3`.
    """
    values = np.atleast_2d(np.asarray(theta, dtype=float))
    if pcb is None:
        pcb = load_pklin_emulator().predict(values, return_std=False)
    pcb = np.atleast_2d(np.asarray(pcb, dtype=float))
    if pcb.shape[0] != values.shape[0]:
        raise ValueError(f"pcb has {pcb.shape[0]} rows but theta has {values.shape[0]}")

    out = pcb.copy()
    massive = values[:, 7] > 0.0
    if not np.any(massive):
        return out

    if ratio is None:
        ratio = ratio_at(values[massive])
    ratio = np.atleast_2d(np.asarray(ratio, dtype=float))
    if ratio.shape[0] != int(np.count_nonzero(massive)):
        raise ValueError(
            f"ratio has {ratio.shape[0]} rows but {int(np.count_nonzero(massive))} "
            "rows have mnu > 0"
        )

    h0 = values[massive, 2] / 100.0
    omeganu = values[massive, 7] * MNU_TO_OMEGA_H2 / h0**2
    omegam = values[massive, 1]
    omega_m = omegam + omeganu
    f_cb = (omegam / omega_m)[:, None]
    f_nu = (omeganu / omega_m)[:, None]
    out[massive] = pcb[massive] * (f_cb + f_nu * np.sqrt(ratio)) ** 2
    return out


class PkmmProvider:
    """A drop-in P(k) provider returning the total-matter spectrum.

    The interface matches :class:`jet.emulator.emulator.Emulator` closely
    enough for :func:`jet.emulator.pklin.sigma8_of_pk`: ``predict`` returns
    the flat data vector in the same layout as the Pcb bundle, and ``x_spec``
    exposes the parameter bounds. It is consumed by ``Sigma8FromAs`` and the
    sigma8 path of ``spectral_equivalence_as``; the halo mass function and the
    bias/correlation models keep using the Pcb emulator directly.
    """

    def __init__(self, pcb_emulator: Emulator, ratio_emulator: Emulator) -> None:
        self.pcb_emulator = pcb_emulator
        self.ratio_emulator = ratio_emulator

    @property
    def x_spec(self):
        """Input specification of the underlying cosmological model."""
        return self.pcb_emulator.x_spec

    @property
    def y_spec(self):
        """Output specification of the shared Pcb/Pmm data-vector layout."""
        return self.pcb_emulator.y_spec

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Return total-matter ``P(k, z)`` for every row of ``X``."""
        if return_std:
            raise ValueError("PkmmProvider does not report a predictive uncertainty")
        values = np.atleast_2d(np.asarray(X, dtype=float))
        pcb = np.asarray(self.pcb_emulator.predict(values, return_std=False))
        massive = values[:, 7] > 0.0
        ratio = None
        if np.any(massive):
            ratio = np.asarray(self.ratio_emulator.predict(values[massive], return_std=False))
        return total_pk(values, pcb=pcb, ratio=ratio)


def load_pkmm_provider() -> PkmmProvider:
    """Load the bundled Pcb and Pnn/Pcb emulators and compose them."""
    return PkmmProvider(load_pklin_emulator(), load_ratio_emulator())


def sigma8_total(
    theta: np.ndarray,
    R: float = 8.0,
    z: float = 0.0,
) -> np.ndarray:
    r"""Return the total-matter :math:`\sigma_R(z)` for ``theta`` rows."""
    provider = load_pkmm_provider()
    return sigma8_of_pk(provider.predict(theta), R=R, z=z)
