"""
Catalog containers and the HOD plugin interface.

**Design stage only** -- see the package docstring for what is decided and what
is still open. Every method here raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..spec import ParameterSpec

__all__ = ["HaloCatalog", "GalaxyCatalog", "HODModel"]


@dataclass
class HaloCatalog:
    """Halo positions, velocities and masses.

    The single entry point of :mod:`jet.catalog`. Particle snapshots are
    deliberately *not* accepted directly; ``from_particles`` is reserved for a
    future halo finder and is not implemented.

    Attributes
    ----------
    position : ndarray of shape (n_halos, 3)
        Comoving coordinates in Mpc/h.
    velocity : ndarray of shape (n_halos, 3)
        Peculiar velocities in km/s.
    mass : ndarray of shape (n_halos,)
        Halo mass in Msun/h, in the definition named by ``mass_def``.
    mass_def : str
        Mass definition the masses are quoted in, e.g. ``"M200m"`` or
        ``"Mvir"``. Explicit because HOD parameters are only meaningful
        relative to a definition.
    box_size : float or None
        Side length of the periodic box in Mpc/h, or ``None`` for a lightcone.
    concentration : ndarray of shape (n_halos,), optional
        NFW concentration, needed by profile-based HOD variants.
    velocity_dispersion : ndarray of shape (n_halos,), optional
        One-dimensional velocity dispersion in km/s, used by velocity-bias
        satellite models.
    """

    position: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    mass_def: str = "M200m"
    box_size: float | None = None
    concentration: np.ndarray | None = None
    velocity_dispersion: np.ndarray | None = None

    @classmethod
    def from_halos(
        cls, data: Any, mass_def: str = "M200m", box_size: float | None = None
    ) -> HaloCatalog:
        """Read a halo catalog from arrays, a file, or a simulation's output.

        Parameters
        ----------
        data : array-like or path-like
            A structured array, a mapping of column names to arrays, or a path
            to an HDF5/npz file (the latter requires ``jet[io]``).
        mass_def : str, optional
            Mass definition of the ``mass`` column.
        box_size : float, optional
            Side length of the periodic box.

        Returns
        -------
        HaloCatalog
        """
        raise NotImplementedError("HaloCatalog.from_halos is not implemented yet")

    @classmethod
    def from_particles(cls, *args: Any, **kwargs: Any) -> HaloCatalog:
        """Run a halo finder on a particle snapshot.

        Reserved, not planned for the first implementation: it needs a halo
        finder, which is a project of its own. Declared so that the entry point
        is stable if it is ever added.
        """
        raise NotImplementedError(
            "jet does not run a halo finder; run Rockstar/FOF upstream and use "
            "HaloCatalog.from_halos"
        )

    def __len__(self) -> int:
        return len(self.mass)


@dataclass
class GalaxyCatalog:
    """Positions and velocities of mock galaxies, with host-halo provenance.

    Attributes
    ----------
    position : ndarray of shape (n_galaxies, 3)
        Comoving coordinates in Mpc/h.
    velocity : ndarray of shape (n_galaxies, 3)
        Peculiar velocities in km/s.
    host_mass : ndarray of shape (n_galaxies,)
        Mass of the host halo, in Msun/h. Carried for HOD diagnostics and for
        statistics binned by halo mass.
    is_central : ndarray of shape (n_galaxies,), dtype bool
        Central/satellite flag. Carried because most HOD diagnostics are
        meaningless without it, and it is free at population time.
    box_size : float or None
        Side length of the periodic box, or ``None`` for a lightcone.
    """

    position: np.ndarray
    velocity: np.ndarray
    host_mass: np.ndarray
    is_central: np.ndarray
    box_size: float | None = None

    def __len__(self) -> int:
        return len(self.host_mass)


class HODModel(ABC):
    """Base class for a halo occupation distribution prescription.

    A subclass supplies two things: the parameters it expects (as a
    :class:`~jet.spec.ParameterSpec` whose parameters carry ``block="hod"``),
    and the rule that turns a halo catalog into a galaxy catalog.

    Subclasses are intended to be registered by name so that an emulator's
    input spec, a training configuration, and a saved bundle can all refer to
    the same model without importing it explicitly.

    Examples
    --------
    A five-parameter Zheng-style model, written the way a plugin author would::

        class Zheng07(HODModel):
            name = "zheng07"
            parameter_spec = ParameterSpec([
                Param("logMmin", bounds=(11.0, 14.0), block="hod"),
                Param("sigma_logM", bounds=(0.01, 1.0), block="hod"),
                Param("logM0", bounds=(11.0, 15.0), block="hod"),
                Param("logM1", bounds=(12.0, 15.5), block="hod"),
                Param("alpha", bounds=(0.01, 2.0), block="hod"),
            ])

            def populate(self, halos, theta, rng):   # doctest: +SKIP
                ...
    """

    #: Registry key, used in configurations and bundle manifests.
    name: ClassVar[str] = ""

    #: Parameters this model expects. Must use ``block="hod"`` for every entry
    #: so that a joint cosmology+hod spec can be assembled by concatenation.
    parameter_spec: ClassVar[ParameterSpec]

    @abstractmethod
    def populate(
        self,
        halos: HaloCatalog,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> GalaxyCatalog:
        """Place galaxies in halos.

        Parameters
        ----------
        halos : HaloCatalog
            Input halos.
        theta : ndarray of shape (n_params,)
            Parameter values in ``parameter_spec`` order. A single point, not a
            batch: sweeping parameters is the caller's loop, and keeping it that
            way lets the emulator own the notion of a parameter space.
        rng : numpy.random.Generator
            Random source. Passed in rather than created internally so that a
            population can be reproduced exactly, which matters when a training
            set has to be regenerated.

        Returns
        -------
        GalaxyCatalog
            Mock galaxies.
        """
