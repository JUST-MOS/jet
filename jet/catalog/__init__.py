"""
Halo catalogs and the HOD models that populate them with galaxies.

**Design stage only.** The abstractions below fix the shape of the interface;
no implementation is provided yet. They are written down now because the
emulator's input specification (``ParameterSpec`` with ``block="hod"``) is
produced here, and that contract has to exist before either side is built.

Pipeline position
-----------------
``halo catalog`` -> ``HODModel.populate`` -> ``galaxy catalog`` -> ``jet.estimator``

Decided
-------
Entry point
    A halo catalog, not particles. Running a halo finder is a substantial
    project of its own and is assumed to happen upstream (Rockstar, FOF, or a
    collaborator's pipeline). ``HaloCatalog.from_particles`` is reserved but
    unimplemented, so adding it later does not change any caller.
Mass definition
    Must be declared, not assumed. HOD parameters calibrated against M200m and
    against Mvir are not interchangeable. The default is ``"M200m"`` to match
    the reference emulator this project grew out of.
HOD models are plugins
    ``HODModel`` subclasses declare their own ``ParameterSpec``, so an emulator
    trained on a five-parameter Zheng-style model and one trained on a
    ten-parameter extended model are simply different models. Nothing in the
    framework assumes a fixed parameter count or a fixed functional form.

Open questions
--------------
* Redshift-space distortions: applied here, or deferred to the estimator? Doing
  it here keeps the galaxy catalog a faithful mock; doing it there lets one
  catalog serve both real and redshift space.
* Box or lightcone: the current scope is a periodic box. A lightcone needs the
  survey selection function and is a separate piece of work.
* JUST-specific observational realism -- magnitude limits, fibre assignment,
  redshift errors -- has not been designed at all.
* ``GalaxyCatalog`` beyond positions and velocities: what else does the
  estimator need? Host halo mass (for HOD diagnostics) and central/satellite
  flags are cheap to carry and likely worth it.
"""

from __future__ import annotations

from .catalog import GalaxyCatalog, HaloCatalog, HODModel

__all__ = ["HaloCatalog", "GalaxyCatalog", "HODModel"]
