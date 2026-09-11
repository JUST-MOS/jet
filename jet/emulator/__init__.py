"""
The emulator framework: train, save, load and evaluate surrogate models.

The entry point is :class:`Emulator`. A typical run::

    from jet.spec import ParameterSpec, Param, DataVectorSpec
    from jet.emulator import Emulator

    x_spec = ParameterSpec([
        Param("Omegab", bounds=(0.04, 0.06), block="cosmo"),
        Param("ns", bounds=(0.92, 1.00), block="cosmo"),
    ])
    y_spec = DataVectorSpec("wp", n_bins=15, log=True)

    emulator = Emulator(x_spec, y_spec, backend="gp", n_pca=8)
    emulator.fit(X, y)                 # X: (N, 2)   y: (N, 15), raw units
    mean, std = emulator.predict(X_new)
    emulator.save("wp.gp.npz")

Loading a saved model needs neither scikit-learn nor PyTorch::

    from jet.emulator import Emulator
    emulator = Emulator.load("wp.gp.npz")   # resolved against $JET_DATA_DIR

Backends are registered by name, so a new regression method plugs in without
touching the transform layer::

    from jet.emulator.backends import Backend, register_backend
"""

from __future__ import annotations

from .backends import BACKENDS, Backend, get_backend, register_backend
from .bundle import load_bundle, save_bundle
from .emulator import DATA_DIR_ENV, Emulator, resolve_weights_path
from .transforms import (
    PCA,
    TRANSFORMS,
    BoundsNorm,
    Identity,
    Log10,
    StandardScaler,
    Transform,
)

__all__ = [
    "Emulator",
    "resolve_weights_path",
    "DATA_DIR_ENV",
    "Backend",
    "BACKENDS",
    "get_backend",
    "register_backend",
    "Transform",
    "TRANSFORMS",
    "Identity",
    "BoundsNorm",
    "StandardScaler",
    "PCA",
    "Log10",
    "save_bundle",
    "load_bundle",
]
