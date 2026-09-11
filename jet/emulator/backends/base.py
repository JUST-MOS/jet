"""
The regression-backend protocol and its registry.

A backend is the only part of the emulator that actually learns a map from
inputs to outputs. It sees arrays that have *already* been through the
transform chain and returns predictions in that same transformed space -- it
knows nothing about parameter bounds, standardisation, PCA, or physical units.
Keeping the contract that narrow is what makes the Gaussian-process and neural
network backends interchangeable.

Every backend exchanges state as plain NumPy arrays, never as pickled objects.
That is what lets a bundle be loaded, and predictions produced, without
scikit-learn or PyTorch installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "Backend",
    "BACKENDS",
    "register_backend",
    "get_backend",
    "backend_from_state",
]


class Backend(ABC):
    """Abstract regression backend.

    Subclasses declare a short ``name`` (the registry key written into a
    bundle) and implement ``fit``, ``predict``, ``state`` and ``from_state``.

    Notes
    -----
    ``fit`` may require a third-party package (scikit-learn, PyTorch), but
    ``predict`` and ``from_state`` must not: those two run on the prediction
    path, which has to work with NumPy alone.
    """

    #: Registry key, e.g. ``"gp"`` or ``"nn"``.
    name: ClassVar[str] = ""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray) -> Backend:
        """Train on transformed inputs and targets.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Transformed parameter points.
        Y : ndarray of shape (n_samples, n_targets)
            Transformed data vectors.

        Returns
        -------
        Backend
            ``self``, fitted.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict at transformed parameter points.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Transformed parameter points.

        Returns
        -------
        mean : ndarray of shape (n_samples, n_targets)
            Predicted data vectors, in transformed space.
        std : ndarray of shape (n_samples, n_targets) or None
            Per-point predictive standard deviation, or ``None`` when the
            backend does not report one. A Gaussian process always can; the
            neural-network backend currently never does, and returns ``None``
            so that callers are told rather than handed an array of zeros.
        """

    @abstractmethod
    def state(self) -> dict[str, np.ndarray]:
        """Return the fitted state as plain arrays, ready for an ``.npz``."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> Backend:
        """Rebuild a fitted backend from :meth:`state` output."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_2d(a: Any, name: str) -> np.ndarray:
        """Validate and coerce an input to a 2-D float array."""
        a = np.asarray(a, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"{name} must be 2-D (n_samples, n_features), got {a.shape}")
        return a


#: Registry mapping backend keys to classes. Populated by ``register_backend``.
BACKENDS: dict[str, type[Backend]] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    """Class decorator adding a backend to :data:`BACKENDS`.

    Raises
    ------
    ValueError
        If the class does not declare a ``name``, or the name is taken.
    """
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(f"{cls.__name__} must declare a non-empty `name`")
    existing = BACKENDS.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"backend name {name!r} is already registered to {existing.__name__}")
    BACKENDS[name] = cls
    return cls


def get_backend(name: str, **kwargs: Any) -> Backend:
    """Construct a backend by registry key.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``"gp"``.
    **kwargs
        Passed straight to the backend constructor.

    Returns
    -------
    Backend
        An unfitted backend instance.

    Raises
    ------
    KeyError
        If the name is unknown; the message lists what is registered.
    """
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; registered backends are {sorted(BACKENDS)}"
        ) from None
    return cls(**kwargs)


def backend_from_state(name: str, state: Mapping[str, np.ndarray]) -> Backend:
    """Rebuild a fitted backend from its registry key and stored state."""
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; registered backends are {sorted(BACKENDS)}"
        ) from None
    return cls.from_state(state)
