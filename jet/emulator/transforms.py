"""
Composable transforms that sit between raw parameters and a regression backend.

A backend sees only already-transformed arrays: ``fit(X, Y)`` where ``X`` and
``Y`` are the outputs of a forward transform chain. Everything that is *not*
regression -- rescaling to a physical range, standardising, compressing with
PCA, taking logarithms -- lives here instead, so that swapping a Gaussian
process for a neural network never means rewriting the preprocessing.

Every transform is written against plain NumPy, so a saved bundle can be
replayed on a machine that has neither scikit-learn nor PyTorch installed.

All transforms operate on 2-D arrays of shape ``(n_samples, n_features)``.

Uncertainty propagation
-----------------------
``inverse_transform_std`` pushes a Gaussian ``(mean, std)`` back to the
original space. It is *not* a plain matrix multiply on the standard deviations:
PCA components are independent, so variances add

.. math:: \\sigma_j = \\sqrt{\\sum_i c_{ij}^2 \\sigma_i^2}

and ``log10`` is nonlinear, so the delta method applies

.. math:: \\sigma_y = \\ln(10)\\, y \\, \\sigma_{\\log_{10} y}

Both of the corresponding one-liners in the reference implementations are
wrong; see ``notes`` in each method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "Transform",
    "Identity",
    "BoundsNorm",
    "StandardScaler",
    "PCA",
    "Log10",
    "TRANSFORMS",
    "transform_from_state",
    "forward",
    "inverse",
    "inverse_std",
]


class Transform(ABC):
    """Base class for a single step of a preprocessing chain.

    Subclasses declare a short ``name`` used as the key under which the step is
    written into a bundle, and must implement ``transform``,
    ``inverse_transform``, ``inverse_transform_std``, ``state`` and
    ``from_state``.

    ``fit`` returns ``self`` so chains can be built fluently.
    """

    #: Registry key written into the bundle manifest.
    name: ClassVar[str] = ""

    @abstractmethod
    def fit(self, a: np.ndarray) -> Transform:
        """Learn whatever state this transform needs from ``a``.

        Parameters
        ----------
        a : ndarray of shape (n_samples, n_features)
            Training values, in the space this transform consumes.

        Returns
        -------
        Transform
            ``self``, fitted.
        """

    @abstractmethod
    def transform(self, a: np.ndarray) -> np.ndarray:
        """Map ``a`` forward through this transform."""

    @abstractmethod
    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        """Map ``a`` back to the space this transform consumes."""

    @abstractmethod
    def inverse_transform_std(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Push a Gaussian back through this transform.

        Parameters
        ----------
        mean, std : ndarray of shape (n_samples, n_features)
            Mean and standard deviation in the *output* space of this
            transform, i.e. what ``transform`` would produce.

        Returns
        -------
        tuple of ndarray
            ``(mean, std)`` in the input space of this transform.
        """

    @abstractmethod
    def state(self) -> dict[str, np.ndarray]:
        """Return the fitted state as plain arrays, ready for an ``.npz``."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> Transform:
        """Rebuild a fitted transform from :meth:`state` output."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_2d(a: Any, name: str = "a") -> np.ndarray:
        """Validate and coerce an input to a 2-D float array."""
        a = np.asarray(a, dtype=float)
        if a.ndim != 2:
            raise ValueError(f"{name} must be 2-D (n_samples, n_features), got {a.shape}")
        return a

    def _check_width(self, a: np.ndarray, width: int) -> None:
        """Raise if ``a`` does not have the feature count this transform expects."""
        if a.shape[1] != width:
            raise ValueError(f"{type(self).__name__} expects {width} features, got {a.shape[1]}")


class Identity(Transform):
    """A pass-through step.

    Useful as an explicit placeholder in a chain, and as the ``use_pca=False``
    counterpart of :class:`PCA` when a chain is assembled from a configuration.
    """

    name = "identity"

    def fit(self, a: np.ndarray) -> Identity:
        self._width = self._as_2d(a).shape[1]
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        return self._as_2d(a)

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        return self._as_2d(a)

    def inverse_transform_std(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._as_2d(mean, "mean"), self._as_2d(std, "std")

    def state(self) -> dict[str, np.ndarray]:
        return {"width": np.asarray(self._width)}

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> Identity:
        obj = cls()
        obj._width = int(state["width"])
        return obj


class BoundsNorm(Transform):
    """Affine map of a bounded range onto ``[0, 1]``.

    Parameters
    ----------
    low, high : array-like of shape (n_features,)
        Validity range of each feature. Values outside are still mapped (the
        result simply leaves ``[0, 1]``); bounds are a *warning* threshold, not
        a clip. :meth:`jet.emulator.Emulator.predict` reports excursions.

    Examples
    --------
    >>> import numpy as np
    >>> t = BoundsNorm([0.0, 10.0], [1.0, 20.0]).fit(np.zeros((1, 2)))
    >>> t.transform(np.array([[0.5, 15.0]]))
    array([[0.5, 0.5]])
    """

    name = "bounds_norm"

    def __init__(self, low: Sequence[float], high: Sequence[float]) -> None:
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        if low.ndim != 1 or high.ndim != 1 or low.shape != high.shape:
            raise ValueError(
                f"low and high must be 1-D of equal length, got {low.shape} and {high.shape}"
            )
        if np.any(high <= low):
            bad = np.nonzero(high <= low)[0].tolist()
            raise ValueError(f"high must exceed low; offending features: {bad}")
        self._low = low
        self._high = high

    @property
    def width(self) -> np.ndarray:
        """Per-feature range ``high - low``."""
        return self._high - self._low

    def fit(self, a: np.ndarray) -> BoundsNorm:
        self._check_width(self._as_2d(a), self._low.size)
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        a = self._as_2d(a)
        self._check_width(a, self._low.size)
        return (a - self._low) / self.width

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        a = self._as_2d(a)
        self._check_width(a, self._low.size)
        return a * self.width + self._low

    def inverse_transform_std(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Linear map: sigma scales by the same factor as the mean.
        mean = self._as_2d(mean, "mean")
        std = self._as_2d(std, "std")
        self._check_width(mean, self._low.size)
        return mean * self.width + self._low, std * self.width

    def state(self) -> dict[str, np.ndarray]:
        return {"low": self._low, "high": self._high}

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> BoundsNorm:
        return cls(state["low"], state["high"])

    def __repr__(self) -> str:
        return f"BoundsNorm(n_features={self._low.size})"


class StandardScaler(Transform):
    """Centre to zero mean and scale to unit variance, feature by feature.

    Features whose training standard deviation is zero are left unscaled
    (divisor set to 1) rather than producing infinities.
    """

    name = "standard_scaler"

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature training mean."""
        self._require_fitted()
        return self._mean  # type: ignore[return-value]

    @property
    def scale_(self) -> np.ndarray:
        """Per-feature training standard deviation, with zeros replaced by one."""
        self._require_fitted()
        return self._scale  # type: ignore[return-value]

    def fit(self, a: np.ndarray) -> StandardScaler:
        a = self._as_2d(a)
        if a.shape[0] < 1:
            raise ValueError("cannot fit StandardScaler on zero samples")
        self._mean = a.mean(axis=0)
        scale = a.std(axis=0)
        self._scale = np.where(scale > 0.0, scale, 1.0)
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        a = self._as_2d(a)
        self._require_fitted()
        self._check_width(a, self._mean.size)  # type: ignore[union-attr]
        return (a - self._mean) / self._scale  # type: ignore[operator]

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        a = self._as_2d(a)
        self._require_fitted()
        self._check_width(a, self._mean.size)  # type: ignore[union-attr]
        return a * self._scale + self._mean  # type: ignore[operator]

    def inverse_transform_std(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Linear map: sigma scales by the same factor as the mean.
        mean = self._as_2d(mean, "mean")
        std = self._as_2d(std, "std")
        self._require_fitted()
        self._check_width(mean, self._mean.size)  # type: ignore[union-attr]
        return mean * self._scale + self._mean, std * self._scale  # type: ignore[operator]

    def state(self) -> dict[str, np.ndarray]:
        self._require_fitted()
        return {"mean": self._mean, "scale": self._scale}  # type: ignore[dict-item]

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> StandardScaler:
        obj = cls()
        obj._mean = np.asarray(state["mean"], dtype=float)
        obj._scale = np.asarray(state["scale"], dtype=float)
        return obj

    def _require_fitted(self) -> None:
        if self._mean is None or self._scale is None:
            raise RuntimeError("StandardScaler has not been fitted")

    def __repr__(self) -> str:
        width = "-" if self._mean is None else self._mean.size
        return f"StandardScaler(n_features={width})"


class PCA(Transform):
    """Linear compression to the leading principal components.

    Implemented with a plain SVD so that no scikit-learn is needed on the
    prediction path. The sign of each component is arbitrary (SVD is defined up
    to a sign); this is harmless because ``transform`` and
    ``inverse_transform`` use the same ``components_``.

    Parameters
    ----------
    n_components : int
        Number of components to keep. Must not exceed
        ``min(n_samples, n_features)`` of the training set.
    """

    name = "pca"

    def __init__(self, n_components: int) -> None:
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError(f"n_components must be a positive integer, got {n_components!r}")
        self.n_components = int(n_components)
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature training mean, subtracted before projecting."""
        self._require_fitted()
        return self._mean  # type: ignore[return-value]

    @property
    def components_(self) -> np.ndarray:
        """Principal axes, shape ``(n_components, n_features)``."""
        self._require_fitted()
        return self._components  # type: ignore[return-value]

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fraction of total variance captured by each retained component."""
        self._require_fitted()
        return self._explained  # type: ignore[return-value]

    def fit(self, a: np.ndarray) -> PCA:
        a = self._as_2d(a)
        n_samples, n_features = a.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} exceeds min(n_samples={n_samples}, "
                f"n_features={n_features})"
            )
        self._mean = a.mean(axis=0)
        centred = a - self._mean
        # full_matrices=False keeps only min(n_samples, n_features) singular
        # vectors, which is all we can retain anyway.
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        self._components = vt[: self.n_components]
        variance = singular**2 / max(n_samples - 1, 1)
        total = variance.sum()
        self._explained = (
            variance[: self.n_components] / total if total > 0 else variance[: self.n_components]
        )
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        a = self._as_2d(a)
        self._require_fitted()
        self._check_width(a, self._mean.size)  # type: ignore[union-attr]
        return (a - self._mean) @ self._components.T  # type: ignore[operator]

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        a = self._as_2d(a)
        self._require_fitted()
        self._check_width(a, self.n_components)
        return a @ self._components + self._mean  # type: ignore[operator]

    def inverse_transform_std(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Components are mutually independent, so variances add:
        #   var_j = sum_i c_ij^2 * var_i
        # The reference implementation instead computes `std @ components`,
        # which propagates standard deviations linearly and therefore
        # systematically overestimates the uncertainty.
        mean = self._as_2d(mean, "mean")
        std = self._as_2d(std, "std")
        self._require_fitted()
        self._check_width(mean, self.n_components)
        out_mean = mean @ self._components + self._mean  # type: ignore[operator]
        out_std = np.sqrt((std**2) @ self._components**2)  # type: ignore[operator]
        return out_mean, out_std

    def state(self) -> dict[str, np.ndarray]:
        self._require_fitted()
        return {
            "mean": self._mean,  # type: ignore[dict-item]
            "components": self._components,  # type: ignore[dict-item]
            "explained_variance_ratio": self._explained,  # type: ignore[dict-item]
            "n_components": np.asarray(self.n_components),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> PCA:
        obj = cls(int(state["n_components"]))
        obj._mean = np.asarray(state["mean"], dtype=float)
        obj._components = np.asarray(state["components"], dtype=float)
        obj._explained = np.asarray(state["explained_variance_ratio"], dtype=float)
        return obj

    def _require_fitted(self) -> None:
        if self._mean is None or self._components is None:
            raise RuntimeError("PCA has not been fitted")

    def __repr__(self) -> str:
        return f"PCA(n_components={self.n_components})"


class Log10(Transform):
    """Base-10 logarithm, applied elementwise.

    Placed on the *output* chain when a statistic spans several decades, so
    that a stationary kernel or a mean-squared-error loss is not dominated by
    the largest values. Which statistics need it is declared on
    :class:`jet.spec.DataVectorSpec`.

    Notes
    -----
    The inverse is not mean-preserving. ``inverse_transform`` returns
    ``10**mean``, which is the *median* of the implied log-normal, not its
    expectation: for ``std = 0.05`` in log10 the true mean sits about 0.7 per
    cent high. That is the right quantity to compare against a measurement
    (which is also a central value, not an average over realisations), but it
    does mean ``Emulator.predict`` returns a median-like mean whenever the data
    vector is stored as a logarithm. Anything that needs the true expectation
    has to say so explicitly rather than assume the returned mean is one.
    """

    name = "log10"

    def fit(self, a: np.ndarray) -> Log10:
        a = self._as_2d(a)
        if a.size and not np.all(a > 0.0):
            bad = np.argwhere(a <= 0.0)[:5]
            raise ValueError(
                "Log10 requires strictly positive values; "
                f"first offending (row, column) pairs: {bad.tolist()}"
            )
        self._width = a.shape[1]
        return self

    def transform(self, a: np.ndarray) -> np.ndarray:
        a = self._as_2d(a)
        if a.size and not np.all(a > 0.0):
            bad = np.argwhere(a <= 0.0)[:5]
            raise ValueError(
                "Log10 requires strictly positive values; "
                f"first offending (row, column) pairs: {bad.tolist()}"
            )
        return np.log10(a)

    def inverse_transform(self, a: np.ndarray) -> np.ndarray:
        return np.power(10.0, self._as_2d(a))

    def inverse_transform_std(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # Delta method: with y = 10**z, dy/dz = ln(10) * 10**z = ln(10) * y,
        # hence sigma_y = ln(10) * y * sigma_z. The reference implementation
        # passes the standard deviation through `10**` untransformed.
        mean = self._as_2d(mean, "mean")
        std = self._as_2d(std, "std")
        out_mean = np.power(10.0, mean)
        return out_mean, np.log(10.0) * out_mean * std

    def state(self) -> dict[str, np.ndarray]:
        return {"width": np.asarray(getattr(self, "_width", 0))}

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> Log10:
        obj = cls()
        obj._width = int(state["width"])
        return obj

    def __repr__(self) -> str:
        return "Log10()"


#: Registry mapping bundle keys to transform classes.
TRANSFORMS: dict[str, type[Transform]] = {
    Identity.name: Identity,
    BoundsNorm.name: BoundsNorm,
    StandardScaler.name: StandardScaler,
    PCA.name: PCA,
    Log10.name: Log10,
}


def transform_from_state(name: str, state: Mapping[str, np.ndarray]) -> Transform:
    """Rebuild a fitted transform from its registry key and stored state.

    Parameters
    ----------
    name : str
        Key in :data:`TRANSFORMS`, as written into the bundle manifest.
    state : mapping of str to ndarray
        Arrays produced by ``Transform.state``.

    Returns
    -------
    Transform
        The reconstructed, fitted transform.
    """
    try:
        cls = TRANSFORMS[name]
    except KeyError:
        raise KeyError(
            f"unknown transform {name!r}; registered transforms are {sorted(TRANSFORMS)}"
        ) from None
    return cls.from_state(state)


def forward(chain: Iterable[Transform], a: np.ndarray) -> np.ndarray:
    """Apply a chain of transforms in order."""
    for step in chain:
        a = step.transform(a)
    return a


def inverse(chain: Iterable[Transform], a: np.ndarray) -> np.ndarray:
    """Undo a chain of transforms, in reverse order."""
    for step in reversed(list(chain)):
        a = step.inverse_transform(a)
    return a


def inverse_std(
    chain: Iterable[Transform], mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Push a Gaussian ``(mean, std)`` back through a chain, in reverse order.

    Each step applies its own uncertainty rule (exact for affine steps, the
    delta method for ``log10``, variance addition for ``PCA``); see the class
    docstrings for the reasoning.
    """
    for step in reversed(list(chain)):
        mean, std = step.inverse_transform_std(mean, std)
    return mean, std
