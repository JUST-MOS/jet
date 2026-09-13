"""
Gaussian-process backend: scikit-learn for training, NumPy for prediction.

Training and prediction deliberately use different machinery. Fitting needs
hyperparameter optimisation, kernel arithmetic and numerical safeguards -- all
of which scikit-learn already provides, tests, and maintains, so there is no
reason to reimplement them. Prediction needs to run on a machine that may have
nothing but NumPy installed, so it is implemented here in about a hundred
lines.

The two halves meet at :meth:`GPBackend.state`, which holds only plain arrays:
the training inputs, one set of kernel hyperparameters per output, and the
pre-solved dual coefficients. Nothing pickled, nothing version-pinned.

One Gaussian process is fitted per output column, not one for the whole data
vector. Outputs of a cosmology emulator are strongly correlated across bins,
and a single process sharing one set of length scales across all of them would
have to compromise between very different scales. Fitting per column (after the
output transform chain, which is typically where PCA has already decorrelated
them) mirrors what ``csstemu`` does with its per-component ``gprinfo``.

Three kernels are available, all multiplied by a constant amplitude:
``ConstantKernel * RBF``, ``ConstantKernel * Matern(nu=5/2)`` and
``ConstantKernel * Matern(nu=3/2)``. They are the three ``csstemu`` uses -- the
power-spectrum emulator takes the first, the halo mass function the second and
the halo-matter correlation function the third -- so an existing ``csstemu``
bundle is representable in this format whichever it was trained with. That is
the point of the shared ``.npz`` layout.

They differ in smoothness: an RBF sample path is infinitely differentiable, a
Matern-5/2 path only twice, a Matern-3/2 path only once. All are evaluated here
from the Euclidean distance between parameter points, so adding a kernel means
adding one expression to :func:`_kernel_matrix` and a key to :data:`KERNELS`.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular

from .base import Backend, register_backend

__all__ = ["GPBackend"]

#: Maximum number of times the noise floor is multiplied by ten while trying to
#: obtain a Cholesky factorisation of the training covariance.
_MAX_JITTER_STEPS = 8

#: Kernel keys accepted by :class:`GPBackend`.
#:
#: ``"rbf"``
#:     ``ConstantKernel * RBF``, the squared-exponential kernel.
#: ``"matern32"``
#:     ``ConstantKernel * Matern(nu=3/2)``, once-differentiable sample paths.
#: ``"matern52"``
#:     ``ConstantKernel * Matern(nu=5/2)``, twice-differentiable sample paths.
#:
#: All are written in terms of the Euclidean distance ``r`` between two
#: parameter points after dividing each coordinate by its length scale.
KERNELS = ("rbf", "matern32", "matern52")

#: Multiplier of ``r`` inside the Matern-3/2 kernel, ``sqrt(3)``.
_MATERN32_SQRT3 = float(np.sqrt(3.0))

#: Multiplier of ``r`` inside the Matern-5/2 kernel, ``sqrt(5)``.
_MATERN52_SQRT5 = float(np.sqrt(5.0))


def _squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between rows of ``a`` and ``b``.

    Computed through the Gram matrix rather than ``scipy.spatial.distance.cdist``
    because it is markedly faster for the small, dense blocks used here.

    Parameters
    ----------
    a, b : ndarray of shape (n, d) and (m, d)

    Returns
    -------
    ndarray of shape (n, m)
        Squared distances, clipped at zero to absorb round-off.
    """
    a_norm = np.einsum("ij,ij->i", a, a)[:, None]
    b_norm = np.einsum("ij,ij->i", b, b)[None, :]
    return np.maximum(a_norm + b_norm - 2.0 * a @ b.T, 0.0)


@register_backend
class GPBackend(Backend):
    """Gaussian process with an anisotropic kernel, one process per output.

    Parameters
    ----------
    kernel : {"rbf", "matern32", "matern52"}, optional
        Kernel shape, multiplied by a constant amplitude. ``"rbf"`` is the
        squared-exponential kernel; ``"matern32"`` and ``"matern52"`` are the
        Matern kernels with ``nu = 3/2`` (once differentiable) and ``nu = 5/2``
        (twice differentiable). All are anisotropic: one length scale per input
        feature. The reference implementation uses ``"rbf"`` for its linear
        power spectrum, ``"matern32"`` for its halo-matter correlation function
        and ``"matern52"`` for its halo mass function.
    alpha : float, optional
        Noise floor added to the diagonal of the training covariance. This is
        a numerical regulariser (the emulator treats simulation outputs as
        noiseless) rather than a fitted noise level.
    optimizer : str or None, optional
        Optimiser used for the log-marginal-likelihood, passed to
        scikit-learn. ``None`` keeps the initial hyperparameters, which
        reproduces what ``csstemu`` does -- it stores fixed hyperparameters and
        never optimises them. Worth reaching for when ``"fmin_l_bfgs_b"``
        emits a convergence warning per output column, or when the training set
        is small enough that the optimiser is fitting noise.
    n_restarts_optimizer : int, optional
        Number of restarts of the optimiser. Zero uses a single run from the
        initial hyperparameters.
    length_scale_bounds : tuple of float, optional
        Bounds on the kernel length scales handed to scikit-learn. When the
        optimiser reports that a length scale has pinned to a bound, widen
        these rather than accepting the warning: the fitted kernel is then
        constrained by the box, not by the data.
    constant_value_bounds : tuple of float, optional
        Bounds on the kernel amplitude handed to scikit-learn.
    random_state : int, optional
        Seed for the optimiser restarts.

    Notes
    -----
    Constructing the backend is free; scikit-learn is imported inside
    :meth:`fit` so that importing :mod:`jet.emulator` never pulls it in.
    """

    name: ClassVar[str] = "gp"

    def __init__(
        self,
        kernel: str = "rbf",
        alpha: float = 1e-10,
        optimizer: str | None = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        length_scale_bounds: tuple[float, float] = (1e-2, 1e2),
        constant_value_bounds: tuple[float, float] = (1e-3, 1e3),
        random_state: int = 0,
    ) -> None:
        if kernel not in KERNELS:
            raise ValueError(f"kernel must be one of {list(KERNELS)}, got {kernel!r}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.kernel = str(kernel)
        self.alpha = float(alpha)
        self.optimizer = optimizer
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.length_scale_bounds = tuple(length_scale_bounds)
        self.constant_value_bounds = tuple(constant_value_bounds)
        self.random_state = int(random_state)

        self._X_train: np.ndarray | None = None
        self._length_scale: np.ndarray | None = None
        self._constant_value: np.ndarray | None = None
        self._noise: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None
        self._alpha_coef: np.ndarray | None = None  # dual coefficients, (n_out, n_train)
        self._cholesky_cache: dict[int, np.ndarray] = {}
        self._history: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_out(self) -> int:
        """Number of independently fitted output columns."""
        self._require_fitted()
        return int(self._length_scale.shape[0])  # type: ignore[union-attr]

    @property
    def length_scale_(self) -> np.ndarray:
        """Fitted RBF length scales, shape ``(n_out, n_features)``."""
        self._require_fitted()
        return self._length_scale  # type: ignore[return-value]

    @property
    def constant_value_(self) -> np.ndarray:
        """Fitted kernel amplitudes, shape ``(n_out,)``."""
        self._require_fitted()
        return self._constant_value  # type: ignore[return-value]

    @property
    def history(self) -> dict[str, Any]:
        """Diagnostics from the last :meth:`fit`, notably the final noise floor."""
        return dict(self._history)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> GPBackend:
        """Fit one Gaussian process per output column.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Transformed parameter points.
        Y : ndarray of shape (n_samples, n_targets)
            Transformed data vectors.

        Returns
        -------
        GPBackend
            ``self``, fitted.

        Raises
        ------
        ImportError
            If scikit-learn is not installed; the message names the extra to
            install.
        """
        ConstantKernel, kernel_cls, GaussianProcessRegressor = _require_sklearn(self.kernel)

        X = self._as_2d(X, "X")
        Y = self._as_2d(Y, "Y")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples, got {X.shape[0]} and {Y.shape[0]}"
            )
        if X.shape[0] < 2:
            raise ValueError("GPBackend needs at least two training samples")

        n_samples, n_features = X.shape
        n_out = Y.shape[1]

        # Standardise the targets per column ourselves rather than relying on
        # scikit-learn's `normalize_y`. That keeps the saved state explicit and
        # independent of scikit-learn's internal attribute names, which have
        # changed between versions for multi-output targets.
        y_mean = Y.mean(axis=0)
        y_scale = Y.std(axis=0)
        y_scale = np.where(y_scale > 0.0, y_scale, 1.0)
        Y_scaled = (Y - y_mean) / y_scale

        length_scale = np.empty((n_out, n_features), dtype=float)
        constant_value = np.empty(n_out, dtype=float)
        alpha_coef = np.empty((n_out, n_samples), dtype=float)
        noise = np.empty(n_out, dtype=float)
        lml = np.empty(n_out, dtype=float)

        for column in range(n_out):
            kernel = ConstantKernel(
                constant_value=1.0, constant_value_bounds=self.constant_value_bounds
            ) * kernel_cls(
                length_scale=np.ones(n_features),
                length_scale_bounds=self.length_scale_bounds,
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                normalize_y=False,  # handled above, per column
                optimizer=self.optimizer,
                n_restarts_optimizer=self.n_restarts_optimizer,
                random_state=self.random_state,
            )
            gp.fit(X, Y_scaled[:, column])

            # Read the fitted hyperparameters back out, then rebuild the linear
            # algebra in NumPy. Doing the rebuild here means the NumPy
            # prediction path is exercised at save time, so a bundle that
            # cannot be replayed fails loudly at fit time instead of quietly at
            # load time.
            constant_value[column] = float(gp.kernel_.k1.constant_value)
            length_scale[column] = np.atleast_1d(gp.kernel_.k2.length_scale)
            lml[column] = float(gp.log_marginal_likelihood_value_)

            factor, used_alpha = _factorise(
                X,
                constant_value[column],
                length_scale[column],
                self.alpha,
                self.kernel,
                warn=f"output column {column}",
            )
            noise[column] = used_alpha
            alpha_coef[column] = cho_solve((factor, True), Y_scaled[:, column], check_finite=False)

        self._X_train = X
        self._length_scale = length_scale
        self._constant_value = constant_value
        self._noise = noise
        self._y_mean = y_mean
        self._y_std = y_scale
        self._alpha_coef = alpha_coef
        self._cholesky_cache = {}
        self._history = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_out": n_out,
            "log_marginal_likelihood": lml,
            "noise": noise.copy(),
            "alpha_requested": self.alpha,
        }
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with the fitted processes, using NumPy only.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Transformed parameter points.

        Returns
        -------
        mean : ndarray of shape (n_samples, n_out)
        std : ndarray of shape (n_samples, n_out)
            Marginal predictive standard deviation, including the noise floor.
        """
        X = self._as_2d(X, "X")
        self._require_fitted()
        self._check_width(X)

        n_out = self.n_out
        mean = np.empty((X.shape[0], n_out), dtype=float)
        std = np.empty((X.shape[0], n_out), dtype=float)

        for column in range(n_out):
            length_scale = self._length_scale[column]
            amplitude = self._constant_value[column]

            cross = _kernel_matrix(X, self._X_train, amplitude, length_scale, self.kernel)
            scaled_mean = cross @ self._alpha_coef[column]
            mean[:, column] = scaled_mean * self._y_std[column] + self._y_mean[column]

            factor = self._cholesky(column)
            # v = L^-1 K_star^T, so that var = k(x, x) - sum(v**2)
            v = solve_triangular(factor, cross.T, lower=True, check_finite=False)
            variance = np.clip(amplitude - np.einsum("ij,ij->j", v, v), 0.0, None)
            std[:, column] = np.sqrt(variance) * self._y_std[column]

        return mean, std

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def state(self) -> dict[str, np.ndarray]:
        """Return the fitted state as plain arrays.

        Deliberately excludes the Cholesky factors: they are ``O(n_out * n^2)``
        and re-derivable from ``X_train`` and the hyperparameters, which keeps
        a bundle at a few hundred kilobytes instead of tens of megabytes.
        """
        self._require_fitted()
        return {
            "kernel": np.asarray(self.kernel),
            "X_train": self._X_train,
            "length_scale": self._length_scale,
            "constant_value": self._constant_value,
            "noise": self._noise,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "alpha_coef": self._alpha_coef,
            "alpha_requested": np.asarray(self.alpha),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> GPBackend:
        """Rebuild a fitted backend from :meth:`state` output, using NumPy only."""
        required = (
            "X_train",
            "length_scale",
            "constant_value",
            "noise",
            "y_mean",
            "y_std",
            "alpha_coef",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(f"GP state is missing {missing}")

        # Bundles written before the kernel was selectable have no "kernel"
        # entry; they are all RBF, which is also the default.
        kernel = str(state["kernel"]) if "kernel" in state else "rbf"
        obj = cls(kernel=kernel, alpha=float(state.get("alpha_requested", 1e-10)))
        obj._X_train = np.asarray(state["X_train"], dtype=float)
        obj._length_scale = np.asarray(state["length_scale"], dtype=float)
        obj._constant_value = np.asarray(state["constant_value"], dtype=float)
        obj._noise = np.asarray(state["noise"], dtype=float)
        obj._y_mean = np.asarray(state["y_mean"], dtype=float)
        obj._y_std = np.asarray(state["y_std"], dtype=float)
        obj._alpha_coef = np.asarray(state["alpha_coef"], dtype=float)
        obj._cholesky_cache = {}
        obj._history = {}

        # A per-column length scale is stored as a 2-D array; a scalar kernel
        # would have been broadcast to shape (1, 1) on save, so restore the
        # isotropic case when every entry of a row is identical.
        if obj._length_scale.ndim == 1:
            obj._length_scale = obj._length_scale[None, :]
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cholesky(self, column: int) -> np.ndarray:
        """Return (and cache) the Cholesky factor for one output column."""
        cached = self._cholesky_cache.get(column)
        if cached is None:
            factor, used_alpha = _factorise(
                self._X_train,
                self._constant_value[column],
                self._length_scale[column],
                float(self._noise[column]),
                self.kernel,
                warn=None,
            )
            cached = factor
            self._cholesky_cache[column] = cached
        return cached

    def _check_width(self, X: np.ndarray) -> None:
        expected = self._X_train.shape[1]  # type: ignore[union-attr]
        if X.shape[1] != expected:
            raise ValueError(f"expected {expected} input features, got {X.shape[1]}")

    def _require_fitted(self) -> None:
        if self._X_train is None:
            raise RuntimeError("GPBackend has not been fitted")

    def __repr__(self) -> str:
        if self._X_train is None:
            return f"GPBackend(kernel={self.kernel!r}, alpha={self.alpha:g}, unfitted)"
        return (
            f"GPBackend(kernel={self.kernel!r}, n_train={self._X_train.shape[0]}, "
            f"n_out={self.n_out}, alpha={self.alpha:g})"
        )


def _kernel_matrix(
    a: np.ndarray,
    b: np.ndarray,
    constant_value: float,
    length_scale: np.ndarray,
    kernel: str = "rbf",
) -> np.ndarray:
    """Evaluate the amplitude times the kernel between rows of ``a`` and ``b``.

    Parameters
    ----------
    a, b : ndarray of shape (n, d) and (m, d)
        Parameter points, already through the input transform chain.
    constant_value : float
        Kernel amplitude.
    length_scale : ndarray of shape (d,)
        One length scale per feature.
    kernel : {"rbf", "matern32", "matern52"}, optional
        Kernel shape. See :data:`KERNELS`.

    Returns
    -------
    ndarray of shape (n, m)
        Covariance block.

    Notes
    -----
    Both Matern kernels are written in terms of the Euclidean distance ``r``,
    so the squared distances the Gram trick returns have to be square-rooted.
    The square root is taken of a value already clipped at zero, so there is no
    branch point at ``r = 0``. The kernel *value* is well defined there for
    every ``nu`` -- the Matern-3/2 path has a kink in its derivative at the
    origin, but evaluating it needs no special case.

    That kink is the reason ``nu = 3/2`` is only safe to fit with a fixed
    kernel: a hyperparameter optimiser differentiating through ``r = 0`` follows
    a subgradient that changes discontinuously under a perturbation of the
    length scale. Every bundle stored with this key was trained by the
    reference, which fixed its hyperparameters, and the bundles are rebuilt
    rather than refitted here, so the gradient path is never exercised.
    """
    squared = _squared_distance(a / length_scale, b / length_scale)
    if kernel == "rbf":
        # Kept on the squared distance: routing it through a square root and
        # back would cost accuracy for nothing.
        return constant_value * np.exp(-0.5 * squared)
    if kernel == "matern32":
        scaled = _MATERN32_SQRT3 * np.sqrt(squared)
        return constant_value * (1.0 + scaled) * np.exp(-scaled)
    if kernel == "matern52":
        scaled = _MATERN52_SQRT5 * np.sqrt(squared)
        return constant_value * (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)
    raise ValueError(f"unknown kernel {kernel!r}; expected one of {list(KERNELS)}")


def _factorise(
    X_train: np.ndarray,
    constant_value: float,
    length_scale: np.ndarray,
    alpha: float,
    kernel: str,
    warn: str | None,
) -> tuple[np.ndarray, float]:
    """Cholesky-factorise the training covariance, inflating the noise floor if needed.

    A kernel matrix built from near-duplicate training points can fail to be
    positive definite. Rather than letting ``LinAlgError`` escape with no
    context, the noise floor is multiplied by ten up to
    :data:`_MAX_JITTER_STEPS` times; the value actually used is returned so it
    can be stored alongside the model.

    Parameters
    ----------
    X_train : ndarray of shape (n, d)
        Training inputs.
    constant_value, length_scale : float and ndarray
        Kernel hyperparameters.
    alpha : float
        Starting noise floor.
    warn : str, optional
        Context for the warning message, or ``None`` to stay silent (used when
        re-factorising at prediction time, where a warning would repeat).

    Returns
    -------
    factor : ndarray of shape (n, n)
        Lower-triangular Cholesky factor.
    alpha_used : float
        Noise floor that succeeded.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the matrix is still not positive definite after the last step.
    """
    covariance = _kernel_matrix(X_train, X_train, constant_value, length_scale, kernel)
    n_samples = covariance.shape[0]
    current = float(alpha)

    for step in range(_MAX_JITTER_STEPS + 1):
        try:
            factor = cholesky(covariance + current * np.eye(n_samples), lower=True)
        except np.linalg.LinAlgError:
            if step == _MAX_JITTER_STEPS:
                raise
            current *= 10.0
            continue
        if step > 0 and warn is not None:
            warnings.warn(
                f"{warn}: the noise floor had to be raised from {alpha:g} to "
                f"{current:g} for a stable factorisation; predictions will be "
                "slightly smoother than requested.",
                RuntimeWarning,
                stacklevel=3,
            )
        return factor, current

    raise AssertionError("unreachable: the loop either returns or re-raises")


def _require_sklearn(kernel: str = "rbf") -> tuple[Any, Any, Any]:
    """Import the scikit-learn pieces needed for training.

    Parameters
    ----------
    kernel : {"rbf", "matern32", "matern52"}
        Which kernel class to return alongside the amplitude and the regressor.

    Returns
    -------
    tuple
        ``(ConstantKernel, kernel class, GaussianProcessRegressor)``.

    Raises
    ------
    ImportError
        If scikit-learn is missing, naming the extra to install.
    """
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "training a Gaussian-process emulator requires scikit-learn; "
            "install it with `pip install jet[gp]`"
        ) from exc

    if kernel == "rbf":
        cls = RBF
    elif kernel == "matern32":
        cls = functools.partial(Matern, nu=1.5)
    elif kernel == "matern52":
        cls = functools.partial(Matern, nu=2.5)
    else:  # pragma: no cover - GPBackend validates before reaching here
        raise ValueError(f"unknown kernel {kernel!r}; expected one of {list(KERNELS)}")
    return ConstantKernel, cls, GaussianProcessRegressor
