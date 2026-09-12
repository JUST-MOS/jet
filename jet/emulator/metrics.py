"""
Accuracy measures and cross-validation for emulators.

The number that matters for a surrogate is not how well it fits the points it
was trained on, but how well it predicts points it has never seen. Every
function here that evaluates accuracy therefore takes out-of-fold predictions
rather than training-set ones -- a model that memorises its training set scores
a perfect :func:`r2_score` and is worthless.

``csstemu`` advertises its accuracy with a leave-one-out plot; :func:`loo_predict`
reproduces that. It refits the emulator once per sample, so it is genuinely
expensive -- use :func:`kfold_predict` while iterating and reserve leave-one-out
for the final number.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .emulator import Emulator

__all__ = ["r2_score", "relative_error", "kfold_predict", "loo_predict", "summarise"]


def r2_score(
    y_true: np.ndarray, y_pred: np.ndarray, per_column: bool = False
) -> float | np.ndarray:
    """Coefficient of determination.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (n_samples, n_targets)
        Reference and predicted data vectors.
    per_column : bool, optional
        Return one score per output column instead of the pooled score. Useful
        for spotting the two or three bins that carry all the error, which a
        pooled number hides.

    Returns
    -------
    float or ndarray of shape (n_targets,)
        ``1`` is perfect prediction, ``0`` is predicting the mean, negative is
        worse than predicting the mean.

    Notes
    -----
    ``per_column=False`` pools every entry of the array, so the reference
    "mean" is the mean over *all* entries, not the per-column mean. Pooling is
    the cruder measure: when the output columns have very different variances
    -- a number density next to a correlation function, say -- the pooled score
    is dominated by the largest column. Use ``per_column=True`` to see how each
    bin is actually doing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if per_column:
        residual = ((y_true - y_pred) ** 2).sum(axis=0)
        total = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    else:
        residual = ((y_true - y_pred) ** 2).sum()
        total = ((y_true - y_true.mean()) ** 2).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        score = 1.0 - residual / total
    return np.where(np.isfinite(score), score, np.nan)


def relative_error(
    y_true: np.ndarray, y_pred: np.ndarray, per_column: bool = False
) -> float | np.ndarray:
    """Typical fractional error, ``median(|pred - true| / |true|)``.

    The median rather than the mean: emulator residuals are heavy-tailed, and a
    handful of bins near a zero crossing would otherwise dominate the average.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (n_samples, n_targets)
        Reference and predicted data vectors. Values must be non-zero.
    per_column : bool, optional
        Return one value per output column instead of the pooled value.

    Returns
    -------
    float or ndarray of shape (n_targets,)
        Fractional error; ``0.01`` means one percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    magnitude = np.abs(y_true)
    if np.any(magnitude == 0.0):
        raise ValueError("relative_error is undefined where y_true is zero")

    fractional = np.abs(y_pred - y_true) / magnitude
    return np.median(fractional, axis=0) if per_column else float(np.median(fractional))


def kfold_predict(
    make_emulator: Callable[[], Emulator],
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Return out-of-fold predictions for every sample.

    The caller supplies a factory rather than an emulator, because each fold
    needs a freshly fitted model -- reusing one would leak the held-out samples
    into the training set.

    Parameters
    ----------
    make_emulator : callable
        Zero-argument callable returning a new, unfitted
        :class:`~jet.emulator.Emulator` with the desired configuration.
    X : ndarray of shape (n_samples, n_params)
        Raw parameter values.
    y : ndarray of shape (n_samples, n_targets)
        Raw data vectors.
    k : int, optional
        Number of folds. Must satisfy ``2 <= k <= n_samples``.
    seed : int, optional
        Seed for the fold assignment.

    Returns
    -------
    ndarray of shape (n_samples, n_targets)
        Each sample predicted by a model that did not see it.

    Examples
    --------
    >>> import numpy as np
    >>> from jet.spec import ParameterSpec, Param, DataVectorSpec
    >>> from jet.emulator import Emulator
    >>> from jet.emulator.metrics import kfold_predict, r2_score
    >>> spec = ParameterSpec([Param("x", bounds=(0.0, 1.0))])
    >>> dv = DataVectorSpec("demo", n_bins=1)
    >>> X = np.linspace(0.0, 1.0, 40)[:, None]
    >>> y = np.sin(X)
    >>> make = lambda: Emulator(spec, dv, backend="gp")   # doctest: +SKIP
    >>> oof = kfold_predict(make, X, y, k=4)              # doctest: +SKIP
    >>> r2_score(y, oof) > 0.99                           # doctest: +SKIP
    True
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y disagree on sample count: {X.shape[0]} vs {y.shape[0]}")
    n_samples = X.shape[0]
    if not 2 <= k <= n_samples:
        raise ValueError(f"k must satisfy 2 <= k <= {n_samples}, got {k}")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    folds = np.array_split(order, k)

    predictions = np.empty_like(y, dtype=float)
    for fold in folds:
        mask = np.ones(n_samples, dtype=bool)
        mask[fold] = False
        model = make_emulator()
        model.fit(X[mask], y[mask])
        predictions[fold] = model.predict(X[fold], return_std=False)

    return predictions


def loo_predict(make_emulator: Callable[[], Emulator], X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions: :func:`kfold_predict` with one fold per sample.

    Refits the emulator ``n_samples`` times. For a few hundred samples and a
    Gaussian process this is minutes; for a neural network it is usually hours,
    in which case prefer :func:`kfold_predict`.

    Parameters
    ----------
    make_emulator : callable
        Zero-argument callable returning a new, unfitted emulator.
    X, y : ndarray
        Raw parameter values and data vectors.

    Returns
    -------
    ndarray of shape (n_samples, n_targets)
        Each sample predicted from a model trained on all the others.
    """
    return kfold_predict(make_emulator, X, y, k=np.asarray(X).shape[0])


def summarise(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Bundle the usual accuracy numbers into one dictionary.

    Parameters
    ----------
    y_true, y_pred : ndarray of shape (n_samples, n_targets)

    Returns
    -------
    dict
        ``r2``, ``r2_per_column``, ``relative_error``, ``relative_error_per_column``
        and ``max_abs_error``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "r2_per_column": r2_score(y_true, y_pred, per_column=True),
        "relative_error": relative_error(y_true, y_pred),
        "relative_error_per_column": relative_error(y_true, y_pred, per_column=True),
        "max_abs_error": float(np.abs(y_true - y_pred).max()),
    }
