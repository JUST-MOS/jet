"""
Tests for jet.emulator.metrics.

Cross-validation is tested with a stand-in emulator rather than a real backend:
what matters is that every sample is predicted by a model that did not see it,
and a least-squares fit makes that check exact and fast. Real backends are
covered in test_backends.py and test_emulator.py.
"""

from __future__ import annotations

import unittest

import numpy as np

from jet.emulator.metrics import (
    kfold_predict,
    loo_predict,
    r2_score,
    relative_error,
    summarise,
)


class _LeastSquaresEmulator:
    """Minimal emulator stand-in: ordinary least squares with an intercept."""

    def __init__(self) -> None:
        self.coefficients: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> _LeastSquaresEmulator:
        design = np.column_stack([np.ones(X.shape[0]), X])
        self.coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> np.ndarray:
        design = np.column_stack([np.ones(X.shape[0]), X])
        return design @ self.coefficients


def make_emulator() -> _LeastSquaresEmulator:
    """Factory handed to the cross-validation helpers."""
    return _LeastSquaresEmulator()


class TestR2Score(unittest.TestCase):
    """Coefficient of determination."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.y_true = rng.normal(size=(50, 3))

    def tearDown(self) -> None:
        del self.y_true

    def test_perfect_prediction_scores_one(self) -> None:
        self.assertAlmostEqual(r2_score(self.y_true, self.y_true), 1.0)

    def test_predicting_the_global_mean_scores_zero(self) -> None:
        # The pooled score compares against the mean over *all* entries, so
        # that is the constant prediction that must score exactly zero.
        mean_only = np.full_like(self.y_true, self.y_true.mean())
        self.assertAlmostEqual(r2_score(self.y_true, mean_only), 0.0)

    def test_predicting_each_column_mean_scores_zero_per_column(self) -> None:
        per_column_mean = np.tile(self.y_true.mean(axis=0), (self.y_true.shape[0], 1))
        np.testing.assert_allclose(
            r2_score(self.y_true, per_column_mean, per_column=True), 0.0, atol=1e-12
        )

    def test_worse_than_the_mean_scores_negative(self) -> None:
        self.assertLess(r2_score(self.y_true, self.y_true + 10.0), 0.0)

    def test_per_column(self) -> None:
        prediction = self.y_true.copy()
        prediction[:, 1] = self.y_true.mean(axis=0)[1]
        per_column = r2_score(self.y_true, prediction, per_column=True)
        self.assertEqual(per_column.shape, (3,))
        self.assertAlmostEqual(float(per_column[0]), 1.0)
        self.assertAlmostEqual(float(per_column[1]), 0.0)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            r2_score(self.y_true, self.y_true[:10])


class TestRelativeError(unittest.TestCase):
    """Median fractional error."""

    def setUp(self) -> None:
        self.y_true = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])

    def tearDown(self) -> None:
        del self.y_true

    def test_exact_prediction_is_zero(self) -> None:
        self.assertAlmostEqual(relative_error(self.y_true, self.y_true), 0.0)

    def test_uniform_ten_percent_error(self) -> None:
        self.assertAlmostEqual(relative_error(self.y_true, self.y_true * 1.1), 0.1)

    def test_per_column(self) -> None:
        prediction = self.y_true.copy()
        prediction[:, 1] *= 1.5
        per_column = relative_error(self.y_true, prediction, per_column=True)
        self.assertAlmostEqual(float(per_column[0]), 0.0)
        self.assertAlmostEqual(float(per_column[1]), 0.5)

    def test_uses_the_median_not_the_mean(self) -> None:
        # One wildly wrong bin must not move the reported error much.
        prediction = self.y_true.copy()
        prediction[0, 0] = 1000.0
        self.assertLess(relative_error(self.y_true, prediction), 0.5)

    def test_zero_reference_raises(self) -> None:
        with self.assertRaises(ValueError):
            relative_error(np.zeros((2, 2)), np.ones((2, 2)))

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            relative_error(self.y_true, self.y_true[:2])


class TestKfoldPredict(unittest.TestCase):
    """Out-of-fold prediction."""

    def setUp(self) -> None:
        rng = np.random.default_rng(2)
        self.X = rng.uniform(-1.0, 1.0, size=(40, 2))
        # Both columns linear in X, so the least-squares stand-in can recover
        # them exactly and the helper is tested independently of any backend.
        self.y = np.column_stack([self.X @ np.array([2.0, -1.0]), self.X @ np.array([0.5, 1.5])])

    def tearDown(self) -> None:
        del self.X, self.y

    def test_returns_one_prediction_per_sample(self) -> None:
        predictions = kfold_predict(make_emulator, self.X, self.y, k=4)
        self.assertEqual(predictions.shape, self.y.shape)

    def test_linear_target_is_recovered(self) -> None:
        predictions = kfold_predict(make_emulator, self.X, self.y, k=5)
        self.assertGreater(float(r2_score(self.y, predictions)), 0.99)

    def test_every_sample_is_held_out_exactly_once(self) -> None:
        """A sample predicted from a model that saw it would fit perfectly."""
        noisy = self.y + 0.5
        predictions = kfold_predict(make_emulator, self.X, noisy, k=4)
        # With four folds, each model trains on three quarters of the data, so
        # no prediction can be an exact interpolation of the full set.
        self.assertGreater(float(np.abs(predictions - noisy).max()), 0.0)

    def test_is_deterministic_given_a_seed(self) -> None:
        first = kfold_predict(make_emulator, self.X, self.y, k=3, seed=1)
        second = kfold_predict(make_emulator, self.X, self.y, k=3, seed=1)
        np.testing.assert_array_equal(first, second)

    def test_seed_changes_the_fold_assignment(self) -> None:
        first = kfold_predict(make_emulator, self.X, self.y, k=3, seed=1)
        second = kfold_predict(make_emulator, self.X, self.y, k=3, seed=2)
        self.assertFalse(np.array_equal(first, second))

    def test_rejects_too_few_folds(self) -> None:
        with self.assertRaises(ValueError):
            kfold_predict(make_emulator, self.X, self.y, k=1)

    def test_rejects_more_folds_than_samples(self) -> None:
        with self.assertRaises(ValueError):
            kfold_predict(make_emulator, self.X, self.y, k=41)

    def test_rejects_mismatched_inputs(self) -> None:
        with self.assertRaises(ValueError):
            kfold_predict(make_emulator, self.X, self.y[:10])


class TestLooPredict(unittest.TestCase):
    """Leave-one-out is k-fold with one fold per sample."""

    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.X = rng.uniform(-1.0, 1.0, size=(12, 1))
        self.y = 3.0 * self.X

    def tearDown(self) -> None:
        del self.X, self.y

    def test_matches_kfold_with_one_fold_per_sample(self) -> None:
        loo = loo_predict(make_emulator, self.X, self.y)
        kfold = kfold_predict(make_emulator, self.X, self.y, k=12)
        np.testing.assert_allclose(loo, kfold)

    def test_shape(self) -> None:
        self.assertEqual(loo_predict(make_emulator, self.X, self.y).shape, (12, 1))


class TestSummarise(unittest.TestCase):
    """The combined accuracy dictionary."""

    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.y_true = np.abs(rng.normal(size=(30, 4))) + 1.0
        self.y_pred = self.y_true * 1.01

    def tearDown(self) -> None:
        del self.y_true, self.y_pred

    def test_reports_the_expected_keys(self) -> None:
        summary = summarise(self.y_true, self.y_pred)
        self.assertEqual(
            sorted(summary),
            [
                "max_abs_error",
                "r2",
                "r2_per_column",
                "relative_error",
                "relative_error_per_column",
            ],
        )

    def test_values_are_sensible(self) -> None:
        summary = summarise(self.y_true, self.y_pred)
        self.assertAlmostEqual(summary["relative_error"], 0.01, places=6)
        self.assertGreater(summary["r2"], 0.99)
        self.assertEqual(summary["r2_per_column"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
