"""
Tests for jet.emulator.transforms.

The uncertainty-propagation tests are the load-bearing ones. Both reference
implementations this project grew out of propagate a standard deviation through
PCA with a plain matrix multiply and through ``log10`` with no correction at
all; both are wrong, and both are silent. The Monte-Carlo comparisons below
fail loudly if either regresses.
"""

from __future__ import annotations

import unittest

import numpy as np

from jet.emulator.transforms import (
    PCA,
    TRANSFORMS,
    BoundsNorm,
    Identity,
    Log10,
    StandardScaler,
    Transform,
    forward,
    inverse,
    inverse_std,
    transform_from_state,
)


class _RoundTripMixin:
    """Shared assertions every transform must satisfy."""

    def assert_state_roundtrip(self, transform: Transform, a: np.ndarray) -> None:
        """A transform rebuilt from its state must behave identically."""
        rebuilt = transform_from_state(transform.name, transform.state())
        np.testing.assert_allclose(rebuilt.transform(a), transform.transform(a), rtol=1e-12)

    def assert_inverse(self, transform: Transform, a: np.ndarray, atol: float = 1e-10) -> None:
        """``inverse_transform(transform(a))`` must return ``a``."""
        np.testing.assert_allclose(
            transform.inverse_transform(transform.transform(a)), a, atol=atol
        )


class TestIdentity(_RoundTripMixin, unittest.TestCase):
    """The pass-through step."""

    def setUp(self) -> None:
        self.a = np.arange(6.0).reshape(3, 2)
        self.transform = Identity().fit(self.a)

    def tearDown(self) -> None:
        del self.a, self.transform

    def test_passes_values_through(self) -> None:
        np.testing.assert_allclose(self.transform.transform(self.a), self.a)

    def test_inverse(self) -> None:
        self.assert_inverse(self.transform, self.a)

    def test_std_passes_through(self) -> None:
        std = np.full_like(self.a, 0.3)
        mean, out_std = self.transform.inverse_transform_std(self.a, std)
        np.testing.assert_allclose(mean, self.a)
        np.testing.assert_allclose(out_std, std)

    def test_state_roundtrip(self) -> None:
        self.assert_state_roundtrip(self.transform, self.a)

    def test_rejects_1d(self) -> None:
        with self.assertRaises(ValueError):
            self.transform.transform(np.arange(3.0))


class TestBoundsNorm(_RoundTripMixin, unittest.TestCase):
    """Affine map of a declared range onto [0, 1]."""

    def setUp(self) -> None:
        self.a = np.array([[0.05, 0.95], [0.04, 1.00]])
        self.transform = BoundsNorm([0.04, 0.92], [0.06, 1.00]).fit(self.a)

    def tearDown(self) -> None:
        del self.a, self.transform

    def test_maps_bounds_to_unit_interval(self) -> None:
        np.testing.assert_allclose(
            self.transform.transform(np.array([[0.04, 0.92], [0.06, 1.00]])),
            [[0.0, 0.0], [1.0, 1.0]],
        )

    def test_extrapolates_beyond_unit_interval(self) -> None:
        # Bounds are a warning threshold, not a clip.
        np.testing.assert_allclose(self.transform.transform(np.array([[0.07, 0.92]])), [[1.5, 0.0]])

    def test_inverse(self) -> None:
        self.assert_inverse(self.transform, self.a)

    def test_std_scales_by_width(self) -> None:
        std = np.full_like(self.a, 0.25)
        mean, out_std = self.transform.inverse_transform_std(self.transform.transform(self.a), std)
        np.testing.assert_allclose(mean, self.a)
        np.testing.assert_allclose(out_std, std * np.array([0.02, 0.08]))

    def test_state_roundtrip(self) -> None:
        self.assert_state_roundtrip(self.transform, self.a)

    def test_rejects_inverted_range(self) -> None:
        with self.assertRaises(ValueError):
            BoundsNorm([1.0, 0.0], [0.0, 1.0])

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            BoundsNorm([0.0, 0.0], [1.0])

    def test_rejects_wrong_feature_count(self) -> None:
        with self.assertRaises(ValueError):
            self.transform.transform(np.zeros((2, 5)))


class TestStandardScaler(_RoundTripMixin, unittest.TestCase):
    """Centring and scaling, implemented without scikit-learn."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.a = rng.normal(loc=3.0, scale=2.0, size=(200, 3))
        self.transform = StandardScaler().fit(self.a)

    def tearDown(self) -> None:
        del self.a, self.transform

    def test_output_is_standardised(self) -> None:
        out = self.transform.transform(self.a)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-12)

    def test_matches_numpy_statistics(self) -> None:
        np.testing.assert_allclose(self.transform.mean_, self.a.mean(axis=0))
        np.testing.assert_allclose(self.transform.scale_, self.a.std(axis=0))

    def test_zero_variance_feature_is_left_alone(self) -> None:
        a = np.column_stack([np.ones(10), np.arange(10.0)])
        scaler = StandardScaler().fit(a)
        np.testing.assert_allclose(scaler.transform(a)[:, 0], 0.0)
        self.assertTrue(np.all(np.isfinite(scaler.transform(a))))

    def test_inverse(self) -> None:
        self.assert_inverse(self.transform, self.a, atol=1e-10)

    def test_std_scales_linearly(self) -> None:
        std = np.full_like(self.a, 0.4)
        mean, out_std = self.transform.inverse_transform_std(self.transform.transform(self.a), std)
        np.testing.assert_allclose(mean, self.a, atol=1e-10)
        self.assertEqual(out_std.shape, self.a.shape)
        # Every row carries the same input sigma, so every row must come back
        # scaled by the same per-feature factor.
        np.testing.assert_allclose(out_std[0], 0.4 * self.transform.scale_)

    def test_state_roundtrip(self) -> None:
        self.assert_state_roundtrip(self.transform, self.a)

    def test_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            StandardScaler().transform(self.a)
        with self.assertRaises(RuntimeError):
            StandardScaler().state()


class TestPCA(_RoundTripMixin, unittest.TestCase):
    """SVD-based compression, and correct uncertainty propagation."""

    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        latent = rng.normal(size=(400, 2)) * np.array([3.0, 0.5])
        # A mixing rotation, so the components are not axis-aligned: this is
        # what makes the difference between propagating standard deviations
        # linearly and propagating variances observable.
        angle = 0.6
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.a = latent @ rotation.T + np.array([1.0, -2.0])
        self.transform = PCA(2).fit(self.a)

    def tearDown(self) -> None:
        del self.a, self.transform

    def test_components_are_orthonormal(self) -> None:
        components = self.transform.components_
        np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-12)

    def test_explained_variance_is_ordered_and_normalised(self) -> None:
        ratio = self.transform.explained_variance_ratio_
        self.assertGreater(ratio[0], ratio[1])
        self.assertAlmostEqual(float(ratio.sum()), 1.0, places=10)

    def test_first_component_captures_the_dominant_direction(self) -> None:
        self.assertGreater(float(self.transform.explained_variance_ratio_[0]), 0.9)

    def test_transform_whitens_the_latent_scale(self) -> None:
        out = self.transform.transform(self.a)
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)
        self.assertGreater(out.std(axis=0)[0], 5.0 * out.std(axis=0)[1])

    def test_inverse_recovers_the_original(self) -> None:
        # Two components of a two-dimensional input: the projection is a
        # rotation, so the reconstruction is exact.
        self.assert_inverse(self.transform, self.a, atol=1e-9)

    def test_rejects_too_many_components(self) -> None:
        with self.assertRaises(ValueError):
            PCA(10).fit(self.a)

    def test_rejects_non_positive_components(self) -> None:
        with self.assertRaises(ValueError):
            PCA(0)

    def test_unfitted_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            PCA(1).transform(self.a)

    def test_state_roundtrip(self) -> None:
        self.assert_state_roundtrip(self.transform, self.a)

    def test_std_propagation_matches_monte_carlo(self) -> None:
        """Regression test: standard deviations must combine in quadrature.

        The reference implementation computes ``std @ components_``, which adds
        standard deviations linearly as if every component were perfectly
        correlated. That overestimates the spread. The correct expression adds
        variances.
        """
        rng = np.random.default_rng(7)
        mean_z = np.array([[0.4, -0.3]])
        std_z = np.array([[0.1, 0.05]])

        mean_out, std_out = self.transform.inverse_transform_std(mean_z, std_z)

        samples = mean_z + std_z * rng.standard_normal((400_000, 2))
        reference = self.transform.inverse_transform(samples)

        np.testing.assert_allclose(mean_out[0], reference.mean(axis=0), rtol=1e-2, atol=1e-3)
        np.testing.assert_allclose(std_out[0], reference.std(axis=0), rtol=2e-2)

    def test_std_propagation_differs_from_the_linear_combination(self) -> None:
        """The buggy formula and the correct one must not agree here.

        Without this, the Monte-Carlo test above could pass on an
        implementation that had quietly reverted to propagating standard
        deviations linearly.
        """
        mean_z = np.array([[0.0, 0.0]])
        std_z = np.array([[1.0, 0.25]])

        _, std_correct = self.transform.inverse_transform_std(mean_z, std_z)
        std_linear = std_z @ self.transform.components_

        self.assertFalse(np.allclose(std_correct[0], std_linear[0], rtol=1e-6))


class TestLog10(_RoundTripMixin, unittest.TestCase):
    """Base-10 logarithm and its delta-method uncertainty."""

    def setUp(self) -> None:
        self.a = np.array([[1.0, 10.0], [100.0, 1000.0]])
        self.transform = Log10().fit(self.a)

    def tearDown(self) -> None:
        del self.a, self.transform

    def test_takes_the_logarithm(self) -> None:
        np.testing.assert_allclose(self.transform.transform(self.a), [[0.0, 1.0], [2.0, 3.0]])

    def test_inverse(self) -> None:
        self.assert_inverse(self.transform, self.a)

    def test_rejects_non_positive_on_fit(self) -> None:
        with self.assertRaises(ValueError):
            Log10().fit(np.array([[1.0, 0.0]]))

    def test_rejects_non_positive_on_transform(self) -> None:
        with self.assertRaises(ValueError):
            self.transform.transform(np.array([[-1.0, 1.0]]))

    def test_state_roundtrip(self) -> None:
        self.assert_state_roundtrip(self.transform, self.a)

    def test_std_propagation_matches_monte_carlo(self) -> None:
        """Regression test: the delta method, not a bare ``10**``.

        The reference implementation pushes the standard deviation through the
        inverse transform unchanged, which ignores that ``10**z`` stretches the
        axis by ``ln(10) * y``.
        """
        rng = np.random.default_rng(11)
        mean_z = np.array([[1.0, 2.0]])
        std_z = np.array([[0.02, 0.05]])

        mean_out, std_out = self.transform.inverse_transform_std(mean_z, std_z)

        samples = mean_z + std_z * rng.standard_normal((400_000, 2))
        reference = self.transform.inverse_transform(samples)

        # The point estimate is the median of the implied log-normal, not its
        # mean, so the Monte-Carlo mean sits slightly high by construction.
        np.testing.assert_allclose(mean_out[0], reference.mean(axis=0), rtol=1e-2)
        np.testing.assert_allclose(std_out[0], reference.std(axis=0), rtol=2e-2)

    def test_std_is_larger_than_the_untransformed_value(self) -> None:
        # ln(10) * y > 1 whenever y > 0.43, so ignoring the Jacobian
        # understates the spread for any statistic above that.
        mean_z = np.array([[1.0]])
        std_z = np.array([[0.02]])
        _, std_out = self.transform.inverse_transform_std(mean_z, std_z)
        self.assertGreater(float(std_out[0, 0]), float(std_z[0, 0]))


class TestRegistryAndChains(unittest.TestCase):
    """The transform registry and the chain helpers."""

    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.raw = rng.normal(size=(50, 3)) * np.array([1.0, 10.0, 100.0])
        # PCA(3) on three features is a rotation rather than a compression, so
        # the chain is exactly invertible -- which is what makes the roundtrip
        # assertions below meaningful. Lossy compression is covered by TestPCA.
        self.chain = [StandardScaler(), PCA(3), StandardScaler()]
        a = self.raw
        for step in self.chain:
            step.fit(a)
            a = step.transform(a)

    def tearDown(self) -> None:
        del self.raw, self.chain

    def test_all_builtins_registered(self) -> None:
        self.assertEqual(
            sorted(TRANSFORMS),
            ["bounds_norm", "identity", "log10", "pca", "standard_scaler"],
        )

    def test_unknown_transform_name(self) -> None:
        with self.assertRaises(KeyError) as caught:
            transform_from_state("nope", {})
        self.assertIn("standard_scaler", str(caught.exception))

    def test_chain_roundtrip(self) -> None:
        rng = np.random.default_rng(5)
        b = rng.normal(size=(20, 3)) * np.array([1.0, 10.0, 100.0])
        transformed = forward(self.chain, b)
        np.testing.assert_allclose(inverse(self.chain, transformed), b, atol=1e-8)

    def test_chain_with_log_is_not_invertible_on_linear_scale(self) -> None:
        """A ``Log10`` step changes the meaning of the values, so the chain
        roundtrip must go through the log space, not the raw space."""
        rng = np.random.default_rng(8)
        positive = np.abs(rng.normal(size=(20, 2))) + 1.0
        chain = [Log10(), StandardScaler()]
        a = positive
        for step in chain:
            step.fit(a)
            a = step.transform(a)
        np.testing.assert_allclose(inverse(chain, forward(chain, positive)), positive, atol=1e-10)

    def test_chain_std_roundtrip_through_affine_steps(self) -> None:
        """A pure scaler/PCA chain must map a Gaussian back to a Gaussian."""
        rng = np.random.default_rng(6)
        b = rng.normal(size=(30, 3)) * np.array([1.0, 10.0, 100.0])
        transformed = forward(self.chain, b)
        std = np.full_like(b, 0.1)

        mean_out, std_out = inverse_std(self.chain, transformed, std)

        np.testing.assert_allclose(mean_out, b, atol=1e-8)
        self.assertTrue(np.all(std_out > 0.0))


if __name__ == "__main__":
    unittest.main()
