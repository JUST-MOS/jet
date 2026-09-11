"""
Tests for the regression backends.

The Gaussian-process tests need scikit-learn and the neural-network tests need
PyTorch, so both are skipped when the relevant extra is absent. That is not
merely a convenience: ``jet[gp]`` and ``jet[nn]`` being optional is a promise
the package makes, and a suite that failed without them would break it.
"""

from __future__ import annotations

import unittest

import numpy as np

from jet.emulator.backends import BACKENDS, backend_from_state, get_backend
from jet.emulator.backends.gp import KERNELS, GPBackend

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_SKLEARN = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_TORCH = False


def smooth_dataset(n_samples: int = 60, n_features: int = 2, n_targets: int = 2, seed: int = 0):
    """Build a small, smooth, noise-free regression problem.

    Smoothness matters: a Gaussian process with an RBF kernel should
    interpolate such a target to high accuracy, so a poor score means a real
    defect rather than an under-parameterised model.

    Returns
    -------
    tuple of ndarray
        ``(X, Y)`` with shapes ``(n_samples, n_features)`` and
        ``(n_samples, n_targets)``.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    Y = np.empty((n_samples, n_targets))
    for j in range(n_targets):
        Y[:, j] = np.sin((j + 1) * X[:, 0]) + 0.3 * X[:, 1] ** 2 + 0.1 * j
    return X, Y


class TestBackendRegistry(unittest.TestCase):
    """Both built-in backends must register themselves on import."""

    def test_builtins_registered(self) -> None:
        self.assertEqual(sorted(BACKENDS), ["gp", "nn"])

    def test_get_backend_by_name(self) -> None:
        self.assertEqual(get_backend("gp").name, "gp")
        self.assertEqual(get_backend("nn").name, "nn")

    def test_get_backend_forwards_kwargs(self) -> None:
        backend = get_backend("gp", alpha=1e-6)
        self.assertAlmostEqual(backend.alpha, 1e-6)

    def test_unknown_backend_lists_registered(self) -> None:
        with self.assertRaises(KeyError) as caught:
            get_backend("random-forest")
        self.assertIn("gp", str(caught.exception))

    def test_backend_from_state_unknown(self) -> None:
        with self.assertRaises(KeyError):
            backend_from_state("nope", {})


@unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn (pip install jet[gp])")
class TestGPBackend(unittest.TestCase):
    """Training with scikit-learn, prediction with NumPy."""

    @classmethod
    def setUpClass(cls) -> None:
        # Fitting with the optimiser enabled takes several seconds, and every
        # test here only reads the result, so one fit serves the whole class.
        cls.X, cls.Y = smooth_dataset()
        cls.backend = get_backend("gp")
        cls.backend.fit(cls.X, cls.Y)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.X, cls.Y, cls.backend

    def test_n_out_matches_target_columns(self) -> None:
        self.assertEqual(self.backend.n_out, self.Y.shape[1])

    def test_one_length_scale_per_input_and_output(self) -> None:
        self.assertEqual(self.backend.length_scale_.shape, (2, 2))  # (n_out, n_features)
        self.assertEqual(self.backend.constant_value_.shape, (2,))  # (n_out,)

    def test_predict_shapes(self) -> None:
        mean, std = self.backend.predict(self.X[:5])
        self.assertEqual(mean.shape, (5, 2))
        self.assertEqual(std.shape, (5, 2))

    def test_interpolates_the_training_points(self) -> None:
        mean, _ = self.backend.predict(self.X)
        np.testing.assert_allclose(mean, self.Y, atol=1e-4)

    def test_predicts_held_out_points_accurately(self) -> None:
        X_test, Y_test = smooth_dataset(n_samples=25, seed=99)
        mean, _ = self.backend.predict(X_test)
        self.assertGreater(
            1.0 - ((Y_test - mean) ** 2).sum() / ((Y_test - Y_test.mean()) ** 2).sum(), 0.99
        )

    def test_uncertainty_grows_away_from_the_training_data(self) -> None:
        _, std_inside = self.backend.predict(np.array([[0.5, 0.5]]))
        _, std_outside = self.backend.predict(np.array([[5.0, 5.0]]))
        self.assertGreater(float(std_outside.sum()), float(std_inside.sum()))

    def test_uncertainty_is_positive(self) -> None:
        _, std = self.backend.predict(self.X)
        self.assertTrue(np.all(std > 0.0))

    def test_rejects_wrong_feature_count(self) -> None:
        with self.assertRaises(ValueError):
            self.backend.predict(np.zeros((3, 5)))

    def test_state_roundtrip_preserves_predictions(self) -> None:
        rebuilt = backend_from_state("gp", self.backend.state())
        mean_a, std_a = self.backend.predict(self.X[:7])
        mean_b, std_b = rebuilt.predict(self.X[:7])
        np.testing.assert_allclose(mean_a, mean_b, rtol=1e-12)
        np.testing.assert_allclose(std_a, std_b, rtol=1e-10)

    def test_state_holds_no_large_derived_arrays(self) -> None:
        """The Cholesky factors are deliberately not stored."""
        state = self.backend.state()
        self.assertNotIn("L", state)
        self.assertEqual(state["X_train"].shape, self.X.shape)

    def test_unfitted_predict_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            get_backend("gp").predict(np.zeros((1, 2)))

    def test_rejects_invalid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("gp", alpha=0.0)

    def test_rejects_single_sample(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("gp").fit(np.zeros((1, 2)), np.zeros((1, 1)))

    def test_optimizer_can_be_disabled(self) -> None:
        """``optimizer=None`` reproduces the reference's fixed hyperparameters."""
        X, Y = smooth_dataset(n_samples=30)
        frozen = get_backend("gp", optimizer=None)
        frozen.fit(X, Y)
        np.testing.assert_allclose(frozen.constant_value_, np.ones(2), rtol=1e-6)

    def test_rejects_an_unknown_kernel(self) -> None:
        with self.assertRaises(ValueError) as caught:
            GPBackend(kernel="matern32")
        self.assertIn("matern52", str(caught.exception))

    def test_kernel_name_survives_a_state_roundtrip(self) -> None:
        for kernel in KERNELS:
            backend = GPBackend(kernel=kernel, optimizer=None).fit(self.X, self.Y)
            self.assertEqual(backend.state()["kernel"].item(), kernel)
            restored = backend_from_state(
                "gp", {key: np.asarray(value) for key, value in backend.state().items()}
            )
            self.assertEqual(restored.kernel, kernel)

    def test_a_bundle_without_a_kernel_key_reads_back_as_rbf(self) -> None:
        """Bundles written before the kernel was selectable must still load."""
        state = {key: np.asarray(value) for key, value in self.backend.state().items()}
        del state["kernel"]
        self.assertEqual(backend_from_state("gp", state).kernel, "rbf")

    def test_history_records_the_fit(self) -> None:
        history = self.backend.history
        self.assertEqual(history["n_out"], 2)
        self.assertEqual(history["log_marginal_likelihood"].shape, (2,))


@unittest.skipUnless(HAS_TORCH, "requires PyTorch (pip install jet[nn])")
class TestNNBackend(unittest.TestCase):
    """Training with PyTorch, prediction with NumPy, no uncertainty."""

    @classmethod
    def setUpClass(cls) -> None:
        # Training is the expensive part; the tests only inspect the trained
        # model, so the ensemble is fitted once for the class.
        cls.X, cls.Y = smooth_dataset(n_samples=80)
        cls.backend = get_backend(
            "nn",
            hidden_dims=(32, 32),
            n_epochs=120,
            batch_size=16,
            lr=5e-3,
            validation_fraction=0.2,
            verbose=False,
        )
        cls.backend.fit(cls.X, cls.Y)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.X, cls.Y, cls.backend

    def test_predict_shapes(self) -> None:
        mean, std = self.backend.predict(self.X[:5])
        self.assertEqual(mean.shape, (5, 2))
        self.assertIsNone(std)

    def test_predicts_held_out_points(self) -> None:
        X_test, Y_test = smooth_dataset(n_samples=25, seed=99)
        mean, _ = self.backend.predict(X_test)
        r2 = 1.0 - ((Y_test - mean) ** 2).sum() / ((Y_test - Y_test.mean()) ** 2).sum()
        self.assertGreater(r2, 0.9)

    def test_numpy_forward_matches_torch(self) -> None:
        """The NumPy reimplementation must agree with PyTorch.

        Prediction is deliberately reimplemented so that a saved bundle can be
        served without PyTorch installed. That is only safe if the two paths
        compute the same function, so they are compared directly.
        """
        torch = __import__("torch")
        member = self.backend._weights[0]

        # Rebuild the same architecture in torch, then load the stored NumPy
        # weights into it.
        linears: list[torch.nn.Linear] = []
        previous = member[0][0].shape[1]
        for weight, _ in member[:-1]:
            linears.append(torch.nn.Linear(previous, weight.shape[0]))
            previous = weight.shape[0]
        linears.append(torch.nn.Linear(previous, member[-1][0].shape[0]))

        with torch.no_grad():
            for layer, (weight, bias) in zip(linears, member, strict=True):
                layer.weight.copy_(torch.as_tensor(weight, dtype=layer.weight.dtype))
                layer.bias.copy_(torch.as_tensor(bias, dtype=layer.bias.dtype))

            reference = self.X[:6]
            a = torch.as_tensor(reference, dtype=torch.float32)
            for index, layer in enumerate(linears):
                a = layer(a)
                if index < len(linears) - 1:
                    a = torch.nn.functional.silu(a)
            expected = a.numpy()

        mean, _ = self.backend.predict(reference)
        np.testing.assert_allclose(mean, expected, rtol=1e-5, atol=1e-6)

    def test_state_roundtrip_preserves_predictions(self) -> None:
        rebuilt = backend_from_state("nn", self.backend.state())
        mean_a, _ = self.backend.predict(self.X[:7])
        mean_b, _ = rebuilt.predict(self.X[:7])
        np.testing.assert_allclose(mean_a, mean_b, rtol=1e-12)

    def test_state_roundtrip_recovers_architecture(self) -> None:
        rebuilt = backend_from_state("nn", self.backend.state())
        self.assertEqual(rebuilt.hidden_dims, (32, 32))
        self.assertEqual(rebuilt.n_members, 1)

    def test_ensemble_averages_members(self) -> None:
        X, Y = smooth_dataset(n_samples=40)
        backend = get_backend(
            "nn",
            hidden_dims=(16,),
            n_epochs=40,
            batch_size=8,
            ensemble_size=3,
            seed=1,
        )
        backend.fit(X, Y)
        self.assertEqual(backend.n_members, 3)
        mean, std = backend.predict(X[:4])
        self.assertEqual(mean.shape, (4, 2))
        self.assertIsNone(std)

    def test_rejects_unknown_activation(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("nn", activation="gelu")

    def test_rejects_bad_ensemble_size(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("nn", ensemble_size=0)

    def test_rejects_bad_validation_fraction(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("nn", validation_fraction=1.0)

    def test_unfitted_predict_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            get_backend("nn").predict(np.zeros((1, 2)))

    def test_history_has_one_entry_per_member(self) -> None:
        history = self.backend.history
        self.assertEqual(len(history), 1)
        self.assertIn("train_loss", history[0])

    def test_rejects_wrong_feature_count(self) -> None:
        with self.assertRaises(ValueError):
            self.backend.predict(np.zeros((3, 5)))


if __name__ == "__main__":
    unittest.main()
