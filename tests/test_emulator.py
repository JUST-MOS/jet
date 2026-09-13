"""
Tests for the Emulator class and the on-disk bundle.

Validation and chain-assembly tests run on any installation. Anything that
actually fits a model is skipped without scikit-learn, matching the promise
that the training backend is an optional extra.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jet.emulator import Emulator, resolve_weights_path
from jet.emulator.backends import get_backend
from jet.emulator.bundle import load_bundle
from jet.emulator.transforms import (
    PCA,
    BoundsNorm,
    Log10,
    StandardScaler,
    transform_from_state,
)
from jet.spec import DataVectorSpec, Param, ParameterSpec

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_SKLEARN = False


def fast_gp(**kwargs):
    """A Gaussian-process backend with hyperparameter optimisation switched off.

    scikit-learn's L-BFGS-B search over the log-marginal-likelihood dominates
    the runtime of this module and contributes nothing to what it tests: the
    subject here is the wiring between specs, transforms, backends and bundles,
    not the quality of a fitted kernel. The optimised path is exercised in
    test_backends.py, where a single fit is enough to cover it.

    Returns
    -------
    jet.emulator.backends.Backend
        An unfitted backend with ``optimizer=None``.
    """
    return get_backend("gp", optimizer=None, **kwargs)


def toy_problem(n_samples: int = 50, seed: int = 0):
    """Return ``(x_spec, y_spec, X, y)`` for a small, smooth problem."""
    rng = np.random.default_rng(seed)
    x_spec = ParameterSpec(
        [
            Param("Omegab", bounds=(0.04, 0.06), block="cosmo"),
            Param("ns", bounds=(0.92, 1.00), block="cosmo"),
            Param("logMcut", bounds=(12.0, 13.8), block="hod"),
        ]
    )
    y_spec = DataVectorSpec("wp", n_bins=3)
    X = np.column_stack(
        [
            rng.uniform(0.04, 0.06, n_samples),
            rng.uniform(0.92, 1.00, n_samples),
            rng.uniform(12.0, 13.8, n_samples),
        ]
    )
    y = np.column_stack(
        [
            np.sin(20.0 * X[:, 0]) + 0.5 * X[:, 1],
            X[:, 0] * X[:, 2],
            np.cos(5.0 * X[:, 1]) + X[:, 0] ** 2,
        ]
    )
    return x_spec, y_spec, X, y


class TestWeightsPathResolution(unittest.TestCase):
    """Resolving a weight file given by name rather than by path."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_existing_path_is_returned_as_is(self) -> None:
        target = self.root / "model.npz"
        target.write_bytes(b"")
        self.assertEqual(resolve_weights_path(target), target)

    def test_name_is_resolved_through_the_environment(self) -> None:
        (self.root / "model.npz").write_bytes(b"")
        with patch.dict(os.environ, {"JET_DATA_DIR": str(self.root)}):
            self.assertEqual(resolve_weights_path("model.npz"), self.root / "model.npz")

    def test_suffix_is_appended(self) -> None:
        (self.root / "model.npz").write_bytes(b"")
        with patch.dict(os.environ, {"JET_DATA_DIR": str(self.root)}):
            self.assertEqual(resolve_weights_path("model"), self.root / "model.npz")

    def test_missing_file_lists_candidates_and_hints(self) -> None:
        with patch.dict(os.environ, {"JET_DATA_DIR": str(self.root), "JET_NO_AUTO_FETCH": "1"}):
            with self.assertRaises(FileNotFoundError) as caught:
                resolve_weights_path("missing")
        message = str(caught.exception)
        self.assertIn("missing", message)
        self.assertIn("JET_DATA_DIR", message)

    def test_missing_file_without_environment_says_so(self) -> None:
        with patch.dict(os.environ, {"JET_NO_AUTO_FETCH": "1"}, clear=True):
            with self.assertRaises(FileNotFoundError) as caught:
                resolve_weights_path("missing")
        self.assertIn("Set JET_DATA_DIR", str(caught.exception))


class TestReleaseAutoFetch(unittest.TestCase):
    """Automatic download of missing weights from GitHub Release."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_fetches_a_missing_bare_name(self) -> None:
        def _fake_fetch(name: str, dest_dir: Path) -> Path:
            target = Path(dest_dir) / name
            target.write_bytes(b"downloaded")
            return target

        with (
            patch.dict(os.environ, {"JET_DATA_DIR": str(self.root)}),
            patch("jet.emulator.emulator._fetch_release_weights", side_effect=_fake_fetch) as fetch,
        ):
            resolved = resolve_weights_path("model.npz")
        self.assertEqual(resolved, self.root / "model.npz")
        self.assertEqual((self.root / "model.npz").read_bytes(), b"downloaded")
        fetch.assert_called_once_with("model.npz", self.root)

    def test_fetch_respects_the_suffix(self) -> None:
        def _fake_fetch(name: str, dest_dir: Path) -> Path:
            target = Path(dest_dir) / name
            target.write_bytes(b"x")
            return target

        with (
            patch.dict(os.environ, {"JET_DATA_DIR": str(self.root)}),
            patch("jet.emulator.emulator._fetch_release_weights", side_effect=_fake_fetch) as fetch,
        ):
            resolved = resolve_weights_path("model")
        self.assertEqual(resolved, self.root / "model.npz")
        fetch.assert_called_once_with("model.npz", self.root)

    def test_fetch_failure_falls_back_to_file_not_found(self) -> None:
        with (
            patch.dict(os.environ, {"JET_DATA_DIR": str(self.root)}),
            patch("jet.emulator.emulator._fetch_release_weights", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaises(FileNotFoundError) as caught:
                resolve_weights_path("missing")
        message = str(caught.exception)
        self.assertIn("Automatic download", message)
        self.assertIn("boom", message)

    def test_disabled_fetch_does_not_touch_the_network(self) -> None:
        with (
            patch.dict(os.environ, {"JET_DATA_DIR": str(self.root), "JET_NO_AUTO_FETCH": "1"}),
            patch("jet.emulator.emulator._fetch_release_weights") as fetch,
        ):
            with self.assertRaises(FileNotFoundError):
                resolve_weights_path("missing")
        fetch.assert_not_called()

    def test_an_absolute_path_never_fetches(self) -> None:
        missing = self.root / "definitely-not-here.npz"
        with (
            patch.dict(os.environ, {"JET_DATA_DIR": str(self.root)}),
            patch("jet.emulator.emulator._fetch_release_weights") as fetch,
        ):
            with self.assertRaises(FileNotFoundError):
                resolve_weights_path(str(missing))
        fetch.assert_not_called()


class TestEmulatorConstruction(unittest.TestCase):
    """Argument validation and transform-chain assembly, without fitting."""

    def setUp(self) -> None:
        self.x_spec, self.y_spec, self.X, self.y = toy_problem()

    def tearDown(self) -> None:
        del self.x_spec, self.y_spec, self.X, self.y

    def test_rejects_non_spec_arguments(self) -> None:
        with self.assertRaises(TypeError):
            Emulator(["not a spec"], self.y_spec)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Emulator(self.x_spec, "not a spec")  # type: ignore[arg-type]

    def test_rejects_unknown_backend(self) -> None:
        with self.assertRaises(KeyError):
            Emulator(self.x_spec, self.y_spec, backend="random-forest")

    def test_rejects_bad_normalisation_mode(self) -> None:
        with self.assertRaises(ValueError):
            Emulator(self.x_spec, self.y_spec, x_normalize="magic")

    def test_rejects_backend_kwargs_with_an_instance(self) -> None:
        with self.assertRaises(ValueError):
            Emulator(
                self.x_spec,
                self.y_spec,
                backend=get_backend("gp"),
                backend_kwargs={"alpha": 1e-5},
            )

    def test_default_chain_is_a_single_scaler(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec)
        chain = emulator._build_x_chain()
        self.assertEqual([step.name for step in chain], ["standard_scaler"])

    def test_bounds_chain(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec, x_normalize="bounds")
        chain = emulator._build_x_chain()
        self.assertEqual([step.name for step in chain], ["bounds_norm"])
        self.assertIsInstance(chain[0], BoundsNorm)

    def test_bounds_plus_standardize_chain(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec, x_normalize="bounds+standardize")
        self.assertEqual(
            [step.name for step in emulator._build_x_chain()],
            ["bounds_norm", "standard_scaler"],
        )

    def test_none_chain_is_empty(self) -> None:
        self.assertEqual(
            Emulator(self.x_spec, self.y_spec, x_normalize="none")._build_x_chain(), []
        )

    def test_bounds_chain_rejects_unbounded_spec(self) -> None:
        emulator = Emulator(ParameterSpec([Param("free")]), self.y_spec, x_normalize="bounds")
        with self.assertRaises(ValueError) as caught:
            emulator._build_x_chain()
        self.assertIn("free", str(caught.exception))

    def test_log_data_vector_adds_a_log_step(self) -> None:
        emulator = Emulator(self.x_spec, DataVectorSpec("wp", n_bins=3, log=True))
        self.assertEqual(
            [step.name for step in emulator._build_y_chain()],
            ["log10", "standard_scaler"],
        )

    def test_no_log_step_without_the_flag(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec)
        self.assertEqual([step.name for step in emulator._build_y_chain()], ["standard_scaler"])

    def test_pca_adds_two_steps(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec, n_pca=2)
        self.assertEqual(
            [step.name for step in emulator._build_y_chain()],
            ["standard_scaler", "pca", "standard_scaler"],
        )

    def test_rejects_bad_n_pca(self) -> None:
        with self.assertRaises(ValueError):
            Emulator(self.x_spec, self.y_spec, n_pca=0)

    def test_unfitted_predict_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            Emulator(self.x_spec, self.y_spec).predict(self.X)

    def test_spec_hashes_are_exposed(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec)
        self.assertEqual(emulator.x_spec_hash, self.x_spec.hash())
        self.assertEqual(emulator.y_spec_hash, self.y_spec.hash())


@unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn (pip install jet[gp])")
class TestEmulatorFitting(unittest.TestCase):
    """Training, prediction, persistence and self-checks."""

    def setUp(self) -> None:
        self.x_spec, self.y_spec, self.X, self.y = toy_problem()
        self.emulator = Emulator(self.x_spec, self.y_spec, backend=fast_gp())
        self.emulator.fit(self.X, self.y)
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        del self.x_spec, self.y_spec, self.X, self.y, self.emulator

    def test_is_fitted(self) -> None:
        self.assertTrue(self.emulator.is_fitted)

    def test_fit_returns_self(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec, backend=fast_gp())
        self.assertIs(emulator.fit(self.X, self.y), emulator)

    def test_predict_shapes(self) -> None:
        mean, std = self.emulator.predict(self.X[:4])
        self.assertEqual(mean.shape, (4, 3))
        self.assertEqual(std.shape, (4, 3))

    def test_accepts_a_single_point_as_1d(self) -> None:
        mean, _ = self.emulator.predict(self.X[0])
        self.assertEqual(mean.shape, (1, 3))

    def test_predict_without_std(self) -> None:
        self.assertEqual(self.emulator.predict(self.X[:4], return_std=False).shape, (4, 3))

    def test_recovers_the_training_data(self) -> None:
        mean, _ = self.emulator.predict(self.X)
        np.testing.assert_allclose(mean, self.y, atol=1e-3)

    def test_rejects_wrong_input_width(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.emulator.predict(np.zeros((2, 5)))
        self.assertIn("Omegab", str(caught.exception))

    def test_rejects_mismatched_y_width(self) -> None:
        with self.assertRaises(ValueError):
            Emulator(self.x_spec, self.y_spec, backend=fast_gp()).fit(self.X, self.y[:, :2])

    def test_rejects_mismatched_sample_counts(self) -> None:
        with self.assertRaises(ValueError):
            Emulator(self.x_spec, self.y_spec, backend=fast_gp()).fit(self.X[:10], self.y[:5])

    def test_extrapolation_warns(self) -> None:
        outside = self.X[:1].copy()
        outside[0, 0] = 0.5  # far outside the (0.04, 0.06) bound
        with self.assertWarns(RuntimeWarning) as caught:
            self.emulator.predict(outside)
        self.assertIn("Omegab", str(caught.warning))

    def test_extrapolation_warning_can_be_disabled(self) -> None:
        emulator = Emulator(
            self.x_spec, self.y_spec, backend=fast_gp(), warn_on_extrapolation=False
        )
        emulator.fit(self.X, self.y)
        outside = self.X[:1].copy()
        outside[0, 0] = 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emulator.predict(outside)

    def test_bounds_slack_suppresses_a_marginal_warning(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec, backend=fast_gp(), bounds_slack=0.5)
        emulator.fit(self.X, self.y)
        marginal = self.X[:1].copy()
        marginal[0, 0] = 0.061  # one tenth of the range beyond the bound
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            emulator.predict(marginal)

    def test_save_and_load_roundtrip(self) -> None:
        path = self.emulator.save(self.dir / "wp.gp")
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix, ".npz")

        restored = Emulator.load(path)
        mean_a, std_a = self.emulator.predict(self.X[:5])
        mean_b, std_b = restored.predict(self.X[:5])
        np.testing.assert_allclose(mean_a, mean_b, rtol=1e-12)
        np.testing.assert_allclose(std_a, std_b, rtol=1e-10)

    def test_save_appends_the_suffix(self) -> None:
        self.assertEqual(self.emulator.save(self.dir / "plain").name, "plain.npz")

    def test_roundtrip_preserves_specs_and_chains(self) -> None:
        path = self.emulator.save(self.dir / "wp.gp")
        restored = Emulator.load(path)
        self.assertEqual(restored.x_spec, self.x_spec)
        self.assertEqual(restored.y_spec, self.y_spec)
        # No log flag and no PCA on this data vector, so each chain is a single
        # standardisation step.
        self.assertEqual([step.name for step in restored.x_chain], ["standard_scaler"])
        self.assertEqual([step.name for step in restored.y_chain], ["standard_scaler"])
        self.assertIsInstance(restored.y_chain[0], StandardScaler)

    def test_verify_accepts_matching_specs(self) -> None:
        self.emulator.verify(x_spec=self.x_spec, y_spec=self.y_spec)

    def test_verify_rejects_a_different_input_spec(self) -> None:
        other = ParameterSpec(
            [
                Param("Omegab", bounds=(0.04, 0.06)),
                Param("ns", bounds=(0.92, 1.00)),
                Param("logM1", bounds=(12.0, 15.0)),
            ]
        )
        with self.assertRaises(ValueError) as caught:
            self.emulator.verify(x_spec=other)
        self.assertIn("input spec mismatch", str(caught.exception))

    def test_verify_rejects_a_different_output_spec(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.emulator.verify(y_spec=DataVectorSpec("wp", n_bins=15))
        self.assertIn("output spec mismatch", str(caught.exception))

    def test_load_can_verify_immediately(self) -> None:
        path = self.emulator.save(self.dir / "wp.gp")
        Emulator.load(path, x_spec=self.x_spec, y_spec=self.y_spec)
        with self.assertRaises(ValueError):
            Emulator.load(path, y_spec=DataVectorSpec("wp", n_bins=99))

    def test_manifest_records_provenance(self) -> None:
        path = self.emulator.save(self.dir / "wp.gp", extra={"training_set": "kunsim-v1"})
        restored = Emulator.load(path)
        self.assertEqual(restored.manifest["backend"], "gp")
        self.assertEqual(restored.manifest["extra"]["training_set"], "kunsim-v1")
        self.assertEqual(restored.manifest["x_spec_hash"], self.x_spec.hash())

    def test_state_is_plain_arrays_no_pickle(self) -> None:
        """A bundle must load with ``allow_pickle=False``."""
        path = self.emulator.save(self.dir / "wp.gp")
        with np.load(path, allow_pickle=False) as archive:
            self.assertIn("manifest", archive)
            self.assertIn("backend.gp.X_train", archive)

    def test_load_rejects_a_foreign_npz(self) -> None:
        foreign = self.dir / "foreign.npz"
        np.savez(foreign, foo=np.arange(3))
        with self.assertRaises(ValueError) as caught:
            Emulator.load(foreign)
        self.assertIn("not a jet emulator bundle", str(caught.exception))

    def test_bundle_carries_extra_arrays(self) -> None:
        """A model can ship a supporting grid in its own file."""
        path = self.emulator.save(self.dir / "wp.gp", extra_arrays={"k": np.arange(4.0)})
        extra = load_bundle(path)["extra_arrays"]
        np.testing.assert_array_equal(extra["k"], np.arange(4.0))
        # The loaded emulator exposes the same arrays on ``extra_arrays``,
        # with whatever keys the writer chose.
        restored = Emulator.load(path)
        np.testing.assert_array_equal(restored.extra_arrays["k"], np.arange(4.0))

    def test_load_without_extra_arrays_has_empty_dict(self) -> None:
        path = self.emulator.save(self.dir / "wp.plain.gp")
        self.assertEqual(Emulator.load(path).extra_arrays, {})

    def test_saved_bundle_honours_the_umask(self) -> None:
        """A model written to shared storage must be readable by the group.

        The file is assembled in a NamedTemporaryFile, which is created 0600,
        and os.replace preserves that -- so without an explicit chmod every
        bundle jet writes is private to its author.
        """
        path = self.emulator.save(self.dir / "wp.mode.gp")
        umask = os.umask(0)
        os.umask(umask)
        self.assertEqual(path.stat().st_mode & 0o777, 0o666 & ~umask)

    def test_pca_chain_roundtrip(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec, backend=fast_gp(), n_pca=2)
        emulator.fit(self.X, self.y)
        path = emulator.save(self.dir / "wp.pca.gp")
        restored = Emulator.load(path)
        self.assertEqual(restored.n_pca, 2)
        self.assertIsInstance(restored.y_chain[1], PCA)
        mean_a, _ = emulator.predict(self.X[:4])
        mean_b, _ = restored.predict(self.X[:4])
        np.testing.assert_allclose(mean_a, mean_b, rtol=1e-10)

    def test_log_data_vector_roundtrip(self) -> None:
        y_spec = DataVectorSpec("wp", n_bins=3, log=True)
        y = np.abs(self.y) + 1.0
        emulator = Emulator(self.x_spec, y_spec, backend=fast_gp())
        emulator.fit(self.X, y)

        path = emulator.save(self.dir / "wp.log.gp")
        restored = Emulator.load(path)
        self.assertIsInstance(restored.y_chain[0], Log10)

        mean, std = restored.predict(self.X[:4])
        # Back in linear units, so the prediction must be positive and its
        # uncertainty must have been scaled by the log Jacobian.
        self.assertTrue(np.all(mean > 0.0))
        self.assertTrue(np.all(std > 0.0))

    def test_load_resolves_through_the_environment(self) -> None:
        self.emulator.save(self.dir / "shared.npz")
        with patch.dict(os.environ, {"JET_DATA_DIR": str(self.dir)}):
            restored = Emulator.load("shared.npz")
        np.testing.assert_allclose(
            restored.predict(self.X[:3], return_std=False),
            self.emulator.predict(self.X[:3], return_std=False),
            rtol=1e-12,
        )

    def test_repr_reports_state(self) -> None:
        self.assertIn("fitted", repr(self.emulator))
        self.assertIn("unfitted", repr(Emulator(self.x_spec, self.y_spec)))


@unittest.skipUnless(HAS_SKLEARN, "requires scikit-learn (pip install jet[gp])")
class TestEmulatorChainsAreReusable(unittest.TestCase):
    """Transforms fitted during ``fit`` must survive a save/load byte-for-byte."""

    def setUp(self) -> None:
        self.x_spec, self.y_spec, self.X, self.y = toy_problem()
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        del self.x_spec, self.y_spec, self.X, self.y

    def test_chain_state_is_identical_after_reload(self) -> None:
        emulator = Emulator(self.x_spec, self.y_spec, backend=fast_gp(), n_pca=2)
        emulator.fit(self.X, self.y)
        before = emulator.y_chain[2].state()

        restored = Emulator.load(emulator.save(self.dir / "m.npz"))
        after = restored.y_chain[2].state()

        self.assertEqual(set(before), set(after))
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])

    def test_identity_transform_survives_the_registry(self) -> None:
        rebuilt = transform_from_state("identity", {"width": np.asarray(3)})
        np.testing.assert_allclose(rebuilt.transform(np.eye(3)), np.eye(3))


if __name__ == "__main__":
    unittest.main()
