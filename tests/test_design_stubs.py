"""
Tests for the catalog, estimator and inference skeletons.

These frameworks are design-stage only, so the tests assert the two things that
must already hold: the modules import without their eventual heavy
dependencies, and every unimplemented entry point fails loudly rather than
returning something plausible. A stub that quietly returns ``None`` would be
worse than no stub at all.
"""

from __future__ import annotations

import unittest

import numpy as np

from jet.catalog import GalaxyCatalog, HaloCatalog, HODModel
from jet.estimator import Covariance, DataVector, Statistic
from jet.inference import Likelihood, Sampler
from jet.spec import DataVectorSpec, Param, ParameterSpec


class TestCatalogSkeleton(unittest.TestCase):
    """Containers and the HOD plugin interface."""

    def setUp(self) -> None:
        self.halos = HaloCatalog(
            position=np.zeros((5, 3)),
            velocity=np.zeros((5, 3)),
            mass=np.arange(5.0),
            mass_def="M200m",
            box_size=100.0,
        )
        self.galaxies = GalaxyCatalog(
            position=np.zeros((3, 3)),
            velocity=np.zeros((3, 3)),
            host_mass=np.arange(3.0),
            is_central=np.array([True, False, False]),
        )

    def tearDown(self) -> None:
        del self.halos, self.galaxies

    def test_catalogs_report_their_size(self) -> None:
        self.assertEqual(len(self.halos), 5)
        self.assertEqual(len(self.galaxies), 3)

    def test_mass_definition_defaults_to_m200m(self) -> None:
        halos = HaloCatalog(position=np.zeros((1, 3)), velocity=np.zeros((1, 3)), mass=np.ones(1))
        self.assertEqual(halos.mass_def, "M200m")

    def test_optional_columns_default_to_none(self) -> None:
        self.assertIsNone(self.halos.concentration)
        self.assertIsNone(self.halos.velocity_dispersion)

    def test_from_halos_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            HaloCatalog.from_halos({"mass": np.ones(3)})

    def test_from_particles_explains_why(self) -> None:
        with self.assertRaises(NotImplementedError) as caught:
            HaloCatalog.from_particles()
        self.assertIn("halo finder", str(caught.exception))

    def test_hod_model_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            HODModel()  # type: ignore[abstract]

    def test_hod_subclass_can_declare_a_spec(self) -> None:
        class Zheng07(HODModel):
            name = "zheng07"
            parameter_spec = ParameterSpec([Param("logMmin", bounds=(11.0, 14.0), block="hod")])

            def populate(self, halos, theta, rng):
                raise NotImplementedError

        self.assertEqual(Zheng07().parameter_spec.names, ("logMmin",))
        self.assertEqual(Zheng07().parameter_spec.blocks, ("hod",))


class TestEstimatorSkeleton(unittest.TestCase):
    """Statistics, measured vectors and covariance."""

    def setUp(self) -> None:
        self.spec = DataVectorSpec("wp", n_bins=3, log=True)
        self.vector = DataVector(values=np.array([1.0, 2.0, 3.0]), spec=self.spec)

    def tearDown(self) -> None:
        del self.spec, self.vector

    def test_data_vector_carries_its_spec(self) -> None:
        self.assertEqual(len(self.vector), 3)
        self.assertIs(self.vector.spec, self.spec)

    def test_data_vector_rejects_a_width_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            DataVector(values=np.array([1.0, 2.0]), spec=self.spec)

    def test_statistic_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Statistic()  # type: ignore[abstract]

    def test_statistic_subclass_can_declare_a_spec(self) -> None:
        class GalaxyClustering(Statistic):
            name = "wp"
            data_vector_spec = self.spec  # type: ignore[assignment]

            def measure(self, catalog, **options):
                raise NotImplementedError

        self.assertEqual(GalaxyClustering().data_vector_spec.dim, 3)

    def test_random_catalog_is_not_implemented(self) -> None:
        class GalaxyClustering(Statistic):
            name = "wp"
            data_vector_spec = DataVectorSpec("wp", n_bins=3)

            def measure(self, catalog, **options):
                raise NotImplementedError

        with self.assertRaises(NotImplementedError):
            GalaxyClustering().random_catalog(None, n_randoms=10)  # type: ignore[arg-type]

    def test_covariance_records_its_provenance(self) -> None:
        covariance = Covariance(matrix=np.eye(3), spec=self.spec, n_realisations=100, method="mock")
        self.assertEqual(covariance.method, "mock")
        self.assertEqual(covariance.n_realisations, 100)
        self.assertIs(covariance.spec, self.spec)

    def test_covariance_copy_is_independent(self) -> None:
        covariance = Covariance(matrix=np.eye(3), spec=self.spec)
        clone = covariance.copy()
        self.assertIsNot(clone.matrix, covariance.matrix)
        np.testing.assert_array_equal(clone.matrix, covariance.matrix)

    def test_covariance_from_mocks_is_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            Covariance.from_mocks(np.zeros((10, 3)), self.spec)

    def test_covariance_inverse_is_not_implemented(self) -> None:
        """Deliberately absent: with too few realisations the inverse would be
        biased, and returning one silently would produce a confidently wrong
        likelihood."""
        covariance = Covariance(matrix=np.eye(3), spec=self.spec, n_realisations=4)
        with self.assertRaises(NotImplementedError):
            covariance.inverse()


class TestInferenceSkeleton(unittest.TestCase):
    """Likelihood construction and the sampler interface."""

    def setUp(self) -> None:
        self.spec = DataVectorSpec("wp", n_bins=3)
        self.parameter_spec = ParameterSpec([Param("ns", bounds=(0.92, 1.00))])

    def tearDown(self) -> None:
        del self.spec, self.parameter_spec

    def test_likelihood_exposes_the_sampled_parameters(self) -> None:
        class _Emulator:
            x_spec = self.parameter_spec

        likelihood = Likelihood(
            measurement=DataVector(values=np.ones(3), spec=self.spec),
            covariance=Covariance(matrix=np.eye(3), spec=self.spec),
            emulator=_Emulator(),  # type: ignore[arg-type]
        )
        self.assertIs(likelihood.parameter_spec, self.parameter_spec)

    def test_likelihood_evaluation_is_not_implemented(self) -> None:
        class _Emulator:
            x_spec = self.parameter_spec

        likelihood = Likelihood(
            measurement=DataVector(values=np.ones(3), spec=self.spec),
            covariance=Covariance(matrix=np.eye(3), spec=self.spec),
            emulator=_Emulator(),  # type: ignore[arg-type]
        )
        theta = np.array([0.96])
        with self.assertRaises(NotImplementedError):
            likelihood.log_likelihood(theta)
        with self.assertRaises(NotImplementedError):
            likelihood.log_prior(theta)
        with self.assertRaises(NotImplementedError):
            likelihood.log_posterior(theta)

    def test_sampler_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            Sampler()  # type: ignore[abstract]

    def test_nested_sampling_is_reachable_through_options(self) -> None:
        """The sampler signature must accept a nested-sampling switch, since the
        README names PocoMC as the intended nested-sampling backend."""

        class _Emcee(Sampler):
            name = "emcee"

            def run(
                self, likelihood, *, n_steps, n_walkers=None, initial=None, rng=None, **options
            ):
                return {"nested": options.get("nested", False)}

        self.assertTrue(_Emcee().run(None, n_steps=10, nested=True)["nested"])  # type: ignore[arg-type]
        self.assertFalse(_Emcee().run(None, n_steps=10)["nested"])  # type: ignore[arg-type]


class TestSkeletonsImportWithoutHeavyDependencies(unittest.TestCase):
    """Importing the design skeletons must not pull in a training stack."""

    def test_modules_import(self) -> None:
        import jet.catalog
        import jet.estimator
        import jet.inference

        self.assertTrue(jet.catalog.__doc__)
        self.assertTrue(jet.estimator.__doc__)
        self.assertTrue(jet.inference.__doc__)

    def test_docstrings_record_the_open_questions(self) -> None:
        import jet.estimator
        import jet.inference

        self.assertIn("cross-covariance", jet.estimator.__doc__)
        self.assertIn("Cross-statistic covariance", jet.inference.__doc__)


if __name__ == "__main__":
    unittest.main()
