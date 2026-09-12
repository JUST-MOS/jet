"""
Tests for derived parameters: the conversions, the spec-level frame, and the
way an emulator reads its input through one.

Most of this file deliberately uses :class:`LinearCombination` rather than
:class:`Sigma8FromAs`, because the point being tested is the *mechanism* and an
exact relation makes the assertions exact too. The sigma8 conversion gets its
own class, guarded on the bundled data being present: the weights are not
tracked by git, so a clone without them must skip rather than fail.
"""

from __future__ import annotations

import json
import unittest
import warnings

import numpy as np

from jet.derived import (
    CONVERSIONS,
    Conversion,
    LinearCombination,
    Sigma8FromAs,
    conversion_from_state,
    register_conversion,
)
from jet.emulator import Emulator
from jet.emulator.pklin import BUNDLE_NAME, data_dir
from jet.spec import DataVectorSpec, Param, ParameterSpec

# The bundled weights are not tracked by git, so a fresh clone has no data and
# the tests that need it must skip rather than fail.
HAS_DATA = (data_dir() / BUNDLE_NAME).exists()


def _omegam_conversion() -> LinearCombination:
    """``Omegam = Omegab + Omegac``, the reference relation used throughout."""
    return LinearCombination("Omegam", "Omegac", {"Omegab": 1.0, "Omegac": 1.0})


def _cosmo_spec(derived=None) -> ParameterSpec:
    """A small, self-contained cosmology-like spec."""
    return ParameterSpec(
        [
            Param("Omegab", bounds=(0.04, 0.06), block="cosmo"),
            Param("Omegac", bounds=(0.20, 0.34), block="cosmo"),
            Param("H0", bounds=(60.0, 80.0), block="cosmo"),
        ],
        derived=derived,
    )


def _state_of(conversion: Conversion) -> dict:
    """Round-trip a conversion through its serialised form."""
    return json.loads(json.dumps(conversion.to_dict()))


class TestConversionProtocol(unittest.TestCase):
    """Validation shared by every conversion."""

    def test_registry_is_populated_by_import(self) -> None:
        self.assertIn("linear", CONVERSIONS)
        self.assertIn("sigma8_as", CONVERSIONS)

    def test_rejects_self_mapping(self) -> None:
        with self.assertRaises(ValueError):
            LinearCombination("Omegac", "Omegac", {"Omegac": 1.0})

    def test_rejects_requires_overlapping_base(self) -> None:
        with self.assertRaises(ValueError):
            LinearCombination("Omegam", "Omegac", {"Omegac": 1.0, "Omegam": 2.0})

    def test_rejects_duplicate_requires(self) -> None:
        # Neither public conversion can produce this -- LinearCombination takes
        # its terms as a mapping, which cannot repeat a key, and Sigma8FromAs
        # derives its axes from a fixed tuple -- so it is driven from a local
        # subclass rather than left uncovered.
        class Duplicated(Conversion):
            kind = "test-duplicated"

            def __init__(self) -> None:
                super().__init__("derived", "base", requires=("dup", "dup"))

            def to_derived(self, X, columns):  # pragma: no cover - never called
                raise NotImplementedError

            def to_base(self, X, values, columns):  # pragma: no cover
                raise NotImplementedError

            def state(self):  # pragma: no cover
                return {}

            @classmethod
            def from_state(cls, state):  # pragma: no cover
                return cls()

        with self.assertRaises(ValueError):
            Duplicated()

    def test_missing_reports_absent_axes(self) -> None:
        conversion = _omegam_conversion()
        self.assertEqual(conversion.missing({"Omegab": 0}), ["Omegac"])
        self.assertEqual(conversion.missing({"Omegab": 0, "Omegac": 1}), [])

    def test_needs_lists_base_and_inputs(self) -> None:
        self.assertEqual(set(_omegam_conversion().names), {"Omegab", "Omegac"})

    def test_evaluating_without_the_axes_raises(self) -> None:
        with self.assertRaises(KeyError) as caught:
            _omegam_conversion().to_derived(np.zeros((1, 1)), {"Omegab": 0})
        self.assertIn("Omegac", str(caught.exception))

    def test_unknown_kind_lists_registered(self) -> None:
        with self.assertRaises(KeyError) as caught:
            conversion_from_state({"kind": "no-such-thing"})
        self.assertIn("linear", str(caught.exception))

    def test_register_rejects_empty_kind(self) -> None:
        with self.assertRaises(ValueError):

            @register_conversion
            class Nameless(Conversion):
                def to_derived(self, X, columns):  # pragma: no cover - never called
                    raise NotImplementedError

                def to_base(self, X, values, columns):  # pragma: no cover
                    raise NotImplementedError

                def state(self):  # pragma: no cover
                    return {}

                @classmethod
                def from_state(cls, state):  # pragma: no cover
                    return cls()


class TestLinearCombination(unittest.TestCase):
    """The exact relation, where the assertions can be to machine precision."""

    def setUp(self) -> None:
        self.conversion = _omegam_conversion()
        self.columns = {"Omegab": 0, "Omegac": 1}
        self.X = np.array([[0.049, 0.260], [0.055, 0.300]])

    def test_to_derived_sums_the_terms(self) -> None:
        np.testing.assert_allclose(self.conversion.to_derived(self.X, self.columns), [0.309, 0.355])

    def test_to_base_solves_the_base_axis(self) -> None:
        got = self.conversion.to_base(self.X, np.array([0.31, 0.35]), self.columns)
        np.testing.assert_allclose(got, [[0.049, 0.261], [0.055, 0.295]])

    def test_round_trip_is_exact(self) -> None:
        derived = self.conversion.to_derived(self.X, self.columns)
        back = self.conversion.to_base(self.X, derived, self.columns)
        np.testing.assert_allclose(back, self.X, rtol=1e-15)

    def test_to_base_leaves_other_columns_alone(self) -> None:
        got = self.conversion.to_base(self.X, np.array([0.31, 0.35]), self.columns)
        np.testing.assert_array_equal(got[:, 0], self.X[:, 0])

    def test_to_base_rejects_a_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            self.conversion.to_base(self.X, np.array([0.31]), self.columns)

    def test_rejects_unknown_base_axis(self) -> None:
        with self.assertRaises(ValueError):
            LinearCombination("Omegam", "Omegac", {"Omegab": 1.0})

    def test_rejects_a_zero_coefficient_on_the_base(self) -> None:
        with self.assertRaises(ValueError):
            LinearCombination("Omegam", "Omegac", {"Omegac": 0.0, "Omegab": 1.0})

    def test_state_round_trip(self) -> None:
        rebuilt = conversion_from_state(_state_of(self.conversion))
        self.assertEqual(rebuilt.name, "Omegam")
        self.assertEqual(rebuilt.base, "Omegac")
        np.testing.assert_allclose(
            rebuilt.to_derived(self.X, self.columns),
            self.conversion.to_derived(self.X, self.columns),
        )

    def test_hash_is_stable_and_content_dependent(self) -> None:
        self.assertEqual(self.conversion.hash(), _omegam_conversion().hash())
        other = LinearCombination("Omegam", "Omegac", {"Omegab": 2.0, "Omegac": 1.0})
        self.assertNotEqual(self.conversion.hash(), other.hash())


class TestParameterSpecDerived(unittest.TestCase):
    """Declaring derived parameters on a spec, and the frame they expose."""

    def test_dim_is_unchanged_by_a_derived_parameter(self) -> None:
        self.assertEqual(_cosmo_spec().dim, _cosmo_spec([_omegam_conversion()]).dim)

    def test_names_does_not_gain_the_derived_name(self) -> None:
        spec = _cosmo_spec([_omegam_conversion()])
        self.assertNotIn("Omegam", spec.names)
        self.assertEqual(spec.derived_names, ("Omegam",))

    def test_has_derived_reflects_the_declaration(self) -> None:
        self.assertFalse(_cosmo_spec().has_derived)
        self.assertTrue(_cosmo_spec([_omegam_conversion()]).has_derived)

    def test_frame_renames_exactly_one_column(self) -> None:
        frame = _cosmo_spec([_omegam_conversion()]).frame("Omegam")
        self.assertEqual(frame.names, ("Omegab", "Omegam", "H0"))
        self.assertEqual(frame.dim, 3)
        self.assertEqual(frame.base_names, ("Omegab", "Omegac", "H0"))

    def test_frame_of_an_undeclared_name_raises(self) -> None:
        with self.assertRaises(KeyError) as caught:
            _cosmo_spec().frame("Omegam")
        self.assertIn("derived parameters are []", str(caught.exception))

    def test_rejects_a_derived_name_that_shadows_an_axis(self) -> None:
        clash = LinearCombination("H0", "Omegac", {"Omegab": 1.0, "Omegac": 1.0})
        with self.assertRaises(ValueError) as caught:
            _cosmo_spec([clash])
        self.assertIn("clashes", str(caught.exception))

    def test_rejects_a_conversion_whose_inputs_are_absent(self) -> None:
        with self.assertRaises(ValueError) as caught:
            _cosmo_spec([_omegam_conversion(), Sigma8FromAs()])
        self.assertIn("not declared on this spec", str(caught.exception))

    def test_rejects_a_non_conversion(self) -> None:
        with self.assertRaises(TypeError):
            _cosmo_spec(["Omegam"])

    def test_rejects_duplicate_derived_names(self) -> None:
        with self.assertRaises(ValueError):
            _cosmo_spec([_omegam_conversion(), _omegam_conversion()])

    def test_frame_converts_both_ways(self) -> None:
        frame = _cosmo_spec([_omegam_conversion()]).frame("Omegam")
        base = np.array([[0.049, 0.261, 67.66]])

        derived = frame.from_base(base)
        np.testing.assert_allclose(derived[:, 1], [0.31])
        np.testing.assert_array_equal(derived[:, [0, 2]], base[:, [0, 2]])

        np.testing.assert_allclose(frame.to_base(derived), base, rtol=1e-15)

    def test_frame_validates_the_width(self) -> None:
        frame = _cosmo_spec([_omegam_conversion()]).frame("Omegam")
        with self.assertRaises(ValueError):
            frame.to_base(np.zeros((2, 5)))

    def test_frame_accepts_a_single_row(self) -> None:
        frame = _cosmo_spec([_omegam_conversion()]).frame("Omegam")
        self.assertEqual(frame.to_base(np.array([0.049, 0.31, 67.66])).shape, (1, 3))

    def test_subset_keeps_a_conversion_that_survives(self) -> None:
        spec = _cosmo_spec([_omegam_conversion()])
        self.assertEqual(spec.subset(names=["Omegab", "Omegac"]).derived_names, ("Omegam",))

    def test_subset_drops_a_conversion_it_cannot_evaluate(self) -> None:
        spec = _cosmo_spec([_omegam_conversion()])
        self.assertEqual(spec.subset(names=["Omegab", "H0"]).derived_names, ())

    def test_dict_round_trip_preserves_the_derived_declaration(self) -> None:
        spec = _cosmo_spec([_omegam_conversion()])
        rebuilt = ParameterSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(rebuilt.derived_names, ("Omegam",))
        self.assertEqual(rebuilt, spec)
        self.assertEqual(rebuilt.hash(), spec.hash())

    def test_hash_ignores_the_conversion(self) -> None:
        """The hash binds a model to the axes, and conversions are the caller's business.

        A model never reads a conversion -- the caller applies it before the
        array reaches the model -- so declaring one must not invalidate a
        bundle. The declarations still travel in ``to_dict``, which is what lets
        a loaded model report the frames its spec offers.
        """
        plain = _cosmo_spec()
        terms = {"Omegab": 2.0, "Omegac": 1.0}
        doubled = _cosmo_spec([LinearCombination("Omegam", "Omegac", terms)])

        self.assertEqual(plain.hash(), doubled.hash())
        self.assertNotEqual(plain.to_dict(), doubled.to_dict())
        self.assertNotEqual(plain, doubled)


class TestDerivedParametersWithAnEmulator(unittest.TestCase):
    """An emulator trained on one axis, driven along another by the caller.

    The substitution lives outside the model on purpose: whatever the caller
    passes to :meth:`predict` means exactly what ``x_spec`` says it means, so
    there is only ever one reading of an array at the point it is consumed.
    """

    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.spec = _cosmo_spec([_omegam_conversion()])
        self.y_spec = DataVectorSpec("demo", n_bins=2)

        bounds = self.spec.bounds_array()
        self.X = np.column_stack([rng.uniform(low, high, 30) for low, high in bounds])
        self.y = np.column_stack([self.X[:, 0] * 100.0, self.X[:, 2]])

        self.emulator = Emulator(
            self.spec, self.y_spec, backend="gp", backend_kwargs={"optimizer": None}
        ).fit(self.X, self.y)

    def test_converting_before_the_call_reproduces_the_base_frame(self) -> None:
        frame = self.spec.frame("Omegam")
        derived = frame.from_base(self.X[:6])
        np.testing.assert_allclose(
            self.emulator.predict(frame.to_base(derived), return_std=False),
            self.emulator.predict(self.X[:6], return_std=False),
            rtol=1e-10,
        )

    def test_the_declaration_survives_a_bundle_round_trip(self) -> None:
        """A loaded model still names the frames its spec offers.

        The model cannot act on the declaration, but it carries it, so a caller
        holding nothing but the bundle can find out that ``Omegam`` is available
        and convert through it.
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = self.emulator.save(Path(directory) / "framed.npz")
            restored = Emulator.load(path)

        self.assertEqual(restored.x_spec.derived_names, ("Omegam",))
        frame = restored.x_spec.frame("Omegam")
        np.testing.assert_allclose(
            restored.predict(frame.to_base(frame.from_base(self.X[:6])), return_std=False),
            self.emulator.predict(self.X[:6], return_std=False),
            rtol=1e-10,
        )

    def test_verify_catches_a_changed_axis_but_not_a_changed_declaration(self) -> None:
        dropped = _cosmo_spec([_omegam_conversion()]).subset(["Omegab", "H0"])
        with self.assertRaises(ValueError):
            self.emulator.verify(x_spec=dropped)

        # A different conversion over the same axes is not a mismatch: the model
        # reads the axes, and the conversions are the caller's to choose.
        terms = {"Omegab": 2.0, "Omegac": 1.0}
        doubled = _cosmo_spec([LinearCombination("Omegam", "Omegac", terms)])
        self.emulator.verify(x_spec=doubled)


@unittest.skipUnless(HAS_DATA, f"requires the bundled data under {data_dir()}")
class TestSigma8FromAs(unittest.TestCase):
    """The built-in conversion, against the reference it was packaged from."""

    #: Computed by the reference implementation for the cosmology below.
    REFERENCE_SIGMA8 = 0.8139727843066471

    #: Omegab, Omegam, H0, ns, As, w0, wa, mnu -- the emulator's own axes.
    COSMOLOGY = np.array([[0.049, 0.310, 67.66, 0.9665, 2.105e-9, -1.0, 0.0, 0.06]])

    COLUMNS = {
        "Omegab": 0,
        "Omegam": 1,
        "H0": 2,
        "ns": 3,
        "As": 4,
        "w0": 5,
        "wa": 6,
        "mnu": 7,
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.conversion = Sigma8FromAs()
        cls.spec = ParameterSpec(
            [
                Param("Omegab", (0.04, 0.06)),
                Param("Omegam", (0.24, 0.40)),
                Param("H0", (60.0, 80.0)),
                Param("ns", (0.92, 1.00)),
                Param("As", (1.7e-9, 2.5e-9)),
                Param("w0", (-1.3, -0.7)),
                Param("wa", (-0.5, 0.5)),
                Param("mnu", (0.0, 0.3)),
            ],
            derived=[cls.conversion],
        )
        cls.sigma8 = float(cls.conversion.to_derived(cls.COSMOLOGY, cls.COLUMNS)[0])

    def test_reproduces_the_reference_value(self) -> None:
        """The packaged emulator must agree with the one it was built from.

        The tolerance is loose enough for the two to differ in how they
        evaluate the same mathematics -- a Gram-matrix distance against
        scipy's cdist, a re-derived Cholesky against a stored one -- and tight
        enough that a wrong kernel, a mis-ordered grid or a missing
        normalisation would fail it.
        """
        self.assertAlmostEqual(self.sigma8 / self.REFERENCE_SIGMA8 - 1.0, 0.0, places=8)

    def test_is_a_plausible_sigma8(self) -> None:
        self.assertGreater(self.sigma8, 0.5)
        self.assertLess(self.sigma8, 1.5)

    def test_scales_as_the_square_root_of_the_amplitude(self) -> None:
        """P(k) is proportional to A_s, so sigma8 is proportional to its root.

        Nothing tells the emulator this -- it has to have learned it from the
        training spectra -- so it holds only to the emulator's own interpolation
        accuracy. Measuring the spread rather than the value at a single point
        is also what makes this a check of the *physics* rather than of one
        prediction: a kernel with the wrong amplitude would shift the constant,
        but only a wrong dependence on A_s would tilt it.
        """
        amplitudes = np.array([1.70, 2.00, 2.105, 2.35, 2.50]) * 1e-9
        rows = np.repeat(self.COSMOLOGY, amplitudes.size, axis=0)
        rows[:, 4] = amplitudes

        sigma8 = self.conversion.to_derived(rows, self.COLUMNS)
        self.assertTrue(np.all(np.diff(sigma8) > 0.0), "sigma8 must rise with A_s")

        ratio = sigma8 / np.sqrt(amplitudes)
        self.assertLess(float(np.ptp(ratio) / ratio.mean()), 1e-3)

    def test_inversion_round_trips(self) -> None:
        target = np.array([0.75, 0.85])
        rows = np.repeat(self.COSMOLOGY, 2, axis=0)
        solved = self.conversion.to_base(rows, target, self.COLUMNS)

        # Compare relatively: the amplitudes are of order 1e-9, so any absolute
        # tolerance loose enough to be safe at that magnitude would also swallow
        # the shift being tested for.
        moved = np.abs(solved[:, 4] / rows[:, 4] - 1.0)
        self.assertTrue(np.all(moved > 1e-3), f"the inversion barely moved A_s: {moved}")

        np.testing.assert_allclose(
            self.conversion.to_derived(solved, self.COLUMNS), target, atol=1e-6
        )

    def test_inversion_needs_a_positive_target(self) -> None:
        with self.assertRaises(ValueError):
            self.conversion.to_base(self.COSMOLOGY, np.array([-1.0]), self.COLUMNS)

    def test_a_target_out_of_reach_warns(self) -> None:
        """sigma8 = 5 is far outside anything this cosmology can produce.

        Left unsaid, the iteration would run its course and hand back an
        enormous A_s that looks like an answer. The budget is trimmed here only
        to keep the test quick -- every iteration costs a full emulator
        evaluation whatever the target, and the default hundred would spend
        twenty seconds proving the same point.
        """
        conversion = Sigma8FromAs(max_iterations=8)
        with self.assertWarns(RuntimeWarning) as caught:
            solved = conversion.to_base(self.COSMOLOGY, np.array([5.0]), self.COLUMNS)
        self.assertIn("did not reach sigma8", str(caught.warning))
        self.assertGreater(solved[0, 4], 1e-8)

    def test_a_short_iteration_budget_warns_rather_than_lying(self) -> None:
        """Convergence is not assumed: a budget too small to reach the
        tolerance is reported, not quietly accepted."""
        starved = Sigma8FromAs(max_iterations=1, tolerance=1e-14)
        with self.assertWarns(RuntimeWarning):
            starved.to_base(self.COSMOLOGY, np.array([0.82]), self.COLUMNS)

    def test_a_reachable_target_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.conversion.to_base(self.COSMOLOGY, np.array([0.82]), self.COLUMNS)

    def test_rejects_a_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            self.conversion.to_base(self.COSMOLOGY, np.array([0.8, 0.9]), self.COLUMNS)

    def test_requires_the_cosmological_axes(self) -> None:
        with self.assertRaises(KeyError):
            self.conversion.to_derived(self.COSMOLOGY, {"As": 4})

    def test_only_z_zero_is_supported(self) -> None:
        with self.assertRaises(NotImplementedError):
            Sigma8FromAs(z=1.0)

    def test_rejects_nonsense_settings(self) -> None:
        with self.assertRaises(ValueError):
            Sigma8FromAs(R=0.0)
        with self.assertRaises(ValueError):
            Sigma8FromAs(tolerance=0.0)
        with self.assertRaises(ValueError):
            Sigma8FromAs(max_iterations=0)

    def test_state_round_trip_preserves_the_result(self) -> None:
        rebuilt = conversion_from_state(_state_of(self.conversion))
        self.assertIsInstance(rebuilt, Sigma8FromAs)
        self.assertAlmostEqual(
            float(rebuilt.to_derived(self.COSMOLOGY, self.COLUMNS)[0]), self.sigma8, places=12
        )

    def test_state_of_a_live_emulator_is_refused(self) -> None:
        """A conversion holding an in-memory model cannot be written to a bundle."""
        live = Sigma8FromAs(pklin=self.conversion.emulator)
        with self.assertRaises(TypeError) as caught:
            live.state()
        self.assertIn("save()", str(caught.exception))

    def test_frame_converts_both_ways_through_the_emulator(self) -> None:
        frame = self.spec.frame("sigma8")
        derived = frame.from_base(self.COSMOLOGY)
        np.testing.assert_allclose(derived[:, 4], [self.sigma8], rtol=1e-9)
        np.testing.assert_allclose(frame.to_base(derived), self.COSMOLOGY, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
