"""
Tests for the halo bias emulator and its conversions.

The splitting is the same as in :mod:`tests.test_xihm`: the layout and the
spline plumbing are checked without weights, because a wrong axis order is a
property of the module rather than of the data; the values need the weights and
skip on a fresh clone.

The golden numbers came from the reference implementation and were confirmed
against it by ``tools/build_xihm_bundle.py``. Note that two of the three
conversions here are *not* regressions -- the density-to-mass inversion and the
mass-threshold finite difference -- so they fail for reasons the box test cannot
see, which is why they have tests of their own.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from jet.emulator.bhm import (
    BUNDLE_NAME,
    LGDEN_GRID,
    LGDEN_INVERSION_GRID,
    LGDEN_INVERSION_MAX,
    LGDEN_INVERSION_MIN,
    MASS_EDGES,
    MASS_PERTURBATION,
    N_BINS,
    N_LGDEN,
    N_Z,
    Z_GRID,
    BhmEmulator,
    _evaluate_spline,
    bias_grid,
    bias_ratio_at,
    data_vector_spec,
    theta_spec,
)
from jet.emulator.hmf import BUNDLE_NAME as HMF_BUNDLE_NAME
from jet.emulator.pklin import data_dir

HAS_DATA = (data_dir() / BUNDLE_NAME).exists()
HAS_HMF = (data_dir() / HMF_BUNDLE_NAME).exists()
HAS_BOTH = HAS_DATA and HAS_HMF

#: The first training cosmology of the reference's Sobol design, with the
#: primordial amplitude converted from the reference's ``A_s * 1e9``.
TRAINING_COSMOLOGY = np.array(
    [
        4.8974680e-02,
        3.0969282e-01,
        6.7660000e01,
        9.6650000e-01,
        2.1050000e-09,
        -1.0000000e00,
        0.0000000e00,
        6.0000000e-02,
    ]
)

#: The bias correction at ``z = 0``, in :data:`LGDEN_GRID` order, from the
#: reference. Unlike the bias itself it is not monotonic in the threshold: it is
#: a correction to an analytic baseline, and the correction peaks in the middle
#: of the range.
GOLDEN_RATIO_Z0 = np.array(
    [
        1.3196412283289174,
        1.3461777144249307,
        1.3451521282590284,
        1.3296565701903407,
        1.3093694816489790,
        1.2807723786571577,
    ]
)

PROBE_Z = np.array([0.0, 0.5, 1.5, 3.0])
PROBE_LGDEN = np.array([-5.0, -4.0, -3.0, -2.5])
PROBE_MASS = np.array([1e12, 1e13, 1e14])

#: ``n(>= M)`` inverted back to a mass, in ``Msun/h``, at the probe points.
GOLDEN_MASS = np.array(
    [
        [2.10546933e14, 4.47911798e13, 5.36000031e12, 1.56958525e12],
        [1.13210739e14, 2.83224000e13, 4.22514164e12, 1.36022216e12],
        [3.46572089e13, 1.08544194e13, 2.27395718e12, 8.75549984e11],
        [7.69956480e12, 2.87054170e12, 7.92124557e11, 3.60570442e11],
    ]
)

#: The bias at those density thresholds.
GOLDEN_BIAS_LGDEN = np.array(
    [
        [3.40197570, 2.12793830, 1.21225039, 0.96593651],
        [4.09564365, 2.71083182, 1.59991220, 1.25068931],
        [5.57278858, 3.97058643, 2.55808191, 2.03438675],
        [7.34607317, 5.71090566, 4.04983328, 3.35755879],
    ]
)

#: The bias at fixed *mass* thresholds, which is a different quantity: the
#: sample is selected by mass rather than by abundance.
GOLDEN_BIAS_MASS = np.array(
    [
        [0.831521728, 1.092570619, 1.958991725],
        [1.019824437, 1.578354549, 3.213792998],
        [1.721127781, 3.356328270, 7.404664574],
        [3.601605717, 7.840568648, 18.119507871],
    ]
)


class TestBhmGrid(unittest.TestCase):
    """The two axes, which are module constants rather than bundle contents."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_sizes_are_consistent(self) -> None:
        self.assertEqual(len(Z_GRID), N_Z)
        self.assertEqual(len(LGDEN_GRID), N_LGDEN)
        self.assertEqual(N_BINS, N_Z * N_LGDEN)
        self.assertEqual(N_BINS, 72)

    def test_axes_are_ascending(self) -> None:
        self.assertTrue(np.all(np.diff(Z_GRID) > 0.0))
        self.assertTrue(np.all(np.diff(LGDEN_GRID) > 0.0))

    def test_the_other_model_shares_the_same_axes(self) -> None:
        """The correlation-function box imports these; they must not diverge."""
        from jet.emulator import xihm

        np.testing.assert_array_equal(xihm.Z_GRID, Z_GRID)
        np.testing.assert_array_equal(xihm.LGDEN_GRID, LGDEN_GRID)
        self.assertEqual(xihm.N_LGDEN, N_LGDEN)

    def test_the_inversion_grid_covers_more_than_the_box(self) -> None:
        """The inversion is tabulated wider than the box so its corners are safe."""
        self.assertLess(LGDEN_INVERSION_GRID[0], LGDEN_GRID[0])
        self.assertLess(LGDEN_INVERSION_MIN, LGDEN_GRID[0])
        self.assertGreater(LGDEN_INVERSION_MAX, LGDEN_GRID[-1])
        self.assertTrue(np.all(np.diff(LGDEN_INVERSION_GRID) > 0.0))

    def test_the_mass_grid_spans_the_emulated_range(self) -> None:
        self.assertTrue(np.all(np.diff(MASS_EDGES) > 0.0))
        self.assertAlmostEqual(float(MASS_EDGES[0]), 1e11, places=0)
        self.assertAlmostEqual(float(MASS_EDGES[-1]), 1e16, places=0)

    def test_the_perturbation_is_a_small_finite_difference_step(self) -> None:
        """It is a numerical step, not a physical quantity; one per cent."""
        self.assertGreater(MASS_PERTURBATION, 0.0)
        self.assertLess(MASS_PERTURBATION, 0.1)

    def test_specs_describe_what_the_module_documents(self) -> None:
        self.assertEqual(theta_spec().dim, 8)
        self.assertEqual(data_vector_spec().n_bins, N_BINS)
        self.assertFalse(data_vector_spec().log)


class TestSplineEvaluationIsOrderAgnostic(unittest.TestCase):
    """``RectBivariateSpline`` needs increasing queries; callers need not give them."""

    def setUp(self) -> None:
        from scipy.interpolate import RectBivariateSpline

        self.z = np.linspace(0.0, 1.0, 5)
        self.x = np.linspace(0.0, 1.0, 6)
        values = np.add.outer(self.z, self.x)
        self.spline = RectBivariateSpline(self.z, self.x, values, kx=1, ky=1)

    def tearDown(self) -> None:
        pass

    def test_unsorted_queries_match_sorted_ones(self) -> None:
        z_query = np.array([0.9, 0.1, 0.5])
        x_query = np.array([0.7, 0.2, 0.55, 0.05])
        got = _evaluate_spline(self.spline, z_query, x_query)
        expected = np.add.outer(z_query, x_query)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_the_result_is_the_outer_product_in_the_asked_for_order(self) -> None:
        got = _evaluate_spline(self.spline, np.array([0.2, 0.8]), np.array([0.4, 0.1]))
        np.testing.assert_allclose(got, np.add.outer([0.2, 0.8], [0.4, 0.1]), rtol=1e-12)


class TestBiasGrid(unittest.TestCase):
    """Reshaping and reshaping back."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_rejects_the_wrong_width(self) -> None:
        with self.assertRaises(ValueError) as caught:
            bias_grid(np.zeros((1, N_BINS - 1)))
        self.assertIn(str(N_BINS), str(caught.exception))

    def test_reshapes_into_two_axes(self) -> None:
        values = np.arange(2 * N_BINS, dtype=float).reshape(2, N_BINS)
        grid = bias_grid(values)
        self.assertEqual(grid.shape, (2, N_Z, N_LGDEN))
        np.testing.assert_array_equal(grid[1], values[1].reshape(N_Z, N_LGDEN))


class TestBiasRatioAt(unittest.TestCase):
    """The two-axis interpolation, on a synthetic box."""

    def setUp(self) -> None:
        self.values = np.arange(N_BINS, dtype=float)[None, :]
        self.grid = bias_grid(self.values)[0]

    def tearDown(self) -> None:
        pass

    def test_returns_the_reference_axis_order(self) -> None:
        out = bias_ratio_at(self.values, z=Z_GRID[:3], lgden=LGDEN_GRID[:2])
        self.assertEqual(out.shape, (1, 3, 2))

    def test_a_node_returns_the_box_entry(self) -> None:
        for i_z in (0, 5, N_Z - 1):
            for i_den in (0, 3, N_LGDEN - 1):
                out = bias_ratio_at(self.values, z=Z_GRID[i_z], lgden=LGDEN_GRID[i_den])
                self.assertAlmostEqual(
                    float(out[0, 0, 0]),
                    float(self.grid[i_z, i_den]),
                    places=9,
                    msg=f"node (z={Z_GRID[i_z]}, lgden={LGDEN_GRID[i_den]})",
                )

    def test_warns_when_the_box_is_extrapolated(self) -> None:
        for kwargs in (
            {"z": 3.5, "lgden": -4.0},
            {"z": 0.5, "lgden": -6.0},
        ):
            with self.assertWarns(RuntimeWarning):
                bias_ratio_at(self.values, **kwargs)

    def test_stays_quiet_inside_the_box(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            bias_ratio_at(self.values, z=0.5, lgden=-4.0)


@unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
class TestBhmEmulator(unittest.TestCase):
    """The bundled model, against golden numbers from the reference."""

    @classmethod
    def setUpClass(cls):
        cls.emu = BhmEmulator.load()
        cls.theta = TRAINING_COSMOLOGY[None, :]

    def test_ratio_has_the_documented_shape(self) -> None:
        ratio = self.emu.ratio(self.theta)
        self.assertEqual(ratio.shape, (1, N_BINS))
        self.assertTrue(np.all(np.isfinite(ratio)))
        # A correction to a bias, so of order one -- not a bias, which would
        # reach tens at the high-mass end.
        self.assertGreater(float(ratio.min()), 0.5)
        self.assertLess(float(ratio.max()), 3.0)

    def test_golden_ratio_at_zero_redshift(self) -> None:
        box = bias_grid(self.emu.ratio(self.theta))[0]
        np.testing.assert_allclose(box[0], GOLDEN_RATIO_Z0, rtol=1e-6)

    def test_specs_come_from_the_bundle(self) -> None:
        self.assertEqual(self.emu.x_spec, theta_spec())
        self.assertEqual(self.emu.y_spec, data_vector_spec())

    def test_a_node_interpolates_back_to_the_box(self) -> None:
        grid = bias_grid(self.emu.ratio(self.theta))[0]
        got = self.emu.ratio_at(self.theta, z=Z_GRID, lgden=LGDEN_GRID)[0]
        np.testing.assert_allclose(got, grid, rtol=1e-9)

    def test_different_cosmologies_give_different_answers(self) -> None:
        other = TRAINING_COSMOLOGY.copy()
        other[1] += 0.05  # Omega_m
        first = self.emu.ratio_at(self.theta, z=0.5, lgden=-4.0)[0]
        second = self.emu.ratio_at(other[None, :], z=0.5, lgden=-4.0)[0]
        self.assertFalse(np.allclose(first, second))


@unittest.skipUnless(HAS_BOTH, f"{BUNDLE_NAME} or {HMF_BUNDLE_NAME} is not present")
class TestBhmConversions(unittest.TestCase):
    """The parts that are not a regression, against golden numbers."""

    @classmethod
    def setUpClass(cls):
        cls.emu = BhmEmulator.load()
        cls.theta = TRAINING_COSMOLOGY[None, :]

    def test_golden_mass_from_density(self) -> None:
        got = self.emu.mass_from_lgden(self.theta, z=PROBE_Z, lgden=PROBE_LGDEN)[0]
        np.testing.assert_allclose(got, GOLDEN_MASS, rtol=1e-6)

    def test_golden_bias_at_a_density_threshold(self) -> None:
        got = self.emu.bias_lgnbar_threshold(self.theta, z=PROBE_Z, lgden=PROBE_LGDEN)[0]
        np.testing.assert_allclose(got, GOLDEN_BIAS_LGDEN, rtol=1e-6)

    def test_golden_bias_at_a_mass_threshold(self) -> None:
        """The 1e14 bin sits above the densities the bias box covers.

        So this call extrapolates and says so, in terms of the mass the caller
        named rather than the threshold it implies. Asserting the warning keeps
        the behaviour visible instead of letting it show up as test noise.
        """
        with self.assertWarns(RuntimeWarning):
            got = self.emu.bias_mass(self.theta, z=PROBE_Z, M=PROBE_MASS)[0]
        np.testing.assert_allclose(got, GOLDEN_BIAS_MASS, rtol=1e-6)

    def test_the_mass_threshold_bias_is_not_the_density_threshold_bias(self) -> None:
        """The two selections are genuinely different, so this is worth pinning.

        The golden values above would pass either way if the conversion were a
        relabelling, which is exactly the mistake the finite difference exists
        to avoid. Two thresholds are needed because the baseline is itself a
        numerical derivative in mass.
        """
        density = self.emu.bias_lgnbar_threshold(self.theta, z=0.0, lgden=np.array([-3.2, -3.0]))[
            0, 0, 1
        ]
        mass = self.emu.bias_mass(self.theta, z=0.0, M=np.array([8e12, 1e13]))[0, 0, 1]
        self.assertNotAlmostEqual(float(density), float(mass), places=2)

    def test_a_single_threshold_works_through_a_local_stencil(self) -> None:
        """One threshold has no neighbour to difference against, so one is built.

        The answer is close to the multi-threshold one but not equal, because it
        is a different finite difference -- the separation is about ``1e-4``,
        which is the stencil's own discretisation error rather than anything
        wrong.
        """
        single = self.emu.bias_lgnbar_threshold(self.theta, z=0.0, lgden=np.array([-3.5]))[0, 0, 0]
        pair = self.emu.bias_lgnbar_threshold(self.theta, z=0.0, lgden=np.array([-3.5, -3.4]))[
            0, 0, 0
        ]
        self.assertAlmostEqual(float(single) / float(pair), 1.0, delta=1e-2)
        self.assertNotEqual(float(single), float(pair))

    def test_a_single_mass_works_too(self) -> None:
        """The same gap one level up: the perturbation needs a baseline per mass."""
        got = self.emu.bias_mass(self.theta, z=0.0, M=np.array([1e13]))
        self.assertEqual(got.shape, (1, 1, 1))
        self.assertTrue(np.all(np.isfinite(got)))

    def test_bias_grows_towards_rarer_objects(self) -> None:
        """Both threshold definitions must order the same way physically.

        A *larger* ``lgden`` is a *larger* cumulative abundance, hence a lower
        mass threshold and a weaker bias, so that axis runs the other way from
        the mass one.
        """
        by_density = self.emu.bias_lgnbar_threshold(self.theta, z=0.0, lgden=LGDEN_GRID)[0, 0]
        self.assertTrue(np.all(np.diff(by_density) < 0.0))
        # The 1e12 end of this range implies a threshold below the box, so the
        # call warns; monotonicity still has to hold through the extrapolation.
        with self.assertWarns(RuntimeWarning):
            by_mass = self.emu.bias_mass(self.theta, z=0.0, M=np.logspace(12, 14.5, 8))[0, 0]
        self.assertTrue(np.all(np.diff(by_mass) > 0.0))

    def test_mass_decreases_with_the_density_threshold(self) -> None:
        """A larger cumulative abundance is a lower mass threshold."""
        mass = self.emu.mass_from_lgden(self.theta, z=0.0, lgden=LGDEN_GRID)[0, 0]
        self.assertTrue(np.all(np.diff(mass) < 0.0))

    def test_accepts_an_unsorted_threshold_request(self) -> None:
        """Callers plot from high mass to low; the answer must not depend on that."""
        ascending = self.emu.mass_from_lgden(self.theta, z=0.5, lgden=np.array([-4.5, -3.5]))[0, 0]
        descending = self.emu.mass_from_lgden(self.theta, z=0.5, lgden=np.array([-3.5, -4.5]))[0, 0]
        np.testing.assert_allclose(descending[::-1], ascending, rtol=1e-12)


class TestMissingBundleMessage(unittest.TestCase):
    """A fresh clone must be told how to get the weights."""

    def test_reader_explains_what_to_do(self) -> None:
        import jet.emulator.bhm as bhm_module

        missing = data_dir() / "definitely-not-here.npz"
        with self.assertRaises(FileNotFoundError) as caught:
            bhm_module._data_path(missing.name)
        message = str(caught.exception)
        self.assertIn("tools/build_xihm_bundle.py", message)
        self.assertIn("JET_DATA_DIR", message)


if __name__ == "__main__":
    unittest.main()
