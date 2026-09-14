"""
Tests for the halo-matter correlation function emulator.

Two layers are tested separately on purpose. The *box* -- the grid the bundled
weights live on, and the layout that flattens it -- is checked without the
weights at all, because a wrong axis order is a property of the module and not
of the data. The *values* need the weights, which are not tracked by git, and
skip on a fresh clone.

The golden numbers were produced by the reference implementation and confirmed
against it by ``tools/build_xihm_bundle.py``; they are here so that a future
change cannot pass by being merely self-consistent. That script is also where
the tightest comparison lives, since it needs a checkout of the reference to
run at all.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from jet.cosmology import cosmo_parameter_spec
from jet.emulator.hmf import BUNDLE_NAME as HMF_BUNDLE_NAME
from jet.emulator.pklin import BUNDLE_NAME as PKLIN_BUNDLE_NAME
from jet.emulator.pklin import data_dir
from jet.emulator.xihm import (
    BUNDLE_NAME,
    LGDEN_GRID,
    N_BINS,
    N_LGDEN,
    N_R,
    N_Z,
    R_MID,
    Z_GRID,
    XiHMEmulator,
    brhm_grid,
    brhm_interpolate,
    data_vector_spec,
    lgden_grid,
    r_mid,
    theta_spec,
    z_grid,
)

# The bundled weights are not tracked by git, so a fresh clone has no data and
# the tests that need it must skip rather than fail.
HAS_DATA = (data_dir() / BUNDLE_NAME).exists()
HAS_HMF = (data_dir() / HMF_BUNDLE_NAME).exists()
HAS_PKLIN = (data_dir() / PKLIN_BUNDLE_NAME).exists()
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

#: Coordinates of the golden values below: number-density thresholds, redshifts
#: and separations, all of them exact nodes of the box.
PROBE_LGDEN = np.array([-5.0, -3.5, -2.5])
PROBE_Z = np.array([0.0, 0.5, 1.5, 3.0])
PROBE_R = np.array([1.0, 10.0, 42.5])

#: ``B_hm`` at the probe nodes, from the reference's ``get_Brhm``, in its own
#: ``(lgden, z, r)`` axis order.
#:
#: The values fall with the density threshold and with separation, which is the
#: physical way round: ``lgden`` is the *log of the cumulative number density*,
#: so a larger value is a lower mass threshold and a weaker bias.
GOLDEN_BRHM = np.array(
    [
        [
            [3.55164648e01, 2.87511348e00, 3.24225608e00],
            [3.55888470e01, 3.62394853e00, 3.55364187e00],
            [2.57000950e01, 5.14893739e00, 5.23342447e00],
            [1.66375123e01, 7.12974851e00, 7.42665668e00],
        ],
        [
            [5.90935413e00, 1.41661049e00, 1.52062479e00],
            [6.91179744e00, 1.88778518e00, 1.98642750e00],
            [7.24604633e00, 2.99119384e00, 3.08031225e00],
            [9.14933186e00, 4.60488670e00, 4.71595797e00],
        ],
        [
            [1.96482247e00, 9.15872497e-01, 9.92209806e-01],
            [2.48577022e00, 1.19353583e00, 1.25362677e00],
            [3.77384912e00, 1.96593886e00, 2.00043747e00],
            [5.72614142e00, 3.26528637e00, 3.32512682e00],
        ],
    ]
)

#: A point strictly inside the box, so that this golden value exercises the
#: interpolation rather than landing on a node where it would be skipped.
GOLDEN_OFF_NODE = 3.0914183338789907
OFF_NODE_Z = 0.4
OFF_NODE_LGDEN = -4.1
OFF_NODE_R = 3.0


class TestBoxGrid(unittest.TestCase):
    """The three axes, which are module constants rather than bundle contents."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_sizes_are_consistent(self) -> None:
        self.assertEqual(len(Z_GRID), N_Z)
        self.assertEqual(len(LGDEN_GRID), N_LGDEN)
        self.assertEqual(R_MID.size, N_R)
        self.assertEqual(N_BINS, N_Z * N_LGDEN * N_R)
        self.assertEqual(N_BINS, 2664)

    def test_axes_are_ascending(self) -> None:
        self.assertTrue(np.all(np.diff(Z_GRID) > 0.0))
        self.assertTrue(np.all(np.diff(LGDEN_GRID) > 0.0))
        self.assertTrue(np.all(np.diff(R_MID) > 0.0))

    def test_redshift_axis_starts_at_zero(self) -> None:
        self.assertEqual(Z_GRID[0], 0.0)
        self.assertEqual(Z_GRID[-1], 3.0)

    def test_separation_axis_ends_where_the_reference_binning_does(self) -> None:
        """The reference filters ``r_mid < 50``; the last midpoint is 42.5."""
        self.assertAlmostEqual(float(R_MID[-1]), 42.5, places=12)
        self.assertLess(float(R_MID[-1]), 50.0)

    def test_specs_describe_what_the_module_documents(self) -> None:
        self.assertEqual(theta_spec().dim, 8)
        self.assertEqual(data_vector_spec().n_bins, N_BINS)
        self.assertFalse(data_vector_spec().log)


class TestCosmologyIsSharedWithTheOtherModels(unittest.TestCase):
    """One parameter box for every bundled model, so the three cannot drift apart."""

    def test_theta_spec_is_the_shared_one(self) -> None:
        shared = cosmo_parameter_spec()
        self.assertEqual(theta_spec(), shared)
        self.assertEqual(
            [param.name for param in theta_spec().params],
            [param.name for param in shared.params],
        )


class TestBrhmGrid(unittest.TestCase):
    """Reshaping and reshaping back."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_rejects_the_wrong_width(self) -> None:
        with self.assertRaises(ValueError) as caught:
            brhm_grid(np.zeros((1, N_BINS - 1)))
        self.assertIn(str(N_BINS), str(caught.exception))

    def test_reshapes_into_three_axes(self) -> None:
        values = np.arange(2 * N_BINS, dtype=float).reshape(2, N_BINS)
        grid = brhm_grid(values)
        self.assertEqual(grid.shape, (2, N_Z, N_LGDEN, N_R))
        np.testing.assert_array_equal(grid[1], values[1].reshape(N_Z, N_LGDEN, N_R))

    def test_round_trips(self) -> None:
        values = np.arange(N_BINS, dtype=float)[None, :]
        np.testing.assert_array_equal(brhm_grid(values).ravel(), values[0])


class TestBrhmInterpolate(unittest.TestCase):
    """The three-axis interpolation, on a synthetic box."""

    def setUp(self) -> None:
        # A box whose every entry is a known linear function of the flat index,
        # so that a wrong axis order shows up as a wrong value rather than as a
        # plausible number.
        self.values = np.arange(N_BINS, dtype=float)[None, :]
        self.grid = brhm_grid(self.values)[0]

    def tearDown(self) -> None:
        pass

    def test_returns_the_reference_axis_order(self) -> None:
        out = brhm_interpolate(self.values, z=Z_GRID[:3], lgden=LGDEN_GRID[:2], r=R_MID[:5])
        self.assertEqual(out.shape, (1, 2, 3, 5))

    def test_a_node_returns_the_box_entry(self) -> None:
        """The strongest layout invariant, and it needs no reference to state."""
        for i_z in (0, 5, N_Z - 1):
            for i_den in (0, 3, N_LGDEN - 1):
                for i_r in (0, 7, N_R - 1):
                    out = brhm_interpolate(
                        self.values,
                        z=Z_GRID[i_z],
                        lgden=LGDEN_GRID[i_den],
                        r=R_MID[i_r],
                    )
                    # The query is a single scalar separation, so its axis has
                    # length one while the other two keep their sizes.
                    self.assertEqual(out.shape, (1, 1, 1, 1))
                    self.assertAlmostEqual(
                        float(out[0, 0, 0, 0]),
                        float(self.grid[i_z, i_den, i_r]),
                        places=9,
                        msg=f"node (z={Z_GRID[i_z]}, lgden={LGDEN_GRID[i_den]}, r={R_MID[i_r]})",
                    )

    def test_a_node_is_invariant_to_the_other_axes(self) -> None:
        """Reversing the query order must reverse the output, nothing else."""
        forward = brhm_interpolate(self.values, z=Z_GRID, lgden=LGDEN_GRID, r=R_MID)[0]
        backward = brhm_interpolate(
            self.values, z=Z_GRID[::-1], lgden=LGDEN_GRID[::-1], r=R_MID[::-1]
        )[0]
        np.testing.assert_allclose(backward, forward[::-1, ::-1, ::-1], rtol=1e-12)

    def test_interpolates_linearly_in_log_r(self) -> None:
        """Halving the log separation must halve the step, not the radius."""
        r_lo, r_hi = R_MID[8], R_MID[9]
        mid = 10.0 ** (0.5 * (np.log10(r_lo) + np.log10(r_hi)))
        lo = brhm_interpolate(self.values, z=Z_GRID[0], lgden=LGDEN_GRID[0], r=r_lo)[0, 0, 0, 0]
        hi = brhm_interpolate(self.values, z=Z_GRID[0], lgden=LGDEN_GRID[0], r=r_hi)[0, 0, 0, 0]
        at = brhm_interpolate(self.values, z=Z_GRID[0], lgden=LGDEN_GRID[0], r=mid)[0, 0, 0, 0]
        self.assertAlmostEqual(float(at), 0.5 * (float(lo) + float(hi)), places=9)
        # In linear r the midpoint is somewhere else entirely, so this check has
        # content only if the interpolation really is in log10(r).
        linear_mid = 0.5 * (r_lo + r_hi)
        other = brhm_interpolate(self.values, z=Z_GRID[0], lgden=LGDEN_GRID[0], r=linear_mid)[
            0, 0, 0, 0
        ]
        self.assertNotAlmostEqual(float(at), float(other), places=6)

    def test_rejects_a_non_positive_separation(self) -> None:
        with self.assertRaises(ValueError):
            brhm_interpolate(self.values, z=0.0, lgden=-4.0, r=0.0)
        with self.assertRaises(ValueError):
            brhm_interpolate(self.values, z=0.0, lgden=-4.0, r=np.array([1.0, -1.0]))

    def test_warns_when_the_box_is_extrapolated(self) -> None:
        for kwargs in (
            {"z": 3.5, "lgden": -4.0, "r": 1.0},
            {"z": 0.5, "lgden": -6.0, "r": 1.0},
            {"z": 0.5, "lgden": -4.0, "r": 100.0},
        ):
            with self.assertWarns(RuntimeWarning):
                brhm_interpolate(self.values, **kwargs)

    def test_stays_quiet_inside_the_box(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            brhm_interpolate(self.values, z=0.5, lgden=-4.0, r=np.array([1.0, 40.0]))


class TestBlendWeights(unittest.TestCase):
    r"""``exp(-(r/40)^4)``, whose shape decides where extrapolation matters.

    The weights are pinned because the module docstring quotes them, and because
    they are the reason asking for separations beyond the box's ``42.5`` is
    survivable. An earlier version of that docstring claimed the weight was
    already ``1e-17`` at ``r = 60``; it is ``6e-3`` there, and ``1e-17`` only at
    ``r = 100``. These tests exist so the next revision of the prose has to
    agree with the arithmetic.
    """

    def setUp(self) -> None:
        from jet.emulator.xihm import R_SWITCH

        self.r_switch = R_SWITCH

    def tearDown(self) -> None:
        pass

    def _weight(self, r: float) -> float:
        return float(np.exp(-((r / self.r_switch) ** 4)))

    def test_the_switch_is_where_the_weight_is_one_over_e(self) -> None:
        self.assertAlmostEqual(self._weight(40.0), 1.0 / np.e, places=12)

    def test_the_documented_weights(self) -> None:
        for radius, expected in (
            (42.5, 2.796e-01),
            (50.0, 8.704e-02),
            (60.0, 6.330e-03),
            (80.0, 1.125e-07),
            (100.0, 1.085e-17),
        ):
            self.assertAlmostEqual(
                self._weight(radius), expected, delta=abs(expected) * 1e-3, msg=f"r = {radius}"
            )

    def test_the_weight_falls_by_four_orders_over_the_decade_after_the_box(self) -> None:
        """The quantitative version of "the extrapolation is damped quickly"."""
        self.assertGreater(self._weight(42.5) / self._weight(80.0), 1e6)


class TestBlend(unittest.TestCase):
    """The blend itself, on synthetic inputs."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_beyond_a_hundred_the_baseline_is_the_whole_answer(self) -> None:
        """Not merely close to it: the emulated term is below double precision."""
        from jet.emulator.xihm import _blend

        direct = np.full((1, 1, 4), 1.0)
        tree = np.full((1, 1, 4), 2.0)
        got = _blend(direct, tree, np.array([10.0, 100.0, 150.0, 300.0]))
        np.testing.assert_array_equal(got[0, 0, 1:], 2.0)

    def test_well_inside_the_switch_the_emulated_term_dominates(self) -> None:
        from jet.emulator.xihm import _blend

        got = _blend(np.full((1, 1, 1), 1.0), np.full((1, 1, 1), 100.0), np.array([1.0]))
        self.assertLess(float(got[0, 0, 0]), 1.01)

    def test_the_radial_axis_is_the_last_one(self) -> None:
        """The weight must broadcast along ``r`` regardless of how many leading axes."""
        from jet.emulator.xihm import _blend

        r = np.array([0.0 + 1.0, 1000.0])
        for shape in ((2, 3, 2), (2, 2), (2, 3, 4, 2)):
            direct = np.ones(shape)
            tree = np.zeros(shape)
            got = _blend(direct, tree, r)
            self.assertEqual(got.shape, shape)
            np.testing.assert_array_equal(got[..., 1], 0.0)


class TestPositiveSeparation(unittest.TestCase):
    """The interpolation is in ``log10 r``, so a non-positive radius is meaningless."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_rejects_zero_and_negative(self) -> None:
        from jet.emulator.xihm import _require_positive_r

        for bad in (0.0, -1.0, np.array([1.0, 0.0])):
            with self.assertRaises(ValueError):
                _require_positive_r(bad)

    def test_accepts_and_returns_an_array(self) -> None:
        from jet.emulator.xihm import _require_positive_r

        self.assertEqual(_require_positive_r(5.0).shape, (1,))
        self.assertEqual(_require_positive_r(np.array([1.0, 2.0])).shape, (2,))


@unittest.skipUnless(HAS_PKLIN, f"{PKLIN_BUNDLE_NAME} is not present")
class TestTransformGrids(unittest.TestCase):
    """The wavenumber grids the transforms run on, which the values depend on.

    Needs the power-spectrum bundle because the check is that the sampled
    wavenumbers fall inside the emulator's own range.
    """

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_the_matter_transform_uses_the_reference_grids(self) -> None:
        """512 sampled and 1024 transformed, not the emulator's own grid."""
        from jet.emulator import pklin, xihm

        self.assertEqual(xihm.XIMM_K.size, 512)
        self.assertEqual(xihm.XIMM_KFFT.size, 1024)
        # Every sampled wavenumber must be inside the emulator's range, or the
        # transform would be fed an extrapolation it did not ask for.
        self.assertGreaterEqual(float(xihm.XIMM_K[0]), float(pklin.k_grid()[0]))
        self.assertLessEqual(float(xihm.XIMM_K[-1]), float(pklin.k_grid()[-1]))

    def test_the_baseline_transform_uses_a_different_grid(self) -> None:
        """Different from the matter one, so the two do not share their ringing.

        They share their lower limit and differ at the upper one, which is the
        part that matters: the high-k end is what sets FFTLog's truncation.
        """
        from jet.emulator.xihm import TREE_K, XIMM_KFFT

        self.assertEqual(TREE_K.size, 1024)
        self.assertNotEqual(float(TREE_K[-1]), float(XIMM_KFFT[-1]))
        self.assertLess(float(TREE_K[-1]), float(XIMM_KFFT[-1]))

    def test_the_grids_are_uniformly_log_spaced(self) -> None:
        from jet.emulator.xihm import TREE_K, XIMM_K, XIMM_KFFT

        for grid in (XIMM_K, XIMM_KFFT, TREE_K):
            steps = np.diff(np.log(grid))
            np.testing.assert_allclose(steps, steps[0], rtol=1e-12)


@unittest.skipUnless(HAS_PKLIN, f"{PKLIN_BUNDLE_NAME} is not present")
class TestXimmLinear(unittest.TestCase):
    """The matter correlation function, which needs only the linear P(k)."""

    @classmethod
    def setUpClass(cls):
        cls.theta = TRAINING_COSMOLOGY[None, :]

    def test_shape_and_realness(self) -> None:
        from jet.emulator.xihm import ximm_linear

        got = ximm_linear(self.theta, z=np.array([0.0, 1.0, 3.0]), r=np.logspace(-1.5, 1.2, 10))
        self.assertEqual(got.shape, (1, 3, 10))
        self.assertTrue(np.all(np.isfinite(got)))

    def test_is_positive_on_the_scales_the_transform_resolves(self) -> None:
        """xi_mm is positive below the baryon acoustic scale's turnover region."""
        from jet.emulator.xihm import ximm_linear

        got = ximm_linear(self.theta, z=0.0, r=np.logspace(-1.5, 0.5, 8))[0, 0]
        self.assertTrue(np.all(got > 0.0))

    def test_decreases_with_redshift(self) -> None:
        """Structure grows, so the correlation at fixed r is weaker later."""
        from jet.emulator.xihm import ximm_linear

        got = ximm_linear(self.theta, z=np.array([0.0, 3.0]), r=np.array([5.0]))[0, :, 0]
        self.assertGreater(float(got[0]), float(got[1]))


@unittest.skipUnless(HAS_BOTH, f"{BUNDLE_NAME} or {HMF_BUNDLE_NAME} is not present")
class TestXihm(unittest.TestCase):
    """The full correlation function, which needs both bundles."""

    @classmethod
    def setUpClass(cls):
        cls.emu = XiHMEmulator.load()
        cls.theta = TRAINING_COSMOLOGY[None, :]

    def test_density_threshold_shape_and_axis_order(self) -> None:
        got = self.emu.xihm_lgnbar_threshold(
            self.theta, z=np.array([0.0, 1.5]), r=np.logspace(-1.0, 1.0, 6), lgden=LGDEN_GRID[:3]
        )
        self.assertEqual(got.shape, (1, 3, 2, 6))
        self.assertTrue(np.all(np.isfinite(got)))

    def test_mass_threshold_shape_and_axis_order(self) -> None:
        # A 1e13 Msun/h threshold sits above the densities the bias box covers,
        # so this extrapolates and says so; the shape is the point here.
        self.emu.warn_on_extrapolation = False
        try:
            got = self.emu.xihm_mass(
                self.theta,
                z=np.array([0.0, 1.5]),
                r=np.logspace(-1.0, 1.0, 6),
                M=np.array([1e12, 1e13]),
            )
        finally:
            self.emu.warn_on_extrapolation = True
        self.assertEqual(got.shape, (1, 2, 2, 6))
        self.assertTrue(np.all(np.isfinite(got)))

    def test_small_radius_is_the_emulated_term_alone(self) -> None:
        r"""At :math:`r \lesssim 5` the baseline contributes under ``1e-4``.

        The weight is ``exp(-(r/40)^4)``, so this is a statement about the blend
        expressed through the two pieces the wrapper already exposes: the answer
        must be the ratio times the matter correlation function, to better than
        the baseline's share. Measured, the agreement is ``1e-6``.
        """
        for radius in (1.0, 5.0):
            got = self.emu.xihm_lgnbar_threshold(
                self.theta, z=0.0, r=[radius], lgden=LGDEN_GRID[2:4]
            )[0]
            ratio = self.emu.brhm(self.theta, z=0.0, lgden=LGDEN_GRID[2:4], r=[radius])[0]
            ximm = self.emu.ximm(self.theta, z=0.0, r=[radius])[0, 0, 0]
            np.testing.assert_allclose(got[:, 0, 0], ratio[:, 0, 0] * ximm, rtol=1e-4)

    def test_far_out_the_ratio_approaches_the_linear_bias(self) -> None:
        r"""At large :math:`r`, :math:`\xi_{hm}/\xi_{mm} \to b`.

        That is the point of the blend, and it is checkable without the
        reference because all three quantities are exposed. Measured, it holds
        to about 0.4 per cent from ``r = 50`` outward; the residual is the
        difference between the baseline's own transform and the matter one, not
        the emulator, which contributes nothing there.
        """
        self.emu.warn_on_extrapolation = False
        try:
            for radius in (50.0, 200.0, 400.0):
                got = self.emu.xihm_lgnbar_threshold(
                    self.theta, z=0.0, r=[radius], lgden=LGDEN_GRID[2:4]
                )[0, :, 0, 0]
                ximm = self.emu.ximm(self.theta, z=0.0, r=[radius])[0, 0, 0]
                bias = self.emu.bias_lgnbar_threshold(self.theta, z=0.0, lgden=LGDEN_GRID[2:4])[
                    0, 0
                ]
                # Both xi_hm and xi_mm are negative beyond the BAO peak, so the
                # ratio is positive and the comparison is sign-preserving.
                np.testing.assert_allclose(got / ximm, bias, rtol=1e-2)
        finally:
            self.emu.warn_on_extrapolation = True

    def test_warn_on_extrapolation_covers_the_whole_stack(self) -> None:
        """One switch, not one per layer: the interface has three axes and five layers."""
        self.emu.warn_on_extrapolation = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                self.emu.xihm_lgnbar_threshold(self.theta, z=0.0, r=[200.0], lgden=LGDEN_GRID[2:4])
        finally:
            self.emu.warn_on_extrapolation = True


@unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
class TestXiHMEmulator(unittest.TestCase):
    """The bundled model, against golden numbers from the reference."""

    @classmethod
    def setUpClass(cls):
        cls.emu = XiHMEmulator.load()
        cls.theta = TRAINING_COSMOLOGY[None, :]

    def test_bias_ratio_has_the_documented_shape(self) -> None:
        ratio = self.emu.bias_ratio(self.theta)
        self.assertEqual(ratio.shape, (1, N_BINS))
        self.assertTrue(np.all(np.isfinite(ratio)))
        # A ratio of two correlation functions, so positive and of order one to
        # a few hundred at the small-r end where the one-halo term dominates.
        self.assertGreater(float(ratio.min()), 0.0)
        self.assertLess(float(ratio.max()), 1e5)

    def test_golden_node_values(self) -> None:
        got = self.emu.brhm(self.theta, z=PROBE_Z, lgden=PROBE_LGDEN, r=PROBE_R)[0]
        self.assertEqual(got.shape, GOLDEN_BRHM.shape)
        np.testing.assert_allclose(got, GOLDEN_BRHM, rtol=1e-6)

    def test_golden_off_node_value(self) -> None:
        got = self.emu.brhm(self.theta, z=OFF_NODE_Z, lgden=OFF_NODE_LGDEN, r=OFF_NODE_R)[0]
        np.testing.assert_allclose(got, GOLDEN_OFF_NODE, rtol=1e-6)

    def test_bias_falls_with_the_number_density_threshold(self) -> None:
        """Physical orientation: a higher threshold is a lower mass and a weaker bias."""
        got = self.emu.brhm(self.theta, z=0.5, lgden=LGDEN_GRID, r=10.0)[0, :, 0, 0]
        self.assertTrue(np.all(np.diff(got) < 0.0))

    def test_the_box_is_the_interpolation_source(self) -> None:
        """Sampling every node must reproduce the box entry by entry."""
        # The box is stored redshift-first; the interpolated result uses the
        # reference's (lgden, z, r) order, so only a transpose separates them.
        grid = brhm_grid(self.emu.bias_ratio(self.theta))[0].transpose(1, 0, 2)
        got = self.emu.brhm(self.theta, z=Z_GRID, lgden=LGDEN_GRID, r=R_MID)[0]
        np.testing.assert_allclose(got, grid, rtol=1e-9)

    def test_different_cosmologies_give_different_answers(self) -> None:
        """The reference caches its interpolator across cosmologies; this must not."""
        other = TRAINING_COSMOLOGY.copy()
        other[1] += 0.05  # Omega_m
        first = self.emu.brhm(self.theta, z=0.5, lgden=-4.0, r=10.0)[0]
        second = self.emu.brhm(other[None, :], z=0.5, lgden=-4.0, r=10.0)[0]
        repeated = self.emu.brhm(self.theta, z=0.5, lgden=-4.0, r=10.0)[0]
        self.assertFalse(np.allclose(first, second))
        np.testing.assert_array_equal(first, repeated)

    def test_specs_come_from_the_bundle(self) -> None:
        self.assertEqual(self.emu.x_spec, theta_spec())
        self.assertEqual(self.emu.y_spec, data_vector_spec())

    def test_the_bundle_axes_match_the_module_constants(self) -> None:
        """The module constants are the fallback when the bundle is absent.

        That fallback is only sound while the two agree, and nothing else
        compares them: the grid accessors prefer the bundle's copy silently.
        """
        np.testing.assert_array_equal(z_grid(), np.asarray(Z_GRID))
        np.testing.assert_array_equal(lgden_grid(), np.asarray(LGDEN_GRID))
        np.testing.assert_allclose(r_mid(), R_MID, rtol=0.0, atol=0.0)


class TestMissingBundleMessage(unittest.TestCase):
    """A fresh clone must be told how to get the weights."""

    def test_reader_explains_what_to_do(self) -> None:
        import jet.emulator.xihm as xihm_module

        missing = data_dir() / "definitely-not-here.npz"
        with self.assertRaises(FileNotFoundError) as caught:
            xihm_module._data_path(missing.name)
        message = str(caught.exception)
        self.assertIn("tools/build_xihm_bundle.py", message)
        self.assertIn("JET_DATA_DIR", message)


if __name__ == "__main__":
    unittest.main()
