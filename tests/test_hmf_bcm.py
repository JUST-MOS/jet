"""
Tests for the baryonic halo mass function.

This is the one bundled model with no cosmological parameters, so the spec tests
here have something to say that the other model's do not: that the input really
is baryons-only, that every parameter is declared under ``block="baryon"``, and
that nothing has quietly acquired a cosmic block it never had data for.

The bundling conventions are the same as everywhere else -- the layout is
checked without weights, the values with them, and a fresh clone skips rather
than fails -- with one addition. The raw emulator and the wrapper disagree by a
factor of ``1e9`` in volume units, faithfully reproducing a disagreement in the
reference between its model class and its published accessor, so that relation
is pinned by a test rather than left to a comment.
"""

from __future__ import annotations

import unittest

import numpy as np

from jet.emulator.hmf_bcm import (
    ALPHA,
    BUNDLE_NAME,
    N_BINS,
    PARAMETER_DEFAULTS,
    REFERENCE_VOLUME_SCALE,
    BCMHFEmulator,
    bin_width,
    data_vector_spec,
    mass_grid,
    theta_spec,
)
from jet.emulator.pklin import data_dir

HAS_DATA = (data_dir() / BUNDLE_NAME).exists()

#: The reference's bounds, transcribed from its ``utils.bary_limits``. Duplicated
#: here deliberately: if the module's spec drifts from the reference's, the
#: bundle stops reproducing it, and this is the test that says so first.
REFERENCE_BOUNDS = {
    "logMc": (10.0, 16.0),
    "thej": (1.0, 10.0),
    "mu": (0.1, 10.0),
    "delta": (4.0, 10.0),
}

#: Three parameter points the golden values below were taken at, spanning the
#: box: the reference's default, a corner near the low end, and one near the
#: high end where the feedback is strongest.
GOLDEN_POINTS = {
    "defaults": np.array([13.0, 3.0, 1.0, 4.0]),
    "low": np.array([10.5, 1.5, 0.5, 4.5]),
    "high": np.array([15.0, 8.0, 8.0, 9.0]),
}

#: ``n(>= M)`` in :math:`(h/\\mathrm{Mpc})^3` at the first five and last two
#: masses of the grid, from the reference's own model class with its published
#: volume factor applied.
#:
#: The tail is tiny -- ``1e-18`` at the top mass bin -- which is what the model
#: is: essentially no haloes above :math:`7.9\\times10^{15} M_\\odot/h`.
GOLDEN_CUMULATIVE = {
    "defaults": np.array(
        [
            1.30180642e-02,
            1.07617331e-02,
            8.84751780e-03,
            7.23375999e-03,
            5.88179883e-03,
            1.05970228e-08,
            9.26424774e-09,
        ]
    ),
    "low": np.array(
        [
            1.32198963e-02,
            1.09273865e-02,
            8.98430438e-03,
            7.34738016e-03,
            5.97668635e-03,
            1.01619905e-09,
            6.36501162e-10,
        ]
    ),
    "high": np.array(
        [
            1.30300530e-02,
            1.07708593e-02,
            8.85424363e-03,
            7.23851578e-03,
            5.88497370e-03,
            4.74417490e-18,
            2.22602491e-19,
        ]
    ),
}


class TestBaryonSpec(unittest.TestCase):
    """The input is four feedback parameters and nothing else."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_four_parameters_in_the_reference_order(self) -> None:
        self.assertEqual(theta_spec().dim, 4)
        self.assertEqual(list(theta_spec().names), ["logMc", "thej", "mu", "delta"])

    def test_every_parameter_is_declared_baryon(self) -> None:
        """The whole point of this model: one block, and it is not the cosmic one."""
        self.assertEqual(theta_spec().blocks, ("baryon", "baryon", "baryon", "baryon"))

    def test_no_cosmological_parameter_leaks_in(self) -> None:
        """The model has no cosmological axis, so a cosmic column would be a lie."""
        self.assertNotIn("cosmo", theta_spec().blocks)
        for name in ("Omegab", "Omegam", "H0", "ns", "As", "w0", "wa", "mnu"):
            self.assertNotIn(name, theta_spec().names)

    def test_bounds_are_the_reference_limits(self) -> None:
        for param in theta_spec().params:
            self.assertEqual(param.bounds, REFERENCE_BOUNDS[param.name])

    def test_spec_hash_is_driven_by_the_block_label(self) -> None:
        """``block`` is part of the hash, so a relabelling invalidates every bundle.

        Pinned because it is a one-word change with a whole-file consequence.
        """
        from jet.spec import Param, ParameterSpec

        relabelled = ParameterSpec(
            [Param(p.name, bounds=p.bounds, block="hod") for p in theta_spec().params]
        )
        self.assertNotEqual(theta_spec().hash(), relabelled.hash())

    def test_output_spec_is_a_logged_rectangle(self) -> None:
        spec = data_vector_spec()
        self.assertEqual(spec.n_bins, N_BINS)
        self.assertEqual(spec.n_bins, 46)
        self.assertTrue(spec.log)

    def test_defaults_are_inside_the_box(self) -> None:
        """A default outside the training range would be a trap, not a default."""
        self.assertEqual(PARAMETER_DEFAULTS.shape, (theta_spec().dim,))
        found = theta_spec().check_within_bounds(PARAMETER_DEFAULTS[None, :])
        self.assertEqual(found, [])

    def test_the_noise_floor_is_the_shared_negligible_one(self) -> None:
        self.assertEqual(ALPHA, 1e-10)


@unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
class TestMassGrid(unittest.TestCase):
    """The grid the data vector is evaluated on, which travels in the bundle."""

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_size_matches_the_data_vector(self) -> None:
        self.assertEqual(mass_grid().size, N_BINS)

    def test_is_ascending_and_uniform_in_log_mass(self) -> None:
        grid = mass_grid()
        self.assertTrue(np.all(np.diff(grid) > 0.0))
        steps = np.diff(np.log10(grid))
        np.testing.assert_allclose(steps, steps[0], rtol=1e-12)

    def test_starts_and_ends_where_the_reference_grid_does(self) -> None:
        """``logspace(10, 16, 61)[14:]``, lower edges: 1e11.4 to 1e15.9."""
        grid = mass_grid()
        self.assertAlmostEqual(float(np.log10(grid[0])), 11.4, places=12)
        self.assertAlmostEqual(float(np.log10(grid[-1])), 15.9, places=12)

    def test_bin_width_is_the_grid_spacing(self) -> None:
        self.assertAlmostEqual(bin_width(), 0.1, places=12)


@unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
class TestBCMHFEmulator(unittest.TestCase):
    """The bundled model, against golden numbers from the reference."""

    @classmethod
    def setUpClass(cls):
        cls.emu = BCMHFEmulator.load()

    def test_cumulative_has_the_documented_shape(self) -> None:
        got = self.emu.cumulative(PARAMETER_DEFAULTS[None, :])
        self.assertEqual(got.shape, (1, N_BINS))
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(np.all(got > 0.0))

    def test_cumulative_decreases_with_mass(self) -> None:
        r"""A cumulative abundance cannot rise: :math:`n(\geq M)` falls with M."""
        got = self.emu.cumulative(PARAMETER_DEFAULTS[None, :])[0]
        self.assertTrue(np.all(np.diff(got) < 0.0))

    def test_golden_values(self) -> None:
        for label, point in GOLDEN_POINTS.items():
            with self.subTest(point=label):
                got = self.emu.cumulative(point[None, :])[0]
                expected = GOLDEN_CUMULATIVE[label]
                np.testing.assert_allclose(np.concatenate([got[:5], got[-2:]]), expected, rtol=1e-6)

    def test_golden_values_are_in_jet_volume_units(self) -> None:
        """The leading bin is ~1e-2, not ~1e7: the 1e-9 factor is applied once."""
        got = self.emu.cumulative(GOLDEN_POINTS["defaults"][None, :])[0]
        self.assertLess(float(got[0]), 1.0)
        self.assertGreater(float(got[0]), 1e-3)

    def test_log10_cumulative_is_the_logarithm(self) -> None:
        point = PARAMETER_DEFAULTS[None, :]
        np.testing.assert_allclose(
            self.emu.log10_cumulative(point), np.log10(self.emu.cumulative(point)), rtol=1e-12
        )

    def test_dndlgM_is_positive_and_finite(self) -> None:
        got = self.emu.dndlgM(PARAMETER_DEFAULTS[None, :])
        self.assertEqual(got.shape, (1, N_BINS))
        self.assertTrue(np.all(np.isfinite(got)))
        self.assertTrue(np.all(got > 0.0))

    def test_dndlgM_integrates_back_to_the_cumulative(self) -> None:
        r"""The first bin of the differential is the cumulative at the first edge.

        The reference's differencing convention -- bin ``i`` against bin
        ``i+1``, and the *last* bin against zero -- means this identity is
        checkable without the reference, and a reversed axis or a wrong bin
        width breaks it.
        """
        point = PARAMETER_DEFAULTS[None, :]
        cumulative = self.emu.cumulative(point)[0]
        differential = self.emu.dndlgM(point)[0]
        np.testing.assert_allclose(
            differential[:-1] * bin_width(), cumulative[:-1] - cumulative[1:], rtol=1e-9
        )
        # The last bin differences against zero rather than against an edge the
        # grid does not hold, so it stands alone.
        self.assertAlmostEqual(
            float(differential[-1] * bin_width()), float(cumulative[-1]), places=30
        )

    def test_the_raw_emulator_uses_the_other_volume_unit(self) -> None:
        """A factor of ``1e9`` between two accessors that look interchangeable.

        Pinned because it reproduces a real disagreement in the reference -- its
        model class returns ``(h/Gpc)^3`` and its published wrapper converts --
        and because nothing else in the package would notice the day someone
        "simplifies" one of the two away.
        """
        point = PARAMETER_DEFAULTS[None, :]
        raw = self.emu.emulator.predict(point, return_std=False)
        np.testing.assert_allclose(
            self.emu.cumulative(point), raw * REFERENCE_VOLUME_SCALE, rtol=1e-12
        )
        self.assertAlmostEqual(REFERENCE_VOLUME_SCALE, 1e-9, places=20)

    def test_different_parameters_give_different_answers(self) -> None:
        first = self.emu.cumulative(GOLDEN_POINTS["low"][None, :])[0]
        second = self.emu.cumulative(GOLDEN_POINTS["high"][None, :])[0]
        self.assertFalse(np.allclose(first, second))

    def test_specs_come_from_the_bundle(self) -> None:
        self.assertEqual(self.emu.x_spec, theta_spec())
        self.assertEqual(self.emu.y_spec, data_vector_spec())

    def test_batches_agree_with_single_rows(self) -> None:
        """The wrapper must not care how many points arrive at once.

        Not an exact equality even though it looks like one: the covariance
        product is a BLAS call, and a two-row and a one-row call can land on
        different kernels. Measured at ``2e-11``, so the tolerance is well above
        the blocking and well below anything a batching mistake would produce.
        """
        stacked = np.stack([GOLDEN_POINTS["low"], GOLDEN_POINTS["high"]])
        together = self.emu.cumulative(stacked)
        self.assertEqual(together.shape, (2, N_BINS))
        for index, label in enumerate(["low", "high"]):
            np.testing.assert_allclose(
                together[index],
                self.emu.cumulative(GOLDEN_POINTS[label][None, :])[0],
                rtol=1e-8,
            )


class TestMissingBundleMessage(unittest.TestCase):
    """A fresh clone must be told how to get the weights."""

    def test_reader_explains_what_to_do(self) -> None:
        import jet.emulator.hmf_bcm as module

        missing = data_dir() / "definitely-not-here.npz"
        with self.assertRaises(FileNotFoundError) as caught:
            module._data_path(missing.name)
        message = str(caught.exception)
        self.assertIn("tools/build_hmf_bcm_bundle.py", message)
        self.assertIn("JET_DATA_DIR", message)


if __name__ == "__main__":
    unittest.main()
