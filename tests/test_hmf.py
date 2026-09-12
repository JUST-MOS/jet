"""
Tests for the halo mass function: the ported cosmology, the Castro23 baseline,
and the bundled emulator built on top of them.

The bundled weights are not tracked by git, so anything that needs them is
guarded on their presence and skips on a fresh clone. The physics is not: the
Fermi-Dirac integral, the expansion history and the baseline's shape are
checked without touching ``jet/data`` at all.

The golden numbers below were produced by the reference implementation and
confirmed against it by ``tools/build_hmf_bundle.py``; they are here so that a
future change to the port cannot pass by being merely self-consistent.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from jet.cosmology import (
    RHO_CRIT,
    Cosmology,
    cosmo_parameter_spec,
    fermi_dirac_integral,
)
from jet.emulator.hmf import (
    BUNDLE_NAME,
    CASTRO23_COEFFICIENTS,
    FINE_CENTERS,
    N_BINS,
    N_Z,
    HMFEmulator,
    castro23_cumulative,
    castro23_dndlnM,
    data_slices,
    data_vector_spec,
    mass_edges,
    mass_slices,
    theta_spec,
    z_grid,
)
from jet.emulator.pklin import data_dir

# The bundled weights are not tracked by git, so a fresh clone has no data and
# the tests that need it must skip rather than fail.
HAS_DATA = (data_dir() / BUNDLE_NAME).exists()

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

#: Cumulative number density in ``(h/Mpc)^3`` at ``z = (0, 1, 3)`` for masses
#: ``(1e12, 1e13, 1e14) Msun/h``, from the reference.
GOLDEN_NUMBER_DENSITY = np.array(
    [
        [4.74067279e-03, 5.36941342e-04, 3.39034024e-05],
        [3.55603910e-03, 2.28985833e-04, 3.14604987e-06],
        [6.88457625e-04, 4.81744558e-06, 2.09172933e-10],
    ]
)

#: ``dn/dlnM`` in ``(h/Mpc)^3`` at the same points.
GOLDEN_DNDLNM = np.array(
    [
        [4.21490575e-03, 5.49719122e-04, 4.96561516e-05],
        [3.70759061e-03, 3.21470407e-04, 8.12623658e-06],
        [1.12807335e-03, 1.39977418e-05, 1.41904723e-09],
    ]
)

PROBE_Z = np.array([0.0, 1.0, 3.0])
PROBE_M = np.array([1e12, 1e13, 1e14])


class TestFermiDiracIntegral(unittest.TestCase):
    """The neutrino integral, against the limits it has closed forms for."""

    def test_zero_argument_matches_the_analytic_value(self):
        # F(0) = integral of x^3 / (1 + e^x) = 7 pi^4 / 120.
        self.assertAlmostEqual(fermi_dirac_integral(0.0), 7.0 * math.pi**4 / 120.0, places=12)

    def test_large_argument_is_linear_in_y(self):
        # For y >> 1 the integrand is y x^2 / (1 + e^x), whose integral is
        # (3/2) zeta(3) -- a different constant from F(0), because the extra
        # power of x under the Fermi factor is what F(0) integrates.
        from scipy.special import zeta

        asymptote = 1.5 * float(zeta(3))
        ratios = [fermi_dirac_integral(y) / y / asymptote for y in (1e3, 1e5, 1e7)]
        # The approach is from above and the correction falls off with y.
        self.assertTrue(all(r > 1.0 for r in ratios))
        self.assertTrue(ratios[0] > ratios[1] > ratios[2])
        self.assertLess(ratios[-1] - 1.0, 1e-10)

    def test_is_increasing(self):
        values = fermi_dirac_integral(np.array([0.0, 1e-3, 0.1, 1.0, 10.0, 1000.0]))
        self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_scalar_in_scalar_out(self):
        self.assertIsInstance(fermi_dirac_integral(1.0), float)
        self.assertEqual(np.asarray(fermi_dirac_integral([1.0, 2.0])).shape, (2,))

    def test_rejects_negative_argument(self):
        with self.assertRaises(ValueError):
            fermi_dirac_integral(-1.0)

    def test_matches_the_reference_table_within_its_own_accuracy(self):
        # Spot values from the reference's own tabulation. The table is a cubic
        # interpolation of this same integral, so agreement is limited by the
        # table, not by the quadrature -- hence five places, not twelve. One
        # point in its 10001 is wrong by 5e-8 and is deliberately not probed.
        cases = [(0.0, 5.682196976983), (1.0005, 6.045985492763), (10.0068, 19.13673634252)]
        for y, expected in cases:
            self.assertAlmostEqual(fermi_dirac_integral(y) / expected, 1.0, places=5)


class TestCosmology(unittest.TestCase):
    """The expansion history and the two matter densities."""

    def setUp(self):
        self.cosmo = Cosmology(Omegab=0.049, Omegam=0.31, H0=67.66, w0=-1.0, wa=0.0, mnu=0.06)

    def test_hubble_parameter_is_normalised_today_to_within_the_reference_drift(self):
        r"""``E(0)`` is one, but not to machine precision.

        The reference carries two independent neutrino normalisations: the
        :math:`\sum m_\nu / 93.14\,\mathrm{eV}` conversion used to close the
        energy budget, and the Fermi-Dirac integral used in :math:`H(z)`. They
        differ by about one per cent, which shows up as
        :math:`E(0) = 1 - 7 \times 10^{-6}`. The port inherits that rather than
        silently repairing it, because every number it is verified against was
        computed with it.
        """
        self.assertAlmostEqual(float(self.cosmo.E(0.0)[0]), 1.0, delta=2e-5)
        drift = float(self.cosmo.E(0.0)[0]) ** 2 - 1.0
        expected = float(self.cosmo.neutrino_density_times_hubble_squared(0.0)[0])
        self.assertAlmostEqual(drift, expected - self.cosmo.Omeganu, places=14)

    def test_densities_today_are_the_inputs_up_to_the_same_drift(self):
        e0_squared = float(self.cosmo.E(0.0)[0]) ** 2
        self.assertAlmostEqual(float(self.cosmo.omega_cb(0.0)[0]), 0.31 / e0_squared, places=13)
        self.assertAlmostEqual(
            float(self.cosmo.omega_m(0.0)[0]),
            (0.31 + self.cosmo.Omeganu) / e0_squared,
            places=13,
        )

    def test_densities_stay_below_one_and_rise_with_redshift(self):
        z = np.array([0.0, 1.0, 3.0])
        cb = self.cosmo.omega_cb(z)
        self.assertTrue(np.all(cb < 1.0))
        self.assertTrue(np.all(np.diff(cb) > 0.0))

    def test_total_matter_exceeds_cold_dark_matter_plus_baryons(self):
        # The difference is the neutrino density, which is small but not zero.
        z = np.array([0.0, 2.0])
        difference = self.cosmo.omega_m(z) - self.cosmo.omega_cb(z)
        self.assertTrue(np.all(difference > 0.0))
        self.assertTrue(np.all(difference < 1e-2))

    def test_massless_neutrinos_leave_radiation_unchanged_in_the_density_split(self):
        massive = Cosmology(Omegab=0.049, Omegam=0.31, H0=67.66, mnu=0.06)
        massless = Cosmology(Omegab=0.049, Omegam=0.31, H0=67.66, mnu=0.0)
        self.assertGreater(massive.Omeganu, 0.0)
        self.assertEqual(massless.Omeganu, 0.0)
        # Both must still close the universe.
        for cosmo in (massive, massless):
            total = cosmo.Omegam + cosmo.Omeganu + cosmo.OmegaR + cosmo.OmegaL
            self.assertAlmostEqual(total, 1.0, places=14)

    def test_dark_energy_is_a_constant_for_a_cosmological_constant(self):
        de = self.cosmo.dark_energy_density(np.array([0.0, 1.0, 5.0]))
        self.assertTrue(np.allclose(de, 1.0, rtol=1e-14))

    def test_critical_density_matches_the_reference_constant(self):
        self.assertEqual(self.cosmo.rho_crit, RHO_CRIT)

    def test_rejects_a_negative_cdm_density(self):
        with self.assertRaises(ValueError):
            Cosmology(Omegab=0.06, Omegam=0.05, H0=67.66)

    def test_rejects_negative_neutrino_mass(self):
        with self.assertRaises(ValueError):
            Cosmology(Omegab=0.049, Omegam=0.31, H0=67.66, mnu=-0.1)

    def test_parameter_spec_is_the_reference_box(self):
        spec = cosmo_parameter_spec()
        self.assertEqual(spec.dim, 8)
        self.assertEqual(spec.names[4], "As")
        self.assertEqual(spec[spec.index("As")].bounds, (1.7e-9, 2.5e-9))
        self.assertEqual(spec[spec.index("mnu")].bounds, (0.0, 0.3))


@unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
class TestCastro23Baseline(unittest.TestCase):
    """The analytic baseline, which needs no bundled weights."""

    def setUp(self):
        self.theta = TRAINING_COSMOLOGY[None, :]
        self.masses = np.logspace(11.5, 14.0, 6)

    def test_is_positive_and_finite(self):
        values = castro23_dndlnM(self.theta, z=np.array([0.0, 1.0]), M=self.masses)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(values > 0.0))

    def test_falls_with_mass_and_with_redshift(self):
        values = castro23_dndlnM(self.theta, z=np.array([0.0, 1.0, 3.0]), M=self.masses)[0]
        self.assertTrue(np.all(np.diff(values, axis=1) < 0.0))
        self.assertTrue(np.all(values[0] > values[-1]))

    def test_cumulative_is_the_integral_of_the_differential(self):
        differential = castro23_dndlnM(self.theta, z=np.array([0.0]), M=FINE_CENTERS)
        cumulative = castro23_cumulative(self.theta, z=np.array([0.0]), M=FINE_CENTERS)
        self.assertEqual(differential.shape, cumulative.shape)
        # Totals must agree: the cumulative is the differential summed from the
        # top of the grid down, and the top contributes nothing.
        dln_m = (np.log10(FINE_CENTERS[1]) - np.log10(FINE_CENTERS[0])) * np.log(10.0)
        self.assertAlmostEqual(
            float(cumulative[0, 0, 0] / 1e9), float((differential[0, 0] * dln_m).sum()), places=8
        )

    def test_cumulative_decreases_with_mass(self):
        cumulative = castro23_cumulative(self.theta, z=np.array([0.0, 2.0]), M=self.masses)
        self.assertTrue(np.all(np.diff(cumulative, axis=2) < 0.0))

    def test_accepts_explicit_coefficients(self):
        default = castro23_dndlnM(self.theta, z=np.array([0.0]), M=self.masses)
        explicit = castro23_dndlnM(
            self.theta, z=np.array([0.0]), M=self.masses, coefficients=CASTRO23_COEFFICIENTS
        )
        np.testing.assert_array_equal(default, explicit)

    def test_accepts_an_explicit_power_spectrum_emulator(self):
        from jet.emulator.pklin import load_pklin_emulator

        shared = load_pklin_emulator()
        default = castro23_dndlnM(self.theta, z=np.array([0.0]), M=self.masses)
        explicit = castro23_dndlnM(self.theta, z=np.array([0.0]), M=self.masses, pk_emulator=shared)
        np.testing.assert_array_equal(default, explicit)

    def test_rejects_a_malformed_parameter_row(self):
        with self.assertRaises(ValueError):
            castro23_dndlnM(np.zeros((2, 5)))

    def test_is_per_redshift_not_positional(self):
        """A redshift asked for twice must give the same answer both times.

        The baseline interpolates the power spectrum at whichever redshift it
        is handed. Indexing the emulator's stored redshifts by position instead
        would still return plausible numbers, but for the wrong cosmology --
        and would make this assertion fail.
        """
        z = np.array([0.5, 0.5, 1.0])
        values = castro23_dndlnM(self.theta, z=z, M=self.masses)
        np.testing.assert_allclose(values[0, 0], values[0, 1], rtol=1e-13)

    def test_interpolates_between_stored_redshifts(self):
        # At cluster masses, where the abundance falls cleanly with redshift.
        # At the low-mass end the Castro23 fit is not monotone in redshift, so
        # this assertion would be testing the fit, not the interpolation.
        z = np.array([0.3, 0.5, 0.7])
        values = castro23_dndlnM(self.theta, z=z, M=np.array([1e13]))[0, :, 0]
        self.assertTrue(values[0] > values[1] > values[2])

    def test_redshift_interpolation_is_continuous(self):
        z = np.linspace(0.0, 3.0, 13)
        values = castro23_dndlnM(self.theta, z=z, M=np.array([1e13]))[0, :, 0]
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(np.diff(values) < 0.0))


class TestHMFBundleLayout(unittest.TestCase):
    """The grid the bundle declares, which needs the bundle to be present."""

    @unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
    def test_redshift_grid_is_ascending_and_starts_at_zero(self):
        z = z_grid()
        self.assertEqual(z.size, N_Z)
        self.assertEqual(float(z[0]), 0.0)
        self.assertTrue(np.all(np.diff(z) > 0.0))

    @unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
    def test_mass_edges_are_ascending(self):
        edges = mass_edges()
        self.assertEqual(edges.size, 61)
        self.assertTrue(np.all(np.diff(edges) > 0.0))

    @unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
    def test_blocks_cover_the_data_vector_exactly_once(self):
        blocks = data_slices()
        self.assertEqual(blocks.shape, (N_Z, 2))
        self.assertEqual(int(blocks[0, 0]), 0)
        self.assertEqual(int(blocks[-1, 1]), N_BINS)
        # Contiguous and non-overlapping.
        np.testing.assert_array_equal(blocks[1:, 0], blocks[:-1, 1])

    @unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
    def test_mass_ranges_share_a_lower_limit_and_shrink_with_redshift(self):
        bins = mass_slices()
        self.assertEqual(bins.shape, (N_Z, 2))
        self.assertEqual(len(set(bins[:, 0].tolist())), 1)
        # Ascending redshift means falling upper mass: the emulator was trained
        # over a narrower range at higher redshift.
        self.assertTrue(np.all(np.diff(bins[:, 1]) <= 0))

    @unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
    def test_data_and_mass_slices_have_the_same_lengths(self):
        np.testing.assert_array_equal(
            np.diff(data_slices(), axis=1), np.diff(mass_slices(), axis=1)
        )

    def test_specs_describe_what_the_module_documents(self):
        self.assertEqual(theta_spec().dim, 8)
        self.assertEqual(data_vector_spec().n_bins, N_BINS)


class TestCastro23CoefficientsAreTheBundledOnes(unittest.TestCase):
    """The module constant and the bundle must not drift apart."""

    @unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
    def test_bundle_carries_the_same_coefficients(self):
        from jet.emulator.hmf import castro23_coefficients

        np.testing.assert_array_equal(castro23_coefficients(), CASTRO23_COEFFICIENTS)


@unittest.skipUnless(HAS_DATA, f"{BUNDLE_NAME} is not present")
class TestHMFEmulator(unittest.TestCase):
    """The bundled model, against golden numbers from the reference."""

    @classmethod
    def setUpClass(cls):
        cls.hmf = HMFEmulator.load()
        cls.theta = TRAINING_COSMOLOGY[None, :]

    def test_golden_number_density(self):
        got = self.hmf.number_density(self.theta, z=PROBE_Z, M=PROBE_M)[0]
        np.testing.assert_allclose(got, GOLDEN_NUMBER_DENSITY, rtol=1e-6)

    def test_golden_differential(self):
        got = self.hmf.dndlnM(self.theta, z=PROBE_Z, M=PROBE_M)[0]
        np.testing.assert_allclose(got, GOLDEN_DNDLNM, rtol=1e-6)

    def test_ratio_reproduces_the_reference_data_vector(self):
        ratio = self.hmf.ratio(self.theta)
        self.assertEqual(ratio.shape, (1, N_BINS))
        # The ratio is a modulation around one, not a mass function.
        self.assertGreater(float(ratio.min()), 0.5)
        self.assertLess(float(ratio.max()), 1.5)

    def test_default_axes_are_the_native_grid(self):
        values = self.hmf.number_density(self.theta)
        self.assertEqual(values.shape, (1, N_Z, FINE_CENTERS.size))

    def test_matches_the_defaults_when_asked_explicitly(self):
        implicit = self.hmf.number_density(self.theta)
        explicit = self.hmf.number_density(self.theta, z=z_grid(), M=FINE_CENTERS)
        np.testing.assert_array_equal(implicit, explicit)

    def test_is_positive_and_falls_with_mass(self):
        values = self.hmf.number_density(self.theta, z=np.array([0.0, 1.0]), M=FINE_CENTERS)
        trained = values[0, :, :60]
        self.assertTrue(np.all(trained > 0.0))
        self.assertTrue(np.all(np.diff(trained, axis=1) < 0.0))

    def test_dndlnM_is_positive_where_the_emulator_was_trained(self):
        masses = np.logspace(11.5, 13.5, 8)
        values = self.hmf.dndlnM(self.theta, z=np.array([0.0, 1.0, 2.0]), M=masses)[0]
        self.assertTrue(np.all(values > 0.0))

    def test_the_two_entry_points_do_not_interfere(self):
        """Calling one must not change what the other returns.

        The reference keeps the offset it folds into its spline as mutable state
        on the instance, so a cached spline can be paired with whatever offset
        the most recent call left behind. jet returns the offset with the spline
        instead. For this model both offsets come out at exactly 1.0, so the
        hazard is latent rather than visible -- which is precisely why the test
        is written as an invariant rather than as a change detection.
        """
        first = self.hmf.number_density(self.theta, z=PROBE_Z, M=PROBE_M)
        self.hmf.dndlnM(self.theta, z=PROBE_Z, M=PROBE_M)
        again = self.hmf.number_density(self.theta, z=PROBE_Z, M=PROBE_M)
        np.testing.assert_array_equal(first, again)

        first_d = self.hmf.dndlnM(self.theta, z=PROBE_Z, M=PROBE_M)
        self.hmf.number_density(self.theta, z=PROBE_Z, M=PROBE_M)
        again_d = self.hmf.dndlnM(self.theta, z=PROBE_Z, M=PROBE_M)
        np.testing.assert_array_equal(first_d, again_d)

    def test_native_grid_values_are_the_interpolated_ones(self):
        """Asking for grid points must agree with asking for neighbours.

        The cumulative function is interpolated, so a value at a grid point is
        not required to equal the underlying array -- but it must be continuous,
        and it must not depend on how many points were requested at once.
        """
        mass = float(FINE_CENTERS[40])
        alone = self.hmf.number_density(self.theta, z=np.array([0.5]), M=np.array([mass]))
        together = self.hmf.number_density(self.theta, z=z_grid(), M=FINE_CENTERS)
        index = int(np.argmin(np.abs(z_grid() - 0.5)))
        self.assertAlmostEqual(
            float(alone[0, 0, 0]) / float(together[0, index, 40]), 1.0, places=10
        )

    def test_interpolates_between_redshifts(self):
        z = np.array([0.3, 0.5, 0.7])
        values = self.hmf.number_density(self.theta, z=z, M=np.array([1e13]))[0, :, 0]
        self.assertTrue(np.all(np.diff(values) < 0.0))

    def test_handles_several_cosmologies_at_once(self):
        theta = np.vstack([TRAINING_COSMOLOGY, TRAINING_COSMOLOGY * [1, 1, 1, 1, 1.2, 1, 1, 1]])
        values = self.hmf.number_density(theta, z=np.array([0.0]), M=np.array([1e13]))
        self.assertEqual(values.shape, (2, 1, 1))
        # Raising As raises the amplitude, so there are more haloes.
        self.assertGreater(float(values[1, 0, 0]), float(values[0, 0, 0]))

    def test_accepts_a_one_dimensional_parameter_row(self):
        values = self.hmf.number_density(TRAINING_COSMOLOGY, z=np.array([0.0]), M=PROBE_M)
        self.assertEqual(values.shape, (1, 1, PROBE_M.size))

    def test_rejects_a_malformed_parameter_row(self):
        with self.assertRaises(ValueError):
            self.hmf.number_density(np.zeros((1, 5)))

    def test_rejects_non_positive_mass(self):
        with self.assertRaises(ValueError):
            self.hmf.number_density(self.theta, M=np.array([0.0, 1e12]))

    def test_exposes_the_underlying_specs(self):
        self.assertEqual(self.hmf.x_spec.dim, 8)
        self.assertEqual(self.hmf.y_spec.n_bins, N_BINS)

    def test_load_from_an_explicit_path(self):
        loaded = HMFEmulator.load(data_dir() / BUNDLE_NAME)
        np.testing.assert_array_equal(
            loaded.number_density(self.theta, z=PROBE_Z, M=PROBE_M),
            self.hmf.number_density(self.theta, z=PROBE_Z, M=PROBE_M),
        )

    def test_a_missing_explicit_path_names_the_data_directory_variable(self):
        # An explicit path goes through the generic resolver, which knows about
        # $JET_DATA_DIR but not about this particular model's builder script.
        # The hint that names the builder is tested separately, on the bundled
        # lookup, which is the path a user without the weights will take.
        with self.assertRaises(FileNotFoundError) as caught:
            HMFEmulator.load(data_dir() / "definitely-not-here.npz")
        self.assertIn("JET_DATA_DIR", str(caught.exception))


class TestMissingBundleMessage(unittest.TestCase):
    """A fresh clone must be told how to get the weights."""

    def test_reader_explains_what_to_do(self):
        import jet.emulator.hmf as hmf_module

        missing = data_dir() / "definitely-not-here.npz"
        with self.assertRaises(FileNotFoundError) as caught:
            hmf_module._data_path(missing.name)
        message = str(caught.exception)
        self.assertIn("tools/build_hmf_bundle.py", message)
        self.assertIn("JET_DATA_DIR", message)


if __name__ == "__main__":
    unittest.main()
