"""Tests for the total-matter power-spectrum composition.

``pkmm`` combines the bundled :math:`P_{cb}` emulator with the bundled
:math:`P_{nn}/P_{cb}` ratio. These tests pin the two properties that matter:
``mnu = 0`` must reduce exactly to :math:`P_{cb}`, and for ``mnu > 0`` the
total-matter sigma8 must match the reference (and be below the cb value).
"""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from jet.emulator.pklin import BUNDLE_NAME, data_dir, load_pklin_emulator, sigma8_of_pk
from jet.emulator.pkmm import (
    BUNDLE_NAME as RATIO_BUNDLE_NAME,
)
from jet.emulator.pkmm import (
    load_pkmm_provider,
    ratio_at,
    sigma8_total,
    total_pk,
)

HAS_DATA = (data_dir() / BUNDLE_NAME).exists()
HAS_RATIO = (data_dir() / RATIO_BUNDLE_NAME).exists()
HAS_TOTAL_DATA = HAS_DATA and HAS_RATIO
HAS_CAMB = importlib.util.find_spec("camb") is not None

THETA_MNU = np.array([0.049, 0.31, 67.66, 0.9665, 2.105e-9, -1.0, 0.0, 0.06])
THETA_MNU0 = np.array([0.045, 0.28, 72.0, 0.95, 2.1e-9, -0.9, 0.3, 0.0])

#: csstemu's total-matter sigma8 for ``THETA_MNU``.
REFERENCE_SIGMA8_TOTAL = 0.8104854320171299


@unittest.skipUnless(HAS_TOTAL_DATA, "bundled P(k) or ratio data is not present")
class TestPkmm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pcb = load_pklin_emulator()

    def test_mnu_zero_reduces_to_pcb(self):
        theta = THETA_MNU0[None, :]
        composed = total_pk(theta, pcb=self.pcb.predict(theta, return_std=False))
        np.testing.assert_array_equal(composed, self.pcb.predict(theta, return_std=False))

    def test_ratio_is_positive_on_the_grid(self):
        ratio = ratio_at(THETA_MNU[None, :])
        self.assertEqual(ratio.shape, (1, 12000))
        self.assertTrue(np.all(ratio > 0.0))

    def test_total_sigma8_matches_the_reference(self):
        got = sigma8_total(THETA_MNU[None, :])[0]
        self.assertAlmostEqual(got / REFERENCE_SIGMA8_TOTAL - 1.0, 0.0, places=6)

    def test_total_sigma8_is_below_the_cb_value(self):
        pcb = self.pcb.predict(THETA_MNU[None, :], return_std=False)
        cb = float(sigma8_of_pk(pcb, R=8.0, z=0.0)[0])
        total = float(sigma8_total(THETA_MNU[None, :])[0])
        self.assertLess(total, cb)
        # The neutrino correction is sub-percent at this mass.
        self.assertGreater(total / cb, 0.99)

    def test_provider_reports_the_underlying_bounds(self):
        provider = load_pkmm_provider()
        np.testing.assert_array_equal(
            provider.x_spec.bounds_array(), self.pcb.x_spec.bounds_array()
        )

    @unittest.skipUnless(HAS_CAMB, "camb is not installed")
    def test_sigma8_total_agrees_with_camb(self):
        """The emulated total-matter sigma8 against CAMB's ``delta_tot``.

        The two are independent reconstructions -- a GP composition against a
        Boltzmann solve -- so the tolerance is set by the reference's own
        sub-percent agreement, not by the emulator's internal precision.
        """
        from jet._boltzmann import camb_sigma8

        thetas = np.array(
            [
                THETA_MNU,
                [0.055, 0.32, 68.0, 0.97, 2.3e-9, -1.1, 0.2, 0.10],
                THETA_MNU0,
            ]
        )
        for z in (0.0, 0.5):
            mine = sigma8_total(thetas, z=z)
            reference = np.array([camb_sigma8(row, R=8.0, z=z)[0] for row in thetas])
            np.testing.assert_allclose(mine, reference, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
