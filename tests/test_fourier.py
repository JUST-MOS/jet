"""
Tests for the Hankel transforms.

Nothing here needs bundled weights: the transforms are closed-form and are
checked against the closed forms, plus a direct quadrature of the same integral.
The point of the tight comparison against the reference implementation is not
testable from pytest -- it lives in ``tools/build_xihm_bundle.py``, which needs
a checkout of the reference to run at all -- so what is pinned here is that the
transforms compute the right integral, not that they compute it bit for bit the
way the reference does.

The tolerances are loose on purpose. FFTLog is an FFT over a finite,
log-spaced period, so its answer carries a ringing floor of order ``1e-9`` of
the peak; the comparison window below therefore stops where the signal is still
far above that floor. Tightening the tolerance would only make the test fail on
the ringing, not on anything real.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn

from jet.fourier import fftlog, pk2dwp, pk2wp, pk2xi, xi2pk

#: Power spectrum used throughout: ``P(k) = exp(-c k^2)``.
_C = 1.0

#: Wavenumbers to transform. Wide enough that the Gaussian is resolved at both
#: ends, and a power of two so that no padding is inserted.
_K = np.logspace(-5, 3, 1024)


def _power_spectrum(k: np.ndarray) -> np.ndarray:
    """``exp(-k^2)``, a Gaussian whose transform has a closed form."""
    return np.exp(-_C * k**2)


def _analytic_xi(r: np.ndarray) -> np.ndarray:
    """The exact transform of :func:`_power_spectrum`.

    For :math:`P(k) = e^{-c k^2}`,

    .. math::
        \\xi(r) = \\frac{1}{8\\pi^{3/2} c^{3/2}} e^{-r^2/(4c)}

    from :math:`\\int_0^\\infty k^2 e^{-ck^2} j_0(kr)\\,dk
    = \\frac{\\sqrt{\\pi}}{4c^{3/2}} e^{-r^2/(4c)}`.
    """
    return np.exp(-(r**2) / (4.0 * _C)) / (8.0 * np.pi**1.5 * _C**1.5)


def _quadrature_xi(r: float) -> float:
    """The same integral by adaptive quadrature in :math:`\\ln k`."""

    def integrand(ln_k: float) -> float:
        k = np.exp(ln_k)
        return k**3 * np.exp(-_C * k**2) * spherical_jn(0, k * r) / (2.0 * np.pi**2)

    return quad(integrand, np.log(_K[0]), np.log(_K[-1]), limit=500)[0]


class TestPk2xi(unittest.TestCase):
    """The spherical-Bessel transform, against its closed form."""

    def setUp(self) -> None:
        self.r, self.xi = pk2xi(_K, _power_spectrum(_K))
        # Only compare where the signal is far above FFTLog's ringing floor,
        # which for this normalisation sits near 1e-9.
        self.window = (self.r > 0.2) & (self.r < 3.0)

    def tearDown(self) -> None:
        pass

    def test_shapes_and_realness(self) -> None:
        self.assertEqual(self.r.shape, _K.shape)
        self.assertEqual(self.xi.shape, _K.shape)
        self.assertTrue(np.all(np.isfinite(self.xi)))

    def test_matches_the_closed_form(self) -> None:
        expected = _analytic_xi(self.r[self.window])
        np.testing.assert_allclose(self.xi[self.window], expected, rtol=5e-2)

    def test_matches_direct_quadrature(self) -> None:
        """An independent evaluation of the same integral, not of the same code."""
        for probe in (0.2, 0.5, 1.0, 2.0, 3.0):
            index = int(np.argmin(np.abs(self.r - probe)))
            self.assertAlmostEqual(
                float(self.xi[index] / _quadrature_xi(float(self.r[index]))),
                1.0,
                delta=5e-2,
                msg=f"mismatch at r = {self.r[index]:.4f}",
            )

    def test_round_trip_recovers_the_power_spectrum(self) -> None:
        """Transforming back must return the spectrum, ringing aside."""
        k_back, p_back = xi2pk(self.r, self.xi)
        usable = (k_back > 1e-3) & (k_back < 3.0) & (p_back > 1e-3 * p_back.max())
        self.assertGreater(int(usable.sum()), 100)
        # Compared in the log, which is where a wrong normalisation shows up as
        # an additive offset rather than as a badly scaled ratio.
        np.testing.assert_allclose(
            np.log(p_back[usable]), np.log(_power_spectrum(k_back[usable])), atol=0.2
        )


class TestPk2Wp(unittest.TestCase):
    """The cylindrical transforms used by the projected statistics."""

    def setUp(self) -> None:
        self.pk = _power_spectrum(_K)

    def tearDown(self) -> None:
        pass

    def test_projected_correlation_is_positive_and_decaying(self) -> None:
        r_p, w_p = pk2wp(_K, self.pk)
        self.assertEqual(r_p.shape, w_p.shape)
        self.assertTrue(np.all(np.isfinite(w_p)))
        # w_p is a line-of-sight integral of a correlation function that is
        # positive on these scales, so it must be positive and fall with r_p.
        self.assertGreater(float(w_p[0]), 0.0)
        self.assertLess(float(w_p[-1]), float(w_p[0]))

    def test_excess_surface_density_kernel_is_finite(self) -> None:
        """``pk2dwp`` uses ``J_2``, which changes sign; only finiteness is pinned."""
        r, ds = pk2dwp(_K, self.pk)
        self.assertEqual(r.shape, ds.shape)
        self.assertTrue(np.all(np.isfinite(ds)))
        # J_2 is positive at the origin and this kernel must inherit that sign
        # there, unlike w_p's all-positive J_0.
        self.assertGreater(float(ds[0]), 0.0)


class TestFftlogCore(unittest.TestCase):
    """The core transform's argument handling and failure modes."""

    def setUp(self) -> None:
        self.f = np.exp(-(_K**2))

    def tearDown(self) -> None:
        pass

    def test_the_singular_bias_raises(self) -> None:
        """``mu + 1 + q = 0`` is a pole of the Gamma ratio and must not be silently run."""
        with self.assertRaises(ValueError) as caught:
            fftlog(_K, self.f, q=-1.5, mu=0.5)
        self.assertIn("singular", str(caught.exception))

    def test_a_power_of_two_grid_is_left_alone(self) -> None:
        """With no padding due, the conjugate grid is returned unchanged in size."""
        y, _ = fftlog(_K, self.f, q=0.0, mu=0.5, ext=1)
        self.assertEqual(y.size, _K.size)

    def test_the_conjugate_grid_is_the_reciprocal_of_the_input(self) -> None:
        """``y * x`` is constant: the transform exchanges a length for its inverse."""
        y, _ = fftlog(_K, self.f, q=0.0, mu=0.5)
        # Working in the product rather than in log y keeps the comparison at
        # the precision of the exponentials; taking logs of 10**s values loses
        # a couple of digits and the constant is then only good to ~1e-13.
        np.testing.assert_allclose(y[::-1] * _K, y[-1] * _K[0], rtol=1e-10)

    def test_the_log_spacing_is_preserved(self) -> None:
        y, _ = fftlog(_K, self.f, q=0.0, mu=0.5)
        delta_x = np.log(_K[1]) - np.log(_K[0])
        delta_y = np.log(y[1]) - np.log(y[0])
        self.assertAlmostEqual(delta_y, delta_x, places=12)

    def test_extrapolation_is_inserted_and_trimmed_back(self) -> None:
        """A grid that is *not* a power of two is where padding actually happens."""
        k = _K[:600]
        f = 1.0 / (1.0 + k**2) ** 2
        y, f_y = fftlog(k, f, q=0.0, mu=0.5, ext=1)
        self.assertEqual(y.size, k.size)
        y_ext, f_ext = fftlog(k, f, q=0.0, mu=0.5, ext=1, return_ext=True)
        self.assertEqual(y_ext.size, 1024)
        self.assertGreater(y_ext.size, k.size)
        self.assertEqual(f_ext.size, y_ext.size)

    def test_a_requested_range_is_reached(self) -> None:
        """``range`` extends until the padded grid covers it, in repeated doublings."""
        k = _K[:600]
        f = 1.0 / (1.0 + k**2) ** 2
        y, _ = fftlog(k, f, q=0.0, mu=0.5, ext=3, range=(1e-8, 1e8), return_ext=True)
        self.assertLessEqual(float(y[0]), 1e-8)
        self.assertGreaterEqual(float(y[-1]), 1e8)


if __name__ == "__main__":
    unittest.main()
