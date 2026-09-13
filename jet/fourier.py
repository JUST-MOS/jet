r"""
Hankel transforms between power spectra and correlation functions.

The :math:`P(k) \leftrightarrow \xi(r)` bridge used by the correlation-function
emulators, and the machinery behind the linear-bias baseline they blend onto at
large separations. The algorithm is FFTLog, which evaluates

.. math::
    f(y) = \int_0^\infty F(x)\,(xy)^q J_\mu(xy)\, y\,dx

with two real FFTs, so a transform over 1024 log-spaced points costs
milliseconds and returns the whole :math:`r` range at once. The alternative --
one spherical-Bessel quadrature per :math:`(z, r)` pair -- is straightforward
but far too slow to sit inside a likelihood.

**Provenance.** This is a port of ``hankl`` 1.1.0 by Minas Karamanis
(https://hankl.readthedocs.io, MIT licence), the copy vendored inside the
reference ``csstemu`` package. The mathematics and the numerical steps are
reproduced operation for operation on purpose: the reference emulators blend
their FFTLog :math:`\xi_{mm}` with a GP prediction, and the build scripts can
only cross-check the two implementations point by point while both run the same
algorithm. A mathematically equivalent rewrite would differ at the level of
FFTLog's low-ringing artefacts -- large enough to hide a structural error
behind a loosened tolerance. ``tools/build_xihm_bundle.py`` is where that
comparison happens.

The two ingredients of the core algorithm are the power-law bias :math:`q`
(which keeps the integrand well behaved at both ends) and the argument
:math:`(xy)`. The latter is fixed by :func:`_lowring_xy` when ``lowring`` is
set, which suppresses the ringing that otherwise leaks into the answer.

Nothing here is specific to any one emulator; the transforms are the standard
ones of galaxy clustering, so the module sits at the top level rather than
under :mod:`jet.emulator`.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma

__all__ = [
    "fftlog",
    "p2xi",
    "xi2p",
    "pk2xi",
    "xi2pk",
    "pk2wp",
    "pk2dwp",
]

#: Argument of the Gamma ratio at which the direct evaluation is abandoned in
#: favour of a Stirling expansion. Past this the difference of two large
#: log-Gammas loses all its significant digits.
_GAMMA_CUTOFF = 200.0


def _gamma_term(mu: float, x: np.ndarray, cutoff: float = _GAMMA_CUTOFF) -> np.ndarray:
    r"""Return :math:`\Gamma[(\mu+1+x)/2] / \Gamma[(\mu+1-x)/2]` for complex ``x``.

    Equation 16 of Hamilton (2000). For a purely real ``x`` a single ``gamma``
    call suffices; for a large imaginary part the ratio is evaluated from the
    Stirling expansion, because the two Gamma functions individually overflow
    long before their ratio does.

    Parameters
    ----------
    mu : float
        Order of the Bessel function :math:`J_\mu`; any real number.
    x : ndarray of complex
        Arguments, one per Fourier mode.
    cutoff : float, optional
        Absolute imaginary part beyond which the Stirling branch is used.

    Returns
    -------
    ndarray of complex
        The ratio, with the singular point :math:`x = \mu + 1` mapped to zero.
    """
    imag_x = np.imag(x)
    g_m = np.zeros(x.size, dtype=complex)

    asym_x = x[np.absolute(imag_x) > cutoff]
    asym_plus = (mu + 1.0 + asym_x) / 2.0
    asym_minus = (mu + 1.0 - asym_x) / 2.0

    good = (np.absolute(imag_x) <= cutoff) & (x != mu + 1.0 + 0.0j)
    x_good = x[good]
    g_m[good] = gamma((mu + 1.0 + x_good) / 2.0) / gamma((mu + 1.0 - x_good) / 2.0)

    # Stirling expansion of the log-gamma ratio, carried to 1/z^5.
    g_m[np.absolute(imag_x) > cutoff] = np.exp(
        (asym_plus - 0.5) * np.log(asym_plus)
        - (asym_minus - 0.5) * np.log(asym_minus)
        - asym_x
        + 1.0 / 12.0 * (1.0 / asym_plus - 1.0 / asym_minus)
        + 1.0 / 360.0 * (1.0 / asym_minus**3.0 - 1.0 / asym_plus**3.0)
        + 1.0 / 1260.0 * (1.0 / asym_plus**5.0 - 1.0 / asym_minus**5.0)
    )

    g_m[x == mu + 1.0 + 0.0j] = 0.0 + 0.0j

    return g_m


def _lowring_xy(mu: float, q: float, span: float, n: int, xy: float = 1.0) -> float:
    r"""Return the low-ringing value of :math:`xy` nearest the input.

    The transform is exact only in the limit of an infinite period; over a
    finite one the truncation aliases back as ringing. :math:`xy` may be chosen
    so that the aliased contribution is real, which costs nothing and removes
    most of it.

    Parameters
    ----------
    mu : float
        Order of :math:`J_\mu`.
    q : float
        Power-law bias exponent.
    span : float
        :math:`\ln(x_{\max} / x_{\min})` of the input grid.
    n : int
        Number of points in the (already padded) grid.
    xy : float, optional
        Starting value.

    Returns
    -------
    float
        The low-ringing value closest to ``xy``.
    """
    delta = span / float(n)
    x = q + 1j * np.pi / delta

    phi_plus = np.imag(np.log(gamma((mu + 1.0 + x) / 2.0)))
    phi_minus = np.imag(np.log(gamma((mu + 1.0 - x) / 2.0)))

    arg = np.log(2.0 / xy) / delta + (phi_plus - phi_minus) / np.pi
    iarg = np.rint(arg)
    if arg != iarg:
        xy = xy * np.exp((arg - iarg) * delta)

    return float(xy)


def _u_m_term(m: np.ndarray, mu: float, q: float, xy: float, span: float) -> np.ndarray:
    r"""Return the :math:`u_m(\mu, q)` weights of equation 18.

    Parameters
    ----------
    m : ndarray
        Fourier mode numbers.
    mu, q, xy, span : float
        Bessel order, bias exponent, transform argument, and
        :math:`\ln(x_{\max}/x_{\min})`.

    Returns
    -------
    ndarray of complex
        The mode weights. The Nyquist entry is forced real, since its imaginary
        part is an artefact of the FFT rather than a property of the transform.
    """
    omega = 1j * 2.0 * np.pi * m / float(span)
    x = q + omega
    # The two factors are multiplied in this order on purpose. Floating-point
    # multiplication does not associate, and the reference computes
    # ``(xy ** -omega) * (2 ** x * gamma_term)``; writing it any other way
    # shifts the last couple of digits and costs the bit-for-bit comparison
    # that the build scripts rely on.
    u_m = xy ** (-omega) * (2**x * _gamma_term(mu, x))
    u_m[m.size - 1] = np.real(u_m[m.size - 1])
    return u_m


def _padding(
    x: np.ndarray,
    f: np.ndarray,
    ext_left: int = 0,
    ext_right: int = 0,
    n_ext: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    r"""Extend a log-spaced grid to a power-of-two length on either side.

    Parameters
    ----------
    x : ndarray
        Uniformly log-spaced abscissae.
    f : ndarray
        Values at ``x``.
    ext_left, ext_right : int, optional
        Extrapolation mode per end: ``0`` none, ``1`` zero, ``2`` constant,
        ``3`` power law. A grid is only extended at an end whose mode is
        positive.
    n_ext : int, optional
        Additional doubling, so that the padded length is
        :math:`2^{\lceil\log_2 N\rceil + n_\mathrm{ext}}`.

    Returns
    -------
    x_pad, f_pad : ndarray
        The extended arrays, or the inputs unchanged when no extension is due.
    n_left, n_right : int
        Number of points added at each end.
    """
    n = x.size
    if n < 2:
        raise ValueError("Size of input arrays needs to be larger than 2")
    n_prime = 2 ** ((n - 1).bit_length() + n_ext)

    if n_prime <= n:
        return x, f, 0, 0

    n_tails = n_prime - n
    if ext_left > 0 and ext_right > 0:
        n_left = n_tails // 2
        n_right = n_tails - n_left
    elif ext_left > 0 and ext_right < 1:
        n_left = n_tails
        n_right = 0
    elif ext_left < 1 and ext_right > 0:
        n_left = 0
        n_right = n_tails
    elif ext_left < 1 and ext_right < 1:
        return x, f, 0, 0
    else:
        raise ValueError("Please provide valid values for ext argument (i.e. 0, 1, 2, 3)")

    delta = (np.log10(np.max(x)) - np.log10(np.min(x))) / float(n - 1)
    x_prime = np.logspace(
        np.log10(x[0]) - n_left * delta, np.log10(x[-1]) + n_right * delta, n_prime
    )

    if n_left > 0:
        if ext_left == 1:
            f_left = np.zeros(n_left)
        elif ext_left == 2:
            f_left = np.full(n_left, f[0])
        elif ext_left == 3:
            f_left = f[0] * (f[1] / f[0]) ** np.arange(-n_left, 0)
        else:
            raise ValueError("Please provide valid values for ext argument (i.e. 0, 1, 2, 3)")
    else:
        f_left = np.array([])

    if n_right > 0:
        if ext_right == 1:
            f_right = np.zeros(n_right)
        elif ext_right == 2:
            f_right = np.full(n_right, f[-1])
        elif ext_right == 3:
            f_right = f[-1] * (f[-1] / f[-2]) ** np.arange(1, n_right + 1)
        else:
            raise ValueError("Please provide valid values for ext argument (i.e. 0, 1, 2, 3)")
    else:
        f_right = np.array([])

    # The power-law branch divides by the second sample at each end, so a tail
    # that has underflowed to zero turns the padding into nan. That is reachable
    # in normal use -- a power spectrum decays faster than any power law at high
    # k -- and the reference drops those nan to zero, so the padding is zero
    # there rather than nan.
    f_prime = np.nan_to_num(np.concatenate((f_left, f, f_right)))

    return x_prime, f_prime, n_left, n_right


def preprocess(
    x: np.ndarray,
    f: np.ndarray,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extend a log-spaced grid to a power-of-two length.

    FFTLog assumes the input is one period of a periodic function, so the
    arrays are grown until they hold a power of two samples, and the values in
    the newly added tails are chosen by ``ext``. Extending ``range`` past what
    the first doubling reaches repeats the operation.

    Parameters
    ----------
    x : ndarray
        Uniformly log-spaced abscissae.
    f : ndarray
        Values at ``x``.
    ext : int or tuple of int, optional
        Extrapolation mode: ``0`` none, ``1`` zero padding, ``2`` constant,
        ``3`` power-law. A tuple gives ``(left, right)`` separately.
    range : tuple of float, optional
        Minimum range to extend to, as ``(x_min, x_max)``. ``None`` stops at
        the first power of two.

    Returns
    -------
    x_ext, f_ext : ndarray
        The extended arrays.
    n_left, n_right : int
        Number of points added at each end, so the result can be cropped back.
    """
    if range is not None:
        try:
            x_min, x_max = range
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Please enter valid x range in the form of a tuple (x_min, x_max) "
                "or list [x_min, x_max]."
            ) from exc
    else:
        x_min, x_max = None, None

    try:
        ext_left, ext_right = ext
    except TypeError:
        ext_left = ext_right = ext

    x, f, n_left, n_right = _padding(x, f, ext_left=ext_left, ext_right=ext_right, n_ext=0)

    if x_min is not None and x_max is not None:
        if ext_left > 0 and ext_right > 0:
            while x[0] > x_min and x[-1] < x_max:
                x, f, dl, dr = _padding(x, f, ext_left=ext_left, ext_right=ext_right, n_ext=1)
                n_left += dl
                n_right += dr

    if x_min is not None:
        if ext_left > 0:
            while x[0] > x_min and (x_max is None or x[-1] >= x_max):
                x, f, dl, dr = _padding(x, f, ext_left=ext_left, ext_right=0, n_ext=1)
                n_left += dl
                n_right += dr

    if x_max is not None:
        if ext_right > 0:
            while x[-1] < x_max and (x_min is None or x[0] <= x_min):
                x, f, dl, dr = _padding(x, f, ext_left=0, ext_right=ext_right, n_ext=1)
                n_left += dl
                n_right += dr

    return x, f, n_left, n_right


def fftlog(
    x: np.ndarray,
    f_x: np.ndarray,
    q: float,
    mu: float,
    xy: float = 1.0,
    lowring: bool = False,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
    return_ext: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Hankel transform of a log-spaced series.

    Computes

    .. math::
        f(y) = \int_0^\infty F(x)\,(xy)^q J_\mu(xy)\, y\, dx

    by the FFTLog algorithm of Talman (1978) and Hamilton (2000): the integral
    becomes a convolution in :math:`\ln x`, which the FFT evaluates in
    :math:`O(N \ln N)` and, because both grids are log-spaced, for every
    :math:`y` at once.

    Parameters
    ----------
    x : ndarray
        Uniformly log-spaced abscissae.
    f_x : ndarray
        Values of :math:`F(x)` at ``x``.
    q : float
        Power-law bias exponent, positive for a forward transform and negative
        for an inverse one.
    mu : float
        Order of :math:`J_\mu`.
    xy : float, optional
        Transform argument.
    lowring : bool, optional
        Pick the low-ringing :math:`xy` nearest to ``xy`` instead of using it
        as given.
    ext : int or tuple of int, optional
        Extrapolation mode; see :func:`preprocess`.
    range : tuple of float, optional
        Minimum range to extend to; see :func:`preprocess`.
    return_ext : bool, optional
        Return the untrimmed arrays. By default the padding is cropped back off,
        so ``y`` spans the same range as ``x``.

    Returns
    -------
    y, f_y : ndarray
        The conjugate abscissae and transformed values.

    References
    ----------
    J. D. Talman. Numerical Fourier and Bessel Transforms in Logarithmic
    Variables. Journal of Computational Physics, 29:35-48, 1978.

    A. J. S. Hamilton. Uncorrelated modes of the non-linear power spectrum.
    MNRAS, 312:257-284, 2000.
    """
    if mu + 1.0 + q == 0.0:
        raise ValueError("The FFTLog Hankel Transform is singular when mu + 1 + q = 0.")

    x, f_x, n_left, n_right = preprocess(x, f_x, ext=ext, range=range)

    n = f_x.size
    span = np.log(np.max(x)) - np.log(np.min(x))
    delta = span / float(n - 1)
    x0 = np.exp(np.log(x[n // 2]))

    c_m = np.fft.rfft(f_x)
    m = np.fft.rfftfreq(n, d=1.0) * float(n)

    if lowring:
        xy = _lowring_xy(mu, q, span, n, xy)

    log_y0 = np.log(xy / x0)

    m_y = np.arange(-n // 2, n // 2)
    id_ = np.fft.fftshift(m_y)

    s = delta * (-m_y) + log_y0
    y = 10 ** (s[id_] / np.log(10))

    b = c_m * _u_m_term(m, mu, q, xy, span)
    a_m = np.fft.irfft(b)
    f_y = a_m[id_]

    f_y = f_y[::-1]
    y = y[::-1]

    if q != 0:
        f_y = f_y * y ** (-float(q))

    if return_ext:
        return y, f_y
    if n_right == 0:
        return y[n_left:], f_y[n_left:]
    return y[n_left:-n_right], f_y[n_left:-n_right]


def p2xi(
    k: np.ndarray,
    p: np.ndarray,
    l: int,  # noqa: E741
    n: int = 0,
    lowring: bool = False,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
    return_ext: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert a power-spectrum multipole to a correlation-function multipole.

    Computes

    .. math::
        \xi_l^{(n)}(r) = i^l \int_0^\infty \frac{k^2 dk}{2\pi^2}\, (kr)^{-n}
        P_l^{(n)}(k)\, j_l(kr)

    which is :func:`fftlog` with :math:`\mu = l + 1/2` and a bias
    :math:`q = -n`, followed by the powers of :math:`k`, :math:`r` and
    :math:`i` that turn Bessel-:math:`J` into spherical-Bessel-:math:`j_l`.

    Parameters
    ----------
    k : ndarray
        Uniformly log-spaced wavenumbers in :math:`h/\mathrm{Mpc}`.
    p : ndarray
        Power spectrum values, in :math:`(\mathrm{Mpc}/h)^3`.
    l : int
        Multipole degree.
    n : int, optional
        Expansion order; ``0`` is the plane-parallel case.
    lowring, ext, range, return_ext
        Passed to :func:`fftlog`.

    Returns
    -------
    r, xi : ndarray
        Separations and the (complex) multipole. The imaginary part vanishes
        for a real input; callers usually take ``np.real``.
    """
    r, f = fftlog(
        k,
        p * k**1.5,
        q=-n,
        mu=l + 0.5,
        lowring=lowring,
        ext=ext,
        range=range,
        return_ext=return_ext,
    )
    return r, f * (2.0 * np.pi) ** (-1.5) * r ** (-1.5) * (1j) ** l


def xi2p(
    r: np.ndarray,
    xi: np.ndarray,
    l: int,  # noqa: E741
    n: int = 0,
    lowring: bool = False,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
    return_ext: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert a correlation-function multipole to a power-spectrum multipole.

    The inverse of :func:`p2xi`:

    .. math::
        P_l^{(n)}(k) = 4\pi (-i)^l \int_0^\infty r^2 dr\, (kr)^n \xi_l^{(n)}(r)\, j_l(kr)

    Parameters
    ----------
    r : ndarray
        Uniformly log-spaced separations in :math:`h/\mathrm{Mpc}`.
    xi : ndarray
        Correlation function values.
    l : int
        Multipole degree.
    n : int, optional
        Expansion order.
    lowring, ext, range, return_ext
        Passed to :func:`fftlog`.

    Returns
    -------
    k, p : ndarray
        Wavenumbers and the (complex) multipole.
    """
    k, f = fftlog(
        r,
        xi * r**1.5,
        q=n,
        mu=l + 0.5,
        lowring=lowring,
        ext=ext,
        range=range,
        return_ext=return_ext,
    )
    return k, f * (2.0 * np.pi) ** 1.5 * k ** (-1.5) * (-1j) ** l


def pk2xi(
    k: np.ndarray,
    pk: np.ndarray,
    lowring: bool = False,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert a real-space isotropic power spectrum to :math:`\xi(r)`.

    .. math::
        \xi(r) = \int_0^\infty \frac{k^2 dk}{2\pi^2}\, P(k)\, j_0(kr)

    Parameters
    ----------
    k : ndarray
        Uniformly log-spaced wavenumbers in :math:`h/\mathrm{Mpc}`.
    pk : ndarray
        Power spectrum in :math:`(h^{-1}\mathrm{Mpc})^3`.
    lowring, ext, range
        Passed to :func:`fftlog`.

    Returns
    -------
    r, xi : ndarray
        Separations in :math:`h^{-1}\mathrm{Mpc}` and the real correlation
        function.
    """
    r, xi = p2xi(k, pk, l=0, n=0, lowring=lowring, ext=ext, range=range)
    return r, np.real(xi)


def xi2pk(
    r: np.ndarray,
    xi: np.ndarray,
    lowring: bool = False,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert a real-space correlation function to :math:`P(k)`.

    .. math::
        P(k) = 4\pi \int_0^\infty r^2 dr\, \xi(r)\, j_0(kr)

    Parameters
    ----------
    r : ndarray
        Uniformly log-spaced separations in :math:`h^{-1}\mathrm{Mpc}`.
    xi : ndarray
        Correlation function.
    lowring, ext, range
        Passed to :func:`fftlog`.

    Returns
    -------
    k, pk : ndarray
        Wavenumbers and the real power spectrum.
    """
    k, pk = xi2p(r, xi, l=0, n=0, lowring=lowring, ext=ext, range=range)
    return k, np.real(pk)


def pk2wp(
    k: np.ndarray,
    pk: np.ndarray,
    lowring: bool = False,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert a power spectrum to the projected correlation function.

    .. math::
        w_p(r_p) = \int_0^\infty \frac{k\,dk}{2\pi}\, P(k)\, J_0(k r_p)

    The cylindrical Bessel function of order zero is the one-dimensional
    projection of :math:`\xi(r)`, so no spherical factor appears.

    Parameters
    ----------
    k : ndarray
        Uniformly log-spaced wavenumbers in :math:`h/\mathrm{Mpc}`.
    pk : ndarray
        Power spectrum in :math:`(h^{-1}\mathrm{Mpc})^3`.
    lowring, ext, range
        Passed to :func:`fftlog`.

    Returns
    -------
    rp, wp : ndarray
        Projected separations and :math:`w_p` in :math:`h^{-1}\mathrm{Mpc}`.
    """
    r, f = fftlog(k, k * pk / (2.0 * np.pi), q=0, mu=0, lowring=lowring, ext=ext, range=range)
    # The transform returns r * w_p(r); divide it back out.
    return r, f / r


def pk2dwp(
    k: np.ndarray,
    pk: np.ndarray,
    lowring: bool = False,
    ext: int | tuple[int, int] = 0,
    range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Convert a power spectrum to the excess-surface-density kernel.

    .. math::
        \Delta\tilde{\Sigma}(R) = \int_0^\infty \frac{k\,dk}{2\pi}\, P(k)\, J_2(kR)

    .. note::
        This is *without* the mean matter density factor
        :math:`\bar{\rho}_m`. Multiply the result by it to obtain the physical
        :math:`\Delta\Sigma(R)`.

    Parameters
    ----------
    k : ndarray
        Uniformly log-spaced wavenumbers in :math:`h/\mathrm{Mpc}`.
    pk : ndarray
        Power spectrum in :math:`(h^{-1}\mathrm{Mpc})^3`.
    lowring, ext, range
        Passed to :func:`fftlog`.

    Returns
    -------
    R, ds : ndarray
        Projected separations and the unscaled kernel.
    """
    r, f = fftlog(k, k * pk / (2.0 * np.pi), q=0, mu=2, lowring=lowring, ext=ext, range=range)
    return r, f / r
