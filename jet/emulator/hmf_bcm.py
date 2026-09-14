r"""
The bundled cumulative halo mass function with baryonic feedback.

Predicts :math:`n(\geq M)` for a halo population whose gas physics is described
by the Arico et al. baryon correction model, at a **fixed cosmology and
redshift**. It is the one bundled model that takes no cosmological parameters
at all: its four inputs are the feedback parameters, declared under
``block="baryon"``. That is the pairing jet's design anticipated -- one
:class:`~jet.spec.ParameterSpec` per input combination, never a masked model --
and this is its first instance.

.. warning::
    The weights encode **one cosmology (``c0000``) at one redshift
    (:math:`z = 0.5`)**, and the model has no axis along which either can vary.
    The mass grid and both of those facts travel in the bundle as provenance;
    they are not extrapolated over.

The four parameters
-------------------
``logMc``, ``thej``, ``mu`` and ``delta`` are, in order, :math:`\log_{10}` of
the halo mass above which feedback suppresses the gas, the AGN temperature, the
fraction of gas ejected, and the slope of the ejection. Their bounds are the
training box.

Simplicity, relative to the cosmological mass function
-----------------------------------------------------
:mod:`jet.emulator.hmf` carries a twelve-redshift box with variable-length
per-redshift mass blocks, an analytic Castro23 baseline and a two-level
interpolation, because the reference it reproduces does. This one has none of
that: a single fixed mass grid, a plain regression, and no baseline. The output
is a rectangle and the input is four numbers, so the module is mostly
documentation.

Why the kernel is a plain Matern
--------------------------------
The reference stores a fitted ``WhiteKernel`` noise level alongside the
``ConstantKernel * Matern`` amplitude and length scales, and builds a kernel of
``ConstantKernel * Matern + WhiteKernel`` from them. A ``WhiteKernel`` is the
usual way to say "these training points carry noise": it enters the training
covariance and deliberately drops out of the cross-covariance at prediction
time, since a new point's noise is not correlated with the training set's.

What the reference's stored dual coefficients show is that on these weights the
term contributed nothing to the arithmetic either way -- they reproduce
``ConstantKernel * Matern`` with the same negligible floor every other bundled
model uses, not that kernel plus the stored noise level. So the bundle carries
:data:`ALPHA` and not ``noise + alpha``, and the stored levels ride along in the
manifest as provenance. Reproducing the numbers is the point; the levels are
recorded so that the day they start mattering, the difference is visible rather
than mysterious.

The practical consequence is that this model is an *almost* exact interpolant:
with the floor at ``1e-10`` the process sits ``alpha`` away from its targets
rather than on them. That is invisible in the score space it is fitted in and
becomes ``1e-7`` in the abundance, because the transform chain exponentiates and
the smallest bin is of order ``1e-3``. The build script leans on the reference's
own process rather than on its stored table for that reason; see its docstring.

.. warning::
    The weights under ``jet/data`` are not tracked by git; see the note in
    ``jet/data/README.md``. They are rebuilt from the reference's published
    arrays by ``tools/build_hmf_bcm_bundle.py``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..spec import DataVectorSpec, Param, ParameterSpec
from .emulator import Emulator, data_dir, resolve_weights_path

__all__ = [
    "BUNDLE_NAME",
    "MASS_GRID_KEY",
    "N_BINS",
    "PARAMETER_DEFAULTS",
    "ALPHA",
    "REFERENCE_VOLUME_SCALE",
    "data_dir",
    "theta_spec",
    "data_vector_spec",
    "mass_grid",
    "bin_width",
    "load_bcm_hmf_emulator",
    "BCMHFEmulator",
]

#: Name of the ``.npz`` bundle under the data directory.
BUNDLE_NAME = "hmf_bcm_m200m.gp.npz"

#: Bundle key holding the mass grid of the data vector, in :math:`M_\odot/h`.
#: It travels inside the bundle rather than as a module constant so that a
#: weights file is self-describing, and so that the reader never has to assume
#: the grid is the one the reference happened to use.
MASS_GRID_KEY = "mass_grid"

#: Entries in the data vector, one per mass bin.
N_BINS = 46

#: The reference's default parameter point, in :func:`theta_spec` order. Not a
#: special point scientifically -- it is the reference's own fallback when a
#: caller sets only some of the four -- but a valid one, and a better starting
#: point than anything a caller would invent.
PARAMETER_DEFAULTS = np.array([13.0, 3.0, 1.0, 4.0])

#: Noise floor added to the training covariance. The reference stores a fitted
#: ``WhiteKernel`` level too, but see the module docstring: its arithmetic never
#: uses it, so neither does this. The value matches every other bundled model.
ALPHA = 1e-10

#: Factor taking the reference's internal abundance unit to :math:`(h/\mathrm{Mpc})^3`.
#:
#: The stored weights predict the abundance in :math:`(h/\mathrm{Gpc})^3`, which
#: is what the reference's model class returns and therefore what this bundle's
#: Gaussian process outputs. Its *published* accessor applies this factor before
#: returning, and so does :meth:`BCMHFEmulator.cumulative` -- but the raw
#: :attr:`BCMHFEmulator.emulator` does not, which is a factor of ``1e9`` between
#: two things that look alike.
REFERENCE_VOLUME_SCALE = 1e-9


def theta_spec() -> ParameterSpec:
    r"""Return the input specification of the bundled emulator.

    Four baryonic feedback parameters, all under ``block="baryon"``. There are
    no cosmological parameters: this model was trained at one fixed cosmology
    and has no axis for it.

    Returns
    -------
    ParameterSpec
        Four parameters, in the reference's column order.
    """
    return ParameterSpec(
        [
            Param("logMc", bounds=(10.0, 16.0), block="baryon", doc="log10 Mc [Msun/h]"),
            Param("thej", bounds=(1.0, 10.0), block="baryon", doc="log10 T_AGN [K]"),
            Param("mu", bounds=(0.1, 10.0), block="baryon", doc="ejected gas fraction"),
            Param("delta", bounds=(4.0, 10.0), block="baryon", doc="ejection slope"),
        ]
    )


def data_vector_spec() -> DataVectorSpec:
    """Return the output specification of the bundled emulator.

    ``log`` is true because the reference emulates :math:`\\log_{10}` of the
    cumulative abundance and exponentiates at the end; declaring it here is what
    puts :class:`~jet.emulator.transforms.Log10` into the transform chain.
    """
    return DataVectorSpec(name="hmf_bcm_m200m", n_bins=N_BINS, log=True)


def _data_path(name: str) -> Path:
    """Resolve one bundled data file, auto-fetching from GitHub Release if needed."""
    try:
        return resolve_weights_path(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"the bundled data file {name!r} is missing from {data_dir()}. It is not "
            "tracked by git; regenerate it with tools/build_hmf_bcm_bundle.py, or set "
            "JET_DATA_DIR to a directory that already holds it."
        ) from exc


@lru_cache(maxsize=1)
def _extra_arrays() -> dict[str, np.ndarray]:
    """Return the bundle's auxiliary arrays, read once per process.

    A missing bundle is not an error here, so that the grid accessor can be
    exercised on a copy of the package that has no weights. Callers that want
    the *values* go through :func:`load_bcm_hmf_emulator`, which does raise.
    """
    from .bundle import load_bundle

    try:
        return dict(load_bundle(_data_path(BUNDLE_NAME))["extra_arrays"])
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=1)
def mass_grid() -> np.ndarray:
    r"""Return the masses the data vector is evaluated at, in :math:`M_\odot/h`.

    The grid is the reference's: the lower edges of the bins of
    ``numpy.logspace(10, 16, 61)[14:]``. It is uniform in
    :math:`\log_{10} M`, which is what lets :func:`bin_width` describe the whole
    spacing with one number, and what makes :func:`BCMHFEmulator.dndlgM` a plain
    difference rather than a quadrature.

    Returns
    -------
    ndarray of shape (N_BINS,)
        Ascending masses, in :math:`M_\odot/h`.
    """
    extras = _extra_arrays()
    if MASS_GRID_KEY not in extras:
        raise KeyError(
            f"the bundled emulator has no {MASS_GRID_KEY!r} array; it was written "
            "without one and cannot be interpreted. Rebuild it with "
            "tools/build_hmf_bcm_bundle.py"
        )
    return extras[MASS_GRID_KEY]


@lru_cache(maxsize=1)
def bin_width() -> float:
    r"""Return the :math:`\log_{10}` spacing of :func:`mass_grid`.

    Computed the way the reference computes its ``lgM``, from the first two
    entries of the edge array. The grid is asserted uniform by the builder, so
    this one number describes every bin.
    """
    grid = mass_grid()
    if grid.size < 2:
        raise ValueError("the mass grid needs at least two entries to have a spacing")
    width = float(np.log10(grid[1]) - np.log10(grid[0]))
    if not np.allclose(np.diff(np.log10(grid)), width, rtol=1e-12, atol=0.0):
        raise ValueError("the bundled mass grid is not uniform; dndlgM would be wrong")
    return width


@lru_cache(maxsize=1)
def load_bcm_hmf_emulator() -> Emulator:
    r"""Load the bundled baryonic mass function emulator.

    The result is cached; the file is read once per process.

    Returns
    -------
    Emulator
        Predicts :math:`n(\geq M)` in :math:`(h/\mathrm{Mpc})^3` on
        :func:`mass_grid`, in the reference's column order.

    Raises
    ------
    FileNotFoundError
        If the data file is absent.
    """
    # Extrapolation warnings are suppressed for the same reason as in the other
    # bundled readers: this model is always probed over whatever box the caller
    # has in mind, and the check that matters is the one the wrapper does
    # against the parameter bounds.
    emulator = Emulator.load(_data_path(BUNDLE_NAME))
    emulator.warn_on_extrapolation = False
    return emulator


class BCMHFEmulator:
    r"""The bundled cumulative halo mass function with baryonic feedback.

    Wraps the generic :class:`~jet.emulator.Emulator`. There is no baseline to
    multiply back in and no axis to interpolate over, so this class is thin:
    it names the two things the regression is worth reading as, and leaves the
    rest to the emulator it holds.

    Parameters
    ----------
    emulator : Emulator, optional
        The fitted model. Defaults to the bundled one.
    """

    def __init__(self, emulator: Emulator | None = None) -> None:
        self._emulator = load_bcm_hmf_emulator() if emulator is None else emulator

    @classmethod
    def load(cls, path: str | Path | None = None) -> BCMHFEmulator:
        """Load a model from ``path``, or the bundled one."""
        if path is None:
            return cls()
        emulator = Emulator.load(path)
        emulator.warn_on_extrapolation = False
        return cls(emulator)

    @property
    def x_spec(self) -> ParameterSpec:
        """Input specification of the wrapped model."""
        return self._emulator.x_spec

    @property
    def y_spec(self) -> DataVectorSpec:
        """Output specification of the wrapped model."""
        return self._emulator.y_spec

    @property
    def emulator(self) -> Emulator:
        r"""The underlying generic emulator, for callers that want the raw model.

        .. warning::
            Its predictions are in :math:`(h/\mathrm{Gpc})^3`, not the
            :math:`(h/\mathrm{Mpc})^3` that :meth:`cumulative` returns -- a
            factor of :data:`REFERENCE_VOLUME_SCALE`. The two accessors differ
            because the reference's model class and its published wrapper differ
            the same way, and reproducing both is how the values stay
            comparable with each.
        """
        return self._emulator

    def log10_cumulative(self, theta: np.ndarray) -> np.ndarray:
        r"""Return :math:`\log_{10} n(\geq M)`.

        The quantity the Gaussian process actually predicts, before the
        exponentiation the transform chain undoes. Useful because a scatter in
        the log is a relative error, which is how these abundances are usually
        compared.

        Parameters
        ----------
        theta : ndarray of shape (n_samples, 4)
            Feedback parameters in :func:`theta_spec` order.

        Returns
        -------
        ndarray of shape (n_samples, N_BINS)
            :math:`\log_{10} n(\geq M)`, with ``n`` in
            :math:`(h/\mathrm{Mpc})^3`.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        return np.log10(self.cumulative(theta))

    def cumulative(self, theta: np.ndarray) -> np.ndarray:
        r"""Return the cumulative halo abundance :math:`n(\geq M)`.

        Parameters
        ----------
        theta : ndarray of shape (n_samples, 4)
            Feedback parameters in :func:`theta_spec` order.

        Returns
        -------
        ndarray of shape (n_samples, N_BINS)
            :math:`n(\geq M)` in :math:`(h/\mathrm{Mpc})^3`, on
            :func:`mass_grid`. Multiply by a volume in
            :math:`(\mathrm{Mpc}/h)^3` for an expected halo count.
        """
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        raw = np.asarray(self._emulator.predict(theta, return_std=False))
        return raw * REFERENCE_VOLUME_SCALE

    def dndlgM(self, theta: np.ndarray) -> np.ndarray:
        r"""Return the differential abundance :math:`dn/d\log_{10}M`.

        A plain difference of the cumulative abundance divided by the bin
        width, which is what the reference does and is why the grid has to be
        uniform. Note the reference's differencing convention: bin ``i`` is
        :math:`n(M_i) - n(M_{i+1})`, and the **last** bin is :math:`n(M_{N-1})`
        on its own, because the difference there is taken against zero rather
        than against an edge the grid does not hold. Reproduced as it stands so
        the values are comparable point by point.

        The result is affected by the binning, so it is a coarser quantity than
        :meth:`cumulative`; the reference says as much in its own docstring.

        Parameters
        ----------
        theta : ndarray of shape (n_samples, 4)
            Feedback parameters in :func:`theta_spec` order.

        Returns
        -------
        ndarray of shape (n_samples, N_BINS)
            :math:`dn/d\log_{10}M` in :math:`(h/\mathrm{Mpc})^3` per dex.
        """
        cumulative = self.cumulative(theta)
        # Reversed so the difference runs against the next *lower* mass, as the
        # reference does it; the prepended zero is what makes the last bin
        # stand alone.
        return np.diff(cumulative[:, ::-1], axis=1, prepend=0.0)[:, ::-1] / bin_width()

    def __repr__(self) -> str:
        return f"BCMHFEmulator(fitted={self._emulator.is_fitted}, n_bins={N_BINS})"
