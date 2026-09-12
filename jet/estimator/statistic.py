"""
Statistics, measured data vectors, and their covariance.

**Design stage only** -- see the package docstring for what is decided and what
is still open. Every method here raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..catalog import GalaxyCatalog, HaloCatalog
from ..spec import DataVectorSpec

__all__ = ["DataVector", "Statistic", "Covariance"]


@dataclass
class DataVector:
    """A measured statistic, together with the specification that defines it.

    The pair travels as a unit because a bare array of 15 numbers is not
    interpretable: it matters which 15, in what order, and whether they are
    logarithms. Every consumer -- the emulator, the likelihood, a plot -- needs
    the spec to make sense of the values, so they are never separated.

    Attributes
    ----------
    values : ndarray of shape (n_bins,)
        The measured statistic.
    spec : DataVectorSpec
        What the entries mean, including the binning and the log flag.
    covariance : Covariance, optional
        Uncertainty on this particular measurement, when known.
    metadata : mapping, optional
        Provenance: redshift, sample selection, estimator settings.
    """

    values: np.ndarray
    spec: DataVectorSpec
    covariance: Covariance | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(np.asarray(self.values).size)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != (self.spec.dim,):
            raise ValueError(
                f"expected {self.spec.dim} values to match {self.spec!r}, "
                f"got shape {self.values.shape}"
            )


class Statistic(ABC):
    """Base class for a measurable clustering statistic.

    A subclass declares the data vector it produces and knows how to measure it
    from a catalog. The declaration is a class-level
    :class:`~jet.spec.DataVectorSpec`, so an emulator can be configured for a
    statistic without instantiating it.

    Subclasses are intended to be registered by name, for the same reason
    :class:`~jet.catalog.HODModel` subclasses are: a configuration file or a
    bundle manifest should be able to name a statistic without importing it.

    Examples
    --------
    ::

        class GalaxyClustering(Statistic):
            name = "wp"
            data_vector_spec = DataVectorSpec(
                name="wp", n_bins=15, log=True,
                bin_edges=(0.1, 0.2, ..., 50.0),
            )

            def measure(self, catalog, **options):   # doctest: +SKIP
                ...
    """

    #: Registry key.
    name: ClassVar[str] = ""

    #: The data vector this statistic produces. An emulator bound to a
    #: different spec will refuse to load a model trained on this one.
    data_vector_spec: ClassVar[DataVectorSpec]

    @abstractmethod
    def measure(
        self,
        catalog: GalaxyCatalog | HaloCatalog,
        *,
        rng: np.random.Generator | None = None,
        **options: Any,
    ) -> DataVector:
        """Measure the statistic from a catalog.

        Parameters
        ----------
        catalog : GalaxyCatalog or HaloCatalog
            The sample to measure.
        rng : numpy.random.Generator, optional
            Random source, for estimators that resample (bootstrap errors,
            jackknife regions). Passed in rather than created internally so
            that a measurement is reproducible.
        **options
            Estimator-specific settings: binning overrides, the random catalog
            for Landy-Szalay, projection depth, and so on.

        Returns
        -------
        DataVector
            The measured values, their spec, and any covariance the estimator
            can produce.
        """

    def random_catalog(self, catalog: GalaxyCatalog, *, n_randoms: int) -> GalaxyCatalog:
        """Generate an unclustered catalog with the same selection as ``catalog``.

        Needed by Landy-Szalay-style estimators, which subtract the
        uncorrelated baseline. The default implementation is not written yet
        because "the same selection" is survey-specific: a box is uniform, a
        lightcone is not.
        """
        raise NotImplementedError("random_catalog is not implemented yet")


@dataclass
class Covariance:
    """Covariance matrix of a data vector, with its provenance.

    Attributes
    ----------
    matrix : ndarray of shape (n_bins, n_bins)
        Covariance, in the same units as the data vector. When the data vector
        is stored as log10, this must be the covariance *of the logged values*,
        not of the linear ones -- the likelihood has no way to tell.
    spec : DataVectorSpec
        The data vector this describes. Carried so that a covariance can be
        checked against the vector it will be applied to.
    n_realisations : int or None
        Number of mock realisations the sample covariance was estimated from,
        or ``None`` for an analytic covariance.
    method : str
        How it was obtained: ``"mock"``, ``"analytic"``, ``"external"``.

    Notes
    -----
    When a covariance is estimated from ``n_realisations`` mocks and then
    inverted, the inverse is biased unless the Hartlap correction
    ``(n - p - 2) / (n - 1)`` is applied for ``p`` bins. Applying it is the
    caller's job, but the realisation count is carried here so a likelihood can
    decide whether it is needed and whether it is even possible (with
    ``n <= p + 2`` the covariance is singular and no correction helps).
    """

    matrix: np.ndarray
    spec: DataVectorSpec
    n_realisations: int | None = None
    method: str = "mock"

    @classmethod
    def from_mocks(
        cls, measurements: Sequence[np.ndarray] | np.ndarray, spec: DataVectorSpec
    ) -> Covariance:
        """Estimate a sample covariance from mock realisations.

        Parameters
        ----------
        measurements : array-like of shape (n_realisations, n_bins)
            One measured data vector per mock.
        spec : DataVectorSpec
            The spec all realisations share.

        Returns
        -------
        Covariance
            With ``method="mock"`` and ``n_realisations`` recorded.
        """
        raise NotImplementedError("Covariance.from_mocks is not implemented yet")

    def inverse(self) -> np.ndarray:
        """Return the inverse covariance, applying the Hartlap correction.

        Raises
        ------
        NotImplementedError
            Not written yet. It must refuse to guess: with
            ``n_realisations <= n_bins + 2`` the sample covariance is singular
            and silently returning a pseudo-inverse would produce a likelihood
            that is confidently wrong.
        """
        raise NotImplementedError("Covariance.inverse is not implemented yet")

    def copy(self) -> Covariance:
        """Return an independent copy."""
        return Covariance(
            matrix=self.matrix.copy(),
            spec=self.spec,
            n_realisations=self.n_realisations,
            method=self.method,
        )
