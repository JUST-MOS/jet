"""
Likelihood construction and the sampler interface.

**Design stage only** -- see the package docstring for what is decided and what
is still open. Every method here raises :class:`NotImplementedError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..emulator import Emulator
from ..estimator import Covariance, DataVector
from ..spec import ParameterSpec

__all__ = ["Likelihood", "Sampler"]


@dataclass
class Likelihood:
    """A Gaussian likelihood comparing a measured data vector to an emulator.

    The emulator uncertainty is held separately from the data covariance on
    purpose. They have different origins -- one is modelling error, the other
    is measurement noise -- and only the latter shrinks as more data are
    collected. Merging them into a single matrix understates the parameter
    errors.

    Attributes
    ----------
    measurement : DataVector
        The observed statistic.
    covariance : Covariance
        Measurement covariance, matching ``measurement.spec``. It must be the
        covariance of the values *as stored*: if the data vector is in log10,
        this is the covariance of the logged values.
    emulator : Emulator
        The surrogate used as the forward model. Its ``x_spec`` defines the
        sampled parameters.
    emulator_error : ndarray of shape (n_bins,), optional
        Additional variance from the surrogate, added to the diagonal of the
        data covariance. Typically taken from the emulator's own predictive
        spread on a validation set, which is why it is supplied here rather
        than read off the emulator at call time.

    Notes
    -----
    With both terms on the diagonal, the log-likelihood is

    .. math::

        -2 \\ln \\mathcal{L} = (d - \\mu(\\theta))^T
        \\left[ C_{\\rm data} + \\mathrm{diag}(\\sigma^2_{\\rm emu}) \\right]^{-1}
        (d - \\mu(\\theta))

    The block-diagonal approximation for cross-statistic correlations lives
    here too, and should be an explicit, named option rather than a silent
    default.
    """

    measurement: DataVector
    covariance: Covariance
    emulator: Emulator
    emulator_error: np.ndarray | None = None

    @property
    def parameter_spec(self) -> ParameterSpec:
        """Parameters being sampled -- the emulator's input specification."""
        return self.emulator.x_spec

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Evaluate the log-likelihood at a parameter point.

        Parameters
        ----------
        theta : ndarray of shape (n_params,)
            A single point, in ``parameter_spec`` order. Samplers that evaluate
            batches of points should loop, or a vectorised variant should be
            added once a backend actually needs it.

        Returns
        -------
        float
            Log-likelihood. Must not raise on an out-of-range ``theta`` --
            samplers explore the prior box and will probe its edges -- but
            should return ``-inf`` when the emulator would be extrapolating
            outside the region it was trained for.
        """
        raise NotImplementedError("Likelihood.log_likelihood is not implemented yet")

    def log_prior(self, theta: np.ndarray) -> float:
        """Evaluate the log-prior.

        The default prior is uniform over the bounds declared on the
        emulator's ``ParameterSpec``. That is the reason bounds are documented
        as a *validity range* rather than a normalisation convenience: they do
        double duty as the prior box.
        """
        raise NotImplementedError("Likelihood.log_prior is not implemented yet")

    def log_posterior(self, theta: np.ndarray) -> float:
        """Return ``log_prior + log_likelihood``."""
        raise NotImplementedError("Likelihood.log_posterior is not implemented yet")


class Sampler(ABC):
    """Base class for posterior-sampling backends.

    A subclass wraps one sampling library and knows nothing about cosmology.
    The likelihood is the only interface it sees.

    Subclasses are intended to be registered by name, so a run configuration can
    say ``sampler = "emcee"`` without importing it.

    Examples
    --------
    ::

        class EmceeSampler(Sampler):
            name = "emcee"

            def run(self, likelihood, n_steps, **options):   # doctest: +SKIP
                ...
    """

    #: Registry key.
    name: ClassVar[str] = ""

    @abstractmethod
    def run(
        self,
        likelihood: Likelihood,
        *,
        n_steps: int,
        n_walkers: int | None = None,
        initial: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        **options: Any,
    ) -> Mapping[str, Any]:
        """Sample the posterior.

        Parameters
        ----------
        likelihood : Likelihood
            The posterior to sample.
        n_steps : int
            Number of sampling steps. The meaning is backend-specific (MCMC
            steps, or nested-sampling likelihood evaluations), so it is a
            required argument rather than a defaulted one.
        n_walkers : int, optional
            Ensemble size, for samplers that have one.
        initial : ndarray of shape (n_walkers, n_params), optional
            Starting positions. When omitted, the sampler draws them from the
            prior, which is why the prior box must be finite.
        rng : numpy.random.Generator, optional
            Random source, so a run can be reproduced.
        **options
            Backend-specific settings, e.g. ``nested`` for PocoMC::

                Sampler.run(likelihood, n_steps=5000, nested=True)

        Returns
        -------
        mapping
            Backend-agnostic results: ``samples`` of shape
            ``(n_samples, n_params)``, ``log_posterior``, ``weights``,
            ``acceptance_fraction``, plus whatever else the backend offers,
            under its own keys.
        """
