"""
Parameter inference from measured statistics, using an emulator as the model.

**Design stage only.** Interfaces and decisions are recorded; no implementation
is provided yet.

Pipeline position
-----------------
``DataVector`` + ``Covariance`` + ``Emulator`` -> ``Likelihood`` -> ``Sampler`` -> posterior

Decided
-------
Two error terms, kept apart
    The data covariance describes noise that differs between realisations; the
    emulator's uncertainty is a modelling error that is the same for every
    realisation. They are carried as separate terms in the likelihood rather
    than summed into one matrix, because summing them understates parameter
    errors -- the emulator error does not average down with more data.

    This is also why the emulator backends are required to report an honest
    uncertainty rather than an array of zeros: a zeros return would silently
    claim a perfect surrogate.

Backends are pluggable
    Sampling is delegated. ``emcee`` is a light, pure-Python dependency behind
    ``jet[inf]``; ``PocoMC`` is the nested-sampling alternative named in the
    project README and would go behind its own extra, since it pulls in JAX.
    Neither is imported at module import time.

Open questions
--------------
* **Cross-statistic covariance.** With one emulator per statistic (see
  :mod:`jet.estimator`), nothing provides the off-diagonal block between, say,
  ``w_p`` and ``DeltaSigma``. Joint constraints depend on it. Options not yet
  chosen: estimate it from mock realisations, predict it with a halo model, or
  accept a block-diagonal approximation and say so explicitly.
* **The data-vector space.** Whether to sample in the space the emulator
  predicts (often log10) or convert to linear space before comparing with data.
  The covariance has to match whichever is chosen; a mismatch here is a
  silent, and serious, error.
* **Statistics selection.** The README mentions a greedy-Fisher statistic
  selection step. It needs a way to evaluate the information content of a
  subset of bins, which in turn needs the covariance and the emulator's Jacobian
  with respect to the parameters. That Jacobian is cheap for a Gaussian process
  and cheap for a network via autograd, but neither path is implemented.
* **Emulator uncertainty propagation**: sampling with ``std`` as a fixed
  additional variance is an approximation. The rigorous version marginalises
  over the surrogate's posterior, which for the Gaussian-process backend is
  analytic but not yet wired up.
"""

from __future__ import annotations

from .inference import Likelihood, Sampler

__all__ = ["Likelihood", "Sampler"]
