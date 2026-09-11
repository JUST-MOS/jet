"""
Measurement of clustering statistics from galaxy and halo samples.

**Design stage only.** Interfaces and decisions are recorded; no implementation
is provided yet.

Pipeline position
-----------------
``galaxy catalog`` -> ``Statistic.measure`` -> ``DataVector`` -> emulator or inference

Decided
-------
One statistic per emulator
    ``w_p``, ``DeltaSigma``, ``n_bar_g`` and the rest are measured and emulated
    independently. Each :class:`~jet.estimator.statistic.Statistic` publishes its
    own :class:`~jet.spec.DataVectorSpec`, and that spec is what an
    :class:`~jet.emulator.Emulator` is bound to.

    The consequence has to be stated plainly: **the cross-covariance between
    different statistics is not produced by this framework.** Joint constraints
    from ``w_p`` and ``DeltaSigma`` come from their off-diagonal covariance
    block, and with one emulator per statistic nothing supplies it. Filling that
    gap -- analytically, from mock realisations, or by accepting a block-
    diagonal approximation -- is an open task, recorded in
    :mod:`jet.inference`.

Emulator uncertainty is not measurement uncertainty
    A surrogate's predictive spread and a survey's statistical error are
    different things with different origins. Folding the former into the data
    covariance would understate parameter errors, because it is a *modelling*
    error that repeats across the data while a covariance describes noise that
    does not. They are kept as separate terms; see
    :class:`~jet.inference.likelihood.Likelihood`.

Two implementations, one interface
    The core implementation uses NumPy only, so ``pip install jet`` can measure
    statistics out of the box. A fast implementation built on ``Corrfunc`` sits
    behind ``jet[est]``; the difference matters because pair counting is
    O(N^2) in NumPy against O(N log N) with a cell list, which is the difference
    between seconds and hours at survey-scale galaxy counts. Both must return
    identical data vectors for identical inputs, and a test should assert that.

Open questions
--------------
* Which statistics are in the first release? ``n_bar_g`` (one number, the
  easiest end-to-end target) and ``w_p`` (the 15-bin case the reference
  emulator used) are the natural starting pair.
* Redshift-space or real-space, and at how many redshifts? Each additional
  redshift multiplies the training-set size and the emulator's output width.
* Covariance estimation: how many mock realisations are available, and is the
  Hartlap correction (needed whenever the covariance is inverted and the number
  of realisations is not far above the number of bins) enough?
"""

from __future__ import annotations

from .statistic import Covariance, DataVector, Statistic

__all__ = ["Statistic", "DataVector", "Covariance"]
