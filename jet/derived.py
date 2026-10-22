"""
Derived parameters: axes that are re-expressions of other axes.

A :class:`~jet.spec.ParameterSpec` describes the axes an emulator is trained on.
Sometimes the same physical model is more usefully addressed along a different
axis: an emulator trained on the primordial amplitude :math:`A_s` is more
conveniently *driven* by :math:`\\sigma_8`. That is not a new degree of freedom
-- the two are invertible functions of each other, so both parameterisations
describe the same model.

This module holds the machinery for that substitution. A :class:`Conversion`
names one derived parameter and the axis it stands in for, and can move values
in *both* directions:

``to_derived``
    given a point in the spec's own (base) frame, evaluate the derived
    parameter;
``to_base``
    given desired derived values, return the base-frame point that produces
    them -- for a nonlinear relation this is a root find, not an algebra step.

The conversion is deliberately *not* an axis of its own. It never adds a column,
so ``X`` keeps its ``(n_samples, spec.dim)`` shape and every downstream
assumption about it survives. See :meth:`jet.spec.ParameterSpec.frame`.

Conversions are registered by ``kind``, mirroring the backend and transform
registries, so a bundle can record which conversion it used by name rather than
pickling a callable.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .emulator.emulator import Emulator
    from .emulator.pkmm import PkmmProvider

__all__ = [
    "Conversion",
    "LinearCombination",
    "Sigma8FromAs",
    "CONVERSIONS",
    "register_conversion",
    "conversion_from_state",
]

#: Maps parameter names to their column index in the frame being converted.
Columns = Mapping[str, int]


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """Return a short, stable hash of a JSON-serialisable mapping."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class Conversion(ABC):
    """A bijection between one axis of a spec and a derived parameter.

    Subclasses declare a registry ``kind`` and implement :meth:`to_derived`,
    :meth:`to_base`, :meth:`state` and :meth:`from_state`. Both directions
    receive the *whole* base-frame row, not just the axis being substituted,
    because a derived parameter generally depends on the other axes too --
    :math:`\\sigma_8` is not a function of :math:`A_s` alone.

    Parameters
    ----------
    name : str
        Name of the derived parameter, e.g. ``"sigma8"``.
    base : str
        Name of the spec axis this parameter stands in for, e.g. ``"As"``.
    requires : sequence of str
        Names of the further axes the conversion reads. Must not include
        ``base`` or ``name``.
    """

    #: Registry key written into a bundle manifest.
    kind: ClassVar[str] = ""

    def __init__(self, name: str, base: str, requires: Sequence[str] = ()) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"conversion name must be a non-empty string, got {name!r}")
        if not isinstance(base, str) or not base:
            raise ValueError(f"conversion base must be a non-empty string, got {base!r}")

        requires = tuple(requires)
        if base == name:
            raise ValueError(f"a conversion cannot map {name!r} onto itself")
        if base in requires or name in requires:
            raise ValueError(
                f"'requires' must list only the *other* axes; got {list(requires)} "
                f"for {name!r} <- {base!r}"
            )
        if len(set(requires)) != len(requires):
            raise ValueError(f"duplicate names in requires: {list(requires)}")

        self.name = name
        self.base = base
        self.requires = requires

    @property
    def names(self) -> tuple[str, ...]:
        """Every spec axis this conversion reads or writes."""
        return (self.base, *self.requires)

    def missing(self, columns: Columns) -> list[str]:
        """Return the axes this conversion needs that ``columns`` does not offer."""
        return [n for n in self.names if n not in columns]

    def _check(self, columns: Columns) -> None:
        """Raise if the spec this conversion is being applied to lacks an axis."""
        missing = self.missing(columns)
        if missing:
            raise KeyError(
                f"conversion {self.name!r} (kind {self.kind!r}) needs parameters "
                f"{missing}, which are not in {sorted(columns)}"
            )

    @abstractmethod
    def to_derived(self, X: np.ndarray, columns: Columns) -> np.ndarray:
        """Evaluate the derived parameter at base-frame points.

        Parameters
        ----------
        X : ndarray of shape (n_samples, spec.dim)
            Points in the spec's own (base) frame.
        columns : mapping of str to int
            Parameter name to column index for the base frame.

        Returns
        -------
        ndarray of shape (n_samples,)
        """

    @abstractmethod
    def to_base(self, X: np.ndarray, values: np.ndarray, columns: Columns) -> np.ndarray:
        """Substitute derived values back into a base-frame array.

        The base column of ``X`` is ignored; every other column is carried over
        unchanged.

        Parameters
        ----------
        X : ndarray of shape (n_samples, spec.dim)
            Base-frame points. Only the columns named in :attr:`requires` are
            read.
        values : ndarray of shape (n_samples,)
            Desired value of the derived parameter for each row.
        columns : mapping of str to int
            Parameter name to column index for the base frame.

        Returns
        -------
        ndarray of shape (n_samples, spec.dim)
            A copy of ``X`` whose base column has been solved for.
        """

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Return the conversion's settings as JSON-serialisable values."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: Mapping[str, Any]) -> Conversion:
        """Rebuild a conversion from :meth:`state` output plus ``kind``/``name``/``base``."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description, used for hashing and bundles."""
        return {
            "kind": self.kind,
            "name": self.name,
            "base": self.base,
            "requires": list(self.requires),
            "state": self.state(),
        }

    def hash(self) -> str:
        """Return a short content hash, binding a model to its derived frame."""
        return _canonical_hash(self.to_dict())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r} <- {self.base!r})"


#: Registry mapping conversion ``kind`` to class.
CONVERSIONS: dict[str, type[Conversion]] = {}


def register_conversion(cls: type[Conversion]) -> type[Conversion]:
    """Register a conversion class under its ``kind``."""
    if not cls.kind:
        raise ValueError(f"{cls.__name__} must define a non-empty 'kind'")
    CONVERSIONS[cls.kind] = cls
    return cls


def conversion_from_state(state: Mapping[str, Any]) -> Conversion:
    """Rebuild a conversion from a manifest entry.

    Parameters
    ----------
    state : mapping
        A :meth:`Conversion.to_dict` mapping.

    Returns
    -------
    Conversion
        The rebuilt, configured conversion.
    """
    kind = state.get("kind")
    if kind not in CONVERSIONS:
        raise KeyError(
            f"unknown conversion kind {kind!r}; registered kinds are {sorted(CONVERSIONS)}"
        )
    return CONVERSIONS[kind].from_state(state)


@register_conversion
class LinearCombination(Conversion):
    """A derived parameter that is a linear combination of other axes.

    Covers the affine bookkeeping relations that show up in a cosmological
    parameter space, such as :math:`h = H_0 / 100`.

    Parameters
    ----------
    name, base : str
        As on :class:`Conversion`.
    terms : mapping of str to float
        Coefficient of each axis in the combination, including ``base``. The
        coefficient on ``base`` must be non-zero, since it is what
        :meth:`to_base` solves for.

    Examples
    --------
    >>> import numpy as np
    >>> conv = LinearCombination("h", "H0", {"H0": 0.01})
    >>> cols = {"H0": 0}
    >>> X = np.array([[67.66]])
    >>> conv.to_derived(X, cols)
    array([0.6766])
    >>> conv.to_base(X, np.array([0.7]), cols)
    array([[70.]])
    """

    kind = "linear"

    def __init__(self, name: str, base: str, terms: Mapping[str, float]) -> None:
        terms = {str(k): float(v) for k, v in terms.items()}
        if base not in terms:
            raise ValueError(f"terms must include the base axis {base!r}; got {sorted(terms)}")
        if terms[base] == 0.0:
            raise ValueError(f"the coefficient on the base axis {base!r} must be non-zero")

        super().__init__(name, base, tuple(n for n in terms if n != base))
        self.terms = terms

    def to_derived(self, X: np.ndarray, columns: Columns) -> np.ndarray:
        self._check(columns)
        return sum(
            coefficient * np.asarray(X, dtype=float)[:, columns[name]]
            for name, coefficient in self.terms.items()
        )

    def to_base(self, X: np.ndarray, values: np.ndarray, columns: Columns) -> np.ndarray:
        self._check(columns)
        X = np.asarray(X, dtype=float)
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != X.shape[0]:
            raise ValueError(f"got {values.shape[0]} derived values for {X.shape[0]} rows of X")

        coefficient = self.terms[self.base]
        rest = sum(c * X[:, columns[name]] for name, c in self.terms.items() if name != self.base)
        out = X.copy()
        out[:, columns[self.base]] = (values - rest) / coefficient
        return out

    def state(self) -> dict[str, Any]:
        return {"terms": dict(self.terms)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LinearCombination:
        return cls(state["name"], state["base"], state["state"]["terms"])


@register_conversion
class Sigma8FromAs(Conversion):
    """:math:`\\sigma_8` standing in for the primordial amplitude :math:`A_s`.

    The relation is not analytic, so it is evaluated through the same machinery
    ``csstemu`` uses: a linear-theory **total-matter** :math:`P(k)` emulator,
    integrated against a top-hat window at :math:`R = 8\\,\\mathrm{Mpc}/h`. The
    total spectrum is composed from :math:`P_{cb}` and the
    :math:`P_{nn}/P_{cb}` ratio, so massive neutrinos are included in
    :math:`\\sigma_8`. Two properties of that setup make the conversion cheap:

    * :math:`P(k) \\propto A_s` at fixed cosmology, hence
      :math:`\\sigma_8 \\propto \\sqrt{A_s}`. One evaluation of the emulator
      gives :math:`\\sigma_8` at *any* :math:`A_s`, so the inversion is a fixed
      point that converges in a couple of steps::

          A_s <- A_s * (sigma8_target / sigma8(A_s))**2

    * the :math:`P(k)` emulator is only ever asked at :math:`z = 0`, so the
      redshift axis of its data vector is sliced off immediately.

    Parameters
    ----------
    name, base : str, optional
        Defaults to ``"sigma8"`` and ``"As"``.
    pklin : Emulator, str or None, optional
        The :math:`P_{cb}(k)` emulator, the first factor of the total-matter
        composition. ``None`` (the default) uses the copy bundled with ``jet``
        under ``jet/data``; a string is resolved like any other weight file,
        through :func:`jet.emulator.resolve_weights_path`. A live
        :class:`Emulator` works for in-memory use but cannot be written to a
        bundle -- save it and pass the path instead.
    ratio_emulator : Emulator, str or None, optional
        The :math:`P_{nn}/P_{cb}` ratio emulator. ``None`` uses the bundled
        copy. Together with ``pklin`` it defines the total-matter spectrum used
        for :math:`\\sigma_8`.
    R : float, optional
        Smoothing scale in :math:`\\mathrm{Mpc}/h`.
    z : float, optional
        Redshift at which :math:`\\sigma_8` is defined. Only ``0.0`` is
        supported by the bundled emulator's window grid.
    tolerance : float, optional
        Absolute tolerance on :math:`\\sigma_8` for the fixed-point iteration.
        The default is far tighter than the reference implementation's ``1e-4``.
        That costs one or two extra iterations -- the step is Newton-like in
        :math:`A_s`, so the error squares each time -- and buys a round trip
        through the frame that is faithful to about ``1e-12`` instead of
        ``1e-4``, which matters when the conversion sits inside an optimisation
        loop and its noise would be differentiated.
    initial_as : float, optional
        Starting guess for the inversion. The default is the value ``csstemu``
        starts from; because the fixed point is nearly exact in one step, this
        only affects how many iterations are needed.
    max_iterations : int, optional
        Safety limit on the inversion.
    fallback_to_camb : bool, optional
        Policy for inputs outside the P(k) emulator's parameter bounds. The
        default raises :class:`ValueError`. When true, :math:`\\sigma_8` is
        obtained from a linear CAMB CDM+baryon spectrum instead, and a
        :class:`UserWarning` names the offending parameters.
    """

    kind = "sigma8_as"

    #: Cosmological axes the bundled emulator is trained on, in its own order.
    PKLIN_PARAMETERS = ("Omegab", "Omegam", "H0", "ns", "As", "w0", "wa", "mnu")

    def __init__(
        self,
        name: str = "sigma8",
        base: str = "As",
        pklin: Emulator | str | None = None,
        ratio_emulator: Emulator | str | None = None,
        R: float = 8.0,
        z: float = 0.0,
        tolerance: float = 1e-8,
        initial_as: float = 2.105e-9,
        max_iterations: int = 100,
        fallback_to_camb: bool = False,
    ) -> None:
        super().__init__(
            name,
            base,
            # PKLIN_PARAMETERS minus the amplitude, which the inversion solves
            # for rather than reads.
            requires=("Omegab", "Omegam", "H0", "ns", "w0", "wa", "mnu"),
        )
        if R <= 0.0:
            raise ValueError(f"R must be positive, got {R}")
        if z != 0.0:
            raise NotImplementedError(
                f"the bundled P(k) emulator is only wired up at z = 0, got z = {z}"
            )
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be positive, got {tolerance}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")

        self.R = float(R)
        self.z = float(z)
        self.tolerance = float(tolerance)
        self.initial_as = float(initial_as)
        self.max_iterations = int(max_iterations)
        self.fallback_to_camb = bool(fallback_to_camb)

        self._pklin = pklin
        self._ratio = ratio_emulator
        self._emulator: PkmmProvider | None = None

    # ------------------------------------------------------------------
    # The P(k) emulator
    # ------------------------------------------------------------------
    @property
    def emulator(self) -> PkmmProvider:
        """The total-matter :math:`P(k)` provider, loaded on first use."""
        if self._emulator is None:
            from .emulator.emulator import Emulator as _Emulator
            from .emulator.pklin import load_pklin_emulator
            from .emulator.pkmm import PkmmProvider, load_ratio_emulator

            if self._pklin is None:
                pcb = load_pklin_emulator()
            elif isinstance(self._pklin, str):
                pcb = _Emulator.load(self._pklin)
            else:
                pcb = self._pklin

            if self._ratio is None:
                ratio = load_ratio_emulator()
            elif isinstance(self._ratio, str):
                ratio = _Emulator.load(self._ratio)
            else:
                ratio = self._ratio

            self._emulator = PkmmProvider(pcb, ratio)
        return self._emulator

    def _theta(self, X: np.ndarray, columns: Columns, As: np.ndarray) -> np.ndarray:
        """Assemble a ``(n_samples, 8)`` block in the P(k) emulator's own order."""
        X = np.asarray(X, dtype=float)
        values = {
            "Omegab": X[:, columns["Omegab"]],
            "Omegam": X[:, columns["Omegam"]],
            "H0": X[:, columns["H0"]],
            "ns": X[:, columns["ns"]],
            "As": As,
            "w0": X[:, columns["w0"]],
            "wa": X[:, columns["wa"]],
            "mnu": X[:, columns["mnu"]],
        }
        return np.column_stack([values[p] for p in self.PKLIN_PARAMETERS])

    def _bounds_mask(self, theta: np.ndarray) -> np.ndarray:
        """Return rows of ``theta`` outside the P(k) emulator's bounds."""
        bounds = self.emulator.x_spec.bounds_array()
        return np.any((theta < bounds[:, 0]) | (theta > bounds[:, 1]), axis=1)

    def _warn_camb(self, theta: np.ndarray, mask: np.ndarray, method: str) -> None:
        """Warn that CAMB is supplying sigma8 for the masked rows."""
        violations = self.emulator.x_spec.check_within_bounds(theta[mask], slack=0.0)
        original_rows = np.flatnonzero(mask)
        details = []
        for param_name, row, value in violations:
            spec = self.emulator.x_spec
            low, high = spec.params[spec.index(param_name)].bounds  # type: ignore[misc]
            details.append(
                f"row {int(original_rows[row])}: {param_name}={value:.6g} "
                f"outside [{low:.6g}, {high:.6g}]"
            )
        warnings.warn(
            f"{method}: {int(mask.sum())} sample(s) outside the P(k) emulator's "
            "validity range; using CAMB sigma8:\n  " + "\n  ".join(details),
            UserWarning,
            stacklevel=3,
        )

    def _sigma8(
        self,
        theta: np.ndarray,
        *,
        allow_out_of_bounds: bool,
        warn: bool,
    ) -> np.ndarray:
        """Evaluate :math:`\\sigma_8` for rows of emulator-frame parameters."""
        from ._boltzmann import camb_sigma8
        from .emulator.pklin import sigma8_of_pk

        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        outside = self._bounds_mask(theta)
        if np.any(outside):
            if not allow_out_of_bounds and not self.fallback_to_camb:
                self.emulator.x_spec.require_within_bounds(theta, name="theta")
            if warn and self.fallback_to_camb:
                self._warn_camb(theta, outside, "Sigma8FromAs")

        out = np.empty(theta.shape[0], dtype=float)
        inside = ~outside
        if np.any(inside):
            Pk = self.emulator.predict(theta[inside], return_std=False)
            out[inside] = sigma8_of_pk(Pk, R=self.R, z=self.z)
        if np.any(outside):
            if self.fallback_to_camb:
                for index in np.flatnonzero(outside):
                    out[index] = float(camb_sigma8(theta[index], R=self.R, z=self.z)[0])
            else:
                # This path is only reached for intermediate inversion steps;
                # the caller checks the final solution separately.
                Pk = self.emulator.predict(theta[outside], return_std=False)
                out[outside] = sigma8_of_pk(Pk, R=self.R, z=self.z)
        return out

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------
    def to_derived(self, X: np.ndarray, columns: Columns) -> np.ndarray:
        self._check(columns)
        X = np.asarray(X, dtype=float)
        theta = self._theta(X, columns, X[:, columns[self.base]])
        return self._sigma8(theta, allow_out_of_bounds=False, warn=True)

    def to_base(self, X: np.ndarray, values: np.ndarray, columns: Columns) -> np.ndarray:
        self._check(columns)
        X = np.asarray(X, dtype=float)
        target = np.asarray(values, dtype=float).reshape(-1)
        if target.shape[0] != X.shape[0]:
            raise ValueError(f"got {target.shape[0]} derived values for {X.shape[0]} rows of X")
        if np.any(target <= 0.0):
            raise ValueError("sigma8 must be positive")

        # The base column of ``X`` holds the *derived* values in this
        # direction, not an As guess. Use the configured initial amplitude.
        initial_theta = self._theta(X, columns, np.full(X.shape[0], self.initial_as))
        initial_outside = self._bounds_mask(initial_theta)
        if np.any(initial_outside):
            if self.fallback_to_camb:
                self._warn_camb(initial_theta, initial_outside, "Sigma8FromAs.to_base")
            else:
                self.emulator.x_spec.require_within_bounds(initial_theta, name="theta")

        As = np.full(X.shape[0], self.initial_as)
        converged = np.zeros(X.shape[0], dtype=bool)
        exhausted = True

        for _ in range(self.max_iterations):
            # Intermediate As values may leave the box; only the input and the
            # final solution are required to be trustworthy.
            guess = self._sigma8(
                self._theta(X, columns, As),
                allow_out_of_bounds=True,
                warn=False,
            )
            if np.any(guess <= 0.0):
                raise RuntimeError(
                    "the P(k) emulator returned a non-positive sigma8; the input "
                    "cosmology is outside the region it was trained on"
                )
            # sigma8 scales as sqrt(As) at fixed cosmology, so this is a Newton
            # step in As for every row that has not yet converged.
            As = np.where(converged, As, As * (target / guess) ** 2)
            converged = np.abs(guess - target) < self.tolerance
            if converged.all():
                exhausted = False
                break

        final_theta = self._theta(X, columns, As)
        final_outside = self._bounds_mask(final_theta)
        if np.any(final_outside):
            if self.fallback_to_camb:
                if not np.any(initial_outside):
                    self._warn_camb(final_theta, final_outside, "Sigma8FromAs.to_base")
            else:
                self.emulator.x_spec.require_within_bounds(final_theta, name="theta")

        if exhausted:
            # The step squares the error and is exact for a power law, so failing
            # to converge means the target is out of reach -- a sigma8 the
            # emulator cannot produce at any A_s -- rather than slow convergence.
            # Returning the best guess quietly would turn that into a plausible
            # looking amplitude, so say so.
            warnings.warn(
                f"{int((~converged).sum())} of {X.shape[0]} row(s) did not reach "
                f"sigma8 to within {self.tolerance:g} in {self.max_iterations} "
                "iterations; the requested value is probably outside the range this "
                "emulator can produce. The returned A_s is the closest guess.",
                RuntimeWarning,
                stacklevel=2,
            )

        out = X.copy()
        out[:, columns[self.base]] = As
        return out

    def state(self) -> dict[str, Any]:
        if self._pklin is None:
            pklin: Any = "builtin"
        elif isinstance(self._pklin, str):
            pklin = self._pklin
        else:
            raise TypeError(
                "this conversion holds a live Emulator, which cannot be written to a "
                "bundle; call Emulator.save() on it first and construct the conversion "
                "with the resulting path (or leave 'pklin' as None to use the bundled one)"
            )
        if self._ratio is None:
            ratio: Any = "builtin"
        elif isinstance(self._ratio, str):
            ratio = self._ratio
        else:
            raise TypeError(
                "this conversion holds a live ratio Emulator, which cannot be written "
                "to a bundle; save it and pass the path, or leave 'ratio_emulator' as "
                "None to use the bundled one"
            )
        return {
            "pklin": pklin,
            "ratio": ratio,
            "R": self.R,
            "z": self.z,
            "tolerance": self.tolerance,
            "initial_as": self.initial_as,
            "max_iterations": self.max_iterations,
            "fallback_to_camb": self.fallback_to_camb,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Sigma8FromAs:
        settings = state["state"]
        pklin = settings["pklin"]
        ratio = settings.get("ratio", "builtin")
        return cls(
            name=state["name"],
            base=state["base"],
            pklin=None if pklin == "builtin" else pklin,
            ratio_emulator=None if ratio == "builtin" else ratio,
            R=settings["R"],
            z=settings["z"],
            tolerance=settings["tolerance"],
            initial_as=settings["initial_as"],
            max_iterations=settings["max_iterations"],
            fallback_to_camb=settings.get("fallback_to_camb", False),
        )
