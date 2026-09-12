"""
Declarative specifications shared by the ``jet`` frameworks.

Two kinds of specifications live here:

``ParameterSpec``
    The *input* side. A sample is a point in a parameter space whose axes are
    named and (optionally) bounded. The same object describes the emulator
    input, the sampled parameters of an HOD model, and the prior box of an
    inference run -- which is why it sits at the top level of the package
    rather than inside ``jet.emulator``.

``DataVectorSpec``
    The *output* side. A sample is a vector of measured statistics, laid out
    as a fixed number of bins along some coordinate (e.g. 15 radial bins of
    the projected correlation function).

Both objects are plain, JSON-serialisable descriptions: they carry no data and
no behaviour beyond construction, validation, and hashing. A specification is
the identity card of a trained model -- a bundle records the hashes of the
specs it was trained against, and loading verifies them, so that a model can
never be silently applied to a differently-shaped parameter space.

A ``ParameterSpec`` may additionally declare *derived parameters*
(:mod:`jet.derived`): alternative names for axes that already exist, related to
them by an invertible function. ``sigma8`` derived from ``As`` adds no column --
it lets the same spec be addressed along either axis, by converting at the
boundary, and it is reached through :meth:`ParameterSpec.frame`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from .derived import Conversion, conversion_from_state

__all__ = ["Param", "ParameterSpec", "DerivedFrame", "DataVectorSpec"]


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """Return a short, stable hash of a JSON-serialisable mapping.

    The payload is dumped with sorted keys and no whitespace so that the digest
    depends only on the content, not on dict insertion order or formatting.

    Parameters
    ----------
    payload : mapping
        JSON-serialisable description of a specification.

    Returns
    -------
    str
        First 12 hex characters of the SHA-256 digest.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class Param:
    """A single scalar parameter (one axis of a parameter space).

    Parameters
    ----------
    name : str
        Column name. Must be unique within a :class:`ParameterSpec`.
    bounds : tuple of float, optional
        Validity range ``(low, high)`` with ``low < high``. Bounds describe
        where the model is *trustworthy*, not merely where its training samples
        happened to land: they are used to normalise the input, to warn when a
        prediction point falls outside the trained region, and to declare prior
        boxes for inference. ``None`` means unbounded.
    block : str, optional
        Logical group the parameter belongs to, e.g. ``"cosmo"`` or ``"hod"``.
        Purely descriptive -- it lets a caller restrict a prediction point to a
        subset of axes (``spec.subset(block="cosmo")``) without naming each one.
    doc : str, optional
        Free-form note. A common use is pinning down a convention that the name
        alone does not fix, e.g. ``"A_s * 1e9"``.

    Notes
    -----
    Instances are immutable in practice (attributes are not re-assigned after
    construction) but are not enforced as frozen, so that they stay cheap to
    pickle and compare.
    """

    __slots__ = ("name", "bounds", "block", "doc")

    def __init__(
        self,
        name: str,
        bounds: tuple[float, float] | None = None,
        block: str = "",
        doc: str = "",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Param name must be a non-empty string, got {name!r}")

        if bounds is not None:
            try:
                low, high = float(bounds[0]), float(bounds[1])
            except (TypeError, IndexError, ValueError) as exc:
                raise ValueError(
                    f"Param {name!r}: bounds must be a (low, high) pair, got {bounds!r}"
                ) from exc
            if not low < high:
                raise ValueError(
                    f"Param {name!r}: bounds must satisfy low < high, got ({low}, {high})"
                )
            bounds = (low, high)

        self.name = name
        self.bounds = bounds
        self.block = str(block)
        self.doc = str(doc)

    @property
    def width(self) -> float | None:
        """Width of the validity range, or ``None`` when unbounded."""
        if self.bounds is None:
            return None
        return self.bounds[1] - self.bounds[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this parameter."""
        return {
            "name": self.name,
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "block": self.block,
            "doc": self.doc,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Param:
        """Rebuild a parameter from :meth:`to_dict` output."""
        bounds = data.get("bounds")
        return cls(
            name=data["name"],
            bounds=tuple(bounds) if bounds is not None else None,
            block=data.get("block", ""),
            doc=data.get("doc", ""),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Param):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"Param({self.name!r}, bounds={self.bounds}, block={self.block!r})"


class ParameterSpec:
    """An ordered collection of :class:`Param` objects.

    The order of the parameters defines the column order of every array that
    this spec describes: a sample ``X`` of shape ``(N, len(spec))`` has
    ``X[:, spec.index("ns")]`` equal to the ``ns`` column.

    Parameters
    ----------
    params : iterable of Param
        The parameters, in column order. Names must be unique.
    derived : iterable of Conversion, optional
        Derived parameters, i.e. alternative names for axes that are already
        declared. Each one names the axis it stands in for and adds no column,
        so ``dim`` is unaffected. See :mod:`jet.derived`.

    Examples
    --------
    >>> spec = ParameterSpec([
    ...     Param("Omegab", bounds=(0.04, 0.06), block="cosmo"),
    ...     Param("ns", bounds=(0.92, 1.00), block="cosmo"),
    ...     Param("logMcut", bounds=(12.0, 13.8), block="hod"),
    ... ])
    >>> spec.dim
    3
    >>> spec.index("logMcut")
    2
    >>> len(spec.subset(block="hod"))
    1
    """

    def __init__(
        self,
        params: Iterable[Param],
        derived: Iterable[Conversion] | None = None,
    ) -> None:
        params = tuple(params)
        if not params:
            raise ValueError("ParameterSpec must contain at least one Param")

        seen: dict[str, int] = {}
        for param in params:
            if not isinstance(param, Param):
                raise TypeError(f"expected Param, got {type(param).__name__}")
            if param.name in seen:
                raise ValueError(
                    f"duplicate parameter name {param.name!r} "
                    f"(positions {seen[param.name]} and {len(seen)})"
                )
            seen[param.name] = len(seen)

        self.params = params
        self._derived = _index_conversions(params, seen, derived)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self) -> Iterator[Param]:
        return iter(self.params)

    def __getitem__(self, index: int) -> Param:
        return self.params[index]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._names

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParameterSpec):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"ParameterSpec({list(self._names)!r})"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Number of parameters, i.e. the number of input columns."""
        return len(self.params)

    @property
    def derived(self) -> Mapping[str, Conversion]:
        """Derived parameters of this spec, keyed by name."""
        return dict(self._derived)

    @property
    def derived_names(self) -> tuple[str, ...]:
        """Names of the derived parameters, in declaration order."""
        return tuple(self._derived)

    @property
    def has_derived(self) -> bool:
        """Whether this spec declares any derived parameter."""
        return bool(self._derived)

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names in column order."""
        return self._names

    @property
    def blocks(self) -> tuple[str, ...]:
        """Block label of each parameter, in column order."""
        return tuple(p.block for p in self.params)

    @property
    def has_bounds(self) -> bool:
        """Whether every parameter declares a validity range."""
        return all(p.bounds is not None for p in self.params)

    # ------------------------------------------------------------------
    # Lookup and slicing
    # ------------------------------------------------------------------
    def index(self, name: str) -> int:
        """Return the column index of ``name``.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        int
            Column index.

        Raises
        ------
        KeyError
            If the name is not part of this spec. The message lists the known
            names, because a mismatch here is the usual symptom of feeding a
            model the wrong columns.
        """
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(
                f"unknown parameter {name!r}; known parameters are {list(self._names)}"
            ) from None

    def indices(self, names: Sequence[str]) -> list[int]:
        """Return the column indices of ``names``, in the order given."""
        return [self.index(name) for name in names]

    def subset(
        self,
        names: Sequence[str] | None = None,
        block: str | None = None,
    ) -> ParameterSpec:
        """Return a new spec restricted to a subset of the parameters.

        Exactly one of ``names`` or ``block`` must be given. The relative order
        of the surviving parameters is preserved, so a value array can be
        sliced with :meth:`indices` using the same selection.

        Parameters
        ----------
        names : sequence of str, optional
            Explicit parameter names to keep.
        block : str, optional
            Keep every parameter whose ``block`` equals this label.

        Returns
        -------
        ParameterSpec
            The restricted specification.

        Raises
        ------
        ValueError
            If neither or both selectors are given, or the selection is empty.
        """
        if (names is None) == (block is None):
            raise ValueError("give exactly one of 'names' or 'block'")

        if block is not None:
            kept = tuple(p for p in self.params if p.block == block)
        else:
            requested = set(names or ())
            unknown = requested - set(self._names)
            if unknown:
                raise KeyError(
                    f"unknown parameter(s) {sorted(unknown)}; "
                    f"known parameters are {list(self._names)}"
                )
            kept = tuple(p for p in self.params if p.name in requested)

        if not kept:
            raise ValueError(f"subset selection produced an empty spec ({names=}, {block=})")

        # Keep the derived parameters that survive the selection intact. A
        # conversion whose base or inputs were dropped cannot be evaluated any
        # more, and carrying it along would turn a call to `frame` into an
        # obscure KeyError at prediction time.
        surviving = {p.name for p in kept}
        derived = [c for c in self._derived.values() if set(c.names) <= surviving]
        return ParameterSpec(kept, derived=derived)

    def frame(self, name: str) -> DerivedFrame:
        """Return the view of this spec addressed along a derived parameter.

        Parameters
        ----------
        name : str
            Name of a derived parameter, e.g. ``"sigma8"``.

        Returns
        -------
        DerivedFrame
            An object with the same ``dim`` whose ``names`` differ in one
            position, and which converts value arrays between the two frames.

        Raises
        ------
        KeyError
            If ``name`` is not a derived parameter of this spec. The message
            lists the ones that are.
        """
        if name not in self._derived:
            raise KeyError(
                f"{name!r} is not a derived parameter of this spec; "
                f"derived parameters are {list(self._derived)}"
            )
        return DerivedFrame(self, name)

    def bounds_array(self) -> np.ndarray:
        """Return bounds as an array of shape ``(dim, 2)``.

        Raises
        ------
        ValueError
            If any parameter is unbounded.
        """
        if not self.has_bounds:
            missing = [p.name for p in self.params if p.bounds is None]
            raise ValueError(f"parameters without bounds: {missing}")
        return np.asarray([p.bounds for p in self.params], dtype=float)

    def check_within_bounds(
        self, X: np.ndarray, slack: float = 0.0
    ) -> list[tuple[str, int, float]]:
        """Find rows of ``X`` that fall outside the declared validity ranges.

        Parameters
        ----------
        X : ndarray of shape (N, dim)
            Sample points, columns in spec order.
        slack : float, optional
            Tolerated fractional excursion beyond a bound. A point at
            ``low - slack * (high - low)`` is still accepted.

        Returns
        -------
        list of tuple
            One ``(name, row_index, value)`` entry per parameter that is out of
            range. Empty when every point is inside (this is the common case,
            so callers can use the result directly as a boolean).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"expected X with shape (N, {self.dim}), got {X.shape}")
        if not self.has_bounds:
            return []

        violations: list[tuple[str, int, float]] = []
        for column, param in enumerate(self.params):
            low, high = param.bounds  # type: ignore[misc]
            margin = slack * (high - low)
            values = X[:, column]
            for row in np.nonzero((values < low - margin) | (values > high + margin))[0]:
                violations.append((param.name, int(row), float(values[row])))
        return violations

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this spec.

        Carries the derived declarations as well as the axes, so that a model
        loaded from a bundle can still say which conversions its spec offers.
        Those declarations are deliberately *not* what :meth:`hash` covers; see
        there.
        """
        return {
            "params": [p.to_dict() for p in self.params],
            "derived": [c.to_dict() for c in self._derived.values()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParameterSpec:
        """Rebuild a spec from :meth:`to_dict` output."""
        derived = [conversion_from_state(entry) for entry in data.get("derived", [])]
        return cls((Param.from_dict(p) for p in data["params"]), derived=derived)

    def hash(self) -> str:
        """Return a short content hash of the axes, used to bind a model to its spec.

        Derived declarations carry no weight here even though :meth:`to_dict`
        stores them. A model reads the axes of its spec and nothing else -- a
        conversion is applied by the caller, before the model sees the array --
        so two specs differing only in which conversions they declare present
        the same input space, and a bundle is bound to the axes alone.

        The practical consequence is that declaring a derived parameter leaves
        existing bundles valid, which is what makes doing so cheap. ``__eq__``
        still compares the whole content, so two specs can be unequal and hash
        alike; that direction of the relationship is an ordinary collision, not
        a contradiction of the equality invariant.
        """
        return _canonical_hash({"params": [p.to_dict() for p in self.params]})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @property
    def _names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def _index(self) -> dict[str, int]:
        return {p.name: i for i, p in enumerate(self.params)}


def _index_conversions(
    params: Sequence[Param],
    seen: Mapping[str, int],
    derived: Iterable[Conversion] | None,
) -> dict[str, Conversion]:
    """Validate derived declarations and key them by name, in order."""
    if not derived:
        return {}

    indexed: dict[str, Conversion] = {}
    for conversion in derived:
        if not isinstance(conversion, Conversion):
            raise TypeError(
                f"expected a Conversion, got {type(conversion).__name__}; "
                "see jet.derived for the available ones"
            )
        if conversion.name in seen:
            raise ValueError(
                f"derived parameter {conversion.name!r} clashes with a declared axis "
                f"of the same name; a derived parameter must be a new name for an "
                f"existing axis, not a replacement for the axis list"
            )
        if conversion.name in indexed:
            raise ValueError(f"duplicate derived parameter {conversion.name!r}")

        missing = [n for n in conversion.names if n not in seen]
        if missing:
            raise ValueError(
                f"derived parameter {conversion.name!r} refers to parameters that are "
                f"not declared on this spec: {missing}; declared parameters are "
                f"{list(seen)}"
            )
        indexed[conversion.name] = conversion

    return indexed


class DerivedFrame:
    """A spec viewed along one of its derived parameters.

    The frame has exactly the same shape as the spec it came from -- ``dim`` is
    unchanged and so is every column except one, which is renamed. Neither frame
    is privileged: :meth:`to_base` and :meth:`from_base` both work, whichever
    direction the underlying relation is naturally written in.

    Instances are produced by :meth:`ParameterSpec.frame`, not constructed
    directly.

    Examples
    --------
    >>> from jet.derived import LinearCombination
    >>> spec = ParameterSpec(
    ...     [Param("Omegab", bounds=(0.04, 0.06)), Param("H0", bounds=(60.0, 80.0))],
    ...     derived=[LinearCombination("h", "H0", {"H0": 0.01})],
    ... )
    >>> frame = spec.frame("h")
    >>> frame.names
    ('Omegab', 'h')
    >>> import numpy as np
    >>> frame.to_base(np.array([[0.049, 0.7]]))
    array([[4.9e-02, 7.0e+01]])
    """

    def __init__(self, spec: ParameterSpec, name: str) -> None:
        self.spec = spec
        self.name = name
        self.conversion = spec.derived[name]
        self.base = self.conversion.base

    @property
    def dim(self) -> int:
        """Number of columns, identical to the base spec's."""
        return self.spec.dim

    @property
    def names(self) -> tuple[str, ...]:
        """Column names in this frame, i.e. the base names with one renamed."""
        return tuple(self.name if p.name == self.base else p.name for p in self.spec)

    @property
    def base_names(self) -> tuple[str, ...]:
        """Column names of the spec this frame was taken from."""
        return self.spec.names

    def to_base(self, X: np.ndarray) -> np.ndarray:
        """Convert a value array from this frame into the spec's own frame.

        Parameters
        ----------
        X : array-like of shape (n_samples, dim)
            Points whose ``name`` column holds the derived parameter and whose
            remaining columns are already in base-frame units.

        Returns
        -------
        ndarray of shape (n_samples, dim)
            The same points, with the ``base`` column filled in.

        Raises
        ------
        ValueError
            If ``X`` has the wrong width, or a derived value the base frame
            cannot reproduce.
        """
        X = self._as_2d(X, "X")
        return self.conversion.to_base(X, X[:, self._column(self.name)], self.spec._index)

    def from_base(self, X: np.ndarray) -> np.ndarray:
        """Convert a value array from the spec's own frame into this one.

        Parameters
        ----------
        X : array-like of shape (n_samples, dim)
            Base-frame points.

        Returns
        -------
        ndarray of shape (n_samples, dim)
            The same points, with the ``base`` column replaced by the derived
            parameter.
        """
        X = self._as_2d(X, "X")
        out = X.copy()
        out[:, self._column(self.name)] = self.conversion.to_derived(X, self.spec._index)
        return out

    def __len__(self) -> int:
        return self.dim

    def __repr__(self) -> str:
        return f"DerivedFrame({list(self.names)!r}, from {self.base!r})"

    def _column(self, name: str) -> int:
        """Column index of a name, in whichever frame it belongs to.

        The two frames share a layout, so a derived name occupies the column of
        the axis it stands in for.
        """
        return self.spec.index(self.base if name == self.name else name)

    def _as_2d(self, X: Any, label: str) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(
                f"{label} must have shape (n_samples, {self.dim}) to match "
                f"{list(self.names)}, got {X.shape}"
            )
        return X


class DataVectorSpec:
    """Description of the measured statistics an emulator predicts.

    A data vector is a flat array; this spec says what its entries mean. It is
    deliberately coarse -- it records how many bins there are and whether the
    quantity was log-transformed, not a formula for computing it. The estimator
    that *produces* the vector owns the physics; this spec only has to be
    enough to lay out a prediction, to attach axis labels to a plot, and to
    detect a mismatch between a model and the data it is asked to fit.

    Parameters
    ----------
    name : str
        Short identifier of the statistic, e.g. ``"wp"`` or ``"dsigma"``.
    n_bins : int
        Number of entries in the data vector.
    log : bool, optional
        Whether the emulator should regress the base-10 logarithm of the
        values rather than the values themselves. This drives the ``Log10``
        step of the output transform chain, and it applies to *stored* data:
        training arrays and predictions both stay in physical units, and the
        log is taken and undone inside the emulator. Set it for a statistic
        that spans several decades (``w_p``), where a stationary kernel or a
        mean-squared-error loss would otherwise be dominated by the largest
        bins.
    bin_edges : sequence of float, optional
        Bin boundaries of length ``n_bins + 1``. Retained for labelling and
        plotting; not used in the regression itself.
    labels : sequence of str, optional
        Per-bin labels of length ``n_bins``, replacing auto-generated ones.

    Examples
    --------
    >>> spec = DataVectorSpec("wp", n_bins=15, log=True)
    >>> spec.dim
    15
    >>> spec.axis_label
    'wp'
    """

    def __init__(
        self,
        name: str,
        n_bins: int,
        log: bool = False,
        bin_edges: Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"DataVectorSpec name must be a non-empty string, got {name!r}")
        if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")

        n_bins = int(n_bins)

        if bin_edges is not None:
            edges = np.asarray(bin_edges, dtype=float)
            if edges.ndim != 1 or edges.size != n_bins + 1:
                raise ValueError(
                    f"bin_edges must be 1-D with n_bins + 1 = {n_bins + 1} entries, "
                    f"got shape {edges.shape}"
                )
        else:
            edges = None

        if labels is not None:
            labels = tuple(str(x) for x in labels)
            if len(labels) != n_bins:
                raise ValueError(f"labels must have n_bins = {n_bins} entries, got {len(labels)}")

        self.name = name
        self.n_bins = n_bins
        self.log = bool(log)
        self.bin_edges = edges
        self.labels = labels

    @property
    def dim(self) -> int:
        """Number of entries in the data vector."""
        return self.n_bins

    @property
    def axis_label(self) -> str:
        """Label for the value axis, reflecting the log transform."""
        return f"log10({self.name})" if self.log else self.name

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this spec."""
        return {
            "name": self.name,
            "n_bins": self.n_bins,
            "log": self.log,
            "bin_edges": self.bin_edges.tolist() if self.bin_edges is not None else None,
            "labels": list(self.labels) if self.labels is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DataVectorSpec:
        """Rebuild a spec from :meth:`to_dict` output."""
        return cls(
            name=data["name"],
            n_bins=data["n_bins"],
            log=data.get("log", False),
            bin_edges=data.get("bin_edges"),
            labels=data.get("labels"),
        )

    def hash(self) -> str:
        """Return a short content hash, used to bind a model to its spec."""
        return _canonical_hash(self.to_dict())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataVectorSpec):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return f"DataVectorSpec({self.name!r}, n_bins={self.n_bins}, log={self.log})"
