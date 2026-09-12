"""
The emulator: a transform chain wrapped around a regression backend.

An :class:`Emulator` owns three things and nothing else:

1. the two specifications that define what the input columns and output bins
   *mean* (:class:`~jet.spec.ParameterSpec`, :class:`~jet.spec.DataVectorSpec`);
2. a chain of :mod:`~jet.emulator.transforms` steps that maps raw physical
   values onto the well-conditioned space a regressor wants;
3. a backend that does the regression in that space.

Nothing about the regression lives outside the backend, and nothing about the
physics lives inside it. Adding a new backend therefore means implementing
``fit``/``predict`` and registering a name; it does not mean restating how
parameters are normalised.

Example
-------
>>> import numpy as np
>>> from jet.spec import ParameterSpec, Param, DataVectorSpec
>>> from jet.emulator import Emulator
>>> spec = ParameterSpec([Param("x", bounds=(0.0, 1.0))])
>>> dv = DataVectorSpec("demo", n_bins=2)
>>> X = np.linspace(0.0, 1.0, 20)[:, None]
>>> y = np.column_stack([np.sin(X[:, 0]), np.cos(X[:, 0])])
>>> em = Emulator(spec, dv, backend="gp").fit(X, y)   # doctest: +SKIP
>>> mean, std = em.predict(np.array([[0.5]]))         # doctest: +SKIP
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.parse
import urllib.request
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import DataVectorSpec, ParameterSpec
from .backends import Backend, get_backend
from .bundle import load_bundle, save_bundle
from .transforms import (
    PCA,
    BoundsNorm,
    Log10,
    StandardScaler,
    Transform,
    forward,
    inverse,
    inverse_std,
)

__all__ = ["Emulator", "resolve_weights_path", "data_dir"]

#: Environment variable consulted when a weight file is given by name rather
#: than by path. Point it at the shared location holding the ``.npz`` files.
DATA_DIR_ENV = "JET_DATA_DIR"

#: Any non-empty value disables the automatic fetch from GitHub Release.
NO_AUTO_FETCH_ENV = "JET_NO_AUTO_FETCH"

#: Repository whose release assets hold the bundled weights. Override for a
#: fork or mirror with ``JET_RELEASE_REPO``.
RELEASE_REPO_ENV = "JET_RELEASE_REPO"
DEFAULT_RELEASE_REPO = "czymh/jet"

#: Pin a specific release tag instead of the latest one.
RELEASE_TAG_ENV = "JET_RELEASE_TAG"

#: Environment variables consulted for a GitHub token (private repositories).
_TOKEN_ENVS = ("GITHUB_TOKEN", "JET_DATA_TOKEN")

#: HTTP timeout for the GitHub API lookups and asset downloads.
_HTTP_TIMEOUT = 10.0

#: Accepted values of the ``x_normalize`` argument.
_X_NORMALIZE_MODES = ("standardize", "bounds", "bounds+standardize", "none")


def cache_data_dir() -> Path:
    """Return the per-user cache directory weights are fetched into.

    Follows the XDG convention: ``$XDG_CACHE_HOME/jet/data``, falling back to
    ``~/.cache/jet/data``. This is where the automatic GitHub Release fetch
    lands, so a plain ``pip install`` needs no environment configuration.

    Returns
    -------
    pathlib.Path
        The cache directory. Not guaranteed to exist; callers create it.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "jet" / "data"


def data_dir() -> Path:
    """Return the directory holding jet's own bundled data files.

    Resolution order:

    1. :data:`DATA_DIR_ENV` when set -- a shared cluster copy, say;
    2. the package's own ``data`` directory when it already holds files --
       a checkout or wheel that ships weights;
    3. :func:`cache_data_dir`, where weights fetched from GitHub Release land
       on first use.

    The environment override exists so that a shared installation can be
    pointed at a copy living on a cluster filesystem without moving the
    package; the cache fallback exists so that an unconfigured installation
    still resolves -- and fetches -- sensibly.

    Returns
    -------
    pathlib.Path
        Directory that bundled data files live in. Not guaranteed to exist:
        the files themselves are not tracked by git, and the readers raise
        with instructions when they look inside an empty one.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    pkg_data = Path(__file__).resolve().parent.parent / "data"
    if pkg_data.is_dir() and any(pkg_data.iterdir()):
        return pkg_data
    return cache_data_dir()


def _github_headers(token: str | None, *, octet: bool = False) -> dict[str, str]:
    """Headers for a GitHub API call, with optional token and media type."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "jet"}
    if octet:
        headers["Accept"] = "application/octet-stream"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _release_tag() -> str:
    """Resolve the release tag the weights should come from.

    ``JET_RELEASE_TAG`` pins one; otherwise the latest release of the
    repository is used.
    """
    pinned = os.environ.get(RELEASE_TAG_ENV)
    if pinned:
        return pinned
    repo = os.environ.get(RELEASE_REPO_ENV, DEFAULT_RELEASE_REPO)
    token = next((os.environ.get(k) for k in _TOKEN_ENVS if os.environ.get(k)), None)
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=_github_headers(token)), timeout=_HTTP_TIMEOUT
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["tag_name"])


def _download_release_asset(name: str, dest: Path) -> Path:
    """Download ``name`` from the project's release assets into ``dest``.

    Uses the GitHub API throughout (public or token-authenticated), streams to
    a temporary sibling, and moves the file into place so an interrupted
    download cannot leave a half-written bundle behind -- the same discipline
    :func:`jet.emulator.bundle.save_bundle` applies on write.
    """
    repo = os.environ.get(RELEASE_REPO_ENV, DEFAULT_RELEASE_REPO)
    token = next((os.environ.get(k) for k in _TOKEN_ENVS if os.environ.get(k)), None)
    tag = _release_tag()

    assets_url = (
        f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    )
    with urllib.request.urlopen(
        urllib.request.Request(assets_url, headers=_github_headers(token)), timeout=_HTTP_TIMEOUT
    ) as response:
        release = json.loads(response.read().decode("utf-8"))

    matches = [a for a in release.get("assets", []) if a.get("name") == name]
    if not matches:
        available = sorted(a.get("name", "") for a in release.get("assets", []))
        raise FileNotFoundError(
            f"release {tag!r} of {repo} has no asset named {name!r}; "
            f"available: {available or 'none'}"
        )

    asset_url = f"https://api.github.com/repos/{repo}/releases/assets/{matches[0]['id']}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(asset_url, headers=_github_headers(token, octet=True))
    try:
        with (
            urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response,
            open(tmp, "wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _fetch_release_weights(name: str, dest_dir: Path) -> Path:
    """Fetch one weight file from GitHub Release into ``dest_dir``."""
    return _download_release_asset(name, Path(dest_dir) / name)


def resolve_weights_path(spec: str | os.PathLike[str], suffix: str = ".npz") -> Path:
    """Turn a weight file given by name or path into an existing path.

    Resolution order:

    1. the argument itself, if it is an existing file;
    2. under the data directory (``$JET_DATA_DIR``, the package ``data``
       directory, or the user cache -- whichever :func:`data_dir` resolves
       to), appending ``suffix`` when the name has none;
    3. when the argument is a bare name (no directory component), the file is
       fetched from the project's GitHub Release and the lookup is retried.

    The automatic fetch exists so that a plain ``pip install`` needs no
    environment configuration: the first ``Emulator.load("wp.gp.npz")`` pulls
    the weights into the user cache and every later call hits the cache. It
    can be disabled with :data:`NO_AUTO_FETCH_ENV`. A name that resolves
    nowhere raises and lists the directories that were tried, because the
    failure mode this guards against -- silently loading a stale copy from a
    relative path -- is otherwise very hard to spot.

    Parameters
    ----------
    spec : str or path-like
        A filesystem path, or a bare filename relative to the data directory.
    suffix : str, optional
        Default suffix appended when the name has none.

    Returns
    -------
    pathlib.Path
        The resolved, existing path.

    Raises
    ------
    FileNotFoundError
        If no candidate exists (and the fetch, when attempted, failed).
    """
    given = Path(spec).expanduser()

    if given.exists():
        return given

    root = os.environ.get(DATA_DIR_ENV)
    base = Path(root).expanduser() if root else data_dir()
    candidates: list[Path] = [base / given]
    if given.suffix == "":
        candidates.append(base / f"{given}{suffix}")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    download_note = ""
    if given.parent == Path(".") and not os.environ.get(NO_AUTO_FETCH_ENV):
        name = given.name if given.suffix else f"{given.name}{suffix}"
        try:
            _fetch_release_weights(name, candidates[0].parent)
        except Exception as exc:  # any failure falls back to the standard error
            download_note = f" Automatic download from GitHub Release failed: {exc}"
        else:
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            download_note = f" Automatic download wrote nothing to {candidates[0].parent}."

    tried = [str(p) for p in [given, *candidates]]
    hint = (
        f"Set {DATA_DIR_ENV} to the directory holding the weight files, or pass an absolute path."
        if root is None
        else f"{DATA_DIR_ENV}={root!r}"
    )
    raise FileNotFoundError(f"no weight file found; tried {tried}. {hint}{download_note}")


class Emulator:
    """A trained (or trainable) surrogate for one statistic.

    Parameters
    ----------
    x_spec : ParameterSpec
        Input parameters, in column order.
    y_spec : DataVectorSpec
        The statistic being predicted, defining the output width.
    backend : str or Backend, optional
        Either a registry key (``"gp"``, ``"nn"``) or a configured backend
        instance. Passing an instance is how backend-specific settings --
        kernel bounds, network width, ensemble size -- are supplied; passing a
        string keeps the common case short.
    n_pca : int, optional
        Number of principal components to retain on the output side. ``None``
        (the default) appends no PCA step, so the backend regresses the data
        vector directly. Retaining components decorrelates the outputs and is
        usually worth it when a Gaussian process is used and the bins are
        smooth.
    backend_kwargs : mapping, optional
        Keyword arguments forwarded to the backend constructor when ``backend``
        is a string. Rejected when ``backend`` is already an instance.
    x_normalize : str, optional
        How to map the inputs before regression:

        ``"standardize"``
            Centre and scale (the default; what both backends expect).
        ``"bounds"``
            Affine map of the declared bounds onto ``[0, 1]``.
        ``"bounds+standardize"``
            Both, bounds first.
        ``"none"``
            Pass raw values through.

        Bounds are used to *warn* about extrapolation regardless of this
        setting; the choice here only affects conditioning.
    bounds_slack : float, optional
        Fractional tolerance for the extrapolation warning, e.g. ``0.05``
        accepts a point five percent of the range beyond a bound.
    warn_on_extrapolation : bool, optional
        Emit a :class:`RuntimeWarning` when a prediction point lies outside the
        declared bounds. The reference implementations extrapolate silently,
        which is how a model ends up quoted outside the region it was trained
        for.

    Attributes
    ----------
    manifest : dict
        Provenance recorded at save time; empty until :meth:`save` or
        :meth:`load` has run.

    Notes
    -----
    The model reads the axes of ``x_spec`` and nothing else. When ``x_spec``
    also declares derived parameters (:mod:`jet.derived`), converting into them
    is the caller's job and happens before the call::

        emulator.predict(emulator.x_spec.frame("sigma8").to_base(X))

    Keeping the substitution outside means a parameter array has exactly one
    interpretation at the point the model sees it, and the conversion is
    visible in the code that relies on it.
    """

    def __init__(
        self,
        x_spec: ParameterSpec,
        y_spec: DataVectorSpec,
        backend: str | Backend = "gp",
        n_pca: int | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        x_normalize: str = "standardize",
        bounds_slack: float = 0.0,
        warn_on_extrapolation: bool = True,
    ) -> None:
        if not isinstance(x_spec, ParameterSpec):
            raise TypeError(f"x_spec must be a ParameterSpec, got {type(x_spec).__name__}")
        if not isinstance(y_spec, DataVectorSpec):
            raise TypeError(f"y_spec must be a DataVectorSpec, got {type(y_spec).__name__}")
        if x_normalize not in _X_NORMALIZE_MODES:
            raise ValueError(
                f"x_normalize must be one of {list(_X_NORMALIZE_MODES)}, got {x_normalize!r}"
            )
        if n_pca is not None and n_pca < 1:
            raise ValueError(f"n_pca must be a positive integer or None, got {n_pca}")

        self.x_spec = x_spec
        self.y_spec = y_spec
        self.n_pca = n_pca
        self.x_normalize = x_normalize
        self.bounds_slack = float(bounds_slack)
        self.warn_on_extrapolation = bool(warn_on_extrapolation)

        if isinstance(backend, str):
            self._backend = get_backend(backend, **dict(backend_kwargs or {}))
        elif isinstance(backend, Backend):
            if backend_kwargs:
                raise ValueError(
                    "backend_kwargs cannot be combined with an already-constructed "
                    "backend instance; pass the settings to the constructor instead"
                )
            self._backend = backend
        else:
            raise TypeError(
                f"backend must be a registry key or a Backend, got {type(backend).__name__}"
            )

        self.x_chain: list[Transform] = []
        self.y_chain: list[Transform] = []
        self.manifest: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def backend(self) -> Backend:
        """The regression backend instance."""
        return self._backend

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` or :meth:`load` has produced a usable model."""
        return bool(self.x_chain) and bool(self.y_chain)

    @property
    def x_spec_hash(self) -> str:
        """Content hash of the input specification."""
        return self.x_spec.hash()

    @property
    def y_spec_hash(self) -> str:
        """Content hash of the output specification."""
        return self.y_spec.hash()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> Emulator:
        """Train the transform chains and the backend.

        Parameters
        ----------
        X : array-like of shape (n_samples, x_spec.dim)
            Raw parameter values, columns in ``x_spec`` order. Arbitrary points,
            not a grid: the emulator knows nothing about Cartesian structure.
        y : array-like of shape (n_samples, y_spec.dim)
            Raw data vectors, matching ``y_spec``.

        Returns
        -------
        Emulator
            ``self``, fitted.
        """
        X = self._validate_X(X, "X")
        y = self._validate_y(y, "y")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}"
            )

        self.x_chain = self._build_x_chain()
        self.y_chain = self._build_y_chain()
        X_transformed = self._fit_chain(self.x_chain, X)
        y_transformed = self._fit_chain(self.y_chain, y)

        self._backend.fit(X_transformed, y_transformed)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict data vectors at raw parameter points.

        Parameters
        ----------
        X : array-like of shape (n_samples, x_spec.dim)
            Raw parameter values, columns in ``x_spec`` order.
        return_std : bool, optional
            Also return the predictive standard deviation. When the backend
            does not report one -- the neural-network backend currently does
            not -- this raises rather than returning zeros, because an array of
            zeros downstream reads as perfect certainty.

        Returns
        -------
        mean : ndarray of shape (n_samples, y_spec.dim)
            Predicted data vectors, back in physical units.
        std : ndarray of shape (n_samples, y_spec.dim)
            Only when ``return_std`` is true. Includes both the backend's own
            predictive spread and the effect of the inverse transform chain.
        """
        self._require_fitted()
        X = self._validate_X(X, "X")
        if self.warn_on_extrapolation:
            self._warn_if_extrapolating(X)

        X_transformed = forward(self.x_chain, X)
        mean_transformed, std_transformed = self._backend.predict(X_transformed)

        if not return_std:
            return inverse(self.y_chain, mean_transformed)

        if std_transformed is None:
            raise ValueError(
                f"backend {self._backend.name!r} does not report a predictive "
                "uncertainty. Call predict(..., return_std=False), or use a backend "
                "that does -- the Gaussian-process backend always returns one."
            )

        mean, std = inverse_std(self.y_chain, mean_transformed, std_transformed)
        return mean, std

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(
        self,
        path: str | os.PathLike[str],
        extra: Mapping[str, Any] | None = None,
        extra_arrays: Mapping[str, np.ndarray] | None = None,
    ) -> Path:
        """Write the trained model to a single ``.npz``.

        Parameters
        ----------
        path : path-like
            Destination. ``.npz`` is appended when missing.
        extra : mapping, optional
            Extra provenance recorded in the manifest, e.g. the training-set
            origin. Values must be JSON-serialisable.
        extra_arrays : mapping of str to ndarray, optional
            Further arrays stored in the file verbatim, e.g. a grid the data
            vector is defined on. Read back through
            :func:`jet.emulator.bundle.load_bundle`.

        Returns
        -------
        pathlib.Path
            The path written.
        """
        self._require_fitted()
        written = save_bundle(
            path,
            x_spec=self.x_spec,
            y_spec=self.y_spec,
            x_chain=self.x_chain,
            y_chain=self.y_chain,
            backend=self._backend,
            extra=extra,
            extra_arrays=extra_arrays,
        )
        self.manifest = load_bundle(written)["manifest"]
        return written

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        x_spec: ParameterSpec | None = None,
        y_spec: DataVectorSpec | None = None,
    ) -> Emulator:
        """Load a trained model from a bundle.

        Nothing beyond NumPy is needed, whatever backend wrote the file.

        Parameters
        ----------
        path : str or path-like
            A path, or a name resolved through :func:`resolve_weights_path`.
        x_spec, y_spec : specs, optional
            When given, the bundle's recorded specs are checked against them
            and a mismatch raises. Supply these whenever the model will be fed
            data from a pipeline, so that a column or bin reordering is caught
            here rather than showing up as a subtly wrong prediction.

        Returns
        -------
        Emulator
            The restored model.
        """
        resolved = resolve_weights_path(path)
        loaded = load_bundle(resolved)

        emulator = cls(
            x_spec=loaded["x_spec"],
            y_spec=loaded["y_spec"],
            backend=loaded["backend"],
            n_pca=_pca_components(loaded["y_chain"]),
        )
        emulator.x_chain = loaded["x_chain"]
        emulator.y_chain = loaded["y_chain"]
        emulator.manifest = loaded["manifest"]

        if x_spec is not None or y_spec is not None:
            emulator.verify(x_spec=x_spec, y_spec=y_spec)
        return emulator

    def verify(
        self,
        x_spec: ParameterSpec | None = None,
        y_spec: DataVectorSpec | None = None,
    ) -> None:
        """Check this model against caller-supplied specifications.

        Parameters
        ----------
        x_spec, y_spec : specs, optional
            Specifications to compare against; whichever is ``None`` is skipped.

        Raises
        ------
        ValueError
            If a supplied spec differs from the one the model was trained on.
            The message names the offending side, because ``ParameterSpec``
            column order or ``DataVectorSpec`` bin count being wrong is the
            usual cause of a silent mismatch.
        """
        self._require_fitted()
        if x_spec is not None and x_spec.hash() != self.x_spec_hash:
            raise ValueError(
                "input spec mismatch: this model was trained on "
                f"{list(self.x_spec.names)} (hash {self.x_spec_hash}), "
                f"but was given {list(x_spec.names)} (hash {x_spec.hash()})"
            )
        if y_spec is not None and y_spec.hash() != self.y_spec_hash:
            raise ValueError(
                f"output spec mismatch: this model predicts {self.y_spec!r} "
                f"(hash {self.y_spec_hash}), but was given {y_spec!r} "
                f"(hash {y_spec.hash()})"
            )

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------
    def _build_x_chain(self) -> list[Transform]:
        """Assemble the input transform chain from the spec and settings."""
        chain: list[Transform] = []

        if "bounds" in self.x_normalize:
            if not self.x_spec.has_bounds:
                missing = [p.name for p in self.x_spec if p.bounds is None]
                raise ValueError(
                    f"x_normalize={self.x_normalize!r} needs bounds on every parameter; "
                    f"missing: {missing}"
                )
            bounds = self.x_spec.bounds_array()
            chain.append(BoundsNorm(bounds[:, 0], bounds[:, 1]))

        if "standardize" in self.x_normalize:
            chain.append(StandardScaler())

        return chain

    def _build_y_chain(self) -> list[Transform]:
        """Assemble the output transform chain from the spec and settings."""
        chain: list[Transform] = []

        if self.y_spec.log:
            chain.append(Log10())
        chain.append(StandardScaler())

        if self.n_pca is not None:
            chain.append(PCA(self.n_pca))
            # The PCA scores are not unit-variance; a second standardisation
            # puts them on the scale both backends assume.
            chain.append(StandardScaler())

        return chain

    @staticmethod
    def _fit_chain(chain: Sequence[Transform], a: np.ndarray) -> np.ndarray:
        """Fit each step of a chain in turn, returning the fully transformed data."""
        for step in chain:
            step.fit(a)
            a = step.transform(a)
        return a

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_X(self, X: Any, name: str) -> np.ndarray:
        """Coerce and shape-check an input array."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.x_spec.dim:
            raise ValueError(
                f"{name} must have shape (n_samples, {self.x_spec.dim}) to match "
                f"{list(self.x_spec.names)}, got {X.shape}"
            )
        return X

    def _validate_y(self, y: Any, name: str) -> np.ndarray:
        """Coerce and shape-check a target array."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1 and self.y_spec.dim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[1] != self.y_spec.dim:
            raise ValueError(
                f"{name} must have shape (n_samples, {self.y_spec.dim}) to match "
                f"{self.y_spec!r}, got {y.shape}"
            )
        return y

    def _warn_if_extrapolating(self, X: np.ndarray) -> None:
        """Warn when prediction points leave the declared parameter ranges."""
        violations = self.x_spec.check_within_bounds(X, slack=self.bounds_slack)
        if not violations:
            return
        preview = ", ".join(
            f"{name}={value:.4g} (row {row})" for name, row, value in violations[:5]
        )
        more = "" if len(violations) <= 5 else f" and {len(violations) - 5} more"
        warnings.warn(
            f"{len(violations)} prediction value(s) fall outside the trained parameter "
            f"range: {preview}{more}. The emulator is extrapolating and its output is "
            "not trustworthy there.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Emulator has not been fitted; call fit() or load() first")

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return (
            f"Emulator({state}, backend={self._backend.name!r}, "
            f"n_params={self.x_spec.dim}, {self.y_spec!r})"
        )


def _pca_components(chain: Sequence[Transform]) -> int | None:
    """Read the component count back out of a loaded chain, if it has a PCA step."""
    for step in chain:
        if isinstance(step, PCA):
            return step.n_components
    return None
