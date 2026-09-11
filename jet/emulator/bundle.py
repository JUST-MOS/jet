"""
On-disk format for a trained emulator: one ``.npz`` per model.

The layout is chosen so that a bundle is readable with nothing but NumPy, and
so that it survives library upgrades. Every array is stored raw -- no pickled
scikit-learn estimators, no ``torch.save`` blobs -- because a pickle pins the
exact library version that wrote it and produces the ``InconsistentVersion``
warnings the reference implementation had to suppress on load.

A bundle holds::

    manifest                      JSON string: specs, chain order, provenance
    x.<i>.<name>.<key>            state array of transform i on the input side
    y.<i>.<name>.<key>            state array of transform i on the output side
    backend.<name>.<key>          state array of the regression backend
    extra.<name>                  any further array the writer asked to store

The manifest repeats the transform *names* in order, so the numbered keys can
be reassembled into a chain without guessing. It also carries the spec hashes,
which :meth:`jet.emulator.Emulator.verify` compares against a caller-supplied
spec: applying a model to a differently-shaped parameter space is a silent
garbage-in-garbage-out failure, and the hash turns it into a loud one.

``extra`` arrays exist so that a model needing a supporting grid -- the
wavenumbers of a tabulated :math:`P(k)`, say -- can carry it in the same file
rather than in a sibling that can drift out of step or go missing in transit.

Dots, not slashes, separate key components: NumPy writes each array as a member
of the underlying zip archive, and a slash would be interpreted as a directory.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import DataVectorSpec, ParameterSpec
from .backends import Backend, backend_from_state
from .transforms import Transform, transform_from_state

__all__ = ["MANIFEST_KEY", "EXTRA_PREFIX", "save_bundle", "load_bundle"]

#: Key under which the JSON manifest is stored inside the archive.
MANIFEST_KEY = "manifest"

#: Prefix of the keys holding arrays the writer stored verbatim.
EXTRA_PREFIX = "extra."


def _json_default(value: Any) -> Any:
    """Coerce NumPy scalars and arrays into JSON-serialisable objects."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__} into a bundle manifest")


def _chain_keys(prefix: str, chain: Sequence[Transform]) -> dict[str, np.ndarray]:
    """Flatten a transform chain into bundle keys."""
    arrays: dict[str, np.ndarray] = {}
    for index, step in enumerate(chain):
        for key, value in step.state().items():
            arrays[f"{prefix}.{index}.{step.name}.{key}"] = np.asarray(value)
    return arrays


def _rebuild_chain(
    prefix: str, names: Sequence[str], archive: Mapping[str, np.ndarray]
) -> list[Transform]:
    """Reassemble a transform chain from numbered bundle keys."""
    chain: list[Transform] = []
    for index, name in enumerate(names):
        head = f"{prefix}.{index}.{name}."
        state = {key[len(head) :]: value for key, value in archive.items() if key.startswith(head)}
        if not state:
            raise KeyError(f"bundle is missing the state of {prefix} transform {index} ({name})")
        chain.append(transform_from_state(name, state))
    return chain


def save_bundle(
    path: str | os.PathLike[str],
    *,
    x_spec: ParameterSpec,
    y_spec: DataVectorSpec,
    x_chain: Sequence[Transform],
    y_chain: Sequence[Transform],
    backend: Backend,
    extra: Mapping[str, Any] | None = None,
    extra_arrays: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Write a trained emulator to ``path``.

    The file is written to a temporary sibling and moved into place, so an
    interrupted write cannot leave a half-written bundle behind -- which matters
    on the shared network filesystems this is expected to run on.

    Parameters
    ----------
    path : path-like
        Destination ``.npz``. The suffix is appended when missing.
    x_spec, y_spec : specs
        The specifications the model was trained against; their hashes are
        recorded so that :meth:`jet.emulator.Emulator.verify` can check them.
    x_chain, y_chain : sequence of Transform
        The fitted preprocessing chains, in application order.
    backend : Backend
        The fitted regression backend.
    extra : mapping, optional
        Additional provenance recorded in the manifest, e.g. training-set
        origin. Values must be JSON-serialisable.
    extra_arrays : mapping of str to ndarray, optional
        Further arrays to store in the file verbatim, read back under
        ``extra_arrays`` by :func:`load_bundle`.

    Returns
    -------
    pathlib.Path
        The path actually written.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    from .. import __version__ as jet_version

    manifest: dict[str, Any] = {
        "format": "jet-emulator-bundle",
        "format_version": 1,
        "jet_version": jet_version,
        "x_spec": x_spec.to_dict(),
        "y_spec": y_spec.to_dict(),
        "x_spec_hash": x_spec.hash(),
        "y_spec_hash": y_spec.hash(),
        "x_chain": [step.name for step in x_chain],
        "y_chain": [step.name for step in y_chain],
        "backend": backend.name,
        "backend_repr": repr(backend),
    }
    if extra:
        manifest["extra"] = dict(extra)

    arrays: dict[str, np.ndarray] = {
        MANIFEST_KEY: np.asarray(json.dumps(manifest, default=_json_default))
    }
    arrays.update(_chain_keys("x", x_chain))
    arrays.update(_chain_keys("y", y_chain))
    for key, value in backend.state().items():
        arrays[f"backend.{backend.name}.{key}"] = np.asarray(value)
    for key, value in (extra_arrays or {}).items():
        arrays[f"{EXTRA_PREFIX}{key}"] = np.asarray(value)

    handle = tempfile.NamedTemporaryFile(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
        # NamedTemporaryFile creates its file 0600, and os.replace carries that
        # mode over to the destination. On a shared filesystem that silently
        # produces a model nobody else can read, so put the mode back to what a
        # plain open() would have given. The umask is read and restored around a
        # zeroing call because Python does not expose it any other way; this is
        # the documented idiom, and the window is a single syscall wide.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(handle.name, 0o666 & ~umask)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise

    return path


def load_bundle(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a bundle back into specs, fitted chains and a fitted backend.

    Parameters
    ----------
    path : path-like
        An ``.npz`` written by :func:`save_bundle`.

    Returns
    -------
    dict
        Keys ``x_spec``, ``y_spec``, ``x_chain``, ``y_chain``, ``backend``,
        ``manifest`` and ``extra_arrays``.

    Raises
    ------
    ValueError
        If the file is not a jet emulator bundle, or its format version is
        newer than this code understands.
    KeyError
        If the file is missing pieces the manifest says it should contain.
    """
    path = Path(path)

    with np.load(path, allow_pickle=False) as archive:
        if MANIFEST_KEY not in archive:
            raise ValueError(f"{path} is not a jet emulator bundle: no {MANIFEST_KEY!r} entry")
        manifest = json.loads(str(archive[MANIFEST_KEY]))
        arrays = {key: archive[key] for key in archive.files if key != MANIFEST_KEY}

    format_version = manifest.get("format_version", 0)
    if format_version > 1:
        raise ValueError(
            f"{path} was written by a newer jet (bundle format {format_version}, "
            "this build understands 1)"
        )

    x_chain = _rebuild_chain("x", manifest.get("x_chain", []), arrays)
    y_chain = _rebuild_chain("y", manifest.get("y_chain", []), arrays)

    backend_name = manifest["backend"]
    prefix = f"backend.{backend_name}."
    backend_state = {
        key[len(prefix) :]: value for key, value in arrays.items() if key.startswith(prefix)
    }
    if not backend_state:
        raise KeyError(f"bundle is missing the state of backend {backend_name!r}")

    extras = {
        key[len(EXTRA_PREFIX) :]: value
        for key, value in arrays.items()
        if key.startswith(EXTRA_PREFIX)
    }

    return {
        "x_spec": ParameterSpec.from_dict(manifest["x_spec"]),
        "y_spec": DataVectorSpec.from_dict(manifest["y_spec"]),
        "x_chain": x_chain,
        "y_chain": y_chain,
        "backend": backend_from_state(backend_name, backend_state),
        "manifest": manifest,
        "extra_arrays": extras,
    }
