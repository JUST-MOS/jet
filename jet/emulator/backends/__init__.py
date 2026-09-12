"""
Regression backends for the emulator, and the registry that resolves them.

Importing this package registers every built-in backend. Neither
:mod:`~jet.emulator.backends.gp` nor :mod:`~jet.emulator.backends.nn` imports
its heavy training dependency at module level -- scikit-learn and PyTorch are
imported inside ``fit`` -- so this import stays cheap and the prediction path
never drags in a training stack.
"""

from __future__ import annotations

from . import gp, nn  # noqa: F401  (imported for their registration side effect)
from .base import (
    BACKENDS,
    Backend,
    backend_from_state,
    get_backend,
    register_backend,
)

__all__ = [
    "Backend",
    "BACKENDS",
    "register_backend",
    "get_backend",
    "backend_from_state",
]
