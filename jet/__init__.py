"""
jet: JUST Emulator Toolkit.

Emulator pipeline for the JUST (Jiao Tong University Spectroscopic Telescope)
survey, following the ACM (Alternative Clustering Methods) framework:
mocks -> clustering statistics -> neural emulator -> cosmological inference.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jet")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
