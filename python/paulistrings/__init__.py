"""Pauli propagation library — Python entry point.

The user-visible surface mirrors the design doc §11. The compiled extension
lives at ``paulistrings._paulistrings``; this package re-exports the high-level
classes and exposes the ``gates``, ``noise``, and ``truncation`` factory
submodules.
"""

from . import _paulistrings
from ._paulistrings import Circuit, PauliSum, reset_log_cache
from . import gates, noise, truncation

__all__ = [
    "Circuit",
    "PauliSum",
    "gates",
    "noise",
    "truncation",
    "reset_log_cache",
]
