"""Pauli propagation library — Python entry point.

The user-visible surface is a thin layer over width-monomorphized enums that
dispatch once outside any hot loop (see ARCHITECTURE.md §Python-Bindings). The
compiled extension lives at ``paulistrings._paulistrings``; this package
re-exports the high-level classes and exposes the ``gates``, ``noise``, and
``truncation`` factory submodules.
"""

from . import _paulistrings
from ._paulistrings import Circuit, PauliSum, PropagationStats, reset_log_cache
from . import gates, noise, truncation
from . import interop

__all__ = [
    "Circuit",
    "PauliSum",
    "PropagationStats",
    "gates",
    "noise",
    "truncation",
    "interop",
    "reset_log_cache",
]
