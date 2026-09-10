"""Pauli propagation library — Python entry point.

The user-visible surface is a thin layer over width-monomorphized enums that
dispatch once outside any hot loop (see ARCHITECTURE.md §Python-Bindings). The
compiled extension lives at ``paulistrings._paulistrings``; this package
re-exports the high-level classes and exposes the ``gates``, ``noise``, and
``truncation`` factory submodules.

``PauliSum.propagate`` also runs the sum split across partitions: ``partitions=``
places one pinned thread pool per NUMA domain in this process, ``comm=`` takes
an ``mpi4py`` communicator and places one partition per rank. ``numa_nodes()``
reports what ``partitions="auto"`` has to place against, and
``mpi_available()`` whether this build was compiled with the ``mpi`` feature.
"""

from . import _paulistrings
from ._paulistrings import (
    DEFAULT_SMALL_SUM_THRESHOLD,
    Circuit,
    PartitionStats,
    PauliSum,
    PropagationStats,
    mpi_available,
    numa_nodes,
    reset_log_cache,
)
from . import gates, noise, truncation
from . import interop
from . import io

__all__ = [
    "Circuit",
    "PauliSum",
    "PropagationStats",
    "PartitionStats",
    "DEFAULT_SMALL_SUM_THRESHOLD",
    "gates",
    "noise",
    "truncation",
    "interop",
    "io",
    "mpi_available",
    "numa_nodes",
    "reset_log_cache",
]
