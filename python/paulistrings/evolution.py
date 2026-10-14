"""Trotterized time evolution as circuits."""

from __future__ import annotations

from ._paulistrings import Circuit, PauliTerms

__all__ = ["trotter"]


def trotter(
    hamiltonian: PauliTerms, dt: float, num_qubits: int, *, steps: int = 1, order: int = 1
) -> Circuit:
    """``steps`` Trotter steps of ``exp(-i H dt)``, each built by ``Circuit.trotter``.

    For a time series, build one step and propagate it repeatedly (Manual: Incremental propagation).
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    circuit = Circuit(num_qubits)
    for _ in range(steps):
        circuit.trotter(hamiltonian, dt, order)
    return circuit