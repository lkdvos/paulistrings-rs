"""Copying forms of the in-place ``PauliSum.propagate`` and ``PauliSum.propagate_with_stats``."""


def propagated(pauli_sum, *args, **kwargs):
    """A copy of ``pauli_sum`` propagated by ``PauliSum.propagate(*args, **kwargs)``; ``pauli_sum`` itself is untouched."""
    evolved = pauli_sum.copy()
    evolved.propagate(*args, **kwargs)
    return evolved


def propagated_with_stats(pauli_sum, *args, **kwargs):
    """``(evolved, stats)``: a copy of ``pauli_sum`` propagated by ``PauliSum.propagate_with_stats(*args, **kwargs)``, and its ``PropagationStats``."""
    evolved = pauli_sum.copy()
    stats = evolved.propagate_with_stats(*args, **kwargs)
    return evolved, stats
