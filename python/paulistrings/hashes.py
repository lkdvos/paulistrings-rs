"""GF(2) row matrices for ``PauliSum.with_hash(rows=...)`` and ``propagate(partition_rows=...)``.

A matrix has shape ``(rows, 2 * num_qubits)`` and ``bool`` entries: row ``i`` reads coordinate ``c``
when ``[i, c]`` is set, the x-bit of qubit ``c`` for ``c < num_qubits`` and the z-bit of qubit
``c - num_qubits`` above, the layout ``PauliSum.hash_rows()`` returns. Every builder draws from
``numpy.random.default_rng(seed)`` and keeps its rows linearly independent as far as the family
reaches, redrawing a dependent candidate; ``exclude`` keeps the named qubits' x- and z-bits out of
every row, as ``partition_row_exclude`` does for the seeded draw.
"""

import numpy as np

__all__ = ["dense", "sparse", "projection", "rank"]

# Candidates tried for one row before the last is taken as drawn.
_REDRAW_ATTEMPTS = 4096


def _readable(num_qubits, exclude):
    """The coordinates a row may read, after ``exclude={"x": [...], "z": [...]}``."""
    exclude = exclude or {}
    unknown = set(exclude) - {"x", "z"}
    if unknown:
        raise ValueError(f"exclude takes the keys 'x' and 'z', not {sorted(unknown)}")
    out = np.ones(2 * num_qubits, dtype=bool)
    for half, offset in (("x", 0), ("z", num_qubits)):
        for q in exclude.get(half, ()):
            if not 0 <= q < num_qubits:
                raise ValueError(f"excluded {half}-qubit {q} is outside 0..{num_qubits}")
            out[offset + q] = False
    return np.flatnonzero(out)


class _Basis:
    """An incrementally grown GF(2) row-echelon basis."""

    def __init__(self):
        self.pivots = {}

    def insert(self, row):
        """Add ``row`` and return whether it was independent of the rows so far."""
        row = row.copy()
        while row.any():
            lead = int(np.flatnonzero(row)[-1])
            if lead not in self.pivots:
                self.pivots[lead] = row
                return True
            row ^= self.pivots[lead]
        return False


def rank(matrix):
    """The GF(2) rank of a boolean matrix."""
    basis = _Basis()
    return sum(basis.insert(row) for row in np.asarray(matrix, dtype=bool))


def _fill(num_qubits, rows, seed, draw, reachable):
    """``rows`` rows from ``draw(rng)``, independent until ``reachable`` are and nonzero after."""
    rng = np.random.default_rng(seed)
    out = np.zeros((rows, 2 * num_qubits), dtype=bool)
    basis = _Basis()
    for i in range(rows):
        for _ in range(_REDRAW_ATTEMPTS):
            out[i] = draw(rng)
            accepted = basis.insert(out[i]) if len(basis.pivots) < reachable else out[i].any()
            if accepted:
                break
    return out


def dense(num_qubits, rows, seed, exclude=None):
    """``rows`` rows each reading every readable coordinate with probability 1/2, full rank up to their count."""
    readable = _readable(num_qubits, exclude)

    def draw(rng):
        row = np.zeros(2 * num_qubits, dtype=bool)
        row[readable] = rng.integers(0, 2, size=readable.size, dtype=bool)
        return row

    return _fill(num_qubits, rows, seed, draw, min(rows, readable.size))


def sparse(num_qubits, rows, weight, seed, exclude=None):
    """``rows`` rows each reading exactly ``weight`` readable coordinates; an even ``weight`` spans only the even-weight subspace, one dimension short."""
    readable = _readable(num_qubits, exclude)
    if not 1 <= weight <= readable.size:
        raise ValueError(f"weight {weight} must lie in 1..={readable.size}, the readable coordinates")
    if weight == readable.size:
        reachable = 1
    elif weight % 2:
        reachable = readable.size
    else:
        reachable = readable.size - 1

    def draw(rng):
        row = np.zeros(2 * num_qubits, dtype=bool)
        row[rng.choice(readable, size=weight, replace=False)] = True
        return row

    return _fill(num_qubits, rows, seed, draw, min(rows, reachable))


def projection(num_qubits, rows, seed, exclude=None):
    """``rows`` rows each reading one readable coordinate, distinct until they run out and cycling after: the bucket is a few bits of the string."""
    readable = _readable(num_qubits, exclude)
    order = np.random.default_rng(seed).permutation(readable) if readable.size else readable
    out = np.zeros((rows, 2 * num_qubits), dtype=bool)
    for i in range(rows):
        if order.size:
            out[i, order[i % order.size]] = True
    return out
