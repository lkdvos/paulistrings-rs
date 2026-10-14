# PauliTerms

An ordered list of sparse Pauli terms `c * P`, with no qubit count.
Each term names only the sites it acts on.
Terms keep the order they were written in, repeats included; binding to a register merges them.

## Constructors

| Signature | Builds |
|---|---|
| `paulistrings.pt(label, sites)` | coefficient-`1` terms, `label[k]` on `sites[k]` |
| `PauliTerms()` | the empty list |

| `sites` | Terms |
|---|---|
| an int, `pt("X", 3)` | one term |
| a tuple of `len(label)` ints, `pt("ZZ", (0, 1))` | one term |
| a list of tuples, `pt("ZZ", [(0, 1), (1, 2)])` | one term per tuple |
| a list of ints with a one-letter label, `pt("X", range(4))` | one term per site |
| an empty list | no terms |

`label` uses `X`, `Y` and `Z` only; identity sites are left out.
A label/site length mismatch, a site repeated within a term, or a character outside `XYZ` is a `ValueError`; a `bool` for `sites` is a `TypeError`.

## Arithmetic

| Call | Returns |
|---|---|
| `a + b`, `a - b` | the terms of `a`, then those of `b` (negated for `-`); nothing is merged |
| `-a`, `a * c`, `c * a`, `a / c` | every coefficient scaled by, or divided by, a complex or real `c` |
| `sum(parts, PauliTerms())` | every `PauliTerms` in `parts`, concatenated in order |

`*` between two `PauliTerms` is a `TypeError`, and `a / 0` is a `ZeroDivisionError`.
NumPy scalars work on either side of `*`.
A bare `sum(parts)`, starting from `0`, also works.

## Accessors

| Call | Returns |
|---|---|
| `len(terms)` | term count, repeats included |
| `iter(terms)` | `(label, sites, coefficient)` triples in order, `sites` a tuple |
| `str(terms)`, `repr(terms)` | `coefficient*P...` for the first few terms, e.g. `-1*Z0Z1`; `0` when empty |

## Binding

| Call | Gives |
|---|---|
| `PauliSum(num_qubits, terms)` | the operator on `num_qubits` qubits; repeated strings summed, exact zeros dropped |
| `circuit.trotter(terms, dt, order=1)`, `paulistrings.trotter(terms, dt, num_qubits, ...)` | a Trotter circuit, one `pauli_rotation` per term, in order; see [Circuit](circuit.md#trotter-steps) |

A site at or beyond the register size is a `ValueError` in both.

## Lattices and models

| Call | Returns |
|---|---|
| `paulistrings.lattices.chain(L, *, periodic=False)` | bonds `(j, j + 1)`, plus `(L - 1, 0)` when periodic |
| `paulistrings.lattices.square(lx, ly, *, periodic=False)` | nearest-neighbour bonds, site `(x, y)` at index `y * lx + x`, each site's right then lower neighbour |
| `paulistrings.models.transverse_ising(bonds, sites, J=1.0, h=1.0)` | `-J sum Z_j Z_j' - h sum X_j`, bonds first |
| `paulistrings.models.xxz(bonds, J=1.0, Jz=1.0, sites=(), h=0.0)` | `J sum (X_j X_j' + Y_j Y_j') + Jz sum Z_j Z_j' - h sum Z_j`, ordered XX, YY, ZZ, fields |

A periodic lattice with a side shorter than 3 is a `ValueError`.
The models leave out any group whose coupling is exactly zero.

See [Sparse terms](../manual/operators.md#sparse-terms) for worked examples.