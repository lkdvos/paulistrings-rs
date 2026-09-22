# PauliString

One `IXYZ` string in the symplectic encoding, with no coefficient attached.
Immutable and hashable; `str` and `repr` both print the label.

## Constructors

| Signature | Builds |
|---|---|
| `PauliString.identity(num_qubits)` | the all-`I` string |
| `PauliString.x(qubit, num_qubits)` | `X` on `qubit`, identity elsewhere |
| `PauliString.y(qubit, num_qubits)` | `Y` on `qubit`, identity elsewhere |
| `PauliString.z(qubit, num_qubits)` | `Z` on `qubit`, identity elsewhere |
| `PauliString.from_label(label)` | from its `IXYZ` label; `num_qubits` is the label's length |
| `paulistrings.p(label)` | shorthand for `from_label` |

Character `i` addresses qubit `i`, as in [`PauliSum.from_strings`](pauli-sum.md#constructors), and `Y` is the `(x=1, z=1)` key with no phase of its own.
A `qubit` at or beyond `num_qubits` is a `ValueError`, as is a character outside `IXYZ`.

## Accessors

| Call | Returns |
|---|---|
| `.num_qubits` | qubit count |
| `.weight` | number of non-identity factors |
| `.label` | the `IXYZ` string |
| `str(p)`, `repr(p)` | the label |
| `p == q` | equality on `(label, num_qubits)` |

## Operations

| Call | Returns |
|---|---|
| `.commutes_with(other)` | `bool` |
| `.anticommutes_with(other)` | `not commutes_with(other)` |
| `.mul(other)` | `(coefficient, product)`, the coefficient being the `i^k` phase |
| `.commutator(other)` | `(coefficient, product)`; `2·mul`'s coefficient when the two anticommute, `0` otherwise |
| `.anticommutator(other)` | the same with the two cases swapped |

All five require both operands to have the same `num_qubits`; anything else is a `ValueError`.
The product string is `mul`'s either way, so a `0` coefficient is an exact algebraic zero.
See [Pauli strings](../manual/operators.md#the-paulistring-type) for the encoding and worked examples.

## Arithmetic

| Call | Returns |
|---|---|
| `p * c`, `c * p` | a one-term `PauliSum`, coefficient `c` on `p` |

`c` must be a complex or real number — `p * q` between two `PauliString`s is a `TypeError` pointing at `.mul(other)`, the Pauli product with its phase, since `*` on a `PauliString` is scalar-only, matching `PauliSum`.
This is what lets a sum be built directly out of strings, e.g. `p("XYZ") * 2 + p("YZI") * 3`; see [The `PauliSum` type](../manual/operators.md#the-paulisum-type).
