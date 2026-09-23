# Measurement

## `expectation(state="x+")`

Expectation value in a single-qubit product state. `state` is either:

| Form | Meaning |
|---|---|
| `"x+"` / `"y+"` / `"z+"` | uniform product state, the `+1` eigenstate of that Pauli on every qubit (case-insensitive) |
| a per-qubit label string, `num_qubits` characters | one character per qubit, qiskit's `Statevector.from_label` alphabet: `0`/`1` = Z±, `+`/`-` = X±, `r`/`l` = Y±, case-sensitive |

Cost is one masked pass over the terms either way. Returns a Python `complex`; take `.real` for a Hermitian operator.

## `expectation_stabilizer(generators)`

Expectation value in a stabilizer state given by its generators.

`generators` is a list of exactly `num_qubits` signed Pauli strings — `"+XX"`, `"-ZZ"`, or bare `"ZIZ"` for `+` — same alphabet and qubit indexing as `from_strings`.
They must be pairwise commuting and independent over GF(2); anything else raises `ValueError`.

Reads any stabilizer state (Bell, GHZ, cluster, Clifford-circuit output), where `expectation` reads only product states.
Cost is `O(terms · num_qubits² / 64)` after a one-off `O(num_qubits³ / 64)` reduction of the generators.

`paulistrings.interop.stabilizers_from_stim(src, num_qubits=None)` builds the `generators` list from a stim `Tableau`/`Circuit`/`TableauSimulator`.

## `overlap(other)`

Hilbert–Schmidt overlap `tr(self* . other) / 2^n`, i.e. `sum(conj(a_i) * b_i)` over shared keys. Both sums must share `num_qubits`.

## `identity_coefficient()`

Coefficient of the `I...I` term, i.e. `tr(O) / 2^n`.

## `propagate_with_stats`

`propagate_with_stats(...)` returns `(evolved, PropagationStats)`; see [propagate / propagate_with_stats](propagate.md) for the full signature and stats fields.
