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

## `anticommute_histogram(sites, axis="x", comm=None)`

`w[n]`, the total `|c|**2` of the strings that anticommute with exactly `n` of the generators `G_q`, `q in sites`, with `G = X` (`axis="x"`) or `Z` (`axis="z"`); `len(sites) + 1` entries summing to `sum(|c|**2)`.
`paulistrings.diagonal_echo(w, delta)` turns it into the diagonal echo `sum_n w[n] cos(2 delta)**n / sum_n w[n]`.

## `rotated_overlap(sites, delta, axis="x", comm=None)`

The exact operator Loschmidt echo `2**-n Tr(A V^dag A V)`, `V = prod_{q in sites} exp(-i delta G_q)`, without materializing `V^dag A V`; `sum(|c|**2)` at `delta = 0`.
Cost is dominated by the largest class of strings equal away from the flipped bits on `sites`.

Both read-outs raise `ValueError` for an axis other than `"x"`/`"z"` or a repeated or out-of-range site.

### Under `comm=`

On a `propagate(..., comm=comm)` result both are collective and return the whole distributed sum's value on every rank.
`rotated_overlap` also needs the result's partition rows to read none of the coordinates `V` flips (x-bits of `sites` for `axis="x"`, z-bits for `"z"`), and raises `ValueError` on every rank otherwise:

| Propagated with | `axis="x"` | `axis="z"` |
|---|---|---|
| `partition_row_exclude={"x": sites}` | yes | if the rows happen to avoid |
| `partition_row_exclude={"z": sites}` | if the rows happen to avoid | yes |
| `partition_row_blocks=...` | yes (cut rows read only z-bits) | if the rows happen to avoid |
| `result="gather"` | yes | yes |
| default seeded rows | if the rows happen to avoid | if the rows happen to avoid |

A sum that was not a `comm=` result, or was since added to another, raises `ValueError` under `comm=`; scaling keeps the split.

## `propagate_with_stats`

`propagate_with_stats(...)` returns `(evolved, PropagationStats)`; see [propagate / propagate_with_stats](propagate.md) for the full signature and stats fields.
