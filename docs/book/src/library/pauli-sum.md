# PauliSum

## Constructors

| Signature | Builds |
|---|---|
| `PauliSum(num_qubits)` | the empty sum |
| `PauliSum.from_strings(terms, *, num_qubits=None)` | from a `{pauli_string: coefficient}` dict; each key is `num_qubits` characters of `I/X/Y/Z`, index `i` addresses qubit `i` |
| `PauliSum.from_strings(labels, coefficients, *, num_qubits=None)` | the same, from two equal-length sequences instead of a dict; a label repeated in `labels` accumulates rather than raising |
| `PauliSum.from_arrays(x, z, coefficients, num_qubits)` | from raw symplectic arrays — the inverse of `x_array`/`z_array`/`coefficients_array` |

`num_qubits` is inferred from the first label's length when omitted from either `from_strings` form; an empty `terms`/`labels` with no explicit `num_qubits` is a `ValueError`, since there is nothing to infer it from.
`from_strings` and `from_arrays` both use the crate's Hermitian convention: a coefficient multiplies the literal Pauli string, and `Y` carries no phase of its own.
Duplicate keys/rows/labels sum their coefficients; exact-zero coefficients are dropped.

`from_arrays` parameters:

| Parameter | Type | Notes |
|---|---|---|
| `x`, `z` | `uint64` array, shape `(n_terms, w)` | symplectic key words, `1 <= w <= ` the compile-time width tier `num_qubits` picks; narrower than the tier is zero-padded |
| `coefficients` | 1-D array, length `n_terms` | `complex128` or a real-float dtype |
| `num_qubits` | `int` | a set bit at or beyond this qubit is a `ValueError` |

## Accessors

| Call | Returns |
|---|---|
| `.num_qubits` | qubit count |
| `.width` | active compile-time width tier (words per term) |
| `len(sum)` | term count |
| `.num_buckets` | current bucket count (grow-only, reflects the last `propagate`/`rebucket`) |
| `.coefficients()` | coefficient column as a list of Python `complex` |
| `.coefficients_array()` | coefficient column as a 1-D `complex128` NumPy array |
| `.x_array()` | X-part column, 2-D `uint64` array, shape `(len, width)` |
| `.z_array()` | Z-part column, 2-D `uint64` array, shape `(len, width)` |
| `str(sum)`, `repr(sum)` | `coefficient*label` for the first few terms, `+`-joined, `... (N more terms)` past that; `0` for the empty sum |

All four array/list accessors return the sum's canonical order (partition-bucket index ascending, then lexicographic `(x, z)`), consistently across calls.
`str`/`repr` show that same order's *prefix*, never sorted by coefficient magnitude — a magnitude sort would cost `O(len log len)` just to print a preview, on a type whose whole point is staying cheap at a huge term count.

## Arithmetic

| Call | Returns |
|---|---|
| `a + b`, `a - b` | a new sum, coefficients combined on matching strings and the rest kept |
| `a += b`, `a -= b` | the same merge, in place |
| `a * c`, `c * a`, `a *= c` | every coefficient scaled by a complex or real `c` |

Both operands of `+`/`-` must have the same `num_qubits`; anything else is a `ValueError`.
Terms whose coefficients cancel exactly are dropped, as is every term when `c` is exactly zero.
`*` is scalar-only: multiplying two sums is a full operator product, which this library does not implement.

## Saving, loading, importing

| Call | Does |
|---|---|
| `paulistrings.io.save(path, pauli_sum)` | write as `.npz` (`paulistrings-npz-v1` format) |
| `paulistrings.io.load(path)` | read back a `PauliSum` |
| `paulistrings.interop.load_task(path)` | parse a schema-v1 task JSON file into a `Task`, whose `.observable` is a `PauliSum` when the task defines one |

See [Saving and loading](../manual/operators.md#saving-and-loading) for the recipe.
