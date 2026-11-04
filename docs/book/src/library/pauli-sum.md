# PauliSum

## Constructors

| Signature | Builds |
|---|---|
| `PauliSum(num_qubits, *, dtype=None)` | the empty sum |
| `PauliSum.from_strings(terms, *, num_qubits=None, dtype=None)` | from a `{pauli_string: coefficient}` dict; each key is `num_qubits` characters of `I/X/Y/Z`, index `i` addresses qubit `i` |
| `PauliSum.from_strings(labels, coefficients, *, num_qubits=None, dtype=None)` | the same, from two equal-length sequences instead of a dict; a label repeated in `labels` accumulates rather than raising |
| `PauliSum.from_arrays(x, z, coefficients, num_qubits, *, dtype=None)` | from raw symplectic arrays — the inverse of `to_arrays` |

`dtype` is the [coefficient type](../manual/operators.md#dtypes) in any spelling `numpy.dtype` accepts: `float64`, `float32`, `complex128` (the default for `None`) or `complex64`, anything else a `TypeError`.
A real dtype raises `ValueError` on a coefficient with a nonzero imaginary part.
`num_qubits` is inferred from the first label's length when omitted from either `from_strings` form; an empty `terms`/`labels` with no explicit `num_qubits` is a `ValueError`, since there is nothing to infer it from.
`from_strings` and `from_arrays` both use the crate's Hermitian convention: a coefficient multiplies the literal Pauli string, and `Y` carries no phase of its own.
Duplicate keys/rows/labels sum their coefficients; exact-zero coefficients are dropped.

`from_arrays` parameters:

| Parameter | Type | Notes |
|---|---|---|
| `x`, `z` | `uint64` array, shape `(n_terms, w)` | symplectic key words, `1 <= w <= ` the compile-time width tier `num_qubits` picks; narrower than the tier is zero-padded |
| `coefficients` | 1-D array, length `n_terms` | `float64`, `float32`, `complex128` or `complex64`, converted to the sum's `dtype` |
| `num_qubits` | `int` | a set bit at or beyond this qubit is a `ValueError` |

## Accessors

| Call | Returns |
|---|---|
| `.num_qubits` | qubit count |
| `.dtype` | coefficient type, a `numpy.dtype` |
| `.copy()`, `copy.copy(sum)`, `copy.deepcopy(sum)` | an independent copy; [`propagated`](propagate.md) propagates one |
| `.astype(dtype)` | a copy at another coefficient type; `ValueError` when going real would drop a nonzero imaginary part |
| `.width` | active compile-time width tier (words per term) |
| `len(sum)` | term count |
| `.num_buckets` | current bucket count, as the last `propagate` left it |
| `.hash_seed` | the seed the bucket hash's drawn rows came from |
| `.hash_rows()` | the bucket hash's GF(2) rows, a `bool` array of shape `(min(2 * num_qubits, 31), 2 * num_qubits)`, x-bits first; see [`paulistrings.hashes`](propagate.md#paulistringshashes) |
| `.bucket_lengths()` | terms per bucket, a `uint64` array in bucket order |
| `.with_hash(rows=None, seed=None)` | a copy bucketed by another hash: the given rows lead, dense rows drawn from `seed` (default: this sum's) fill the rest; dependent rows are a `RuntimeWarning` |
| `.partition(partitions=None, ..., comm=None)` | a [`PartitionedPauliSum`](propagate.md#partitionedpaulisum) that stays split between calls, over NUMA partitions or MPI ranks, so its partition and bucket rows can change mid-circuit |
| `.coefficients()` | coefficient column as a list of Python `float` for a real dtype, `complex` otherwise |
| `.coefficients_array()` | coefficient column as a 1-D NumPy array of the sum's dtype |
| `.x_array()` | X-part column, 2-D `uint64` array, shape `(len, width)` |
| `.z_array()` | Z-part column, 2-D `uint64` array, shape `(len, width)` |
| `.to_arrays()` | `(x_array(), z_array(), coefficients_array())` from one snapshot, the inverse of `from_arrays`; `PartitionedPauliSum.local()` gives a rank's own terms |
| `str(sum)`, `repr(sum)` | `coefficient*label` for the first few terms, `+`-joined, `... (N more terms)` past that; `0` for the empty sum |

All five array/list accessors return the sum's canonical order (partition-bucket index ascending, then lexicographic `(x, z)`), consistently across calls.
`str`/`repr` show that same order's *prefix*, never sorted by coefficient magnitude — a magnitude sort would cost `O(len log len)` just to print a preview, on a type whose whole point is staying cheap at a huge term count.

## Arithmetic

| Call | Returns |
|---|---|
| `a + b`, `a - b` | a new sum, coefficients combined on matching strings and the rest kept, at NumPy's promotion of the two dtypes |
| `a += b`, `a -= b` | the same merge and dtype, in place |
| `a * c`, `c * a`, `a *= c` | every coefficient scaled by a complex or real `c`, at `a`'s dtype |

Both operands of `+`/`-` must have the same `num_qubits`; anything else is a `ValueError`.
A real-dtype `a` raises `ValueError` on a `c` with a nonzero imaginary part.
Terms whose coefficients cancel exactly are dropped, as is every term when `c` is exactly zero.
`*` is scalar-only: multiplying two sums is a full operator product, which this library does not implement.

## Saving, loading, importing

| Call | Does |
|---|---|
| `paulistrings.io.save(path, pauli_sum)` | write as `.npz` (`paulistrings-npz-v1` format) |
| `paulistrings.io.load(path)` | read back a `PauliSum` |
| `paulistrings.interop.load_task(path)` | parse a schema-v1 task JSON file into a `Task`, whose `.observable` is a `PauliSum` when the task defines one |

See [Saving and loading](../manual/operators.md#saving-and-loading) for the recipe.
