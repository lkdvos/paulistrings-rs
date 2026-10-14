# Library

Signatures and tables for the Python API. Usage and rationale are in the [Manual](../manual/index.md).

| Page | Covers |
|---|---|
| [PauliString](pauli-string.md) | constructors, accessors, commutator/product operations |
| [PauliSum](pauli-sum.md) | constructors, accessors, save/load/import pointers |
| [PauliTerms](pauli-terms.md) | `pt`, sparse terms, binding, lattices and models |
| [Circuit](circuit.md) | gate/noise methods, broadcasting, Trotter steps, composition, conventions |
| [propagate / propagate_with_stats](propagate.md) | full signatures, engine/bucket/partition knobs, stats field tables |
| [Truncation policies](truncation.md) | `coeff`/`weight`/`topn`/`approx_topn`/`collapse_sample`, `&`/`\|` combinators |
| [Direction semantics](direction.md) | `"forward"`/`"heisenberg"`, push-order note |
| [Measurement](measurement.md) | `expectation`, `expectation_stabilizer`, `overlap`, `identity_coefficient`, `anticommute_histogram`, `rotated_overlap` |
| [Module helpers](module-helpers.md) | `p`, `pt`, `trotter`, `numa_nodes`, `mpi_available`, `DEFAULT_SMALL_SUM_THRESHOLD`, `reset_log_cache`, `diagonal_echo` |