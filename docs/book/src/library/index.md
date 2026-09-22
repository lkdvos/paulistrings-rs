# Library

Signatures and tables for the Python API. Usage and rationale are in the [Manual](../manual/index.md).

| Page | Covers |
|---|---|
| [PauliString](pauli-string.md) | constructors, accessors, commutator/product operations |
| [PauliSum](pauli-sum.md) | constructors, accessors, save/load/import pointers |
| [Circuit](circuit.md) | gate/noise methods, composition, one-gate-per-channel and rotation-angle conventions |
| [propagate / propagate_with_stats](propagate.md) | full signatures, engine/bucket/partition knobs, stats field tables |
| [Truncation policies](truncation.md) | `coeff`/`weight`/`topn`/`approx_topn`, `&`/`\|` combinators |
| [Direction semantics](direction.md) | `"forward"`/`"heisenberg"`, push-order note |
| [Measurement](measurement.md) | `expectation`, `expectation_stabilizer`, `overlap`, `identity_coefficient` |
| [Module helpers](module-helpers.md) | `numa_nodes`, `mpi_available`, `DEFAULT_SMALL_SUM_THRESHOLD`, `reset_log_cache` |
