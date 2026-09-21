# Explanation

Background and rationale, read occasionally rather than searched. See [How-to guides](../how-to/index.md) for recipes and [Reference](../reference/index.md) for signatures and tables.

| Page | Covers |
|---|---|
| [Truncation](truncation.md) | what truncation actually does to the result: per-channel application, no variational bound |
| [Propagation engine](propagation-engine.md) | the bucketed layout, the GF(2)-linear hash, coset parallelism, and why the loop needs no locks |
| [Performance](performance.md) | measured numbers behind the engine design's claims |
| [NUMA nodes](numa.md) | the in-process partitioned engine, when it helps |
| [MPI ranks](mpi.md) | the distributed engine, one partition per rank |
| [Comparisons](comparisons.md) | how this engine's numbers relate to other simulators |
| [Case studies](case-studies/index.md) | measured showcases and benchmarks, each a record with independent cross-checks |
