# How-to guides

Short, goal-oriented recipes. Each assumes the [Tutorial](../tutorial/index.md)'s basics.

- [Observable from Pauli strings](build-observable-from-strings.md) — `PauliSum.from_strings`.
- [Observable from raw arrays](build-observable-from-arrays.md) — `from_arrays` and the array accessors.
- [Observable from a Hamiltonian](build-observable-from-hamiltonian.md) — accumulating a weighted sum of Pauli strings.
- [PauliSum save and load](save-load-pauli-sum.md) — the `paulistrings.io` `.npz` format.
- [Circuit noise](add-noise-to-a-circuit.md) — the noise channels and the `noise` factories.
- [stim and qiskit circuit import](import-circuit-from-stim-or-qiskit.md) — `interop.circuit_from_stim`/`circuit_from_qiskit`/`circuit_from_json`.
- [Truncation policy](choose-a-truncation-policy.md) — which of `coeff`/`weight`/`topn` to reach for.
- [Validate a result](validate-a-result.md) — the cutoff sweep, the retained norm, and when a value may be quoted.
- [Propagation direction](choose-propagation-direction.md) — Heisenberg vs forward.
- [Expectation values](compute-expectation-values.md) — `expectation`, `expectation_stabilizer`, `overlap`.
- [Observable vs time](propagate-a-time-series.md) — a Trotterized time series in one incremental pass.
- [Propagation stats and logs](read-propagation-stats-and-logs.md) — `propagate_with_stats` and the `log` facade.
- [NUMA partitions](run-on-numa-partitions.md) — `partitions=` and `RAYON_NUM_THREADS`.
- [MPI ranks](run-under-mpi.md) — `comm=` and the distributed engine.
