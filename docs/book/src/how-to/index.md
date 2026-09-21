# How-to guides

Short, goal-oriented recipes. Each assumes the [Tutorial](../tutorial/index.md)'s basics.

- [Build an observable from Pauli strings](build-observable-from-strings.md) — `PauliSum.from_strings`.
- [Build an observable from raw arrays](build-observable-from-arrays.md) — `from_arrays` and the array accessors.
- [Save and load a PauliSum](save-load-pauli-sum.md) — the `paulistrings.io` `.npz` format.
- [Add noise to a circuit](add-noise-to-a-circuit.md) — the noise channels and the `noise` factories.
- [Import a circuit from stim or qiskit](import-circuit-from-stim-or-qiskit.md) — `interop.circuit_from_stim`/`circuit_from_qiskit`/`circuit_from_json`.
- [Choose a truncation policy](choose-a-truncation-policy.md) — which of `coeff`/`weight`/`topn` to reach for.
- [Choose a propagation direction](choose-propagation-direction.md) — Heisenberg vs forward.
- [Compute expectation values](compute-expectation-values.md) — `expectation`, `expectation_stabilizer`, `overlap`.
- [Read propagation stats and logs](read-propagation-stats-and-logs.md) — `propagate_with_stats` and the `log` facade.
- [Run across NUMA partitions](run-on-numa-partitions.md) — `partitions=` and `RAYON_NUM_THREADS`.
- [Run under MPI](run-under-mpi.md) — `comm=` and the distributed engine.
