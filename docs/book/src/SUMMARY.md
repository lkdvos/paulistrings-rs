# Summary

[paulistrings-rs](index.md)
[Installation](installation.md)

# Tutorial

- [Your first propagation](tutorial/index.md)

# How-to guides

- [Overview](how-to/index.md)
- [Build an observable from Pauli strings](how-to/build-observable-from-strings.md)
- [Build an observable from raw arrays](how-to/build-observable-from-arrays.md)
- [Save and load a PauliSum](how-to/save-load-pauli-sum.md)
- [Add noise to a circuit](how-to/add-noise-to-a-circuit.md)
- [Import a circuit from stim or qiskit](how-to/import-circuit-from-stim-or-qiskit.md)
- [Choose a truncation policy](how-to/choose-a-truncation-policy.md)
- [Choose a propagation direction](how-to/choose-propagation-direction.md)
- [Compute expectation values](how-to/compute-expectation-values.md)
- [Read propagation stats and logs](how-to/read-propagation-stats-and-logs.md)
- [Run across NUMA partitions](how-to/run-on-numa-partitions.md)
- [Run under MPI](how-to/run-under-mpi.md)

# Reference

- [Overview](reference/index.md)
- [PauliSum](reference/pauli-sum.md)
- [Circuit](reference/circuit.md)
- [propagate / propagate_with_stats](reference/propagate.md)
- [Truncation policies](reference/truncation.md)
- [Direction semantics](reference/direction.md)
- [Measurement](reference/measurement.md)
- [Module helpers](reference/module-helpers.md)

# Explanation

- [How it works](explanation/index.md)
- [Performance](explanation/performance.md)
- [Running across NUMA nodes](explanation/numa.md)
- [Running across MPI ranks](explanation/mpi.md)
- [Comparisons against other tools](explanation/comparisons.md)
- [Case studies](explanation/case-studies/index.md)
  - [B1 — Operator scrambling](explanation/case-studies/b1-operator-scrambling.md)
  - [B2 — Noisy circuit verification](explanation/case-studies/b2-noisy-verification.md)
  - [B5 — Hybrid depth reduction](explanation/case-studies/b5-operator-backpropagation.md)
  - [B6 — Resource probes of the evolved operator](explanation/case-studies/b6-resource-probes.md)
  - [B7 — Stabilizer-state preparation](explanation/case-studies/b7-stabilizer-prep.md)
  - [A — Clifford point](explanation/case-studies/a-clifford.md)
  - [B — Kick-angle sweep](explanation/case-studies/b-theta-sweep.md)
  - [C — Deep Trotter circuits](explanation/case-studies/c-deep-trotter.md)
  - [D — XXZ chain scaling](explanation/case-studies/d-xxz-chain.md)
  - [E — Random SU(4) brickwork](explanation/case-studies/e-su4-brickwork.md)
