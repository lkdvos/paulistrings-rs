# Showcases

Five measured applications, each one a physics question the engine was pointed
at rather than a demonstration written around a known answer. Every one carries
an independent cross-check (a dense reference computed by a route that shares no
code with the engine) and a truncation-convergence panel, and every one says
where its converged window ends.

| | what it shows | independent check | cost |
|---|---|---|---|
| [B1 Operator scrambling](b1-operator-scrambling.md) | support growth, light cones, OTOCs and butterfly velocity — 1D chain, then a 2D quench, then a measured 3D cost projection | dense `2ⁿ×2ⁿ` Kronecker construction; worst gap 5.8·10⁻¹⁵ (1D) / 2.1·10⁻¹⁴ (2D) | minutes (1D), ~15 min (2D), tens of GB |
| [B2 Noisy circuit verification](b2-noisy-verification.md) | on a 127-qubit kicked-Ising circuit, **noise makes the simulation cheaper**: 651× fewer peak terms and 1078× less wall time at `p = 3e-2` | hand-rolled Kraus density-matrix evolution, all five channels, both directions, `1e-10` | 25.7 min single-threaded |
| [B5 Operator backpropagation](b5-operator-backpropagation.md) | hybrid depth reduction: back-propagate the tail classically, hand a QPU the shorter front circuit and an evolved observable | qiskit-Aer statevector, gap 1.7·10⁻¹⁶; task file round-trip gap exactly `0.0` | ~3 s |
| [B6 Resource probes](b6-resource-probes.md) | difficulty of the evolved operator: Pauli-spectrum entropy (the cost model for *this* engine) against operator entanglement (the cost model for MPO methods) | brute force over all `4ⁿ` traces and a dense SVD; every gap ≤ 8.9·10⁻¹⁶ | under a minute |
| [B7 Stabilizer-prep](b7-stabilizer-prep.md) | stim prepares a 36-qubit 2D cluster state, a non-Clifford tail is propagated, and the expectation is contracted against the stabilizer state in `O(m·n²/64)`, avoiding a 1.0 TiB state vector | dense statevector and a projector from the generators alone at `n ≤ 12`, plus qiskit Aer; worst gap 2.2·10⁻¹⁵ | 116 s, 10.2 GiB peak RSS (`--quick`: 40 s, 1.5 GiB) |

B3 (variational pre-training) and B4 (QML/QCNN) are not part of this suite.

## What every showcase page carries

A convergence panel on every truncated result: a single number from a single
cutoff is not a result here. The retained Hilbert–Schmidt norm `N = Σ|c_P|²`
alongside it, conserved under exact unitary evolution and equal to 1 for a
single Pauli seed, so `1 − N` is exactly the deleted fraction of the operator
under truncation. Named dependencies rather than silent approximations: where
a reference was not reachable, the page says so and what it would cost.

## Reproducing any of them

```bash
./scripts/setup.sh
source .venv/bin/activate
pip install -e ".[examples]"
maturin develop --release -m crates/paulistrings-py/Cargo.toml

RAYON_NUM_THREADS=1 python examples/b1_operator_scrambling/run_b1_1d.py
RAYON_NUM_THREADS=1 python examples/b2_noisy_verification/run_b2.py
RAYON_NUM_THREADS=1 python examples/b5_operator_backpropagation/run_b5.py
RAYON_NUM_THREADS=1 python examples/b6_resource_probes/run_b6.py
RAYON_NUM_THREADS=1 python examples/b7_stabilizer_prep/run_b7.py
```

Each script rewrites every figure and JSON file next to itself. Each showcase
also has a CI-visible correctness gate under `python/paulistrings/tests/` that
runs in about a second on numpy alone — the physics is checked on every commit
even though the full runs are manual.

**Source:**
[`examples/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/README.md).
