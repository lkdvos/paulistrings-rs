# Showcases

Four measured applications, each one a physics question the engine was pointed
at rather than a demonstration written around a known answer. Every one carries
an independent cross-check (a dense reference computed by a route that shares no
code with the engine) and a truncation-convergence panel, and every one says
where its converged window ends.

| | what it shows | independent check | cost |
|---|---|---|---|
| [**B1** Operator scrambling](b1-operator-scrambling.md) | support growth, light cones, OTOCs and butterfly velocity — 1D chain, then a 2D quench, then a measured 3D cost projection | dense `2ⁿ×2ⁿ` Kronecker construction; worst gap 5.8·10⁻¹⁵ (1D) / 2.1·10⁻¹⁴ (2D) | minutes (1D), ~15 min (2D), tens of GB |
| [**B2** Noisy verification](b2-noisy-verification.md) | on a 127-qubit kicked-Ising circuit, **noise makes the simulation cheaper**: 651× fewer peak terms and 1078× less wall time at `p = 3e-2` | hand-rolled Kraus density-matrix evolution, all five channels, both directions, `1e-10` | 25.7 min single-threaded |
| [**B5** Operator backpropagation](b5-operator-backpropagation.md) | hybrid depth reduction: back-propagate the tail classically, hand a QPU the shorter front circuit and an evolved observable | qiskit-Aer statevector, gap 1.7·10⁻¹⁶; task file round-trip gap exactly `0.0` | ~3 s |
| [**B6** Resource probes](b6-resource-probes.md) | how hard is the evolved operator? Pauli-spectrum entropy (the cost model for *this* engine) against operator entanglement (the cost model for MPO methods) | brute force over all `4ⁿ` traces and a dense SVD; every gap ≤ 8.9·10⁻¹⁶ | under a minute |

B3 (variational pre-training), B4 (QML/QCNN) and B7 (stabilizer-prep →
PP-estimate) are design stubs, blocked on a symbolic-coefficient core redesign
or on stabilizer-membership contraction. They are named as absent rather than
approximated.

## What every showcase page is required to carry

These are the suite's own global rules, not editorial preference:

- **A convergence panel on every truncated result.** A single number from a
  single cutoff is not a result here.
- **The retained Hilbert–Schmidt norm `N = Σ|c_P|²` alongside it.** Under exact
  unitary evolution `N` is conserved and equals 1 for a single Pauli seed; under
  truncation it only falls, and `1 − N` is exactly the fraction of the operator
  that was deleted. It is the one diagnostic that says a curve has gone
  meaningless.
- **Named dependencies, never silent approximations.** Where a reference was not
  reachable, the page says so and says what it would cost.
- **Recorded cuts.** Where a grid was shortened to fit a time box, the pilot
  measurement and the projection that justified the cut are in the driver and on
  the page.

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
```

Each script rewrites every figure and JSON file next to itself. Each showcase
also has a CI-visible correctness gate under `python/paulistrings/tests/` that
runs in about a second on numpy alone — the physics is checked on every commit
even though the full runs are manual.

**Source:**
[`examples/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/README.md).
