# Case studies

Ten measured pieces of work — five showcases and five benchmarks — each one a
physics question or a performance question the engine was pointed at rather
than a demonstration written around a known answer. Every one carries an
independent cross-check (a dense reference computed by a route that shares no
code with the engine, or an exact oracle such as `stim`) and, where the result
is truncated, a convergence panel that says where the converged window ends.

## The flagship worked example: a 2D Ising quench {#the-2d-ising-quench}

![Average X magnetization vs time for the 2D Ising quench, 4×4 and 6×6 lattices](../../assets/ising-quench/ising_quench.svg)

The site's landing figure is a transverse-field Ising quench on a periodic 4×4 and 6×6 lattice at `J = h = 1`, first-order Trotterized at `δt = 0.05` out to `t = 2`.
The average X magnetization is Heisenberg-propagated through the 40-step circuit and read against `|+…+⟩` at every step, under `coeff(1e-10) & topn(k)` with `k` of 50 000 (4×4) and 200 000 (6×6).
The 6×6 case is `2³⁶` amplitudes, already out of reach for exact diagonalization; the quench runs in ~11–12 s on the reference host.

The honest error bar there is not floating point but the `topn` tie rule: `topn` keeps or drops a magnitude-tied symmetry orbit whole, and an alternative tiebreak that always keeps exactly `k` terms moves the trajectory by up to 1.7% (4×4) and 0.37% (6×6).
The larger lattice is the better-resolved one, because the observable averages over more sites and the truncation error self-averages.

The walkthrough is a Rust one — it is the crate's own worked example, embedded into its rustdoc — and is the hand-off point for Rust users rather than part of this book:
[`crates/paulistrings/docs/examples/ising_2d_quench.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/docs/examples/ising_2d_quench.md), with the source at [`ising_2d_quench.rs`](https://github.com/lkdvos/paulistrings-rs/blob/main/crates/paulistrings/examples/ising_2d_quench.rs).
For the same pattern in Python — one Trotter step at a time, an expectation value per step — see [Observable vs time](../../how-to/propagate-a-time-series.md).

## Showcases

Five measured applications.

| | what it shows | independent check | cost |
|---|---|---|---|
| [B1 Operator scrambling](b1-operator-scrambling.md) | support growth, light cones, OTOCs and butterfly velocity — 1D chain, then a 2D quench, then a measured 3D cost projection | dense `2ⁿ×2ⁿ` Kronecker construction; worst gap 5.8·10⁻¹⁵ (1D) / 2.1·10⁻¹⁴ (2D) | minutes (1D), ~15 min (2D), tens of GB |
| [B2 Noisy circuit verification](b2-noisy-verification.md) | on a 127-qubit kicked-Ising circuit, **noise makes the simulation cheaper**: 651× fewer peak terms and 1078× less wall time at `p = 3e-2` | hand-rolled Kraus density-matrix evolution, all five channels, both directions, `1e-10` | 25.7 min single-threaded |
| [B5 Operator backpropagation](b5-operator-backpropagation.md) | hybrid depth reduction: back-propagate the tail classically, hand a QPU the shorter front circuit and an evolved observable | qiskit-Aer statevector, gap 1.7·10⁻¹⁶; task file round-trip gap exactly `0.0` | ~3 s |
| [B6 Resource probes](b6-resource-probes.md) | difficulty of the evolved operator: Pauli-spectrum entropy (the cost model for *this* engine) against operator entanglement (the cost model for MPO methods) | brute force over all `4ⁿ` traces and a dense SVD; every gap ≤ 8.9·10⁻¹⁶ | under a minute |
| [B7 Stabilizer-prep](b7-stabilizer-prep.md) | stim prepares a 36-qubit 2D cluster state, a non-Clifford tail is propagated, and the expectation is contracted against the stabilizer state in `O(m·n²/64)`, avoiding a 1.0 TiB state vector | dense statevector and a projector from the generators alone at `n ≤ 12`, plus qiskit Aer; worst gap 2.2·10⁻¹⁵ | 116 s, 10.2 GiB peak RSS (`--quick`: 40 s, 1.5 GiB) |

B3 (variational pre-training) and B4 (QML/QCNN) are not part of this suite.

### Contents common to every showcase page

A convergence panel on every truncated result: a single number from a single
cutoff is not a result here. The retained Hilbert–Schmidt norm `N = Σ|c_P|²`
alongside it, conserved under exact unitary evolution and equal to 1 for a
single Pauli seed, so `1 − N` is exactly the deleted fraction of the operator
under truncation. Named dependencies rather than silent approximations: where
a reference was not reachable, the page says so and what it would cost.
[Validate a result](../../how-to/validate-a-result.md) is that panel as a recipe, for your own runs.

### Reproducing a showcase

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
also has a CI-visible correctness gate under `examples/tests/` that
runs in about a second on numpy alone — the physics is checked on every commit
even though the full runs are manual.

**Source:**
[`examples/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/examples/README.md).

## Benchmarks

Five benchmarks, each with a setup, an oracle, and a result.
Two of the five results are negative.

| | setup | oracle | headline result |
|---|---|---|---|
| [A Clifford point](a-clifford.md) | 127-qubit heavy-hex kicked Ising at `θ_h = π/2`, weight-10 and weight-17 published observables | `stim` — exact ±1 at the Clifford point | exact integers reproduced **bit-exactly at every cutoff**; per-layer parity with `PauliPropagation.jl` on all 1355 layers before any timing was allowed |
| [B Kick-angle sweep](b-theta-sweep.md) | same circuit, 5 Trotter steps, six kick angles, three observables | causal-cone exact (19 q and 30 q cones), self-converged at 59 q | the accuracy target met in milliseconds where the cone is small; **8/8** self-convergence estimates conservative where an exact answer exists; 9/9 cross-engine parity on 12 195 per-layer counts |
| [C Deep Trotter](c-deep-trotter.md) | same circuit, `Z_62`, depth ladder 5/9/15/20 steps, dyadic cutoffs | exact 19-qubit cone at 5 steps; self-converged beyond | the headline is a **reachability boundary**: 0.01 accuracy in 0.11 s at 5 steps; at 15–20 steps in the hard interior neither the target nor a reference to score it against is reachable |
| [D XXZ chain](d-xxz-chain.md) | Trotterized XXZ chain, `n = 20…100`, free and interacting regimes | statevector at `n ≤ 26`, plus an *analytic* growth law | quadratic term growth confirmed as **exactly `16s²`**; the cross-engine ranking changes sign between 3·10³ and 3·10⁴ terms |
| [E Random SU(4) brickwork](e-su4-brickwork.md) | 36 qubits, an independent Haar-random SU(4) block per brickwork site | statevector at `n ≤ 24` | the generic worst case: no Clifford structure, no light-cone shortcut. Rise, plateau, then **collapse to zero terms**; the two engines within noise of each other |

### Comparability rules

Four of them are worth stating up front, because they are what makes the tables
comparable:

1. One gate per channel, everywhere. Truncation is applied after every
   channel, so fusing two gates into one channel changes the answer. Every
   circuit in the suite is built one gate per `Circuit` method call, which is
   also what makes a *per-layer* comparison against another engine meaningful.
2. Term-count parity blocks timing. No cross-engine wall time is reported for
   a configuration whose evolved Pauli sums diverge term-for-term at matched
   truncation. The parity gate runs first, untimed, and compares every
   per-layer count in application order, not just the final one — a divergence
   that cancels by the end is exactly the bug the check exists to catch.
3. Single-threaded, warm, with input generation outside the timed region.
   `RAYON_NUM_THREADS=1` exported before the interpreter starts. References are
   exempt: an oracle is not a timing measurement, so reference sweeps are allowed
   threads (and are run in a spawned child, which also confines qiskit-aer's
   persistent OpenMP pool).
4. Every truncated result ships with a convergence panel, and a
   self-converged reference may only be quoted if its plateau test passed *and*
   its reported uncertainty is inside half the accuracy bar. Rows failing either
   test are printed as `not claimable` and no value is quoted from them.

### The plateau criterion

The obvious self-convergence test is "tighten the cutoff until two successive
values agree to `tol`". That test is wrong here, and Benchmark B caught it.
Run against an *exact* reference at a small kick angle, it declared convergence
with an estimated uncertainty of **exactly zero** while the value was still
5.6·10⁻⁷ from the truth, because at a small kick angle the only terms
contributing to `⟨0|O|0⟩` are those rotated all the way to pure `Z`, so
loosening the cutoff by a decade admits thousands of new terms *none of which is
pure `Z`*, and the expectation does not move at all while the sum keeps growing.
An exactly-zero difference there means "no relevant term has arrived yet".

So the criterion in force requires the two small successive differences and
one of: the term count has stopped growing (the sum has saturated, and the
plateau is the exact answer), or both differences are strictly nonzero (an
ordinary slowly-converging series). A flat value with a still-growing sum is
rejected. A sum truncated to zero terms is rejected outright, however flat it
looks. The fix is worth a measured **190×** in accuracy.

Benchmarks C and B2 import that criterion as a function object rather than
re-implementing it, and a test asserts it is the same object.
[Validate a result](../../how-to/validate-a-result.md) states it as user guidance, with the sweep that feeds it.

### Reproducing a benchmark

```bash
./scripts/setup.sh && source .venv/bin/activate
pip install -e ".[examples,bench]"
maturin develop --release -m crates/paulistrings-py/Cargo.toml

RAYON_NUM_THREADS=1 pytest benchmarks/python/bench_a_clifford.py --benchmark-only
RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py --validate-convergence
RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py --validate-convergence
RAYON_NUM_THREADS=1 python examples/xxz_chain/run_benchmark_d.py all
RAYON_NUM_THREADS=1 python benchmarks/python/bench_e_su4.py
```

None of these is in CI. Each has a CI-safe correctness gate at smaller scale
under `benchmarks/python/tests/`, so the physics is checked on every commit.

There is a further benchmark surface this section does not cover: the cross-*library*
construction/conjugation comparison against `qiskit.SparsePauliOp` and
`openfermion.QubitOperator` in `benchmarks/python/bench_baseline.py`.

### Caveat for every benchmark page

**Wall times are indicative of shape, not campaign-grade.** They were taken on a
shared workstation (Intel Xeon Gold 6244 @ 3.60 GHz, `ccqlin038`) whose stated
single-thread run-to-run noise is ±5–8% — and concurrent load was heavier than
that at times: the same configuration in Benchmark B measured 13.8 s and 28.6 s
in two probes minutes apart. Term counts, expectation values, parity outcomes and
convergence verdicts are load-independent, and those are the numbers to quote.
Anything under ~10% needs `scripts/ab-compare.sh` (two prebuilt binaries
alternated adjacent in time, paired per-run deltas, acceptance by direction
consistency across every pair), not these tables.

**Sources:**
[`benchmarks/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/README.md)
and the per-benchmark READMEs linked from each page.
