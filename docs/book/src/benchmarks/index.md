# Benchmarks

<p class="lead">Five benchmarks, each with a setup, an oracle, and a result.
Two of the five results are negative.</p>

| | setup | oracle | headline result |
|---|---|---|---|
| [A Clifford point](a-clifford.md) | 127-qubit heavy-hex kicked Ising at `θ_h = π/2`, weight-10 and weight-17 published observables | `stim` — exact ±1 at the Clifford point | exact integers reproduced **bit-exactly at every cutoff**; per-layer parity with `PauliPropagation.jl` on all 1355 layers before any timing was allowed |
| [B Kick-angle sweep](b-theta-sweep.md) | same circuit, 5 Trotter steps, six kick angles, three observables | causal-cone exact (19 q and 30 q cones), self-converged at 59 q | the accuracy target met in milliseconds where the cone is small; **8/8** self-convergence estimates conservative where an exact answer exists; 9/9 cross-engine parity on 12 195 per-layer counts |
| [C Deep Trotter](c-deep-trotter.md) | same circuit, `Z_62`, depth ladder 5/9/15/20 steps, dyadic cutoffs | exact 19-qubit cone at 5 steps; self-converged beyond | the headline is a **reachability boundary**: 0.01 accuracy in 0.11 s at 5 steps; at 15–20 steps in the hard interior neither the target nor a reference to score it against is reachable |
| [D XXZ chain](d-xxz-chain.md) | Trotterized XXZ chain, `n = 20…100`, free and interacting regimes | statevector at `n ≤ 26`, plus an *analytic* growth law | quadratic term growth confirmed as **exactly `16s²`**; the cross-engine ranking changes sign between 3·10³ and 3·10⁴ terms |
| [E Random SU(4) brickwork](e-su4-brickwork.md) | 36 qubits, an independent Haar-random SU(4) block per brickwork site | statevector at `n ≤ 24` | the generic worst case: no Clifford structure, no light-cone shortcut. Rise, plateau, then **collapse to zero terms**; the two engines within noise of each other |

## The rules these ran under

Four of them are worth stating up front, because they are what makes the tables
comparable:

1. One gate per channel, everywhere. Truncation is applied after every
   channel, so fusing two gates into one channel changes the answer. Every
   circuit in the suite is built one gate per `Circuit.push`, which is also
   what makes a *per-layer* comparison against another engine meaningful.
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

## The plateau criterion

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

## Reproducing

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
under `python/paulistrings/tests/`, so the physics is checked on every commit.

There are two further benchmark surfaces this section does not cover: the Rust
criterion microbenchmarks (`cargo bench -p paulistrings`, for tight inner-loop
work — multiplication, commutator, weight, hashing) and the cross-*library*
construction/conjugation comparison against `qiskit.SparsePauliOp` and
`openfermion.QubitOperator` in `benchmarks/python/bench_baseline.py`.

## Caveat that applies to every page in this section

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
