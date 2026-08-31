# Benchmark E -- random SU(4) brickwork

`research/plans/2026-08-31-examples-benchmarks-suite.md` §6 Part A, row "E". Driver:
[`../bench_e_su4.py`](../bench_e_su4.py). CI-safe correctness gate:
[`python/paulistrings/tests/test_benchmark_e_su4.py`](../../../python/paulistrings/tests/test_benchmark_e_su4.py).

```
RAYON_NUM_THREADS=1 python benchmarks/python/bench_e_su4.py
```

## What this benchmark is for

Every other Part-A benchmark in this suite (A, B, C, D) is built from a physically structured
circuit -- kicked Ising, an XXZ chain -- with commuting layers, Clifford points, or both, so
truncation has real structure to exploit. Benchmark E is the deliberate opposite: `n=36` qubits,
`examples/common/circuits.py`'s `random_su4_staircase(n, depth, seed)` draws an **independent
Haar-random SU(4) block** for every brickwork site (even layers on `(0,1),(2,3),...`, odd layers on
`(1,2),(3,4),...`), observable `Z_18` (the central qubit). There is no stabilizer structure, no
commuting sublattice, no light-cone shortcut past nearest-neighbour causality -- this is the
generic worst case Pauli propagation actually faces, and every number below should be read as "how
does the engine behave with no help from problem structure," not as a favorable case.

Seed **20260831**, fixed everywhere in this file and in `bench_e_su4.py`'s `SEED` constant.

## 1. Validation against the statevector oracle

`n in {8, 12, 16, 20, 24}`, depth 3, **no truncation** (the operator's light cone at depth 3 has at
most 15 non-identity Pauli letters spread over 7 qubits either side of center, so an untruncated
propagation is cheap at any of these sizes -- 4095 terms in every case, independent of `n`, since
the light cone does not reach the boundary):

| n | oracle | engine | \|error\| |
|---|---|---|---|
| 8 | -0.0428035729 | -0.0428035729 | 1.25e-16 |
| 12 | +0.4658107269 | +0.4658107269 | 4.44e-16 |
| 16 | -0.1628731605 | -0.1628731605 | 6.66e-16 |
| 20 | -0.6075332763 | -0.6075332763 | 2.22e-16 |
| 24 | -0.1677957052 | -0.1677957052 | 2.16e-15 |

Agreement to double-precision floating-point tolerance at every size, against qiskit Aer's dense
statevector simulator -- two engines sharing no simulation code. `python/paulistrings/tests/
test_benchmark_e_su4.py` re-derives this (and the truncated case, and every uniform product state)
as a CI-visible, `importorskip`d gate.

## 2. Term-count explosion vs. depth (the headline curve)

`n=36`, `Z_18`, three `min_abs_coeff` truncations. Figure: [`term_count_vs_depth.svg`](term_count_vs_depth.svg).

There is **no Clifford savings anywhere in this circuit** -- every one of the plan's caveats about
Haar circuits applies literally. Two regimes are visible in the numbers, not just the rising half:

* **Pre-saturation growth.** At the loosest truncation practical to run here (`min_abs_coeff=1e-4`),
  peak term count goes 15 -> 253 -> 4038 -> 58079 -> 667088 -> 2296294 -> 4110454 for depth 1..7 --
  roughly an order of magnitude every one to two layers, well inside the plan's "~4^depth until
  saturation" expectation (`4^7 ~ 1.6e5` undercounts the true growth here, since a generic 2-qubit
  Haar block's adjoint action is a dense ~15x15 real orthogonal map on the traceless Pauli algebra,
  not a 4-way branch; empirically growth is faster than `4^depth`, not slower).
* **Post-saturation collapse.** At looser cutoffs (`1e-2`, `1e-3`), the *peak* plateaus quickly
  (~1300-1700 terms for `1e-2`, ~80000-110000 for `1e-3`) and then **final term count falls to
  zero**: by depth 10 (`1e-2`) or depth 20 (`1e-3`), no coefficient anywhere in the operator's
  support survives the cutoff -- the Heisenberg-evolved `Z_18` has genuinely spread its amplitude so
  thin that every individual term is below threshold. This is not the propagation "giving up"; it is
  the expected fate of a local operator diffusing through a non-integrable, non-Clifford brickwork:
  eventually `⟨Z_18(t)⟩` really is (to truncation resolution) the maximally mixed value.

| min_abs_coeff | depth | final_terms | peak_terms | time (s) |
|---|---|---|---|---|
| 1e-2 | 1..6, 8 | 15 -> 1483 (peak, depth 4) -> 37 | 15..1483 | <0.02 each |
| 1e-2 | 10, 12, 16, 20 | 0 | 1258-1719 | <0.02 each |
| 1e-3 | 1..6 | 15 -> 84836 | 15..84836 | 0.0006-0.65 |
| 1e-3 | 8, 10, 12, 16 | 32567 -> 1624 -> 9 | 78595-107864 | 1.2-1.4 |
| 1e-3 | 20 | 0 | 85174 | 1.4 |
| 1e-4 | 1..7 | 15 -> 4110454 (still rising) | 15..4110454 | 0.0006-25.0 |

**Recorded cut.** The `1e-4` grid stops at depth 7 (25.0 s; depth 6 was already 7.9 s). This
checkout (single-threaded, ccqlin038) measured the per-depth-step cost growing ~2.3-3x at this
cutoff (7.9 s -> 25.0 s from depth 6 to 7), so depth 8 alone projects to roughly a minute and depth
9-10 to several minutes -- out of proportion for a benchmark classed `manual-short` in the plan's
own runtime table. The two looser truncations already show the full rise-plateau-collapse shape
cheaply (their peak resident set is orders of magnitude smaller precisely because they prune to
resolution earlier), so the qualitative claim ("no Clifford savings, generic exponential-then-
truncation-bounded fanout") does not need the tighter grid extended further to be supported.

## 3. Error vs. runtime, fixed (n, depth), swept over truncation

`n=16`, depth=6 (statevector-checkable; deep enough that fanout is well underway). Figure:
[`error_vs_runtime.svg`](error_vs_runtime.svg). Oracle: -0.0497243601.

| min_abs_coeff | final_terms | time (s) | \|error\| |
|---|---|---|---|
| 1e-1 | 0 | 0.0002 | 4.97e-2 |
| 1e-2 | 391 | 0.008 | 8.11e-2 |
| 1e-3 | 62569 | 0.44 | 1.75e-2 |
| 1e-4 | 2257674 | 6.82 | 1.75e-3 |
| 1e-5 | 12539219 | 14.5 | 1.57e-4 |
| 1e-6 | 16251799 | 15.3 | 5.41e-6 |
| 1e-8 | 16771992 | 17.9 | 7.02e-9 |

Error falls roughly with the truncation cutoff itself (each decade in `eps` buys roughly a decade of
accuracy) until `~1e-6`, where term count and runtime saturate (adding two more decades of cutoff
tightness, `1e-6 -> 1e-8`, adds under 4% more terms and no measurable runtime, but still buys three
more decades of accuracy -- the residual error there is dominated by which *specific* low-weight
terms survive, not by raw term count). Note the **non-monotone** row at `eps=0.01`: its error
(8.1e-2) is *larger* than the untruncated-relative-to-oracle error implied by `eps=0.1` (4.97e-2) --
truncation error is not guaranteed monotone in the cutoff for a generic (non-sign-coherent) operator,
which is exactly why this curve is reported as measured, not smoothed or asserted monotone.

## 4. Time / memory vs. n, fixed depth

Depth=6, `min_abs_coeff=1e-4`, `n` from 8 to the 36-qubit headline. Figure:
[`time_memory_vs_n.svg`](time_memory_vs_n.svg). Oracle-checked (statevector) up to `n=24`;
`n in {28, 32, 36}` is unchecked and labeled `oracle_checked: false` in `results.json`'s
`extra` field -- self-converged only, per the plan's rule that every truncated claim beyond
oracle range says so plainly rather than borrowing an oracle check it didn't get.

Wall time is dominated by term count, not `n` directly (this circuit's causal structure means the
light cone from the central qubit, not `n` itself, sets how many terms accumulate by a fixed depth);
see `results.json` for the full per-point table. Every point's `peak_memory_kb_delta` reads exactly
`0.0` -- **not because these runs use no memory**, but because this driver runs everything in one
process (module docstring of `examples/common/harness.py`: `VmHWM` is a process-lifetime high-water
mark), and `error_vs_runtime()`'s own eps=1e-6/1e-8 points (16.2M-16.8M terms, run earlier in the
same process) already set a higher peak than any `size_scaling` point reaches. A size/memory figure
that needs a real per-point number would need one process per point; this driver does not do that,
and says so rather than reporting a misleadingly flat `0.0` curve as if it meant something.

## 5. Determinism

The plan's explicit requirement for this benchmark ("same seed twice -> identical term counts and
expectation to 1e-12") is asserted in `bench_e_su4.py`'s `check_determinism()` (smoke re-run,
printed above) and, authoritatively, in
`python/paulistrings/tests/test_benchmark_e_su4.py::test_same_seed_twice_gives_identical_term_counts_and_expectation`
(always runs in CI, no optional dependency) plus a negative control confirming two distinct seeds
generically disagree.

## 6. PauliPropagation.jl comparison

Schema v1's `unitary_2q` gate is exactly the SU(4) block this circuit is built from, and jl 0.8.2
defines no `_toschrodinger` method for `TransferMapGate`
(`benchmarks/julia/README.md`, "Known gaps"), so **only `direction="heisenberg"` is comparable** --
the same restriction the rest of the suite already documents for any `unitary_1q`/`unitary_2q`/noise
gate against this jl version. `min_abs_coeff=1e-4` is deliberately **non-dyadic**: the one known
coefficient-boundary divergence between the engines (`benchmarks/julia/README.md` §P3) is triggered
by *exact* dyadic coefficients landing on the cutoff, which requires a Clifford point; every
coefficient in a Haar-random SU(4) circuit is an irrational float, so the boundary case cannot arise
here regardless, and the non-dyadic choice removes it as a possible objection anyway.

**Per-layer term-count parity holds exactly**, checked before any timing was recorded (plan §7 rule
2):

| n | depth | rust final_terms | jl final_terms | parity |
|---|---|---|---|---|
| 6 (smoke) | 3 | 4051 | 4051 | exact, all layers |
| 10 (timed) | 5 | 381654 | 381654 | exact, all layers |

Both per-layer term counts (not just the final one) and the final expectation (to `1e-12`) matched
at both sizes, reusing `benchmarks/python/test_julia_parity.py`'s `compare()` -- the same gate the
rest of the suite's cross-engine comparisons use, so this reconfirms the already-documented
vocabulary parity specifically for the `unitary_2q` gates this circuit is built from, rather than
re-deriving a separate comparison mechanism.

**Timed comparison** (n=10, depth=5, single-threaded both sides, warm/JIT-excluded on the jl side):

| engine | version | time (s) | final_terms |
|---|---|---|---|
| paulistrings | (see `results.json` provenance) | 0.496 | 381654 |
| PauliPropagation.jl | 0.8.2 | 0.492 | 381654 |

The two engines are within measurement noise of each other on this point (see CLAUDE.md's noise-
floor note: ±5-8% single-threaded is expected between *any* two runs on this host, let alone two
different implementations) -- this is a parity-gated same-ballpark result, not a speed claim in
either direction. Repeated end-to-end runs of this driver during development landed rust/jl in the
0.47-0.53 s range each time, always within a few percent of each other, so the "same ballpark"
reading is not a one-off.

## Provenance

Seed 20260831 throughout. `results.json` carries the full per-run provenance block (commit, CPU,
rustc/julia/PauliPropagation.jl versions, thread count) for every record above. Single-threaded
(`RAYON_NUM_THREADS=1`) throughout; `RunRecord.provenance.thread_count` records `None` rather than a
possibly-misleading count on the runs that follow a qiskit-aer statevector call in the same process
(see `bench_e_su4.py`'s `threads=None` comment -- Aer's own thread pool defeats the harness's
process-wide thread-delta pinning heuristic once it has run; `RAYON_NUM_THREADS=1`, asserted once at
the top of the driver before anything propagates, is what actually pins the engine).
