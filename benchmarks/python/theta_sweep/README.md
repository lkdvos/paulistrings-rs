# Benchmark B — the θ_h sweep at 5 Trotter steps

Part A entry **B** of `research/plans/2026-08-31-examples-benchmarks-suite.md` (§6). Heavy-hex kicked
Ising, **n = 127**, **5 Trotter steps**, `θ_zz = −π/2`, six kick angles
**θ_h ∈ {0, 0.2, π/8, π/4, 3π/8, π/2}**, three observables from Kim et al. (2023) — `Z_62` (Fig. 4b),
the weight-10 operator (Fig. 3b) and the weight-17 operator (Fig. 3c). Heisenberg picture, contracted
against `|0…0⟩`, single-threaded, warm timings, 1355 channels per circuit (one gate per channel).

| file | what it is |
|---|---|
| `../bench_b_theta_sweep.py` | the driver — also the record of every measurement that shaped it |
| `results.json` | 236 `report.RunRecord`s (227 paulistrings, 9 PauliPropagation.jl) with full provenance |
| `summary.json` | references, endpoint checks, time-to-accuracy selections, parity outcomes, cuts |
| `error-vs-min-abs-coeff.svg` | the convergence panel: \|error\| vs coefficient cutoff, one curve per θ_h |
| `error-vs-max-weight.svg` | the same for the weight cap |
| `error-vs-runtime.svg` | \|error\| vs warm wall time — the plot the benchmark exists for |
| `term-count-vs-truncation.svg` | term count vs cutoff, paulistrings against PauliPropagation.jl (the curves coincide exactly, so only one is visible) |
| `parity-per-layer-terms.svg` | all 1355 per-layer term counts, both engines (thick solid under thin dashed) |

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py --validate-convergence
```

`RAYON_NUM_THREADS=1` must be exported **before** the interpreter starts (Rayon builds its global
pool at the first propagate and never resizes it); the driver refuses to run otherwise. The CI-safe
gate for the same physics on a 20-qubit sublattice is
`python/paulistrings/tests/test_benchmark_b_sweep.py`.

Recorded run: commit `94077fa`, clean worktree, ccqlin038 (Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0,
python 3.11.11, paulistrings 0.1.0, numpy 2.4.6, qiskit 2.5.2, stim 1.16.0, julia 1.12.6 +
PauliPropagation 0.8.2. **39.6 min** end to end.

## 1. Reference strategy — what is actually reachable

The plan asked for a light-cone exact reference at every point for all three observables. It is not
reachable, and the reason is the cone size, which `oracles.light_cone` recomputes on every run
(19 / 30 / 59 qubits, the same at every θ_h). What each observable gets instead:

| observable | cone | interior θ_h reference | exact? | per point |
|---|---|---|---|---|
| `Z_62` | 19 q | `light_cone_exact(method="both")` — Aer statevector **and** untruncated Pauli propagation over the same cone, required to agree | **yes** | 15 s |
| weight-10 | 30 q | `light_cone_exact(method="statevector", max_statevector_qubits=30)` | **yes** | 142–197 s |
| weight-17 | 59 q | `self_converged_reference` — tighten `min_abs_coeff` until the plateau is real | **no** | 8–66 s |

Both endpoints (θ_h ∈ {0, π/2}) are Clifford points for every observable and get exact integers from
`oracles.stim_clifford_exact` in under 0.1 s.

### Why weight-10 uses a statevector and not Pauli propagation over the same cone

Both paths are exact, so the choice is pure cost. Measured at θ_h = 0.2, single-threaded, by growing
the applied-gate prefix of the 30-qubit reduced cone circuit (Heisenberg propagation applies gates in
reverse, so the tail of the gate list is a genuine prefix of the work):

| applied gates | peak terms | wall |
|---|---|---|
| 40 | 6.4e1 | 0.00 s |
| 80 | 5.2e4 | 0.03 s |
| 120 | 4.3e6 | 0.93 s |
| 160 | 4.3e8 | 87 s |
| 305 (the whole reduced cone) | — | aborted at a 26 GiB address-space cap |

The statevector over the same cone is **~150 s and 16.1 GiB**, and does not care about depth. So it
is the cheaper path at every θ_h, and it is what the driver uses. For `Z_62`'s 19-qubit cone the
Pauli path *is* affordable (12.8 s against 2.8 s for the statevector), which is why that observable
can afford `method="both"` and get two independent simulations behind one reference.

### Why weight-17 cannot have an exact interior reference

59 qubits rules out any dense method (`2**59 * 16` bytes). Untruncated Pauli propagation over a
59-qubit, 403-gate cone is far past the weight-10 wall above. Nothing else is exact. Hence the
self-converged reference, labelled `self_converged` in every record — `reference_exact` is `false`
and `reference_uncertainty` carries the estimate.

## 2. The self-convergence criterion, and the measured reason it is not the obvious one

The obvious criterion is "tighten the cutoff until two successive expectation values agree to `tol`".
**That criterion is wrong here, and the validation leg caught it.** Run against the *exact* `Z_62`
reference at θ_h = 0.2, it declares convergence with an estimated uncertainty of **exactly zero**
while the value is still **5.6e-7** from the truth:

```
eps=1e-03  <O>=+0.981076800  Δ=  —        terms=     59
eps=1e-04  <O>=+0.981076800  Δ= 0.00e+00  terms=    225
eps=1e-05  <O>=+0.981076800  Δ= 0.00e+00  terms=    728
...   bit-identical through eps=1e-07, then
eps=1e-08  <O>=+0.981077278  Δ= 4.78e-07  terms=  21595
exact                +0.981077357894
```

A four-decade plateau, and it is not convergence. The mechanism: at a small kick angle the only terms
that contribute to `⟨0|O|0⟩` are the ones rotated all the way to pure `Z`, and each rotation costs a
factor `sin θ_h`. Loosening the cutoff by a decade admits thousands of new terms, but for several
decades *none of them is pure `Z`*, so the expectation does not move at all while the sum keeps
growing. An exactly-zero difference there means "no relevant term has arrived yet".

`_plateau_is_real` therefore requires the two small successive differences **and** one of:

* the term count has stopped growing (equal `final_terms` on the last two points) — the sum has
  saturated, every Pauli string above the cutoff is present, and the plateau is the exact answer; or
* both differences are strictly **nonzero** — the ordinary picture of a series converging slowly.

A flat value with a still-growing sum is rejected and the sweep goes on. A sum truncated to **zero
terms** is rejected outright, however flat and "saturated" it looks. All three branches are
load-bearing in this run: `Z_62` at θ_h = π/4 converges *only* by saturation, `Z_62` at θ_h = 0.2
*only* by the nonzero-difference branch, and weight-17 at θ_h = π/4 empties out completely at
`min_abs_coeff` 1e-3.

The fix is worth the measured 190× in accuracy: with the plateau test in place, `Z_62` at θ_h = 0.2
self-converges to a true error of **2.98e-9** with an estimate of 4.78e-7, instead of 5.58e-7 with an
estimate of 0.

## 3. Results

### 3.1 Reference values

| observable | θ_h | reference | method | exact | uncertainty |
|---|---|---|---|---|---|
| `z62` | 0 | +1.000000000000 | `stim_clifford_exact` | yes | — |
| `z62` | 0.2 | +0.981077357894 | `light_cone_exact:both` | yes | — |
| `z62` | π/8 | +0.922334053224 | `light_cone_exact:both` | yes | — |
| `z62` | π/4 | +0.519411017552 | `light_cone_exact:both` | yes | — |
| `z62` | 3π/8 | +0.059299068661 | `light_cone_exact:both` | yes | — |
| `z62` | π/2 | +0.000000000000 | `stim_clifford_exact` | yes | — |
| `weight_10` | 0 | +0.000000000000 | `stim_clifford_exact` | yes | — |
| `weight_10` | 0.2 | −0.000650289249 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | π/8 | −0.007714801813 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | π/4 | −0.012840895482 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | 3π/8 | +0.317475056355 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | π/2 | +1.000000000000 | `stim_clifford_exact` | yes | — |
| `weight_17` | 0 | +0.000000000000 | `stim_clifford_exact` | yes | — |
| `weight_17` | 0.2 | +0.000000076744 | `self_converged` — **not** converged | **no** | 7.67e-08 |
| `weight_17` | π/8 | +0.000000000000 | `self_converged` — **not** converged | **no** | 0 |
| `weight_17` | π/4 | −0.000036246290 | `self_converged` — **not** converged | **no** | 3.62e-05 |
| `weight_17` | 3π/8 | −0.133657950824 | `self_converged` — **not** converged | **no** | 1.57e-03 |
| `weight_17` | π/2 | −1.000000000000 | `stim_clifford_exact` | yes | — |

**The four weight-17 interior references did not converge — none of them.** The plateau test refused
every one, and the budget guard stopped each sweep before the next tightening (`summary.json` carries
the full evidence). The reason is physical, not a budget accident: reaching `⟨0|O|0⟩` from a
weight-17 seed means rotating all seventeen `X`/`Y` sites to `Z`, and the resulting coefficients sit
far below any affordable cutoff except near θ_h = π/2. At θ_h = π/8 the expectation is *identically
zero* out to `min_abs_coeff = 1e-5` and 5.2e7 terms; at θ_h = 0.2 the first non-zero contribution
appears only at 1e-7 and 1.0e8 terms. Treat those two numbers as "consistent with zero at the
resolution reached", not as measurements — and note that **an error computed against a
self-converged reference is not an error against the truth**: for weight-17 the reference *is* this
engine's own tightest run, so the tightest point of each sweep has an error of ~0 by construction.
That circularity is exactly why `Z_62` and weight-10 carry exact references and why §3.4 exists.

### 3.2 Clifford endpoints vs. Benchmark A's integers

Scored over the coefficient sweep (8 cutoffs from 1e-2 to 1e-9) at each endpoint:

| observable | θ_h | exact integer | worst deviation, 8 runs | weight caps that emptied the sum |
|---|---|---|---|---|
| `z62` | 0 | +1 | **0** | — |
| `z62` | π/2 | 0 | **0** | 2, 4, 6, 8, 10, 12 |
| `weight_10` | 0 | 0 | **0** | 2, 4, 6, 8 |
| `weight_10` | π/2 | +1 | **0** | 2, 4, 6, 8 |
| `weight_17` | 0 | 0 | **0** | 2, 4, 6, 8, 10, 12 |
| `weight_17` | π/2 | −1 | **0** | 2, 4, 6, 8, 10, 12 |

Every endpoint is reproduced **bit-exactly at every cutoff**: `+1` for weight-10 and `−1` for
weight-17 at θ_h = π/2, matching Benchmark A. The `weight_10` and `weight_17` values at θ_h = 0 are
`0` rather than ±1 because at that angle `rx(0)` is the identity and every `ZZ` generator maps the
operator to another string with `X`/`Y` content, whose `|0…0⟩` expectation is exactly zero.

The last column is a genuine finding about the *weight* knob, not a failure: at a Clifford angle the
back-evolved operator passes through weight ~30–40 mid-circuit even though it lands on a single
low-weight string, so `max_weight ≤ 8` (and for weight-17, `≤ 12`) truncates the whole sum to zero
terms. `weight_10` at θ_h = π/2 recovers the exact `+1` as soon as the cap reaches 10. Endpoint
scoring therefore uses the coefficient sweep only; the weight-sweep figures live in their own
`summary.json` fields.

### 3.3 Time to |error| < 1e-3

`harness.time_to_accuracy`, grids ordered loosest to tightest; "cheapest" is the fastest passing
point by wall time.

| observable | θ_h | reached? | cheapest truncation | wall | terms |
|---|---|---|---|---|---|
| `z62` | 0 | yes | 1e-09 | 0.0012 s | 1 |
| `z62` | 0.2 | yes | 1e-04 | 0.0040 s | 225 |
| `z62` | π/8 | yes | 1e-03 | 0.0061 s | 527 |
| `z62` | π/4 | yes | 1e-05 | 2.44 s | 2 146 372 |
| `z62` | 3π/8 | yes | 1e-02 | 0.018 s | 1 056 |
| `z62` | π/2 | yes | 1e-09 | 0.0014 s | 1 |
| `weight_10` | 0 | yes | 1e-09 | 0.0013 s | 1 |
| `weight_10` | 0.2 | yes | 1e-02 | 0.0041 s | 175 |
| `weight_10` | π/8 | yes | 1e-03 | 0.27 s | 10 480 |
| `weight_10` | π/4 | **no** | — | — | — |
| `weight_10` | 3π/8 | **no** | — | — | — |
| `weight_10` | π/2 | yes | 1e-08 | 0.0013 s | 1 |
| `weight_17` | 0 | yes | 1e-08 | 0.0013 s | 1 |
| `weight_17` | 0.2 | (degenerate) | 1e-02 | 0.020 s | 945 |
| `weight_17` | π/8 | (degenerate) | 1e-02 | 0.074 s | 12 |
| `weight_17` | π/4 | (degenerate) | 1e-02 | 0.049 s | 0 |
| `weight_17` | 3π/8 | yes | 1e-04 | 101 s | 4 291 840 |
| `weight_17` | π/2 | yes | 1e-09 | 0.0027 s | 1 |

Two rows of honest bad news, both consequences of the recorded cuts in §5:

* **weight-10 at π/4 and 3π/8 never reach 1e-3.** The cut grid stops at `min_abs_coeff = 1e-4`, where
  the errors are 2.00e-3 and 2.29e-3 against the exact statevector references. The next decade would
  cost ~212 s and ~9.3e7 terms per point. The bar is reachable, just not inside this run's box.
* **The three weight-17 rows marked (degenerate) pass the bar vacuously.** Their references are
  ~1e-7 to ~4e-5, i.e. already inside 1e-3, so returning `0.0` — which at θ_h = π/4 and
  `min_abs_coeff = 1e-2` it does with *zero terms* — "passes". A pass whose truncated sum is empty
  measures nothing. This is a property of a 1e-3 bar applied to a signal three or more orders below
  it, not of the engine.

### 3.3a Truncation error is not monotone in the cutoff

Worth stating plainly because it shapes how the convergence panel must be read: loosening
`min_abs_coeff` by a decade does **not** always make the error worse. weight-10 at the two largest
interior angles, against exact references:

| observable | θ_h | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| weight-10 | π/4 | 1.10e-3 | 2.31e-3 | 2.00e-3 |
| weight-10 | 3π/8 | 3.40e-3 | 9.36e-3 | 2.29e-3 |

The loosest cutoff is *closest* at π/4. That is not a bug and not noise: the discarded terms carry
signs, so a partial sum can sit nearer the truth than a larger partial sum does, and a truncated
Pauli sum has no variational bound to forbid it. Two consequences, both already built in: the
"cheapest truncation that suffices" selection in §3.3 evaluates the whole grid rather than stopping at
the first pass, and `test_benchmark_b_sweep.py` asserts convergence *across* its grid
(`error[tightest] ≤ error[loosest]`, plus non-decreasing term counts) rather than point to point.

### 3.4 Self-convergence validation, where the exact answer *is* known

The weight-17 procedure run against exact references, scored by whether its estimated uncertainty
stayed honest (`true_error ≤ max(10 × estimate, 1e-12)`):

| observable | θ_h | exact | self-converged | true error | estimate | honest? |
|---|---|---|---|---|---|---|
| `z62` | 0.2 | +0.981077357894 | +0.981077354909 | 2.98e-09 | 4.78e-07 | yes |
| `z62` | π/8 | +0.922334053224 | +0.922334054565 | 1.34e-09 | 2.37e-07 | yes |
| `z62` | π/4 | +0.519411017552 | +0.519411017552 | 1.25e-14 | 0 | yes |
| `z62` | 3π/8 | +0.059299068661 | +0.059299068661 | 6.94e-18 | 0 | yes |
| `weight_10` | 0.2 | −0.000650289249 | −0.000650114250 | 1.75e-07 | 6.50e-06 | yes |
| `weight_10` | π/8 | −0.007714801813 | −0.007707770864 | 7.03e-06 | 4.57e-04 | yes |
| `weight_10` | π/4 | −0.012840895482 | −0.010838146388 | 2.00e-03 | 4.31e-03 | yes |
| `weight_10` | 3π/8 | +0.317475056355 | +0.317468054253 | 7.00e-06 | 7.07e-03 | yes |

**8/8 conservative**, with the estimate over-stating the true error by 1.6×–1000×. The two zero
estimates are plateaus reached by *saturation*, where the remaining error is summation rounding
(1.3e-14, 6.9e-18) — which is why the comparison carries a floating-point floor. The two rows where
the procedure was stopped by its budget (weight-10 at π/4 and 3π/8) are still honest, which is the
useful part: a budget-truncated sweep reports a larger uncertainty, not a falsely confident one.

`test_benchmark_b_sweep.py` runs the same check on the 20-qubit sublattice for three observables ×
six angles and asserts the same bar in CI.

## 4. Term-count parity against PauliPropagation.jl

Matched truncation, one gate per channel on both sides (schema-v1 task JSON drives both engines from
one description), Heisenberg picture, `|0…0⟩`, single-threaded, θ_h = 0.2,
`min_abs_coeff ∈ {1e-3, 1e-4, 1e-5}` — three cutoffs rather than one, because "identical counts at
one cutoff" is a much weaker statement than "identical counts along a sweep".

The comparison is **per applied layer, not just the final count** — all 1355 of them. A divergence
that cancels by the end is exactly the coefficient-boundary or truncation-schedule bug the check
exists to catch (`benchmarks/julia/README.md` §P3, §P5). Both engines report counts in application
order, so the lists line up index by index with no reversal.

Cutoffs are **powers of ten, deliberately not dyadic**: this repo drops `|c| <= eps` and
PauliPropagation.jl keeps `|c| == eps` (§P3), a divergence that is measure-zero for a non-dyadic
threshold and *not* measure-zero for `2**-14`-style cutoffs at Clifford angles. **No eps perturbation
was needed** — every case matched on the nose.

| observable | min_abs_coeff | per-layer counts | final terms (both) | \|Δ⟨O⟩\| | verdict |
|---|---|---|---|---|---|
| `z62` | 1e-3 | 1355 / 1355 identical | 59 | 0 | **OK** |
| `z62` | 1e-4 | 1355 / 1355 identical | 225 | 0 | **OK** |
| `z62` | 1e-5 | 1355 / 1355 identical | 728 | 0 | **OK** |
| `weight_10` | 1e-3 | 1355 / 1355 identical | 1 415 | 0 | **OK** |
| `weight_10` | 1e-4 | 1355 / 1355 identical | 9 641 | 0 | **OK** |
| `weight_10` | 1e-5 | 1355 / 1355 identical | 53 310 | 2.2e-19 | **OK** |
| `weight_17` | 1e-3 | 1355 / 1355 identical | 32 859 | 0 | **OK** |
| `weight_17` | 1e-4 | 1355 / 1355 identical | 351 100 | 0 | **OK** |
| `weight_17` | 1e-5 | 1355 / 1355 identical | 2 853 283 | 0 | **OK** |

**9/9 pass**: every one of the 12 195 compared per-layer term counts is identical, and eight of the
nine expectation values agree to the last bit. That clears plan §7 rule 2, so the cross-engine
numbers in `results.json` are reportable.

Two observations from the parity runs that are *not* headline comparisons:

* **Memory.** On the heaviest case (`weight_17` at 1e-5, 2.85e6 resident terms) PauliPropagation.jl's
  dict backend peaked at **67.6 GiB** RSS, against ~1.2 GiB for this engine's bucketed columns on the
  same sum. Part of jl's figure is the separate untimed `@countpaulis` pass the runner does for
  per-layer counts, so this is not a clean like-for-like — it is recorded because 67 GiB on a shared
  workstation is worth knowing before anyone reruns this.
* **Wall time is not compared here.** Julia's timings in `results.json` are `wall_warm_s` from a
  single warm repeat, taken on a loaded machine (see §6); the repo's own discipline puts anything
  under ~10% behind `scripts/ab-compare.sh`, and a one-repeat number is nowhere near that bar.
  Benchmark D's report is where the cross-engine timing crossover is characterised.

## 5. Recorded cuts

Plan §6/D15's time-box policy is "pilot, project, then shrink the grid and record the cut". The cuts
are a table in the driver (`COEFF_GRID_CUTS`), not an adaptive stop, so they are reviewable and do not
change shape with machine load. The pilot measurements they come from, single-threaded:

| observable | θ_h | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| `z62` | 3π/8 | 0.8 s | 1.8 s | 2.3 s | 2.4 s |
| weight-10 | 0.2 | 0.1 s | 0.3 s | 1.1 s | 3.6 s |
| weight-10 | π/8 | 1.1 s | 6.7 s | 37 s | 140 s |
| weight-10 | π/4 | 14 s | 212 s | — | — |
| weight-17 | 0.2 | 2.4 s | 16 s | 98 s | — |
| weight-17 | π/8 | 37 s | — | — | — |

Resulting grids (the full grid is eight points, 1e-2 … 1e-9):

| observable | θ_h | points kept | tightest cutoff |
|---|---|---|---|
| `z62` | all | 8 | 1e-9 |
| weight-10 | 0, π/2 | 8 | 1e-9 |
| weight-10 | 0.2 | 6 | 1e-7 |
| weight-10 | π/8 | 5 | 1e-6 |
| weight-10 | π/4, 3π/8 | 3 | 1e-4 |
| weight-17 | 0, π/2 | 8 | 1e-9 |
| weight-17 | 0.2 | 4 | 1e-5 |
| weight-17 | π/8, π/4, 3π/8 | 3 | 1e-4 |

`z62` is never cut: its whole eight-point grid is under 3 s at every angle, because a `Z` seed's
reachable set saturates (2 146 372 terms at π/4 from 1e-5 down, unchanged by four further decades).
The `max_weight` sweep needs no cut table at all — it is run at the tightest coefficient cutoff whose
*uncapped* run came in under `WEIGHT_SWEEP_TIME_BUDGET_S = 10 s`, and a weight cap can only remove
terms, so every capped run is cheaper than that already-measured one. Which cutoff each weight sweep
used is recorded per run in `results.json`, and it is **not** the same across curves — the
`error-vs-max-weight.svg` axis label says so.

The one thing the cuts cost is §3.3's two unmet accuracy bars. Everything else in this report is
inside its grid.

## 6. Caveats

* **`min_abs_coeff ≥ 1e-12` everywhere.** `cos(π/2) == 6.123233995736766e-17`, not zero, so at a
  Clifford angle every rotation leaves a numerically-dead residual branch; below that floor an
  untruncated 127-qubit propagation fans out without bound. `MIN_SAFE_COEFF` enforces it and the
  driver refuses a grid that goes lower.
* **Reference computations run in a spawned child process.** qiskit-aer's statevector simulator
  spawns an OpenMP pool that persists for the life of the process, which would trip
  `harness.assert_single_threaded` on every later timed run. The child's threads die with it, and only
  plain data crosses back. The weight-17 self-convergence child is additionally given
  `REFERENCE_THREADS = 16` Rayon workers: a reference is an oracle, not a timing measurement, so the
  single-thread rule does not bind it, and the threads buy cutoff reach. That is what let the
  θ_h = 0.2 reference get to 1.0e8 terms and see a non-zero value at all.
* **Timings were taken on a shared workstation with other work running.** The repo's stated
  single-thread campaign noise is ±5–8%; concurrent load here was heavier than that at times — the
  same weight-10 configuration measured 13.8 s and 28.6 s in two probes minutes apart. Term counts,
  expectation values and parity outcomes are unaffected, being load-independent. Treat the wall times
  as indicative of *shape*, not as campaign-grade numbers; `scripts/ab-compare.sh` is the tool for
  anything under ~10%.
* **Two derived fields in `summary.json` were recomputed after the run** from the recorded
  measurements, and `summary.json`'s own `notes` say so: `endpoints[]` (to score the coefficient
  sweep only — see §3.2) and `self_convergence_validation[].conservative` (to apply the
  floating-point floor of §3.4). No measurement was changed, and re-running the driver now produces
  both directly.
