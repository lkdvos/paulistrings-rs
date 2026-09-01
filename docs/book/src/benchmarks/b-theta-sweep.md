# Benchmark B — The kick-angle sweep

<p class="lead">Six kick angles, three published observables, and the question
of what an <em>exact</em> reference costs. This is the benchmark that built the
self-convergence machinery the rest of the suite imports — and the one that
measured why the obvious version of it is wrong.</p>

![Absolute error against warm wall time, one curve per kick angle](../assets/theta-sweep/error-vs-runtime.svg)

*The plot the benchmark exists for: `|error|` against warm wall time, over the
truncation grid, for every observable and angle.*

## Setup

Heavy-hex kicked Ising, **n = 127**, **5 Trotter steps**, `θ_zz = −π/2`, six kick
angles **θ_h ∈ {0, 0.2, π/8, π/4, 3π/8, π/2}**, and three observables from the
utility experiment — `Z_62` (weight 1), a weight-10 operator and a weight-17
operator. Heisenberg picture, contracted against `|0…0⟩`, single-threaded, warm
timings, 1355 channels per circuit at one gate per channel.

Recorded run: ccqlin038 (Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0, Python
3.11.11, numpy 2.4.6, qiskit 2.5.2, stim 1.16.0, julia 1.12.6 +
PauliPropagation 0.8.2. **39.6 min** end to end, 236 records.

## Oracle: exact where one is reachable, and the cost of finding out

The plan asked for a light-cone exact reference at every point for all three
observables. **It is not reachable**, and the reason is the cone size — which the
driver recomputes from the gate list on every run rather than assuming:

| observable | cone | interior reference | exact? | per point |
|---|---|---|---|---|
| `Z_62` | 19 q | `light_cone_exact(method="both")` — an Aer statevector **and** an untruncated Pauli propagation over the same cone, required to agree | **yes** | 15 s |
| weight-10 | 30 q | `light_cone_exact(method="statevector")` | **yes** | 142–197 s |
| weight-17 | 59 q | self-converged — tighten `min_abs_coeff` until the plateau is real | **no** | 8–66 s |

Both endpoints (`θ_h ∈ {0, π/2}`) are Clifford for every observable and get exact
integers from `stim` in under 0.1 s.

**Why weight-10 uses a statevector rather than Pauli propagation over the same
cone.** Both paths are exact, so the choice is pure cost — measured by growing
the applied-gate prefix of the 30-qubit reduced cone circuit:

| applied gates | peak terms | wall |
|---|---|---|
| 40 | 6.4e1 | 0.00 s |
| 80 | 5.2e4 | 0.03 s |
| 120 | 4.3e6 | 0.93 s |
| 160 | 4.3e8 | 87 s |
| 305 (the whole reduced cone) | — | aborted at a 26 GiB address-space cap |

The statevector over the same cone is ~150 s and 16.1 GiB and does not care about
depth, so it is the cheaper path at every angle. For `Z_62`'s 19-qubit cone the
Pauli path *is* affordable (12.8 s against 2.8 s), which is why that observable
can afford `method="both"` and get two independent simulations behind one
reference.

**Why weight-17 cannot have an exact interior reference.** 59 qubits rules out
any dense method, and untruncated Pauli propagation over a 59-qubit, 403-gate cone
is far past the weight-10 wall above. Nothing else is exact — hence the
self-converged reference, labelled as such in every record.

## The self-convergence criterion, and the measured reason it is not the obvious one

The obvious criterion is "tighten the cutoff until two successive expectation
values agree to `tol`". **That criterion is wrong here, and the validation leg
caught it.** Run against the *exact* `Z_62` reference at `θ_h = 0.2`, it declares
convergence with an estimated uncertainty of **exactly zero** while the value is
still 5.6·10⁻⁷ from the truth:

```text
eps=1e-03  <O>=+0.981076800  Δ=  —        terms=     59
eps=1e-04  <O>=+0.981076800  Δ= 0.00e+00  terms=    225
eps=1e-05  <O>=+0.981076800  Δ= 0.00e+00  terms=    728
...   bit-identical through eps=1e-07, then
eps=1e-08  <O>=+0.981077278  Δ= 4.78e-07  terms=  21595
exact                +0.981077357894
```

A four-decade plateau, and it is not convergence. The mechanism: at a small kick
angle the only terms contributing to `⟨0|O|0⟩` are the ones rotated all the way
to pure `Z`, and each rotation costs a factor `sin θ_h`. Loosening the cutoff by
a decade admits thousands of new terms, but for several decades *none of them is
pure `Z`*, so the expectation does not move at all while the sum keeps growing.
An exactly-zero difference there means "no relevant term has arrived yet".

The criterion therefore requires the two small successive differences **and** one
of:

- the term count has stopped growing — the sum has saturated, every Pauli string
  above the cutoff is present, and the plateau *is* the exact answer; or
- both differences are strictly **nonzero** — the ordinary picture of a series
  converging slowly.

A flat value with a still-growing sum is rejected and the sweep goes on. A sum
truncated to **zero terms** is rejected outright, however flat and "saturated" it
looks. All three branches are load-bearing in this run. The fix is worth a
measured **190×** in accuracy: with the plateau test in place, `Z_62` at
`θ_h = 0.2` self-converges to a true error of 2.98·10⁻⁹ with an estimate of
4.78·10⁻⁷, instead of 5.58·10⁻⁷ with an estimate of 0.

## Results

### Reference values

![Absolute error against the coefficient cutoff](../assets/theta-sweep/error-vs-min-abs-coeff.svg)

| observable | θ_h | reference | method | exact | uncertainty |
|---|---|---|---|---|---|
| `z62` | 0 | +1.000000000000 | `stim_clifford_exact` | yes | — |
| `z62` | 0.2 | +0.981077357894 | `light_cone_exact:both` | yes | — |
| `z62` | π/8 | +0.922334053224 | `light_cone_exact:both` | yes | — |
| `z62` | π/4 | +0.519411017552 | `light_cone_exact:both` | yes | — |
| `z62` | 3π/8 | +0.059299068661 | `light_cone_exact:both` | yes | — |
| `z62` | π/2 | +0.000000000000 | `stim_clifford_exact` | yes | — |
| `weight_10` | 0.2 | −0.000650289249 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | π/8 | −0.007714801813 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | π/4 | −0.012840895482 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | 3π/8 | +0.317475056355 | `light_cone_exact:statevector` | yes | — |
| `weight_10` | π/2 | +1.000000000000 | `stim_clifford_exact` | yes | — |
| `weight_17` | 0.2 | +0.000000076744 | `self_converged` — **not** converged | **no** | 7.67e-08 |
| `weight_17` | π/8 | +0.000000000000 | `self_converged` — **not** converged | **no** | 0 |
| `weight_17` | π/4 | −0.000036246290 | `self_converged` — **not** converged | **no** | 3.62e-05 |
| `weight_17` | 3π/8 | −0.133657950824 | `self_converged` — **not** converged | **no** | 1.57e-03 |
| `weight_17` | π/2 | −1.000000000000 | `stim_clifford_exact` | yes | — |

> **The four weight-17 interior references did not converge — none of them.** The
> plateau test refused every one, and the budget guard stopped each sweep before
> the next tightening.

The reason is physical, not a budget accident: reaching `⟨0|O|0⟩` from a
weight-17 seed means rotating all seventeen `X`/`Y` sites to `Z`, and the
resulting coefficients sit far below any affordable cutoff except near
`θ_h = π/2`. At `θ_h = π/8` the expectation is *identically zero* out to
`min_abs_coeff = 1e-5` and 5.2·10⁷ terms; at `θ_h = 0.2` the first non-zero
contribution appears only at `1e-7` and 1.0·10⁸ terms. Treat those two numbers as
"consistent with zero at the resolution reached", not as measurements — and note
that **an error computed against a self-converged reference is not an error
against the truth**: for weight-17 the reference *is* this engine's own tightest
run, so the tightest point of each sweep has an error of ~0 by construction.

### Clifford endpoints {#clifford-endpoints}

![Absolute error against the weight cap](../assets/theta-sweep/error-vs-max-weight.svg)

Scored over the coefficient sweep (8 cutoffs from `1e-2` to `1e-9`) at each
endpoint:

| observable | θ_h | exact integer | worst deviation, 8 runs | weight caps that emptied the sum |
|---|---|---|---|---|
| `z62` | 0 | +1 | **0** | — |
| `z62` | π/2 | 0 | **0** | 2, 4, 6, 8, 10, 12 |
| `weight_10` | 0 | 0 | **0** | 2, 4, 6, 8 |
| `weight_10` | π/2 | +1 | **0** | 2, 4, 6, 8 |
| `weight_17` | 0 | 0 | **0** | 2, 4, 6, 8, 10, 12 |
| `weight_17` | π/2 | −1 | **0** | 2, 4, 6, 8, 10, 12 |

Every endpoint is reproduced **bit-exactly at every cutoff**, matching
[Benchmark A](a-clifford.md).

The last column is a genuine finding about the *weight* knob, not a failure: at a
Clifford angle the back-evolved operator passes through weight ~30–40 mid-circuit
even though it lands on a single low-weight string, so `max_weight <= 8` (and for
weight-17, `<= 12`) truncates the whole sum to zero terms. `weight_10` at
`θ_h = π/2` recovers the exact `+1` as soon as the cap reaches 10.

### Time to |error| < 1e-3

Grids ordered loosest to tightest; "cheapest" is the fastest passing point.

| observable | θ_h | reached? | cheapest truncation | wall | terms |
|---|---|---|---|---|---|
| `z62` | 0.2 | yes | 1e-04 | 0.0040 s | 225 |
| `z62` | π/8 | yes | 1e-03 | 0.0061 s | 527 |
| `z62` | π/4 | yes | 1e-05 | 2.44 s | 2 146 372 |
| `z62` | 3π/8 | yes | 1e-02 | 0.018 s | 1 056 |
| `weight_10` | 0.2 | yes | 1e-02 | 0.0041 s | 175 |
| `weight_10` | π/8 | yes | 1e-03 | 0.27 s | 10 480 |
| `weight_10` | π/4 | **no** | — | — | — |
| `weight_10` | 3π/8 | **no** | — | — | — |
| `weight_17` | 0.2 | (degenerate) | 1e-02 | 0.020 s | 945 |
| `weight_17` | π/8 | (degenerate) | 1e-02 | 0.074 s | 12 |
| `weight_17` | π/4 | (degenerate) | 1e-02 | 0.049 s | 0 |
| `weight_17` | 3π/8 | yes | 1e-04 | 101 s | 4 291 840 |

Two rows of honest bad news, both consequences of the recorded cuts:

- **weight-10 at π/4 and 3π/8 never reach 1e-3.** The cut grid stops at
  `min_abs_coeff = 1e-4`, where the errors are 2.00·10⁻³ and 2.29·10⁻³ against
  the exact statevector references. The next decade would cost ~212 s and
  ~9.3·10⁷ terms per point. The bar is reachable, just not inside this run's box.
- **The three weight-17 rows marked (degenerate) pass the bar vacuously.** Their
  references are ~10⁻⁷ to ~4·10⁻⁵, i.e. already inside 1e-3, so returning `0.0` —
  which at `θ_h = π/4` and `min_abs_coeff = 1e-2` it does with *zero terms* —
  "passes". A pass whose truncated sum is empty measures nothing. This is a
  property of a 1e-3 bar applied to a signal three or more orders below it, not
  of the engine.

### Truncation error is not monotone in the cutoff

Worth stating plainly, because it shapes how every convergence panel on this site
must be read. weight-10 at the two largest interior angles, against exact
references:

| observable | θ_h | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| weight-10 | π/4 | **1.10e-3** | 2.31e-3 | 2.00e-3 |
| weight-10 | 3π/8 | 3.40e-3 | 9.36e-3 | **2.29e-3** |

The loosest cutoff is *closest* at π/4. That is not a bug and not noise: the
discarded terms carry signs, so a partial sum can sit nearer the truth than a
larger partial sum does, and a truncated Pauli sum has no variational bound to
forbid it. Two consequences, both already built in: the "cheapest truncation that
suffices" selection above evaluates the whole grid rather than stopping at the
first pass, and the CI test asserts convergence *across* its grid
(`error[tightest] ≤ error[loosest]`, plus non-decreasing term counts) rather than
point to point.

### Self-convergence validation, where the exact answer *is* known

The weight-17 procedure run against exact references, scored by whether its
estimated uncertainty stayed honest (`true_error ≤ max(10 × estimate, 1e-12)`):

| observable | θ_h | true error | estimate | honest? |
|---|---|---|---|---|
| `z62` | 0.2 | 2.98e-09 | 4.78e-07 | yes |
| `z62` | π/8 | 1.34e-09 | 2.37e-07 | yes |
| `z62` | π/4 | 1.25e-14 | 0 | yes |
| `z62` | 3π/8 | 6.94e-18 | 0 | yes |
| `weight_10` | 0.2 | 1.75e-07 | 6.50e-06 | yes |
| `weight_10` | π/8 | 7.03e-06 | 4.57e-04 | yes |
| `weight_10` | π/4 | 2.00e-03 | 4.31e-03 | yes |
| `weight_10` | 3π/8 | 7.00e-06 | 7.07e-03 | yes |

**8/8 conservative**, with the estimate over-stating the true error by 1.6×–1000×.
The two zero estimates are plateaus reached by *saturation*, where the remaining
error is summation rounding (1.3·10⁻¹⁴, 6.9·10⁻¹⁸) — which is why the comparison
carries a floating-point floor. The two rows where the procedure was stopped by
its budget are still honest, which is the useful part: a budget-truncated sweep
reports a larger uncertainty, not a falsely confident one.

*(That last statement does **not** survive to Benchmark C's depth — see
[C §2.3](c-deep-trotter.md#the-uncertainty-estimate-is-not-a-bound).)*

## Cross-engine parity

![All 1355 per-layer term counts, both engines](../assets/theta-sweep/parity-per-layer-terms.svg)

Matched truncation, one gate per channel on both sides, Heisenberg, `|0…0⟩`,
single-threaded, `θ_h = 0.2`, `min_abs_coeff ∈ {1e-3, 1e-4, 1e-5}` — three
cutoffs rather than one, because "identical counts at one cutoff" is a much weaker
statement than "identical counts along a sweep". The comparison is **per applied
layer, not just the final count** — all 1355 of them.

![Term count against the cutoff, both engines](../assets/theta-sweep/term-count-vs-truncation.svg)

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

**9/9 pass: every one of the 12 195 compared per-layer term counts is identical,**
and eight of the nine expectation values agree to the last bit. Cutoffs are
powers of ten, deliberately not dyadic, and **no eps perturbation was needed** —
every case matched on the nose. See [Comparisons](../comparisons.md) for why that
choice matters.

Two observations from the parity runs that are *not* headline comparisons:

- **Memory.** On the heaviest case (2.85·10⁶ resident terms)
  `PauliPropagation.jl`'s dict backend peaked at **67.6 GiB** RSS against ~1.2 GiB
  for this engine's bucketed columns. Part of the jl figure is a separate untimed
  counting pass, so this is not a clean like-for-like — it is recorded because
  67 GiB on a shared workstation is worth knowing before anyone reruns it. **And
  it does not reproduce:** Benchmark C re-measured the same quantity with a
  per-process sampler and got 30–50× lower. See
  [C §6.1](c-deep-trotter.md#memory).
- **Wall time is not compared here.** The jl timings in this benchmark's
  `results.json` are a single warm repeat taken on a loaded machine. Benchmark D
  is where the cross-engine timing crossover is characterised.

## Recorded cuts

The time-box policy is "pilot, project, then shrink the grid and record the cut",
and the cuts are a table in the driver — reviewable, and not changing shape with
machine load. The pilot, single-threaded:

| observable | θ_h | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|
| `z62` | 3π/8 | 0.8 s | 1.8 s | 2.3 s | 2.4 s |
| weight-10 | 0.2 | 0.1 s | 0.3 s | 1.1 s | 3.6 s |
| weight-10 | π/8 | 1.1 s | 6.7 s | 37 s | 140 s |
| weight-10 | π/4 | 14 s | 212 s | — | — |
| weight-17 | 0.2 | 2.4 s | 16 s | 98 s | — |
| weight-17 | π/8 | 37 s | — | — | — |

`z62` is never cut: its whole eight-point grid is under 3 s at every angle,
because a `Z` seed's reachable set saturates (2 146 372 terms at π/4 from `1e-5`
down, unchanged by four further decades). The one thing the cuts cost is the two
unmet accuracy bars above; everything else is inside its grid.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py --validate-convergence
pytest python/paulistrings/tests/test_benchmark_b_sweep.py    # the CI gate, on a 20-qubit sublattice
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts; the driver
refuses to run otherwise.

## Caveats

- **Reference computations run in a spawned child process.** qiskit-aer's
  statevector simulator spawns an OpenMP pool that persists for the life of the
  process, which would trip the harness's single-thread assertion on every later
  timed run. The child's threads die with it, and only plain data crosses back.
  The weight-17 self-convergence child is additionally given 16 Rayon workers — a
  reference is an oracle, not a timing measurement, and the threads buy cutoff
  reach; that is what let the `θ_h = 0.2` reference get to 1.0·10⁸ terms and see
  a non-zero value at all.
- **Timings were taken on a shared workstation with other work running.** The
  same weight-10 configuration measured 13.8 s and 28.6 s in two probes minutes
  apart. Term counts, expectation values and parity outcomes are unaffected,
  being load-independent.
- **`min_abs_coeff ≥ 1e-12` everywhere.** `cos(π/2) == 6.123233995736766e-17`,
  not zero, so at a Clifford angle every rotation leaves a numerically dead
  residual branch; below that floor an untruncated 127-qubit propagation fans out
  without bound. The driver refuses a grid that goes lower.
- **Two derived fields in the committed `summary.json` were recomputed after the
  run** from the recorded measurements, and that file's own `notes` say so. No
  measurement was changed, and re-running the driver now produces both directly.

**Source for every number on this page:**
[`benchmarks/python/theta_sweep/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/theta_sweep/README.md),
with the 236 raw records in `results.json` and the verdicts in `summary.json`
next to it.
