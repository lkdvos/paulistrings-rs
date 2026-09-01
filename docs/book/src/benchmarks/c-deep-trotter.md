# Benchmark C — Deep Trotter, and a reachability boundary

<p class="lead">The headline benchmark, and the one whose headline is a negative
result. At 5 Trotter steps the accuracy target is met in a tenth of a second
against an exact reference. At 15–20 steps in the hard interior <strong>neither
the target nor a reference to score it against is reachable</strong> — and the
published record agrees, since the paper that set the bar publishes no exact
20-step value either.</p>

![Absolute error against warm wall time, one curve per depth, with the 0.01 target drawn](../assets/deep-trotter/error-vs-runtime.svg)

*`|error|` against warm wall time, one curve per depth in the ladder, with the
0.01 accuracy target drawn. Read the [`claimable` column](#time-to-error) below,
not the curve's crossing point.*

## Setup

Heavy-hex kicked Ising, **n = 127**, `θ_zz = −π/2`, observable **`Z_62`** — the
weight-1 operator whose 20-step point is the marquee number of the utility
experiment — a depth ladder of **5 / 9 / 15 / 20** Trotter steps, two kick angles
in the hard interior **θ_h ∈ {7π/32, 5π/16}**, and a **dyadic** truncation grid
`min_abs_coeff ∈ {2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸}` extended with three looser dyadics.
Heisenberg, contracted against `|0…0⟩`, single-threaded, warm timings, one gate
per channel — 5 420 channels at 20 steps.

Recorded run: ccqlin038 (Intel Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0, Python
3.11.11, julia 1.12.6 + PauliPropagation 0.8.2. **85.7 min** end to end, 42
records.

### Why these angles, and why there is a rung at 9

The plan suggested "θ_h ≈ 0.6–1.0". This benchmark uses **7π/32 = 0.687223…** and
**5π/16 = 0.981748…**, and puts a rung at **9** steps rather than 10, for one
reason: that is where the published exact benchmarks live. The upstream data
(Begušić, Gray & Chan, arXiv:2308.05077, Zenodo doi 10.5281/zenodo.10223349) has
an `exact.csv` on a `k·π/32` grid whose figure 5a is `⟨Z₆₂⟩` after 9 steps and
5b after 20 — and **`exact.csv` has a `5a` column and no `5b` column.** So the
paper that introduced the "<0.01 absolute accuracy" bar publishes **no exact
20-step value either**: independent, external corroboration of the reachability
finding below.

**Nothing was checked in.** The only egress available in that environment was a
*summarizing* fetch, not a byte-exact one, and a reference file transcribed
through a summarizer is not a citation — so the repository's references directory
still ships with no data files, and every number in this benchmark is computed by
an oracle in the repository. What the angle and depth choice buys is that the
follow-up is *one step*: fetch `exact.csv` byte-exactly, drop it in, and compare
its `5a` column against the 9-step rung row for row, no interpolation, because
both angles are on the grid. A test pins the anchor and pins that no
float-looking payload has leaked into it.

## Oracle: exact at 5 steps, self-converged beyond

The commutation-aware backward cone of `Z_62`, recomputed from the gate list on
every run:

| steps | cone | gates in the reduced circuit | reference |
|---|---|---|---|
| 5 | **19 q** | 83 | `light_cone_exact(method="both")` — **exact** |
| 9 | 65 q | 471 | self-converged |
| 15 | 127 q | 1 823 | self-converged |
| 20 | 127 q | 3 178 | self-converged |

`method="both"` runs *two independent simulations* over the same cone — an Aer
statevector and an untruncated Pauli propagation — and requires them to agree, so
the 5-step reference is not just exact but cross-checked. From 9 steps on the cone
is past any dense method, untruncated Pauli propagation over it fans out without
bound, and no published exact value exists either.

The self-convergence machinery is
[Benchmark B's](b-theta-sweep.md#the-self-convergence-criterion-and-the-measured-reason-it-is-not-the-obvious-one),
imported as a function object — a test asserts it is the same object — with two
retunings recorded in `summary.json`: the tolerance is `1e-3` rather than B's
`1e-5` (the bar here is 0.01, so a plateau resolved to `1e-3` leaves 10×
headroom, and B's `1e-5` is unreachable at 20 steps at any affordable cutoff),
and **the reference grid is extended two dyadic powers past the tightest timed
grid point**, so the error of every timed run — including the tightest — is
measured against something strictly tighter than itself.

### `claimable`: when a self-converged reference may be quoted

A reference is claimable if it is exact, or if it is self-converged **and** the
plateau test converged **and** the reported uncertainty is inside `0.01 / 2`.
Without the second and third conditions an "achieved" row is circular: the
reference is this engine's own tightest run, so the tightest timed point agrees
with it by construction and the error it reports says nothing about the truth.
Every accuracy row carries both `achieved` and `claimable`, and **only
`claimable` rows are quoted as results.**

### The uncertainty estimate is not a bound, and its bias flips sign {#the-uncertainty-estimate-is-not-a-bound}

Measured on the 20-qubit sublattice at 20 steps, where a dense statevector gives
the truth:

| θ_h | true error | reported uncertainty | ratio |
|---|---|---|---|
| 7π/32 | 9.57e-2 | 2.60e-1 | 2.7× **over** |
| 5π/16 | 9.43e-3 | 3.02e-2 | 3.2× **over** |

Conservative over the full grid. But stop the same sweep after **two** points —
which is what a budget guard does — and the estimate is a single difference taken
before the series has moved:

| θ_h | true error | reported uncertainty | ratio |
|---|---|---|---|
| 5π/16 | 3.96e-2 | 4.55e-3 | 8.7× **under** |
| 1.0 (off-grid probe) | 2.87e-2 | 1.83e-3 | **15.7× under** |

The second row is past the 10× slack Benchmark B's validation allows, so **B's "a
budget-truncated sweep reports a larger uncertainty, not a falsely confident one"
does not survive to this depth.** The plateau test still does its job —
`converged` stays `false`, because two successive small differences were never
seen — and that is the point: at this depth `converged = false` must be read as
**"no usable uncertainty estimate"**, not as "a slightly weaker estimate".

## Results

### Reference values, with convergence evidence

![Expectation against cutoff, against each reference](../assets/deep-trotter/convergence-vs-truncation.svg)

| θ_h | steps | reference ⟨Z₆₂⟩ | method | exact? | uncertainty | converged | **claimable** |
|---|---|---|---|---|---|---|---|
| 7π/32 | 5 | **+0.655563050749** | `light_cone_exact:both`, 19-q cone | **yes** | — | — | **yes** |
| 7π/32 | 9 | +0.627635952626 | self-converged, stopped at `2⁻¹⁸` | no | 1.05e-2 | no | no |
| 7π/32 | 15 | +0.486980851624 | self-converged, stopped at `2⁻¹⁶` | no | 2.12e-2 | no | no |
| 7π/32 | 20 | +0.397165406356 | self-converged, stopped at `2⁻¹⁶` | no | **1.44e-1** | no | no |
| 5π/16 | 5 | **+0.238477118019** | `light_cone_exact:both`, 19-q cone | **yes** | — | — | **yes** |
| 5π/16 | 9 | +0.125689481581 | self-converged, stopped at `2⁻¹⁶` | no | 3.38e-3 | no | no |
| 5π/16 | 15 | +0.040918596491 | self-converged, stopped at `2⁻¹⁶` | no | 2.08e-3 | no | no |
| 5π/16 | 20 | **+0.016131374386** | self-converged, plateau at `2⁻¹⁴` | no | 7.52e-4 | **yes** | **yes** |

**7π/32 at 20 steps** is the marquee point and the clearest negative result here.
The reference sweep reached `2⁻¹⁶` with 3.9·10⁷ terms in 276 s at 16 threads, and
the value never settled:

| cutoff | ⟨Z₆₂⟩ | Δ vs previous | final terms | peak terms | wall (16 threads) |
|---|---|---|---|---|---|
| 2⁻⁸ | +0.520968403928 | — | 363 | 1 838 | 0.39 s |
| 2⁻¹⁰ | +0.395887579771 | 1.25e-1 | 8 046 | 17 659 | 0.42 s |
| 2⁻¹² | +0.246415020524 | 1.49e-1 | 138 220 | 204 728 | 1.63 s |
| 2⁻¹⁴ | +0.253480590101 | 7.07e-3 | 2 441 936 | 3 108 582 | 18.6 s |
| 2⁻¹⁶ | +0.397165406356 | **1.44e-1** | 38 791 218 | 45 418 769 | 276 s |

*Stopped: the next tightening projects to ~6.2·10⁸ terms, over the 4·10⁸ guard.*

The partial sums swing between 0.25 and 0.52 as the cutoff tightens by factors of
four. That is not noise and not a bug — the discarded terms carry signs, so a
truncated Pauli sum has no variational bound — but it means the grid is nowhere
near resolving this point, and the driver refuses to quote it (uncertainty
1.44·10⁻¹, i.e. 14× the target).

**5π/16 at 20 steps** is the one deep point that *does* converge: 0.011941,
0.014730, 0.015481, 0.016131 at `2⁻⁸ … 2⁻¹⁴`, differences 2.79·10⁻³, 7.52·10⁻⁴,
6.50·10⁻⁴ — two successive below `1e-3`, both strictly nonzero, so the plateau
test accepts. It converges because the observable has *decayed*: `⟨Z₆₂⟩ ≈ 0.016`,
and the sum that carries it collapses from a 1.4·10⁷-term transient to 34 698
resident terms.

### Time to |error| < 0.01 {#time-to-error}

| θ_h | steps | reference | reached? | **claimable?** | cheapest cutoff | wall | terms |
|---|---|---|---|---|---|---|---|
| 7π/32 | 5 | exact | yes | **yes** | `2⁻¹²` | **0.11 s** | 59 336 |
| 7π/32 | 9 | self-conv. | yes | no | `2⁻¹⁴` | 37.3 s | 2 195 788 |
| 7π/32 | 15 | self-conv. | yes | no | `2⁻¹⁰` | 0.83 s | 11 571 |
| 7π/32 | 20 | self-conv. | yes | no | `2⁻¹⁰` | 1.23 s | 8 046 |
| 5π/16 | 5 | exact | yes | **yes** | `2⁻¹⁰` | **0.14 s** | 72 352 |
| 5π/16 | 9 | self-conv. | yes | no | `2⁻¹²` | 17.6 s | 155 416 |
| 5π/16 | 15 | self-conv. | yes | **no** | `2⁻⁸` | 0.13 s | 54 |
| 5π/16 | 20 | self-conv. | yes | **yes** | `2⁻⁸` | 0.13 s | 11 |

> **Read the `claimable` column, not the `reached?` column.** Every row "reaches"
> the bar, and most of those passes are meaningless.

- **The two exact-reference rows are the real measurements.** At 5 steps the 0.01
  target is met in **0.11 s / 5.9·10⁴ terms** and **0.14 s / 7.2·10⁴ terms**
  against a reference with no truncation anywhere, and `2⁻¹⁸` reproduces the
  exact value to 3.9·10⁻¹⁵ and 2.2·10⁻¹⁶ in ~2.4 s.
- **7π/32 at 15 and 20 steps "pass" at `2⁻¹⁰` by luck.** The reference is itself
  unresolved, and the error against it is *not* monotone in the cutoff — at 20
  steps it runs 1.24e-1, **1.28e-3**, 1.51e-1, 1.44e-1 as the cutoff tightens
  through `2⁻⁸ … 2⁻¹⁴`. The `2⁻¹⁰` "pass" is one partial sum happening to cross
  the (wrong) reference. This is exactly the failure the claimability test exists
  to catch.
- **5π/16 at 15 and 20 steps pass vacuously**, in Benchmark B's sense: the signal
  is 0.041 and 0.016, so a 0.01 *absolute* bar is 25% and 62% of the whole
  answer. At 20 steps **11 resident terms** clear it. The bar is only a demanding
  test where the signal is ≫ 0.01, which on this observable means shallow
  circuits or small kick angles.

> **So the honest headline is a reachability boundary, not a single number.** The
> 0.01 target is met, against an exact reference, in a tenth of a second at 5
> steps; at 9 steps it needs `2⁻¹⁴`, 2.2·10⁶ terms and 37 s, and the reference
> behind it is only resolved to `1e-2`; and at 15–20 steps in the hard interior
> neither the target nor a reference to score it against is reachable inside this
> benchmark's box.

### What it would take at 20 steps, θ_h = 7π/32 {#what-it-would-take}

Extrapolating the measured growth (15.9× in terms and ~15× in wall time per
factor of four in the cutoff, against a measured error law of only ~2.2× error
reduction per the same step):

| cutoff | projected terms | projected wall, 32 threads | projected columns |
|---|---|---|---|
| `2⁻¹⁶` | 4.5e7 *(measured)* | 276 s *(measured, 16 threads)* | 2.2 GiB |
| `2⁻¹⁸` | ~7e8 | ~1.1 h | ~37 GiB |
| `2⁻²⁰` | ~1e10 | ~17 h | ~560 GiB |

The 0.01 bar needs roughly `2⁻²⁰`–`2⁻²²` at this point. That is out of reach of a
workstation — consistent with the published record having no exact 20-step value
either.

[Showcase B2](../showcases/b2-noisy-verification.md#the-reachability-boundary-with-and-without-noise)
revisits this exact point *with noise* and finds 48× fewer peak terms and a 154×
smaller last difference — and still not a pass.

### The sanity envelope

![Peak resident terms against cutoff, with the expected envelope shaded](../assets/deep-trotter/term-count-vs-truncation.svg)

12 records fall on the plan's three cutoffs and are scored against the handoff's
1.2·10⁶–9.3·10⁶ tracked-set envelope. **None required a semantics
investigation.** Nine of twelve are inside or expectedly below; the three "above
ceiling" readings are all the same 1.44·10⁷-term transient at 5π/16 and `2⁻¹⁴`,
whose *final* counts are 3.2·10⁶, 2.4·10⁵ and 3.5·10⁴ — a three-order peak/final
collapse. The envelope lands squarely on the 20-step, `2⁻¹⁴` point it was quoted
for: **3 108 582 peak terms, dead centre.**

That matters for reading the accuracy shortfall: the setup is behaving as
expected, so the shortfall is not a symptom of a mis-specified circuit — **it is
what that tracked-set size actually buys at this depth.**

### Methodology validation at n = 127

Running the self-convergence procedure at the two depths where an exact reference
exists, at the *real* system size rather than only on the CI gate's sublattice:

| θ_h | steps | true error | estimated | conservative? |
|---|---|---|---|---|
| 7π/32 | 5 | **3.44e-15** | 3.71e-05 | yes |
| 5π/16 | 5 | **2.22e-16** | 0 | yes |

2/2, both converging by saturation to the exact answer at floating-point
precision. That is the procedure working where it can work; the sign-flip
measurement above is the record of where it cannot.

## The dyadic cutoffs and the one-ulp mitigation

The plan fixes exact dyadic cutoffs, which is the one case where this engine and
`PauliPropagation.jl` provably disagree: this repository drops `|c| <= eps`, jl
keeps `|c| == eps`. At a Clifford `θ_zz` the coefficients are exact dyadics too,
so an exact straddle is **not** a measure-zero event — which is why Benchmark B
could use powers of ten and ignore this, and C cannot.

The documented mitigation, applied and reported:

- **paulistrings runs use the dyadic verbatim.** `2⁻¹⁴` is `2⁻¹⁴`.
- **jl runs get `math.nextafter(eps, inf)`.** jl drops `|c| < eps′`, and there is
  no float strictly between `eps` and `eps′`, so `|c| < eps′` is exactly
  `|c| <= eps`: jl's rule becomes this engine's rule, bit for bit, with **no
  coefficient touched.** The perturbation is one ulp — 1.1·10⁻¹⁹ absolute at
  `2⁻¹⁴`.

`summary.json` records both thresholds and their difference for every parity
case, and two tests pin the two halves of the argument.

## Cross-engine parity, at the deepest point

![All 5420 per-layer term counts, both engines](../assets/deep-trotter/parity-per-layer-terms.svg)

Matched truncation at `θ_h = 7π/32`, **20 Trotter steps** — the deepest, heaviest
point in the benchmark — one gate per channel on both sides, Heisenberg, `|0…0⟩`,
single-threaded, at all three dyadic cutoffs the memory gate allowed. Per applied
layer, all 5 420 of them:

| cutoff | jl threshold (+1 ulp) | per-layer counts | final terms (both) | peak terms (both) | \|Δ⟨O⟩\| | verdict |
|---|---|---|---|---|---|---|
| `2⁻¹⁰` | +2.17e-19 | **5 420 / 5 420 identical** | 8 046 | 17 659 | 5.55e-17 | **OK** |
| `2⁻¹²` | +5.42e-20 | **5 420 / 5 420 identical** | 138 220 | 204 728 | 2.78e-17 | **OK** |
| `2⁻¹⁴` | +1.36e-20 | **5 420 / 5 420 identical** | 2 441 936 | 3 108 582 | 5.55e-17 | **OK** |

**3/3 pass: every one of the 16 260 compared per-layer term counts is identical**,
final *and* peak counts agree exactly, and the expectations agree to ≤ 5.6·10⁻¹⁷
against a `1e-9` bar. This is the first place in the suite where the one-ulp
mitigation is load-bearing.

### Memory — and a figure from Benchmark B that does not reproduce {#memory}

Every jl leg ran; **none was skipped for memory.** The gate's affine model,
refitted from each leg's directly-sampled RSS:

| after leg | model | projection for the next leg | measured |
|---|---|---|---|
| (prior) | 4.00 GiB + 23.44 KiB/term | 4.39 GiB at `2⁻¹⁰` | 0.66 GiB |
| `2⁻¹⁰` | prior slope kept — one point cannot fit a slope | 8.58 GiB at `2⁻¹²` | 0.79 GiB |
| `2⁻¹²` | 0.64 GiB + **0.74 KiB/term** *(fitted)* | 2.84 GiB at `2⁻¹⁴` | **2.00 GiB** |
| `2⁻¹⁴` | 0.70 GiB + **0.44 KiB/term** *(fitted)* | — | — |

So `PauliPropagation.jl`'s dict backend costs **~0.44–0.74 KiB per resident
term** on this host, plus a ~0.7 GiB fixed footprint — **~30–50× lower than the
24 KiB/term implied by Benchmark B's "67.6 GiB at 2.85·10⁶ terms".**
Extrapolating this fit to B's case gives ~2–3 GiB, not 67.6 GiB.

The two measurements are not directly comparable, and this benchmark does not
claim B's is wrong about what it measured — but the discrepancy is large enough to
name. What is different: the figure above is sampled **directly off the
`runner.jl` process** (`/proc/<pid>/status` `VmRSS`, polled twice a second),
whereas `getrusage(RUSAGE_CHILDREN).ru_maxrss` — the obvious alternative, and what
this driver originally used — is a process-lifetime running maximum over *all*
reaped children, which in this driver is dominated by its own multi-gigabyte
reference children. That conflation was observed during development: the same
1 925-term jl task read 3.68 GiB by `getrusage` and 0.66 GiB by direct sampling.
**Anyone re-deriving B's memory claim should re-measure it with a per-process
sampler before quoting it.**

For scale on this engine's side: the `2⁻¹⁴` sum's bucketed columns are ~0.15 GiB
by construction (3.1·10⁶ terms × 48 B for `x`/`z`/coefficient at `W=2`), and the
whole 42-run campaign's process high-water was 1.11 GiB. A clean per-run
comparison would need one run per process, so the ratio is left unstated rather
than computed from a monotone process-lifetime figure.

### Wall time — reported, not claimed {#wall-time-reported-not-claimed}

| cutoff | paulistrings (warm, 1 thread) | PauliPropagation.jl (1 warm repeat, 1 thread) |
|---|---|---|
| `2⁻¹⁰` | 1.21 s | 1.73 s |
| `2⁻¹²` | 14.7 s | 33.2 s |
| `2⁻¹⁴` | 201 s | 454 s |

jl is 1.4×, 2.3×, 2.3× slower on these three points. **This is not a benchmark
claim** — it is a single warm repeat per point on a shared workstation. A ~2.3×
gap is well outside the ±5–8% noise band and the direction is consistent across
three points spanning two orders of magnitude in problem size, so it is worth
recording; but the numbers to quote from this benchmark are the term counts and
the accuracy rows, which are load-independent. [Benchmark
D](d-xxz-chain.md#cross-engine-timing-and-the-crossover) is where the crossover
is characterised properly.

## Recorded cuts

The pilot behind the cuts, run at the round angles 0.7 / 1.0 before the
grid-aligned angles were adopted, gives dyad-to-dyad ratios at `θ_h = 0.7`, 20
steps of 4×, 12×, 15.5×, 16.4× in wall time and 9.6×, 11×, 15×, 15× in peak terms
per factor of four in the cutoff. So `2⁻¹⁸` projects to ~6.6·10³ s at 32 threads
(~7·10⁴ s single-threaded) and ~7·10⁸ resident terms (~37 GiB of columns) — out
of the whole time box for a single grid point.

**The pilot's projections held.** Measured in the recorded run at `θ_h = 7π/32`,
20 steps, single-threaded: 0.33 s, 1.23 s, 14.8 s, 202 s at `2⁻⁸ … 2⁻¹⁴`, against
the pilot's 0.31 s, 1.3 s, 16 s, 246 s at the neighbouring `θ_h = 0.7` — within
the noise band plus the 2% angle difference, except the `2⁻¹⁴` point, which came
in 18% *faster* than the pilot (which had been running against a 10-core orphan
process).

So the timed grid runs the full six dyadics at 5 steps and stops at `2⁻¹⁴` at 9,
15 and 20. Two further cuts, both recorded rather than hidden: **two kick angles,
not three** (a third would have cost ~40 min and told the same story; the two
chosen bracket the behaviour — 7π/32 keeps an O(0.5) signal at 20 steps, 5π/16 has
decayed to O(0.01)), and **the `2⁻¹⁸` reference at 20 steps**, cut by the driver's
budget guard *on the projection* rather than after paying for it — it would not
have made that reference claimable anyway.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py --validate-convergence
pytest python/paulistrings/tests/test_benchmark_c_deep.py    # the CI gate: 25 tests, ~50 s
```

## Caveats

- **References run in a spawned child with 16 Rayon workers.** A reference is an
  oracle, not a timing measurement, so the single-thread rule does not bind it,
  and the threads buy cutoff reach (measured 11.2× at 32 threads on the 20-step,
  `2⁻¹⁴` point). The child also confines qiskit-aer's persistent OpenMP pool.
- **Timings on a shared workstation.** The recorded run itself started on a quiet
  box (load 12, 238 GiB free) and stayed there, but two earlier aborted attempts
  left orphaned 10-core reference children that polluted the pilot before they
  were found and killed.
- **`min_abs_coeff ≥ 1e-12` everywhere.** The tightest cutoff used here is
  `2⁻²²` ≈ 2.4·10⁻⁷, far above the floor.
- **Peak vs final term count.** At 5π/16 and 20 steps the sum peaks at ~1.5·10⁷
  resident terms and lands on ~2·10⁴ — three orders apart. Everything
  term-count-shaped here uses the **peak**, because that is what a run has to
  hold; the results JSON carries both.

**Source for every number on this page:**
[`benchmarks/python/deep_trotter/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/deep_trotter/README.md),
with the raw records in `results.json` and the full convergence evidence in
`summary.json` next to it.
