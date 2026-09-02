# Benchmark C — Deep Trotter circuits

<p class="lead">At 5 Trotter steps the 0.01 accuracy target is met in a tenth
of a second against an exact reference. At 15–20 steps in the hard interior
<strong>neither the target nor a reference to score it against is
reachable</strong>, and the published record agrees: the paper that set the
bar publishes no exact 20-step value either. That reachability boundary, not
a single accuracy number, is this benchmark's headline.</p>

![Absolute error against warm wall time, one curve per depth, with the 0.01 target drawn](../assets/deep-trotter/error-vs-runtime.svg)

*`|error|` against warm wall time, one curve per depth in the ladder, with the
0.01 accuracy target drawn. Read the [`claimable` column](#time-to-error)
below, not the curve's crossing point.*

## Setup

Heavy-hex kicked Ising, n = 127, `θ_zz = −π/2`, observable `Z_62`
(the weight-1 operator whose 20-step point is the marquee number of the
utility experiment), a depth ladder of 5 / 9 / 15 / 20 Trotter steps, two
kick angles in the hard interior θ_h ∈ {7π/32, 5π/16}, and a dyadic
truncation grid `min_abs_coeff ∈ {2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸}` extended with three
looser dyadics. Heisenberg, contracted against `|0…0⟩`, single-threaded, warm
timings, one gate per channel: 5 420 channels at 20 steps.

Recorded run: ccqlin038 (Intel Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0,
Python 3.11.11, julia 1.12.6 + PauliPropagation 0.8.2. **85.7 min** end to
end, 42 records.

### Angle, depth, grid and reference

7π/32 = 0.687223… and 5π/16 = 0.981748… sit on the `k·π/32` grid of the
published exact benchmarks (Begušić, Gray & Chan, arXiv:2308.05077), which
also fixes the rung at 9 steps rather than 10: that data's `exact.csv` has a
`5a` column (`⟨Z₆₂⟩` after 9 steps) and no `5b` column (after 20), so the
paper that set the 0.01-accuracy bar publishes no exact 20-step value either
— independent corroboration of the finding below. No reference file is
checked in, so every number here is computed by an oracle in this
repository. The full six-dyadic grid runs at 5 steps; at 9, 15 and 20 steps
it stops at `2⁻¹⁴`, since past that point at 20 steps the term count
projects to ~6.2·10⁸, over the 4·10⁸ guard. The two angles bracket the
behaviour: 7π/32 keeps an O(0.5) signal at 20 steps, 5π/16 decays to
O(0.01).

The reference is the commutation-aware backward cone of `Z_62`: exact at 5
steps (19 q, 83 gates), where `light_cone_exact(method="both")` requires an
Aer statevector and an untruncated Pauli propagation to agree, and
self-converged beyond (65 q at 9 steps, 127 q at 15 and 20). Past 5 steps no
dense method or published exact value exists, so the tolerance is `1e-3`
(10× headroom under the 0.01 bar) and the grid extends two dyadic powers
past the tightest timed point, measuring every timed run against something
strictly tighter than itself — the machinery is
[Benchmark B's](b-theta-sweep.md#the-self-convergence-criterion-and-the-measured-reason-it-is-not-the-obvious-one).

A reference is **claimable** if it is exact, or self-converged with the
plateau test converged and the uncertainty inside `0.01 / 2` — otherwise an
"achieved" row is circular, since the tightest timed point agrees with its
own tightest run by construction. Only `claimable` rows are quoted below.

## Results

### Reference values, with convergence evidence

![Expectation against cutoff, against each reference](../assets/deep-trotter/convergence-vs-truncation.svg)

| θ_h | steps | reference ⟨Z₆₂⟩ | method | exact? | uncertainty | converged | claimable |
|---|---|---|---|---|---|---|---|
| 7π/32 | 5 | **+0.655563050749** | `light_cone_exact:both`, 19-q cone | yes | — | — | **yes** |
| 7π/32 | 9 | +0.627635952626 | self-converged, stopped at `2⁻¹⁸` | no | 1.05e-2 | no | no |
| 7π/32 | 15 | +0.486980851624 | self-converged, stopped at `2⁻¹⁶` | no | 2.12e-2 | no | no |
| 7π/32 | 20 | +0.397165406356 | self-converged, stopped at `2⁻¹⁶` | no | 1.44e-1 | no | no |
| 5π/16 | 5 | **+0.238477118019** | `light_cone_exact:both`, 19-q cone | yes | — | — | **yes** |
| 5π/16 | 9 | +0.125689481581 | self-converged, stopped at `2⁻¹⁶` | no | 3.38e-3 | no | no |
| 5π/16 | 15 | +0.040918596491 | self-converged, stopped at `2⁻¹⁶` | no | 2.08e-3 | no | no |
| 5π/16 | 20 | **+0.016131374386** | self-converged, plateau at `2⁻¹⁴` | no | 7.52e-4 | yes | **yes** |

**7π/32 at 20 steps** is the clearest negative result here: the reference
sweep reached `2⁻¹⁶` (3.9·10⁷ terms, 276 s at 16 threads) and never settled.

| cutoff | ⟨Z₆₂⟩ | Δ vs previous | final terms | peak terms | wall (16 threads) |
|---|---|---|---|---|---|
| 2⁻⁸ | +0.520968403928 | — | 363 | 1 838 | 0.39 s |
| 2⁻¹⁰ | +0.395887579771 | 1.25e-1 | 8 046 | 17 659 | 0.42 s |
| 2⁻¹² | +0.246415020524 | 1.49e-1 | 138 220 | 204 728 | 1.63 s |
| 2⁻¹⁴ | +0.253480590101 | 7.07e-3 | 2 441 936 | 3 108 582 | 18.6 s |
| 2⁻¹⁶ | +0.397165406356 | **1.44e-1** | 38 791 218 | 45 418 769 | 276 s |

The partial sums swing between 0.25 and 0.52 as the cutoff tightens by
factors of four — the signature of discarded terms that carry signs, so a
truncated Pauli sum has no variational bound. The grid is nowhere near
resolving this point: uncertainty 1.44·10⁻¹, 14× the target.

**5π/16 at 20 steps** is the one deep point that converges (0.011941,
0.014730, 0.015481, 0.016131 at `2⁻⁸ … 2⁻¹⁴`, two successive differences
below `1e-3`), because the observable has decayed to `⟨Z₆₂⟩ ≈ 0.016` and the
sum collapses from a 1.4·10⁷-term transient to 34 698 resident terms.

### The uncertainty estimate is not a bound {#the-uncertainty-estimate-is-not-a-bound}

Its bias flips sign. Measured on the 20-qubit sublattice at 20 steps, where
a dense statevector gives the truth, over the full grid it is conservative;
stopped after two points, what a budget guard does, it is not:

| θ_h | regime | true error | reported uncertainty | ratio |
|---|---|---|---|---|
| 7π/32 | full grid | 9.57e-2 | 2.60e-1 | 2.7× over |
| 5π/16 | full grid | 9.43e-3 | 3.02e-2 | 3.2× **over** |
| 5π/16 | 2-point guard | 3.96e-2 | 4.55e-3 | 8.7× under |
| 1.0 (off-grid probe) | 2-point guard | 2.87e-2 | 1.83e-3 | **15.7× under** |

The last row is past the 10× slack Benchmark B's validation allows: a
budget-truncated sweep no longer just reports a weaker uncertainty here, it
can report a falsely confident one. `converged` still stays `false`, since
two successive small differences were never seen, and at this depth that
must be read as no usable estimate, not a weaker one.

### Time to |error| < 0.01 {#time-to-error}

| θ_h | steps | reference | reached? | claimable? | cheapest cutoff | wall | terms |
|---|---|---|---|---|---|---|---|
| 7π/32 | 5 | exact | yes | **yes** | `2⁻¹²` | **0.11 s** | 59 336 |
| 7π/32 | 9 | self-conv. | yes | no | `2⁻¹⁴` | 37.3 s | 2 195 788 |
| 7π/32 | 15 | self-conv. | yes | no | `2⁻¹⁰` | 0.83 s | 11 571 |
| 7π/32 | 20 | self-conv. | yes | no | `2⁻¹⁰` | 1.23 s | 8 046 |
| 5π/16 | 5 | exact | yes | **yes** | `2⁻¹⁰` | **0.14 s** | 72 352 |
| 5π/16 | 9 | self-conv. | yes | no | `2⁻¹²` | 17.6 s | 155 416 |
| 5π/16 | 15 | self-conv. | yes | no | `2⁻⁸` | 0.13 s | 54 |
| 5π/16 | 20 | self-conv. | yes | **yes** | `2⁻⁸` | 0.13 s | 11 |

Read the `claimable` column, not `reached?`: every row reaches the bar, and
most of those passes are meaningless.

- The two exact-reference rows are the real measurements: the 0.01
  target is met in **0.11 s / 5.9·10⁴ terms** and **0.14 s / 7.2·10⁴
  terms** against a reference with no truncation anywhere.
- 7π/32 at 15 and 20 steps "pass" at `2⁻¹⁰` **by luck**. The reference is
  itself unresolved and the error against it is not monotone in the cutoff
  (at 20 steps: 1.24e-1, **1.28e-3**, 1.51e-1, 1.44e-1 through `2⁻⁸ … 2⁻¹⁴`);
  the `2⁻¹⁰` pass is one partial sum crossing the wrong reference, exactly
  what the claimability test exists to catch.
- 5π/16 at 15 and 20 steps pass **vacuously**: the signal is 0.041 and
  0.016, so a 0.01 absolute bar is 25% and 62% of the whole answer, and 11
  resident terms clear it at 20 steps.

The 0.01 target is met, against an exact reference, in a tenth of a second
at 5 steps; at 9 steps it needs `2⁻¹⁴`, 2.2·10⁶ terms and 37 s against a
reference resolved only to `1e-2`; at 15–20 steps neither the target nor a
reference to score it against is reachable inside this benchmark's box.

### What it would take at 20 steps, θ_h = 7π/32 {#what-it-would-take}

Extrapolating the measured growth (15.9× in terms and ~15× in wall time per
factor of four in the cutoff, against a measured error law of only ~2.2×
error reduction per the same step):

| cutoff | projected terms | projected wall, 32 threads | projected columns |
|---|---|---|---|
| `2⁻¹⁶` | 4.5e7 *(measured)* | 276 s *(measured, 16 threads)* | 2.2 GiB |
| `2⁻¹⁸` | ~7e8 | ~1.1 h | ~37 GiB |
| `2⁻²⁰` | ~1e10 | ~17 h | ~560 GiB |

The 0.01 bar needs roughly `2⁻²⁰`–`2⁻²²` here, out of reach of a
workstation. [Showcase
B2](../showcases/b2-noisy-verification.md#the-reachability-boundary-with-and-without-noise)
revisits this point with noise: 48× fewer peak terms, a 154× smaller last
difference, and still not a pass.

### Sanity envelope and methodology check

![Peak resident terms against cutoff, with the expected envelope shaded](../assets/deep-trotter/term-count-vs-truncation.svg)

12 records fall on the three cutoffs tracked here, scored against a
1.2·10⁶–9.3·10⁶ tracked-set envelope: nine of twelve are inside or
expectedly below, and the three "above ceiling" readings are the same
1.44·10⁷-term transient at 5π/16 and `2⁻¹⁴`. The envelope lands squarely on
the 20-step, `2⁻¹⁴` point it was quoted for, **3 108 582 peak terms**, dead
centre: the accuracy shortfall above is what that tracked-set size actually
buys at this depth, not a mis-specified circuit.

Running the self-convergence procedure at the real system size, at the two
depths where an exact reference exists, both converge by saturation to the
exact answer at floating-point precision: true error **3.44e-15** against an
estimated 3.71e-05 at 7π/32, and **2.22e-16** against 0 at 5π/16 — both
conservative.

## The dyadic cutoffs and the one-ulp mitigation {#the-dyadic-cutoffs-and-the-one-ulp-mitigation}

Exact dyadic cutoffs are the one case where this engine and
`PauliPropagation.jl` provably disagree: this repository drops `|c| <= eps`,
jl keeps `|c| == eps`, and at a Clifford `θ_zz` the coefficients are exact
dyadics too, so an exact straddle is not measure-zero, unlike Benchmark B's
powers of ten. The mitigation: paulistrings runs use the dyadic verbatim,
and jl runs get `math.nextafter(eps, inf)` — since no float sits strictly
between `eps` and `eps′`, `|c| < eps′` is exactly `|c| <= eps`, jl's rule
becomes this engine's rule bit for bit, with no coefficient touched. The
perturbation is one ulp: 1.1·10⁻¹⁹ absolute at `2⁻¹⁴`.

## Cross-engine parity at the deepest point {#cross-engine-parity-at-the-deepest-point}

![All 5420 per-layer term counts, both engines](../assets/deep-trotter/parity-per-layer-terms.svg)

Matched truncation at `θ_h = 7π/32`, 20 Trotter steps, the deepest point in
the benchmark, one gate per channel on both sides, Heisenberg, `|0…0⟩`,
single-threaded, at the three dyadic cutoffs the memory gate allowed, per
applied layer, all 5 420 of them:

| cutoff | jl threshold (+1 ulp) | per-layer counts | final terms (both) | peak terms (both) | \|Δ⟨O⟩\| | verdict |
|---|---|---|---|---|---|---|
| `2⁻¹⁰` | +2.17e-19 | 5 420 / 5 420 identical | 8 046 | 17 659 | 5.55e-17 | OK |
| `2⁻¹²` | +5.42e-20 | 5 420 / 5 420 identical | 138 220 | 204 728 | 2.78e-17 | OK |
| `2⁻¹⁴` | +1.36e-20 | **5 420 / 5 420 identical** | 2 441 936 | 3 108 582 | 5.55e-17 | **OK** |

**3/3 pass**: every one of the 16 260 compared per-layer term counts is
identical, final and peak counts agree exactly, and the expectations agree
to ≤ 5.6·10⁻¹⁷ against a `1e-9` bar. This is the first place in the suite
where the one-ulp mitigation is load-bearing.

## Memory {#memory}

A figure from Benchmark B does not reproduce here: extrapolating this
section's fit to B's case gives ~2–3 GiB, not the 67.6 GiB B reported. Every
jl leg ran, none skipped for memory; the gate's affine model, refitted from
each leg's directly-sampled RSS, converges on:

| after leg | model (fitted) | measured |
|---|---|---|
| `2⁻¹⁰` | — (one point cannot fit a slope) | 0.66 GiB |
| `2⁻¹²` | 0.64 GiB + 0.74 KiB/term | **2.00 GiB** |
| `2⁻¹⁴` | 0.70 GiB + 0.44 KiB/term | — |

So jl's dict backend costs **~0.44–0.74 KiB per resident term** on this
host plus a ~0.7 GiB fixed footprint, ~30–50× lower than the 24 KiB/term
implied by B's figure. The two measurements are not directly comparable, but
the gap is worth naming: this figure samples `/proc/<pid>/status` `VmRSS` on
the `runner.jl` process directly, twice a second, while
`getrusage(RUSAGE_CHILDREN).ru_maxrss` is a process-lifetime maximum over
all reaped children, dominated here by multi-gigabyte reference children —
the same 1 925-term jl task read 3.68 GiB by `getrusage` and 0.66 GiB by
direct sampling. Re-measure with a per-process sampler before quoting B's
figure again. For scale, this engine's `2⁻¹⁴` sum is ~0.15 GiB by
construction (3.1·10⁶ terms × 48 B at `W=2`), and the whole 42-run
campaign's process high-water was 1.11 GiB.

## Wall time — reported, not claimed {#wall-time-reported-not-claimed}

| cutoff | paulistrings (warm, 1 thread) | PauliPropagation.jl (1 warm repeat, 1 thread) |
|---|---|---|
| `2⁻¹⁰` | 1.21 s | 1.73 s |
| `2⁻¹²` | 14.7 s | 33.2 s |
| `2⁻¹⁴` | 201 s | 454 s |

jl is 1.4×, 2.3×, 2.3× slower here, a single warm repeat per point on a
shared workstation, not a benchmark claim — but the gap is well outside the
±5–8% noise band and consistent in direction across two orders of magnitude
in problem size. The numbers to quote from this benchmark are the term
counts and accuracy rows, which are load-independent; [Benchmark
D](d-xxz-chain.md#cross-engine-timing-and-the-crossover) characterises the
crossover properly.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py --validate-convergence
pytest python/paulistrings/tests/test_benchmark_c_deep.py    # the CI gate: 25 tests, ~50 s
```

## Caveats

- **References run in a spawned child with 16 Rayon workers** (measured
  11.2× cutoff reach at 32 threads on the 20-step, `2⁻¹⁴` point), since a
  reference is an oracle, not a timing measurement; the child also confines
  qiskit-aer's persistent OpenMP pool.
- **Timings were taken on a shared workstation.** The recorded run started
  on a quiet box (load 12, 238 GiB free) and stayed there.
- **`min_abs_coeff ≥ 1e-12` everywhere**; the tightest cutoff used here is
  `2⁻²²` ≈ 2.4·10⁻⁷, far above the floor.
- **Peak vs final term count.** At 5π/16 and 20 steps the sum peaks at
  ~1.5·10⁷ resident terms and lands on ~2·10⁴; every term-count figure here
  uses the peak, since that is what a run has to hold.

**Numbers:** every value on this page is computed by
[`benchmarks/python/deep_trotter/bench_c_deep_trotter.py`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/deep_trotter/README.md),
with the raw records in `results.json` and the full convergence evidence in
`summary.json` next to it.
