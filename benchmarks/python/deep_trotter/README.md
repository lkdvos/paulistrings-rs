# Benchmark C — deep Trotter, time to fixed accuracy

Part A entry **C**, the headline, of `research/plans/2026-08-31-examples-benchmarks-suite.md` (§6).
Heavy-hex kicked Ising, **n = 127**, `θ_zz = −π/2`, observable **`Z_62`** (the weight-1 operator of
Kim et al. 2023, whose 20-step point is the marquee number of the utility experiment), a depth ladder
of **5 / 9 / 15 / 20 Trotter steps**, two kick angles in the plan's hard interior
**θ_h ∈ {7π/32, 5π/16}** = {0.687223…, 0.981748…}, and the plan's **dyadic** truncation grid
`min_abs_coeff ∈ {2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸}` extended with three looser dyadics. Heisenberg picture,
contracted against `|0…0⟩`, single-threaded, warm timings, one gate per channel
(`steps × (144 + 127)` channels, so 5 420 at 20 steps).

| file | what it is |
|---|---|
| `../bench_c_deep_trotter.py` | the driver — also the record of every measurement that shaped it |
| `results.json` | every `report.RunRecord` with full provenance |
| `summary.json` | references + convergence evidence, envelope checks, time-to-accuracy rows, parity outcomes, cuts, the published anchor |
| `error-vs-runtime.svg` | **the headline**: \|error\| vs warm wall time, one curve per depth, with the 0.01 target drawn |
| `term-count-vs-truncation.svg` | peak resident terms vs cutoff, with the handoff's 1.2e6–9.3e6 envelope shaded |
| `convergence-vs-truncation.svg` | plan §7 rule 4's convergence panel: ⟨Z₆₂⟩ vs cutoff against each reference |
| `parity-per-layer-terms.svg` | all 5 420 per-layer term counts, both engines (thick solid under thin dashed) |

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_c_deep_trotter.py --validate-convergence
```

`RAYON_NUM_THREADS=1` must be exported **before** the interpreter starts (Rayon builds its global
pool at the first propagate and never resizes it); the driver refuses to run otherwise. The CI-safe
gate for the same physics on a 20-qubit sublattice is
`python/paulistrings/tests/test_benchmark_c_deep.py` (25 tests, ~50 s, 208 MiB peak RSS).

## 0. The short version

Four results, in order of how much they should change what a reader believes:

1. **At 5 steps the plan's accuracy target is met easily and provably.** `2⁻¹²` reaches
   \|error\| < 0.01 against an *exact* causal-cone reference in **0.11 s** with 5.9e4 terms, and
   `2⁻¹⁸` reproduces the exact value to 3.9e-15 in 2 s. Nothing here is self-referential.
2. **At 15–20 steps in the hard interior, neither the target nor a reference to score it against is
   reachable.** At θ_h = 7π/32, 20 steps the reference sweep reached 3.9e7 terms and the value still
   swung by 1.4e-1 on the last tightening — the plan's grid does not resolve this point at all. The
   handoff's 1.2e6–9.3e6 tracked-set envelope *is* met there (3.1e6 peak terms, dead centre), so the
   two halves of the plan's acceptance gate are **not simultaneously satisfiable**: that tracked-set
   size buys ~1e-1, not <1e-2. Projected cost to reach 0.01 at that point: ~1e10 terms, ~560 GiB,
   ~17 h at 32 threads (§3.3). The published record agrees it is out of reach — the paper that set the
   0.01 bar publishes no exact 20-step value either (§1).
3. **PauliPropagation.jl agrees term for term, exactly, at 20 Trotter steps.** All **16 260** compared
   per-layer term counts are identical across three dyadic cutoffs, final and peak counts agree
   exactly, expectations to ≤ 5.6e-17 — and this is the first place in the suite where the one-ulp
   dyadic mitigation is load-bearing (§4, §6).
4. **Benchmark B's "67.6 GiB at 2.85e6 terms" for jl does not reproduce.** Sampling the `runner.jl`
   process directly gives **0.44–0.74 KiB/term** plus ~0.7 GiB fixed — 30–50× lower, and 2.00 GiB
   measured on a 3.1e6-term sum. §6.1 gives the likely reason (`getrusage(RUSAGE_CHILDREN)` conflates
   sibling children) and recommends re-measuring B's figure before it is quoted again.

Everything below is the evidence.

## 1. Why the angles are 7π/32 and 5π/16, and the depth ladder has a 9

The plan suggests "θ_h ≈ 0.6–1.0, e.g. 0.7 and 1.0". This benchmark uses **7π/32 = 0.687223…** and
**5π/16 = 0.981748…** instead, and puts a rung at **9** steps rather than 10, for one reason: that is
where the published exact benchmarks live.

`driver.PUBLISHED_ANCHOR` records what was established (2026-08-31):

* The upstream data is Begušić, Gray & Chan, *"Fast and converged classical simulations of evidence
  for the utility of quantum computing before fault tolerance"* (arXiv:2308.05077), data repository
  `tbegusic/arxiv-2308.05077-data` (Zenodo doi 10.5281/zenodo.10223349). Its `exact.csv` has header
  row `theta_h,4a,4b,4c,4d,5a` and 16 data rows on a `k·π/32` grid.
* From that paper's figure captions: Fig. 4a–d are the **5-step** observables (magnetization,
  weight-10, weight-17, weight-17-modified), and **Fig. 5a is ⟨Z₆₂⟩ after 9 steps, Fig. 5b after 20
  steps**.
* `exact.csv` has a `5a` column and **no `5b` column.** So the paper that introduced the "<0.01
  absolute accuracy" bar this benchmark is scored against publishes **no exact 20-step value
  either** — independent, external corroboration of plan decision D5 and of the cone sizes in §2.
* A rendering of column `4b` reproduced this repo's own independent exact weight-10 references
  (`../theta_sweep/README.md` §3.1) to **12 significant figures** at θ_h = π/8, π/4 and 3π/8, and the
  Clifford endpoints of `4b`/`4c`/`4d` reproduced the exact `0` / `+1` / `−1` integers. That is
  strong evidence the upstream file uses this suite's conventions — lattice, `θ_zz = −π/2`, layer
  order, Hermitian `Y`, `|0…0⟩`.

**Nothing was checked in.** The only egress available here was a *summarizing* fetch, not a
byte-exact one (`curl`/`wget` are blocked in this environment), and
`examples/data/references/README.md` is explicit: "the header is the citation, so it must be written
from the fetch, not from memory." A reference file transcribed through a summarizer is not a
citation, so `examples/data/references/` still ships with no data files and every number in this
report is computed by an oracle in this repo.

What the angle and depth choice buys is that the follow-up is **one step**: fetch `exact.csv`
byte-exactly, drop it in as `begusic2023_exact.csv` with the header that directory specifies, and
compare its `5a` column against this benchmark's 9-step rung — row for row, no interpolation, because
both angles are on the `k·π/32` grid.
`test_the_published_anchor_is_recorded_and_not_transcribed` pins all of this, including that no
float-looking payload has leaked into the anchor.

## 2. References — exact where one exists, self-converged where none does

The commutation-aware backward cone of `Z_62`, recomputed from the gate list by
`oracles.light_cone` on every run:

| steps | cone | gates in the reduced circuit | reference |
|---|---|---|---|
| 5 | **19 q** | 83 | `light_cone_exact(method="both")` — **exact** |
| 9 | 65 q | 471 | self-converged |
| 10 | 81 q | 638 | — (not a rung; shown for the trend) |
| 15 | 127 q | 1 823 | self-converged |
| 20 | 127 q | 3 178 | self-converged |

`method="both"` at 5 steps runs *two independent simulations* over the same cone — an Aer statevector
and an untruncated Pauli propagation — and requires them to agree, so the 5-step reference is not
just exact but cross-checked. From 9 steps on, the cone is past any dense method, untruncated Pauli
propagation over it fans out without bound, and (§1) no published exact value exists either. Those
references are **self-converged** and labelled `self_converged` in every record, with
`reference_exact = false`.

### 2.1 The self-convergence machinery is Benchmark B's, imported

`bench_c_deep_trotter` imports `bench_b_theta_sweep.self_converged_reference` and its
`_plateau_is_real` rather than re-implementing them, and
`test_c_reuses_benchmark_bs_plateau_criterion` asserts the *function object* is the same one. That is
deliberate: B measured that the obvious criterion ("two successive values agree") is **wrong**,
declaring convergence with an estimated uncertainty of exactly zero while the value was still 5.6e-7
from the truth, because at a small kick angle the expectation can sit bit-identical across four
decades of cutoff while the sum keeps growing. `_plateau_is_real` requires the two small successive
differences **and** either a saturated term count or two strictly-nonzero differences, and rejects a
zero-term sum outright.

Two things are retuned for C, both recorded in `summary.json`:

* **`SELF_CONVERGENCE_TOL = 1e-3`** (B uses `1e-5`). The bar here is the plan's 0.01, so a plateau
  resolved to 1e-3 leaves 10× headroom; B's 1e-5 is unreachable at 20 steps at any affordable cutoff.
* **The reference grid is extended two dyadic powers past the tightest *timed* grid point**, so the
  error of every timed run — including the tightest — is measured against something strictly tighter
  than itself. Where the extension is unaffordable the sweep stops and reports `converged = false`.

### 2.2 `claimable`: when a self-converged reference may be quoted against the 0.01 bar

`reference_is_claimable` returns `True` for an exact reference, and for a self-converged one only if
the plateau test converged **and** the reported uncertainty is inside `0.01 / 2`. Without the second
condition an "achieved" row is the circularity B flagged for weight-17: the reference is this
engine's own tightest run, so the tightest timed point agrees with it by construction and the error
it reports says nothing about the truth. Every accuracy row in `summary.json` carries both `achieved`
and `claimable`, and **only `claimable` rows are quoted as results below.**

### 2.3 The uncertainty estimate is not a bound, and its bias flips sign

Measured on the 20-qubit sublattice at 20 steps, where a dense statevector gives the truth
(`test_benchmark_c_deep.py`):

| θ_h | value at 2⁻¹² | exact | true error | reported uncertainty | ratio |
|---|---|---|---|---|---|
| 7π/32 | +0.498910481 | +0.594614476873 | 9.57e-2 | 2.60e-1 | 2.7× **over** |
| 5π/16 | +0.034707098 | +0.044136003756 | 9.43e-3 | 3.02e-2 | 3.2× **over** |

Conservative over the full grid. But stop the same sweep after **two** points — which is what a
budget guard does — and the estimate is a single difference taken before the series has moved:

| θ_h | reference | exact | true error | reported uncertainty | ratio |
|---|---|---|---|---|---|
| 5π/16 | +0.004546671 | +0.044136003756 | 3.96e-2 | 4.55e-3 | 8.7× **under** |
| 1.0 (off-grid probe) | +0.001831526 | +0.030567143 | 2.87e-2 | 1.83e-3 | **15.7× under** |

The second row is past the 10× slack Benchmark B's §3.4 validation allows, so **B's "a
budget-truncated sweep reports a larger uncertainty, not a falsely confident one" does not survive to
this depth.** The plateau test still does its job — `converged` stays `false`, because two successive
small differences were never seen — and that is the point: at this depth `converged = false` must be
read as *"no usable uncertainty estimate"*, not as *"a slightly weaker estimate"*.
`reference_is_claimable` encodes that reading, and
`test_a_budget_truncated_sweep_can_understate_the_true_error` pins it.

## 3. Results

Recorded run: commit `e024d8b`, clean worktree, ccqlin038 (Intel Xeon Gold 6244 @ 3.60 GHz), rustc
1.94.0, python 3.11.11, paulistrings 0.1.0, numpy 2.4.6, qiskit 2.5.2, stim 1.16.0, julia 1.12.6 +
PauliPropagation 0.8.2. **85.7 min** end to end, 42 `RunRecord`s (39 paulistrings, 3
PauliPropagation.jl), `MemAvailable` 238 GiB at exit.

### 3.1 Reference values, with convergence evidence

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

The two exact references are each *two* independent simulations of the 19-qubit reduced cone — an Aer
statevector and an untruncated Pauli propagation — required to agree by
`light_cone_exact(method="both")`, in 3.3–3.4 s.

The convergence evidence for every self-converged reference is in `summary.json`
(`references[...].reference_evidence.points`). The three that matter most:

**7π/32, 20 steps** — the marquee point, and the clearest negative result in this report. The
reference sweep reached `2⁻¹⁶` with 3.9e7 terms in 276 s at 16 threads and the value never settled:

| cutoff | ⟨Z₆₂⟩ | Δ vs previous | final terms | peak terms | wall (16 threads) |
|---|---|---|---|---|---|
| 2⁻⁸ | +0.520968403928 | — | 363 | 1 838 | 0.39 s |
| 2⁻¹⁰ | +0.395887579771 | 1.25e-1 | 8 046 | 17 659 | 0.42 s |
| 2⁻¹² | +0.246415020524 | 1.49e-1 | 138 220 | 204 728 | 1.63 s |
| 2⁻¹⁴ | +0.253480590101 | 7.07e-3 | 2 441 936 | 3 108 582 | 18.6 s |
| 2⁻¹⁶ | +0.397165406356 | **1.44e-1** | 38 791 218 | 45 418 769 | 276 s |

*stopped: the next tightening projects to ~6.2e8 terms (growth 15.9×), over `max_terms = 4e8`.*

The partial sums swing between 0.25 and 0.52 as the cutoff tightens by factors of four. That is not
noise and not a bug — the discarded terms carry signs, so a truncated Pauli sum has no variational
bound — but it means **the plan's grid is nowhere near resolving this point**, and the driver refuses
to quote it (`claimable = false`, uncertainty 1.44e-1 = 14× the target).

**7π/32, 9 steps** — the deepest rung the reference sweep pushed to `2⁻¹⁸`, reaching **2.6e8 terms**
in 299 s at 16 threads: 0.690283, 0.733785, 0.674054, 0.631624, 0.621171, 0.627636 at
`2⁻⁸ … 2⁻¹⁸`, differences 4.4e-2, 6.0e-2, 4.2e-2, 1.0e-2, 6.5e-3. Monotonically shrinking at the end,
but never two below `1e-3`, so `converged = false` and uncertainty 1.05e-2.

**5π/16, 20 steps** — the one deep point that *does* converge: 0.011941, 0.014730, 0.015481, 0.016131
at `2⁻⁸ … 2⁻¹⁴`, differences 2.79e-3, 7.52e-4, 6.50e-4 — two successive below `1e-3` and both
strictly nonzero, so the plateau test accepts. Uncertainty 7.52e-4, comfortably inside the bar, so
this row is claimable. It converges because the observable has *decayed*: ⟨Z₆₂⟩ ≈ 0.016, and the sum
that carries it collapses from a 1.4e7-term transient to 34 698 resident terms.

### 3.2 Time to |error| < 0.01

`harness.time_to_accuracy`, dyadic grid loosest-first, single-threaded, warm (the warm-up pass is
dropped once a run exceeds 3 s — recorded per record as `extra["warm"]`).

| θ_h | steps | reference | reached? | **claimable?** | cheapest cutoff | wall | terms |
|---|---|---|---|---|---|---|---|
| 7π/32 | 5 | exact | yes | **yes** | `2⁻¹²` | **0.11 s** | 59 336 |
| 7π/32 | 9 | self-conv. | yes | no | `2⁻¹⁴` | 37.3 s | 2 195 788 |
| 7π/32 | 15 | self-conv. | yes | no | `2⁻¹⁰` | 0.83 s | 11 571 |
| 7π/32 | 20 | self-conv. | yes | no | `2⁻¹⁰` | 1.23 s | 8 046 |
| 5π/16 | 5 | exact | yes | **yes** | `2⁻¹⁰` | **0.14 s** | 72 352 |
| 5π/16 | 9 | self-conv. | yes | no | `2⁻¹²` | 17.6 s | 155 416 |
| 5π/16 | 15 | self-conv. | yes | no | `2⁻⁸` | 0.13 s | 54 |
| 5π/16 | 20 | self-conv. | yes | **yes** | `2⁻⁸` | 0.13 s | 11 |

**Read the `claimable` column, not the `reached?` column.** Every row "reaches" the bar, and most of
those passes are meaningless:

* **The two exact-reference rows are the real measurements.** At 5 steps the plan's 0.01 target is met
  in **0.11 s / 5.9e4 terms** (7π/32) and **0.14 s / 7.2e4 terms** (5π/16) against a reference with
  no truncation anywhere, and `2⁻¹⁸` reproduces the exact value to 3.9e-15 and 2.2e-16 in ~2.4 s.
* **7π/32 at 15 and 20 steps "pass" at `2⁻¹⁰` by luck.** The reference is itself unresolved (2.1e-2
  and 1.4e-1), and the error against it is *not* monotone in the cutoff — at 20 steps it runs
  1.24e-1, **1.28e-3**, 1.51e-1, 1.44e-1 as the cutoff tightens through `2⁻⁸ … 2⁻¹⁴`. The `2⁻¹⁰`
  "pass" is one partial sum happening to cross the (wrong) reference. This is exactly the failure
  `reference_is_claimable` exists to catch.
* **5π/16 at 15 and 20 steps pass vacuously**, in Benchmark B's sense: the signal is 0.041 and 0.016,
  so a 0.01 *absolute* bar is 25 % and 62 % of the whole answer. At 20 steps 11 resident terms clear
  it. The bar is only a demanding test where the signal is ≫ 0.01, which on this observable means
  shallow circuits or small kick angles.

So the honest headline is a **reachability boundary, not a single number**: the plan's 0.01 target is
met, against an exact reference, in a tenth of a second at 5 steps; at 9 steps it needs `2⁻¹⁴`,
2.2e6 terms and 37 s (and the reference behind it is only resolved to 1e-2); and at 15–20 steps in
the hard interior neither the target nor a reference to score it against is reachable inside this
benchmark's box.

### 3.3 What it would take at 20 steps, θ_h = 7π/32

Extrapolating the measured growth (15.9× in terms and ~15× in wall time per factor of four in the
cutoff, and the CI gate's measured error law of only ~2.2× error reduction per the same step):

| cutoff | projected terms | projected wall, 32 threads | projected columns |
|---|---|---|---|
| `2⁻¹⁶` | 4.5e7 *(measured)* | 276 s *(measured, 16 threads)* | 2.2 GiB |
| `2⁻¹⁸` | ~7e8 | ~1.1 h | ~37 GiB |
| `2⁻²⁰` | ~1e10 | ~17 h | ~560 GiB |

The 0.01 bar needs roughly `2⁻²⁰`–`2⁻²²` at this point. That is out of reach of a workstation, which
is consistent with the published record: the paper that set the "<0.01 absolute accuracy" bar
publishes exact benchmarks for the 5-step observables and for ⟨Z₆₂⟩ at 9 steps, and **none at 20
steps** (§1).

### 3.4 The sanity envelope

12 records fall on the plan's three cutoffs and are scored against the handoff's 1.2e6–9.3e6
tracked-set envelope. **None required a semantics investigation.**

| θ_h | steps | cutoff | final terms | peak terms | verdict |
|---|---|---|---|---|---|
| 7π/32 | 5 | 2⁻¹⁴ | 329 534 | 329 534 | below floor — expected at 5 of 20 steps |
| 7π/32 | 5 | 2⁻¹⁶ | 882 375 | 882 375 | below floor — expected at 5 of 20 steps |
| 7π/32 | 5 | 2⁻¹⁸ | 1 865 949 | 1 865 949 | **inside** |
| 7π/32 | 9 | 2⁻¹⁴ | 2 195 788 | 2 245 594 | **inside** |
| 7π/32 | 15 | 2⁻¹⁴ | 1 817 494 | 2 254 084 | **inside** |
| 7π/32 | 20 | 2⁻¹⁴ | 2 441 936 | 3 108 582 | **inside** |
| 5π/16 | 5 | 2⁻¹⁴ | 1 544 083 | 1 544 083 | **inside** |
| 5π/16 | 5 | 2⁻¹⁶ | 2 121 774 | 2 121 774 | **inside** |
| 5π/16 | 5 | 2⁻¹⁸ | 2 146 424 | 2 146 424 | **inside** |
| 5π/16 | 9 | 2⁻¹⁴ | 3 196 258 | 14 396 463 | above ceiling (peak) |
| 5π/16 | 15 | 2⁻¹⁴ | 239 480 | 14 396 463 | above ceiling (peak) |
| 5π/16 | 20 | 2⁻¹⁴ | 34 698 | 14 396 463 | above ceiling (peak) |

Nine of twelve are inside or expectedly below; the three "above ceiling" readings are all the same
1.44e7-term transient at 5π/16 and `2⁻¹⁴`, whose *final* counts are 3.2e6, 2.4e5 and 3.5e4 — the
three-orders-of-magnitude peak/final collapse §7 describes. The handoff's envelope lands squarely on
the 20-step, `2⁻¹⁴` point it was quoted for: **3 108 582 peak terms**, dead centre of 1.2e6–9.3e6. So
the setup is behaving as the handoff expected, and the accuracy shortfall in §3.2 is not a symptom of
a mis-specified circuit — it is what that tracked-set size actually buys at this depth.

### 3.5 Methodology validation at n = 127

`--validate-convergence` runs the self-convergence procedure at the two depths where an exact
reference exists, and scores its estimate against the truth — at the *real* system size, not only on
the CI gate's 20-qubit sublattice:

| θ_h | steps | exact | self-converged | true error | estimated | conservative? |
|---|---|---|---|---|---|---|
| 7π/32 | 5 | +0.655563050749 | +0.655563050749 | **3.44e-15** | 3.71e-05 | yes |
| 5π/16 | 5 | +0.238477118019 | +0.238477118019 | **2.22e-16** | 0 | yes |

2/2, both converging by saturation to the exact answer at floating-point precision, with the estimate
overstating by ten orders of magnitude in the first case and legitimately zero in the second. That is
the procedure working where it can work; §2.3 is the record of where it cannot.

## 4. The dyadic cutoffs and the one-ulp mitigation

The plan fixes `min_abs_coeff ∈ {2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸}`. Those are exact dyadics, which is the one case
where this engine and PauliPropagation.jl provably disagree: this repo drops `|c| ≤ eps`, jl keeps
`|c| == eps` (`benchmarks/julia/README.md` §P3). At a Clifford `θ_zz` the coefficients are exact
dyadics too, so an exact straddle is **not** a measure-zero event — which is why Benchmark B could
use powers of ten and ignore this, and C cannot.

The documented mitigation, applied and reported:

* **paulistrings runs use the dyadic verbatim.** `2⁻¹⁴` is `2⁻¹⁴`.
* **jl runs get `math.nextafter(eps, inf)`.** jl drops `|c| < eps′`, and there is no float strictly
  between `eps` and `eps′`, so `|c| < eps′` is exactly `|c| ≤ eps`: jl's rule becomes this engine's
  rule, bit for bit, with **no coefficient touched**. The perturbation is one ulp — 1.1e-19 absolute
  at `2⁻¹⁴`, 1.1e-16 relative.

`julia_min_abs_coeff` does the conversion, `summary.json` records both thresholds and their
difference for every parity case, and two tests pin the two halves:
`test_this_engine_drops_a_coefficient_equal_to_the_cutoff` builds a one-term sum at exactly `2⁻¹⁴` and
checks it is dropped (and that `nextafter(2⁻¹⁴, ∞)` survives), and
`test_julia_one_ulp_perturbation_matches_this_engines_rule` checks the two rules agree on the
boundary value and both its neighbours, at all three of the plan's cutoffs.

## 5. Recorded cuts

Plan §6/D15's time-box policy is "pilot, project, then shrink the grid and record the cut". The cuts
are a table in the driver (`COEFF_GRID_CUTS`), not an adaptive stop, so they are reviewable and do not
change shape with machine load.

The pilot they come from (`_MEASURED_PILOT` in the driver, run at the round angles 0.7 / 1.0 before
the grid-aligned angles were adopted — a 2 % difference in θ_h, which does not move an
order-of-magnitude cost projection):

| θ_h | steps | cutoff | ⟨Z₆₂⟩ | final terms | peak terms | wall |
|---|---|---|---|---|---|---|
| 0.7 | 5 | 2⁻¹⁴ | +0.639626084 | 389 804 | 389 804 | 1.0 s |
| 0.7 | 5 | 2⁻¹⁶ | +0.638708946 | 1 011 254 | 1 011 254 | 1.2 s |
| 0.7 | 5 | 2⁻¹⁸ | +0.638708946 | 2 079 048 | 2 079 048 | 2.3 s |
| 0.7 | 10 | 2⁻¹⁴ | +0.487348444 | 2 235 674 | 2 582 120 | 64 s |
| 0.7 | 20 | 2⁻⁸ | +0.493356018 | 372 | 2 012 | 0.31 s |
| 0.7 | 20 | 2⁻¹⁰ | +0.380652188 | 7 787 | 19 336 | 1.3 s |
| 0.7 | 20 | 2⁻¹² | +0.238520762 | 133 109 | 219 016 | 16 s |
| 0.7 | 20 | 2⁻¹⁴ | +0.228420592 | 2 399 125 | 3 237 089 | 246 s |
| 0.7 | 20 | 2⁻¹⁴ | +0.228420592 | 2 399 125 | 3 237 089 | 22 s *(32 threads)* |
| 0.7 | 20 | 2⁻¹⁶ | +0.368476045 | 38 840 616 | 47 644 820 | 404 s *(32 threads)* |
| 1.0 | 5 | 2⁻¹⁴ | +0.215511154 | 1 543 616 | 1 543 616 | 1.7 s |
| 1.0 | 5 | 2⁻¹⁶ | +0.215535472 | 2 072 871 | 2 072 871 | 2.3 s |
| 1.0 | 5 | 2⁻¹⁸ | +0.215535472 | 2 146 412 | 2 146 412 | 2.5 s |
| 1.0 | 10 | 2⁻¹⁴ | +0.081905150 | 1 437 964 | 15 288 166 | 256 s |
| 1.0 | 20 | 2⁻⁸ | +0.000000000 | 6 | 6 625 | 0.20 s |
| 1.0 | 20 | 2⁻¹⁰ | +0.009333665 | 84 | 82 868 | 1.5 s |
| 1.0 | 20 | 2⁻¹² | +0.010138239 | 1 570 | 1 112 920 | 30 s |
| 1.0 | 20 | 2⁻¹⁴ | +0.010388188 | 20 140 | 15 288 166 | 355 s |

Two readings decide the cuts:

* **At 5 steps the sum saturates.** `2⁻¹⁶` and `2⁻¹⁸` give the same value to 1e-15, which is also the
  exact light-cone answer, and the tightest point costs 2.5 s. So the plan's full grid is kept there.
* **At 20 steps it does not.** Dyad-to-dyad ratios at θ_h = 0.7, 20 steps are 4×, 12×, 15.5×, 16.4×
  in wall time and 9.6×, 11×, 15×, 15× in peak terms per factor of four in the cutoff. `2⁻¹⁸` then
  projects to **~6.6e3 s at 32 threads (~7e4 s single-threaded) and ~7e8 resident terms (~37 GiB of
  columns)** — out of the plan's whole time box for a single grid point, never mind eight.

**The pilot's projections held.** Measured in the recorded run at θ_h = 7π/32, 20 steps,
single-threaded — 0.33 s, 1.23 s, 14.8 s, 202 s at `2⁻⁸ … 2⁻¹⁴` — against the pilot's 0.31 s, 1.3 s,
16 s, 246 s at the neighbouring θ_h = 0.7. Within the ±5–8 % single-thread noise band plus the 2 %
angle difference, except the `2⁻¹⁴` point, which came in 18 % *faster* than the pilot (the pilot ran
against a 10-core orphan process; §7).

So the timed grid runs the full six dyadics at **5 steps** and stops at **2⁻¹⁴** at 9, 15 and 20
steps. The `2⁻¹⁶` value at the deeper rungs is still *reported* — the reference sweep reaches it,
with threads, because a reference is an oracle and not a timing measurement (plan §7 rule 3 exists to
make cross-engine wall times comparable, and no reference wall time is quoted as a benchmark number).

Two further cuts, both recorded rather than hidden:

* **Two kick angles, not three.** A third would have cost a full grid plus a reference at every rung
  (~40 min measured at these depths) and told the same story; the two chosen bracket the interesting
  behaviour — 7π/32 keeps an O(0.5) signal at 20 steps, 5π/16 has decayed to O(0.01).
* **The `2⁻¹⁸` reference at 20 steps.** Reaching it would not have made the 20-step reference
  *claimable* anyway — §3.3's extrapolation puts that at `2⁻²⁰`–`2⁻²²` — but it would have given one
  more point on the curve. It was cut by the driver's own budget guard, which fired on the projection
  (~6.2e8 terms, over `max_terms = 4e8`) rather than after paying for it; the alternative was ~1.1 h
  at 32 threads with ~37 GiB of columns on a shared workstation. The projection is recorded in
  `summary.json` and the consequence is stated in §3.1 rather than papered over.
* **The `2⁻¹⁶` reference at 9 and 15 steps, and `2⁻¹⁸` at 9 steps for `5π/16`.** All stopped by the
  same guards, on projections, with the reason recorded per reference in
  `summary.json` (`stopped_early`). The 7π/32, 9-step sweep is the one that *did* reach `2⁻¹⁸`, at
  2.6e8 resident terms and 299 s with 16 threads — the largest single propagation in this benchmark.

## 6. Term-count parity and memory against PauliPropagation.jl

Matched truncation at θ_h = 7π/32, **20 Trotter steps** — the deepest, heaviest point in the
benchmark — one gate per channel on both sides (schema-v1 task JSON drives both engines from one
description), Heisenberg picture, `|0…0⟩`, single-threaded, at all three dyadic cutoffs the memory
gate allowed. The comparison is **per applied layer, not just the final count** — all 5 420 of them.

| cutoff | jl threshold (+1 ulp) | per-layer counts | final terms (both) | peak terms (both) | \|Δ⟨O⟩\| | verdict |
|---|---|---|---|---|---|---|
| `2⁻¹⁰` | +2.17e-19 | **5 420 / 5 420 identical** | 8 046 | 17 659 | 5.55e-17 | **OK** |
| `2⁻¹²` | +5.42e-20 | **5 420 / 5 420 identical** | 138 220 | 204 728 | 2.78e-17 | **OK** |
| `2⁻¹⁴` | +1.36e-20 | **5 420 / 5 420 identical** | 2 441 936 | 3 108 582 | 5.55e-17 | **OK** |

**3/3 pass: every one of the 16 260 compared per-layer term counts is identical**, final *and* peak
counts agree exactly, and the expectation values agree to ≤ 5.6e-17 against a 1e-9 bar. That clears
plan §7 rule 2, so the cross-engine records in `results.json` are reportable.

This is the first place in the suite where the **one-ulp mitigation is load-bearing**: the cutoffs are
exact dyadics and `θ_zz = −π/2` is a Clifford angle, so coefficients land on the threshold
bit-exactly, and the two engines' boundary rules differ there (§4). With
`math.nextafter(eps, ∞)` handed to jl the rules coincide and all three legs match on the nose. No
coefficient was touched, and `summary.json` records both thresholds and their difference per leg.

### 6.1 Memory — and a figure from Benchmark B that does not reproduce

Every jl leg ran; **none was skipped for memory.** The gate's affine model, refitted from each leg's
directly-sampled RSS:

| after leg | model | projection for the next leg | measured |
|---|---|---|---|
| (prior) | 4.00 GiB + 23.44 KiB/term | 4.39 GiB at `2⁻¹⁰` | 0.66 GiB |
| `2⁻¹⁰` | 4.00 GiB + 23.44 KiB/term *(prior slope kept — one point cannot fit a slope)* | 8.58 GiB at `2⁻¹²` | 0.79 GiB |
| `2⁻¹²` | 0.64 GiB + **0.74 KiB/term** *(fitted)* | 2.84 GiB at `2⁻¹⁴` | **2.00 GiB** |
| `2⁻¹⁴` | 0.70 GiB + **0.44 KiB/term** *(fitted)* | — | — |

So PauliPropagation.jl's dict backend costs **~0.44–0.74 KiB per resident term** on this host, plus a
~0.7 GiB fixed footprint. **That is ~30–50× lower than the 24 KiB/term implied by Benchmark B's
"67.6 GiB at 2.85e6 terms".** Extrapolating this fit to B's case gives ~2–3 GiB, not 67.6 GiB.

The two measurements are not directly comparable and this report does not claim B's is wrong about
what it measured — but the discrepancy is large enough to name. What is different here: the figure
above is sampled **directly off the `runner.jl` process** (`/proc/<pid>/status` `VmRSS`, polled twice
a second by `JuliaRssSampler`), whereas `getrusage(RUSAGE_CHILDREN).ru_maxrss` — the obvious
alternative, and what this driver originally used — is a process-lifetime running maximum over *all*
reaped children, which in this driver is dominated by its own multi-gigabyte reference children. That
conflation is real and was observed during development: the same 1 925-term jl task read 3.68 GiB by
`getrusage` and 0.66 GiB by direct sampling. **Anyone re-deriving B's memory claim should re-measure
it with a per-process sampler before quoting it.**

For scale on this engine's side: the `2⁻¹⁴` sum's bucketed columns are ~0.15 GiB by construction
(3.1e6 terms × 48 B for `x`/`z`/coefficient at W=2), and the whole 42-run campaign's process
high-water was 1.11 GiB. A clean per-run comparison would need one run per process (`harness`'s
memory-accounting note), so the ratio is left unstated rather than computed from a monotone
process-lifetime figure.

### 6.2 Wall time — reported, not claimed

| cutoff | paulistrings (warm, 1 thread) | PauliPropagation.jl (1 warm repeat, 1 thread) |
|---|---|---|
| `2⁻¹⁰` | 1.21 s | 1.73 s |
| `2⁻¹²` | 14.7 s | 33.2 s |
| `2⁻¹⁴` | 201 s | 454 s |

jl is 1.4×, 2.3×, 2.3× slower on these three points. **This is not a benchmark claim.** It is a
single warm repeat per point (`PARITY_WARM_REPEATS = 1`) on a shared workstation, and the repo's
discipline puts anything under ~10 % behind `scripts/ab-compare.sh`. A ~2.3× gap is well outside that
noise band and the direction is consistent across three points spanning two orders of magnitude in
problem size, so it is worth recording — but the numbers to quote from this benchmark are the term
counts and the accuracy rows, which are load-independent.

## 7. Caveats

* **Timings were taken on a shared workstation.** The repo's stated single-thread campaign noise is
  ±5–8 %. The recorded run itself started on a quiet box (load 12, 238 GiB free) and stayed there,
  but two earlier aborted attempts left orphaned 10-core reference children that polluted the pilot
  measurements before they were found and killed — which is why the `2⁻¹⁴` pilot number in §5 is 18 %
  slower than the run's. Term counts, expectation values, envelope readings and parity outcomes are
  load-independent and are the numbers to quote; treat the wall times as indicative of *shape*, not
  as campaign-grade figures. `scripts/ab-compare.sh` is the tool for anything under ~10 %.
* **References run in a spawned child with `REFERENCE_THREADS = 16` Rayon workers.** A reference is
  an oracle, not a timing measurement, so the single-thread rule does not bind it, and the threads
  buy cutoff reach (measured 11.2× at 32 threads on the 20-step, `2⁻¹⁴` point). The child is also
  what confines qiskit-aer's persistent OpenMP pool, which would otherwise trip
  `harness.assert_single_threaded` on every later timed run.
* **`min_abs_coeff ≥ 1e-12` everywhere,** inherited from Benchmark B's `MIN_SAFE_COEFF`:
  `cos(π/2) == 6.123233995736766e-17`, not zero, so at a Clifford angle every rotation leaves a
  numerically-dead residual branch and an untruncated 127-qubit propagation fans out without bound.
  The driver refuses a grid that goes lower. The tightest cutoff used here is `2⁻²²` ≈ 2.4e-7, far
  above it.
* **Peak vs final term count.** At 5π/16 and 20 steps the sum peaks at ~1.5e7 resident terms and
  lands on ~2e4 — three orders apart. Everything term-count-shaped in this report (the envelope
  check, the jl memory model, the figures) uses the **peak**, because that is what a run has to hold;
  `results.json` carries both.
