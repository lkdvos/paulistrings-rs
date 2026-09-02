# B — Kick-angle sweep

<p class="lead">Six kick angles, three published observables, 5 Trotter steps:
how far a truncated Pauli sum drifts from the exact answer, and what an
<em>exact</em> reference costs at each cone size.</p>

![Absolute error against warm wall time, one curve per kick angle](../assets/theta-sweep/error-vs-runtime.svg)

*The plot the benchmark exists for: `|error|` against warm wall time, over the
truncation grid, for every observable and angle.*

## Setup

Heavy-hex kicked Ising, n = 127, 5 Trotter steps, `θ_zz = −π/2`, six kick
angles θ_h ∈ {0, 0.2, π/8, π/4, 3π/8, π/2}, three observables from the
utility experiment (`Z_62`, a weight-10 operator, a weight-17 operator).
Heisenberg picture, contracted against `|0…0⟩`, single-threaded, warm
timings, 1355 channels per circuit at one gate per channel. Recorded run:
ccqlin038 (Xeon Gold 6244 @ 3.60 GHz), rustc 1.94.0, Python 3.11.11, numpy
2.4.6, qiskit 2.5.2, stim 1.16.0, julia 1.12.6 + PauliPropagation 0.8.2,
39.6 min end to end, 236 records.

## Oracle

Reachable exactness is set by cone size, which the driver recomputes from the
gate list on every run rather than assuming it:

| observable | cone | interior reference | exact | per point |
|---|---|---|---|---|
| `Z_62` | 19 q | `light_cone_exact(method="both")` — an Aer statevector and an untruncated Pauli propagation, required to agree | yes | 15 s |
| weight-10 | 30 q | `light_cone_exact(method="statevector")` | yes | 142–197 s |
| weight-17 | 59 q | self-converged — tighten `min_abs_coeff` until the plateau is real | no | 8–66 s |

Both endpoints (θ_h ∈ {0, π/2}) are Clifford for every observable and get
exact integers from `stim` in under 0.1 s.

Weight-10 uses a statevector instead of Pauli propagation over the same
cone on cost alone: Pauli terms over the 30-qubit cone hit 4.3e8 at 160
applied gates (87 s) and abort past a 26 GiB cap before the 305-gate cone
completes, while the statevector is a flat ~150 s and 16.1 GiB regardless of
depth. `Z_62`'s 19-qubit cone is small enough for the Pauli path to be
affordable instead (12.8 s against 2.8 s), which is why it gets both methods
behind one reference. Weight-17 has neither option: 59 qubits rules out any
dense method, and untruncated propagation over its 59-qubit, 403-gate cone
is far past the weight-10 wall above, hence the self-converged reference.

## The self-convergence criterion {#the-self-convergence-criterion-and-the-measured-reason-it-is-not-the-obvious-one}

A criterion that tightens the cutoff until two successive expectation values
agree to `tol` reports convergence too early here. Run against the exact
`Z_62` reference at θ_h = 0.2, it settles on an estimated uncertainty of
**exactly zero** while the value is still 5.6·10⁻⁷ from the truth:

```text
eps=1e-03  <O>=+0.981076800  Δ=  —        terms=     59
eps=1e-04  <O>=+0.981076800  Δ= 0.00e+00  terms=    225
...   bit-identical through eps=1e-07, then
eps=1e-08  <O>=+0.981077278  Δ= 4.78e-07  terms=  21595
exact                +0.981077357894
```

At a small kick angle only terms rotated all the way to pure `Z` contribute
to `⟨0|O|0⟩`, and each rotation costs a factor `sin θ_h`. Loosening the
cutoff by a decade admits thousands of new terms, but for several decades
none of them is pure `Z`, so the expectation does not move: a zero
difference here means no relevant term has arrived yet, not convergence.

The criterion used in this benchmark also requires the term count to have
stopped growing (the sum has saturated and the plateau is exact), or both
successive differences to be strictly nonzero (ordinary slow convergence). A
flat value on a still-growing sum is rejected, and a sum truncated to zero
terms is rejected outright however flat it looks. With this criterion,
`Z_62` at θ_h = 0.2 self-converges to a true error of 2.98·10⁻⁹ with an
estimate of 4.78·10⁻⁷ — a **~190×** tighter estimate than the naive rule,
which reports the same run's 5.58·10⁻⁷ true error as zero.

## Reference values

![Absolute error against the coefficient cutoff](../assets/theta-sweep/error-vs-min-abs-coeff.svg)

`Z_62` and weight-10 carry exact references at every interior angle (12
decimal digits, in the committed `summary.json`). Weight-17's four interior
references (7.67e-08, 0, 3.62e-05, 1.57e-03 uncertainty at θ_h = 0.2, π/8,
π/4, 3π/8) did not converge: the plateau test refused every one, and the
budget guard stopped each sweep first. Reaching `⟨0|O|0⟩` from a weight-17
seed means rotating all seventeen `X`/`Y` sites to `Z`, so coefficients sit
far below any affordable cutoff except near θ_h = π/2 — at π/8 the
expectation is identically zero out to `1e-5` and 5.2·10⁷ terms, consistent
with zero at the resolution reached rather than a measurement. An error
scored against a self-converged reference is not an error against the
truth: for weight-17 the reference is this engine's own tightest run, so
the tightest point of each sweep has an error of ~0 by construction.

## Clifford endpoints {#clifford-endpoints}

![Absolute error against the weight cap](../assets/theta-sweep/error-vs-max-weight.svg)

Scored over the coefficient sweep (8 cutoffs from `1e-2` to `1e-9`) at each
endpoint:

| observable | θ_h | exact integer | worst deviation, 8 runs | weight caps that emptied the sum |
|---|---|---|---|---|
| `z62` | 0 | +1 | 0 | — |
| `z62` | π/2 | 0 | 0 | 2, 4, 6, 8, 10, 12 |
| `weight_10` | 0 | 0 | 0 | 2, 4, 6, 8 |
| `weight_10` | π/2 | +1 | 0 | 2, 4, 6, 8 |
| `weight_17` | 0 | 0 | 0 | 2, 4, 6, 8, 10, 12 |
| `weight_17` | π/2 | −1 | 0 | 2, 4, 6, 8, 10, 12 |

Every endpoint reproduces **bit-exactly** at every cutoff, matching
[Benchmark A](a-clifford.md). The weight-cap column is a genuine finding: at
a Clifford angle the back-evolved operator passes through weight ~30–40
mid-circuit even though it lands on a single low-weight string, so
`max_weight <= 8` (`<= 12` for weight-17) empties the sum; `weight_10`
recovers the exact `+1` as soon as the cap reaches 10.

## Time to |error| < 1e-3

`z62` and weight-10 clear the bar at every angle in under 3 s, except
weight-10 at π/4 and 3π/8, which never clear it inside this run's grid (cut
off at 1e-4: errors 2.00·10⁻³ and 2.29·10⁻³; the next decade would cost
~212 s and ~9.3·10⁷ terms). `z62`'s priciest passing point is π/4: 2.44 s at
2 146 372 terms. Three weight-17 rows pass vacuously — references of
~10⁻⁷–10⁻⁴ already sit inside 1e-3, so a `0.0` returned with zero terms
"passes" without measuring anything; the fourth, at 3π/8, needs 101 s and
4 291 840 terms.

## Truncation error is not monotone in the cutoff {#truncation-error-is-not-monotone-in-the-cutoff}

weight-10 at the two largest interior angles, against exact references:

| observable | θ_h | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| weight-10 | π/4 | 1.10e-3 | 2.31e-3 | 2.00e-3 |
| weight-10 | 3π/8 | 3.40e-3 | 9.36e-3 | 2.29e-3 |

The loosest cutoff is closest at π/4: discarded terms carry signs, so a
partial sum can sit nearer the truth than a larger one, and a truncated
Pauli sum has no variational bound against it. The CI test asserts
convergence across its grid rather than point to point.

## Self-convergence validation

The weight-17 procedure, run at every `z62`/weight-10 angle against exact
references and scored by whether its estimated uncertainty stayed honest
(`true_error ≤ max(10 × estimate, 1e-12)`), is **8/8** conservative, the
estimate over-stating the true error by 1.6×–1000× (full row breakdown in
the committed `summary.json`). The two zero estimates are plateaus reached
by saturation, the remaining error being summation rounding (1.3·10⁻¹⁴,
6.9·10⁻¹⁸), which sets the floating-point floor in the comparison. The two
budget-stopped rows are still honest: a larger uncertainty, not a falsely
confident one. That property does not survive to Benchmark C's depth — see
[C](c-deep-trotter.md#the-uncertainty-estimate-is-not-a-bound).

## Cross-engine parity {#cross-engine-parity}

![All 1355 per-layer term counts, both engines](../assets/theta-sweep/parity-per-layer-terms.svg)

Matched truncation, one gate per channel on both sides, Heisenberg, `|0…0⟩`,
single-threaded, θ_h = 0.2, `min_abs_coeff ∈ {1e-3, 1e-4, 1e-5}` — three
cutoffs rather than one, compared per applied layer, all 1355 of them.

![Term count against the cutoff, both engines](../assets/theta-sweep/term-count-vs-truncation.svg)

Per-layer counts are 1355/1355 identical for all nine (observable, cutoff)
pairs. Final terms and |Δ⟨O⟩|:

| observable | final terms @ 1e-3 / 1e-4 / 1e-5 | \|Δ⟨O⟩\| |
|---|---|---|
| `z62` | 59 / 225 / 728 | 0 |
| `weight_10` | 1 415 / 9 641 / 53 310 | 0 / 0 / 2.2e-19 |
| `weight_17` | 32 859 / 351 100 / 2 853 283 | 0 |

**9/9 pass:** all 12 195 compared per-layer term counts are identical, and
eight of the nine expectation values agree to the last bit. Cutoffs are
powers of ten, deliberately not dyadic, and no eps perturbation was needed.
See [Comparisons](../comparisons.md) for why that choice matters.

Two side findings. On the heaviest case (2.85·10⁶ resident terms)
`PauliPropagation.jl`'s dict backend peaked at **67.6 GiB** RSS against
**~1.2 GiB** for this engine's bucketed columns, not a clean like-for-like
(part of the jl figure is a separate untimed counting pass), but worth
knowing before anyone reruns it on a shared workstation. It also does not
reproduce: a per-process sampler in [Benchmark C](c-deep-trotter.md#memory)
measured the same quantity 30–50× lower. Wall time is not compared here —
the jl timings are a single warm repeat on a loaded machine.

## Reproducing

```bash
source .venv/bin/activate
RAYON_NUM_THREADS=1 python benchmarks/python/bench_b_theta_sweep.py --validate-convergence
pytest python/paulistrings/tests/test_benchmark_b_sweep.py    # the CI gate, on a 20-qubit sublattice
```

`RAYON_NUM_THREADS=1` must be exported before the interpreter starts; the
driver refuses to run otherwise.

## Caveats

- Reference computations run in a spawned child process, since qiskit-aer's
  statevector simulator spawns an OpenMP pool that would trip the
  single-thread assertion later; only plain data crosses back. The
  weight-17 self-convergence child additionally gets 16 Rayon workers, since
  a reference is an oracle rather than a timing measurement; the workers
  buy the cutoff reach that reached 1.0·10⁸ terms at θ_h = 0.2.
- Timings were taken on a shared workstation with other work running: the
  same weight-10 configuration measured 13.8 s and 28.6 s minutes apart.
  Term counts, expectation values and parity outcomes are load-independent.
- `min_abs_coeff ≥ 1e-12` everywhere: `cos(π/2) == 6.123233995736766e-17`,
  not zero, so a Clifford-angle rotation leaves a numerically dead residual
  branch, and below that floor an untruncated propagation fans out unbounded.
- Two derived fields in the committed `summary.json` were recomputed from
  the recorded measurements; that file's `notes` field says so.

**Numbers:** every figure on this page traces to
[`benchmarks/python/theta_sweep/README.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/python/theta_sweep/README.md),
with the 236 raw records in `results.json` and the verdicts in `summary.json`
next to it.
