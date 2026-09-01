# SU(4) brickwork: the matrix-gate path, measured

> ## ✅ The matrix-gate path is **not** a large-`m` liability — it is a **mid-`m`** one
>
> The parent study's biggest blind spot is now measured. `unitary_2q` against PauliPropagation.jl's
> dense `TransferMapGate` reaches ratio **1.974 at 2.30 × 10⁶ peak terms and is still rising** — higher
> than *either* rotation curve in the parent study at comparable size. What the matrix path does have
> is the study's **deepest mid-size deficit**: **0.620 at 1.29 × 10⁴ peak terms**, and a crossover at
> **≈ 8 × 10⁴ peak terms**, **21× kicked-Ising's** and the highest of the four workloads measured so
> far.

The parent study is [`../README.md`](../README.md); its protocol, ratio convention (`ratio =
t_julia / t_paulistrings`, > 1 means we are faster), acceptance rule and caveats apply here unchanged
and are not restated. This document reports only what this run adds.

Driver: `benchmarks/python/bench_jl_performance.py`, workload `su4` — **unedited**, at its committed
cutoff grid. Data: `results.json`, `summary.json`, transcript `run.log`, figures alongside. The
extension binary is the one the parent study and the deep-kicked-Ising run measured (mtime
`2026-09-01 03:19:44`); no Rust code was touched and nothing was rebuilt.

## What is different about this workload

| | rotation workloads (parent study, deep-KI) | this run |
|---|---|---|
| our channel | `pauli_rotation`, fanout 2 | **`unitary_2q`** — a local PTM, fanout up to **16** |
| jl's channel | `PauliRotation` (its fast path) | **`TransferMapGate`** — a dense 16×16 PTM per block |
| channels | 1355 / 1782 / 5420 | **105** (n = 36, depth 6 brickwork) |
| dispatch width | `W = 2` (100–127 qubits) → 48 B/term payload | **`W = 1`** (36 qubits) → **32 B/term** payload |

Two independent things therefore move at once relative to the rotation curves: a much higher fanout
per term, and a much lower channel count. They pull the curve in opposite directions, and the data
below separates them.

## Results

All five configurations passed the parity gate: **all 105 per-layer term counts identical** on every
one, expectations agreeing to ≤ 1.2 × 10⁻¹⁶ against a 1e-9 bar. No configuration was disqualified,
none was cut. At `min_abs_coeff = 1e-3` the expectation is
`-0.0030947264746490136` — **bit-identical** to the value benchmark E committed and the parent study's
mirror table pins, so this really is that circuit.

| `min_abs_coeff` | final terms | peak terms | rust s | jl s | median ratio | pairs agree | faster |
|---|---|---|---|---|---|---|---|
| 1e-2 | 193 | 1 416 | 0.00917 | 0.00909 | **0.983** | 0/5 | **indistinguishable** |
| 3e-3 | 7 089 | 12 924 | 0.1574 | 0.09754 | **0.620** | 5/5 | Julia |
| 1e-3 | 84 836 | 84 836 | 0.6171 | 0.6286 | **1.027** | 0/5 | **indistinguishable** |
| 3e-4 | 573 826 | 573 826 | 2.630 | 4.417 | **1.676** | 5/5 | paulistrings |
| 1e-4 | 2 296 294 | 2 296 294 | 7.806 | 15.46 | **1.974** ← still rising | 5/5 | paulistrings |

Per-pair ratios (the acceptance rule is about their agreement in sign, so all are shown):

| `min_abs_coeff` | pair 0 | pair 1 | pair 2 | pair 3 | pair 4 | spread |
|---|---|---|---|---|---|---|
| 1e-2 | 1.007 | 0.999 | 0.972 | 0.983 | 0.977 | 3.6 % |
| 3e-3 | 0.552 | 0.598 | 0.662 | 0.689 | 0.620 | 22.2 % |
| 1e-3 | 1.036 | 1.027 | 1.022 | 0.995 | 1.040 | 4.4 % |
| 3e-4 | 1.644 | 1.692 | 1.676 | 1.667 | 1.699 | 3.3 % |
| 1e-4 | 1.999 | 1.934 | 1.989 | 1.940 | 1.974 | 3.3 % |

**This run produced the study's first indistinguishable configurations** — 16 configurations in the
parent study and 5 in the deep-KI run were all unanimous; two of these five are not. Both sit at a
ratio of ~1, which is what a tie should look like. The 22 % spread at 3e-3 is entirely on our side —
our leg ranges 0.142–0.176 s across the five pairs while Julia's holds 0.097–0.099 s — on a 0.16 s
measurement, and all five pairs still agree in sign.

The `abba` alternation earns its keep at the heaviest point: rust-first pairs median 1.989,
julia-first 1.937, a consistent 2.7 % order effect. The quoted 1.974 is the median across both orders.

![ratio vs term count](ratio-vs-terms.svg)

![time vs term count](time-vs-terms.svg)

## Crossover: ≈ 8 × 10⁴ peak terms, and the sweep landed a configuration on it

Interpolated **8.01 × 10⁴ peak terms**, bracketed by the two sign-consistent neighbours
12 924 @ 0.620 and 573 826 @ 1.676. `summary.json` flags it
`inside_indistinguishable_zone: true`, and that flag needs reading carefully:

* The driver's "zone" is the min–max span over *all* mixed-sign configurations, here
  **1 416 … 84 836** — but the 12 924 point inside that span is unanimous at 0.620. The band is **not**
  uniformly unresolved.
* What the flag actually reflects is that the **1e-3 configuration is itself a measured tie**
  (median 1.027, pairs 0.995–1.040 straddling 1) at 84 836 peak terms. That is not a failure to
  resolve the crossover; it is landing on it. The crossover is at ~8 × 10⁴ peak terms and one
  configuration sits there.
* The 1 416-term tie is a **separate** phenomenon at the small-`m` end, on the far side of a region
  where Julia is clearly faster. The su4 ratio curve is **tie → dip → tie → rise**, not the rotation
  workloads' monotone climb.

| workload | channels | crossover (peak terms) | vs su4 |
|---|---|---|---|
| `kicked_ising` | 1355 | 3.79 × 10³ | **21.1×** lower |
| `kicked_ising_deep` | 5420 | 9.32 × 10³ | 8.6× lower |
| `xxz` | 1782 | 1.65 × 10⁴ | 4.9× lower |
| **`su4`** | **105** | **8.01 × 10⁴** | — |

**So yes: the crossover really is ~20× kicked-Ising's** (21.1×), which was the pilot's claim, and it
is the widest workload spread in the study — 21× across four workloads, against the parent study's
4.4× across two. It is only 4.9× XXZ's, so "20×" is specifically against kicked-Ising, not against
every rotation workload.

## Per-term cost: ours keeps falling, Julia's is flat

![per-term cost](per-term-cost.svg)

| peak terms | rust ns/peak-term | jl ns/peak-term | ratio |
|---|---|---|---|
| 1 416 | 6 477 | 6 417 | 0.983 |
| 12 924 | 12 175 | 7 547 | 0.620 |
| 84 836 | 7 274 | 7 410 | 1.027 |
| 573 826 | 4 584 | 7 697 | 1.676 |
| 2 296 294 | 3 399 | 6 732 | 1.974 |

Above the dip our cost per term falls **−72 %** (12 175 → 3 399 ns) and is still falling **−26 %**
across the last 4× alone; Julia's moves **−11 %** overall and is non-monotone
(7 547 / 7 410 / 7 697 / 6 732). This is the **XXZ pattern, not the 5-step-kicked-Ising pattern**:
Julia gets no per-term discount here, because this sweep is nowhere near closure (a 3× tightening
still multiplies the term count 4.0×), which is exactly what
[`../deep-kicked-ising/README.md`](../deep-kicked-ising/README.md) established the discount depends on.

Normalizer-free version — growth in time per growth in term count, step by step:

| step (final terms) | terms | rust time | jl time |
|---|---|---|---|
| 193 → 7 089 | ×36.7 | **×17.2** | ×10.7 |
| 7 089 → 84 836 | ×12.0 | **×3.92** | ×6.44 |
| 84 836 → 573 826 | ×6.76 | **×4.26** | ×7.03 |
| 573 826 → 2 296 294 | ×4.00 | **×2.97** | ×3.50 |

Our time grows less than Julia's on every step but the first. The entire deficit is created in that
one step, between 193 and 7 089 terms, and then repaid.

## Memory

Floors reproduce the parent study's to within 0.1 MB: **37.8 MB** for us, **0.601 GiB** for Julia
(×16.3).

| peak terms | rust peak | rust B/term | jl peak | jl B/term | jl / rust | capacity slack | rust slack-normalized |
|---|---|---|---|---|---|---|---|
| 1 416 | 0.038 GiB | 793 † | 0.657 GiB | 42 629 † | 53.8× | 1.45 | — |
| 12 924 | 0.044 GiB | 581 † | 0.659 GiB | 4 777 † | 8.2× | 1.27 | — |
| 84 836 | 0.049 GiB | **151** | 0.712 GiB | **1 404** | 9.3× | 1.55 | 98 |
| 573 826 | 0.091 GiB | **102** | 0.955 GiB | **662** | 6.5× | 1.83 | **56** |
| 2 296 294 | 0.235 GiB | **93** | 1.660 GiB | **495** | 5.3× | 1.83 | **51** |

† below ~10⁵ terms the floor-subtracted figure is allocator granularity, not payload.

Our bytes per term fall monotonically — **no repeat of the parent study's 91 → 125 B/term step**, and
no step is expected here: the su4 grid happens to place both heavy points at the *same*
power-of-two capacity slack (1.83), so this sweep adds two consistent points to the
[capacity-doubling model](../deep-kicked-ising/README.md#4-memory) without testing it further. Divided
by that slack, 56 and 51 B/term against `W = 1`'s **32 B/term** payload arithmetic — an overhead
factor of 1.6–1.8× where the two `W = 2` campaigns' six large points gave 63–73 against 48, i.e.
1.3–1.5×. The overhead does not shrink with `W`, which is what a fixed 16-byte coefficient plus
`W`-independent transient buffers predicts.

Julia pays **495 B/term** here at 36 qubits (a single `UInt64` key) against 350 B/term on XXZ's 100
qubits, so its dict overhead is not dominated by key width either.

![memory per term](memory-per-term.svg)

## What this says about the `gu2q` path

1. **Fanout-16 gather/merge is not a large-`m` problem.** At 2.30 × 10⁶ peak terms this path is at
   **1.974** — above the parent study's kicked-Ising (**1.431** at 2.12 × 10⁶) and XXZ (**1.798** at
   2.66 × 10⁶) at comparable size, second only to deep-KI's 2.197 at 3.11 × 10⁶ — and rising at
   +17.8 % per 4× terms. Nothing in this data supports optimizing `gu2q` *for large `m`*.
2. **It is a mid-`m` problem, and the deficit is quantified: 1.61× at 1.29 × 10⁴ peak terms.** That is
   the deepest point any workload reaches in the 6 × 10³–2 × 10⁴ band (kicked-Ising 1.126 at 6 311,
   XXZ 0.895 at 9 918, deep-KI 1.462 at 17 659), and it is what pushes the crossover out to 8 × 10⁴.
   An optimization aimed at the matrix-gate path buys the most between ~10⁴ and ~10⁵ terms; above
   5 × 10⁵ the path already wins by 1.7–2.0×.
3. **It is not the fixed per-layer cost.** At 1 416 peak terms we *tie* (0.983) where the rotation
   workloads sat at 0.32–0.46, because 105 channels pay our rebucket → permute → coset-loop overhead
   105 times instead of 1355–5420 times. The parent study's "our fixed overheads at small term counts"
   thread is real but is not what this curve's deficit is made of.
4. **Where the deficit comes from is not settled by this data.** It is created entirely in the
   193 → 7 089-term step (our time ×17.2 against Julia's ×10.7). A plausible reading is that a sum of
   ~10⁴ terms is still small enough to be cache-resident for a hash map, so Julia's dict path is near
   its cheapest exactly there, while our gather → sort → merge over a 16-way expansion pays its full
   cost from the first term. That is a hypothesis consistent with Julia's flat per-term curve above
   8 × 10⁴, not a measured mechanism; a `phase-timing` breakdown of a `unitary_2q` layer at 10⁴ terms
   would settle it and is the obvious next probe.

## Protocol deviations

**None.** The parent study's protocol was followed exactly, at its own pair count. Three procedural
notes, all decided before the timed run and recorded in
`research/notes/2026-09-01-large-m-campaign-log.md`:

1. **The workload was not extended.** The handoff authorized appending tighter cutoffs if 1e-4 landed
   short of ~10⁶ peak terms. A rust-only probe put it at 2.30 × 10⁶, so the committed five-cutoff grid
   was used unchanged and the driver was not edited — these are the numbers for the workload the study
   declared.
2. **A rust-only term-count probe and a 1-pair pilot preceded the timed run**, to size it and to
   project the pair count. Both are decision aids; none of their timings are reported. The pilot's
   five single-pair ratios (0.976 / 0.642 / 1.076 / 1.672 / 1.977) reproduce the timed run's medians
   inside the per-pair spread, which is the only use made of them.
3. **5 pairs**, the parent study's count (the deep-KI run's 3 was a time-box floor). The full run took
   **13.4 min**.

The box was quiet throughout (load ≤ 1.0, no other tenant), 240 GiB free at both ends.

## Reproducing

```bash
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload su4 --pairs 5 \
    --out benchmarks/python/jl_performance/su4-curve

# re-render figures from the committed data, no measurement
python benchmarks/python/jl_performance_figures.py \
    benchmarks/python/jl_performance/su4-curve/summary.json

# the CI protocol gate, which pins this workload's mirror (no julia, < 1 s)
pytest python/paulistrings/tests/test_jl_performance_protocol.py
```

Host ccqlin038 (2 × Xeon Gold 6244, `powersave`), Julia 1.12.6 + PauliPropagation.jl 0.8.2,
`PP_BACKEND=dict`, `PP_FUSED=0`, rustc 1.94.0, driver commit `35ff414` with a clean tree.
