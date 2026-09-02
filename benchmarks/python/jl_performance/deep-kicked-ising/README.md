# Kicked-Ising at 20 Trotter steps: the ratio far from closure

The same circuit family as the 5-step curve, run deep enough that 3 × 10⁶ terms is far from the reachable Pauli
set's closure. This separates how much of the cross-engine ratio depends on the tracked set nearing closure from
how much depends on its size. The protocol, ratio convention (`ratio = t_julia / t_paulistrings`, `> 1` means
`paulistrings` is faster), acceptance rule and caveats live in [`../README.md`](../README.md).

Driver: `benchmarks/python/bench_jl_performance.py`, workload `kicked_ising_deep`, driver commit `e4aeccd`.
Engine `crates/` tree `4768fe4`, extension built 2026-09-01 03:19:44, the same binary the sweeps in [`../`](../)
and [`../su4-curve/`](../su4-curve/) measure, so all three are directly comparable.

## Configuration

| | 5-step curve ([`../`](../)) | this sweep |
|---|---|---|
| circuit | heavy-hex kicked-Ising, 127 q, `theta_zz = -pi/2` | same |
| observable / state | `Z_62` / `z+`, Heisenberg | same |
| Trotter steps | 5 (1355 channels) | 20 (5420 channels) |
| `theta_h` | `5pi/16` | `7pi/32` |
| cutoffs | 2⁻⁴ … 2⁻¹⁸, dyadic | 2⁻⁸, 2⁻¹⁰, 2⁻¹², 2⁻¹³, 2⁻¹⁴, dyadic |
| pairs per configuration | 5 | 3 |

The angle and depth are benchmark C's, which already proved per-layer parity at `theta_h = 7pi/32`, 20 steps,
`min_abs_coeff = 2⁻¹⁴`; this sweep reproduces its committed term counts exactly, 2 441 936 final and 3 108 582
peak.

**The sweep is far from closure**, which is what makes it the right control. Two consecutive halvings each
multiply the term count by ~4.2 (×4.191 for 2⁻¹² → 2⁻¹³, ×4.215 for 2⁻¹³ → 2⁻¹⁴), a clean `terms ∝ eps^-2.07`
power law with no sign of flattening at 2.4 × 10⁶ terms. Peak terms grow ×3.97 and ×3.82 over the same steps. The
5-step sweep at its tight end is the opposite case: a 4× tighter cutoff adds 1.16 % more terms.

## Results

All five configurations passed the per-layer parity gate: all 5420 counts identical on every one, expectations
agreeing to ≤ 2.2 × 10⁻¹⁶ against a 1e-9 bar. No configuration was disqualified, and all 15 pairs agreed in sign.

| `min_abs_coeff` | final terms | peak terms | rust s | jl s | rust ns/term | jl ns/term | ratio | pairs | faster |
|---|---|---|---|---|---|---|---|---|---|
| 2⁻⁸ = 0.003906 | 363 | 1 838 | 0.3285 | 0.1252 | 178 733 | 68 099 | 0.381 | 3/3 | Julia |
| 2⁻¹⁰ = 9.766e-4 | 8 046 | 17 659 | 1.206 | 1.763 | 68 273 | 99 822 | 1.462 | 3/3 | paulistrings |
| 2⁻¹² = 2.441e-4 | 138 220 | 204 728 | 14.89 | 33.40 | 72 723 | 163 164 | 2.244 | 3/3 | paulistrings |
| 2⁻¹³ = 1.221e-4 | 579 312 | 813 262 | 55.18 | 121.6 | 67 852 | 149 550 | 2.212 | 3/3 | paulistrings |
| 2⁻¹⁴ = 6.104e-5 | 2 441 936 | 3 108 582 | 202.0 | 443.7 | 64 973 | 142 721 | 2.197 | 3/3 | paulistrings |

| `min_abs_coeff` | pair 0 | pair 1 | pair 2 | spread |
|---|---|---|---|---|
| 2⁻⁸ | 0.340 | 0.381 | 0.395 | 16 % |
| 2⁻¹⁰ | 1.430 | 1.494 | 1.462 | 4.5 % |
| 2⁻¹² | 2.244 | 2.274 | 2.227 | 2.1 % |
| 2⁻¹³ | 2.212 | 2.271 | 2.202 | 3.1 % |
| 2⁻¹⁴ | 2.197 | 2.172 | 2.273 | 4.6 % |

**Crossover ≈ 9.32 × 10³ peak terms**, bracketed by 1 838 @ 0.381 and 17 659 @ 1.462, no indistinguishable zone.
That is 2.5× the 5-step workload's 3.79 × 10³, consistent with this engine's fixed per-layer cost, which a 4×
deeper circuit pays 4× more often at small term counts.

![ratio vs term count](ratio-vs-terms.svg)

![time vs term count](time-vs-terms.svg)

## The ratio is flat in peak terms

Over 8.1 × 10⁵ → 3.1 × 10⁶ peak terms the ratio moves 2.212 → 2.197, which is −0.7 %. Widening to the full
measured range, 2.0 × 10⁵ → 3.1 × 10⁶ (a 15.2× span), it moves 2.244 → 2.197, or −2.1 %, inside the 2–5 %
per-pair spread at those points. The 5-step sweep on the same binary decays over a span of the same width, 1.925
at 6.4 × 10⁵ falling to 1.389 at 2.1 × 10⁶, so that decay is not a property of large `m` on this engine.

Per-term cost says where the difference sits:

| sweep | span (peak terms) | rust ns/peak-term | jl ns/peak-term |
|---|---|---|---|
| 5-step | 637 219 → 2 146 424 | 1 252 → 1 120 (−10.6 %) | 2 423 → 1 567 (−35.3 %) |
| deep | 204 728 → 3 108 582 | 72 731 → 64 981 (−10.7 %) | 163 143 → 142 734 (−12.5 %) |
| deep, tail only | 813 262 → 3 108 582 | 67 850 → 64 981 (−4.2 %) | 149 521 → 142 734 (−4.5 %) |

This engine's amortization is identical in the two sweeps (−10.6 % against −10.7 %). Julia's is not: −35.3 % where
the sum is closing, −12.5 % where it is not, and in the deep tail the two amortize at the same rate, which is why
the ratio is flat. Near closure almost every gate application lands on a key the sum already contains, which for a
hash map is a lookup and an add with no insert, no rehash and no dict growth; a bucketed gather → sort → merge
gets no such discount, its cost being essentially independent of whether the keys are new.

At matched size the picture is the same. The 5-step sweep reads 1.431 at 2.12 × 10⁶ peak terms; this one reads
2.197 at 3.11 × 10⁶, 47 % more terms and a 1.54× larger advantage. Nothing on this engine's side degrades anywhere
in the sweep, with per-term cost still falling at 3.1 × 10⁶ terms. Worth recording as a limit on the reading: the
ratio plateaus at ~2.2 rather than climbing the way XXZ's does, so removing saturation removes the decay without
turning it into growth.

![per-term cost](per-term-cost.svg)

Absolute ns/peak-term is not comparable between the two workloads: the deep circuit applies 4× the channels and
holds a large sum across most of them, whereas the 5-step run reaches its peak only in its final layers. Only the
trend within each curve carries meaning.

## Memory

Floors are 37.7 MB for this engine against Julia's 0.601 GiB, a factor of 16.0.

| peak terms | rust peak | rust B/term | jl peak | jl B/term | jl / rust |
|---|---|---|---|---|---|
| 1 838 | 0.042 GiB | 3 078 † | 0.655 GiB | 32 021 † | 10.4× |
| 17 659 | 0.045 GiB | 471 † | 0.654 GiB | 3 251 † | 6.9× |
| 204 728 | 0.058 GiB | 112 | 0.765 GiB | 855 | 7.6× |
| 813 262 | 0.106 GiB | 92 | 1.091 GiB | 646 | 7.0× |
| 3 108 582 | 0.323 GiB | 99 | 1.970 GiB | 473 | 4.8× |

† below ~10⁵ terms the floor-subtracted figure is allocator granularity, not payload.

Bytes per term move 92 → 99 across a 3.8× increase, with peak RSS growing 3.05× for 3.82× the terms, i.e.
sub-linearly. The 5-step sweep shows a 91 → 125 B/term step across 1.54 × 10⁶ → 2.12 × 10⁶ terms, which this grid
jumps over. A capacity-doubling model reconciles the two: normalized by allocated-capacity slack
`2^ceil(log2 m) / m`, the six large points of the two sweeps collapse to 63–73 B/term of live payload against
48 B/term of `W = 2` payload arithmetic (5-step: 1.65 slack at 105 B/term, 1.36 at 91, 1.98 at 125, 1.95 at 123;
deep: 1.29 at 92, 1.35 at 99). The 91 → 125 step is then 1.36 → 1.98 slack, the term count having crossed 2²¹ so
that every bucket doubled at once. Exact-size final allocation, or a gentler growth factor, is worth up to ~1.5×
of peak RSS in the worst case and nothing in the best. This is a model consistent with both datasets, not a
measured mechanism; a direct allocator probe would settle it.

Julia's side corroborates the closure reading. Its 237 B/term at the 5-step sweep's 2.12 × 10⁶ terms, where its
RSS plateaus at 1.07 GiB across two configurations, is the cheapest per-term figure in either sweep, and is what a
dict that has stopped growing looks like. Here, still inserting, it pays 473 B/term at 3.11 × 10⁶ terms.

![memory per term](memory-per-term.svg)

## Protocol

The parent protocol was followed with one documented change: **3 pairs per configuration, not 5**. Five pairs
projected ~192 min against a ~75 min budget; three projected ~116 min and took 116.9. Three is the floor, matching
`benchmarks/PROFILING.md`'s A/B harness bar, and it is affordable because the signal being resolved is the
difference between ~2.2 and ~1.4, far outside the 2–5 % per-pair spread. Dropping the 2⁻¹⁴ point instead was never
an option, since it is the point the reading rests on. A restricted pilot and a rust-only term-count probe
preceded the timed run as sizing aids, both through the driver's own `parity_gate` and `run_pairs`; none of their
timings are reported.

No parity failure, no disqualification, no cut leg, no mixed-sign configuration. The box was quiet throughout
(load ≤ 1.5, no other tenant), 240 GiB free at both ends.

## Reproducing

```bash
RAYON_NUM_THREADS=1 python benchmarks/python/bench_jl_performance.py \
    --curves --workload kicked_ising_deep --pairs 3 \
    --out benchmarks/python/jl_performance/deep-kicked-ising

# re-render figures from the committed data, no measurement
python benchmarks/python/jl_performance_figures.py \
    benchmarks/python/jl_performance/deep-kicked-ising/summary.json

# the CI protocol gate, which pins this workload's mirror (no julia, < 1 s)
pytest python/paulistrings/tests/test_jl_performance_protocol.py
```

Host ccqlin038 (2 × Xeon Gold 6244, `powersave`), Julia 1.12.6 with PauliPropagation.jl 0.8.2, `PP_BACKEND=dict`,
`PP_FUSED=0`, rustc 1.94.0, driver commit `e4aeccd` with a clean tree.
