# Phase-1 fact sheet: where the time actually goes, per phase, versus `m`

Host ccqlin038 (reference host), 2026-09-01, single socket-pair Xeon Gold 6244 (16 physical cores / 32 threads,
1 MiB L2 per core, 24.75 MiB L3 per socket, powersave governor), rustc 1.94.0, commit `6715918`, load 0.4–1.2
throughout, box otherwise idle. Raw probe output, perf logs and provenance:
`benchmarks/results/2026-09-01-ccqlin038/` (gitignored; every number quoted here is in it).

Instrument: `cargo run --release --features phase-timing --example phase_breakdown`, extended in `6715918` with a
`su4` layer and a `--truncation` knob (see §0.2). `RUST_LOG` unset for every timed run.

This sheet answers one question — **which phase costs what, as a function of `m` and channel type** — and turns the
answer into per-experiment go/no-go evidence for Phase 2 (§7). It does not propose or benchmark any optimization.

## The culprits, in one table

| regime | dominant phase | number | where |
|---|---|---|---|
| **any `m`, sparse PTM** (`rotation_zz`, `cnot`; fanout ≤ 1.7) | `gather` **46–51 %** + `merge` **42–44 %** | 8.2–9.9 ns/gathered row; sort only 7 % | §1 |
| **any `m`, dense PTM** (`su4`; fanout 14.9) | **`sort` 58–60 %** | 16.1 ns/sorted row at best configuration | §1 |
| **large `m` (10⁶–10⁷)** | nothing new — per-term cost is **flat to ±10 % over 1000×** | only `gather` drifts, +19 % | §1 |
| **small `m` (≲ 10²)** | **`prepare`** — 61–79 % of a **1.43 µs/layer** fixed cost | serial pipeline is only **0.19 µs** | §2 |
| **mid `m` (< 8192), dense PTM, `W = 2`** | **`sort`, 2.2× its asymptote** (3.3× per sorted row) | branch-miss 2.50 % vs 0.71 %; bucket count = 1 | §3.1 |
| **any `m`, dense PTM, `W = 1`** | **`sort`, 1.9× the `W = 2` cost** for half the key bytes | 30.6 vs 16.1 ns/sorted row | §3.2 |
| **whenever `TopN` is active** | **`finalize_layer` 61–71 % of the layer** | 52–64 ns/term, flat in `m`; ~half of it `hypot` | §6 |
| **dense PTM at ≥ 16 threads** | memory controller | write traffic at **100–117 % of the write ceiling** | §4 |

Two things that are **not** culprits, against expectation: no phase is superlinear in `m` anywhere out to
`m` = 2.1 × 10⁷ (§1), and the bucketed serial pipeline — the thing the study README blamed for the small-`m`
regime loss — costs 0.19 µs per layer (§2).

---

## 0. Two methodology corrections that change what earlier small-`m` numbers mean

### 0.1 A short timed region on this host is measured at 1200 MHz, not 3600 MHz

The governor is `powersave`. The probe's default `--reps 8` makes a small-`m` cell run for **microseconds**, and the
core never leaves its idle frequency. Measured, `rotation_zz` at `m = 150`, one thread:

| `--reps` | timed region | µs/layer | ns/term |
|---|---|---|---|
| 8 | 0.25 ms | 31.78 | 143.13 |
| 100 | 3.2 ms | 32.06 | 144.40 |
| 1 000 | 12 ms | 12.26 | 55.21 |
| 10 000 | 88 ms | 8.80 | 39.64 |
| 30 000 | 262 ms | 8.74 | 39.39 |
| 100 000 | 884 ms | 8.84 | 39.81 |

**3.62×** between the two plateaus — exactly the 1200 → 3600 MHz ratio. It converges once the timed region (and
therefore the probe's equally-long warm-up call, which does the ramping) exceeds ~50 ms. `rotation_zz` at
`m = 15 000` shows the same thing: 1047.95 µs/layer at `--reps 8`, 465.52 at 100, 460.98 at 1000.

**Consequence.** Every cell in this sheet was run twice: a throwaway calibration pass at `--reps 8` to learn the
per-layer time, then the recorded pass at `reps = clamp(200 ms / per-layer, 4, 40 000)`. Any small-`m`
`phase_breakdown` number taken at the default `--reps` — including the first batch of this campaign, kept as
`small-m.log` next to the corrected `small-m2.log` — is up to 3.6× pessimistic and should not be compared against
anything. The large-`m` cells (`m ≥ 10⁵`) are unaffected: `--reps 8` there is already hundreds of ms.

### 0.2 `gu2q` = `sqrt(SWAP)` is not a matrix-gate proxy; `su4` was added

The existing `gu2q` cell is `sqrt(SWAP)`, and two measured properties disqualify it as a stand-in for the
`unitary_2q` path the cross-engine su4 curve exercises:

- **Its PTM is sparse.** Steady-state fanout is **3.65** gathered rows per input term, against **14.94** for a
  Haar-random SU(4) — i.e. it exercises about a quarter of the dense-PTM work.
- **`sqrt(SWAP)² = SWAP` is Clifford**, so repeating one block drives the term count into a **period-2 cycle**
  (10 000 → 32 503 → 10 000 … at `--n 10000`) rather than to a fixed point. `--reps` parity then changes which
  end of the cycle the probe reports as `n`, and the `m` grid comes out irregular.

`6715918` adds `--layers su4`: one fixed Haar-random SU(4) block, drawn from `circuits.py::haar_su4` under
`default_rng(0xC0FFEE)` — the same distribution `bench_jl_performance.py::su4_gates` and benchmark E use. Fanout
14.94, a true fixed point, no cycle. **`su4`, not `gu2q`, is the matrix-gate row to read below.** `gu2q` is kept
because it is what every prior committed campaign measured.

---

## 1. `m`-sweep: no phase grows superlinearly. Anywhere.

Single thread, `--qubits 128` (`W = 2`), `--truncation keep`, `m` = mean pre-layer term count
(`terms_in / layers`). ns/term = wall / `terms_in`. Every serial phase not shown (`rebucket`, `prepare`,
`span_plan`, `permute`, `unpermute`, `recount`, `finalize`) is **≤ 0.5 % of the layer** at every `m ≥ 10⁵` and is
tabulated separately in §2.

### rotation_zz (`PauliRotation`, weight-2 generator, fanout 1.67, 40 % of rows sorted)

| `m` | ns/term | gather | sort | merge | ns/gathered row (gather) | ns/sorted row |
|---|---|---|---|---|---|---|
| 14 974 | 30.55 | 13.91 | 2.58 | 13.23 | 8.36 | 3.88 |
| 149 957 | 29.33 | 13.87 | 2.06 | 13.23 | 8.33 | 3.09 |
| 450 162 | 29.23 | 13.92 | 2.02 | 13.17 | 8.35 | 3.02 |
| 1 499 667 | 29.90 | 14.53 | 2.03 | 13.20 | 8.72 | 3.05 |
| 4 499 928 | 32.31 | 16.46 | 2.07 | 13.59 | 9.88 | 3.10 |
| 15 000 086 | 30.84 | 15.41 | 2.02 | 13.27 | 9.25 | 3.03 |

### gu2q (`sqrt(SWAP)`, fanout 3.65, 73 % sorted)

| `m` | ns/term | gather | sort | merge |
|---|---|---|---|---|
| 21 252 | 70.94 | 20.52 | 29.52 | 20.35 |
| 219 848 | 62.72 | 20.63 | 21.88 | 20.01 |
| 704 201 | 61.93 | 20.87 | 21.54 | 19.38 |
| 2 124 444 | 68.12 | 23.29 | 22.56 | 22.07 |
| 6 374 655 | 67.17 | 22.50 | 22.48 | 21.94 |
| 21 249 295 | 66.35 | 21.59 | 22.30 | 22.26 |

### su4 (Haar SU(4), fanout 14.94, 93 % sorted)

| `m` | ns/term | gather | **sort** | merge | ns/sorted row |
|---|---|---|---|---|---|
| 9 884 | 426.07 | 127.25 | **269.33** | 27.79 | 19.32 |
| 99 386 | 387.01 | 128.44 | **230.63** | 27.72 | 16.54 |
| 297 402 | 383.91 | 128.58 | **227.57** | 27.57 | 16.32 |
| 989 268 | 393.35 | 130.18 | **234.62** | 28.41 | 16.83 |
| 2 967 118 | 394.87 | 136.26 | **229.93** | 28.51 | 16.50 |
| 9 891 098 | 389.83 | 133.21 | **228.50** | 27.88 | 16.39 |

### Verdict

- **Per-term cost is flat in `m` to ±10 % over three decades** — `rotation_zz` **29.2–32.3** ns/term over
  `m` = 1.5 × 10⁴ → 1.5 × 10⁷ (1000×), `gu2q` **61.9–70.9** over 2.1 × 10⁴ → 2.1 × 10⁷ (1000×), `su4`
  **377.9–426.1** over 9.9 × 10³ → 9.9 × 10⁶ (1000×), where the 426.1 at `m` = 9 884 is the tail of the §3.1
  small-`m` cliff and everything from `m` = 3 × 10⁴ up sits in 377.9–394.9. There is **no superlinear phase and no
  large-`m` cliff**. This is an independent, direct confirmation of the deep-kicked-Ising cross-engine verdict
  (`benchmarks/python/jl_performance/deep-kicked-ising/README.md`), now from inside the engine.
- **The only phase that moves at all with `m` is `gather`**, and by little: `rotation_zz` 13.87 → 16.46 ns/term
  (+19 %) between `m` = 1.5 × 10⁵ and 4.5 × 10⁶, then back to 15.41 at 1.5 × 10⁷; `su4` 128.4 → 136.3 (+8 %).
  `sort` and `merge` are flat to ±3 % everywhere. §4 shows `gather`'s drift is LLC-miss-rate, not bandwidth.
- **Dominant phase depends on the channel's fanout, not on `m`:**

  | channel class | fanout | gather | sort | merge |
  |---|---|---|---|---|
  | `rotation_zz` / `cnot` (sparse PTM) | 1.0–1.7 | **46–51 %** | 7 % | **42–44 %** |
  | `gu2q` (`sqrt(SWAP)`) | 3.65 | 33 % | **33 %** | 32 % |
  | `su4` (dense 16×16 PTM) | 14.94 | 33 % | **58–60 %** | 7 % |

  Symbol-level profiles agree (`perf-report.txt`): `su4` at `m ≈ 10⁶` is `drift::sort` 46.4 % +
  `sort_rows_with_scratch` 10.7 % + `gather_local_output_major` 32.2 % + `fill_coset` (merge) 6.6 %;
  `rotation_zz` at `m ≈ 10⁵` is `gather_local_input_major` 48.1 % + `fill_coset` 42.7 % + sort 6.5 %.

- **RSS.** No memory step, and no cell was memory-limited: `rotation_zz` 1.48 GB peak at `m` = 1.5 × 10⁷ (99
  B/term), `su4` 1.65 GB at 9.9 × 10⁶ (167 B/term), `gu2q` 3.3 GB at 2.1 × 10⁷ (157 B/term). No cell was skipped
  for memory or time; §6 lists the only skips.

---

## 2. Small-`m` fixed-cost anatomy: 1.4 µs per layer, 70 % of it `prepare`

Per-layer wall time as `m → 0`, one thread, `W = 2`, 128 qubits, `--reps` auto-scaled per §0.1
(`fixed-cost.jsonl`; the `m → 0` row is the smallest cell the layer admits, at which the term-proportional work is
≤ 0.6 µs).

| layer | µs/layer at `m → 0` | `prepare` | `rebucket` | `span_plan` | `permute`+`unpermute` | `recount`+`finalize` | coset-loop intercept |
|---|---|---|---|---|---|---|---|
| `rotation_zz` (2q `PauliRotation`) | **1.432** (m=2) | 1.001 (70 %) | 0.017 | 0.058 | 0.037 | 0.054 | ~0.17 |
| `cnot` (`Clifford2Q`) | **1.468** (m=2) | 1.015 (69 %) | 0.018 | 0.051 | 0.037 | 0.051 | ~0.20 |
| `gu2q` (`GeneralUnitary2Q`, sqrt-SWAP) | **2.936** (m=4) | 2.307 (79 %) | 0.017 | 0.061 | 0.039 | 0.051 | ~0.32 |
| `su4` (`GeneralUnitary2Q`, Haar) | **~5.4** (see note) | 4.188 (78 %) | 0.017 | 0.072 | 0.041 | 0.052 | ~1.0 |
| `depolarizing` (rescale fast path) | **0.535** (m=2) | 0.326 (61 %) | 0.017 | 0 | 0 | 0.031 | ~0.10 |

`su4`'s smallest admissible cell is `m = 30` (its closed key set is ~14× the seed), where the wall is 17.14 µs and
the term-proportional work at its 390 ns/term asymptote is already 11.7 µs; the fixed part is therefore
4.188 (`prepare`) + 0.18 (serial phases) + ~1.0 (coset-loop entry) ≈ **5.4 µs**, less precisely resolved than the
other rows.

Three things this settles:

1. **The bucketed serial pipeline is not the fixed cost.** `rebucket → span_plan → permute → coset-loop entry →
   unpermute → recount → finalize` together cost **0.16–0.19 µs per layer** and are *independent of both `m` and the
   channel* (`rebucket` 0.017 µs, `span_plan` 0.051–0.072, `permute`+`unpermute` 0.037–0.041,
   `recount`+`finalize` 0.051–0.054 — the same to three digits across five channel types and six decades of `m`).
   The study README's reading — "a bucketed rebucket → permute → coset-loop → unpermute pipeline whose per-layer
   cost is nearly independent of the term count" — is *true* but the quantity is **0.19 µs**, not something that
   could produce a 1.6–3.1× deficit.
2. **`prepare` is the fixed cost.** 61–79 % of it for every channel with a local PTM, and 95 % of `su4`'s. It is
   `Channel::prepare` — PTM derivation and delta-plan construction, genuinely O(1) in the term count. For the
   general two-qubit unitary it is **2.31 µs (sparse PTM) to 4.19–5.71 µs (dense PTM)** *per gate*: 256 4×4
   complex triple products plus traces. A circuit of distinct SU(4) blocks pays it once per block and cannot cache
   it.
3. **Break-even `m`** (fixed cost = term-proportional work, at each layer's own asymptotic ns/term):
   `rotation_zz` **49 terms** (1.43 µs / 29.2 ns), `cnot` **35** (1.47 / 42.4), `gu2q` **44** (2.94 / 66.4),
   `su4` **14** (5.4 / 390). At `m = 1497`, `rotation_zz`'s fixed cost is **3.3 %** of the layer (1.43 of 42.83 µs).

So above ~10² terms the fixed cost cannot be the small-`m` regime loss. The measured per-term cost on the rotation
path is *already flat*: 41.2 ns/term at `m = 150`, 38.8 at 513, 28.6 at 1497, 30.5 at 1.5 × 10⁴, 29.0–29.4 above.
**On the rotation path the small-`m` loss is Julia being cheaper per term at 10²–10⁴ terms, not us being more
expensive than our own asymptote.** The only exception is the matrix-gate path, which does have a genuine
small/mid-`m` per-term penalty — §3.

---

## 3. The matrix-gate mid-`m` deficit: a configuration cliff in the sort, and it is `W`-specific

### 3.1 At `W = 2` there is a 2.2× cliff below 8192 terms, entirely in `sort`

`su4`, one thread, 128 qubits, `--reps` auto-scaled. `B` = bucket count = runs per layer (`su4-fine.jsonl`):

| `m` | `B` | rows/run | ns/term | `sort` | `gather` | `merge` | **ns/sorted row** |
|---|---|---|---|---|---|---|---|
| 980 | 1 | 14 630 | **841.2** | 732.7 | 71.9 | 30.1 | **52.5** |
| 2 002 | 2 | 14 966 | 806.7 | 702.0 | 71.5 | 29.5 | 50.3 |
| 3 976 | 4 | 14 854 | 657.1 | 553.1 | 73.7 | 28.4 | 39.7 |
| 7 868 | 8 | 14 686 | 535.4 | 379.7 | 126.0 | 26.2 | 27.3 |
| 9 884 | 128 | 1 153 | 429.8 | 271.7 | 128.3 | 28.2 | 19.5 |
| 15 806 | 128 | 1 844 | 380.8 | 226.8 | 125.8 | 27.3 | 16.3 |
| ≥ 31 696 | 128–16 384 | 3 700–14 900 | **377.9–394.9** | 224–235 | 126–136 | 27–29 | **16.1–16.8** |

**2.2× on the layer, 3.3× per sorted row**, and it is not run length: rows/run is ~14 700 at `B` = 1, 2, 4 and 8
alike (because `m` grows with `B`), yet ns/sorted-row falls 52.5 → 27.3; and at `B ≥ 128` ns/sorted-row is a flat
16.1–16.8 for rows/run anywhere from 1 153 to 14 900. It tracks `B`.

`B` is set by `bucket/sum.rs::desired_bits`: the `worth_splitting` floor requires
`len ≥ DEFAULT_MIN_BUCKETS × MIN_TERMS_PER_TASK = 128 × 64 = 8192`; below that `B` is only
`ceil(len / DEFAULT_TARGET_BUCKET_LEN=1024)` rounded up to a power of two, so `m ≤ 1024` gets a single bucket.
**8192 terms is exactly where the cliff is.**

**Mechanism is branch prediction, not cache.** `perf stat` at fixed `--reps 2000` (whole-process; the warm-up call
matches the timed one, so per-row figures are over `2 × rows_gathered`):

| `m` | `B` | branch-miss rate | insn / gathered row | IPC | cycles / gathered row | LLC load-miss |
|---|---|---|---|---|---|---|
| 980 | 1 | **2.50 %** | 535 | 2.14 | 250 | 0.88 % |
| 7 868 | 8 | 0.81 % | 439 | 2.82 | 156 | 0.40 % |
| 15 806 | 128 | 0.71 % | 329 | 2.93 | 112 | 0.43 % |

Branch misses per gathered row: 2.50 → 0.66 → 0.44 (**5.8×**). LLC load-miss rate is under 1 % throughout — the
whole `su4` run is cache-resident, so **capacity is not the mechanism**. The cycle ratio (2.23×) factors cleanly
into 1.63× more instructions and 1.37× worse IPC: the stable sort does more comparisons *and* mispredicts more when
a run draws its rows from few source buckets. `sort_rows_with_scratch` sorts a `Vec<u32>` permutation through an
indirect comparator, so comparison count and misprediction are the whole cost.

### 3.2 At `W = 1` the cliff does not exist — and the asymptote is 1.9× worse

One-qubit flip, `q = 64` (`W = 1`) against `q = 65` (`W = 2`), same everything else, runs adjacent in time
(`width.jsonl`):

| layer | fanout | `W` | `m` | `B` | ns/term | ns/sorted row |
|---|---|---|---|---|---|---|
| `su4` | 14.94 | 1 | 980 | 1 | 519.5 | 30.6 |
| `su4` | 14.94 | 2 | 980 | 1 | 840.8 | 52.6 |
| `su4` | 14.94 | 1 | 63 364 | 128 | **578.7** | **30.9** |
| `su4` | 14.94 | 2 | 63 518 | 128 | **378.6** | **16.1** |
| `gu2q` | 3.65 | 1 | 100 140 | 256 | 60.5 | 10.1 |
| `gu2q` | 3.65 | 2 | 100 007 | 256 | 63.0 | 8.3 |
| `cnot` | 1.00 | 1 | 47 000 | 128 | 33.7 | 10.4 |
| `cnot` | 1.00 | 2 | 47 000 | 128 | 42.4 | 12.4 |
| `rotation_zz` | 1.67 | 1 | 100 374 | 128 | 23.7 | 3.2 |
| `rotation_zz` | 1.67 | 2 | 100 189 | 128 | 29.2 | 3.0 |

At `W = 1` the `su4` sort is a flat ~30.5 ns/sorted row at *every* `B` (confirmed at `q` = 36, 64 and, per
`su4-w1.jsonl`, across `m` = 980 → 283 074: ns/term 510–586, no trend). At `W = 2` it is 16.1 once `B ≥ 128` and
52.5 at `B = 1`. The `W = 1` penalty scales with duplicate-key density: 1.90× at fanout 14.94, 1.22× at 3.65,
~1.0× at 1.67, and *reversed* (0.84×) at fanout 1.0.

So the honest framing is not "small `B` is slow" but: **the stable sort has a fast regime worth ~16 ns/row on
high-duplicate matrix-gate runs, and whether a configuration reaches it depends on `B` *and* on `W` — a 3.3×
spread (16.1 / 27.3 / 30.6 / 52.5 ns/row) driven purely by configuration, at identical algorithmic work.** The
`W = 1` half of that is a defect on its face: `W = 1` moves half the key bytes and is 1.9× slower per row. The
mechanism there is not identified (the comparator is the derived `Ord` for `[u64; W]`; a codegen difference between
`N = 1` and `N = 2` is the obvious suspect and is cheap to A/B).

### 3.3 What this does and does not say about the cross-engine su4 deficit

The committed su4 curve (`benchmarks/python/jl_performance/su4-curve/README.md`) is **`n = 36` qubits, i.e.
`W = 1`**, 105 channels, Heisenberg. Its deficit is created in one step, 193 → 7089 final terms, our time ×17.2
against Julia's ×10.7.

At `W = 1` our measured per-term cost is **flat** across that whole band (§3.2). Both engines process
parity-verified identical per-layer term counts, so Σ`m` is the same for both; a flat `c_ours` means Σ`m` itself
grew ×17.2, and Julia's per-term cost therefore **fell 17.2 / 10.7 = 1.61×** over the step. Julia got better; we
did not get worse.

**Answer to "does the mid-`m` deficit correlate with a specific phase of ours?" — No. It is uniform.** At the
curve's own width every phase's per-term cost is flat, so the deficit is created on Julia's side. The
cache-residency-of-the-dictionary reading survives: Julia's cost per term is non-monotone in `m` (it dips into its
best regime around 10⁴ terms — where its ratio is 0.620 — and rises again by 8.5 × 10⁴, where the ratio returns to
1.027), while ours does not move.

**But** §3.1 is a real, separate `W = 2` opportunity, not a restatement of the same thing: at 65–128 qubits — the
width every rotation workload in the study uses, and the width a 127-qubit matrix-gate circuit would use — the
matrix-gate path costs 2.2× its asymptote below 8192 terms. Nothing in the committed study measured that corner.

---

## 4. Roofline: one phase is bandwidth-bound, and only above 8 threads

Model per `benchmarks/PROFILING.md` §Roofline at `T = 48` (`W = 2`):
`bytes/layer = m·T + 2(rows_gathered − rows_id)·T + 2·rows_id·16 + 2·rows_sorted·T + terms_out·T`, giving **256
B/term** (`rotation_zz`), **636** (`gu2q`), **2804** (`su4`). Ceilings from
`research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`. Measured DRAM is the uncore IMC pass of
`scripts/perf-stat.sh` (idle baseline subtracted).

### Single thread — ceilings: read 11.3, copy 9.5, triad 11.3 GB/s

| cell | `m` | model GB/s | % of copy ceiling (model) | **measured DRAM** | **% of copy (measured)** | LLC load-miss | IPC | verdict |
|---|---|---|---|---|---|---|---|---|
| `rotation_zz` | 10⁶ | 8.7 | 92 % | **2.46** | **26 %** | 28.4 % | 1.87 | latency-bound |
| `rotation_zz` | 3 × 10⁶ | 8.5 | 89 % | **3.86** | **41 %** | 36.1 % | 1.48 | latency-bound |
| `gu2q` | 3 × 10⁶ | 9.5 | 100 % | **2.60** | **27 %** | 32.4 % | 2.02 | latency-bound |
| `su4` | 10⁶ | 7.1 | 75 % | **0.61** | **6 %** | 6.5 % | 2.73 | **compute-bound** |

The model over-counts by **2.2–11.7×** at one thread because most of the modelled traffic is served from cache —
exactly the reading PROFILING.md prescribes. **Do not call any single-threaded phase bandwidth-bound.**
`rotation_zz` and `gu2q` at `m ≥ 10⁶` sit at 26–41 % of the mixed-traffic ceiling *with* a 28–36 % LLC load-miss
rate and IPC falling 1.87 → 1.48: that is a **latency / working-set** problem in `gather`, which is also the only
phase whose ns/term grows with `m` (§1). `su4` is unambiguously **compute-bound** at one thread (0.6 GB/s, LLC miss
6.5 %, IPC 2.7).

### Threads — the dense-PTM path saturates the write path at 16 threads

`su4` at `m ≈ 10⁶`, `--reps 40` (input generation diluted to noise), uncore IMC summed over both sockets. Ceilings
from the fact sheet, both sockets: **16 physical → read 45.0, write 25.3**; **32 threads → read 48.8, write 23.1**.

| threads | ns/term | speedup vs 1t | IPC | LLC load-miss | measured read | measured write | **% of read ceiling** | **% of write ceiling** |
|---|---|---|---|---|---|---|---|---|
| 1 | 393.4 | 1.0× | 2.73 | 6.5 % | — (0.61 total) | — | 5 % | — |
| 8 | 69.3 | 5.7× | 2.20 | 45.0 % | 23.37 | 15.90 | 52 % | 63 % |
| 16 | **57.7** | **6.8×** | 1.39 | 47.1 % | 36.79 | **25.19** | 82 % | **100 %** |
| 32 | 62.7 | 6.3× | 0.70 | 45.4 % | 35.08 | **26.94** | 72 % | **117 %** |

**`su4` is bandwidth-bound from 16 threads up.** Its write stream is at **100 % of the measured write ceiling at 16
threads and 117 % at 32** (where the ceiling itself is lower), with reads at 72–82 % — both over the 70 % bar. And
**32 threads buy zero extra bandwidth (62.0 vs 62.0 GB/s total) while costing 8 % of wall time** (57.7 → 62.7
ns/term) as IPC halves, 1.39 → 0.70. Measured traffic *exceeds* the model (61.8 against a modelled 48.6 at 16
threads, 44.8 at 32), so the sort moves ~1.3–1.4× more than the two passes the model charges it.

The transition is between 1 and 8 threads, and it is a cache transition: at fixed `m` the LLC load-miss rate goes
**6.5 % → 45.0 %** from 1 to 8 threads and then stays flat (45.0 / 47.1 / 45.4 at 8 / 16 / 32). Hyperthreading is
therefore *not* the cause — 16 threads (one per physical core, full 1 MiB L2 each) is already saturated.

A candidate mechanism, **not verified here**: `DEFAULT_TARGET_BUCKET_LEN = 1024` is justified in-source by "a bucket
plus its gather scratch stays L2-resident: 1024 terms is ~48 KB against 1 MiB of L2", but the object that has to
stay resident is the **gather run**, which at fanout 14.94 is `14.94 × 1024 × 48 B ≈ 737 KB` — one per live coset.
At 1 thread one such run plus its slice of the 48 MB source sum has the whole 24.75 MB L3 to itself; at 8+ threads
eight or more independent runs stream from scattered parts of the same 48 MB sum. Whether the binding constraint is
L2 per run or L3 against the source sum is a separate measurement (a `--bucket-target` knob would settle it), and
Phase-2 experiment (2) should settle it before tuning anything.

`rotation_zz` and `gu2q` do not have this problem. At 32 threads and `m ≈ 3 × 10⁶` they measure
**read 4.02 / write 4.70** and **read 5.24 / write 5.27 GB/s** — 8–11 % of the read ceiling and 20–23 % of the
write ceiling, against modelled 96.8 and 121.1 GB/s, i.e. ~91 % of modelled traffic cache-served. Their poor
32-thread IPC (0.93, 1.14) with a 30–34 % LLC miss rate says latency, not bandwidth. Speedups 1 → 8 → 32 threads:
`rotation_zz` **7.2–7.3× → 11.3–13.6×**, `gu2q` **7.3–7.9× → 12.8–13.3×** (range over `m` = 1 × 10⁶ and
3 × 10⁶). Parallel efficiency (`Σbusy / (coset_loop × threads)`) stays 0.87–0.99 in every cell measured, so none of
this is load imbalance — it is per-thread slowdown.

---

## 5. Perf counters, full grid (one thread)

`scripts/perf-stat.sh`, `--reps` per §0.1 (`perf-stat.log`). `cycles/string` is whole-process and converges from
above; use it for same-`--reps` ratios only.

| layer | `m` | IPC | LLC load-miss | branch-miss | attributable DRAM |
|---|---|---|---|---|---|
| `rotation_zz` | 1 004 | 2.79 | 50.0 % | 0.70 % | 0.01 GB/s |
| `rotation_zz` | 10 008 | 2.42 | 15.8 % | 1.51 % | 0.00 |
| `rotation_zz` | 100 189 | 2.42 | 14.2 % | 1.39 % | 0.08 |
| `rotation_zz` | 1 001 633 | 1.87 | 28.4 % | 1.81 % | 2.46 |
| `rotation_zz` | 2 999 955 | 1.48 | 36.1 % | 2.30 % | 3.86 |
| `gu2q` | 1 002 | 2.79 | 37.2 % | 0.94 % | 0.09 |
| `gu2q` | 9 978 | 2.63 | 12.8 % | 1.09 % | −0.10 |
| `gu2q` | 100 007 | 2.67 | 19.5 % | 0.85 % | 0.12 |
| `gu2q` | 998 081 | 2.22 | 28.7 % | 1.05 % | 2.21 |
| `gu2q` | 2 995 766 | 2.02 | 32.4 % | 1.17 % | 2.60 |
| `su4` | 980 | 2.14 | 1.0 % | 2.51 % | −0.12 |
| `su4` | 9 945 | 2.81 | 1.2 % | 0.98 % | 0.02 |
| `su4` | 100 108 | 2.85 | 3.1 % | 0.62 % | 0.14 |
| `su4` | 996 292 | 2.73 | 6.5 % | 0.66 % | 0.61 |

Highlights: the two engines-within-the-engine are cleanly separated. The **sparse-PTM path degrades in memory**
(LLC miss 14 % → 36 %, IPC 2.42 → 1.48, DRAM 0.08 → 3.86 GB/s from `m` = 10⁵ to 3 × 10⁶). The **dense-PTM path
degrades in branches** (LLC miss stays 1–6.5 %, IPC 2.7–2.85, and the only bad cell is `m = 980` where branch-miss
jumps to 2.51 %). The `m ≈ 10³` cells' high LLC-miss *rate* with near-zero absolute DRAM is a small-denominator
artifact (very few LLC loads), not a signal.

Flamegraphs: `flamegraph-probe-gu2q-6715918.html` (regenerated per cell, so only the last survives; the quotable
form is `perf-report.txt`, §1). At `--reps 8` the input-generation frames (`BuildAccumulator::add_term/finalize`,
`refine_bucket`, ~11 % combined) are visible and must be ignored; the `--reps 3000` gu2q profile has none.

---

## 6. `TopN` costs more than the entire rest of the layer

`--truncation` comparison on `rotation_zz`, one thread, `W = 2` (`topn.jsonl`, `topn-mt.jsonl`).
`CoefficientThreshold` has no `finalize_layer` at all, so it is the zero-finalize control.

| policy | `m` | ns/term | `gather` | `sort` | `merge` | **`finalize`** | finalize % of wall |
|---|---|---|---|---|---|---|---|
| `keep` | 100 189 | 28.98 | 13.73 | 2.01 | 13.09 | 0.00 | 0 % |
| `keep` | 1 001 633 | 29.43 | 14.21 | 1.99 | 13.12 | 0.00 | 0 % |
| `keep` | 2 999 955 | 29.97 | 14.58 | 2.03 | 13.21 | 0.00 | 0 % |
| `coeff:0.0` | 100 189 | **40.78** | 13.89 | 2.06 | **24.67** | 0.00 | 0 % |
| `coeff:0.0` | 1 001 633 | **41.45** | 14.51 | 2.05 | **24.76** | 0.00 | 0 % |
| `coeff:0.0` | 2 999 955 | **44.40** | 17.03 | 2.04 | **25.16** | 0.00 | 0 % |
| `topn:100000` | 100 000 | **85.73** | 14.51 | 1.60 | 17.01 | **52.44** | **61.2 %** |
| `topn:1000000` | 1 000 000 | **91.12** | 15.27 | 1.59 | 17.98 | **56.12** | **61.6 %** |
| `topn:3000000` | 3 000 000 | **100.15** | 15.63 | 1.65 | 18.50 | **64.17** | **64.1 %** |

Thread scaling of `TopN` at `m = 10⁶` (its `finalize_layer` is `par_iter` throughout):

| threads | ns/term | `finalize` ns/term | finalize % of wall | speedup of finalize |
|---|---|---|---|---|
| 1 | 90.47 | 55.77 | 61.6 % | 1.0× |
| 8 | 20.11 | 13.81 | 68.7 % | 4.0× |
| 32 | 15.63 | 11.09 | **70.9 %** | 5.0× |

### Verdict

- **`TopN` selection is a big constant, not superlinear.** 52.44 → 64.17 ns/term is **+22 % over a 30× `m` range**
  — sub-logarithmic, consistent with three O(`m`) passes plus an O(`m`)-expected `select_nth_unstable`. Nothing here
  motivates an asymptotically better selection algorithm; the constant is the target.
- **The constant is enormous:** `finalize_layer` is **1.6–1.8× the cost of the entire rest of the layer** and
  **61–71 % of layer wall time** at every `m` and every thread count. It parallelizes worse than the coset loop
  (5.0× vs 11–14× on 32 threads), so its share *rises* with threads.
- **A named, measured sub-cause: `Complex64::norm()` is `hypot`.** `num-complex`'s `norm()` is
  `self.re.hypot(self.im)` (`src/lib.rs:217`) — a libm call. The `coeff:0.0` control isolates exactly one
  `keep_term` doing one `c.norm() > t` per merged term, and it costs **+11.8 to +14.4 ns/term, all of it in the
  merge** (13.1 → 24.7–25.2, i.e. merge cost nearly doubles) — a 42–52 cycle penalty per call at 3.6 GHz, which is
  what `hypot` costs. `TopN::finalize_layer` calls `.norm()` **twice per candidate** (once building `mags`, once in
  the compaction predicate), so **~24–29 of its ~56 ns/term is `hypot`**. It also allocates and fills a fresh
  `Vec<f64>` of the candidate count every layer — 12 MB at `m` = 1.5 × 10⁶.

---

## 7. Implications for the Phase-2 experiments

Verdicts are against the evidence above only.

### (1) `TopN` histogram selection — **strongly supported, but re-scope it**

`finalize_layer` is 61–71 % of any layer running `TopN` (§6). The evidence says the win is in the *constant*, not
the selection algorithm, and it ranks the sub-wins:

- **(1a) Replace both `.norm()` calls with `.norm_sqr()` against `t²`.** Directly measured worth: the identical
  substitution in `CoefficientThreshold::keep_term` accounts for 11.8–14.4 ns/term of merge cost, and `TopN` makes
  two such calls per candidate — so ~24 of 56 ns/term, ~40 % of finalize. This is *not* a histogram change; it is
  a one-line change with a much better evidence-to-risk ratio than the rest. **Do it first and separately.**
  Correctness rider: `TopN`'s documented tie rule turns on *exact* `f64` equality at the threshold. Squaring is
  monotone but not exactly representable, so the tie-group semantics must be re-derived on squared magnitudes (or
  the threshold compared in squared space throughout) before this is a free win. That is a design question, not a
  perf one.
- **(1b) Stop materializing the magnitude `Vec`** (12 MB/layer at `m` = 1.5 × 10⁶).
- **(1c) Exponent-histogram selection replacing `select_nth_unstable`.** Supported, but it is the *smallest* of
  the three: after (1a) and (1b) the remaining budget is ~2 linear passes plus the select.

Also worth fixing while there: `CoefficientThreshold` itself costs **+40–48 % of total layer time** versus
`keep` (§6). Every Python caller passing `min_abs_coeff` pays that, and (1a) applies verbatim.

### (2) Bucket count tuning — **strongly supported, narrowly**

Supported **only** for the dense-PTM path at `W = 2` and `m < 8192`, where it is worth **2.2× on the layer /
3.3× per sorted row** (§3.1), and the trigger is identified exactly: `desired_bits`'s `worth_splitting` floor at
`DEFAULT_MIN_BUCKETS × MIN_TERMS_PER_TASK = 8192`. Two testable forms: lower the floor, or make
`DEFAULT_TARGET_BUCKET_LEN` fanout-aware — its own doc comment justifies 1024 by L2 residency of *a bucket plus
its gather scratch* (48 KB), but the object that must stay resident is the **gather run**, which at fanout 14.94 is
737 KB (§4). Adding a probe/API knob for the target and the min-bucket floor is a prerequisite for this experiment
and would also settle the open L2-vs-L3 question in §4.

Evidence explicitly does **not** support it for:
- `rotation_zz` / `cnot` at any `m` — `sort` is 7 % of those layers, so even a 3× sort win is ≤ +14 %.
- any channel at `m ≥ 10⁴` — per-sorted-row cost is flat 16.1–16.8 ns from rows/run 1 153 to 14 900 (§3.1).
- the cross-engine su4 curve's numbers — that curve is `W = 1`, where the cliff is absent (§3.2, §3.3). Do not
  project this experiment onto the 0.620 ratio.

### (3) SIMD kernels — **partially supported, and the obvious target is the wrong one**

- **`gather` (46–51 % of sparse-PTM layers, 33 % of dense): do not vectorize first.** It runs at 8.2–9.9 ns per
  gathered row single-threaded with a 14–36 % LLC load-miss rate and IPC falling to 1.48 (§4, §5). It is
  latency-bound on scattered source reads. Prefetching, blocking, or a smaller per-run working set will move it;
  wider arithmetic will not.
- **The sort (58–60 % of dense-PTM layers): the prize, but as a replacement, not a vectorization.** Its cost is
  comparison count and branch misprediction through an indirect `Vec<u32>` permutation comparator (§3.1), and its
  per-row cost varies **3.3× with configuration at identical work** (16.1 / 27.3 / 30.6 / 52.5 ns/row across
  `B` and `W`). A key-only radix sort would remove the comparator, the branches and the fragility together. Read
  the `#[inline]` comments in `engine/merge.rs` first — the code layout there is A/B-verified load-bearing in both
  directions.
- **Unplanned but cheapest: the `W = 1` sort is 1.9× slower per row than `W = 2` on high-duplicate runs** (§3.2),
  reproduced across a one-qubit `q = 64` → `q = 65` flip and confirmed at `q = 36`. `W = 1` moves half the key
  bytes and does strictly less work, so this is a defect, and `W = 1` covers every workload up to 64 qubits in the
  PyO3 dispatch. Mechanism unidentified; the derived `Ord` for `[u64; 1]` versus `[u64; 2]` is the suspect and the
  A/B is one afternoon.
- **Do not vectorize the merge.** It is a serial two-pointer walk at 1.8–8.2 ns/gathered row, and
  `engine/merge.rs` already records that restructuring it into gallop + segment copies cost +20–35 %.

### (4) Dictionary / hybrid path for small `m` and saturated regimes — **supported for small `m` with a different rationale; not evidenced for saturated**

The stated rationale ("our fixed per-layer pipeline costs nearly the same at 10² terms as at 10⁴") is **not
supported**: the serial pipeline is 0.19 µs/layer, the whole fixed cost is 1.43 µs (70 % of it `prepare`, which a
dictionary engine also pays in some form), and it breaks even at `m = 48` and is 3.3 % of the layer by `m = 1497`
(§2). A hybrid cannot win by removing the pipeline.

What a hybrid *would* remove is the **sort**, and that is worth something precisely where §3 says: the dense-PTM
path below 8192 terms at `W = 2` (2.2×) and at any `m` at `W = 1` (1.9× versus its own `W = 2` sibling). Re-scope
the experiment around that, and note that (2) and (3) attack the same cost with far less new machinery.

For the **saturated / near-closed** regime this breakdown measured nothing: per-phase costs are flat to ±10 % out to
`m` = 2.1 × 10⁷ and no phase misbehaves (§1). The no-new-keys merge fast path remains justified only by the
cross-engine note, not by anything here.

### (5) Memory-step smoothing — **not supported by this data**

No memory step and no memory-limited cell: 99 B/term at `m` = 1.5 × 10⁷ (`rotation_zz`), 167 at 9.9 × 10⁶
(`su4`), 157 at 2.1 × 10⁷ (`gu2q`) (§1). Per-phase timings are `m`-independent to ±10 %, so nothing here says the
power-of-two capacity slack costs *time*. If it is done, it is for peak-RSS reasons — already bounded at ≤ 1.5×
worst case in `2026-09-01-large-m-campaign-log.md` — and should not be sold as a throughput experiment.

### (6) Merge / finalize traffic — **implicated as arithmetic, not as traffic**

`merge` is 42–44 % of the sparse-PTM layer, at 7.9–8.2 ns per gathered row. Its modelled traffic is ~91 %
cache-served at 32 threads (§4), so **traffic is not the target**. The one measured merge win is arithmetic: the
`hypot` in `keep_term` (§6, item 1a), worth 11.8–14.4 ns/term to every caller who passes a coefficient threshold.
`finalize` traffic matters only when `TopN` is active, where it is item (1).

### (7) Unplanned, and it gates every other multi-thread measurement

**`su4` is bandwidth-saturated from 16 threads and 32 threads is net harmful** (57.7 → 62.7 ns/term; write traffic
at 100–117 % of the measured write ceiling at both; §4). Any matrix-gate throughput experiment measured at 32 threads will be
measuring the memory controller, not the change. Measure the dense-PTM path at ≤ 8 threads, or at 16 with the
bandwidth number quoted alongside.

---

## 8. Reproduction

```bash
# per-cell auto-reps is mandatory below m ~ 1e5 (§0.1): reps = 200ms / per-layer-time
cargo build --release --features phase-timing -p paulistrings --example phase_breakdown

# §1 m-sweep
./target/release/examples/phase_breakdown --n 10000   --qubits 128 --threads 1 --layers rotation_zz --reps 186
./target/release/examples/phase_breakdown --n 210000  --qubits 128 --threads 1 --layers su4         --reps 4
# §3.1 the W=2 cliff
for n in 70 140 280 560 1120 4480; do ./target/release/examples/phase_breakdown \
    --n $n --qubits 128 --threads 1 --layers su4 --reps 400 --format json; done
# §3.2 the width flip
for q in 64 65; do ./target/release/examples/phase_breakdown \
    --n 4480 --qubits $q --threads 1 --layers su4 --reps 400 --format json; done
# §6 TopN against its zero-finalize control
./target/release/examples/phase_breakdown --n 668000  --threads 1 --layers rotation_zz --reps 8 --truncation coeff:0.0
./target/release/examples/phase_breakdown --n 1000000 --threads 1 --layers rotation_zz --reps 8 --truncation topn:1000000
# §4, §5 counters
scripts/perf-stat.sh --n 70500 --qubits 128 --threads 16 --layers su4 --reps 40 --format json
```
