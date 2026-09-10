# SIMD evaluation — Stage 0 (ISA baseline and the scalar parity fold)

Measured 2026-09-10 on the reference host (ccqlin038, 2× Xeon Gold 6244 Cascade Lake-SP,
16c/32t, governor `powersave`), rustc 1.94.0. Working tree at `a07cc50` plus the fold
described in §3.

This is the first stage of an evaluation of whether SIMD — and specifically a word-planar
storage layout — can buy anything in the engine kernels. Stage 0 is the *control*: it
establishes what the instruction set alone is worth, because a hand-written vector kernel
must beat an autovectorized build, not a baseline one.

## 1. The finding that reframes every prior measurement: the build is SSE2

`.cargo/config.toml` sets no `target-cpu` and no `target-feature`; nothing else in the
repo does either. `rustc --print cfg` reports exactly three features:

```
target_feature="fxsr" target_feature="sse" target_feature="sse2"
```

Disassembling the shipped `phase_breakdown` release binary (327 952 instructions):

| | baseline | `-C target-cpu=x86-64-v3` |
|---|---:|---:|
| `popcnt` instructions | **0** | **154** |
| AVX (`vp*` / `ymm` / `zmm`) | 0 | present |
| SWAR popcount constants (`0x5555…`, `0x3333…`, `0x0f0f…`) | 54 | — |

So **every `count_ones()` in the codebase lowers to the ~12-operation SWAR sequence**, and
has done for every measurement recorded in `research/notes/`. The affected hot sites are
4 popcounts per word in `PauliString::mul_assign`, one per word in `weight()` and
`WeightCutoff::keep_term` (per merged row), and `bits × 2W` per term in
`Gf2Hash::bucket_of`.

`x86-64-v2` (Nehalem, 2009+) is enough to enable `popcnt`, so this costs nothing in
portability to fix — which made it look like free performance. It is not; see §2.

## 2. `-C target-cpu=x86-64-v3`: one win, one universal regression in the merge

Interleaved A/B, one binary per side built up front, `abba` order, pinned to one physical
core, `--reps 20`, `RUST_LOG` unset, quiet box. Acceptance is direction consistency across
every pair; median Δ% is the effect size. Negative = v3 faster. 7 pairs per cell.

| layer | wall Δ% | gather Δ% | sort Δ% | merge Δ% |
|---|---:|---:|---:|---:|
| `cnot` | **−3.56%** (7/7) | **−9.35%** (7/7) | ns | +6.48% (7/7) |
| `rotation_zz` | **+3.31%** (7/7) | ns | −0.86% (7/7) | **+8.71%** (7/7) |
| `trotter` | **+4.53%** (7/7) | +1.67% (7/7) | +3.16% (7/7) | **+9.08%** (7/7) |

("ns" = pairs disagree in sign, no consistent change.) Every cell is 7/7 consistent, and
per-cell spreads are ~1–2%, so these are firm.

**The mechanism is single and universal: enabling AVX2 makes `merge2_into` 6.5–9.1%
slower, in every cell, 7/7 every time.** It simultaneously makes `cnot`'s gather 9.35%
faster. The net sign per layer is then just whichever phase dominates that layer's budget:
`cnot` is gather-heavy (54%) with only 24% merge, so it wins; `rotation_zz` and `trotter`
carry ~40% merge, so they lose.

This is consistent with everything this repo has recorded about `merge2_into`: it is a
serial two-pointer walk that has repeatedly measured hypersensitive to codegen (one
`#[inline]` worth +20–34%, a segment-copy restructure +20–35%). Autovectorizing around it
is another way to perturb it, and the perturbation is reliably negative.

**Consequences.**

1. **`-C target-cpu` must not be set globally and called an improvement.** On the priority
   sparse workloads it is net *negative* on two of three layers.
2. **But the gather win is real and worth keeping.** The obvious shape is to get AVX2 into
   the gather while keeping the merge on baseline codegen — i.e. targeted
   `#[target_feature(enable="avx2")]` on the gather with the global build left alone,
   rather than a global flag. That is a concrete, data-driven Stage-1 experiment, and it
   is a much narrower change than the word-planar refactor this study set out to evaluate.
3. It also means any future SIMD measurement must report the merge phase separately. A
   global-flag A/B nets two opposing effects and can read as "no change" while both phases
   moved substantially.

Caveat on an earlier pass: a first run of this comparison was taken while a `cargo test`
was compiling on the same box and showed 35–64 ms swings for identical cells. Those
numbers were discarded and are not the ones above.

## 3. The XOR-fold: a real 2× on the function, invisible in the layer

`commutes_with`, `Gf2Hash::row_parity` and `PartitionRows::partition_of` each accumulated
`2W` full popcounts and then used only the low bit. Popcount parity is GF(2)-linear —
`popcount(a) + popcount(b) = popcount(a ^ b) + 2·popcount(a & b)`, so mod 2 the masked
words can be XOR-folded first and reduced by a **single** `count_ones`. That is `2W` → 1,
and `bucket_of` over `bits` rows goes from `bits · 2W` to `bits`. (The doc comment on
`row_parity` already described this cheaper form; only the implementation was expensive.)

Isolated microbenchmark, low-weight keys, one physical core:

| W | current ns/row | folded ns/row | speedup |
|---:|---:|---:|---:|
| 1 | 2.75 | 1.32 | **2.07×** |
| 2 | 2.27 | 2.27 | 1.00× |
| 4 | 2.56 | 2.56 | 1.00× |
| 8 | 11.45 | 3.47 | **3.30×** |
| 16 | 23.93 | 6.61 | **3.62×** |

The fold is also **ISA-insensitive** by construction: folded W=16 runs 6.61 ns/row at
baseline and 6.35 at `native`, whereas the unfolded form ranges 23.93 → 9.97. It removes
the dependence on `popcnt` rather than requiring it.

**End-to-end, however, it does nothing.** `rotation_zz`, W=1, 7 pairs, quiet box:

```
median Δ% -0.12   min -0.44   max +0.11
5/7 negative, 2/7 positive — no consistent change
```

Wall times were 37.39–37.74 ms across all 14 runs (~1% spread), so this is a tight null,
not a noisy one. `commutes_with` is simply too small a fraction of a gather that is
latency-bound, which is exactly what `2026-09-01-large-m-phase-breakdown.md` §7(3)
predicted: *"Prefetching, blocking, or a smaller per-run working set will move it; wider
arithmetic will not."*

**The fold is kept anyway**: it is strictly fewer instructions, bitwise-identical (pinned
by new tests at `W ∈ {1,2,4,8,16}` against the per-word form as an independent oracle), it
removes an ISA dependence, and it costs nothing. It is recorded here as a *null* on layer
time so nobody re-measures it expecting a win.

## 4. Method notes worth keeping

- `perf_event_paranoid == 0` on this host, so in-process `perf_event_open` counters work.
- **`rdtsc` must not be used as the cycle metric here**: `constant_tsc`/`nonstop_tsc` mean
  it counts *reference* cycles at the 3.6 GHz nominal, so under `powersave` at 1200 MHz it
  over-reports core cycles by ~3×.
- AVX-512 intrinsics, including `_mm512_mask_compressstoreu_epi64`, **compile on stable
  1.94** under `#[target_feature]`. No nightly and no crate is needed for the compress
  primitive. Host has `avx512f`+`avx512vl` but **not** `vpopcntdq`, GFNI or VBMI.
- Pinning to one physical core with `--reps 20` reduced the spread from the documented
  ±5–8% to ~1%. Do this for every single-threaded cell.
- Do not run an A/B while a build is running on the same box. It cost one full pass here.

## 5. Status and what Stage 0 implies for Stages 1–3

Stage 0's gate was: if the ISA baseline plus the scalar fold already capture most of the
plausible headroom, the expected value of the SIMD stages drops. The answer is more
interesting than either branch:

- the scalar fold captures **nothing** at the layer level, confirming latency-boundness;
- the ISA is worth **−3.6% on `cnot`** but **+3.3% on `rotation_zz`** and **+4.5% on
  `trotter`**, i.e. it is net negative on two of the three priority layers and is not a
  global win.

Neither result supports a broad SIMD program for the sparse regime, and both are
consistent with the standing verdict that gather is latency-bound and merge must not be
disturbed.

The one positive signal is `cnot`'s gather at **−9.35%, 7/7 consistent**, under AVX2 —
paired with a merge regression of +6.5 to +9.1% that is equally consistent and, on
merge-heavy layers, larger. **Isolating the first from the second is the highest-value
Stage-1 experiment**, and it is a far narrower change than the word-planar refactor this
study set out to evaluate: put `#[target_feature(enable="avx2")]` on the gather, leave the
global build at baseline so the merge keeps its current codegen, and re-run this matrix.

Still open (Stage 1): whether a word-planar layout helps, which the decisive cheap gate
`G-MERGE-1` (planar `merge2_into` ≤ +2% at W=2) should settle before any refactor.
