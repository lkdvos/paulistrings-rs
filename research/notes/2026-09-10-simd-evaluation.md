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

> **Retracted 2026-09-10 — see `2026-09-10-hot-path-code-size.md` §5.** The universal
> `merge2_into` regression below was DSB eviction, not the ISA. Re-measured with JCC padding
> applied to both sides it **disappears in all three cells**, and `x86-64-v3` becomes a
> consistent win on every layer (−1.19 / −4.85 / −2.14, 7/7). The consequences drawn here do
> not follow.

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

## 3a. What actually dominates each phase, at instruction level

`perf annotate` on the baseline (`rotation_zz`, W=2, 128 qubits, 1 thread). Symbol shares
of the whole run: `gather_local_input_major` 39.3%, `fill_coset` 33.8% (carries the merge),
sort family ~10% (`quicksort` 4.9% + `sort_rows_with_scratch` 3.5% + `drift::sort` 1.8%).

**Gather (39.3%)** — no single dominant op; five roughly equal costs:

| % of fn | instruction | what it is |
|---:|---|---|
| 6.1 | `mov (%rcx,%rax,1),%rdi` | indexed load of the source key word |
| 6.1 | `mulpd %xmm2,%xmm4` | the complex multiply `coeff * amp` — already SSE2-vectorized |
| 5.9 | `jp` | parity flag after `ucomisd`: the **`if a == ZERO` amplitude test** |
| 5.7 | `movapd %xmm4,%xmm0` | shuffling the complex value |
| ~6 | `shr %cl` / `and $0x8` / `bt %rcx,%rdi` | `support_bits` extraction |
| 2.0 | `movups %xmm6,(%rax,%rcx,1)` | the scattered store into the run |

Worth noting the amplitude zero-test alone is ~6% of the largest phase, and the complex
multiply is *already* vectorized at baseline SSE2 — there is no scalar FP to speed up.

**Merge (33.8%, inlined into `fill_coset`)** — one branch dominates:

| % of fn | instruction | what it is |
|---:|---|---|
| **8.7** | `jne` | the data-dependent two-pointer advance — the misprediction hotspot |
| 6.8 | `mov %rbp,0x8(%rax,%rcx,1)` | writing the key's second word to `dst` |
| 5.9 | `movupd (%rax),%xmm2` | loading a coefficient |
| 4.5 | `mov %r13,(%rax,%rcx,1)` | writing the key's first word |
| 1.8 | `addpd %xmm0,%xmm2` | the equal-key coefficient sum |
| ~4 | `setb %cl` / `cmp` / `test $0x1,%cl` | materializing the `take_a` decision |

A single conditional branch at 8.7% is the merge's real cost, which is exactly why it does
not vectorize and why the "do not restructure" verdict holds.

**Sort (~10%)** — the dependent indexed load, as the sort-kernel note argued:

| % of fn | instruction | what it is |
|---:|---|---|
| 13.2 | `movdqu %xmm0,(%r11)` | permutation-gather store of a 16-byte key |
| **11.8** | `mov (%r12,%r10,4),%ecx` | **loading the permutation index** — the dependent load |
| 6.6 / 5.3 | `mov 0x8(%rdi),%rcx` / `mov 0x8(%rsi),%rcx` | the comparator's z-word loads |

In `quicksort` the cost is `movups` 16-byte element moves plus `xor %edi,%edi` /
`mov $0x1,%edi` at ~9% each — the branchless comparison result.

## 3b. Targeted `#[target_feature]` on the gather: real ISA win, larger layout damage

§2 suggested the obvious next move: give the *gather* AVX2 while leaving the merge on
baseline codegen, capturing `cnot`'s −9.35% without the merge's +6.5..+9.1%. Implemented
as a runtime `is_x86_feature_detected!("avx2")` check dispatching to a
`#[target_feature(enable = "avx2")]` wrapper around an `#[inline(always)]` copy of the
body — dispatch once per coset task, never per row.

It works, in the narrow sense: the AVX2 monomorphizations are emitted (204 VEX-encoded
instructions; 128-bit `xmm` rather than `ymm`, since `W ∈ {1,2}` keys are small), and
`cnot`'s gather does move. Isolating A/B against a fold-only binary, 7 pairs:

| layer | wall Δ% | gather Δ% | sort Δ% | merge Δ% |
|---|---:|---:|---:|---:|
| `cnot` | +1.80% (7/7) | **−5.33%** (7/7) | **+11.97%** (7/7) | +8.81% (7/7) |
| `rotation_zz` | **+8.92%** (7/7) | +1.45% (7/7) | **+34.66%** (7/7) | +14.33% (7/7) |

**The sort — which this change does not touch in any way — got 34.66% slower, 7/7
consistent, min +34.35 max +35.60.**

This is not noise and it is not a mistake in the measurement. Applying the LTO
discriminator the protocol requires, every work counter is bit-identical between the two
binaries:

```
terms_in, terms_out, rows_gathered, rows_sorted, rows_id, cosets, runs, layers
  -> identical in both cells
```

The engine performed exactly the same work, row for row, and an untouched function ran a
third slower. This is pure code placement.

**The structural reason, and why it is not tunable away.** To recompile a body under a
wider feature set, `#[target_feature]` *requires* a second copy of that body — a
`#[target_feature]` function cannot be inlined into a caller lacking those features, so
the scalar path needs its own instantiation. In a `lto = "fat"`, `codegen-units = 1`
binary the duplicated gather body (× 2 widths) displaces everything around it, and the
sort and merge pay for it. The tension is inherent: the mechanism that delivers the ISA
win is the same mechanism that causes the damage.

**Verdict: reverted.** Targeted `#[target_feature]` dispatch is not viable in this tree at
this granularity. A −5.33% gather bought at the price of +34.66% on the sort is not a
trade worth making, and there is no obvious knob that keeps the first without the second.

This also retires the Stage-2 plan's recommended dispatch strategy before it was built,
which is the cheapest possible time to learn it. Any future attempt needs either a
separate codegen unit for the kernels (breaking the fat-LTO assumption the whole crate is
built on) or a global build flag — and §2 already showed the global flag is net negative
on two of three priority layers.

## 3c. The mechanism: uop-cache (DSB) eviction, and the engine is already frontend-bound

> **Superseded 2026-09-10 by `2026-09-10-hot-path-code-size.md`.** The observations below
> stand; the diagnosis does not. The cause is not hot-path code *size* but the **JCC erratum**
> (SKX102): with the mitigating microcode loaded, a 32-byte window whose jump touches the
> 32-byte boundary is excluded from the DSB outright. One build flag
> (`-Cllvm-args=-x86-branches-within-32B-boundaries`, now in `.cargo/config.toml`) takes DSB
> residency from 45.8% to **97.9%** and wall time down **7.5–12.6%** on all three priority
> layers, 7/7 pairs, at bit-identical work counters.

§3b's +34.66% on an untouched sort demanded a mechanism. It is the **decoded-uop cache
(DSB)**, and the evidence is unambiguous.

**It is not a codegen change.** `sort_rows_with_scratch`, `merge2_into` and every
`fill_coset` have *identical sizes* in both binaries, and a normalized disassembly diff of
the sort is byte-for-byte identical (303 instructions). Only the addresses moved. The
duplicated AVX2 gather body added **336 bytes**, and **336 mod 32 = 16** — so every
downstream function shifted by half a 32-byte DSB window, turning 32-byte-aligned
functions into 16-byte-misaligned ones and vice versa.

**Counters** (`rotation_zz`, W=2, 1 thread, `--reps 20`, pinned core):

| counter | pb-fold | pb-gavx |
|---|---:|---:|
| `idq.dsb_uops` | 10.59e9 | **3.35e9** (−68%) |
| `idq.mite_uops` | 12.46e9 | **17.96e9** (+44%) |
| instructions | 20.83e9 | 20.24e9 (**fewer**) |
| cycles | 9.40e9 | 10.23e9 (+8.9%) |
| IPC | 2.22 | 1.98 |
| branch-misses | 55.5M | 54.9M (flat) |
| `icache_64b.iftag_miss` | 8.0M | 8.5M (flat) |

Fewer instructions, more cycles, flat I-cache and flat branch misses: this is purely the
front end. Per-symbol attribution of MITE (legacy-decode) uops localizes it exactly:

| symbol | pb-fold | pb-gavx |
|---|---:|---:|
| `sort_rows_with_scratch` | 0.27% | **5.32%** |
| `quicksort` | 0.35% | **3.04%** |
| `drift::sort` | 0.10% | **2.98%** |
| **sort family** | **0.72%** (0.09e9 uops) | **11.34%** (2.04e9 uops) |

**A 22.7× increase in legacy-decoded uops in the sort** — from uop-cache resident to
re-decoded essentially every iteration. That is the +34.66%.

**The baseline is itself only 45.8% DSB.** This is the more consequential finding: the
engine already delivers less than half its uops from the uop cache, i.e. it is
substantially front-end bound before anything is added. The reason is code size — the DSB
is 32 sets × 8 ways × 6 uops, and a 32-byte window that needs more than ~3 ways cannot be
cached at all, while `fill_coset` monomorphizations run to 0x13d0 (5 072) bytes.

**Two obvious remedies were measured and both made it worse:**

| variant | DSB share | cycles vs fold |
|---|---:|---:|
| baseline (`pb-fold`) | 45.8% | — |
| `-C llvm-args=-align-all-functions=5` | 42.5% | +0.9% |
| `#[inline(never)]` on `merge2_into` | 31.5% | +5.7% |

Forcing 32-byte function alignment pads the binary and costs more footprint than the
alignment buys; outlining the merge costs more in call overhead and lost context than the
size saves. Both reverted.

**This retroactively explains several previously-unexplained results in this repo**, which
is the main reason to record it:

- *"adding `#[inline]` to `merge2_into` measured +20–34%"* — inlining it at more call
  sites inflates code size and evicts the uop cache.
- *"the `#[inline]` on `sort_rows_with_scratch` is worth ~6%"* — the same effect with the
  sign flipped.
- *"adding ANY code, even `cfg(test)`-only code, moves untouched hot paths ±4–7%
  direction-consistently"* — address shifts remap DSB sets.

The general statement: **this binary sits at a fragile local optimum in a front-end-bound
regime, and its sensitivity to code motion is a uop-cache phenomenon, not a mystery.** Any
change that adds code to the hot path pays a front-end tax that is frequently larger than
the arithmetic it saves — which is precisely why §3b's targeted SIMD failed, and why it
would fail for a word-planar kernel set too.

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

The one positive signal was `cnot`'s gather at **−9.35%, 7/7 consistent**, under AVX2. §3b
tried to isolate it with targeted `#[target_feature]` dispatch and found that the
isolation mechanism costs **+34.66% on the untouched sort** at bit-identical work counts.
That avenue is closed.

**What this means for Stages 1-3.** The study set out to ask whether a word-planar layout
and SIMD kernels would pay. Stage 0 has instead established something more binding: in
this binary, *the code-placement cost of adding a second kernel path exceeds the
arithmetic it buys*. That applies to any SIMD program here, planar or not, because every
one of them adds a second path. The remaining honest options are narrow:

1. A **global** build flag, per-workload — net negative on 2 of 3 priority layers (§2), so
   at best a documented opt-in for gather-dominated circuits like `cnot`.
2. Moving the kernels into a **separate codegen unit / crate**, giving up the whole-program
   fat-LTO layout the engine currently depends on. That is a large, risky change to
   evaluate against a single-digit-percent prize.
3. Doing nothing, which the phase budget and the latency-bound verdict both support.

> **Withdrawn 2026-09-10 — see `2026-09-10-hot-path-code-size.md`.** This recommendation
> rests on §2 and §3b, both of which were measuring an unpadded binary in a JCC-erratum
> regime. The study should be reopened and every Stage-0 measurement retaken on the padded
> baseline.

**Recommendation: do not proceed to Stages 1-3 as scoped.** The decisive layout gate
(`G-MERGE-1`) was never reached, because a cheaper experiment closed the door upstream of
it. If the question is revisited, the first thing to settle is not SIMD but whether the
kernels can live outside the fat-LTO unit without losing more than they gain.

Still open (Stage 1): whether a word-planar layout helps, which the decisive cheap gate
`G-MERGE-1` (planar `merge2_into` ≤ +2% at W=2) should settle before any refactor.
