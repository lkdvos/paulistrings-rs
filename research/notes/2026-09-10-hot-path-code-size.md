# Hot-path code size and DSB residency — the JCC erratum was the whole story

Measured 2026-09-10 on the reference host (ccqlin038, 2× Xeon Gold 6244 Cascade Lake-SP,
16c/32t, governor `powersave`, microcode `0x5003901`), rustc 1.94.0, working tree at
`11d2844`. Follow-on to `2026-09-10-simd-evaluation.md` §3c, which established that the
engine is front-end bound at **45.8% DSB residency** and left open whether that was
recoverable.

**It is recoverable, essentially in full, by one build flag: 45.8% → 97.9% DSB and
−7.5..−12.6% wall on all three priority layers, 7/7 pairs, at bit-identical work counters.**
The cause was never hot-path code *size*. It was the **JCC erratum**.

## 1. Pre-registered gate

Stated before any measurement: a change must (a) raise DSB share by ≥5 percentage points on
the reference cell, (b) show direction-consistent wall improvement across ≥7 paired runs on
at least one of `rotation_zz` / `cnot` / `trotter` with **bit-identical work counters**, and
(c) show no direction-consistent regression on the other two. The winner clears (a) by 52
percentage points and (b)/(c) on all three layers.

Cell throughout: `--n 1000000 --threads 1 --layers <L> --qubits 128 --reps 20`, `taskset -c 6`,
`RUST_LOG` unset. Baseline rebuilt from scratch was **byte-identical** to §3c's `pb-fold`, so
the two notes' numbers are directly comparable.

## 2. Localizing the MITE uops first

§3c reported the aggregate. Per-symbol attribution (`perf record -e idq.mite_uops -c 2000000`,
cross-referenced against the same record on `idq.dsb_uops`) says the deficit is not spread over
the binary at all — it is one function:

| symbol | MITE uops | DSB uops | DSB share |
|---|---:|---:|---:|
| `gather_local_input_major` | 7.82e9 (62.4% of all MITE) | 2.05e9 | **20.8%** |
| `fill_coset` | 4.09e9 (32.7%) | 4.61e9 | 53.0% |
| sort family (`sort_rows_with_scratch`, `quicksort`, `drift::sort`) | ~0.09e9 (0.7%) | 1.88e9 | ~95% |

This immediately falsifies the code-size hypothesis in §3c's own framing. `gather_local_input_major`
is a *small*, tight, branchy double loop — nothing like `fill_coset`'s 5 072 bytes — and it is the
one that cannot stay in the uop cache. The 10 `fill_coset` monomorphizations are also a red herring
for a different reason: only one or two of them execute in any given layer, and code that never
executes never occupies a DSB way. Monomorphization count can only ever matter here through
*address placement*, i.e. as lottery, never as a systematic cost.

What `gather_local_input_major` has in abundance is **conditional branches per byte**.

## 3. The mechanism: JCC erratum (SKX102)

Every Skylake-derived core, Cascade Lake included, carries the jump-conditional-code erratum.
The mitigating microcode — this host runs `0x5003901`, far past the fix revision — makes the
front end **refuse to cache in the DSB any 32-byte fetch window whose jump instruction crosses
or ends on the 32-byte boundary**. Such a window re-decodes through legacy MITE on every single
iteration. A loop dense in conditional branches has a high probability that at least one of them
lands badly, and the whole window is then permanently MITE.

LLVM ships the standard mitigation: `-mbranches-within-32B-boundaries` pads with segment
prefixes and nops so no branch touches a boundary. Through rustc that is
`-Cllvm-args=-x86-branches-within-32B-boundaries`, stable, one flag, no source change.

### 3.1 Counters

`perf stat`, same cell, two runs per side in `abba` order (values are per-run, spread <1%):

| counter | baseline | +JCC padding |
|---|---:|---:|
| `idq.dsb_uops` | 10.60e9 | **25.68e9** |
| `idq.mite_uops` | 12.52e9 | **0.55e9** (−23×) |
| **DSB share** | **45.9%** | **97.9%** |
| instructions | 20.86e9 | 21.43e9 (+2.7%, the padding) |
| cycles | 9.34e9 | **8.68e9 (−7.1%)** |
| IPC | 2.23 | **2.47** |

All three layers:

| layer | DSB share, base → JCC | cycles Δ |
|---|---|---:|
| `rotation_zz` | 45.9% → **97.9%** | −7.1% |
| `cnot` | 45.8% → **98.5%** | −10.2% |
| `trotter` | 48.4% → **93.4%** | −7.9% |

Per-symbol, `gather_local_input_major`'s MITE uops go **7.82e9 → 0.06e9**, a 130× drop. The
symbol that was 20.8% DSB-resident is now effectively fully resident.

### 3.2 Wall clock, paired

Two prebuilt binaries, alternated `abba`, 7 pairs, one physical core, `scripts/ab-report.py`:

| layer | wall Δ% | gather Δ% | sort Δ% | merge Δ% |
|---|---:|---:|---:|---:|
| `rotation_zz` | **−9.33%** (7/7) | −10.92% (7/7) | ns | −9.19% (7/7) |
| `cnot` | **−12.57%** (7/7) | −15.22% (7/7) | −6.04% (7/7) | −12.89% (7/7) |
| `trotter` | **−7.48%** (7/7) | −4.93% (7/7) | −2.54% (7/7) | −12.74% (7/7) |

Per-cell spreads are ~0.5%, tighter than anything else this repo has measured. **LTO
discriminator applied**: `terms_in`, `terms_out`, `rows_gathered`, `rows_sorted`, `rows_id`,
`cosets`, `runs`, `layers` are bit-identical between the two sides on all three layers. The
engine did exactly the same work and did it 7–13% faster.

At 16 threads (`taskset -c 0-15`, same 7-pair protocol) the effect is real but smaller —
`rotation_zz` −3.53% (7/7), `cnot` −13.33% (7/7), `trotter` ns (6/7) — as expected once the
workload is bandwidth- rather than front-end-bound. No regression anywhere.

**Shipped** in `.cargo/config.toml` under `[target.'cfg(target_arch = "x86_64")']`. Cost is
~2% more instructions and ~2% larger binaries. `scripts/profile.sh` and `benchmarks/PROFILING.md`
were updated in the same commit: an exported `RUSTFLAGS` **replaces** the config's `rustflags`
wholesale, so anything setting `RUSTFLAGS` for a benchmark build must re-append the flag or it
will silently profile a different binary from the one that ships.

## 4. What this retracts and what it explains

This supersedes §3c's closing framing. §3c's *observations* all stand; its *diagnosis* —
"hot-path code size, `fill_coset` at 5 072 bytes" — was wrong, and the two remedies it tried
failed precisely because neither addresses branch/boundary alignment:

| §3c variant | DSB share | cycles vs base |
|---|---:|---:|
| baseline | 45.8% | — |
| `-align-all-functions=5` | 42.5% | +0.9% |
| `#[inline(never)]` on `merge2_into` | 31.5% | +5.7% |
| **`-x86-branches-within-32B-boundaries`** | **97.9%** | **−7.1%** |

Function-start alignment is the wrong granularity: the erratum is about where *branches* fall,
not where functions begin. `-align-all-nofallthru-blocks=5` stacked on top of the JCC flag was
also measured and adds nothing (97.9% → 98.0%, cycles within noise) — with the branches padded
there is nothing left to recover.

More importantly, this is the mechanism behind the repo's oldest piece of folklore:

- *"adding ANY code, even `cfg(test)`-only code, moves untouched hot paths ±4–7%
  direction-consistently"* — shifting code by any amount re-rolls, for every branch in the
  binary, whether it lands on a 32-byte boundary. That is a genuine lottery with a large
  variance, and it is now **pinned**: with padding on, a branch never touches a boundary
  regardless of where the function starts.
- *"`#[inline]` on `merge2_into` cost +20–34%"*, *"`#[inline]` on `sort_rows_with_scratch` is
  worth ~6%"*, *"a segment-copy restructure cost +20–35%"* — all the same coin flip. The
  `#[inline]` set in `engine/merge.rs` should be **re-A/B'd under the padded build** before its
  comments are trusted again; the measurements that justify them were taken in the unpadded
  regime and may not reproduce.
- §3b's headline anomaly — a targeted `#[target_feature]` gather making the *untouched* sort
  34.66% slower — is fully explained. The added 336 bytes shifted everything downstream by
  16 bytes and re-rolled every branch.

## 5. The consequence for the SIMD study: §2's verdict is void

§2 concluded that `-C target-cpu=x86-64-v3` "must not be set globally", on the strength of a
**universal** merge regression: `merge2_into` 6.5–9.1% slower, 7/7, in every cell. §3b then
closed the whole targeted-SIMD avenue on top of it.

That regression was DSB eviction, not the ISA. Re-run with the JCC padding applied to both
sides (`jcc` vs `jcc + target-cpu=x86-64-v3`, 7 pairs, work counters bit-identical):

| layer | wall Δ% | gather Δ% | sort Δ% | merge Δ% |
|---|---:|---:|---:|---:|
| `rotation_zz` | **−1.19%** (7/7) | −1.73% (7/7) | −0.72% (7/7) | ns |
| `cnot` | **−4.85%** (7/7) | −8.75% (7/7) | ns | ns |
| `trotter` | **−2.14%** (7/7) | −5.69% (7/7) | +1.01% (7/7) | ns |

**The merge regression is gone in all three cells** (no consistent change, where §2 had
+6.5..+9.1% at 7/7), and AVX2 is now a consistent win on every layer instead of net negative
on two of three. §2's asymmetric picture — "gather wins, merge loses, sign depends on the
phase mix" — was an artifact of measuring in the unpadded regime.

`target-cpu=x86-64-v3` is **not** shipped here: that is a portability decision (it drops
pre-Haswell hosts) and belongs to the user, not to this change. But the reason the repo had for
rejecting it no longer exists, and the recommendation in `2026-09-10-simd-evaluation.md` §5
("do not proceed to Stages 1–3") rests on evidence that has now been withdrawn. **The SIMD
study should be reopened**, with every Stage-0 measurement retaken on the padded baseline.

## 6. Negative results, recorded as such

Investigated and **not** pursued, each for a reason established above rather than by
measurement fatigue:

- **Reducing `fill_coset`'s 10 monomorphizations** (type-erasing the `TruncationPolicy` at the
  coset boundary, excluding unused `W`). Not pursued: unexecuted code occupies no DSB way, so
  the only channel by which monomorphization count could matter is address placement — the very
  lottery the JCC flag removes. There is no systematic win here, and type erasure at the coset
  boundary would risk the `keep_term` inlining that `merge2_into` depends on.
- **Cold-path outlining** (`#[cold]` on the output-major gather, the radix-sort path, the
  partitioned/export path). Listed a priori as the most promising direction; superseded. The
  measurement says the deficit was never in cold code sharing hot windows — it was branch
  alignment inside one hot loop — and §3c's `#[inline(never)] merge2_into` result (31.5% DSB,
  +5.7%) is direct evidence that outlining in this tree costs more than it saves.
- **`get_unchecked` / bounds-check removal.** Not attempted. It was a last-resort item, the
  motivating mechanism turned out to be something else, and it would have added `unsafe` to a
  tree that has almost none for no measured reason.
- **Unroll caps on cold-but-instantiated widths.** Same disposition as monomorphization count.
- **`-align-all-nofallthru-blocks=5`** on top of the JCC flag: 97.9% → 98.0% DSB, cycles within
  noise, binary +12%. No effect; not shipped.
- **`opt-level=2`**: not measured. With the front end at 97.9% there is no front-end headroom
  left for it to buy, and it would trade away inlining the merge depends on.

## 7. Open

1. **Re-A/B the `engine/merge.rs` `#[inline]` set under the padded build.** Its comments encode
   numbers taken in the unpadded regime; some or all may be void. This is the highest-value
   follow-up and it is cheap.
2. **Reopen the SIMD evaluation** (§5). `target-cpu=x86-64-v3` is now −1.2 / −4.9 / −2.1 across
   the priority layers, and the §3b `#[target_feature]` mechanism should be retried on the
   padded baseline, where a 336-byte insertion no longer re-rolls the binary.
3. **Verify on the cluster's other microarchitectures.** The flag is scoped to `x86_64` and is
   a pure win on Skylake-derived parts, but Rome/Genoa (Zen) and Ice Lake do not have the
   erratum and pay the ~2% instruction cost for nothing. An `sbatch` on `ccq` across node types
   is the check; hardware counters do **not** work on cluster nodes, so that comparison is
   wall-clock paired only.
4. Whether the remaining 2.1% MITE (0.55e9 uops) is worth chasing. Almost certainly not.
