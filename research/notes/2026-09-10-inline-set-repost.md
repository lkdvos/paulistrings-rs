# The `engine/merge.rs` `#[inline]` set, re-measured on the padded build — nothing survives

Measured 2026-09-10 on the reference host (ccqlin038, 2× Xeon Gold 6244 Cascade Lake-SP,
16c/32t, governor `powersave`, microcode `0x5003901`), rustc 1.94.0, working tree at `3136639`.
Load average 1.9–3.2 throughout (other users' `julia` and the usual agent processes; all
measurements pinned to `taskset -c 6`, one physical core on NUMA node 0, no build running
concurrently with any timed run).

Follow-up item 1 of `2026-09-10-hot-path-code-size.md` §7. That note established that the
repo's long-standing "adding code moves untouched hot paths ±4–7%" folklore was the **JCC erratum**
(SKX102), and that `-Cllvm-args=-x86-branches-within-32B-boundaries` — now in
`.cargo/config.toml` — pins it (45.8% → 98% DSB residency). It flagged the `engine/merge.rs`
`#[inline]` comments as measured in that confounded regime and therefore suspect.

**They are worse than suspect: three of the four are measuring nothing at all.** Under the
shipping profile (`lto = "fat"`, `codegen-units = 1`) adding or removing any of the three
`#[inline]` hints leaves `.text` **byte-identical**. There was never a code difference to
measure. The fourth claim — the *stable adaptive sort* — survives on CNOT at half the recorded
size, and does not reproduce at all on `rotation_zz`.

## 1. Protocol

Cell: `--n 1000000 --threads 1 --layers <L> --qubits 128 --reps 20`, `taskset -c 6`,
`RUST_LOG` unset (`--reps 2` for `su4`, whose layer is already ~4.1 s). Both sides built up
front, binaries copied aside, then alternated `abba` for 7 pairs; report by
`scripts/ab-report.py --all-phases`. Acceptance is direction consistency across every pair;
the median Δ% is the effect size.

Padding verified in effect on every binary before trusting anything — baseline DSB share
97.8% (`rotation_zz`), 98.4% (`cnot`), 99.1% (`su4`), 98.1% (`gu2q`), matching the 97.9/98.5%
in the code-size note. No `RUSTFLAGS` was exported at any point (`env -u RUSTFLAGS` on every
build), so the config's `rustflags` list applied intact.

**LTO discriminator applied to every campaign**: `terms_in`, `terms_out`, `rows_gathered`,
`rows_sorted`, `rows_id`, `cosets`, `runs`, `layers` bit-identical between sides in all five.

## 2. The three `#[inline]` claims are codegen no-ops

Before measuring anything, compare the binaries. `objcopy -O binary --only-section=.text`:

| variant | change | `.text` md5 | verdict |
|---|---|---|---|
| base | — | `6079b077…` | — |
| v1 | **remove** `#[inline]` from `sort_rows_with_scratch` | `6079b077…` | **identical** |
| v2i | **add** `#[inline]` to `merge2_into` | `6079b077…` | **identical** |
| v3 | **add** `#[inline]` to `sort_rows_radix_with_scratch` | `6079b077…` | **identical** |
| v2n | `#[inline(never)]` on `merge2_into` | `f142452a…` | differs (addresses shift) |

`cmp -l base v1` finds 51 differing bytes in the whole ELF: 20 in `.note.gnu.build-id` and 31
single bytes at stride 0x18 in `.data.rel.ro` — one byte per 24-byte `core::panic::Location`
record, i.e. the source line numbers that moved when the attribute line was deleted. Every hot
symbol keeps not only its size but its **address**.

This is exactly what the profile predicts. `#[inline]` in Rust does two things: it makes the
function's MIR available across codegen units, and it raises LLVM's inline cost threshold. At
`codegen-units = 1` the first is vacuous, and the second cannot fire on functions this size —
the four `merge2_into` monomorphizations are 1 095 / 1 186 / 1 309 / 1 379 bytes and the two
`sort_rows_with_scratch` ones 1 260 / 1 276. Nothing was ever going to inline them.

The paired A/Bs agree, and double as a noise floor for the protocol:

| claim | change | layer | wall median Δ% | pairs |
|---|---|---|---:|---|
| "`#[inline]` on `sort_rows_with_scratch` is worth ~6%" | remove it | `rotation_zz` | **+0.01** | 3/7 neg — ns |
| | | `cnot` | **+0.35** | 3/7 neg — ns |
| "`#[inline]` on `merge2_into` cost +20–34%" | add it | `rotation_zz` | **−1.02** | 5/7 neg — ns |
| | | `cnot` | **+0.08** | 3/7 neg — ns |
| "no measurement either way yet" (radix sort) | add it | `gu2q` | **+0.01** | 3/7 neg — ns |
| | | `su4` | **+0.03** | 3/7 neg — ns |

Per-pair spread with byte-identical `.text` is ±0.7% typical, with one −4.55% outlier in 28
pairs. That is the honest resolution of this protocol at 1 thread on this box today, and it is
worth remembering: a −4.6% single pair means nothing.

## 3. `#[inline(never)]` on `merge2_into` — the one real codegen change, now nearly free

This is the third arm, and the direct retest of the code-size note's §4 table entry
(unpadded: 31.5% DSB, **+5.7% cycles**).

| arm | layer | cycles | instructions | IPC | DSB share |
|---|---|---:|---:|---:|---:|
| base | `rotation_zz` | 8.821e9 | 21.42e9 | 2.43 | 97.8% |
| `#[inline(never)]` | `rotation_zz` | 8.694e9 | 21.80e9 | 2.51 | **97.9%** |
| base | `cnot` | 8.286e9 | 16.66e9 | 2.01 | 98.4% |
| `#[inline(never)]` | `cnot` | 8.115e9 | 16.88e9 | 2.08 | **98.4%** |

The DSB collapse is gone: outlining `merge2_into` no longer evicts anything, because with the
branches padded there is no boundary lottery left to lose. Paired wall clock, 7 pairs:

| layer | wall Δ% | gather Δ% | sort Δ% | merge Δ% |
|---|---:|---:|---:|---:|
| `rotation_zz` | +0.02 (3/7 neg — ns) | −0.28 ns | +0.83 (7/7) | +0.10 ns |
| `cnot` | +0.33 (1/7 neg — ns) | +0.01 ns | −0.35 ns | **+1.68 (7/7)** |

The only surviving signal is merge busy +1.68% on `cnot`, 7/7 — a real, small call-overhead
cost that never reaches wall time. **`+5.7% cycles` → nothing.** The unpadded number was the
erratum, measured one more time.

## 4. The sort *algorithm* claim: half-confirmed, and now conditional

`sort_unstable_by` (pdqsort) in place of the stable adaptive `sort_by` (driftsort) at
`merge.rs:136`. This one is algorithmic, not layout, and it was expected to survive. It does —
on one of the two layers.

| layer | wall Δ% | sort Δ% | gather Δ% | merge Δ% |
|---|---:|---:|---:|---:|
| `cnot` | **+44.29 (7/7, +42.6…+45.8)** | **+188.99 (7/7)** | +0.28 ns | −0.30 ns |
| `rotation_zz` | **−0.43 (5/7 neg — ns)** | −2.89 (6/7 neg — ns) | +0.01 ns | +0.46 ns |

Counters corroborate: `cnot` 8.286e9 → 10.713e9 cycles (+29%) and 16.66e9 → 24.04e9
instructions (+44%); `rotation_zz` 8.821e9 → 8.652e9 cycles at 21.42e9 → 21.45e9 instructions,
i.e. the sort did not change its work at all.

**Why the layers split, from the probe's own counters:**

| layer | `cosets` | `runs` | runs/coset | `rows_id` | `rows_sorted` |
|---|---:|---:|---:|---:|---:|
| `cnot` | 5 120 | 20 480 | **4** | 0 | 15.02e6 |
| `rotation_zz` | 20 480 | 40 960 | 2 (one is the id stream) | 30.0e6 | 19.99e6 |

`rotation_zz`'s comparison sort sees a **single** non-identity stream per coset — one
already-ascending run, which pdqsort's presorted-input pre-check handles in one linear pass,
exactly as cheaply as driftsort's run detection. `cnot` hands it four streams to merge, and
pdqsort has to quicksort them. So the constraint is real but **conditional on ≥2 streams**, and
the recorded "+77% on a 10⁶ `rotation_zz` layer" is void — the layer it names is precisely the
one where the choice is free. The companion "+43% on CNOT" reproduces almost exactly (+44.3%).

Note the sort is only 8.6% of `rotation_zz` wall (69 ms of 801 ms) and there is no path by
which a sort-phase change could have produced +77% wall there in any regime; read the original
as a pre-padding sort-phase measurement that the layout lottery inflated.

## 5. Decisions

- **`#[inline]` on `sort_rows_with_scratch`: kept, comment rewritten.** Removal is a provable
  null change under the shipping profile, so removing it would be manufacturing a diff; the
  hint is still meaningful for a consumer building without fat LTO. What is removed is its
  status as a *constraint*.
- **`merge2_into` and `sort_rows_radix_with_scratch`: left unhinted, comments rewritten.** Same
  reasoning in the other direction, plus the +1.68% merge-busy datum against outlining.
- **`sort_by`: kept, and it remains a real constraint** — restated with the run-count condition
  and the CNOT number.
- `ARCHITECTURE.md` §Engine and §Performance-Model binding constraint #3 updated to match;
  constraint #3 now also states explicitly that there is *no* accompanying `#[inline]`
  constraint. `CLAUDE.md`'s performance-discipline bullet and
  `examples/delta_span_diagnostics.rs`'s header updated for the same reason.

## 6. What this frees

The `#[inline]` set was cited as a danger zone in
`research/plans/2026-09-01-large-m-optimization-campaign.md` §3 (the SIMD campaign),
`2026-09-01-sort-kernel.md` §3.2, `2026-09-01-bucket-cliff.md` and
`2026-09-01-topn-finalize.md`. Those citations are historical records and are left as written,
but the constraint they describe no longer exists: **on the padded build, attribute changes in
`engine/merge.rs` are not a performance hazard and do not need their own campaign.** Combined
with §5 of the code-size note (the `target-cpu=x86-64-v3` merge regression also being an
artifact), the reopened SIMD study has one fewer tripwire to work around.

## 7. Open

1. The `merge2_into` outlining cost (+1.68% merge busy, 7/7, `cnot`) is the only measurable
   attribute effect left in the file and it is below wall-clock resolution. Not worth chasing.
2. A 16-thread arm was not run. At 1 thread the three hint variants are byte-identical code, so
   there is nothing a thread count could reveal for them; for `#[inline(never)]` and the sort
   algorithm a multi-thread cell would be the softest number on a shared box and was skipped
   deliberately.
3. The pdqsort/driftsort split by run count suggests the reverse question: on the *single*-run
   sparse layers the comparison sort is spending ~3.5 ns/row on data it could detect as sorted
   in one pass. Whether a cheap `is_sorted` pre-check in `sort_rows_with_scratch` beats
   driftsort's own is unmeasured.
