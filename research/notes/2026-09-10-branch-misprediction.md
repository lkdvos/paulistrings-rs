# Branch misprediction — measured, attributed, and half of it removed

Measured 2026-09-10 on the reference host (ccqlin038, 2× Xeon Gold 6244 Cascade Lake-SP,
16c/32t, governor `powersave`, microcode `0x5003901`), rustc 1.94.0, working tree at
`50adee0` (baseline) through `4ee8033`. Load average 2.1–5.9 throughout (the usual agent
processes); every timed run pinned to `taskset -c 6`, one physical core on NUMA node 0, and
no build ever running concurrently with a timed run.

Third of the front-end trilogy. `2026-09-10-hot-path-code-size.md` fixed branch *placement*
(the JCC erratum, 45.8% → 98% DSB residency, −7.5..−12.6% wall);
`2026-09-10-inline-set-repost.md` cleared the `merge.rs` attribute folklore off the same
build. Branch *prediction* was the piece neither touched. It was worth **4.8–9.2% of
cycles**, and two changes took roughly half of that:

| layer | wall Δ% vs `50adee0`, 1 thread, 7/7 pairs | `br_misp_retired` |
|---|---:|---|
| `rotation_zz` | **−14.68** | 56.7M → 33.9M |
| `cnot` | **−9.65** | 76.8M → 44.4M |
| `trotter` | **−2.77** | |

Neither change is the one the framing predicted. A branchless `merge2_into` — the textbook
fix, and the prior favourite — **lost**, for the reason `2026-09-01-sort-kernel.md` §1.2
already recorded. §5 is that negative result.

## 1. Protocol

Cell throughout: `--n 1000000 --threads 1 --layers <L> --qubits 128 --reps 20`
(`--reps 2` for `su4`), `taskset -c 6`, `RUST_LOG` unset. Both sides built up front,
alternated `abba`, 7 pairs, reported by `scripts/ab-report.py --all-phases`; acceptance is
direction consistency across every pair, median Δ% the effect size.

**Padding verified in effect on every binary**: DSB share 97.8–98.4% on both sides of every
comparison below, matching the code-size note's 97.9/98.5%. No `RUSTFLAGS` was exported at
any point (`env -u RUSTFLAGS` on every build), so `.cargo/config.toml`'s list applied intact.

**LTO discriminator applied to every campaign**: `terms_in`, `terms_out`, `rows_gathered`,
`rows_sorted`, `rows_id`, `cosets`, `runs`, `layers` bit-identical between sides in all of
them. Every number below is the same engine doing byte-for-byte the same work.

Counters are per-thread `perf stat` on the pinned process (not `-a -C 6`: the sibling
hyperthread pollutes the core-wide slot events, and perf's own `tma_bad_speculation` metric
halves SLOTS for SMT and so reads ~2× high on a core this process has to itself).
Attribution is `perf record -e br_misp_retired.all_branches:pp -c 1500` — precise sampling,
without which the IP lands on the wrong instruction.

## 2. How much is actually lost — three honest conversions

The task's opening estimate was "55.5M misses × 16–20 cycles ÷ 8.57e9 = 10–13% of runtime".
That is one of three defensible conversions and it is the most generous. All three, on the
baseline:

| | `rotation_zz` | `cnot` |
|---|---:|---:|
| cycles | 8.622e9 | 7.939e9 |
| retired conditional branches | 3.516e9 | 3.059e9 |
| `br_misp_retired.all_branches` | 56.7e6 | 76.8e6 |
| miss rate | 1.61% | 2.51% |
| `machine_clears.count` | 0.72e6 | 0.56e6 |
| **(a)** bad-speculation slots: `(UOPS_ISSUED − UOPS_RETIRED.RETIRE_SLOTS + 4·RECOVERY_CYCLES) / 4·CLKS` | **18.7%** | **28.3%** |
| **(b)** stall cycles: `(INT_MISC.RECOVERY_CYCLES + INT_MISC.CLEAR_RESTEER_CYCLES) / CLKS` | **6.9%** | **9.2%** |
| **(c)** count × 17-cycle penalty | 10.9% | 16.5% |

(a) is TMA's own measure but it counts *slots*, not time: at IPC 2.2–2.5 of a 4-wide machine
there is spare issue width, so wrong-path uops are partly free and (a) is an upper bound on
what removing them can return. (b) counts only the two stall components the machine
attributes to a clear (pipeline recovery, and the front end idling on the resteer) and misses
the wasted-work component, so it is a lower bound. (c) is the textbook arithmetic and lands
between them, closer to (a).

**The honest reading is 7–16% of cycles, most likely near the low end**, and machine clears
are 1.3% of the clears — this is branch misprediction and nothing else. Not 3%: worth
attacking. The eventual outcome (§6) lands at −8.7% and −7.0% cycles for the two layers,
i.e. between (b) and (c), which is the right place for a change that removed ~40% of the
misses.

## 3. Which branches — two instructions, 78% of the misses

Precise attribution, baseline, 37K/51K samples:

**`rotation_zz` (56.7M misses)**

| symbol | share | where inside it |
|---|---:|---|
| `fill_coset` (carries the inlined merge) | 39.9% | **95.8% of it on one macro-fused `cmp`/`jne`** — the first-word test of the lexicographic key compare in `merge.rs`'s `take_a` |
| `gather_local_input_major` | 39.6% | **99.4% of it on one `jne`** — the `ucomisd` after `let a = d.amp[s]`, i.e. `if a == ZERO { continue }` |
| `quicksort` (std) | 15.5% | the partition loop's `jae` (`quicksort.rs:223`) |
| `refine_bucket`, `sort_rows_with_scratch`, `drift::sort` | 3.2% | |

**`cnot` (76.8M misses)**

| symbol | share | where inside it |
|---|---:|---|
| `gather_local_input_major` | 43.4% | the same zero test, spread over three sibling monomorphizations (35.2 / 33.7 / 30.9%) |
| `drift::sort` (std) | 29.2% | the stable merge's take-left branches (`stable/merge.rs:92`, `:124`) |
| `fill_coset` | 14.1% | 97.9% of it on the equal-key drain test |
| `quicksort` (std) | 11.2% | |

Two source lines — one in the gather, one in the merge — carry **78% of `rotation_zz`'s
mispredicts**. The task's priors were right about *which* branches; §5 shows they were wrong
about the fix for one of them.

Two details confirm the mechanism rather than assuming it. The `jp` immediately after the
gather's `ucomisd` — the NaN leg of `Complex64`'s `!=` — takes **0.03%** of the misses: it
never fires and predicts perfectly, exactly as expected, so the whole cost is the zero test.
And the `take_a` compare's miss *rate*, from the matching `br_inst_retired.conditional:pp`
record, is **≈35%** — against a 50% ceiling. That is a coin flip, not a warm-up artifact.
Both branches are asking a genuinely data-dependent question about one row.

## 4. What shipped

### 4.1 `merge2_into` as three loops (`06777e3`)

Not a branchiness fix — found while instrumenting one, and reported here because it is
where a third of the win came from. The fused walk guarded both streams' bounds inside the
hot loop:

```rust
while i < an || j < bn {
    let take_a = j >= bn || (i < an && (a_x[i], a_z[i]) <= (b_x[j], b_z[j]));
```

Two extra conditional branches per output row, plus a loop test that both sides can end,
plus — for every Clifford layer, where the identity stream is empty (`cnot`: `rows_id == 0`)
— running the whole two-stream apparatus to compute `take_a == false` a hundred million
times. Split into a both-live main walk and two drains, the main loop tests only the key
comparison and the `a`-empty case gets a single-stream reduction with no `a`-side test at all.

| layer | wall Δ% | merge Δ% | sort Δ% | gather Δ% |
|---|---:|---:|---:|---:|
| `rotation_zz` | **−4.80** (7/7) | −11.81 (7/7) | ns | ns |
| `cnot` | **−1.96** (7/7) | −7.94 (7/7) | ns | ns |
| `trotter` | **−3.09** (7/7) | −6.56 (7/7) | ns | −1.67 (7/7) |

Counters say plainly that this is *not* a prediction effect: `rotation_zz` retires **11.9%
fewer conditional branches and 3.9% fewer instructions**, cycles −2.8%, and mispredicts move
only −2.3%. It is branch and instruction count.

### 4.2 Branchless zero-amplitude filter in the gather (`4ee8033`)

The one real branchiness fix. `if a == ZERO { continue }` becomes: always materialize the
row, publish it with `len += (a != 0)`.

```rust
runs[i ^ coords[e] as usize].push_if(nonzero(a), kx, kz, src.coeff[t] * a);
```

`GatherRun::push_if` writes the three columns at `len` and then `set_len(len + keep)`. The
`nonzero` helper uses `|` rather than `||` so the imaginary-part test does not reintroduce a
branch (and is pinned by test to agree with `Complex64`'s own `!= ZERO` on signed zeros and
NaN). `reset` now reserves **one slot past** the plan's exact per-run capacity — the slot a
discarded row is written into — which is the invariant the `unsafe` rests on: `cap_rest`
counts every (source row, delta) pair that can target the run, so `len ≤ cap_rest < capacity`
at every call. The same commit fixes an under-reservation on the way past: `reserve(cap −
capacity)` on a just-cleared `Vec` asks for `cap − capacity`, not `cap`.

No AVX-512 was needed. `vpcompressq` was the suggested instrument; a scalar
store-then-conditional-length is the same idea one row at a time, portable, and removes the
`Vec::push` capacity check as a side effect.

| layer | wall Δ% | gather Δ% | note |
|---|---:|---:|---|
| `rotation_zz` | **−10.71** (7/7) | −19.48 (7/7) | 1 rest delta, amp zero on the commuting half |
| `cnot` | **−7.91** (7/7) | −14.37 (7/7) | Clifford: 1 of 4 deltas nonzero per pattern |
| `gu2q` | **−4.11** (7/7) | −11.93 (7/7) | |
| `su4` | ns | ns | dense 16×16 PTM — nothing to filter, and no overhead added |
| `trotter` | ns | ns | wide-rotation plan, not this code path |
| `depolarizing` | ns | ns | key-preserving: no gather at all |

The mechanism is visible in the counters, which is the gate this change had to clear:

(both sides carry §4.1, so this isolates the filter alone.)

| | `rotation_zz` before → after | `cnot` before → after |
|---|---|---|
| `br_misp_retired` | 55.4M → **33.7M** (−39%) | 76.7M → **44.2M** (−42%) |
| cycles | 8.38e9 → **7.65e9** (−8.7%) | 7.99e9 → **7.42e9** (−7.0%) |
| instructions | 20.63e9 → 20.55e9 | 16.23e9 → **19.79e9 (+22%)** |
| IPC | 2.46 → 2.69 | 2.03 → 2.67 |

**`cnot` retires 22% more instructions and is 7.9% faster.** That is the whole trade stated
in one line: the emit work spent on rows that are then discarded is far cheaper than the
mispredicts it buys off. `su4` is the control — a dense PTM has no zeros, so the filter never
discards, and the cell is a clean null in both directions.

At 16 threads (`taskset -c 0-15`, same 7-pair protocol) the gather phase still moves 7/7 on
`rotation_zz` (−6.85%) and 6/7 on `cnot`; wall is "no consistent change" in both, which is
what this box's 16-thread noise does to a single-digit effect and not evidence against.
Peak RSS is +0.16% at worst (the spare slot and the corrected reservation).

## 5. Negative result: the branchless merge loses, again

The prior favourite, and the reason this task existed: replace `take_a`'s branch with a
`cmov` select — branchless lexicographic `key_le` over all `2W` words, `cmov` on the key
words, `i += take_a; j += !take_a`. Built, tested green, measured:

| layer | wall Δ% | merge Δ% |
|---|---:|---:|
| `rotation_zz` | **+4.83** (7/7) | +11.64 (7/7) |
| `trotter` | **+4.64** (7/7) | +10.55 (7/7) |
| `cnot` | −1.29 (7/7) | −5.35 (7/7) |

(`cnot`'s win is not branchlessness: that variant also split the loop, and `cnot`'s identity
stream is empty, so it is §4.1's effect arriving early. Re-measured as a pure loop split it
is −1.96%.)

Two things went wrong, and the second is the interesting one.

**It did not remove the mispredicts.** Conditional branches fell 18% (3.516e9 → 2.884e9 — the
lexicographic early-exits are genuinely gone) but `br_misp_retired` did not move at all:
56.7M → 58.4M. Re-profiling says why: **97.9% of `fill_coset`'s misses are now on the
equal-key drain test** `b_x[j] == key_x`, at the same ~35% rate. Of course they are. The
merge's per-row decision — do the two streams' next keys coincide — carries about a bit of
real entropy per row, and it has to be resolved somewhere. Moving it from the compare to the
drain changes which instruction the counter names, not the information the predictor is being
asked for. **A branchless rewrite of one branch in a loop that needs the same bit downstream
buys nothing.**

**And it lengthened the loop-carried chain**, exactly as `2026-09-01-sort-kernel.md` §1.2–1.3
recorded for driftsort's 22-instruction cmov merge: load `a[i]`/`b[j]` → `2W` chained `cmov`s
→ `take_a` → `i`/`j` → the next load, ~10 cycles that cannot overlap, against a branchy
version whose predicted path issues the next iteration's loads immediately. Estimated before
building; the +11.6% merge busy is that estimate arriving.

**This is now the second independent measurement in this repo of the same rule**: branchless
converts a mispredict into a dependency chain, and only wins when the branch genuinely misses
*and nothing downstream re-asks the question*. The gather passes both tests — the discarded
row is dead, nothing depends on it — and wins 19%. The merge fails the second and loses 12%.

Not committed. Reproduce from `06777e3`'s parent by replacing the comparison with a
`cmov`-select; it is ~40 lines.

## 6. Where the budget stands now

Baseline → shipped, same three conversions as §2:

| | `rotation_zz` | `cnot` |
|---|---|---|
| bad-speculation slots (a) | 18.7% → **9.5%** | 28.3% → **14.1%** |
| clear stall cycles (b) | 6.9% → **4.8%** | 9.2% → **5.7%** |
| `br_misp_retired` | 56.7M → 33.9M | 76.8M → 44.4M |
| cycles | 8.62e9 → 7.65e9 | 7.94e9 → 7.42e9 |
| DSB share | 97.9% → 97.8% | 98.4% → 98.4% |

Residual attribution at `4ee8033` — the gather has left the profile entirely (below 1% of
samples):

| | `rotation_zz` (33.7M) | `cnot` (44.2M) |
|---|---:|---:|
| `fill_coset` — the merge's `take_a` compare / drain test | **65.2%** | 24.7% |
| `drift::sort` (std stable merge) | 1.5% | **51.6%** |
| `quicksort` (std) | 25.3% | 19.5% |

So what is left is (i) the merge's irreducible one-bit-per-row decision, which §5 shows is
not addressable by making it branchless, and (ii) the comparison sorts' own partition and
merge branches, inside std, on data that is unpredictable by construction — and the sort
*algorithm* is a standing constraint (`sort_unstable_by` costs +44% wall on `cnot`,
re-measured 2026-09-10).

## 7. Not pursued, and why

- **Bounds checks** (11 `jae`-to-panic in the merge, 10 in the gather). Perfectly predicted —
  they contribute ~0 to the misses measured here — so `get_unchecked` would buy instruction
  count and BTB pressure only. The two shipped changes already took 3.9% of instructions out
  of `rotation_zz` without any unsafe indexing, and `push_if` is as much `unsafe` as this
  hot path should carry on the strength of one campaign. Left as a proposal.
- **`Vec::push` capacity checks.** Gone for the gather's rest and id streams as a side effect
  of §4.2; the claim that the engine sizes every run exactly was verified in the code and is
  true (with the reservation bug §4.2 fixes). Not separately measured.
- **A per-support-pattern list of nonzero delta entries** (CSR over `amp[s]`), which would
  skip the zero entries rather than filter them. Attractive for sparse Cliffords — `cnot`
  would iterate 1 delta instead of 4 — but it does not help the layer that needs it most:
  `rotation_zz` has exactly one rest delta, so the list is length 0 or 1 and its loop-exit
  branch is the same coin flip §3 measured. §4.2 gets the same win with no data-structure
  change. Worth revisiting only for dense-fanout sparse PTMs.

## 8. Open

1. The merge's ~35%-miss decision is 65% of what remains on `rotation_zz` (≈22M misses,
   ≈4% of cycles). Neither branchy nor branchless resolves it; the only remaining shapes are
   algorithmic — emit both candidate rows and compact (the gather's trick applied to the
   merge, but the equal-key sum makes it not obviously expressible), or change the stream
   layout so the interleaving stops being random. Neither is a small change.
2. `drift::sort` is now the plurality of `cnot`'s misses. Off-limits as an algorithm; whether
   a cheaper *comparator* (the sort-kernel note's open item 3, a presorted pre-check) moves it
   is unmeasured.
3. The 16-thread arms are all "no consistent change" on wall while their phase deltas hold
   direction. Nothing here is contradicted by that, but neither is the 1-thread effect
   confirmed at scale; a quiet-box multi-thread campaign is the way to settle it.
4. `gather_local_output_major` still has the branchy zero test. No built-in channel reaches
   it (`GATHER_OUTPUT_MAJOR_MIN_R = 3`), so it was left alone rather than changed unmeasured.

## 9. Cross-references

- `2026-09-10-hot-path-code-size.md` — the JCC erratum and the padding flag. Everything here
  is measured on top of it; without the padding these branches would have been re-measuring
  layout for the third time.
- `2026-09-10-inline-set-repost.md` — why `merge.rs` is no longer an attribute danger zone,
  which is what made §4.1 and §5 cheap to try.
- `2026-09-01-sort-kernel.md` §1.2–1.3 — the first measurement of "branchless loses to a
  well-predicted branch with a shorter chain". §5 is the second.
- `2026-09-10-simd-evaluation.md` §3a — the instruction-level phase breakdown that named both
  branches (`jp`/`ucomisd` at 5.9% of the gather, the merge's `jne` at 8.7%) before anyone
  knew what fraction of them missed. Its "do not restructure the merge walk" verdict still
  stands and is untouched by §4.1: the rejected variant was gallop + bulk segment copy, an
  algorithmic change; the loop split is the same walk with its bounds tests hoisted.
