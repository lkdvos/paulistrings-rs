# E5 memory-growth smoothing: a negative result

*2026-09-01. Phase-2 experiment E5 (`expt/mem-growth`), scoped in
`research/notes/2026-09-01-large-m-campaign-log.md` from
`research/notes/2026-09-01-large-m-phase-breakdown.md` §7(5). Demoted to
RSS-only by the fact sheet: "No memory step and no memory-limited cell ...
Per-phase timings are `m`-independent to ±10%, so nothing here says the
power-of-two capacity slack costs *time*. If it is done, it is for peak-RSS
reasons ... and should not be sold as a throughput experiment."*

## Verdict up front

**Reverted.** Both candidate fixes are mechanistically real — confirmed by
deterministic unit tests that pin the exact `Vec`-doubling mechanism — but
end-to-end `VmHWM` measurement on real propagation workloads shows no net win
for the low-fanout case and a **2.5–2.6× regression** for the high-fanout
case. `expt/mem-growth` HEAD (`0749b86`) carries no source change relative to
the campaign tip (`f592c43`); the fix commit (`abea4fa`) is reverted, not
deleted, so the mechanism and the reasoning that killed it stay in history.

## Where `VmHWM` is actually set

`ARCHITECTURE.md §Data-Model`'s "capacity is retained across layers" is not a
one-way retention — it is a **circulation** between two physical `Vec`
identities per bucket-in-a-coset, and both count toward RSS:

1. The `PauliSum`'s own persistent bucket (`sum.buckets[b]`, a `BucketCols<W>`
   with `x`/`z`/`coeff` columns), and
2. That coset member's slot in the `LayerScratch`'s per-worker
   `CosetScratch.old` array — scratch, but *not* transient in the "freed
   after the layer" sense: it is a `LayerScratch` field, alive for the whole
   `propagate` call.

Every layer, `fill_coset` (`engine/bucketed.rs:516-519`) `mem::swap`s these
two: the bucket's *previous* content becomes this layer's gather **source**
(now living in `old[i]`), and the *scratch's* previous (cleared) content
becomes this layer's write **destination** (now living in `chunk[i]` ==
`sum.buckets[b]`). So on odd layers object A plays "live bucket" and object B
plays "scratch"; on even layers the roles swap. Both objects are permanently
allocated for the process lifetime — this is *why* the existing
`capacity_stabilizes_across_repeated_layers` test in `bucketed.rs` sums
`bucket_cap + old_cap + run_cap + sort_cap + perm + staging` as one figure:
that whole sum is what sets `VmHWM`, not the `PauliSum`'s columns alone.

`merge2_into` (`engine/merge.rs`) writes the destination via three naked
`.push()` calls with **no reservation at all** — the only allocation site in
the hot per-layer path with none. `refine_bucket`
(`bucket/sum.rs::refine_bucket`, called from `PauliSum::refine`, itself
called by the grow-only `rebucket` before each layer once `m` crosses a
`desired_bits` threshold) builds its "up" split half the same way: `up =
BucketCols::new()` (capacity 0), then pushes into it row by row. Both are
genuine, un-reserved `Vec` growth sites feeding directly into the circulating
capacity described above, and both are exercised every single layer
(`merge2_into`, once per coset member) or every bucket-count doubling
(`refine_bucket`, `O(log2(final_m / initial_m))` times over a run) — i.e.
exactly the two places "the engine predicts output bucket sizes" (the
mission's own framing) but never *uses* that prediction to allocate.

## The fix that was tried

- **`merge2_into`**: `dst` is always freshly cleared before this call
  (`fill_coset`'s swap+clear), so its length is 0 and `an + bn` (identity
  stream length + rest stream length) is a safe upper bound on the output —
  truncation and zero-drop only ever *remove* rows. First version:
  `dst_x.reserve_exact(an + bn)` (and `z`/`coeff` likewise). Second version,
  after the first showed a throughput problem in the m~4.2×10⁶ `rotation_zz`
  cell (`--n 2800000`, wall +13.5% in one interleaved pair, +4.5% in a
  second): switched to `.reserve()`, whose amortized heuristic
  (`max(2×capacity, needed)`) degrades to exactly `reserve_exact`'s behavior
  when capacity is 0 (the cold-start case right after `refine`, or after a
  fanout spike starves a circulating buffer) and to plain `push`'s existing
  growth otherwise.
- **`refine_bucket`**: a first pass over the same `hash.row_parity` test the
  split already uses counts the exact "up" size before any row moves (no
  estimation needed — a partition has no duplicate keys, so the count *is*
  the exact final size), then `up.{x,z,coeff}.reserve_exact(n_up)` once,
  before the split loop.

Both were pinned with deterministic red/green unit tests before any
real-workload measurement:

| test | before | after | bound |
|---|---|---|---|
| `merge2::tests::merge2_into_does_not_leave_doubled_capacity_on_dst` (dst starts empty, `an+bn`=1030, just above the 1024→2048 push-doubling step) | capacity 2048, ratio **1.988×** | capacity ≈1030, ratio ≈1.0 | ≤1.10× |
| `bucket::sum::tests::refine_upper_half_capacity_is_not_doubled` (16 pre-split buckets sized so each upper half lands ~1050, just above the same step) | aggregate ratio **1.666×** | aggregate ratio ≈1.0 | ≤1.10× |

Both went from a clean RED to a clean GREEN. The mechanism the mission
described is real, present, and exactly where the fact sheet's "the engine
predicts output bucket sizes" pointed.

## Real-workload measurement: it backfires

Method: `cargo run --release --features phase-timing --example
phase_breakdown`, `RUST_LOG` unset, single thread (per `ARCHITECTURE.md`'s
own phase-1 finding that the dense-PTM path is bandwidth-saturated above 8
threads — irrelevant here but kept for cleanliness), `VmHWM` read from
`/proc/self/status` at the end of two chained `propagate` calls (untimed
warm-up + timed). Three binaries built from three commits/working states at
identical source otherwise: `before` (`f592c43`, no fix), `after`
(`abea4fa`, `reserve_exact` in both sites), `after2` (working tree with
`merge2_into` switched to `.reserve()`, `refine_bucket` unchanged),
`refineonly` (working tree with `merge2_into` reverted to plain `push`,
`refine_bucket`'s fix kept in isolation). Repeated runs of the *same* binary
at the *same* config are reproducible to ≤0.24% at `m` ≳ 10⁶ (the noise
floor for these numbers) and up to ~2% at `m` ~ 10⁶ — deltas quoted below
exceed that floor except where noted.

### `rotation_zz` (fanout 1.67, low key-collision density), single repeated channel, 30 layers

`vmhwm_kb`, raw (not floor-subtracted) B/term = `vmhwm_kb × 1024 / (terms_in /
layers)`:

| `m` (mean terms/layer) | before | after (`reserve_exact`, both sites) | after2 (`reserve`, both sites) | refineonly (`refine_bucket` alone) |
|---|---|---|---|---|
| 1,049,610 | 183,244 kB (178.7 B/term) | 187,776 kB (183.3 B/term, **+2.5%**) | 189,112 kB (184.6 B/term, **+3.2%**) | 185,288 kB (180.8 B/term, **+1.1%**, inside the ~2% noise band at this `m`) |
| 4,199,916 | 582,124 kB (141.9 B/term) | 585,640 kB (142.7 B/term, **+0.6%**) | 600,604 kB (146.4 B/term, **+3.2%**) | 589,560 kB (143.7 B/term, **+1.3%**, outside `before`'s own 0.24% noise floor at this `m` — real) |

Every single fix variant is **equal or worse**, never better, at every point
measured. Wall time for `merge2_into`'s fix at `m` = 4.2M: `before`
3934–3940 ms across two runs; `after` (`reserve_exact`) 3935–4468 ms (+0.02%
to +13.5% across two interleaved pairs — consistent direction, noisy
magnitude); `after2` (`reserve`) 3949 ms (+0.2%, within noise, so switching
to `.reserve()` did fix the throughput problem — it just didn't fix the
memory problem it was meant to solve).

### `su4` (Haar-random dense 16×16 PTM, fanout 14.94, heavy within-run key collision), `m` = 283,816

| variant | `vmhwm_kb` |
|---|---|
| before | 27,368 / 27,624 (repeat) |
| after (`reserve_exact`, both sites) | **71,908** |
| after2 (`reserve`, both sites) | **70,376** |
| refineonly (`refine_bucket` alone) | 27,108 (repeat of `before`, within noise) |

**2.55–2.62× worse**, reproducible across the `reserve_exact` and `reserve`
variants alike (both collapse to the same behavior here, per the mechanism
below), and localized entirely to `merge2_into` — `refineonly` matches
`before`.

## Why it backfires: the upper bound is not the size

`an + bn` is a *safe* upper bound on `merge2_into`'s output — it can never be
exceeded, so no correctness issue exists — but it is a **loose** one exactly
where a channel's gather run has heavy within-run key collisions, because
the merge's whole job is to *reduce* those collisions via the segmented sum
before writing. `su4`'s fact sheet numbers (`2026-09-01-large-m-phase-breakdown.md`
§1) already named this: fanout 14.94, "93% sorted" — most of every gather
run collapses onto far fewer unique output keys than `an + bn` counts. For
`rotation_zz` (fanout 1.67, "40% sorted") the same effect exists but is much
smaller, which is exactly the gradient the two measured channels show: a
mild, consistent loss for `rotation_zz`, a dramatic one for `su4`.

The loose reservation is not a one-layer cost, either. `Vec` capacity never
shrinks on its own, and the `fill_coset` swap keeps circulating whatever
capacity a buffer was ever given — so one layer's over-reservation for a
high-collision channel becomes the **permanent floor** for every later turn
that buffer plays "destination" for the rest of the run. This is the
opposite of what the mission's framing ("the engine predicts output bucket
sizes") assumed was available: a *true* size prediction here would need the
*post-dedup* count, which is not knowable before doing the segmented-reduction
work the merge itself performs — computing it separately would mean walking
the merge twice, a real throughput cost the fact sheet explicitly ruled this
experiment out of taking on.

`refine_bucket`'s fix has no such loose-bound problem — a partition's "up"
count is exactly the final size, no collisions possible — and it measured
benign on `su4` (matches `before`). But it still showed a small, reproducible
regression on `rotation_zz` at `m` = 4.2M (+1.3%, outside `before`'s own
noise floor at that size). The most likely mechanism: naive push-driven
doubling always lands on a power of two, and glibc's allocator is
demonstrably better at recycling power-of-two-sized chunks across many calls
than the arbitrary exact sizes `reserve_exact` requests here — a heap
fragmentation effect at the allocator level, not anything wrong with the
`Vec`-capacity accounting itself (the unit tests, which measure `Vec`
capacity directly and not process RSS, are unaffected by this and stay
green).

## What this rules out, and what it doesn't

- **Ruled out**: reserving a computed *safe upper bound* (exact or amortized)
  in `merge2_into`'s hot per-layer path, for any channel whose gather runs
  have non-trivial within-run key collisions. This is not a narrow
  implementation bug to patch around — it is the general shape of "reserve a
  correctness-safe upper bound before a reduction," which is fundamentally
  loose whenever the reduction removes a lot.
- **Not ruled out** by this experiment: `shrink_to_fit` at a well-chosen
  point (e.g. once `propagate` detects `m` has stabilized across a few
  layers, or once at the end of a `propagate` call) was in-scope per the
  mission but not attempted — the real-workload measurements above already
  showed the *reservation* side backfiring badly enough that adding a
  *post-hoc trim* on top would only be worth trying after re-deriving a
  reservation policy that doesn't lose to plain `push`, which this
  experiment did not find. A `shrink_to_fit`-only approach (no change to
  growth, just a trim after the fact) remains a live, distinct idea for a
  future attempt and would not inherit this note's failure mode, since it
  never over-reserves — it only ever removes existing slack. It was not
  attempted here because of the time-box on a "deliberately small" E5.
- **Not ruled out**: the true "smoothing" `research/notes/2026-09-01-large-m-campaign-log.md`
  observed (91→99 B/term over 3.8× terms in the deep-KI cross-engine run vs.
  the committed study's 91→125 B/term step) may already be adequately
  explained by that run simply not landing near as unfavorable a
  power-of-two boundary as the committed study's own workload did — i.e.
  there may be nothing left to fix here at all, consistent with the fact
  sheet's own "~0 best case" framing for this bound.

## Proposed timing-neutrality gate command

Moot for this branch specifically — `expt/mem-growth` HEAD carries no source
diff against the campaign tip, so any `ab-compare.sh` run against it is
guaranteed neutral by construction. For the record, had the fix been kept,
the intended gate was:

```bash
scripts/ab-compare.sh --a <campaign-tip-sha> --b expt/mem-growth \
    --layers rotation_zz,su4 --n 1e5,1e6 --threads 1
```

(`su4` included deliberately, not just the mission's `rotation_zz`+`su4`
pairing at threads=1 — it is exactly the channel that caught the regression
`rotation_zz` alone would have missed.)

## Reproduction

```bash
cargo build --release --features phase-timing -p paulistrings --example phase_breakdown

# rotation_zz, m ~ 1.05e6 and ~4.2e6 (mean terms/layer = terms_in / layers)
./target/release/examples/phase_breakdown --n 700000  --qubits 128 --threads 1 \
    --layers rotation_zz --reps 30 --format tsv
./target/release/examples/phase_breakdown --n 2800000 --qubits 128 --threads 1 \
    --layers rotation_zz --reps 30 --format tsv

# su4, m = 283,816 -- the cell that shows the 2.5-2.6x regression
./target/release/examples/phase_breakdown --n 5000 --qubits 128 --threads 1 \
    --layers su4 --reps 4 --format tsv
```

Read `vmhwm_kb` (last-but-one TSV column) and `terms_in`/`layers` for mean
`m`. The reverted fix (for anyone who wants to reproduce the regression
directly) is commit `abea4fa` on this branch, one `git cherry-pick` away from
`0749b86`'s parent.
