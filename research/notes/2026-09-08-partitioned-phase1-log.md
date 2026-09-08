# Partitioned engine, phase 1 — decision log

Running log of gates and decisions while the partitioned (NUMA/multi-node) engine lands on branch
`partitioned-engine`. Plan: `~/.claude/plans/i-would-like-to-cheerful-mitten.md` (user-approved
2026-09-08). Numbers here come from `benchmarks/results/2026-09-08-ccqlin038/` (gitignored).

## 2026-09-08 — S7a `ExtraRows` hook on `fill_coset`: accepted

`scripts/ab-compare.sh s7a-extrarows --a e7de227 --b . --probe '--n 1000000 --qubits 128 --layers
rotation_zz,cnot,su4 --threads 1,16 --reps 8' --pairs 3 --order abba`, load 4–7, `RUST_LOG` unset.
The B side adds the zero-cost `ExtraRows` generic (`NoExtra` on the existing path), the
`apply_layer_bucketed_with` wrapper level, and `pub(super)` visibility; no merge-kernel change.

| cell | wall median Δ% | pairs | work counters (`rows_gathered/sorted/id`, `runs`, `cosets`) |
|---|---|---|---|
| rotation_zz 1t | +3.8 | 3/3 up | identical |
| rotation_zz 16t | +4.4 | 3/3 up | identical |
| cnot 1t | +1.7 | mixed | identical |
| cnot 16t | −17.9 | 3/3 down | identical |
| su4 1t | −3.0 | 3/3 down | identical |
| su4 16t | −18.8 | mixed | identical |

Verdict: **layout artifact, accepted.** Every work counter is bit-identical between sides, and the
consistent deltas have opposite signs on different layers (rotation up, su4/cnot down), inside the
calibrated ±4–7% LTO layout band (`research/notes/2026-09-01-large-m-campaign-log.md`). The fallback
(a textual `fill_coset_recv` copy) would perturb layout just the same. Re-check with the full
`--partitions 1` path after S9 lands, as the plan requires.

## 2026-09-08 — pre-existing flaky stack overflow in the debug test binary

`cargo test -p paulistrings` (debug, full parallelism) aborts with `fatal runtime error: stack
overflow` on an unnamed Rayon worker in roughly 1 run in 4; reproduced on the scaffold commit
`e7de227` (before any partitioned code) and never with `RUST_MIN_STACK=16777216`. Worked around by
`.cargo/config.toml` `[env] RUST_MIN_STACK = "16777216"`; root cause (leading hypothesis: Rayon's
adaptive split depth under stealing × debug frame sizes) deferred to the phase-4 cleanup.

## 2026-09-08 — phase 1 landed

The partitioned engine is on `partitioned-engine`, end to end: a sum split by `PartitionRows`
across `P ≤ 16` pinned Rayon pools, a push-model per-layer exchange, collective bucket-count
agreement and collective truncation, scatter/gather at the run boundary, and a trace. Design
reference: `ARCHITECTURE.md` §Partitioning (written in S12).

| step | commits |
|---|---|
| S1 partition rows | `6846149` |
| S2 scatter/gather on `PauliSum` | `a8abf01` |
| S3 transport traits + wire format | `8ee435e`, `e0695f2` |
| S4 topology, pinning, pools | `dd04018` |
| S5 `PartitionPlan` + row emitters | `6fc46a8`, `65f3bad` |
| S6 export pass | `7039d8e` |
| S7a `ExtraRows` hook | `7f863fc` |
| S7b `apply_layer_partitioned` | `1335239` |
| S8 `PartitionedTruncation` | `70d2f31`, `7a23a34` |
| S9 driver | `7215685` |
| S10 trace and phase counters | `5902c47` |
| measurement tooling | `58e9378`, `e88975f`, `8be2ddc`, `6019589`, `a807e3e`, `93bef8e`, `3d0ae8b` |
| Slurm templates | `ae26017` |
| S12 documentation | `ede6646`, `9fedb47`, `a372220` |

Support commits: `b3778f4` (rustdoc link targets), `75988a5` (`.cargo/config.toml` stack size).

**`P = 1` is bitwise `propagate`.** `one_partition_matches_propagate_bitwise` compares `to_arrays()`
on a 64-channel TFIM Trotter step under `ApproxTopN(2000)`, both directions, plus a keep-all prefix;
it passed on the first run in debug and release. The three places that could have broken it did not:
`allreduce_max_u8` is the identity at size 1, so the collective rebucket reduces to `rebucket`'s own
grow-only body; `scatter_bits` is a no-op at `pbits = 0` and `filter_partition` under
`PartitionRows::none` copies every bucket in order, so scatter and `merge_partitions` are the
identity; and `apply_layer_partitioned` takes its `!plan.has_remote()` branch into
`apply_layer_bucketed` with `LayerKnobs::default()`. The other tripwires
(`layer_fingerprints_are_stable`, the thread-count byte-identity tests) were never regenerated at
any step.

### Gotcha 1 — the retained table can report `is_key_preserving`

The plan assumed the key-preserving `rescale_in_place` fast path was unreachable once a layer had
remote deltas. It is not. The partitioned layer hands the bucketed engine a `retain_entries` copy
holding only the *local* entries, and a channel whose every non-identity delta is remote (a
Hadamard under a partition row that sees its mask — the common case) leaves an identity-only table
for which `is_key_preserving()` is true. Without the `!X::NEEDS_BETA` guard the layer rescales in
place and silently drops every received row. Verified red: both differential nets fail without it.

### Gotcha 2 — the bucket-count all-reduce fails by ballooning, not by erroring

Dropping the collective bits agreement (each partition keeping its own `desired_bits`) did not fail
a test. A partner's block was read at the wrong CSR offsets, produced garbage segment lengths, and
the run allocated until it had to be killed. The fix in the tree is the tripwire
`debug_assert_eq!(block.num_buckets(), local.num_buckets())` on every received block, which turns
the same break into a 0.5 s failure naming the mismatch. **An MPI transport needs the equivalent
check on its receive path**, alongside an equivalent of the debug collective-order counter.

### Decisions worth carrying forward

- `Or` does **not** forward `finalize_layer_partitioned` to its children, because its unpartitioned
  `finalize_layer` is the trait's no-op default rather than either child's; forwarding would make
  `Or(_, ApproxTopN(n))` truncate under partitioning and not without it.
- The transport group is built **per `map_partitions` call** and moved into the partitions. A
  runtime-owned group would keep every sender alive while a partition unwinds, hanging a partner
  blocked in `recv`; the runtime owns only the expensive thing, the pools.
- `scatter_bits(bits, pbits, want) = want.max(bits − pbits).min(bits)`: the bucket count summed over
  partitions equals the unpartitioned one, and the rule is the identity at `P = 1`.
- Exact `TopN` is rejected at compile time by the `PartitionedTruncation` bound. A distributed
  `k`-th selection is the phase-6 follow-up.

### Not measured

No campaign or A/B has been run against the partitioned path. The S7a verdict above is the only
measurement in this log; the post-S9 `--partitions 1` re-check and the P=1 vs P=N runtime-knob A/B
(`scripts/slurm/ab-campaign.sbatch`) are still open. Nothing on the docs site quotes a partitioned
number.

## 2026-09-08 — post-S9 A/B of the untouched path (`e7de227` vs phase-1 tip `c6b11e2`): accepted

Same protocol as the S7a gate (`rotation_zz,cnot,su4` × threads 1,16, `--n 1e6`, 3 pairs abba, load 4–6).
Wall: rotation_zz 1t **+4.4%** (3/3), cnot 1t **+3.4%** (3/3), every 16-thread cell and su4 1t mixed-sign.
Work counters bit-identical on all six cells. Because the 1-thread rotation cell moved the same way in
both A/Bs, a sharper discriminator was run on the archived binaries (`perf stat -e instructions,cycles`,
`rotation_zz,cnot`, 1 thread, `--reps 8`, two runs each):

| side | instructions | cycles |
|---|---|---|
| A `e7de227` | 16.48e9, 16.48e9 | 9.42e9, 9.24e9 |
| B phase-1 tip | 16.55e9, 16.53e9 | 9.91e9, 9.96e9 |

**+0.3% instructions, +5–7% cycles**: the same work at a lower IPC — code placement acting on the
layout-sensitive kernels (CLAUDE.md §Performance discipline), not added work. The only hot-path source
changes are the `NoExtra` generic (compiles away) and one loop-invariant `gen_local` test in the
rotation arm. Accepted. Follow-up for the phase-4 cleanup: test the layout hypothesis directly with a
function-alignment flag A/B (`-C llvm-args=-align-all-functions=6` or similar) rather than chasing it now.

## 2026-09-08 — first end-to-end pinned run (smoke, not a measurement)

`phase_breakdown --partitions 1,2 --partition-cpus "0-7,16-23;8-15,24-31" --threads 32 --n 200000
--reps 2`, load ~5: `rotation_local` P=2 exports 0 rows and matches P=1 wall; `rotation_remote` P=2
exports every anticommuting row (400 300 rows / 2 layers at m = 3.0e5) and its wall doubles (4.4 → 9.4 ms);
`su4` at m = 2.8e6 exports 4.2e7 rows (2.0 GB) over 2 layers with random partition rows and runs 2.2×
slower at P=2 (323 → 716 ms). Partition imbalance 1.001–1.004. This is the phase-1 cost model as
predicted (roughly half of a dense 2Q gate's deltas are remote under random rows) — the input to the
hash-tuning research, not a verdict on the design.
