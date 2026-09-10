# Partitioned engine — decision log

Chronological record of the gates, measurements and decisions behind the partitioned engine on
branch `partitioned-engine`, phase 1 (in-process) through phase 4 (cleanup).

## Index

- **Design of record**: `ARCHITECTURE.md` §Partitioning and its `### Transport composition`.
- **Results**: `2026-09-08-numa-partitioning-results.md` — the P=1 vs P=2 tables across four hosts,
  the counter pass, and the MPI weak-scaling table. Every partitioned number on the docs site traces
  there.
- **Plan**: `~/.claude/plans/i-would-like-to-cheerful-mitten.md` (user-approved 2026-09-08).
- **Optimization history lives here and nowhere else.** A remote rotation layer went from 19× a
  local one to 10×, then 4.6×, then 3.8× over three passes; the entries below say what each pass
  moved and what it rejected. The design doc carries only the final measured cost model.

## Measurement protocol

Unless an entry says otherwise: **ccqlin038** (2 sockets × Xeon Gold 6244, 8 cores + HT per socket,
shared box, load noted per run), release build, `RUST_LOG` unset, seeded input outside the timed
region, medians of 3 runs, **ms per layer**. Distributed cells run `mpirun -n 2 --map-by ppr:1:numa
--bind-to numa` at 8 threads per rank, with `rotation_local` as the load control. Campaign output
lands in `benchmarks/results/<date>-<host>/` (gitignored). A/B verdicts follow CLAUDE.md
§Performance discipline: direction consistency across every pair, median Δ% as the effect size.

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

### Not measured (at this point)

No campaign or A/B has been run against the partitioned path. The S7a verdict above is the only
measurement in this log; the post-S9 `--partitions 1` re-check and the P=1 vs P=N runtime-knob A/B
(`scripts/slurm/ab-campaign.sbatch`) are still open. Nothing on the docs site quotes a partitioned
number. — Closed by the next entry and by `2026-09-08-numa-partitioning-results.md`.

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

## 2026-09-08 (evening) — priorities revised by the user

The primary goal is a **multi-node engine with bounded overhead**, so that sums that do not fit one
node become reachable; gains from aggregate bandwidth on the local kernels are a bonus; the intra-node
NUMA speedup measured in phase 2 (`2026-09-08-numa-partitioning-results.md`: −4 to −18% on
exchange-free dense layers) is secondary. The regime of interest is very large `m`.

Consequences for the plan:
- Phase 3 (MPI via rsmpi) is next. Preference: **one rank per NUMA domain (D = 1)** — process placement
  gives NUMA locality for free; the hybrid in-process-domains-per-rank router is deferred unless a
  measurement asks for it. Partition 0 on the calling thread stays (harmless).
- The pull-model in-process exchange is dropped from the near-term list (the inter-node copy is
  inevitable; the in-process path is now mainly the CI stand-in). The spin-wait collective (already
  in progress, small) lands if it is clean.
- Phase-3 measurement targets the large-`m` regime: weak scaling (fixed terms per rank, 1→2→4→8 ranks
  on Rusty) for local and remote layers; a capacity run past one node's RAM; and **peak RSS per rank
  including exchange transients** as a first-class metric — with random partition rows a dense gate
  exports ~7.5 m rows per layer (2 GB per layer at m = 2.8e6 in the smoke run), which at large `m`
  can exceed the resident sum. Mitigations in scope: chunked export/exchange (bounded transient) and
  the phase-5 locality rows.

## 2026-09-08 (evening) — workload priority: Pauli rotations first

The user's primary target is rotation-heavy circuits (Trotter / kicked-Ising style layers of Pauli
rotations); dense two-qubit unitaries are secondary. Consequences:
- Exchange volume per remote rotation layer is at most one row per anticommuting term (the cos pass
  is always local), so the exchange transient is ≤ 1× the resident sum, not the ~7.5× of a dense gate:
  chunked exchange is a lesser concern for the primary workload.
- Locality is about generators: for a partition row with zero x-bits and z-bits equal to the indicator
  of one side of a cut, every single-qubit X rotation is local and a ZZ(i, j) rotation is remote iff
  the edge (i, j) crosses the cut. For kicked-Ising on heavy-hex, only the cut-crossing ZZ layers
  exchange. This is the phase-5 hypothesis to test first, on the presentation workload.
- Phase-3 measurement uses the `trotter` layer / kicked-Ising circuits at large `m` as the headline,
  `su4` as a secondary stress case.

## 2026-09-09 — phase 3 (MPI transport) landed

Commits `6c7082e`..`a9a545f` (feature `mpi` + `build.rs` rpath; `engine/partitioned/mpi.rs`
`MpiTransport` over rsmpi 0.8.2; `distributed.rs` `DistributedSum<W, X: Transport>` — one partition
per process, `run_layers` shared with the in-process driver; `tests/mpi_ranks.rs` (`harness = false`,
16 cases, singleton under `cargo test`, 2/4 ranks under `scripts/mpi-test.sh`); CI job `mpi`;
`ARCHITECTURE.md §Partitioning → ### Transport composition`; probe `--mpi` with per-rank sidecars and
`vmhwm_kb`). Verified on the merged tree: default workspace 699 tests, `--features mpi` singleton 705,
`mpirun -n 2` and `-n 4` all 16 cases ok; `lto = "fat"` builds with rsmpi.

Decisions/deviations worth remembering:
- **Thread level is `MPI_THREAD_SERIALIZED`, not FUNNELED**: the layer loop runs inside
  `rayon::ThreadPool::install`, so MPI calls are issued from a pool worker. mpi4py users need
  `mpi4py.rc.thread_level = 'serialized'` or `'multiple'` (its default) before `from mpi4py import MPI`.
- Wire parts travel as bytes chunked at 1 GiB (`with_chunk_bytes`), not as a `u64` view: the header
  and CSR offsets are 4-byte aligned.
- `check_consistency` (fingerprint all-reduce, once per propagate) and `gather_to_root` are defaulted
  trait methods; `MpiTransport` overrides the gather. A rank with a different circuit fails with a
  named error instead of hanging (tested).
- Pre-existing flaky test `runtime::tests::a_panicking_partition_does_not_hang_the_group` asserted
  *which* partner's death surfaces first; now asserts termination only.
- Slurm: `scripts/slurm/mpi-ranks.sbatch` now builds a pinned commit (`PS_REV`) in a private worktree,
  runs the rank matrix (`ranks = 2^k ≤ nodes × numa`, one per NUMA domain, `--cpu-bind=ldoms`), then
  the probe with `--mpi` (env `LAYERS`, `N`).

## 2026-09-09 — Python `comm=` landed; the reported communicator "leak" is not one

Bindings: `propagate(..., comm=<mpi4py comm>, result="gather"|"local")`, `mpi_available()`,
`PartitionStats.rank/.size`, `tests/test_mpi.py` (any world size), `scripts/mpi-test.sh --python`,
`docs/book/src/design/mpi.md`. The binding validates everything non-collective (mode conflict,
placement, exact-`topn` gate, thread level ≥ SERIALIZED, `MPI._sizeof` ABI guard) *before* the
collective `MPI_Comm_dup`, so a rank that raises never enters a collective its partners wait on.
The handoff flagged `MpiTransport::adopt` returning `SizeNotPowerOfTwo` after the dup as a leak: it is
not — rsmpi's `SimpleCommunicator::from_raw` owns the handle and `Drop` calls `MPI_Comm_free`
(`mpi-0.8.2/src/topology/sealed.rs:152`), and `adopt` takes the communicator by value.

## 2026-09-09 — first large-`m` MPI numbers (Slurm 7008424: Icelake node, 2 ranks × 32 threads, UCX shm)

`phase_breakdown --mpi --n 8000000` (replicated input, 6.0e6 terms per rank for rotations), per layer:

| layer | wall | export | exchange | coset loop | rows exported/layer | peak RSS/rank |
|---|---|---|---|---|---|---|
| rotation_local | **7.4 ms** | 0 | 0 | 6.6 ms | 0 | 2.2 GB |
| rotation_remote | **140 ms** | 35–39 ms | 95 ms | 6.9 ms | 4.0e6 (192 MB) | 2.3 GB |
| rotation_zz (random rows: remote) | 143 ms | 39 ms | 95 ms | 7.1 ms | 4.0e6 | 1.6 GB |
| cnot | 70 ms | 19 ms | 43 ms | 6 ms | 2.0e6 | 2.4 GB |

A remote rotation layer costs **19× a local one**, and the interconnect is not the reason (192 MB over
UCX shared memory is ~10 ms). The exchange path allocates fresh send and receive buffers every layer
(page faults + zeroing of ~400 MB), decodes received bytes word by word (`pod_read_unaligned` loop),
copies received rows a second time into the gather runs, and the export pass (two passes over 6e6
terms) takes 5× the whole coset loop. Under the revised priorities this is the overhead to bound;
target: remote layer ≤ 3× local. Optimization pass launched (buffer reuse across layers, zero-copy
encode/decode into typed columns, unzeroed fills, sub-phase laps inside `exchange`).

Also: peak RSS is ~370 B/term per rank here, of which the probe's *replicated input* (all 1.2e7 terms
built on every rank before filtering) is a large share — a probe artefact for capacity runs; real
drivers should ingest distributed. Multi-node jobs 7008425/26 failed because the build dir was on
node-local `/tmp`; the template now builds on the shared filesystem.

## 2026-09-09 — the exchange path optimized: remote rotation layer 10× → 4.6× a local one

Standard protocol, `--layers rotation_local,rotation_remote --reps 6`. Two commits: `33194aa`
(sub-phase laps) and `a2a7633` (the fix).

### Where the time went (the laps, before any fix)

`--n 2000000` (1.5e6 terms/rank, 1e6 rows = 48 MB exported per layer):

| phase | local | remote |
|---|---|---|
| wall | 6.7 | 68.2 |
| export | — | 18.7 = count 4.7 + **fill 13.6** |
| exchange | — | 29.8 = post 0.03 + header 0.4 + **alloc 7.2** + data 13.7 + **decode 8.5** |
| coset loop | 6.5 | 15.0 (append_into 7.9 busy) |

The interconnect was 13.7 ms of a 68 ms layer. The rest was the engine paying for its own megabytes:
a fresh zeroed allocation for the export block (most of "fill"), a fresh zeroed allocation for the
receive buffers, and a word-by-word `pod_read_unaligned` decode into a *second* copy.

### The fix

Grow-only `ExchangeBlock` columns (`set_counts` re-aims a block, `header.rows` is the authority, not
`x.len()`), a payload pool on `PartitionState` threaded through `Transport::exchange(send, spare)`,
and `Payload::recv_into` handing MPI mutable byte views of the receiving payload's own typed columns
so the decode pass disappears. The in-process transport ignores the pool — it moves the payload, so
the pool circulates through the partners.

| cell | phase | before | after |
|---|---|---|---|
| `--n 2e6` | local wall | 6.7 | 6.5 |
| | remote wall | 68.2 | **32–34** |
| | export | 18.7 | 8.0 (count 3.6 + fill 4.4) |
| | exchange | 29.8 | 13.7 (data wait 13.4; alloc/decode 0.0) |
| | coset loop | 15.0 | 10.0–11.3 |
| | remote/local | 10.2× | **5.0×** |
| `--n 8e6` | local wall | 24.7 | 25.3 |
| | remote wall | 238.4 | **116.7** |
| | export | 68.5 | 27.8 (count 8.5 + fill 19.3) |
| | exchange | 128.3 | 51.4 |
| | coset loop | 39.3 | 35.3 |
| | remote/local | 9.7× | **4.6×** |

The coset loop got faster too (gather 47 → 33 ms busy at 2e6): the received rows now live in pages
that were already touched.

**Memory.** Peak RSS per rank is unchanged — 544 MB at 2e6 both sides; at 8e6 the three runs span
1593–1716 MB before and 1605–1725 MB after, one band. What does rise is the *between-layer* resident
set (`vmrss` 1.0 → 1.4 GB at 8e6): the pool holds one export volume of send buffers plus one of
receive buffers instead of returning them to the allocator every layer. Peak is what a capacity run
is bounded by, so this is the intended trade; a `shrink` hook on `DistributedSum` is the escape hatch
if a driver ever needs the memory back between circuits.

### What is left, and what was rejected

- **The transfer, 13.4 ms (2e6) / 50.9 ms (8e6).** 96 MB and 384 MB per rank per layer across the
  socket, i.e. ~7.5 GB/s both sizes — a single-threaded cross-socket copy, which is the hardware.
  Only overlap can hide it: the CSR offsets (8 KB) are all `ExtraRows::count` needs to size the runs,
  so a two-phase exchange could let the coset loop's *gather* run while the rows are still in flight
  and have the first `append_into` complete the receive. It needs a non-blocking `Transport` shape
  (rsmpi requests outliving `multiple_scope`) and the deadlock argument written down; it is the next
  idea, not this pass's.
- **Dropping the export's count pass: rejected on arithmetic.** The count pass reads the keys only
  (192 MB at 8e6, 8.5 ms, 23 GB/s — bandwidth-bound). Sizing each bucket's segment at its upper bound
  instead would write 288 MB and then read and rewrite 384 MB to compact, and raise the transient. The
  two passes are already at the memory ceiling (fill: 480 MB in 19.3 ms = 25 GB/s).
- **`append_into`'s second copy: not worth it yet.** 7 ms busy at 2e6 and 36 ms at 8e6, i.e. ~1 and
  ~4.5 ms of wall on 8 threads. Removing it means receiving *per destination bucket* straight into
  the gather run — the sender's CSR by source bucket is the receiver's `β′ ⊕ bd`, a bijection, so the
  addresses are computable before the receive — but the runs are per-coset-task scratch that does not
  exist until the coset loop is running, so it needs the same deferred-receive shape as the overlap
  idea. Design them together.

## 2026-09-09 — the transfer hidden under the coset loop: remote rotation layer 4.5× → 3.8× a local one

Standard protocol, `--layers rotation_local,rotation_remote --reps 6`, load 1.4–3.5 (the local layer
is the control and did not move). Two commits: `4260cde` (the destination-coset order) and `58bf6d7`
(the two-phase exchange).

### The idea

After the previous pass the remote layer was export 27.8 + exchange 51.4 + coset loop 35.3. The 51 ms
is 384 MB per rank per layer over the socket at ~7.5 GB/s — a single-threaded copy at the hardware
limit, and nothing makes it cheaper. The only lever is to run the coset loop while it happens.

Two pieces:

1. **The block is laid out in the receiver's coset order.** `Gf2Span::perm_index` is a pure function
   of the local bucket deltas and the (collectively agreed) bucket count, so both ranks compute it;
   the sender permutes its count and offset arrays and writes segment `p` for the receiver's position
   `p`, and the receiver reads `segment(position_of(β′))`. A coset is then contiguous in the block, so
   a contiguous range of positions is a whole number of cosets.
2. **`Transport::exchange_layer(send, spare, map, body)`.** The framing splits: the *early* parts
   (block headers + CSR offsets, tens of kilobytes — all `ExtraRows::count` needs) are waited out
   before `body` runs; the *bulk* parts (x/z/coeff) are cut at the chunk edges, posted chunk-major, and
   completed under it. `ChunkWait::wait_chunk` is called once per coset task at the top of
   `append_into`. On the MPI side `ChunkPipeline` holds the request collection behind a `try_lock`ed
   mutex: whichever worker gets in is inside `MPI_Waitsome` over *all* requests (driving its partner's
   rendezvous too), the rest back off on per-chunk atomics. One thread in MPI at a time is the
   `MPI_THREAD_SERIALIZED` already required. Deadlock argument: all sends posted before the first
   receive, a waiter services its partner, and the closing `finish` waits out anything the loop never
   asked for.

### Before/after

| n | phase | before | after |
|---|---|---|---|
| 8e6 | rotation_local wall | 26.2 | 25.9 |
| | **rotation_remote wall** | **116.9** | **99.6** |
| | export | 27.6 (count 8.1 + fill 19.5) | 30.2 (count 10.8 + fill 19.3) |
| | exchange | 51.6 (data wait 50.9) | **1.2** (header+early 1.0, residual data wait 0.02) |
| | coset loop | 37.2 | 66.7 (of which `chunk_wait` 274 ms busy ≈ 34 ms wall on 8 workers) |
| | **remote / local** | **4.46×** | **3.84×** |
| | peak RSS/rank | 1654 / 1711 MB | 1648 / 1696 MB |
| 2e6 | rotation_local wall | 6.7 | 6.4 |
| | **rotation_remote wall** | **32.5** | **25.1** |
| | **remote / local** | **5.0×** | **3.90×** |

`exchange_ns` is small now *by construction* — the transfer moved inside the coset loop. The new
counter `chunk_wait_ns` (worker busy time, a part of `append_ns`) is what the loop failed to hide;
read the two together, and see `benchmarks/PROFILING.md`.

### What the chunk sweep says

`PAULISTRINGS_EXCHANGE_CHUNKS` overrides the chunk count (read once per process, so `mpirun -x`).
At 8e6, wall per remote layer: **121.3 at K=1**, 99.0 at K=8, 100.2 at K=32, 102.7 at K=128. K=1 is
the two-phase shape with no pipeline and it is *worse* than the blocking exchange it replaced — so
the win is the pipelining, not the reshaping, and 8 is the default.

### Rejected, measured

- **Lanes (aligning the chunks with Rayon's contiguous split).** The model was: `par_chunks_mut`
  gives worker `j` the range `[jN/T, (j+1)N/T)`, so with `K` contiguous chunks worker `j` starts in
  chunk `j` and worker `T−1` blocks for the whole transfer. The fix would be to cut the range into
  `L` lanes first and make chunk `k` the `k`-th slice of *every* lane, so the whole pool walks the
  chunks together. Implemented (`ChunkMap::piece`, chunk-major piece order on the wire) and measured
  at 8e6: `L=1` 100.5, `L=2` 96.4, `L=4 K=4` 101.7, `L=8` 105.3 — and **`chunk_wait_ns` was
  269–282 ms in every one of them**, i.e. the lane structure changed the waiting not at all. The
  model is wrong; reverted.
- **A sleep-first waiter** (`PIPELINE_SPINS`/`YIELDS` to 0, sleep 50 µs), on the theory that seven
  spinning workers slow the driving worker's copy: 101.8 ms, no change. Reverted.

### What remains

- **~15 ms per layer of the transfer is still not hidden at 8e6.** Real coset-loop work is ~243
  worker-ms (30 ms wall on 8 workers) against a 51 ms transfer, so the floor for this shape is
  `export + transfer + one chunk's work` ≈ 30 + 51 + 4 = 85 ms (3.3×), and we are at 99.6. The gap is
  a combination of the coset scheduling (which lanes did *not* explain) and the transfer running
  slower with seven workers churning beside it. Worth a `perf` look before another guess.
- **The coordinator's gather/merge split** — run each batch's local gather *before* waiting on its
  chunk, then append/sort/merge after — is not reachable without either editing
  `engine/bucketed.rs` or duplicating `fill_coset` (≈250 lines over four private types) in
  `layer.rs`, because it presupposes a *batched* coset loop and the batching itself is what
  `par_chunks_mut` does not let a caller control. Design it together with a `LayerKnobs`-level
  batching hook if it is worth the diff.
- **The real ceiling is the copy, and intra-node it need not exist.** Both ranks are on one node, so
  an `MPI_Win_allocate_shared` window the export filled directly would let the receiver's
  `append_into` read the sender's buffer — deleting the 51 ms outright rather than hiding it. It does
  not generalize off-node and it moves the export's allocation into MPI's hands, so it is a separate
  design, not an increment on this one.
- **`export_count_ns` drifted 8.1 → 10.8 ms** across the pass. Nothing in the count path changed;
  it tracked the box's load. Worth re-measuring on a quiet node before reading anything into it.

## 2026-09-09 — phase-3 weak-scaling numbers in; cleanup phase starts

Slurm 7010761–63 (1/2/4 Icelake nodes, 2/4/8 ranks, 6e6 terms per rank): remote rotation layer 3.5×
local intra-node, 4.4–4.7× inter-node, flat from 4 to 8 ranks; the layer is transfer-bound (~13 GB/s
per node over IB with two ranks sharing the NIC), compute fully hidden. Full table in
`2026-09-08-numa-partitioning-results.md`. Decision: proceed to phase 4 (cleanup) per the plan; the
intra-node zero-copy handoff (in-process domains per rank) stays a scoped follow-up — it would take the
2-rank case from 3.5× to ~1.6× but does nothing for the inter-node share.

## 2026-09-09 — phase 4 (cleanup): the code half

Eight commits `cbea2f4`..`a7b865f` on top of `04ae45b`, behaviour unchanged: `engine/bucketed.rs`
and `engine/merge.rs` are byte-for-byte untouched, the fingerprint net and the thread-count
byte-identity tests pass with their literals unregenerated. Net **−325 lines of code** (partitioned
module −174, the six partitioned test files −200, `test_support` +49) against +101 lines of comment.

Surface narrowed: the concrete wire types (`AlreadyHere`, `BlockHeader`, `ExchangeBlock`,
`PartnerPayload`), the topology pinning helpers and `Payload::from_byte_parts` are `pub(crate)` or
gone — a `Transport` moves an opaque `P: Payload` and never names them. `mpi::DEFAULT_CHUNK_BYTES`
is private (`with_chunk_bytes` is the documented override), and
`MpiTransport::{try_from_communicator, communicator}` are gone. **`Transport::exchange_layer` is now
the required method** and `exchange` the provided one over an empty `ChunkMap`; it was the other way
round. `PhaseStats::decode_ns` is removed with the blocking exchange that fed it, and with it the
column in the probe's TSV and JSON sidecar.

Shared where it was duplicated: `driver::scatter_local`, `driver::PartitionCtx`,
`PartitionWork::take`, `mpi::note_slot`, and the partitioned fixtures (`KeepAll`, `zz_rotation`,
`trotter_circuit`, `unpinned_partitions`) in `test_support`. `tests/mpi_ranks.rs` gained coverage of
`propagate_mpi`, the one public entry point nothing exercised.

Deliberately left whole: `MpiTransport::exchange_layer` at 210 lines, because top to bottom it *is*
the deadlock argument (post every send → wait the headers → post and wait the early parts → post the
bulk → body → finish) and splitting it buys no borrow-checker win;
`post_layer_send`'s nine arguments, which are borrows the request scope has to outlive.

Smoke check, standard protocol, `--n 2000000 --threads 8 --reps 4`, three runs alternated, load 2–3:
remote/local 3.73× before and 3.62× after, remote wall 98.4 → 98.6 ms, peak RSS per rank +0.27%. Not
an A/B; the point was that nothing moved.

## 2026-09-10 — phase 4 (cleanup): the docs half

`ARCHITECTURE.md` §Partitioning rewritten as the current state of the whole engine, with the
optimization history removed from it — the 19× → 3.8× sequence lives in this log, and the design doc
carries only the two committed result tables. §Parallelism and §Performance-Model updated to match.

The book's `design/numa.md` and `design/mpi.md` stop saying "no committed measurement yet" and open
with what to expect; `benchmarks/PROFILING.md` contract (a) lists the fields the probe emits now and
gains a rank-axis section; CLAUDE.md gains the sidecar-contract rule and the probe's replicated-input
artefact under Known gaps.

Still open, in rough order of expected value:

1. **Locality rows** (phase 5). The whole in-process story turns on it, and inter-node the only
   remaining lever is fewer bytes. Hypothesis to test first: a row with zero x-bits whose z-bits are
   the indicator of one side of a spatial cut makes every single-qubit rotation local and only
   cut-crossing ZZ layers remote.
2. **Ingest distributed.** The replicated input caps a capacity run at what one rank can build.
3. **Intra-node zero-copy handoff** (`MPI_Win_allocate_shared`, or in-process domains per rank):
   3.5× → ~1.6× on the 2-rank case, nothing off-node.
4. **Split `collective_ns`** into a publish lap and a wait lap, so arrival skew and transport are
   separate columns.
5. **The debug stack-overflow flake** (`RUST_MIN_STACK` workaround in `.cargo/config.toml`) still has
   no root cause, and the function-alignment A/B that would test the LTO-layout hypothesis behind the
   S7a and post-S9 verdicts was never run.
6. **Exact `TopN` under partitioning**, as a distributed *k*-th selection.

## 2026-09-09 — phase 5 (partition-row tuning) first results

`2026-09-09-partition-row-tuning-results.md`: cut rows make the heavy-hex kicked-Ising step **15–21%
faster at P=2 than the single-process engine** (4 of 271 layers remote instead of 139; exported rows
10× fewer; imbalance ≤ 1.09); the chain is a wash. Two fixes on the way: the partition-row salt equalled
the default hash seed (`f39341a`), and the selector's tie order picks the cut location (balance-scored
restarts in progress).

## 2026-09-10 — phase-5 C2 (MPI, cut vs random rows) and the per-layer collective floor

Cut rows: 1.9–2.4× faster than random at 2/4/8 ranks on heavy-hex (Slurm 7015679–85; table in the
phase-5 results note). At ≤ 3e5 terms per rank the per-layer bucket-bits all-reduce (~30–70 µs over IB
× 271 layers per step) is the dominant cost of exchange-free layers; amortizing it (every K layers and
before remote layers) is the next engine change (in progress).

## 2026-09-10 — bucket-bits collective on a schedule; phase 5 results complete

`19a7971`..`9c14435`: the bucket-count all-reduce runs only on layers with a remote delta, on the first
`BITS_AGREE_EVERY = 16` layers, and every 16th layer after that; non-finalizing policies skip
`finalize_layer_partitioned`. Heavy-hex step with cut rows, 2 ranks on one node: 1355 → 120 collectives
per call, wall unchanged (shared-memory all-reduce is µs) — the effect this targets is the 30–70 µs per
layer measured over InfiniBand at 4–8 ranks (Slurm 7015681/83: 133/136 ms per step); re-measure there.
`collective_ns` did not fall 11×: it is arrival skew, and fewer sync points each absorb more of it.

Phase 5 status: hypothesis confirmed on both workloads (cut rows: 4/271 layers remote per heavy-hex
step, P=2 in-process 15–21% faster than single-process, 1.9–2.4× over random rows on IB); selector fixed
(balance-scored restarts) and compared to hand cuts; recommendation recorded in the results note.
Open after this: the IB re-measurement of the bits schedule, distributed ingestion for capacity runs,
the small-message regime of cut-crossing layers, intra-node zero-copy handoff.

## 2026-09-10 — bits schedule confirmed over InfiniBand

Slurm 7015753/54: heavy-hex step with cut rows 133 → 86 ms (4 ranks) and 136 → 81 ms (8 ranks) per
step; multi-node strong scaling now visible (125/86/81 at 2/4/8). Phase 5 closed; table in the results
note.

## 2026-09-10 — pre-merge trim (PR #6)

Review size, not behaviour: the PR against `main` was 24.6k added lines, of which the partitioned module
was 12.2k (4.1k code, 3.9k comments, 3.7k inline tests). Two removals, each its own commit, verified by the
full gate set (default / `phase-timing` / `mpi`, `mpi-test.sh --ranks 2,4 --python`, pytest):

- `e4c6c6b` — measurement scaffolding whose numbers are recorded above: `ExchangeTimings` and the four
  MPI exchange laps, the `export_count_ns`/`export_fill_ns` sub-split, the probe's `--p1-path` (P=1
  partitioned was measured byte-identical and equal in wall to the classic path, which is now the only
  P=1 path), `DistributedSum::is_empty_local`. `append_ns`/`chunk_wait_ns` stay: they are the documented
  "what the coset loop failed to hide" diagnostic.
- `fd085ec` — the greedy MAX-XOR-SAT selector (`select_rows`, `select_rows_with`, `SelectOptions`,
  `RowSelection`, `--partition-rows select`). Verdict it rests on: on heavy-hex it saved 2 remote layers
  per step at an imbalance of 1.30 against 1.085 for the hand cut; `cut` is the recommendation for a known
  lattice. Last present at `da86546`; `PartitionRows::cut`, `circuit_generators` and `layer_locality` are
  the supported tools. Re-introduce from that commit if a workload without a known lattice needs it.

Net −1.75k lines (24.6k → 22.9k against `main`); module code 4.1k → 3.9k. Not done, on purpose: comment
density (32%, in the repo's register) and the layer-level tests in `layer.rs`, which test a different level
than the driver nets in `tests/propagate_partitioned.rs`.
