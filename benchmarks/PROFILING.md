# Profiling

Operational reference for the performance-measurement framework: how to run a measurement campaign, read its output, and decide what it means.
See `benchmarks/README.md` for the three benchmark surfaces; this doc is about the tools underneath the Rust microbenchmark surface.

Two protocols share the same probe (`examples/phase_breakdown.rs`) and `phase-timing` instrumentation, differing only in how runs are scheduled and compared.
**Campaigns** (`bench-campaign.sh`) run once and log everything to a dated results directory — good for exploring, and for effects large enough to clear this host's single-shot noise floor.
**Interleaved A/B** (`ab-compare.sh`) builds two binaries and alternates them adjacent in time — the only protocol that can resolve effects the campaign noise floor would otherwise swallow.

## The standard loop

1. Change engine code.
2. Run a campaign:
   ```bash
   scripts/bench-campaign.sh <name> \
     criterion:apply_layer_bucketed \
     probe:'--n 1000000 --threads 1,8,32' \
     perf-stat:'--n 1000000 --threads 32 --layers rotation_zz'
   ```
3. Open the HTML report rendered at the end (`benchmarks/results/<date>-<host>/<name>-report.html`), then diff against the last snapshot:
   ```bash
   python3 scripts/criterion-report.py compare <old-snapshot>.json <new>.json
   ```
   Record anything that moved enough to matter in `research/FINDINGS.md`.

Snapshots live next to the campaign's `.txt` file, at `benchmarks/results/<date>-<host>/<campaign-name>.json`.
`criterion:` items write it via `criterion-report.py snapshot --merge`, so several `criterion:` items in one campaign accumulate into the same file.
The `.txt` sibling carries a provenance header (commit, rustc version, thread count, governor, CPU, load average at start/end) followed by one `=== item ===` section per item.
`criterion-report.py compare` is a reporting tool, not a gate — it always exits 0; treat REGRESSION/IMPROVED annotations (±5% default threshold) as prompts to look, not pass/fail.

If the effect is smaller than the noise floor below, a campaign-to-campaign comparison can't resolve it — use the A/B protocol.

## Interleaved A/B protocol

This host's single-shot campaign noise is ~±5–8% at 1 thread and ~±10–26% at 8–32 threads, which exceeds the ~5–10% effects most engine changes produce.
The protocol: build both sides up front into separate binaries, alternate them adjacent in time for several pairs, and compare pairs (run *i* of A against run *i* of B) instead of two independently-noisy means.

```bash
scripts/ab-compare.sh g1d-borrow \
  --a HEAD~3 --b . \
  --probe '--n 1000000 --threads 1,32 --layers rotation_zz' \
  --pairs 5 --order abba
```

`--a`/`--b` take a git revision or `.` for the working tree; each side builds in its own tree so the two builds never evict each other from a shared `target/`.
`--pairs` defaults to 3.
`--order abba` alternates the within-pair order so a monotone drift in machine state can't masquerade as a consistent B-is-faster signal.

The run phase alternates the two binaries, appending each cell to a per-side `--json-out` sidecar, then hands off to `ab-report.py`, which pairs runs by position within each `(layer, threads)` cell.

- **Acceptance rule: direction consistency, not a p-value.** Every pair in a cell must move the same way; the median Δ% is then the effect size to quote. Pairs disagreeing in sign are "no consistent change" — not a small win, not a trend.
- `--field wall_ns` (default) drives the comparison; `--all-phases` also summarizes the coset-loop worker busy fields (`gather_ns`, `sort_ns`, `merge_ns`) per cell to explain why wall time moved — not as an effect size, since they sum over every Rayon worker and never sum to wall time.

`ab-report.py` is re-invocable on the archived sidecars at any time (stdlib only, no cargo, no benchmarks):

```bash
python3 scripts/ab-report.py <name>-a.probe.jsonl <name>-b.probe.jsonl \
    --all-phases --label-a "A=HEAD~3" --label-b "B=."
```

Both sides being the working tree, or both resolving to the same clean commit, is flagged with a warning — that configuration measures this host's own noise floor, not a code change.

## Tool one-liners

- `scripts/bench-campaign.sh <name> <item>...` — runs a sequence of `criterion:<filter>`, `probe:<args>`, `perf-stat:<args>`, `scaling:<placement>`, `macro`, or `bandwidth` items, logged to `benchmarks/results/<date>-<host>/<name>.txt` (append-on-rerun, never overwrite). `--help` prints the full item menu.
- `scripts/host-topology.sh` — sourced, not executed, by `bench-campaign.sh` and `bandwidth.sh`. Single source of truth for host-specific placement: `PLACEMENT_PREFIX` (named placement → command prefix, keyed on `hostname -s`), `BANDWIDTH_RUNS`, and `CEILING_MAP` (thread-count → `bandwidth.txt` section label, consumed by `perf-viz.py`). An unrecognized host falls back to an uncalibrated default plus a stderr warning; add a new `case` arm rather than editing an existing one.
- `scripts/ab-compare.sh <name> --a <rev|.> --b <rev|.> --probe '<args>' [options]` — the interleaved A/B harness. `--probe-b '<args>'` gives side B its own probe args (default: same as `--probe`) for a runtime-knob A/B — e.g. `--partitions` — on one binary rather than a code change; see the P=1-vs-P=2 recipe under Threading below. When `--probe-b` is given, both sides sharing a tree/commit gets a note instead of the usual smoke-mode warning, and the report runs with `--pair-on layer,threads`.
- `python3 scripts/ab-report.py A.jsonl B.jsonl [--field wall_ns] [--all-phases] [--pair-on layer,threads[,partitions]]` — the paired-delta reporter, re-invocable on archived sidecars. `--pair-on` (default `layer,threads,partitions`) sets which fields identify a cell; drop `partitions` to pair a P=1 run against a P=2 run of the same `(layer, threads)`.
- `scripts/profile.sh probe --n 1000000 --threads 32 --layers rotation_zz` — flamegraph the `phase_breakdown` probe.
- `scripts/profile.sh bench apply_layer_bucketed 10` — flamegraph a criterion bench group via `--profile-time <seconds>` (default 10s).
- `scripts/profile.sh bin ./target/release/some_bin --args` — flamegraph an arbitrary prebuilt binary, no build step.
- `scripts/perf-stat.sh --n 1000000 --threads 32 --layers rotation_zz` — hardware counters (cycles/string, IPC, LLC miss rate) plus a DRAM pass for one `(layer, thread count)` cell. `PROBE=/path/to/binary` skips the `cargo build` and measures a prebuilt binary instead.
- `scripts/bandwidth.sh` — the host's STREAM-style bandwidth ceiling across the placement matrix in `host-topology.sh`; run once per host or after hardware changes. First stdout line is the `# ceiling-map:` header `perf-viz.py` parses.
- `python3 scripts/criterion-report.py snapshot out.json --filter <substr> [--merge]` — snapshot `target/criterion/` to JSON. `--merge` updates an existing `out.json` instead of overwriting.
- `python3 scripts/criterion-report.py compare old.json new.json` — markdown Δ-table between two snapshots.
- `python3 scripts/perf-viz.py benchmarks/results/<date>-<host>/<campaign> [--compare OLD.json]` — render one campaign's data into a self-contained `<campaign>-report.html`; `--compare` adds Δ% columns to the criterion table. `bench-campaign.sh` runs this automatically at the end of every campaign.
- `python3 benchmarks/python/bench_baseline.py` — the cross-library baseline harness under `benchmarks/python/`; see `benchmarks/README.md` for its scope.

## Machine contracts (what is logged where)

The tools communicate through file formats, not each other's code — a format change in one silently breaks a consumer elsewhere unless the coupling is written down.

**(a) The probe's `--json-out` sidecar** (one JSON object per line) is the sole input to `perf-viz.py`'s phase-breakdown section and, via `ab-compare.sh`'s per-side sidecars, to `ab-report.py`. Fields, from `phase_breakdown.rs`'s `json_line` — verify against `PhaseStats` in
`crates/paulistrings/src/engine/stats.rs` before relying on any name: `layer`, `truncation`, `threads`,
`n`, `reps`, `qubits`, `seed`, `hash_seed`, `bucket_bits`, `wall_ns`; the wall-clock phases
`rebucket_ns`, `prepare_ns`, `rescale_ns`,
`span_plan_ns`, `permute_ns`, `coset_loop_ns`, `unpermute_ns`, `recount_ns`, `finalize_ns`; the worker
busy-time phases `swap_ns`, `size_ns`, `gather_ns`, `sort_ns`, `merge_ns`, `clear_ns`; and the counters
`layers`, `cosets`, `runs`, `rows_gathered`, `rows_sorted`, `rows_id`, `terms_in`, `terms_out`, `vmrss_kb`,
`vmhwm_kb`, `target_bucket_len`, `min_buckets`. `n` is the *steady-state* term count after an untimed warm-up call, not necessarily the
requested `--n` — see Phase timing below. `perf-viz.py` keeps only the *last* line per
`(layer, threads, partitions)` key, so an appended re-run overwrites the earlier one in the rendered
report.

The partitioned engine (`--partitions <csv>`, default `1`) adds: `partitions` (int), `partition_cpus`
(string, the `--partition-cpus` argument echoed back — `"auto"`, `"unpinned"`, or the `"<list>;<list>"`
lists), `pin_memory` (0/1, from `--bind-memory`), `gen_qubits` (2-element list, the `(q0, q1)` a
`rotation_*` layer rotated about — `[0, 1]` for every other layer), `local_layers`, `remote_layers`,
`rows_exported`, `bytes_exported`, `partition_terms_in` (list, one entry per partition),
`partition_imbalance` (float), `export_ns`, `exchange_ns`, `barrier_ns`, and `partition_coset_loop_ns`
(list, one entry per partition).

Six further keys carry the workload and row-policy axes the two Trotter-step
layers (`tfim_step`, `heavyhex_step`) brought with them, and are written on every row:
`initial` (string, `"random"` or `"z0"` — the cell's input sum, whose default is per layer, so
read it rather than assuming the run's flag), `partition_rows` (string, `"random"` / `"cut"`),
`rows_remote_gens` (int) and `rows_remote_weight` (float) — the distinct key-delta
masks those rows leave remote and the number of *layers* carrying one, the same scale for all
three policies — and the two per-layer series `partition_imbalance_by_layer` (list of floats, one
`max/mean` of the partitions' input term counts per layer, in application order) and
`terms_by_layer` (list of ints, the group's total terms in per layer). The cell-level
`partition_imbalance` sums the layers before dividing and so hides the mixing dynamics; the series
is what shows them, and `terms_by_layer` is the growth curve to read it against. Both are long —
a four-step heavy-hex cell is 1084 layers — and both are empty on an unpartitioned row.
`--format tsv` carries all six as trailing columns of the same names, the two series
`|`-joined. `hash_seed` on a partitioned row is the seed actually used, which a `cut`/`select`
cell may have re-drawn to keep its rows independent of `H`'s (it says so on stderr); that also
moves the coset dimension, so do not compare such a cell's phase timings against a differently
seeded one.

A distributed cell (`--mpi`) adds two more, and only there: `rank` and `ranks`. Each rank writes its
own sidecar file, `<--json-out path>.rank<N>`.

Two further keys are **sub-phases, contained in the phase above rather than additional to it**, so
never add them to a total: `append_ns`, worker busy time inside `gather_ns`, for merging received rows
into the output buckets' rest streams, of which `chunk_wait_ns` is the part spent blocked waiting for a
chunk of those rows to land.

**The exchange is two-phase, so `exchange_ns` is small and the transfer shows up inside the coset
loop** (`ARCHITECTURE.md` §Partitioning): the rows arrive while the layer runs, and `chunk_wait_ns` is
what the loop failed to hide. Read the pair together — a remote layer whose `chunk_wait_ns / threads`
is close to `exchange_ns` hid nothing, and one where it is near zero is compute-bound.

**Every consumer of this sidecar must read every one of these — old and
new alike — with `row.get(key, default)`, never a bare index/key lookup**: a sidecar written before the
partitioned engine landed has none of them, and `partitions` defaults to `1` in that case (an
unpartitioned probe run is P=1, not absent data). `ab-report.py` and `perf-viz.py` both follow this rule;
match it in any new consumer.

Two details of how the probe fills those fields:

- **Every row carries all of them, partitioned or not** — one schema per campaign. An unpartitioned
  (P=1) row has `partitions: 1`, zero counters, empty `partition_terms_in` /
  `partition_coset_loop_ns` lists and `partition_imbalance: 1.0`, and its `partition_cpus`/`pin_memory`
  echo the flags even though no placement was applied. A consequence for `ab-report.py`: in a P=1-vs-P=2
  knob A/B, `export_ns`/`exchange_ns`/`barrier_ns` are 0 on side A, so `--all-phases` reports them as
  "not present in these runs" (a Δ% against zero is undefined) — read side B's absolute values from the
  sidecar or the table format instead.
- `barrier_ns` is the sidecar's name for the engine's `PhaseStats::collective_ns`: the driver's per-layer
  bucket-count all-reduce, which is the one collective every layer makes (an all-local layer issues no
  other transport call). There is no barrier in the engine.
- The wall-clock phases of a partitioned row are the **maximum over partitions** (the group's critical
  path), the busy-time phases and every counter are **sums** over partitions, and `layers` is the
  driver's layer count rather than a sum — the fold lives in `phase_breakdown.rs::fold_partition_stats`.

**(b) The probe's stdout `cell` line.** Every cell prints one `cell layer=<name> threads=<n> n=<n> layers=<n> wall_ms=<f> trunc=<spec>` line, in every `--format`; the partitioned engine appends ` partitions=<P>` after `trunc=`.
`perf-stat.sh`'s awk greps this literal shape (`n=` and `layers=` field prefixes) to compute cycles/string, scanning every field for those two prefixes rather than assuming a fixed field count.
Change the line's fields or order, fix `perf-stat.sh`'s awk in the same change.

**(c) Criterion snapshot JSON.** `criterion-report.py snapshot` and `bench-campaign.sh`'s `criterion:`/`scaling:` items write `{full_id: {median_ns, mean_ns, stddev_ns, throughput_elems, melem_per_s}}`, consumed by `compare` and `perf-viz.py`'s criterion and scaling sections.
Thread-scaling groups rely on a naming contract: a `BenchmarkId` of `<group>/<threads>` where `<group>` starts with `thread_scaling` and `<threads>` is a bare integer; `perf-viz.py` splits on the last `/` and parses the tail as an int, silently skipping anything else.

**(d) `bandwidth.txt`**, written by `bandwidth.sh` and read by `perf-viz.py`'s bandwidth section and roofline model: an optional first line `# ceiling-map: <key>=<label>;...` (keys are thread counts or `default`), then one or more `=== <section label> ===` headers each followed by `kernel=<name> threads=<n> mib=<n> reps=<n> best_gbps=<f> avg_gbps=<f>` lines from `crates/membench`.
Without a ceiling-map header, `perf-viz.py` falls back to a hard-coded ccqlin038-shaped thread-count → section table.

**(e) The campaign results directory** (`benchmarks/results/<date>-<host>/`). Per campaign name:
`<name>.txt`, `<name>.json`, `<name>-probe.json` (the probe's JSONL sidecar, despite the `.json` extension), one `<name>-scaling-<placement>.json` per `scaling:` placement, and `<name>-report.html`.
`bandwidth.txt` and `flamegraph-<name>-<shortcommit>[-dirty].html` (+ `.meta.txt`) live directly in the dated directory, shared across every campaign there.
A/B runs add `<name>-ab.log`, `<name>-{a,b}.probe.jsonl`, and the archived binaries `<name>-{a,b}-<sha|worktree>[-dirty]`.

## Phase timing

The `phase-timing` Cargo feature (`crates/paulistrings/src/engine/stats.rs`) gates a per-phase counter breakdown of the propagation engine.
It is measurement-only and never in the default feature set — the default build carries no timing code and no stats fields at all.
The same bitwise-identity tests (fingerprint net, thread-count/bucket-count/seed determinism) run with the feature enabled in CI, as the acceptance test that instrumentation doesn't perturb output.

Counters are read via `LayerScratch::take_stats()` after driving layers through `propagate_with_scratch`.
Two clock domains are deliberately mixed in one `PhaseStats`:

- **Wall-clock phases** (`rebucket_ns` … `finalize_ns`) — measured once per layer on the calling thread; they sum to approximately the layer's wall time.
- **Worker busy-time phases** (`swap_ns` … `clear_ns`) — summed across every coset task on every Rayon worker. Under a `t`-thread pool they sum to `coset_loop_ns × t × efficiency`, not to `coset_loop_ns` itself. `Σbusy / (coset_loop_ns × t)` is the coset loop's parallel efficiency.

The probe runs each `(layer, threads)` cell twice inside a dedicated Rayon pool: an untimed warm-up call, then a timed call whose input is the warm-up's output, so the timed call measures steady-state cost.
`trotter` additionally self-caps its input at `TROTTER_MAX_N` regardless of `--n` (64 distinct generators under no truncation grow combinatorially rather than closing).
The probe also prints its own timer-overhead estimate (`PhaseStats::timer_reads() × stats::TIMER_READ_OVERHEAD_NS`) so you can see when the measurement pollutes itself.

### The partition axis

`--partitions <csv>` (default `1`) sweeps partition counts alongside `--layers` and `--threads`.
A `P > 1` cell scatters the input across a `PartitionRuntime` of `P` pinned pools outside the timed region, then runs the same warm-up + timed pair through `PartitionedSum::propagate_with_options`.

- **`--threads` is the TOTAL thread count** at every `P`: each partition's pool gets `threads / P` workers. A `--threads` value not divisible by `--partitions` is a startup error, and so is `P > threads`.
- `--partition-cpus` takes `auto` (default — `Placement::Auto`, one partition per NUMA node in the mask, capped at `P`), `unpinned` (`Placement::Unpinned`, no pinning at all), or the `'<list>;<list>'` cpulists of `Placement::Explicit` — the same string `host-topology.sh`'s `PARTITION_CPUS` holds.
- `--bind-memory 0|1` (default 1) is `PartitionConfig::bind_memory`; `--partition-seed <u64>` fixes the GF(2) partition rows instead of deriving them from the sum's hash seed.
- `P = 1` runs the unpartitioned engine — the same code path, policy value and output as before the partition axis existed.
- Layers `rotation_local` and `rotation_remote` are a `ZZ` rotation on `(0, q)` with `q` the smallest qubit whose layer has, respectively, no remote delta and at least one, under that cell's partition rows. Both collapse to `rotation_zz` at `P = 1`; the chosen pair goes to stderr and into the sidecar's `gen_qubits`.
- `--partition-rows random|cut` (default `random`) chooses the rows: `random` is `PartitionRows::from_seed`; `cut` is `PartitionRows::cut` over `P` contiguous qubit blocks, chosen by an exact DP over the layer's own graph to cross as few two-qubit generators as possible at ±25% size balance. Both report `rows_remote_gens` / `rows_remote_weight` in the sidecar.
- Layers `tfim_step` (a 1D open chain of `--qubits` qubits) and `heavyhex_step` (the fixed 127-qubit Eagle r3 lattice, `--qubits >= 127`) are the rotation-only kicked-Ising Trotter steps the row policies exist for: `--reps` is the step count, both default to `--initial z0`, and both need a truncation policy to converge (`coeff:1.220703125e-4` is `2^-13`).
- `--truncation topn:<N>` is refused for a partitioned cell: `TopN` has no `PartitionedTruncation` impl. `atopn:<N>`, `coeff:<t>` and `keep` all run.

### The rank axis

`--mpi` (feature `mpi`) runs the same cells as one partition per process, taking the group from `MPI_COMM_WORLD` instead of from `--partitions`, which must stay at its default `1`.
Leave `--partition-cpus` at `auto`: the launcher's affinity mask is the placement.

```bash
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH=$(llvm-config --libdir)
cargo build --release --features phase-timing,mpi --example phase_breakdown

mpirun -n 4 --map-by ppr:1:numa --bind-to numa \
  target/release/examples/phase_breakdown --mpi --threads 32 \
  --layers rotation_local,rotation_remote --n 4000000 --reps 8 --json-out out.jsonl
```

- **`--threads` is per rank** here, not a total to divide.
- **The input is replicated.** Every rank builds `--n` terms and keeps its own share, so `vmhwm_kb` grows with the rank count at constant terms per rank — a probe artifact, not the engine; hold terms per rank fixed by scaling `--n` with the rank count.
- **Each rank writes `<path>.rank<N>`** and prints its own `cell` line. Take medians over ranks; a spread between ranks on the same cell is arrival skew, visible in `exchange_ns` and `barrier_ns`.
- **`chunk_wait_ns` is only filled here.** The in-process transport has no transfer to wait on, so it is zero for a `--partitions` cell and nonzero for an `--mpi` one.
- **`PAULISTRINGS_EXCHANGE_CHUNKS`** overrides the pipeline's chunk count, read once per process (needs `mpirun -x`). `K = 1` is the two-phase shape with no pipelining, the control for "did the overlap do anything".
- Reference numbers and the weak-scaling table: `research/HARDWARE.md`. On Rusty, `scripts/slurm/mpi-ranks.sbatch` runs the differential net and then this probe across the allocation.

## Flamegraphs

`scripts/profile.sh` has three modes: `probe` (builds and profiles `phase_breakdown` under the `profiling` Cargo profile), `bench` (profiles a criterion bench group via `--profile-time`, built with `cargo bench --no-run`), and `bin` (profiles an arbitrary prebuilt binary, no build step).

The `[profile.profiling]` build inherits `release` (same `lto = "fat"`, same `codegen-units = 1`) and adds `debug = "line-tables-only"` plus `strip = "none"` so `perf`/`addr2line` can expand LTO-inlined frames.
`PROFILE_MODE` picks the stack-walking method: `dwarf` (default, `perf record --call-graph dwarf,16384`) or `fp` (frame pointers, forcing `RUSTFLAGS="-Cforce-frame-pointers=yes $JCC_RUSTFLAGS"` and a full rebuild for `probe`/`bench`; for `bin` it only changes the `perf record` flag, so you are responsible for having built that binary with frame pointers).

Output: `benchmarks/results/<date>-<host>/flamegraph-<name>-<shortcommit>[-dirty].html` plus a sidecar `.meta.txt` (host, date, full commit, rustc version, mode, frequency, exact command line).

Caveats:
- Criterion's `--profile-time` loops the routine without statistical sampling — ignore `criterion::` and setup frames in the graph.
- Rayon idle spinning shows up as `crossbeam_epoch::*` frames; don't mistake it for real work.
- `perf.data` with DWARF call graphs is GB-scale; `profile.sh` records into a scratch temp dir cleaned up automatically after conversion.
- `#[inline(never)]` is attribution-of-last-resort for pinning a frame down in a flamegraph — never add it just to make quoted numbers look cleaner, and never quote a number that depended on it.

## Counters & bandwidth

`scripts/perf-stat.sh` runs two passes over the `phase_breakdown` probe:

- **Pass A** (per-process): `cycles`, `instructions`, `LLC-loads`, `LLC-load-misses`, `branches`, `branch-misses`. Derives IPC, LLC miss rate, and cycles/input-string. Cycles/string is frequency-robust but whole-process, so with more than one cell in a run it's a blend that converges from above as `--n`/`--reps` grow.
- **Pass B** (system-wide uncore IMC): `uncore_imc/cas_count_read/` and `uncore_imc/cas_count_write/`, `--per-socket`, unavoidably `-a` (whole box) on a shared host. An idle baseline (`IDLE_SECS`, default 3s) is measured immediately before, also `--per-socket`, and subtracted per socket. Units come pre-scaled to MiB by `perf` itself. Per-socket attributable read/write GB/s is reported alongside `% of per-socket ceiling` against the ccqlin038 one-socket ceilings (read 39.0 GB/s, write 18.6 GB/s — `research/HARDWARE.md`); these two constants are ccqlin038-specific and need updating before trusting the percentage on another host.
- **Pass C** (per-process retired NUMA loads): `mem_load_l3_miss_retired.local_dram`, `.remote_dram`, `.remote_hitm`, printed as raw counts plus `remote load share = remote/(local+remote)`. These are retired loads only; the write stream's NUMA locality is visible only in pass B. Guarded by `perf list | grep -q mem_load_l3_miss_retired.remote_dram`; a host/kernel/perf build without these events prints a one-line skip instead of failing.

## Roofline model

Measured bandwidth ceilings come from `scripts/bandwidth.sh` (membench, `crates/membench/src/main.rs`): STREAM-convention nominal bytes (copy = 16 B/elem, triad = 24 B/elem), plain (write-allocating) stores, read-for-ownership deliberately uncorrected — this matches the propagation engine's own store pattern.

Bytes-moved model per layer, at `W = 2` (48 B/term = 32 B key + 16 B coeff), over `n` input rows and `n_out` gathered rows:

- **gather** ≈ read `n × 48` + write `n_out × 48`
- **sort** ≈ a few read+write passes over `n_out × 48`
- **merge** ≈ read `n_out × 48` + `n_existing × 48`, write `n_merged × 48`

Serial phases (rebucket, recount) compare against the single-core ceiling; the coset loop against the all-core ceiling for whatever placement was used to measure it.

`perf-viz.py` computes this automatically per probe cell ("DRAM: X GB/s = Y% of ceiling") from `rows_gathered`, `rows_sorted`, and `rows_id`: the identity-delta stream skips the sort and carries no tag byte, and a dense identity row materializes only its 16-byte coefficient, modeled as coset-cache-resident:

`bytes/layer = terms_in×T + 2×(rows_gathered−rows_id)×T + 2×rows_id×16 + 2×rows_sorted×T + terms_out×T
              + 2×bytes_exported`

with `T = 16·W + 16` on the coset path, or `2×terms_in×T` on the rescale path.
A probe line carrying `rows_gathered`/`rows_sorted` but no `rows_id` field is priced with `rows_id` treated as 0; a line carrying neither `rows_id` nor `rows_sorted` falls back to the older `4×rows_gathered×(T+1)` model.
The `2×bytes_exported` term (zero on a P=1 line, absent on a pre-partitioning line) prices a partition's exported rows the same as any row written once and read once elsewhere.
This is divided by wall time and by the membench triad ceiling at a comparable core count (per the host's `# ceiling-map:` header — Machine contracts (d)).
Read the result as a classification: over 100% means the modeled traffic is mostly cache-served; near 100% is genuinely at the wall; far below 100% with high wall time points at latency, serial phases, or imbalance instead.

Rule of thumb: at or above ~70% of the measured ceiling, the phase is bandwidth-bound. Far below the ceiling with a high LLC miss rate points at a latency/working-set problem instead.

## Threading

`scripts/host-topology.sh` is the single place CPU/NUMA placements are defined, keyed on `hostname -s`.
The reference host (ccqlin038: 2 sockets, node0 physical CPUs 0-7 / HT 16-23, node1 physical CPUs 8-15 / HT 24-31) is calibrated as:

| placement | prefix | isolates |
|---|---|---|
| `default` | (none) | whatever the ambient scheduler does |
| `node0` | `numactl --cpunodebind=0 --membind=0` | NUMA cost (vs `default`/spread) |
| `phys16` | `taskset -c 0-15` | both sockets, physical cores only |
| `smt16` | `taskset -c 0-7,16-23 numactl --membind=0` | HT yield vs cross-socket (compare to `phys16` / node0 8-core) |
| `phys8` | `taskset -c 0-7 numactl --membind=0` | pure single-node core scaling |

An unrecognized host gets only the no-op `default` placement plus an uncalibrated `BANDWIDTH_RUNS`/`CEILING_MAP` and a stderr warning; add a case arm in `host-topology.sh` to calibrate a new host.

**Partitioned cells run under no placement prefix.** The partitioned engine pins its own worker threads via `--partition-cpus "<list>;<list>"` (`host-topology.sh`'s `PARTITION_CPUS` map, ccqlin038: `node2x8="0-7;8-15"`, `node2x16="0-7,16-23;8-15,24-31"`), so wrapping a P>1 run in a `PLACEMENT_PREFIX` entry fights that pinning: `numactl --membind` forces every page onto one node regardless of which partition touches it, and `--cpunodebind`/`taskset` shrink the CPU mask the engine's own `Auto` placement reads.
The one legitimate combination is P=1 under the existing `node0`/`phys8` placements, used as the one-socket reference point.

**P=1 vs P=2, same binary (a runtime-knob A/B):**
```bash
scripts/ab-compare.sh partitions-1v2 --a . --b . \
  --probe '--n 1000000 --threads 16 --layers rotation_zz --partitions 1' \
  --probe-b '--n 1000000 --threads 16 --layers rotation_zz --partitions 2' \
  --pairs 5 --order abba
```
One `(layer, threads)` cell per invocation keeps the report's cell-by-cell pairing unambiguous.
Because `--a`/`--b` are the same tree, `ab-compare.sh` recognizes `--probe-b` and swaps the usual smoke-mode warning for a note that this measures the probe-arg difference, not a code change; it also passes `--pair-on layer,threads` to `ab-report.py`.

Both sides carry the same `--threads` because that flag is the total thread count on either side, which is what makes the comparison a partitioning comparison rather than a thread-count one; it must be divisible by `P`.
Add `--partition-cpus '<list>;<list>'` to side B to pin the split to sockets instead of taking `Auto`'s NUMA-node reading of the mask, and remember that a P>1 side takes no placement prefix.
`scripts/slurm/ab-campaign.sbatch` runs exactly this shape on an exclusive node, deriving `P` and the lists from the node's sysfs.

`node0` vs a spread/default placement isolates NUMA cost; `phys16` vs `smt16` isolates hyperthread yield against cross-socket cost; `node0` scaled 1→8 threads is pure core scaling, useful for Amdahl/USL fits since it has no NUMA or HT crossover to confound the fit.

## Host caveats

- This host's single-shot campaign noise floor is ~±5–8% at 1 thread and ~±10–26% at 8–32 threads.
  Trust an absolute campaign-to-campaign comparison only above that floor; below it, use the interleaved A/B protocol.
  `bench-campaign.sh` logs load average at the start and end of every campaign — check it before trusting a close call.
- The CPU governor is `powersave` and cannot be pinned to `performance` (no root on the reference host).
  Prefer frequency-robust ratios (IPC, percent of the measured bandwidth ceiling, speedup ratios, cycles/string) over absolute millisecond figures across days.
- Run every campaign (and every A/B run) with `RUST_LOG` unset (and no logger installed): with no logger the engine's per-layer progress logging is a single static level check and allocates nothing, whereas an enabled `debug` filter adds a formatted line and a clock read per layer.
