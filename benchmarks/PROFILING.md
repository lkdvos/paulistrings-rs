# Profiling

Operational reference for the performance-measurement framework: how to
run a measurement campaign, read its output, and decide what it means.
See `benchmarks/README.md` for the three benchmark *surfaces*; this doc
is about the tools that sit underneath the Rust microbenchmark surface.

## The standard loop

The point of the whole framework is to make "did my change help" a
repeatable, three-command loop:

1. Change engine code.
2. Run a campaign:
   ```bash
   scripts/bench-campaign.sh <name> \
     criterion:apply_layer_bucketed \
     probe:'--n 1000000 --threads 1,8,32' \
     perf-stat:'--n 1000000 --threads 32 --layers rotation_zz'
   ```
3. Open the HTML report the campaign rendered at the end
   (`benchmarks/results/<date>-<host>/<name>-report.html` — phase-breakdown
   bars, throughput and scaling charts, criterion table, bandwidth ceilings;
   self-contained, no network), then diff against the last snapshot:
   ```bash
   python3 scripts/criterion-report.py compare <old-snapshot>.json <new>.json
   ```
   and attach the resulting Δ-table (and a flamegraph, if something moved
   enough to be worth explaining) to a `research/notes/` entry.

Snapshots live next to the campaign's raw-output `.txt` file, in
`benchmarks/results/<date>-<host>/<campaign-name>.json` — `criterion:`
items write it, `compare` reads two of them (from the same or different
campaigns/days). Everything a campaign does is logged into the `.txt`
sibling: provenance header (commit, rustc version, thread count,
governor, CPU, load average at start and end) followed by one `=== item
===` section per item with that tool's combined stdout+stderr. A
`criterion-report.py compare` is a reporting tool, not a gate — it always
exits 0, so treat REGRESSION/IMPROVED annotations (±5% default threshold)
as prompts to look, not as pass/fail.

## Tool one-liners

- `scripts/bench-campaign.sh <name> <item>...` — runs a sequence of
  `criterion:<filter>`, `probe:<args>`, `perf-stat:<args>`,
  `scaling:<placement>`, `macro`, or `bandwidth` items and logs them to
  `benchmarks/results/<date>-<host>/<name>.txt` (append-on-rerun, never
  overwrite). `scripts/bench-campaign.sh --help` prints the full item
  menu.
- `scripts/profile.sh probe --n 1000000 --threads 32 --layers rotation_zz`
  — flamegraph the `phase_breakdown` probe.
- `scripts/profile.sh bench apply_layer_bucketed 10` — flamegraph a
  criterion bench group via `--profile-time <seconds>` (default 10s).
- `scripts/profile.sh bin ./target/release/some_bin --args` — flamegraph
  an arbitrary prebuilt binary, no build step.
- `scripts/perf-stat.sh --n 1000000 --threads 32 --layers rotation_zz` —
  hardware counters (cycles/string, IPC, LLC miss rate) plus a DRAM
  bandwidth pass for one `(layer, thread count)` cell.
- `scripts/bandwidth.sh` — the host's STREAM-style bandwidth ceiling
  across the placement matrix; run once per host or after hardware
  changes.
- `python3 scripts/criterion-report.py snapshot out.json --filter <substr>`
  — snapshot `target/criterion/` to JSON.
- `python3 scripts/criterion-report.py compare old.json new.json` —
  markdown Δ-table between two snapshots.
- `python3 scripts/fit_scaling.py --group thread_scaling_bucketed_gu2q`
  (or `--group all`) — Amdahl/USL fits over a `thread_scaling*` criterion
  group.
- `python3 scripts/perf-viz.py benchmarks/results/<date>-<host>/<campaign>`
  — render one campaign's data (`.txt`, `.json`, `-probe.json`,
  `-scaling-*.json`, plus the directory's `bandwidth.txt`) into a single
  self-contained `<campaign>-report.html`; `--compare OLD.json` adds Δ%
  columns to the criterion table. `bench-campaign.sh` runs this
  automatically at the end of every campaign; the probe feeds it via its
  `--json-out` sidecar (one JSON line per cell, independent of `--format`).

## Phase timing

The `phase-timing` Cargo feature (`crates/paulistrings/src/engine/stats.rs`)
gates a per-phase counter breakdown of the propagation engine. It is
**measurement-only and never in the default feature set** — the default
build carries no timing code and no stats fields at all; the same
bitwise-identity tests (the fingerprint net, thread-count/bucket-count/seed
determinism tests) run *with the feature enabled* in CI as the acceptance
test that instrumentation doesn't perturb output, since the timers only
read the clock and add to plain integers.

Counters are read via `LayerScratch::take_stats()` after driving layers
through `propagate_with_scratch`. There are two clock domains, deliberately
mixed in one `PhaseStats`:

- **Wall-clock phases** (`rebucket_ns` … `finalize_ns`) — measured once
  per layer on the calling thread; they sum to approximately the
  layer's wall time.
- **Worker busy-time phases** (`swap_ns` … `clear_ns`) — summed across
  every coset task on every Rayon worker. Under a `t`-thread pool they
  sum to `coset_loop_ns × t × efficiency`, not to `coset_loop_ns` itself.
  `Σbusy / (coset_loop_ns × t)` **is** the coset loop's parallel
  efficiency — the gap between the two domains is the load-balance
  signal.

The probe (`cargo run --release --features phase-timing --example
phase_breakdown`) prints its own timer-overhead estimate
(`PhaseStats::timer_reads() × stats::TIMER_READ_OVERHEAD_NS`) next to
the breakdown, so you can see when the measurement is polluting itself
(tiny cosets, many runs inflate the read count).

## Flamegraphs

`scripts/profile.sh` has three modes: `probe` (builds and profiles
`phase_breakdown` under the `profiling` Cargo profile), `bench` (profiles
a criterion bench group via `--profile-time`, built with `cargo bench
--no-run` and located under `target/release/deps/`), and `bin` (profiles
an arbitrary prebuilt binary, no build step).

The `[profile.profiling]` build (`Cargo.toml`) inherits `release` (same
`lto = "fat"`, same `codegen-units = 1` — it profiles what ships) and adds
`debug = "line-tables-only"` plus `strip = "none"` so `perf`/`addr2line`
can expand LTO-inlined frames. `PROFILE_MODE` picks the stack-walking
method: `dwarf` (default) — `perf record --call-graph dwarf,16384`,
works against the normal profiling build; or `fp` — frame pointers,
which forces `RUSTFLAGS="-Cforce-frame-pointers=yes"` and a full rebuild
for `probe`/`bench` (codegen differs from the cached artifacts), and for
`bin` only changes the `perf record` flag — you're responsible for having
built that binary with frame pointers yourself.

Output: `benchmarks/results/<date>-<host>/flamegraph-<name>-<shortcommit>[-dirty].html`
plus a sidecar `.meta.txt` (host, date, full commit, rustc version, mode,
frequency, exact command line).

Caveats:
- Criterion's `--profile-time` loops the routine without statistical
  sampling — ignore `criterion::` and setup frames in the graph.
- Rayon idle spinning shows up as `crossbeam_epoch::*` frames; don't
  mistake it for real work.
- `perf.data` with DWARF call graphs is GB-scale; `profile.sh` records
  into a scratch temp dir that's cleaned up automatically after
  conversion to HTML.
- `#[inline(never)]` is attribution-of-last-resort for pinning a frame
  down in a flamegraph — never add it just to make quoted numbers look
  cleaner, and never quote a number that depended on it.

## Counters & bandwidth

`scripts/perf-stat.sh` runs two passes over the `phase_breakdown` probe:

- **Pass A** (per-process): `cycles`, `instructions`, `LLC-loads`,
  `LLC-load-misses`, `branches`, `branch-misses`. Derives IPC, LLC miss
  rate, and cycles/input-string. Cycles/string is frequency-robust
  (matters on a powersave-governed host) but is **whole-process** — it
  includes input generation and warm-up, so with more than one cell in a
  run it's a blend (the script warns), and it converges from above as
  `--n`/`--reps` grow.
- **Pass B** (system-wide uncore IMC): `uncore_imc/cas_count_read/` and
  `uncore_imc/cas_count_write/`, `--per-socket`, unavoidably `-a` (whole
  box) on a shared host. An idle baseline (`IDLE_SECS`, default 3s) is
  measured immediately before and subtracted as a rate — still
  approximate when load average is non-trivial. Units come pre-scaled to
  MiB by `perf` itself; don't multiply by 64 again.

## Roofline model

Measured bandwidth ceilings come from `scripts/bandwidth.sh` (membench,
`crates/membench/src/main.rs`): STREAM-convention nominal bytes (copy =
16 B/elem, triad = 24 B/elem), plain (write-allocating) stores, read-for-
ownership deliberately **uncorrected** — this matches the propagation
engine's own store pattern, so the nominal figure is the right ceiling to
compare phases against, not the true DRAM CAS traffic (cross-check that
with the uncore pass if you want it).

Bytes-moved model per layer, at `W = 2` (48 B/term = 32 B key + 16 B
coeff, +1 B gather tag), over `n` input rows and `n_out` gathered rows:

- **gather** ≈ read `n × 48` + write `n_out × 49`
- **sort** ≈ a few read+write passes over `n_out × 49`
- **merge** ≈ read `n_out × 49` + `n_existing × 48`, write `n_merged × 48`

Serial phases (rebucket, recount) compare against the **single-core**
ceiling; the coset loop itself compares against the **all-core** ceiling
for whatever placement was used to measure it.

`perf-viz.py` computes this automatically per probe cell ("DRAM: X GB/s =
Y% of ceiling") from `PhaseStats::rows_gathered`, `rows_sorted`, and
`rows_id` (v0.5: the id stream skips the sort, and there is no tag byte;
v0.6 G1d: a *dense* identity row — rotations, general unitaries —
materializes only its 16-byte coefficient, its keys borrowed in place from
the source bucket and modeled as coset-cache-resident):
`bytes/layer = terms_in×T + 2×(rows_gathered−rows_id)×T + 2×rows_id×16 +
2×rows_sorted×T + terms_out×T`
with `T = 16·W + 16` on the coset path, or `2×terms_in×T` on the rescale path
(v0.5 probe JSON without `rows_id` prices every gathered row at full T;
pre-v0.5 lines without `rows_sorted` fall back to the old
`4×rows_gathered×(T+1)` model),
divided by the layer's wall time and by the membench **triad** ceiling at a
comparable core count (1 / ≤8 / ≤16 / 32). Read it as a classification, not
a gauge: **over 100% means the modeled traffic is mostly served from cache**
(small per-coset working sets — not DRAM-bound); near 100% is genuinely at
the wall (e.g. the Trotter step at 8 threads); far below 100% with high
wall time points at latency, serial phases (gu2q's rebucket), or imbalance.

Rule of thumb: at or above ~70% of the measured ceiling, the phase is
bandwidth-bound — stop optimizing arithmetic. Far below the ceiling with
a high LLC miss rate points at a latency/working-set problem instead.

## Threading

The placement matrix (from `bench-campaign.sh`'s `scaling:<placement>`
items and `bandwidth.sh`) is calibrated to ccqlin038's topology: 2
sockets, node0 physical CPUs 0-7 (HT 16-23), node1 physical CPUs 8-15 (HT
24-31). Adjust the masks for other hosts.

| placement | prefix | isolates |
|---|---|---|
| `default` | (none) | whatever the ambient scheduler does |
| `node0` | `numactl --cpunodebind=0 --membind=0` | NUMA cost (vs `default`/spread) |
| `phys16` | `taskset -c 0-15` | both sockets, physical cores only |
| `smt16` | `taskset -c 0-7,16-23 numactl --membind=0` | HT yield vs cross-socket (compare to `phys16` / node0 8-core) |
| `phys8` | `taskset -c 0-7 numactl --membind=0` | pure single-node core scaling |

Comparisons: `node0` vs a spread/default placement isolates NUMA cost;
`phys16` (16 physical cores, both sockets) vs `smt16` (8 physical + 8 HT
siblings, one socket) isolates hyperthread yield against cross-socket
cost; `node0` scaled 1→8 threads is pure core scaling, useful for
Amdahl/USL fits since it has no NUMA or HT crossover to confound the fit.

`scripts/fit_scaling.py --group <thread_scaling_group>` fits Amdahl's law
and the Universal Scalability Law to a `thread_scaling*` criterion group
and reports the serial fraction / σ,κ with R². Cross-check the fitted
serial fraction against the probe's own measured serial share (wall-clock
phases minus the coset loop, divided by total wall time) — the two
should roughly agree; if they don't, something outside the modeled coset
loop is eating parallelism.

## Host caveats

- The reference host is a shared box: expect a ±5-10% noise floor.
  `bench-campaign.sh` logs load average at the start and end of every
  campaign for exactly this reason — check it before trusting a close
  call.
- The CPU governor is `powersave` and cannot be pinned to `performance`
  (no root on the reference host). Prefer ratios that are frequency-robust
  — IPC, percent of the measured bandwidth ceiling, speedup ratios,
  cycles/string — over absolute millisecond figures when comparing across
  days.
