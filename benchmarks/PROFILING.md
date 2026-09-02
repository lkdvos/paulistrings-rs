# Profiling

Operational reference for the performance-measurement framework: how to run a measurement campaign, read its
output, and decide what it means. See `benchmarks/README.md` for the three benchmark *surfaces*; this doc is
about the tools underneath the Rust microbenchmark surface.

Two protocols share the same probe (`examples/phase_breakdown.rs`) and `phase-timing` instrumentation,
differing only in how runs are scheduled and compared. **Campaigns** (`bench-campaign.sh`) run once and log
everything to a dated results directory — good for exploring, and for effects large enough to clear this
host's single-shot noise floor. **Interleaved A/B** (`ab-compare.sh`) builds two binaries and alternates
them adjacent in time — the only protocol that can resolve effects the campaign noise floor would otherwise
swallow.

## The standard loop

1. Change engine code.
2. Run a campaign:
   ```bash
   scripts/bench-campaign.sh <name> \
     criterion:apply_layer_bucketed \
     probe:'--n 1000000 --threads 1,8,32' \
     perf-stat:'--n 1000000 --threads 32 --layers rotation_zz'
   ```
3. Open the HTML report rendered at the end (`benchmarks/results/<date>-<host>/<name>-report.html` —
   phase-breakdown bars, throughput and scaling charts, criterion table, bandwidth ceilings; self-contained, no
   network), then diff against the last snapshot:
   ```bash
   python3 scripts/criterion-report.py compare <old-snapshot>.json <new>.json
   ```
and attach the Δ-table (and a flamegraph, if something moved enough to be worth explaining) to a
`research/notes/` entry.

Snapshots live next to the campaign's `.txt` file, at
`benchmarks/results/<date>-<host>/<campaign-name>.json`. `criterion:` items write it via
`criterion-report.py snapshot --merge`, so several `criterion:` items in one campaign accumulate into the
same file instead of the later one overwriting the earlier. The `.txt` sibling carries a provenance header
(commit, rustc version, thread count, governor, CPU, load average at start/end) followed by one `=== item
===` section per item with that tool's combined stdout+stderr. `criterion-report.py compare` is a reporting
tool, not a gate — always exits 0, so treat REGRESSION/IMPROVED annotations (±5% default threshold) as
prompts to look, not pass/fail.

If the effect you're chasing is smaller than the noise floor below, a campaign-to-campaign comparison can't
resolve it — use the A/B protocol.

## Interleaved A/B protocol

This host's single-shot campaign noise is **~±5–8% at 1 thread and ~±10–26% at 8–32 threads**, and
untouched code moves that much between two otherwise-identical campaigns. A difference of two
independent campaign means cannot resolve the ~5–10% effects most engine changes actually produce.
What can: build *both* sides up front into separate binaries, alternate them adjacent in time for
several pairs, and compare *pairs* (run *i* of A against run *i* of B), instead of two
independently-noisy means.

```bash
scripts/ab-compare.sh g1d-borrow \
  --a HEAD~3 --b . \
  --probe '--n 1000000 --threads 1,32 --layers rotation_zz' \
  --pairs 5 --order abba
```

`--a`/`--b` take a git revision or `.` for the working tree (uncommitted changes included); each side builds
in its own tree (a detached worktree, or the working tree for `.`) so the two builds never evict each other
from a shared `target/`, and the working tree's `Cargo.lock` is seeded into each worktree so both sides
resolve identical dependencies. `--pairs` defaults to 3. `--order abba` alternates the within-pair order
instead of always running A first, so a monotone drift in machine state can't masquerade as a consistent
B-is-faster signal.

The run phase alternates the two binaries, appending each cell to a per-side `--json-out` sidecar, then
hands off to `ab-report.py`, which pairs runs by position within each `(layer, threads)` cell:

- **Acceptance rule: direction consistency, not a p-value.** With a handful of pairs there is nothing
  statistically meaningful to compute, and none is. Every pair in a cell must move the same way; the median Δ%
  is then the effect size to quote. Pairs disagreeing in sign are reported as "no consistent change" — not a
  small win, not a trend, not something to average over.
- `--field wall_ns` (default) drives the comparison; `--all-phases` also summarizes the coset-loop worker
  busy fields (`gather_ns`, `sort_ns`, `merge_ns`) per cell to explain *why* wall time moved — not as an
  effect size, since they sum over every Rayon worker and never sum to wall time.

`ab-report.py` is re-invocable on the archived sidecars at any time (stdlib only, no cargo, no benchmarks):

```bash
python3 scripts/ab-report.py <name>-a.probe.jsonl <name>-b.probe.jsonl \
    --all-phases --label-a "A=HEAD~3" --label-b "B=."
```

Both sides being the working tree, or both resolving to the same clean commit, is flagged with a warning —
that configuration measures this host's own noise floor, not a code change.

## Tool one-liners

- `scripts/bench-campaign.sh <name> <item>...` — runs a sequence of `criterion:<filter>`, `probe:<args>`,
  `perf-stat:<args>`, `scaling:<placement>`, `macro`, or `bandwidth` items, logged to
  `benchmarks/results/<date>-<host>/<name>.txt` (append-on-rerun, never overwrite). `--help` prints the full
  item menu.
- `scripts/host-topology.sh` — sourced, not executed, by `bench-campaign.sh` and `bandwidth.sh`. Single
  source of truth for host-specific placement: `PLACEMENT_PREFIX` (named placement → command prefix, keyed on
  `hostname -s`), `BANDWIDTH_RUNS` (the matrix `bandwidth.sh` measures), and `CEILING_MAP` (thread-count →
  `bandwidth.txt` section label, consumed by `perf-viz.py`). An unrecognized host falls back to an
  uncalibrated default (1 and `nproc` threads, no NUMA placement) plus a stderr warning; add a new `case` arm
  rather than editing an existing one.
- `scripts/ab-compare.sh <name> --a <rev|.> --b <rev|.> --probe '<args>' [options]` — the interleaved A/B
  harness; see the section above for the full flag list and an example.
- `python3 scripts/ab-report.py A.jsonl B.jsonl [--field wall_ns] [--all-phases]` — its paired-delta
  reporter, re-invocable on archived sidecars; see above.
- `scripts/profile.sh probe --n 1000000 --threads 32 --layers rotation_zz` — flamegraph the
  `phase_breakdown` probe.
- `scripts/profile.sh bench apply_layer_bucketed 10` — flamegraph a criterion bench group via
  `--profile-time <seconds>` (default 10s).
- `scripts/profile.sh bin ./target/release/some_bin --args` — flamegraph an arbitrary prebuilt binary, no
  build step.
- `scripts/perf-stat.sh --n 1000000 --threads 32 --layers rotation_zz` — hardware counters (cycles/string,
  IPC, LLC miss rate) plus a DRAM pass for one `(layer, thread count)` cell. `PROBE=/path/to/binary` skips the
  `cargo build` and measures a prebuilt binary instead — e.g. a binary `ab-compare.sh` archived.
- `scripts/bandwidth.sh` — the host's STREAM-style bandwidth ceiling across the placement matrix in
  `host-topology.sh`; run once per host or after hardware changes. First stdout line is the `# ceiling-map:`
  header `perf-viz.py` parses.
- `python3 scripts/criterion-report.py snapshot out.json --filter <substr> [--merge]` — snapshot
  `target/criterion/` to JSON. `--merge` loads an existing `out.json` and updates it with new entries instead
  of overwriting, so multiple `--filter`-scoped snapshots accumulate.
- `python3 scripts/criterion-report.py compare old.json new.json` — markdown Δ-table between two snapshots.
- `python3 scripts/fit_scaling.py [--group <name>|all] [--snapshot FILE...]` — Amdahl/USL fits over one or
  every `thread_scaling*` criterion group (default: every group). `--snapshot` reads one or more
  `criterion-report.py snapshot` files instead of `target/criterion` directly (later files win on collisions).
- `python3 scripts/perf-viz.py benchmarks/results/<date>-<host>/<campaign> [--compare OLD.json]` — render
  one campaign's data (`.txt`, `.json`, `-probe.json`, `-scaling-*.json`, plus the directory's
  `bandwidth.txt`) into a self-contained `<campaign>-report.html`; `--compare` adds Δ% columns to the
  criterion table. `bench-campaign.sh` runs this automatically at the end of every campaign.

## Machine contracts (what is logged where)

The tools communicate through file formats, not each other's code — a format change in one silently breaks a
consumer elsewhere unless the coupling is written down.

**(a) The probe's `--json-out` sidecar** (one JSON object per line) is the sole input to `perf-viz.py`'s
phase-breakdown section and, via `ab-compare.sh`'s per-side sidecars, to `ab-report.py`. Fields, from
`phase_breakdown.rs`'s `json_line` — verify against `PhaseStats` in
`crates/paulistrings/src/engine/stats.rs` before relying on any name: `layer`, `threads`, `n`, `reps`,
`qubits`, `seed`, `wall_ns`; the wall-clock phases `rebucket_ns`, `prepare_ns`, `rescale_ns`,
`span_plan_ns`, `permute_ns`, `coset_loop_ns`, `unpermute_ns`, `recount_ns`, `finalize_ns`; the worker
busy-time phases `swap_ns`, `size_ns`, `gather_ns`, `sort_ns`, `merge_ns`, `clear_ns`; and the counters
`layers`, `cosets`, `runs`, `rows_gathered`, `rows_sorted`, `rows_id`, `terms_in`, `terms_out`, `vmrss_kb`,
`vmhwm_kb`. `n` is the *steady-state* term count after an untimed warm-up call, not necessarily the
requested `--n` — see Phase timing below. `perf-viz.py` keeps only the *last* line per `(layer, threads)`
key, so an appended re-run overwrites the earlier one in the rendered report.

**(b) The probe's stdout `cell` line.** Every cell prints exactly one `cell layer=<name> threads=<n> n=<n>
layers=<n> wall_ms=<f>` line, in every `--format`. `perf-stat.sh`'s awk greps this literal shape (`n=` and
`layers=` field prefixes) to compute cycles/string — a coupling documented here and nowhere else in the
code. Change the line's fields or order, fix `perf-stat.sh`'s awk in the same change.

**(c) Criterion snapshot JSON.** `criterion-report.py snapshot` and `bench-campaign.sh`'s
`criterion:`/`scaling:` items write `{full_id: {median_ns, mean_ns, stddev_ns, throughput_elems,
melem_per_s}}`, consumed by `compare`, `fit_scaling.py --snapshot`, and `perf-viz.py`'s criterion and
scaling sections. Thread-scaling groups additionally rely on a naming contract: a `BenchmarkId` of
`<group>/<threads>` where `<group>` starts with `thread_scaling` and `<threads>` is a bare integer —
`fit_scaling.py` and `perf-viz.py` both split on the last `/` and parse the tail as an int; anything else is
silently skipped.

**(d) `bandwidth.txt`**, written by `bandwidth.sh` and read by `perf-viz.py`'s bandwidth section and
roofline model: an optional first line `# ceiling-map: <key>=<label>;...` (keys are thread counts or
`default`; format is pinned to `perf-viz.py`'s parser), then one or more `=== <section label> ===` headers
each followed by `kernel=<name> threads=<n> mib=<n> reps=<n> best_gbps=<f> avg_gbps=<f>` lines from
`crates/membench`. Without a ceiling-map header, `perf-viz.py` falls back to a hard-coded ccqlin038-shaped
thread-count → section table.

**(e) The campaign results directory** (`benchmarks/results/<date>-<host>/`). Per campaign name:
`<name>.txt`, `<name>.json`, `<name>-probe.json` (the probe's JSONL sidecar — note the `.json` extension
despite being JSON-*Lines*), one `<name>-scaling-<placement>.json` per `scaling:` placement, and
`<name>-report.html`. `bandwidth.txt` and `flamegraph-<name>-<shortcommit>[-dirty].html` (+ `.meta.txt`)
live directly in the dated directory, shared across every campaign there. A/B runs add `<name>-ab.log`,
`<name>-{a,b}.probe.jsonl` (`.jsonl`, distinct from the campaign sidecar's `.json`), and the archived
binaries `<name>-{a,b}-<sha|worktree>[-dirty]`. Results directories may still hold hand-rolled A/B
logs/binaries predating `ab-compare.sh` (not following this naming) — safe to delete.

## Phase timing

The `phase-timing` Cargo feature (`crates/paulistrings/src/engine/stats.rs`) gates a per-phase counter
breakdown of the propagation engine. It is **measurement-only and never in the default feature set** — the
default build carries no timing code and no stats fields at all; the same bitwise-identity tests (the
fingerprint net, thread-count/bucket-count/seed determinism tests) run *with the feature enabled* in CI as
the acceptance test that instrumentation doesn't perturb output, since the timers only read the clock and
add to plain integers.

Counters are read via `LayerScratch::take_stats()` after driving layers through `propagate_with_scratch`.
Two clock domains are deliberately mixed in one `PhaseStats`:

- **Wall-clock phases** (`rebucket_ns` … `finalize_ns`) — measured once per layer on the calling thread;
  they sum to approximately the layer's wall time.
- **Worker busy-time phases** (`swap_ns` … `clear_ns`) — summed across every coset task on every Rayon
  worker. Under a `t`-thread pool they sum to `coset_loop_ns × t × efficiency`, not to `coset_loop_ns` itself.
  `Σbusy / (coset_loop_ns × t)` is the coset loop's parallel efficiency — the gap between the two domains
  is the load-balance signal.

The probe (`cargo run --release --features phase-timing --example phase_breakdown`) runs each `(layer,
threads)` cell twice inside a dedicated Rayon pool: an untimed warm-up call, then the timed call whose input
is the warm-up's output. For the single-generator layers this drives the input to its closed fixed point
first, so the timed call measures steady-state cost rather than first-layer growth (see Machine contracts
(a) for what that does to the reported `n`); `trotter` additionally self-caps its input at `TROTTER_MAX_N`
regardless of `--n` (64 distinct generators under no truncation grow combinatorially rather than closing —
see that constant's doc comment). The probe also prints its own timer-overhead estimate
(`PhaseStats::timer_reads() × stats::TIMER_READ_OVERHEAD_NS`) next to the breakdown, so you can see when the
measurement pollutes itself (tiny cosets, many runs inflate the read count).

## Flamegraphs

`scripts/profile.sh` has three modes: `probe` (builds and profiles `phase_breakdown` under the `profiling`
Cargo profile), `bench` (profiles a criterion bench group via `--profile-time`, built with `cargo bench
--no-run` and located under `target/release/deps/`), and `bin` (profiles an arbitrary prebuilt binary, no
build step).

The `[profile.profiling]` build (`Cargo.toml`) inherits `release` (same `lto = "fat"`, same
`codegen-units = 1`, so it profiles what ships) and adds `debug = "line-tables-only"` plus
`strip = "none"` so `perf`/`addr2line` can expand LTO-inlined frames. `PROFILE_MODE` picks the
stack-walking method: `dwarf` (default, `perf record --call-graph dwarf,16384`, works against the
normal profiling build) or `fp` (frame pointers, which forces
`RUSTFLAGS="-Cforce-frame-pointers=yes"` and a full rebuild for `probe`/`bench` since codegen differs
from cached artifacts, and for `bin` only changes the `perf record` flag — you are responsible for
having built that binary with frame pointers yourself).

Output: `benchmarks/results/<date>-<host>/flamegraph-<name>-<shortcommit>[-dirty].html` plus a sidecar
`.meta.txt` (host, date, full commit, rustc version, mode, frequency, exact command line).

Caveats:
- Criterion's `--profile-time` loops the routine without statistical sampling — ignore `criterion::` and
  setup frames in the graph.
- Rayon idle spinning shows up as `crossbeam_epoch::*` frames; don't mistake it for real work.
- `perf.data` with DWARF call graphs is GB-scale; `profile.sh` records into a scratch temp dir cleaned up
  automatically after conversion.
- `#[inline(never)]` is attribution-of-last-resort for pinning a frame down in a flamegraph — never add it
  just to make quoted numbers look cleaner, and never quote a number that depended on it.

## Counters & bandwidth

`scripts/perf-stat.sh` runs two passes over the `phase_breakdown` probe:

- **Pass A** (per-process): `cycles`, `instructions`, `LLC-loads`, `LLC-load-misses`, `branches`,
  `branch-misses`. Derives IPC, LLC miss rate, and cycles/input-string. Cycles/string is frequency-robust
  (matters on a powersave-governed host) but whole-process — it includes input generation and warm-up, so
  with more than one cell in a run it's a blend (the script warns), converging from above as `--n`/`--reps`
  grow.
- **Pass B** (system-wide uncore IMC): `uncore_imc/cas_count_read/` and `uncore_imc/cas_count_write/`,
  `--per-socket`, unavoidably `-a` (whole box) on a shared host. An idle baseline (`IDLE_SECS`, default 3s) is
  measured immediately before and subtracted as a rate — still approximate under nontrivial load average.
  Units come pre-scaled to MiB by `perf` itself; don't multiply by 64 again.

## Roofline model

Measured bandwidth ceilings come from `scripts/bandwidth.sh` (membench, `crates/membench/src/main.rs`):
STREAM-convention nominal bytes (copy = 16 B/elem, triad = 24 B/elem), plain (write-allocating) stores,
read-for-ownership deliberately **uncorrected** — this matches the propagation engine's own store pattern,
so the nominal figure is the right ceiling to compare phases against (cross-check with the uncore pass if
you want true CAS traffic).

Bytes-moved model per layer, at `W = 2` (48 B/term = 32 B key + 16 B coeff), over `n` input rows and `n_out`
gathered rows:

- **gather** ≈ read `n × 48` + write `n_out × 48`
- **sort** ≈ a few read+write passes over `n_out × 48`
- **merge** ≈ read `n_out × 48` + `n_existing × 48`, write `n_merged × 48`

Serial phases (rebucket, recount) compare against the single-core ceiling; the coset loop against the
all-core ceiling for whatever placement was used to measure it.

`perf-viz.py` computes this automatically per probe cell ("DRAM: X GB/s = Y% of ceiling") from
`rows_gathered`, `rows_sorted`, and `rows_id`: the identity-delta stream skips the sort and carries no tag
byte, and a *dense* identity row (rotations, general unitaries) materializes only its 16-byte coefficient —
keys borrowed in place from the source bucket, modeled as coset-cache-resident:

`bytes/layer = terms_in×T + 2×(rows_gathered−rows_id)×T + 2×rows_id×16 + 2×rows_sorted×T + terms_out×T`

with `T = 16·W + 16` on the coset path, or `2×terms_in×T` on the rescale path. A probe line carrying
`rows_gathered`/`rows_sorted` but no `rows_id` field is priced with `rows_id` treated as 0 (every gathered
row at full `T`, no dense-identity discount); a line carrying neither `rows_id` nor `rows_sorted` falls back
to the older `4×rows_gathered×(T+1)` model (every row priced as if tagged and passed through a sort read and
write). This is divided by wall time and by the membench triad ceiling at a comparable core count (per
the host's `# ceiling-map:` header — Machine contracts (d)). Read the result as a classification, not a
gauge: **over 100% means the modeled traffic is mostly served from cache** (small per-coset working sets,
not DRAM-bound); near 100% is genuinely at the wall; far below 100% with high wall time points at latency,
serial phases, or imbalance instead.

Rule of thumb: at or above ~70% of the measured ceiling, the phase is bandwidth-bound — stop optimizing
arithmetic. Far below the ceiling with a high LLC miss rate points at a latency/working-set problem instead.

## Threading

`scripts/host-topology.sh` is the single place CPU/NUMA placements are defined. `bench-campaign.sh`'s
`scaling:<placement>` items and `bandwidth.sh` both source it, keyed on `hostname -s`. The reference host
(ccqlin038: 2 sockets, node0 physical CPUs 0-7 / HT 16-23, node1 physical CPUs 8-15 / HT 24-31) is
calibrated as:

| placement | prefix | isolates |
|---|---|---|
| `default` | (none) | whatever the ambient scheduler does |
| `node0` | `numactl --cpunodebind=0 --membind=0` | NUMA cost (vs `default`/spread) |
| `phys16` | `taskset -c 0-15` | both sockets, physical cores only |
| `smt16` | `taskset -c 0-7,16-23 numactl --membind=0` | HT yield vs cross-socket (compare to `phys16` / node0 8-core) |
| `phys8` | `taskset -c 0-7 numactl --membind=0` | pure single-node core scaling |

An unrecognized host gets only the no-op `default` placement plus an uncalibrated
`BANDWIDTH_RUNS`/`CEILING_MAP` and a stderr warning; add a case arm in `host-topology.sh` to calibrate a new
host rather than editing an existing one.

`node0` vs a spread/default placement isolates NUMA cost; `phys16` (16 physical, both sockets) vs `smt16` (8
physical + 8 HT, one socket) isolates hyperthread yield against cross-socket cost; `node0` scaled 1→8
threads is pure core scaling, useful for Amdahl/USL fits since it has no NUMA or HT crossover to confound
the fit.

`scripts/fit_scaling.py --group <name>` (or `--group all`, the default) fits Amdahl's law and the Universal
Scalability Law to a `thread_scaling*` criterion group and reports the serial fraction / σ,κ with R².
Cross-check the fitted serial fraction against the probe's own measured serial share (wall-clock phases
minus the coset loop, over total wall time) — the two should roughly agree; if not, something outside the
modeled coset loop is eating parallelism.

## Host caveats

- This host's single-shot campaign noise floor is **~±5–8% at 1 thread and ~±10–26% at 8–32 threads**,
  and untouched code moves that much between otherwise-identical campaigns. Trust an absolute
  campaign-to-campaign comparison only above that floor; below it, use the interleaved A/B protocol.
  `bench-campaign.sh` logs load average at the start and end of every campaign — check it before
  trusting a close call.
- The CPU governor is `powersave` and cannot be pinned to `performance` (no root on the reference
  host). Prefer frequency-robust ratios (IPC, percent of the measured bandwidth ceiling, speedup
  ratios, cycles/string) over absolute millisecond figures across days.
- Run every campaign (and every A/B run) with `RUST_LOG` **unset** (and no logger installed): with no logger
  the engine's per-layer progress logging is a single static level check and allocates nothing, whereas an
  enabled `debug` filter adds a formatted line and a clock read per layer.
