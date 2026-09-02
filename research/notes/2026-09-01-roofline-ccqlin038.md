# Roofline fact sheet — ccqlin038, 2026-09-01

Standalone measurement of the propagation engine's achieved memory traffic against the host's
measured bandwidth ceilings. This sheet is the committed source for the numbers on the docs site's
Design → Performance page.

## Provenance

- Host: ccqlin038.flatironinstitute.org — 2× Xeon Gold 6244 @ 3.60 GHz (16c/32t, 2 NUMA nodes),
  governor `powersave`. Ceiling denominators: `research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`
  (unchanged hardware; not re-measured).
- Engine: commit `94b3364`, rustc 1.94.0, `--release` + `--features phase-timing`; `RUST_LOG` unset;
  working tree carried documentation-only edits (`crates/` clean).
- Instrument: `target/release/examples/phase_breakdown` at `--qubits 128` (`W = 2`), truncation
  `keep`, default seed; counters via `scripts/perf-stat.sh` (one cell per invocation, `--reps 40`,
  idle DRAM baseline subtracted; baselines measured 0.4–0.5 GB/s).
- Byte model: `benchmarks/PROFILING.md` §Roofline model, `T = 48` B/term at `W = 2`, evaluated from
  the probe's row counts (`terms_in`, `rows_gathered`, `rows_id`, `rows_sorted`, `terms_out`).
- Raw output: `benchmarks/results/2026-09-01-ccqlin038-roofline/` (gitignored; every number quoted
  here is in it). Phase-share grid at default `--reps 8` (all cells are large-`m`, where the default
  is converged); counter cells at `--reps 40`. Start load 1.19.
- `m` below is the probe's steady-state term count for the cell (the probe's `--n` is the build
  target before closure: `--n 1e6` closes at `m = 1.50e6` for `rotation_zz`, `1.0e6` for `gu2q`,
  `1.41e7` for `su4`; `--n 3e6` at `4.50e6` / `3.0e6` / `4.24e7`).

## Table A — single thread (ceilings, 1 core node-local: read 11.3 / copy 9.5 / triad 11.3 GB/s)

| cell | m | ns/term | model GB/s | measured GB/s | % of copy | model / measured | IPC | LLC load-miss | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `rotation_zz` 1.50e6 | 1.50e6 | 30.6 | 8.5 | 2.53 | 27% | 3.3× | 2.26 | 41.1% | latency-bound |
| `rotation_zz` 4.50e6 | 4.50e6 | 30.5 | 8.3 | 3.29 | 35% | 2.5× | 2.24 | 38.3% | latency-bound |
| `gu2q` 3.0e6 | 3.0e6 | 141.3 | 9.0 | 2.08 | 22% | 4.3× | 2.62 | 34.9% | latency-bound |
| `gu2q` 1.0e6 | 1.0e6 | 138.5 | 9.8 | 2.07 | 22% | 4.7× | 2.68 | 33.6% | latency-bound |
| `su4` 1.41e7 | 1.41e7 | 327.3 | 8.6 | 0.67 | 7% | 12.8× | 2.98 | 2.1% | compute-bound |

Single-thread verdict: no layer class is bandwidth-bound at one thread. The byte model over-counts
DRAM traffic by 2.5–12.8× because most modeled traffic is served from cache; what limits the sparse
layers is load latency (IPC ≈ 2.2–2.7 with 34–41% LLC load-miss), and `su4` is compute-bound
(IPC 2.98, 2.1% LLC load-miss — its sort is 51–52% of busy time).

Per-term cost is flat in `m`: `rotation_zz` 30.5–30.6 ns/term across 1.50e6→4.50e6,
`gu2q` 138.5–141.3 across 1.0e6→3.0e6, `su4` 322–327 (grid, 1.41e7→4.24e7).

## Table B — thread scaling (ceilings, both sockets: 8/16t read 45.0 / write 25.3; 32t read 48.8 / write 23.1)

`su4` at m = 1.41e7 (dense two-qubit PTM — the bandwidth-heavy class):

| threads | ns/term | speedup | IPC | LLC load-miss | read GB/s | write GB/s | % read ceil | % write ceil | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 327.3 | — | 2.98 | 2.1% | 0.61 | 0.27 | 5% | 1% | compute-bound |
| 8 | 56.2 | 5.8× | 2.21 | 75.4% | 25.9 | 18.2 | 58% | 72% | approaching write ceiling |
| 16 | **49.3** | **6.6×** | 1.26 | 79.5% | 39.3 | **28.1** | 87% | **111%** | write-bandwidth-bound |
| 32 | 55.3 | 5.9× | 0.58 | 71.0% | 39.5 | 27.6 | 81% | 120% | write-bandwidth-bound |

At 16 threads `su4` sits at the machine's write ceiling (the >100% figures are against the STREAM
plain-store nominal ceiling, which the engine's write-allocating stores share, so "at the ceiling"
is the correct reading). 32 threads buy zero additional bandwidth (67.0 vs 67.0 GB/s attributable)
and cost 12% of wall time; 16 threads is the optimum for this class on this host.

Sparse layers at 32 threads, m = 4.50e6 (`rotation_zz`) / 3.0e6 (`gu2q`):

| cell | ns/term | speedup vs 1t | IPC | LLC load-miss | read GB/s | write GB/s | % read ceil | % write ceil | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `rotation_zz` 32t | 2.7 | 11.3× | 0.97 | 34.4% | 8.9 | 11.5 | 18% | 50% | latency-bound |
| `gu2q` 32t | 10.8 | 13.1× | 1.29 | 29.4% | 11.7 | 11.3 | 24% | 49% | latency-bound |

The sparse classes scale to 11.3–13.1× at 32 threads and stop at half the write ceiling: they stay
latency-bound, not bandwidth-bound. Their measured traffic at 32t is 20–23 GB/s against modeled
70–113 GB/s, i.e. ~70–90% of modeled traffic is cache-served even at full thread count.

## Phase shares (grid, share of summed worker busy time, gather/sort/merge)

| layer | 1t | 16t | 32t |
|---|---|---|---|
| `rotation_zz` (m=4.50e6) | 54/9/37 | 53/10/36 | 68/9/22 |
| `gu2q` (m=3.0e6) | 38/33/29 | 35/31/33 | 39/29/31 |
| `su4` (m=4.24e7) | 41/51/8 | 35/55/10 | 26/65/9 |

Gather + merge dominate the sparse classes (85–91% of busy); the sort dominates only the dense PTM
class. Parallel efficiency (busy / (coset-loop wall × threads)) is 0.99 for `su4` at 16t; the
scaling limits above are memory-system effects, not load imbalance.

## Verdict summary

One phase regime is bandwidth-bound: dense two-qubit PTMs (`su4`) from ~8 threads up, pinned to the
write ceiling by 16 threads. Everything else — every class at one thread, and the sparse classes at
any thread count — is latency- or compute-bound, with the byte model over-counting DRAM traffic
2.5–12.8× at one thread because the working set is largely cache-served. Thread guidance on this
host: 16 threads for dense-PTM-heavy circuits, 32 threads for sparse-rotation circuits (11–13×).
