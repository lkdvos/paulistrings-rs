# Performance

What the engine's speed rests on, and where its measured limits are. Layout and dispatch decisions
come first; the second half is a roofline analysis of the current engine against the reference
host's measured memory bandwidth. Every number is copied from a committed source, named inline and
in the footer.

## Layout: structure-of-arrays, monomorphized width

Terms live in per-bucket parallel columns: `Vec<[u64; W]>` for the `x` and `z` key words,
`Vec<Complex64>` for coefficients — 48 B per term at `W = 2` (32 B key + 16 B coefficient).
Coefficient-only scans (truncation, expectation values) and key-only scans (weight, commutation)
each stream only the bytes they use, and each column maps directly to a GPU device buffer. Buckets
retain their capacity across layers, so a steady-state propagation loop allocates nothing.

The width `W` is a const generic: monomorphization unrolls all bit operations and keeps
`PauliString` a `Copy` value type. The Python bindings instantiate `W ∈ {1, 2, 4, 8, 16}` (64–1024
qubits) and dispatch once, outside any hot loop.

## Where a layer's time goes

Measured on the reference host (2× Xeon Gold 6244, 16c/32t) with the repo's phase-timing probe,
per layer class, as a share of summed worker busy time (gather / sort / merge):

| layer class | 1 thread | 16 threads | 32 threads |
|---|---|---|---|
| ZZ rotation (m = 4.50e6) | 54 / 9 / 37 | 53 / 10 / 36 | 68 / 9 / 22 |
| general 2q unitary, sparse PTM (m = 3.0e6) | 38 / 33 / 29 | 35 / 31 / 33 | 39 / 29 / 31 |
| dense 2q PTM, `su4` (m = 4.24e7) | 41 / 51 / 8 | 35 / 55 / 10 | 26 / 65 / 9 |

Gather + merge dominate the sparse classes; the sort dominates only dense two-qubit PTMs. Parallel
efficiency (busy time over coset-loop wall × threads) is 0.99 for `su4` at 16 threads: load balance
is a solved problem, and the scaling limits below are memory-system effects, not imbalance.

Per-term cost is flat in the sum size over the measured range: 30.5–30.6 ns/term for the ZZ
rotation across m = 1.50e6 → 4.50e6, 138.5–141.3 for the sparse 2q unitary, 322–327 for `su4`
(single thread).

## The memory wall, measured

The ceiling comes from a STREAM-style probe (`crates/membench` via `scripts/bandwidth.sh`),
nominal bytes with plain write-allocating stores — the same store pattern the engine uses. Best
GB/s on the reference host:

| placement | read | write | copy | triad |
|---|---:|---:|---:|---:|
| 1 core, node-local | 11.3 | 10.1 | 9.5 | 11.3 |
| one socket, 8 physical | 39.0 | 18.6 | 35.6 | 28.1 |
| both sockets, 16 physical | 45.0 | 25.3 | 40.0 | 38.1 |
| both sockets, 32 threads | 48.8 | 23.1 | 33.8 | 36.5 |

Two structural facts from the ceiling measurement: hyperthreads add no bandwidth (39.0 → 39.2
GB/s), and the second socket adds only 15–25% under first-touch page placement with work-stealing,
not 2×.

The modeled traffic side prices each layer from its row counts at `T = 48` B/term
(`benchmarks/PROFILING.md` §Roofline model):

```text
bytes/layer = terms_in×T + 2×(rows_gathered−rows_id)×T + 2×rows_id×16
              + 2×rows_sorted×T + terms_out×T
```

Reading the ratio of measured DRAM traffic to ceiling as a classification: at or above ~70% of the
measured ceiling a phase is bandwidth-bound; modeled traffic far above measured means the working
set is cache-served; far below ceiling with a high LLC miss rate points at latency, not bandwidth.

### Single thread: nothing is bandwidth-bound

Measured 2026-09-01 on the current engine (fact sheet:
`research/notes/2026-09-01-roofline-ccqlin038.md`; ceilings per the 1-core row above):

| cell | ns/term | model GB/s | measured GB/s | % of copy ceiling | IPC | LLC load-miss | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| ZZ rotation, m = 4.50e6 | 30.5 | 8.3 | 3.29 | 35% | 2.24 | 38.3% | latency-bound |
| sparse 2q, m = 3.0e6 | 141.3 | 9.0 | 2.08 | 22% | 2.62 | 34.9% | latency-bound |
| `su4`, m = 1.41e7 | 327.3 | 8.6 | 0.67 | 7% | 2.98 | 2.1% | compute-bound |

The byte model over-counts DRAM traffic by 2.5–12.8× at one thread because most modeled traffic is
served from cache. The sparse classes are limited by load latency (IPC ≈ 2.2–2.6 with 34–41% LLC
load-miss); `su4` is compute-bound in its sort (IPC 2.98, 2.1% LLC load-miss).

### Threads: the dense-PTM class hits the write ceiling

`su4` at m = 1.41e7 (both-socket ceilings: read 45.0 / write 25.3 at 8–16t, 48.8 / 23.1 at 32t):

| threads | ns/term | speedup | read GB/s | write GB/s | % read ceil | % write ceil | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 327.3 | — | 0.61 | 0.27 | 5% | 1% | compute-bound |
| 8 | 56.2 | 5.8× | 25.9 | 18.2 | 58% | 72% | approaching write ceiling |
| 16 | **49.3** | **6.6×** | 39.3 | **28.1** | 87% | **111%** | write-bandwidth-bound |
| 32 | 55.3 | 5.9× | 39.5 | 27.6 | 81% | 120% | write-bandwidth-bound |

By 16 threads the dense-PTM class sits at the machine's write ceiling — the figures above 100% are
against the STREAM nominal-store ceiling, which the engine's write-allocating stores share, so "at
the ceiling" is the correct reading. 32 threads buy zero additional bandwidth (67.0 vs 67.0 GB/s
attributable) and cost 12% of wall time.

The sparse classes at 32 threads stay latency-bound and keep scaling:

| cell | ns/term | speedup vs 1t | read GB/s | write GB/s | % write ceil | verdict |
|---|---:|---:|---:|---:|---:|---|
| ZZ rotation, m = 4.50e6 | 2.7 | 11.3× | 8.9 | 11.5 | 50% | latency-bound |
| sparse 2q, m = 3.0e6 | 10.8 | 13.1× | 11.7 | 11.3 | 49% | latency-bound |

They stop at half the write ceiling with 70–90% of their modeled traffic cache-served: bandwidth is
not what limits them, and they take the full thread count profitably.

## Thread-count guidance

On a comparable two-socket host: run dense-PTM-heavy circuits (general two-qubit unitaries with
dense transfer matrices) at ~16 threads — beyond that the write ceiling is already saturated and
extra threads only add contention. Run sparse-rotation and Clifford circuits at full thread count
(11–13× at 32 threads here). Hyperthreads add no bandwidth and help only latency-bound phases.

## Noise floor

Single-shot campaign noise on the reference host is ±5–8% single-threaded and ±10–26% at 8–32
threads; untouched code moves that much between campaigns. Effects below that need the interleaved
A/B protocol (`scripts/ab-compare.sh`), whose acceptance criterion is direction consistency across
every pair.

Sources:
[`research/notes/2026-09-01-roofline-ccqlin038.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/notes/2026-09-01-roofline-ccqlin038.md)
(all achieved numbers, phase shares, verdicts);
[`research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/notes/2026-08-30-bandwidth-ceiling-ccqlin038.md)
(ceilings);
[`benchmarks/PROFILING.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/PROFILING.md)
(byte model, interpretation rules, noise floor);
[`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md)
§Data-Model, §Width, §Performance-Model (layout and dispatch).
