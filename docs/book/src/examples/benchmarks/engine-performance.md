# Engine performance

Where the engine's measured limits are: a roofline analysis of the current engine against the reference host's measured memory bandwidth, then the partitioned and distributed measurements.
Every number is copied from a committed source, named inline and in the footer.

*Manual: [Stats, memory and logging](../../manual/propagation/settings.md), [NUMA partitions](../../manual/propagation/partitions.md), [MPI ranks](../../manual/propagation/mpi.md)*

The design these numbers measure is [Inside the engine](../../manual/propagation/engine.md); the thread-count advice they support is in [Stats, memory and logging](../../manual/propagation/settings.md#threads).

## Layer time breakdown

Measured on the reference host (2× Xeon Gold 6244, 16c/32t) with the repo's phase-timing probe, per layer class, as a share of summed worker busy time (gather / sort / merge):

| layer class | 1 thread | 16 threads | 32 threads |
|---|---|---|---|
| ZZ rotation (m = 4.50e6) | 54 / 9 / 37 | 53 / 10 / 36 | 68 / 9 / 22 |
| general 2q unitary, sparse PTM (m = 3.0e6) | 38 / 33 / 29 | 35 / 31 / 33 | 39 / 29 / 31 |
| dense 2q PTM, `su4` (m = 4.24e7) | 41 / 51 / 8 | 35 / 55 / 10 | 26 / 65 / 9 |

Gather + merge dominate the sparse classes; the sort dominates only dense two-qubit PTMs.
Parallel efficiency (busy time over coset-loop wall × threads) is 0.99 for `su4` at 16 threads: load balance is a solved problem, and the scaling limits below are memory-system effects, not imbalance.

![Share of worker busy time in gather, sort and merge per layer class, one thread](../../assets/design/phase-shares.svg)

Per-term cost is flat in the sum size over the measured range: 30.5–30.6 ns/term for the ZZ rotation across m = 1.50e6 → 4.50e6, 138.5–141.3 for the sparse 2q unitary, 322–327 for `su4` (single thread).

## Memory wall

The ceiling comes from a STREAM-style probe (`crates/membench` via `scripts/bandwidth.sh`), nominal bytes with plain write-allocating stores — the same store pattern the engine uses.
Best GB/s on the reference host:

| placement | read | write | copy | triad |
|---|---:|---:|---:|---:|
| 1 core, node-local | 11.3 | 10.1 | 9.5 | 11.3 |
| one socket, 8 physical | 39.0 | 18.6 | 35.6 | 28.1 |
| both sockets, 16 physical | 45.0 | 25.3 | 40.0 | 38.1 |
| both sockets, 32 threads | 48.8 | 23.1 | 33.8 | 36.5 |

Two structural facts from the ceiling measurement: hyperthreads add no bandwidth (39.0 → 39.2 GB/s), and the second socket adds only 15–25% under first-touch page placement with work-stealing, not 2×.

The modeled traffic side prices each layer from its row counts at `T = 48` B/term (`benchmarks/PROFILING.md` §Roofline model):

```text
bytes/layer = terms_in×T + 2×(rows_gathered−rows_id)×T + 2×rows_id×16
              + 2×rows_sorted×T + terms_out×T
```

Reading the ratio of measured DRAM traffic to ceiling as a classification: at or above ~70% of the measured ceiling a phase is bandwidth-bound; modeled traffic far above measured means the working set is cache-served; far below ceiling with a high LLC miss rate points at latency, not bandwidth.

## Single-thread roofline

Measured on the current engine (fact sheet: `research/HARDWARE.md` §Engine roofline, single thread; ceilings per the 1-core row above):

| cell | ns/term | model GB/s | measured GB/s | % of copy ceiling | IPC | LLC load-miss | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| ZZ rotation, m = 4.50e6 | 30.5 | 8.3 | 3.29 | 35% | 2.24 | 38.3% | latency-bound |
| sparse 2q, m = 3.0e6 | 141.3 | 9.0 | 2.08 | 22% | 2.62 | 34.9% | latency-bound |
| `su4`, m = 1.41e7 | 327.3 | 8.6 | 0.67 | 7% | 2.98 | 2.1% | compute-bound |

The byte model over-counts DRAM traffic by 2.5–12.8× at one thread because most modeled traffic is served from cache.
The sparse classes are limited by load latency (IPC ≈ 2.2–2.6 with 34–41% LLC load-miss); `su4` is compute-bound in its sort (IPC 2.98, 2.1% LLC load-miss).

## Multi-thread roofline {#multi-thread-roofline}

`su4` at m = 1.41e7 (both-socket ceilings: read 45.0 / write 25.3 at 8–16t, 48.8 / 23.1 at 32t):

| threads | ns/term | speedup | read GB/s | write GB/s | % read ceil | % write ceil | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 327.3 | — | 0.61 | 0.27 | 5% | 1% | compute-bound |
| 8 | 56.2 | 5.8× | 25.9 | 18.2 | 58% | 72% | approaching write ceiling |
| 16 | **49.3** | **6.6×** | 39.3 | **28.1** | 87% | **111%** | write-bandwidth-bound |
| 32 | 55.3 | 5.9× | 39.5 | 27.6 | 81% | 120% | write-bandwidth-bound |

By 16 threads the dense-PTM class sits at the machine's write ceiling — the figures above 100% are against the STREAM nominal-store ceiling, which the engine's write-allocating stores share, so "at the ceiling" is the correct reading.
32 threads buy zero additional bandwidth (67.0 vs 67.0 GB/s attributable) and cost 12% of wall time.

![su4 attributable DRAM traffic against thread count, with the measured read and write ceilings](../../assets/design/roofline-threads.svg)

The sparse classes at 32 threads stay latency-bound and keep scaling:

| cell | ns/term | speedup vs 1t | read GB/s | write GB/s | % write ceil | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| ZZ rotation, m = 4.50e6 | 2.7 | 11.3× | 8.9 | 11.5 | 50% | latency-bound |
| sparse 2q, m = 3.0e6 | 10.8 | 13.1× | 11.7 | 11.3 | 49% | latency-bound |

They stop at half the write ceiling with 70–90% of their modeled traffic cache-served: bandwidth is not what limits them, and they take the full thread count profitably.

## Partitioned: P = 2 vs P = 1 {#partitioned}

Two facts above — the second socket adding 15–25% rather than 2×, and the dense-PTM class pinned to the write ceiling — are the same effect: pages are placed by first touch and the thread pool then steals work across sockets, so roughly half of the second socket's reads are remote.
The engine can instead be run partitioned, one pinned pool and one share of the sum per NUMA domain, with only the rows a layer moves across a domain boundary exchanged ([NUMA partitions](../../manual/propagation/partitions.md)).

**Locality decides everything.**
A layer whose gates leave the partition rows fixed runs entirely inside its own domain and gains; a layer that moves any row across a boundary pays for the export pass, the copy and the receiver's larger merge stream.
Measured at `P = 2` against `P = 1` on the same binary, on two-socket hosts from 8 to 48 cores per socket:

| layer class | `P = 2` vs `P = 1` |
|---|---|
| dense two-qubit unitaries, no row crossing | **4–18% faster**, direction-consistent on every host |
| Pauli rotations, no row crossing | coset loop 12–22% faster on two of three hosts |
| any layer that exports rows (random rows) | **2–8.4× slower**, and worse at higher core counts |

The gain grows with cores per socket, as a memory-system effect should: −4% on an 8-core-per-socket workstation, −18% on a 48-core Genoa at 96 threads.
Load imbalance across partitions stays within 1.000–1.004 with random rows.
Random partition rows put roughly half of a dense two-qubit gate's deltas across a boundary, so the default draw lands a mixed circuit in the bottom row of that table.

The per-host cells behind the summary, copied from [`research/HARDWARE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/HARDWARE.md) §Partitioned engine, P=1 vs P=2 in-process: `scripts/ab-compare.sh` runtime-knob A/B, 5 pairs `abba`, `--n 1000000 --qubits 128` (`W = 2`), random partition rows, each partition pinned to its socket with `MPOL_BIND`; steady-state `m` ≈ 1.5e6 for rotations, 1.0e6 for `gu2q`, 1.4e7 for `su4`/`su4_local`.
Wall time per layer, P=2 vs P=1 (median Δ%, pairs agreeing):

| cell | ccqlin038 16t | ccqlin038 32t | Icelake 32t | Icelake 64t | Genoa 48t | Genoa 96t |
|---|---|---|---|---|---|---|
| `su4` (random rows) | +128% 6/6 | +118% 5/5 | +241% 5/5 | +274% 5/5 | +403% 5/5 | +373% 5/5 |
| `gu2q` | — | +282% 5/5 | — | — | +597% 5/5 | +574% 5/5 |
| `rotation_remote` | +204% 5/5 | +217% 5/5 | +310% 5/5 | +461% 5/5 | +730% 5/5 | +737% 5/5 |
| `rotation_local` (no exchange) | −13.6% 4/5 | −4.9% 4/5 | +16% 4/5 | +21% 5/5 | −12.0% 5/5 | −29.9% 4/5 |
| `su4_local` (no exchange) | −4.1% 5/5 | −3.5% 5/5 | −6.6% 5/5 | −7.3% 5/5 | −8.7% 5/5 | −18.2% 5/5 |

Exchange-free rotation layer, per layer (from the probe sidecars):

| host, threads | P=1 wall | P=2 wall | P=2 collective (`barrier_ns`) | coset loop P=1 → P=2 |
|---|---|---|---|---|
| ccqlin038, 16 | 4.24 ms | 3.33 ms | 0.11 ms | 4.15 → 3.25 ms (−22%) |
| Genoa, 48 | 1.10 ms | 0.97 ms | 0.04 ms | 1.02 → 0.88 ms (−14%) |
| Icelake, 32 | 1.62 ms | 2.14 ms | 0.30 ms | 1.54 → 1.77 ms (+15%) |

Counters, ccqlin038, dense layers at 16 threads (8 per socket at P=2); `scripts/perf-stat.sh`, shared box:

| cell | IPC | LLC load-miss | cycles/string | S0 read / write GB/s (% of 39.0 / 18.6) | S1 read / write | total GB/s |
|---|---|---|---|---|---|---|
| `su4` P=1 | 1.13 | 76% | 7178 | 15.8 / 14.3 (40% / 77%) | 15.8 / 12.8 (41% / 69%) | 60.2 |
| `su4` P=2 | 1.03 | 89% | 8229 | 12.4 / 9.7 (32% / 52%) | 11.8 / 9.5 (30% / 51%) | 44.0 |
| `su4_local` P=1 | 1.09 | 75% | — | 16.4 / 13.8 (42% / 74%) | 16.4 / 13.4 (42% / 72%) | 60.7 |
| `su4_local` P=2 | 1.27 | 75% | — | 18.6 / 12.9 (48% / 70%) | 18.8 / 13.1 (48% / 70%) | 64.0 |

`partition_imbalance` is 1.000–1.004 everywhere; exchange volume is `bytes ≈ rows × 48`.

### Benchmarking

`P = 1` versus `P = N` is a runtime-knob A/B of one binary, not a code A/B:

```bash
scripts/ab-compare.sh partitions-1v2 --a . --b . \
  --probe   '--n 1000000 --threads 32 --layers su4_local --partitions 1' \
  --probe-b '--n 1000000 --threads 32 --layers su4_local --partitions 2 \
             --partition-cpus "0-15,32-47;16-31,48-63"'
```

`--threads` is the total at every `P`.
A partitioned cell runs under **no** placement prefix: `numactl` and `taskset` both defeat the split the engine is making.
On an exclusive cluster node, `scripts/slurm/ab-campaign.sbatch` builds the CPU lists from the node's own topology.
The protocol and the sidecar fields are in [`benchmarks/PROFILING.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/PROFILING.md).

## Distributed: weak scaling {#distributed}

The same split across *processes* is one partition per MPI rank ([MPI ranks](../../manual/propagation/mpi.md)), and there the goal is capacity rather than bandwidth.
Measured on Icelake `ccq` nodes (2 × 32 cores), one rank per NUMA domain, 32 threads per rank, UCX shared memory within a node and InfiniBand between nodes, at 6·10⁶ terms per rank:

| | 2 ranks (1 node) | 4 ranks (2 nodes) | 8 ranks (4 nodes) |
|---|---|---|---|
| layer with no row crossing | 10.6 ms | 11.1 ms | 10.3 ms |
| Pauli rotation whose generator crosses | 37.5 ms (**3.5×**) | 48.3 ms (**4.4×**) | 48.6 ms (**4.7×**) |

**Weak scaling is flat once the exchange leaves the node**: the remote layer costs the same at 4 and 8 ranks, and local layers cost ~10.5 ms whatever the rank count.
The step from 2 to 4 is shared memory giving way to InfiniBand.
A remote layer is **transfer-bound** — the pipeline runs the coset loop under the transfer, so what is left is the export pass plus the bytes on the wire — which means the lever is fewer bytes (locality rows, a lower truncation), not more threads.

The full per-layer breakdown, copied from [`research/HARDWARE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/HARDWARE.md) §Partitioned engine, MPI weak scaling: `phase_breakdown --mpi`, replicated input `--n 4e6 × ranks` → 6.0e6 terms per rank (4.0e6 for `cnot`), `--reps 8`; medians over ranks, ms per layer.

| ranks (nodes) | layer | wall | export | hidden transfer (chunk wait, busy/32) | coset loop | remote/local | peak RSS/rank |
|---|---|---|---|---|---|---|---|
| 2 (1) | rotation_local | 10.6 | 0 | 0 | 8.6 | 1 | 2.3 GB |
| 2 (1) | rotation_remote | 37.5 | 8.0 | ~20 | 26.1 | 3.5× | 2.3 GB |
| 2 (1) | cnot | 29.5 | 10.0 | ~9 | 14.6 | — | 2.4 GB |
| 4 (2) | rotation_local | 11.1 | 0 | 0 | 6.9 | 1 | 3.3 GB |
| 4 (2) | rotation_remote | 48.3 | 7.2 | ~27 | 32.7 | 4.4× | 3.3 GB |
| 4 (2) | cnot | 28.7 | 8.8 | ~11 | 16.3 | — | 3.3 GB |
| 8 (4) | rotation_local | 10.3 | 0 | 0 | 6.8 | 1 | 6.3 GB |
| 8 (4) | rotation_remote | 48.6 | 7.3 | ~28 | 33.3 | 4.7× | 6.3 GB |
| 8 (4) | cnot | 42.8 | — | — | — | — | 6.3 GB |

The remote layer is transfer-bound: ~6–8 ms compute, 7–8 ms export, ~30 ms transfer for 192 MB per rank (two ranks share a NIC → ~13 GB/s per node).
`vmhwm` growth with rank count is the probe's replicated input, not the engine; engine-side peak per rank is flat.

## Noise floor

Single-shot campaign noise on the reference host is ±5–8% single-threaded and ±10–26% at 8–32 threads; untouched code moves that much between campaigns.
Effects below that need the interleaved A/B protocol (`scripts/ab-compare.sh`), whose acceptance criterion is direction consistency across every pair.

Sources: [`research/HARDWARE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/research/HARDWARE.md) (roofline tables, phase shares, bandwidth ceilings, partitioned and distributed tables, all achieved numbers and verdicts); [`benchmarks/PROFILING.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/benchmarks/PROFILING.md) (byte model, interpretation rules, noise floor); [`ARCHITECTURE.md`](https://github.com/lkdvos/paulistrings-rs/blob/main/ARCHITECTURE.md) §Data-Model, §Width, §Performance-Model (layout and dispatch).
