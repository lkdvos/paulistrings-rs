# Hardware facts

Measured host data, cited as roofline denominators.
Numbers are as measured; do not round or re-derive them.
Full measurement write-ups are in git history under the deleted `research/notes/`.

## ccqlin038 — reference workstation

2× Xeon Gold 6244 @ 3.60 GHz, 16c/32t, 2 NUMA nodes, governor `powersave`, microcode `0x5003901`, shared box.
Cascade Lake-SP: affected by the JCC erratum (SKX102).

### Memory bandwidth ceilings

`crates/membench` via `scripts/bandwidth.sh`; STREAM-convention nominal bytes, plain (write-allocating) stores, best of 5 reps over 512 MiB f64 arrays.
Rerun only after hardware changes.

| placement | read | write | copy | triad |
|---|---:|---:|---:|---:|
| 1 core, node-local | 11.3 | 10.1 | 9.5 | 11.3 |
| 1 core, remote (cross-socket) | 7.8 | 7.2 | 5.5 | 8.4 |
| one socket, 8 physical (either node — symmetric) | 39.0 | 18.6 | 35.6 | 28.1 |
| one socket, 8 phys + 8 HT | 39.2 | 18.4 | 34.9 | 27.6 |
| both sockets, 16 physical | 45.0 | 25.3 | 40.0 | 38.1 |
| both sockets, 16 phys, interleaved pages | 41.2 | 21.3 | 31.8 | 33.5 |
| both sockets, 32 threads | 48.8 | 23.1 | 33.8 | 36.5 |

GB/s. Uncore cross-check (`perf stat -a uncore_imc/cas_count_*`, node0 read run): 38.3 vs 39.0 GB/s.
Consistent with 2 of 6 memory channels populated per socket (2 × 23.4 = 46.9 GB/s nominal; 39/46.9 = 83% efficiency), not the 140.8 GB/s/socket DDR4-2933 spec figure.
Hyperthreads add nothing (39.0 → 39.2); the second socket adds ~15–25%, not 2×; remote streams run at 7.8 GB/s/core.

### Engine roofline, single thread

`target/release/examples/phase_breakdown --qubits 128` (`W = 2`), truncation `keep`, counters via `scripts/perf-stat.sh --reps 40`, idle DRAM baseline (0.4–0.5 GB/s) subtracted; byte model `T = 48` B/term.
Ceilings: 1 core node-local read 11.3 / copy 9.5 / triad 11.3 GB/s.

| cell | m | ns/term | model GB/s | measured GB/s | % of copy | model / measured | IPC | LLC load-miss | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `rotation_zz` | 1.50e6 | 30.6 | 8.5 | 2.53 | 27% | 3.3× | 2.26 | 41.1% | latency-bound |
| `rotation_zz` | 4.50e6 | 30.5 | 8.3 | 3.29 | 35% | 2.5× | 2.24 | 38.3% | latency-bound |
| `gu2q` | 3.0e6 | 141.3 | 9.0 | 2.08 | 22% | 4.3× | 2.62 | 34.9% | latency-bound |
| `gu2q` | 1.0e6 | 138.5 | 9.8 | 2.07 | 22% | 4.7× | 2.68 | 33.6% | latency-bound |
| `su4` | 1.41e7 | 327.3 | 8.6 | 0.67 | 7% | 12.8× | 2.98 | 2.1% | compute-bound |

Per-term cost is flat in `m`: `rotation_zz` 30.5–30.6 ns/term over 1.50e6→4.50e6, `gu2q` 138.5–141.3 over 1.0e6→3.0e6, `su4` 322–327 over 1.41e7→4.24e7.

### Engine roofline, thread scaling

`su4` at m = 1.41e7. Ceilings: 8/16t read 45.0 / write 25.3; 32t read 48.8 / write 23.1 GB/s.

| threads | ns/term | speedup | IPC | LLC load-miss | read GB/s | write GB/s | % read ceil | % write ceil | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 327.3 | — | 2.98 | 2.1% | 0.61 | 0.27 | 5% | 1% | compute-bound |
| 8 | 56.2 | 5.8× | 2.21 | 75.4% | 25.9 | 18.2 | 58% | 72% | approaching write ceiling |
| 16 | 49.3 | 6.6× | 1.26 | 79.5% | 39.3 | 28.1 | 87% | 111% | write-bandwidth-bound |
| 32 | 55.3 | 5.9× | 0.58 | 71.0% | 39.5 | 27.6 | 81% | 120% | write-bandwidth-bound |

Sparse layers at 32 threads, m = 4.50e6 (`rotation_zz`) / 3.0e6 (`gu2q`):

| cell | ns/term | speedup vs 1t | IPC | LLC load-miss | read GB/s | write GB/s | % read ceil | % write ceil | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `rotation_zz` 32t | 2.7 | 11.3× | 0.97 | 34.4% | 8.9 | 11.5 | 18% | 50% | latency-bound |
| `gu2q` 32t | 10.8 | 13.1× | 1.29 | 29.4% | 11.7 | 11.3 | 24% | 49% | latency-bound |

Phase shares (share of summed worker busy time, gather/sort/merge):

| layer | 1t | 16t | 32t |
|---|---|---|---|
| `rotation_zz` (m=4.50e6) | 54/9/37 | 53/10/36 | 68/9/22 |
| `gu2q` (m=3.0e6) | 38/33/29 | 35/31/33 | 39/29/31 |
| `su4` (m=4.24e7) | 41/51/8 | 35/55/10 | 26/65/9 |

Parallel efficiency (busy / (coset-loop wall × threads)) is 0.99 for `su4` at 16t.
Thread guidance on this host: 16 threads for dense-PTM-heavy circuits, 32 for sparse-rotation circuits.

## `ccq` cluster node types

`scripts/slurm/jcc-portability.sbatch`, one exclusive node each, governor `performance`; family/model read from `/proc/cpuinfo`.

| node | CPU | family/model | JCC erratum | sockets × cores | NUMA |
|---|---|---|---|---|---|
| rome | AuthenticAMD Zen2 | 23 / 49 | no | — | — |
| genoa | AuthenticAMD Zen4 | 25 / 17 | no | 2 × EPYC 9474F, 48c | 2 (NPS1) |
| icelake | GenuineIntel Ice Lake-SP | 6 / 106 | no | 2 × Xeon Platinum 8362, 32c, no SMT | 2 |

### JCC branch-padding cost off Skylake

`-Cllvm-args=-x86-branches-within-32B-boundaries`, 7 pairs `abba` per cell, `--reps 20`, 1 thread.

| quantity | value |
|---|---|
| code size, padded vs unpadded (same commit, all three parts) | 2 044 024 B vs 2 003 272 B, +40 752 B = +2.03% |
| direction-consistent (7/7) phase results, rome + genoa + icelake | 13 padded-slower, 0 padded-faster |
| magnitude range | +0.60% to +3.78% per phase, ~+1% wall where wall resolves |
| same flag on ccqlin038 (Cascade Lake) | DSB residency 45.8% → 98.0%, −9..−13% wall |

Multi-thread cells of that campaign are unusable: `--n 1000000` leaves ~10–20k terms per worker at 64–128 threads (rome `rotation_zz` at 128 threads spans −33.31% to +32.55%).
A multi-thread campaign needs `--n` scaled with core count, roughly `3e7` on a 96-core node.

## Partitioned engine, P=1 vs P=2 in-process

`scripts/ab-compare.sh` runtime-knob A/B, 5 pairs `abba`, `--n 1000000 --qubits 128` (`W = 2`), random partition rows, each partition pinned to its socket with `MPOL_BIND`.
Steady-state `m` ≈ 1.5e6 for rotations, 1.0e6 for `gu2q`, 1.4e7 for `su4`/`su4_local`.

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

## Partitioned engine, MPI weak scaling

Icelake `ccq` nodes, one rank per NUMA domain, 32 threads per rank, InfiniBand between nodes and UCX shared memory within one; `phase_breakdown --mpi`, replicated input `--n 4e6 × ranks` → 6.0e6 terms per rank (4.0e6 for `cnot`), `--reps 8`.
Medians over ranks, ms per layer.

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

## Measurement noise

Single-shot campaign noise on ccqlin038 is ±5–8% single-threaded and ±10–26% at 8–32 threads.
Pinning to one physical core with `--reps 20` reduces the single-thread spread to ~1%.
`rdtsc` must not be used as the cycle metric on this host: `constant_tsc`/`nonstop_tsc` count reference cycles at the 3.6 GHz nominal, so under `powersave` at 1200 MHz it over-reports core cycles by ~3×.
