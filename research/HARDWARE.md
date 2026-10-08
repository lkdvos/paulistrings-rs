# Hardware facts

Measured host facts and roofline denominators; numbers are as measured, never rounded or re-derived.
Method write-ups are in git history (`git log --diff-filter=D -- research/`).

## ccqlin038 — reference workstation

2× Xeon Gold 6244 @ 3.60 GHz, 16c/32t, 2 NUMA nodes, governor `powersave`, microcode `0x5003901`, shared box.
Cascade Lake-SP: affected by the JCC erratum (SKX102).

### Memory bandwidth ceilings

`scripts/bandwidth.sh` (`crates/membench`): STREAM-convention bytes, write-allocating stores, best of 5 over 512 MiB f64 arrays.

| placement | read | write | copy | triad |
|---|---:|---:|---:|---:|
| 1 core, node-local | 11.3 | 10.1 | 9.5 | 11.3 |
| 1 core, remote (cross-socket) | 7.8 | 7.2 | 5.5 | 8.4 |
| one socket, 8 physical (either node — symmetric) | 39.0 | 18.6 | 35.6 | 28.1 |
| one socket, 8 phys + 8 HT | 39.2 | 18.4 | 34.9 | 27.6 |
| both sockets, 16 physical | 45.0 | 25.3 | 40.0 | 38.1 |
| both sockets, 16 phys, interleaved pages | 41.2 | 21.3 | 31.8 | 33.5 |
| both sockets, 32 threads | 48.8 | 23.1 | 33.8 | 36.5 |

GB/s; the uncore IMC counters agree (38.3 against 39.0 read).
2 of 6 memory channels are populated per socket (46.9 GB/s nominal, 83% achieved); hyperthreads add nothing and the second socket adds ~15–25%.

### Engine roofline, single thread

`phase_breakdown --qubits 128` (`W = 2`), truncation `keep`, `scripts/perf-stat.sh --reps 40` with the idle DRAM baseline subtracted; byte model 48 B/term against the 1-core node-local ceilings.

| cell | m | ns/term | model GB/s | measured GB/s | % of copy | model / measured | IPC | LLC load-miss | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `rotation_zz` | 1.50e6 | 30.6 | 8.5 | 2.53 | 27% | 3.3× | 2.26 | 41.1% | latency-bound |
| `rotation_zz` | 4.50e6 | 30.5 | 8.3 | 3.29 | 35% | 2.5× | 2.24 | 38.3% | latency-bound |
| `gu2q` | 3.0e6 | 141.3 | 9.0 | 2.08 | 22% | 4.3× | 2.62 | 34.9% | latency-bound |
| `gu2q` | 1.0e6 | 138.5 | 9.8 | 2.07 | 22% | 4.7× | 2.68 | 33.6% | latency-bound |
| `su4` | 1.41e7 | 327.3 | 8.6 | 0.67 | 7% | 12.8× | 2.98 | 2.1% | compute-bound |

Per-term cost is flat in `m` (`su4` 322–327 ns/term over 1.41e7 → 4.24e7).

### Engine roofline, thread scaling

`su4` at m = 1.41e7, against the both-socket ceilings above.

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

Parallel efficiency is 0.99 for `su4` at 16t; use 16 threads for dense-PTM circuits and 32 for sparse-rotation circuits on this host.

## ccqlin038 — GPU

NVIDIA RTX A6000 (GA102, sm_86, 48 GB GDDR6, 768 GB/s spec), driver-managed clocks at 1800 MHz SM / 7601 MHz memory under load; shared box.

### Device memory bandwidth ceilings

`scripts/bandwidth.sh --device 0`: STREAM-convention bytes, grid-stride f64 kernels, best of 5.

| arrays | read | write | copy | triad |
|---|---:|---:|---:|---:|
| 512 MiB | 703.7 | 706.6 | 674.4 | 676.2 |
| 2 GiB | 709.7 | 711.4 | 658.8 | 673.0 |

GB/s; read and write reach 92% of the 768 GB/s spec.

### Device layer vs host, first table

`phase_breakdown --device 0` against `--threads 16,32`, `--qubits 128`, truncation `keep`, `--reps 5` after one untimed application; `m` is the steady-state term count, the ratio host over device.

| cell | m | device ms/layer | device ns/term | host 16t ns/term | host 32t ns/term | ratio vs 16t | ratio vs 32t |
|---|---|---:|---:|---:|---:|---:|---:|
| `rotation_zz` | 1.50e6 | 1.845 | 1.23 | 2.48 | 2.59 | 2.0× | 2.1× |
| `rotation_zz` | 6.00e6 | 6.920 | 1.15 | 2.63 | 2.80 | 2.3× | 2.4× |
| `rotation_zz` | 2.40e7 | 26.84 | 1.12 | 4.17 | 3.34 | 3.7× | 3.0× |
| `cnot` | 1.00e6 | 1.109 | 1.11 | 3.37 | 3.61 | 3.0× | 3.3× |
| `cnot` | 4.00e6 | 3.948 | 0.99 | 4.66 | 3.91 | 4.7× | 4.0× |
| `cnot` | 1.60e7 | 15.35 | 0.96 | 3.76 | 3.74 | 3.9× | 3.9× |
| `gu2q` | 3.25e6 | 3.528 | 1.09 | 3.07 | 3.05 | 2.8× | 2.8× |
| `gu2q` | 1.30e7 | 13.58 | 1.04 | 3.88 | 3.25 | 3.7× | 3.1× |
| `gu2q` | 5.20e7 | 52.25 | 1.00 | 3.90 | 3.04 | 3.9× | 3.0× |
| `su4` | 1.41e7 | 65.65 | 4.65 | 51.3 | 61.4 | 11.0× | 13.2× |
| `su4` | 5.65e7 | 263.3 | 4.66 | 74.3 | 69.8 | 15.9× | 15.0× |
| `heavyhex_step` (5 steps, `coeff:2^-13`, 1355 layers, final m) | 1.16e6 | 2.122 | 1.84 | 3.53 | 3.52 | 1.9× | 1.9× |
| `trotter` (64 layers, 100 → 6.7e4 terms) | 6.7e4 | 9.316 | 139.5 | 130.8 | 121.7 | 0.94× | 0.87× |

`su4` at 2.3e8 steady terms does not fit the 48 GB device; the host `su4` row at 5.65e7 ran under load average 6–11 (quiet: 49–58 ns/term).

Device phases per layer (`phase-timing`, CUDA events; K1+K2 = `gather_ns`, K3 = `merge_ns`, K4 = `compact_ns`, `coset_loop_ns` the driving thread's wall):

| cell | m | K1+K2 | K3 | K4 | coset loop | records/layer | ns per record (K3) |
|---|---|---:|---:|---:|---:|---:|---:|
| `rotation_zz` | 1.50e6 | 0.126 | 1.370 | 0.283 | 1.840 | 2.50e6 | 0.55 |
| `cnot` | 1.00e6 | 0.085 | 0.760 | 0.200 | 1.103 | 1.00e6 | 0.76 |
| `gu2q` | 3.25e6 | 0.208 | 2.900 | 0.349 | 3.521 | 8.65e6 | 0.34 |
| `su4` | 1.41e7 | 1.044 | 61.98 | 2.440 | 65.64 | 2.11e8 | 0.29 |
| `su4` | 5.65e7 | 4.035 | 248.7 | 9.838 | 263.3 | 8.44e8 | 0.29 |

ms; the fused kernel is 94% of a dense layer and 69–82% of a sparse one.
Both tables' `cnot` rows are the fused layer; the permutation path (`FINDINGS.md §GPU Clifford permutation path`) runs the same cell at 0.386 ms per layer at 1.00e6 and 5.07 ms at 1.60e7.
In-layer copies are 0.02–0.5 ms per layer; `to_host` is 0.8 s at 1.41e7 and 3.1 s at 5.65e7 terms.

## `gpu` cluster nodes — one process, several devices

`scripts/slurm/gpu-devices.sbatch`, same conventions as the ccqlin038 tables, ms per layer; the multi-device rows keep the single device's total `m` (strong scaling).

| node | GPUs | interconnect (`nvidia-smi topo -m`) | SM / memory clock under load | job |
|---|---|---|---|---|
| workergpu068, A100-SXM4-80GB | 2 of 4 | NV4 between the pair | 1410 / 1593 MHz | 7101047, rev 01df3eb |
| workergpu065, A100-SXM4-80GB | 4 | NV4 between every pair | 1410 / 1593 MHz | 7099959, rev 8234874 |
| workergpu046, A100-SXM4-80GB | 2 of 4 | NV4 between the pair | not recorded | 7110164, rev 54b7bd2 |

| node | cell | m | 1 device ms/layer | 1 device ns/term | 2 devices ms/layer | 2 devices ns/term | speedup | export ms | exchange ms | barrier ms | bytes exported/layer |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A100 | `rotation_zz` | 6.00e6 | 5.743 | 0.96 | 3.105 | 0.52 | 1.85× | 0 | 0 | 0.03 | 0 |
| A100 | `cnot` | 4.00e6 | 3.329 | 0.83 | 8.319 | 2.08 | 0.40× | 0.48 | 5.19 | 1.26 | 1.12e8 |
| A100 | `gu2q` | 1.30e7 | 10.66 | 0.82 | 51.27 | 3.94 | 0.21× | 1.83 | 38.27 | 11.97 | 9.41e8 |
| A100 | `su4` | 5.65e7 | 204.1 | 3.61 | 1249 | 22.11 | 0.16× | 42.78 | 884.0 | 308.0 | 2.35e10 |
| A100 | `heavyhex_step` | 1.16e6 | 1.731 | 1.50 | 1.308 | 1.13 | 1.32× | 0.05 | 0.26 | 0.04 | 4.86e6 |
| A100 | `rotation_remote` | 6.00e6 | — | — | 12.86 | 2.14 | — | 0.53 | 9.50 | 2.51 | 2.24e8 |

Four devices, workergpu065 (one device 5.729 / 3.326 / 11.02 / 204.4 / 1.733 ms per layer on the same cells):

| cell | 4 devices ms/layer | speedup | export ms | exchange ms | barrier ms | bytes exported/layer |
|---|---:|---:|---:|---:|---:|---:|
| `rotation_zz` | 1.782 | 3.21× | 0 | 0 | 0.1 | 0 |
| `cnot` | 8.148 | 0.41× | 0.5 | 6.0 | 1.4 | 1.68e8 |
| `gu2q` | 25.64 | 0.43× | 1.1 | 19.1 | 5.4 | 9.41e8 |
| `su4` | 1542 | 0.13× | 34.0 | 1102 | 363.1 | 3.53e10 |
| `rotation_remote` | 7.232 | — | 0.3 | 5.0 | 1.1 | 2.24e8 |
| `heavyhex_step` | 1.023 | 1.69× | 0.1 | 0.3 | 0.1 | 6.85e6 |

These two tables lack the memory-pool peer grant and the sender-side merge, so their exchange staged through the host (≈ 27 GB/s aggregate for `su4`); the table below has both.
Unmerged exports are 7.4 rows per steady-state term on `su4`, 56 bytes each at `W = 2`.
One A100 runs `su4` at 3.61 ns/term against the A6000's 4.66.

Two devices with the peer-pool grant and the sender-side merge, workergpu046 (job 7110164, rev 54b7bd2):

| cell | m | 1 device ms/layer | 2 devices ms/layer | 2 devices ns/term | speedup | export ms | exchange ms | barrier ms | bytes exported/layer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cnot` | 4.00e6 | 3.323 | 4.207 | 1.05 | 0.79× | 0.48 | 1.54 | 0.17 | 1.12e8 |
| `su4` | 5.65e7 | 211.2 | 194.4 | 3.44 | 1.09× | 74.40 | 38.36 | 13.45 | 3.17e9 |
| `rotation_remote` | 6.00e6 | — | 5.892 | 0.98 | — | 0.53 | 2.50 | 0.92 | 2.24e8 |
| `heavyhex_step` | 1.16e6 | 1.734 | 1.109 | 0.96 | 1.56× | 0.05 | 0.07 | 0.02 | 4.86e6 |

Peer copies on an A100 pair run at 93.9 GB/s each way (88 GB/s for a kernel touching its peer), and 91.6–92.2 GB/s on every ordered pair of workergpu063's four devices.
The `su4` exchange moves the merged 3.17e9 bytes in 38.4 ms, ≈ 83 GB/s.

## `gpu` cluster nodes — one device per MPI rank

Replicated input (`heavyhex_step` split), one GPU and 8 CPUs per rank, rank 0 shown, ms per layer.
These rows exchange through host memory, since replaced by NCCL (next section).

| ranks (nodes) | node | layer | m per rank | wall | export | exchange | chunk wait | coset loop | h2d + d2h | bytes exported/rank | peak RSS/rank |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 (1) | A100, job 7101048 | `rotation_zz` | 6.00e6 | 5.75 | 0 | 0 | 0 | 5.72 | 0.05 | 0 | 2.1 GB |
| 2 (1) | A100 | `cnot` | 4.00e6 | 45.66 | 17.00 | 0.20 | 14.53 | 28.40 | 26.16 | 9.61e7 | 2.2 GB |
| 2 (1) | A100 | `gu2q` | 1.30e7 | 296.4 | 89.92 | 7.40 | 105.3 | 197.9 | 165.5 | 8.07e8 | 4.9 GB |
| 2 (1) | A100 | `su4` | 5.65e7 | 7357 | 2253 | 172.7 | 2669 | 4871 | 4145 | 2.02e10 | 58.7 GB |
| 2 (1) | A100 | `rotation_remote` | 6.00e6 | 72.83 | 22.01 | 0.08 | 26.04 | 50.70 | 39.98 | 1.92e8 | 58.7 GB |
| 2 (1) | A100 | `heavyhex_step` | 5.77e5 | 1.95 | 0.47 | 0.04 | 0.29 | 1.42 | 0.65 | 2.08e6 | 0.6 GB |
| 4 (1) | A100, job 7110165 | `rotation_zz` | 6.00e6 | 5.81 | 0 | 0 | 0 | 5.72 | 0.06 | 0 | 3.3 GB |
| 4 (1) | A100 | `cnot` | 4.00e6 | 70.43 | 26.37 | 0.79 | 23.82 | 42.80 | 40.42 | 1.44e8 | 4.3 GB |
| 4 (1) | A100 | `gu2q` | 1.30e7 | 138.7 | 46.15 | 0.20 | 48.78 | 92.26 | 66.42 | 2.88e8 | 4.3 GB |
| 4 (1) | A100 | `su4` | 5.65e7 | 3607 | 1307 | 7.17 | 1303 | 2292 | 1767 | 8.13e9 | 46.2 GB |
| 4 (1) | A100 | `rotation_remote` | 6.00e6 | 85.70 | 24.54 | 0.08 | 30.85 | 57.31 | 44.31 | 1.92e8 | 46.2 GB |
| 4 (1) | A100 | `heavyhex_step` | 2.89e5 | 1.37 | 0.36 | 0.03 | 0.25 | 0.96 | 0.50 | 1.47e6 | 0.5 GB |

`rotation_zz` weak-scales flat; peak RSS is the probe's replicated input; the 2-rank rows are unmerged, the 4-rank rows have the sender-side merge.

## `gpu` cluster nodes — NCCL exchange between MPI ranks

One A100-SXM4-80GB node, every GPU visible to every rank, one GPU and 8 CPUs per rank, replicated input, rank 0, ms per layer, medians of five alternating host/NCCL pairs.
NCCL 2.23.4 chose `P2P/CUMEM/read` between every pair of ranks.

| ranks | node, job | layer | m per rank | host staging | NCCL | speedup | NCCL exchange | bytes exported/rank |
|---|---|---|---|---:|---:|---:|---:|---:|
| 2 | workergpu070, 7125331 | `rotation_zz` | 6.00e6 | 5.78 | 5.80 | 1.00× | 0 | 0 |
| 2 | | `cnot` | 4.00e6 | 43.62 | 6.92 | 6.3× | 2.49 | 9.61e7 |
| 2 | | `rotation_remote` | 6.00e6 | 72.35 | 10.70 | 6.8× | 4.36 | 1.92e8 |
| 2 | | `su4` | 5.65e7 | 1250 | 356.9 | 3.5× | 57.50 | 2.72e9 |

Every pair agrees in sign; a four-rank bring-up on workergpu063 takes 10.3 s of NCCL init and 0.7 s of warm-up.

## `ccq` cluster node types

One exclusive node each, governor `performance`; family/model from `/proc/cpuinfo`.

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

A multi-thread campaign on these nodes needs `--n` scaled with core count, roughly `3e7` on a 96-core node; `--n 1000000` is noise at 64–128 threads.

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
