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

## ccqlin038 — GPU

NVIDIA RTX A6000 (GA102, sm_86, 48 GB GDDR6, 768 GB/s spec), driver-managed clocks: 210 / 405 MHz idle, 1800 MHz SM / 7601 MHz memory under load, 44–66 °C, 175–183 W during the cells below.
Shared box; load average 2–5 during the device cells, 6–11 during the host cells.

### Device memory bandwidth ceilings

`crates/membench --device 0` (feature `cuda`) via `scripts/bandwidth.sh --device 0`; STREAM-convention nominal bytes, grid-stride f64 kernels, best of 5 reps.

| arrays | read | write | copy | triad |
|---|---:|---:|---:|---:|
| 512 MiB | 703.7 | 706.6 | 674.4 | 676.2 |
| 2 GiB | 709.7 | 711.4 | 658.8 | 673.0 |

GB/s; read and write reach 92% of the 768 GB/s spec.

### Device layer vs host, first table

`phase_breakdown --device 0` against `phase_breakdown --threads 16,32` (`scripts/jcc-rustflags.sh` sourced), `--qubits 128` (`W = 2`), truncation `keep`, `--reps 5`: one untimed application drives the sum to its steady state, the timed call applies the layer five more times, both sides identically.
`m` is the steady-state term count; ns per term is wall per layer over `m`; the ratio is host over device.

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

`su4` at `--n 16000000` (2.3e8 steady terms) does not fit the 48 GB device and was not run.
The host `su4` row at 5.65e7 (74.3 ns/term at 16t) is above the 49–58 ns/term of earlier quiet measurements; the box carried a load average of 6–11 during it.

Device phases per layer (`phase-timing`, CUDA events; K1+K2 = `gather_ns`, K3 = `merge_ns`, K4 = `compact_ns`, `coset_loop_ns` the driving thread's wall):

| cell | m | K1+K2 | K3 | K4 | coset loop | records/layer | ns per record (K3) |
|---|---|---:|---:|---:|---:|---:|---:|
| `rotation_zz` | 1.50e6 | 0.126 | 1.370 | 0.283 | 1.840 | 2.50e6 | 0.55 |
| `cnot` | 1.00e6 | 0.085 | 0.760 | 0.200 | 1.103 | 1.00e6 | 0.76 |
| `gu2q` | 3.25e6 | 0.208 | 2.900 | 0.349 | 3.521 | 8.65e6 | 0.34 |
| `su4` | 1.41e7 | 1.044 | 61.98 | 2.440 | 65.64 | 2.11e8 | 0.29 |
| `su4` | 5.65e7 | 4.035 | 248.7 | 9.838 | 263.3 | 8.44e8 | 0.29 |

ms; the fused kernel is 94% of a dense layer and 69–82% of a sparse one, where the fixed per-block cost (0.55–0.76 ns per record at ~1000 records per block) dominates.
In-layer copies are 0.02–0.5 ms per layer (`h2d_ns` + `d2h_ns`); the per-process NVRTC compile is 3.5 s inside the first cell's `upload_ns`, and `download_ns` (`to_host`, pinned D2H plus the host re-sort into `PauliSum`) is 0.8 s at 1.41e7 and 3.1 s at 5.65e7 terms.

## `gpu` cluster nodes — one process, several devices

From `scripts/slurm/gpu-devices.sbatch` runs (`scripts/slurm/README.md`, The GPU jobs).
Same conventions as the ccqlin038 tables: `--qubits 128`, truncation `keep` (`heavyhex_step` five steps under `coeff:2^-13`, 1355 layers), `--reps 5`, `m` the steady-state term count, ms per layer, device exchange (`PAULISTRINGS_GPU_EXCHANGE` unset).
The partitioned row keeps the single device's total `m`, so its speedup is strong scaling.

| node | GPUs | interconnect (`nvidia-smi topo -m`) | SM / memory clock under load | job |
|---|---|---|---|---|
| workergpu068, A100-SXM4-80GB | 2 of 4 | NV4 between the pair | 1410 / 1593 MHz | 7101047, rev 01df3eb |
| workergpu065, A100-SXM4-80GB | 4 | NV4 between every pair | 1410 / 1593 MHz | 7099959, rev 8234874 |
| H100-SXM5 | 4 | | | |

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

A layer without remote deltas scales; a layer with them is exchange-bound, at ≈ 27 GB/s aggregate for `su4` (2.35e10 bytes in 884 ms) against the 100 GB/s per direction of four NVLink3 links.
Every row above predates `cuMemPoolSetAccess` in `enable_peer_access`: cudarc allocates from the stream-ordered pool, which `cuCtxEnablePeerAccess` does not map, so the exchange staged through the host and direct peer loads faulted (`gpu_peer`, jobs 7101880 and 7102281: 21.7 GB/s peer copies against 26 GB/s pinned host copies and 880 GB/s same-device, `CUDA_ERROR_ILLEGAL_ADDRESS` from a kernel reading its peer, `nvidia-smi topo -p2p` OK).
The exported bytes are pre-dedup deltas, 7.4 rows per steady-state term on `su4`, 56 bytes each at `W = 2`.
One A100 runs `su4` at 3.61 ns/term against the A6000's 4.66.

## `gpu` cluster nodes — one device per MPI rank

From `scripts/slurm/mpi-gpu-ranks.sbatch` runs.
Replicated input, so `m` per rank equals the single-device `m` above (`heavyhex_step` excepted: its sum is split, 5.8e5 per rank); one GPU and 8 CPUs per rank, rank 0 shown (rank 1 within 4%), ms per layer.
Exchange goes through host memory (`h2d` + `d2h` is 55–57% of a remote layer's wall).

| ranks (nodes) | node | layer | m per rank | wall | export | exchange | chunk wait | coset loop | h2d + d2h | bytes exported/rank | peak RSS/rank |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 (1) | A100, job 7101048 | `rotation_zz` | 6.00e6 | 5.75 | 0 | 0 | 0 | 5.72 | 0.05 | 0 | 2.1 GB |
| 2 (1) | A100 | `cnot` | 4.00e6 | 45.66 | 17.00 | 0.20 | 14.53 | 28.40 | 26.16 | 9.61e7 | 2.2 GB |
| 2 (1) | A100 | `gu2q` | 1.30e7 | 296.4 | 89.92 | 7.40 | 105.3 | 197.9 | 165.5 | 8.07e8 | 4.9 GB |
| 2 (1) | A100 | `su4` | 5.65e7 | 7357 | 2253 | 172.7 | 2669 | 4871 | 4145 | 2.02e10 | 58.7 GB |
| 2 (1) | A100 | `rotation_remote` | 6.00e6 | 72.83 | 22.01 | 0.08 | 26.04 | 50.70 | 39.98 | 1.92e8 | 58.7 GB |
| 2 (1) | A100 | `heavyhex_step` | 5.77e5 | 1.95 | 0.47 | 0.04 | 0.29 | 1.42 | 0.65 | 2.08e6 | 0.6 GB |
| 4 (1) | A100 | | | | | | | | | | |
| 8 (2) | A100 | | | | | | | | | | |

`rotation_zz` weak-scales flat (5.75 vs 5.74 ms on one device); the peak RSS is the probe's replicated input, carried from `su4` into the later cells.

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
