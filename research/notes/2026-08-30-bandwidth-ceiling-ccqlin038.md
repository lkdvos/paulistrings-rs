# Memory-bandwidth ceiling — ccqlin038

Measured 2026-08-30 with `crates/membench` via `scripts/bandwidth.sh` (raw:
`benchmarks/results/2026-08-30-ccqlin038/bandwidth.txt`). STREAM-convention
nominal bytes, plain (write-allocating) stores, best of 5 reps over 512 MiB
f64 arrays. Rerun only after hardware changes.

## Headline numbers (best GB/s)

| placement | read | write | copy | triad |
|---|---:|---:|---:|---:|
| 1 core, node-local | 11.3 | 10.1 | 9.5 | 11.3 |
| 1 core, remote (cross-socket) | 7.8 | 7.2 | 5.5 | 8.4 |
| one socket, 8 physical (either node — symmetric) | 39.0 | 18.6 | 35.6 | 28.1 |
| one socket, 8 phys + 8 HT | 39.2 | 18.4 | 34.9 | 27.6 |
| both sockets, 16 physical | 45.0 | 25.3 | 40.0 | 38.1 |
| both sockets, 16 phys, interleaved pages | 41.2 | 21.3 | 31.8 | 33.5 |
| both sockets, 32 threads | 48.8 | 23.1 | 33.8 | 36.5 |

Uncore cross-check (`perf stat -a uncore_imc/cas_count_*` during the node0
read run): 38.3 vs 39.0 GB/s — the CAS→GB/s conversion and the kernel agree
within 2%.

## What this means

- **The real per-socket ceiling is ~39 GB/s read / ~28 GB/s triad**, not the
  6-channel spec figure (140.8 GB/s/socket DDR4-2933) earlier notes implicitly
  assumed when calling things "bandwidth-bound". The measured value is
  consistent with **2 of 6 memory channels populated per socket**
  (2 × 23.4 = 46.9 GB/s nominal; 39/46.9 = 83% efficiency, typical). Every
  prior roofline intuition on this box was optimistic by ~3.6×.
- **Hyperthreads add nothing to bandwidth** (39.0 → 39.2). They only help
  latency-bound phases.
- **The second socket adds only ~15–25% in practice** (39 → 45–49 read), not
  2×, because pages are placed by first touch and Rayon work-stealing then
  reads ~half of them remotely; remote streams run at 7.8 GB/s/core.
  Interleaving pages does not help (41.2). A perfectly NUMA-partitioned
  workload would see ~78 GB/s aggregate; the engine's current design cannot,
  and this — not core count — is the likely wall behind the flat scaling
  above ~16 threads.
- Write is ~half of read on a socket (18.6 vs 39.0 nominal): read-for-
  ownership doubles the actual write traffic. The engine also uses plain
  stores, so compare its phases against these nominal numbers as-is.

Serial phases compare against the 1-core row; the coset loop against the row
matching the run's placement.
