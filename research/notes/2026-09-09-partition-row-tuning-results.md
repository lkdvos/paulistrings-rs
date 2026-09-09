# Partition-row tuning — results (phase 5, C1: in-process P=2 on ccqlin038)

Plan: `research/plans/2026-09-09-partition-row-tuning.md`. Engine state: branch `partitioned-engine` at
`f39341a` (cut rows `PartitionRows::cut`, greedy selector `select_rows`, probe layers `tfim_step` /
`heavyhex_step`, `--partition-rows random|cut|select`). Raw: `benchmarks/results/2026-09-09-ccqlin038/p5-*`,
log `benchmarks/results/2026-09-09-ccqlin038-phase5-c1.log`.

## Setup

`scripts/ab-compare.sh --a . --b . --probe '<cell>' --probe-b '<cell> --partitions 2 --partition-cpus … --partition-rows <rows>'`,
3 pairs `abba`, load 0.5–10 (shared box), `RUST_LOG` unset. Cells: `--qubits 128 --reps 5 --truncation
coeff:0.00024` (≈ 2⁻¹²), initial operator `Z` on qubit 64 (`--initial z0`), `--partition-seed 7`
(**never the default seed**: the partition-row salt equalled the default hash seed before `f39341a`).
`tfim_step` = open 128-chain, ZZ layer then X layer per step (255 channels/step); `heavyhex_step` = the
127-qubit heavy-hex kicked-Ising step of the presentation workload (271 channels/step); θzz = −π/2,
θh = 5π/16. Peak resident ≈ 1e6 terms (2⁻¹² over 5 steps) — the *volume* metrics are what this
experiment is about; wall at this size is a sanity check of the direction.

## Table — P=2 (one partition per socket) vs the single-process engine, per Trotter step

| workload | rows | threads | P=1 ms/step | P=2 ms/step | Δ wall (pairs) | remote layers / step | rows exported / step | imbalance (final, max) |
|---|---|---|---|---|---|---|---|---|
| heavyhex_step | random | 16 | 272 | 563 | **+94%** 3/3 | 139 / 271 | 5.0e6 | 1.00, 1.01 |
| heavyhex_step | random | 32 | 270 | 553 | **+104%** 3/3 | 139 / 271 | 5.0e6 | 1.00, 1.01 |
| heavyhex_step | **cut** | 16 | 271 | 208 | **−20.7%** 3/3 | **4 / 271** | **4.8e5** | 1.006, 1.09 |
| heavyhex_step | **cut** | 32 | 264 | 221 | **−14.5%** 3/3 | 4 / 271 | 4.8e5 | 1.006, 1.09 |
| tfim_step | random | 16 | 109 | 236 | **+118%** 3/3 | 134 / 255 | 1.7e6 | 1.00, 1.01 |
| tfim_step | random | 32 | 99 | 273 | **+152%** 3/3 | 134 / 255 | 1.7e6 | 1.00, 1.01 |
| tfim_step | **cut** | 16 | 105 | 96 | **−8.6%** 3/3 | **1 / 255** | **9.6e4** | 1.03, 1.12 |
| tfim_step | **cut** | 32 | 100 | 107 | **+6.9%** 3/3 | 1 / 255 | 9.6e4 | 1.03, 1.12 |

Cut construction: chain bisected at qubit 64 (1 of 127 ZZ edges crosses); heavy-hex blocks `[0, 65)`,
`[65, 128)` (4 of 144 edges cross). Every single-qubit X rotation is local under a z-only row.

## Reading

- **The hypothesis holds.** Cut rows leave exactly the cut-crossing ZZ layers remote (1 per step on the
  chain, 4 on heavy-hex) against ~half of all layers under random rows; exported rows drop 17× (chain) and
  10× (heavy-hex). Imbalance from a single-site start settles at 1.03 (chain) / 1.006 (heavy-hex) with a
  transient max of 1.12 / 1.09.
- **On the primary workload the partitioned engine is now faster than the single-process one**: heavy-hex
  kicked Ising at P=2 runs 15–21% faster per step (3/3 pairs at both thread counts) — the NUMA gain
  measured in phase 2 on exchange-free layers, no longer eaten by the exchange. The chain is a wash
  (−9% at 16 threads, +7% at 32).
- `select` reproduces the chain's 1-remote-layer count and finds a 2-edge separator on heavy-hex, but the
  greedy's tie order — not balance — decides *which* edge is rejected, so on the chain it chose the
  `{0}`-vs-rest cut and left one partition empty; a warmed probe sum does not fix that. Balance-scored
  restarts are being added to the selector (measure `select` again after).
- Acceptance from the plan: remote fraction ≤ cut edges ✓; rows exported ≥ 5× down ✓ (10–17×); imbalance
  ≤ 1.2 ✓; P=2 wall within 1.3× of P=1 ✓ (faster on heavy-hex).

## Next

C2: MPI weak scaling with `--partition-rows cut` on Rusty (`scripts/slurm/mpi-ranks.sbatch` with
`PROBE_ARGS`), 2/4/8 ranks, `heavyhex_step` and `tfim_step`, against the random-rows rotation numbers in
`2026-09-08-numa-partitioning-results.md`; then the default-policy recommendation.
