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

## C2 — MPI, Rusty Icelake nodes, cut vs random rows (2026-09-10, head `f328e1f`, Slurm 7015679–85)

One rank per NUMA domain, 32 threads per rank, `--qubits 128 --reps 5 --initial z0 --partition-seed 7`;
the sum size is set by the physics (cutoff), so across rank counts this is **strong scaling of a small
problem**: at 2⁻¹² heavy-hex ends at 6.4e5 terms total (3.2e5 per rank at 2 ranks, 7.6e4 at 8); the
chain at 2⁻¹² is tiny (2.8e3 terms) and uninformative at 8 ranks. Medians over ranks, ms per step:

| ranks (nodes) | rows | heavy-hex ms/step | export | coset loop | remote layers/step | rows exported/step | chain ms/step |
|---|---|---|---|---|---|---|---|
| 2 (1) | random | 303 | 126 | 136 | 139 | 2.5e6 | 176 |
| 2 (1) | **cut** | **125** | 3.7 | 86 | **4** | 2.4e5 | 60 |
| 4 (2) | random | 292 | 110 | 111 | 204 | 1.8e6 | 191 |
| 4 (2) | **cut** | **133** | 6.6 | 60 | **12** | 1.4e5 | 62 |
| 8 (4) | random | 257 | 79 | 85 | 238 | 1.1e6 | 152 |
| 8 (4) | **cut** | **136** | 8.1 | 46 | **25** | 9.5e4 | 72 |
| 8 (4), cutoff 2⁻¹⁵ | cut | 2875 | 344 | 960 | 25 | 8.3e6 | — |

Reading:
- **Cut rows win 2.4× (2 ranks) to 1.9× (8 ranks) over random rows on the primary workload, at every
  rank count**, by leaving 4/12/25 layers per step remote instead of 139/204/238.
- **A floor at small per-rank size**: with cut rows the coset loop scales (86 → 60 → 46 ms) but the wall
  does not (125 → 133 → 136). The difference — 39, 73, 90 ms per step over 271 layers — is 30–70 µs per
  layer: the bucket-bits all-reduce every layer (plus the skew it exposes) over InfiniBand, on layers
  that otherwise need no communication. At the large per-rank sizes the engine targets a layer takes
  ≥ 10 ms and this is noise; at 1e5 terms per rank it is half the step. Fix in progress: bits collective
  only every K layers and before any remote layer, so an exchange-free layer makes no MPI call.
- The 2⁻¹⁵ point (peak ~1e6 terms per rank): 25 remote layers per step cost ~63 ms each against ~1 ms
  for a local layer — 16 MB messages per partner per layer in 2 MB pipeline chunks are latency-, not
  bandwidth-, bound at this size, and the export pass is 10× slower per term than at 6e6 terms per rank
  (14 ns/term). Both are small-message regimes outside the capacity target, but they price the
  cut-crossing layers for medium problems: look after the collective fix (fewer, larger chunks when a
  layer is small; export parallelism with few buckets).

## Selector after the balance-scored restarts (`38e07e3`, `afb99d7`; 2026-09-10, in-process P=2, 4 steps, cutoff 4e-3)

`select_rows` now runs the greedy under several tie orders (each accepted generator moved last in turn,
then seeded shuffles within weight/support tie groups) and ranks the answers: non-conserved → every
partition populated on the probe → least-loaded share in eighths of the ideal → remote weight → row
weight. Without a probe it is the old greedy bit for bit. Setup cost 7–14 ms once per run.

| workload | rows | remote layers / step | rows exported (4 steps) | final imbalance |
|---|---|---|---|---|
| chain (64 q) | select, before fix | 1 (at the chain's end) | 0 | one partition empty |
| chain (64 q) | select, after | 1 | 5.7e3 | 1.014 |
| chain (64 q) | cut | 1 | 6.0e3 | 1.01 |
| heavy-hex (127 q) | cut | 4 | 7.3e3 | 1.085 |
| heavy-hex (127 q) | select | **2** | 6.4e3 | **1.30** |

`select` finds a 2-edge separator on heavy-hex (half the remote layers of the hand bisection) but the
region it cuts off is smaller, so balance is worse (1.30 vs 1.085); the exported volume is similar
because the remote layers move the same anticommuting terms. Both are an order of magnitude below
random rows. Recommendation: for a known lattice use `cut` with a balanced bisection; `select` for
circuits without an obvious geometry, with the balance band as the knob if imbalance matters more than
remote weight.

## C2b — the bits-collective schedule over InfiniBand (2026-09-10, head `6355bdc`, Slurm 7015753/54)

Same cells as C2 (heavy-hex step, cut rows, 2⁻¹², 32 threads per rank), with the bucket-bits all-reduce
on the 16-layer schedule instead of every layer:

| ranks (nodes) | before ms/step (7015681/83) | after ms/step | coset loop | collectives per call | remote layers/step |
|---|---|---|---|---|---|
| 4 (2) | 133 | **85.6** (−36%) | 40 | 160 (was 1355) | 12 |
| 8 (4) | 136 | **81.1** (−40%) | 31 | 219 (was 1355) | 25 |

The per-layer collective was the floor: removing it takes the 4- and 8-rank steps from flat (125 → 133
→ 136 ms at 2/4/8 ranks) to scaling (125 → 86 → 81). What remains above the coset loop (~45–50 ms per
step) is the 12–25 cut-crossing layers at this small per-rank size (latency-bound 16 MB exchanges) plus
arrival skew. Random rows at the same cells: 292 / 257 ms — cut rows plus the schedule are now
**3.4× (4 ranks) and 3.2× (8 ranks) faster than random rows**.
