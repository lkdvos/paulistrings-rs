# NUMA partitioning, phase 2 — first measurements (P=1 vs P=2)

Paired A/B of the partitioned engine at P=2 against the same binary at P=1 (runtime-knob A/B:
`scripts/ab-compare.sh --a <rev> --b <rev> --probe ... --probe-b '... --partitions 2 --partition-cpus ...'`,
5 pairs, `abba`, `RUST_LOG` unset). `--threads` is the total; each partition gets half, pinned to its
socket, with `MPOL_BIND`. Partition rows are the default random rows (seeded from the hash). Raw data:
`benchmarks/results/2026-09-08-{ccqlin038,worker6140,worker6141,worker6142,worker7230}/` and the Slurm
logs `benchmarks/results/slurm-70049{70,71,72,73}.out`, `slurm-700500{1,2}.out` (worker6127 Icelake, worker7224 Genoa; gitignored).
Decision log: `2026-09-08-partitioned-phase1-log.md`. Success criterion (user, 2026-09-08): only the
intra-node NUMA effect must be a speedup; multi-node exists for memory capacity and may be slower.

## Provenance

| host | CPUs | NUMA | governor | who ran it | engine commit |
|---|---|---|---|---|---|
| ccqlin038 (workstation) | 2× Xeon Gold 6244, 8c/socket + HT | 2 | powersave, shared box, load 5–28 during the runs | this session | `4182a48`–`5c20312` |
| worker6140–6142 (Rusty `ccq`, exclusive) | 2× Xeon Platinum 8362, 32c/socket, no SMT | 2 | performance | Slurm 7004970/71/73 | `6254e70`'s parent tree |
| worker7230 (Rusty `ccq`, exclusive) | 2× EPYC 9474F, 48c/socket, no SMT | 2 (NPS1) | performance | Slurm 7004972 | same |

`--n 1000000 --qubits 128` (`W = 2`): steady-state `m` ≈ 1.5e6 for rotations, 1.0e6 for `gu2q`,
1.4e7 for `su4`/`su4_local`. Rotation cells `--reps 40`, dense cells `--reps 8`. Cluster nodes have no
`perf` counter access; counters below are ccqlin038 only (`scripts/perf-stat.sh`, shared box, approximate).

## Table A — wall time per layer, P=2 vs P=1 (median Δ%, pairs agreeing)

| cell | ccqlin038 16t | ccqlin038 32t | Icelake 32t | Icelake 64t | Genoa 48t | Genoa 96t |
|---|---|---|---|---|---|---|
| `su4` (random rows: ~half of 15 deltas remote) | **+128%** 6/6 | **+118%** 5/5 | +241% 5/5 | +274% 5/5 | +403% 5/5 | +373% 5/5 |
| `gu2q` | — | **+282%** 5/5 | — | — | +597% 5/5 | +574% 5/5 |
| `rotation_remote` (generator remote) | **+204%** 5/5 | **+217%** 5/5 | +310% 5/5 | +461% 5/5 | +730% 5/5 | +737% 5/5 |
| `rotation_local` (no exchange) | −13.6% 4/5 | −4.9% 4/5 | +16% 4/5 | **+21%** 5/5 | **−12.0%** 5/5 | −29.9% 4/5 |
| `su4_local` (no exchange) | **−4.1%** 5/5 | **−3.5%** 5/5 | **−6.6%** 5/5 | **−7.3%** 5/5 | **−8.7%** 5/5 | **−18.2%** 5/5 |

Bold = direction-consistent per the protocol. **`su4_local` is the NUMA result**: the dense,
bandwidth-heavy class with zero exchange is faster at P=2 on every host, 5/5 pairs everywhere, from
−4% on the 8-core-per-socket workstation to −18% on Genoa at 96 threads — the gain grows with cores
per socket, as a memory-system effect should. Every cell that exports rows is 2–8× slower at P=2,
and the penalty grows with core count: the exchange (export pass + copy + the receiver's larger rest
stream) costs more than the layer it feeds. `rotation_remote` at 3 ms/layer ships ~1e6 rows/layer.

## Table B — where the time goes in the exchange-free rotation layer (per layer, from the sidecars)

| host, threads | P=1 wall | P=2 wall | P=2 collective (`barrier_ns`) | coset loop P=1 → P=2 |
|---|---|---|---|---|
| ccqlin038, 16 | 4.24 ms | 3.33 ms | 0.11 ms | 4.15 → 3.25 ms (**−22%**) |
| Genoa, 48 | 1.10 ms | 0.97 ms | 0.04 ms | 1.02 → 0.88 ms (**−14%**) |
| Icelake, 32 | 1.62 ms | 2.14 ms | 0.30 ms | 1.54 → 1.77 ms (+15%) |

The NUMA effect on the layer work is real on two of three hosts. The `barrier_ns` column is the
engine's `collective_ns`, stamped from the top of the layer through the bucket-bits all-reduce — so
it is almost entirely **arrival skew** between the two partitions (the faster one waiting for the
slower), not transport: it scales with the layer (80 µs on a 3.1 ms layer, 3.4 µs on a 43 µs layer)
while the all-reduce mechanism is fixed. Rewriting the in-process collectives on spin-waiting atomics
(commit `4dda4ad`) took the mechanism from 2.9 µs to 0.25 µs per call and left this column unchanged,
as it should. Follow-up: split `collective_ns` into a publish lap and a wait lap so imbalance and
transport are separate columns. Icelake's slower coset loop at P=2 (+15%) is unexplained — no counters
on the node; 16 threads per socket there vs 8 here — and is not on the critical path now that the
priority is multi-node capacity.

## Table C — counters on ccqlin038, dense layers at 16 threads (8 per socket at P=2)

| cell | IPC | LLC load-miss | cycles/string | S0 read / write GB/s (% of 39.0 / 18.6) | S1 read / write | total GB/s |
|---|---|---|---|---|---|---|
| `su4` P=1 | 1.13 | 76% | 7178 | 15.8 / 14.3 (40% / 77%) | 15.8 / 12.8 (41% / 69%) | 60.2 |
| `su4` P=2 | 1.03 | 89% | 8229 | 12.4 / 9.7 (32% / 52%) | 11.8 / 9.5 (30% / 51%) | 44.0 |
| `su4_local` P=1 | 1.09 | 75% | — | 16.4 / 13.8 (42% / 74%) | 16.4 / 13.4 (42% / 72%) | 60.7 |
| `su4_local` P=2 | 1.27 | 75% | — | 18.6 / 12.9 (48% / 70%) | 18.8 / 13.1 (48% / 70%) | 64.0 |

Reading: at P=1 the first-touch spread already balances the dense layer's traffic over both memory
controllers, each at ~72–77% of a single socket's write ceiling. Partitioning keeps that split (RSS
1207 vs 1129 MB per node under `numastat`) and raises IPC by 16%, but the wall gain is 3.5–4%: on
this box the dense class is not limited by DRAM write bandwidth once both controllers are in use, and
locality buys only the remote-latency share. The plan's `su4` acceptance (≤ −10%) is **not met**; the
consistent −4% is the honest size of the effect for dense layers here. `su4` with random rows at P=2
moves *less* DRAM traffic (44 vs 60 GB/s) while taking 2.3× longer: the exchange is extra work and
copies, not bandwidth — a design cost, fixable (pull model), not a hardware wall.

## Verdicts against the plan's acceptance list

- `su4` P=2 direction-consistent, bound ≈ 1.3×: **fails** (2.2–2.3× slower with random rows; −4% when
  exchange-free). The bound assumed write-bandwidth-bound P=1 with single-socket traffic; P=1 already
  uses both sockets' controllers.
- `rotation_local`/`gu2q` expected null ±5%: rotation is a **gain** on ccqlin038/Genoa (−12 to −22% on
  the coset loop) and a loss on Icelake; `gu2q` (fanout 3.65, half the deltas remote) is a large loss.
- `rotation_remote − rotation_local` = exchange cost: **3–8× the layer**, `bytes ≈ rows × 48` confirmed.
- Per-socket write ≥ 70% of ceiling at P=2: 70% on `su4_local`, 51% on exporting `su4`.
- `partition_imbalance ≤ 1.05`: 1.000–1.004 everywhere.

## What follows from this

1. **Locality is the whole game.** A layer with any remote delta costs 2–8× its local time under the
   push exchange; a fully local layer gains 4–22%. Phase 5 (partition rows as cuts of the gate graph)
   is therefore the lever for real circuits, and the `su4_local`/`rotation_local` cells are its
   upper bound on this hardware.
2. ~~Remove the collective's wake-up latency~~ — done (`4dda4ad`); the column was arrival skew, see
   Table B. Remaining: instrument skew separately from the mechanism.
3. **Pull-based in-process exchange** (receiver reads the partner's input buckets directly, one
   interconnect crossing, no export pass, no copy) to bound the residual remote-layer cost;
   MPI keeps push. Under the success criterion this is a bound, not a target.
4. The untouched path (`--partitions 1`) shows no regression on a quiet node (Slurm 7004973: all four
   cells within ±1.3%, mixed sign); the ±4% seen on the workstation is placement/layout noise.
