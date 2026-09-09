# Partition-row tuning for the partitioned engine (phase 5)

Parent plan: the partitioned-engine plan (phases 1–4 landed; see
`research/notes/2026-09-08-partitioned-phase1-log.md` and
`research/notes/2026-09-08-numa-partitioning-results.md`). Priority (user, 2026-09-08): multi-node
capacity for very large sums; Pauli-rotation circuits (Trotter, kicked Ising) are the primary workload.

## Why

A layer whose deltas all have zero partition bits touches no transport and costs 1×; a remote rotation
layer costs 3.5× (shared memory) to 4.7× (InfiniBand) at 6e6 terms per rank and is transfer-bound. With
random partition rows roughly half of all rotation generators are remote. The rows are a free
parameter: `part(v) = R·v` for any `p` rows `R` over GF(2)^{2n}. A generator `g` (its key mask) is local
iff `R·g = 0`. So the row choice is a **max-weighted XOR-SAT** problem: pick `R` making `R·g = 0` for as
much circuit weight (layer count × rows exported) as possible, subject to **balance** — a row with
`R·g = 0` for *every* generator is a conserved quantity of the dynamics and puts all terms in one
partition. Balance must be measured, not assumed: terms migrate between partitions only through remote
layers, so few remote layers also means slow mixing.

For 1- and 2-local Pauli generators this is a graph cut: a row with no x-bits and z-bits equal to the
indicator of one side of a qubit cut makes every single-qubit X rotation local and a ZZ(i, j) rotation
remote iff the edge (i, j) crosses the cut; `p` rows are `p` cuts (labels of 2^p blocks). Hypothesis to
test: on a chain and on heavy-hex, cut rows reduce the remote-layer fraction from ~1/2 to
(cut edges)/(edges) — a handful of layers per Trotter step — at acceptable imbalance.

## Deliverables

**A. Row construction and selection (core crate).**
- `PartitionRows::cut(num_qubits, blocks: &[Vec<u32>])` — `log2(blocks)` z-only rows labelling the
  blocks (block `b` gets partition label `b`); `PartitionRows::from_rows` stays the general hook.
- `partition_rows::select(circuit, p, weights) -> PartitionRows` — greedy weighted XOR-SAT over the
  circuit's prepared generator masks: order generators by weight (layer count; optionally × expected
  exported rows), add the constraint `r·g = 0` while the system stays consistent *and* keeps a nonzero
  solution space of dimension ≥ p; pick `p` independent rows from the solution space, preferring rows
  that are **not** conserved (some generator has `r·g = 1`) and that are balanced on a probe set of
  terms (the initial sum after a few layers, if provided). Deterministic. Report which generators end
  up remote.
- `layer_locality(circuit, rows, hash, direction) -> Vec<bool>` (exists as `count_remote_deltas`; wrap).
- Tests: hand cases (chain of 4 with cut {0,1}|{2,3}: X's local, ZZ(1,2) remote only; heavy-hex toy);
  `select` reproduces the cut on a chain; conserved-row rejection; balance of a cut row on a random
  low-weight sum; property: `rows.partition_of(mask(g)) == 0` for every generator the selector reports
  local.

**B. Workloads and instrumentation (probe).**
- Probe layers: `tfim_step` (1D chain on `--qubits`: X on every qubit + ZZ on every edge, one Trotter
  step; repeated `--reps` times), `heavyhex_step` (the 127-qubit heavy-hex kicked-Ising step used in
  `presentation/bench` — copy the edge list into `test_support`), both with angles as in the
  presentation workload; `su4_brickwork` stays secondary.
- `--partition-rows random|cut|select` (default `random` = today); `cut` bisects the qubit range /
  heavy-hex graph into `P` contiguous blocks; `select` runs the selector on the cell's circuit.
- Per-cell outputs already exist (`local_layers`, `remote_layers`, `rows_exported`, `partition_terms_in`,
  `partition_imbalance`); add per-layer imbalance to the JSON (`partition_imbalance_by_layer`) so mixing
  dynamics are visible.

**C. Experiments.**
1. In-process, P=2 and P=4 on ccqlin038 (no cluster needed for the volume metrics): for each workload
   × rows ∈ {random, cut, select}: remote-layer fraction, rows exported per step, imbalance per layer
   over 10 steps starting from a single-site Z observable and from a random low-weight sum, wall per
   step (paired A/B, `--probe-b`).
2. MPI weak scaling with `cut` rows, 2/4/8 ranks on Icelake (`scripts/slurm/mpi-ranks.sbatch` with
   `LAYERS=tfim_step,heavyhex_step` and the rows flag), against the random-rows numbers in the results
   note.
3. Write `research/notes/2026-09-XX-partition-row-tuning-results.md`: tables, the imbalance dynamics,
   and a recommendation for the default row policy (keep random unless the circuit is known? or
   `select` by default when a `Circuit` is available at scatter time).

## Acceptance

Cut/selected rows on the chain and heavy-hex step: remote-layer fraction ≤ (cut edges)/(edges) + X
layers local; rows exported per step down ≥ 5× vs random; imbalance settles ≤ 1.2 within a few steps
from a spread observable (report the single-site case as is); per-step wall at P=2 in-process within
1.3× of the P=1 step (i.e. the NUMA gain is no longer eaten by exchange). Anything else is a negative
result to record.
