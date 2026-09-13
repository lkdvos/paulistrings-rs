# Campaign contract — quera-talk-data/campaign-2026-09-11

Frozen scientific task, hardware class, timing semantics, and constraints for this campaign.
This is an index into `tasks/T01-variants.json` and `tasks/T02-canonical-task.md`, not a restatement of them.
Independent of the prior `presentation` branch campaign per explicit user decision (2026-09-11) — no reuse of its data/figures.
I cannot execute `sbatch`/`scancel`/`scontrol update` under org policy; I prepare and validate every job script, the user submits.

## Canonical workload (source: `tasks/T02-canonical-task.md`)

127-qubit heavy-hex kicked Ising, ibm_sherbrooke (FakeSherbrooke) lattice, 144 edges.
Layer order x-then-zz per Trotter step; `theta_zz = -pi/2` exactly; `theta_h = 7pi/32` primary, `5pi/16` secondary.
Depth: 20 Trotter steps headline, 9-step rung as the one point with a published-anchor cross-check (Begušić/Gray/Chan `exact.csv` col `5a`; no `5b` reference exists at 20 steps).
Observable `Z_62` (Eagle numbering); initial state `z+ = |0...0>^127`; direction `heisenberg` (backward/adjoint).
Accuracy bar: 0.01 absolute against the self-converged 20-step reference (no exact published value at 20 steps).
Julia comparison leg pinned by `git-tree-sha1 fe2bc2552caf975532a8b1372bd8bde1e1cd3f3f` (PauliPropagation.jl), not the unverifiable "0.8.2" label.

## Variant registry (source: `tasks/T01-variants.json`, hash-corrected 2026-09-11)

| Label | Commit | Role |
| --- | --- | --- |
| naive_baseline | `d410f4e5985ad917146867be31511143fde8f893` | unbucketed serial sort-merge — E0 internal baseline |
| direct_small_sum_path | `e56f021e54f3f64c3ddb8e2f688c39d91433d721` | small-m direct-apply path, orthogonal axis |
| bucketed_engine_serial | `f08db7df8bcd771f25383db0120e111cfb018bd2` | v0.2 B.5, GF(2) bucketing, serial — E4 isolation point |
| bucketed_engine_parallel | `ef037012e645d4f63013f26eaf3dbd6ce6299660` | v0.2 C.1-C.3, Rayon parallel — E5 isolation point |
| presentation_bench_crate_variants | `81b922c5a57e971a97f1630b74bce85f8fc8750c` | isolated naive/threadmaps/mergesort/bucketed with agreement tests — E1/E2 |
| partitioned_numa_engine | `a07cc5073fd53fa572f48616b400bc5134fce2d5` | PR #6 merge, NUMA + MPI — E6/E7/E8 |
| jcc_erratum_and_branch_prediction | `df28850f73a0331cd9375ee4258e4343fce402e6` | PR #7, front-end campaign — E1 kernel-improvement candidate |
| attempted_rejected_variants | git history + `research/FINDINGS.md` only | narrative only, not separately benchmarked |

Three of the seven SHAs from the T01 recon subagent (haiku, low effort) were fabricated/garbled and have been corrected in place in `T01-variants.json` against `git log -1 --format='%H %s' <sha>`; see that file's `metadata.verification_note`.
Any narrative numbers quoted in T01/commit messages (e.g. "7.93x", "-14.68%") are historical claims for labeling only — E1/E2/E4/E5 require fresh measurement on the frozen hardware class, never reuse of those quoted numbers.

## Hardware contract (mandatory, single architecture, physical cores only)

Frozen class: **`genoa`** on partition `ccq` — `worker7xxx`, 2 sockets x 48 cores = 96 physical cores/node, 1.5 TB RAM, `ib-genoa` interconnect.
Rationale: most single-node memory headroom before the distributed extension is needed (E7), and a large-enough core count for the thread-scaling ladder (C3); `icelake` (64 cores/node) and `rome` (128 cores/node, 1 TB) are alternates if `genoa` availability blocks a cell — switching requires a fresh baseline per the mandatory-hardware-contract rule, not a mid-curve swap.
Slurm reports `ThreadsPerCore=1` for all three classes (no SMT exposed to the scheduler) — still must be verified with `lscpu` on an actual allocation before the first timed run, not trusted from this login-node-side check.
A `rocky8` reservation covers part of the fleet (`worker[...,7001-7072,7074-7144,...]`) and excludes account `-temp`; observed not to block normal submission, but re-check `scontrol show reservation` if a genoa job queues unexpectedly.

## Timing semantics

Per-gate: start before prepare/rebucketing, stop after merge, cutoff filtering, global finalization (if selected), and required completion sync. Monotonic wall clock, integer ns, in-memory buffer, written outside the timed region.
Throughput: `R_j = N_j / t_j` per gate; aggregate/binned throughput is `sum(N_j)/sum(t_j)`, never an unweighted mean of rates.
Distributed: no diagnostic global barrier/all-reduce per gate; combine traces post-run; max rank-local gate duration is an explicitly labeled critical-rank proxy, not synchronized global time; min/median/max rank durations retained to expose skew.

## Schema version

`runs.jsonl` / `gates.rank-N.jsonl` schema v1 — see handoff §"Data contract" for the authoritative field list until a versioned schema file is written here (T06).

## Status

Phase A (workload + variant freeze): **done** 2026-09-11 (T01, T02 complete and hash-verified).
T03 (cluster/toolchain survey): partial — module/topology facts gathered above; toolchain build-from-source check (maturin/cargo/mpi module combo actually builds) not yet run.
T04-T10: not started. See `tasks.json`.
