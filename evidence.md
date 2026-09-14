# Evidence mapping — campaign-2026-09-11

Status as of 2026-09-11. **No cluster job has been submitted.** Every E0–E9 row below is either
*tooling-ready* (the measurement plumbing exists and is tested) or *blocked* (a genuine prerequisite is
missing). None is populated with real numbers, and none may be until a real Slurm allocation runs.

| ID | Evidence | Slide use | Status | Blocking on |
| --- | --- | --- | --- | --- |
| E0 | Baseline + external libraries | 10, every recurrence | tooling-ready | Rust leg: `jobs/campaign-genoa.sbatch` (bucketed_current only). Julia leg: `jobs/run_cell_julia.py` + `jobs/campaign-genoa-julia.sbatch`, real end-to-end local proof at small scale 2026-09-13 (`decisions.md` #20); real 20-step canonical-depth run not yet submitted |
| E1 | Actual kernel improvement | 11 | blocked | Historical-revision worktree-checkout build machinery not implemented (`tasks.json#T07`); variant identified in `tasks/T01-variants.json` (`jcc_erratum_and_branch_prediction`) |
| E2 | Attempted threading approach | 12 | blocked | Same as E1; variant `presentation_bench_crate_variants` |
| E3 | Memory diagnosis | 13-14 | not started | No task yet drives `crates/membench`/`scripts/bandwidth.sh` for this campaign's host |
| E4 | Bucketed single-thread gain | 25 | tooling-ready | `campaign-genoa.sbatch` C2 stage (threads=1 cells) |
| E5 | Bucketed multithread gain | 26-27 | tooling-ready | `campaign-genoa.sbatch` C3 stage (thread ladder) — needs C2's peak_terms first to pick fixed cutoffs |
| E6 | Multiprocess/distributed behavior | 28 | **real data** | `raw/2026-09-13-distributed-4ranks`: eps=2⁻¹⁸, 4 ranks/2 nodes, `wall_time_s=468.77`, `final_terms=583,393,599` — matches the single-node eps=2⁻¹⁸/threads=96 final term count exactly, a real overlap-consistency check between the in-process and distributed engines |
| E7 | Beyond-single-node capacity | 28 | **real data** | `raw/2026-09-13-distributed-16ranks`: eps=2⁻²⁰, 16 ranks/8 nodes, `wall_time_s=1658.8`, `peak_terms=8,923,556,570`, `peak_rss_kb≈3.04 TB` summed across ranks — **double** a single genoa node's 1.5 TB, a genuine beyond-single-node-capacity result (`decisions.md` #22-23). First attempt at this eps (8 ranks/4 nodes, job 7032060) OOM-crashed: the random partition-row draw put one rank at ~1.20 TiB RSS (343% above the 8-rank average per `seff`); doubling to 16 ranks spread the same random draw thin enough to complete cleanly |
| E8 | Communication-aware hash | 28/28b | **real data** | `raw/2026-09-14-worker7173-e8`: eps=2⁻¹⁶, `partitions=2`, real genoa allocation (`hardware_valid=true`), 127-qubit canonical task, 20 Trotter steps. `random` exported 4,519,274,260 rows / 217,248,351,640 bytes and took `wall_time_s=91.87`; `cut` (BFS locality heuristic) exported 408,053,676 rows / 19,601,061,744 bytes and took `wall_time_s=39.57` — a **~91% export-volume reduction and ~2.3x wall-clock speedup** for the same cell, same correctness (`final_terms`/`peak_terms` identical between policies). Figure: `figures/real/hash_communication.png`. The distributed (`comm=`) path still only supports `"random"` |
| E9 | Observable consistency/convergence | 24 | tooling-ready | Live pilot already ran (5 steps, θh=7π/32, ε=2⁻⁶): 1355/1355 per-layer term counts identical, `|Δ⟨O⟩|=0` between engines (`decisions.md` #10) — this is a plumbing validation at shallow depth, not the headline 20-step accuracy claim, which still needs the real campaign |

## Headline numbers

None yet. This table will carry one row per completed, hardware-valid run once `raw/` is populated;
see `analysis/normalize.py::runtime_tolerance` for the exact fields.

## What "tooling-ready" means precisely

The Rust engine change (`GateTrace`, T04), the Python binding surface (`PropagationStats`/`PartitionStats`
new fields), the schema/validator/normalizer (T06), the preflight+driver+Slurm template (T07), and the
figure-generation code (T09) are all implemented, tested (42 tests across `analysis/`, `jobs/`, `figures/`,
plus the core crate's 549 Rust tests + `mpi`-feature tests + 451 Python tests — see `decisions.md`), and
integration-checked against each other by the main agent. What remains is exactly one thing: a human
running `./reproduce.sh submit-c1c2`'s printed command on the real cluster.

## Known gaps carried into any real run

- `hash_communication` / E8: schema/driver/normalize/figure support is real now (`decisions.md` #21), validated
  with small local runs — not with a real cluster allocation. The distributed (`comm=`) path has no
  explicit-rows plumbing, so a genuine multi-node "cut" comparison isn't possible yet either; only the
  in-process partitioned engine (no MPI needed) supports both policies today.
- `jobs/preflight.py`'s CPU fingerprint is MEDIUM confidence and cannot distinguish Genoa from Bergamo;
  confirm against a real genoa allocation's `/proc/cpuinfo` before trusting `hardware_valid=true` there.
- `setup_time_s`/`scatter_time_s`/`gather_time_s` and per-gate `support_weight`/`bucket_bits`/`partner_count`
  are always `null` from the current driver — not fabricated, but not available either.
- E1/E2/E3/E6/E7/E8 need additional implementation (worktree builds, membench wiring, multi-node template,
  schema field) beyond what T01-T09 built; each is logged as `blocked` above with its specific prerequisite,
  not silently absent.
- E0's Julia leg (`jobs/run_cell_julia.py`) never emits a per-gate trace: `trace_enabled` is always `False`
  and no `gates.rank-N.jsonl` records are written for it, because PauliPropagation.jl has no per-gate
  wall-time instrumentation (decision #10) and `validate_gate` requires `nanos` as a real, non-null int —
  there is no honest per-gate record this leg could produce. Real per-layer term counts are captured instead,
  in the run record's `extra.per_layer_terms`, since gate records have no home for them without `nanos`.
- The real 20-step canonical-depth Julia run (`jobs/campaign-genoa-julia.sbatch`) has not been submitted;
  only a small local proof (8 qubits, 1 Trotter step) has actually executed end-to-end and validated clean.
