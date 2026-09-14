# Evidence mapping — campaign-2026-09-11

Status as of 2026-09-11. **No cluster job has been submitted.** Every E0–E9 row below is either
*tooling-ready* (the measurement plumbing exists and is tested) or *blocked* (a genuine prerequisite is
missing). None is populated with real numbers, and none may be until a real Slurm allocation runs.

| ID | Evidence | Slide use | Status | Blocking on |
| --- | --- | --- | --- | --- |
| E0 | Baseline + external libraries | 10, every recurrence | tooling-ready | Rust leg: `jobs/campaign-genoa.sbatch` (bucketed_current only). Julia leg: `jobs/run_cell_julia.py` + `jobs/campaign-genoa-julia.sbatch`, real end-to-end local proof at small scale 2026-09-13 (`decisions.md` #20); real 20-step canonical-depth run not yet submitted |
| E1 | Actual kernel improvement | 11 | blocked | Variant `jcc_erratum_and_branch_prediction` not attempted this pass -- time-boxed to the four variants named in the historical-variants task instead (see the new row below) |
| E2 | Attempted threading approach | 12 | blocked | Variant `presentation_bench_crate_variants` is a standalone `presentation/bench/` crate removed from the current tree, own JSON output shape, confirmed out of scope, not attempted |
| E4/E5 historical baseline | Bucketed single/multithread gain vs. the pre-bucketing/pre-parallel historical commits | 25-27 | **real data (toy scale)** | `jobs/run_cell_historical.py` + `jobs/campaign-genoa-historical.sbatch`, real genoa allocation (job 7033716, after fixing a real `ModuleNotFoundError` the first cluster attempt exposed -- `decisions.md` #31): all four historical commits from `tasks/T01-variants.json` (`naive_baseline`, `direct_small_sum_path`, `bucketed_engine_serial`, `bucketed_engine_parallel`) built via real worktree checkouts + real `cargo`/`maturin` builds and run at a reduced scale (n=12, 2 Trotter steps, eps=1e-6, chosen so the pre-bucketing engine survives), each `status=completed` and passing `analysis/schema.py::validate_run` (5 runs, 46 gate records, 0 problems). Real genoa wall times: naive_baseline 2.96e-5s/14 terms, direct_small_sum_path 1.14e-3s/10 terms, bucketed_engine_serial 7.82e-3s/10 terms, bucketed_engine_parallel 4.99e-3s/10 terms, bucketed_current (same scale) 7.36e-4s/10 terms. Figure: `figures/real/recurring_stage6.png` (stages 1-6, `naive_baseline` through `bucketed_engine_parallel`; the efficiency/left panel is honestly empty -- none of the four historical strategies exposes a per-gate stats object). `naive_baseline`'s higher term count (14 vs. 10) is a disclosed confounder (different ZZ-gate decomposition, no `rx`/`rz`/`cnot` sugar at that commit), not a bug. `direct_small_sum_path` only exercises its default (sorted) engine, not its distinguishing direct-apply path (no `engine=` kwarg at that commit; backport ruled out of scope). At this toy scale the numbers reflect fixed overhead, not real algorithmic cost -- they demonstrate all four variants build and run for real on the frozen hardware class, not yet a real speedup ordering claim; that needs a re-run closer to the full 127-qubit scale, which the weakest (unbucketed) variant may not survive |
| E3 | Memory diagnosis | 13-14 | not started | No task yet drives `crates/membench`/`scripts/bandwidth.sh` for this campaign's host |
| E4 | Bucketed single-thread gain | 25 | tooling-ready | `campaign-genoa.sbatch` C2 stage (threads=1 cells) |
| E5 | Bucketed multithread gain | 26-27 | tooling-ready | `campaign-genoa.sbatch` C3 stage (thread ladder) — needs C2's peak_terms first to pick fixed cutoffs |
| E6 | Multiprocess/distributed behavior | 28 | **real data** | `raw/2026-09-13-distributed-4ranks`: eps=2⁻¹⁸, 4 ranks/2 nodes, `wall_time_s=468.77`, `final_terms=583,393,599` — matches the single-node eps=2⁻¹⁸/threads=96 final term count exactly, a real overlap-consistency check between the in-process and distributed engines |
| E7 | Beyond-single-node capacity | 28 | **real data** | `raw/2026-09-13-distributed-16ranks`: eps=2⁻²⁰, 16 ranks/8 nodes, `wall_time_s=1658.8`, `peak_terms=8,923,556,570`, `peak_rss_kb≈3.04 TB` summed across ranks — **double** a single genoa node's 1.5 TB, a genuine beyond-single-node-capacity result (`decisions.md` #22-23). First attempt at this eps (8 ranks/4 nodes, job 7032060) OOM-crashed: the random partition-row draw put one rank at ~1.20 TiB RSS (343% above the 8-rank average per `seff`); doubling to 16 ranks spread the same random draw thin enough to complete cleanly |
| E8 | Communication-aware hash | 28/28b | **real data** (point) + **prepared** (cutoff sweep) | `raw/2026-09-14-worker7173-e8`: eps=2⁻¹⁶, `partitions=2`, real genoa allocation (`hardware_valid=true`), 127-qubit canonical task, 20 Trotter steps. `random` exported 4,519,274,260 rows / 217,248,351,640 bytes and took `wall_time_s=91.87`; `cut` (BFS locality heuristic) exported 408,053,676 rows / 19,601,061,744 bytes and took `wall_time_s=39.57` — a **~91% export-volume reduction and ~2.3x wall-clock speedup** for the same cell, same correctness (`final_terms`/`peak_terms` identical between policies). Figure: `figures/real/hash_communication.png`. Per user request 2026-09-14 (same rationale as E9's convergence sweep, `decisions.md` #30), `campaign-genoa-e8.sbatch` now sweeps the same 4-point cutoff grid {2⁻¹²,2⁻¹⁴,2⁻¹⁶,2⁻¹⁸} instead of one point, and `normalize.hash_communication()`/new `make_hash_communication_vs_cutoff_figure` plot export volume vs. cutoff — does the cut policy's advantage hold, grow, or shrink as the cutoff tightens? Not yet run at the extended grid. The distributed (`comm=`) path still only supports `"random"` |
| E9 | Observable consistency/convergence | 24 | **real data** (point) + **prepared** (trajectory) | Single-point headline claim achieved: full 20-step canonical depth, eps=2⁻¹⁶, `paulistrings` (`raw/2026-09-14-worker7169`) vs PauliPropagation.jl (`raw/2026-09-14-worker7169-julia`) agree to `abs_delta=1.67e-16` — floating-point noise, not a real discrepancy (`decisions.md` #27). Figure: `figures/real/accuracy.png`. Per user request 2026-09-14, a single point is a weak plot; `jobs/run_convergence_sweep.py` + `jobs/campaign-genoa-convergence.sbatch` (`decisions.md` #29) instead sweep `<O>` at every Trotter step for `min_abs_coeff` in {2⁻¹², 2⁻¹⁴, 2⁻¹⁶, 2⁻¹⁸} via `Circuit` slicing (no engine change), giving one real trajectory line per cutoff. `figures/make_compact_figures.py::make_convergence_figure` renders it. Not yet run on the real cluster |

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
- E1/E2/E3 need additional implementation (jcc_erratum/presentation_bench worktree builds, membench wiring)
  beyond what this pass and T01-T09 built; each is logged as `blocked` above with its specific prerequisite,
  not silently absent.
- The E4/E5 historical-baseline row above is real but only at a toy reduced scale on a non-genoa workstation --
  `jobs/campaign-genoa-historical.sbatch` (not yet submitted) is what turns it into cluster-scale, hardware-valid
  data. `direct_small_sum_path` never exercises its distinguishing direct-apply engine path (no `engine=` kwarg
  at that commit); `naive_baseline`'s term counts are not directly comparable to the other three (different ZZ
  gate decomposition, a disclosed confounder, not a bug -- see the E4/E5 row above and decisions.md).
- E0's Julia leg (`jobs/run_cell_julia.py`) never emits a per-gate trace: `trace_enabled` is always `False`
  and no `gates.rank-N.jsonl` records are written for it, because PauliPropagation.jl has no per-gate
  wall-time instrumentation (decision #10) and `validate_gate` requires `nanos` as a real, non-null int —
  there is no honest per-gate record this leg could produce. Real per-layer term counts are captured instead,
  in the run record's `extra.per_layer_terms`, since gate records have no home for them without `nanos`.
- The real 20-step canonical-depth Julia run (`jobs/campaign-genoa-julia.sbatch`) has not been submitted;
  only a small local proof (8 qubits, 1 Trotter step) has actually executed end-to-end and validated clean.
