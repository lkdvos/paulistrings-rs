# Evidence mapping — campaign-2026-09-11

Status as of 2026-09-11. **No cluster job has been submitted.** Every E0–E9 row below is either
*tooling-ready* (the measurement plumbing exists and is tested) or *blocked* (a genuine prerequisite is
missing). None is populated with real numbers, and none may be until a real Slurm allocation runs.

| ID | Evidence | Slide use | Status | Blocking on |
| --- | --- | --- | --- | --- |
| E0 | Baseline + external libraries | 10, every recurrence | tooling-ready | `jobs/campaign-genoa.sbatch` (bucketed_current only); Julia leg verified working (`decisions.md` #10) |
| E1 | Actual kernel improvement | 11 | blocked | Historical-revision worktree-checkout build machinery not implemented (`tasks.json#T07`); variant identified in `tasks/T01-variants.json` (`jcc_erratum_and_branch_prediction`) |
| E2 | Attempted threading approach | 12 | blocked | Same as E1; variant `presentation_bench_crate_variants` |
| E3 | Memory diagnosis | 13-14 | not started | No task yet drives `crates/membench`/`scripts/bandwidth.sh` for this campaign's host |
| E4 | Bucketed single-thread gain | 25 | tooling-ready | `campaign-genoa.sbatch` C2 stage (threads=1 cells) |
| E5 | Bucketed multithread gain | 26-27 | tooling-ready | `campaign-genoa.sbatch` C3 stage (thread ladder) — needs C2's peak_terms first to pick fixed cutoffs |
| E6 | Multiprocess/distributed behavior | 28 | tooling-ready | `jobs/campaign-genoa-distributed.sbatch` + `jobs/run_cell_distributed.py`, tested at 1/2/4 ranks (2026-09-13); not yet run on the real cluster |
| E7 | Beyond-single-node capacity | 28 | blocked | Same driver as E6 works, but no lower-tolerance point that actually exceeds single-node capacity has been attempted yet |
| E8 | Communication-aware hash | 28/28b | blocked | `partition_row_policy` field missing from the run-record schema (`decisions.md` #13); also needs E6's template |
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

- `hash_communication` / E8: schema needs a `partition_row_policy` field before this table can ever be non-empty.
- `jobs/preflight.py`'s CPU fingerprint is MEDIUM confidence and cannot distinguish Genoa from Bergamo;
  confirm against a real genoa allocation's `/proc/cpuinfo` before trusting `hardware_valid=true` there.
- `setup_time_s`/`scatter_time_s`/`gather_time_s` and per-gate `support_weight`/`bucket_bits`/`partner_count`
  are always `null` from the current driver — not fabricated, but not available either.
- E1/E2/E3/E6/E7/E8 need additional implementation (worktree builds, membench wiring, multi-node template,
  schema field) beyond what T01-T09 built; each is logged as `blocked` above with its specific prerequisite,
  not silently absent.
