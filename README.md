# campaign-2026-09-11

Benchmark-data campaign for the QuEra technical talk, driven by `quera-benchmark-agent-handoff.md`.
Independent of the `presentation` branch's prior campaign (user decision, `decisions.md` #1): no reuse of its data or figures.

## Workload

127-qubit heavy-hex kicked Ising, `ibm_sherbrooke` lattice (144 edges), layer order x-then-zz.
`theta_zz = -pi/2` exactly; `theta_h = 7pi/32` primary (`5pi/16` secondary); 20 Trotter steps headline, 9-step published-anchor rung.
Observable `Z_62`, initial state `z+`, direction `heisenberg`.
Full citations and the unverifiable-claims list: `tasks/T02-canonical-task.md`.

## Hardware

Frozen to the **genoa** class on Slurm partition `ccq` (2x48 physical cores, 1.5 TB RAM) — see `contract.md#hardware-contract` for the rationale and the fallback classes.
Every timed run must pass `jobs/preflight.py` first; a failed preflight writes a `status="invalid_hardware"` record rather than a timing.

## Layout

```
contract.md       frozen task/hardware/timing-semantics/schema decisions
campaign.json     planned matrix, resource bounds, stage status (nothing submitted yet)
tasks.json        T01-T10 status and per-task outputs
decisions.md       numbered decision log with rationale
job-ledger.jsonl   one line per Slurm submission (empty: none yet)
tasks/            T01 variant registry, T02 canonical-task citation
agent-results/     compact per-task subagent reports
analysis/          schema validation + trace normalization + normalized-table builders (T06)
jobs/              hardware preflight, single-cell run driver, Slurm template (T07)
figures/           recurring two-panel figure + compact figures (T09); figures/_synth/ is
                   gitignored synthetic-preview output ONLY, never real data
raw/               real runs.jsonl / gates.rank-N.jsonl land here once a job completes
tables/            normalized CSVs, built from raw/ once it exists
evidence.md        E0-E9 mapping, headline table (currently: everything blocked on real data)
reproduce.sh       explicit prepare/test/preflight/submit-print/collect/tables/figures commands
```

## Metrics

Per-gate: `terms_in`, complete-gate elapsed nanoseconds (`GateTrace`/`PartitionStats.nanos` — see
`crates/paulistrings/src/engine/bucketed.rs::GateTrace` and `ARCHITECTURE.md §Engine`), gate identity
(`circuit_index`, `application_index`, `gate_name`). Aggregate throughput is `sum(terms_in)/sum(nanos)`,
never a mean of per-gate rates (`analysis/normalize.py::efficiency_binned`).
Whole-run timing (`wall_time_s`) comes from a separate, untraced `propagate` call so tracing overhead never
pollutes the reported number (`jobs/run_cell.py`).

## Commands

```bash
./reproduce.sh test-tooling     # 42 tests, no cluster access needed
./reproduce.sh preflight        # hardware report on the current host
./reproduce.sh submit-c1c2      # prints (never runs) the sbatch command
./reproduce.sh collect          # validates raw/ once a job has produced it
```

See `reproduce.sh`'s own `usage()` for the rest, and `evidence.md` for what each command is blocking on.
