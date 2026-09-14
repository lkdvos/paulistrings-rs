# Decisions log — campaign-2026-09-11

1. **Independent fresh campaign, not a reuse of the `presentation` branch.** User-confirmed 2026-09-11 after main
   agent flagged that branch's existing data/figures/deck. No reuse of its numbers or figures.
2. **Cannot execute `sbatch`/`scancel`/`scontrol update`** regardless of the handoff's task-specific
   authorization — org policy overrides. All job scripts prepared and validated; submission commands handed to
   the user.
3. **Hardware class frozen to `genoa`** (ccq partition, 96 physical cores/node, 1.5 TB) over `rome`/`icelake`.
   See `contract.md#hardware-contract` for rationale; not yet confirmed by an actual allocation (SMT check,
   toolchain build) — that's the remainder of T03.
4. **T01's low-effort (haiku) recon fabricated 3 of 7 commit SHAs.** Repaired by the main agent via
   `git log -1 --format='%H %s' <sha>` verification against every claimed hash before freezing `contract.md`.
   Escalation trigger from the handoff's routing table ("failed meaningful acceptance check") — did not re-run
   the whole task on a stronger model, targeted repair was sufficient since the commit *messages* T01 gave were
   correct, only the hashes were wrong.
5. **T02's canonical task recommendation accepted as-is** — well-cited, flagged its own unverifiable claims
   (Kyiv vs Sherbrooke lattice identity, no exact 20-step reference, PauliPropagation.jl version string vs
   tree-hash). No repair needed.
6. **T04 (gate tracing) implemented directly by the main agent, not delegated** — the handoff's routing table
   flags this work as ownership/concurrency-sensitive and default-path-regression-prone, and the main agent
   already had the codebase context loaded from Phase A; delegating and then reviewing would have cost more
   context than doing it once.
7. **New `GateTrace` type, not an extension of the existing `TermTrace`.** `TermTrace` is documented as
   always-compiled and zero-extra-cost; adding timing fields to it would break that contract for every existing
   caller. `GateTrace` mirrors its opt-in state-machine shape but is a separate type, sharing one hoisted
   `Instant::now()` with the existing per-layer `DEBUG` log line so an untraced/non-debug layer's cost is
   unchanged. Same pattern extended into `PartitionLayerRecord`/`PartitionLayerRow` (new `circuit_index`,
   `application_index`, `gate_name`, `nanos` fields) rather than a parallel structure, since `run_layers` is
   shared by the in-process partitioned and MPI-distributed drivers — one edit covers both.
8. **No per-gate Trotter-step index stored in the engine.** `Circuit<W>` has no notion of step boundaries (it's
   a flat channel list), so a step index would be a fiction the engine invents. `circuit_index / channels_per_step`
   downstream (T06's normalization script, once `channels_per_step` is known from the task config) recovers it
   without adding step-awareness to the core crate. Documented on `GateTrace`.
10. **T05 required no Julia/wrapper code changes.** The existing `benchmarks/julia/runner.jl` gate vocabulary
   (`rx`, `pauli_rotation`, `cnot`, ...) already covers `heavy_hex_kicked_ising` exactly, `test_julia_parity.py`
   passes 32/32 on this host/revision, and a live pilot of the real canonical circuit (θ_h=7π/32, 5 steps,
   ε=2⁻⁶, `--parity-theta`/`--parity-steps` matched) through `bench_c_deep_trotter.py` reproduced 1355/1355
   identical per-layer term counts and `|Δ⟨O⟩|=0` between the two engines. T04's new `PropagationStats.nanos`/
   `gate_name`/`circuit_index` already flow through the existing `propagate_with_stats` call used by
   `_rust_leg` with no wrapper edit needed. Julia's per-gate wall-time gap is real, pre-existing (documented in
   `runner.jl`'s own output `notes`), and accepted rather than fabricated, per the handoff's explicit fallback
   rule for a library lacking equivalent instrumentation — the efficiency-panel per-gate series for
   PauliPropagation.jl stays marked missing; only its total-time reference is used.
11. **T06/T07 delegated to fresh subagents, both independently re-verified by the main agent** (re-ran their
   test suites, and for T07 also re-ran `preflight.py` live on this host to confirm it honestly reports
   `preflight_passed=false`/`node_class_guess="unknown"` on a Cascade Lake login node rather than guessing
   "genoa"). Consistent with the routing table's cheap/medium tiers for this work and the "don't repeat a
   completed subagent inspection without a concrete inconsistency" rule — a light re-run sufficed, no full
   re-review.
12. **RESOLVED 2026-09-13**: job 7030090 landed on `worker7277`, `lscpu` reported `AMD EPYC 9474F` (a real
   Genoa-generation part) and `preflight.py` independently matched it to the `(AuthenticAMD, family=25,
   model=17)` table entry, reporting `hardware_valid=true`/`node_class=genoa` — the fingerprint table is now
   empirically confirmed against a real allocation, not just secondary-source guesswork. (It still cannot
   distinguish Genoa from Bergamo by construction; no Bergamo allocation has been seen to test that edge.)
13. **`hash_communication()` (E8) is a documented placeholder**, not a silent gap: the run-record schema has no
   `partition_row_policy` field yet. Needs a schema addition (`"random"`/`"cut"`) before E8 data can be
   collected; deferred rather than bolted on speculatively.
15. **Integration bug caught while wiring T10's `reproduce.sh`**: T06 and T09 each independently added a
   `tests/__init__.py` to their own test directory, so running `analysis/tests` and `figures/tests` together
   under pytest collided on the top-level module name `tests` (`ModuleNotFoundError`). Fixed by removing both
   `__init__.py` files — every test file has a unique basename across the three directories, so pytest's
   basename-based rootdir insertion handles them without a shared package name. Re-verified: 42/42 pass
   together. This is exactly the kind of cross-task integration issue the main agent is responsible for
   catching (handoff: "the main agent integrates T04-T07... runs the required acceptance gates").
19. **Built the distributed (MPI) campaign path (C4/E6) on 2026-09-13**, at the user's request for
   distributed data ahead of the "by tomorrow" deadline. New: `jobs/run_cell_distributed.py` (a separate
   driver, not a branch inside `run_cell.py`, since MPI's startup order — `mpi4py.rc.thread_level` before
   importing `mpi4py.MPI` — and the "no rank-dependent branch around a collective" rule are load-bearing and
   easiest to keep correct in one focused file), using `propagate(..., comm=COMM, result="local")` (never
   gathers the full sum, per the handoff's capacity-demonstration rule) and a sum-of-local-expectations
   allreduce for a true (not per-rank-partial) observable value. Only rank 0 writes `runs.jsonl` (avoids
   concurrent writers on one shared file); every rank writes its own `gates.rank-<r>.jsonl`. Built and tested
   against a real `--features mpi` extension in a new `.venv-mpi` (the documented CLAUDE.md steps needed a
   module-name correction: `python-mpi/3.12.9` must be loaded via three separate `module load` calls, not one
   combined command, for Lmod to resolve the extension-provides relationship) at 1, 2, and 4 ranks, with
   `preflight.run_preflight` monkeypatched to a passing report (this login host is not genoa) — 6/6 new
   regression tests (`test_run_cell_distributed.py`) pass at every rank count, and `analysis/validate_campaign.py`
   accepts the real 4-rank output cleanly (0 problems).
   New Slurm template `jobs/campaign-genoa-distributed.sbatch` mirrors `scripts/slurm/mpi-ranks.sbatch`'s
   proven multi-node placement (`--cpu-bind=ldoms --mpi=pmix`, shared-filesystem build, ranks rounded to a
   power of two) but launches the real canonical task instead of the differential-test binary. Defaults to
   `min_abs_coeff=2^-16` deliberately — the same point already trusted from `campaign-genoa.sbatch`, so the
   first distributed cell is an apples-to-apples overlap comparison, not a blind leap to an untested tolerance.
   **Not yet run on the real cluster** — E6 is prepared, not evidenced; E7 (a run that exceeds single-node
   capacity) and E8 (hash-row comparison, still blocked on the missing `partition_row_policy` schema field)
   remain out of scope for this pass.
17. **First real job (7030090) revealed a genuine T06/T07 schema mismatch**, invisible to synthetic-fixture
   tests because both sides' fixtures independently matched their own (slightly wrong) assumptions:
   `run_cell.py` wrote `build_features`/`compiler_version` as `None` (never actually collected),
   `engine="paulistrings"` (not in the schema's closed enum), `partitions=None` for an unpartitioned run
   (schema wants `1`), `slurm_job_id` as an `int` (schema wants `str`), and omitted `trotter_step` from every
   gate record entirely (the field itself must be present, possibly null — my own T06/T07 dispatch prompts
   disagreed on this, an error in the handoff, not either subagent's). Fixed in `run_cell.py`: `rustc
   --version` now captured for real, `build_features=[]` (matches what the sbatch template actually passes to
   `maturin develop`), `engine`/`partitions` derived from `spec.partitions`, `slurm_job_id` stringified, and
   `trotter_step` computed as `circuit_index // (len(circuit) // spec.trotter_steps)` since the driver has
   both quantities in scope. The 6 already-collected real records (~50 min of real compute) were migrated in
   place rather than re-run — `analysis/validate_campaign.py` now passes clean (0 problems) on the real data.
18. **T10 (`campaign.json`/`README.md`/`evidence.md`/`reproduce.sh`) written by the main agent directly**,
   not delegated — small, mostly-documentation glue work, and it needed the accumulated context of every
   prior task's exact output paths and caveats, which a fresh agent would have had to re-read anyway.
   `evidence.md` marks every E0-E9 row honestly as `tooling-ready` or `blocked` with its precise prerequisite;
   no invented numbers anywhere, per the handoff's explicit prohibition.
14. **T08 (timing-semantics review) done by the main agent directly**, not delegated: it requires judgment
   across files already in this agent's context (T04's engine changes) plus the newly-delegated T06/T07 output,
   so a fresh subagent would cost more in context-transfer than it saved. No correctness issues found beyond
   the two gaps already logged (genoa fingerprint confidence, hash_communication placeholder).
20. **Built the E0 external-baseline campaign glue for real, on 2026-09-13**: `jobs/run_cell_julia.py`, given
    the same `cell.json` shape `run_cell.py` consumes, builds the circuit/observable through
    `run_cell._build_circuit`/`_build_observable` (real `paulistrings` objects, not a second construction
    path), converts them to task-JSON schema v1 via `common.oracles.as_circuit_spec`/`pauli_terms`, and drives
    `benchmarks/julia/runner.jl` through the existing `benchmarks/python/julia_baseline.py` wrapper
    (`make_task`/`run_task`) — no new task-JSON construction, per the handoff's explicit instruction. Preflight-
    gated exactly like `run_cell.py` (same hardware contract, same node in a real run). Writes one run record
    per cell to `runs.jsonl`, `engine="pauli_propagation_jl"`, `variant_id="external_pauli_propagation_jl"`
    (a cell's own `variant_id`, a Rust-commit-registry label, is meaningless for this leg and is ignored for
    that purpose — only `config_id` is preserved, so a Rust cell and its Julia counterpart join on it). Never
    writes a gate record: `validate_gate` requires `nanos` as a real non-null int and PauliPropagation.jl has
    no per-gate wall-time instrumentation (decision #10), so `trace_enabled` is always `False` and there is no
    honest per-gate record to produce. Real per-layer term counts (`PP_LAYER_COUNTS=1`, on by default) go into
    the run record's `extra.per_layer_terms` instead, since gate records have no home for them.

    **Real schema mismatches found and fixed** (`analysis/validate_campaign.py` run against real local output,
    same pattern as decision #17): (a) `schema.py`'s `_ENGINES` enum had no value for an external-library leg
    at all — `unpartitioned`/`partitioned`/`distributed` describe this engine's own run topology, not which
    software produced the run — fixed by adding `"pauli_propagation_jl"` as a fourth, distinct value rather
    than overloading `"unpartitioned"` onto a different axis of meaning. (b) `config_id` and `policy` were
    required non-null `str` in `validate_run`, but a cell that never gets far enough to build a policy object
    (`invalid_hardware`, or any pre-run failure) genuinely has neither — both are real optional fields
    (`CellSpec.config_id` already defaults to `None`), so both were changed to nullable. This also fixed a
    latent bug in `run_cell.py`'s own `invalid_hardware` path, discovered incidentally: it already emitted
    `config_id=None`/`policy=None` and was already failing `validate_run` on those two fields before this fix,
    just never caught because no test called `validate_run` on that path. (c) `validate_run` requires a
    `"wall_time_s_reason"` string whenever `wall_time_s` is null, on *any* status, but neither driver's
    `RUN_FIELDS`/`_empty_run_record` ever populated that key — fixed in both `run_cell.py` and
    `run_cell_julia.py` by setting `wall_time_s_reason` to the same `failure_reason` text on every
    non-completed record.

    **Real data generated and validated**: an 8-qubit, 1-Trotter-step smoke cell ran end-to-end through the
    real `julia`/PauliPropagation.jl installation on this host (juliaup, `julia 1.12.6`, already-instantiated
    `benchmarks/julia` project — `Manifest.toml` pins `PauliPropagation.jl 0.8.2`, tree-sha
    `fe2bc2552caf975532a8b1372bd8bde1e1cd3f3f`, matching `contract.md`'s pin) and its `runs.jsonl` record passed
    `analysis/validate_campaign.py` with 0 problems. `quera-talk-data/campaign-2026-09-11/jobs/tests/
    test_run_cell_julia.py` adds 4 tests (2 real `julia` subprocess invocations, skipped cleanly if `julia`
    is unavailable), including a cross-engine term-count check against `run_cell.run_cell` on the identical
    tiny circuit — the miniature version of decision #10's pilot, cheap enough to run on every test invocation.
    All 46 pre-existing tests under `quera-talk-data/campaign-2026-09-11/` still pass (no regression from the
    schema changes above).

    **Not done, and explicitly out of scope for this pass**: the real 20-step canonical-depth Julia run —
    `jobs/campaign-genoa-julia.sbatch` is prepared (mirrors `campaign-genoa.sbatch`'s worktree/venv/maturin
    preamble, since this leg still needs `paulistrings` importable to build the circuit, plus `module load
    julia` and `Pkg.instantiate()` on the shared filesystem) but **not submitted** — that is the user's step,
    per org policy (decision #2). E1/E2 (historical Rust commit variants) were not touched, per this task's
    explicit scope boundary.

21. **Closed the E8 gap for real, on 2026-09-13**: decision #13 deferred `hash_communication()` (random-vs-cut
    partition-row communication volume) on the missing `partition_row_policy` schema field. This pass adds it,
    end to end, and validates it with real local runs (not fabricated numbers) — the in-process partitioned
    engine, which needs no MPI, is the right tool: this is a placement/row-selection question, not a
    distributed-capacity one.

    **Rust/PyO3 (`crates/paulistrings-py/src/sum.rs`)**: `parse_partitions` used to hardcode
    `partition_row_seed: None` when building the core `PartitionConfig` — there was no way at all to choose a
    seed from Python. Fixed by threading a new `partition_row_seed: int | None = None` kwarg through
    `propagate`/`propagate_with_stats` into the existing Rust field (no new Rust algorithm; `PartitionRows::
    from_seed` already took a seed). A second, alternative kwarg `partition_row_blocks: list[list[int]] | None`
    gives an explicit "cut": one disjoint qubit block per partition, fed straight to
    `crates/paulistrings/src/bucket/hash.rs`'s **already-existing** `PartitionRows::cut(num_qubits, blocks)` —
    that constructor (Z-only rows labelling blocks, `partition_of_pauli` = XOR of the odd-z-weight blocks' labels)
    predates this task; no new row-construction algorithm was needed, only exposing it. Both kwargs are validated
    (row/block count vs. `partitions=`'s resolved count, qubit bounds, block disjointness) with the GIL held,
    *before* `allow_threads` releases it — a caller mistake is a `ValueError`, never a panic across the FFI
    boundary. Mutually exclusive with each other and, for now, with `comm=` (no `PartitionConfig`-equivalent
    plumbing exists on the MPI path yet; raises a clear `ValueError` naming the gap rather than silently
    dropping the knob). `partition_row_seed=None`/`partition_row_blocks=None` (both defaults) reproduce today's
    seeded-random path exactly — the performance-discipline rule that a new opt-in knob is a no-op at its
    default.

    **Rust tests**: `cargo test --workspace` (549 core-crate tests, unchanged) plus 3 new `paulistrings-py`
    unit tests (`sum::partition_row_knob_tests`) proving two different explicit block sets assign a term to
    different partitions, that a mismatched block count / an overlapping block is rejected before
    `PartitionRows::cut` would panic on the same condition, and a round-trip check that a named block's qubit
    lands in that partition. These test plain-Rust helper functions
    (`validate_partition_row_blocks_impl`/`build_partition_rows`), not `parse_partitions`/`parse_run_mode`
    themselves: this crate is `extension-module`-only (loaded *by* Python, never embedding it), so a
    `Python::with_gil` call inside `cargo test` fails to link (`PyErr_*`/`PyUnicode_*` undefined symbols) —
    the end-to-end Python-facing behavior is covered by 11 new tests in
    `python/paulistrings/tests/test_partitioned.py` instead (built via `maturin develop --release`, run via
    `pytest`). `cargo clippy --workspace --all-targets -- -D warnings` clean.

    **Schema (`analysis/schema.py`)**: added `partition_row_policy`, nullable `"random"`/`"cut"`, required null
    when `engine == "unpartitioned"` (there is no row policy without a partitioned/distributed run) and
    otherwise one of the two closed values. `analysis/tests/test_schema_and_normalize.py` gained 5 new
    validator tests plus 2 `hash_communication` tests.

    **Driver (`jobs/run_cell.py`)**: `CellSpec.partition_row_policy: str = "random"` (default preserves today's
    behavior when `partitions is None`, where the field is simply ignored and recorded `None`). When
    `partitions` is set, `"cut"` calls a new `_cut_blocks(spec, num_partitions)` helper — a breadth-first
    traversal of `circuits.heavy_hex_sublattice(n)`'s edges from qubit 0, sliced into `num_partitions` equal
    contiguous chunks of visit order. **This is a simple first-pass heuristic** (visiting physically adjacent
    qubits consecutively keeps each chunk local without solving an actual min-cut) — **not** the "open research"
    optimal row-tuning CLAUDE.md's Known Gaps section refers to, and it is not oversold as such anywhere in
    code or docs. An unknown `partition_row_policy` value raises `ValueError` rather than silently falling back
    to random. `jobs/run_cell_distributed.py` also gained the field (for `RUN_FIELDS` parity — it's a shared
    tuple) but rejects anything other than `"random"` with a clear collective error: its `comm=` path has no
    explicit-rows plumbing (see the Rust paragraph above), so writing a `"cut"`-labeled record without cut rows
    actually applied would be a lie. `jobs/campaign-genoa-distributed.sbatch` gained a `PARTITION_ROW_POLICY`
    env knob (default `random`, preserving today's behavior byte-for-byte) that fails the job immediately with
    a clear message if set to `cut`. `jobs/run_cell_julia.py` (the external-baseline leg, unrelated to
    partitioning) needed only `"partition_row_policy": None` added to its two record dicts to keep
    `RUN_FIELDS` parity, since it shares that tuple with `run_cell.py`.

    **`analysis/normalize.py::hash_communication`**: implemented for real. Takes `(run_records, gate_records)`,
    groups by `run_id` (not merged across runs, so same-`config_id` "random" vs "cut" runs stay comparable
    side by side), sums each run's `rows_exported`/`bytes_exported` across its own gate records — the exact
    metric `PartitionStats`/`PartitionLayerRow` already exposed (decision #7's `GateTrace`/`PartitionLayerRecord`
    work), no new metric invented. Raises `ValueError` (not a silent empty list) only when literally no run in
    the input carries a non-null `partition_row_policy` — the old "field doesn't exist" placeholder condition
    is gone since the field now exists; a genuinely input-less call still refuses to fabricate a plot, with a
    precise cause instead of a vague one.

    **`figures/make_compact_figures.py::make_hash_communication_figure`**: plots real rows now — one bar per
    `(config_id, partition_row_policy)` pair, `total_rows_exported` on the y-axis. Still raises
    `NotImplementedError` (not a blank/fabricated plot) on a genuinely empty row list, which can now only mean
    every tagged run had zero gate records (e.g. a zero-layer circuit), not "the schema lacks the field."

    **Real local validation** (this host is not genoa; same monkeyed-preflight precedent as decisions #17/#19/
    #20 — this is local plumbing validation, not a cluster measurement): ran two otherwise-identical cells
    through the real `jobs/run_cell.py` CLI path — 32 qubits, 4 Trotter steps, `partitions=2`, one with
    `partition_row_policy="random"`, one `"cut"` — both `status="completed"`, both records/gate-records passed
    `analysis/schema.py`'s validators and `analysis/validate_campaign.py` (0 problems) with 264 real gate
    records each. Real numbers: **random** exported 8975 rows / 294120 bytes total across the run; **cut**
    exported 100 rows / 3584 bytes — a ~98.9% reduction in row export volume for this cell, from the BFS cut
    alone. `analysis.normalize.hash_communication` on the real `runs.jsonl`/`gates.rank-0.jsonl` this produced
    returns exactly that comparison, and `figures.make_compact_figures.make_hash_communication_figure` plots it
    without error. This is real evidence that random and cut policies genuinely differ in export volume — the
    entire point of E8 — from a small local run, not a cluster-scale measurement (that's the sbatch command
    below, not yet run, per decision #2).

    **Not run**: the real cluster-scale, 127-qubit E8 comparison. `run_cell.py` supports
    `partitions=`/`partition_row_policy=` for real today, but neither `campaign-genoa.sbatch` (single-node,
    in-process partitioned) nor `campaign-genoa-distributed.sbatch` (multi-node MPI) exposes a real "cut" cell
    at cluster scale yet: `campaign-genoa.sbatch` hardcodes `"partitions": null` in its cell.json template (no
    env knob for it at all — out of this pass's scope, which only touched the distributed template per the
    task), and the distributed path itself has no explicit-rows plumbing (above). A genuine cluster-scale E8
    result needs either (a) a `campaign-genoa.sbatch` variant that sets `partitions=`/`partition_row_policy=`
    in its cell.json (straightforward — the driver and PyO3 support already exist — but not built this pass),
    or (b) the MPI-side explicit-rows plumbing. Neither was run; only prepared and reasoned about.

9. **Caught and fixed a real regression before it shipped**: the direct/small-sum path (`engine::direct`) has
   its own layer loop, separate from the sorted engine's, and initially had no gate-trace wiring at all — so
   `engine="auto"`/`"direct"` runs recorded only the sorted-suffix layers, truncating `PropagationStats.layers`
   from 76 to 7 on the existing `test_per_layer_term_counts_match_between_engines` pytest. Fixed by mirroring
   the same want_timer/record_gate_trace wiring into `direct.rs::run_direct_prefix`, and pinned with a new
   assertion block in `small_sum_path.rs::assert_engines_agree` so it can't silently regress again.

22. **Root-caused job 7032060's OOM crash (eps=2^-20, 8 ranks/4 nodes): genuine capacity shortfall
    compounded by random-partition-row imbalance, not a code bug.** Investigated per the E7 completion
    task: full re-read of `raw/slurm-7032060.out`, `sacct -j 7032060` broken out per step
    (`MaxRSS`/`AveRSS`/`ReqMem`), `seff 7032060`, `jobs/run_cell_distributed.py` end to end, and a diff
    of `jobs/campaign-genoa-distributed.sbatch` against the proven `scripts/slurm/mpi-ranks.sbatch`.

    **Timeline (real, from sacct/seff, not guessed)**: job ran 01:33:01 wall and Slurm marked it
    `COMPLETED` overall — the "srun launcher appears hung" note in `job-ledger.jsonl` was written before
    the job's final state settled and is superseded by this entry. Step `7032060.0` ran 01:31:55 before
    `OUT_OF_MEMORY`. `seff` reports 2.73 TB utilized of 5.87 TB requested (46.45%) but flags "the task
    which had the largest memory consumption differs by 343.84% from the average" — `sacct`'s per-step
    `MaxRSS=1257805740K` (~1.20 TiB, one single rank, task 1 on `worker7216`) vs `AveRSS=365811870K`
    (~349 GiB average across the 8 ranks). One rank alone consumed ~82% of its node's usable RAM
    (`ReqMem`/`seff` confirm the node got its full ~1.47 TiB, so the earlier "missing `--mem`" theory in
    the task brief is **refuted** — `--exclusive` with no `--mem` correctly grants the whole node, matching
    `mpi-ranks.sbatch`'s established, working behavior; no difference in UCX env vars between the two
    scripts either, and `/dev/shm` on this login host is 126G tmpfs, not a plausible independent culprit).

    **Backtrace re-read**: the SIGBUS-in-`MPI_Bcast`-under-`mpi4py` trace does **not** localize to the
    `run_id` bcast at `run_cell_distributed.py:128` as speculated in the task brief. mpi4py's pickle-based
    object collectives (`Comm.bcast`, `Comm.allreduce` on non-buffer Python objects) are implemented
    internally via reduce+broadcast regardless of which Python-level method is called, so every
    `COMM.allreduce(...)` call in the file (lines ~134, ~198, ~208, ~244-247) is an equally plausible site
    for a `MPI_Bcast` frame — the trace alone can't distinguish them. Given 93 real minutes elapsed and
    that `observable.propagate(..., comm=COMM, result="local")` (the ~90-minute Rust-side computation,
    line 190) uses rsmpi directly and would show Rust frames if it were the crash site (it shows none),
    the far more likely site is the very next Python-level collective after `propagate` returns —
    `COMM.allreduce(local_exp, op=MPI.SUM)` at line 198 — i.e., the crash landed exactly when the already-
    OOM-adjacent rank needed one more small allocation (pickling/MPI internal buffers) to do the
    post-computation reduction. This is consistent with, not contradictory to, the OOM read: the collective
    was the straw, not the load.

    **No collective-order bug found.** Re-verified the module's own invariant ("no rank-dependent branch
    around a collective"): every conditional (`spec.variant_id != "bucketed_current"`, `not group_ok`,
    `spec.partition_row_policy != "random"`) is evaluated from `spec` (parsed identically by every rank from
    one shared `cell.json`, written once by the sbatch script before any rank starts) or from `group_ok`
    (itself already collectively agreed via the `allreduce` immediately above it). This cell used
    `partition_row_policy="random"` on every rank (the only value the sbatch script ever writes into
    `cell.json`, no per-rank variation possible from a single shared-filesystem file written before `srun`)
    — the "cut... race" scenario in the task brief did not apply and there is no evidence of one.

    **Real capacity extrapolation**: eps=2^-16 -> final_terms=38,791,220; eps=2^-18 ->
    final_terms=583,393,599 / peak_terms=635,371,364, a ~15x jump. Naively continuing that ratio puts
    eps=2^-20's peak in the 9-10 billion term range. At `W=2` (127 qubits needs the 128-qubit dispatch
    width) the core payload is 2×u64 (x) + 2×u64 (z) + complex128 coefficient = 48 bytes/term; 10B terms is
    ~480 GB of raw payload before any merge/scratch doubling, Vec capacity slack, or per-thread (48
    threads/rank) Rayon accumulator overhead — all of which are real and unaccounted for in that minimum.
    The observed total RSS (~2.79 TB, `AveRSS × NTasks`) implies an effective bytes/term several times the
    48-byte floor, which is the expected shape for a parallel merge engine, not evidence of a leak.

    **The imbalance is the sharper, more actionable half of the finding.** Total available memory (8 ranks
    × ~1.47 TiB/node ÷ 2 ranks/node = 8 × ~735 GiB ≈ 5.87 TiB) would have comfortably covered an *evenly
    split* ~10B-term sum. The crash happened because `partition_row_policy="random"` (the only policy this
    path supports — decision #21) drew a partition-row set that put roughly 3.4x the average share of
    terms on one rank, and that rank shares its node with a second rank, so the node-level ceiling (not the
    cluster-level one) was hit first. `research/`/Known Gaps already documents partition-row imbalance as
    open research ("roughly half of a dense two-qubit gate's deltas cross at P=2... tuning the rows is open
    research") — this run is a real, measured instance of that same phenomenon causing an actual failure,
    not a new mechanism.

    **Conclusion: genuine capacity/imbalance limit at this scale, not a bug — case (b), no code fix
    applied.** `run_cell_distributed.py` and `campaign-genoa-distributed.sbatch` are correct as written;
    changing them would either paper over a real result (e.g. silently capping `min_abs_coeff`) or require
    building the distributed explicit-rows plumbing decision #21 explicitly deferred (a multi-day task, out
    of scope for an OOM-triage pass). The concrete, low-risk lever available today without any code change
    is more nodes: doubling to `--nodes=8` (16 ranks, 1 per NUMA domain as today) roughly doubles both
    total capacity and the number of partitions the random draw spreads terms over, which should shrink
    the worst-rank absolute footprint even if the relative skew ratio persists. Recommended resubmit for a
    real eps=2^-20 completion attempt:

    ```
    env -u SBATCH_RESERVATION --nodes=8 MIN_ABS_COEFF=9.5367432e-07 \
        sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-distributed.sbatch
    ```

    If this still OOMs, the next lever is `--nodes=16` (32 ranks) before concluding eps=2^-20 needs the
    distributed explicit-rows (`"cut"`) plumbing built first; a looser point (eps=2^-19, `MIN_ABS_COEFF=
    1.9073486e-06`) is the fallback if node budget is constrained, as a real (if less ambitious) E7 capacity
    point instead of a repeated crash at 2^-20.

23. **E7 achieved for real on 2026-09-14**: resubmitting at 16 ranks/8 nodes (double decision #22's
   8-rank attempt, same eps=2^-20, same `partition_row_policy=random`) completed cleanly:
   `wall_time_s=1658.8`, `peak_terms=8,923,556,570`, `peak_rss_kb≈3.04 TB` summed across ranks
   (`raw/2026-09-13-distributed-16ranks/runs.jsonl`). 3.04 TB is double a single genoa node's 1.5 TB
   RAM, so this is a genuine beyond-single-node-capacity demonstration, not merely a slow single-node
   equivalent — the same random partition draw that OOM'd one rank at 8 ranks (decision #22: one rank at
   ~1.20 TiB, 343% above average) spread thin enough across 16 ranks to leave headroom everywhere.
   `evidence.md` E6 and E7 both updated from tooling-ready/blocked to real data.

24. **Prepared `jobs/campaign-genoa-e8.sbatch` on 2026-09-14** for a real, hardware-valid E8
   comparison: single genoa node, in-process partitioned engine (`partitions=2`), same canonical
   task and eps=2^-16 as `campaign-genoa.sbatch`'s trusted overlap point, run once under
   `partition_row_policy=random` and once under `"cut"`. Needed because decision #21's random-vs-cut
   numbers came from a preflight-monkeypatched local test (this login host isn't genoa), which proved
   the mechanism but isn't schema-legitimate campaign data — confirmed 2026-09-14 by actually trying
   `run_cell.py` locally: it correctly returned `status=invalid_hardware` with no bypass available,
   by design. Not yet submitted (org policy: submission is the user's step).
