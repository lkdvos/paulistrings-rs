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

25. **E8 achieved for real on 2026-09-14** (`raw/2026-09-14-worker7173-e8`, job 7033128): the
   locality-aware `"cut"` partition-row policy cut export volume by ~91% (408M vs. 4.52B rows;
   19.6GB vs. 217GB) and wall time by ~2.3x (39.6s vs. 91.9s) relative to `"random"`, at the same
   correctness (`final_terms`/`peak_terms` identical between policies) — the first schema-legitimate,
   hardware-valid E8 data point, superseding decision #21's preflight-monkeypatched local proof.
   Note: `campaign-genoa-e8.sbatch` originally embedded the policy name in `config_id`, which would
   have defeated `make_hash_communication_figure`'s per-`config_id` grouping; fixed to share one
   `config_id` across both policies (only `partition_row_policy` differs) before generating the real
   figure at `figures/real/hash_communication.png`. `evidence.md` E8 moved tooling-ready to real data.

26. **Found and fixed a real gap while wiring E9's accuracy figure, 2026-09-14**: `run_cell.py`
   called `evolved.expectation(spec.state)` and discarded the result -- a real computation thrown
   away every run, and the exact value `analysis/normalize.py::accuracy` needs. Also found
   `accuracy()` itself was unusable for a real cross-engine comparison: it expected one run record
   to carry both a `reference_value` and an `observed_value` in its own `extra` field, but the Rust
   and Julia drivers each write their own engine's `expectation_re`/`expectation_im` to their own,
   separate run record (decisions.md #20) -- no single record ever has both. Fixed: `run_cell.py`
   now stores `extra.expectation_re`/`expectation_im` (matching the Julia driver's shape exactly);
   `accuracy()` rewritten to pair a `pauli_propagation_jl` run against every other completed run
   sharing `(min_abs_coeff, n_qubits)`, emitting `reference_value`/`observed_value`/`abs_delta`.
   34/34 analysis+jobs tests still pass. No existing Rust run record has `expectation_re` (all
   predate this fix, including the 2026-09-13 `worker7277`/`worker7327` data) -- a fresh, cheap
   single cell (threads=96, eps=2^-16, matching the real Julia point in `raw/2026-09-14-worker7169-julia`)
   is needed to get the first real paired accuracy data point.

28. **Built the historical-variants driver (E1/E2 prep, really E4/E5's historical-baseline
   comparison) for real on 2026-09-14**: `jobs/run_cell_historical.py`, a separate driver
   (not folded into `run_cell.py`) because the four historical commits in
   `tasks/T01-variants.json` drift from the current Python API in four genuinely different
   ways -- `naive_baseline` (d410f4e) has `todo!()`-stubbed PyO3 bindings entirely,
   `direct_small_sum_path` (e56f021) has no `engine=` kwarg on `propagate`, and
   `bucketed_engine_serial`/`bucketed_engine_parallel` (f08db7d/ef03701) predate
   `examples/common/` entirely. A single shared code path through `run_cell.py`'s existing
   `CellSpec`/`run_cell` shape was not realistic; a `VARIANT_ENTRY` registry
   (`VARIANT_REGISTRY: dict[str, VariantEntry]`) maps each `variant_id` to its commit SHA and
   one of two strategies instead.

   **Reduced scale, chosen and fixed for all five variants (four historical + a same-scale
   `bucketed_current`)**: n_qubits=12 (a straight 11-edge line on `heavy_hex_sublattice`, well
   clear of the isolated-qubit sizes CLAUDE.md/circuits.py warn about), 2 Trotter steps,
   theta_h=7pi/32 (contract.md's primary point, kept for continuity even off the frozen
   127-qubit task), theta_zz=-pi/2 (contract.md's fixed value), min_abs_coeff=1e-6. Chosen
   empirically: `naive_baseline`'s unbucketed, single-threaded, no-prepare engine completed
   this workload in ~1e-4s, comfortably "seconds, not hours" with room to spare, while still
   producing a real Trotterized circuit (46-90 channels depending on the ZZ-gate decomposition)
   rather than a degenerate single-gate smoke test. This is explicitly NOT the frozen 127-qubit
   canonical task (`contract.md`) -- it exists only so all variants can complete and be compared
   on equal footing; a real cluster-scale comparison still needs the 127-qubit task, which none
   of these four historical commits can build via `examples/common/heavy_hex_kicked_ising`
   (three of the four predate that module, and the fourth's PyO3 surface is unusable).

   **Two build/run strategies**, both driven from a throwaway `git worktree add --detach`
   (never the live checkout, always removed in a `finally`, verified after every run this
   pass left zero stray worktrees registered against the live repo):
   - `naive_baseline`: `strategy="rust_harness"`. A small Rust example
     (`_NAIVE_EXAMPLE_SRC`, templated and written into the worktree's
     `crates/paulistrings/examples/historical_smoke.rs`) built with
     `cargo build --release --example`, run as a subprocess, one JSON line of stdout parsed
     for `wall_time_s`/`initial_terms`/`final_terms`. Because this commit predates
     `Circuit::rx`/`rz`/`cnot` sugar, the ZZ interaction is built with the engine's native
     two-qubit `PauliRotation` generator directly (`gen_z` bits at both qubits, `theta_zz`) --
     not the CNOT-RZ-CNOT sandwich the other three variants use, since that identity requires
     gate methods this commit doesn't have. This is a disclosed, real confounder: the two
     decompositions are exactly equivalent unitaries, but per-channel truncation sees a
     different intermediate circuit, so `naive_baseline`'s final term count (14) is not
     expected to match the other three's (10) -- not a bug, and not silently glossed over
     (see the registry entry's `notes` field and `evidence.md`'s new E4/E5 row).
   - `direct_small_sum_path`/`bucketed_engine_serial`/`bucketed_engine_parallel`:
     `strategy="pyo3_handrolled"`. Built via `maturin develop --release` into a per-variant
     venv (created once per variant, reused across reps), then run through
     `_HANDROLLED_WORKLOAD_SRC`, a hand-built kicked-Ising circuit using only
     `Circuit.rx`/`.cnot`/`.rz` (all three commits expose exactly this surface, confirmed by
     inspection before writing the shared template) and the CNOT-RZ-CNOT sandwich for the ZZ
     interaction. `direct_small_sum_path` runs its default (sorted) engine only -- decisions.md's
     option (a) backport of current HEAD's `engine=` kwarg into that commit's `sum.rs` was
     time-boxed out of scope for this pass (real diff inspected, confirmed mechanically
     straightforward but non-trivial to backport safely without its own test pass); the
     direct-apply path this variant exists to demonstrate is therefore NOT exercised, and this
     is stated plainly in the registry entry's `notes` and in `evidence.md`, not silently
     dropped.

   **Peak RSS**: `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` before/after each subprocess
   call, delta reported with `peak_rss_provenance="rusage_children_maxrss_delta"` -- cheap and
   real, with the same "lower bound, not an exact per-run figure" caveat `harness.py` already
   documents for `/proc/self/status`'s VmHWM.

   **Preflight gating**: `run_cell_historical.py` calls the same `preflight.run_preflight()`
   `run_cell.py` does; a non-genoa host gets a real `invalid_hardware` record, no worktree
   touched. `PS_HIST_SKIP_PREFLIGHT=1` bypasses this for local-dev validation only (documented
   in the module docstring), mirroring the precedent already set for `run_cell.py`/
   `run_cell_julia.py` in decisions #17/#19/#20/#21 -- `campaign-genoa-historical.sbatch` never
   sets it.

   **Real local validation, all four historical variants plus `bucketed_current` at the same
   scale**: every one of the five cells produced a `status="completed"` record; all five passed
   `analysis/schema.py::validate_run` with 0 problems (checked directly, and via
   `analysis/validate_campaign.py` on the combined `runs.jsonl`/`gates.rank-0.jsonl`: "PASS: 6
   run(s), 46 gate record(s), 0 problems" -- the 6th run and the gate records are from the
   `bucketed_current` cell, the only one of the five with a gate trace). Real wall times on this
   non-genoa workstation: naive_baseline 1.04e-4s (14 terms), direct_small_sum_path 1.11e-3s
   (10 terms), bucketed_engine_serial 6.09e-3s (10 terms), bucketed_engine_parallel 3.31e-3s
   (10 terms), bucketed_current 1.54e-4s (10 terms). These numbers are NOT evidence of a real
   speedup ordering -- the reduced scale is dominated by fixed per-call overhead (venv/import,
   worktree build artifacts already warm from repeated local runs), not algorithmic cost; they
   demonstrate that all four historical variants build and run for real, which real cluster-scale
   data (not yet collected) needs as its foundation.

   **Tests**: `jobs/tests/test_run_cell_historical.py`, 11 tests -- registry shape/SHA
   cross-check against `tasks/T01-variants.json`, `cell.json` unknown-field rejection, the
   honest real-preflight `invalid_hardware` path on this (non-genoa) host with no monkeypatch,
   the `PS_HIST_SKIP_PREFLIGHT` bypass exercised with the build strategy monkeypatched out (fast,
   no real build), and 4 real end-to-end build+run+`validate_run` tests (one per historical
   variant), gated behind `PYTEST_RUN_SLOW_HISTORICAL_BUILDS=1` (no `slow` pytest marker exists
   elsewhere in this suite, so a skipif env-var gate matches the existing convention rather than
   introducing one) since a fresh build can take over a minute -- same spirit as
   `test_run_cell_distributed.py`'s MPI-build tests. Ran both ways: 7 passed/4 skipped by default
   (0.45s), and separately with the env var set, all 4 real-build tests passed in 139s. The full
   existing suite (`analysis/tests`, `jobs/tests`, `figures/tests`) still passes: 67 passed, 5
   skipped, no regression.

   **New Slurm template** `jobs/campaign-genoa-historical.sbatch`, mirroring
   `campaign-genoa.sbatch`'s toolchain/module/JCC-rustflags preamble for the `bucketed_current`
   leg and calling `run_cell_historical.py` directly (module-loaded `python3.11`, no separate
   venv needed at the driver level -- it manages its own per-variant venvs) for the four
   historical legs, all writing to one `out_dir`/`runs.jsonl`. `bash -n` syntax-checked clean.
   Not submitted -- submission is the user's step (decision #2):

   ```
   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-historical.sbatch
   ```

   **Corners cut, given the time budget**: (1) the `engine=` kwarg backport for
   `direct_small_sum_path` (documented above, not silently dropped); (2) `jcc_erratum_and_
   branch_prediction` (E1) and `presentation_bench_crate_variants` (E2, already out of scope
   per the task brief) were not touched this pass -- the task named exactly the four variants
   built here; (3) no real genoa cluster run yet, only local validation on this workstation
   (same "local proof, cluster run is the user's step" pattern as decisions #17/#19/#20/#21/#24).

27. **E9 headline accuracy result achieved for real on 2026-09-14** (`raw/2026-09-14-worker7169`
   paired with `raw/2026-09-14-worker7169-julia`): at the full 20-step canonical depth, eps=2^-16,
   `paulistrings`' expectation value (0.39716532998468246) agrees with PauliPropagation.jl's
   (0.3971653299846826) to `abs_delta=1.67e-16` -- floating-point noise, not a real discrepancy.
   This is the headline claim decision #10's shallow 5-step pilot could only gesture at.
   `figures/real/accuracy.png` generated via the fixed `normalize.accuracy()` (decision #26).
   evidence.md E9 moved from "plumbing validation only" to real headline data.

29. **Replaced the single-point accuracy plot with a convergence-trajectory sweep, 2026-09-14**,
   per user request: a single (reference, observed) point is a weak figure. New
   `jobs/run_convergence_sweep.py` propagates `circuit[:k*channels_per_step]` for every
   `k in 1..trotter_steps` and calls `.expectation(state)` on each prefix -- `Circuit.__getitem__`'s
   slice support already exists (`crates/paulistrings-py/src/circuit.rs`), so this needed no engine
   change, only a new driver. A prefix propagation is correctness-preserving (truncation only ever
   depends on the sum's own history, never on gates not yet applied), at the cost of redoing the
   shared prefix work once per step -- accepted since there is no incremental/checkpointed propagate
   call to build on instead. Cutoff grid fixed to {2^-12, 2^-14, 2^-16, 2^-18} per the user's explicit
   time-budget instruction (avoid tighter cutoffs than what's already been measured). Output is a
   bespoke `convergence.jsonl` shape (`CONVERGENCE_FIELDS`), not `analysis/schema.py`'s `RUN_FIELDS`
   -- one row is one (cutoff, step) point on a curve, not a campaign cell, and forcing that shape
   would have added fields with no meaning here. Smoke-tested locally (n=10, 2 cutoffs, preflight
   monkeypatched in a throwaway script only): term counts diverge between cutoffs as expected while
   the two cutoffs still had not visibly diverged in `<O>` at this toy scale/depth. New
   `figures/make_compact_figures.py::make_convergence_figure` and
   `jobs/campaign-genoa-convergence.sbatch` (mirrors `campaign-genoa.sbatch`'s preamble). Existing
   figures/jobs test suites (23 tests) still pass. Not yet run on the real cluster.

30. **Extended E8 to a cutoff sweep, 2026-09-14**, per user request (same rationale as decision
   #29's accuracy -> convergence upgrade): a single (eps=2^-16) random-vs-cut point doesn't show
   whether the cut policy's advantage holds as the sum grows. `campaign-genoa-e8.sbatch` now loops
   over the same 4-point grid {2^-12, 2^-14, 2^-16, 2^-18} as `campaign-genoa-convergence.sbatch`
   (both policies at each point) instead of one. `analysis/normalize.py::hash_communication` gained
   a `min_abs_coeff` field per row (pulled straight from the run record; previously the join had no
   way to know which cutoff a row came from) and its sort key changed from `config_id` to
   `min_abs_coeff` to support that. New `figures/make_compact_figures.py::
   make_hash_communication_vs_cutoff_figure` plots export volume vs. cutoff, one line per policy
   (log-log), alongside the existing per-`config_id` bar chart (kept, not replaced -- still useful
   for a single-point comparison). 2 new normalize tests + 2 new figure tests, all green (46/46
   across analysis/figures/jobs). Not yet run on the real cluster at the extended grid; the
   eps=2^-16 point already has real data (evidence.md E8).

31. **Real cluster run (job 7033257) exposed a real bug the subagent's local testing missed,
   2026-09-14**: all four historical-variant cells failed `ModuleNotFoundError: No module named
   'paulistrings'` (only the `bucketed_current` cell, which builds its own worktree/venv inline in
   the sbatch script, succeeded). Root cause: `run_cell_historical.py`'s top-level `from run_cell
   import RUN_FIELDS, ...` eagerly executes all of `run_cell.py`, which imports `examples/common/
   circuits.py`, which imports `paulistrings` at module scope -- before this driver has built any
   of its own per-variant throwaway venvs. The subagent's local validation had `.venv` (already
   containing paulistrings from earlier campaign work) active on `sys.path`, masking this entirely;
   the real cluster job invokes the bare module Python, which has nothing installed yet. Fixed by
   duplicating the four small, genuinely paulistrings-independent helpers (`RUN_FIELDS` tuple,
   `_append_jsonl`, `_compiler_version`, `_slurm_job_id`) directly into `run_cell_historical.py`
   instead of importing them, removing the eager dependency entirely -- verified by importing the
   module under the bare `module load python/3.11.11` interpreter (no `.venv`) and confirming no
   `paulistrings` import is triggered. `RUN_FIELDS` must now be kept in sync with `run_cell.py`'s
   copy by hand; `analysis/schema.py`'s own `RUN_FIELDS`-equivalent validator is the actual
   authority either side answers to, so a drift here fails loudly there, not silently. Existing
   tests (7 passed, 4 skipped -- the slow real-build ones) still pass. Lesson for future delegated
   work: local validation under an environment that happens to have extra state pre-installed
   (`.venv` with prior campaign packages) is not equivalent to the real job's from-scratch
   environment -- this is exactly the class of gap real Slurm submission has caught before
   (decisions.md #17, #22).

32. **Real genoa historical-variant data achieved on 2026-09-14** (job 7033716, after decision
   #31's fix): all four historical commits completed on real genoa hardware at the reduced toy
   scale, 0 schema problems. Generated `figures/real/recurring_stage6.png` (stages 1-6 of the
   recurring figure) from real `runtime_tolerance()` rows -- the efficiency/left panel is empty
   for every stage since none of the four historical strategies produces a per-gate trace (module
   docstring). `bucketed_current`'s reduced-scale cell was run for the sbatch's own fair-comparison
   purpose but isn't itself one of the 7 `STAGE_VARIANTS` (that slot is `bucketed_engine_parallel`,
   "the canonical bucketed engine as shipped"), so it's excluded from the figure. `evidence.md`
   E4/E5-historical-baseline row upgraded from toy-scale-non-genoa to real genoa data.

33. **Convergence sweep completed for real on 2026-09-14** (job 7033650, 80 points, all
   `status=completed`): eps=2^-16 and eps=2^-18's final (step 20) points exactly match the
   previously-measured `final_terms` (38,791,220 and 583,393,599 respectively) from the main
   campaign's own 127-qubit runs, confirming the prefix-propagation approach's correctness
   end-to-end, not just at toy scale. Per a follow-up user request ("I'm also still missing a
   figure that contains the comparison with PauliPropagation.jl"), `make_convergence_figure`
   gained an optional `julia_points=` overlay (black stars) -- `runner.jl` only computes a final
   expectation value, never a per-layer one (its `PP_LAYER_COUNTS` gives per-layer term counts
   only), so this is a real endpoint overlay, not a Julia trajectory line. The one real Julia point
   we have (eps=2^-16, step 20, decisions.md #27) is plotted; getting more would need additional,
   expensive Julia runs at the other three cutoffs (not requested, not run). Output renamed
   `figures/real/accuracy.png` -> `convergence.png` to match its actual content. 3 new figure
   tests, all green (19/19).

34. **Quera-talk full-scale historical sweep, 2026-09-14** (per explicit user request: "a full
   scale 127 qubit version with series for various tolerances, for the expensive versions simply
   only look at the less strict tolerances"). Real local feasibility investigation on this
   (non-genoa, 32-core Xeon Gold 6244) workstation before touching the driver, using
   `run_cell_historical.py`'s own `run_cell()` directly with `PS_HIST_SKIP_PREFLIGHT=1`:

   **Finding #1 (unexpected, changes the plan): at the original `trotter_steps=2`, cost is flat
   in BOTH n_qubits and min_abs_coeff.** Timed `naive_baseline` at n_qubits in {12, 32, 64, 96,
   127} at the loosest cutoff (2^-12): wall_time_s stayed in [4.5e-5, 4.1e-4]s and final_terms
   stayed at exactly 14 for every n_qubits. Then swept the full 4-point cutoff grid at n=127,
   trotter_steps=2: final_terms stayed at 14 for every cutoff too. Root cause: this is a backward
   (Heisenberg) propagation of a single local-site observable; after only 2 Trotter layers the
   operator's light cone has not reached the chain's boundary or grown enough to approach any of
   the four cutoffs, so scaling n_qubits alone (as literally read from the task) would give every
   one of the four variants an identical, trivial, cost-flat line -- not the "expensive variants
   fall behind at tight tolerance" story the recurring figure's cost panel needs. Depth, not
   qubit count, is the real cost driver for this construction.

   **Finding #2: n_qubits is free at every depth tested** -- re-confirmed at trotter_steps=6:
   naive_baseline's cost and term count did not move between n=12 and n=127. This means fixing
   n_qubits=127 for all four variants costs nothing extra relative to a smaller n, so there is no
   real feasibility tension on the qubit-count axis at all -- the tension is entirely on depth x
   cutoff.

   **Decision: raise `trotter_steps` from 2 to 10 (real, tractable, cutoff-sensitive) and fix
   n_qubits=127 for every variant**, extending `run_cell_historical.py`'s `HistoricalCellSpec`
   with `n_qubits` (default `DEFAULT_N_QUBITS=12`, unchanged) and `trotter_steps` (default
   `TROTTER_STEPS=2`, unchanged) fields, both threaded into the two build templates (n_qubits was
   already a template placeholder; trotter_steps needed the same treatment). `min_abs_coeff` was
   also changed from a single float to a tuple normalized from either a scalar or a list in
   `HistoricalCellSpec.from_dict`, and `run_cell()` now builds each variant's worktree/venv
   **once** and loops over every requested cutoff against that one build -- the Rust harness's
   `MIN_ABS_COEFF` moved from a compile-time const to a runtime CLI arg (`argv[1]`) specifically
   so a tolerance sweep never pays a second `cargo build` per point. `run_cell()`'s return type
   changed from a single dict to `list[dict]` (one schema-v1 record per cutoff); `main()` and all
   of `test_run_cell_historical.py`'s call sites were updated to match, plus new tests for the
   scalar/list normalization, the n_qubits/trotter_steps defaults, and (mocked, fast) confirmation
   that a multi-cutoff sweep builds exactly once.

   **Real feasibility table at n_qubits=127, trotter_steps=10, this non-genoa workstation** (the
   `direct_small_sum_path`/`bucketed_engine_serial`/`bucketed_engine_parallel` circuit is the
   shared CNOT-RZ-CNOT-sandwich one, so their final_terms agree exactly at each cutoff;
   `naive_baseline`'s native-generator ZZ decomposition is the pre-existing disclosed confounder,
   not measured at every cutoff here since it only gets one):

   | variant | eps=2^-12 | eps=2^-14 | eps=2^-16 | eps=2^-18 |
   | --- | --- | --- | --- | --- |
   | naive_baseline | 133.4s / 3,018,683 terms | *(not attempted)* | *(not attempted)* | *(not attempted)* |
   | direct_small_sum_path | 1.11s / 232,432 | 1.72s / 696,172 | 2.76s / 1,791,652 | 4.76s / 3,936,794 |
   | bucketed_engine_serial | 12.96s / 232,432 | 34.25s / 696,172 | 68.63s / 1,791,652 | 128.14s / 3,936,794 |
   | bucketed_engine_parallel | 31.45s / 232,432 | 69.76s / 696,172 | 130.34s / 1,791,652 | 240.38s / 3,936,794 |

   (`naive_baseline`'s own decomposition gave 3,018,683 terms at eps=2^-12 vs the other three's
   232,432 -- consistent with the pre-existing confounder, and also confirms its unbucketed
   sort-merge engine is doing real, non-trivial work at this scale, not an artifact of the toy
   circuit.)

   **A genuinely surprising, disclosed result: `bucketed_engine_parallel` was ~2.2-2.4x SLOWER
   than `bucketed_engine_serial` at every cutoff on this 32-core, single-socket workstation**, not
   faster. This is real data, not a bug in the driver (both engines produce identical
   `final_terms` at every cutoff, confirming correctness; only wall-clock differs). Two credible,
   undistinguished-by-this-pass explanations: (1) this historical commit's Rayon parallelism
   (`ef03701`, "v0.2 C.1-C.3") may have a genuine per-task overhead issue at this workload's
   bucket/task granularity that a later commit fixed, consistent with `research/FINDINGS.md`
   cataloguing several "obviously good" ideas that measured worse; (2) single-socket, 32-thread
   contention on this workstation is not representative of the frozen 2-socket, 96-core genoa
   class this campaign targets. Not investigated further (out of scope for a driver/scheduling
   task) -- flagged here explicitly so the real cluster run is not read as a foregone conclusion
   for stage 5 -> stage 6 of the recurring figure.

   **Final per-variant plan implemented** (`jobs/campaign-genoa-historical.sbatch`, all at
   n_qubits=127, trotter_steps=10):
   - `naive_baseline`: **one point only**, the loosest cutoff (2^-12) -- 133s for that single point
     already dominates the other three variants' entire 4-point grids combined, and the
     unbucketed engine's own term generation is not cutoff-sensitive at the depths sampled (an
     earlier trotter_steps=6 probe showed its term count flat across the whole grid), so a second
     point would cost roughly the same again for no additional signal.
   - `direct_small_sum_path`, `bucketed_engine_serial`, `bucketed_engine_parallel`: the full
     4-point campaign grid {2^-12, 2^-14, 2^-16, 2^-18} -- all comfortably tractable (max single
     point 240s, max full-grid total ~472s, both well inside the sbatch's wall-time cap).
   - `bucketed_current` (not itself a `STAGE_VARIANTS` entry, run for a fair current-vs-historical
     sanity overlay per the task's optional item 3): same n_qubits=127, trotter_steps=10, and the
     same full 4-point grid as the strongest historical variant, via `run_cell.py`'s existing
     one-cutoff-per-invocation CLI (bash loop in the sbatch template; that driver was
     intentionally not touched).

   This directly implements the requested narrative device for
   `figures/make_recurring_figure.py::_draw_cost_panel` (confirmed by reading it, not modified):
   it groups `tolerance_rows` by `variant_id` and plots only `status == "completed"` points --
   `naive_baseline`'s line will have exactly one point where the other three have four, with no
   failed/OOM record needed for the untried cutoffs, exactly the "this regime didn't exist until
   the engine improved" visual the user asked for.

   **Validated locally, not yet on a real cluster.** Fast test suite (registry/schema/gating/
   sweep-builds-once, no real build): `jobs/tests/test_run_cell_historical.py`, 12 passed / 4
   skipped in 0.22s. Full existing suite (`analysis/`, `jobs/`, `figures/`): 76 passed, 5 skipped,
   no regression. **Bare-Python import check (decision #31's exact lesson) repeated for this
   change**: `env -u VIRTUAL_ENV ... module load modules/2.4-20250724 python/3.11.11 && python3.11
   -c "import run_cell_historical"` with no `.venv` on `sys.path` -- imports cleanly,
   `"paulistrings" not in sys.modules` confirmed. `bash -n` on the rewritten
   `campaign-genoa-historical.sbatch` is clean. The feasibility table above and the
   `bucketed_engine_parallel` anomaly are real numbers from real builds/runs on this workstation,
   not estimates -- but they are not genoa numbers; a real cluster run at this exact
   n_qubits=127/trotter_steps=10/per-variant-grid plan is still the user's next step:

   ```
   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-historical.sbatch
   ```

   **Corners cut**: the slow real-build tests (`PYTEST_RUN_SLOW_HISTORICAL_BUILDS=1`) were not
   re-run end-to-end against the new full-scale defaults in this pass (they still exercise the
   original small toy scale via `_spec()`'s defaults, which is what they were written to check --
   the driver's build/run machinery, not a specific scale); the fast suite plus the manual
   feasibility runs above are the real evidence for the full-scale path specifically.
   `figures/make_recurring_figure.py` and `figures/make_compact_figures.py` were read (to confirm
   `_draw_cost_panel`'s missing-point behavior) but deliberately not modified, per the task. The
   E8/convergence-sweep jobs (7033663, 7033650) were running concurrently in this shared worktree
   during this work; nothing here touched their scripts, `raw/` output directories, or `cell.json`
   scratch paths (this driver's scratch lives under a separate `/tmp` prefix).

35. **Real Julia convergence trajectory built and cost-estimated, 2026-09-14**, per user
   follow-up ("I'm also still missing a figure that contains the comparison with
   PauliPropagation.jl" -> "make that comparison a real full trajectory, not just one
   endpoint"). Two real questions had to be answered before writing any code, per the task's
   explicit instruction not to guess: (a) does PauliPropagation.jl 0.8.2 expose an
   incremental/checkpointed propagation API that would make a per-step trajectory cheap, and
   (b) if not, is genuine from-scratch re-propagation of every prefix affordable single-threaded.

   **(a) API investigation, real, not assumed.** Read the installed package source directly
   (`~/.julia/packages/PauliPropagation/kdA9q` -- confirmed as the 0.8.2/tree-sha
   `fe2bc2552caf975532a8b1372bd8bde1e1cd3f3f` install by matching `Manifest.toml`'s pinned
   tree-sha against `Project.toml`'s version field in each of the two locally-installed
   copies; a second, older 0.3.0 copy at `.../Z3w2l` was NOT the one used). `src/Propagation/
   generics.jl` and `propagationcache.jl`: `propagate`/`propagate!` and every
   `AbstractPauliPropagationCache` variant always take the WHOLE given circuit and run it in
   one call; the cache types (`PauliPropagationCache`, `VectorPauliPropagationCache`) are
   allocation-reuse buffers, not checkpoints across separate circuit segments.

   More importantly, **this is not an API gap but a mathematical fact about what a growing
   Heisenberg-picture prefix means**, worked out by hand: for `direction="heisenberg"`,
   propagating prefix `circuit[:k*cps]` conjugates the observable by gate `k*cps` FIRST
   (innermost) and gate 1 LAST (outermost) -- `_preparecircuit`'s `toheisenberg` reversal,
   confirmed in `generics.jl`. Going from step `k` to step `k+1` prepends the new step's gates
   at the INNERMOST position, ahead of everything already baked into step `k`'s result; the
   common tail (gates `1..k*cps`, reversed) is the SAME fixed linear operator `T_k` applied to
   two DIFFERENT starting operands (`O` for step `k`, `O` already conjugated by the new step's
   gates for step `k+1`) -- `T_k` itself is never stored as a reusable object, only realized by
   actually running `propagate` over that many gates, so there is no way to reuse step `k`'s
   result to get step `k+1`'s cheaper. (The reverse pass -- one forward sweep applying gates
   `N, N-1, ..., 1` once and recording intermediate results -- computes a real but DIFFERENT
   quantity: Heisenberg conjugation by a circuit SUFFIX of length `j`, not a prefix of length
   `k`; these coincide only in the trivial `j=k=N` case.) Conclusion: **prefix re-propagation
   is the only correct approach; there is no incremental shortcut, confirmed rather than
   assumed.**

   **(b) Real/estimated cost.** Real Rust data from the existing convergence sweep
   (`raw/2026-09-14-worker7172-convergence/convergence.jsonl`, 96 threads) gives, per cutoff,
   both the term-count/wall-time growth curve (which saturates by step ~9 for eps in
   {2^-12,2^-14,2^-16}, so the total-sweep-vs-final-step-alone redundancy ratio is ~6.7-8x, not
   ~20x) and the final-step (full-circuit) wall time. Combined with the one real single-thread
   Julia timing point (decisions.md #27: 4874.94s at eps=2^-16) and the real Rust
   single-thread/96-thread ratio at the same cutoff (~33.6x, from the ~1650s single-thread
   figure already on record), a same-ratio extrapolation to the other cutoffs plus the
   Julia/Rust ~2.95x single-thread factor (decisions.md #10/#27) gives: eps=2^-12 ≈ 15 min,
   eps=2^-14 ≈ 49 min, eps=2^-16 ≈ 9 hours, all single-threaded. eps=2^-18 is excluded: the
   RUST engine's own single-thread full run at that cutoff has twice failed to finish inside
   an 8h cap (job-ledger.jsonl 7030090/7031789) with zero completed reps, so a ~3x-slower
   Julia attempt would almost certainly repeat that failure at higher cost, not merely be
   slow. **These are estimates, not measurements** -- flagged as such everywhere they appear
   (`evidence.md`, `campaign.json`'s new stage) -- built from real data on both sides but never
   validated by an actual Julia run past toy scale.

   **Implementation.** `benchmarks/julia/runner.jl` gained `PP_LAYER_EXPECTATION=0|1` (default
   0) and `PP_TROTTER_STEPS=N`, following `PP_LAYER_COUNTS`'s exact pattern (documented in the
   header env-var table): for `step in 1:N`, propagate `circuit[1:step*cps]` fresh from a
   `deepcopy` of the observable (mirroring `run_convergence_sweep.py`'s
   `circuit[:k*channels_per_step]` exactly) and record `(trotter_step, re, im, final_terms,
   wall_time_s)` into a new `result.per_layer_expectation` array, in the same `extra`-shaped
   output slot as `per_layer_terms`. Requires `task.run.state` (a term count needs none, an
   expectation does); a clear `TaskError` otherwise. `benchmarks/python/julia_baseline.py`
   gained a matching `JuliaResult.per_layer_expectation` property; `run_task`'s existing
   `extra_env` passthrough needed no signature change. Every existing env-var-gated behavior
   (`PP_LAYER_COUNTS`, `PP_FUSED`, `PP_BACKEND`, etc.) is untouched when
   `PP_LAYER_EXPECTATION` is left at its default 0.

   **Real correctness check** (the actual trust bar for a comparison figure, mirroring
   `test_julia_parity.py`'s existing rigor): new
   `test_per_step_expectation_trajectory_parity` in that same file runs a real
   `julia` subprocess on a shared 6-qubit/4-step task and a Rust-side prefix trajectory
   (`run_rust_prefix_trajectory`, mirroring `run_convergence_sweep.py`'s own slicing) and
   compares every step's `final_terms` exactly and `<O>` to `EXPECTATION_TOL=1e-12`. **Passing,
   real, on this host** (juliaup, not module-loaded Julia -- acceptable per decisions #17/#19/
   #20's local-plumbing-validation precedent; a real module-loaded-Julia cluster run is still
   the confirming step). Also validated end-to-end with a preflight-monkeypatched throwaway
   script at n=8/2 steps through the new sweep driver directly.

   **New driver**: `jobs/run_convergence_sweep_julia.py` (a separate file from
   `run_cell_julia.py`, following `run_convergence_sweep.py`'s own precedent of a bespoke
   sweep driver rather than folding into the single-cell driver) builds the circuit/observable
   through `run_cell._build_circuit`/`_build_observable` exactly like `run_cell_julia.py`,
   then calls `julia_baseline.run_task` ONCE PER CUTOFF (not once per step) with
   `PP_LAYER_EXPECTATION=1`/`PP_TROTTER_STEPS=<n>`/`PP_WARM_REPEATS=0`/`PP_LAYER_COUNTS=0` --
   the latter two deliberately skip the runner's own redundant full-circuit warm-repeat and
   per-gate-term-count passes, each costing roughly one more full propagation, which this
   driver does not need. Writes `JULIA_CONVERGENCE_FIELDS` rows (reusing every field name from
   `run_convergence_sweep.py`'s `CONVERGENCE_FIELDS` whose meaning matches, plus
   `engine`/`runtime_version`) to `convergence_julia.jsonl`, one row per `(min_abs_coeff,
   trotter_step)` point, appended cutoff-by-cutoff so a later cutoff's timeout does not lose
   earlier real data. New `jobs/campaign-genoa-julia-convergence.sbatch` mirrors
   `campaign-genoa-julia.sbatch`'s worktree/venv/Julia-project preamble; default grid
   `{2^-12, 2^-14, 2^-16}` (NOT 2^-18, per the cost estimate above), `--time=16:00:00`. `bash
   -n` clean. **Not submitted** -- submission is the user's step (decision #2):

   ```
   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-julia-convergence.sbatch
   ```

   **Figure**: `figures/make_compact_figures.py::make_convergence_figure`'s `julia_points=`
   parameter (a flat list of single points) replaced with `julia_rows=` accepting the SAME row
   shape as the Rust `rows` argument. A cutoff with more than one Julia row draws a real dashed
   line in the SAME color as the Rust line at that cutoff (visually paired, distinguished only
   by linestyle); a cutoff with exactly one row (the existing real eps=2^-16/step-20 endpoint,
   decisions.md #27/#33, which has no natural multi-point shape) falls back to the original
   black-star rendering, so the one real data point already in hand keeps rendering exactly as
   before. Colors are keyed off the UNION of Rust and Julia cutoffs so a Julia-only cutoff
   cannot crash the lookup. 3 figure tests updated/added (single-point fallback, real dashed
   line, Julia-only-cutoff color safety); 21/21 green.

   **Not done, and explicitly out of scope for this pass**: the real 127-qubit cluster run at
   any cutoff -- `campaign-genoa-julia-convergence.sbatch` is prepared and `bash -n`-checked
   but not submitted, per org policy (decision #2). All cost figures above are estimates from
   real but indirect data (Rust's own term-growth curve plus one Julia timing point), not
   measurements of the actual Julia sweep; a real run is needed to confirm them before trusting
   the estimated ~9-hour eps=2^-16 figure enough to budget cluster time against it.

36. **E8 cutoff sweep completed for real on 2026-09-14** (job 7033663, all 8 cells, 0 schema
   problems): the `"cut"` partition-row policy's export-volume advantage over `"random"` holds
   consistently at ~11x across the full {2^-12, 2^-14, 2^-16, 2^-18} grid (10.7x, 10.8x, 11.1x,
   11.2x respectively) -- the advantage does not erode, and if anything grows slightly, as the sum
   grows by 4 orders of magnitude. `figures/real/hash_communication.png` regenerated as the real
   cutoff-sweep line plot (superseding the single-point eps=2^-16 bar chart).

37. **Added real Julia multithreading support, 2026-09-14**, per user request. Investigated the
   installed PauliPropagation.jl 0.8.2 source directly rather than assuming `julia -tN` alone
   helps: `propagate`'s own docstring says `thread=true` (the default) "disables multithreading in
   every function on the VectorPauliSum backend that can multithread" -- meaning the campaign's
   default "dict" backend (`PauliSum`) is single-threaded regardless of `-t`, and only `PP_BACKEND=
   vector` actually engages parallelism. This was existing, already-wired runner.jl functionality
   (`PP_BACKEND` env var, already read at line ~642) -- no new Julia code needed. Spot-checked
   correctness at a tiny scale (8 qubits, 5 gates): `dict`/1-thread and `vector`/4-thread agree on
   `final_terms` (2 vs 2). Added `CellSpec.backend` (shared dataclass, `run_cell.py`; default
   "dict", ignored by the Rust leg) and threaded it through `run_cell_julia.py`'s `run_task(...,
   backend=spec.backend)` call (already recorded in `extra.julia_backend`, unused until now).
   `campaign-genoa-julia.sbatch` gained `JULIA_THREADS`/`JULIA_BACKEND` env knobs (defaults
   preserve today's single-thread/dict behavior exactly). Not yet parity-tested at full campaign
   scale (only the tiny spot-check above) and not yet run on the real cluster -- see the submit
   command below for a first real multithreaded point at the same eps=2^-16 the single-thread
   baseline already covers, for a direct comparison.

38. **Multithreaded Julia achieved for real on 2026-09-14** (job 7034021, eps=2^-16, 96 threads,
   vector backend): `wall_time_s=899.49` vs. the existing single-thread `dict`-backend baseline's
   `4874.94` -- a real ~5.4x speedup. `final_terms=38,791,220` and `expectation_re=0.3971653299846822`
   both match the single-thread run to floating-point tolerance, confirming the `vector` backend's
   correctness at full 127-qubit campaign scale (not just decision #37's tiny 8-qubit spot-check).

39. **Julia thread ladder + Rust-vs-Julia efficiency overlay, 2026-09-14**, per user request.
   `campaign-genoa-julia.sbatch`'s `JULIA_THREADS` now accepts a space-separated list (looped,
   mirroring `campaign-genoa.sbatch`'s `THREADS` ladder pattern) instead of one value; each thread
   count only makes sense combined with `JULIA_BACKEND=vector` (decision #37: the default `dict`
   backend is single-threaded regardless of `-tN`). `figures/make_compact_figures.py::
   make_thread_scaling_figure` gained an `other_rows=`/`other_label=` overlay: each series is
   normalized against its OWN 1-thread baseline (`normalize.thread_scaling()`'s existing contract),
   so the comparison is of relative parallel efficiency, not absolute wall-clock speed -- a much
   slower engine in absolute terms can still be plotted on the same relative-speedup axes. One
   shared "ideal (y=x)" reference line covers both series. New figure test, all green (22/22
   figures, 25/25 analysis+jobs). Not yet run on the real cluster -- see the submit command below
   for the real ladder, matching Rust's own thread points {1,2,4,8,16,32,48,96} at the same eps=2^-16
   already covered by both engines' single-thread and 96-thread points.
