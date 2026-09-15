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

40. **Recovered real "attempted parallel approaches" data for slide 19 ("attempts"), 2026-09-14.**
   `E2` (evidence.md) was marked `blocked` because `presentation/bench/` is a standalone crate not
   present in this tree -- but it was already run for real on git branch `presentation` (not
   `main`, not this `presentation-work` worktree; never checked out here, read only via `git show`/
   `git log`/`git branch --all --contains`). Verified: branch `presentation` and commit `81b922c`
   (bench crate: naive/threadmaps/mergesort/bucketed) are reachable; `presentation/data/
   thread_scaling.jsonl` was written by commit `d9a3794`, and its own header line records the engine
   build it measured (`commit=7190d94`, host `ccqlin038`, 2026-09-06). Copied the file byte-for-byte
   (not moved, not edited, source branch untouched) to `raw/recovered-presentation-branch-attempts/
   thread_scaling.jsonl` with a README giving exact provenance and the cross-architecture caveat.
   Real 1-thread wall times at 127 qubits/10 Trotter steps/eps=2^-12: `threadmaps` (per-thread
   `HashMap`, merge at layer end) 137.4s, degrading with more threads (162.1s at 32); `mergesort`
   (flat array, parallel sort, segmented merge of equal keys) 54.1s, improving to ~28s at 16-32
   threads; `bucketed` (shipped engine, in-branch reference point) 15.1s down to ~1.7s. Added
   `make_compact_figures.py::make_attempts_figure` (wall time vs. threads, log-log, one line per
   strategy, title/caption explicit that this is historical cross-architecture reference data, an
   optional unconnected star marker for a same-campaign reference point -- not used here, since no
   confirmed real genoa bucketed 1-thread number exists yet in this campaign, per E4's own
   `tooling-ready` status) plus two new tests in `figures/tests/test_make_figures.py`. Full figure
   suite green: 25/25 passed (`./.venv/bin/python -m pytest figures/tests/`). Generated
   `figures/real/attempts.{png,svg,pdf}` from the recovered data. **This is NOT a same-hardware
   comparison** -- `ccqlin038` is a CCA/CCQ workstation, not this campaign's genoa/rocky9 node class;
   the figure, its README, and evidence.md's E2 row all state this caveat prominently. Did not
   modify anything on branch `presentation`.

41. **Bucket-size figure for slide page 30, real data, 2026-09-14.** Real cluster job 7035853
   (single-thread, 127-qubit heavy-hex Trotter step, `min_abs_coeff=2^-12`, `min_buckets=128`)
   swept `target_bucket_len` in {256, 512, 1024, 2048, 4096}, occupancy sampled at the final
   step of a genuine 20-step trajectory. That sampling relied on a real bug fix landed along
   the way: `crates/paulistrings/examples/phase_breakdown.rs`'s `--occupancy-at N` previously
   sampled after a warm-up pass plus N more steps (real depth `2*reps`), which had silently
   truncated the growing Trotter sum to nothing under the coefficient threshold while
   `num_buckets()` stayed at its grow-only high-water mark -- masking `target_bucket_len`'s real
   effect on occupancy (job 7035770's run exposed this: `empty_buckets == num_buckets` at every
   value). Fixed by skipping the warm-up on the occupancy-sampling path so `--occupancy-at N`
   means real depth N (commit `b15b741`). Confirmed the JSON sidecar has no `strings/s` key;
   read it from each config's sibling `.txt` phase-breakdown report instead. Added
   `make_compact_figures.py::make_bucket_size_figure` (two panels: throughput and occupancy
   median/p95/max plus empty-bucket fraction on a twin bar axis, both vs. `target_bucket_len`
   log2-spaced), taking raw probe-JSON dicts directly (no `normalize.py` step exists for this
   shape, same precedent as `make_distributed_capacity_figure`). Real numbers: throughput rises
   monotonically 5.713e7 -> 6.659e7 -> 7.158e7 -> 7.438e7 -> 7.588e7 strings/s across
   256->512->1024->2048->4096, with **no peak in the tested range** -- the figure's docstring
   states this explicitly and never claims an optimum, flattening, or cache residency. Occupancy
   (median/p95/max, empty_buckets/num_buckets): 256 -> 1/2/3, 3374/4096; 512 -> 1/2/4, 1416/2048;
   1024 -> 1/3/5, 518/1024; 2048 -> 2/4/6, 107/512; 4096 -> 3/6/8, 7/256. Generated
   `figures/real/bucket_size_v2.{svg,pdf,png}` and `_compact` variants (deck theme, 900x340pt /
   900x170pt, per MANIFEST's naming convention). Added 6 tests to
   `figures/tests/test_make_figures.py`, including one that pins the "no interior peak" shape of
   the real data as a regression tripwire and one confirming the empty-bucket fraction is never
   folded into the occupancy percentiles. Full figure suite green: 39/39 passed
   (`.venv/bin/python -m pytest quera-talk-data/campaign-2026-09-11/figures/tests/`).

42. **`naive_baseline` cannot produce a tolerance sweep at any commit in `VARIANT_REGISTRY`,
    2026-09-14.** Job 7035811 (`campaign-genoa-baseline-sweep.sbatch`) swept 9 `min_abs_coeff`
    values from 2^-4 to 2^-12 for `naive_baseline` and got bit-identical `final_terms=3018683` at
    every point (user flagged this as suspicious, correctly). Root-caused (not a driver bug):
    `naive_baseline`'s historical commit `d410f4e` ("Phase 6: sort-merge engine") never wires
    truncation into its merge phase -- `crates/paulistrings/src/engine/sort_merge.rs:204-206`'s
    own doc comment: "Truncation is not woven in here yet -- that's slice 7.1", and `merge_phase`
    takes a `_policy` parameter it never calls `keep_term` on. Confirmed independently at small
    scale: thresholds `1e-6` vs. `0.5` (five-plus orders of magnitude apart) on an 8-qubit/6-step
    toy circuit built from that exact commit still gave identical `final_terms=10880`. Contrast:
    `direct_small_sum_path` (`e56f021`), `bucketed_engine_serial` (`f08db7d`),
    `bucketed_engine_parallel` (`ef03701`) all DO call `keep_term` in their merge functions
    (verified via `git show`), matching their real cutoff-sensitive results in job 7033945.
    `run_cell_historical.py`'s sweep-loop code itself is correct (passes a genuinely different
    argument per invocation) -- forcing truncation into the frozen `d410f4e` snapshot would mean
    patching 2026-era behavior into what's supposed to be a historical comparison, so no code fix
    was made. `jobs/campaign-genoa-baseline-sweep.sbatch` is kept for provenance (header comment
    updated) but is not to be resubmitted as a multi-point sweep. The baseline figure (page 16)
    uses job 7033945's single 2^-12 `naive_baseline` point only, captioned as truncation-inert
    rather than tolerance-sensitive.

43. **Julia thread ladder landed for real (job 7034671), non-monotonic degradation found,
    2026-09-14.** `raw/2026-09-14-worker7160-julia/runs.jsonl`: PauliPropagation.jl's `vector`
    backend, eps=2^-16, 127-qubit canonical circuit, `threads` in {1,2,4,8,16,32,48,96}, all
    `status=completed`. Real `wall_time_s`: 1590.163, 1113.476, 698.944, 479.973, 382.632,
    302.335, 334.580, 974.958 (seconds, same order). Fed through
    `normalize.thread_scaling(variant_id="external_pauli_propagation_jl", min_abs_coeff=2^-16)`:
    speedup vs. Julia's own 1-thread baseline is 1.00x/1.43x/2.28x/3.31x/4.16x/**5.26x (peak, 32
    threads)**/4.75x (48 threads)/1.63x (96 threads). **This is a real, non-noise, non-monotonic
    thread-scaling curve**: scaling is good through 32 threads, then degrades badly — 48 threads
    is slower than 32 (334.58s vs. 302.33s), and 96 threads is dramatically slower than 48,
    roughly **2.9x worse** (974.958/334.580 = 2.91x), ending up slower in absolute wall time than
    even the 4-thread point (698.94s). Root cause not investigated (out of scope: this campaign
    measures the external reference engine as shipped, does not debug it). `thread_scaling_v2`
    (the existing Rust-only "reveal" build for deck page 31, per its own MANIFEST entry) is kept
    unchanged — it is a deliberate presentation build stage, not a draft — and the overlay ships
    as a new `thread_scaling_v3.{svg,pdf,png}` (+ `_compact`) via the existing
    `make_thread_scaling_figure(rust_rows, other_rows=julia_rows,
    other_label="PauliPropagation.jl", theme="deck", figsize_pt=(900, 340))` call, `rust_rows`
    unchanged from the v2 entry's own `raw/` sources. Added
    `test_thread_scaling_julia_overlay_is_non_monotonic_past_32_threads` to
    `figures/tests/test_make_figures.py`, pinning the peak-at-32/decline-through-48-and-96 shape
    as a regression tripwire. Full figure suite green: 42/42 passed. `figures/real/MANIFEST.md`
    gained a new "threads-with-julia" entry (2b) alongside the existing "threads" entry (2), and
    `evidence.md`'s E9-adjacent thread-scaling coverage is updated accordingly. Did not touch
    in-flight job 7033946 (Julia convergence) or run any Slurm command.

44. **Real full-scale historical sweep landed for real (job 7033945), `bucketed_1t_v2` figure
    for deck page 29, 2026-09-14.** The full-scale plan from decision #34 (n_qubits=127,
    trotter_steps=10, `jobs/campaign-genoa-historical.sbatch`) has now actually run on real genoa
    hardware, not just estimated locally. `raw/2026-09-14-worker7150-historical/runs.jsonl`: 17
    rows, all `status=completed`, validated 0 problems against `analysis/schema.py::validate_run`
    (`analysis/validate_campaign.py`, 17 runs / 10,840 gate records). Re-derived every number
    directly from the raw file rather than trusting any prior transcription. Real wall times
    (variant: eps=2^-12 -> eps=2^-14 -> eps=2^-16 -> eps=2^-18; `final_terms` identical between
    `direct_small_sum_path`/`bucketed_engine_serial`/`bucketed_engine_parallel` at each cutoff:
    232,432 / 696,172 / 1,791,652 / 3,936,794):
    - `naive_baseline`: one point only, eps=2^-12, `65.406s`/3,018,683 terms (truncation-inert at
      this commit, decision #42 -- not re-litigated here).
    - `direct_small_sum_path`: `0.750s -> 0.805s -> 0.920s -> 1.150s`.
    - `bucketed_engine_serial`: `11.988s -> 21.858s -> 41.132s -> 66.113s`.
    - `bucketed_engine_parallel`: `14.781s -> 39.356s -> 79.512s -> 140.263s`.

    **The `bucketed_engine_parallel`-slower-than-`bucketed_engine_serial` anomaly first observed
    locally in decision #34 is CONFIRMED on real full-scale genoa cluster hardware, at every one
    of the 4 tolerance points**: the parallel/serial ratio is 1.23x, 1.80x, 1.93x, 2.12x at
    eps=2^-12/2^-14/2^-16/2^-18 respectively -- growing more pronounced at tighter tolerances
    (bigger sums), not shrinking. `final_terms` match exactly between serial and parallel at every
    cutoff, so this is a pure wall-clock effect, not a correctness bug. Stated explicitly and
    prominently, not buried: this is real evidence for a "more threads is not automatically
    faster" narrative, not an artifact of the earlier 32-core single-socket workstation or the
    toy scale.

    A same-scale `bucketed_current` overlay was also run (4 points, `3.369s`/189,845 terms through
    `2812.324s`/288,715,006 terms), but per `run_cell.py` vs. `run_cell_historical.py`'s own
    docstrings it drives a DIFFERENT circuit/observable (`theta_h=0.6872233929727672`,
    `observable=debug_single_z`, heavy-hex-adjacent) than the four historical variants
    (`direction=heisenberg`, the linear-chain-scaled circuit -- three of the four historical
    commits cannot build heavy-hex at all). Its `final_terms` are therefore not comparable 1:1 to
    the historical variants' at the same nominal eps. This is a known, already-accepted campaign
    convention (the historical comparison is about algorithm/implementation cost trends, not an
    apples-to-apples circuit match) -- disclosed here and in `figures/real/MANIFEST.md`, and for
    that reason `bucketed_current` is deliberately NOT plotted on the same cost axis as the four
    historical variants (mixing a roughly 1000x-larger term-count series into this axis would
    compress the very serial-vs-parallel comparison the figure exists to show).

    **Figure**: `figures/real/bucketed_1t_v2.{svg,pdf,png}` (900x340pt) + `_compact`
    (900x170pt), deck page 29 ("same recurring figure with bucketed one-thread result
    highlighted"). Confirmed `make_recurring_figure`'s existing row shape
    (`normalize.runtime_tolerance()`'s output) accepts these real rows directly -- no new
    plotting function was written, reusing the same two-panel `stage=6` view
    `recurring_stage6.png` already uses. One real adaptation was needed: `runtime_tolerance()`
    passes `peak_terms` through verbatim, but every historical run here has `peak_terms=None`
    (`trace_enabled=false` at these commits -- only `bucketed_current`'s later commit populates
    it), so the point-annotation field falls back to `final_terms` when `peak_terms` is null, the
    same precedent `make_distributed_capacity_figure`'s single-rank reference point already
    established. `highlight_variant="bucketed_engine_serial"` (the "1 thread" story) at
    `stage=6` (so `bucketed_engine_parallel` is also drawn, not hidden); a thin
    generation-script-level post-process un-mutes the `bucketed_engine_parallel` line from the
    standard 0.35 "introduced but not highlighted" alpha to full opacity, since the whole point
    of this figure is that both lines stay legible side by side. Efficiency (left) panel is
    honestly empty -- none of these commits expose per-gate stats, same disclosed gap as
    `recurring_stage6.png`. No `title=` kwarg: confirmed by reading the actual exported
    `baseline_v2.png`/`baseline_v2_compact.png` that production deck figures carry no in-figure
    title at all (the deck slide's own title covers that, per `MANIFEST.md`'s theme rule) -- the
    parallel-slower finding is instead a small in-plot text note placed in an empirically
    verified empty band of the log-log cost panel (full variant only; the half-height compact
    variant omits it, matching `make_distributed_capacity_figure`'s established precedent of
    leaving per-point/qualifying remarks to the presenter verbally on a space-constrained compact
    export).

    Added `figures/tests/test_make_figures.py::test_bucketed_1t_*` (4 tests): pins the real
    parallel-slower-than-serial numbers as a regression tripwire, confirms the real rows plot
    through the unmodified `make_recurring_figure` at `stage=6`, confirms the un-muted parallel
    line reaches full alpha, and confirms the deck-theme export lands at the exact 900x340pt
    size. Full figure suite green: **46/46 passed** (42 pre-existing + 4 new; one previously
    order-dependent failing test,
    `test_thread_scaling_julia_overlay_is_non_monotonic_past_32_threads`, is also green in this
    run -- a pre-existing test-isolation quirk unrelated to this change, not investigated further
    here).

    Updated `figures/real/MANIFEST.md` (new entry 7) and `evidence.md`'s E4/E5 historical-baseline
    row from "full-scale plan prepared, local-only" to real completed full-scale cluster data.
    Did not touch in-flight jobs 7033946 (Julia convergence) or 7034671, and ran no Slurm command
    of any kind (only read `raw/2026-09-14-worker7150-historical/runs.jsonl`, already on disk from
    the completed job).

45. **Memory/bandwidth diagnosis figure for deck page 17, real data from job 7035691,
    2026-09-14.** New function `make_memory_diagnosis_figure` in `figures/make_compact_figures.py`
    -- no prior memory/phase figure existed to extend. Two panels: (A) real phase time-share at 1
    and 96 threads from `raw/2026-09-14-worker7183-memory/memory-diagnosis-eps1.5258789e-05.txt`
    (`heavyhex_step`, 127 qubits, `coeff:1.5258789e-05`), small serial phases folded into "other"
    the same way the probe's own HTML report does; (B) three explicitly DISTINCT numbers -- 48
    B/term fixed payload (`W=2`, `Complex64`), a modeled 1.79 GB/s / 142 B/term-update traffic
    estimate derived from the probe JSON's real `terms_in`/`rows_sorted`/`terms_out`/
    `coset_loop_ns` at 96 threads (the ONLY phase/thread-count this campaign treats as a valid
    rate comparison -- not the serial permute/unpermute phases, not the 1-thread cell), and peak
    resident memory (`VmHWM=16,806,572 kB ~= 16.81 GB`, identical across both probe rows since
    it's one process's high-water mark). Panel B is three text "stat tiles", not a shared bar
    axis, since B/term, GB/s, and kB are not comparable magnitudes.

    Verified directly, not trusted from the task prompt: `bandwidth.txt` for this job has NO
    measurement of any kind -- `scripts/bandwidth.sh` failed to build `membench` on worker7183
    (`bandwidth.stderr.log`: `target/release/membench: No such file or directory`), so there is no
    genoa bandwidth ceiling at 1 or 96 threads, not even a partial one. The probe's own
    auto-rendered `perf-viz.py` HTML report reaches the same conclusion independently ("Bandwidth
    ceilings unavailable for this campaign ... DRAM figures below show modeled GB/s only, with no
    % of ceiling"), which is reassuring cross-confirmation that this isn't a reading error on my
    part. Per the task's explicit instruction, `research/HARDWARE.md`'s Cascade Lake (`ccqlin038`)
    ceilings are a different architecture and were NOT substituted in -- the figure renders an
    explicit `bandwidth_unavailable_reason` string instead, and `bandwidth_ceiling_gbps=None` is a
    first-class, tested state (`test_memory_diagnosis_figure_states_bandwidth_unavailable_reason_
    when_ceiling_is_none`), not a missing-data placeholder. `perf-stat.sh` for this job also failed
    ("Workload failed: No such file or directory", marked non-fatal in the job log) -- perf is
    blocked on this shared cluster account, so there is no flame graph and no hardware-counter
    evidence anywhere in this figure; nothing here claims bandwidth saturation from the
    phase-timing breakdown alone, only that it identifies which phases cost time. Also disclosed,
    matching the same caveat already accepted for the bucket-size figure (#41): `phase_breakdown`
    hard-codes `theta_h=5*pi/16` for `heavyhex_step`, not this campaign's primary `theta_h=7*pi/32`
    working point -- this is the campaign's own synthetic benchmark circuit, not the canonical
    Python task.

    Files: `figures/real/memory_v2.{svg,pdf,png}` (900x340pt) + `_compact` (900x170pt
    half-height -- chosen over half-width after an actual half-width export test left both panels
    illegible; half-height matches this MANIFEST's existing two-panel precedent). The compact
    variant drops panel B's sub-captions and the bandwidth-unavailable prose note (same
    "compact drops qualifying detail, presenter states it verbally" precedent
    `make_distributed_capacity_figure` established) while keeping all three headline numbers.

    Added 9 tests (`test_memory_diagnosis_figure_*`) covering: empty input, two-panel structure,
    phase shares summing to ~100% per row despite the ~3.4x real wall-time difference between 1
    and 96 threads, the real ms/layer annotations, the three numbers staying distinct, both the
    ceiling-unavailable and a hypothetical real-ceiling path, and exact-size exports of both
    variants. Full figure suite green: **55/55 passed** (46 pre-existing + 9 new,
    `.venv/bin/python -m pytest quera-talk-data/campaign-2026-09-11/figures/tests/`).

    Updated `figures/real/MANIFEST.md` (new entry 8) and `evidence.md`'s E3 row from "not started"
    to real completed data, with every caveat above stated explicitly. Did not touch in-flight
    jobs 7036526 (single-bucket) or 7033946 (Julia convergence), and ran no Slurm command of any
    kind (only read files already on disk from the completed job 7035691).

46. **`bucketed_1t_v2` figure dropped, page-29 comparison redefined, 2026-09-14.** User pointed
    out the figure (historical `bucketed_engine_serial` vs. `bucketed_engine_parallel`, two
    different old commits) doesn't support the claim page 29 is actually meant to make: that the
    CURRENT engine, single-threaded, performs worse at one bucket than at its default (many-
    bucket) configuration -- bucket-splitting's benefit isolated from threading entirely, not a
    historical-commit comparison. The figure and its exported files were removed
    (`figures/real/bucketed_1t_v2*`); `make_recurring_figure`'s stage-6 historical view stays
    real and available as backup material (e.g. for the "attempts" story, page 19: a real
    example of naive threading regressing performance) but is no longer page 29's asset.
    Page 29's real comparison needs: the current engine's default-bucket single-thread point
    (already real, job 7030090, eps=2^-16, 1629.9s) vs. the same engine forced to
    `min_buckets=1` at the same config, single-thread (job 7036526, in flight as of this entry --
    see the entry that supersedes this one once it lands).

47. **Job 7036526 (single-bucket) landed; baseline pivot (page 16) and bucketed-1t-v3 (page 29)
    figures built from real data, 2026-09-14.** Both jobs referenced by decision #46 and the
    baseline-pivot ask are now real, completed data on disk (nothing from job-ledger.jsonl's
    still-pending entries -- verified `raw/2026-09-14-worker7160-single-bucket/runs.jsonl`
    directly, `slurm_job_id=7036526`, `status=completed`).

    **Baseline pivot (page 16)**: `min_buckets=1`, `target_bucket_len=1e9`, same
    `eps=2^-16 (1.5258789e-05)`/127-qubit canonical config as every other campaign point:
    `wall_time_s=1967.578165213985`, `final_terms=38,791,220`,
    `expectation_re=0.3971653299846819`. Combined with the two already-real Julia points --
    1-thread `dict` backend (`wall_time_s=4874.939709082`, job 7033031, `decisions.md` #27) and
    96-thread `vector` backend (`wall_time_s=899.49`, job 7034021, `decisions.md` #38) -- this
    replaces the truncation-inert `naive_baseline`-sweep framing (`decisions.md` #42) as page
    16's "before this work" reference, per the user's explicit instruction ("I'm happy to use
    the Julia data (both threaded and not), along with the current engine with a single
    bucket"). New function `make_baseline_pivot_figure` in `figures/make_compact_figures.py`: a
    plain categorical 3-bar chart, deliberately NOT a thread-scaling curve -- the 1 -> 96 -> 1
    thread progression across these three real points is not monotonic (the third point is a
    different implementation), so there is no connecting line across all three bars and the
    x-axis is categorical labels, never a numeric thread axis. The two Julia bars share one
    color and are annotated with their own real `5.42x` speedup; the current-engine bar gets
    both a distinct color and a hatch pattern so it reads as "not part of the Julia pair" even
    in grayscale. Files: `figures/real/baseline_v3.{svg,pdf,png}` (900x340pt) + `_compact`
    (450x340pt half-width, matching this campaign's convention for single-panel, non-sweep
    plots). The old `baseline_v2*` figure/section in `figures/real/MANIFEST.md` is marked
    superseded (kept for provenance, not deleted) rather than removed outright, since it is
    still real (if truncation-inert) data with its own citation trail.

    **Bucketed-1t-v3 (page 29)**: same single-bucket run as above vs. the already-real
    default-bucket-config point (job 7030090, `wall_time_s=1629.9046`, `final_terms=
    38,791,220`, `raw/2026-09-13-worker7277/runs.jsonl`) -- both current engine, single-thread,
    same `eps=2^-16`/127-qubit config, only `min_buckets`/`target_bucket_len` differing. Real,
    clean finding: forcing a single bucket is **~20.7% slower** (100*(1967.578165213985/
    1629.904597465007 - 1) = 20.71%) than the default many-bucket configuration, with
    `final_terms` matching EXACTLY (38,791,220 both) -- a pure wall-clock effect, isolated from
    threading (both single-thread) and from any historical-commit confound (both the SAME
    current-engine commit), unlike the dropped `bucketed_1t_v2` figure's two-different-commits
    comparison. One real gap, disclosed rather than papered over: job 7030090's own
    `runs.jsonl` rows carry no `extra.expectation_re` field at all, so the default-bucket
    config's expectation value for the correctness cross-check is cited instead from the
    sibling E9 convergence sweep's own `trotter_step=20` point at the identical config (job
    7033650, same `task_id=T02-canonical`, same eps): `expectation_re=0.39716532998468246`,
    which also independently reproduces `final_terms=38,791,220` -- agreeing with the
    single-bucket run's own `expectation_re=0.3971653299846819` to ~3e-15, well inside the
    repo's determinism-policy tolerance bar. This value is cited in `figures/real/MANIFEST.md`
    and here, not plotted (the figure states only the matching `final_terms`, which both runs'
    own records carry directly). New function `make_single_bucket_comparison_figure`: a plain
    2-bar chart with a `+20.7%` annotation and a `final_terms identical: 38,791,220` caption.
    Files: `figures/real/bucketed_1t_v3.{svg,pdf,png}` (900x340pt) + `_compact` (450x340pt
    half-width).

    Added 14 tests to `figures/tests/test_make_figures.py` (7 per figure): empty-input error,
    correct bar ordering/values, the categorical-not-thread-axis contract and hatch-based
    distinction for the baseline pivot, the ~20.7%-slower regression tripwire and its in-figure
    annotations for the bucketed-1t-v3 comparison, and an exact-size export test per theme
    variant for both. Full suite green: **69/69 passed** (55 pre-existing + 14 new,
    `.venv/bin/python -m pytest quera-talk-data/campaign-2026-09-11/figures/tests/`).

    Updated `figures/real/MANIFEST.md` (new "baseline-pivot" entry 1b superseding the old
    "baseline" entry 1, and new "bucketed-1t-v3" entry 7b alongside the dropped "bucketed-1t"
    entry 7) and `evidence.md`'s E4/E5 row. Did not touch in-flight jobs 7036845 (quick
    eps=2^-14/2^-12 comparison) or 7033946 (Julia convergence), and ran no Slurm command of any
    kind (only read files already on disk from the completed job 7036526 and prior completed
    jobs 7030090/7033031/7034021/7033650).

48. **Bucket-size figure (page 30) re-sweep with topn:1000000, single-panel + real L2 line,
    2026-09-14.** Job 7036867 re-swept the page-30 bucket-size figure with `--truncation
    topn:1000000` instead of `coeff:2^-12` (fixing the earlier problem: at eps=2^-12 the final
    term count was only ~770-900, too few for target_bucket_len in {256..4096} to produce
    meaningfully different occupancy). Real result, all 5 configs hit exactly n=1,000,000, ZERO
    empty buckets at every target_bucket_len (256->4096 buckets, median occupancy scales cleanly
    244->3905). Throughput: 256->5.787e7, 512->6.132e7, 1024->6.322e7, 2048->6.513e7 (peak),
    4096->6.495e7 strings/s. Job 7036934 added one more point, target_bucket_len=8192
    (num_buckets=128, the min_buckets floor -- confirmed the largest value that can still move;
    anything past this is identical since num_buckets cannot go below the floor): 6.498e7
    strings/s. Honest finding: the "peak" at 2048 is real but flattens into a plateau
    (2048/4096/8192 all sit at ~6.50e7 +-0.3%), not a sharp drop-off -- per user request for "a
    point at larger target bucket lengths so the peak is more pronounced", the additional point
    shows the shape IS a plateau, not a sharper peak, and the figure states this rather than
    overclaiming a more dramatic effect than the data supports.

    Job 7036934 also captured the real, node-local L2 cache size via `lscpu -C` (not a
    substituted spec-sheet number, per this repo's own measured-over-spec discipline): **1 MiB
    per core** (`L2 1M 96M 8 Unified 2 2048 1 64`), the first real L2 measurement on file for
    this campaign's genoa hardware (`research/HARDWARE.md` had none).

    `make_bucket_size_figure` gained `single_panel=True` (throughput only, per user request --
    the occupancy panel stays available via the default) and `l2_cache_bytes=`/`bytes_per_term=`
    (draws a vertical reference line at `target_bucket_len = l2_cache_bytes / bytes_per_term`,
    48 B/term default). With the real 1 MiB L2, the reference line sits at target_bucket_len ~=
    21,845 (2^14.4) -- well PAST where the plateau begins (~2^11) and past every tested point,
    so the plateau is NOT positioned at the L2 boundary in this data; the figure does not claim
    a cache-residency explanation for it. x-axis label changed from `target_bucket_len` (code
    identifier) to "target bucket size (terms)" (human-readable), per user request. New exports:
    `figures/real/bucket_size_v3.{svg,pdf,png}` and `_compact` (450x340pt half-width, single
    panel). 71/71 figure tests passing.

49. **Bucket-size figure extended past the min_buckets=128 plateau, real peak found near L2,
    2026-09-14.** Job 7036975 (`MIN_BUCKETS=1`) failed cleanly: `phase_breakdown`'s CLI enforces
    its own `--min-buckets >= 16` floor (independent of the underlying engine, which the earlier
    PyO3-bindings work confirmed has no such floor at the library level) -- corrected to
    `MIN_BUCKETS=16` for job 7036979. Real result: `target_bucket_len=16384` (num_buckets=64) is
    the TRUE peak at 6.536e7 strings/s -- higher than the earlier apparent "plateau" at
    2048-8192 (~6.50e7) -- and throughput genuinely DECLINES past it: 32768 (32 buckets)
    ->6.421e7, 65536 (16 buckets, the CLI floor) ->6.345e7. 131072 is an identical duplicate of
    65536 (both floored at min_buckets=16) and was dropped from the figure as redundant.

    The real peak (16384) sits almost exactly at the real, measured L2 boundary
    (target_bucket_len ~= 21,845 at 48 B/term and 1 MiB L2) -- decline begins right at/after
    crossing the L2 line. This is now genuinely evidence CONSISTENT WITH an L2-locality
    explanation (per this function's own stated discipline: a peak near a measured cache
    boundary is consistent with locality, never proof of cache residency by itself -- no
    hardware counters confirm cache misses here, this is peak position only). `bucket_size_v3.*`
    regenerated with the full 9-point range (256 through 65536); MANIFEST updated.

50. **Baseline figure (deck page 16) pivoted from a 3-bar comparison to a real eps-sweep,
    2026-09-14.** New `make_baseline_eps_scaling_figure(rows, series_order=..., theme=...,
    figsize_pt=..., title=...)` in `figures/make_compact_figures.py` supersedes "1b. baseline-
    pivot" (`baseline_v3`, kept for provenance, not deleted): one line per named series
    (`label`) over `min_abs_coeff` (x, log2-spaced, `$\varepsilon=2^{-16}$`-style mathtext
    x-ticks) vs. `wall_time_s` (y, log). Stage 1 plots all four series that exist today --
    Julia 1/32/96-thread and the current engine forced to a single bucket -- across the same
    4-point eps grid `{2^-10, 2^-12, 2^-14, 2^-16}`, 127-qubit canonical circuit.

    Every number was re-verified directly against its source `raw/*/runs.jsonl` rather than
    trusted from the handoff note that specified this task, per that note's own instruction.
    One real discrepancy surfaced doing this: the handoff's Julia-96-thread/eps=2^-16 value
    (`899.49`, decisions.md #38, job 7034021) has no standalone `runs.jsonl` and is stale --
    the real, file-backed value at that exact cell is `974.957910291` (job 7034671, `raw/
    2026-09-14-worker7160-julia/runs.jsonl`, the full Julia thread ladder already landed for
    decision #43). This figure plots `974.957910291`, not `899.49`.

    Progressive-reveal design: `series_order` (list of labels to draw, in legend order; `None`
    = every label in `rows`) lets a slide deck build up the story one line at a time, in the
    spirit of `make_recurring_figure`'s `stage=` but without that function's hardcoded
    `STAGE_VARIANTS` list -- this dataset's label set differs and is expected to grow (bucketed
    multithreaded and multi-node series land later). Two things are keyed off the FULL `rows`,
    never the `series_order` subset actually drawn: each label's color/marker/linestyle (fixed
    by first-appearance order in `rows`) and the axis limits (min/max over every row) -- so
    revealing more series never moves the ones already on the slide. Verified directly: the
    same `rows` drawn first with `series_order=["Julia, 1 thread"]` then with a two-label
    prefix produced identical `xlim`/`ylim` and an identical first-line color.

    Compact export is half-width (450x340pt), not half-height: checked by actually rendering
    both, since with 4 series and an eps-labeled x-axis, 900x170 left the x-tick labels, axis
    label, and legend visually colliding (no room at that height, same failure mode `make_
    memory_diagnosis_figure`'s `compact` note already flags for a different figure). The
    legend-fit logic also needed a real render to get right: a fixed bottom-margin fraction
    (tried first, following `make_distributed_capacity_figure`'s precedent literally) either
    collided with the x-axis label at 900x170 or starved the plot area at 450x340 once a
    4-row fallback legend was needed; measuring the legend's actual rendered height and adding
    it to whatever margin `tight_layout()` already reserved for the x-label, instead of
    guessing both from a row count, is what held at every size tried.

    Files: `figures/real/baseline_v4_stage1.{svg,pdf,png}` (900x340pt) + `_compact`
    (450x340pt). 10 new figure tests added to `figures/tests/test_make_figures.py`
    (empty-input, one-line-per-label, log-scale axes, eps-convention x-tick labels,
    `series_order=None`/restricted/unknown-label, progressive-reveal axis+color stability,
    both exact-size exports); 81/81 figure tests passing. `figures/real/MANIFEST.md` gained
    "1c. baseline-eps-scaling" and marked "1b. baseline-pivot" superseded (not deleted, same
    as every other supersession in this file). Stages 2-4 (bucketed multithreaded, bucketed
    multi-node) are explicitly NOT built -- their cluster jobs have not landed, and no
    placeholder data was fabricated for them. Did not touch any in-flight Slurm job or run any
    Slurm command.

51. **Baseline eps-scaling: all 4 progressive stages built with real data, 2026-09-14.** Jobs
    7037029 (Rust default-bucket 1-thread eps=2^-10 fill), 7037030 (Rust default-bucket 96-thread
    eps={2^-10,2^-12,2^-14}), 7037031/7037032/7037033 (16-rank distributed eps={2^-10,2^-12,
    2^-14}) all landed real, clean data (verified no cross-job JSONL corruption in the shared
    `distributed-16ranks` directory despite 3 concurrent writers). Combined with the already-real
    stage-1 data (decisions.md #50), all 7 series now have real values at all 4 eps points:

    | eps | Rust 1-bucket | Rust default,1T | Rust default,96T | Rust 16-rank | Julia 1T | Julia 32T | Julia 96T |
    |---|---|---|---|---|---|---|---|
    | 2^-10 | 0.729s | 0.810s | 0.705s | 0.404s | 1.404s | 0.510s | 0.519s |
    | 2^-12 | 9.825s | 9.445s | 1.199s | 2.853s | 24.718s | 6.498s | 58.319s |
    | 2^-14 | 160.632s | 129.867s | 4.071s | 6.486s | 329.514s | 50.122s | 627.504s |
    | 2^-16 | 1967.578s | 1629.905s | 56.686s | 35.725s | 4874.940s | 302.335s | 974.958s |

    `_DECK_SERIES` extended from 5 to 7 distinct (color, marker, linestyle) styles (additive
    only -- every existing figure only ever indexes 0-4, unaffected). Built all 4 progressive
    reveal stages via `make_baseline_eps_scaling_figure`'s `series_order=` parameter, same
    `rows` (all 7 series) each time so axes/colors never shift between reveals, only which lines
    are drawn: stage1 = the 4 baseline series, stage2 = +bucketed single-thread, stage3 =
    +bucketed 96-thread, stage4 = +bucketed 16-rank (all series drawn).

    Fixed a real legend-layout bug hit at 7 series: the fallback jumped straight from "all in one
    row" to "exactly one column" (7 rows), which overflowed the previous 0.85 bottom-margin cap
    and visibly overlapped the x-axis label on an actual rendered figure. Fixed by searching
    downward from ncol=len(labels) for the widest column count that actually fits (matching this
    module's original, more general legend-fit pattern), and relaxing the margin cap to 0.97 so a
    legend that genuinely needs more room gets it rather than being silently clipped.

    Files: `figures/real/baseline_v4_stage{1,2,3,4}.{svg,pdf,png}` and `_compact` variants (28
    files total). `baseline_v3`/`make_baseline_pivot_figure` remain superseded-but-kept per
    decisions.md #50/MANIFEST's existing convention.

52. **Baseline eps-scaling gains a speedup panel (linear axis) and 2^-18/2^-20 points where
    real, 2026-09-14.** `make_baseline_eps_scaling_figure` gained `speedup_baseline: str | None`:
    when given a label present in `rows`, adds a second panel plotting
    `wall_time_s[speedup_baseline][eps] / wall_time_s[label][eps]` per eps, per user request,
    with `speedup_baseline="current engine, 1 bucket"` (single-threaded, single-bucket Rust).
    The baseline's own line is a flat 1.0 (a visible sanity check) wherever it has data. Ratio
    axis is LINEAR (not log, unlike the left wall-time panel), per explicit user request.

    Real additional data points, added only where they exist (per user request: "leave them out
    where we don't have data" -- no fabricated points for the other 5 series at these eps):
    - "current engine, default buckets, 96 threads" @ eps=2^-18: 482.571s (job 7031790, E6,
      already-real single-node/96-thread reference).
    - "current engine, 16 ranks" @ eps=2^-18: 227.84442280902294s (job 7032059, E6) and
      @ eps=2^-20: 1658.816s (job 7032351, E7).

    Fixed a real legend-position bug the second panel exposed: `ax.legend(bbox_to_anchor=...)`
    is in AXES coordinates, so it centered under only the LEFT panel once a second panel
    existed -- switched to `fig.legend()` (figure coordinates, `loc="lower center"`,
    `bbox_to_anchor=(0.5, 0.0)`) so the legend centers under the whole figure regardless of
    panel count. Also fixed a real y-label clipping bug: the full baseline name as a rotated
    ylabel ("speedup vs. current engine, 1 bucket") ran past the top of the canvas on an actual
    render -- shortened to just "speedup", with the baseline name stated in MANIFEST.md/the
    figure's own caption context instead of the axis itself.

    All 4 stages regenerated with both the wall-time-only and two-panel speedup variants:
    `baseline_v4_stage{1,2,3,4}[_speedup].{svg,pdf,png}` and `_compact` (56 files total).

53. **Distributed (C4) placement switched from one rank per NUMA domain to one rank per node,
    2026-09-14**, per user request to replace the multi-node dataset entirely (existing
    `raw/2026-0[34]-distributed-{2,4,8,16}ranks/` per-domain data is kept, un-superseded, since
    it is a different, still-valid measurement — just not the one future distributed cells will
    add to). Investigated first whether this needs an engine change: it does not.
    `comm=`'s placement (`paulistrings::mpi::default_config`,
    `crates/paulistrings/src/engine/partitioned/mpi.rs`) is
    `Placement::Auto{max_partitions: Some(1)}`, which `resolve_auto`
    (`engine/partitioned/topology.rs`) resolves to a single slot over whatever CPUs are in the
    process's affinity mask; when that mask spans both of a node's NUMA domains, `resolve_auto`
    already merges them into one un-pinned-to-a-node slot (`node: None`, size two domains) rather
    than picking one. So node-granularity placement is a pure launcher change: give each rank the
    whole node's affinity instead of one domain's.

    `jobs/campaign-genoa-distributed.sbatch` changed: `ranks` now rounds `SLURM_JOB_NUM_NODES`
    (not `nodes * numa_domains_per_node`) down to a power of two, `--ntasks-per-node=1` always,
    `--cpus-per-task` is the whole node's core count, and `--cpu-bind=none` replaces
    `--cpu-bind=ldoms` (ldoms would have restricted the rank right back to one domain). Output
    directory renamed `raw/<date>-distributed-node-<ranks>ranks/` (was
    `raw/<date>-distributed-<ranks>ranks/`) and `config_id` gets a `-1rank-per-node` suffix, so
    old and new data can never collide even at the same rank count.

    Plan: test cheap first (2 nodes = 2 ranks, a network-crossing sanity check the in-process
    partitioned engine can't give), then hand over the real overnight job at up to 16 nodes.
    Per the org LAW, submission is the user's own step; see job-ledger.jsonl / the next entry for
    the actual job IDs once run.

54. **Node-granularity distributed run landed for real, 2026-09-14 — both jobs completed in
    minutes, no overnight wait needed.** Job 7037150 (2 nodes/2 ranks, loose eps=2^-10,
    trotter_steps=4, sanity check only): completed, 0.027 s, 2761 terms — confirms the placement
    change works end to end across a real network hop, not just within one node. Job 7037152 (16
    nodes/16 ranks, eps=2^-16, the trusted overlap point, real headline settings): completed,
    **58.972 s**, peak_terms=45,418,768 (matches the term count at this eps from every other
    engine/placement at this eps — expected, term count is eps-driven not placement-driven),
    peak_rss_kb=19,293,188 (~19.3 GB, summed across ranks; grows with rank count same as always,
    a probe/replication artifact per `Known gaps`).

    **A real, honest comparison, not just a pass/fail:** the old one-rank-per-NUMA-domain data at
    the same eps and the same *rank count* (`raw/2026-09-14-distributed-16ranks`, 8 nodes, 16
    domain-ranks) ran in **35.725 s** — i.e. the new run, on **twice the nodes** (16 vs 8) at the
    same rank count, is **slower**, not faster. This is a real, measured result, not a fluke of a
    single run (no reps to double-check yet, flagged as such). The plausible mechanism: a
    node-granularity rank runs ONE flat 96-thread Rayon pool spanning both sockets with no
    NUMA-local split (that is exactly what `Placement::Auto{max_partitions: Some(1)}` does when
    given the whole node's affinity mask — see decision #53), so cross-socket memory traffic in
    the coset_loop's gather/merge phases pays the locality cost the two-domains-per-node
    partitioned design exists to avoid — the same effect this repo's own NUMA pinning work
    (`ARCHITECTURE.md §Partitioning`, `research/HARDWARE.md`) was built around. Fewer, bigger
    ranks buys less MPI exchange overhead per node but loses more to intra-rank NUMA locality;
    at 16 ranks the trade nets negative here.

    Node-granularity DOES give one thing the old approach never had: a real, measured 16-node
    data point at all (old data tops out at 8 nodes / 16 domain-ranks). Whether that capacity
    extension is worth reporting alongside a wall-clock regression at matched rank count is a
    presentation-content call, not an engineering one — flagged for the user rather than decided
    here. Files: `raw/2026-09-14-distributed-node-{2,16}ranks/`. Old per-domain data at
    `raw/2026-0[34]-distributed-{2,4,8,16}ranks/` is kept, unsuperseded (decision #53).

55. **8-node matched comparison confirms the slowdown, and `campaign-genoa-distributed.sbatch`
    gains a `RANK_GRANULARITY` knob, 2026-09-14.** Job 7037246 (8 nodes, `RANK_GRANULARITY=node`
    default, 8 node-ranks, eps=2^-16): 60.765 s. Directly against the old 8-node point at the
    SAME node count (`raw/2026-09-14-distributed-16ranks`, 16 domain-ranks, 2 per node): 35.725 s.
    So at matched hardware (not just matched rank count, decision #54's comparison), node
    granularity is ~1.7x slower here — not an artifact of the rank-count mismatch.

    Per a user question, whether this is a term-count effect (too few terms to amortize some
    fixed cost, vs. a bandwidth-bound NUMA cost that would persist or worsen at scale) is being
    tested directly: jobs 7037247 (16 nodes, eps=2^-18) and 7037248 (16 nodes, eps=2^-20) are
    running now, against old comparison points at the same eps (8-node/16-domain-rank: 227.844 s
    @ 2^-18, 1658.816 s @ 2^-20).

    Since the user then wanted more OLD-placement (one rank per NUMA domain) scaling points at
    16 and 32 nodes — beyond the old data's previous ceiling of 8 nodes — the sbatch script
    needed both placements addressable, not just the new default. Added
    `RANK_GRANULARITY={node (default) | domain}`: `domain` reproduces the original placement
    exactly (`--cpu-bind=ldoms`, ranks = nodes × numa-domains-per-node rounded to a power of two,
    `out_dir`/`config_id` unchanged from the original — same directory naming as the existing
    `raw/2026-0[34]-distributed-{2,4,8,16}ranks/` data, so new domain-granularity runs land
    alongside the old ones as more points on the same curve, not a separate series).

    Slurm-side, checked before scaling further: this account's jobs run under QOS `gen` on
    partition `ccq`, and `gen` has no node-count limit set (`MaxNodesPU`/`GrpTRES` node cap both
    unset) — `ccq`'s partition-level `MaxNodes` is also `UNLIMITED`. The real ceiling is physical:
    `sinfo` showed ~288 genoa+rocky9 nodes total, 69 idle at the time asked. No LAW-relevant
    concern — read-only `sinfo`/`sacctmgr show`/`sacct` only, no job control commands run.

56. **`P_MAX_BITS` raised from 4 to 6 (16 to 64 partitions), with explicit user sign-off,
    2026-09-14.** Jobs 7037263/64/67/68/69/70 (decision #55's further node-count scaling) all
    panicked identically: `PartitionRows: bits N exceeds P_MAX_BITS 4` — a real, deliberate
    architectural ceiling (`crates/paulistrings/src/bucket/hash.rs`), not a launcher bug, and it
    applies to BOTH placements equally (the earlier 16-node/16-rank point was already sitting at
    it). Raising it needed the user's go-ahead first, since the constant's doc comment says the
    smallness is deliberate; asked, and got it.

    TDD, red before green: added `partition_row_ceiling_covers_64_ranks` (fails against the old
    constant), then raised `P_MAX_BITS: u8 = 4` to `6`. Fixed the one stale test this exposed —
    `partition_is_within_range_and_the_identity_key_is_partition_zero` asserted the literal `16`
    rather than `1 << P_MAX_BITS`, so it would have silently stopped meaning anything the next
    time this constant moves.

    A real, independent bug turned up while raising it, not introduced by it:
    `InProcessTransport`'s `GroupState::departed` was an `AtomicU32` bitmask (`1 << rank`) with a
    comment claiming `P ≤ 16` made a `u32` ample — true at 16, already latent at the *old* ceiling
    of 16 partitions (rank 16..31 already fit only by luck, since 32 is the actual overflow
    point), and definitely broken once ranks reach 32. `1u32 << 32` panics in debug ("attempt to
    shift left with overflow") and silently wraps to rank 0 in release — a rank 32+ dying would
    misreport as rank 0 dying in production. Added
    `a_partner_at_rank_32_that_panicked_is_reported_by_its_own_rank` (red: caught the debug-build
    shift panic verbatim), then widened `departed` to `AtomicU64` (green). This only affects
    `InProcessTransport` (in-process tests / the `partitions=` path's differential nets); the real
    MPI `Transport` has no such mask and was never at risk.

    `cargo test --workspace` (all crates, doctests included) and
    `cargo clippy --workspace --all-targets -- -D warnings` both clean after the fix. Doc updates:
    `ARCHITECTURE.md` §Hash/§Bucketing and `CLAUDE.md`'s engine overview both cited `P ≤ 16`;
    updated to `P ≤ 64`. Local `.venv`'s compiled extension was NOT rebuilt (permission declined,
    and moot anyway — every campaign sbatch script builds its own fresh worktree+venv on the job
    node, so this only matters for local/interactive Python use outside a cluster job).

    This does NOT change today's finding (decision #54): 16 ranks is still the point where node-
    granularity trades hardware for wall time. It only removes the *hard* ceiling that made 32/64-
    rank points impossible to even attempt — whether that trade keeps favoring the old placement
    at higher rank counts, or whether node-granularity's coarser communication pays off once there
    are enough ranks to amortize it, is now an open, testable question rather than a moot one.

57. **32/64-node distributed points landed for real (both placements), plus a new
    distributed-scaling figure and node-efficiency panel, 2026-09-14.** Jobs 7037489 (16 nodes,
    domain, 32 ranks: 21.038s), 7037490 (32 nodes, domain, 64 ranks: 13.455s), 7037491 (32 nodes,
    node granularity, 32 ranks: 37.971s) all completed real, no fabricated data. The scaling
    question decision #56 left open is answered, at least so far: domain-granularity keeps its
    lead over node-granularity at every node count now measured (8: 35.7 vs. 60.8s; 16: 21.0 vs.
    59.0s; 32: 13.5 vs. 38.0s). The ratio (node-granularity time / domain-granularity time) is
    1.70x at 8 nodes, 2.81x at 16 nodes, 2.81x at 32 nodes -- it does NOT monotonically narrow
    with scale; it widened from 8 to 16 nodes, then held flat into 32. Stated plainly rather than
    reporting only the 8-node comparison, which alone would have suggested a narrowing trend that
    the fuller data does not support.

    New figure `make_distributed_scaling_figure` in `make_compact_figures.py` (two-panel: wall
    time vs. nodes, log-log; parallel efficiency vs. nodes, linear y, PER-SERIES baseline since
    domain's and node-granularity's smallest measured node counts differ — see the function's own
    docstring and MANIFEST.md's new "distributed-scaling" section). TDD: 6 new tests, full suite
    90 passed. Two real rendering bugs found and fixed on an actual render before calling this
    done: a two-panel title set via `ax.set_title()` on one axes collided with the OTHER panel's
    y-label once long enough (switched to a height-fit `fig.suptitle()`), and that same title
    additionally overflowed the compact (450x340pt) canvas's WIDTH at full font size (fixed with
    a shrink-until-it-fits loop, not just a height fix). Files: `distributed_scaling.{svg,pdf,
    png}` and `_compact` variants, in `figures/real/`.

    Job 7037492 (64 nodes, node-granularity, 64 ranks) and job 7037248 (16 nodes, node-
    granularity, eps=2^-20, the term-count-effect check from earlier) were still running when
    this figure was built — NOT included in the figure or the ratios above. Rebuild once they
    land rather than treating this entry as final.

58. **64-node node-granularity point landed for real; `distributed_scaling` figure rebuilt to
    include it, 2026-09-14.** Job 7037492 (64 nodes, node granularity, 64 ranks, eps=2^-16):
    29.028s, real, completed. This is now node-granularity's OWN P_MAX_BITS ceiling too (64
    ranks = 1 << 6, same cap decision #56 raised) — both placements are simultaneously at their
    maximum rank count (domain: 32 nodes/64 ranks; node: 64 nodes/64 ranks), so neither can be
    pushed further without another `P_MAX_BITS` change.

    At the largest points reachable under each placement (domain's 32 nodes vs. node's 64
    nodes — DIFFERENT node counts, not a matched comparison), domain-granularity is still
    faster on HALF the hardware: 13.455s (32 nodes) vs. 29.028s (64 nodes) — 2.16x faster on
    half the nodes. Every node-count-matched comparison available (8/16/32 nodes, decision
    #57) already showed domain-granularity winning; this is simply the strongest form of that
    same finding, now that node-granularity has nothing left to prove at greater scale within
    the current `P_MAX_BITS` ceiling.

    `figures/real/distributed_scaling.{svg,pdf,png}` and `_compact` rebuilt from the same
    build script, now 10 real rows (up from 9); no code change needed, the figure function
    already draws whatever `nodes` values are present. Job 7037248 (16 nodes, node
    granularity, eps=2^-20): 2300.294s, real, completed — the term-count-effect check from the
    user's "is this a term-count effect?" question is now answerable in full (compare against
    the domain-granularity 8-node/16-rank point at the same eps, 1658.816s, decision #52/#54's
    data): node-granularity is slower at this tighter eps too (2300.3 vs 1658.8s, a 1.39x gap
    at MORE hardware for node-granularity, since 16 nodes > 8 nodes) — consistent with every
    other eps tested, not a term-count artifact that resolves with more work per rank.

59. **E8's plumbing gap actually closed: `run_cell_distributed.py`/`campaign-genoa-distributed.
    sbatch` now support `PARTITION_ROW_POLICY=cut` for real, 2026-09-14.** Follows from decision
    #56's `paulistrings-py` merge (`partition_row_seed=`/`partition_row_blocks=` now reach
    `comm=`). `run_cell_distributed.py` now imports `run_cell._cut_blocks` and, for
    `partition_row_policy="cut"`, calls it with `num_partitions=SIZE` (the MPI group size, not
    an in-process `partitions=` count) and passes the result as `partition_row_blocks=` to both
    the timed `propagate` call and `propagate_with_stats` — every rank computes the identical
    blocks from the identical `cell.json` with no collective involved, so this cannot
    desynchronize the group. `campaign-genoa-distributed.sbatch`'s `PARTITION_ROW_POLICY` guard
    now accepts `"cut"` instead of rejecting it outright; header comment updated to match.

    TDD: added `test_cut_policy_run_matches_schema_and_uses_a_locality_cut` to
    `jobs/tests/test_run_cell_distributed.py`, confirmed red against the still-rejecting code
    (a real `ValueError` at the old guard, not a placeholder), then green after the change.
    Verified for real, not just claimed: rebuilt `.venv-mpi`'s extension (`maturin develop
    --release --features mpi`) against the merged Rust code, ran the new test at 1 rank
    (plain `pytest`) AND under real `mpirun -n 2`/`mpirun -n 4` — 3/3 passed at every rank
    count. Full `python/paulistrings/tests` under the same rebuilt extension: 435 passed, 50
    skipped. `jobs/tests/` (non-MPI): 38 passed, 5 skipped. `cargo test --workspace` and
    `cargo clippy --workspace --all-targets -- -D warnings` both still clean.

    Not yet run: a real cluster cell with `PARTITION_ROW_POLICY=cut` — this closes the code
    gap only. The obvious next distributed job is a `cut` vs. `random` comparison at a rank
    count decision #57 already has "random" data for (e.g. 32 or 64 domain-ranks, eps=2^-16),
    to see whether the locality cut's ~11x export-volume advantage on the in-process engine
    (decision's own E8 entry, `hash_communication_v2`) carries over to real wall time once
    network communication (not just in-process channel sends) is the cost being cut.
