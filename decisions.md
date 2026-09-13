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
12. **Open item before trusting real genoa data**: T07's CPU fingerprint table (`jobs/preflight.py`) is MEDIUM
   confidence from secondary sources and cannot distinguish Genoa from Bergamo (both family 25h). The first
   real job on a genoa allocation should capture its actual `/proc/cpuinfo` `family`/`model` and either confirm
   the table or correct it — do not trust `hardware_valid=true` on the real allocation until that's done once.
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
16. **T10 (`campaign.json`/`README.md`/`evidence.md`/`reproduce.sh`) written by the main agent directly**,
   not delegated — small, mostly-documentation glue work, and it needed the accumulated context of every
   prior task's exact output paths and caveats, which a fresh agent would have had to re-read anyway.
   `evidence.md` marks every E0-E9 row honestly as `tooling-ready` or `blocked` with its precise prerequisite;
   no invented numbers anywhere, per the handoff's explicit prohibition.
14. **T08 (timing-semantics review) done by the main agent directly**, not delegated: it requires judgment
   across files already in this agent's context (T04's engine changes) plus the newly-delegated T06/T07 output,
   so a fresh subagent would cost more in context-transfer than it saved. No correctness issues found beyond
   the two gaps already logged (genoa fingerprint confidence, hash_communication placeholder).
9. **Caught and fixed a real regression before it shipped**: the direct/small-sum path (`engine::direct`) has
   its own layer loop, separate from the sorted engine's, and initially had no gate-trace wiring at all — so
   `engine="auto"`/`"direct"` runs recorded only the sorted-suffix layers, truncating `PropagationStats.layers`
   from 76 to 7 on the existing `test_per_layer_term_counts_match_between_engines` pytest. Fixed by mirroring
   the same want_timer/record_gate_trace wiring into `direct.rs::run_direct_prefix`, and pinned with a new
   assertion block in `small_sum_path.rs::assert_engines_agree` so it can't silently regress again.
