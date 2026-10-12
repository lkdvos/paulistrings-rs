# Guided review of `crates/paulistrings` — decision log

**Delete before merging.** Its substance moves into the PR description.

## Process

Chunk by chunk: Claude presents purpose, reading order, design decisions, proposals and questions; the user decides; agreed items are implemented by a subagent as verified commits on branch `paulistrings-rust-review`, recorded here as decisions with "As applied" SHAs.
Verification per commit: `cargo fmt --check`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`; behaviour-preserving changes also get an old-vs-new equivalence check against the base commit (to `assert_terms_close` tolerance, per the determinism policy); performance claims need a same-node A/B (`scripts/ab-compare.sh`) submitted by the user.

## Global map

Base commit: `89bdcca`. ~24k production lines, ~19.5k in-source test lines, 6.5k integration-test lines, 1.5k CUDA.

Layers (dependency direction, bottom-up):

1. `phase`, `rng`, `pauli_string` — leaf algebra.
2. `bucket/hash` (`Gf2Hash`, `PartitionRows`) → `bucket/sum` (the real `PauliSum` storage) → `pauli_sum` (re-export + `ProductBasis`/`PauliAxis`) → `accumulator` (`BuildAccumulator`).
3. `channel/*` (`Channel` trait, built-ins, `prepared::Prepared` PTM tables keyed on the hash), `circuit`.
4. `truncation/*` (`TruncationPolicy`, builtins, `BuiltinTruncation` tree).
5. `engine/{merge,coset,bucketed,direct,stats}`, `engine/mod` (`propagate*`).
6. `engine/partitioned/*` (in-process NUMA partitions, `Transport`, `DistributedSum`, `mpi`).
7. `engine/gpu/*` (CUDA, builds on partitioned's transport/runtime types).
8. Read-outs: `stabilizer`, `echo`.

Wrong-direction production edges:

- `bucket/sum` imports `stabilizer::StabilizerState` (storage layer knows a read-out).
- `bucket/hash` imports `echo::RotationAxis`.
- `truncation/tree` imports `engine::partitioned::{Collectives, PartitionedTruncation}`.
- `engine/partitioned/topology` calls `engine::gpu::device` — a cycle with gpu → partitioned.
- `pauli_sum.rs` is mostly docs; `PauliSum` lives in `bucket/sum.rs`.

Main call path: `propagate` → `propagate_with_scratch_and_options` (`engine/mod.rs:292`) → per layer `PauliSum::rebucket` → `Channel::prepare(hash, adjoint)` → `apply_layer_bucketed` (`engine/bucketed.rs:453`, cosets from `engine/coset.rs`, sort/merge in `engine/merge.rs`) → `TruncationPolicy::finalize_layer`.
Optional prefix: `engine/direct.rs` small-sum path via `EngineSelection`.

Public surface: four `propagate*` variants plus `propagate_partitioned*`, ~25 partitioned re-exports at crate root, `gpu::*` behind `cuda`, `mpi::*` behind `mpi`, `test_support` `#[doc(hidden)] pub` behind `test-utils`.

Comment density (production code, all `//` lines including docs): root 26%, bucket 18%, channel 23%, truncation 36%, engine 22%, partitioned 27%, gpu 10%.

## Chunk plan

| # | Chunk | Files | Status |
|---|---|---|---|
| 0 | Tour of the main call path | engine/mod, bucketed (skim), coset, merge, partitioned/driver (skim) | done (A/B pending) |
| 1 | Leaf algebra | phase, rng, pauli_string | done |
| 2 | Partition hash | pauli_sum/hash | done |
| 3 | Storage | pauli_sum/{storage,partition,accumulator} | done |
| 4 | Channels I | channel/mod, clifford, identity, rotation, circuit | done |
| 5 | Channels II | channel/noise, unitary, prepared | done |
| 6 | Truncation | truncation/* | done except D42 (awaiting the user's A/B) |
| 7 | Sort/merge kernels | engine/merge, engine/coset | pending |
| 8 | Bucketed layer | engine/bucketed, engine/stats | pending |
| 9 | Front door | engine/mod, engine/direct | pending |
| 10 | Read-outs | stabilizer, echo, examples | pending |
| 11 | Partitioned: placement | topology, runtime, rows, plan | pending |
| 12 | Partitioned: transport | transport | pending |
| 13 | Partitioned: layer | export, layer, backend | pending |
| 14 | Partitioned: driver | driver, trace, truncation | pending |
| 15 | Distributed | distributed, mpi | pending |
| 16 | GPU: plumbing | device, module, kernel_cache, error, columns, staging, scan, fingerprint | pending |
| 17 | GPU: layer | sum, prepared, layer, kernels/*.cu | pending |
| 18 | GPU: finalize/export | export, finalize, truncation, payload, partition | pending |
| 19 | GPU: wire | wire, wire/peer, nccl | pending |
| 20 | GPU: drivers | driver, rank, rank/affinity | pending |
| 21 | Organisation revisit | — | pending |
| 22 | Test infrastructure | test_support, tests/, in-source tests | pending |
| 23 | Benches and examples | benches/, examples/ (phase_breakdown 3k lines) | pending |

## Decisions

Conventions (2026-10-08), applied first by a preparatory pass before the chunk 0 tour; every chunk applies them afterwards.

- **D1 Comments default to absent.** Only a safety invariant, a non-obvious algorithm choice, or a contract an implementor must honour survives; private items get at most one `///` line; the six CLAUDE.md comment rules are enforced as written.
- **D2 FINDINGS citations as one-liners.** History and measurement prose goes; where a reader would otherwise "fix" code back to a rejected idea, one line `// Not X: research/FINDINGS.md §Y` stays.
- **D3 Public docs are short.** One summary line, a paragraph only for a non-obvious contract, doc examples only on the front door (`propagate`, `PauliSum`, `BuildAccumulator`, `Channel`, `TruncationPolicy`); `lib.rs` is an abstract, the quick example and a link to the mdBook.
- **D4 Tests leave the source files.** `foo.rs` keeps one `#[cfg(test)] mod tests;` line and the tests move to `foo/tests.rs` (private access kept). Move-only in the preparatory pass; trimming waits for the coverage audit in chunk 22.
- **D5 Minimal public surface.** Modules private, crate root re-exports the user API, `gpu`/`mpi` public modules behind their features, everything else `pub(crate)`; `test_support` stays `#[doc(hidden)]` behind `test-utils`; the Python bindings get narrow public replacements for internal paths they use; the four `propagate*` variants collapse. Breaking changes are acceptable.
- **D6 Organisation.** Dependencies point downward only (fix `bucket/sum`→`stabilizer`, `bucket/hash`→`echo`, `truncation/tree`→`partitioned`, `partitioned/topology`↔`gpu`); `PauliSum`, its storage, the GF(2) hash and the accumulator merge into one `pauli_sum/` folder (no separate `bucket/`); read-outs (`ProductBasis`, `StabilizerState`, echo) into `readout/`; `engine/{partitioned,gpu}` stay nested; ~800 production lines per file as a soft cap, split on real seams; moves are their own commits and sweep `ARCHITECTURE.md` citations and the CLAUDE.md layout in the same commit.
- **D7 Code-volume cuts are mechanical only.** Dead code, duplication, needless `pub`, repetitive `phase-timing` cfg blocks; feature-level cuts (e.g. `EngineSelection`/direct path, `TermTrace`/`GateTrace`, `PropagateOptions` knobs) are listed as candidates for the user, not removed.
- **D8 Correctness bar.** No behaviour change, no intended performance change; every commit fmt/test/clippy/pytest green and builds under `cuda` and `phase-timing` (and `mpi` where the modules load); small LTO layout shifts accepted, one `ab-compare` handed to the user at the end of the pass.
- **D9 CLAUDE.md and research docs.** Rewrite the testing and layout sections to match, add a short "Code organisation" section (D5, D6, D10), and trim CLAUDE.md, `research/FINDINGS.md` and `research/HARDWARE.md` to current-state essentials.
- **D10 Naming favours readability.** No abbreviations in names (`accumulator` not `acc`, `stamp` not `st`); established domain acronyms (`GF2`, `PTM`, `NUMA`, `MPI`) stay but are defined once in docs.

- **D11 Seal `Transport`.** `Transport`/`Collectives` stay public (`DistributedSum` names the transport), sealed so no external impl; `Payload`, `ChunkMap`, `ChunkWait`, `InProcessTransport` crate-private, `InProcessTransport` reachable for integration tests through `test_support`.
- **D12 Tuning constants off the root.** `DEFAULT_MIN_BUCKETS`, `DEFAULT_TARGET_BUCKET_LEN`, `TIMER_READ_OVERHEAD_NS` move to `test_support`.
- **D13 Narrow externally unused methods.** `Gf2Hash::same_rows_as`, `PartitionRuntime::wait_timeout`, `DistributedSum::scatter_with_runtime`, `ChunkMap::chunk_of_position`, `InProcessTransport::group_with_timeout`, `PauliSum::empty_with_hash` become `pub(crate)`.
- **D14 Collapse method pairs.** `propagate`/`propagate_with_options` on `PartitionedSum`, `DistributedSum`, `GpuPartitionedSum`, `GpuDistributedSum`, and the four `DistributedSum::scatter*`, each to one plain call plus one taking options.
- Feature-level cuts (D7 candidates: direct path, `TermTrace`/`GateTrace`, bucket knobs, partition-row diagnostics, GPU fault hooks) are decided in the chunk that tours them (8, 9, 14).

Chunk 0 (2026-10-09):

- **D15 `prepare` is crate-internal.** `Channel` keeps `max_fanout`, `support`, `apply`, `apply_adjoint`, `debug_name`; `prepare` moves to a crate-private extension trait, so `Prepared`/`LocalPtm`/`DeltaEntry`/`PreparedRotation` leave the public surface. Applied in chunk 5.
- **D16 One layer loop.** `propagate`/`propagate_with` run `partitioned::driver::run_layers` with one partition on the caller's Rayon pool and a trivial transport; a solo fast path keeps rebucket-before-prepare (no second `prepare` on growth layers). Gated on an A/B against `89bdcca`; if it regresses, fall back to a shared per-layer helper.
- **D17 Cut the small-sum direct path now.** Delete `engine/direct.rs`, `EngineSelection`, `small_sum_threshold`, `DEFAULT_SMALL_SUM_THRESHOLD`, Python `engine=`/`small_sum_threshold=`, `tests/small_sum_path.rs`; opt-in users lose up to 2.3× at small `m` (FINDINGS §Direct-apply path for small sums). Default users are unaffected.

Chunk 1 (2026-10-09):

- **D18 `PauliString` trims.** Derive `Ord`/`PartialOrd`/`Hash` (identical to the hand-written lex `x` then `z`); drop `unsafe impl Pod/Zeroable` (nothing casts `PauliString`); rename `mul` → `product`.
- **D19 Qubit indices are `usize`** on every public API (string constructors, `support_mask`, channel constructors).
- **D20 `Phase` stays** (exact, swap-and-negate `apply`, natural return of `mul_assign`; no phase bits in the key — the coefficient carries it and the key must be the operator); `BuildAccumulator::add_term(string, coeff)` drops the `Phase` argument (66 of 71 call sites passed `Phase::ONE`).
- **D21 `#[inline]` only where it can matter.** Remove from generic and private functions crate-wide; keep on small non-generic `pub` functions; keep `always`/`never`/`#[cold]` only with a stated reason; verified by a byte comparison of the release probe's `.text`, differences either kept with a reason or queued for A/B.

Chunk 0 follow-ups (2026-10-09):

- **D22 One per-layer DEBUG format.** `layer k/n [name]: a -> b terms, x ms (partition r/P, d remote deltas, m rows in)` at every P; the benchmark parser's anchored prefix still matches.
- **D23 One truncation trait.** `finalize_layer_partitioned(&self, local, collectives)` becomes a default method of `TruncationPolicy` (one partition: `finalize_layer`; more: panic if `finalizes_layer()` and not overridden); `PartitionedTruncation` and `SoloPolicy` deleted; `Collectives` moves below `truncation` (sealed, move-only); exact `TopN` above one partition is a run-time error checked at call start, its message noting it is not yet supported. Partitioned `TopN` stays welcome later (exact selection by refining `ApproxTopN`'s all-reduced histogram).

Chunk 2 (2026-10-09):

- **D24 `Gf2Matrix`.** One crate-private GF(2) matrix type (draw, from rows, apply, row parity, row) under both `Gf2Hash` (`{matrix, active bits, seed}`) and `PartitionRows` (`{matrix}`); public API unchanged.
- **D25** Drop `bucket_of_pauli` and `partition_of_pauli`; the `(x, z)` forms stay.
- **D26** `PartitionRows::from_rows` moves to `test_support`.
- Open design candidate for chunks 11–13: partition rows as the leading rows of one hash matrix, so remote-delta planning reuses `Gf2Span` over `p + b` bits.

Chunk 3 (2026-10-09):

- **D27 `overlap` aligns.** `PauliSum::overlap` aligns `other` to `self`'s partition like `add` (it asserted equal bucket counts, so Python `small.overlap(propagated)` hit a Rust panic); tests in Rust and Python.
- **D28 Partition management is internal.** `refine`, `coarsen`, `rebucket`, `with_hash`, `bucket(b)`, `bucket_len` become crate-private (tests via `test_support`); `num_buckets` and `hash()` stay read-only.
- **D29 Storage cleanups.** One `merge_sorted(a, b, on_equal)` two-pointer helper (overlap keeps its scalar loop); one run-merger; `with_hash` scatters then sorts per bucket instead of a global merge; one serial-or-parallel helper; `refine_bucket` takes `b` only under `debug_assertions`; accumulator sorts with `sort_unstable_by` on the derived order.

Chunk 4 (2026-10-09):

- **D30** Drop `Channel::max_fanout`; `OutputBuffer` becomes opaque (private growable storage, public `push`, crate-private constructor). Its fixed capacity only served the deleted direct path.
- **D31** `apply_adjoint` is a required trait method (the silent default was correct only for self-adjoint channels).
- **D32** (superseded by D35) shared Clifford table inversion, a real `Clifford2Q` adjoint, private table fields, no `from_table`.
- **D33** `PauliRotation` stores `generator: PauliString<W>`; `Circuit::push` asserts the channel's support lies within `num_qubits`; `Circuit` fields crate-private with `num_qubits()`/`channels()` accessors.
- **D34** `IdentityChannel` moves to `test_support`.

Chunk 5 (2026-10-09):

- **D35 One PTM built-in.** `PtmChannel` (support qubits plus compact PTM storage: permutation and signs for Cliffords, the diagonal for diagonal noise, sparse entries otherwise) replaces `Clifford1Q`, `Clifford2Q`, `GeneralUnitary1Q/2Q`, `Depolarizing`, `Dephasing`, `PauliChannel`, `Depolarizing2Q`, `AmplitudeDamping`, with named constructors; adjoint is the transpose; its `prepare` builds the delta list directly without probing `apply`. Python keeps its names. Check memory per gate and `prepare` cost per layer before/after.
- **D36 Validating constructors.** Real `assert!`s for distinct qubits and `p, γ ∈ [0, 1]` (the release-mode `Depolarizing2Q` equal-qubit hole), no public fields.

Chunk 4–5 follow-ups and chunk 6 (2026-10-11):

- **D37** `debug_name` reports constructor names (`hadamard`, `cnot`, `depolarizing`, …), so trace and Python `gate_name` strings change.
- **D38** `PtmChannel` shrinks to about 48 bytes (one-byte kind instead of a name, `u32` qubits internally).
- **D39** `unitary_1q`/`unitary_2q` assert unitarity (`‖U†U − I‖` within a small tolerance) with a clear message; Python maps it to `ValueError`.
- **D40 `Or` combines per-term filters only.** Constructing an `Or` with a layer-pass child (`TopN`, `ApproxTopN`, `CollapseSample`, or an `And` holding one) is an error (`Or::new` asserts, tuple fields private; checked `BuiltinTruncation` constructor; Python `ValueError`); `contains_exact_top_n` becomes `!supports_partitioned()`. Rejected alternatives: sequential passes (makes `Or` equal `And` for passes) and true union via statistic-plus-selection layer passes (the useful "threshold with a TopN floor" case; not pursued now).
- **D41 `TopN` and `ApproxTopN` keep exactly `n` terms** when the sum exceeds `n`: everything strictly above the boundary (the `n`-th magnitude, or the boundary octave) plus as many boundary terms as fit, chosen in bucket order (rank order across partitions, via an all-reduce of per-partition boundary counts); no wipe to empty, no overshoot. Which tied terms survive is not a semantic promise. Device paths follow the same rule.
- **D42 Per-term filter stays monomorphized unless measured otherwise.** Before unifying the generic built-ins and `BuiltinTruncation`, a probe flag runs `CoefficientThreshold` either generically or as `BuiltinTruncation::Coefficient`, and a same-node runtime-knob A/B decides: no consistent difference → one `Truncation` enum; otherwise keep the generic per-term path (and dispatch common Python trees onto it).
- **D43** Small cleanups: `TopN`'s compaction via `retain`, one `LOG_TARGET`, `contains_exact_top_n` removed per D40.

As applied: `00b4dc5` D37 (`adjoint()` toggles an `_adjoint` suffix, so `sdg` reports `phase_gate_adjoint`), `187ba3b` D38 (80 → 40 bytes; sparse table one `Box<[SparseEntry]>`; `PtmChannel::qubits()` returns an iterator), `213f089` D39 (largest `|(U†U − I)_ij|` ≤ 1e-10, NaN fails; Python checks the same quantity), `4e982f6` D40 (the `Or` check runs in `Or::finalizes_layer`, before the first layer, because `Or::new` has no `W`; `BuiltinTruncation::or`/`try_or` check at construction), `084d45e` D41 (host `retain_with_boundary_quota` over `PauliSum::retain_per_bucket`; partitioned quota via one extra length-P all-reduce; device `k_boundary_counts` + scan + quota-aware `k_retain`, `k_topn_counts` gone; `test_support::assert_runs_agree` compares count only when a layer pass cut a tie), `22d796d` D42 probe flag `--truncation-dispatch` plus `PROBE_EXTRA`/`KNOB_B` in `ab-campaign.sbatch`, `96714e0` D43, `601b391` (Python MPI test policy above the circuit's peak). Follow-up (user: also check the smallest values): `0740138`, `945f5c3` make `assert_runs_agree(policy, layers, …)` check, after one cut, sorted kept magnitudes and above-boundary terms (TopN) or the smallest kept term's octave and above-octave terms (ApproxTopN); after more than one cutting layer only the count holds (survivors evolve differently: magnitudes differ up to 3e-2, one device run straddles an octave edge), so every net gained a single-cut case. Equivalence vs `2e1b64a`, 604 cases: 404 without `TopN`/`ApproxTopN`/`Or` bitwise (44 differ only in D37 trace names); single-pass cuts keep exactly `n` with every above-boundary term bitwise; multi-layer cuts diverge only after a layer where the old rule kept fewer than `n`. 762 workspace, 893 cuda, 504 pytest, MPI 2/4 ranks green.

### As applied, chunks 4–5

`38353a4` D35+D36 (`PtmChannel` with `hadamard`, `phase_gate`, `pauli_x/y/z`, `cnot`, `cz`, `swap`, `unitary_1q/2q`, `depolarizing`, `dephasing`, `pauli_channel`, `depolarizing_2q`, `amplitude_damping`, `adjoint()`; storage `SignedPermutation`/`Diagonal`/`Sparse`; constructors assert distinct qubits and parameters in [0, 1] incl. NaN; per-constructor tests compare the direct table bitwise against probing `apply`), `12318fc` D30+D31+D15 (`OutputBuffer<W>` owns growable columns; `apply_adjoint` required; `prepare` via a `#[doc(hidden)] prepare_for_layer(.., PrepareToken)` whose token type is unnameable outside the crate), `86cc82d` D33 (plus a new public `Circuit::push_boxed`), `8e506c6` D34.
Channel production lines 1671 → 1457. `prepare` per layer (indicative, one core): H 426 → 229 ns, CNOT 1058 → 581, dense 2Q 4224 → 1639, depolarizing 368 → 166, amplitude damping 437 → 223. Memory per gate: `size_of::<PtmChannel>()` = 80 bytes (+0 Clifford, +32–128 diagonal, +108 to +4068 sparse) vs 16–48 bytes for the old Clifford/noise structs.
`debug_name` keeps the old type names ("Clifford1Q", …) so traces and Python `gate_name` are unchanged. Python: `depolarize`, `dephase`, `amplitude_damping`, `pauli_channel` now raise `ValueError` on out-of-range or NaN parameters (10 new tests).
Equivalence vs `3fef926`: 480 cases bitwise identical (new shim per side; added S†, reversed 2Q gates, all noise parameters, random circuits). Verified: fmt/clippy matrix, rustdoc, 751 workspace tests, `cuda` 881 on the A6000, pytest 496/101, `mpi-test.sh --ranks 2,4` Rust and `--python`.

### As applied, chunk 3

`6072ee6` D27 (red→green: two Rust tests and one Python test failed with "bucket count mismatch" before the fix; `align_to` returns `Cow`, so neither `add` nor `overlap` clones an already-aligned operand), `b6f21d9` D28 (`test_support` wrappers `with_hash`, `bucket`, `bucket_len`, `refine`, `coarsen`), `e49da4b` D29 (`storage.rs` 632 → 589, `partition.rs` 228 → 205; `merge_sorted` with `keys_are_disjoint`; `merge_runs` deleted; `with_hash` scatters then sorts per bucket; `map_buckets` helper). Deviation accepted: `refine` runs `assert_invariants()` under debug instead of passing `b` to `refine_bucket` (stronger check, same debug test time). Note: on corrupted input a disjoint merge in release drops both duplicate terms where the old code kept both.
Equivalence vs `12a6979`: 404 cases bitwise identical (316 new storage cases: `with_hash` to 7 hashes ± bits, `add` over 7×7 partitions, `overlap`; full bucket contents compared). Verified: fmt/clippy matrix, rustdoc, 739 workspace tests, `cuda` 869 on the A6000, pytest 486/101, `mpi-test.sh --ranks 2,4` Rust and `--python`.

### As applied, chunks 0–2 follow-ups

`26595cf` D22; `74aa287` D23 (`TruncationPolicy::finalize_layer_partitioned` default + `supports_partitioned` fail-fast checked in `PartitionedSum`/`DistributedSum::propagate_on_backend` before anything detaches; `ApproxTopN`/`CollapseSample` call plain `finalize_layer` at one partition, so P=1 traces count fewer collectives; exact `TopN` now runs at P=1 and one-rank `DistributedSum`; `Collectives` in `src/collectives.rs`); `24a76d7` D24; `f3c2c3b` D25; `e8b6587` D26; `2e5ba53` CLAUDE.md GPU-per-rank recipe loads `python-mpi`/`uv`.
Equivalence vs `b9a4878` (harness now with partitioned P = 1, 2, 4 and hash-index cases): 88 cases bitwise identical. Verified: fmt/clippy matrix, rustdoc, 739 workspace tests, `cuda` lib 674 + device nets on the A6000, pytest 485/101, `--features mpi` 741, `mpi-test.sh --ranks 2,4` Rust and `--python`.

### As applied, chunk 1

`af7e971` D18 (derived order/hash checked against every sort, merge, radix and GPU key; `Pod` claims in ARCHITECTURE and README fixed; Python `.mul` unchanged), `be9c4c7` D19 (`usize` indices on string constructors, `support_mask`, every channel constructor and `support` field, `LocalPtm::qubits()` iterator, `PartitionRows::cut`, `PartitionRowPolicy`; `u32` kept in `LocalPtm.qubits` and GPU kernel args; Python: a qubit index ≥ 2^32 now raises `ValueError` from the bounds check instead of `OverflowError`), `52bceb4` D20 (five tests that only re-tested `Phase::apply` removed; 737 → 732 tests), `34fe078` D21 (`#[inline]` 164 → 7, `always` 6 → 0, `never` 2 → 1, `#[cold]` 2 → 1; `.text` probe 1,606,324 → 1,609,108, cdylib 3,007,018 → 3,016,714; bisection stopped by the user, partial per-file codegen map in the agent report).
Equivalence vs `b3e0b2b`: 49 cases bitwise identical. Verified: fmt/clippy matrix, rustdoc, 732 workspace tests, `cuda` 862 on the A6000, pytest 485/101, `mpi-test.sh --ranks 2,4` Rust and `--python`.

### As applied, chunk 0

`fab0549` D17: direct path, `EngineSelection`, `small_sum_threshold` and Python `engine=`/`small_sum_threshold=` removed (+195/−1721, 37 files); workspace tests 758 → 737, pytest 523 → 485 passed (101 skipped).
`bfb19b3` D16: `propagate_with` runs `run_layers` as one partition on the caller's pool via a crate-private `SoloTransport` and `SoloPolicy` (keeps exact `TopN` and `dyn` policies for `propagate`), `PartitionStorage::refine_unprepared` keeps rebucket-before-one-`prepare` at P=1 (also for `PartitionedSum` P=1 and one-rank `DistributedSum`), `PartitionPlan` rebuilt in place (no per-layer allocation at any P), traces/stats filled from `run_layers`' records; net about +45 lines (single loop, not fewer lines).
Equivalence vs `c14ea2a` (harness in scratchpad `equiv/`): 48 cases bitwise identical incl. traces and bucket counts; panic text gains a "partition 0," prefix; a policy with `finalizes_layer() == false` now skips `finalize_layer` (allowed by contract).
Local A/B (4 ABAB pairs, indicative): `su4` 1e6 8T −6.5% (4/4, likely layout), `rotation_zz` 1e6 1T +2.6% (4/4); small-n cells noise. Same-node A/B pending: `A_REV=89bdcca PS_REV=bfb19b3 LAYERS="rotation_zz cnot su4" NS="1000 1000000" REPS=40 sbatch --export=ALL scripts/slurm/ab-campaign.sbatch`.
Verified: fmt/clippy matrix, rustdoc, 737 workspace tests, `cuda` 867 on the A6000, pytest 485/101, `mpi-test.sh --ranks 2,4` Rust and `--python`.

`c2411cb`: a probe over `Gf2Hash` seeds 1–3 at 4/7/10 bits gave `haar_su4` coset dimension `r = 4` (output-major gather) at 7 and 10 bits, `cnot` `r = 2`, `rotation_zz` `r = 1`; three docs that said no built-in reaches output-major fixed.

### As applied, stage A (D4, D5, D6)

`5f44228`..`0b0f466` (16 commits): `pauli_sum/{storage,partition,hash,accumulator}`, `readout/{product_state,stabilizer,echo}`, the four wrong-way edges fixed (read-out methods as `impl PauliSum` blocks in `readout/`, `keeps_flip_classes` to `readout/echo`, `PartitionedTruncation for BuiltinTruncation` to `engine/partitioned/truncation`, new `engine/cuda_context` breaks the topology↔gpu cycle), seven files split on seams, 54 sibling `tests.rs` files, private modules with flat root re-exports, `propagate` + `propagate_with(…, &mut scratch, options)`, `propagate_partitioned` takes options.
Verified: 799 workspace tests (warm wall 42.5 s → 30.3 s), pytest 523 passed / 101 skipped (unchanged), `cuda` tests 929 passed on the A6000, `cuda,mpi,test-utils` 971 passed, `mpi-test.sh --ranks 2,4` green; Python API unchanged.
Deviation: the out-of-memory warning at CUDA context creation now prints the driver error text rather than `GpuError`'s.
Left in source: `cfg(test)` fields woven into `gpu/wire/peer.rs`, `nccl.rs`, `ExchangeBlock::with_counts`, `skip_sequence_for_test`.

### As applied, stage B0 (D11–D14)

`533c666` D11 (`Collectives: sealed::Sealed`, which seals `Transport` too; `Payload`/`ChunkMap`/`ChunkWait`/`InProcessTransport` unnameable outside, `InProcessTransport` via `test_support`), `d371565` D12, `feda6c5` D13 (`PartitionRuntime::wait_timeout` and `DistributedSum::scatter_with_runtime` removed as dead), `67bbebb` D14 (`propagate_with` on every sum type; `DistributedSum::scatter` + `scatter_with(sum, transport, ScatterOptions { runtime, rows: ScatterRows::{Policy, Explicit} })`).
Verified: workspace fmt/test/clippy per commit, clippy across py `cuda`/`cuda,mpi`, `cuda` tests on the A6000, pytest 523/101, `mpi-test.sh --ranks 2,4` Rust and `--python`; Python API unchanged.
Follow-ups folded into stage B: `PartitionedSum::scatter_with_rows` and the GPU `scatter_to_device(s)_with_rows` pairs renamed to the `scatter`/`scatter_with` shape (cross-folder, so in B8); the `DistributedSum` doc example removed per D3; `ChunkMap`'s unreachable `pub` methods narrowed.

### As applied, stage C (D9)

`538b7b0` CLAUDE.md 232 → 189 lines (new "Code organisation" section, testing policy for sibling `tests.rs`, repo layout current; dropped architecture detail, uv version facts, some MPI/CUDA recipe variants, perf numbers now pointing at HARDWARE.md), `9a7b417` FINDINGS.md 487 → 300 (all headings kept; shipped/superseded GPU items relabelled), `2060f53` HARDWARE.md 299 → 284 (tables kept; host-staged GPU numbers labelled as such).
Found: Python-side code cites `FINDINGS.md §A3/§A5/§A7/§A8-ii`, labels that match no heading (pre-existing). SIMD entry's "build is SSE2" still holds (no `target-cpu` in `.cargo/config.toml`).

### As applied, stage B (D1–D3, D7, D10, D14 follow-up)

Per-folder passes cherry-picked from parallel worktrees: B7 `52b7c21`..`8b6ab4f`, B5 `44897f8`..`f7bcb04`, B4 `23120de`..`8d56e04`, B1 `4759caa`..`0fa3596`, B2 `01d9ae6`..`79ab14f`, B6 `5d92df1`..`1b44d22`, B3 ..`ddfbd09`; then B8 `cd9e950`..`2489250` (cross-folder renames, every `scatter` in `scatter`/`scatter_with` shape via `ScatterOptions`/`ScatterRows::resolve`, `assert_invariants` via `test_support`, `RemoteDelta::partition_delta` dropped).
Totals vs `89bdcca`: production 22.9k → 20.5k lines, production comment lines 5036 → 2340, tests + `test_support` 19.9k → 19.6k; 41 doc examples removed (each checked covered elsewhere; workspace 799 → 758 tests); non-comment production lines roughly flat (renames rewrap, split files add headers).
Doc bugs fixed: `Gf2Hash::coarsen` merges `(b, b+B/2)` not `(2i, 2i+1)`; pipeline early-slot tag; `ExchangeBlock::header` `num_buckets` meaning; in-process collectives use atomics not channels; `ExportScratch` derives `Default`.
Python: `Truncation.__repr__` kept `Coeff(...)` via a `tree_repr` in the bindings.
Verified at `2489250`: fmt/clippy (default, `cuda,phase-timing,test-utils`, py `cuda`, `cuda,mpi`), rustdoc `-D warnings`, 758 workspace tests, `cuda` 888 passed on the A6000 (device tests ran), pytest 523/101, `mpi-test.sh --ranks 2,4` Rust and `--python`, `cuda,mpi` `mpi_ranks` one rank 33 ok (two ranks on one GPU unsupported by NCCL).
Not yet measured: code A/B against `89bdcca` (D8).
- **D44** Cross-configuration run comparisons use the normalised operator overlap `Re⟨a, b⟩ / (‖a‖ ‖b‖)` against a tolerance of the order of the truncation (plus equal counts), replacing the magnitude/octave/count-only rules.
  As applied: `7f4aad3`, `749ec81`: `assert_runs_agree(policy, discarded, got, want, tol, what)` with bound `1 − cos ≤ 2Δ²/(‖got‖‖want‖)`, `Δ` = norm the reference run's cut dropped (measured by `test_support::DiscardMeter`, no engine change) plus `√len·tol`; refuses cuts too heavy to catch a sign flip on the largest term. Truncating nets now cut lightly (inputs spread over 40 octaves, larger `n`); observed/bound median 2.7e-3, max 0.50 (tied cut). No public `norm` (`overlap(self).re.sqrt()`).
  Heavy cuts (user): `9645f99`, `5c10cb7` add `assert_runs_agree_count_only` (exactly `n`, finite, and asserts the cut is too heavy for the overlap check) with one `heavy_cut` case (12 qubits, 800 terms, n = 400, 24 cutting layers) per configuration: partitioned, in-process distributed, MPI, GPU, GPU partitioned, Python.
- **D45** Chunk 7: `merge2_into` takes `ColumnsRef`/`BucketColumns` instead of 10 slices; one inline emit helper (no A/B, the user remeasures later); `SortKernel` owns the radix gate and its constants; a real `u32` length assert replaces the radix fallback; `Gf2Span` drops `space_mask`, privatises `is_representative`/`rank_of_representative`, renames `r` → `dimension`, `coord_of` → `coordinate_of`; `merge2_into` → `merge_identity_and_rest`, sorts → `sort_by_comparison`/`sort_by_radix`.
  As applied: cherry-picked `79baa4e`..`27b0c8d`; 604 equivalence cases bitwise; unpadded su4 instructions +0.37% (`SortKernel::sort` inlines both kernels), user remeasures later.
- **D46** Chunk 8: one `LayerTrace` (the old `GateTrace`, with `peak_terms`; `enable_trace`/`take_trace`) replaces `TermTrace` and `GateTrace`; `phase-timing` scaffolding folded crate-wide behind no-op macros/ZST, gated on a byte-identical default-build `.text`; the parallel coset loop's fresh-scratch fallback becomes an `expect`; `PhaseStats` stays one struct (stage-B split item closed); `ExtraRows::append_into` takes `&mut BucketColumns`, `id_*` → `identity_*`, `push_row` inlined if trivial, `rescale_in_place` via `retain_per_bucket` if codegen holds.
  As applied: cherry-picked `929a4b7`..`c89fca2`. `engine/timing.rs` macros (`stamp!`, `lap!`, `tally!`, `timed!`); phase-timing cfg attributes 122 → 54. Default-build `.text` same size, differing only in `log!` line-number immediates (masked disassembly identical). `LayerTrace` in place of both traces (Python API unchanged); `PauliSum::update_per_bucket` backs `retain_per_bucket` and `rescale_in_place` (keep-all depolarizing layer 30–35M → 20–23M instructions). 604 equivalence cases bitwise.
- **D47** Chunk 9: bucket knobs stay public and are validated (`min_buckets ≥ 16`, `target_bucket_len ≥ 1`; Rust assert, Python `ValueError`); Rust `propagate_with` keeps a required `PropagateOptions`, Python takes concrete defaults instead of `None` plumbing. Rust docs are a developer guide, Python and the book are the user guide: remove the Rust user example (`examples/ising_2d_quench.rs`, its plot script, output, README entry, `crates/paulistrings/docs/`, `pub mod examples`) and rustdoc doc examples (crate root, `propagate`, `PauliSum`, `BuildAccumulator`, `Channel`, `TruncationPolicy`) unless one is the only test of something; the crate header points to the book and ARCHITECTURE.md; `phase_breakdown` stays (probe). `propagate`'s `# Panics` names `MAX_LOCAL_SUPPORT` with `PauliRotation` exempt.
  As applied: `ee88e62` (`PropagateOptions::validate`, asserted at every propagate entry point incl. partitioned, distributed and GPU; Python literal defaults `1024`/`128`, `ValueError` below the bounds, `None` now a `TypeError`; tests that used `min_buckets: 1` for one bucket now use a huge floor), `448e3ea` (Ising example, `crates/paulistrings/docs/`, `pub mod examples`, all 10 doctests removed; the `Channel` doctest became `tests/propagate.rs::a_channel_implemented_downstream_propagates_both_ways`; book flagship section, `macro` campaign item, CI `cargo test --doc`/`--examples` removed), `18c5e65`, `4379169`. Open: README and crate README Rust quickstarts.
- **D48** Chunk 10: `ProductState` folds into `ProductBasis::uniform(PauliAxis)` with one `expectation_product`, `ProductBasis` fields private; `StabilizerState::sign_of` crate-private, `expectation_of` removed, `StabilizerError::InternalPhase` becomes an assert, `Row::neg` → `negative`; one crate-private parallel per-term fold for the read-outs; one range-checked `qubit_mask` in `pauli_string.rs` behind `support_mask`.
- **D49** The operator-Loschmidt-echo read-outs leave the library: `readout/echo.rs` (`RotationAxis`, `diagonal_echo`, `anticommute_histogram`, `rotated_overlap`, `keeps_flip_classes`), the `DistributedSum` versions and their Python bindings go; Rust provides only the primitive, `PauliSum.to_arrays()` (inverse of `from_arrays`, also on a local MPI share), and the estimators move to numpy in `examples/b8_ole/ole.py`, validated against the Rust versions before deletion.
  As applied (D48, D49): `23e36a5` (read-outs; `support_mask` now range-checks in release), `f6f1db8` (`PauliSum.to_arrays()`), `9ba0d28` (numpy estimators in `examples/b8_ole/ole.py`, `comm=` via allreduce; agree with Rust to ≤ 1e-14; about 17× slower, 3.1 s vs 0.18 s at 20M terms, 1.7% of propagate time), `a529ab1` (echo removed from Rust, bindings, tests, docs; a `comm=` local share is now a plain `PauliSum`), `67b73dd` (README Rust quickstarts → developer note). Noticed, not chased: `.venv-mpi` python without `mpirun` segfaults on run_b8; mpirun default binding makes B8 MPI runs 200–300× slower than `--bind-to none`.
- **D50** Chunk 11: `PartitionConfig::threads_per_partition: Option<usize>` for every placement replaces `Placement::Unpinned`'s field and `PartitionRuntime::with_threads_per_partition`; `PartitionSlot` and `PartitionRuntime::slots()` crate-private (via `test_support`); one placement `LOG_TARGET` (`paulistrings::partitioned`) in `partitioned/mod.rs`, used by `cuda_context` too; row diagnostics (`circuit_generators`, `GeneratorWeight`, `count_remote_deltas`, `PartitionPlan::new`) move to `test_support`; `prev_power_of_two` via `ilog2`; one `LocalPtm::rest_streams()`. Single hash matrix: issue #22.
  As applied: `312255f`..`c8e33c1` (the branch's broken intermediate commit squashed with its binding fix). `threads_per_partition` also overrides device partitions' 4 host workers; new `PartitionConfig::num_partitions()` since `resolve` is crate-private; target const named `PLACEMENT_LOG_TARGET` (MPI uses it too); diagnostics in `test_support/rows.rs`; `LocalPtm::has_identity()` added. 604 equivalence cases bitwise incl. P=1/2/4.
- **D51** Chunk 12: one crate-private coset order (permutation, inverse) owned by `LayerScratch` and shared by `ChunkMap`; `InProcessTransport` collectives become a mutex/condvar group (departure flag and timeout kept), gated on a P = 2/4 same-node A/B on small layers that the user submits; D10 renames (`perm`, `inv`, `r`, `seq`, `src`, `dst`, `col`, `recv_from`); one wait-timeout constant. Env knobs `PAULISTRINGS_EXCHANGE_CHUNKS`/`PAULISTRINGS_NCCL_TIMEOUT_S` decided in chunks 15/17.
  As applied: `9deb5ad` (`CosetOrder` in `engine/coset.rs`, owned by `LayerScratch`, lent to the coset loop via `LayerKnobs::order`; `ChunkMap` only chunks and is `Copy`), `244175a` (mutex/condvar rendezvous; 7 `unsafe` sites → 0; departure, timeout and mismatch panics kept; new tests), `a091b47`, `820b035`, `be40a40`. 603 equivalence cases bitwise. Local indicative A/B: P = 4 cnot +31% wall, consistent in 4/4 pairs (about 15–20 µs per layer); gate pending the user's jobs (A `9deb5ad`, B `244175a`, P = 2 and P = 4 unpinned). Fallback if confirmed: bounded spin on a completed-generation atomic before blocking.
- **D52** Chunk 13: `ColumnsMut` beside `ColumnsRef` replaces `export::BlockColumns`; one dense-entry test shared by export and `DeltaPlan`; D10 renames (`RecvRows` → `ReceivedRows`, `recv`, `bx`/`bz`/`bc`, …), `ExportCounts` folds into `LayerExchangeCounts` (`rows_sent`/`bytes_sent`); `PartitionStorage`/`HostPartition` `#[doc(hidden)]`; `PAULISTRINGS_EXCHANGE_CHUNKS` kept and documented in CLAUDE.md beside `PAULISTRINGS_GPU_EXCHANGE_BYTES`. Queued after D51.

## Possible improvements

- Cache the hash-independent part of a channel's prepared table (probe + masks) so a layer recomputes only `δ = H·d`; `prepare` is 4.2–5.7 µs per dense two-qubit gate and dominates small-`m` runs now that the direct path is gone (issue to open).

## Open items

- `PtmChannel` is 80 bytes (name 16, qubits `[usize; 2]` 16, storage 40); a one-byte kind and `u32` qubits would give about 48.
- `unitary_1q`/`unitary_2q` do not check unitarity.

- A custom policy overriding `finalize_layer_partitioned` must also override `supports_partitioned` (documented; Rust cannot detect overrides).
- `Or(TopN, …)`: `supports_partitioned` is true (Or runs no layer pass) while Python and the GPU drivers reject any `TopN` in the tree; and Or dropping its children's layer passes is itself questionable semantics (chunk 6).

- Re-evaluate which `#[inline]` annotations matter by A/B. Removing them alone changed codegen in `channel/{clifford,noise,unitary}.rs`, `engine/{coset,mod}.rs` and `partitioned/export.rs` (both binaries), and in `channel/rotation.rs`, `engine/merge.rs` and `partitioned/driver.rs` (cdylib only).
- `mpi-test.sh --python` needs `python-mpi/3.12.9` and `uv` loaded on top of the MPI module list (document in CLAUDE.md).

- `ApproxTopN` wipes a sum whose top octave alone exceeds `n` to empty (documented contract; seen at 1e5-start cases with `ApproxTopN(4000)`): surprising behaviour to revisit in chunk 6.
- `benchmarks/python/jl_performance/README.md` and `post-optimization-auto/` still describe `engine="auto"` as historical records (chunk 23).

- `PAULISTRINGS_NCCL_TIMEOUT_S` also bounds the in-process `PeerWire` (misleading env-var name; chunk 19).
- `GpuDistributedSum::propagate_with` only `debug_assert!`s a stale device error, `GpuPartitionedSum` returns it (chunk 20).
- `Prepared` is the public return type of `Channel::prepare` but unnameable outside `test_support`, so custom `prepare` overrides cannot be written (chunk 5/9).
- Fold `phase-timing` cfg blocks in the layer loop and coset kernel behind a no-op stamp; split partitioned/device-only `PhaseStats` fields (chunk 8, needs A/B).
- Shared two-pointer merge skeleton (`merge_two`, `merge_two_adding`, `PauliSum::overlap`) (chunk 3).
- ARCHITECTURE.md contradictions: §Truncation "TopN remains the default" vs ApproxTopN partitioned default; §Channels PTM constructor that doesn't exist, missing `PauliChannel`/`Depolarizing2Q`; §Truncation sketch missing `finalizes_layer`/`CollapseSample`.
- Python code cites `FINDINGS.md §A3/§A5/§A7/§A8-ii`, which match no heading.
- `arch_static` leaks one string per unknown compute arch (bounded).
- Files still over the soft cap: `mpi.rs`, `gpu/layer.rs`, `distributed.rs`, `test_support.rs`.

## Resume here

Chunks 0–6 done (chunk 6 at `601b391`); D42 waits on job 7216669 (`ab-campaign.sbatch`, `KNOB_B="--truncation-dispatch builtin"`, PS_REV `b4df80d`), D16 on its own A/B. Chunk 7 (D45) and D44 implementing; chunk 8 (D46) implementing on top of `chunk7`; chunk 9 (D47) implementing; chunk 12 (D51) implementing; D52 implementing; D51 collectives gated on the user's A/B; next chunk 14 (driver, trace, partitioned truncation).
