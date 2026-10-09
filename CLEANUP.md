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
| 2 | Partition hash | pauli_sum/hash | D24–D26 implementing |
| 3 | Storage | bucket/sum, pauli_sum, accumulator | pending |
| 4 | Channels I | channel/mod, clifford, identity, rotation, circuit | pending |
| 5 | Channels II | channel/noise, unitary, prepared | pending |
| 6 | Truncation | truncation/* | pending |
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

## Possible improvements

- Cache the hash-independent part of a channel's prepared table (probe + masks) so a layer recomputes only `δ = H·d`; `prepare` is 4.2–5.7 µs per dense two-qubit gate and dominates small-`m` runs now that the direct path is gone (issue to open).

## Open items

- Re-evaluate which `#[inline]` annotations matter by A/B. Removing them alone changed codegen in `channel/{clifford,noise,unitary}.rs`, `engine/{coset,mod}.rs` and `partitioned/export.rs` (both binaries), and in `channel/rotation.rs`, `engine/merge.rs` and `partitioned/driver.rs` (cdylib only).
- `mpi-test.sh --python` needs `python-mpi/3.12.9` and `uv` loaded on top of the MPI module list (document in CLAUDE.md).

- `ApproxTopN` wipes a sum whose top octave alone exceeds `n` to empty (documented contract; seen at 1e5-start cases with `ApproxTopN(4000)`): surprising behaviour to revisit in chunk 6.
- `benchmarks/python/jl_performance/README.md` and `post-optimization-auto/` still describe `engine="auto"` as historical records (chunk 23).

- Release-mode guard: `Depolarizing2Q` with equal qubits is only `debug_assert`ed (chunk 5).
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

Preparatory pass (stages A, B0, B1–B8, C) complete and verified at `2489250`; code A/B vs `89bdcca` handed to the user. Next: chunk 0 tour on the reorganised tree (update the global map paths first).
