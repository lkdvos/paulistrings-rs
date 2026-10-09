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
| 0 | Tour of the main call path | engine/mod, bucketed (skim), coset, merge, partitioned/driver (skim) | pending |
| 1 | Leaf algebra | phase, rng, pauli_string | pending |
| 2 | Partition hash | bucket/hash | pending |
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

## Open items

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
