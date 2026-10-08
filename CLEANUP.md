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

### As applied, stage A (D4, D5, D6)

`5f44228`..`0b0f466` (16 commits): `pauli_sum/{storage,partition,hash,accumulator}`, `readout/{product_state,stabilizer,echo}`, the four wrong-way edges fixed (read-out methods as `impl PauliSum` blocks in `readout/`, `keeps_flip_classes` to `readout/echo`, `PartitionedTruncation for BuiltinTruncation` to `engine/partitioned/truncation`, new `engine/cuda_context` breaks the topology↔gpu cycle), seven files split on seams, 54 sibling `tests.rs` files, private modules with flat root re-exports, `propagate` + `propagate_with(…, &mut scratch, options)`, `propagate_partitioned` takes options.
Verified: 799 workspace tests (warm wall 42.5 s → 30.3 s), pytest 523 passed / 101 skipped (unchanged), `cuda` tests 929 passed on the A6000, `cuda,mpi,test-utils` 971 passed, `mpi-test.sh --ranks 2,4` green; Python API unchanged.
Deviation: the out-of-memory warning at CUDA context creation now prints the driver error text rather than `GpuError`'s.
Left in source: `cfg(test)` fields woven into `gpu/wire/peer.rs`, `nccl.rs`, `ExchangeBlock::with_counts`, `skip_sequence_for_test`.

## Possible improvements

## Open items

## Resume here

Conventions D1–D10 agreed. Preparatory pass running (stage A: organisation, test extraction, pub surface; stage B: comments, naming, mechanical cuts by folder; stage C: CLAUDE.md and research docs). Chunk 0 tour after it lands.
