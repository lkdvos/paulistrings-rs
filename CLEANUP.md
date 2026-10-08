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

(none yet)

## Possible improvements

## Open items

## Resume here

Setup done; waiting for the user to adjust the chunk plan and state cross-cutting conventions before chunk 0.
