//! Partitioned execution: the sum split across NUMA domains or MPI ranks by
//! designated *partition rows* of the GF(2) hash, one pinned Rayon pool per
//! partition, and a push-model exchange of the rows a layer moves across
//! partitions. See ARCHITECTURE.md §Partitioning.
//!
//! Module map:
//! - `topology` — CPU sets, NUMA node discovery, thread/memory pinning, pinned pools.
//! - `transport` — the `Collectives`/`Transport`/`Payload` seam, the
//!   `ExchangeBlock` wire format, and the in-process channel-matrix transport.
//! - `plan` — per-layer classification of a prepared channel's deltas into
//!   local and remote (partner) deltas.
//! - `export` — the export pass building per-partner exchange blocks.
//! - `layer` — `apply_layer_partitioned`: export → exchange → coset loop.
//! - `mpi` — the distributed transport, one partition per rank (`mpi` feature).
//! - `truncation` — `PartitionedTruncation` (collective `ApproxTopN`).
//! - `runtime` — `PartitionRuntime`: the resolved placement, its pools, and the
//!   scoped fan-out a partitioned call runs inside.
//! - `driver` — `PartitionedSum`, `propagate_partitioned`, `run_layers`, and
//!   the per-layer collective schedule (`BITS_AGREE_EVERY`).
//! - `distributed` — `DistributedSum`: `run_layers` again, with one partition
//!   per process, over any `Transport`.
//! - `trace` — `PartitionTrace`, the opt-in per-layer record of term counts,
//!   bucket bits and exchange volume.
//!
//! # Why there are two drivers
//!
//! [`PartitionedSum`] holds `P` partitions inside one process and fans out to
//! them per call; [`DistributedSum`] *is* one partition and its peers are other
//! processes. Everything below the fan-out is literally one implementation —
//! `run_layers` is the layer loop for both, `scatter_local` the scatter body,
//! `PartitionWork` the per-partition payload, and `apply_layer_partitioned` the
//! layer — and the two differ only above it, in three ways that do not
//! reconcile:
//!
//! - **The transport group's lifetime.** `PartitionedSum` builds a fresh
//!   in-process group *per call* and moves the endpoints into the partitions,
//!   so a partition that panics drops its senders and its partners fail by name
//!   instead of blocking in `recv`. A `DistributedSum` owns one endpoint for
//!   its whole life, because an `MPI_Comm` is not something to duplicate per
//!   layer.
//! - **Scatter and gather.** In-process, the input is *one* sum split by
//!   `filter_partition` and merged back locally, bitwise, with no wire format
//!   in sight. Distributed, the input is *replicated* on every rank and the
//!   gather goes through the transport's byte framing to rank 0 alone.
//! - **The consistency check.** One process cannot hand its own partitions
//!   different circuits; separate processes can, so only the distributed driver
//!   pays for `check_consistency`.
//!
//! Expressing one as the other would mean either a runtime-owned transport
//! group (which is the hang the per-call group exists to prevent) or a
//! replicated-input, byte-framed scatter for the in-process case, which is
//! strictly more work for the same answer.
//!
//! # What is an error and what is a panic
//!
//! **Setting a run up returns `Result`; running it panics.** Exactly two things
//! can fail without being a bug in the caller's code, and both happen before a
//! term is touched:
//!
//! - resolving a [`PartitionConfig`] against the machine and building its pools
//!   — [`TopologyError`], from [`PartitionRuntime::new`] and everything that
//!   calls it;
//! - adopting a communicator — `MpiError`, from
//!   `MpiTransport::from_raw_handle` (`from_communicator` is the panicking
//!   convenience over it; `mpi` feature).
//!
//! Everything after that is a **contract violation**, and the engine panics
//! naming the partition and what it expected: a group size that is not a power
//! of two, partition rows that do not match the sum, a channel whose `prepare`
//! declines, a policy that finalizes layers without a collective form, a
//! partner that desynchronized or died. None of them is recoverable — the group
//! is already out of step — and turning them into `Result` would only move the
//! `unwrap` to the caller.
//!
//! # Naming
//!
//! - **`size` is a transport's group cardinality** ([`Collectives::size`]);
//!   **`num_partitions` is a placement's or a row set's**
//!   ([`PartitionRuntime::num_partitions`],
//!   [`PartitionRows::num_partitions`](crate::PartitionRows::num_partitions)).
//!   They are equal in any well-formed run, and the drivers assert it at
//!   scatter. "Rank" is a partition index, used where the peer is a process.
//! - **`bind_memory` is the Rust name throughout** (the [`PartitionConfig`]
//!   field, `build_pool`'s argument, the probe's `--bind-memory`): it installs
//!   `MPOL_BIND` on the slot's NUMA node, which is a *memory* policy and not a
//!   thread affinity. The Python surface calls the same knob `pin_memory=`, and
//!   the probe's JSON sidecar uses that spelling to match it; those two are the
//!   only places the other name appears.

pub(crate) mod distributed;
pub(crate) mod driver;
pub(crate) mod export;
pub(crate) mod layer;
#[cfg(feature = "mpi")]
pub mod mpi;
pub(crate) mod plan;
pub(crate) mod rows;
pub(crate) mod runtime;
pub(crate) mod topology;
pub(crate) mod trace;
pub(crate) mod transport;
pub(crate) mod truncation;

// The front door: a sum split across partitions, and the one-shot entry points.
pub use distributed::DistributedSum;
#[cfg(feature = "phase-timing")]
pub use driver::PartitionPhaseStats;
pub use driver::{
    propagate_partitioned, propagate_partitioned_with_options, PartitionedSum, BITS_AGREE_EVERY,
};
pub use plan::count_remote_deltas;
// Choosing the partition rows instead of drawing them: the circuit's generator
// masks, the weighted MAX-XOR-SAT selector over them, and the per-layer
// locality the chosen rows produce.
pub use rows::{circuit_generators, layer_locality, GeneratorWeight};
pub use runtime::PartitionRuntime;
// Where the partitions run: the machine's CPU sets and NUMA nodes, and the
// placement a caller asks for.
pub use topology::{numa_nodes, CpuSet, PartitionConfig, PartitionSlot, Placement, TopologyError};
// What a partitioned run did, layer by layer: term counts per partition,
// bucket bits, and who sent how many rows to whom.
pub use trace::{PartitionLayerRecord, PartitionTrace};
// The seam a transport is written against. The concrete wire types
// (`ExchangeBlock`, `PartnerPayload`, `BlockHeader`) are deliberately not here:
// a transport moves an opaque `P: Payload` and never names them.
pub use transport::{ChunkMap, ChunkWait, Collectives, InProcessTransport, Payload, Transport};
// The collective form of a layer finalization: what a truncation policy has
// to provide before it can run with the sum split across partitions.
pub use truncation::PartitionedTruncation;
