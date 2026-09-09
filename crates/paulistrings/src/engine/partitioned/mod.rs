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
//! - `driver` — `PartitionedSum`, `propagate_partitioned`, and `run_layers`.
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

pub(crate) mod distributed;
pub(crate) mod driver;
pub(crate) mod export;
pub(crate) mod layer;
#[cfg(feature = "mpi")]
pub mod mpi;
pub(crate) mod plan;
pub(crate) mod runtime;
pub(crate) mod topology;
pub(crate) mod trace;
pub(crate) mod transport;
pub(crate) mod truncation;

// The front door: a sum split across partitions, and the one-shot entry points.
pub use distributed::DistributedSum;
#[cfg(feature = "phase-timing")]
pub use driver::PartitionPhaseStats;
pub use driver::{propagate_partitioned, propagate_partitioned_with_options, PartitionedSum};
pub use plan::count_remote_deltas;
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
