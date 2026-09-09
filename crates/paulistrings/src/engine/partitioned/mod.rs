//! Partitioned execution: the sum split across NUMA domains (and, later, MPI
//! ranks) by designated *partition rows* of the GF(2) hash, one pinned Rayon
//! pool per partition, and a push-model exchange of the rows a layer moves
//! across partitions. See ARCHITECTURE.md §Partitioning.
//!
//! Module map (each lands in its own step; see the plan in the research notes):
//! - `topology` — CPU sets, NUMA node discovery, thread/memory pinning, pinned pools.
//! - `transport` — `Collectives`/`Transport` traits, the `ExchangeBlock` wire
//!   format, and the in-process channel-matrix transport.
//! - `plan` — per-layer classification of a prepared channel's deltas into
//!   local and remote (partner) deltas.
//! - `export` — the export pass building per-partner exchange blocks.
//! - `layer` — `apply_layer_partitioned`: export → exchange → coset loop.
//! - `mpi` — the distributed transport, one partition per rank (`mpi` feature).
//! - `truncation` — `PartitionedTruncation` (collective `ApproxTopN`).
//! - `runtime` — `PartitionRuntime`: the resolved placement, its pools, and the
//!   scoped fan-out a partitioned call runs inside.
//! - `driver` — `PartitionedSum` and `propagate_partitioned`: the layer loop,
//!   the collective bucket-count agreement, scatter and gather.
//! - `distributed` — `DistributedSum`: the same layer loop with one partition
//!   per process, over any `Transport`.
//! - `trace` — `PartitionTrace`, the opt-in per-layer record of term counts,
//!   bucket bits and exchange volume.

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
