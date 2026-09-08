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
//! - `truncation` — `PartitionedTruncation` (collective `ApproxTopN`).
//! - `runtime` — `PartitionRuntime`: the resolved placement, its pools, and the
//!   scoped fan-out a partitioned call runs inside.
//! - `driver` — `PartitionedSum` and `propagate_partitioned`: the layer loop,
//!   the collective bucket-count agreement, scatter and gather.
//! - `trace` — `PartitionTrace`, the opt-in per-layer record of term counts,
//!   bucket bits and exchange volume.

pub(crate) mod driver;
pub(crate) mod export;
pub(crate) mod layer;
pub(crate) mod plan;
pub(crate) mod runtime;
pub(crate) mod topology;
pub(crate) mod trace;
pub(crate) mod transport;
pub(crate) mod truncation;

// The front door: a sum split across partitions, and the one-shot entry points.
#[cfg(feature = "phase-timing")]
pub use driver::PartitionPhaseStats;
pub use driver::{propagate_partitioned, propagate_partitioned_with_options, PartitionedSum};
pub use plan::count_remote_deltas;
pub use runtime::PartitionRuntime;
// What a partitioned run did, layer by layer: term counts per partition,
// bucket bits, and who sent how many rows to whom.
pub use topology::{
    allowed_cpus, bind_current_thread_memory, current_cpu, numa_nodes, pin_current_thread, CpuSet,
    PartitionConfig, PartitionSlot, Placement, TopologyError,
};
pub use trace::{PartitionLayerRecord, PartitionTrace};
// The exchange wire format: what a layer's cross-partition traffic looks like
// on the wire, and the trait an MPI transport implements to move it.
pub use transport::{
    BlockHeader, Collectives, ExchangeBlock, InProcessTransport, PartnerPayload, Payload, Transport,
};
// The collective form of a layer finalization: what a truncation policy has
// to provide before it can run with the sum split across partitions.
pub use truncation::PartitionedTruncation;
