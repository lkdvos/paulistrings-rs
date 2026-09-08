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

pub(crate) mod plan;
pub(crate) mod topology;
pub(crate) mod transport;

pub use plan::count_remote_deltas;
pub use topology::{
    allowed_cpus, bind_current_thread_memory, current_cpu, numa_nodes, pin_current_thread, CpuSet,
    PartitionConfig, PartitionSlot, Placement, TopologyError,
};
