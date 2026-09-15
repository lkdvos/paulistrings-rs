//! Partitioned execution: the sum split across NUMA domains or MPI ranks by designated partition rows of the GF(2) hash, one pinned Rayon pool per partition, and a push-model exchange of the rows a layer moves across partitions.
//! See ARCHITECTURE.md §Partitioning.
//! `PartitionedSum` (`driver`) holds `P` partitions in one process and fans out per call; `DistributedSum` (`distributed`) *is* one partition, its peers other processes; both share `run_layers`, `PartitionWork`, and `apply_layer_partitioned` (`layer`).
//! Setting a run up returns `Result` (`TopologyError`, `MpiError`); everything past that is a contract violation and panics, since the group is already out of step by then.
//! `size` ([`Collectives::size`]) is a transport's group cardinality; `num_partitions` ([`PartitionRuntime::num_partitions`], [`PartitionRows::num_partitions`](crate::PartitionRows::num_partitions)) is a placement's or row set's — equal in any well-formed run.

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
pub use distributed::{DistributedSum, PartitionRowPolicy};
#[cfg(feature = "phase-timing")]
pub use driver::PartitionPhaseStats;
pub use driver::{
    propagate_partitioned, propagate_partitioned_with_options, PartitionedSum, BITS_AGREE_EVERY,
};
pub use plan::count_remote_deltas;
// Choosing partition rows: the circuit's generator masks and the weighted MAX-XOR-SAT selector over them.
pub use rows::{circuit_generators, GeneratorWeight};
pub use runtime::PartitionRuntime;
// Where partitions run: CPU sets, NUMA nodes, and the placement a caller asks for.
pub use topology::{numa_nodes, CpuSet, PartitionConfig, PartitionSlot, Placement, TopologyError};
// What a partitioned run did, layer by layer: term counts, bucket bits, exchange volume.
pub use trace::{PartitionLayerRecord, PartitionTrace};
// The seam a transport is written against; a transport moves an opaque `P: Payload`, never the wire types.
pub use transport::{ChunkMap, ChunkWait, Collectives, InProcessTransport, Payload, Transport};
// The collective form of a layer finalization a truncation policy must provide.
pub use truncation::PartitionedTruncation;
