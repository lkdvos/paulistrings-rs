//! Partitioned execution: the sum split across NUMA domains or MPI ranks by GF(2) partition rows, one pinned Rayon pool per partition, and the push-model exchange between them (ARCHITECTURE.md §Partitioning).
//! `PartitionedSum` holds every partition in one process, `DistributedSum` is one partition of a group of processes; both run `driver::run_layers`.
//! Setting a run up returns `Result`; everything past that is a contract violation and panics, since the group is already out of step.

pub(crate) mod backend;
pub(crate) mod distributed;
pub(crate) mod driver;
pub(crate) mod export;
pub(crate) mod layer;
#[cfg(feature = "mpi")]
pub mod mpi;
pub(crate) mod plan;
#[cfg(any(test, feature = "test-utils"))]
pub(crate) mod rows;
pub(crate) mod runtime;
pub(crate) mod sum;
pub(crate) mod topology;
pub(crate) mod trace;
pub(crate) mod transport;

pub use distributed::{DistributedSum, PartitionRowPolicy, ScatterOptions, ScatterRows};
pub use driver::propagate_partitioned;
pub use runtime::PartitionRuntime;
#[cfg(feature = "phase-timing")]
pub use sum::PartitionPhaseStats;
pub use sum::PartitionedSum;
pub use topology::{numa_nodes, CpuSet, PartitionConfig, PartitionSlot, Placement, TopologyError};
pub use trace::{PartitionLayerRecord, PartitionTrace};
pub use transport::Transport;
