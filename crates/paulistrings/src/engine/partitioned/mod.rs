//! Partitioned execution: the sum split across NUMA domains or MPI ranks by GF(2) partition rows, one pinned Rayon pool per partition, and the push-model exchange between them (ARCHITECTURE.md §Partitioning).
//! `PartitionedSum` holds every partition in one process, `DistributedSum` is one partition of a group of processes; both run `engine::driver::run_layers`.
//! Setting a run up returns `Result`; everything past that is a contract violation and panics, since the group is already out of step.

/// `log` target of placement, pinning and transport setup, separate from the progress target so it filters on its own.
pub(crate) const PLACEMENT_LOG_TARGET: &str = "paulistrings::partitioned";

pub(crate) mod distributed;
pub(crate) mod driver;
pub(crate) mod export;
pub(crate) mod layer;
#[cfg(feature = "mpi")]
pub mod mpi;
pub(crate) mod plan;
pub(crate) mod runtime;
pub(crate) mod sum;
pub(crate) mod topology;
pub(crate) mod transport;

pub use distributed::{DistributedSum, PartitionRowPolicy, ScatterOptions, ScatterRows};
pub use driver::propagate_partitioned;
pub use runtime::PartitionRuntime;
pub use sum::PartitionedSum;
pub use topology::{numa_nodes, CpuSet, PartitionConfig, Placement, TopologyError};
pub use transport::Transport;
