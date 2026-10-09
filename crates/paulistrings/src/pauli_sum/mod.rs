//! [`PauliSum<W>`], its bucketed column storage, the GF(2)-linear hash that partitions it, and the [`BuildAccumulator`] that ingests it (ARCHITECTURE.md §Data-Model, §Bucketing).
//!
//! [`BuildAccumulator`]: crate::BuildAccumulator

pub(crate) mod accumulator;
pub(crate) mod hash;
mod partition;
pub(crate) mod storage;

pub use hash::{Gf2Hash, PartitionRows, P_MAX_BITS};
pub use storage::PauliSum;

#[cfg(test)]
mod tests;
