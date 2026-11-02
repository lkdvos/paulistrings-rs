//! [`PauliSum<W>`], its bucketed column storage, the GF(2)-linear hash that partitions it, and the [`BuildAccumulator`] that ingests it.
//!
//! [`BuildAccumulator`]: crate::BuildAccumulator

pub(crate) mod accumulator;
pub(crate) mod hash;
mod partition;
pub(crate) mod storage;

pub use hash::{max_hash_bits, Gf2Hash, PartitionRows};
pub use storage::PauliSum;

#[cfg(test)]
mod tests;
