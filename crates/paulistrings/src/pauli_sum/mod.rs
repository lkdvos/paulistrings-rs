//! [`PauliSum<W, C>`], its bucketed column storage, the GF(2)-linear hash that partitions it, and the [`BuildAccumulator`] that ingests it.
//!
//! [`BuildAccumulator`]: crate::BuildAccumulator

pub(crate) mod accumulator;
pub(crate) mod hash;
mod partition;
mod product;
pub(crate) mod storage;
mod table;

pub use hash::{max_hash_bits, Gf2Hash, PartitionRows};
pub use storage::PauliSum;

#[cfg(test)]
mod tests;
