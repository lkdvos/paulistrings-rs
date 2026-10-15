//! Per-layer classification of a prepared channel's deltas into local (`part(d) = 0`) and remote ones.
//!
//! The partition rows are linear, so a key delta `d` moves every term by the same `part(d)`: remoteness is a property of the delta alone, every partition reaches the same verdict without communicating, and the partner set is symmetric.

use crate::channel::prepared::Prepared;
use crate::pauli_sum::hash::PartitionRows;

/// One delta of a prepared channel that crosses a partition boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RemoteDelta {
    /// Index into `ptm.deltas()`; for a rotation, `0` is the identity pass and `1` the generator pass.
    pub entry: usize,
    /// `rank ^ part(d)`, never `rank` itself.
    pub partner: u32,
    /// `h(d)`.
    pub bucket_delta: u32,
}

/// How one prepared channel's deltas split under a partitioning, seen from one partition.
#[derive(Clone, Debug, Default)]
pub(crate) struct PartitionPlan {
    /// Per entry (numbered as [`RemoteDelta::entry`]), whether its partition delta is zero.
    pub local_entries: Vec<bool>,
    /// The local deltas' bucket deltas, ascending, deduplicated and always containing `0`.
    pub local_bucket_deltas: Vec<u32>,
    /// Ascending by [`RemoteDelta::entry`].
    pub remote: Vec<RemoteDelta>,
    /// Non-identity realized deltas, local and remote together, so the sort-kernel gate sees the channel's whole fanout.
    pub rest_streams_total: usize,
}

impl PartitionPlan {
    /// Classify `prepared`'s deltas under `rows`, as seen from partition `rank`, keeping the allocations across layers.
    pub(crate) fn rebuild<const W: usize>(
        &mut self,
        prepared: &Prepared<W>,
        rows: &PartitionRows<W>,
        rank: u32,
    ) {
        debug_assert!(
            rows.num_partitions() > rank as usize,
            "PartitionPlan: rank {rank} outside {} partitions",
            rows.num_partitions(),
        );
        self.local_entries.clear();
        self.local_bucket_deltas.clear();
        self.local_bucket_deltas.push(0);
        self.remote.clear();

        let mut classify =
            |entry: usize, mask_x: &[u64; W], mask_z: &[u64; W], bucket_delta: u32| {
                let partition_delta = rows.partition_of(mask_x, mask_z);
                self.local_entries.push(partition_delta == 0);
                if partition_delta == 0 {
                    self.local_bucket_deltas.push(bucket_delta);
                } else {
                    self.remote.push(RemoteDelta {
                        entry,
                        partner: rank ^ partition_delta,
                        bucket_delta,
                    });
                }
            };
        match prepared {
            Prepared::Local(ptm) => {
                for (entry, delta) in ptm.deltas().iter().enumerate() {
                    let (mask_x, mask_z) = delta.mask();
                    classify(entry, &mask_x, &mask_z, delta.bucket_delta);
                }
            }
            Prepared::Rotation(rotation) => {
                let (generator_x, generator_z) = rotation.generator_mask();
                classify(0, &[0u64; W], &[0u64; W], rotation.bucket_delta_identity);
                classify(1, &generator_x, &generator_z, rotation.bucket_delta_generator);
            }
        }

        self.rest_streams_total = match prepared {
            Prepared::Local(ptm) => {
                debug_assert!(
                    !ptm.has_identity() || self.local_entries[0],
                    "the identity delta must be local: part(0) = 0",
                );
                ptm.rest_streams()
            }
            Prepared::Rotation(_) => {
                debug_assert!(self.local_entries[0], "the identity pass must be local");
                1
            }
        };

        self.local_bucket_deltas.sort_unstable();
        self.local_bucket_deltas.dedup();
    }

    /// Whether this layer moves anything across a partition boundary.
    pub(crate) fn has_remote(&self) -> bool {
        !self.remote.is_empty()
    }

    /// The index of remote delta `remote`'s block among those its partner sends this partition: the `j`-th remote delta with partner `q` is the `j`-th block received from `q`, since remoteness is symmetric.
    pub(crate) fn received_block_index(&self, remote: usize) -> usize {
        self.index_among_partner(remote)
    }

    /// The index of remote delta `remote`'s block among those this partition sends its partner, which an export fills in plan order per partner.
    pub(crate) fn sent_block_index(&self, remote: usize) -> usize {
        self.index_among_partner(remote)
    }

    fn index_among_partner(&self, remote: usize) -> usize {
        let partner = self.remote[remote].partner;
        self.remote[..remote].iter().filter(|other| other.partner == partner).count()
    }
}

#[cfg(test)]
mod tests;
