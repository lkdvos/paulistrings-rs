//! Per-layer classification of a prepared channel's deltas into local (`part(d) = 0`) and remote ones (ARCHITECTURE.md §Partitioning).

use crate::channel::prepared::Prepared;
#[cfg(any(test, feature = "test-utils"))]
use crate::circuit::Circuit;
#[cfg(any(test, feature = "test-utils"))]
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::hash::PartitionRows;

/// One delta of a prepared channel that crosses a partition boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RemoteDelta {
    /// Index into `ptm.deltas()`; for a rotation, `0` is the identity pass and `1` the generator pass.
    pub entry: usize,
    /// `rank ^ partition_delta`.
    pub partner: u32,
    /// `h(d)`.
    pub bucket_delta: u32,
    /// `part(d)`, nonzero by construction.
    pub partition_delta: u32,
}

/// How one prepared channel's deltas split under a partitioning, seen from one partition.
#[derive(Clone, Debug)]
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
    /// Classify `prepared`'s deltas under `rows`, as seen from partition `rank`.
    pub(crate) fn new<const W: usize>(
        prepared: &Prepared<W>,
        rows: &PartitionRows<W>,
        rank: u32,
    ) -> Self {
        debug_assert!(
            rows.num_partitions() > rank as usize,
            "PartitionPlan: rank {rank} outside {} partitions",
            rows.num_partitions(),
        );

        // `(mask_x, mask_z, bucket_delta)` per entry, in entry order.
        let entries: Vec<([u64; W], [u64; W], u32)> = match prepared {
            Prepared::Local(ptm) => ptm
                .deltas()
                .iter()
                .map(|d| {
                    let (mx, mz) = d.mask();
                    (mx, mz, d.bucket_delta)
                })
                .collect(),
            Prepared::Rotation(rotation) => {
                let (gx, gz) = rotation.generator_mask();
                vec![
                    ([0u64; W], [0u64; W], rotation.bucket_delta_identity),
                    (gx, gz, rotation.bucket_delta_generator),
                ]
            }
        };

        let mut local_entries: Vec<bool> = Vec::with_capacity(entries.len());
        let mut local_bucket_deltas: Vec<u32> = vec![0];
        let mut remote: Vec<RemoteDelta> = Vec::new();

        for (entry, (mask_x, mask_z, bucket_delta)) in entries.iter().enumerate() {
            let partition_delta = rows.partition_of(mask_x, mask_z);
            local_entries.push(partition_delta == 0);
            if partition_delta == 0 {
                local_bucket_deltas.push(*bucket_delta);
            } else {
                remote.push(RemoteDelta {
                    entry,
                    partner: rank ^ partition_delta,
                    bucket_delta: *bucket_delta,
                    partition_delta,
                });
            }
        }

        // `engine::bucketed`'s `rest_streams`, over the whole realized delta set (ARCHITECTURE.md §Bucketing).
        let rest_streams_total = match prepared {
            Prepared::Local(ptm) => {
                let has_identity = ptm.deltas().first().is_some_and(|d| d.local_delta == 0);
                debug_assert!(
                    !has_identity || local_entries[0],
                    "the identity delta must be local: part(0) = 0",
                );
                ptm.deltas().len() - has_identity as usize
            }
            Prepared::Rotation(_) => {
                debug_assert!(local_entries[0], "the identity pass must be local");
                1
            }
        };

        local_bucket_deltas.sort_unstable();
        local_bucket_deltas.dedup();

        Self {
            local_entries,
            local_bucket_deltas,
            remote,
            rest_streams_total,
        }
    }

    /// Whether this layer moves anything across a partition boundary.
    pub(crate) fn has_remote(&self) -> bool {
        !self.remote.is_empty()
    }

    /// The remote deltas destined for partition `q`, ascending by entry.
    pub(crate) fn remote_for_partner(&self, q: u32) -> impl Iterator<Item = &RemoteDelta> {
        self.remote.iter().filter(move |r| r.partner == q)
    }
}

/// Per layer in application order, `(local, remote)` delta counts of `circuit`'s channels prepared against `hash` under `rows`.
///
/// # Panics
///
/// If any channel declines [`Channel::prepare`](crate::Channel::prepare).
#[cfg(any(test, feature = "test-utils"))]
pub fn count_remote_deltas<const W: usize>(
    circuit: &Circuit<W>,
    hash: &Gf2Hash<W>,
    rows: &PartitionRows<W>,
    adjoint: bool,
) -> Vec<(usize, usize)> {
    let n = circuit.channels.len();
    (0..n)
        .map(|k| if adjoint { n - 1 - k } else { k })
        .map(|index| {
            let prepared = circuit.channels[index]
                .prepare(hash, adjoint)
                .unwrap_or_else(|| {
                    panic!("count_remote_deltas: layer {index} declined Channel::prepare")
                });
            let plan = PartitionPlan::new(&prepared, rows, 0);
            let local = plan.local_entries.iter().filter(|b| **b).count();
            (local, plan.remote.len())
        })
        .collect()
}

#[cfg(test)]
mod tests;
