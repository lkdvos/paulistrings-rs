//! Per-layer classification of a prepared channel's deltas into *local* and
//! *remote*. See ARCHITECTURE.md §Engine for the delta set itself.
//!
//! A partitioning splits keys by `part(v) = P·v` ([`PartitionRows`]) on top of
//! the bucket split `loc(v) = H·v` ([`Gf2Hash`]). Both are GF(2)-linear, so a
//! prepared channel's key delta `d` moves a term by a *constant* partition
//! delta `pd = part(d)` and bucket delta `bd = h(d)`, whatever the term. That
//! is what makes this a per-layer plan rather than a per-term decision:
//!
//! * `pd == 0` — the delta is **local**: it lands in the same partition, and the ordinary coset loop handles it with no communication.
//! * `pd != 0` — the delta is **remote**: every row it produces belongs to partition `rank ^ pd`, so the layer exports one stream per such delta to that one partner.
//!
//! The identity delta has mask `0` and `part(0) = 0`, so it is always local: a partition never has to ship a term to itself.

use crate::channel::prepared::Prepared;
#[cfg(any(test, feature = "test-utils"))]
use crate::circuit::Circuit;
#[cfg(any(test, feature = "test-utils"))]
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::hash::PartitionRows;

/// One delta of a prepared channel that crosses a partition boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RemoteDelta {
    /// Index into `ptm.deltas()`.
    /// For [`Prepared::Rotation`] the two implicit entries are numbered the way the engine's plan numbers them: the identity pass is `0`, the generator pass is `1`.
    pub entry: usize,
    /// The partition every row of this delta belongs to, `rank ^ partition_delta`.
    pub partner: u32,
    /// `h(d)` — the entry's bucket delta, carried so the export pass can name
    /// the destination bucket without re-hashing.
    pub bucket_delta: u32,
    /// `part(d)`, nonzero by construction.
    pub partition_delta: u32,
}

/// How one prepared channel's deltas split under a partitioning, from the
/// point of view of one partition (`rank`).
#[derive(Clone, Debug)]
pub(crate) struct PartitionPlan {
    /// Per entry, `true` if the entry's partition delta is zero.
    ///
    /// Length is `ptm.deltas().len()` for [`Prepared::Local`] and 2 for [`Prepared::Rotation`] (identity pass, generator pass).
    /// Indices line up with [`RemoteDelta::entry`], so the two together cover every entry exactly once.
    pub local_entries: Vec<bool>,
    /// The local deltas' bucket deltas, ascending and deduplicated — the input to `Gf2Span::new` for the coset loop that runs inside this partition.
    ///
    /// Always contains `0`: the identity delta is local, and `0` is in every span regardless, so including it unconditionally cannot change the span it generates.
    pub local_bucket_deltas: Vec<u32>,
    /// The remote deltas, ascending by [`RemoteDelta::entry`].
    pub remote: Vec<RemoteDelta>,
    /// Non-identity *realized* deltas, local and remote together.
    ///
    /// The gather's sort-kernel choice (`merge::RADIX_MIN_REST_STREAMS`) turns on how wide a channel's fanout is, which is a property of the channel, not of how this partition happens to see it — so the gate reads the total, not the local count.
    pub rest_streams_total: usize,
}

impl PartitionPlan {
    /// Classify `prep`'s deltas under `rows`, as seen from partition `rank`.
    ///
    /// With [`PartitionRows::none`] every delta is local and `local_bucket_deltas` is exactly `prep.bucket_deltas()`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `rank` is not a partition of `rows`, or if the identity delta somehow classifies as remote (it cannot: `part(0) = 0`).
    pub(crate) fn new<const W: usize>(
        prep: &Prepared<W>,
        rows: &PartitionRows<W>,
        rank: u32,
    ) -> Self {
        debug_assert!(
            rows.num_partitions() > rank as usize,
            "PartitionPlan: rank {rank} outside {} partitions",
            rows.num_partitions(),
        );

        // `(mask_x, mask_z, bucket_delta)` per entry, in entry order.
        // Built once so the classification below is one loop for both variants; this runs once per layer, so the small allocation is free.
        let entries: Vec<([u64; W], [u64; W], u32)> = match prep {
            Prepared::Local(ptm) => ptm
                .deltas()
                .iter()
                .map(|d| {
                    let (mx, mz) = d.mask();
                    (mx, mz, d.bucket_delta)
                })
                .collect(),
            // The rotation's two implicit entries: the identity pass (mask 0) and the generator pass (mask `P`).
            Prepared::Rotation(r) => {
                let (gx, gz) = r.gen_mask();
                vec![
                    ([0u64; W], [0u64; W], r.bucket_delta_identity),
                    (gx, gz, r.bucket_delta_gen),
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

        // `deltas()` is the *realized* delta set (ARCHITECTURE.md §Bucketing), so its length minus the identity entry is the number of streams a gather concatenates into a run's rest columns — the quantity `engine::bucketed`'s own plan calls `rest_streams`.
        // The rotation has exactly one non-identity pass at any generator weight.
        let rest_streams_total = match prep {
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

    /// `true` if this layer moves anything across a partition boundary.
    pub(crate) fn has_remote(&self) -> bool {
        !self.remote.is_empty()
    }

    /// The remote deltas destined for partition `q`, ascending by entry.
    pub(crate) fn remote_for_partner(&self, q: u32) -> impl Iterator<Item = &RemoteDelta> {
        self.remote.iter().filter(move |r| r.partner == q)
    }
}

/// Per layer, `(local delta count, remote delta count)` under `rows`, for `circuit`'s channels prepared against `hash`.
///
/// Layers are reported in **application order**: circuit order for `adjoint == false`, reverse order for `adjoint == true`, matching [`Direction::Heisenberg`](crate::Direction::Heisenberg).
///
/// A diagnostic answering "how much of this circuit crosses a partition boundary?" without running a layer.
/// The counts do not depend on `rank`, so it reports from partition 0.
///
/// # Panics
///
/// Panics if any channel declines [`Channel::prepare`](crate::Channel::prepare), the same condition on which `propagate` panics.
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
        .map(|idx| {
            let prep = circuit.channels[idx]
                .prepare(hash, adjoint)
                .unwrap_or_else(|| {
                    panic!("count_remote_deltas: layer {idx} declined Channel::prepare")
                });
            let plan = PartitionPlan::new(&prep, rows, 0);
            let local = plan.local_entries.iter().filter(|b| **b).count();
            (local, plan.remote.len())
        })
        .collect()
}

#[cfg(test)]
mod tests;
