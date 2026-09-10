//! Per-layer classification of a prepared channel's deltas into *local* and
//! *remote*. See ARCHITECTURE.md §Engine for the delta set itself.
//!
//! A partitioning splits keys by `part(v) = P·v` ([`PartitionRows`]) on top of
//! the bucket split `loc(v) = H·v` ([`Gf2Hash`]). Both are GF(2)-linear, so a
//! prepared channel's key delta `d` moves a term by a *constant* partition
//! delta `pd = part(d)` and bucket delta `bd = h(d)`, whatever the term. That
//! is what makes this a per-layer plan rather than a per-term decision:
//!
//! * `pd == 0` — the delta is **local**: it lands in the same partition, and
//!   the ordinary coset loop handles it with no communication.
//! * `pd != 0` — the delta is **remote**: every row it produces belongs to
//!   partition `rank ^ pd`, so the layer exports one stream per such delta to
//!   that one partner.
//!
//! The identity delta has mask `0` and `part(0) = 0`, so it is always local: a
//! partition never has to ship a term to itself.

use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::channel::prepared::Prepared;
use crate::circuit::Circuit;

/// One delta of a prepared channel that crosses a partition boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RemoteDelta {
    /// Index into `ptm.deltas()`. For [`Prepared::Rotation`] the two implicit
    /// entries are numbered the way the engine's plan numbers them: the
    /// identity pass is `0`, the generator pass is `1`.
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
    /// Length is `ptm.deltas().len()` for [`Prepared::Local`] and 2 for
    /// [`Prepared::Rotation`] (identity pass, generator pass). Indices line up
    /// with [`RemoteDelta::entry`], so the two together cover every entry
    /// exactly once.
    pub local_entries: Vec<bool>,
    /// The local deltas' bucket deltas, ascending and deduplicated — the input
    /// to `Gf2Span::new` for the coset loop that runs inside this partition.
    ///
    /// Always contains `0`: the identity delta is local, and `0` is in every
    /// span regardless, so including it unconditionally cannot change the
    /// span it generates.
    pub local_bucket_deltas: Vec<u32>,
    /// The remote deltas, ascending by [`RemoteDelta::entry`].
    pub remote: Vec<RemoteDelta>,
    /// Non-identity *realized* deltas, local and remote together.
    ///
    /// The gather's sort-kernel choice (`merge::RADIX_MIN_REST_STREAMS`) turns
    /// on how wide a channel's fanout is, which is a property of the channel,
    /// not of how this partition happens to see it — so the gate reads the
    /// total, not the local count.
    pub rest_streams_total: usize,
}

impl PartitionPlan {
    /// Classify `prep`'s deltas under `rows`, as seen from partition `rank`.
    ///
    /// With [`PartitionRows::none`] every delta is local and
    /// `local_bucket_deltas` is exactly `prep.bucket_deltas()`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `rank` is not a partition of `rows`, or if the
    /// identity delta somehow classifies as remote (it cannot: `part(0) = 0`).
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

        // `(mask_x, mask_z, bucket_delta)` per entry, in entry order. Built
        // once so the classification below is one loop for both variants; this
        // runs once per layer, so the small allocation is free.
        let entries: Vec<([u64; W], [u64; W], u32)> = match prep {
            Prepared::Local(ptm) => ptm
                .deltas()
                .iter()
                .map(|d| {
                    let (mx, mz) = d.mask();
                    (mx, mz, d.bucket_delta)
                })
                .collect(),
            // The rotation's two implicit entries: the identity pass (mask 0)
            // and the generator pass (mask `P`).
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

        // `deltas()` is the *realized* delta set (ARCHITECTURE.md §Bucketing),
        // so its length minus the identity entry is the number of streams a
        // gather concatenates into a run's rest columns — the quantity
        // `engine::bucketed`'s own plan calls `rest_streams`. The rotation has
        // exactly one non-identity pass at any generator weight.
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

/// Per layer, `(local delta count, remote delta count)` under `rows`, for
/// `circuit`'s channels prepared against `hash`.
///
/// Layers are reported in **application order**: circuit order for
/// `adjoint == false`, reverse order for `adjoint == true`, matching
/// [`Direction::Heisenberg`](crate::Direction::Heisenberg).
///
/// A research-phase diagnostic — it answers "how much of this circuit crosses a
/// partition boundary?" without running a layer. The counts do not depend on
/// `rank`, so it reports from partition 0.
///
/// # Panics
///
/// Panics if any channel declines [`Channel::prepare`](crate::Channel::prepare),
/// the same condition on which `propagate` panics.
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
impl PartitionPlan {
    /// The partitions this layer exports to, distinct and ascending.
    ///
    /// The engine routes by [`Self::remote`] directly; this is the tests' way
    /// of asking the same question as a set.
    fn partners(&self) -> impl Iterator<Item = u32> + '_ {
        let mut v: Vec<u32> = self.remote.iter().map(|r| r.partner).collect();
        v.sort_unstable();
        v.dedup();
        v.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::rotation::PauliRotation;
    use crate::channel::{Channel, GeneralUnitary2Q};
    use crate::pauli_string::PauliString;
    use crate::test_support::{haar_su4_matrix, Xs64};

    const NQ: usize = 128;

    fn hash() -> Gf2Hash<2> {
        Gf2Hash::<2>::new(NQ, 8, 0xB00C)
    }

    /// One partition row that is `Z` on `qubit` and nothing else: `part(v)` is
    /// then the z-bit of `v` on that qubit (the row's z-mask is what meets the
    /// key's z-word in `partition_of`).
    fn z_row(qubit: u32) -> PartitionRows<2> {
        let mut rz = [0u64; 2];
        rz[qubit as usize / 64] = 1u64 << (qubit % 64);
        PartitionRows::<2>::from_rows(NQ, vec![[0u64; 2]], vec![rz])
    }

    fn prep_of<C: Channel<2>>(ch: &C) -> Prepared<2> {
        ch.prepare(&hash(), false).unwrap()
    }

    // ---- the trivial partitioning keeps everything local ----

    #[test]
    fn without_partition_rows_nothing_is_remote() {
        let rows = PartitionRows::<2>::none(NQ);
        for (name, ch) in [
            ("h", Box::new(Clifford1Q::h(3)) as Box<dyn Channel<2>>),
            ("cnot", Box::new(Clifford2Q::cnot(1, 4))),
            (
                "haar",
                Box::new(GeneralUnitary2Q::from_matrix(1, 5, haar_su4_matrix())),
            ),
        ] {
            let prep = ch.prepare(&hash(), false).unwrap();
            let plan = PartitionPlan::new(&prep, &rows, 0);
            assert!(!plan.has_remote(), "{name}: unexpected remote deltas");
            assert!(plan.remote.is_empty(), "{name}");
            assert_eq!(plan.partners().count(), 0, "{name}");
            assert!(plan.local_entries.iter().all(|b| *b), "{name}");
            assert_eq!(
                plan.local_bucket_deltas,
                prep.bucket_deltas(),
                "{name}: local span input differs from the channel's own",
            );
            let Prepared::Local(ptm) = &prep else {
                panic!("{name}: expected Local")
            };
            assert_eq!(plan.rest_streams_total, ptm.num_deltas() - 1, "{name}");
        }
    }

    #[test]
    fn a_wide_rotation_without_partition_rows_is_all_local() {
        let mut gen = PauliString::<2>::z(1);
        for q in [5u32, 66, 100] {
            gen.mul_assign(&PauliString::<2>::z(q));
        }
        let prep = prep_of(&PauliRotation::new(gen, 0.41));
        assert!(matches!(prep, Prepared::Rotation(_)));
        let plan = PartitionPlan::new(&prep, &PartitionRows::<2>::none(NQ), 0);
        assert_eq!(plan.local_entries, vec![true, true]);
        assert!(!plan.has_remote());
        assert_eq!(plan.local_bucket_deltas, prep.bucket_deltas());
        assert_eq!(plan.rest_streams_total, 1);
    }

    // ---- a crafted row makes one delta remote ----

    #[test]
    fn a_delta_whose_mask_sets_the_partition_bit_is_remote() {
        // H(3)'s delta set is {0, XZ on qubit 3}. A single partition row that
        // is Z on qubit 3 reads the z-bit there, which that mask sets, so
        // `part(XZ_3) = 1`.
        let rows = z_row(3);
        assert_eq!(rows.num_partitions(), 2);
        let prep = prep_of(&Clifford1Q::h(3));
        let Prepared::Local(ptm) = &prep else {
            panic!("expected Local")
        };
        assert_eq!(ptm.num_deltas(), 2);

        for rank in 0..2u32 {
            let plan = PartitionPlan::new(&prep, &rows, rank);
            assert_eq!(plan.local_entries, vec![true, false], "rank {rank}");
            assert!(plan.has_remote(), "rank {rank}");
            assert_eq!(
                plan.remote,
                vec![RemoteDelta {
                    entry: 1,
                    partner: rank ^ 1,
                    bucket_delta: ptm.deltas()[1].bucket_delta,
                    partition_delta: 1,
                }],
                "rank {rank}",
            );
            // Only the identity delta stays behind, so the local coset loop is
            // over a single bucket.
            assert_eq!(plan.local_bucket_deltas, vec![0], "rank {rank}");
            // ...but the sort-kernel gate still sees the channel's full fanout.
            assert_eq!(plan.rest_streams_total, 1, "rank {rank}");
            assert_eq!(
                plan.partners().collect::<Vec<_>>(),
                vec![rank ^ 1],
                "rank {rank}",
            );
            assert_eq!(
                plan.remote_for_partner(rank ^ 1).count(),
                1,
                "rank {rank}: partner stream",
            );
            assert_eq!(plan.remote_for_partner(rank).count(), 0, "rank {rank}");
        }
    }

    #[test]
    fn a_rotation_whose_generator_crosses_is_one_remote_delta() {
        let mut gen = PauliString::<2>::z(1);
        for q in [5u32, 66, 100] {
            gen.mul_assign(&PauliString::<2>::z(q));
        }
        // A Z row on qubit 1 reads the generator's z-bit there, which is set —
        // the generator is a product of `Z`s, so an x-row would read nothing.
        let rows = z_row(1);
        let prep = prep_of(&PauliRotation::new(gen, 0.41));
        let Prepared::Rotation(r) = &prep else {
            panic!("expected Rotation")
        };

        let plan = PartitionPlan::new(&prep, &rows, 1);
        assert_eq!(plan.local_entries, vec![true, false]);
        assert_eq!(plan.local_bucket_deltas, vec![0]);
        assert_eq!(
            plan.remote,
            vec![RemoteDelta {
                entry: 1,
                partner: 0,
                bucket_delta: r.bucket_delta_gen,
                partition_delta: 1,
            }],
        );
        assert_eq!(plan.rest_streams_total, 1);
    }

    // ---- the two classes partition the entry set ----

    #[test]
    fn local_and_remote_cover_every_entry_exactly_once() {
        let mut rng = Xs64::new(0xA11CE);
        let prep = prep_of(&GeneralUnitary2Q::from_matrix(1, 5, haar_su4_matrix()));
        let Prepared::Local(ptm) = &prep else {
            panic!("expected Local")
        };
        // The dense fixture realizes all sixteen deltas.
        assert_eq!(ptm.num_deltas(), 16);

        for trial in 0..32 {
            let bits = 1 + (trial % 3);
            let rows_x: Vec<[u64; 2]> = (0..bits).map(|_| rng.next_array::<2>()).collect();
            let rows_z: Vec<[u64; 2]> = (0..bits).map(|_| rng.next_array::<2>()).collect();
            let rows = PartitionRows::<2>::from_rows(NQ, rows_x, rows_z);
            let rank = (rng.next_u64() as u32) % rows.num_partitions() as u32;
            let plan = PartitionPlan::new(&prep, &rows, rank);

            assert_eq!(plan.local_entries.len(), ptm.num_deltas());
            let n_local = plan.local_entries.iter().filter(|b| **b).count();
            assert_eq!(n_local + plan.remote.len(), ptm.num_deltas());
            // Remote entries are exactly the false slots, in ascending order.
            let remote_entries: Vec<usize> = plan.remote.iter().map(|r| r.entry).collect();
            let expected: Vec<usize> = plan
                .local_entries
                .iter()
                .enumerate()
                .filter(|(_, b)| !**b)
                .map(|(e, _)| e)
                .collect();
            assert_eq!(remote_entries, expected, "trial {trial}");
            for r in &plan.remote {
                assert_ne!(r.partition_delta, 0);
                assert_eq!(r.partner, rank ^ r.partition_delta);
                assert_ne!(r.partner, rank, "a partition never ships to itself");
                assert_eq!(r.bucket_delta, ptm.deltas()[r.entry].bucket_delta);
            }
            // Partners: distinct and ascending, and every remote delta's
            // partner appears.
            let partners: Vec<u32> = plan.partners().collect();
            assert!(partners.windows(2).all(|w| w[0] < w[1]), "trial {trial}");
            for r in &plan.remote {
                assert!(partners.contains(&r.partner), "trial {trial}");
            }
            assert_eq!(
                partners
                    .iter()
                    .map(|&q| plan.remote_for_partner(q).count())
                    .sum::<usize>(),
                plan.remote.len(),
                "trial {trial}",
            );
            // The local span input is exactly the local entries' bucket deltas
            // (plus 0, which the identity always contributes anyway).
            let mut want: Vec<u32> = vec![0];
            for (e, keep) in plan.local_entries.iter().enumerate() {
                if *keep {
                    want.push(ptm.deltas()[e].bucket_delta);
                }
            }
            want.sort_unstable();
            want.dedup();
            assert_eq!(plan.local_bucket_deltas, want, "trial {trial}");
            assert_eq!(plan.rest_streams_total, 15, "trial {trial}");
        }
    }

    // ---- the diagnostic ----

    #[test]
    fn count_remote_deltas_reports_layers_in_application_order() {
        // Two layers with hand-checkable delta sets under a Z-row on qubit 3:
        //   H(3):  deltas {0, XZ_3} -> part = {0, 1}  -> (1 local, 1 remote)
        //   S(9):  deltas {0,  Z_9} -> part = {0, 0}  -> (2 local, 0 remote)
        // (S maps X -> Y and Y -> -X, so its only nonzero delta is a z-bit.)
        let rows = z_row(3);
        let h = hash();
        let mut circuit = Circuit::<2>::new(NQ);
        circuit.push(Clifford1Q::h(3));
        circuit.push(Clifford1Q::s(9));

        assert_eq!(
            count_remote_deltas(&circuit, &h, &rows, false),
            vec![(1, 1), (2, 0)],
        );
        // Heisenberg order is the reverse. `S` is not self-adjoint, but its
        // adjoint's delta set is the same subspace, so the counts do not move.
        assert_eq!(
            count_remote_deltas(&circuit, &h, &rows, true),
            vec![(2, 0), (1, 1)],
        );
    }

    #[test]
    fn count_remote_deltas_without_rows_is_all_local() {
        let rows = PartitionRows::<2>::none(NQ);
        let h = hash();
        let mut circuit = Circuit::<2>::new(NQ);
        circuit.push(Clifford2Q::cnot(1, 4));
        circuit.push(Clifford1Q::h(3));
        assert_eq!(
            count_remote_deltas(&circuit, &h, &rows, false),
            vec![(4, 0), (2, 0)],
        );
    }
}
