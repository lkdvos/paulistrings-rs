use super::*;
use crate::channel::{Channel, Prepare};
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{
    assert_same_terms, assert_terms_close, differential_channels_w1, differential_channels_w2,
    median_octave_cut, rand_sum,
};
use crate::truncation::octave_window::BoundaryCut;

fn detach_leaves_an_empty_partition_under_the_same_hash<const W: usize>(num_qubits: usize) {
    let mut sum = rand_sum::<W>(3_000, num_qubits, 0xDE7AC4 + W as u64);
    sum.refine();
    sum.refine();
    let want = sum.clone();
    let mut part = HostPartition::new(sum);

    let moved = part.detach();

    assert_eq!(part.len(), 0);
    assert!(part.sum.is_empty());
    assert_eq!(part.hash().bits(), want.hash().bits());
    assert!(part.hash().same_rows_as(want.hash()));
    assert_eq!(part.sum.num_qubits(), want.num_qubits());
    assert_eq!(part.sum.num_buckets(), want.num_buckets());
    part.sum.assert_invariants();

    assert_eq!(moved.len(), want.len());
    assert_same_terms(&moved.sum, &want, "detached partition");
    moved.sum.assert_invariants();
}

#[test]
fn detach_leaves_an_empty_partition_under_the_same_hash_w1() {
    detach_leaves_an_empty_partition_under_the_same_hash::<1>(40);
}

#[test]
fn detach_leaves_an_empty_partition_under_the_same_hash_w2() {
    detach_leaves_an_empty_partition_under_the_same_hash::<2>(100);
}

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// One layer on every partition of `parts` through [`HostPartition::apply_layer`], each with its pending cut, merged.
fn host_layer<const W: usize>(
    parts: Vec<(PauliSum<W>, Option<BoundaryCut>)>,
    prepared: &Prepared<W>,
    rows: &PartitionRows<W>,
) -> (PauliSum<W>, Vec<LayerExchangeCounts>) {
    let transports = InProcessTransport::group(parts.len() as u32);
    let results: Vec<(PauliSum<W>, LayerExchangeCounts)> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .into_iter()
            .zip(transports)
            .map(|((local, cut), transport)| {
                scope.spawn(move || {
                    let mut part = HostPartition::new(local);
                    if let Some(cut) = cut {
                        part.state.layer.octaves.defer(cut);
                    }
                    let plan = PartitionPlan::new(prepared, rows, transport.rank());
                    let counts =
                        part.apply_layer(prepared, &plan, rows, &AlwaysKeep, &transport, false);
                    part.sum.assert_invariants();
                    (part.sum, counts)
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().expect("a partition panicked")).collect()
    });
    let (sums, counts): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    (PauliSum::merge_partitions(sums), counts)
}

/// A layer that exchanges under a pending cut gives what it gives once the cut is applied first, sending the same rows.
fn a_pending_cut_survives_an_exchanging_layer<const W: usize>(
    input: &PauliSum<W>,
    channels: &[(&'static str, Box<dyn Channel<W>>)],
) {
    let num_qubits = input.num_qubits();
    for (name, channel) in channels {
        for pbits in [1u8, 2] {
            let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, 0x2468);
            let whole = input.clone().with_hash(Gf2Hash::<W>::new(num_qubits, 5, 0xAB));
            let prepared = channel.prepare(whole.hash(), false).expect("prepare");
            let mut deferred = Vec::new();
            let mut applied = Vec::new();
            for rank in 0..rows.num_partitions() as u32 {
                let local = whole.filter_partition(&rows, rank);
                let cut = median_octave_cut(&local);
                let mut cut_local = local.clone();
                cut.apply(&mut cut_local);
                deferred.push((local, Some(cut)));
                applied.push((cut_local, None));
            }
            let (got, got_counts) = host_layer(deferred, &prepared, &rows);
            let (want, want_counts) = host_layer(applied, &prepared, &rows);
            let what = format!("{name} P={}", rows.num_partitions());
            assert_eq!(got_counts, want_counts, "{what}");
            assert_terms_close(&got, &want, 1e-12, &what);
        }
    }
}

#[test]
fn a_pending_cut_survives_an_exchanging_layer_w1() {
    let input = rand_sum::<1>(3_000, 8, 0xC07);
    a_pending_cut_survives_an_exchanging_layer(&input, &differential_channels_w1());
}

#[test]
fn a_pending_cut_survives_an_exchanging_layer_w2() {
    let input = rand_sum::<2>(3_000, 128, 0xC08);
    a_pending_cut_survives_an_exchanging_layer(&input, &differential_channels_w2());
}
