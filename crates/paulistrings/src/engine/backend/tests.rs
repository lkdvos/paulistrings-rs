use super::*;
use crate::channel::{Channel, Prepare};
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{
    assert_same_terms, assert_terms_close, differential_channels_w1, differential_channels_w2,
    rand_sum,
};

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

/// Per partition: a warm-up layer and its layer pass, so a window exists, then `first` counting ahead for `second` if `count_ahead`, its deferred layer pass, and `second`; merged, with `second`'s exchange counts and whether its export found counts ahead.
fn two_layers<const W: usize>(
    parts: &[PauliSum<W>],
    layers: [&Prepared<W>; 3],
    rows: &PartitionRows<W>,
    n: usize,
    count_ahead: bool,
) -> (PauliSum<W>, Vec<LayerExchangeCounts>, bool) {
    use crate::truncation::builtin::ApproxTopN;
    let transports = InProcessTransport::group(parts.len() as u32);
    let results: Vec<(PauliSum<W>, LayerExchangeCounts, bool)> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .iter()
            .zip(transports)
            .map(|(local, transport)| {
                scope.spawn(move || {
                    let mut part = HostPartition::new(local.clone());
                    let rank = transport.rank();
                    let policy = ApproxTopN(n);
                    let [warm_up, first, second] = layers;
                    let apply = |part: &mut HostPartition<W>, layer, ahead| {
                        let plan = PartitionPlan::new(layer, rows, rank);
                        let counts =
                            part.apply_layer(layer, &plan, rows, &policy, &transport, false, ahead);
                        part.finalize_layer(&policy, &transport);
                        counts
                    };
                    apply(&mut part, warm_up, None);
                    let second_plan = PartitionPlan::new(second, rows, rank);
                    let ahead = match second_plan.remote.as_slice() {
                        [remote] if count_ahead => Some(RowEmitter::of(second, remote)),
                        _ => None,
                    };
                    apply(&mut part, first, ahead.as_ref());
                    let found = part.state.layer.octaves.has_ahead()
                        && part.state.layer.octaves.pending().is_some();
                    let counts = apply(&mut part, second, None);
                    part.settle();
                    part.sum.assert_invariants();
                    (part.sum, counts, found)
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().expect("a partition panicked")).collect()
    });
    let found = results.iter().any(|(_, _, found)| *found);
    let (sums, counts): (Vec<_>, Vec<_>) =
        results.into_iter().map(|(sum, counts, _)| (sum, counts)).unzip();
    (PauliSum::merge_partitions(sums), counts, found)
}

/// An exchanging layer exports from the counts the layer before took ahead, under the cut deferred between them, what it exports from its own count pass.
fn counts_ahead_survive_a_deferred_cut<const W: usize>(
    input: &PauliSum<W>,
    channels: &[(&'static str, Box<dyn Channel<W>>)],
) {
    let num_qubits = input.num_qubits();
    let whole = input.clone().with_hash(Gf2Hash::<W>::new(num_qubits, 5, 0xAB));
    let prepare = |channel: &dyn Channel<W>| channel.prepare(whole.hash(), false).expect("prepare");
    let warm_up = prepare(channels[0].1.as_ref());
    let mut found_any = false;
    // Every third channel leads, which keeps the pairs few while every channel follows.
    for (first_name, first) in channels.iter().step_by(3) {
        let first = prepare(first.as_ref());
        for (second_name, second) in channels {
            let second = prepare(second.as_ref());
            for pbits in [1u8, 2] {
                let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, 0x2468);
                let parts: Vec<PauliSum<W>> = (0..rows.num_partitions() as u32)
                    .map(|rank| whole.filter_partition(&rows, rank))
                    .collect();
                let layers = [&warm_up, &first, &second];
                let n = input.len() / 2;
                let (got, got_counts, found) = two_layers(&parts, layers, &rows, n, true);
                let (want, want_counts, _) = two_layers(&parts, layers, &rows, n, false);
                let what = format!("{first_name} -> {second_name} P={}", rows.num_partitions());
                assert_eq!(got_counts, want_counts, "{what}");
                assert_terms_close(&got, &want, 1e-12, &what);
                found_any |= found;
            }
        }
    }
    assert!(found_any, "no layer pair exported from counts ahead under a pending cut");
}

#[test]
fn counts_ahead_survive_a_deferred_cut_w1() {
    let input = crate::test_support::tie_heavy_sum::<1>(3_000, 8, 0xC09);
    counts_ahead_survive_a_deferred_cut(&input, &differential_channels_w1());
}

#[test]
fn counts_ahead_survive_a_deferred_cut_w2() {
    let input = crate::test_support::tie_heavy_sum::<2>(3_000, 128, 0xC0A);
    counts_ahead_survive_a_deferred_cut(&input, &differential_channels_w2());
}
