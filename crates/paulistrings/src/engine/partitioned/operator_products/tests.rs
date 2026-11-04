use num_complex::Complex64;

use super::*;
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::test_support::{assert_terms_close, low_weight_sum, KeepAll, Xs64};
use crate::truncation::{CoefficientThreshold, TopN};

/// `sum` cut into one share per partition of `rows`.
fn split<const W: usize, C: Coefficient>(
    sum: &PauliSum<W, C>,
    rows: &PartitionRows<W>,
) -> Vec<PauliSum<W, C>> {
    (0..rows.num_partitions() as u32).map(|r| sum.filter_partition(rows, r)).collect()
}

/// `f` on every share, one thread per rank over an in-process transport group.
fn on_ranks<const W: usize, C: Coefficient>(
    shares: &[PauliSum<W, C>],
    f: impl Fn(&PauliSum<W, C>, &InProcessTransport) -> PauliSum<W, C> + Sync,
) -> Vec<PauliSum<W, C>> {
    let transports = InProcessTransport::group(shares.len() as u32);
    std::thread::scope(|scope| {
        let handles: Vec<_> = shares
            .iter()
            .zip(transports)
            .map(|(share, transport)| {
                let f = &f;
                scope.spawn(move || f(share, &transport))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank panicked")).collect()
    })
}

fn assert_shares_belong<const W: usize, C: Coefficient>(
    shares: &[PauliSum<W, C>],
    rows: &PartitionRows<W>,
) {
    for (rank, share) in shares.iter().enumerate() {
        share.assert_invariants();
        assert!(
            share.iter().all(|(x, z, _)| rows.partition_of(x, z) == rank as u32),
            "rank {rank}"
        );
    }
}

/// Every product of a nonlocal generator and a split operand against the single-process kernel.
fn agrees_with_one_process<const W: usize>(partitions: usize, num_qubits: usize, seed: u64) {
    let generator = low_weight_sum::<W>(40, num_qubits, 2, seed);
    let operand = low_weight_sum::<W>(6_000, num_qubits, 4, seed + 1);
    let rows =
        PartitionRows::<W>::from_seed(num_qubits, partitions.trailing_zeros() as u8, seed + 2);
    let shares = split(&operand, &rows);
    let what = format!("P={partitions} W={W}");

    let got = on_ranks(&shares, |s, t| liouvillian_share(&generator, s, &rows, t, &KeepAll));
    assert_shares_belong(&got, &rows);
    let want = generator.liouvillian(&operand);
    assert_terms_close(
        &PauliSum::merge_partitions(got),
        &want,
        1e-12,
        &format!("{what}: liouvillian"),
    );

    let got = on_ranks(&shares, |s, t| product_share(&generator, s, &rows, t, &KeepAll));
    assert_shares_belong(&got, &rows);
    assert_terms_close(
        &PauliSum::merge_partitions(got),
        &generator.product(&operand),
        1e-12,
        &format!("{what}: product"),
    );

    let got = on_ranks(&shares, |s, t| anticommutator_share(&generator, s, &rows, t, &KeepAll));
    assert_terms_close(
        &PauliSum::merge_partitions(got),
        &generator.anticommutator(&operand),
        1e-12,
        &format!("{what}: anticommutator"),
    );
}

#[test]
fn every_product_agrees_with_one_process_w1() {
    agrees_with_one_process::<1>(2, 30, 1);
    agrees_with_one_process::<1>(4, 30, 2);
}

#[test]
fn every_product_agrees_with_one_process_w2() {
    agrees_with_one_process::<2>(2, 90, 3);
    agrees_with_one_process::<2>(4, 90, 4);
}

#[test]
fn shares_at_different_bucket_counts_agree() {
    let generator = low_weight_sum::<1>(30, 30, 2, 5);
    let operand = low_weight_sum::<1>(20_000, 30, 4, 6);
    let rows = PartitionRows::<1>::from_seed(30, 1, 7);
    let mut shares = split(&operand, &rows);
    shares[1].refine();
    assert_ne!(shares[0].hash().bits(), shares[1].hash().bits());
    let got = on_ranks(&shares, |s, t| liouvillian_share(&generator, s, &rows, t, &KeepAll));
    assert_terms_close(
        &PauliSum::merge_partitions(got),
        &generator.liouvillian(&operand),
        1e-12,
        "bits",
    );
}

#[test]
fn a_filter_policy_sees_the_whole_key() {
    let generator = low_weight_sum::<1>(30, 30, 2, 8);
    let operand = low_weight_sum::<1>(6_000, 30, 4, 9);
    let rows = PartitionRows::<1>::from_seed(30, 2, 10);
    let policy = CoefficientThreshold(0.3);
    let got = on_ranks(&split(&operand, &rows), |s, t| {
        liouvillian_share(&generator, s, &rows, t, &policy)
    });
    let want = generator.liouvillian_with(&operand, &policy);
    assert!(want.len() < generator.liouvillian(&operand).len());
    assert_terms_close(&PauliSum::merge_partitions(got), &want, 1e-12, "threshold");
}

#[test]
#[should_panic(expected = "a rank panicked")]
fn a_layer_pass_without_a_partitioned_form_is_rejected() {
    let generator = low_weight_sum::<1>(10, 20, 2, 11);
    let operand = low_weight_sum::<1>(500, 20, 4, 12);
    let rows = PartitionRows::<1>::from_seed(20, 1, 13);
    on_ranks(&split(&operand, &rows), |s, t| liouvillian_share(&generator, s, &rows, t, &TopN(10)));
}

#[test]
#[should_panic(expected = "a rank panicked")]
fn ranks_holding_different_generators_fail() {
    let operand = low_weight_sum::<1>(500, 20, 4, 14);
    let rows = PartitionRows::<1>::from_seed(20, 1, 15);
    // `Xs64` seeds with `seed | 1`, so the two seeds must differ above bit 0.
    let generators = [low_weight_sum::<1>(10, 20, 2, 16), low_weight_sum::<1>(10, 20, 2, 18)];
    let shares = split(&operand, &rows);
    let transports = InProcessTransport::group(2);
    std::thread::scope(|scope| {
        let handles: Vec<_> = shares
            .iter()
            .zip(transports)
            .zip(&generators)
            .map(|((share, transport), generator)| {
                let rows = &rows;
                scope.spawn(move || liouvillian_share(generator, share, rows, &transport, &KeepAll))
            })
            .collect();
        for h in handles {
            h.join().expect("a rank panicked");
        }
    });
}

/// The Heisenberg chain's bonds with random coefficients, so no two contributions cancel exactly.
fn heisenberg_bonds(num_qubits: usize, seed: u64) -> PauliSum<1> {
    let mut rng = Xs64::new(seed);
    let mut accumulator = BuildAccumulator::<1>::new(num_qubits);
    for j in 0..num_qubits - 1 {
        for f in [PauliString::<1>::x, PauliString::<1>::y, PauliString::<1>::z] {
            let mut p = f(j);
            p.mul_assign(&f(j + 1));
            let c = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
            accumulator.add_term(p, Complex64::new(c, 0.0));
        }
    }
    accumulator.finalize()
}

#[test]
fn cut_rows_send_a_halo_of_anticommuting_terms() {
    let n = 32;
    let generator = heisenberg_bonds(n, 18);
    let operand = low_weight_sum::<1>(20_000, n, 4, 19);
    let blocks: Vec<Vec<usize>> = (0..4).map(|b| (8 * b..8 * b + 8).collect()).collect();
    let rows = PartitionRows::<1>::cut(n, &blocks);
    let shares = split(&operand, &rows);

    let mut by_delta: ByDelta<1, Complex64> = BTreeMap::new();
    for (x, z, c) in generator.iter() {
        by_delta.entry(rows.partition_of(x, z)).or_default().push(Entry { x: *x, z: *z, c });
    }
    // Only the YY and ZZ bonds across the three block boundaries move a term between partitions.
    assert_eq!(by_delta.iter().filter(|(&d, _)| d != 0).map(|(_, t)| t.len()).sum::<usize>(), 6);

    let share = &shares[1];
    let index = SupportIndex::new(share);
    let mut sent = 0;
    for (&delta, terms) in by_delta.iter().filter(|(&d, _)| d != 0) {
        let payload = encode_reachable(share, Some(&index), terms);
        let slice: PauliSum<1> = decode_rank(&payload.0, share.hash().clone(), n, 1);
        slice.assert_invariants();
        let reaches = |x: &[u64; 1], z: &[u64; 1]| {
            let key = PauliString { x: *x, z: *z };
            terms.iter().any(|h| !PauliString { x: h.x, z: h.z }.commutes_with(&key))
        };
        assert!(
            slice.iter().all(|(x, z, _)| reaches(x, z)),
            "delta {delta}: an unreachable term was sent"
        );
        let want = share.iter().filter(|(x, z, _)| reaches(x, z)).count();
        assert_eq!(slice.len(), want, "delta {delta}: a reachable term was held back");
        sent += slice.len();
    }
    // Rank 1 holds block 1 (qubits 8..16), so its halo acts on a qubit of the boundary bonds (7, 8) or (15, 16).
    assert!(sent > 0 && sent < share.len(), "sent {sent} of {} terms", share.len());

    let got = on_ranks(&shares, |s, t| liouvillian_share(&generator, s, &rows, t, &KeepAll));
    assert_terms_close(
        &PauliSum::merge_partitions(got),
        &generator.liouvillian(&operand),
        1e-12,
        "cut rows",
    );
}
