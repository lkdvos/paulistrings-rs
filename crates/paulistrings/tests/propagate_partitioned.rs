//! `propagate_partitioned` against the unpartitioned `propagate`.
//!
//! The partitioned engine's contract is that splitting the sum across
//! partitions changes nothing observable: the gathered output agrees with the
//! unpartitioned engine to floating-point tolerance (ARCHITECTURE.md
//! §Determinism), and at `P = 1` it is the unpartitioned engine, bit for bit.
//!
//! Every configuration here is `test_support::unpinned_partitions`, so the
//! tests run on any box (one node, no NUMA, a `taskset`ed CI container) —
//! placement itself is covered by `engine::partitioned::topology`'s own tests.

use paulistrings::engine::partitioned::{
    propagate_partitioned, propagate_partitioned_with_options, PartitionConfig, Placement,
};
use paulistrings::test_support::{
    assert_terms_close, rand_sum, rand_sum_real, random_circuit, trotter_circuit,
    unpinned_partitions, zz_rotation, KeepAll,
};
use paulistrings::truncation::{And, ApproxTopN, CoefficientThreshold, WeightCutoff};
use paulistrings::{
    propagate, Circuit, Direction, PartitionedTruncation, PauliSum, PropagateOptions,
    TruncationPolicy,
};

const TOL: f64 = 1e-11;
/// The Trotter angle every `trotter_circuit` fixture here rotates by.
const THETA: f64 = 0.1;

/// `P` unpinned partitions of two threads each: the shape of a partitioned run
/// without its placement.
fn config(partitions: usize) -> PartitionConfig {
    unpinned_partitions(partitions, 2, 0x5EED_C0FF_EE00_1234)
}

/// One policy against one circuit: the unpartitioned engine is the oracle, and
/// every partition count has to reproduce it — same keys (which
/// `assert_terms_close` checks), close coefficients, and the same term count.
fn check<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    name: &str,
    partitions: &[usize],
) where
    T: PartitionedTruncation<W> + ?Sized,
{
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let want = propagate(circuit, sum.clone(), policy, direction);
        for &p in partitions {
            let got = propagate_partitioned(circuit, sum.clone(), policy, direction, &config(p))
                .expect("topology resolves");
            let what = format!("{name} P={p} {direction:?}");
            assert_terms_close(&got, &want, TOL, &what);
            assert_eq!(got.len(), want.len(), "{what}: term count");
        }
    }
}

const PS: [usize; 3] = [1, 2, 4];

/// The 64-channel TFIM Trotter step at `W = 1`, under the two policies whose
/// layer pass is collective.
#[test]
fn trotter_matches_propagate_w1() {
    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = rand_sum_real::<1>(2_000, 32, 0x71A0);
    check(&circuit, &sum, &ApproxTopN(3_000), "trotter approx", &PS);
    check(
        &circuit,
        &sum,
        &And(CoefficientThreshold(1e-9), ApproxTopN(3_000)),
        "trotter and",
        &PS,
    );
}

/// The same recipe at `W = 2`: two-word keys, one word live.
#[test]
fn trotter_matches_propagate_w2() {
    let circuit = trotter_circuit::<2>(32, THETA);
    let sum = rand_sum_real::<2>(1_500, 32, 0x71A1);
    check(&circuit, &sum, &ApproxTopN(2_000), "trotter w2", &PS);
}

/// `BuiltinTruncation` drives the partitioned engines through its own `finalize_layer_partitioned`, one collective per layer as the builtin `And` it names.
#[test]
fn builtin_truncation_tree_matches_propagate() {
    use paulistrings::truncation::BuiltinTruncation as T;
    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = rand_sum_real::<1>(2_000, 32, 0x71A2);
    let tree = T::And(Box::new(T::Coeff(1e-9)), Box::new(T::ApproxTopN(3_000)));
    check(&circuit, &sum, &tree, "trotter tree", &PS);
}

/// A 30-layer random circuit over every channel class at `W = 1`.
///
/// Six qubits, so the whole Pauli group is 4⁶ = 4096 keys and an untruncated
/// run is bounded by construction — which is what makes the `KeepAll` cell
/// affordable.
#[test]
fn random_circuit_matches_propagate_w1() {
    let circuit = random_circuit::<1>(6, 30, 0x9AA1, true);
    let sum = rand_sum::<1>(300, 6, 0x9AA2);
    check(&circuit, &sum, &KeepAll, "w1 keep", &PS);
    check(&circuit, &sum, &CoefficientThreshold(1e-9), "w1 eps", &PS);
    check(&circuit, &sum, &WeightCutoff(4), "w1 weight", &PS);
    check(&circuit, &sum, &ApproxTopN(500), "w1 approx", &PS);
    check(
        &circuit,
        &sum,
        &And(CoefficientThreshold(1e-9), ApproxTopN(500)),
        "w1 and",
        &PS,
    );
}

/// A 20-layer random circuit at `W = 2` over 70 qubits: keys straddle the word
/// boundary, and both a per-term cap and a collective one are exercised.
#[test]
fn random_circuit_matches_propagate_w2() {
    let circuit = random_circuit::<2>(70, 20, 0x9BB1, true);
    let sum = rand_sum_real::<2>(1_500, 70, 0x9BB2);
    check(&circuit, &sum, &WeightCutoff(2), "w2 weight", &PS);
    check(&circuit, &sum, &ApproxTopN(2_000), "w2 approx", &PS);
    check(
        &circuit,
        &sum,
        &And(CoefficientThreshold(1e-9), ApproxTopN(2_000)),
        "w2 and",
        &PS,
    );
}

/// The tripwire that `P = 1` is *today's* engine: same rebucket policy, same
/// layer, same finalization, same bits out — not merely tolerance-equal.
#[test]
fn one_partition_matches_propagate_bitwise() {
    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = rand_sum_real::<1>(2_000, 32, 0x71A0);
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let want = propagate(&circuit, sum.clone(), &ApproxTopN(2_000), direction);
        let got = propagate_partitioned(
            &circuit,
            sum.clone(),
            &ApproxTopN(2_000),
            direction,
            &config(1),
        )
        .expect("topology resolves");
        assert_eq!(
            got.to_arrays(),
            want.to_arrays(),
            "P=1 {direction:?} is not bitwise identical to propagate",
        );
    }

    // And with no truncation at all, on a short prefix so the sum stays small.
    let mut short = Circuit::<1>::new(8);
    for q in 0..6u32 {
        short.push(zz_rotation::<1>(q, (q + 1) % 8, 0.2));
    }
    let small = rand_sum::<1>(500, 8, 0x71A3);
    let want = propagate(&short, small.clone(), &KeepAll, Direction::Forward);
    let got = propagate_partitioned(&short, small, &KeepAll, Direction::Forward, &config(1))
        .expect("topology resolves");
    assert_eq!(got.to_arrays(), want.to_arrays(), "P=1 keep-all");
}

/// `PropagateOptions` reaches the partitioned layer loop: a coarse bucket
/// policy changes the bucket count, not the answer.
#[test]
fn options_are_honoured() {
    let circuit = random_circuit::<1>(6, 12, 0x9AA1, true);
    let sum = rand_sum::<1>(300, 6, 0x9AA2);
    let options = PropagateOptions {
        target_bucket_len: 32,
        min_buckets: 16,
        ..PropagateOptions::default()
    };
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    for &p in &PS {
        let got = propagate_partitioned_with_options(
            &circuit,
            sum.clone(),
            &KeepAll,
            Direction::Forward,
            &config(p),
            options,
        )
        .expect("topology resolves");
        assert_terms_close(&got, &want, TOL, &format!("coarse buckets P={p}"));
    }
}

/// An empty sum, a single term, and a zero-layer circuit.
#[test]
fn edge_cases() {
    let circuit = random_circuit::<1>(6, 8, 0x9AA1, true);
    for &p in &PS {
        let empty = PauliSum::<1>::empty(6);
        let out = propagate_partitioned(&circuit, empty, &KeepAll, Direction::Forward, &config(p))
            .expect("topology resolves");
        assert!(out.is_empty(), "P={p}: an empty sum stays empty");

        let one = rand_sum::<1>(1, 6, 0x1);
        assert_eq!(one.len(), 1);
        let want = propagate(&circuit, one.clone(), &KeepAll, Direction::Forward);
        let got = propagate_partitioned(&circuit, one, &KeepAll, Direction::Forward, &config(p))
            .expect("topology resolves");
        assert_terms_close(&got, &want, TOL, &format!("single term P={p}"));

        // A zero-layer circuit is the identity, bit for bit.
        let sum = rand_sum::<1>(300, 6, 0x9AA2);
        let empty_circuit = Circuit::<1>::new(6);
        let got = propagate_partitioned(
            &empty_circuit,
            sum.clone(),
            &KeepAll,
            Direction::Heisenberg,
            &config(p),
        )
        .expect("topology resolves");
        assert_eq!(
            got.to_arrays(),
            sum.to_arrays(),
            "P={p}: a zero-layer circuit is the identity",
        );
    }
}

/// `Placement::Auto` on whatever box this is — one partition per NUMA node,
/// which is `P = 1` on a single-node box and `P = 2` on the reference host.
/// Either way the answer is the unpartitioned one.
#[test]
fn auto_placement_agrees() {
    let circuit = random_circuit::<1>(6, 15, 0x9AA1, true);
    let sum = rand_sum::<1>(300, 6, 0x9AA2);
    let config = PartitionConfig {
        placement: Placement::Auto {
            max_partitions: Some(2),
        },
        bind_memory: false,
        partition_row_seed: None,
    };
    let want = propagate(&circuit, sum.clone(), &ApproxTopN(400), Direction::Forward);
    let got = propagate_partitioned(&circuit, sum, &ApproxTopN(400), Direction::Forward, &config)
        .expect("topology resolves");
    assert_terms_close(&got, &want, TOL, "auto placement");
}

/// A policy with a layer finalization and no collective form is a compile-time
/// error for `TopN` and a panic for a user policy that lies about it — the
/// `PartitionedTruncation` default body's assertion, reached through the
/// driver.
#[test]
#[should_panic(expected = "has a layer finalization but no partitioned one")]
fn a_finalizing_policy_without_a_collective_form_panics() {
    struct Liar;
    impl<const W: usize> TruncationPolicy<W> for Liar {
        fn finalizes_layer(&self) -> bool {
            true
        }
    }
    impl<const W: usize> PartitionedTruncation<W> for Liar {}

    let circuit = random_circuit::<1>(6, 3, 0x9AA1, false);
    let sum = rand_sum::<1>(100, 6, 0x9AA2);
    let _ = propagate_partitioned(&circuit, sum, &Liar, Direction::Forward, &config(2));
}

/// A partition that dies mid-run must not leave its partners blocked in a
/// collective: the transport reports a dead partner and the scoped join
/// propagates the first panic.
#[test]
fn partner_panic_does_not_hang() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    /// Panics on partition 1's third layer, after the exchange — so the
    /// partners are inside the *next* layer's collective when it dies.
    struct PanicOnLayer {
        seen: Vec<AtomicUsize>,
    }
    impl<const W: usize> TruncationPolicy<W> for PanicOnLayer {}
    impl<const W: usize> PartitionedTruncation<W> for PanicOnLayer {
        fn finalize_layer_partitioned(
            &self,
            _local: &mut PauliSum<W>,
            coll: &dyn paulistrings::engine::partitioned::Collectives,
        ) {
            let rank = coll.rank() as usize;
            let k = self.seen[rank].fetch_add(1, Ordering::Relaxed);
            if rank == 1 && k == 2 {
                panic!("partition 1 fell over");
            }
        }
    }

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let circuit = random_circuit::<1>(6, 8, 0x9AA1, false);
        let sum = rand_sum::<1>(300, 6, 0x9AA2);
        let policy = PanicOnLayer {
            seen: (0..2).map(|_| AtomicUsize::new(0)).collect(),
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            propagate_partitioned(&circuit, sum, &policy, Direction::Forward, &config(2))
        }));
        let _ = tx.send(outcome.is_err());
    });

    match rx.recv_timeout(Duration::from_secs(120)) {
        Ok(panicked) => assert!(panicked, "the run should have panicked, not returned"),
        Err(_) => panic!("a dead partition left the group hanging"),
    }
}
