//! `propagate_partitioned` against the unpartitioned `propagate`.
//!
//! The partitioned engine's contract is that splitting the sum across
//! partitions changes nothing observable: the gathered output agrees with the
//! unpartitioned engine to floating-point tolerance.
//!
//! Every configuration here is `test_support::unpinned_partitions`, so the
//! tests run on any box (one node, no NUMA, a `taskset`ed CI container) —
//! placement itself is covered by `engine::partitioned::topology`'s own tests.

use paulistrings::test_support::{
    assert_runs_agree, assert_runs_agree_count_only, assert_same_terms, assert_terms_close,
    heavy_cut, propagate_discarding, rand_sum, rand_sum_real, random_circuit, single_cuts,
    spread_magnitudes, trotter_circuit, unpinned_partitions, KeepAll, PlainApproxTopN,
};
use paulistrings::{
    propagate, BuiltinTruncation, Circuit, Direction, PauliSum, PropagateOptions, TruncationPolicy,
};
use paulistrings::{
    propagate_partitioned, Gf2Hash, PartitionConfig, PartitionRows, PartitionedSum, Placement,
};
use paulistrings::{And, ApproxTopN, CoefficientThreshold, PerPartition, TopN, WeightCutoff};

const TOL: f64 = 1e-11;
/// The Trotter angle every `trotter_circuit` fixture here rotates by.
const THETA: f64 = 0.01;

/// `P` unpinned partitions of two threads each: the shape of a partitioned run
/// without its placement.
fn config(partitions: usize) -> PartitionConfig {
    unpinned_partitions(partitions, 2)
}

/// One policy against one circuit: the unpartitioned engine is the oracle, and
/// every partition count has to reproduce it as `assert_runs_agree` defines.
fn check<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    name: &str,
    partitions: &[usize],
) where
    T: TruncationPolicy<W> + Clone + Into<BuiltinTruncation>,
{
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let (want, discarded) = propagate_discarding(circuit, sum.clone(), policy, direction);
        for &p in partitions {
            let got = partitioned(circuit, sum, policy, direction, p);
            let what = format!("{name} P={p} {direction:?}");
            assert_runs_agree(policy, discarded, &got, &want, TOL, &what);
        }
    }
}

fn partitioned<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    partitions: usize,
) -> PauliSum<W>
where
    T: TruncationPolicy<W>,
{
    propagate_partitioned(
        circuit,
        sum.clone(),
        policy,
        direction,
        &config(partitions),
        PropagateOptions::default(),
    )
    .expect("topology resolves")
}

/// Every layer of [`heavy_cut`] cuts through ties or an octave, too heavily for the operator check (`assert_runs_agree_count_only` asserts so): the counts must still match, `ApproxTopN` at `P = 2, 4` and exact `TopN` at `P = 1`.
#[test]
fn heavy_cuts_keep_the_reference_count() {
    let (circuit, sum) = heavy_cut::<1>(12, 1, 800, 0x4EA1);
    for direction in [Direction::Forward, Direction::Heisenberg] {
        let policies: [(BuiltinTruncation, &[usize]); 2] =
            [(ApproxTopN(400).into(), &[2, 4]), (TopN(400).into(), &[1])];
        for (policy, partitions) in policies {
            let (want, discarded) = propagate_discarding(&circuit, sum.clone(), &policy, direction);
            for &p in partitions {
                let got = partitioned(&circuit, &sum, &policy, direction, p);
                let what = format!("heavy {policy:?} P={p} {direction:?}");
                assert_runs_agree_count_only(policy.clone(), discarded, &got, &want, &what);
            }
        }
    }
}

const PS: [usize; 3] = [1, 2, 4];

/// The 64-channel TFIM Trotter step at `W = 1`, under the two policies whose
/// layer pass is collective.
#[test]
fn trotter_matches_propagate_w1() {
    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<1>(2_000, 32, 0x71A0), 40, 0x71A0);
    check(&circuit, &sum, &ApproxTopN(6_000), "trotter approx", &PS);
    check(&circuit, &sum, &And(CoefficientThreshold(1e-9), ApproxTopN(6_000)), "trotter and", &PS);
}

/// The same recipe at `W = 2`: two-word keys, one word live.
#[test]
fn trotter_matches_propagate_w2() {
    let circuit = trotter_circuit::<2>(32, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<2>(1_500, 32, 0x71A1), 40, 0x71A1);
    check(&circuit, &sum, &ApproxTopN(4_000), "trotter w2", &PS);
}

/// `BuiltinTruncation` drives the partitioned engines through its own `finalize_layer_partitioned`, one collective per layer as the builtin `And` it names.
#[test]
fn builtin_truncation_tree_matches_propagate() {
    use paulistrings::BuiltinTruncation as T;
    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<1>(2_000, 32, 0x71A2), 40, 0x71A2);
    let tree = T::And(Box::new(T::Coefficient(1e-9)), Box::new(T::ApproxTopN(6_000)));
    check(&circuit, &sum, &tree, "trotter tree", &PS);
}

/// Two calls of fused `ApproxTopN(n)` on one split, each partition against the same split under [`PlainApproxTopN`]: the plain pass decides each partition's kept set from all-reduced counts and the partition rows alone, so the fused pass must keep the same set on every partition after every call, and `len()` must count the terms the split holds.
fn check_fused_keeps_the_plain_set<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    n: usize,
    name: &str,
) {
    for direction in [Direction::Forward, Direction::Heisenberg] {
        for p in PS {
            let scatter =
                || PartitionedSum::scatter(sum.clone(), &config(p)).expect("topology resolves");
            let (mut got, mut want) = (scatter(), scatter());
            for call in 0..2 {
                got.propagate(circuit, &ApproxTopN(n), direction);
                want.propagate(circuit, &PlainApproxTopN(n), direction);
                let what = format!("{name} P={p} {direction:?} call {call}");
                assert_eq!(want.len(), n, "{what}: the plain run must cut");
                for r in 0..p {
                    assert_same_terms(
                        got.partition(r),
                        want.partition(r),
                        &format!("{what} partition {r}"),
                    );
                }
                let held: usize = (0..p).map(|r| got.partition(r).len()).sum();
                assert_eq!(got.len(), held, "{what}: len() after the call");
            }
        }
    }
}

/// The 64-channel TFIM Trotter step, both widths, the fused pass against the plain one on the same partitions.
#[test]
fn fused_approx_top_n_keeps_the_plain_set_partitioned() {
    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<1>(2_000, 32, 0x71A0), 40, 0x71A0);
    check_fused_keeps_the_plain_set(&circuit, &sum, 6_000, "trotter w1");
    let circuit = trotter_circuit::<2>(70, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<2>(1_500, 70, 0x71A1), 40, 0x71A1);
    check_fused_keeps_the_plain_set(&circuit, &sum, 4_000, "trotter w2");
}

/// ZZ bonds with a depolarizing layer every seventh, then an X field, on `num_qubits` qubits.
fn trotter_with_rescales<const W: usize>(num_qubits: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        let bond =
            paulistrings::test_support::zz_rotation::<W>(q, (q + 1) % num_qubits, 2.0 * THETA);
        circuit.push(bond);
        if q % 7 == 6 {
            circuit.push(paulistrings::PtmChannel::depolarizing(q, 0.02));
        }
    }
    for q in 0..num_qubits {
        let generator = paulistrings::PauliString::<W>::x(q);
        circuit.push(paulistrings::PauliRotation::new(generator, 2.0 * THETA));
    }
    circuit
}

/// `ApproxTopN` with rescale layers between rotations: the deferred cut is applied by the next coset loop, a rescale, an exchange's export, a rebucket, or the end of the call.
#[test]
fn approx_top_n_through_rescale_layers_matches_propagate() {
    let circuit = trotter_with_rescales::<1>(32);
    let sum = spread_magnitudes(&rand_sum_real::<1>(2_000, 32, 0x71A3), 40, 0x71A3);
    check(&circuit, &sum, &ApproxTopN(6_000), "trotter rescale w1", &PS);
    check_fused_keeps_the_plain_set(&circuit, &sum, 6_000, "trotter rescale w1");
    let circuit = trotter_with_rescales::<2>(32);
    let sum = spread_magnitudes(&rand_sum_real::<2>(1_500, 32, 0x71A3), 40, 0x71A3);
    check(&circuit, &sum, &ApproxTopN(4_000), "trotter rescale w2", &PS);
    check_fused_keeps_the_plain_set(&circuit, &sum, 4_000, "trotter rescale w2");
}

/// Three kicked-Ising steps: Clifford ZZ bonds, then an X kick, on `num_qubits` qubits.
fn kicked_ising<const W: usize>(num_qubits: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for _ in 0..3 {
        for q in 0..num_qubits - 1 {
            let quarter = std::f64::consts::FRAC_PI_2;
            circuit.push(paulistrings::test_support::zz_rotation::<W>(q, q + 1, quarter));
        }
        for q in 0..num_qubits {
            let generator = paulistrings::PauliString::<W>::x(q);
            circuit.push(paulistrings::PauliRotation::new(generator, THETA));
        }
    }
    circuit
}

/// A kicked Ising chain whose Clifford bond layers cannot grow the sum, so every partition skips their layer pass on the agreed bound.
#[test]
fn approx_top_n_skipping_clifford_layers_matches_propagate() {
    let circuit = kicked_ising::<1>(32);
    let sum = spread_magnitudes(&rand_sum_real::<1>(2_000, 32, 0x71A4), 40, 0x71A4);
    check(&circuit, &sum, &ApproxTopN(6_000), "kicked ising w1", &PS);
    check_fused_keeps_the_plain_set(&circuit, &sum, 6_000, "kicked ising w1");
    let circuit = kicked_ising::<2>(32);
    let sum = spread_magnitudes(&rand_sum_real::<2>(1_500, 32, 0x71A4), 40, 0x71A4);
    check(&circuit, &sum, &ApproxTopN(4_000), "kicked ising w2", &PS);
    check_fused_keeps_the_plain_set(&circuit, &sum, 4_000, "kicked ising w2");
}

/// Amplitude damping at `gamma = 1 - 2^-40` after a Trotter step scales every non-identity coefficient on its qubit down by at least `2^-40` in `|c|²`, pushing the boundary out of the window the last pass centred, so those passes fall back to the plain one.
fn trotter_then_damping<const W: usize>(num_qubits: usize) -> Circuit<W> {
    let mut circuit = trotter_circuit::<W>(num_qubits, THETA);
    for q in 0..4 {
        circuit.push(paulistrings::PtmChannel::amplitude_damping(q, 1.0 - 2f64.powi(-40)));
        let generator = paulistrings::PauliString::<W>::x(q);
        circuit.push(paulistrings::PauliRotation::new(generator, THETA));
    }
    circuit
}

/// The fused pass falls back to the plain pass where the boundary leaves the window, and still keeps the plain set.
#[test]
fn approx_top_n_window_miss_keeps_the_plain_set() {
    let circuit = trotter_then_damping::<1>(32);
    let sum = rand_sum_real::<1>(2_000, 32, 0x71A5);
    check_fused_keeps_the_plain_set(&circuit, &sum, 3_000, "damping w1");
    let circuit = trotter_then_damping::<2>(70);
    let sum = rand_sum_real::<2>(1_500, 70, 0x71A5);
    check_fused_keeps_the_plain_set(&circuit, &sum, 2_500, "damping w2");
}

/// One cut that splits a boundary group, dropping little weight.
#[test]
fn single_cuts_match_propagate() {
    for (name, circuit, sum, n) in single_cuts::<1>(16, 0x51C1) {
        check(&circuit, &sum, &ApproxTopN(n), name, &PS);
    }
    for (name, circuit, sum, n) in single_cuts::<2>(70, 0x51C2) {
        check(&circuit, &sum, &ApproxTopN(n), name, &PS);
    }
}

/// A 30-layer random circuit over every channel class at `W = 1`.
///
/// Six qubits, so the whole Pauli group is 4⁶ = 4096 keys and an untruncated
/// run is bounded by construction — which is what makes the `KeepAll` cell
/// affordable.
#[test]
fn random_circuit_matches_propagate_w1() {
    let circuit = random_circuit::<1>(6, 30, 0x9AA1, true);
    let sum = spread_magnitudes(&rand_sum::<1>(300, 6, 0x9AA2), 40, 0x9AA2);
    check(&circuit, &sum, &KeepAll, "w1 keep", &PS);
    check(&circuit, &sum, &CoefficientThreshold(1e-9), "w1 eps", &PS);
    check(&circuit, &sum, &WeightCutoff(4), "w1 weight", &PS);
    check(&circuit, &sum, &ApproxTopN(1_000), "w1 approx", &PS);
    check(&circuit, &sum, &And(CoefficientThreshold(1e-9), ApproxTopN(1_000)), "w1 and", &PS);
}

/// A 20-layer random circuit at `W = 2` over 70 qubits: keys straddle the word
/// boundary, and both a per-term cap and a collective one are exercised.
#[test]
fn random_circuit_matches_propagate_w2() {
    let circuit = random_circuit::<2>(70, 20, 0x9BB1, true);
    let sum = spread_magnitudes(&rand_sum_real::<2>(1_500, 70, 0x9BB2), 40, 0x9BB2);
    check(&circuit, &sum, &WeightCutoff(2), "w2 weight", &PS);
    check(&circuit, &sum, &ApproxTopN(4_000), "w2 approx", &PS);
    check(&circuit, &sum, &And(CoefficientThreshold(1e-9), ApproxTopN(4_000)), "w2 and", &PS);
}

/// `PerPartition` of `TopN(n)` or `ApproxTopN(n)` holds every partition to `n` terms after every layer, remote ones included, alone and under a global `ApproxTopN`.
#[test]
fn per_partition_caps_every_partition_on_every_layer() {
    let circuit = trotter_circuit::<1>(32, 0.3);
    let sum = spread_magnitudes(&rand_sum_real::<1>(2_000, 32, 0x9A29), 40, 0x9A29);
    const N: usize = 700;
    let policies: [(BuiltinTruncation, &str); 3] = [
        (PerPartition(TopN(N)).into(), "topn"),
        (PerPartition(ApproxTopN(N)).into(), "approx"),
        (And(ApproxTopN(3 * N), PerPartition(TopN(N))).into(), "topn under approx"),
    ];
    for (policy, name) in policies {
        for &p in &PS {
            let mut split = PartitionedSum::scatter(sum.clone(), &config(p)).expect("topology");
            split.enable_trace();
            split.propagate(&circuit, &policy, Direction::Forward);
            let trace = split.take_trace().expect("tracing on");
            assert!(p == 1 || trace.layers.iter().any(|layer| layer.remote_deltas > 0));
            let largest = trace.layers.iter().flat_map(|layer| &layer.terms_out).max();
            assert_eq!(largest, Some(&N), "{name} P={p}: the cap must bind and hold");
        }
    }
}

/// `PropagateOptions` reaches the partitioned layer loop: a coarse bucket
/// policy changes the bucket count, not the answer.
#[test]
fn options_are_honoured() {
    let circuit = random_circuit::<1>(6, 12, 0x9AA1, true);
    let sum = rand_sum::<1>(300, 6, 0x9AA2);
    let options = PropagateOptions { target_bucket_len: 32, min_buckets: 16 };
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    for &p in &PS {
        let got = propagate_partitioned(
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
        let out = propagate_partitioned(
            &circuit,
            empty,
            &KeepAll,
            Direction::Forward,
            &config(p),
            PropagateOptions::default(),
        )
        .expect("topology resolves");
        assert!(out.is_empty(), "P={p}: an empty sum stays empty");

        let one = rand_sum::<1>(1, 6, 0x1);
        assert_eq!(one.len(), 1);
        let want = propagate(&circuit, one.clone(), &KeepAll, Direction::Forward);
        let got = propagate_partitioned(
            &circuit,
            one,
            &KeepAll,
            Direction::Forward,
            &config(p),
            PropagateOptions::default(),
        )
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
            PropagateOptions::default(),
        )
        .expect("topology resolves");
        assert_eq!(got.to_arrays(), sum.to_arrays(), "P={p}: a zero-layer circuit is the identity",);
    }
}

/// `Placement::Auto` on whatever box this is — one partition per NUMA node,
/// which is `P = 1` on a single-node box and `P = 2` on the reference host.
/// Either way the answer is the unpartitioned one.
#[test]
fn auto_placement_agrees() {
    let circuit = random_circuit::<1>(6, 15, 0x9AA1, true);
    let sum = spread_magnitudes(&rand_sum::<1>(300, 6, 0x9AA2), 40, 0x9AA2);
    let config = PartitionConfig {
        placement: Placement::Auto { max_partitions: Some(2) },
        bind_memory: false,
        threads_per_partition: None,
    };
    let (want, discarded) =
        propagate_discarding(&circuit, sum.clone(), &ApproxTopN(400), Direction::Forward);
    let got = propagate_partitioned(
        &circuit,
        sum,
        &ApproxTopN(400),
        Direction::Forward,
        &config,
        PropagateOptions::default(),
    )
    .expect("topology resolves");
    assert_runs_agree(ApproxTopN(400), discarded, &got, &want, TOL, "auto placement");
}

/// A policy with a layer pass and no collective form panics before the first layer.
#[test]
#[should_panic(expected = "cannot run on 2 partitions")]
fn a_finalizing_policy_without_a_collective_form_panics() {
    struct Liar;
    impl<const W: usize> TruncationPolicy<W> for Liar {
        fn finalizes_layer(&self) -> bool {
            true
        }
    }

    let circuit = random_circuit::<1>(6, 3, 0x9AA1, false);
    let sum = rand_sum::<1>(100, 6, 0x9AA2);
    let _ = propagate_partitioned(
        &circuit,
        sum,
        &Liar,
        Direction::Forward,
        &config(2),
        PropagateOptions::default(),
    );
}

/// Exact `TopN` runs at one partition, matching `propagate`.
#[test]
fn exact_top_n_runs_at_one_partition() {
    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = rand_sum_real::<1>(2_000, 32, 0x70B1);
    let want = propagate(&circuit, sum.clone(), &TopN(1_500), Direction::Forward);
    let got = propagate_partitioned(
        &circuit,
        sum,
        &TopN(1_500),
        Direction::Forward,
        &config(1),
        PropagateOptions::default(),
    )
    .expect("topology resolves");
    assert_terms_close(&got, &want, TOL, "TopN at P = 1");
}

/// Exact `TopN` above one partition panics before the first layer, as the bare policy and inside a `BuiltinTruncation` tree.
#[test]
fn exact_top_n_above_one_partition_panics_before_the_first_layer() {
    use paulistrings::BuiltinTruncation as T;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `TopN` that counts the terms the engine shows it.
    struct CountingTopN(AtomicUsize);
    impl<const W: usize> TruncationPolicy<W> for CountingTopN {
        fn keep_term(&self, _x: &[u64; W], _z: &[u64; W], _c: num_complex::Complex64) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            true
        }
        fn finalize_layer(&self, sum: &mut PauliSum<W>) {
            TopN(10).finalize_layer(sum);
        }
    }

    let circuit = trotter_circuit::<1>(32, THETA);
    let sum = rand_sum_real::<1>(500, 32, 0x70B2);
    let run = |policy: &dyn TruncationPolicy<1>| {
        let sum = sum.clone();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            propagate_partitioned(
                &circuit,
                sum,
                policy,
                Direction::Forward,
                &config(2),
                PropagateOptions::default(),
            )
        }))
        .expect_err("a partitioned exact TopN must panic");
        let message = panic.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(
            message.contains("not yet supported") && message.contains("ApproxTopN"),
            "{message}"
        );
    };
    run(&TopN(10));
    run(&T::And(Box::new(T::Coefficient(1e-9)), Box::new(T::TopN(10))));
    let counting = CountingTopN(AtomicUsize::new(0));
    run(&counting);
    assert_eq!(counting.0.load(Ordering::Relaxed), 0, "no layer may run");
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
    impl<const W: usize> TruncationPolicy<W> for PanicOnLayer {
        fn supports_partitioned(&self) -> bool {
            true
        }

        fn finalize_layer_partitioned(
            &self,
            _local: &mut PauliSum<W>,
            coll: &dyn paulistrings::Collectives,
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
        let policy = PanicOnLayer { seen: (0..2).map(|_| AtomicUsize::new(0)).collect() };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            propagate_partitioned(
                &circuit,
                sum,
                &policy,
                Direction::Forward,
                &config(2),
                PropagateOptions::default(),
            )
        }));
        let _ = tx.send(outcome.is_err());
    });

    match rx.recv_timeout(Duration::from_secs(120)) {
        Ok(panicked) => assert!(panicked, "the run should have panicked, not returned"),
        Err(_) => panic!("a dead partition left the group hanging"),
    }
}

/// An in-process split repartitioned by other rows, with its buckets rehashed, continues exactly as the unpartitioned run.
#[test]
fn repartitioning_and_rehashing_mid_run_match_propagate() {
    let n = 8;
    let circuit = trotter_circuit::<1>(n, THETA);
    let sum = rand_sum::<1>(300, n, 0x9A27);
    let half = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    let want = propagate(&circuit, half.clone(), &KeepAll, Direction::Forward);
    for partitions in [2usize, 4] {
        let bits = partitions.trailing_zeros() as u8;
        let mut split = PartitionedSum::scatter(sum.clone(), &config(partitions))
            .expect("an unpinned placement resolves");
        split.propagate(&circuit, &KeepAll, Direction::Forward);
        let new_rows = PartitionRows::from_seed(n, bits, 0xFEED);
        split.repartition(new_rows.clone());
        let hash = Gf2Hash::from_rows(n, 3, 3, vec![[0b1], [0b10]], vec![[0], [0]]);
        split.rehash_buckets(hash.clone());
        assert_eq!(split.rows(), &new_rows);
        for r in 0..partitions {
            let part = split.partition(r);
            assert_eq!(part.hash().rows(), hash.rows());
            assert!(
                part.iter().all(|(x, z, _)| new_rows.partition_of(x, z) == r as u32),
                "partition {r} holds foreign keys"
            );
        }
        assert_terms_close(&split.gather(), &half, TOL, &format!("midway partitions={partitions}"));
        split.propagate(&circuit, &KeepAll, Direction::Forward);
        assert_terms_close(&split.gather(), &want, TOL, &format!("partitions={partitions}"));
    }
}
