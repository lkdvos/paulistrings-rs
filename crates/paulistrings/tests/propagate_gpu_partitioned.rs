//! `GpuPartitionedSum` against the unpartitioned `propagate`: `P` virtual device partitions on one device agree to tolerance, and `P = 1` is `GpuPauliSum` bit for bit (ARCHITECTURE.md §Partitioning, §Determinism).
//! Every case returns early without a device.
//! `PAULISTRINGS_GPU_TEST_DEVICES` (a csv of ordinals, default `0`) naming more than one device places the `P == devices.len()` configurations one partition per device; every other `P` stays virtual on the first.

use paulistrings::engine::partitioned::{
    count_remote_deltas, PartitionConfig, PartitionRuntime, Placement,
};
use paulistrings::gpu::{
    GpuBucketPolicy, GpuError, GpuLayerOptions, GpuPartitionedSum, GpuPauliSum,
};
use paulistrings::require_cuda;
use paulistrings::test_support::{
    assert_same_terms, assert_terms_close, rand_sum, rand_sum_real, random_circuit,
    rows_reading_z63, su4_chain, trotter_circuit, x0_terms_identity_on_q63, zz_rotation, KeepAll,
};
use paulistrings::truncation::{
    And, ApproxTopN, BuiltinTruncation, CoefficientThreshold, WeightCutoff,
};
use paulistrings::{
    propagate, propagate_with_options, Circuit, Direction, PartitionRows, PartitionedTruncation,
    PauliSum, PropagateOptions,
};

const TOL: f64 = 1e-11;
const THETA: f64 = 0.1;
const ROW_SEED: u64 = 0x5EED_C0FF_EE00_1234;
const PS: [usize; 3] = [1, 2, 4];

/// The ordinals `PAULISTRINGS_GPU_TEST_DEVICES` names, `[0]` when unset.
fn test_devices() -> Vec<u32> {
    match std::env::var("PAULISTRINGS_GPU_TEST_DEVICES") {
        Ok(csv) if !csv.trim().is_empty() => csv
            .split(',')
            .map(|d| {
                d.trim().parse().unwrap_or_else(|_| {
                    panic!("PAULISTRINGS_GPU_TEST_DEVICES: '{d}' is not a device ordinal")
                })
            })
            .collect(),
        _ => vec![0],
    }
}

/// `P` partitions: one per listed device when there are exactly `P` of them, otherwise `P` virtual ones on the first.
fn config(partitions: usize) -> PartitionConfig {
    let devices = test_devices();
    let placement = if devices.len() > 1 && devices.len() == partitions {
        Placement::Devices {
            devices,
            per_device: 1,
        }
    } else {
        Placement::Devices {
            devices: vec![devices[0]],
            per_device: partitions,
        }
    };
    PartitionConfig {
        placement,
        bind_memory: false,
        partition_row_seed: Some(ROW_SEED),
    }
}

/// A split of `sum` over `p` partitions.
fn split_of<const W: usize>(sum: &PauliSum<W>, p: usize) -> GpuPartitionedSum<W> {
    let runtime = PartitionRuntime::new(&config(p)).expect("placement");
    GpuPartitionedSum::scatter_to_devices(sum, runtime, &config(p)).expect("scatter")
}

/// One propagation and gather.
fn run<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    p: usize,
) -> Result<PauliSum<W>, GpuError>
where
    T: PartitionedTruncation<W> + Clone + Into<BuiltinTruncation>,
{
    let mut split = split_of(sum, p);
    split.propagate(circuit, policy, direction)?;
    split.gather()
}

fn check<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    name: &str,
    partitions: &[usize],
) where
    T: PartitionedTruncation<W> + Clone + Into<BuiltinTruncation>,
{
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let want = propagate(circuit, sum.clone(), policy, direction);
        for &p in partitions {
            let got = run(circuit, sum, policy, direction, p).expect("device propagate");
            let what = format!("{name} P={p} {direction:?}");
            assert_eq!(got.len(), want.len(), "{what}: term count");
            assert_terms_close(&got, &want, TOL, &what);
        }
    }
}

#[test]
fn trotter_matches_propagate_w1() {
    require_cuda!();
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

#[test]
fn trotter_matches_propagate_w2() {
    require_cuda!();
    let circuit = trotter_circuit::<2>(32, THETA);
    let sum = rand_sum_real::<2>(1_500, 32, 0x71A1);
    check(&circuit, &sum, &ApproxTopN(2_000), "trotter w2", &PS);
}

/// Dense: a Haar SU(4) under random rows has about half its deltas remote at `P = 2`.
#[test]
fn random_circuit_matches_propagate_w1() {
    require_cuda!();
    let sum = rand_sum::<1>(300, 6, 0x9AA2);
    let dense = random_circuit::<1>(6, 30, 0x9AA1, true);
    check(&dense, &sum, &KeepAll, "w1 dense keep", &PS);
    check(
        &dense,
        &sum,
        &CoefficientThreshold(1e-9),
        "w1 dense eps",
        &PS,
    );
    check(&dense, &sum, &WeightCutoff(4), "w1 dense weight", &PS);
    check(&dense, &sum, &ApproxTopN(500), "w1 dense approx", &PS);
    check(
        &dense,
        &sum,
        &And(CoefficientThreshold(1e-9), ApproxTopN(500)),
        "w1 dense and",
        &PS,
    );
    let sparse = random_circuit::<1>(6, 40, 0x9AA3, false);
    check(&sparse, &sum, &KeepAll, "w1 sparse keep", &PS);
}

#[test]
fn random_circuit_matches_propagate_w2() {
    require_cuda!();
    let sum = rand_sum_real::<2>(1_500, 70, 0x9BB2);
    let dense = random_circuit::<2>(70, 20, 0x9BB1, true);
    check(&dense, &sum, &WeightCutoff(2), "w2 dense weight", &PS);
    check(&dense, &sum, &ApproxTopN(2_000), "w2 dense approx", &PS);
    check(
        &dense,
        &sum,
        &And(CoefficientThreshold(1e-9), ApproxTopN(2_000)),
        "w2 dense and",
        &PS,
    );
    let sparse = random_circuit::<2>(70, 30, 0x9BB3, false);
    check(&sparse, &sum, &ApproxTopN(3_000), "w2 sparse approx", &PS);
}

/// A rotation whose generator crosses the partition exports one row per anticommuting term; the pair is chosen by `count_remote_deltas` under the rows `config` draws.
#[test]
fn a_rotation_crossing_the_partition_agrees() {
    require_cuda!();
    let nq = 16;
    let sum = rand_sum::<1>(2_000, nq, 0xC2055);
    for p in [2usize, 4] {
        let rows = PartitionRows::<1>::from_seed(nq, p.trailing_zeros() as u8, ROW_SEED);
        let q1 = (1..nq as u32)
            .find(|&q1| {
                let mut c = Circuit::<1>::new(nq);
                c.push(zz_rotation::<1>(0, q1, 0.3));
                count_remote_deltas(&c, sum.hash(), &rows, false)[0].1 > 0
            })
            .expect("some ZZ crosses");
        let mut circuit = Circuit::<1>::new(nq);
        for k in 0..4 {
            circuit.push(zz_rotation::<1>(0, q1, 0.3 + 0.1 * k as f64));
            circuit.push(paulistrings::channel::Clifford1Q::h(
                (k as u32 + 3) % nq as u32,
            ));
        }
        check(&circuit, &sum, &KeepAll, "crossing zz", &[p]);
        check(
            &circuit,
            &sum,
            &ApproxTopN(3_000),
            "crossing zz approx",
            &[p],
        );
    }
}

/// Cut rows: one qubit block per partition, scattered through `scatter_to_devices_with_rows`.
#[test]
fn cut_rows_agree() {
    require_cuda!();
    let nq = 16;
    let sum = rand_sum_real::<1>(1_500, nq, 0xC077);
    let circuit = trotter_circuit::<1>(nq, THETA);
    for p in [2usize, 4] {
        let blocks: Vec<Vec<u32>> = (0..p)
            .map(|k| ((k * nq / p) as u32..((k + 1) * nq / p) as u32).collect())
            .collect();
        let rows = PartitionRows::<1>::cut(nq, &blocks);
        for direction in [Direction::Forward, Direction::Heisenberg] {
            let want = propagate(&circuit, sum.clone(), &ApproxTopN(2_500), direction);
            let runtime = PartitionRuntime::new(&config(p)).expect("placement");
            let mut split =
                GpuPartitionedSum::scatter_to_devices_with_rows(&sum, rows.clone(), runtime)
                    .expect("scatter");
            split.enable_trace();
            split
                .propagate(&circuit, ApproxTopN(2_500), direction)
                .expect("propagate");
            let trace = split.take_trace().expect("tracing on");
            assert!(
                trace.remote_layers() > 0,
                "P={p}: a cut still crosses somewhere"
            );
            let got = split.gather().expect("gather");
            let what = format!("cut P={p} {direction:?}");
            assert_eq!(got.len(), want.len(), "{what}: term count");
            assert_terms_close(&got, &want, TOL, &what);
        }
    }
}

#[test]
fn edge_cases() {
    require_cuda!();
    let circuit = random_circuit::<1>(6, 8, 0x9AA1, true);
    for &p in &PS {
        let empty = PauliSum::<1>::empty(6);
        let out = run(&circuit, &empty, &KeepAll, Direction::Forward, p).expect("empty");
        assert!(out.is_empty(), "P={p}: an empty sum stays empty");

        let one = rand_sum::<1>(1, 6, 0x1);
        let want = propagate(&circuit, one.clone(), &KeepAll, Direction::Forward);
        let got = run(&circuit, &one, &KeepAll, Direction::Forward, p).expect("one term");
        assert_terms_close(&got, &want, TOL, &format!("single term P={p}"));

        let sum = rand_sum::<1>(300, 6, 0x9AA2);
        let got = run(
            &Circuit::<1>::new(6),
            &sum,
            &KeepAll,
            Direction::Heisenberg,
            p,
        )
        .expect("zero layers");
        assert_eq!(
            got.to_arrays(),
            sum.to_arrays(),
            "P={p}: a zero-layer circuit is the identity"
        );
    }
}

/// `P = 1` takes the same path as `GpuPauliSum`, so the two agree bit for bit, and both agree with the host to tolerance.
#[test]
fn one_partition_is_gpu_pauli_sum_bitwise_and_both_match_the_host() {
    require_cuda!();
    let circuit = random_circuit::<2>(70, 15, 0x9BB1, true);
    let sum = rand_sum::<2>(2_000, 70, 0x9BB2);
    for direction in [Direction::Forward, Direction::Heisenberg] {
        let host = propagate(&circuit, sum.clone(), &ApproxTopN(4_000), direction);
        let mut dev = GpuPauliSum::from_host(&sum, test_devices()[0]).expect("upload");
        dev.propagate(&circuit, ApproxTopN(4_000), direction)
            .expect("device");
        let want = dev.gather().expect("download");
        assert_eq!(
            want.len(),
            host.len(),
            "GpuPauliSum {direction:?}: term count"
        );
        assert_terms_close(
            &want,
            &host,
            TOL,
            &format!("GpuPauliSum vs host {direction:?}"),
        );
        let got = run(&circuit, &sum, &ApproxTopN(4_000), direction, 1).expect("partitioned");
        assert_same_terms(&got, &want, &format!("P=1 {direction:?}"));
        assert_eq!(got.to_arrays(), want.to_arrays(), "P=1 {direction:?} bits");
    }
}

/// Three calls on one resident split, against the host chained the same way.
#[test]
fn repeated_propagate_on_one_split() {
    require_cuda!();
    let mut host = rand_sum::<1>(2_000, 8, 0x8E9);
    let mut split = split_of(&host, 2);
    let circuits = [
        random_circuit::<1>(8, 4, 0x1111, false),
        random_circuit::<1>(8, 3, 0x2222, true),
        random_circuit::<1>(8, 5, 0x3333, true),
    ];
    for (i, c) in circuits.iter().enumerate() {
        host = propagate(c, host, &ApproxTopN(3_000), Direction::Heisenberg);
        split
            .propagate(c, ApproxTopN(3_000), Direction::Heisenberg)
            .expect("propagate");
        assert_eq!(split.len(), host.len(), "call {i}");
        assert_terms_close(
            &split.gather().expect("gather"),
            &host,
            TOL,
            &format!("call {i}"),
        );
    }
}

#[test]
fn options_are_honoured() {
    require_cuda!();
    let circuit = random_circuit::<1>(6, 12, 0x9AA1, true);
    let sum = rand_sum::<1>(300, 6, 0x9AA2);
    let options = PropagateOptions {
        target_bucket_len: 32,
        min_buckets: 16,
        ..PropagateOptions::default()
    };
    let want = propagate_with_options(&circuit, sum.clone(), &KeepAll, Direction::Forward, options);
    for &p in &PS {
        let mut split = split_of(&sum, p);
        split
            .propagate_with_options(&circuit, KeepAll, Direction::Forward, options)
            .expect("propagate");
        assert_terms_close(
            &split.gather().expect("gather"),
            &want,
            TOL,
            &format!("coarse buckets P={p}"),
        );
    }
}

/// The trace's exchange columns are consistent: a zero diagonal, local layers ship nothing, and every row sent is a row received.
/// On a remote layer every partition runs at the agreed bucket count, whatever its own bucket policy wants.
#[test]
fn trace_is_consistent_and_remote_layers_run_at_the_agreed_bits() {
    require_cuda!();
    let nq = 12;
    let sum = rand_sum::<1>(3_000, nq, 0x7ACE);
    let circuit = random_circuit::<1>(nq, 16, 0x7ACF, true);
    for p in [2usize, 4] {
        let mut split = split_of(&sum, p);
        split.set_layer_options(GpuLayerOptions {
            bucket_policy: GpuBucketPolicy::TermsPerBucket(8),
            ..GpuLayerOptions::default()
        });
        split.enable_trace();
        split
            .propagate(&circuit, ApproxTopN(6_000), Direction::Forward)
            .expect("propagate");
        let trace = split.take_trace().expect("tracing on");
        assert_eq!(trace.layers.len(), circuit.channels.len());
        assert!(trace.remote_layers() > 0 && trace.local_layers() > 0);
        for (k, layer) in trace.layers.iter().enumerate() {
            assert_eq!(layer.rows_sent.len(), p, "layer {k}");
            for (r, row) in layer.rows_sent.iter().enumerate() {
                assert_eq!(row.len(), p);
                assert_eq!(row[r], 0, "layer {k}: partition {r} ships to itself");
            }
            let sent: u64 = layer.rows_sent.iter().flat_map(|r| r.iter()).sum();
            let received: u64 = layer.rows_received.iter().sum();
            assert_eq!(sent, received, "layer {k}: rows sent equal rows received");
            for q in 0..p {
                let to_q: u64 = layer.rows_sent.iter().map(|r| r[q]).sum();
                assert_eq!(to_q, layer.rows_received[q], "layer {k}: partition {q}");
            }
            if layer.remote_deltas == 0 {
                assert_eq!(sent, 0, "layer {k}: a local layer ships nothing");
            }
        }
        let last = trace.layers.last().unwrap();
        for r in 0..p {
            assert_eq!(
                split.last_layer_counters(r).bits,
                last.bits,
                "P={p} rank {r}"
            );
        }
        let want = propagate(
            &circuit,
            sum.clone(),
            &ApproxTopN(6_000),
            Direction::Forward,
        );
        assert_terms_close(
            &split.gather().expect("gather"),
            &want,
            TOL,
            &format!("traced P={p}"),
        );
    }
}

/// A dense layer at `P = 2` ships several remote deltas to its one partner, the output agrees with the host, and two runs are bitwise equal.
#[test]
fn several_deltas_to_one_partner_agree_and_are_reproducible() {
    require_cuda!();
    let nq = 10;
    let sum = rand_sum::<1>(2_000, nq, 0x5E7);
    let circuit = random_circuit::<1>(nq, 6, 0x5E8, true);
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    let mut split = split_of(&sum, 2);
    split.enable_trace();
    split
        .propagate(&circuit, KeepAll, Direction::Forward)
        .expect("propagate");
    let trace = split.take_trace().expect("tracing on");
    let widest = trace
        .layers
        .iter()
        .map(|l| l.remote_deltas)
        .max()
        .unwrap_or(0);
    assert!(
        widest >= 2,
        "the SU(4) layers must ship at least two remote deltas to the partner"
    );
    let first = split.gather().expect("gather");
    assert_eq!(first.len(), want.len(), "term count");
    assert_terms_close(&first, &want, TOL, "several deltas");
    let again = run(&circuit, &sum, &KeepAll, Direction::Forward, 2).expect("again");
    assert_eq!(
        first.to_arrays(),
        again.to_arrays(),
        "two runs are bitwise equal"
    );
}

/// A partition failing before its exchange returns the error, its partners finish, and the split refuses every later call.
#[test]
fn an_injected_failure_poisons_the_split_and_the_partners_finish() {
    require_cuda!();
    let circuit = random_circuit::<1>(8, 8, 0x1F01, true);
    let sum = rand_sum::<1>(400, 8, 0x1F02);
    for p in [2usize, 4] {
        let mut split = split_of(&sum, p);
        split.inject_failure(1, 2);
        let r = split.propagate(&circuit, KeepAll, Direction::Forward);
        assert!(
            matches!(
                r,
                Err(GpuError::Unsupported("injected before the exchange"))
            ),
            "P={p}: {r:?}"
        );
        let again = split.propagate(&circuit, KeepAll, Direction::Forward);
        assert!(
            matches!(again, Err(GpuError::Poisoned { rank: 1, layer: 2 })),
            "P={p}: {again:?}"
        );
        assert!(
            matches!(split.gather(), Err(GpuError::Poisoned { .. })),
            "P={p}: gather"
        );
    }
}

/// Rank 1 holds no terms, so it pairs every remote layer with empty blocks while rank 0 ships it rows, across rotations and Cliffords that cross and one that stays local.
#[test]
fn an_empty_partition_ships_empty_blocks_and_merges_what_it_receives() {
    require_cuda!();
    let input = x0_terms_identity_on_q63(500, 0xE0);
    let mut circuit = Circuit::<1>::new(64);
    circuit.push(zz_rotation::<1>(0, 63, 0.3));
    circuit.push(paulistrings::channel::Clifford1Q::h(5));
    circuit.push(paulistrings::channel::Clifford2Q::cnot(2, 63));
    circuit.push(zz_rotation::<1>(7, 63, 0.4));
    let rows = rows_reading_z63();
    assert!(
        input.iter().all(|(x, z, _)| rows.partition_of(x, z) == 0),
        "fixture: rank 1 starts empty"
    );
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let runtime = PartitionRuntime::new(&config(2)).expect("placement");
    let mut split =
        GpuPartitionedSum::scatter_to_devices_with_rows(&input, rows, runtime).expect("scatter");
    split
        .propagate(&circuit, KeepAll, Direction::Forward)
        .expect("propagate");
    let got = split.gather().expect("gather");
    assert_eq!(got.len(), want.len());
    assert_terms_close(&got, &want, TOL, "empty partition");
}

/// The tag limit is per received segment, not per block: a block of well over `MAX_BUCKET_LEN` rows split across two positions is merged in full.
#[test]
fn a_received_block_above_the_tag_limit_is_merged_when_every_segment_fits() {
    require_cuda!();
    const MAX_BUCKET_LEN: usize = 4096;
    let mut circuit = Circuit::<1>::new(64);
    circuit.push(zz_rotation::<1>(0, 63, 0.3));
    let options = PropagateOptions {
        target_bucket_len: 1 << 20,
        min_buckets: 1,
        ..PropagateOptions::default()
    };
    let input = x0_terms_identity_on_q63(MAX_BUCKET_LEN + MAX_BUCKET_LEN / 2, 0xF3);
    let seed = input.hash().seed();
    let input = input.with_hash(paulistrings::Gf2Hash::new(64, 1, seed));
    let want = propagate_with_options(
        &circuit,
        input.clone(),
        &KeepAll,
        Direction::Forward,
        options,
    );
    let runtime = PartitionRuntime::new(&config(2)).expect("placement");
    let mut split =
        GpuPartitionedSum::scatter_to_devices_with_rows(&input, rows_reading_z63(), runtime)
            .expect("scatter");
    split.set_layer_options(GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
        ..GpuLayerOptions::default()
    });
    split.enable_trace();
    split
        .propagate_with_options(&circuit, KeepAll, Direction::Forward, options)
        .expect("a block over two fitting segments is merged");
    let trace = split.take_trace().expect("tracing on");
    let sent: u64 = trace.layers[0].rows_sent[0].iter().sum();
    assert!(
        sent as usize > MAX_BUCKET_LEN,
        "fixture: {sent} rows in one block"
    );
    let got = split.gather().expect("gather");
    assert_eq!(got.len(), want.len());
    assert_terms_close(&got, &want, TOL, "block of 6144 rows over two segments");
}

/// An agreed count the device cannot make its blocks fit under is `Unsupported`, and the call returns.
#[test]
fn an_agreed_count_below_the_devices_need_is_unsupported() {
    require_cuda!();
    use paulistrings::channel::GeneralUnitary2Q;
    use paulistrings::test_support::haar_su4_matrix;
    // One bucket in, so the scatter keeps one bucket and the agreement stays there under the capped proposal.
    let sum = rand_sum::<1>(20_000, 10, 0x1F03);
    let sum = sum
        .clone()
        .with_hash(paulistrings::Gf2Hash::new(10, 0, sum.hash().seed()));
    let mut circuit = Circuit::<1>::new(10);
    circuit.push(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix()));
    let mut split = split_of(&sum, 2);
    split.set_layer_options(GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
        max_bits: 0,
        ..GpuLayerOptions::default()
    });
    let options = PropagateOptions {
        target_bucket_len: 1 << 20,
        min_buckets: 1,
        ..PropagateOptions::default()
    };
    let r = split.propagate_with_options(&circuit, KeepAll, Direction::Forward, options);
    assert!(matches!(r, Err(GpuError::Unsupported(_))), "{r:?}");
}

/// Off-schedule layers never refine, so uneven partitions keep equal bucket counts through a long dense run, and the trace and the gather both accept them.
#[test]
fn uneven_cut_partitions_keep_equal_bits_across_seventeen_layers() {
    require_cuda!();
    let nq = 16;
    let sum = rand_sum::<1>(2_000, nq, 0x1F04);
    let circuit = random_circuit::<1>(nq, 20, 0x1F05, true);
    let rows = PartitionRows::<1>::cut(nq, &[(0..3).collect(), (3..16).collect()]);
    let want = propagate(
        &circuit,
        sum.clone(),
        &ApproxTopN(4_000),
        Direction::Forward,
    );
    let runtime = PartitionRuntime::new(&config(2)).expect("placement");
    let mut split = GpuPartitionedSum::scatter_to_devices_with_rows(&sum, rows.clone(), runtime)
        .expect("scatter");
    split.enable_trace();
    split
        .propagate(&circuit, ApproxTopN(4_000), Direction::Forward)
        .expect("propagate");
    let trace = split.take_trace().expect("tracing on");
    assert_eq!(trace.layers.len(), 20);
    assert!(trace.local_layers() > 0);
    let got = split.gather().expect("gather");
    assert_eq!(got.len(), want.len());
    assert_terms_close(&got, &want, TOL, "uneven cut");
}

/// Four cut blocks with gates only inside blocks and between blocks 0 and 1: the partner pairs at partition delta 2 and 3 never exchange.
#[test]
fn cut_rows_at_p4_leave_never_exchanging_pairs_at_zero() {
    require_cuda!();
    use paulistrings::channel::{Clifford1Q, Clifford2Q, GeneralUnitary2Q};
    use paulistrings::test_support::haar_su4_matrix;
    let nq = 16;
    let sum = rand_sum::<1>(1_500, nq, 0x1F06);
    let blocks: Vec<Vec<u32>> = (0..4).map(|k| (4 * k..4 * k + 4).collect()).collect();
    let rows = PartitionRows::<1>::cut(nq, &blocks);
    let mut circuit = Circuit::<1>::new(nq);
    circuit.push(Clifford2Q::cnot(0, 4));
    circuit.push(zz_rotation::<1>(1, 5, 0.3));
    circuit.push(GeneralUnitary2Q::from_matrix(2, 6, haar_su4_matrix()));
    circuit.push(Clifford1Q::h(9));
    circuit.push(Clifford2Q::cnot(4, 7));
    circuit.push(zz_rotation::<1>(0, 3, 0.2));
    circuit.push(GeneralUnitary2Q::from_matrix(1, 2, haar_su4_matrix()));
    circuit.push(Clifford2Q::cnot(8, 10));
    circuit.push(Clifford2Q::cnot(12, 14));
    circuit.push(zz_rotation::<1>(3, 5, 0.25));
    let runtime = PartitionRuntime::new(&config(4)).expect("placement");
    let mut split =
        GpuPartitionedSum::scatter_to_devices_with_rows(&sum, rows, runtime).expect("scatter");
    split.enable_trace();
    split
        .propagate(&circuit, ApproxTopN(6_000), Direction::Forward)
        .expect("propagate");
    let trace = split.take_trace().expect("tracing on");
    // The partition deltas a layer can realize are those of its channel's key-delta masks; every other partner pair must show zero.
    let rows = PartitionRows::<1>::cut(nq, &blocks);
    let hash = sum.hash();
    let mut zero_pairs = 0usize;
    let mut sent = 0u64;
    for layer in &trace.layers {
        let ch = &circuit.channels[layer.circuit_index as usize];
        let mut deltas = std::collections::HashSet::new();
        match ch.prepare(hash, false).expect("prepared") {
            paulistrings::channel::prepared::Prepared::Local(ptm) => {
                for d in ptm.deltas() {
                    deltas.insert(rows.partition_of(&d.mask_x, &d.mask_z));
                }
            }
            paulistrings::channel::prepared::Prepared::Rotation(r) => {
                deltas.insert(rows.partition_of(&r.gen.x, &r.gen.z));
            }
        }
        for r in 0..4 {
            for q in 0..4 {
                if !deltas.contains(&((r ^ q) as u32)) {
                    assert_eq!(
                        layer.rows_sent[r][q], 0,
                        "{}: {r}->{q} never crosses",
                        layer.gate_name
                    );
                    zero_pairs += 1;
                }
                sent += layer.rows_sent[r][q];
            }
        }
    }
    assert!(
        zero_pairs > 0 && sent > 0,
        "the fixture has both idle pairs and traffic"
    );
    let want = propagate(&circuit, sum, &ApproxTopN(6_000), Direction::Forward);
    assert_terms_close(&split.gather().expect("gather"), &want, TOL, "cut P=4");
}

/// Rows every partition shipped over a traced propagation, and the gathered result.
fn traced_rows<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    p: usize,
    premerge: bool,
) -> (u64, Vec<u64>, PauliSum<W>) {
    let mut split = split_of(sum, p);
    split.set_layer_options(GpuLayerOptions {
        premerge,
        ..GpuLayerOptions::default()
    });
    split.enable_trace();
    split
        .propagate(circuit, KeepAll, Direction::Forward)
        .expect("propagate");
    let trace = split.take_trace().expect("tracing on");
    let per_layer: Vec<u64> = trace
        .layers
        .iter()
        .map(|l| l.rows_sent.iter().flatten().sum())
        .collect();
    (
        trace.total_rows_exchanged(),
        per_layer,
        split.gather().expect("gather"),
    )
}

fn premerge_case<const W: usize>(nq: usize, n: usize, seed: u64) {
    let sum = rand_sum::<W>(n, nq, seed);
    let circuit = su4_chain::<W>(nq);
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    for p in [2usize, 4] {
        let what = format!("W={W} P={p}");
        let (plain, plain_layers, off) = traced_rows(&circuit, &sum, p, false);
        let (merged, merged_layers, on) = traced_rows(&circuit, &sum, p, true);
        for (got, tag) in [(&off, "unmerged"), (&on, "merged")] {
            assert_eq!(got.len(), want.len(), "{what} {tag}: term count");
            assert_terms_close(got, &want, TOL, &format!("{what} {tag}"));
        }
        assert!(
            merged < plain,
            "{what}: {merged} rows merged against {plain}"
        );
        for (k, (m, u)) in merged_layers.iter().zip(&plain_layers).enumerate() {
            assert!(m <= u, "{what} layer {k}: {m} rows merged against {u}");
        }
    }
}

/// With the sender-side merge on, a dense layer ships strictly fewer rows, and the result agrees with `propagate` either way.
#[test]
fn premerge_ships_fewer_rows_on_dense_layers_and_agrees_w1() {
    require_cuda!();
    premerge_case::<1>(8, 3_000, 0x9E1);
}

#[test]
fn premerge_ships_fewer_rows_on_dense_layers_and_agrees_w2() {
    require_cuda!();
    premerge_case::<2>(100, 3_000, 0x9E2);
}

/// Each term paired with its swap image at the opposite coefficient: `sqrt(SWAP)`'s symmetric half sends both to the same two keys at amplitude 1/2, so those rows cancel exactly wherever they meet, merged on the sender or on the receiver.
#[test]
fn premerge_with_exactly_cancelling_rows_agrees() {
    require_cuda!();
    use paulistrings::channel::GeneralUnitary2Q;
    use paulistrings::test_support::sqrt_swap_matrix;
    let nq = 8;
    let base = rand_sum_real::<1>(400, nq, 0x9E3);
    let mut acc = paulistrings::BuildAccumulator::<1>::new(nq);
    for (x, z, c) in base.iter() {
        let (x, z) = (x[0], z[0]);
        let swap = |v: u64| (v & !0b11) | ((v & 1) << 1) | ((v >> 1) & 1);
        let v = paulistrings::PauliString::<1> { x: [x], z: [z] };
        let w = paulistrings::PauliString::<1> {
            x: [swap(x)],
            z: [swap(z)],
        };
        if v == w {
            continue;
        }
        acc.add_term(v, paulistrings::Phase::ONE, c);
        acc.add_term(w, paulistrings::Phase::ONE, -c);
    }
    let sum = acc.finalize();
    let mut circuit = Circuit::<1>::new(nq);
    circuit.push(GeneralUnitary2Q::from_matrix(0, 1, sqrt_swap_matrix()));
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    for p in [2usize, 4] {
        let (_, _, got) = traced_rows(&circuit, &sum, p, true);
        let what = format!("cancelling P={p}");
        assert_eq!(got.len(), want.len(), "{what}: term count");
        assert_terms_close(&got, &want, TOL, &what);
    }
}

/// Device bytes of `rows` received rows at width `W`, the unit of `GpuLayerOptions::exchange_bytes`.
fn recv_bytes<const W: usize>(rows: usize) -> usize {
    rows * (2 * W + 3) * 8
}

/// A split under a receive cap of `exchange_bytes` and many small buckets, so a remote layer has positions to cut.
fn capped_split<const W: usize>(
    sum: &PauliSum<W>,
    p: usize,
    exchange_bytes: usize,
) -> GpuPartitionedSum<W> {
    let mut split = split_of(sum, p);
    split.set_layer_options(GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(8),
        exchange_bytes,
        ..GpuLayerOptions::default()
    });
    split
}

/// Under no receive cap, a cap of 128 rows and one of 8, four SU(4) layers run one call each agree with `propagate`, the uncapped receive moving in one chunk and a capped one in several; under a cap the whole circuit agrees in both directions.
fn chunked_receive_case<const W: usize>(nq: usize, n: usize, seed: u64) {
    let sum = rand_sum::<W>(n, nq, seed);
    let circuit = su4_chain::<W>(nq);
    let layers: Vec<Circuit<W>> = [(0u32, 1u32), (1, 2), (0, 1), (2, 3)]
        .iter()
        .map(|&(a, b)| {
            let mut c = Circuit::<W>::new(nq);
            c.push(paulistrings::channel::GeneralUnitary2Q::from_matrix(
                a,
                b,
                paulistrings::test_support::haar_su4_matrix(),
            ));
            c
        })
        .collect();
    let mut want = sum.clone();
    for c in &layers {
        want = propagate(c, want, &KeepAll, Direction::Forward);
    }
    for p in [2usize, 4] {
        for cap in [usize::MAX, recv_bytes::<W>(128), recv_bytes::<W>(8)] {
            let what = format!("W={W} P={p} cap={cap}");
            let mut split = capped_split(&sum, p, cap);
            let mut widest = 0u32;
            for (k, c) in layers.iter().enumerate() {
                split
                    .propagate(c, KeepAll, Direction::Forward)
                    .expect("propagate");
                for r in 0..p {
                    let counters = split.last_layer_counters(r);
                    if cap == usize::MAX && counters.rows_received > 0 {
                        assert_eq!(counters.recv_chunks, 1, "{what} layer {k} rank {r}");
                    }
                    widest = widest.max(counters.recv_chunks);
                }
            }
            let got = split.gather().expect("gather");
            assert_terms_close(&got, &want, TOL, &format!("{what} by layer"));
            if cap == usize::MAX {
                continue;
            }
            assert!(
                widest >= 3,
                "{what}: the widest receive moved in {widest} chunks"
            );
            for direction in [Direction::Forward, Direction::Heisenberg] {
                let whole = propagate(&circuit, sum.clone(), &KeepAll, direction);
                let mut split = capped_split(&sum, p, cap);
                split
                    .propagate(&circuit, KeepAll, direction)
                    .expect("propagate");
                let got = split.gather().expect("gather");
                assert_eq!(got.len(), whole.len(), "{what} {direction:?}: term count");
                assert_terms_close(&got, &whole, TOL, &format!("{what} {direction:?}"));
            }
        }
    }
}

#[test]
fn a_capped_device_receive_moves_in_chunks_and_agrees_w1() {
    require_cuda!();
    chunked_receive_case::<1>(10, 1_500, 0xC4A1);
}

#[test]
fn a_capped_device_receive_moves_in_chunks_and_agrees_w2() {
    require_cuda!();
    chunked_receive_case::<2>(90, 1_500, 0xC4A2);
}

/// A partition failing mid-receive, after its second chunk moved, returns the error, its partners finish, and the split refuses every later call.
/// The cap forces at least 3 chunks and the failing rank's own layer proves it: `recv_chunks` is the planned count, set before the chunk loop runs, so it stands even though the loop broke partway through.
#[test]
fn a_mid_receive_failure_poisons_the_split_and_the_partners_finish() {
    require_cuda!();
    let sum = rand_sum::<1>(2_000, 10, 0xC4A3);
    let circuit = su4_chain::<1>(10);
    for p in [2usize, 4] {
        let mut split = capped_split(&sum, p, recv_bytes::<1>(128));
        split.inject_chunk_oom(1, 1);
        let r = split.propagate(&circuit, KeepAll, Direction::Forward);
        assert!(
            matches!(r, Err(GpuError::OutOfMemory { device: 0, .. })),
            "P={p}: {r:?}"
        );
        assert!(
            split.last_layer_counters(1).recv_chunks > 1,
            "P={p}: the failing layer must have had more than one chunk, got {:?}",
            split.last_layer_counters(1)
        );
        let again = split.propagate(&circuit, KeepAll, Direction::Forward);
        assert!(
            matches!(again, Err(GpuError::Poisoned { rank: 1, .. })),
            "P={p}: {again:?}"
        );
    }
}

/// A Clifford-only circuit across virtual partitions agrees with the host; a layer whose delta stays inside every partition takes the permutation path on every rank, one whose delta crosses keeps the fused layer.
#[test]
fn clifford_circuits_agree_across_partitions_and_local_layers_permute() {
    require_cuda!();
    use paulistrings::channel::Clifford1Q;
    use paulistrings::test_support::random_clifford_circuit;
    let nq = 12;
    let sum = rand_sum::<1>(2000, nq, 0xC11F);
    let circuit = random_clifford_circuit::<1>(nq, 30, 0xC1FF);
    check(&circuit, &sum, &KeepAll, "clifford keep", &[2, 4]);
    check(
        &circuit,
        &sum,
        &And(CoefficientThreshold(0.3), WeightCutoff(9)),
        "clifford and",
        &[2, 4],
    );
    for p in [2usize, 4] {
        let mut split = split_of(&sum, p);
        // `H` on `q` has the one delta `X_q Z_q`; local when the partition rows read it as zero.
        let delta_local = |q: u32| split.rows().partition_of(&[1u64 << q], &[1u64 << q]) == 0;
        let local = (0..nq as u32).find(|&q| delta_local(q)).expect("a local H");
        let remote = (0..nq as u32)
            .find(|&q| !delta_local(q))
            .expect("a crossing H");
        let mut c = Circuit::<1>::new(nq);
        c.push(Clifford1Q::h(local));
        split
            .propagate(&c, KeepAll, Direction::Forward)
            .expect("local H");
        for rank in 0..p {
            let counters = split.last_layer_counters(rank);
            assert!(counters.permuted, "P={p} rank {rank}: {counters:?}");
        }
        let mut c = Circuit::<1>::new(nq);
        c.push(Clifford1Q::h(remote));
        split
            .propagate(&c, KeepAll, Direction::Forward)
            .expect("crossing H");
        for rank in 0..p {
            let counters = split.last_layer_counters(rank);
            assert!(
                !counters.permuted && counters.rows_received > 0,
                "P={p} rank {rank}: {counters:?}"
            );
        }
        let mut both = Circuit::<1>::new(nq);
        both.push(Clifford1Q::h(local));
        both.push(Clifford1Q::h(remote));
        let want = propagate(&both, sum.clone(), &KeepAll, Direction::Forward);
        assert_terms_close(
            &split.gather().unwrap(),
            &want,
            TOL,
            &format!("P={p} two H"),
        );
    }
}
