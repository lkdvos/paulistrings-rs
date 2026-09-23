//! The CUDA backend against the host `propagate`: same keys and term count, coefficients to tolerance (ARCHITECTURE.md §Determinism).
//! Every case returns early without a device.

use paulistrings::channel::{
    Channel, Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation,
};
use paulistrings::gpu::{GpuBucketPolicy, GpuError, GpuLayerOptions, GpuPauliSum};
use paulistrings::test_support::{
    assert_same_terms, assert_terms_close, cancellation_channel, cancellation_sum,
    differential_channels_w1, differential_channels_w2, haar_su4_matrix, rand_sum, random_circuit,
    trotter_circuit, zz_rotation, KeepAll, ShiftX, Xs64,
};
use paulistrings::truncation::{And, ApproxTopN, CoefficientThreshold, WeightCutoff};
use paulistrings::{
    propagate, propagate_with_options, Circuit, Direction, Gf2Hash, PartitionedTruncation,
    PauliString, PauliSum, PropagateOptions,
};

const TOL: f64 = 1e-11;

macro_rules! require_cuda {
    () => {
        if !paulistrings::gpu::cuda_available() {
            return;
        }
    };
}

fn one_layer<const W: usize>(num_qubits: usize, ch: Box<dyn Channel<W>>) -> Circuit<W> {
    let mut c = Circuit::<W>::new(num_qubits);
    c.channels.push(ch);
    c
}

/// Host oracle versus the device in both directions, with the device run under `options`.
fn check_with<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    name: &str,
    options: Option<GpuLayerOptions>,
    extra: &[String],
) where
    T: PartitionedTruncation<W> + ?Sized,
{
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let want = propagate(circuit, sum.clone(), policy, direction);
        let mut dev = GpuPauliSum::from_host_with_options(sum, 0, extra).expect("upload");
        if let Some(o) = options {
            dev.set_layer_options(o);
        }
        dev.propagate(circuit, policy, direction)
            .expect("device propagate");
        let got = dev.to_host().expect("download");
        let what = format!("{name} {direction:?}");
        assert_eq!(got.len(), want.len(), "{what}: term count");
        assert_eq!(dev.len(), want.len(), "{what}: device len");
        assert_terms_close(&got, &want, TOL, &what);
    }
}

fn check<const W: usize, T>(circuit: &Circuit<W>, sum: &PauliSum<W>, policy: &T, name: &str)
where
    T: PartitionedTruncation<W> + ?Sized,
{
    check_with(circuit, sum, policy, name, None, &[]);
}

#[test]
fn differential_channels_match_propagate_w1() {
    require_cuda!();
    let input = rand_sum::<1>(3000, 8, 0xC0FFEE);
    for (name, ch) in differential_channels_w1() {
        check(&one_layer(8, ch), &input, &KeepAll, name);
    }
}

#[test]
fn differential_channels_match_propagate_w2() {
    require_cuda!();
    let input = rand_sum::<2>(3000, 128, 0xC0FFEE);
    for (name, ch) in differential_channels_w2() {
        check(&one_layer(128, ch), &input, &KeepAll, name);
    }
}

#[test]
fn random_circuits_match_propagate_w1() {
    require_cuda!();
    let input = rand_sum::<1>(500, 6, 0x11);
    check(
        &random_circuit::<1>(6, 30, 0x9AA1, true),
        &input,
        &KeepAll,
        "dense W=1",
    );
    check(
        &random_circuit::<1>(6, 40, 0x9AA2, false),
        &input,
        &KeepAll,
        "sparse W=1",
    );
}

#[test]
fn random_circuits_match_propagate_w2() {
    require_cuda!();
    let input = rand_sum::<2>(2000, 70, 0x22);
    check(
        &random_circuit::<2>(70, 20, 0x9BB1, true),
        &input,
        &KeepAll,
        "dense W=2",
    );
    check(
        &random_circuit::<2>(70, 30, 0x9BB2, false),
        &input,
        &KeepAll,
        "sparse W=2",
    );
}

#[test]
fn weight_three_rotation_and_a_long_trotter_run() {
    require_cuda!();
    let mut gen = PauliString::<2>::x(3);
    gen.z[1] |= 1 << 2;
    gen.x[0] |= 1 << 40;
    let input = rand_sum::<2>(5000, 128, 0x33);
    check(
        &one_layer(128, Box::new(PauliRotation::new(gen, 0.7))),
        &input,
        &KeepAll,
        "weight-3 rotation",
    );
    // 2 · 20 = 40 layers from a single term, so the bucket count grows and the device refines mid-run.
    let mut acc = paulistrings::BuildAccumulator::<1>::new(20);
    acc.add_term(
        PauliString::<1>::z(0),
        paulistrings::Phase::ONE,
        num_complex::Complex64::new(1.0, 0.0),
    );
    let small = acc.finalize();
    check(
        &trotter_circuit::<1>(20, 0.1),
        &small,
        &CoefficientThreshold(1e-4),
        "trotter",
    );
}

#[test]
fn exact_cancellation_drops_the_zero_term() {
    require_cuda!();
    let input = cancellation_sum::<1>(8);
    let circuit = one_layer(8, Box::new(cancellation_channel()));
    let dropped = [Direction::Forward, Direction::Heisenberg]
        .iter()
        .any(|&d| propagate(&circuit, input.clone(), &KeepAll, d).len() < input.len());
    assert!(dropped, "the fixture must cancel exactly in some direction");
    check(&circuit, &input, &KeepAll, "cancellation");
    let input2 = cancellation_sum::<2>(128);
    check(
        &one_layer(128, Box::new(cancellation_channel())),
        &input2,
        &KeepAll,
        "cancellation W=2",
    );
}

#[test]
fn truncation_policies_match_the_host_term_for_term() {
    require_cuda!();
    let input = rand_sum::<2>(3000, 128, 0x44);
    let circuit = random_circuit::<2>(128, 12, 0x5555, true);
    check(&circuit, &input, &CoefficientThreshold(1e-3), "coeff 1e-3");
    check(&circuit, &input, &WeightCutoff(4), "weight 4");
    check(
        &circuit,
        &input,
        &CoefficientThreshold(-1.0),
        "coeff negative eps",
    );
}

#[test]
fn finalizing_and_composed_policies_are_rejected_before_the_first_layer() {
    require_cuda!();
    let input = rand_sum::<1>(100, 8, 0x55);
    let circuit = one_layer(8, Box::new(Clifford2Q::cnot(0, 1)));
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    assert!(matches!(
        dev.propagate(&circuit, &ApproxTopN(10), Direction::Forward),
        Err(GpuError::Unsupported(_))
    ));
    assert!(matches!(
        dev.propagate(
            &circuit,
            &And(CoefficientThreshold(1e-3), WeightCutoff(4)),
            Direction::Forward
        ),
        Err(GpuError::Unsupported(_))
    ));
    assert_eq!(dev.len(), input.len(), "nothing ran");
    assert_same_terms(&dev.to_host().unwrap(), &input, "untouched");
}

#[test]
fn short_fingerprints_still_agree() {
    require_cuda!();
    let input = rand_sum::<2>(2000, 128, 0x66);
    let circuit = random_circuit::<2>(128, 8, 0x7777, true);
    for fp in ["-DFP_BITS=8", "-DFP_BITS=0"] {
        check_with(&circuit, &input, &KeepAll, fp, None, &[fp.to_string()]);
    }
}

fn su4_layer<const W: usize>(num_qubits: usize, q0: u32, q1: u32) -> Circuit<W> {
    one_layer(
        num_qubits,
        Box::new(GeneralUnitary2Q::from_matrix(q0, q1, haar_su4_matrix())),
    )
}

/// `to_arrays` with the coefficients as bit patterns, for bitwise run-to-run comparison.
type ArrayBits = (Vec<[u64; 2]>, Vec<[u64; 2]>, Vec<(u64, u64)>);

fn arrays_bits(sum: &PauliSum<2>) -> ArrayBits {
    let (x, z, c) = sum.to_arrays();
    (
        x,
        z,
        c.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect(),
    )
}

/// Every block takes a fallback: with the low fingerprint word zeroed the `g_hi32` passes decide, with `FP_BITS=0` only the full-key sort does.
#[test]
fn fallback_paths_run_resolve_and_are_reproducible() {
    require_cuda!();
    let input = rand_sum::<2>(3000, 128, 0xFA11);
    let circuit = su4_layer::<2>(128, 0, 1);
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let run = |opts: &[String]| {
        let mut dev = GpuPauliSum::from_host_with_options(&input, 0, opts).expect("upload");
        dev.propagate(&circuit, &KeepAll, Direction::Forward)
            .expect("propagate");
        let c = dev.last_layer_counters();
        let got = dev.to_host().unwrap();
        assert_terms_close(&got, &want, TOL, &format!("{opts:?}"));
        (arrays_bits(&got), c)
    };
    let (hi, c) = run(&["-DFP_ZERO_LO".to_string()]);
    assert!(c.fallback_hi > 0, "{c:?}");
    assert_eq!(c.fallback_key, 0, "{c:?}");
    assert_eq!(run(&["-DFP_ZERO_LO".to_string()]).0, hi);
    let (key, c) = run(&["-DFP_BITS=0".to_string()]);
    assert!(c.fallback_hi > 0 && c.fallback_key > 0, "{c:?}");
    assert_eq!(run(&["-DFP_BITS=0".to_string()]).0, key);
    let (_, c) = run(&[]);
    assert_eq!((c.fallback_hi, c.fallback_key), (0, 0));
}

/// A smaller opt-in shared memory lowers the record cap, and the refine loop absorbs it.
#[test]
fn a_small_shared_memory_limit_still_agrees() {
    require_cuda!();
    let input = rand_sum::<2>(3000, 128, 0x5E);
    let circuit = su4_layer::<2>(128, 0, 1);
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let mut dev =
        GpuPauliSum::from_host_with_options(&input, 0, &["-DTEST_SHARED_LIMIT=40000".to_string()])
            .expect("upload");
    dev.propagate(&circuit, &KeepAll, Direction::Forward)
        .expect("propagate");
    let c = dev.last_layer_counters();
    assert!(c.n_cap <= 2048 && c.records_max <= 2048, "{c:?}");
    assert_terms_close(&dev.to_host().unwrap(), &want, TOL, "small shared limit");
}

/// The host schedule leaves buckets longer than the tag can address, so the layer refines before counting again.
#[test]
fn oversize_buckets_trigger_the_refine_and_recount_loop() {
    require_cuda!();
    let input = rand_sum::<2>(100_000, 128, 0x0FF);
    let circuit = su4_layer::<2>(128, 0, 1);
    let options = PropagateOptions {
        target_bucket_len: 1 << 20,
        min_buckets: 16,
        ..PropagateOptions::default()
    };
    let want = propagate_with_options(
        &circuit,
        input.clone(),
        &KeepAll,
        Direction::Forward,
        options,
    );
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    dev.set_layer_options(GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
        ..GpuLayerOptions::default()
    });
    dev.propagate_with_options(&circuit, &KeepAll, Direction::Forward, options)
        .expect("propagate");
    let c = dev.last_layer_counters();
    assert!(c.refine_passes > 0, "{c:?}");
    assert_terms_close(&dev.to_host().unwrap(), &want, TOL, "oversize loop");
}

/// A small arena batches the positions, and the output outgrows the spare's capacity mid-layer.
#[test]
fn multi_batch_output_growth_agrees_and_is_reproducible() {
    require_cuda!();
    let input = rand_sum::<2>(20_000, 128, 0xBA7C);
    let circuit = su4_layer::<2>(128, 5, 70);
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let run = || {
        let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
        dev.set_layer_options(GpuLayerOptions {
            arena_bytes: 1 << 20,
            ..GpuLayerOptions::default()
        });
        dev.propagate(&circuit, &KeepAll, Direction::Forward)
            .expect("propagate");
        let c = dev.last_layer_counters();
        (dev.to_host().unwrap(), c)
    };
    let (got, c) = run();
    assert!(c.batches > 1, "{c:?}");
    assert!(
        got.len() > 2 * input.len(),
        "the output outgrew the input-sized spare"
    );
    assert_terms_close(&got, &want, TOL, "multi-batch");
    assert_eq!(arrays_bits(&run().0), arrays_bits(&got));
}

#[test]
fn wide_words_w16() {
    require_cuda!();
    let input = rand_sum::<16>(300, 1024, 0x16);
    let mut gen = PauliString::<16>::x(1000);
    gen.z[3] |= 1 << 7;
    gen.x[9] |= 1 << 60;
    let mut c = Circuit::<16>::new(1024);
    c.push(Clifford2Q::cnot(3, 900));
    c.push(PauliRotation::new(gen, 0.3));
    c.push(GeneralUnitary2Q::from_matrix(5, 700, haar_su4_matrix()));
    check(&c, &input, &KeepAll, "W=16");
}

/// A layer that cannot fit returns `Err`, leaves the previous layer's output, and the same object continues correctly afterwards.
#[test]
fn a_mid_run_error_leaves_the_last_layer_and_the_sum_resumes() {
    require_cuda!();
    // Two buckets at upload, so the driver's schedule refines before the second layer, which cannot fit at three bits.
    let input = rand_sum::<2>(4000, 128, 0xE44);
    let seed = input.hash().seed();
    let input = input.with_hash(Gf2Hash::new(128, 1, seed));
    let su4 = || GeneralUnitary2Q::from_matrix(2, 3, haar_su4_matrix());
    let mut first = Circuit::<2>::new(128);
    first.push(zz_rotation::<2>(0, 1, 0.3));
    let mut whole = Circuit::<2>::new(128);
    whole.push(zz_rotation::<2>(0, 1, 0.3));
    whole.push(su4());
    let mut rest = Circuit::<2>::new(128);
    rest.push(su4());
    rest.push(Clifford2Q::cnot(7, 90));
    let mut all = Circuit::<2>::new(128);
    all.push(zz_rotation::<2>(0, 1, 0.3));
    all.push(su4());
    all.push(Clifford2Q::cnot(7, 90));

    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    dev.set_layer_options(GpuLayerOptions {
        max_bits: 3,
        ..GpuLayerOptions::default()
    });
    assert!(matches!(
        dev.propagate(&whole, &KeepAll, Direction::Forward),
        Err(GpuError::Unsupported(_))
    ));
    let after_first = propagate(&first, input.clone(), &KeepAll, Direction::Forward);
    assert_eq!(dev.len(), after_first.len());
    assert_terms_close(
        &dev.to_host().unwrap(),
        &after_first,
        TOL,
        "last completed layer",
    );
    dev.set_layer_options(GpuLayerOptions::default());
    dev.propagate(&rest, &KeepAll, Direction::Forward)
        .expect("resumes");
    let want = propagate(&all, input, &KeepAll, Direction::Forward);
    assert_terms_close(&dev.to_host().unwrap(), &want, TOL, "resumed");
}

/// Rescale, growth with refine, and a mixed circuit across three calls on one resident sum.
#[test]
fn repeated_propagate_on_one_resident_sum() {
    require_cuda!();
    let mut host = rand_sum::<1>(2000, 8, 0x8E9);
    let mut dev = GpuPauliSum::from_host(&host, 0).expect("upload");
    let mut c1 = Circuit::<1>::new(8);
    c1.push(Depolarizing {
        support: [3],
        p: 0.1,
    });
    let mut c2 = Circuit::<1>::new(8);
    c2.push(GeneralUnitary2Q::from_matrix(1, 3, haar_su4_matrix()));
    let mut c3 = Circuit::<1>::new(8);
    c3.push(Clifford2Q::cnot(2, 5));
    c3.push(Clifford1Q::h(0));
    for (i, c) in [c1, c2, c3].iter().enumerate() {
        host = propagate(c, host, &KeepAll, Direction::Heisenberg);
        dev.propagate(c, &KeepAll, Direction::Heisenberg)
            .expect("propagate");
        assert_eq!(dev.len(), host.len(), "call {i}");
        assert_terms_close(&dev.to_host().unwrap(), &host, TOL, &format!("call {i}"));
    }
}

/// A table whose entry 0 is not the identity, and a rotation whose generator hashes to bucket delta 0.
#[test]
fn no_identity_delta_and_a_zero_bucket_delta_generator() {
    require_cuda!();
    let input = rand_sum::<2>(3000, 128, 0x0DE);
    let probe = Gf2Hash::<2>::new(128, 6, input.hash().seed());
    let mut rng = Xs64::new(0x6E4);
    let gen = loop {
        let mut g = PauliString::<2> {
            x: [0; 2],
            z: [0; 2],
        };
        for _ in 0..3 {
            let q = (rng.next_u64() % 128) as usize;
            g.x[q / 64] |= 1 << (q % 64);
            let q = (rng.next_u64() % 128) as usize;
            g.z[q / 64] |= 1 << (q % 64);
        }
        if probe.bucket_of(&g.x, &g.z) == 0 && (g.x[0] | g.x[1] | g.z[0] | g.z[1]) != 0 {
            break g;
        }
    };
    let mut c = Circuit::<2>::new(128);
    c.push(ShiftX);
    c.push(PauliRotation::new(gen, 0.45));
    check(&c, &input, &KeepAll, "no identity + zero-delta generator");
}

#[test]
fn fixed_terms_per_bucket_policy_agrees() {
    require_cuda!();
    let input = rand_sum::<1>(4000, 8, 0x77);
    let circuit = random_circuit::<1>(8, 10, 0x8888, true);
    let o = GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(256),
        arena_bytes: 1 << 20,
        ..GpuLayerOptions::default()
    };
    check_with(
        &circuit,
        &input,
        &KeepAll,
        "fixed 256, 1 MiB arena",
        Some(o),
        &[],
    );
}

#[test]
fn output_is_bitwise_reproducible_run_to_run() {
    require_cuda!();
    let input = rand_sum::<2>(5000, 128, 0x88);
    let circuit = random_circuit::<2>(128, 10, 0x9999, true);
    let run = || {
        let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
        dev.propagate(&circuit, &KeepAll, Direction::Forward)
            .expect("propagate");
        let (x, z, c) = dev.to_host().unwrap().to_arrays();
        let bits: Vec<(u64, u64)> = c.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect();
        (x, z, bits)
    };
    let first = run();
    for _ in 0..2 {
        assert_eq!(run(), first);
    }
}

#[test]
fn wide_words_w4() {
    require_cuda!();
    let input4 = rand_sum::<4>(2000, 250, 0x99);
    let mut c4 = Circuit::<4>::new(250);
    c4.push(Clifford2Q::cnot(64, 129));
    check(&c4, &input4, &KeepAll, "W=4 cnot");
    c4.push(PauliRotation::new(PauliString::<4>::x(249), 0.4));
    check(&c4, &input4, &KeepAll, "W=4 cnot+rotation");
    c4.push(GeneralUnitary2Q::from_matrix(3, 200, haar_su4_matrix()));
    check(&c4, &input4, &KeepAll, "W=4 cnot+rotation+su4");
}

#[test]
fn wide_words_w8() {
    require_cuda!();
    let input8 = rand_sum::<8>(500, 512, 0xAA);
    let mut c8 = Circuit::<8>::new(512);
    c8.push(Clifford2Q::cnot(100, 300));
    check(&c8, &input8, &KeepAll, "W=8 cnot");
    c8.push(GeneralUnitary2Q::from_matrix(5, 400, haar_su4_matrix()));
    check(&c8, &input8, &KeepAll, "W=8 cnot+su4");
}

#[test]
fn propagate_gpu_front_door_and_options() {
    require_cuda!();
    let input = rand_sum::<1>(1000, 8, 0xBB);
    let circuit = random_circuit::<1>(8, 6, 0xCCCC, true);
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let got = paulistrings::gpu::propagate_gpu(&circuit, &input, &KeepAll, Direction::Forward, 0)
        .expect("propagate_gpu");
    assert_terms_close(&got, &want, TOL, "front door");
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    dev.enable_trace();
    dev.propagate_with_options(
        &circuit,
        &KeepAll,
        Direction::Forward,
        PropagateOptions::default(),
    )
    .expect("with options");
    let trace = dev.take_trace().expect("tracing on");
    assert_eq!(trace.layers.len(), circuit.channels.len());
    assert_terms_close(&dev.to_host().unwrap(), &want, TOL, "traced");
}
