//! The CUDA backend against the host `propagate`: same keys and term count, coefficients to tolerance (ARCHITECTURE.md §Determinism).
//! Every case returns early without a device.

use paulistrings::channel::{Channel, Clifford2Q, GeneralUnitary2Q, PauliRotation};
use paulistrings::gpu::{GpuBucketPolicy, GpuError, GpuLayerOptions, GpuPauliSum};
use paulistrings::test_support::{
    assert_terms_close, cancellation_channel, cancellation_sum, differential_channels_w1,
    differential_channels_w2, haar_su4_matrix, rand_sum, random_circuit, trotter_circuit, KeepAll,
};
use paulistrings::truncation::{And, ApproxTopN, CoefficientThreshold, WeightCutoff};
use paulistrings::{
    propagate, Circuit, Direction, PartitionedTruncation, PauliString, PauliSum, PropagateOptions,
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
    // 2 · 20 = 40 layers, past BITS_AGREE_EVERY, from a small operator so the bucket count grows mid-run.
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
    assert_terms_close(
        &dev.to_host().unwrap(),
        &input,
        0.0f64.max(TOL),
        "untouched",
    );
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

#[test]
fn fixed_terms_per_bucket_policy_agrees() {
    require_cuda!();
    let input = rand_sum::<1>(4000, 8, 0x77);
    let circuit = random_circuit::<1>(8, 10, 0x8888, true);
    let o = GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(256),
        arena_bytes: 1 << 20,
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
