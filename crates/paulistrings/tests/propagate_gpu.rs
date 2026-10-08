//! The CUDA backend against the host `propagate`: same keys and term count, coefficients to tolerance (ARCHITECTURE.md §Determinism).
//! Every case returns early without a device.

use paulistrings::gpu::{GpuBucketPolicy, GpuError, GpuLayerOptions, GpuPauliSum};
use paulistrings::require_cuda;
use paulistrings::test_support::{
    and, assert_terms_close, cancellation_channel, cancellation_sum, differential_channels_w1,
    differential_channels_w2, haar_su4_matrix, or, rand_sum, random_circuit, trotter_circuit,
    zz_rotation, KeepAll, ShiftX, Xs64,
};
use paulistrings::{
    propagate, propagate_with, Circuit, Direction, Gf2Hash, LayerScratch, PartitionedTruncation,
    PauliString, PauliSum, PropagateOptions,
};
use paulistrings::{And, ApproxTopN, BuiltinTruncation, CoefficientThreshold, Or, WeightCutoff};
use paulistrings::{
    Channel, Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation,
};

const TOL: f64 = 1e-11;

fn one_layer<const W: usize>(num_qubits: usize, ch: Box<dyn Channel<W>>) -> Circuit<W> {
    let mut c = Circuit::<W>::new(num_qubits);
    c.channels.push(ch);
    c
}

/// One device run: the downloaded sum and the last layer's counters.
fn device_run<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    options: Option<GpuLayerOptions>,
    extra: &[String],
) -> (PauliSum<W>, paulistrings::gpu::GpuLayerCounters)
where
    T: PartitionedTruncation<W> + Clone + Into<BuiltinTruncation>,
{
    let mut dev = GpuPauliSum::from_host_with_options(sum, 0, extra).expect("upload");
    if let Some(o) = options {
        dev.set_layer_options(o);
    }
    dev.propagate(circuit, policy, direction)
        .expect("device propagate");
    let got = dev.gather().expect("download");
    assert_eq!(dev.len(), got.len());
    (got, dev.last_layer_counters(0))
}

/// Host oracle versus the device in both directions, the device run under `options` and `extra`; with `permuted`, whether the last layer must take the permutation path.
fn check_with<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    name: &str,
    (options, extra): (Option<GpuLayerOptions>, &[String]),
    permuted: Option<bool>,
) where
    T: PartitionedTruncation<W> + Clone + Into<BuiltinTruncation>,
{
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let want = propagate(circuit, sum.clone(), policy, direction);
        let (got, c) = device_run(circuit, sum, policy, direction, options, extra);
        let what = format!("{name} {direction:?}");
        if let Some(p) = permuted {
            assert_eq!(c.permuted, p, "{what}: {c:?}");
        }
        assert_eq!(got.len(), want.len(), "{what}: term count");
        assert_terms_close(&got, &want, TOL, &what);
    }
}

fn check<const W: usize, T>(circuit: &Circuit<W>, sum: &PauliSum<W>, policy: &T, name: &str)
where
    T: PartitionedTruncation<W> + Clone + Into<BuiltinTruncation>,
{
    check_with(circuit, sum, policy, name, (None, &[]), None);
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

/// The truncation matrix on a dense random circuit, host `propagate` and one device driven by the same `BuiltinTruncation` value.
/// Every policy here is exact on one device, exact `TopN` included, so the term counts must match as well as the terms.
fn truncation_matrix<const W: usize>(num_qubits: usize, layers: usize, seed: u64) {
    use BuiltinTruncation as T;
    let input = rand_sum::<W>(2000, num_qubits, seed);
    let circuit = random_circuit::<W>(num_qubits, layers, seed ^ 0x5555, true);
    // Half the untruncated output, so the octave cut bites.
    let n = propagate(&circuit, input.clone(), &T::Keep, Direction::Forward).len() / 2;
    assert!(n > 1000, "the fixture must grow");
    let matrix = [
        ("keep", T::Keep),
        ("coeff 1e-3", T::Coeff(1e-3)),
        ("coeff negative eps", T::Coeff(-1.0)),
        ("weight 4", T::Weight(4)),
        ("approx n", T::ApproxTopN(n)),
        ("coeff & approx", and(T::Coeff(1e-3), T::ApproxTopN(n))),
        ("approx & weight", and(T::ApproxTopN(n), T::Weight(4))),
        ("coeff | weight", or(T::Coeff(1e-3), T::Weight(4))),
        ("topn n", T::TopN(n)),
        ("coeff & topn", and(T::Coeff(1e-6), T::TopN(n))),
        ("topn | weight", or(T::TopN(n), T::Weight(0))),
    ];
    for (name, policy) in &matrix {
        check(&circuit, &input, policy, &format!("W={W} {name}"));
    }
}

#[test]
fn truncation_matrix_matches_the_host_w1() {
    require_cuda!();
    truncation_matrix::<1>(8, 30, 0x4401);
}

#[test]
fn truncation_matrix_matches_the_host_w2() {
    require_cuda!();
    truncation_matrix::<2>(70, 20, 0x4402);
}

/// The same cells through the builtin policy types, so the lowering from the real policies is what runs.
#[test]
fn builtin_policy_types_lower_and_match_the_host() {
    require_cuda!();
    let input = rand_sum::<2>(3000, 128, 0x4403);
    let circuit = random_circuit::<2>(128, 12, 0x5557, true);
    check(&circuit, &input, &CoefficientThreshold(1e-3), "coeff");
    check(&circuit, &input, &ApproxTopN(500), "approx 500");
    check(
        &circuit,
        &input,
        &And(CoefficientThreshold(1e-3), ApproxTopN(500)),
        "coeff & approx",
    );
    check(
        &circuit,
        &input,
        &And(ApproxTopN(800), WeightCutoff(4)),
        "approx & weight",
    );
    check(
        &circuit,
        &input,
        &Or(CoefficientThreshold(1e-3), WeightCutoff(4)),
        "coeff | weight",
    );
}

/// An `ApproxTopN` that cuts every layer keeps the device and host counts equal through a long run, and the count stays within `n`.
#[test]
fn approx_top_n_bounds_every_layer_like_the_host() {
    require_cuda!();
    let input = rand_sum::<1>(2000, 12, 0x4404);
    let circuit = random_circuit::<1>(12, 30, 0x5558, true);
    let policy = ApproxTopN(1500);
    check(&circuit, &input, &policy, "approx 1500, 30 layers");
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    dev.propagate(&circuit, policy, Direction::Forward)
        .expect("device propagate");
    assert!(dev.len() <= 1500 && !dev.is_empty());
}

fn assert_rejected_untouched<T>(policy: &T, what: &str)
where
    T: PartitionedTruncation<1> + Clone + Into<BuiltinTruncation>,
{
    let input = rand_sum::<1>(100, 8, 0x55);
    let circuit = one_layer(8, Box::new(Clifford2Q::cnot(0, 1)));
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    let r = dev.propagate(&circuit, policy, Direction::Forward);
    assert!(matches!(r, Err(GpuError::Unsupported(_))), "{what}: {r:?}");
    assert_eq!(dev.len(), input.len(), "{what}: nothing ran");
    assert_eq!(
        dev.gather().unwrap().to_arrays(),
        input.to_arrays(),
        "{what}: bitwise untouched"
    );
}

#[test]
fn unlowerable_policies_are_rejected_before_the_first_layer() {
    require_cuda!();
    use BuiltinTruncation as T;
    let chain = (1..9).fold(T::Coeff(0.0), |acc, k| and(acc, T::Weight(k)));
    assert_rejected_untouched(&chain, "17-node program");
}

/// Ties at the boundary (a group that straddles the cut, one that fits exactly), `n = 0` and `n >= len`, all on the device against the host's exact rule.
#[test]
fn exact_top_n_edge_cases_match_the_host() {
    require_cuda!();
    use paulistrings::test_support::tie_heavy_sum;
    use BuiltinTruncation as T;
    // Four equal-size magnitude groups (ARCHITECTURE.md §Truncation): 200 terms, 50 per group.
    let sum = tie_heavy_sum::<1>(200, 8, 0xED9E);
    let circuit = Circuit::<1>::new(8);
    check(&circuit, &sum, &T::TopN(50), "tie group fits exactly");
    check(&circuit, &sum, &T::TopN(80), "tie group straddles the cut");
    let mut dev = GpuPauliSum::from_host(&sum, 0).expect("upload");
    for n in [0usize, 1, sum.len(), sum.len() + 5] {
        let want = propagate(&circuit, sum.clone(), &T::TopN(n), Direction::Forward);
        dev.propagate(&circuit, T::TopN(n), Direction::Forward)
            .expect("device propagate");
        assert_eq!(dev.len(), want.len(), "n={n}");
        assert_terms_close(&dev.gather().unwrap(), &want, TOL, &format!("n={n}"));
        dev = GpuPauliSum::from_host(&sum, 0).expect("re-upload");
    }
}

/// A group member (`per_device > 1`) still rejects exact `TopN`, unlike a lone `GpuPauliSum`.
#[test]
fn exact_top_n_is_unsupported_above_one_partition() {
    require_cuda!();
    use paulistrings::gpu::GpuPartitionedSum;
    use paulistrings::{PartitionConfig, PartitionRuntime, Placement};
    use BuiltinTruncation as T;
    let input = rand_sum::<1>(500, 8, 0x7093);
    let circuit = one_layer(8, Box::new(Clifford2Q::cnot(0, 1)));
    let config = PartitionConfig {
        placement: Placement::Devices {
            devices: vec![0, 0],
            per_device: 1,
        },
        ..PartitionConfig::default()
    };
    let runtime = PartitionRuntime::new(&config).expect("runtime");
    let mut split =
        GpuPartitionedSum::scatter_to_devices(&input, runtime, &config).expect("scatter");
    let r = split.propagate(&circuit, T::TopN(10), Direction::Forward);
    assert!(
        matches!(r, Err(GpuError::Unsupported("exact TopN on device"))),
        "{r:?}"
    );
}

#[test]
fn short_fingerprints_still_agree() {
    require_cuda!();
    let input = rand_sum::<2>(2000, 128, 0x66);
    let circuit = random_circuit::<2>(128, 8, 0x7777, true);
    for fp in ["-DFP_BITS=8", "-DFP_BITS=0"] {
        check_with(
            &circuit,
            &input,
            &KeepAll,
            fp,
            (None, &[fp.to_string()]),
            None,
        );
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
        let (got, c) = device_run(&circuit, &input, &KeepAll, Direction::Forward, None, opts);
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
    let limit = ["-DTEST_SHARED_LIMIT=40000".to_string()];
    let (got, c) = device_run(&circuit, &input, &KeepAll, Direction::Forward, None, &limit);
    assert!(c.n_cap <= 2048 && c.records_max <= 2048, "{c:?}");
    assert_terms_close(&got, &want, TOL, "small shared limit");
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
    let want = propagate_with(
        &circuit,
        input.clone(),
        &KeepAll,
        Direction::Forward,
        &mut LayerScratch::new(),
        options,
    );
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    dev.set_layer_options(GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
        ..GpuLayerOptions::default()
    });
    dev.propagate_with_options(&circuit, KeepAll, Direction::Forward, options)
        .expect("propagate");
    let c = dev.last_layer_counters(0);
    assert!(c.refine_passes > 0, "{c:?}");
    assert_terms_close(&dev.gather().unwrap(), &want, TOL, "oversize loop");
}

/// A small arena batches the positions, and the output outgrows the spare's capacity mid-layer.
#[test]
fn multi_batch_output_growth_agrees_and_is_reproducible() {
    require_cuda!();
    let input = rand_sum::<2>(20_000, 128, 0xBA7C);
    let circuit = su4_layer::<2>(128, 5, 70);
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let small = GpuLayerOptions {
        arena_bytes: 1 << 20,
        ..GpuLayerOptions::default()
    };
    let run = || {
        device_run(
            &circuit,
            &input,
            &KeepAll,
            Direction::Forward,
            Some(small),
            &[],
        )
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
        dev.propagate(&whole, KeepAll, Direction::Forward),
        Err(GpuError::Unsupported(_))
    ));
    let after_first = propagate(&first, input.clone(), &KeepAll, Direction::Forward);
    assert_eq!(dev.len(), after_first.len());
    assert_terms_close(
        &dev.gather().unwrap(),
        &after_first,
        TOL,
        "last completed layer",
    );
    dev.set_layer_options(GpuLayerOptions::default());
    dev.propagate(&rest, KeepAll, Direction::Forward)
        .expect("resumes");
    let want = propagate(&all, input, &KeepAll, Direction::Forward);
    assert_terms_close(&dev.gather().unwrap(), &want, TOL, "resumed");
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
        dev.propagate(c, KeepAll, Direction::Heisenberg)
            .expect("propagate");
        assert_eq!(dev.len(), host.len(), "call {i}");
        assert_terms_close(&dev.gather().unwrap(), &host, TOL, &format!("call {i}"));
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
        (Some(o), &[]),
        None,
    );
}

#[test]
fn output_is_bitwise_reproducible_run_to_run() {
    require_cuda!();
    let input = rand_sum::<2>(5000, 128, 0x88);
    let circuit = random_circuit::<2>(128, 10, 0x9999, true);
    let run =
        || arrays_bits(&device_run(&circuit, &input, &KeepAll, Direction::Forward, None, &[]).0);
    let first = run();
    for _ in 0..2 {
        assert_eq!(run(), first);
    }
}

/// A CNOT, a rotation with a generator in several words and an SU(4) across words, at `nq` qubits.
fn wide_words<const W: usize>(nq: u32, seed: u64) {
    let input = rand_sum::<W>(500, nq as usize, seed);
    let mut gen = PauliString::<W>::x(nq - 1);
    gen.z[W / 2] |= 1 << 7;
    gen.x[W - 2] |= 1 << 60;
    let mut c = Circuit::<W>::new(nq as usize);
    c.push(Clifford2Q::cnot(3, nq - 100));
    c.push(PauliRotation::new(gen, 0.3));
    c.push(GeneralUnitary2Q::from_matrix(5, nq - 60, haar_su4_matrix()));
    check(&c, &input, &KeepAll, &format!("W={W}"));
}

#[test]
fn wide_words_match_propagate() {
    require_cuda!();
    wide_words::<4>(250, 0x99);
    wide_words::<8>(512, 0xAA);
    wide_words::<16>(1024, 0x16);
}

#[test]
fn propagate_gpu_front_door_and_options() {
    require_cuda!();
    let input = rand_sum::<1>(1000, 8, 0xBB);
    let circuit = random_circuit::<1>(8, 6, 0xCCCC, true);
    let want = propagate(&circuit, input.clone(), &KeepAll, Direction::Forward);
    let got = paulistrings::gpu::propagate_gpu(&circuit, &input, KeepAll, Direction::Forward, 0)
        .expect("propagate_gpu");
    assert_terms_close(&got, &want, TOL, "front door");
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    dev.enable_trace();
    dev.propagate_with_options(
        &circuit,
        KeepAll,
        Direction::Forward,
        PropagateOptions::default(),
    )
    .expect("with options");
    let trace = dev.take_trace().expect("tracing on");
    assert_eq!(trace.layers.len(), circuit.channels.len());
    assert_terms_close(&dev.gather().unwrap(), &want, TOL, "traced");
}

/// Every Clifford runs on the permutation path (K12–K14) and agrees with the host under every policy shape, on an input holding exact-zero coefficients too; the knob returns it to the fused layer with the same result, and a rotation, a dense unitary and a key-preserving channel never take it.
fn clifford_layers<const W: usize>(nq: usize, q0: u32, q1: u32, weight: u64) {
    use paulistrings::test_support::{random_clifford_circuit, with_zero_coefficients};
    let input = with_zero_coefficients(&rand_sum::<W>(3000, nq, 0xC11F), 7);
    assert!(input
        .iter()
        .any(|(_, _, c)| c == num_complex::Complex64::new(0.0, 0.0)));
    let gates: Vec<(&str, Box<dyn Channel<W>>)> = vec![
        ("h", Box::new(Clifford1Q::h(q0))),
        ("s", Box::new(Clifford1Q::s(q1))),
        ("cnot", Box::new(Clifford2Q::cnot(q0, q1))),
        ("cz", Box::new(Clifford2Q::cz(q0, q1))),
        ("swap", Box::new(Clifford2Q::swap(q0, q1))),
    ];
    let policies: Vec<(&str, BuiltinTruncation)> = vec![
        ("keep", BuiltinTruncation::Keep),
        ("coeff", BuiltinTruncation::Coeff(0.5)),
        ("weight", BuiltinTruncation::Weight(weight as u32)),
        (
            "and",
            and(
                BuiltinTruncation::Coeff(0.3),
                BuiltinTruncation::Weight(weight as u32),
            ),
        ),
    ];
    for (name, ch) in gates {
        let circuit = one_layer(nq, ch);
        for (pname, policy) in &policies {
            check_with(
                &circuit,
                &input,
                policy,
                &format!("{name} {pname}"),
                (None, &[]),
                Some(true),
            );
        }
    }
    let circuit = random_clifford_circuit::<W>(nq, 24, 0x5EED);
    let last_permutes = !matches!(
        circuit.channels.last().unwrap().prepare(input.hash(), false),
        Some(paulistrings::test_support::Prepared::Local(p)) if p.is_key_preserving()
    );
    for (pname, policy) in &policies {
        let what = format!("random clifford {pname}");
        check_with(
            &circuit,
            &input,
            policy,
            &what,
            (None, &[]),
            Some(last_permutes),
        );
    }
    let off = GpuLayerOptions {
        clifford: false,
        ..GpuLayerOptions::default()
    };
    let cnot = one_layer(
        nq,
        Box::new(Clifford2Q::cnot(q0, q1)) as Box<dyn Channel<W>>,
    );
    check_with(
        &cnot,
        &input,
        &KeepAll,
        "knob off",
        (Some(off), &[]),
        Some(false),
    );
    let (on, _) = device_run(&cnot, &input, &KeepAll, Direction::Forward, None, &[]);
    let (fused, _) = device_run(&cnot, &input, &KeepAll, Direction::Forward, Some(off), &[]);
    assert_eq!(
        on.to_arrays(),
        fused.to_arrays(),
        "one product per key: the two paths agree bit for bit"
    );
    for hook in ["-DFP_BITS=8", "-DFP_ZERO_LO"] {
        let extra = [hook.to_string()];
        check_with(
            &circuit,
            &input,
            &KeepAll,
            hook,
            (None, &extra),
            Some(last_permutes),
        );
    }
    let rot = one_layer(
        nq,
        Box::new(zz_rotation::<W>(q0, q1, 0.4)) as Box<dyn Channel<W>>,
    );
    check_with(&rot, &input, &KeepAll, "rotation", (None, &[]), Some(false));
    check_with(
        &su4_layer::<W>(nq, q0, q1),
        &input,
        &KeepAll,
        "su4",
        (None, &[]),
        Some(false),
    );
    let dep = one_layer(
        nq,
        Box::new(Depolarizing {
            support: [q0],
            p: 0.1,
        }) as Box<dyn Channel<W>>,
    );
    let (_, c) = device_run(&dep, &input, &KeepAll, Direction::Forward, None, &[]);
    assert!(c.rescaled && !c.permuted, "{c:?}");
}

#[test]
fn clifford_layers_take_the_permutation_path_w1() {
    require_cuda!();
    clifford_layers::<1>(8, 3, 5, 6);
}

#[test]
fn clifford_layers_take_the_permutation_path_w2() {
    require_cuda!();
    clifford_layers::<2>(128, 70, 100, 96);
}

/// A Clifford on a sum whose source buckets exceed the fused layer's tag limit and whose blocks exceed its record cap: the scatter path has neither limit and never refines for them.
#[test]
fn the_permutation_path_has_no_bucket_length_cap() {
    require_cuda!();
    let input = rand_sum::<2>(20_000, 128, 0xB16).with_hash(Gf2Hash::new(
        128,
        0,
        rand_sum::<2>(1, 128, 0).hash().seed(),
    ));
    assert_eq!(input.num_buckets(), 1);
    let opts = GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
        ..GpuLayerOptions::default()
    };
    let circuit = one_layer(
        128,
        Box::new(Clifford2Q::cnot(3, 90)) as Box<dyn Channel<2>>,
    );
    let want = propagate_with(
        &circuit,
        input.clone(),
        &KeepAll,
        Direction::Forward,
        &mut LayerScratch::new(),
        PropagateOptions {
            target_bucket_len: 1 << 20,
            min_buckets: 1,
            ..PropagateOptions::default()
        },
    );
    let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
    dev.set_layer_options(opts);
    dev.propagate_with_options(
        &circuit,
        KeepAll,
        Direction::Forward,
        PropagateOptions {
            target_bucket_len: 1 << 20,
            min_buckets: 1,
            ..PropagateOptions::default()
        },
    )
    .expect("device propagate");
    let c = dev.last_layer_counters(0);
    assert!(c.permuted && c.bits == 0 && c.refine_passes == 0, "{c:?}");
    assert_terms_close(&dev.gather().unwrap(), &want, TOL, "one bucket of 20000");
}
