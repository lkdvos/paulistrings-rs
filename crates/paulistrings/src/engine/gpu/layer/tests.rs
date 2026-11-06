use super::fast_paths::permute_threads;
use super::options::parse_bytes;
use super::*;
use crate::channel::Prepare;
use crate::channel::{Channel, PtmChannel};
use crate::engine::gpu::module::CoefficientLayout;
use crate::engine::gpu::GPU_MAX_BITS;
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{
    assert_terms_close, haar_su4_matrix, naive_apply_layer, rand_sum, with_rotation_partners,
    zz_rotation, KeepAll,
};
use crate::{PauliRotation, PauliString, PauliSum};
use num_complex::Complex64;

/// `SWAP·CNOT` as a matrix: its symplectic map has no fixed nonzero vector, so all 16 deltas are realized and every row emits exactly one record.
fn sixteen_delta_permutation() -> PtmChannel {
    let e = |i: usize| -> [Complex64; 4] {
        let mut r = [Complex64::new(0.0, 0.0); 4];
        r[i] = Complex64::new(1.0, 0.0);
        r
    };
    PtmChannel::unitary_2q(0, 1, [e(0), e(3), e(1), e(2)])
}

/// What [`single_bucket_layer`] observed: the bits the layer ran at, its refine passes and its device counters.
#[derive(Debug)]
struct Observed {
    bits: u8,
    refine_passes: u32,
    record: DeviceLayerRecord,
}

/// One layer on a single-bucket device sum under `opts`, against the naive oracle; returns the counters.
/// With `x0`, every term carries `X` on qubit 0, so a two-qubit table on `(0, 1)` never sees the identity pattern and every entry emits for every row.
fn single_bucket_layer(
    n: usize,
    x0: bool,
    channel: &dyn Channel<2>,
    options: GpuLayerOptions,
) -> Result<Observed, GpuError> {
    let mut input = rand_sum::<2>(n, 128, 0x4096 + n as u64);
    if x0 {
        let mut accumulator = crate::pauli_sum::accumulator::BuildAccumulator::<2>::new(128);
        for (x, z, c) in input.iter() {
            let mut x = *x;
            x[0] |= 1;
            accumulator.add_term(crate::pauli_string::PauliString::<2> { x, z: *z }, c);
        }
        input = accumulator.finalize();
    }
    single_bucket_layer_on(input, channel, options)
}

/// [`single_bucket_layer`] on a given input.
fn single_bucket_layer_on(
    input: PauliSum<2>,
    channel: &dyn Channel<2>,
    options: GpuLayerOptions,
) -> Result<Observed, GpuError> {
    let n = input.len();
    let input = input.with_hash(Gf2Hash::new(128, 0, crate::pauli_sum::storage::DEFAULT_HASH_SEED));
    assert_eq!(input.len(), n);
    let mut sum = GpuSum::from_host(&input, 0)?;
    let mut scratch = LayerScratch::new(&sum, options)?;
    let prepared = channel.prepare(sum.hash(), false).expect("prepared");
    let rows = crate::pauli_sum::hash::PartitionRows::<2>::none(128);
    let plan = PartitionPlan::new(&prepared, &rows, 0);
    let solo = InProcessTransport::group(1);
    let layer =
        DeviceLayer { prepared: &prepared, plan: &plan, keep: &KeepProgram::KEEP, target_bits: 0 };
    apply_layer_device(&mut sum, &layer, &mut scratch, &solo[0])?;
    let want = naive_apply_layer(&input, channel, &KeepAll, false);
    assert_terms_close(&sum.to_host()?, &want, 1e-11, "single bucket");
    Ok(Observed {
        bits: sum.hash().bits(),
        refine_passes: scratch.refine_passes,
        record: scratch.record,
    })
}

#[test]
fn a_full_source_bucket_and_a_full_block_fit_and_one_more_row_refines() {
    crate::require_cuda!();
    // The limits under test are the fused layer's, which a permutation table only reaches with the scatter path off.
    let options = GpuLayerOptions {
        records_per_block: 1 << 30,
        clifford: false,
        ..GpuLayerOptions::default()
    };
    let perm = sixteen_delta_permutation();
    let c = single_bucket_layer(MAX_BUCKET_LEN, false, &perm, options).unwrap();
    assert_eq!(
        (c.bits, c.refine_passes, c.record.records, c.record.max_block_records),
        (0, 0, 4096, 4096)
    );
    let c = single_bucket_layer(MAX_BUCKET_LEN + 1, false, &perm, options).unwrap();
    assert!(c.refine_passes > 0 && c.bits > 0, "{c:?}");

    // A Haar SU(4) row with a non-identity pattern emits 15 records: the one entry mapping it onto `I⊗I` is exactly zero.
    let su4 = PtmChannel::unitary_2q(0, 1, haar_su4_matrix());
    let record_cap =
        crate::engine::gpu::module::kernel_set(0, 2, CoefficientLayout::of::<Complex64>())
            .unwrap()
            .layer_cap();
    let c = single_bucket_layer(record_cap / 15, true, &su4, options).unwrap();
    assert_eq!(
        (
            c.bits,
            c.refine_passes,
            c.record.max_block_records as usize,
            c.record.record_capacity as usize
        ),
        (0, 0, 15 * (record_cap / 15), record_cap)
    );
    let c = single_bucket_layer(record_cap / 15 + 1, true, &su4, options).unwrap();
    assert!(c.refine_passes > 0 && c.bits > 0, "{c:?}");
    let capped = GpuLayerOptions { max_bits: 0, ..options };
    assert!(matches!(
        single_bucket_layer(record_cap / 15 + 1, true, &su4, capped),
        Err(GpuError::Unsupported(_))
    ));
}

/// A source bucket of exactly `MAX_BUCKET_LEN` rows, a quarter of them rotation partners of others in the same bucket, joins at the largest table; one more row refines.
#[test]
fn a_full_source_bucket_joins_and_one_more_row_refines() {
    crate::require_cuda!();
    let options = GpuLayerOptions { records_per_block: 1 << 30, ..GpuLayerOptions::default() };
    let mut wide = PauliString::<2> { x: [0; 2], z: [0; 2] };
    wide.x[0] = 0b1011;
    wide.z[1] = 1 << 7;
    for rotation in [zz_rotation::<2>(3, 100, 0.4), PauliRotation::new(wide, -0.3)] {
        let g = rotation.generator();
        for n in [MAX_BUCKET_LEN, MAX_BUCKET_LEN + 1] {
            let partners = n / 4;
            let base = rand_sum::<2>(n - partners, 128, 0x7104 + n as u64);
            let input = with_rotation_partners(&base, &g, partners, 0x7105);
            let c = single_bucket_layer_on(input, &rotation, options).unwrap();
            assert!(c.record.joined, "{c:?}");
            if n == MAX_BUCKET_LEN {
                assert_eq!((c.bits, c.refine_passes), (0, 0), "{c:?}");
            } else {
                assert!(c.refine_passes > 0 && c.bits > 0, "{c:?}");
            }
        }
    }
}

/// A chain of layers on a many-bucket device sum against the naive layer after every step: join layers, whose output keeps a gap after each bucket, feed another join, the sort path, the permutation scatter, the rescale and a refine.
/// `alone` leaves a read-back in flight, as a lone partition's sum does; `forget` drops the longest-bucket bound before every layer, so each join settles the lengths first.
fn join_output_feeds_every_path<const W: usize>(
    num_qubits: usize,
    n: usize,
    seed: u64,
    (alone, forget): (bool, bool),
) {
    let last = num_qubits - 1;
    let rx = |q: usize, theta: f64| {
        let mut g = PauliString::<W> { x: [0; W], z: [0; W] };
        g.x[q / 64] |= 1 << (q % 64);
        PauliRotation::new(g, theta)
    };
    let channels: Vec<(Box<dyn Channel<W>>, bool)> = vec![
        (Box::new(rx(1, 0.37)), true),
        (Box::new(zz_rotation::<W>(0, last, 0.21)), true),
        (Box::new(rx(last, -0.53)), true),
        (Box::new(PtmChannel::cnot(0, last)), false),
        (Box::new(rx(0, 0.71)), true),
        (Box::new(PtmChannel::unitary_2q(1, 2, haar_su4_matrix())), false),
        (Box::new(rx(2, 0.29)), true),
        (Box::new(PtmChannel::depolarizing(1, 0.1)), false),
        (Box::new(rx(3, -0.44)), true),
    ];
    let base = rand_sum::<W>(n, num_qubits, seed);
    let mut want = with_rotation_partners(&base, &rx(1, 0.0).generator(), n / 3, seed ^ 0x5A)
        .with_hash(Gf2Hash::new(num_qubits, 3, crate::pauli_sum::storage::DEFAULT_HASH_SEED));
    let mut sum = GpuSum::from_host_with_options(&want, 0, &[], alone).expect("upload");
    let options = GpuLayerOptions { records_per_block: 256, ..GpuLayerOptions::default() };
    let mut scratch = LayerScratch::new(&sum, options).expect("scratch");
    let rows = crate::pauli_sum::hash::PartitionRows::<W>::none(num_qubits);
    let solo = InProcessTransport::group(1);
    for (step, (channel, joins)) in channels.iter().enumerate() {
        if step == 6 {
            sum.refine().expect("refine");
        }
        let prepared = channel.prepare(sum.hash(), false).expect("prepared");
        let plan = PartitionPlan::new(&prepared, &rows, 0);
        let layer = DeviceLayer {
            prepared: &prepared,
            plan: &plan,
            keep: &KeepProgram::KEEP,
            target_bits: 0,
        };
        if forget {
            sum.length.forget_longest();
        }
        let n_in = sum.len() as u64;
        apply_layer_device(&mut sum, &layer, &mut scratch, &solo[0]).expect("layer");
        assert_eq!(scratch.record.joined, *joins, "W={W} step {step}");
        if *joins {
            // Positions run over every bucket once, so the two source lengths per position sum to twice the input, which its bound bounds.
            assert_eq!(scratch.record.records, 2 * n_in, "W={W} step {step}: sized by lengths");
        }
        want = naive_apply_layer(&want, channel.as_ref(), &KeepAll, false);
        assert!(sum.len() >= want.len(), "W={W} step {step}: the length is a bound");
        sum.debug_check();
        sum.settle_lengths().expect("settle");
        assert_eq!(sum.len(), want.len(), "W={W} step {step}: settled exactly");
        let got = sum.to_host().expect("download");
        assert!(got.hash().bits() >= 3, "W={W} step {step}");
        assert_terms_close(&got, &want, 1e-11, &format!("W={W} step {step}"));
    }
}

#[test]
fn join_output_feeds_every_path_w1() {
    crate::require_cuda!();
    join_output_feeds_every_path::<1>(12, 3_000, 0x7106, (false, false));
    join_output_feeds_every_path::<1>(12, 3_000, 0x7106, (true, false));
}

#[test]
fn join_output_feeds_every_path_w2() {
    crate::require_cuda!();
    join_output_feeds_every_path::<2>(100, 20_000, 0x7107, (false, false));
    join_output_feeds_every_path::<2>(100, 20_000, 0x7107, (true, true));
}

#[test]
fn join_output_feeds_every_path_w4() {
    crate::require_cuda!();
    join_output_feeds_every_path::<4>(256, 20_000, 0x7108, (true, false));
    join_output_feeds_every_path::<4>(256, 20_000, 0x7108, (true, true));
}

#[test]
fn the_scatter_block_follows_the_average_bucket_within_the_fused_width() {
    assert_eq!(permute_threads(1_000_000, 256, 2), 1024);
    assert_eq!(permute_threads(1_000_000, 4096, 2), 256);
    assert_eq!(permute_threads(10_000, 1024, 1), 64);
    assert_eq!(permute_threads(0, 1, 1), 64);
    assert_eq!(permute_threads(1_000_000, 256, 8), 256);
    assert_eq!(permute_threads(1_000_000, 256, 4), 512);
}

#[test]
fn the_exchange_cap_parses_bytes_with_binary_suffixes() {
    assert_eq!(parse_bytes("4096"), Some(4096));
    assert_eq!(parse_bytes("3K"), Some(3 << 10));
    assert_eq!(parse_bytes("2m"), Some(2 << 20));
    assert_eq!(parse_bytes("1G"), Some(1 << 30));
    for bad in ["", "0", "-1", "G", "1.5G", "lots"] {
        assert_eq!(parse_bytes(bad), None, "{bad:?}");
    }
}

/// Eight positions of ten rows under an arena of thirty: batches of three, and a new batch at every chunk start.
#[test]
fn arena_batches_start_anew_at_every_chunk_start() {
    let segment_start: Vec<u32> = (0..=8).map(|p| 10 * p).collect();
    let bytes = 30 * DeviceColumns::<1>::BYTES_PER_TERM;
    assert_eq!(
        arena_batches::<1, Complex64>(&segment_start, bytes, 1, &[]),
        (vec![(0, 3), (3, 6), (6, 8)], 30)
    );
    assert_eq!(
        arena_batches::<1, Complex64>(&segment_start, bytes, 1, &[0, 4, 6]),
        (vec![(0, 3), (3, 4), (4, 6), (6, 8)], 30)
    );
    assert_eq!(
        arena_batches::<1, Complex64>(&segment_start, bytes, 1, &[0, 1, 2, 3, 4, 5, 6, 7]),
        ((0..8).map(|p| (p, p + 1)).collect(), 10)
    );
}

#[test]
fn records_per_block_scales_with_fanout_and_is_grow_only() {
    let p = 4096;
    // 1e6 terms: fanout 1 wants 2^8 buckets (3906 records), fanout 2 one more bit, fanout 16 four more.
    assert_eq!(gpu_desired_bits(1_000_000, 1, p, 0), 8);
    assert_eq!(gpu_desired_bits(1_000_000, 2, p, 0), 9);
    assert_eq!(gpu_desired_bits(1_000_000, 16, p, 0), 12);
    assert_eq!(gpu_desired_bits(1_000_000, 1, p, 11), 11, "never below the current bits");
    assert_eq!(gpu_desired_bits(0, 16, p, 0), 0);
    assert_eq!(gpu_desired_bits(4096, 1, p, 0), 0);
    assert_eq!(gpu_desired_bits(4097, 1, p, 0), 1);
    assert_eq!(gpu_desired_bits(usize::MAX / 2, 16, p, 0), GPU_MAX_BITS);
}

#[test]
fn layer_options_validate_their_bounds() {
    assert_eq!(GpuLayerOptions::default().validate(), Ok(()));
    let bad = |options: GpuLayerOptions| options.validate().unwrap_err();
    assert_eq!(
        bad(GpuLayerOptions { records_per_block: 0, ..GpuLayerOptions::default() }),
        "records_per_block must be at least 1, got 0"
    );
    assert_eq!(
        bad(GpuLayerOptions { arena_bytes: 0, ..GpuLayerOptions::default() }),
        "arena_bytes must be at least 1, got 0"
    );
    assert_eq!(
        bad(GpuLayerOptions { max_bits: GPU_MAX_BITS + 1, ..GpuLayerOptions::default() }),
        format!("max_bits must be at most {GPU_MAX_BITS}, got {}", GPU_MAX_BITS + 1)
    );
}
