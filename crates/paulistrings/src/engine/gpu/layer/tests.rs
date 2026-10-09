use super::fast_paths::perm_threads;
use super::options::parse_bytes;
use super::*;
use crate::channel::{Channel, GeneralUnitary2Q};
use crate::engine::partitioned::transport::InProcessTransport;
use crate::pauli_sum::hash::Gf2Hash;
use crate::test_support::{
    assert_terms_close, haar_su4_matrix, naive_apply_layer, rand_sum, KeepAll,
};
use num_complex::Complex64;

/// `SWAP·CNOT` as a matrix: its symplectic map has no fixed nonzero vector, so all 16 deltas are realized and every row emits exactly one record.
fn sixteen_delta_permutation() -> GeneralUnitary2Q {
    let e = |i: usize| -> [Complex64; 4] {
        let mut r = [Complex64::new(0.0, 0.0); 4];
        r[i] = Complex64::new(1.0, 0.0);
        r
    };
    GeneralUnitary2Q::from_matrix(0, 1, [e(0), e(3), e(1), e(2)])
}

/// One layer on a single-bucket device sum under `opts`, against the naive oracle; returns the counters.
/// With `x0`, every term carries `X` on qubit 0, so a two-qubit table on `(0, 1)` never sees the identity pattern and every entry emits for every row.
fn single_bucket_layer(
    n: usize,
    x0: bool,
    channel: &dyn Channel<2>,
    options: GpuLayerOptions,
) -> Result<GpuLayerCounters, GpuError> {
    let mut input = rand_sum::<2>(n, 128, 0x4096 + n as u64);
    if x0 {
        let mut accumulator = crate::pauli_sum::accumulator::BuildAccumulator::<2>::new(128);
        for (x, z, c) in input.iter() {
            let mut x = *x;
            x[0] |= 1;
            accumulator.add_term(
                crate::pauli_string::PauliString::<2> { x, z: *z },
                crate::phase::Phase::ONE,
                c,
            );
        }
        input = accumulator.finalize();
    }
    let input = input.with_hash(Gf2Hash::new(
        128,
        0,
        crate::pauli_sum::storage::DEFAULT_HASH_SEED,
    ));
    assert_eq!(input.len(), n);
    let mut sum = GpuSum::from_host(&input, 0)?;
    let mut scratch = LayerScratch::new(&sum, options)?;
    let prepared = channel.prepare(sum.hash(), false).expect("prepared");
    let rows = crate::pauli_sum::hash::PartitionRows::<2>::none(128);
    let plan = PartitionPlan::new(&prepared, &rows, 0);
    let solo = InProcessTransport::group(1);
    apply_layer_device(
        &mut sum,
        &prepared,
        &plan,
        &KeepProgram::KEEP,
        &mut scratch,
        0,
        &solo[0],
    )?;
    let want = naive_apply_layer(&input, channel, &KeepAll, false);
    assert_terms_close(&sum.to_host()?, &want, 1e-11, "single bucket");
    Ok(scratch.counters)
}

#[test]
fn a_full_source_bucket_and_a_full_block_fit_and_one_more_row_refines() {
    crate::require_cuda!();
    // The limits under test are the fused layer's, which a permutation table only reaches with the scatter path off.
    let options = GpuLayerOptions {
        bucket_policy: GpuBucketPolicy::TermsPerBucket(1 << 20),
        clifford: false,
        ..GpuLayerOptions::default()
    };
    let perm = sixteen_delta_permutation();
    let c = single_bucket_layer(MAX_BUCKET_LEN, false, &perm, options).unwrap();
    assert_eq!(
        (c.bits, c.refine_passes, c.records, c.records_max),
        (0, 0, 4096, 4096)
    );
    let c = single_bucket_layer(MAX_BUCKET_LEN + 1, false, &perm, options).unwrap();
    assert!(c.refine_passes > 0 && c.bits > 0, "{c:?}");

    // A Haar SU(4) row with a non-identity pattern emits 15 records: the one entry mapping it onto `I⊗I` is exactly zero.
    let su4 = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());
    let record_cap = crate::engine::gpu::module::kernel_set(0, 2)
        .unwrap()
        .layer_cap();
    let c = single_bucket_layer(record_cap / 15, true, &su4, options).unwrap();
    assert_eq!(
        (
            c.bits,
            c.refine_passes,
            c.records_max as usize,
            c.record_capacity as usize
        ),
        (0, 0, 15 * (record_cap / 15), record_cap)
    );
    let c = single_bucket_layer(record_cap / 15 + 1, true, &su4, options).unwrap();
    assert!(c.refine_passes > 0 && c.bits > 0, "{c:?}");
    let capped = GpuLayerOptions {
        max_bits: 0,
        ..options
    };
    assert!(matches!(
        single_bucket_layer(record_cap / 15 + 1, true, &su4, capped),
        Err(GpuError::Unsupported(_))
    ));
}

#[test]
fn the_scatter_block_follows_the_average_bucket_within_the_fused_width() {
    assert_eq!(perm_threads(1_000_000, 256, 2), 1024);
    assert_eq!(perm_threads(1_000_000, 4096, 2), 256);
    assert_eq!(perm_threads(10_000, 1024, 1), 64);
    assert_eq!(perm_threads(0, 1, 1), 64);
    assert_eq!(perm_threads(1_000_000, 256, 8), 256);
    assert_eq!(perm_threads(1_000_000, 256, 4), 512);
}

#[test]
fn the_exchange_cap_parses_bytes_with_binary_suffixes() {
    assert_eq!(parse_bytes(None), usize::MAX);
    assert_eq!(parse_bytes(Some("4096")), 4096);
    assert_eq!(parse_bytes(Some(" 3K ")), 3 << 10);
    assert_eq!(parse_bytes(Some("2m")), 2 << 20);
    assert_eq!(parse_bytes(Some("1G")), 1 << 30);
    for bad in ["", "0", "-1", "G", "1.5G", "lots"] {
        assert_eq!(parse_bytes(Some(bad)), usize::MAX, "{bad:?}");
    }
}

/// Eight positions of ten rows under an arena of thirty: batches of three, and a new batch at every chunk start.
#[test]
fn arena_batches_start_anew_at_every_chunk_start() {
    let segment_start: Vec<u32> = (0..=8).map(|p| 10 * p).collect();
    let bytes = 30 * DeviceColumns::<1>::BYTES_PER_TERM;
    assert_eq!(
        arena_batches::<1>(&segment_start, bytes, 1, &[]),
        (vec![(0, 3), (3, 6), (6, 8)], 30)
    );
    assert_eq!(
        arena_batches::<1>(&segment_start, bytes, 1, &[0, 4, 6]),
        (vec![(0, 3), (3, 4), (4, 6), (6, 8)], 30)
    );
    assert_eq!(
        arena_batches::<1>(&segment_start, bytes, 1, &[0, 1, 2, 3, 4, 5, 6, 7]),
        ((0..8).map(|p| (p, p + 1)).collect(), 10)
    );
}

#[test]
fn records_per_block_policy_scales_with_fanout_and_is_grow_only() {
    let p = GpuBucketPolicy::RecordsPerBlock(4096);
    // 1e6 terms: fanout 1 wants 2^8 buckets (3906 records), fanout 2 one more bit, fanout 16 four more.
    assert_eq!(gpu_desired_bits(1_000_000, 1, p, 0), 8);
    assert_eq!(gpu_desired_bits(1_000_000, 2, p, 0), 9);
    assert_eq!(gpu_desired_bits(1_000_000, 16, p, 0), 12);
    assert_eq!(
        gpu_desired_bits(1_000_000, 1, p, 11),
        11,
        "never below the current bits"
    );
    assert_eq!(gpu_desired_bits(0, 16, p, 0), 0);
    assert_eq!(gpu_desired_bits(4096, 1, p, 0), 0);
    assert_eq!(gpu_desired_bits(4097, 1, p, 0), 1);
    assert_eq!(gpu_desired_bits(usize::MAX / 2, 16, p, 0), B_MAX_BITS);
    let fixed = GpuBucketPolicy::TermsPerBucket(256);
    assert_eq!(gpu_desired_bits(1_000_000, 16, fixed, 0), 12);
    assert_eq!(gpu_desired_bits(1_000_000, 1, fixed, 0), 12);
    assert_eq!(gpu_desired_bits(256, 1, fixed, 0), 0);
    assert_eq!(gpu_desired_bits(257, 1, fixed, 0), 1);
}
