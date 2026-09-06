//! The engine's bucket-count measurement lever.
//!
//! `PropagateOptions::{target_bucket_len, min_buckets}` are the only way to run
//! the bucketed engine with *fewer, larger* buckets than the default policy
//! picks. `PauliSum::rebucket` is grow-only and `desired_bits` clamps the count
//! below at `min_buckets` once `len >= min_buckets * MIN_TERMS_PER_TASK`, so
//! both knobs have to move together — which is what these tests pin, alongside
//! the correctness bar: a coarser partition is the same computation to
//! floating-point tolerance (ARCHITECTURE.md §Bucket-Policy,
//! `research/notes/2026-09-01-bucket-cliff.md` §1.4).

use paulistrings::bucket::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use paulistrings::channel::PauliRotation;
use paulistrings::test_support::{assert_same_terms, assert_terms_close, rand_sum};
use paulistrings::{
    propagate, propagate_with_options, Circuit, Direction, PauliString, PauliSum, PropagateOptions,
    TruncationPolicy,
};

/// No truncation: the partition must not change the surviving terms, so nothing
/// may be dropped on either arm.
struct KeepAll;
impl<const W: usize> TruncationPolicy<W> for KeepAll {}

const QUBITS: usize = 127;

/// Four ZZ rotations on disjoint pairs — a fanout-2 channel, the cheap layer
/// whose bucket policy the note's §3 measures.
fn zz_circuit() -> Circuit<2> {
    let mut circuit = Circuit::<2>::new(QUBITS);
    for (k, (i, j)) in [(0usize, 1usize), (5, 9), (40, 71), (100, 126)]
        .into_iter()
        .enumerate()
    {
        let mut gen = PauliString::<2> {
            x: [0u64; 2],
            z: [0u64; 2],
        };
        gen.z[i / 64] |= 1u64 << (i % 64);
        gen.z[j / 64] |= 1u64 << (j % 64);
        circuit.push(PauliRotation::<2>::new(gen, 0.2 + 0.1 * k as f64));
    }
    circuit
}

fn run(sum: PauliSum<2>, options: PropagateOptions) -> PauliSum<2> {
    propagate_with_options(&zz_circuit(), sum, &KeepAll, Direction::Forward, options)
}

/// Raising both knobs coarsens the partition, and the result is the same sum.
#[test]
fn target_bucket_len_controls_the_partition() {
    let sum = rand_sum::<2>(200_000, QUBITS, 0x5EED_0001);
    let fine = run(sum.clone(), PropagateOptions::default());
    let coarse = run(
        sum,
        PropagateOptions {
            target_bucket_len: 64 * DEFAULT_TARGET_BUCKET_LEN,
            min_buckets: 16,
            ..Default::default()
        },
    );
    assert!(
        coarse.num_buckets() < fine.num_buckets(),
        "coarse arm must use strictly fewer buckets: {} vs {}",
        coarse.num_buckets(),
        fine.num_buckets()
    );
    assert_terms_close(&coarse, &fine, 1e-9, "coarse vs default partition");
}

/// The knobs are additive: the default options are `propagate` exactly.
#[test]
fn default_options_reproduce_propagate_exactly() {
    let sum = rand_sum::<2>(50_000, QUBITS, 0x5EED_0002);
    let circuit = zz_circuit();
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    let got = propagate_with_options(
        &circuit,
        sum,
        &KeepAll,
        Direction::Forward,
        PropagateOptions::default(),
    );
    assert_same_terms(&got, &want, "default options vs propagate");
}

/// Once the sum is worth splitting, `min_buckets` clamps the bucket count from
/// below whatever `target_bucket_len` asks for. At a size where the default
/// partition already sits *on* that floor, raising only `target_bucket_len` is
/// therefore inert — the floor is what is binding, and getting below it needs
/// `min_buckets` to move too. Both knobs have to move together.
#[test]
fn min_buckets_floor_binds_above_the_target() {
    // Sized so the resident count stays inside
    // `[min_buckets * MIN_TERMS_PER_TASK, min_buckets * target_bucket_len]`
    // = [8192, 131072] for the whole run: worth splitting, and the floor —
    // not the target — is what sets the count.
    let sum = rand_sum::<2>(20_000, QUBITS, 0x5EED_0003);
    let fine = run(sum.clone(), PropagateOptions::default());
    let target_only = run(
        sum,
        PropagateOptions {
            target_bucket_len: 65_536,
            ..Default::default()
        },
    );
    assert_eq!(
        fine.num_buckets(),
        DEFAULT_MIN_BUCKETS,
        "fixture must sit on the floor, or the test is vacuous"
    );
    assert_eq!(
        target_only.num_buckets(),
        fine.num_buckets(),
        "the min_buckets floor binds: raising target_bucket_len alone is inert"
    );
}
