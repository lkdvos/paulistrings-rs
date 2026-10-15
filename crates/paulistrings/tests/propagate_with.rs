//! `propagate_with` against `propagate`, to floating-point tolerance: the default options are `propagate`, a reused scratch leaks no state, and the bucket-count knobs coarsen the partition.

use paulistrings::test_support::{assert_terms_close, rand_sum, rand_sum_real};
use paulistrings::test_support::{DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use paulistrings::{
    propagate, propagate_with, Circuit, Direction, LayerScratch, PauliString, PauliSum,
    PropagateOptions, TruncationPolicy,
};
use paulistrings::{PauliRotation, PtmChannel};

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// `exp(-i·0.37·Z₀Z₁/2)`, `CNOT(1, 2)` and depolarizing noise on qubit 3, over eight qubits.
fn test_circuit<const W: usize>() -> Circuit<W> {
    let mut c = Circuit::<W>::new(8);
    let gen = PauliString::<W> {
        x: {
            let mut a = [0u64; W];
            a[0] = 0;
            a
        },
        z: {
            let mut a = [0u64; W];
            a[0] = 0b11;
            a
        },
    };
    c.push(PauliRotation::new(gen, 0.37));
    c.push(PtmChannel::cnot(1, 2));
    c.push(PtmChannel::depolarizing(3, 0.05));
    c
}

fn check_equivalence<const W: usize>() {
    let circuit = test_circuit::<W>();
    let reference = propagate(
        &circuit,
        rand_sum_real::<W>(3000, 8, 0xFEED),
        &AlwaysKeep,
        Direction::Heisenberg,
    );

    let mut scratch = LayerScratch::<W>::new();
    let via_scratch = propagate_with(
        &circuit,
        rand_sum_real::<W>(3000, 8, 0xFEED),
        &AlwaysKeep,
        Direction::Heisenberg,
        &mut scratch,
        PropagateOptions::default(),
    );
    assert_terms_close(&via_scratch, &reference, 1e-12, "default options");

    // Reusing the scratch, with whatever high-water state the first call left behind, must not change a second, different propagation.
    let ref2 = propagate(
        &circuit,
        rand_sum_real::<W>(2000, 8, 0xBEEF),
        &AlwaysKeep,
        Direction::Forward,
    );
    let via2 = propagate_with(
        &circuit,
        rand_sum_real::<W>(2000, 8, 0xBEEF),
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
        PropagateOptions::default(),
    );
    assert_terms_close(&via2, &ref2, 1e-12, "reused scratch");
}

#[test]
fn propagate_with_scratch_matches_propagate_w1() {
    check_equivalence::<1>();
}

#[test]
fn propagate_with_scratch_matches_propagate_w2() {
    check_equivalence::<2>();
}

const QUBITS: usize = 127;

/// Four ZZ rotations on disjoint pairs, a cheap fanout-2 layer.
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
    propagate_with(
        &zz_circuit(),
        sum,
        &AlwaysKeep,
        Direction::Forward,
        &mut LayerScratch::new(),
        options,
    )
}

/// Raising both knobs coarsens the partition, and the result is the same sum.
#[test]
fn target_bucket_len_controls_the_partition() {
    // Rebucketing only grows, so the default arm must outgrow the input's 128 buckets (past `128 * DEFAULT_TARGET_BUCKET_LEN` terms) for the arms to differ.
    let sum = rand_sum::<2>(40_000, QUBITS, 0x5EED_0001);
    let fine = run(sum.clone(), PropagateOptions::default());
    let coarse = run(
        sum,
        PropagateOptions {
            target_bucket_len: 64 * DEFAULT_TARGET_BUCKET_LEN,
            min_buckets: 16,
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

/// Once the sum is worth splitting, `min_buckets` clamps the bucket count from below whatever `target_bucket_len` asks for.
/// At a size where the default partition already sits on that floor, raising only `target_bucket_len` is inert, so both knobs have to move together.
#[test]
fn min_buckets_floor_binds_above_the_target() {
    // Sized so the resident count stays inside `[min_buckets * MIN_TERMS_PER_TASK, min_buckets * target_bucket_len] = [8192, 131072]` for the whole run: worth splitting, and the floor sets the count.
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

#[test]
#[should_panic(expected = "min_buckets must be at least 16, got 15")]
fn min_buckets_below_16_panics() {
    run(
        rand_sum::<2>(10, QUBITS, 1),
        PropagateOptions {
            min_buckets: 15,
            ..PropagateOptions::default()
        },
    );
}

#[test]
#[should_panic(expected = "target_bucket_len must be at least 1, got 0")]
fn zero_target_bucket_len_panics() {
    run(
        rand_sum::<2>(10, QUBITS, 1),
        PropagateOptions {
            target_bucket_len: 0,
            ..PropagateOptions::default()
        },
    );
}

#[test]
fn validate_accepts_the_defaults_and_the_bounds() {
    assert_eq!(PropagateOptions::default().validate(), Ok(()));
    let bounds = PropagateOptions {
        target_bucket_len: 1,
        min_buckets: 16,
    };
    assert_eq!(bounds.validate(), Ok(()));
}
