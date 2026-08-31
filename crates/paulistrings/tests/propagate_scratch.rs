//! `propagate_with_scratch` is `propagate`'s implementation; these tests pin
//! that the two entry points are bitwise-interchangeable and that reusing one
//! scratch across calls does not leak state into the output. They run in both
//! feature configurations (`--features phase-timing` must not change a bit).

use paulistrings::channel::{Clifford2Q, Depolarizing, PauliRotation};
// `rand_sum_real` is byte-for-byte the generator this file used to define
// inline: same xorshift stream, same masking, real coefficients only.
use paulistrings::test_support::rand_sum_real;
use paulistrings::{
    propagate, propagate_with_scratch, Circuit, Direction, LayerScratch, PauliString, PauliSum,
    TruncationPolicy,
};

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

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
            a[0] = 0b11; // Z₀Z₁
            a
        },
    };
    c.push(PauliRotation::new(gen, 0.37));
    c.push(Clifford2Q::cnot(1, 2));
    c.push(Depolarizing {
        support: [3],
        p: 0.05,
    });
    c
}

fn assert_bitwise_eq<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>) {
    let (ax, az, ac) = a.to_arrays();
    let (bx, bz, bc) = b.to_arrays();
    assert_eq!(ax, bx);
    assert_eq!(az, bz);
    assert_eq!(ac.len(), bc.len());
    for (ca, cb) in ac.iter().zip(bc.iter()) {
        assert_eq!(ca.re.to_bits(), cb.re.to_bits());
        assert_eq!(ca.im.to_bits(), cb.im.to_bits());
    }
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
    let via_scratch = propagate_with_scratch(
        &circuit,
        rand_sum_real::<W>(3000, 8, 0xFEED),
        &AlwaysKeep,
        Direction::Heisenberg,
        &mut scratch,
    );
    assert_bitwise_eq(&reference, &via_scratch);

    // Reusing the same scratch (with whatever high-water state the first call
    // left behind) must not change a bit of a second, different propagation.
    let ref2 = propagate(
        &circuit,
        rand_sum_real::<W>(2000, 8, 0xBEEF),
        &AlwaysKeep,
        Direction::Forward,
    );
    let via2 = propagate_with_scratch(
        &circuit,
        rand_sum_real::<W>(2000, 8, 0xBEEF),
        &AlwaysKeep,
        Direction::Forward,
        &mut scratch,
    );
    assert_bitwise_eq(&ref2, &via2);
}

#[test]
fn propagate_with_scratch_matches_propagate_w1() {
    check_equivalence::<1>();
}

#[test]
fn propagate_with_scratch_matches_propagate_w2() {
    check_equivalence::<2>();
}
