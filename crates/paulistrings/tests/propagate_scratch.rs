//! `propagate_with_scratch` is `propagate`'s implementation; these tests pin
//! that the two entry points are bitwise-interchangeable and that reusing one
//! scratch across calls does not leak state into the output. They run in both
//! feature configurations (`--features phase-timing` must not change a bit).

use num_complex::Complex64;
use paulistrings::channel::{Clifford2Q, Depolarizing, PauliRotation};
use paulistrings::{
    propagate, propagate_with_scratch, BuildAccumulator, Circuit, Direction, LayerScratch,
    PauliString, PauliSum, Phase, TruncationPolicy,
};

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// xorshift64* as in `benches/pauli_ops.rs` — deterministic, no `rand` dep.
struct Xs64(u64);
impl Xs64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn random_sum<const W: usize>(n_terms: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n_terms);
    for _ in 0..n_terms {
        let mut p = PauliString::<W> {
            x: [0; W],
            z: [0; W],
        };
        for w in 0..W {
            // Mask each word down to the qubits it actually covers.
            let lo = w * 64;
            let mask = if num_qubits >= lo + 64 {
                u64::MAX
            } else if num_qubits > lo {
                (1u64 << (num_qubits - lo)) - 1
            } else {
                0
            };
            p.x[w] = rng.next_u64() & mask;
            p.z[w] = rng.next_u64() & mask;
        }
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, 0.0));
    }
    acc.finalize()
}

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
        random_sum::<W>(3000, 8, 0xFEED),
        &AlwaysKeep,
        Direction::Heisenberg,
    );

    let mut scratch = LayerScratch::<W>::new();
    let via_scratch = propagate_with_scratch(
        &circuit,
        random_sum::<W>(3000, 8, 0xFEED),
        &AlwaysKeep,
        Direction::Heisenberg,
        &mut scratch,
    );
    assert_bitwise_eq(&reference, &via_scratch);

    // Reusing the same scratch (with whatever high-water state the first call
    // left behind) must not change a bit of a second, different propagation.
    let ref2 = propagate(
        &circuit,
        random_sum::<W>(2000, 8, 0xBEEF),
        &AlwaysKeep,
        Direction::Forward,
    );
    let via2 = propagate_with_scratch(
        &circuit,
        random_sum::<W>(2000, 8, 0xBEEF),
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
