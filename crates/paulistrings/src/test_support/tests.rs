use super::{raw_outputs, IdentityChannel, HEAVY_HEX_127_EDGES};
use crate::channel::Channel;
use crate::pauli_string::PauliString;
use num_complex::Complex64;

#[test]
fn identity_channel_has_empty_support() {
    assert_eq!(
        <IdentityChannel as Channel<1>>::support(&IdentityChannel),
        [0u64; 1]
    );
    assert_eq!(
        <IdentityChannel as Channel<2>>::support(&IdentityChannel),
        [0u64; 2]
    );
}

#[test]
fn identity_channel_emits_its_input_unchanged() {
    let mut input = PauliString::<2>::x(0);
    let _ = input.mul_assign(&PauliString::<2>::y(1));
    let _ = input.mul_assign(&PauliString::<2>::z(70));
    let coeff = Complex64::new(2.0, 3.0);
    for adjoint in [false, true] {
        assert_eq!(
            raw_outputs(&IdentityChannel, adjoint, input, coeff),
            vec![(input, coeff)]
        );
    }
}

/// Pins the transcribed copy: 144 undirected edges over qubits `0..126`, sorted and unique as `(lo, hi)`, degree histogram 2 × 1, 89 × 2, 36 × 3.
#[test]
fn heavy_hex_127_edges_match_the_source_lattice() {
    assert_eq!(HEAVY_HEX_127_EDGES.len(), 144);
    let mut degree = [0usize; 127];
    let mut prev = (0usize, 0usize);
    for (i, &(a, b)) in HEAVY_HEX_127_EDGES.iter().enumerate() {
        assert!(a < b, "edge {i} is not (lo, hi): ({a}, {b})");
        assert!(b < 127, "edge {i} names qubit {b} outside 0..126");
        if i > 0 {
            assert!(prev < (a, b), "edge {i} breaks the sorted-unique order");
        }
        prev = (a, b);
        degree[a] += 1;
        degree[b] += 1;
    }
    let mut histogram = [0usize; 4];
    for d in degree {
        histogram[d] += 1;
    }
    assert_eq!(histogram, [0, 2, 89, 36]);
}
