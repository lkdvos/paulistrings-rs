use super::*;

#[test]
fn support_mask_packs_cross_word_qubits() {
    // Qubit 70 at W=2 lands in word 1, bit 6.
    let mask: [u64; 2] = support_mask(&[5, 70]);
    assert_eq!(mask[0], 1u64 << 5);
    assert_eq!(mask[1], 1u64 << 6);
}

#[test]
fn support_mask_is_order_and_duplicate_insensitive() {
    let a: [u64; 1] = support_mask(&[3, 1, 3]);
    let b: [u64; 1] = support_mask(&[1, 3]);
    assert_eq!(a, b);
}

/// The default `debug_name` must survive erasure to `dyn Channel<W>` — that is how the engine sees every channel — and must trim both the module path and any generic arguments.
#[test]
fn debug_name_through_dyn_trims_path_and_generics() {
    use crate::pauli_string::PauliString;

    let channels: Vec<(Box<dyn Channel<1>>, &str)> = vec![
        (Box::new(PtmChannel::hadamard(0)), "hadamard"),
        (Box::new(PtmChannel::cnot(0, 1)), "cnot"),
        (
            // Generic in `W`, so `type_name` carries a `<1>` to trim.
            Box::new(PauliRotation::new(PauliString::<1>::z(0), 0.3)),
            "PauliRotation",
        ),
        (Box::new(PtmChannel::depolarizing(0, 0.1)), "depolarizing"),
        (Box::new(PtmChannel::dephasing(0, 0.1)), "dephasing"),
        (
            Box::new(PtmChannel::pauli_channel(0, 0.1, 0.1, 0.1)),
            "pauli_channel",
        ),
        (
            Box::new(PtmChannel::depolarizing_2q(0, 1, 0.1)),
            "depolarizing_2q",
        ),
        (
            Box::new(PtmChannel::amplitude_damping(0, 0.1)),
            "amplitude_damping",
        ),
        (
            Box::new(crate::test_support::IdentityChannel),
            "IdentityChannel",
        ),
    ];
    for (channel, expected) in &channels {
        assert_eq!(channel.debug_name(), *expected);
    }
}

#[test]
fn support_mask_of_empty_is_zero() {
    let mask: [u64; 2] = support_mask(&[]);
    assert_eq!(mask, [0, 0]);
}

#[test]
#[should_panic(expected = "outside the 64-qubit register")]
fn support_mask_rejects_a_qubit_past_the_width() {
    let _: [u64; 1] = support_mask(&[64]);
}

#[test]
fn push_appends_in_order_w1() {
    let mut buffer = OutputBuffer::<1>::new();
    buffer.push([0xAA], [0xBB], Complex64::new(1.0, 2.0));
    buffer.push([0xCC], [0xDD], Complex64::new(3.0, 4.0));
    assert_eq!(
        buffer.terms().collect::<Vec<_>>(),
        vec![
            ([0xAA], [0xBB], Complex64::new(1.0, 2.0)),
            ([0xCC], [0xDD], Complex64::new(3.0, 4.0)),
        ]
    );
}

#[test]
fn push_appends_in_order_w2() {
    let mut buffer = OutputBuffer::<2>::new();
    buffer.push([0x11, 0x22], [0x33, 0x44], Complex64::new(5.0, 6.0));
    assert_eq!(
        buffer.terms().collect::<Vec<_>>(),
        vec![([0x11, 0x22], [0x33, 0x44], Complex64::new(5.0, 6.0))]
    );
}

#[test]
fn clear_forgets_every_term() {
    let mut buffer = OutputBuffer::<1>::new();
    buffer.push([0xAA], [0xBB], Complex64::new(1.0, 0.0));
    buffer.push([0xCC], [0xDD], Complex64::new(2.0, 0.0));
    buffer.clear();
    assert_eq!(buffer.terms().len(), 0);
    buffer.push([0xEE], [0xFF], Complex64::new(3.0, 0.0));
    assert_eq!(
        buffer.terms().collect::<Vec<_>>(),
        vec![([0xEE], [0xFF], Complex64::new(3.0, 0.0))]
    );
}
