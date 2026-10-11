use super::*;
use crate::channel::Prepare;
use crate::test_support::{haar_su4_matrix, sqrt_swap_matrix};

/// One of every constructor on `(q0, q1)`, both qubit orders for the two-qubit ones.
fn every_constructor(q0: usize, q1: usize) -> Vec<(&'static str, PtmChannel)> {
    let t_gate = [[ONE, ZERO], [ZERO, Complex64::from_polar(1.0, 0.7)]];
    let mut channels = vec![
        ("hadamard", PtmChannel::hadamard(q0)),
        ("phase_gate", PtmChannel::phase_gate(q0)),
        ("phase_gate_adjoint", PtmChannel::phase_gate(q0).adjoint()),
        ("pauli_x", PtmChannel::pauli_x(q0)),
        ("pauli_y", PtmChannel::pauli_y(q0)),
        ("pauli_z", PtmChannel::pauli_z(q0)),
        ("unitary_1q", PtmChannel::unitary_1q(q0, t_gate)),
        (
            "pauli_channel",
            PtmChannel::pauli_channel(q0, 0.03, 0.04, 0.05),
        ),
    ];
    for p in [0.0, 0.07, 0.75, 1.0] {
        channels.push(("depolarizing", PtmChannel::depolarizing(q0, p)));
        channels.push(("dephasing", PtmChannel::dephasing(q0, p)));
        channels.push(("amplitude_damping", PtmChannel::amplitude_damping(q0, p)));
    }
    for (a, b) in [(q0, q1), (q1, q0)] {
        channels.push(("cnot", PtmChannel::cnot(a, b)));
        channels.push(("cz", PtmChannel::cz(a, b)));
        channels.push(("swap", PtmChannel::swap(a, b)));
        channels.push((
            "unitary_2q sqrt-swap",
            PtmChannel::unitary_2q(a, b, sqrt_swap_matrix()),
        ));
        channels.push((
            "unitary_2q haar",
            PtmChannel::unitary_2q(a, b, haar_su4_matrix()),
        ));
        channels.push(("depolarizing_2q", PtmChannel::depolarizing_2q(a, b, 0.07)));
        channels.push((
            "depolarizing_2q",
            PtmChannel::depolarizing_2q(a, b, 15.0 / 16.0),
        ));
    }
    channels
}

fn assert_same_local_ptm<const W: usize>(a: &Prepared<W>, b: &Prepared<W>, what: &str) {
    let (Prepared::Local(a), Prepared::Local(b)) = (a, b) else {
        panic!("{what}: both preparations must be local");
    };
    assert_eq!(
        a.qubits().collect::<Vec<_>>(),
        b.qubits().collect::<Vec<_>>(),
        "{what}"
    );
    assert_eq!(a.num_deltas(), b.num_deltas(), "{what}");
    for (x, y) in a.deltas().iter().zip(b.deltas()) {
        assert_eq!(x.local_delta, y.local_delta, "{what}");
        assert_eq!(x.bucket_delta, y.bucket_delta, "{what}");
        assert_eq!((x.mask_x, x.mask_z), (y.mask_x, y.mask_z), "{what}");
        for (u, v) in x.amplitude.iter().zip(y.amplitude.iter()) {
            assert_eq!(
                (u.re.to_bits(), u.im.to_bits()),
                (v.re.to_bits(), v.im.to_bits()),
                "{what}: delta {}",
                x.local_delta
            );
        }
    }
}

/// The direct table must be bit for bit what probing `apply` derives, in both directions.
fn check_prepare_matches_the_probe<const W: usize>(q0: usize, q1: usize) {
    for seed in [1u64, 2] {
        let hash = Gf2Hash::<W>::new(64 * W, 10, seed);
        for (name, channel) in every_constructor(q0, q1) {
            for adjoint in [false, true] {
                let direct = Prepare::<W>::prepare(&channel, &hash, adjoint).unwrap();
                let probed = Prepared::derive_local(&channel, &hash, adjoint).unwrap();
                assert_same_local_ptm(&direct, &probed, &format!("{name} adjoint={adjoint}"));
            }
        }
    }
}

#[test]
fn prepare_matches_the_probe_of_apply_w1() {
    check_prepare_matches_the_probe::<1>(3, 9);
}

#[test]
fn prepare_matches_the_probe_of_apply_w2() {
    check_prepare_matches_the_probe::<2>(60, 70);
    check_prepare_matches_the_probe::<2>(5, 127);
}

#[test]
fn adjoint_is_the_transpose() {
    for (name, channel) in every_constructor(2, 6) {
        assert_eq!(
            channel.adjoint().transfer_matrix(false),
            channel.transfer_matrix(true),
            "{name}"
        );
        assert_eq!(
            channel.adjoint().adjoint().transfer_matrix(false),
            channel.transfer_matrix(false),
            "{name}"
        );
    }
}

#[test]
fn each_constructor_picks_its_storage() {
    let kind = |channel: &PtmChannel| match channel.storage {
        PtmStorage::SignedPermutation { .. } => "permutation",
        PtmStorage::Diagonal { .. } => "diagonal",
        PtmStorage::Sparse { .. } => "sparse",
    };
    for (name, channel) in every_constructor(0, 1) {
        let expected = match name {
            "hadamard" | "phase_gate" | "phase_gate_adjoint" | "pauli_x" | "pauli_y"
            | "pauli_z" | "cnot" | "cz" | "swap" => "permutation",
            "depolarizing" | "dephasing" | "pauli_channel" | "depolarizing_2q" => "diagonal",
            _ => "sparse",
        };
        assert_eq!(kind(&channel), expected, "{name}");
    }
}

#[test]
fn debug_name_is_the_constructor_name() {
    for (label, channel) in every_constructor(0, 1) {
        let expected = label.split(' ').next().unwrap();
        assert_eq!(Channel::<1>::debug_name(&channel), expected, "{label}");
        let adjoint_name = Channel::<1>::debug_name(&channel.adjoint());
        if let Some(source) = expected.strip_suffix("_adjoint") {
            assert_eq!(adjoint_name, source);
        } else {
            assert_eq!(adjoint_name, format!("{expected}_adjoint"));
        }
    }
}

#[test]
fn a_channel_fits_in_40_bytes() {
    assert!(core::mem::size_of::<PtmChannel>() <= 40);
}

/// The two-qubit order is the constructor's: `cnot(4, 1)` controls on 4.
#[test]
fn a_reversed_pair_keeps_its_roles() {
    let x_on_control = crate::pauli_string::PauliString::<1>::x(4);
    let out = crate::test_support::outputs(&PtmChannel::cnot(4, 1), false, x_on_control, ONE);
    let mut expected = x_on_control;
    let _ = expected.mul_assign(&crate::pauli_string::PauliString::<1>::x(1));
    assert_eq!(out, vec![(expected.x, expected.z, ONE)]);
}

#[test]
#[should_panic(expected = "PtmChannel::depolarizing: p must lie in [0, 1] (got 1.5)")]
fn depolarizing_rejects_p_above_one() {
    let _ = PtmChannel::depolarizing(0, 1.5);
}

#[test]
#[should_panic(expected = "PtmChannel::dephasing: p must lie in [0, 1] (got -0.1)")]
fn dephasing_rejects_negative_p() {
    let _ = PtmChannel::dephasing(0, -0.1);
}

#[test]
#[should_panic(expected = "PtmChannel::amplitude_damping: gamma must lie in [0, 1] (got NaN)")]
fn amplitude_damping_rejects_nan() {
    let _ = PtmChannel::amplitude_damping(0, f64::NAN);
}

#[test]
#[should_panic(expected = "PtmChannel::pauli_channel: px + py + pz must be at most 1")]
fn pauli_channel_rejects_a_total_above_one() {
    let _ = PtmChannel::pauli_channel(0, 0.5, 0.3, 0.3);
}

#[test]
#[should_panic(expected = "PtmChannel::depolarizing_2q: the two qubits must differ (both are 2)")]
fn depolarizing_2q_rejects_equal_qubits() {
    let _ = PtmChannel::depolarizing_2q(2, 2, 0.1);
}

#[test]
#[should_panic(expected = "PtmChannel::unitary_2q: the two qubits must differ (both are 0)")]
fn unitary_2q_rejects_equal_qubits() {
    let _ = PtmChannel::unitary_2q(0, 0, sqrt_swap_matrix());
}
