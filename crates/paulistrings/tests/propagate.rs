//! Integration tests for `propagate` against hand-computed Pauli algebra.
//!
//! These tests exercise the public API end-to-end: construct a `PauliSum` via
//! `BuildAccumulator`, push one or more channels onto a `Circuit`, propagate,
//! and check the result against algebra worked out by hand. Single-channel
//! cases go through `propagate` on a one-channel circuit — the only way to
//! drive one layer from outside the crate, and the path users actually take.
//!
//! Every fixture here is small enough to live in a single bucket (the
//! partition is trivial below `DEFAULT_TARGET_BUCKET_LEN`), so `bucket(0)` is
//! the whole sum in plain lexicographic key order and the positional
//! assertions below are well defined.
//!
//! `tests/propagate_bucketed.rs` covers the bucketed-engine-specific
//! properties: agreement with the naive oracle over whole circuits, and
//! thread-count stability.

use num_complex::Complex64;
use paulistrings::channel::{Channel, Clifford1Q, Clifford2Q, IdentityChannel, PauliRotation};
use paulistrings::test_support::approx_eq;
use paulistrings::truncation::{CoefficientThreshold, TopN, WeightCutoff};
use paulistrings::{
    propagate, BuildAccumulator, Circuit, Direction, PauliString, PauliSum, Phase, TruncationPolicy,
};

const TOL: f64 = 1e-12;

/// No-op truncation policy: keeps every term, no per-layer pass.
struct NoTruncation;
impl<const W: usize> TruncationPolicy<W> for NoTruncation {}

/// Build a `PauliSum<1>` from a list of `(PauliString, coeff)` pairs via
/// `BuildAccumulator`. Public-API alternative to the test-only
/// `PauliSum::from_strings` helper inside the crate.
fn sum1(num_qubits: usize, terms: &[(PauliString<1>, Complex64)]) -> PauliSum<1> {
    let mut acc = BuildAccumulator::<1>::new(num_qubits);
    for (p, c) in terms {
        acc.add_term(*p, Phase::ONE, *c);
    }
    acc.finalize()
}

fn sum2(num_qubits: usize, terms: &[(PauliString<2>, Complex64)]) -> PauliSum<2> {
    let mut acc = BuildAccumulator::<2>::new(num_qubits);
    for (p, c) in terms {
        acc.add_term(*p, Phase::ONE, *c);
    }
    acc.finalize()
}

/// A one-channel circuit — the smallest unit `propagate` can drive.
fn single<const W: usize, C: Channel<W> + 'static>(num_qubits: usize, ch: C) -> Circuit<W> {
    let mut c = Circuit::<W>::new(num_qubits);
    c.push(ch);
    c
}

/// One forward layer of `ch` on `input`, with `policy` threaded through.
fn layer1<C: Channel<1> + 'static, T: TruncationPolicy<1>>(
    input: &PauliSum<1>,
    num_qubits: usize,
    ch: C,
    policy: &T,
) -> PauliSum<1> {
    propagate(
        &single(num_qubits, ch),
        input.clone(),
        policy,
        Direction::Forward,
    )
}

/// `layer1` at `W = 2`.
fn layer2<C: Channel<2> + 'static, T: TruncationPolicy<2>>(
    input: &PauliSum<2>,
    num_qubits: usize,
    ch: C,
    policy: &T,
) -> PauliSum<2> {
    propagate(
        &single(num_qubits, ch),
        input.clone(),
        policy,
        Direction::Forward,
    )
}

#[test]
fn single_layer_h_conjugates_z_to_x() {
    let input = sum1(1, &[(PauliString::<1>::z(0), Complex64::new(1.0, 0.0))]);
    let out = layer1(&input, 1, Clifford1Q::h(0), &NoTruncation);
    assert_eq!(out.len(), 1);
    let x = PauliString::<1>::x(0);
    assert_eq!(out.bucket(0).0[0], x.x);
    assert_eq!(out.bucket(0).1[0], x.z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn single_layer_h_conjugates_x_to_z() {
    let input = sum1(1, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
    let out = layer1(&input, 1, Clifford1Q::h(0), &NoTruncation);
    assert_eq!(out.len(), 1);
    let z = PauliString::<1>::z(0);
    assert_eq!(out.bucket(0).0[0], z.x);
    assert_eq!(out.bucket(0).1[0], z.z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn single_layer_s_conjugates_x_to_y() {
    // S · X · S† = Y (with phase +1).
    let input = sum1(1, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
    let out = layer1(&input, 1, Clifford1Q::s(0), &NoTruncation);
    assert_eq!(out.len(), 1);
    let y = PauliString::<1>::y(0);
    assert_eq!(out.bucket(0).0[0], y.x);
    assert_eq!(out.bucket(0).1[0], y.z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn single_layer_cnot_propagates_z_control() {
    // The non-trivial generators for CNOT(0,1) are X⊗I → X⊗X and I⊗Z → Z⊗Z.
    // Test the I⊗Z case: input Z on qubit 1 → Z on qubit 0 AND qubit 1.
    let input = sum2(2, &[(PauliString::<2>::z(1), Complex64::new(1.0, 0.0))]);
    let out = layer2(&input, 2, Clifford2Q::cnot(0, 1), &NoTruncation);
    assert_eq!(out.len(), 1);
    let mut expected = PauliString::<2>::z(0);
    let _ = expected.mul_assign(&PauliString::<2>::z(1));
    assert_eq!(out.bucket(0).0[0], expected.x);
    assert_eq!(out.bucket(0).1[0], expected.z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn single_layer_cnot_propagates_x_target() {
    // CNOT(0,1) · (X⊗I) · CNOT = X⊗X (X on the control fans out to target).
    let input = sum2(2, &[(PauliString::<2>::x(0), Complex64::new(1.0, 0.0))]);
    let out = layer2(&input, 2, Clifford2Q::cnot(0, 1), &NoTruncation);
    assert_eq!(out.len(), 1);
    let mut expected = PauliString::<2>::x(0);
    let _ = expected.mul_assign(&PauliString::<2>::x(1));
    assert_eq!(out.bucket(0).0[0], expected.x);
    assert_eq!(out.bucket(0).1[0], expected.z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn single_layer_pauli_rotation_pi_z_flips_x_sign() {
    // exp(-i·π·Z/2) · X · exp(+i·π·Z/2) = -X.
    // `PauliRotation` emits cos(θ)·X + sin(θ)·Y; at θ=π that is (-1)·X plus a
    // Y term with sin(π) = 1.2246e-16, which is not an exact zero and so
    // survives the merge with a tiny coefficient. Hence approx_eq.
    let input = sum1(1, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
    let p = PauliString::<1>::z(0);
    let out = layer1(
        &input,
        1,
        PauliRotation::new(p, std::f64::consts::PI),
        &NoTruncation,
    );
    let x = PauliString::<1>::x(0);
    let y = PauliString::<1>::y(0);
    let mut found_x: Option<Complex64> = None;
    let mut found_y: Option<Complex64> = None;
    for i in 0..out.len() {
        if out.bucket(0).0[i] == x.x && out.bucket(0).1[i] == x.z {
            found_x = Some(out.bucket(0).2[i]);
        } else if out.bucket(0).0[i] == y.x && out.bucket(0).1[i] == y.z {
            found_y = Some(out.bucket(0).2[i]);
        }
    }
    assert!(approx_eq(found_x.unwrap(), Complex64::new(-1.0, 0.0), TOL));
    assert!(approx_eq(
        found_y.unwrap_or(Complex64::new(0.0, 0.0)),
        Complex64::new(0.0, 0.0),
        TOL
    ));
}

#[test]
fn single_layer_identity_channel_passes_sum_through() {
    let input = sum1(
        1,
        &[
            (PauliString::<1>::z(0), Complex64::new(1.0, 0.0)),
            (PauliString::<1>::x(0), Complex64::new(2.0, 0.0)),
        ],
    );
    let out = layer1(&input, 1, IdentityChannel::new(), &NoTruncation);
    assert_eq!(out.len(), 2);
    let (ox, oz, oc) = out.to_arrays();
    let (ix, iz, ic) = input.to_arrays();
    assert_eq!(ox, ix);
    assert_eq!(oz, iz);
    for (o, i) in oc.iter().zip(ic.iter()) {
        assert!(approx_eq(*o, *i, TOL));
    }
}

#[test]
fn single_layer_w2_word_boundary() {
    // H on qubit 64 conjugates Z(qubit 64) → X(qubit 64) — same as W=1
    // but with the bit in word 1.
    let input = sum2(65, &[(PauliString::<2>::z(64), Complex64::new(1.0, 0.0))]);
    let out = layer2(&input, 65, Clifford1Q::h(64), &NoTruncation);
    assert_eq!(out.len(), 1);
    let x = PauliString::<2>::x(64);
    assert_eq!(out.bucket(0).0[0], x.x);
    assert_eq!(out.bucket(0).1[0], x.z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn propagate_zero_channel_circuit_returns_input() {
    let input = sum1(1, &[(PauliString::<1>::z(0), Complex64::new(1.0, 0.0))]);
    let circuit = Circuit::<1>::new(1);
    let out = propagate(&circuit, input.clone(), &NoTruncation, Direction::Forward);
    assert_eq!(out.len(), input.len());
    let (ox, oz, oc) = out.to_arrays();
    let (ix, iz, ic) = input.to_arrays();
    assert_eq!(ox, ix);
    assert_eq!(oz, iz);
    for (o, i) in oc.iter().zip(ic.iter()) {
        assert!(approx_eq(*o, *i, TOL));
    }
}

#[test]
fn propagate_two_h_layers_is_identity() {
    let input = sum1(1, &[(PauliString::<1>::z(0), Complex64::new(1.0, 0.0))]);
    let mut circuit = Circuit::<1>::new(1);
    circuit.push(Clifford1Q::h(0));
    circuit.push(Clifford1Q::h(0));
    let out = propagate(&circuit, input.clone(), &NoTruncation, Direction::Forward);
    assert_eq!(out.len(), 1);
    assert_eq!(out.bucket(0).0[0], PauliString::<1>::z(0).x);
    assert_eq!(out.bucket(0).1[0], PauliString::<1>::z(0).z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn propagate_pauli_rotation_round_trip_via_heisenberg() {
    // Forward then Heisenberg on the same single-channel circuit must
    // round-trip to the input. Adjoint of `exp(-iθP/2)` is `exp(+iθP/2)`,
    // so the composition is the identity.
    let input = sum1(1, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
    let p = PauliString::<1>::z(0);
    let theta = std::f64::consts::FRAC_PI_3;
    let mut circuit = Circuit::<1>::new(1);
    circuit.push(PauliRotation::new(p, theta));
    let after_fwd = propagate(&circuit, input.clone(), &NoTruncation, Direction::Forward);
    let round_trip = propagate(&circuit, after_fwd, &NoTruncation, Direction::Heisenberg);
    // Find X coefficient, which should be ≈ 1 + 0i; tiny float remainder
    // on Y is acceptable.
    let x = PauliString::<1>::x(0);
    let mut found_x: Option<Complex64> = None;
    for i in 0..round_trip.len() {
        if round_trip.bucket(0).0[i] == x.x && round_trip.bucket(0).1[i] == x.z {
            found_x = Some(round_trip.bucket(0).2[i]);
        }
    }
    assert!(approx_eq(found_x.unwrap(), Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn propagate_clifford_s_round_trip_via_heisenberg() {
    // Forward applies S (X → Y); Heisenberg applies S† (Y → X). Net
    // identity on X. This exercises the non-trivial Clifford1Q::adjoint
    // table — if apply_adjoint defaulted to apply, we'd land on Y.
    let input = sum1(1, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
    let mut circuit = Circuit::<1>::new(1);
    circuit.push(Clifford1Q::s(0));
    let after_fwd = propagate(&circuit, input.clone(), &NoTruncation, Direction::Forward);
    // sanity: after forward, the term is Y.
    assert_eq!(after_fwd.bucket(0).0[0], PauliString::<1>::y(0).x);
    assert_eq!(after_fwd.bucket(0).1[0], PauliString::<1>::y(0).z);
    let round_trip = propagate(&circuit, after_fwd, &NoTruncation, Direction::Heisenberg);
    assert_eq!(round_trip.len(), 1);
    assert_eq!(round_trip.bucket(0).0[0], PauliString::<1>::x(0).x);
    assert_eq!(round_trip.bucket(0).1[0], PauliString::<1>::x(0).z);
    assert!(approx_eq(
        round_trip.bucket(0).2[0],
        Complex64::new(1.0, 0.0),
        TOL
    ));
}

#[test]
fn propagate_heisenberg_reverses_channel_order() {
    // Pick a circuit [A, B] where AB ≠ BA. Forward applies A then B.
    // Heisenberg applies B† then A†. For self-adjoint H and non-self-adjoint
    // S, [H, S] forward on Z gives:
    //   H: Z → X
    //   S: X → Y
    // Final: Y.
    // Heisenberg on the same Z input would do:
    //   S†: Z → Z
    //   H†: Z → X
    // Final: X.
    // Distinct outputs prove the reversal is happening.
    let input = sum1(1, &[(PauliString::<1>::z(0), Complex64::new(1.0, 0.0))]);
    let mut circuit = Circuit::<1>::new(1);
    circuit.push(Clifford1Q::h(0));
    circuit.push(Clifford1Q::s(0));
    let fwd = propagate(&circuit, input.clone(), &NoTruncation, Direction::Forward);
    let heis = propagate(&circuit, input, &NoTruncation, Direction::Heisenberg);
    assert_eq!(fwd.bucket(0).0[0], PauliString::<1>::y(0).x);
    assert_eq!(fwd.bucket(0).1[0], PauliString::<1>::y(0).z);
    assert_eq!(heis.bucket(0).0[0], PauliString::<1>::x(0).x);
    assert_eq!(heis.bucket(0).1[0], PauliString::<1>::x(0).z);
}

/// `WeightCutoff(1)` threads through the layer and
/// drops the weight-2 term during the merge.
#[test]
fn single_layer_weight_cutoff_drops_high_weight() {
    // Z on q0 (weight 1) + Z⊗Z on q0,q1 (weight 2).
    let mut zz = PauliString::<2>::z(0);
    let _ = zz.mul_assign(&PauliString::<2>::z(1));
    let input = sum2(
        2,
        &[
            (PauliString::<2>::z(0), Complex64::new(1.0, 0.0)),
            (zz, Complex64::new(2.0, 0.0)),
        ],
    );
    let out = layer2(&input, 2, IdentityChannel::new(), &WeightCutoff(1));
    assert_eq!(out.len(), 1);
    let z0 = PauliString::<2>::z(0);
    assert_eq!(out.bucket(0).0[0], z0.x);
    assert_eq!(out.bucket(0).1[0], z0.z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

/// `TopN(1)` threads through `propagate` via
/// `finalize_layer`. After a single rotation that fans Z's input X into a
/// (X, Y) superposition, the post-layer truncation must keep at most one
/// term — confirming `finalize_layer` is invoked.
#[test]
fn propagate_top_n_truncates_each_layer() {
    let input = sum1(1, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
    let p = PauliString::<1>::z(0);
    let mut circuit = Circuit::<1>::new(1);
    circuit.push(PauliRotation::new(p, std::f64::consts::FRAC_PI_3));
    let out = propagate(&circuit, input, &TopN(1), Direction::Forward);
    assert!(out.len() <= 1);
    // The X term has |cos(π/3)| = 0.5; the Y term has |sin(π/3)| ≈ 0.866;
    // TopN(1) keeps the larger → Y.
    assert_eq!(out.len(), 1);
    let y = PauliString::<1>::y(0);
    assert_eq!(out.bucket(0).0[0], y.x);
    assert_eq!(out.bucket(0).1[0], y.z);
    // `assert_invariants` is debug-only, and integration tests are also built
    // by `cargo test --release`, so the call has to be gated or the release test
    // binary does not compile.
    #[cfg(debug_assertions)]
    out.assert_invariants();
}

/// `CoefficientThreshold` threads through the layer and
/// drops the sub-eps term during the merge. Using `IdentityChannel` keeps
/// the algebra trivial — the test isolates the truncation behavior.
#[test]
fn single_layer_with_threshold_drops_below_eps() {
    let input = sum1(
        1,
        &[
            (PauliString::<1>::z(0), Complex64::new(1.0, 0.0)),
            (PauliString::<1>::x(0), Complex64::new(1e-12, 0.0)),
        ],
    );
    let out = layer1(
        &input,
        1,
        IdentityChannel::new(),
        &CoefficientThreshold(1e-9),
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out.bucket(0).0[0], PauliString::<1>::z(0).x);
    assert_eq!(out.bucket(0).1[0], PauliString::<1>::z(0).z);
    assert!(approx_eq(out.bucket(0).2[0], Complex64::new(1.0, 0.0), TOL));
}

#[test]
fn single_layer_combines_inputs_that_collide_under_channel() {
    // X and Y both on qubit 0, with coeffs 3 and 2. Apply S:
    //   S · X = Y (phase +1), S · Y = -X (phase -1).
    // Output before merge: (Y, +3), (X, -2). After the merge: X with coeff
    // -2 (sorted first since x=1, z=0), Y with coeff 3.
    let input = sum1(
        1,
        &[
            (PauliString::<1>::x(0), Complex64::new(3.0, 0.0)),
            (PauliString::<1>::y(0), Complex64::new(2.0, 0.0)),
        ],
    );
    let out = layer1(&input, 1, Clifford1Q::s(0), &NoTruncation);
    assert_eq!(out.len(), 2);
    let x = PauliString::<1>::x(0);
    let y = PauliString::<1>::y(0);
    // They tie on x[0]=1, so z[0] decides: X has z=0, Y has z=1, so X < Y.
    // Coeffs: X = -2, Y = 3.
    assert_eq!(out.bucket(0).0[0], x.x);
    assert_eq!(out.bucket(0).1[0], x.z);
    assert!(approx_eq(
        out.bucket(0).2[0],
        Complex64::new(-2.0, 0.0),
        TOL
    ));
    assert_eq!(out.bucket(0).0[1], y.x);
    assert_eq!(out.bucket(0).1[1], y.z);
    assert!(approx_eq(out.bucket(0).2[1], Complex64::new(3.0, 0.0), TOL));
}
