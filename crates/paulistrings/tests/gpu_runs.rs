//! Device runs against the host: permutation runs walked in one pass and rotation runs over joined one-qubit tables agree with `propagate` to tolerance.
//! Every case returns early without a device.

use num_complex::Complex64;
use paulistrings::gpu::GpuLayerOptions;
use paulistrings::require_cuda;
use paulistrings::test_support::{
    assert_terms_close_up_to_residues, coefficient_tolerance, device_run_as, low_weight_sum,
    zz_rotation, KeepAll,
};
use paulistrings::{
    propagate, BuildAccumulator, BuiltinTruncation, Circuit, Coefficient, CoefficientThreshold,
    Direction, PauliRotation, PauliString, PauliSum, PtmChannel, WeightCutoff,
};

/// A kicked-Ising step on a chain, `R_X(θ)` on every qubit and a quarter-turn `ZZ` on every bond, then `R_X(θ)` again on the first half: a rotation run, a Clifford run, a rotation run.
fn kicked_chain<const W: usize>(num_qubits: usize, theta: f64) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), theta));
    }
    for q in 0..num_qubits - 1 {
        circuit.push(zz_rotation::<W>(q, q + 1, std::f64::consts::FRAC_PI_2));
    }
    for q in 0..num_qubits / 2 {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), theta));
    }
    circuit
}

/// Weight-two strings with real coefficients, so the untruncated runs stay small.
fn sparse_input<const W: usize>(num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    for (x, z, c) in low_weight_sum::<W>(60, num_qubits, 2, seed).iter() {
        acc.add_term(PauliString::<W> { x: *x, z: *z }, Complex64::new(c.re, 0.0));
    }
    acc.finalize()
}

/// Cliffords alone: Hadamards, CNOTs and quarter-turn `ZZ`s, one run per direction.
fn clifford_block<const W: usize>(num_qubits: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in (0..num_qubits).step_by(3) {
        circuit.push(PtmChannel::hadamard(q));
    }
    for q in 0..num_qubits - 1 {
        circuit.push(PtmChannel::cnot(q, q + 1));
        circuit.push(zz_rotation::<W>(q, (q + 5) % num_qubits, -std::f64::consts::FRAC_PI_2));
    }
    circuit
}

/// `circuit` on `sum` narrowed to `C` on the device, widened back, and how many layers it applied inside a run past the run's first.
fn device_as<const W: usize, C: Coefficient>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &BuiltinTruncation,
    direction: Direction,
) -> (PauliSum<W>, usize) {
    let options = GpuLayerOptions::default();
    let (got, trace) = device_run_as::<W, C>(circuit, sum, policy, direction, options, true);
    let joined = trace.expect("tracing").layers.iter().filter(|l| l.nanos[0] == 0).count();
    (got, joined)
}

/// The device in `f64` and `c128` against the host `propagate`, both directions; returns the fewest layers any device run joined into a run.
fn agrees<const W: usize>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: impl Into<BuiltinTruncation>,
    name: &str,
) -> usize {
    let policy: BuiltinTruncation = policy.into();
    let mut joined = usize::MAX;
    for direction in [Direction::Forward, Direction::Heisenberg] {
        let want = propagate(circuit, sum.clone(), &policy, direction);
        assert!(want.len() > 1, "{name}: the fixture must keep terms");
        let (got, f64_joined) = device_as::<W, f64>(circuit, sum, &policy, direction);
        assert_terms_close_up_to_residues(
            &got,
            &want,
            1e3 * coefficient_tolerance::<f64>(),
            &format!("{name} f64 {direction:?}"),
        );
        let (got, c128_joined) = device_as::<W, Complex64>(circuit, sum, &policy, direction);
        assert_terms_close_up_to_residues(
            &got,
            &want,
            1e3 * coefficient_tolerance::<Complex64>(),
            &format!("{name} c128 {direction:?}"),
        );
        joined = joined.min(f64_joined).min(c128_joined);
    }
    joined
}

fn runs_agree<const W: usize>(num_qubits: usize, seed: u64) {
    let input = sparse_input::<W>(num_qubits, seed);
    let cliffords = clifford_block::<W>(num_qubits);
    let kicked = kicked_chain::<W>(num_qubits, 0.6);
    let mut rotations = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        rotations.push(PauliRotation::new(PauliString::<W>::x(q), 0.6));
    }
    for (name, circuit) in
        [("clifford", &cliffords), ("kicked", &kicked), ("rotations", &rotations)]
    {
        let name = format!("{name} W={W}");
        let joined = agrees(circuit, &input, KeepAll, &format!("{name} keep"));
        assert!(joined > 0, "{name}: the device took no run");
        agrees(circuit, &input, CoefficientThreshold(2e-3), &format!("{name} threshold"));
        agrees(circuit, &input, WeightCutoff((num_qubits / 2) as _), &format!("{name} weight"));
    }
}

/// Strings with `Z` on all of qubits `0..8` and distinct elsewhere, under `R_X` on those eight: each grows 256-fold within its class, so the one device block outgrows its rows and the run falls back to its layers one by one.
fn crowded_classes<const W: usize>(num_qubits: usize, strings: usize) {
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    for i in 0..strings {
        let mut p = PauliString::<W>::z(0);
        for q in 1..8 {
            p.z[q / 64] |= 1 << (q % 64);
        }
        let q = 8 + i % (num_qubits - 8);
        p.x[q / 64] |= 1 << (q % 64);
        if i >= num_qubits - 8 {
            p.z[q / 64] |= 1 << (q % 64);
        }
        acc.add_term(p, Complex64::new(1.0 + 0.1 * i as f64, 0.0));
    }
    let input = acc.finalize();
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..8 {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), 0.4 + 0.1 * q as f64));
    }
    let name = format!("crowded W={W}");
    agrees(&circuit, &input, CoefficientThreshold(1e-3), &name);
    agrees(&circuit, &input, CoefficientThreshold(1e-9), &name);
}

#[test]
fn crowded_rotation_classes_agree_with_the_host() {
    require_cuda!();
    crowded_classes::<1>(20, 12);
    crowded_classes::<2>(90, 24);
}

#[test]
fn device_runs_agree_with_the_host_w1() {
    require_cuda!();
    runs_agree::<1>(12, 0x5EED1);
}

#[test]
fn device_runs_agree_with_the_host_w2() {
    require_cuda!();
    runs_agree::<2>(90, 0x5EED2);
}
