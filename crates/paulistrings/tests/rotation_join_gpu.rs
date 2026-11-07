//! The device join layer for rotations against the host `propagate`, in every coefficient type the device sums take.
//! Every case returns early without a device.

use num_complex::Complex64;
use paulistrings::gpu::GpuPartitionedSum;
use paulistrings::require_cuda;
use paulistrings::test_support::{
    assert_runs_agree, coefficient_tolerance, propagate_discarding, rand_sum_real,
    with_rotation_partners, zz_rotation, KeepAll,
};
use paulistrings::{
    propagate, ApproxTopN, BuiltinTruncation, Circuit, Coefficient, CoefficientThreshold,
    Direction, PauliRotation, PauliString, PauliSum,
};

/// One device layer of `rotation` on `sum` as `C`: the result widened back, and whether the layer joined.
fn device<const W: usize, C: Coefficient>(
    rotation: &PauliRotation<W>,
    sum: &PauliSum<W>,
    policy: &BuiltinTruncation,
) -> (PauliSum<W>, bool) {
    let mut circuit = Circuit::<W>::new(sum.num_qubits());
    circuit.push(rotation.clone());
    let narrowed = sum.cast::<C>(0.0).expect("a real input sum");
    let mut dev = GpuPartitionedSum::<W, C>::from_host(&narrowed, 0).expect("upload");
    dev.enable_trace();
    dev.propagate(&circuit, policy.clone(), Direction::Heisenberg).expect("device propagate");
    let got = dev.gather().expect("download").cast::<Complex64>(0.0).expect("widening");
    let layer = dev.take_trace().expect("tracing").layers.pop().expect("one layer");
    (got, layer.device[0].expect("a device partition").joined)
}

/// The join layer in `C` against the host, for `rotation` on `sum` under every policy.
fn agrees_as<const W: usize, C: Coefficient>(
    rotation: &PauliRotation<W>,
    sum: &PauliSum<W>,
    what: &str,
) {
    let mut circuit = Circuit::<W>::new(sum.num_qubits());
    circuit.push(rotation.clone());
    let tol = coefficient_tolerance::<C>();
    // A light cut of the grown sum, so the cut terms stay comparable.
    let grown = propagate(&circuit, sum.clone(), &KeepAll, Direction::Heisenberg).len();
    let policies = [
        BuiltinTruncation::Keep,
        BuiltinTruncation::from(CoefficientThreshold(0.05)),
        BuiltinTruncation::from(ApproxTopN(grown - grown / 100)),
    ];
    for policy in policies {
        let what = format!("{what} {} {policy:?}", C::NAME);
        let (want, discarded) =
            propagate_discarding(&circuit, sum.clone(), &policy, Direction::Heisenberg);
        let (joined, took) = device::<W, C>(rotation, sum, &policy);
        assert!(took, "{what}: the layer must join");
        assert_runs_agree(policy, discarded, &joined, &want, tol, &what);
    }
}

fn agrees<const W: usize>(rotation: &PauliRotation<W>, sum: &PauliSum<W>, what: &str) {
    agrees_as::<W, f64>(rotation, sum, what);
    agrees_as::<W, f32>(rotation, sum, what);
    agrees_as::<W, Complex64>(rotation, sum, what);
}

fn generator<const W: usize>(xs: &[usize], zs: &[usize]) -> PauliString<W> {
    let mut g = PauliString::<W> { x: [0; W], z: [0; W] };
    for &q in xs {
        g.x[q / 64] |= 1 << (q % 64);
    }
    for &q in zs {
        g.z[q / 64] |= 1 << (q % 64);
    }
    g
}

/// Partners that all miss, all hit, and half hit, for a one-qubit, a two-qubit (tabulated) and a wide rotation.
fn hits_and_misses<const W: usize>(num_qubits: usize, n: usize, seed: u64) {
    let last = num_qubits - 1;
    let rotations = [
        ("rx", PauliRotation::new(generator::<W>(&[1], &[]), 0.37)),
        ("zz", zz_rotation::<W>(0, last, 0.21)),
        ("wide", PauliRotation::new(generator::<W>(&[0, 2, last], &[2, 3, last]), -0.53)),
    ];
    for (name, rotation) in &rotations {
        let g = rotation.generator();
        let base = rand_sum_real::<W>(n, num_qubits, seed);
        let anticommuting = base
            .iter()
            .filter(|(x, z, _)| !PauliString { x: **x, z: **z }.commutes_with(&g))
            .count();
        for (label, count) in [("miss", 0), ("hit", anticommuting), ("both", anticommuting / 2)] {
            let sum = with_rotation_partners(&base, &g, count, seed ^ 0x5A);
            agrees(rotation, &sum, &format!("W={W} {name} {label}"));
        }
    }
}

#[test]
fn rotations_join_with_hits_and_misses_w1() {
    require_cuda!();
    hits_and_misses::<1>(12, 600, 0x7101);
}

#[test]
fn rotations_join_with_hits_and_misses_w2() {
    require_cuda!();
    hits_and_misses::<2>(100, 3_000, 0x7102);
}
