//! Runs of one-to-one channels taken as one pass agree with the same channels applied one call at a time, which no run can span.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use num_complex::Complex64;
use paulistrings::test_support::{
    assert_terms_close, heavy_hex_127_edges, low_weight_sum, rand_sum, unpinned_partitions, Xs64,
};
use paulistrings::{
    propagate, propagate_with, BuildAccumulator, Circuit, CoefficientThreshold, Direction,
    LayerScratch, PartitionRows, PartitionRuntime, PartitionedSum, PauliRotation, PauliString,
    PauliSum, PropagateOptions, PtmChannel, ScatterOptions, ScatterRows, TruncationPolicy,
};

const TOL: f64 = 1e-12;

/// A gate of the test circuits, kept as data so one circuit can be built whole and gate by gate.
#[derive(Clone, Copy, Debug)]
enum Gate {
    /// A rotation about `X` on the first qubit list and `Z` on the second, `Y` where they meet.
    Rotation {
        x: [Option<usize>; 2],
        z: [Option<usize>; 2],
        theta: f64,
    },
    Hadamard(usize),
    Phase(usize),
    Cnot(usize, usize),
    Cz(usize, usize),
    Depolarizing(usize),
}

fn rotation(x: &[usize], z: &[usize], theta: f64) -> Gate {
    let pad = |qubits: &[usize]| [qubits.first().copied(), qubits.get(1).copied()];
    Gate::Rotation { x: pad(x), z: pad(z), theta }
}

fn build<const W: usize>(num_qubits: usize, gates: &[Gate]) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    for &gate in gates {
        match gate {
            Gate::Rotation { x, z, theta } => {
                let mut generator = PauliString::<W>::identity();
                for q in x.into_iter().flatten() {
                    generator.x[q / 64] |= 1 << (q % 64);
                }
                for q in z.into_iter().flatten() {
                    generator.z[q / 64] |= 1 << (q % 64);
                }
                circuit.push(PauliRotation::new(generator, theta));
            }
            Gate::Hadamard(q) => circuit.push(PtmChannel::hadamard(q)),
            Gate::Phase(q) => circuit.push(PtmChannel::phase_gate(q)),
            Gate::Cnot(a, b) => circuit.push(PtmChannel::cnot(a, b)),
            Gate::Cz(a, b) => circuit.push(PtmChannel::cz(a, b)),
            Gate::Depolarizing(q) => circuit.push(PtmChannel::depolarizing(q, 0.05)),
        }
    }
    circuit
}

/// `gates` applied one single-channel call at a time, in the direction's order.
fn gate_by_gate<const W: usize, T: TruncationPolicy<W>>(
    num_qubits: usize,
    gates: &[Gate],
    mut sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W> {
    let order: Vec<&Gate> = match direction {
        Direction::Forward => gates.iter().collect(),
        Direction::Heisenberg => gates.iter().rev().collect(),
    };
    for gate in order {
        sum =
            propagate(&build::<W>(num_qubits, std::slice::from_ref(gate)), sum, policy, direction);
    }
    sum
}

/// One kicked-Ising step on the heavy-hex lattice: `rx(-π/4)` on every qubit, then `ZZ(θ)` on every bond.
fn heavy_hex_step(theta: f64) -> Vec<Gate> {
    let mut gates: Vec<Gate> = (0..127).map(|q| rotation(&[q], &[], -FRAC_PI_4)).collect();
    gates.extend(heavy_hex_127_edges().into_iter().map(|(a, b)| rotation(&[], &[a, b], theta)));
    gates
}

fn z_at(site: usize) -> PauliSum<2> {
    let mut accumulator = BuildAccumulator::<2>::with_capacity(127, 1);
    accumulator.add_term(PauliString::<2>::z(site), Complex64::new(1.0, 0.0));
    accumulator.finalize()
}

#[test]
fn heavy_hex_clifford_steps_agree_gate_by_gate() {
    let mut gates = Vec::new();
    for _ in 0..5 {
        gates.extend(heavy_hex_step(-FRAC_PI_2));
    }
    let policy = CoefficientThreshold(1e-4);
    let circuit = build::<2>(127, &gates);
    let got = propagate(&circuit, z_at(62), &policy, Direction::Heisenberg);
    let want = gate_by_gate(127, &gates, z_at(62), &policy, Direction::Heisenberg);
    assert!(want.len() > 1000, "the series grows: {} terms", want.len());
    assert_terms_close(&got, &want, TOL, "heavy-hex Clifford");
}

/// Clifford runs of every kind between generic rotations, on `num_qubits` qubits.
fn mixed_gates(num_qubits: usize, layers: usize, seed: u64) -> Vec<Gate> {
    let mut rng = Xs64::new(seed);
    let mut draw = |n: usize| (rng.next_u64() % n as u64) as usize;
    let mut gates = Vec::new();
    for _ in 0..layers {
        let a = draw(num_qubits);
        let b = (a + 1 + draw(num_qubits - 1)) % num_qubits;
        let quarter = if draw(2) == 0 { FRAC_PI_2 } else { -FRAC_PI_2 };
        gates.push(match draw(9) {
            0 => Gate::Hadamard(a),
            1 => Gate::Phase(a),
            2 => Gate::Cnot(a, b),
            3 => Gate::Cz(a, b),
            4 => Gate::Depolarizing(a),
            5 => rotation(&[], &[a, b], quarter),
            6 => rotation(&[a], &[a], quarter),
            7 => rotation(&[a], &[b], quarter),
            _ => rotation(&[a], &[b], 0.3),
        });
    }
    gates
}

fn mixed_circuits_agree<const W: usize>(num_qubits: usize) {
    let policy = CoefficientThreshold(1e-3);
    // The larger sum is split into buckets and walked in parallel.
    for (seed, terms) in [(1u64, 10_000), (2, 2000)] {
        let gates = mixed_gates(num_qubits, 60, seed);
        let circuit = build::<W>(num_qubits, &gates);
        let sum = rand_sum::<W>(terms, num_qubits, seed);
        for direction in [Direction::Heisenberg, Direction::Forward] {
            let got = propagate(&circuit, sum.clone(), &policy, direction);
            let want = gate_by_gate(num_qubits, &gates, sum.clone(), &policy, direction);
            assert_terms_close(&got, &want, TOL, &format!("W={W}, seed {seed}, {direction:?}"));
        }
    }
}

#[test]
fn mixed_circuits_agree_gate_by_gate_w1() {
    mixed_circuits_agree::<1>(40);
}

#[test]
fn mixed_circuits_agree_gate_by_gate_w2() {
    mixed_circuits_agree::<2>(100);
}

#[test]
fn a_run_records_one_trace_row_per_layer() {
    let gates = heavy_hex_step(-FRAC_PI_2);
    let circuit = build::<2>(127, &gates);
    let mut scratch = LayerScratch::<2>::new();
    scratch.enable_trace();
    let options = PropagateOptions::default();
    let policy = CoefficientThreshold(1e-3);
    let mut sum = z_at(62);
    for _ in 0..2 {
        sum = propagate_with(&circuit, sum, &policy, Direction::Heisenberg, &mut scratch, options);
    }
    let trace = scratch.take_trace().expect("tracing is on");
    assert_eq!(trace.layers.len(), 2 * gates.len());
    let untimed = trace.layers.iter().filter(|layer| layer.nanos.iter().all(|&n| n == 0)).count();
    assert_eq!(
        untimed,
        2 * (heavy_hex_127_edges().len() - 1),
        "the bonds run as one pass per step"
    );
}

/// Runs of `Z`-type Cliffords (`ZZ`, `CZ`, `S` and `Z` quarter turns) between generic `X` rotations on `num_qubits` qubits.
fn z_type_runs(num_qubits: usize, steps: usize, seed: u64) -> Vec<Gate> {
    let mut rng = Xs64::new(seed);
    let mut draw = |n: usize| (rng.next_u64() % n as u64) as usize;
    let mut gates = Vec::new();
    for _ in 0..steps {
        for _ in 0..4 {
            gates.push(rotation(&[draw(num_qubits)], &[], 0.4));
        }
        for _ in 0..12 {
            let a = draw(num_qubits);
            let b = (a + 1 + draw(num_qubits - 1)) % num_qubits;
            gates.push(match draw(4) {
                0 => rotation(&[], &[a, b], FRAC_PI_2),
                1 => Gate::Cz(a, b),
                2 => Gate::Phase(a),
                _ => rotation(&[], &[a], -FRAC_PI_2),
            });
        }
    }
    gates
}

/// At P > 1 under x-only rows the `Z`-type Cliffords are local and run as one pass between the exchanging `X` rotations, and Forward ends the call on a run.
#[test]
fn runs_that_exchange_nothing_are_taken_on_every_partition_and_agree() {
    let num_qubits = 40;
    let gates = z_type_runs(num_qubits, 4, 21);
    let circuit = build::<1>(num_qubits, &gates);
    let sum = low_weight_sum::<1>(10_000, num_qubits, 4, 21);
    let policy = CoefficientThreshold(1e-3);
    let all = [(1u64 << num_qubits) - 1];
    for ((partitions, bits), direction) in [(2, 1u8), (4, 2)]
        .into_iter()
        .flat_map(|cell| [(cell, Direction::Heisenberg), (cell, Direction::Forward)])
    {
        let want = propagate(&circuit, sum.clone(), &policy, direction);
        // Rows reading x-bits only, so a delta of z-bits alone stays on its partition.
        let rows = PartitionRows::<1>::from_seed_excluding(num_qubits, bits, 5, &[0], &all);
        let options = ScatterOptions {
            runtime: PartitionRuntime::new(&unpinned_partitions(partitions, 2)).unwrap(),
            rows: ScatterRows::Explicit(rows),
        };
        let mut split = PartitionedSum::scatter_with(sum.clone(), options);
        split.enable_trace();
        split.propagate_with(&circuit, &policy, direction, PropagateOptions::default());
        let label = format!("P = {partitions}, {direction:?}");
        assert_terms_close(&split.gather(), &want, TOL, &label);
        let trace = split.take_trace().expect("tracing is on");
        let untimed = trace.layers.iter().filter(|layer| layer.nanos.iter().all(|&n| n == 0));
        assert_eq!(
            untimed.count(),
            4 * 11,
            "{label}: each step's twelve Cliffords run as one pass"
        );
    }
}
