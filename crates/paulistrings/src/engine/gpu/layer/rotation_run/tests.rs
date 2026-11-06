use super::*;
use crate::channel::Prepare;
use crate::engine::gpu::layer::GpuLayerOptions;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::test_support::{assert_terms_close_up_to_residues, coefficient_tolerance};
use crate::{
    propagate, BuiltinTruncation, Circuit, CoefficientThreshold, Direction, PauliRotation,
    PauliString, PauliSum,
};
use num_complex::Complex64;

/// `crowded` distinct strings on qubits `0..8` alone, each with `Z` or `Y` on one of them, so all share the class with nothing elsewhere, and 200 strings that also reach past qubit 8.
fn one_crowded_class<const W: usize>(num_qubits: usize, crowded: usize) -> PauliSum<W> {
    let mut accumulator = BuildAccumulator::<W>::new(num_qubits);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state >> 16
    };
    let mut seen = std::collections::HashSet::new();
    while seen.len() < crowded {
        let pattern = next() & 0xFFFF;
        if pattern >> 8 == 0 || !seen.insert(pattern) {
            continue;
        }
        let mut p = PauliString::<W>::identity();
        p.x[0] = pattern & 0xFF;
        p.z[0] = pattern >> 8;
        accumulator.add_term(p, Complex64::new(0.5 + (next() % 1000) as f64 * 1e-3, 0.0));
    }
    for i in 0..200 {
        let mut p = PauliString::<W>::z(i % 8);
        let q = 8 + (next() as usize) % (num_qubits - 8);
        p.x[q / 64] |= 1 << (q % 64);
        accumulator.add_term(p, Complex64::new(0.2 + i as f64 * 1e-3, 0.0));
    }
    accumulator.finalize()
}

/// A class too large for any block runs fused, over global tables, and agrees with the host's run under a per-term filter.
fn crowded_class_runs_fused<const W: usize>(num_qubits: usize) {
    let input = one_crowded_class::<W>(num_qubits, 700);
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..8 {
        circuit.push(PauliRotation::new(PauliString::<W>::x(q), 0.3 + 0.07 * q as f64));
    }
    for threshold in [1e-9, 1e-3] {
        let policy = BuiltinTruncation::from(CoefficientThreshold(threshold));
        let keep = KeepProgram::lower(&policy).expect("a per-term program");
        let mut sum = GpuSum::<W, Complex64>::from_host(&input, 0).expect("upload");
        let mut scratch = LayerScratch::new(&sum, GpuLayerOptions::default()).expect("scratch");
        let layers: Vec<Prepared<W>> = circuit
            .channels
            .iter()
            .map(|channel| channel.prepare(sum.hash(), false).expect("prepared"))
            .collect();
        let bits = sum.hash().bits();
        let fused = rotation_run_fused(&mut sum, &layers, &keep, bits, &mut scratch).expect("run");
        assert!(fused, "W={W} threshold {threshold}: the crowded class fell back");
        let want = propagate(&circuit, input.clone(), &policy, Direction::Forward);
        assert_terms_close_up_to_residues(
            &sum.to_host().expect("download"),
            &want,
            1e3 * coefficient_tolerance::<Complex64>(),
            &format!("crowded class W={W} threshold {threshold}"),
        );
    }
}

#[test]
fn a_class_outgrowing_every_block_runs_fused() {
    crate::require_cuda!();
    crowded_class_runs_fused::<1>(24);
    crowded_class_runs_fused::<4>(256);
}
