//! The talk's fixed workload: the 127-qubit heavy-hex kicked Ising circuit.
//!
//! `read_edges`, `edge_coloring`, `zz` and `kicked_ising` are copied verbatim
//! from `crates/paulistrings/examples/small_m_ab.rs` (the gate-for-gate port of
//! `examples/common/circuits.py::heavy_hex_kicked_ising`). The copy is pinned
//! by `tests/agreement.rs::workload_is_pinned` to the committed constants
//! (144 edges, 1355 channels, 5038 final / 6311 peak terms at eps 2^-8).

use num_complex::Complex64;
use paulistrings::channel::PauliRotation;
use paulistrings::{BuildAccumulator, Circuit, PauliString, PauliSum, Phase};

pub const W: usize = 2;
pub const QUBITS: usize = 127;
pub const STEPS: usize = 5;
pub const THETA_H: f64 = 5.0 * std::f64::consts::PI / 16.0;
pub const OBSERVABLE_QUBIT: u32 = 62;
/// 2^-13: ~1.16e6 peak terms on this circuit (`presentation/data/term_growth.jsonl`).
pub const DEFAULT_EPS: f64 = 1.220703125e-4;

/// Default edge-list path, resolved against the repository root so the binary
/// works from any working directory.
pub fn default_edges_path() -> String {
    format!("{}/../../examples/data/heavy_hex_127.edges", env!("CARGO_MANIFEST_DIR"))
}

/// Undirected edges, `lo hi` per line, `#` comments.
pub fn read_edges(path: &str) -> Vec<(u32, u32)> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read the heavy-hex edge list at {path}: {e}"));
    let mut edges = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let a: u32 = it.next().expect("edge line has no first index").parse().unwrap();
        let b: u32 = it.next().expect("edge line has no second index").parse().unwrap();
        edges.push((a.min(b), a.max(b)));
    }
    edges.sort_unstable();
    edges
}

/// Greedy first-fit edge coloring in sorted edge order.
pub fn edge_coloring(edges: &[(u32, u32)]) -> Vec<Vec<(u32, u32)>> {
    let mut used: Vec<Vec<usize>> = Vec::new();
    let mut classes: Vec<Vec<(u32, u32)>> = Vec::new();
    let n = edges.iter().map(|&(a, b)| a.max(b) as usize + 1).max().unwrap_or(0);
    used.resize(n, Vec::new());
    for &(a, b) in edges {
        let mut color = 0usize;
        while used[a as usize].contains(&color) || used[b as usize].contains(&color) {
            color += 1;
        }
        while classes.len() <= color {
            classes.push(Vec::new());
        }
        classes[color].push((a, b));
        used[a as usize].push(color);
        used[b as usize].push(color);
    }
    classes
}

fn zz(a: u32, b: u32) -> PauliString<W> {
    let mut g = PauliString::<W>::z(a);
    let (word, bit) = (b as usize / 64, 1u64 << (b % 64));
    g.z[word] |= bit;
    g
}

/// The circuit's channels in order, as `(generator, angle)` pairs.
pub fn kicked_ising_gates(edges: &[(u32, u32)], n: usize, steps: usize, theta_h: f64) -> Vec<(PauliString<W>, f64)> {
    let theta_zz = -std::f64::consts::FRAC_PI_2;
    let zz_order: Vec<(u32, u32)> = edge_coloring(edges).into_iter().flatten().collect();
    let mut g = Vec::with_capacity(steps * (n + zz_order.len()));
    for _ in 0..steps {
        for q in 0..n as u32 {
            g.push((PauliString::<W>::x(q), theta_h));
        }
        for &(a, b) in &zz_order {
            g.push((zz(a, b), theta_zz));
        }
    }
    g
}

/// `heavy_hex_kicked_ising(n, steps, theta_h, -pi/2)`, `x-then-zz` order.
pub fn kicked_ising(edges: &[(u32, u32)], n: usize, steps: usize, theta_h: f64) -> Circuit<W> {
    let mut c = Circuit::<W>::new(n);
    for (g, theta) in kicked_ising_gates(edges, n, steps, theta_h) {
        c.push(PauliRotation::new(g, theta));
    }
    c
}

/// The same circuit as one single-channel `Circuit` per layer (for per-layer timing).
pub fn kicked_ising_layers(edges: &[(u32, u32)], n: usize, steps: usize, theta_h: f64) -> Vec<Circuit<W>> {
    kicked_ising_gates(edges, n, steps, theta_h)
        .into_iter()
        .map(|(g, theta)| {
            let mut c = Circuit::<W>::new(n);
            c.push(PauliRotation::new(g, theta));
            c
        })
        .collect()
}

/// A single-term `Z_q` observable.
pub fn z_observable(num_qubits: usize, q: u32) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, 1);
    acc.add_term(PauliString::<W>::z(q), Phase::ONE, Complex64::new(1.0, 0.0));
    acc.finalize()
}

/// The talk's circuit at `steps` Trotter steps.
pub fn talk_circuit(edges_path: &str, steps: usize) -> Circuit<W> {
    let edges = talk_edges(edges_path);
    let c = kicked_ising(&edges, QUBITS, steps, THETA_H);
    assert_eq!(c.len(), steps * (QUBITS + 144), "kicked-Ising channel count");
    c
}

pub fn talk_edges(edges_path: &str) -> Vec<(u32, u32)> {
    let edges = read_edges(edges_path);
    assert_eq!(edges.len(), 144, "expected the 144-edge Eagle map");
    edges
}

pub fn talk_layers(edges_path: &str, steps: usize) -> Vec<Circuit<W>> {
    kicked_ising_layers(&talk_edges(edges_path), QUBITS, steps, THETA_H)
}
