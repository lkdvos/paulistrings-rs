//! The layer menu: each cell's circuit and input sum.

use num_complex::Complex64;
use paulistrings::test_support::{haar_su4_matrix, low_weight_sum, rand_sum};
use paulistrings::{
    BuildAccumulator, Circuit, PauliRotation, PauliString, PauliSum, PtmChannel, TruncationPolicy,
};

use crate::cli::{Config, Initial, TruncSpec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LayerKind {
    RotationZz,
    /// A `ZZ` rotation whose deltas all stay inside their partition; the same cell as `RotationZz` at `P = 1`.
    RotationLocal,
    /// A `ZZ` rotation with at least one delta crossing partitions every layer.
    RotationRemote,
    Cnot,
    Gu2q,
    Su4,
    /// Haar SU(4) on a pair whose deltas are all local under the partition rows.
    Su4Local,
    Depolarizing,
    Trotter,
    /// One kicked-Ising Trotter step on an open chain; see [`tfim_step_circuit`].
    TfimStep,
    /// One kicked-Ising Trotter step on the 127-qubit heavy-hex lattice; see [`heavy_hex_step_circuit`].
    HeavyHexStep,
}

impl LayerKind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            LayerKind::RotationZz => "rotation_zz",
            LayerKind::RotationLocal => "rotation_local",
            LayerKind::RotationRemote => "rotation_remote",
            LayerKind::Cnot => "cnot",
            LayerKind::Gu2q => "gu2q",
            LayerKind::Su4 => "su4",
            LayerKind::Su4Local => "su4_local",
            LayerKind::Depolarizing => "depolarizing",
            LayerKind::Trotter => "trotter",
            LayerKind::TfimStep => "tfim_step",
            LayerKind::HeavyHexStep => "heavyhex_step",
        }
    }

    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        match s {
            "rotation_zz" => Ok(LayerKind::RotationZz),
            "rotation_local" => Ok(LayerKind::RotationLocal),
            "rotation_remote" => Ok(LayerKind::RotationRemote),
            "cnot" => Ok(LayerKind::Cnot),
            "gu2q" => Ok(LayerKind::Gu2q),
            "su4" => Ok(LayerKind::Su4),
            "su4_local" => Ok(LayerKind::Su4Local),
            "depolarizing" => Ok(LayerKind::Depolarizing),
            "trotter" => Ok(LayerKind::Trotter),
            "tfim_step" => Ok(LayerKind::TfimStep),
            "heavyhex_step" => Ok(LayerKind::HeavyHexStep),
            other => Err(format!(
                "unknown layer '{other}' (expected one of: rotation_zz, rotation_local, \
                 rotation_remote, cnot, gu2q, su4, su4_local, depolarizing, trotter, tfim_step, \
                 heavyhex_step)"
            )),
        }
    }

    /// The two Trotter-step workloads, whose input grows from a single-site `Z` observable.
    pub(crate) fn is_trotter_step(self) -> bool {
        matches!(self, LayerKind::TfimStep | LayerKind::HeavyHexStep)
    }

    /// The layer's two-qubit generator graph for `--partition-rows cut`; empty when it has none.
    pub(crate) fn cut_edges(self, num_qubits: usize) -> Vec<(usize, usize)> {
        match self {
            LayerKind::TfimStep => chain_edges(num_qubits),
            LayerKind::HeavyHexStep => heavy_hex_127_edges(),
            _ => Vec::new(),
        }
    }

    /// Whether the layer's generator qubits are chosen per cell ([`choose_generator`]) rather than fixed at `(0, 1)`.
    pub(crate) fn picks_generator(self) -> bool {
        matches!(
            self,
            LayerKind::RotationLocal | LayerKind::RotationRemote | LayerKind::Su4Local
        )
    }
}

pub(crate) fn default_layers() -> Vec<LayerKind> {
    vec![
        LayerKind::RotationZz,
        LayerKind::Cnot,
        LayerKind::Gu2q,
        LayerKind::Depolarizing,
        LayerKind::Trotter,
    ]
}

/// A weight-2 `ZZ` rotation, verbatim from `benches/pauli_ops.rs::zz_rotation`.
pub(crate) fn zz_rotation<const W: usize>(q0: usize, q1: usize, theta: f64) -> PauliRotation<W> {
    let mut gen = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    gen.z[q0 / 64] |= 1u64 << (q0 % 64);
    gen.z[q1 / 64] |= 1u64 << (q1 % 64);
    PauliRotation::new(gen, theta)
}

/// sqrt(SWAP) on `(q0, q1)`, verbatim from `benches/pauli_ops.rs::sqrt_swap`.
pub(crate) fn sqrt_swap(q0: usize, q1: usize) -> PtmChannel {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    PtmChannel::unitary_2q(
        q0,
        q1,
        [
            [one, zero, zero, zero],
            [zero, h, hc, zero],
            [zero, hc, h, zero],
            [zero, zero, zero, one],
        ],
    )
}

/// One fixed Haar-random SU(4) block on `(q0, q1)`, the probe's stand-in for the general matrix-gate path.
///
/// Unlike [`sqrt_swap`], a generic SU(4) keeps a dense PTM under repeated application rather than cycling.
pub(crate) fn haar_su4_block(q0: usize, q1: usize) -> PtmChannel {
    PtmChannel::unitary_2q(q0, q1, haar_su4_matrix())
}

/// Qubit count of the fixed [`trotter_circuit`] chain.
pub(crate) const TROTTER_QUBITS: usize = 32;

/// Safety cap on `trotter`'s own input size, overriding `--n`: 64 distinct generators under no
/// truncation grow combinatorially rather than closing to a bounded key set, so an uncapped `--n`
/// would try to materialize far too many terms.
pub(crate) const TROTTER_MAX_N: usize = 100;

/// The 64-channel TFIM Trotter step: 32 `ZZ` bond rotations (periodic boundary) then 32
/// transverse-field `X` rotations, fixed at [`TROTTER_QUBITS`] qubits regardless of `--qubits`.
pub(crate) fn trotter_circuit<const W: usize>() -> Circuit<W> {
    let num_qubits = TROTTER_QUBITS;
    let theta = 0.1;
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        circuit.push(zz_rotation::<W>(q, (q + 1) % num_qubits, 2.0 * theta));
    }
    for q in 0..num_qubits {
        let gen = PauliString::<W>::x(q);
        circuit.push(PauliRotation::new(gen, 2.0 * theta));
    }
    circuit
}

/// Qubits in the heavy-hex lattice [`heavy_hex_127_edges`] describes.
pub(crate) const HEAVY_HEX_QUBITS: usize = 127;

/// The `ZZ` angle of the kicked-Ising workload: `-pi/2`, a Clifford entangler.
pub(crate) const THETA_ZZ: f64 = -std::f64::consts::FRAC_PI_2;

/// The transverse-field kick angle of the same workload: `5·pi/16`, the non-Clifford point.
pub(crate) const THETA_H: f64 = 5.0 * std::f64::consts::PI / 16.0;

/// The bonds of a 1D **open** chain: `n - 1` edges `(i, i+1)`.
pub(crate) fn chain_edges(num_qubits: usize) -> Vec<(usize, usize)> {
    (0..num_qubits.saturating_sub(1))
        .map(|i| (i, i + 1))
        .collect()
}

/// The 127-qubit heavy-hex coupling map, from `test_support`.
pub(crate) fn heavy_hex_127_edges() -> Vec<(usize, usize)> {
    paulistrings::test_support::heavy_hex_127_edges()
}

/// Greedy first-fit edge coloring in sorted edge order; a color is a set of disjoint-support
/// edges, i.e. one hardware layer.
pub(crate) fn edge_coloring(edges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let n = edges.iter().map(|&(a, b)| a.max(b) + 1).max().unwrap_or(0);
    let mut used: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut classes: Vec<Vec<(usize, usize)>> = Vec::new();
    for &(a, b) in edges {
        let mut color = 0usize;
        while used[a].contains(&color) || used[b].contains(&color) {
            color += 1;
        }
        while classes.len() <= color {
            classes.push(Vec::new());
        }
        classes[color].push((a, b));
        used[a].push(color);
        used[b].push(color);
    }
    classes
}

/// A `PauliRotation` about `X_q`.
pub(crate) fn x_rotation<const W: usize>(q: usize, theta: f64) -> PauliRotation<W> {
    PauliRotation::new(PauliString::<W>::x(q), theta)
}

/// `steps` Trotter steps of the kicked transverse-field Ising model on an open chain of
/// `num_qubits` qubits, one channel per gate: the `ZZ` layer then the `X` layer, `2n - 1`
/// channels per step. Uses [`THETA_ZZ`] / [`THETA_H`]; not step-comparable with
/// [`heavy_hex_step_circuit`], whose `X`-then-`ZZ` order differs.
pub(crate) fn tfim_step_circuit<const W: usize>(num_qubits: usize, steps: usize) -> Circuit<W> {
    let mut c = Circuit::<W>::new(num_qubits);
    for _ in 0..steps {
        for (a, b) in chain_edges(num_qubits) {
            c.push(zz_rotation::<W>(a, b, THETA_ZZ));
        }
        for q in 0..num_qubits {
            c.push(x_rotation::<W>(q, THETA_H));
        }
    }
    c
}

/// `steps` Trotter steps of the 127-qubit heavy-hex kicked-Ising circuit, one channel per gate:
/// the `X` layer then the `ZZ` layer in [`edge_coloring`] order, 271 channels per step.
/// `num_qubits` must be at least [`HEAVY_HEX_QUBITS`]; anything above is a spectator.
pub(crate) fn heavy_hex_step_circuit<const W: usize>(
    num_qubits: usize,
    steps: usize,
) -> Circuit<W> {
    assert!(
        num_qubits >= HEAVY_HEX_QUBITS,
        "heavy_hex_step_circuit: the lattice needs {HEAVY_HEX_QUBITS} qubits, got {num_qubits}",
    );
    let zz_order: Vec<(usize, usize)> = edge_coloring(&heavy_hex_127_edges())
        .into_iter()
        .flatten()
        .collect();
    let mut c = Circuit::<W>::new(num_qubits);
    for _ in 0..steps {
        for q in 0..HEAVY_HEX_QUBITS {
            c.push(x_rotation::<W>(q, THETA_H));
        }
        for &(a, b) in &zz_order {
            c.push(zz_rotation::<W>(a, b, THETA_ZZ));
        }
    }
    c
}

/// The single-term observable `Z_q` on `num_qubits` qubits, coefficient 1.
pub(crate) fn z_observable<const W: usize>(num_qubits: usize, q: usize) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, 1);
    acc.add_term(PauliString::<W>::z(q), Complex64::new(1.0, 0.0));
    acc.finalize()
}

/// A truncation policy that never drops anything.
pub(crate) struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// [`AlwaysKeep`] for a partitioned cell.
pub(crate) struct AlwaysKeepPartitioned;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeepPartitioned {
    fn finalizes_layer(&self) -> bool {
        false
    }
}

/// Builds a cell's circuit. `gen_qubits` is the `(q0, q1)` pair the `rotation_*` layers rotate about.
pub(crate) fn build_circuit<const W: usize>(
    layer: LayerKind,
    qubits: usize,
    reps: usize,
    gen_qubits: (usize, usize),
) -> Circuit<W> {
    let theta = 0.1;
    match layer {
        LayerKind::RotationZz | LayerKind::RotationLocal | LayerKind::RotationRemote => {
            let (q0, q1) = gen_qubits;
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(zz_rotation::<W>(q0, q1, theta));
            }
            c
        }
        LayerKind::Cnot => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(PtmChannel::cnot(0, 1));
            }
            c
        }
        LayerKind::Gu2q => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(sqrt_swap(0, 1));
            }
            c
        }
        LayerKind::Su4 | LayerKind::Su4Local => {
            let (q0, q1) = gen_qubits;
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(haar_su4_block(q0, q1));
            }
            c
        }
        LayerKind::Depolarizing => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(PtmChannel::depolarizing(3, 0.05));
            }
            c
        }
        LayerKind::Trotter => trotter_circuit::<W>(),
        LayerKind::TfimStep => tfim_step_circuit::<W>(qubits, reps),
        LayerKind::HeavyHexStep => heavy_hex_step_circuit::<W>(qubits, reps),
    }
}

/// The cell's input sum, before any propagation: the seeded generator the layer asks for.
/// Shared by the unpartitioned and the partitioned path, so a `P = 1` and a `P = 2` cell
/// propagate the same terms.
pub(crate) fn build_base_sum<const W: usize>(layer: LayerKind, cfg: &Config) -> PauliSum<W> {
    if layer.is_trotter_step() && cfg.truncation == TruncSpec::Keep {
        eprintln!(
            "phase_breakdown: warning: {} under --truncation keep does not converge — a \
             kicked-Ising step is a fresh set of distinct generators every step, so the term \
             count grows without bound. Pass a coefficient threshold, e.g. \
             --truncation coeff:1.220703125e-4 (2^-13, the presentation's working point) or \
             --truncation coeff:3.90625e-3 (2^-8) for a quick run.",
            layer.name(),
        );
    }
    // `trotter` is 64 distinct generators, unlike the other layers' one generator repeated, so a
    // dense input can blow up combinatorially. It gets a low-weight input sized to its own fixed
    // qubit count instead of `--qubits`, and its own `--n` cap (see `TROTTER_MAX_N`).
    match layer {
        LayerKind::Trotter => {
            if cfg.n > TROTTER_MAX_N {
                eprintln!(
                    "phase_breakdown: note: trotter ignores --n above {TROTTER_MAX_N} \
                     (requested {}) — two chained, untruncated 64-layer passes grow \
                     combinatorially, and an uncapped --n has been measured driving this \
                     layer past tens of GB of RSS. Capping this cell's input to {TROTTER_MAX_N}.",
                    cfg.n,
                );
            }
            low_weight_sum::<W>(cfg.n.min(TROTTER_MAX_N), TROTTER_QUBITS, 3, cfg.seed)
        }
        // Everything else takes `--initial`, defaulting per layer.
        _ => match cfg.initial.unwrap_or_else(|| Initial::default_for(layer)) {
            Initial::Random => rand_sum::<W>(cfg.n, cfg.qubits, cfg.seed),
            Initial::Z0 => z_observable::<W>(cfg.qubits, cfg.qubits / 2),
        },
    }
}
