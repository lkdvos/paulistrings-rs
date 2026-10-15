//! [`PtmChannel`], every built-in one- and two-qubit gate and noise channel as a compact Pauli-transfer matrix (PTM).

mod clifford;
mod noise;
mod unitary;

use super::prepared::{PrepareToken, Prepared, LOCAL_DIM};
use super::{qubit_loc, read_pauli, support_mask, write_pauli, Channel, OutputBuffer};
use crate::pauli_sum::hash::Gf2Hash;
use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// A one- or two-qubit gate or noise channel, stored as its Pauli-transfer matrix.
///
/// Built by the named constructors: the Cliffords [`hadamard`](Self::hadamard), [`phase_gate`](Self::phase_gate), [`pauli_x`](Self::pauli_x), [`pauli_y`](Self::pauli_y), [`pauli_z`](Self::pauli_z), [`cnot`](Self::cnot), [`cz`](Self::cz), [`swap`](Self::swap);
/// the general unitaries [`unitary_1q`](Self::unitary_1q), [`unitary_2q`](Self::unitary_2q);
/// and the noise channels [`depolarizing`](Self::depolarizing), [`dephasing`](Self::dephasing), [`pauli_channel`](Self::pauli_channel), [`depolarizing_2q`](Self::depolarizing_2q), [`amplitude_damping`](Self::amplitude_damping).
///
/// On the support, Paulis are indexed `x | (z << 1)` per qubit (`I = 0, X = 1, Z = 2, Y = 3`), the constructor's first qubit in the low two bits.
/// [`Channel::apply`] maps input Pauli `s` to `Σ_t ptm[s][t] · P_t`, and [`Channel::apply_adjoint`] reads the transpose.
/// [`Channel::debug_name`] is the constructor's name (`"hadamard"`, `"cnot"`, …), with `_adjoint` appended after [`adjoint`](Self::adjoint).
#[derive(Clone, Debug)]
pub struct PtmChannel {
    /// Support qubits in constructor order; only the first `num_support_qubits` are meaningful.
    qubits: [u32; 2],
    num_support_qubits: u8,
    kind: PtmKind,
    /// Whether this is the [`adjoint`](Self::adjoint) of the constructed channel; [`Channel::debug_name`] appends `_adjoint`.
    adjoint: bool,
    storage: PtmStorage,
}

/// The constructor a [`PtmChannel`] came from, which [`Channel::debug_name`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PtmKind {
    Hadamard,
    PhaseGate,
    PauliX,
    PauliY,
    PauliZ,
    Cnot,
    Cz,
    Swap,
    Unitary1q,
    Unitary2q,
    Depolarizing,
    Dephasing,
    PauliChannel,
    Depolarizing2q,
    AmplitudeDamping,
}

impl PtmKind {
    /// The constructor's name, with `_adjoint` appended for the adjoint.
    fn name(self, adjoint: bool) -> &'static str {
        let [plain, adjoint_name] = match self {
            Self::Hadamard => ["hadamard", "hadamard_adjoint"],
            Self::PhaseGate => ["phase_gate", "phase_gate_adjoint"],
            Self::PauliX => ["pauli_x", "pauli_x_adjoint"],
            Self::PauliY => ["pauli_y", "pauli_y_adjoint"],
            Self::PauliZ => ["pauli_z", "pauli_z_adjoint"],
            Self::Cnot => ["cnot", "cnot_adjoint"],
            Self::Cz => ["cz", "cz_adjoint"],
            Self::Swap => ["swap", "swap_adjoint"],
            Self::Unitary1q => ["unitary_1q", "unitary_1q_adjoint"],
            Self::Unitary2q => ["unitary_2q", "unitary_2q_adjoint"],
            Self::Depolarizing => ["depolarizing", "depolarizing_adjoint"],
            Self::Dephasing => ["dephasing", "dephasing_adjoint"],
            Self::PauliChannel => ["pauli_channel", "pauli_channel_adjoint"],
            Self::Depolarizing2q => ["depolarizing_2q", "depolarizing_2q_adjoint"],
            Self::AmplitudeDamping => ["amplitude_damping", "amplitude_damping_adjoint"],
        };
        if adjoint {
            adjoint_name
        } else {
            plain
        }
    }
}

/// The nonzero PTM entries, in the cheapest form the constructor knows to be exact.
#[derive(Clone, Debug)]
enum PtmStorage {
    /// A Clifford: input `s` goes to `image[s]` with coefficient `-1` where bit `s` of `negated` is set, `+1` otherwise.
    SignedPermutation { image: [u8; LOCAL_DIM], negated: u16 },
    /// A Pauli-diagonal channel: input `s` keeps its key and is scaled by `scale[s]`.
    Diagonal { scale: Box<[f64]> },
    /// Everything else, ascending by `(input, output)`.
    Sparse { entries: Box<[SparseEntry]> },
}

/// One nonzero PTM entry: `amplitude` takes local Pauli `input` to `output`.
#[derive(Clone, Copy, Debug)]
struct SparseEntry {
    input: u8,
    output: u8,
    amplitude: Complex64,
}

impl PtmChannel {
    /// A channel on `qubits` (one or two, distinct) from its storage.
    fn new(qubits: &[usize], storage: PtmStorage, kind: PtmKind) -> Self {
        debug_assert!(qubits.len() == 1 || qubits.len() == 2);
        let narrow = |q: usize| u32::try_from(q).expect("PtmChannel: qubit index exceeds u32");
        let mut padded = [narrow(qubits[0]); 2];
        for (slot, &q) in padded.iter_mut().zip(qubits) {
            *slot = narrow(q);
        }
        Self {
            qubits: padded,
            num_support_qubits: qubits.len() as u8,
            kind,
            adjoint: false,
            storage,
        }
    }

    /// The support qubits, in constructor order.
    pub fn qubits(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.qubits[..self.num_support_qubits as usize].iter().map(|&q| q as usize)
    }

    /// The adjoint channel, whose PTM is this one's transpose.
    pub fn adjoint(&self) -> Self {
        let storage = match &self.storage {
            PtmStorage::SignedPermutation { image, negated } => {
                let mut inverse = [0u8; LOCAL_DIM];
                let mut inverse_negated = 0u16;
                for (s, &t) in image[..self.dimension()].iter().enumerate() {
                    let t = t as usize;
                    inverse[t] = s as u8;
                    inverse_negated |= ((negated >> s) & 1) << t;
                }
                PtmStorage::SignedPermutation { image: inverse, negated: inverse_negated }
            }
            PtmStorage::Diagonal { scale } => PtmStorage::Diagonal { scale: scale.clone() },
            PtmStorage::Sparse { entries } => {
                let mut transposed: Vec<((u8, u8), Complex64)> = entries
                    .iter()
                    .map(|entry| ((entry.output, entry.input), entry.amplitude))
                    .collect();
                transposed.sort_by_key(|&(position, _)| position);
                sparse_storage(transposed)
            }
        };
        Self { storage, adjoint: !self.adjoint, ..*self }
    }

    /// `4^k` for `k` support qubits.
    fn dimension(&self) -> usize {
        1 << (2 * self.num_support_qubits)
    }

    /// The local Pauli index of the input's restriction to the support.
    fn local_index<const W: usize>(&self, x: &[u64; W], z: &[u64; W]) -> usize {
        self.qubits()
            .enumerate()
            .map(|(j, q)| {
                let (word, bit, _) = qubit_loc(q);
                read_pauli(x, z, word, bit) << (2 * j)
            })
            .fold(0, |index, part| index | part)
    }

    /// The input with its support overwritten by local Pauli index `t`.
    fn with_local_index<const W: usize>(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        t: usize,
    ) -> ([u64; W], [u64; W]) {
        let mut new_x = *x;
        let mut new_z = *z;
        for (j, q) in self.qubits().enumerate() {
            let (word, bit, mask) = qubit_loc(q);
            write_pauli(&mut new_x, &mut new_z, word, bit, mask, (t >> (2 * j)) & 3);
        }
        (new_x, new_z)
    }

    /// Body of `apply` (`transpose == false`) and `apply_adjoint`.
    fn apply_rows<const W: usize>(
        &self,
        transpose: bool,
        x: &[u64; W],
        z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<W>,
    ) {
        let s = self.local_index(x, z);
        match &self.storage {
            PtmStorage::SignedPermutation { image, negated } => {
                let (source, t) = if transpose {
                    let source = image[..self.dimension()]
                        .iter()
                        .position(|&image| image as usize == s)
                        .expect("a signed permutation is a bijection");
                    (source, source)
                } else {
                    (s, image[s] as usize)
                };
                let (new_x, new_z) = self.with_local_index(x, z, t);
                let signed = if (negated >> source) & 1 == 1 { -coeff } else { coeff };
                out.push(new_x, new_z, signed);
            }
            PtmStorage::Diagonal { scale } => out.push(*x, *z, coeff * scale[s]),
            PtmStorage::Sparse { entries } => {
                for entry in entries.iter() {
                    let (from, to) = if transpose {
                        (entry.output, entry.input)
                    } else {
                        (entry.input, entry.output)
                    };
                    if from as usize == s {
                        let (new_x, new_z) = self.with_local_index(x, z, to as usize);
                        out.push(new_x, new_z, coeff * entry.amplitude);
                    }
                }
            }
        }
    }

    /// Calls `visit(s, t, amplitude)` for every stored entry.
    fn for_each_entry(&self, mut visit: impl FnMut(usize, usize, Complex64)) {
        match &self.storage {
            PtmStorage::SignedPermutation { image, negated } => {
                for (s, &t) in image[..self.dimension()].iter().enumerate() {
                    let amplitude = if (negated >> s) & 1 == 1 { -ONE } else { ONE };
                    visit(s, t as usize, amplitude);
                }
            }
            PtmStorage::Diagonal { scale } => {
                for (s, &value) in scale.iter().enumerate() {
                    visit(s, s, Complex64::new(value, 0.0));
                }
            }
            PtmStorage::Sparse { entries } => {
                for entry in entries.iter() {
                    visit(entry.input as usize, entry.output as usize, entry.amplitude);
                }
            }
        }
    }

    /// The dense `table[s][t]` (transposed when `adjoint`) in ascending-qubit local indices, the order [`Prepared`] uses.
    fn transfer_matrix(&self, adjoint: bool) -> [[Complex64; LOCAL_DIM]; LOCAL_DIM] {
        let reversed = self.num_support_qubits == 2 && self.qubits[0] > self.qubits[1];
        let reorder = |index: usize| {
            if reversed {
                ((index & 3) << 2) | (index >> 2)
            } else {
                index
            }
        };
        let mut table = [[ZERO; LOCAL_DIM]; LOCAL_DIM];
        self.for_each_entry(|s, t, amplitude| {
            let (row, column) = if adjoint { (t, s) } else { (s, t) };
            table[reorder(row)][reorder(column)] += amplitude;
        });
        table
    }
}

/// Sparse storage from `(position, amplitude)` pairs ascending by position, exact zeros dropped.
fn sparse_storage(entries: Vec<((u8, u8), Complex64)>) -> PtmStorage {
    PtmStorage::Sparse {
        entries: entries
            .into_iter()
            .filter(|&(_, amplitude)| amplitude != ZERO)
            .map(|((input, output), amplitude)| SparseEntry { input, output, amplitude })
            .collect(),
    }
}

/// Panics unless the two support qubits differ.
fn assert_distinct(constructor: &str, q0: usize, q1: usize) {
    assert!(q0 != q1, "PtmChannel::{constructor}: the two qubits must differ (both are {q0})");
}

/// Panics unless `value` lies in `[0, 1]`.
fn assert_probability(constructor: &str, name: &str, value: f64) {
    assert!(
        (0.0..=1.0).contains(&value),
        "PtmChannel::{constructor}: {name} must lie in [0, 1] (got {value})"
    );
}

impl<const W: usize> Channel<W> for PtmChannel {
    fn support(&self) -> [u64; W] {
        let mut qubits = [0usize; 2];
        for (slot, q) in qubits.iter_mut().zip(self.qubits()) {
            *slot = q;
        }
        support_mask(&qubits[..self.num_support_qubits as usize])
    }

    fn debug_name(&self) -> &'static str {
        self.kind.name(self.adjoint)
    }

    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<W>,
    ) {
        self.apply_rows(false, input_x, input_z, coeff, out);
    }

    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<W>,
    ) {
        self.apply_rows(true, input_x, input_z, coeff, out);
    }

    /// Tabulated straight from the stored PTM, without probing `apply`; a test holds it equal to the probe's table.
    fn prepare_for_layer(
        &self,
        hash: &Gf2Hash<W>,
        adjoint: bool,
        _: PrepareToken,
    ) -> Option<Prepared<W>> {
        Some(Prepared::from_transfer_matrix(&self.support(), &self.transfer_matrix(adjoint), hash))
    }
}

#[cfg(test)]
mod tests;
