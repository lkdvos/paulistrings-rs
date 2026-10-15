//! The [`Channel`] trait, its output buffer, and the built-in gates and noise channels.

pub(crate) mod prepared;
pub(crate) mod ptm;
pub(crate) mod rotation;

pub use ptm::PtmChannel;
pub use rotation::PauliRotation;

pub(crate) use prepared::Prepare;

use crate::pauli_string::qubit_mask;
use crate::pauli_sum::hash::Gf2Hash;
use num_complex::Complex64;
use prepared::{PrepareToken, Prepared};

/// The terms one [`Channel::apply`] call emits, pushed with [`push`](Self::push).
pub struct OutputBuffer<const W: usize> {
    x: Vec<[u64; W]>,
    z: Vec<[u64; W]>,
    coeff: Vec<Complex64>,
}

impl<const W: usize> OutputBuffer<W> {
    /// Append one term.
    pub fn push(&mut self, x: [u64; W], z: [u64; W], coeff: Complex64) {
        self.x.push(x);
        self.z.push(z);
        self.coeff.push(coeff);
    }

    pub(crate) fn new() -> Self {
        Self {
            x: Vec::new(),
            z: Vec::new(),
            coeff: Vec::new(),
        }
    }

    /// Forget every term, keeping the storage.
    pub(crate) fn clear(&mut self) {
        self.x.clear();
        self.z.clear();
        self.coeff.clear();
    }

    /// The terms in push order.
    pub(crate) fn terms(
        &self,
    ) -> impl ExactSizeIterator<Item = ([u64; W], [u64; W], Complex64)> + '_ {
        self.x
            .iter()
            .zip(&self.z)
            .zip(&self.coeff)
            .map(|((&x, &z), &coeff)| (x, z, coeff))
    }
}

/// Pack qubit indices into a [`Channel::support`] bitmask; order and duplicates do not matter.
///
/// # Panics
///
/// If a qubit is not below `64 · W`.
pub fn support_mask<const W: usize>(qubits: &[usize]) -> [u64; W] {
    qubit_mask(qubits.iter().copied(), 64 * W)
}

/// `(word, bit, 1 << bit)` of qubit `q`.
fn qubit_loc(q: usize) -> (usize, usize, u64) {
    let word = q / 64;
    let bit = q % 64;
    (word, bit, 1u64 << bit)
}

/// The packed single-qubit Pauli index `x | (z << 1)` (`I=0, X=1, Z=2, Y=3`) at `(word, bit)`.
fn read_pauli<const W: usize>(x: &[u64; W], z: &[u64; W], word: usize, bit: usize) -> usize {
    let x_bit = (x[word] >> bit) & 1;
    let z_bit = (z[word] >> bit) & 1;
    (x_bit | (z_bit << 1)) as usize
}

/// Overwrite the qubit at `(word, bit, mask)` with the packed Pauli index `p`.
fn write_pauli<const W: usize>(
    x: &mut [u64; W],
    z: &mut [u64; W],
    word: usize,
    bit: usize,
    mask: u64,
    p: usize,
) {
    let x_bit = (p & 1) as u64;
    let z_bit = ((p >> 1) & 1) as u64;
    x[word] = (x[word] & !mask) | (x_bit << bit);
    z[word] = (z[word] & !mask) | (z_bit << bit);
}

/// A gate or noise channel: maps a Pauli string to a small weighted sum of Pauli strings.
///
/// Built-ins: [`PtmChannel`] (every one- and two-qubit gate and noise channel) and [`PauliRotation`].
/// A custom channel implements `support`, `apply` and `apply_adjoint`; the engine derives its per-layer form by probing `apply` on the at most 16 local basis Paulis.
pub trait Channel<const W: usize>: Send + Sync {
    /// Qubits this channel acts on, as a bitmask built with [`support_mask`].
    ///
    /// Outputs must differ from their input only at these qubits, with amplitudes that depend on the input only through its bits here; the engine's probe of `apply` is sound exactly then, and debug builds re-probe on an all-ones background to catch a violation.
    fn support(&self) -> [u64; W];

    /// Short name for the per-layer progress log.
    ///
    /// The default is the type name without path or generics, which is unhelpful for tuples, closures and boxes; override it there.
    fn debug_name(&self) -> &'static str {
        let full = core::any::type_name::<Self>();
        let head = match full.find('<') {
            Some(i) => &full[..i],
            None => full,
        };
        match head.rfind("::") {
            Some(i) => &head[i + 2..],
            None => head,
        }
    }

    /// Apply the channel to one input term, pushing its outputs to `out`.
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<W>,
    );

    /// Apply the channel's adjoint to one input term; run by `Direction::Heisenberg`.
    ///
    /// A self-adjoint channel calls [`apply`](Self::apply).
    fn apply_adjoint(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut OutputBuffer<W>,
    );

    /// The engine form for one layer: probed from `apply` unless a built-in knows its table.
    /// Neither callable nor overridable outside the crate, since `PrepareToken` and `Prepared` cannot be named there.
    #[doc(hidden)]
    fn prepare_for_layer(
        &self,
        hash: &Gf2Hash<W>,
        adjoint: bool,
        _: PrepareToken,
    ) -> Option<Prepared<W>> {
        Prepared::derive_local(self, hash, adjoint)
    }
}

#[cfg(test)]
mod tests;
