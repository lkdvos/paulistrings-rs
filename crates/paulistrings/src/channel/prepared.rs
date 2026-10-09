//! [`Prepared`], the per-layer engine form of a channel, and the crate-private [`Prepare`] that produces it.
//! See ARCHITECTURE.md §Prepared-Channels.

use num_complex::Complex64;

use super::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::phase::Phase;

/// Largest support handled by [`Prepared::Local`].
pub(crate) const MAX_LOCAL_SUPPORT: usize = 2;

/// `4^MAX_LOCAL_SUPPORT`, the number of local Pauli basis elements.
pub(crate) const LOCAL_DIM: usize = 16;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// The argument that keeps [`Channel::prepare_for_layer`] out of reach: it is `pub` only to appear in that signature, and only this module can build one.
pub struct PrepareToken(());

/// [`Channel::prepare_for_layer`] without the token, for the engine.
pub(crate) trait Prepare<const W: usize> {
    /// The channel's engine form for one layer; `None` makes `propagate` panic.
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>>;
}

impl<const W: usize, C: Channel<W> + ?Sized> Prepare<W> for C {
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>> {
        self.prepare_for_layer(hash, adjoint, PrepareToken(()))
    }
}

/// One key delta, with the amplitude it carries for each input support pattern.
#[derive(Clone, Debug)]
pub(crate) struct DeltaEntry<const W: usize> {
    /// `δ = H·d`; entries may share one.
    pub(crate) bucket_delta: u32,
    /// `d` in local support coordinates: bit `2j` is the x-bit of support qubit `j`, bit `2j+1` its z-bit.
    pub(crate) local_delta: u8,
    /// X-part of `d` as a full-width XOR mask.
    pub(crate) mask_x: [u64; W],
    /// Z-part of `d` as a full-width XOR mask.
    pub(crate) mask_z: [u64; W],
    /// `amplitude[s]` takes support pattern `s` to `s ^ local_delta`; exactly zero means no output.
    pub(crate) amplitude: [Complex64; LOCAL_DIM],
}

impl<const W: usize> DeltaEntry<W> {
    /// The output row for support pattern `s`, computed operation for operation as the gather does; `None` for a zero amplitude.
    pub(crate) fn emit(
        &self,
        s: usize,
        x: &[u64; W],
        z: &[u64; W],
        c: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        let a = self.amplitude[s];
        if a == ZERO {
            return None;
        }
        let mut key_x = *x;
        let mut key_z = *z;
        for w in 0..W {
            key_x[w] ^= self.mask_x[w];
            key_z[w] ^= self.mask_z[w];
        }
        Some((key_x, key_z, c * a))
    }

    /// `(mask_x, mask_z)`.
    pub(crate) fn mask(&self) -> ([u64; W], [u64; W]) {
        (self.mask_x, self.mask_z)
    }
}

/// A channel with support on at most [`MAX_LOCAL_SUPPORT`] qubits, as its local Pauli-transfer matrix grouped by key delta.
/// `pub` in this private module only because [`Prepared`] carries it.
#[derive(Clone, Debug)]
pub struct LocalPtm<const W: usize> {
    /// Support qubits, ascending; only the first `k` are meaningful.
    qubits: [u32; MAX_LOCAL_SUPPORT],
    /// Number of support qubits.
    k: u8,
    /// Ascending by `local_delta`, so the identity entry is first when present.
    deltas: Vec<DeltaEntry<W>>,
}

impl<const W: usize> LocalPtm<W> {
    /// Number of support qubits.
    pub(crate) fn k(&self) -> usize {
        self.k as usize
    }

    /// Support qubits, ascending.
    #[cfg(any(test, feature = "cuda"))]
    pub(crate) fn qubits(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.qubits[..self.k as usize].iter().map(|&q| q as usize)
    }

    /// The delta entries, ascending by `local_delta`.
    pub(crate) fn deltas(&self) -> &[DeltaEntry<W>] {
        &self.deltas
    }

    /// Number of key deltas.
    #[cfg(test)]
    pub(crate) fn num_deltas(&self) -> usize {
        self.deltas.len()
    }

    /// The distinct bucket deltas `h(D)`, ascending.
    pub(crate) fn bucket_deltas(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.deltas.iter().map(|d| d.bucket_delta).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The local support pattern of a key, packed like `local_delta`.
    pub(crate) fn support_bits(&self, x: &[u64; W], z: &[u64; W]) -> usize {
        let mut s = 0usize;
        for j in 0..self.k as usize {
            let q = self.qubits[j] as usize;
            let w = q / 64;
            let b = q % 64;
            s |= (((x[w] >> b) & 1) as usize) << (2 * j);
            s |= (((z[w] >> b) & 1) as usize) << (2 * j + 1);
        }
        s
    }

    /// Whether every key maps to itself, making the layer an in-place coefficient rescale.
    pub(crate) fn is_key_preserving(&self) -> bool {
        self.deltas.len() == 1 && self.deltas[0].local_delta == 0
    }

    /// A copy keeping the entries flagged in `keep`, one flag per entry.
    pub(crate) fn retain_entries(&self, keep: &[bool]) -> LocalPtm<W> {
        debug_assert_eq!(
            keep.len(),
            self.deltas.len(),
            "retain_entries: one flag per delta entry",
        );
        LocalPtm {
            qubits: self.qubits,
            k: self.k,
            deltas: self
                .deltas
                .iter()
                .zip(keep)
                .filter(|(_, &k)| k)
                .map(|(d, _)| d.clone())
                .collect(),
        }
    }

    /// The support qubits of `mask` with no entries yet; `None` above [`MAX_LOCAL_SUPPORT`].
    fn on_support(mask: &[u64; W]) -> Option<Self> {
        let k: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        if k > MAX_LOCAL_SUPPORT {
            return None;
        }
        let mut qubits = [0u32; MAX_LOCAL_SUPPORT];
        let mut n = 0usize;
        for (w, &word) in mask.iter().enumerate() {
            let mut live = word;
            while live != 0 {
                qubits[n] = (64 * w) as u32 + live.trailing_zeros();
                n += 1;
                live &= live - 1;
            }
        }
        Some(LocalPtm {
            qubits,
            k: k as u8,
            deltas: Vec::new(),
        })
    }

    /// One entry per local delta that `table` reaches, ascending.
    fn push_deltas(&mut self, table: &[[Complex64; LOCAL_DIM]; LOCAL_DIM], hash: &Gf2Hash<W>) {
        let dim = 1usize << (2 * self.k);
        let mut has_delta = [false; LOCAL_DIM];
        for (s, row) in table.iter().enumerate().take(dim) {
            for (t, &amplitude) in row.iter().enumerate().take(dim) {
                if amplitude != ZERO {
                    has_delta[s ^ t] = true;
                }
            }
        }
        for d in (0..dim).filter(|&d| has_delta[d]) {
            let mut amplitude = [ZERO; LOCAL_DIM];
            for (s, slot) in amplitude.iter_mut().enumerate().take(dim) {
                *slot = table[s][s ^ d];
            }
            let (mask_x, mask_z) = self.lift(d as u8);
            self.deltas.push(DeltaEntry {
                bucket_delta: hash.bucket_of(&mask_x, &mask_z),
                local_delta: d as u8,
                mask_x,
                mask_z,
                amplitude,
            });
        }
    }

    /// Lift a local delta to full-width XOR masks.
    fn lift(&self, local_delta: u8) -> ([u64; W], [u64; W]) {
        let mut mx = [0u64; W];
        let mut mz = [0u64; W];
        for j in 0..self.k as usize {
            let q = self.qubits[j] as usize;
            let w = q / 64;
            let bit = 1u64 << (q % 64);
            if (local_delta >> (2 * j)) & 1 == 1 {
                mx[w] |= bit;
            }
            if (local_delta >> (2 * j + 1)) & 1 == 1 {
                mz[w] |= bit;
            }
        }
        (mx, mz)
    }
}

/// A rotation with support wider than [`MAX_LOCAL_SUPPORT`], whose amplitudes are computed per term.
/// `pub` in this private module only because [`Prepared`] carries it.
#[derive(Clone, Debug)]
pub struct PreparedRotation<const W: usize> {
    /// The generator `P`.
    pub(crate) generator: PauliString<W>,
    /// `cos(θ)`.
    pub(crate) cos: f64,
    /// `sin(θ)`.
    pub(crate) sin: f64,
    /// Bucket delta of the identity output, `H·0 = 0`.
    pub(crate) bucket_delta_identity: u32,
    /// Bucket delta of the `v ⊕ P` output, `H·P`.
    pub(crate) bucket_delta_generator: u32,
}

impl<const W: usize> PreparedRotation<W> {
    /// The generator-pass row, computed operation for operation as the gather does; `None` if the term commutes with the generator.
    pub(crate) fn emit_generator(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        c: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        let v = PauliString::<W> { x: *x, z: *z };
        if v.commutes_with(&self.generator) {
            return None;
        }
        let mut product = v;
        let phase = product.mul_assign(&self.generator);
        let total = Phase::I + phase;
        Some((product.x, product.z, total.apply(c) * self.sin))
    }

    /// The generator as an XOR mask pair, the one non-identity key delta.
    pub(crate) fn generator_mask(&self) -> ([u64; W], [u64; W]) {
        (self.generator.x, self.generator.z)
    }
}

/// A channel prepared for one layer of the bucketed engine.
/// `pub` in this private module only because the hidden [`Channel::prepare_for_layer`] returns it.
#[derive(Clone, Debug)]
pub enum Prepared<const W: usize> {
    /// Support on at most `MAX_LOCAL_SUPPORT` qubits, tabulated.
    Local(LocalPtm<W>),
    /// A `PauliRotation` with wider support.
    Rotation(PreparedRotation<W>),
}

impl<const W: usize> Prepared<W> {
    /// The distinct bucket deltas `h(D)`.
    pub(crate) fn bucket_deltas(&self) -> Vec<u32> {
        match self {
            Prepared::Local(p) => p.bucket_deltas(),
            Prepared::Rotation(r) => {
                if r.bucket_delta_generator == r.bucket_delta_identity {
                    vec![r.bucket_delta_identity]
                } else {
                    vec![r.bucket_delta_identity, r.bucket_delta_generator]
                }
            }
        }
    }

    /// The XOR masks of the key deltas: every entry's for a table, the generator's for a rotation.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn key_delta_masks(&self) -> Vec<([u64; W], [u64; W])> {
        match self {
            Prepared::Local(ptm) => ptm.deltas().iter().map(|d| d.mask()).collect(),
            Prepared::Rotation(rotation) => vec![rotation.generator_mask()],
        }
    }

    /// [`Prepared::Local`] probed from [`Channel::apply`]; `None` when the support exceeds [`MAX_LOCAL_SUPPORT`] or an output leaves it.
    pub(crate) fn derive_local<C>(channel: &C, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Self>
    where
        C: Channel<W> + ?Sized,
    {
        let mut ptm = LocalPtm::on_support(&channel.support())?;
        let table = probe_table(channel, &ptm, adjoint, false)?;

        // The probe only sees zero background bits; a channel violating the bounded-support contract disagrees on an all-ones background.
        #[cfg(debug_assertions)]
        {
            let shadow = probe_table(channel, &ptm, adjoint, true);
            debug_assert!(
                shadow.as_ref() == Some(&table),
                "Channel::apply depends on bits outside its declared support; \
                 the bounded-support contract of Prepared::derive_local is violated",
            );
        }

        ptm.push_deltas(&table, hash);
        Some(Prepared::Local(ptm))
    }

    /// [`Prepared::Local`] from a known local PTM `table[s][t]` on the support `mask`, in ascending-qubit local indices.
    pub(crate) fn from_transfer_matrix(
        mask: &[u64; W],
        table: &[[Complex64; LOCAL_DIM]; LOCAL_DIM],
        hash: &Gf2Hash<W>,
    ) -> Self {
        let mut ptm = LocalPtm::on_support(mask).expect("a PTM support has at most two qubits");
        ptm.push_deltas(table, hash);
        Prepared::Local(ptm)
    }
}

/// The local PTM `table[s][t]` probed from `channel`, or `None` if an output leaves the support; `background` sets every non-support input bit.
fn probe_table<const W: usize, C>(
    channel: &C,
    ptm: &LocalPtm<W>,
    adjoint: bool,
    background: bool,
) -> Option<[[Complex64; LOCAL_DIM]; LOCAL_DIM]>
where
    C: Channel<W> + ?Sized,
{
    let k = ptm.k as usize;
    let dim = 1usize << (2 * k);
    let mut table = [[ZERO; LOCAL_DIM]; LOCAL_DIM];

    let (support_x, support_z) = {
        let mut mx = [0u64; W];
        let mut mz = [0u64; W];
        for j in 0..k {
            let q = ptm.qubits[j] as usize;
            let bit = 1u64 << (q % 64);
            mx[q / 64] |= bit;
            mz[q / 64] |= bit;
        }
        (mx, mz)
    };

    let mut out = OutputBuffer::<W>::new();

    for (s, row) in table.iter_mut().enumerate().take(dim) {
        let (mut in_x, mut in_z) = ptm.lift(s as u8);
        if background {
            for w in 0..W {
                in_x[w] |= !support_x[w];
                in_z[w] |= !support_z[w];
            }
        }

        out.clear();
        if adjoint {
            channel.apply_adjoint(&in_x, &in_z, Complex64::new(1.0, 0.0), &mut out);
        } else {
            channel.apply(&in_x, &in_z, Complex64::new(1.0, 0.0), &mut out);
        }

        for (out_x, out_z, coeff) in out.terms() {
            for w in 0..W {
                if (out_x[w] ^ in_x[w]) & !support_x[w] != 0
                    || (out_z[w] ^ in_z[w]) & !support_z[w] != 0
                {
                    return None;
                }
            }
            let t = ptm.support_bits(&out_x, &out_z);
            row[t] += coeff;
        }
    }

    Some(table)
}

#[cfg(test)]
mod tests;
