//! [`Prepared`], the per-layer engine form of a channel, and the crate-private [`Prepare`] that produces it.

use num_complex::Complex64;

use crate::coefficient::{Coefficient, REAL_TOLERANCE};

use super::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::phase::Phase;

/// Largest support handled by [`Prepared::Local`].
pub(crate) const MAX_LOCAL_SUPPORT: usize = 2;

/// `4^MAX_LOCAL_SUPPORT`, the number of local Pauli basis elements.
pub(crate) const LOCAL_DIM: usize = 16;

/// The argument that keeps [`Channel::prepare_for_layer`] out of reach: it is `pub` only to appear in that signature, and only this module can build one.
pub struct PrepareToken(());

/// [`Channel::prepare_for_layer`] without the token, for the engine.
pub(crate) trait Prepare<const W: usize> {
    /// The channel's engine form for one layer; `None` makes `propagate` panic.
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>>;
}

impl<const W: usize, Ch: Channel<W> + ?Sized> Prepare<W> for Ch {
    fn prepare(&self, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Prepared<W>> {
        self.prepare_for_layer(hash, adjoint, PrepareToken(()))
    }
}

/// One key delta, with the amplitude it carries for each input support pattern.
#[derive(Clone, Debug)]
pub(crate) struct DeltaEntry<const W: usize, C = Complex64> {
    /// `δ = H·d`; entries may share one.
    pub(crate) bucket_delta: u32,
    /// `d` in local support coordinates: bit `2j` is the x-bit of support qubit `j`, bit `2j+1` its z-bit.
    pub(crate) local_delta: u8,
    /// X-part of `d` as a full-width XOR mask.
    pub(crate) mask_x: [u64; W],
    /// Z-part of `d` as a full-width XOR mask.
    pub(crate) mask_z: [u64; W],
    /// `amplitude[s]` takes support pattern `s` to `s ^ local_delta`; exactly zero means no output.
    pub(crate) amplitude: [C; LOCAL_DIM],
}

impl<const W: usize, C: Coefficient> DeltaEntry<W, C> {
    /// The output row for support pattern `s`, computed operation for operation as the gather does; `None` for a zero amplitude.
    pub(crate) fn emit(
        &self,
        s: usize,
        x: &[u64; W],
        z: &[u64; W],
        c: C,
    ) -> Option<([u64; W], [u64; W], C)> {
        let a = self.amplitude[s];
        if a == C::ZERO {
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
pub struct LocalPtm<const W: usize, C = Complex64> {
    /// Support qubits, ascending; only the first `k` are meaningful.
    qubits: [u32; MAX_LOCAL_SUPPORT],
    /// Number of support qubits.
    k: u8,
    /// Ascending by `local_delta`, so the identity entry is first when present.
    deltas: Vec<DeltaEntry<W, C>>,
}

impl<const W: usize, C: Coefficient> LocalPtm<W, C> {
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
    pub(crate) fn deltas(&self) -> &[DeltaEntry<W, C>] {
        &self.deltas
    }

    /// Whether entry `index` has a nonzero amplitude on every support pattern; `amplitude` is sized `LOCAL_DIM` but only `4^k` entries are populated.
    pub(crate) fn entry_is_dense(&self, index: usize) -> bool {
        let dim = 1usize << (2 * self.k());
        self.deltas[index].amplitude[..dim].iter().all(|a| *a != C::ZERO)
    }

    /// Entries with a nonzero amplitude on some support pattern: the rows-per-term fanout a device block is sized by.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn emitting_entries(&self) -> usize {
        self.deltas.iter().filter(|d| d.amplitude.iter().any(|a| *a != C::ZERO)).count()
    }

    /// Whether the first entry is the identity delta.
    pub(crate) fn has_identity(&self) -> bool {
        self.deltas.first().is_some_and(|d| d.local_delta == 0)
    }

    /// Non-identity realized deltas: the rest streams the sort-kernel gate counts.
    pub(crate) fn rest_streams(&self) -> usize {
        self.deltas.len() - usize::from(self.has_identity())
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
    pub(crate) fn retain_entries(&self, keep: &[bool]) -> LocalPtm<W, C> {
        debug_assert_eq!(keep.len(), self.deltas.len(), "retain_entries: one flag per delta entry",);
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

    /// This table with every amplitude converted to `T`; `None` when a real `T` meets an imaginary amplitude.
    fn convert<T: Coefficient>(&self) -> Option<LocalPtm<W, T>> {
        let deltas = self
            .deltas
            .iter()
            .map(|d| {
                let mut amplitude = [T::ZERO; LOCAL_DIM];
                for (slot, &a) in amplitude.iter_mut().zip(&d.amplitude) {
                    *slot = T::from_complex64(a.to_complex64(), REAL_TOLERANCE)?;
                }
                Some(DeltaEntry {
                    bucket_delta: d.bucket_delta,
                    local_delta: d.local_delta,
                    mask_x: d.mask_x,
                    mask_z: d.mask_z,
                    amplitude,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(LocalPtm { qubits: self.qubits, k: self.k, deltas })
    }
}

impl<const W: usize> LocalPtm<W> {
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
        Some(LocalPtm { qubits, k: k as u8, deltas: Vec::new() })
    }

    /// One entry per local delta that `table` reaches, ascending.
    fn push_deltas(&mut self, table: &[[Complex64; LOCAL_DIM]; LOCAL_DIM], hash: &Gf2Hash<W>) {
        let dim = 1usize << (2 * self.k);
        let mut has_delta = [false; LOCAL_DIM];
        for (s, row) in table.iter().enumerate().take(dim) {
            for (t, &amplitude) in row.iter().enumerate().take(dim) {
                if amplitude != Complex64::ZERO {
                    has_delta[s ^ t] = true;
                }
            }
        }
        for d in (0..dim).filter(|&d| has_delta[d]) {
            let mut amplitude = [Complex64::ZERO; LOCAL_DIM];
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
pub struct PreparedRotation<const W: usize, R = f64> {
    /// The generator `P`.
    pub(crate) generator: PauliString<W>,
    /// `cos(θ)`.
    pub(crate) cos: R,
    /// `sin(θ)`.
    pub(crate) sin: R,
    /// Bucket delta of the identity output, `H·0 = 0`.
    pub(crate) bucket_delta_identity: u32,
    /// Bucket delta of the `v ⊕ P` output, `H·P`.
    pub(crate) bucket_delta_generator: u32,
}

impl<const W: usize, R: Copy> PreparedRotation<W, R> {
    /// The generator-pass row, computed operation for operation as the gather does; `None` if the term commutes with the generator.
    pub(crate) fn emit_generator<C: Coefficient<Real = R>>(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        c: C,
    ) -> Option<([u64; W], [u64; W], C)> {
        let v = PauliString::<W> { x: *x, z: *z };
        if v.commutes_with(&self.generator) {
            return None;
        }
        let mut product = v;
        let phase = product.mul_assign(&self.generator);
        let total = Phase::I + phase;
        Some((product.x, product.z, c.mul_phase(total) * self.sin))
    }

    /// The generator as an XOR mask pair, the one non-identity key delta.
    pub(crate) fn generator_mask(&self) -> ([u64; W], [u64; W]) {
        (self.generator.x, self.generator.z)
    }
}

/// A channel prepared for one layer of the bucketed engine.
/// `pub` in this private module only because the hidden [`Channel::prepare_for_layer`] returns it.
#[derive(Clone, Debug)]
pub enum Prepared<const W: usize, C: Coefficient = Complex64> {
    /// Support on at most `MAX_LOCAL_SUPPORT` qubits, tabulated.
    Local(LocalPtm<W, C>),
    /// A `PauliRotation` with wider support.
    Rotation(PreparedRotation<W, C::Real>),
}

impl<const W: usize, C: Coefficient> Prepared<W, C> {
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
}

/// A layer's channel whose transfer table has an imaginary amplitude, which a real coefficient type cannot carry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImaginaryAmplitude;

impl<const W: usize> Prepared<W> {
    /// This form with amplitudes in the coefficient type `T`.
    ///
    /// # Errors
    ///
    /// [`ImaginaryAmplitude`] when `T` is real and an amplitude's imaginary part exceeds [`REAL_TOLERANCE`]; a rotation is always real, since its `i` meets an odd product phase.
    pub(crate) fn convert<T: Coefficient>(&self) -> Result<Prepared<W, T>, ImaginaryAmplitude> {
        Ok(match self {
            Prepared::Local(ptm) => Prepared::Local(ptm.convert().ok_or(ImaginaryAmplitude)?),
            Prepared::Rotation(r) => Prepared::Rotation(PreparedRotation {
                generator: r.generator,
                cos: T::real(r.cos),
                sin: T::real(r.sin),
                bucket_delta_identity: r.bucket_delta_identity,
                bucket_delta_generator: r.bucket_delta_generator,
            }),
        })
    }

    /// [`Prepared::Local`] probed from [`Channel::apply`]; `None` when the support exceeds [`MAX_LOCAL_SUPPORT`] or an output leaves it.
    pub(crate) fn derive_local<Ch>(channel: &Ch, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Self>
    where
        Ch: Channel<W> + ?Sized,
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
fn probe_table<const W: usize, Ch>(
    channel: &Ch,
    ptm: &LocalPtm<W>,
    adjoint: bool,
    background: bool,
) -> Option<[[Complex64; LOCAL_DIM]; LOCAL_DIM]>
where
    Ch: Channel<W> + ?Sized,
{
    let k = ptm.k as usize;
    let dim = 1usize << (2 * k);
    let mut table = [[Complex64::ZERO; LOCAL_DIM]; LOCAL_DIM];

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
