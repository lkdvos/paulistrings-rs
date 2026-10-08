//! [`Prepared`], the per-layer engine form of a channel. See ARCHITECTURE.md §Prepared-Channels.

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

/// One key delta, with the amplitude it carries for each input support pattern.
#[derive(Clone, Debug)]
pub struct DeltaEntry<const W: usize> {
    /// `δ = H·d`; entries may share one.
    pub bucket_delta: u32,
    /// `d` in local support coordinates: bit `2j` is the x-bit of support qubit `j`, bit `2j+1` its z-bit.
    pub local_delta: u8,
    /// X-part of `d` as a full-width XOR mask.
    pub mask_x: [u64; W],
    /// Z-part of `d` as a full-width XOR mask.
    pub mask_z: [u64; W],
    /// `amp[s]` takes support pattern `s` to `s ^ local_delta`; exactly zero means no output.
    pub amp: [Complex64; LOCAL_DIM],
}

impl<const W: usize> DeltaEntry<W> {
    /// The output row for support pattern `s`, computed operation for operation as the gather does; `None` for a zero amplitude.
    #[inline]
    pub(crate) fn emit(
        &self,
        s: usize,
        x: &[u64; W],
        z: &[u64; W],
        c: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        let a = self.amp[s];
        if a == ZERO {
            return None;
        }
        let mut kx = *x;
        let mut kz = *z;
        for w in 0..W {
            kx[w] ^= self.mask_x[w];
            kz[w] ^= self.mask_z[w];
        }
        Some((kx, kz, c * a))
    }

    /// `(mask_x, mask_z)`.
    #[inline]
    pub(crate) fn mask(&self) -> ([u64; W], [u64; W]) {
        (self.mask_x, self.mask_z)
    }
}

/// A channel with support on at most [`MAX_LOCAL_SUPPORT`] qubits, as its local Pauli-transfer matrix grouped by key delta.
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
    #[inline]
    pub fn k(&self) -> usize {
        self.k as usize
    }

    /// Support qubits, ascending.
    #[inline]
    pub fn qubits(&self) -> &[u32] {
        &self.qubits[..self.k as usize]
    }

    /// The delta entries, ascending by `local_delta`.
    #[inline]
    pub fn deltas(&self) -> &[DeltaEntry<W>] {
        &self.deltas
    }

    /// Number of key deltas.
    #[inline]
    pub fn num_deltas(&self) -> usize {
        self.deltas.len()
    }

    /// The distinct bucket deltas `h(D)`, ascending.
    pub fn bucket_deltas(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.deltas.iter().map(|d| d.bucket_delta).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The local support pattern of a key, packed like `local_delta`.
    #[inline]
    pub fn support_bits(&self, x: &[u64; W], z: &[u64; W]) -> usize {
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
    pub fn is_key_preserving(&self) -> bool {
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
#[derive(Clone, Debug)]
pub struct RotationPrep<const W: usize> {
    /// The generator `P`.
    pub gen: PauliString<W>,
    /// `cos(θ)`.
    pub cos: f64,
    /// `sin(θ)`.
    pub sin: f64,
    /// Bucket delta of the identity output, `H·0 = 0`.
    pub bucket_delta_identity: u32,
    /// Bucket delta of the `v ⊕ P` output, `H·P`.
    pub bucket_delta_gen: u32,
}

impl<const W: usize> RotationPrep<W> {
    /// The generator-pass row, computed operation for operation as the gather does; `None` if the term commutes with the generator.
    #[inline]
    pub(crate) fn emit_gen(
        &self,
        x: &[u64; W],
        z: &[u64; W],
        c: Complex64,
    ) -> Option<([u64; W], [u64; W], Complex64)> {
        let v = PauliString::<W> { x: *x, z: *z };
        if v.commutes_with(&self.gen) {
            return None;
        }
        let mut prod = v;
        let phase = prod.mul_assign(&self.gen);
        let total = Phase::I + phase;
        Some((prod.x, prod.z, total.apply(c) * self.sin))
    }

    /// The generator as an XOR mask pair, the one non-identity key delta.
    pub(crate) fn gen_mask(&self) -> ([u64; W], [u64; W]) {
        (self.gen.x, self.gen.z)
    }
}

/// A channel prepared for one layer of the bucketed engine.
#[derive(Clone, Debug)]
pub enum Prepared<const W: usize> {
    /// Support on at most `MAX_LOCAL_SUPPORT` qubits, tabulated.
    Local(LocalPtm<W>),
    /// A `PauliRotation` with wider support.
    Rotation(RotationPrep<W>),
}

impl<const W: usize> Prepared<W> {
    /// The distinct bucket deltas `h(D)`.
    pub fn bucket_deltas(&self) -> Vec<u32> {
        match self {
            Prepared::Local(p) => p.bucket_deltas(),
            Prepared::Rotation(r) => {
                if r.bucket_delta_gen == r.bucket_delta_identity {
                    vec![r.bucket_delta_identity]
                } else {
                    vec![r.bucket_delta_identity, r.bucket_delta_gen]
                }
            }
        }
    }

    /// [`Prepared::Local`] probed from [`Channel::apply`]; `None` when the support exceeds [`MAX_LOCAL_SUPPORT`] or an output leaves it.
    pub fn derive_local<C>(channel: &C, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Self>
    where
        C: Channel<W> + ?Sized,
    {
        let mask = channel.support();
        let k: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        if k > MAX_LOCAL_SUPPORT {
            return None;
        }

        let mut qubits = [0u32; MAX_LOCAL_SUPPORT];
        let mut n = 0usize;
        for (w, &word) in mask.iter().enumerate() {
            let mut live = word;
            while live != 0 {
                let bit = live.trailing_zeros();
                qubits[n] = (64 * w) as u32 + bit;
                n += 1;
                live &= live - 1;
            }
        }
        debug_assert_eq!(n, k);

        let mut ptm = LocalPtm {
            qubits,
            k: k as u8,
            deltas: Vec::new(),
        };

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

        let dim = 1usize << (2 * k);
        let mut has_delta = [false; LOCAL_DIM];
        for s in 0..dim {
            for t in 0..dim {
                if table[s][t] != ZERO {
                    has_delta[s ^ t] = true;
                }
            }
        }

        for d in 0..dim {
            if !has_delta[d] {
                continue;
            }
            let mut amp = [ZERO; LOCAL_DIM];
            for s in 0..dim {
                amp[s] = table[s][s ^ d];
            }
            let (mask_x, mask_z) = ptm.lift(d as u8);
            ptm.deltas.push(DeltaEntry {
                bucket_delta: hash.bucket_of(&mask_x, &mask_z),
                local_delta: d as u8,
                mask_x,
                mask_z,
                amp,
            });
        }

        Some(Prepared::Local(ptm))
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
    let fanout = channel.max_fanout().max(1);

    let mut table = [[ZERO; LOCAL_DIM]; LOCAL_DIM];

    let (sup_x, sup_z) = {
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

    let mut buf_x = vec![[0u64; W]; fanout];
    let mut buf_z = vec![[0u64; W]; fanout];
    let mut buf_c = vec![ZERO; fanout];

    for (s, row) in table.iter_mut().enumerate().take(dim) {
        let (mut in_x, mut in_z) = ptm.lift(s as u8);
        if background {
            for w in 0..W {
                in_x[w] |= !sup_x[w];
                in_z[w] |= !sup_z[w];
            }
        }

        let mut len = 0usize;
        {
            let mut out = OutputBuffer::<W> {
                x: &mut buf_x,
                z: &mut buf_z,
                coeff: &mut buf_c,
                len: &mut len,
            };
            if adjoint {
                channel.apply_adjoint(&in_x, &in_z, Complex64::new(1.0, 0.0), &mut out);
            } else {
                channel.apply(&in_x, &in_z, Complex64::new(1.0, 0.0), &mut out);
            }
        }

        for i in 0..len {
            for w in 0..W {
                if (buf_x[i][w] ^ in_x[w]) & !sup_x[w] != 0
                    || (buf_z[i][w] ^ in_z[w]) & !sup_z[w] != 0
                {
                    return None;
                }
            }
            let t = ptm.support_bits(&buf_x[i], &buf_z[i]);
            row[t] += buf_c[i];
        }
    }

    Some(table)
}

#[cfg(test)]
mod tests;
