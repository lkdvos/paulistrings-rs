//! Per-layer prepared form of a channel. See ARCHITECTURE.md §Prepared-Channels.
//!
//! Prepares a channel once per layer into a table, reducing the inner loop to one lookup on ≤ 4 extracted bits, one XOR with a precomputed mask, and one complex multiply — instead of redoing the channel's setup (e.g. `theta.cos()/sin()`) per term through a vtable call.
//! The prepared form also carries, for each key delta `d`, the bucket delta `δ = H·d` the engine needs to know which buckets to read.

use num_complex::Complex64;

use super::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::phase::Phase;

/// Largest support size handled by [`Prepared::Local`].
///
/// The dense local Pauli-transfer matrix is `4^k × 4^k`, so `k = 2` is 4 KB of `Complex64` per layer; `k = 3` would be 64 KB, too large to be worth building per layer.
pub const MAX_LOCAL_SUPPORT: usize = 2;

/// `4^MAX_LOCAL_SUPPORT` — the number of local Pauli basis elements.
pub const LOCAL_DIM: usize = 16;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// One key delta, with the amplitude it carries for each input support pattern.
#[derive(Clone, Debug)]
pub struct DeltaEntry<const W: usize> {
    /// `δ = H·d`. Output bucket `β'` reads input bucket `β' ^ bucket_delta` for this delta.
    /// Several entries may share a `bucket_delta` (a hash collision); it costs a wasted read, never correctness.
    pub bucket_delta: u32,
    /// The delta in local support coordinates: bit `2j` is the x-bit of support qubit `j`, bit `2j+1` its z-bit.
    /// The canonical ordering key, since unlike `bucket_delta` it does not depend on the bucket count.
    pub local_delta: u8,
    /// `d` lifted to a full-width XOR mask.
    pub mask_x: [u64; W],
    /// `d` lifted to a full-width XOR mask.
    pub mask_z: [u64; W],
    /// `amp[s]` is the amplitude taking input support pattern `s` to
    /// `s ^ local_delta`. Exactly zero means "no output for this `s`".
    pub amp: [Complex64; LOCAL_DIM],
}

impl<const W: usize> DeltaEntry<W> {
    /// The output row this entry emits for a term with support pattern `s`, or `None` when the amplitude is exactly zero.
    ///
    /// The row-level form of the engine's gather inner loop (`engine::bucketed::gather_local_input_major`): same lookup, same mask XOR, same multiply, in the same order, so the partitioned export pass produces bitwise the rows a local gather would have.
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

    /// The entry's key delta as a full-width XOR mask pair, `(mask_x, mask_z)`.
    ///
    /// What a partitioning reads to classify the entry: `part(mask)` is the partition delta, `h(mask)` the (already stored) [`bucket_delta`](Self::bucket_delta).
    #[inline]
    pub(crate) fn mask(&self) -> ([u64; W], [u64; W]) {
        (self.mask_x, self.mask_z)
    }
}

/// A channel with support on at most [`MAX_LOCAL_SUPPORT`] qubits, as a dense
/// local Pauli-transfer matrix grouped by bucket delta.
#[derive(Clone, Debug)]
pub struct LocalPtm<const W: usize> {
    /// Support qubits, ascending. Only the first `k` are meaningful.
    qubits: [u32; MAX_LOCAL_SUPPORT],
    /// Number of support qubits, `0 ≤ k ≤ MAX_LOCAL_SUPPORT`.
    k: u8,
    /// The delta set, **ascending by `local_delta`** — the canonical construction order (index 0 is the identity entry when present). Does not define summation order; see ARCHITECTURE.md §Determinism.
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

    /// The delta set, ascending by `local_delta` — the canonical construction
    /// order (see the field doc).
    #[inline]
    pub fn deltas(&self) -> &[DeltaEntry<W>] {
        &self.deltas
    }

    /// Number of distinct key deltas, i.e. `|D|`.
    #[inline]
    pub fn num_deltas(&self) -> usize {
        self.deltas.len()
    }

    /// The distinct bucket deltas `h(D)`, ascending. Length is
    /// `2^rank(H|_D)` — the number of input buckets each output bucket reads.
    pub fn bucket_deltas(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.deltas.iter().map(|d| d.bucket_delta).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Extract the local support pattern `s` of a key.
    ///
    /// Bit `2j` is the x-bit of support qubit `j`, bit `2j+1` its z-bit — the
    /// same packing `Clifford2Q` uses (`idx = x0 | z0<<1 | x1<<2 | z1<<3`).
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

    /// `true` if this channel leaves every key bitwise unchanged, so a layer is
    /// an in-place coefficient rescale: no gather, no sort, no merge.
    ///
    /// Covers `IdentityChannel`, `Depolarizing`, `Dephasing` and
    /// `Clifford1Q::{x, y, z}`.
    pub fn is_key_preserving(&self) -> bool {
        self.deltas.len() == 1 && self.deltas[0].local_delta == 0
    }

    /// A copy keeping only the entries with `keep[e] == true`, in order, with the same support qubits and `k`.
    ///
    /// Used at prepare time to split a partitioned layer's table into a local-only one, so the gather loops keep running over a plain `deltas()` slice with no per-entry predicate in the inner loop.
    /// The result is still ascending by `local_delta`, a subsequence of an ascending sequence.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `keep` is not one flag per entry.
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

/// A rotation with support wider than [`MAX_LOCAL_SUPPORT`].
///
/// The delta set is `{0, P}` for any generator weight, so only two buckets are ever read — but the amplitude's `i^k` phase depends on `2w` support bits, which stops being tabulable, so amplitudes are computed per term instead, with `cos`/`sin` hoisted out of the loop.
#[derive(Clone, Debug)]
pub struct RotationPrep<const W: usize> {
    /// The generator `P`.
    pub gen: PauliString<W>,
    /// `cos(θ)`, hoisted.
    pub cos: f64,
    /// `sin(θ)`, hoisted.
    pub sin: f64,
    /// `H·0 = 0`: the bucket delta for the identity output.
    pub bucket_delta_identity: u32,
    /// `H·P`: the bucket delta for the `v ⊕ P` output.
    pub bucket_delta_gen: u32,
}

impl<const W: usize> RotationPrep<W> {
    /// The generator-pass row for a term, or `None` if it commutes with the generator (in which case the rotation leaves the term alone and only the identity pass emits).
    ///
    /// Copied verbatim from the `DeltaPlan::Rotation` arm of `engine::bucketed::fill_coset` so it stays bitwise-equal to the gather: `mul_assign` returns the `i^k` of the Pauli product, the leading `i` of `i · Q · P` folds in as `Phase::I + phase`, applied to the coefficient before the hoisted `sin` multiplies last.
    ///
    /// The identity pass has no emitter: every term contributes one identity row unconditionally, so there is nothing to decide per row.
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

    /// The generator-pass key delta as a full-width XOR mask pair — i.e. the
    /// generator itself, since the delta set is `{0, P}`.
    pub(crate) fn gen_mask(&self) -> ([u64; W], [u64; W]) {
        (self.gen.x, self.gen.z)
    }
}

/// A channel prepared for one layer of the bucketed engine.
#[derive(Clone, Debug)]
pub enum Prepared<const W: usize> {
    /// Amplitudes depend only on ≤ 4 support bits, so they are tabulated.
    /// Covers `Clifford1Q`/`Clifford2Q`, all noise channels,
    /// `GeneralUnitary1Q`/`2Q`, and `PauliRotation` at generator weight ≤ 2.
    Local(LocalPtm<W>),
    /// `PauliRotation` at generator weight > 2.
    Rotation(RotationPrep<W>),
}

impl<const W: usize> Prepared<W> {
    /// Bucket deltas this channel can produce, i.e. `h(D)`.
    ///
    /// Output bucket `β'` reads input buckets `β' ^ δ` for each `δ` here. The
    /// length is `2^rank(H|_D)` — 1, 2, 4 or 16 for the built-ins.
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

    /// Derive the prepared form of a bounded-support channel by probing its own [`Channel::apply`].
    ///
    /// Returns `None` when the channel's support is wider than [`MAX_LOCAL_SUPPORT`], or when it writes outside its declared support; in both cases `propagate` panics rather than proceeding with an unsound preparation (see ARCHITECTURE.md §Prepared-Channels).
    ///
    /// # The soundness precondition
    ///
    /// Exact iff the channel honours the bounded-support contract: the output amplitude may depend on the input only through its support bits.
    /// Probing cannot fully verify that — a channel reading qubit 5 while declaring support `[0]` would produce a table wrong for inputs never tried — so debug builds re-derive with an all-ones background outside the support and assert the two tables agree, and a property test checks every built-in against `apply` on randomized full-width inputs.
    pub fn derive_local<C>(channel: &C, hash: &Gf2Hash<W>, adjoint: bool) -> Option<Self>
    where
        C: Channel<W> + ?Sized,
    {
        let mask = channel.support();
        // Popcount first, and bail before materializing anything, so a wide
        // support never pays for qubit extraction it will just discard.
        let k: usize = mask.iter().map(|w| w.count_ones() as usize).sum();
        if k > MAX_LOCAL_SUPPORT {
            return None;
        }

        // Extract qubit indices ascending via per-word `trailing_zeros`. A
        // bitmask is already a set, so this is automatically sorted and
        // duplicate-free.
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

        #[cfg(debug_assertions)]
        {
            // Same table, but with every non-support bit set. A channel that
            // reads outside its support will disagree.
            let shadow = probe_table(channel, &ptm, adjoint, true);
            debug_assert!(
                shadow.as_ref() == Some(&table),
                "Channel::apply depends on bits outside its declared support; \
                 the bounded-support contract of Prepared::derive_local is violated",
            );
        }

        // Collect the deltas, ascending by local delta -- the canonical,
        // bucket-count-independent order that determinism relies on.
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
            // `d` ascends over this loop, so `deltas` ends up sorted by
            // `local_delta` with no explicit sort.
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

/// Probe `channel.apply` on every local basis Pauli and read off `amp[s][t]`.
///
/// With `background`, every non-support bit of the probe is set — used in debug builds to detect a channel that reads outside its support.
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

    // Bits belonging to the support, so they can be excluded from a background.
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
            // The output must differ from the input only inside the support,
            // otherwise this channel cannot be expressed as a local PTM.
            for w in 0..W {
                if (buf_x[i][w] ^ in_x[w]) & !sup_x[w] != 0
                    || (buf_z[i][w] ^ in_z[w]) & !sup_z[w] != 0
                {
                    return None;
                }
            }
            let t = ptm.support_bits(&buf_x[i], &buf_z[i]);
            // Several outputs can share a `t` only if the channel emits the same
            // Pauli twice; sum rather than overwrite so the table stays faithful.
            row[t] += buf_c[i];
        }
    }

    Some(table)
}

#[cfg(test)]
mod tests;
