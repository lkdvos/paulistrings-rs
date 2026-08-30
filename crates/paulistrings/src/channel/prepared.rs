//! Per-layer prepared form of a channel. See v0.2 design doc §5.
//!
//! v0.1 calls [`Channel::apply`] once per input term, through a vtable, and lets
//! each channel redo its own setup every time — `PauliRotation` recomputes
//! `theta.cos()/sin()` per term, `Clifford1Q::apply_adjoint` rebuilds its inverse
//! table per term. The bucketed engine instead *prepares* a channel once per
//! layer into the form below, reducing the inner loop to one table lookup on ≤ 4
//! extracted bits, one XOR with a precomputed mask, and one complex multiply
//! (v0.2 §2.6).
//!
//! The prepared form also carries what the engine needs to know *which buckets to
//! read*: for each possible key delta `d`, the bucket delta `δ = H·d`.

use num_complex::Complex64;

use super::{Channel, OutputBuffer};
use crate::bucket::hash::Gf2Hash;
use crate::pauli_string::PauliString;

/// Largest support size handled by [`Prepared::Local`].
///
/// The dense local Pauli-transfer matrix is `4^k × 4^k`, so `k = 2` is 16×16 =
/// 4 KB of `Complex64` per layer. `k = 3` would be 64 KB, which stops being a
/// per-layer table worth building.
pub const MAX_LOCAL_SUPPORT: usize = 2;

/// `4^MAX_LOCAL_SUPPORT` — the number of local Pauli basis elements.
pub const LOCAL_DIM: usize = 16;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// One key delta, with the amplitude it carries for each input support pattern.
#[derive(Clone, Debug)]
pub struct DeltaEntry<const W: usize> {
    /// `δ = H·d`. Output bucket `β'` reads input bucket `β' ^ bucket_delta` for
    /// this delta.
    ///
    /// Several entries may share a `bucket_delta`: with a dense random `H`,
    /// `rank(H|_D) = dim D` and each is unique (v0.2 §2.6), but a collision is a
    /// performance wart, not a correctness problem, and is handled by simply
    /// having two entries name the same input bucket.
    pub bucket_delta: u32,
    /// The delta in local support coordinates: bit `2j` is the x-bit of support
    /// qubit `j`, bit `2j+1` its z-bit.
    ///
    /// This is the **canonical ordering key**. Determinism requires iterating
    /// deltas in an order that does not depend on the bucket count, and `δ =
    /// H·d` does depend on it while `d` does not (v0.2 §9.1).
    pub local_delta: u8,
    /// `d` lifted to a full-width XOR mask.
    pub mask_x: [u64; W],
    /// `d` lifted to a full-width XOR mask.
    pub mask_z: [u64; W],
    /// `amp[s]` is the amplitude taking input support pattern `s` to
    /// `s ^ local_delta`. Exactly zero means "no output for this `s`".
    pub amp: [Complex64; LOCAL_DIM],
}

/// A channel with support on at most [`MAX_LOCAL_SUPPORT`] qubits, as a dense
/// local Pauli-transfer matrix grouped by bucket delta.
#[derive(Clone, Debug)]
pub struct LocalPtm<const W: usize> {
    /// Support qubits, ascending. Only the first `k` are meaningful.
    qubits: [u32; MAX_LOCAL_SUPPORT],
    /// Number of support qubits, `0 ≤ k ≤ MAX_LOCAL_SUPPORT`.
    k: u8,
    /// The delta set, **ascending by `local_delta`**.
    ///
    /// This order is the engine's canonical equal-key summation order, and it
    /// must not depend on the bucket count: `local_delta` is `H`-independent
    /// while `bucket_delta = H·d` is not, and duplicate-key summation order is
    /// observable through floating-point non-associativity (v0.2 §9.1). The
    /// coset engine gathers input-bucket-major for locality and restores this
    /// order afterwards, by carrying each entry's index here as a tag through
    /// the per-run sort (v0.3 §2, `sort_merge::sort_phase_tagged`).
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

    /// The delta set, ascending by `local_delta` — the canonical equal-key
    /// summation order; each entry's index here is its sort tag in the engine.
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
    /// `Clifford1Q::{x, y, z}` — which the baselines showed cost a full
    /// `O(n log n)` sort in v0.1 to multiply each coefficient by a scalar.
    pub fn is_key_preserving(&self) -> bool {
        self.deltas.len() == 1 && self.deltas[0].local_delta == 0
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
/// The delta set is `{0, P}` for *any* generator weight (v0.2 §2.3), so only two
/// buckets are ever read — but the amplitude's `i^k` phase depends on `2w`
/// support bits, which stops being tabulable. Amplitudes are computed per term
/// instead, with `cos`/`sin` hoisted out of the loop (which v0.1 did not do).
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
    /// length is `2^rank(H|_D)` — 1, 2, 4 or 16 for the built-ins (v0.2 §2.4).
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

    /// Derive the prepared form of a bounded-support channel by probing its own
    /// [`Channel::apply`].
    ///
    /// Returns `None` when the channel's support is wider than
    /// [`MAX_LOCAL_SUPPORT`], or when it writes outside its declared support. In
    /// both cases the caller must fall back to the whole-sum v0.1 path, which is
    /// correct but not bucketed — a performance fallback, never a correctness
    /// compromise.
    ///
    /// # The soundness precondition
    ///
    /// This is exact **iff** the channel honours the bounded-support contract:
    /// the output amplitude may depend on the input only through its support
    /// bits. Probing cannot fully verify that — a channel that *reads* qubit 5
    /// while declaring support `[0]` would produce a table that is wrong for
    /// inputs this function never tried. Two things mitigate it:
    ///
    /// * Debug builds re-derive with an all-ones background outside the support
    ///   and assert the two tables agree, which catches the common case.
    /// * A property test checks every built-in's derived table against `apply`
    ///   on randomized full-width inputs (v0.2 §5.3).
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
        // duplicate-free -- unlike the old `Vec<u32>` support list, there is
        // no need to sort or check for a caller-supplied duplicate.
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
/// With `background`, every non-support bit of the probe is set — used in debug
/// builds to detect a channel that reads outside its support.
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
mod tests {
    use super::*;
    use crate::channel::clifford::{Clifford1Q, Clifford2Q};
    use crate::channel::identity::IdentityChannel;
    use crate::channel::noise::{AmplitudeDamping, Dephasing, Depolarizing};
    use crate::channel::rotation::PauliRotation;

    const TOL: f64 = 1e-12;

    struct Xs64(u64);
    impl Xs64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
    }

    type Term<const W: usize> = ([u64; W], [u64; W], Complex64);

    /// Outputs of `apply` / `apply_adjoint`, with exact zeros dropped and equal
    /// keys summed — the same normalization the merge phase performs.
    fn via_apply<const W: usize, C: Channel<W> + ?Sized>(
        ch: &C,
        adjoint: bool,
        x: &[u64; W],
        z: &[u64; W],
        coeff: Complex64,
    ) -> Vec<Term<W>> {
        let f = ch.max_fanout().max(1);
        let mut bx = vec![[0u64; W]; f];
        let mut bz = vec![[0u64; W]; f];
        let mut bc = vec![ZERO; f];
        let mut len = 0usize;
        {
            let mut out = OutputBuffer::<W> {
                x: &mut bx,
                z: &mut bz,
                coeff: &mut bc,
                len: &mut len,
            };
            if adjoint {
                ch.apply_adjoint(x, z, coeff, &mut out);
            } else {
                ch.apply(x, z, coeff, &mut out);
            }
        }
        normalize((0..len).map(|i| (bx[i], bz[i], bc[i])).collect())
    }

    /// The same outputs, reconstructed from the prepared table.
    fn via_prepared<const W: usize>(
        prep: &Prepared<W>,
        x: &[u64; W],
        z: &[u64; W],
        coeff: Complex64,
    ) -> Vec<Term<W>> {
        let mut out: Vec<Term<W>> = Vec::new();
        match prep {
            Prepared::Local(p) => {
                let s = p.support_bits(x, z);
                for m in p.deltas() {
                    let a = m.amp[s];
                    if a == ZERO {
                        continue;
                    }
                    let mut ox = *x;
                    let mut oz = *z;
                    for w in 0..W {
                        ox[w] ^= m.mask_x[w];
                        oz[w] ^= m.mask_z[w];
                    }
                    out.push((ox, oz, coeff * a));
                }
            }
            Prepared::Rotation(r) => {
                let input = PauliString::<W> { x: *x, z: *z };
                if input.commutes_with(&r.gen) {
                    out.push((*x, *z, coeff));
                } else {
                    out.push((*x, *z, coeff * r.cos));
                    let mut prod = input;
                    let phase = prod.mul_assign(&r.gen);
                    let total = crate::phase::Phase::I + phase;
                    out.push((prod.x, prod.z, total.apply(coeff) * r.sin));
                }
            }
        }
        normalize(out)
    }

    fn normalize<const W: usize>(mut v: Vec<Term<W>>) -> Vec<Term<W>> {
        v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        let mut out: Vec<Term<W>> = Vec::new();
        for (x, z, c) in v {
            match out.last_mut() {
                Some(last) if last.0 == x && last.1 == z => last.2 += c,
                _ => out.push((x, z, c)),
            }
        }
        out.retain(|t| t.2 != ZERO);
        out
    }

    fn assert_terms_eq<const W: usize>(a: &[Term<W>], b: &[Term<W>], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: term count {a:?} vs {b:?}");
        for (p, q) in a.iter().zip(b.iter()) {
            assert_eq!(p.0, q.0, "{what}: x key");
            assert_eq!(p.1, q.1, "{what}: z key");
            assert!((p.2 - q.2).norm() < TOL, "{what}: coeff {} vs {}", p.2, q.2);
        }
    }

    /// The core B.4 check: the derived table must reproduce `apply` on
    /// **randomized full-width** inputs, not just on the basis probes used to
    /// build it. This is what catches a channel reading outside its declared
    /// support (v0.2 §5.3).
    fn check_agrees_on_random_inputs<const W: usize, C: Channel<W>>(
        ch: &C,
        num_qubits: usize,
        label: &str,
    ) {
        let hash = Gf2Hash::<W>::new(num_qubits, 10, 0xD00D);
        for &adjoint in &[false, true] {
            let prep = ch
                .prepare(&hash, adjoint)
                .unwrap_or_else(|| panic!("{label}: prepare returned None"));
            let mut rng = Xs64::new(0x5EED ^ (adjoint as u64));
            for _ in 0..400 {
                let mut x = [0u64; W];
                let mut z = [0u64; W];
                for w in 0..W {
                    x[w] = rng.next_u64();
                    z[w] = rng.next_u64();
                }
                let coeff = Complex64::new(
                    (rng.next_u64() as i64 as f64) / (i64::MAX as f64),
                    (rng.next_u64() as i64 as f64) / (i64::MAX as f64),
                );
                let direct = via_apply(ch, adjoint, &x, &z, coeff);
                let table = via_prepared(&prep, &x, &z, coeff);
                assert_terms_eq(&direct, &table, &format!("{label} adjoint={adjoint}"));
            }
        }
    }

    // ---- the derived table reproduces `apply`, per built-in channel ----

    #[test]
    fn derived_table_matches_apply_identity() {
        check_agrees_on_random_inputs::<2, _>(&IdentityChannel::new(), 128, "identity");
    }

    #[test]
    fn derived_table_matches_apply_clifford1q() {
        for (name, ch) in [
            ("h", Clifford1Q::h(3)),
            ("s", Clifford1Q::s(3)),
            ("x", Clifford1Q::x(3)),
            ("y", Clifford1Q::y(3)),
            ("z", Clifford1Q::z(3)),
        ] {
            check_agrees_on_random_inputs::<2, _>(&ch, 128, name);
        }
        // Also across a word boundary.
        check_agrees_on_random_inputs::<2, _>(&Clifford1Q::h(70), 128, "h@70");
    }

    #[test]
    fn derived_table_matches_apply_clifford2q() {
        for (name, ch) in [
            ("cnot", Clifford2Q::cnot(1, 4)),
            ("cz", Clifford2Q::cz(1, 4)),
            ("swap", Clifford2Q::swap(1, 4)),
        ] {
            check_agrees_on_random_inputs::<2, _>(&ch, 128, name);
        }
        // Straddling the word boundary.
        check_agrees_on_random_inputs::<2, _>(&Clifford2Q::cnot(60, 70), 128, "cnot@60,70");
    }

    #[test]
    fn derived_table_matches_apply_noise() {
        check_agrees_on_random_inputs::<2, _>(
            &Depolarizing {
                support: [5],
                p: 0.07,
            },
            128,
            "depolarizing",
        );
        check_agrees_on_random_inputs::<2, _>(
            &Dephasing {
                support: [5],
                p: 0.07,
            },
            128,
            "dephasing",
        );
        check_agrees_on_random_inputs::<2, _>(
            &AmplitudeDamping {
                support: [5],
                gamma: 0.3,
            },
            128,
            "amplitude_damping",
        );
    }

    #[test]
    fn derived_table_matches_apply_rotation_weight_1_and_2() {
        check_agrees_on_random_inputs::<2, _>(
            &PauliRotation::new(PauliString::<2>::z(9), 0.37),
            128,
            "rot_z",
        );
        check_agrees_on_random_inputs::<2, _>(
            &PauliRotation::new(PauliString::<2>::y(70), 0.37),
            128,
            "rot_y@70",
        );
        let mut zz = PauliString::<2>::z(9);
        zz.mul_assign(&PauliString::<2>::z(70));
        check_agrees_on_random_inputs::<2, _>(&PauliRotation::new(zz, 0.37), 128, "rot_zz");
    }

    #[test]
    fn functional_form_matches_apply_for_a_wide_rotation() {
        // Weight 4 > MAX_LOCAL_SUPPORT, so this takes the Rotation variant.
        let mut gen = PauliString::<2>::z(1);
        for q in [5u32, 66, 100] {
            gen.mul_assign(&PauliString::<2>::z(q));
        }
        let rot = PauliRotation::new(gen, 0.41);
        assert_eq!(rot.weight(), 4);
        check_agrees_on_random_inputs::<2, _>(&rot, 128, "rot_weight4");
    }

    // ---- delta-set dimensions must match the v0.2 §2.3 table ----

    fn n_bucket_deltas<const W: usize, C: Channel<W>>(ch: &C, adjoint: bool) -> usize {
        // Plenty of bucket bits, so distinct key deltas do not collide.
        let hash = Gf2Hash::<W>::new(128, 16, 0xBEEF);
        ch.prepare(&hash, adjoint).unwrap().bucket_deltas().len()
    }

    #[test]
    fn bucket_fanin_matches_the_design_table() {
        // 1 bucket: key-preserving channels.
        for (name, n) in [
            (
                "identity",
                n_bucket_deltas::<2, _>(&IdentityChannel::new(), false),
            ),
            (
                "depolarizing",
                n_bucket_deltas::<2, _>(
                    &Depolarizing {
                        support: [5],
                        p: 0.1,
                    },
                    false,
                ),
            ),
            (
                "dephasing",
                n_bucket_deltas::<2, _>(
                    &Dephasing {
                        support: [5],
                        p: 0.1,
                    },
                    false,
                ),
            ),
            ("pauli_x", n_bucket_deltas::<2, _>(&Clifford1Q::x(5), false)),
            ("pauli_y", n_bucket_deltas::<2, _>(&Clifford1Q::y(5), false)),
            ("pauli_z", n_bucket_deltas::<2, _>(&Clifford1Q::z(5), false)),
        ] {
            assert_eq!(n, 1, "{name} should read 1 input bucket per output bucket");
        }

        // 2 buckets: 1-dimensional delta sets.
        for (name, n) in [
            ("h", n_bucket_deltas::<2, _>(&Clifford1Q::h(5), false)),
            ("s", n_bucket_deltas::<2, _>(&Clifford1Q::s(5), false)),
            (
                "amplitude_damping",
                n_bucket_deltas::<2, _>(
                    &AmplitudeDamping {
                        support: [5],
                        gamma: 0.3,
                    },
                    false,
                ),
            ),
            (
                "rot_z",
                n_bucket_deltas::<2, _>(&PauliRotation::new(PauliString::<2>::z(5), 0.3), false),
            ),
        ] {
            assert_eq!(n, 2, "{name} should read 2 input buckets per output bucket");
        }

        // 4 buckets: 2-dimensional delta sets.
        for (name, n) in [
            (
                "cnot",
                n_bucket_deltas::<2, _>(&Clifford2Q::cnot(1, 4), false),
            ),
            ("cz", n_bucket_deltas::<2, _>(&Clifford2Q::cz(1, 4), false)),
            (
                "swap",
                n_bucket_deltas::<2, _>(&Clifford2Q::swap(1, 4), false),
            ),
        ] {
            assert_eq!(n, 4, "{name} should read 4 input buckets per output bucket");
        }
    }

    #[test]
    fn a_rotation_reads_two_buckets_at_any_generator_weight() {
        // The headline structural claim: unlike the support-derived bucketing
        // v0.1 §5 envisaged (which would need 4^w buckets), the delta set {0, P}
        // is 1-dimensional at every weight.
        for weight in 1..=6usize {
            let mut gen = PauliString::<2>::z(0);
            for q in 1..weight as u32 {
                gen.mul_assign(&PauliString::<2>::z(q * 13));
            }
            let rot = PauliRotation::new(gen, 0.3);
            assert_eq!(rot.weight(), weight);
            assert_eq!(
                n_bucket_deltas::<2, _>(&rot, false),
                2,
                "weight {weight} should still read 2 buckets",
            );
        }
    }

    #[test]
    fn adjoint_preparations_have_the_same_fanin() {
        // Conjugating by G^-1 has delta set im(S^-1 ^ I), a different subspace of
        // the same dimension, so the bucket count must not change.
        for (name, fwd, adj) in [
            (
                "s",
                n_bucket_deltas::<2, _>(&Clifford1Q::s(5), false),
                n_bucket_deltas::<2, _>(&Clifford1Q::s(5), true),
            ),
            (
                "cnot",
                n_bucket_deltas::<2, _>(&Clifford2Q::cnot(1, 4), false),
                n_bucket_deltas::<2, _>(&Clifford2Q::cnot(1, 4), true),
            ),
            (
                "rot_z",
                n_bucket_deltas::<2, _>(&PauliRotation::new(PauliString::<2>::z(5), 0.3), false),
                n_bucket_deltas::<2, _>(&PauliRotation::new(PauliString::<2>::z(5), 0.3), true),
            ),
            (
                // Included as of B.8: before the adjoint was implemented this
                // was trivially equal, because apply_adjoint *was* apply.
                "amp_damping",
                n_bucket_deltas::<2, _>(
                    &AmplitudeDamping {
                        support: [5],
                        gamma: 0.3,
                    },
                    false,
                ),
                n_bucket_deltas::<2, _>(
                    &AmplitudeDamping {
                        support: [5],
                        gamma: 0.3,
                    },
                    true,
                ),
            ),
        ] {
            assert_eq!(fwd, adj, "{name}: adjoint fan-in differs from forward");
        }
    }

    // ---- key-preserving detection ----

    #[test]
    fn key_preserving_channels_are_detected() {
        let hash = Gf2Hash::<2>::new(128, 8, 0x1);
        let yes: Vec<Box<dyn Channel<2>>> = vec![
            Box::new(IdentityChannel::new()),
            Box::new(Depolarizing {
                support: [5],
                p: 0.1,
            }),
            Box::new(Dephasing {
                support: [5],
                p: 0.1,
            }),
            Box::new(Clifford1Q::x(5)),
            Box::new(Clifford1Q::y(5)),
            Box::new(Clifford1Q::z(5)),
        ];
        for ch in &yes {
            match ch.prepare(&hash, false).unwrap() {
                Prepared::Local(p) => assert!(p.is_key_preserving()),
                _ => panic!("expected a Local preparation"),
            }
        }

        let no: Vec<Box<dyn Channel<2>>> = vec![
            Box::new(Clifford1Q::h(5)),
            Box::new(Clifford2Q::cnot(1, 4)),
            Box::new(PauliRotation::new(PauliString::<2>::z(5), 0.3)),
        ];
        for ch in &no {
            match ch.prepare(&hash, false).unwrap() {
                Prepared::Local(p) => assert!(!p.is_key_preserving()),
                _ => panic!("expected a Local preparation"),
            }
        }
    }

    // ---- support-bit packing ----

    #[test]
    fn support_bits_use_the_clifford2q_packing() {
        let hash = Gf2Hash::<1>::new(64, 8, 0x1);
        let prep = Clifford2Q::cnot(2, 7).prepare(&hash, false).unwrap();
        let Prepared::Local(p) = prep else {
            panic!("expected Local")
        };
        assert_eq!(p.qubits(), &[2, 7]);
        // x on q2 -> bit 0; z on q2 -> bit 1; x on q7 -> bit 2; z on q7 -> bit 3.
        let x2 = PauliString::<1>::x(2);
        assert_eq!(p.support_bits(&x2.x, &x2.z), 0b0001);
        let z2 = PauliString::<1>::z(2);
        assert_eq!(p.support_bits(&z2.x, &z2.z), 0b0010);
        let x7 = PauliString::<1>::x(7);
        assert_eq!(p.support_bits(&x7.x, &x7.z), 0b0100);
        let z7 = PauliString::<1>::z(7);
        assert_eq!(p.support_bits(&z7.x, &z7.z), 0b1000);
        // Bits outside the support are ignored.
        let mut noise = PauliString::<1>::y(30);
        noise.mul_assign(&PauliString::<1>::x(2));
        assert_eq!(p.support_bits(&noise.x, &noise.z), 0b0001);
    }

    // ---- hash collisions among deltas ----

    #[test]
    fn colliding_deltas_share_a_group_rather_than_being_lost() {
        // With 1 bucket bit, CNOT's four key deltas cannot map to four distinct
        // bucket deltas, so rank(H|_D) < dim D and groups must carry multiple
        // members. Correctness must not depend on H being well-chosen
        // (v0.2 §2.6).
        let hash = Gf2Hash::<2>::new(128, 1, 0xC011);
        let prep = Clifford2Q::cnot(1, 4).prepare(&hash, false).unwrap();
        let Prepared::Local(p) = prep else {
            panic!("expected Local")
        };
        assert!(
            p.bucket_deltas().len() <= 2,
            "only 2 bucket values exist with 1 bit",
        );
        // No delta is dropped: the total member count is still |D| = 4.
        assert_eq!(p.num_deltas(), 4);
        // And the table still reproduces `apply`.
        check_agrees_on_random_inputs::<2, _>(&Clifford2Q::cnot(1, 4), 128, "cnot_collided");
    }

    #[test]
    fn deltas_are_ascending_by_local_delta() {
        // Determinism depends on gathering deltas in an order that does not
        // depend on the bucket count (v0.2 §9.1). `local_delta` is that order,
        // and it must hold even when bucket deltas collide (bits = 1 here).
        for bits in [1u8, 4, 16] {
            let hash = Gf2Hash::<2>::new(128, bits, 0xC012);
            let prep = Clifford2Q::swap(1, 4).prepare(&hash, false).unwrap();
            let Prepared::Local(p) = prep else {
                panic!("expected Local")
            };
            for pair in p.deltas().windows(2) {
                assert!(
                    pair[0].local_delta < pair[1].local_delta,
                    "deltas not ascending at bits={bits}",
                );
            }
        }
    }

    #[test]
    fn delta_iteration_order_is_independent_of_bucket_count() {
        // The same claim, checked directly: changing `bits` must not reorder the
        // flattened delta sequence.
        let order = |bits: u8| -> Vec<u8> {
            let hash = Gf2Hash::<2>::new(128, bits, 0xC013);
            let prep = Clifford2Q::cnot(1, 4).prepare(&hash, false).unwrap();
            let Prepared::Local(p) = prep else {
                panic!("expected Local")
            };
            p.deltas().iter().map(|m| m.local_delta).collect()
        };
        let reference = order(16);
        for bits in [1u8, 2, 4, 8, 12] {
            assert_eq!(order(bits), reference, "delta order changed at bits={bits}");
        }
    }

    // ---- when derivation must refuse ----

    #[test]
    fn derive_refuses_support_wider_than_the_local_maximum() {
        // A weight-3 rotation: `derive_local` must decline, even though
        // `PauliRotation::prepare` overrides to the functional form.
        let mut gen = PauliString::<1>::z(0);
        gen.mul_assign(&PauliString::<1>::z(1));
        gen.mul_assign(&PauliString::<1>::z(2));
        let rot = PauliRotation::new(gen, 0.3);
        let hash = Gf2Hash::<1>::new(64, 8, 0x1);
        assert!(Prepared::derive_local(&rot, &hash, false).is_none());
        // The override still produces a usable preparation.
        assert!(matches!(
            rot.prepare(&hash, false),
            Some(Prepared::Rotation(_))
        ));
    }

    /// The popcount check reads straight off the mask, independent of which
    /// qubits are set or which word they land in.
    #[test]
    fn derive_local_rejects_popcount_gt_2() {
        struct ThreeQubits;
        impl<const W: usize> Channel<W> for ThreeQubits {
            fn max_fanout(&self) -> usize {
                1
            }
            fn support(&self) -> [u64; W] {
                let mut mask = [0u64; W];
                mask[0] = 0b111; // qubits 0, 1, 2 -- popcount 3
                mask
            }
            fn apply(
                &self,
                input_x: &[u64; W],
                input_z: &[u64; W],
                coeff: Complex64,
                out: &mut OutputBuffer<'_, W>,
            ) {
                out.push(*input_x, *input_z, coeff);
            }
        }
        let hash = Gf2Hash::<1>::new(64, 8, 0x1);
        assert!(Prepared::derive_local(&ThreeQubits, &hash, false).is_none());
    }

    #[test]
    fn derive_refuses_a_channel_that_writes_outside_its_support() {
        // A deliberately broken channel: declares support [0] but also flips
        // qubit 1. Deriving a local PTM for it would be silently wrong, so
        // derivation must decline and let the engine use the whole-sum path.
        struct Liar;
        impl<const W: usize> Channel<W> for Liar {
            fn max_fanout(&self) -> usize {
                1
            }
            fn support(&self) -> [u64; W] {
                let mut mask = [0u64; W];
                mask[0] = 1;
                mask
            }
            fn apply(
                &self,
                input_x: &[u64; W],
                input_z: &[u64; W],
                coeff: Complex64,
                out: &mut OutputBuffer<'_, W>,
            ) {
                let mut x = *input_x;
                x[0] ^= 0b10; // qubit 1 — outside the declared support
                out.push(x, *input_z, coeff);
            }
        }
        let hash = Gf2Hash::<1>::new(64, 8, 0x1);
        assert!(Prepared::derive_local(&Liar, &hash, false).is_none());
        assert!(Liar.prepare(&hash, false).is_none());
    }
}
