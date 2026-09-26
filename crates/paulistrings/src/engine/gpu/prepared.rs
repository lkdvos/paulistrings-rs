//! [`DevicePrepared`], one prepared channel in the layout `kernels/prelude.cuh`'s `Table` reads. See ARCHITECTURE.md §Prepared-Channels.

use num_complex::Complex64;

use super::fingerprint::FingerprintRows;
use crate::bucket::hash::Gf2Hash;
use crate::channel::prepared::{Prepared, LOCAL_DIM};
use crate::engine::partitioned::plan::RemoteDelta;

/// `rem[e]` of an entry sourced from a local bucket; matches `NO_REMOTE` in `kernels/prelude.cuh`.
pub(crate) const NO_REMOTE: u32 = u32::MAX;

/// Entries with any nonzero amplitude per pattern, averaged over the active patterns, at or above which a table counts as dense.
/// Dense tables reduce by the block-wide segmented scan, sparse ones by the head-serial walk.
pub(crate) const DENSE_ROWS_PER_PATTERN: f64 = 2.0;

/// A prepared channel's tables as the kernels take them; `bucket_delta` is recomputed from the masks for the current hash, so a refine between `prepare` and the layer costs no second `prepare`.
#[derive(Clone, Debug)]
pub(crate) struct DevicePrepared<const W: usize> {
    /// `MODE_LOCAL` or `MODE_ROTATION`.
    pub(crate) mode: u32,
    pub(crate) entries: usize,
    pub(crate) kq: u32,
    pub(crate) q0: u32,
    pub(crate) q1: u32,
    pub(crate) rot_cos: f64,
    pub(crate) rot_sin: f64,
    /// `16 × LOCAL_DIM` complex amplitudes as f64 pairs.
    pub(crate) amp: Vec<f64>,
    /// 16 entries of `(x-mask, z-mask)` word pairs in the prelude's row layout.
    pub(crate) mask: Vec<u64>,
    /// Per entry, bit `s` set when pattern `s` emits.
    pub(crate) nz: Vec<u32>,
    /// `h(mask)` per entry under the hash passed to [`Self::new`].
    pub(crate) bucket_delta: Vec<u32>,
    /// `g(mask)` per entry, unmasked.
    pub(crate) gm: Vec<u64>,
    /// Per entry, the received block's slot (its index in the plan's remote list) or [`NO_REMOTE`].
    pub(crate) rem: Vec<u32>,
    /// Received entries.
    pub(crate) n_remote: usize,
    /// Entries that emit for some pattern, local and received alike; 2 for a rotation.
    pub(crate) fanout: usize,
    /// The reduction choice.
    pub(crate) dense: bool,
    /// One identity entry and no received entry: the K5 rescale path, which would drop every received row (ARCHITECTURE.md §Partitioning).
    pub(crate) key_preserving: bool,
    masks: Vec<([u64; W], [u64; W])>,
}

impl<const W: usize> DevicePrepared<W> {
    /// `remote` names the entries whose rows arrive from a partner instead of a local bucket, in the plan's order.
    pub(crate) fn new(
        prep: &Prepared<W>,
        hash: &Gf2Hash<W>,
        fp: &FingerprintRows<W>,
        remote: &[RemoteDelta],
    ) -> Self {
        let mut amp = vec![0f64; 16 * LOCAL_DIM * 2];
        let mut nz = vec![0u32; 16];
        let mut masks: Vec<([u64; W], [u64; W])> = Vec::new();
        let (mode, kq, q0, q1, rot_cos, rot_sin, dense, key_preserving);
        match prep {
            Prepared::Local(ptm) => {
                let dim = 1usize << (2 * ptm.k());
                let mut rows = 0usize;
                for (e, d) in ptm.deltas().iter().enumerate() {
                    for s in 0..LOCAL_DIM {
                        amp[(e * LOCAL_DIM + s) * 2] = d.amp[s].re;
                        amp[(e * LOCAL_DIM + s) * 2 + 1] = d.amp[s].im;
                        if d.amp[s] != Complex64::new(0.0, 0.0) {
                            nz[e] |= 1 << s;
                            if s < dim {
                                rows += 1;
                            }
                        }
                    }
                    masks.push((d.mask_x, d.mask_z));
                }
                let q = ptm.qubits();
                mode = 0;
                kq = ptm.k() as u32;
                q0 = q.first().copied().unwrap_or(0);
                q1 = q.get(1).copied().unwrap_or(0);
                rot_cos = 0.0;
                rot_sin = 0.0;
                dense = rows as f64 / dim as f64 >= DENSE_ROWS_PER_PATTERN;
                key_preserving = ptm.is_key_preserving();
            }
            Prepared::Rotation(r) => {
                masks.push(([0u64; W], [0u64; W]));
                masks.push(r.gen_mask());
                nz[0] = u32::MAX;
                nz[1] = u32::MAX;
                mode = 1;
                kq = 0;
                q0 = 0;
                q1 = 0;
                rot_cos = r.cos;
                rot_sin = r.sin;
                dense = false;
                key_preserving = false;
            }
        }
        assert!(masks.len() <= 16, "a prepared table has at most 16 entries");
        let entries = masks.len();
        let fanout = (0..entries).filter(|&e| nz[e] != 0).count();
        let mut mask = vec![0u64; 16 * 2 * W];
        let mut gm = vec![0u64; 16];
        for (e, (mx, mz)) in masks.iter().enumerate() {
            mask[(e * 2) * W..(e * 2 + 1) * W].copy_from_slice(mx);
            mask[(e * 2 + 1) * W..(e * 2 + 2) * W].copy_from_slice(mz);
            gm[e] = fp.fingerprint(mx, mz);
        }
        let mut rem = vec![NO_REMOTE; 16];
        for (k, r) in remote.iter().enumerate() {
            assert!(
                r.entry < entries,
                "remote entry {} outside the table",
                r.entry
            );
            rem[r.entry] = k as u32;
        }
        let mut out = Self {
            mode,
            entries,
            kq,
            q0,
            q1,
            rot_cos,
            rot_sin,
            amp,
            mask,
            nz,
            bucket_delta: vec![0u32; 16],
            gm,
            rem,
            n_remote: remote.len(),
            fanout,
            dense,
            key_preserving: key_preserving && remote.is_empty(),
            masks,
        };
        out.rehash(hash);
        out
    }

    /// Recompute every entry's bucket delta under `hash`, after a refine.
    pub(crate) fn rehash(&mut self, hash: &Gf2Hash<W>) {
        for (e, (mx, mz)) in self.masks.iter().enumerate() {
            self.bucket_delta[e] = hash.bucket_of(mx, mz);
        }
    }

    /// The support patterns entry `e` emits into, one bit per pattern; empty for a rotation.
    fn output_patterns(&self, e: usize) -> u32 {
        if self.mode != 0 {
            return 0;
        }
        let bit = |v: &[u64; W], q: u32| ((v[(q / 64) as usize] >> (q % 64)) & 1) as usize;
        let (mx, mz) = &self.masks[e];
        let mut ld = 0usize;
        if self.kq > 0 {
            ld |= bit(mx, self.q0) | bit(mz, self.q0) << 1;
        }
        if self.kq > 1 {
            ld |= bit(mx, self.q1) << 2 | bit(mz, self.q1) << 3;
        }
        (0..LOCAL_DIM)
            .filter(|&s| (self.nz[e] >> s) & 1 != 0)
            .fold(0, |acc, s| acc | 1 << (s ^ ld))
    }

    /// Whether two of `entries` can emit one key: they must reach a common output pattern, since `v ⊕ d_a = w ⊕ d_b` puts both rows on one pattern.
    pub(crate) fn entries_can_collide(&self, entries: &[usize]) -> bool {
        let out: Vec<u32> = entries.iter().map(|&e| self.output_patterns(e)).collect();
        out.iter()
            .enumerate()
            .any(|(i, a)| out[i + 1..].iter().any(|b| a & b != 0))
    }

    /// The table restricted to `entries`, renumbered `0..entries.len()` in that order, every entry sourced from a local bucket.
    pub(crate) fn restrict(&self, entries: &[usize]) -> Self {
        assert!(self.mode == 0, "only a tabulated channel restricts");
        let mut t = self.clone();
        t.entries = entries.len();
        t.amp.fill(0.0);
        t.mask.fill(0);
        t.nz.fill(0);
        t.bucket_delta.fill(0);
        t.gm.fill(0);
        t.rem.fill(NO_REMOTE);
        t.masks.clear();
        let a = LOCAL_DIM * 2;
        for (j, &e) in entries.iter().enumerate() {
            t.amp[j * a..(j + 1) * a].copy_from_slice(&self.amp[e * a..(e + 1) * a]);
            t.mask[j * 2 * W..(j + 1) * 2 * W]
                .copy_from_slice(&self.mask[e * 2 * W..(e + 1) * 2 * W]);
            t.nz[j] = self.nz[e];
            t.bucket_delta[j] = self.bucket_delta[e];
            t.gm[j] = self.gm[e];
            t.masks.push(self.masks[e]);
        }
        t.n_remote = 0;
        t.fanout = (0..t.entries).filter(|&j| t.nz[j] != 0).count();
        t.key_preserving = false;
        t
    }

    /// The distinct bucket deltas of the local entries, ascending, always containing `0`: the input to `Gf2Span::new`, equal to the plan's `local_bucket_deltas`.
    pub(crate) fn bucket_deltas(&self) -> Vec<u32> {
        let mut v: Vec<u32> = (0..self.entries)
            .filter(|&e| self.rem[e] == NO_REMOTE)
            .map(|e| self.bucket_delta[e])
            .collect();
        v.push(0);
        v.sort_unstable();
        v.dedup();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::sum::DEFAULT_HASH_SEED;
    use crate::channel::clifford::Clifford2Q;
    use crate::channel::noise::Depolarizing;
    use crate::channel::rotation::PauliRotation;
    use crate::channel::{Channel, GeneralUnitary2Q};
    use crate::pauli_string::PauliString;
    use crate::test_support::{haar_su4_matrix, sqrt_swap_matrix, zz_rotation};

    fn table<const W: usize>(ch: &dyn Channel<W>, nq: usize) -> DevicePrepared<W> {
        let hash = Gf2Hash::<W>::new(nq, 6, DEFAULT_HASH_SEED);
        let prep = ch.prepare(&hash, false).expect("prepared");
        DevicePrepared::new(&prep, &hash, &FingerprintRows::new(hash.seed()), &[])
    }

    #[test]
    fn classification_and_fanout_per_table() {
        let cnot = table::<1>(&Clifford2Q::cnot(1, 3), 8);
        assert_eq!(
            (cnot.fanout, cnot.dense, cnot.key_preserving),
            (cnot.entries, false, false)
        );
        assert!(cnot.entries > 1);
        let zz = table::<1>(&zz_rotation::<1>(1, 3, 0.3), 8);
        assert_eq!((zz.fanout, zz.dense), (2, false));
        let swap = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, sqrt_swap_matrix()), 8);
        assert!(swap.dense, "sqrt-swap averages 3.65 rows per pattern");
        let su4 = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, haar_su4_matrix()), 8);
        assert_eq!((su4.fanout, su4.dense), (16, true));
        let dep = table::<1>(
            &Depolarizing {
                support: [2],
                p: 0.1,
            },
            8,
        );
        assert!(dep.key_preserving && dep.fanout == 1 && !dep.dense);
        let mut gen = PauliString::<2>::x(3);
        gen.z[1] |= 1 << 2;
        gen.x[0] |= 1 << 40;
        let rot = table::<2>(&PauliRotation::new(gen, 0.7), 128);
        assert_eq!(
            (rot.mode, rot.entries, rot.fanout, rot.dense),
            (1, 2, 2, false)
        );
        assert_eq!(rot.mask[4..6], gen.x);
        assert_eq!(rot.mask[6..8], gen.z);
        assert_eq!(rot.bucket_deltas().len(), 2);
    }

    /// An identity-only retained table with received entries must not take the K5 path, and its position map spans the local deltas only.
    #[test]
    fn received_entries_disable_the_rescale_path_and_leave_the_local_span() {
        use crate::bucket::hash::PartitionRows;
        use crate::engine::partitioned::plan::PartitionPlan;
        let hash = Gf2Hash::<1>::new(8, 4, DEFAULT_HASH_SEED);
        let fp = FingerprintRows::new(hash.seed());
        // `H` has the deltas `{0, X₁Z₁}`; a row reading qubit 1's x-bit makes the one non-identity entry remote.
        let ch = crate::channel::clifford::Clifford1Q::h(1);
        let prep = ch.prepare(&hash, false).unwrap();
        let rows = PartitionRows::<1>::from_rows(8, vec![[0b10u64]], vec![[0u64]]);
        let plan = PartitionPlan::new(&prep, &rows, 0);
        let Prepared::Local(ptm) = &prep else {
            unreachable!()
        };
        let retained = ptm.retain_entries(&plan.local_entries);
        assert!(
            retained.is_key_preserving() || plan.remote.is_empty(),
            "fixture: the retained table must be identity-only"
        );
        let t = DevicePrepared::new(&prep, &hash, &fp, &plan.remote);
        assert!(!plan.remote.is_empty());
        assert!(!t.key_preserving, "received rows force the full path");
        assert_eq!(t.n_remote, plan.remote.len());
        for r in &plan.remote {
            assert_ne!(t.rem[r.entry], NO_REMOTE);
        }
        assert_eq!(t.bucket_deltas(), plan.local_bucket_deltas);
        let local = DevicePrepared::new(&prep, &hash, &fp, &[]);
        assert!(!local.key_preserving && local.n_remote == 0);
    }

    /// A Clifford maps patterns bijectively, so no two of its entries share an output pattern; a dense SU(4) and `sqrt(SWAP)` do, and a rotation never restricts.
    #[test]
    fn collisions_between_entries_follow_the_output_patterns() {
        let all = |t: &DevicePrepared<1>| (0..t.entries).collect::<Vec<_>>();
        let cnot = table::<1>(&Clifford2Q::cnot(1, 3), 8);
        assert!(!cnot.entries_can_collide(&all(&cnot)));
        let su4 = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, haar_su4_matrix()), 8);
        assert!(su4.entries_can_collide(&all(&su4)));
        assert!(
            !su4.entries_can_collide(&[3]),
            "one entry cannot collide with itself"
        );
        let swap = table::<1>(&GeneralUnitary2Q::from_matrix(1, 3, sqrt_swap_matrix()), 8);
        assert!(swap.entries_can_collide(&all(&swap)));
        let mut gen = PauliString::<2>::x(3);
        gen.z[1] |= 1 << 2;
        gen.x[0] |= 1 << 40;
        let rot = table::<2>(&PauliRotation::new(gen, 0.7), 128);
        assert_eq!(rot.mode, 1);
        assert!(!rot.entries_can_collide(&[0, 1]));
    }

    #[test]
    fn restrict_renumbers_the_chosen_entries_and_sources_them_locally() {
        let su4 = table::<2>(
            &GeneralUnitary2Q::from_matrix(1, 70, haar_su4_matrix()),
            128,
        );
        let pick = [9usize, 2, 14];
        let r = su4.restrict(&pick);
        assert_eq!((r.entries, r.n_remote, r.key_preserving), (3, 0, false));
        assert_eq!(r.dense, su4.dense);
        for (j, &e) in pick.iter().enumerate() {
            let a = LOCAL_DIM * 2;
            assert_eq!(r.amp[j * a..(j + 1) * a], su4.amp[e * a..(e + 1) * a]);
            assert_eq!(r.mask[j * 4..(j + 1) * 4], su4.mask[e * 4..(e + 1) * 4]);
            assert_eq!(
                (r.nz[j], r.bucket_delta[j], r.gm[j]),
                (su4.nz[e], su4.bucket_delta[e], su4.gm[e])
            );
        }
        assert!(r.rem.iter().all(|&k| k == NO_REMOTE));
        assert!(r.nz[3..].iter().all(|&m| m == 0));
        assert_eq!(r.fanout, 3);
    }

    #[test]
    fn rehash_tracks_the_hash_and_gm_is_the_fingerprint_of_the_mask() {
        let hash = Gf2Hash::<1>::new(8, 3, DEFAULT_HASH_SEED);
        let ch = Clifford2Q::cnot(1, 3);
        let prep = ch.prepare(&hash, false).unwrap();
        let fp = FingerprintRows::new(hash.seed());
        let mut t = DevicePrepared::new(&prep, &hash, &fp, &[]);
        let Prepared::Local(ptm) = &prep else {
            unreachable!()
        };
        for (e, d) in ptm.deltas().iter().enumerate() {
            assert_eq!(t.bucket_delta[e], d.bucket_delta);
            assert_eq!(t.gm[e], fp.fingerprint(&d.mask_x, &d.mask_z));
        }
        let mut refined = hash.clone();
        refined.refine();
        refined.refine();
        t.rehash(&refined);
        let prep2 = ch.prepare(&refined, false).unwrap();
        let Prepared::Local(ptm2) = &prep2 else {
            unreachable!()
        };
        for (e, d) in ptm2.deltas().iter().enumerate() {
            assert_eq!(t.bucket_delta[e], d.bucket_delta);
        }
        assert_eq!(t.bucket_deltas(), ptm2.bucket_deltas());
    }
}
