//! [`DevicePrepared`], one prepared channel in the layout `kernels/prelude.cuh`'s `Table` reads. See ARCHITECTURE.md §Prepared-Channels.

use num_complex::Complex64;

use super::fingerprint::FingerprintRows;
use crate::bucket::hash::Gf2Hash;
use crate::channel::prepared::{Prepared, LOCAL_DIM};

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
    /// Entries that emit for some pattern; 2 for a rotation.
    pub(crate) fanout: usize,
    /// The reduction choice.
    pub(crate) dense: bool,
    /// One identity entry only: the K5 rescale path.
    pub(crate) key_preserving: bool,
    masks: Vec<([u64; W], [u64; W])>,
}

impl<const W: usize> DevicePrepared<W> {
    pub(crate) fn new(prep: &Prepared<W>, hash: &Gf2Hash<W>, fp: &FingerprintRows<W>) -> Self {
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
            fanout,
            dense,
            key_preserving,
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

    /// The distinct bucket deltas, ascending, always containing `0`: the input to `Gf2Span::new`.
    pub(crate) fn bucket_deltas(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.bucket_delta[..self.entries].to_vec();
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
        DevicePrepared::new(&prep, &hash, &FingerprintRows::new(hash.seed()))
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

    #[test]
    fn rehash_tracks_the_hash_and_gm_is_the_fingerprint_of_the_mask() {
        let hash = Gf2Hash::<1>::new(8, 3, DEFAULT_HASH_SEED);
        let ch = Clifford2Q::cnot(1, 3);
        let prep = ch.prepare(&hash, false).unwrap();
        let fp = FingerprintRows::new(hash.seed());
        let mut t = DevicePrepared::new(&prep, &hash, &fp);
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
