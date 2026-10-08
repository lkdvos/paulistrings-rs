//! [`DevicePrepared`], one prepared channel in the layout `kernels/prelude.cuh`'s `Table` reads. See ARCHITECTURE.md §Prepared-Channels.

use num_complex::Complex64;

use super::fingerprint::FingerprintRows;
use crate::channel::prepared::{Prepared, LOCAL_DIM};
use crate::engine::partitioned::plan::RemoteDelta;
use crate::pauli_sum::hash::Gf2Hash;

/// `rem[e]` of an entry sourced from a local bucket; matches `NO_REMOTE` in `kernels/prelude.cuh`.
pub(crate) const NO_REMOTE: u32 = u32::MAX;

/// `entry_of[s]` of a support pattern no entry emits for; matches `NO_ENTRY` in `kernels/prelude.cuh`.
pub(crate) const NO_ENTRY: u32 = u32::MAX;

/// Emitting entries per active pattern at or above which a table reduces by the segmented scan rather than the head-serial walk.
const DENSE_ROWS_PER_PATTERN: f64 = 2.0;

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
    /// At most one emitting entry per pattern, no two entries reaching one output pattern, no received entry: the scatter path (`kernels/permute.cu`), after K5.
    pub(crate) permutation: bool,
    /// Per support pattern, the one entry that emits for it or [`NO_ENTRY`]; meaningful only when `permutation`.
    pub(crate) entry_of: Vec<u32>,
    masks: Vec<([u64; W], [u64; W])>,
}

impl<const W: usize> DevicePrepared<W> {
    /// `remote` names the entries whose rows arrive from a partner instead of a local bucket, in the plan's order.
    pub(crate) fn new(
        prepared: &Prepared<W>,
        hash: &Gf2Hash<W>,
        fingerprints: &FingerprintRows<W>,
        remote: &[RemoteDelta],
    ) -> Self {
        let mut amp = vec![0f64; 16 * LOCAL_DIM * 2];
        let mut nz = vec![0u32; 16];
        let mut masks: Vec<([u64; W], [u64; W])> = Vec::new();
        let mut entry_of = vec![NO_ENTRY; LOCAL_DIM];
        let mut one_entry_per_pattern = false;
        let (mode, kq, q0, q1, rot_cos, rot_sin, dense, key_preserving);
        match prepared {
            Prepared::Local(ptm) => {
                let dim = 1usize << (2 * ptm.k());
                let mut rows = 0usize;
                one_entry_per_pattern = true;
                for (e, d) in ptm.deltas().iter().enumerate() {
                    for s in 0..LOCAL_DIM {
                        amp[(e * LOCAL_DIM + s) * 2] = d.amp[s].re;
                        amp[(e * LOCAL_DIM + s) * 2 + 1] = d.amp[s].im;
                        if d.amp[s] != Complex64::new(0.0, 0.0) {
                            nz[e] |= 1 << s;
                            if s < dim {
                                rows += 1;
                                if entry_of[s] != NO_ENTRY {
                                    one_entry_per_pattern = false;
                                }
                                entry_of[s] = e as u32;
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
            gm[e] = fingerprints.fingerprint(mx, mz);
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
            permutation: false,
            entry_of,
            masks,
        };
        let all: Vec<usize> = (0..entries).collect();
        out.permutation = mode == 0
            && one_entry_per_pattern
            && remote.is_empty()
            && !out.entries_can_collide(&all);
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
        let mut local_delta = 0usize;
        if self.kq > 0 {
            local_delta |= bit(mx, self.q0) | bit(mz, self.q0) << 1;
        }
        if self.kq > 1 {
            local_delta |= bit(mx, self.q1) << 2 | bit(mz, self.q1) << 3;
        }
        (0..LOCAL_DIM)
            .filter(|&s| (self.nz[e] >> s) & 1 != 0)
            .fold(0, |patterns, s| patterns | 1 << (s ^ local_delta))
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
        t.permutation = false;
        t.entry_of.fill(NO_ENTRY);
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
mod tests;
