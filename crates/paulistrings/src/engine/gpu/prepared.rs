//! [`DevicePrepared`], one prepared channel in the layout `kernels/prelude.cuh`'s `Table` reads.

use num_complex::Complex64;

use super::fingerprint::FingerprintRows;
use crate::channel::prepared::{Prepared, LOCAL_DIM};
use crate::engine::plan::RemoteDelta;
use crate::pauli_sum::hash::Gf2Hash;

/// `received_slot[e]` of an entry sourced from a local bucket; matches `NO_REMOTE` in `kernels/prelude.cuh`.
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
    pub(crate) support_size: u32,
    pub(crate) qubit_0: u32,
    pub(crate) qubit_1: u32,
    pub(crate) rotation_cos: f64,
    pub(crate) rotation_sin: f64,
    /// `16 × LOCAL_DIM` complex amplitudes as f64 pairs.
    pub(crate) amplitude: Vec<f64>,
    /// 16 entries of `(x-mask, z-mask)` word pairs in the prelude's row layout.
    pub(crate) mask: Vec<u64>,
    /// Per entry, bit `s` set when pattern `s` emits.
    pub(crate) nonzero_patterns: Vec<u32>,
    /// `h(mask)` per entry under the hash passed to [`Self::new`].
    pub(crate) bucket_delta: Vec<u32>,
    /// `g(mask)` per entry, unmasked.
    pub(crate) fingerprint_delta: Vec<u64>,
    /// Per entry, the received block's slot (its index in the plan's remote list) or [`NO_REMOTE`].
    pub(crate) received_slot: Vec<u32>,
    /// Received entries.
    pub(crate) received_entries: usize,
    /// Entries that emit for some pattern, local and received alike; 2 for a rotation.
    pub(crate) fanout: usize,
    /// The reduction choice.
    pub(crate) dense: bool,
    /// One identity entry and no received entry: the rescale path, which would drop every received row.
    pub(crate) key_preserving: bool,
    /// At most one emitting entry per pattern, no two entries reaching one output pattern, no received entry: the scatter path (`kernels/permute.cu`), after the rescale.
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
        let mut amplitude = vec![0f64; 16 * LOCAL_DIM * 2];
        let mut nonzero_patterns = vec![0u32; 16];
        let mut masks: Vec<([u64; W], [u64; W])> = Vec::new();
        let mut entry_of = vec![NO_ENTRY; LOCAL_DIM];
        let mut one_entry_per_pattern = false;
        let (
            mode,
            support_size,
            qubit_0,
            qubit_1,
            rotation_cos,
            rotation_sin,
            dense,
            key_preserving,
        );
        match prepared {
            Prepared::Local(ptm) => {
                let dim = 1usize << (2 * ptm.k());
                let mut rows = 0usize;
                one_entry_per_pattern = true;
                for (e, d) in ptm.deltas().iter().enumerate() {
                    for s in 0..LOCAL_DIM {
                        amplitude[(e * LOCAL_DIM + s) * 2] = d.amplitude[s].re;
                        amplitude[(e * LOCAL_DIM + s) * 2 + 1] = d.amplitude[s].im;
                        if d.amplitude[s] != Complex64::new(0.0, 0.0) {
                            nonzero_patterns[e] |= 1 << s;
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
                let mut q = ptm.qubits().map(|q| q as u32);
                mode = 0;
                support_size = ptm.k() as u32;
                qubit_0 = q.next().unwrap_or(0);
                qubit_1 = q.next().unwrap_or(0);
                rotation_cos = 0.0;
                rotation_sin = 0.0;
                dense = rows as f64 / dim as f64 >= DENSE_ROWS_PER_PATTERN;
                key_preserving = ptm.is_key_preserving();
            }
            Prepared::Rotation(r) => {
                masks.push(([0u64; W], [0u64; W]));
                masks.push(r.generator_mask());
                nonzero_patterns[0] = u32::MAX;
                nonzero_patterns[1] = u32::MAX;
                mode = 1;
                support_size = 0;
                qubit_0 = 0;
                qubit_1 = 0;
                rotation_cos = r.cos;
                rotation_sin = r.sin;
                dense = false;
                key_preserving = false;
            }
        }
        assert!(masks.len() <= 16, "a prepared table has at most 16 entries");
        let entries = masks.len();
        let fanout = (0..entries).filter(|&e| nonzero_patterns[e] != 0).count();
        let mut mask = vec![0u64; 16 * 2 * W];
        let mut fingerprint_delta = vec![0u64; 16];
        for (e, (mx, mz)) in masks.iter().enumerate() {
            mask[(e * 2) * W..(e * 2 + 1) * W].copy_from_slice(mx);
            mask[(e * 2 + 1) * W..(e * 2 + 2) * W].copy_from_slice(mz);
            fingerprint_delta[e] = fingerprints.fingerprint(mx, mz);
        }
        let mut received_slot = vec![NO_REMOTE; 16];
        for (k, r) in remote.iter().enumerate() {
            assert!(
                r.entry < entries,
                "remote entry {} outside the table",
                r.entry
            );
            received_slot[r.entry] = k as u32;
        }
        let mut out = Self {
            mode,
            entries,
            support_size,
            qubit_0,
            qubit_1,
            rotation_cos,
            rotation_sin,
            amplitude,
            mask,
            nonzero_patterns,
            bucket_delta: vec![0u32; 16],
            fingerprint_delta,
            received_slot,
            received_entries: remote.len(),
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
        if self.support_size > 0 {
            local_delta |= bit(mx, self.qubit_0) | bit(mz, self.qubit_0) << 1;
        }
        if self.support_size > 1 {
            local_delta |= bit(mx, self.qubit_1) << 2 | bit(mz, self.qubit_1) << 3;
        }
        (0..LOCAL_DIM)
            .filter(|&s| (self.nonzero_patterns[e] >> s) & 1 != 0)
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
        t.amplitude.fill(0.0);
        t.mask.fill(0);
        t.nonzero_patterns.fill(0);
        t.bucket_delta.fill(0);
        t.fingerprint_delta.fill(0);
        t.received_slot.fill(NO_REMOTE);
        t.masks.clear();
        let a = LOCAL_DIM * 2;
        for (j, &e) in entries.iter().enumerate() {
            t.amplitude[j * a..(j + 1) * a].copy_from_slice(&self.amplitude[e * a..(e + 1) * a]);
            t.mask[j * 2 * W..(j + 1) * 2 * W]
                .copy_from_slice(&self.mask[e * 2 * W..(e + 1) * 2 * W]);
            t.nonzero_patterns[j] = self.nonzero_patterns[e];
            t.bucket_delta[j] = self.bucket_delta[e];
            t.fingerprint_delta[j] = self.fingerprint_delta[e];
            t.masks.push(self.masks[e]);
        }
        t.received_entries = 0;
        t.fanout = (0..t.entries)
            .filter(|&j| t.nonzero_patterns[j] != 0)
            .count();
        t.key_preserving = false;
        t.permutation = false;
        t.entry_of.fill(NO_ENTRY);
        t
    }

    /// The distinct bucket deltas of the local entries, ascending, always containing `0`: the input to `Gf2Span::new`, equal to the plan's `local_bucket_deltas`.
    pub(crate) fn bucket_deltas(&self) -> Vec<u32> {
        let mut v: Vec<u32> = (0..self.entries)
            .filter(|&e| self.received_slot[e] == NO_REMOTE)
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
