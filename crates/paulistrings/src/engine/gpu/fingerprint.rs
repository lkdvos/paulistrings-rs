//! The per-term GF(2)-linear fingerprint `g(v) = G·v` and its host oracle for `kernels/fingerprint.cu`.
//!
//! `g` only orders records inside a device bucket; identity is always decided on the full key.

use crate::rng::{mix64, SPLITMIX_GAMMA};

/// Mixed into the hash seed before drawing the fingerprint rows, so `G` is unrelated to the `Gf2Hash` and `PartitionRows` rows of the same seed.
const FINGERPRINT_SALT: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// Rows of `G`, one bit of `g` each.
pub(crate) const FP_ROWS: usize = 64;

/// The 64 rows of `G` as `(x-mask, z-mask)` pairs.
#[derive(Clone, Debug)]
pub(crate) struct FingerprintRows<const W: usize> {
    rows_x: Vec<[u64; W]>,
    rows_z: Vec<[u64; W]>,
}

impl<const W: usize> FingerprintRows<W> {
    /// Rows drawn by splitmix64 from `hash_seed ^ FINGERPRINT_SALT`.
    // Not xorshift: research/FINDINGS.md §`Gf2Hash` rows are splitmix64, not xorshift successors
    pub(crate) fn new(hash_seed: u64) -> Self {
        let mut state = hash_seed ^ FINGERPRINT_SALT;
        let mut next = || {
            state = state.wrapping_add(SPLITMIX_GAMMA);
            mix64(state)
        };
        let mut rows_x = Vec::with_capacity(FP_ROWS);
        let mut rows_z = Vec::with_capacity(FP_ROWS);
        for _ in 0..FP_ROWS {
            rows_x.push(std::array::from_fn::<u64, W, _>(|_| next()));
            rows_z.push(std::array::from_fn::<u64, W, _>(|_| next()));
        }
        Self { rows_x, rows_z }
    }

    /// The unmasked 64-bit `g(x, z)`; the device applies its compile-time `FP_BITS` mask on top.
    pub(crate) fn fingerprint(&self, x: &[u64; W], z: &[u64; W]) -> u64 {
        let mut out = 0u64;
        for r in 0..FP_ROWS {
            let mut parity = 0u64;
            for w in 0..W {
                parity ^= (x[w] & self.rows_x[r][w]) ^ (z[w] & self.rows_z[r][w]);
            }
            out |= u64::from(parity.count_ones() & 1) << r;
        }
        out
    }

    /// The rows in the prelude's device layout.
    pub(crate) fn flat(&self) -> Vec<u64> {
        let mut v = Vec::with_capacity(FP_ROWS * 2 * W);
        for r in 0..FP_ROWS {
            v.extend_from_slice(&self.rows_x[r]);
            v.extend_from_slice(&self.rows_z[r]);
        }
        v
    }
}

#[cfg(test)]
mod tests;
