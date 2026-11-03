//! [`PartnerOrder`], which puts a rotation's partner run in key order in linear time.

use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::coefficient::Coefficient;
use crate::pauli_sum::storage::BucketColumns;

/// Set bits a two-qubit key delta can have.
const MAX_LEVELS: usize = 2 * MAX_LOCAL_SUPPORT;

/// The key order of `k ⊕ g` over a run of distinct `k` in key order.
///
/// Two keys first differ at the same bit before and after the XOR, and their order flips exactly when `g` sets that bit.
/// So within each group of keys sharing every bit above `g`'s most significant set bit, the halves on either side of that bit swap, and so on recursively down `g`'s set bits; below the last one the source order stands.
/// That is a fixed block permutation, applied by [`order`](Self::order) from each adjacent pair's first differing bit, with no comparison sort.
#[derive(Clone, Copy, Debug)]
pub(super) struct PartnerOrder<const W: usize> {
    /// `g`'s set bits as significance ranks in the lex key `(x, z)`, 0 the most significant, ascending.
    levels: [u32; MAX_LEVELS],
    depth: usize,
}

impl<const W: usize> PartnerOrder<W> {
    /// The order for `g = (mask_x, mask_z)`, or `None` if it has more than [`MAX_LEVELS`] set bits.
    pub(super) fn try_new(mask_x: &[u64; W], mask_z: &[u64; W]) -> Option<Self> {
        let bits: u32 = mask_x.iter().chain(mask_z).map(|word| word.count_ones()).sum();
        (bits as usize <= MAX_LEVELS).then(|| Self::new(mask_x, mask_z))
    }

    /// The order for the XOR mask `g = (mask_x, mask_z)`, which has at most [`MAX_LEVELS`] set bits.
    pub(super) fn new(mask_x: &[u64; W], mask_z: &[u64; W]) -> Self {
        let mut levels = [0u32; MAX_LEVELS];
        let mut depth = 0;
        for word in 0..2 * W {
            let mut live = if word < W { mask_x[word] } else { mask_z[word - W] };
            while live != 0 {
                let rank = live.leading_zeros();
                assert!(
                    depth < MAX_LEVELS,
                    "PartnerOrder: a key delta has at most {MAX_LEVELS} set bits"
                );
                levels[depth] = 64 * word as u32 + rank;
                depth += 1;
                live &= !(1u64 << (63 - rank));
            }
        }
        Self { levels, depth }
    }

    /// Write `run`, whose keys are `k ⊕ g` for distinct `k` in key order, into `out` in key order; `divergence` is scratch.
    pub(super) fn order<C: Coefficient>(
        &self,
        run: &BucketColumns<W, C>,
        out: &mut BucketColumns<W, C>,
        divergence: &mut Vec<u32>,
    ) {
        out.clear();
        let len = run.len();
        out.x.reserve(len);
        out.z.reserve(len);
        out.coeff.reserve(len);
        divergence.clear();
        divergence.push(0);
        divergence.extend((1..len).map(|row| first_difference(run, row - 1, row)));
        let mut placer = Placer { run, out, divergence, levels: &self.levels[..self.depth] };
        placer.place(0, len, 0);
    }
}

/// The significance rank of the first bit at which rows `a` and `b` differ.
fn first_difference<const W: usize, C: Coefficient>(
    run: &BucketColumns<W, C>,
    a: usize,
    b: usize,
) -> u32 {
    for w in 0..W {
        let diff = run.x[a][w] ^ run.x[b][w];
        if diff != 0 {
            return 64 * w as u32 + diff.leading_zeros();
        }
    }
    for w in 0..W {
        let diff = run.z[a][w] ^ run.z[b][w];
        if diff != 0 {
            return 64 * (W + w) as u32 + diff.leading_zeros();
        }
    }
    unreachable!("PartnerOrder: the run's keys must be distinct")
}

/// One [`PartnerOrder::order`] call; `divergence[i]` is the first differing bit of rows `i - 1` and `i`.
struct Placer<'a, const W: usize, C: Coefficient = num_complex::Complex64> {
    run: &'a BucketColumns<W, C>,
    out: &'a mut BucketColumns<W, C>,
    divergence: &'a [u32],
    levels: &'a [u32],
}

impl<const W: usize, C: Coefficient> Placer<'_, W, C> {
    fn copy(&mut self, low: usize, high: usize) {
        self.out.x.extend_from_slice(&self.run.x[low..high]);
        self.out.z.extend_from_slice(&self.run.z[low..high]);
        self.out.coeff.extend_from_slice(&self.run.coeff[low..high]);
    }

    /// Rows `low..high`, which share every bit above `levels[level - 1]` and that bit, in key order.
    fn place(&mut self, low: usize, high: usize, level: usize) {
        if level == self.levels.len() || high - low < 2 {
            self.copy(low, high);
            return;
        }
        let bit = self.levels[level];
        // Groups share every bit above `bit`; one split by it has its XORed-set half first, which swaps.
        // Runs of groups without a split pass to the next level whole, since its groups only refine them.
        let (mut pending, mut group_start, mut row) = (low, low, low + 1);
        while row < high {
            let d = self.divergence[row];
            if d < bit {
                group_start = row;
            } else if d == bit {
                let split = row;
                let mut end = split + 1;
                while end < high && self.divergence[end] > bit {
                    end += 1;
                }
                debug_assert!(end == high || self.divergence[end] < bit);
                self.place(pending, group_start, level + 1);
                self.place(split, end, level + 1);
                self.place(group_start, split, level + 1);
                pending = end;
                group_start = end;
                row = end;
            }
            row += 1;
        }
        self.place(pending, high, level + 1);
    }
}

#[cfg(test)]
mod tests;
