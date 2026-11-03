//! The rotation layer: a two-delta table `{0, g}` applied to each bucket's own columns by a scan, a linear partner order and a merge-join.
//!
//! Contracts: the result agrees with the general coset fill to floating-point tolerance, every output passes `keep_term` exactly as there, and on the Clifford arm the identity residues `|a₀| ≤ ε·|a₁|` are dropped where the general path keeps them.

use num_complex::Complex64;

use super::coset_fill::{nonzero, CosetScratch, GatherRun};
use super::partner_order::PartnerOrder;
use super::ZERO;
use crate::channel::prepared::{LocalPtm, LOCAL_DIM};
use crate::engine::merge::merge_identity_and_rest;
use crate::pauli_sum::storage::{BucketColumns, ColumnsRef};
use crate::truncation::TruncationPolicy;

use crate::engine::timing::{lap, stamp, tally};

/// A table shaped like a Pauli rotation about `g`: an identity entry and one generator entry.
///
/// On a pattern the generator entry skips, the identity amplitude is exactly one; the patterns it reaches are closed under `⊕ g`, so a partner never lands on a string that keeps its coefficient.
#[derive(Clone, Debug)]
pub(super) struct RotationLayer<'p, const W: usize> {
    ptm: &'p LocalPtm<W>,
    /// `(mask_x, mask_z)` of `g`.
    mask: ([u64; W], [u64; W]),
    /// `a₀[s]`, the identity entry's amplitude.
    identity: [Complex64; LOCAL_DIM],
    /// `a₁[s]`, the amplitude taking pattern `s` to `s ⊕ g`.
    partner: [Complex64; LOCAL_DIM],
    /// Whether every anticommuting pattern has `|a₀| ≤ ε·|a₁|`, so strings move instead of splitting.
    clifford: bool,
    order: PartnerOrder<W>,
}

impl<'p, const W: usize> RotationLayer<'p, W> {
    /// The rotation shape of `ptm`, or `None` if it has another.
    pub(super) fn of(ptm: &'p LocalPtm<W>) -> Option<Self> {
        let [identity, generator] = ptm.deltas() else { return None };
        if identity.local_delta != 0 {
            return None;
        }
        let dim = 1usize << (2 * ptm.k());
        let delta = generator.local_delta as usize;
        let mut clifford = true;
        for s in 0..dim {
            let (a0, a1) = (identity.amplitude[s], generator.amplitude[s]);
            if a1 == ZERO {
                if a0 != Complex64::new(1.0, 0.0) {
                    return None;
                }
            } else {
                if generator.amplitude[s ^ delta] == ZERO {
                    return None;
                }
                // `cos θ` evaluated at a quarter turn is a rounding residue of `sin θ`, not an amplitude.
                clifford &= a0.norm() <= f64::EPSILON * a1.norm();
            }
        }
        let mask = generator.mask();
        Some(Self {
            ptm,
            mask,
            identity: identity.amplitude,
            partner: generator.amplitude,
            clifford,
            order: PartnerOrder::new(&mask.0, &mask.1),
        })
    }

    /// Append each anticommuting string's partner `(k ⊕ g, c·a₁)` to `partners`, then scale the bucket by `a₀` in place, or on the Clifford arm drop the moved strings.
    fn scan(&self, bucket: &mut BucketColumns<W>, partners: &mut BucketColumns<W>) {
        let len = bucket.len();
        // One slot past the exact bound, for `append_if`'s discarded write.
        partners.x.reserve(len + 1);
        partners.z.reserve(len + 1);
        partners.coeff.reserve(len + 1);
        let BucketColumns { x, z, coeff } = bucket;
        let (mask_x, mask_z) = &self.mask;
        let mut kept = 0usize;
        for row in 0..len {
            let (key_x, key_z, c) = (x[row], z[row], coeff[row]);
            let pattern = self.ptm.support_bits(&key_x, &key_z);
            let amplitude = self.partner[pattern];
            let anticommutes = nonzero(amplitude);
            let partner_x = core::array::from_fn(|w| key_x[w] ^ mask_x[w]);
            let partner_z = core::array::from_fn(|w| key_z[w] ^ mask_z[w]);
            // SAFETY: reserved `len + 1` above, and at most `len` rows are published.
            unsafe {
                GatherRun::append_if(partners, anticommutes, partner_x, partner_z, c * amplitude)
            };
            if self.clifford {
                x[kept] = key_x;
                z[kept] = key_z;
                coeff[kept] = c;
                kept += usize::from(!anticommutes);
            } else {
                // Unconditional: a store only for anticommuting strings mispredicts on every active gate.
                coeff[row] = c * self.identity[pattern];
            }
        }
        if self.clifford {
            x.truncate(kept);
            z.truncate(kept);
            coeff.truncate(kept);
        }
    }
}

/// Apply `layer` to one coset, one bucket or the pair `{β, β ⊕ h(g)}`, each partner run merging into the other member.
pub(super) fn fill_rotation_coset<const W: usize, T>(
    members: &mut [BucketColumns<W>],
    layer: &RotationLayer<'_, W>,
    policy: &T,
    scratch: &mut CosetScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let size = members.len();
    debug_assert!(size == 1 || size == 2);
    // The general path's buffers, idle between its cosets: each member's run holds its partners in `rest` and in key order in `identity`, and `old[0]` is the merge fallback.
    scratch.runs.resize_with(size.max(scratch.runs.len()), GatherRun::default);
    if scratch.old.is_empty() {
        scratch.old.push(BucketColumns::default());
    }
    stamp!(stamp);
    // Every scan precedes every merge, so each partner carries its source's unscaled coefficient.
    for (j, bucket) in members.iter_mut().enumerate() {
        let partners = &mut scratch.runs[j ^ (size - 1)].rest;
        partners.clear();
        // Counted as the general path's identity rows, whose keys stay put.
        tally!(scratch.stats.rows_gathered, bucket.len());
        tally!(scratch.stats.rows_id, bucket.len());
        layer.scan(bucket, partners);
    }
    lap!(stamp, scratch.stats.gather_ns);
    for (j, bucket) in members.iter_mut().enumerate() {
        let run = &mut scratch.runs[j];
        tally!(scratch.stats.rows_gathered, run.rest.len());
        tally!(scratch.stats.rows_sorted, run.rest.len());
        layer.order.order(&run.rest, &mut run.identity, scratch.sort.indices());
        lap!(stamp, scratch.stats.sort_ns);
        #[cfg_attr(not(test), allow(unused_variables))]
        let in_place = merge_partners(bucket, run.identity.view(), policy, &mut scratch.old[0]);
        #[cfg(test)]
        {
            scratch.rotation_merges.0 += usize::from(in_place);
            scratch.rotation_merges.1 += usize::from(!in_place);
        }
        lap!(stamp, scratch.stats.merge_ns);
    }
    tally!(scratch.stats.cosets, 1);
    tally!(scratch.stats.runs, size);
}

/// Merge the strictly ascending `partners` into the strictly ascending `bucket`, summing equal keys bucket-first and keeping each output iff it is nonzero and `keep_term` passes; returns whether it ran in place.
///
/// In place needs spare capacity for every partner that is a new key; otherwise the merge runs into `merged`, which is swapped in, as the general path does.
fn merge_partners<const W: usize, T>(
    bucket: &mut BucketColumns<W>,
    partners: ColumnsRef<'_, W>,
    policy: &T,
    merged: &mut BucketColumns<W>,
) -> bool
where
    T: TruncationPolicy<W> + ?Sized,
{
    let total = bucket.len() + partners.coeff.len() - count_hits(bucket.view(), partners);
    let spare = bucket.x.capacity().min(bucket.z.capacity()).min(bucket.coeff.capacity());
    if spare >= total {
        merge_backward_in_place(bucket, partners, policy, total);
        true
    } else {
        merged.clear();
        merge_identity_and_rest(bucket.view(), partners, merged, policy);
        std::mem::swap(bucket, merged);
        merged.clear();
        false
    }
}

/// Keys `partners` shares with `bucket`, both strictly ascending.
fn count_hits<const W: usize>(bucket: ColumnsRef<'_, W>, partners: ColumnsRef<'_, W>) -> usize {
    let (len, extra) = (bucket.x.len(), partners.x.len());
    let (mut i, mut j, mut hits) = (0usize, 0usize, 0usize);
    while i < len && j < extra {
        let order = (bucket.x[i], bucket.z[i]).cmp(&(partners.x[j], partners.z[j]));
        // Branchless steps: which side advances is the merge's coin flip.
        i += usize::from(order.is_le());
        j += usize::from(order.is_ge());
        hits += usize::from(order.is_eq());
    }
    hits
}

/// [`merge_partners`] in the bucket's own buffer: written from the back of `total`, the length before drops, then shifted to the front over the slots drops left.
fn merge_backward_in_place<const W: usize, T>(
    bucket: &mut BucketColumns<W>,
    partners: ColumnsRef<'_, W>,
    policy: &T,
    total: usize,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let ColumnsRef { x: p_x, z: p_z, coeff: p_c } = partners;
    let (len, extra) = (bucket.len(), p_c.len());
    debug_assert_eq!(total, len + extra - count_hits(bucket.view(), partners));
    debug_assert!(bucket.x.capacity() >= total);
    debug_assert!(bucket.z.capacity() >= total);
    debug_assert!(bucket.coeff.capacity() >= total);
    debug_assert!(p_x.windows(2).zip(p_z.windows(2)).all(|(x, z)| (x[0], z[0]) < (x[1], z[1])));
    let x = bucket.x.as_mut_ptr();
    let z = bucket.z.as_mut_ptr();
    let c = bucket.coeff.as_mut_ptr();
    // `i` and `j` count the unread rows of each input, `w` the first written slot, and `h` the hits among the unread rows.
    // A step writes at most one slot per output key it consumes, so `w ≥ i + j − h ≥ i`, since each unread hit pairs an unread partner: a write never lands on an unread bucket row.
    let (mut i, mut j, mut w) = (len, extra, total);
    // SAFETY: every read is below `i ≤ len`, the initialized prefix; every write is at `w - 1 ≥ i` and below `total ≤ capacity`.
    unsafe {
        let emit = |w: &mut usize, key_x: [u64; W], key_z: [u64; W], sum: Complex64| {
            if sum != ZERO && policy.keep_term(&key_x, &key_z, sum) {
                *w -= 1;
                x.add(*w).write(key_x);
                z.add(*w).write(key_z);
                c.add(*w).write(sum);
            }
        };
        while i > 0 && j > 0 {
            let (b_x, b_z) = (*x.add(i - 1), *z.add(i - 1));
            let (q_x, q_z) = (p_x[j - 1], p_z[j - 1]);
            match (b_x, b_z).cmp(&(q_x, q_z)) {
                std::cmp::Ordering::Greater => {
                    i -= 1;
                    emit(&mut w, b_x, b_z, *c.add(i));
                }
                std::cmp::Ordering::Less => {
                    j -= 1;
                    emit(&mut w, q_x, q_z, p_c[j]);
                }
                std::cmp::Ordering::Equal => {
                    i -= 1;
                    j -= 1;
                    emit(&mut w, b_x, b_z, *c.add(i) + p_c[j]);
                }
            }
        }
        while j > 0 {
            j -= 1;
            emit(&mut w, p_x[j], p_z[j], p_c[j]);
        }
        // Below the last write the bucket's rows may already sit in their slots, and are then not rewritten.
        while i > 0 {
            i -= 1;
            let (key_x, key_z, sum) = (*x.add(i), *z.add(i), *c.add(i));
            if sum != ZERO && policy.keep_term(&key_x, &key_z, sum) {
                w -= 1;
                if w != i {
                    x.add(w).write(key_x);
                    z.add(w).write(key_z);
                    c.add(w).write(sum);
                }
            }
        }
        let out = total - w;
        if w > 0 {
            std::ptr::copy(x.add(w), x, out);
            std::ptr::copy(z.add(w), z, out);
            std::ptr::copy(c.add(w), c, out);
        }
        bucket.x.set_len(out);
        bucket.z.set_len(out);
        bucket.coeff.set_len(out);
    }
}

#[cfg(test)]
mod tests;
