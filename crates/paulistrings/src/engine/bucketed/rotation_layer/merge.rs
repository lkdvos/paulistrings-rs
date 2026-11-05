//! The rotation layer's merge of a bucket's partners into the bucket, in place when its capacity allows.

use crate::coefficient::Coefficient;
use crate::engine::merge::merge_identity_and_rest;
use crate::pauli_sum::storage::{BucketColumns, ColumnsRef};
use crate::truncation::TruncationPolicy;

/// Merge the strictly ascending `partners` into the strictly ascending `bucket`, summing equal keys bucket-first and keeping each output iff it is nonzero and `keep_term` passes; returns whether it ran in place.
///
/// In place needs spare capacity for every partner that is a new key: with room for every partner the hits go uncounted and the merge starts at that bound, else they are counted; without room the merge runs into `merged`, which is swapped in, as the general path does.
/// `dirty`, when given, lists (as the scan packs them) the only rows whose coefficients changed, every other row already passing `keep_term`; a bucket with few such rows and partners merges by [`merge_sparse_in_place`].
pub(super) fn merge_partners<const W: usize, C: Coefficient, T>(
    bucket: &mut BucketColumns<W, C>,
    partners: ColumnsRef<'_, W, C>,
    policy: &T,
    merged: &mut BucketColumns<W, C>,
    dirty: Option<&[u32]>,
) -> bool
where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let bound = bucket.len() + partners.coeff.len();
    let spare = bucket.x.capacity().min(bucket.z.capacity()).min(bucket.coeff.capacity());
    if let Some(dirty) = dirty {
        let events = partners.coeff.len() + dirty.len();
        if spare >= bound && events * SPARSE_MERGE_SEGMENT <= bucket.len() {
            merge_sparse_in_place(bucket, partners, dirty, policy, bound);
            return true;
        }
    }
    let total = if spare >= bound { bound } else { bound - count_hits(bucket.view(), partners) };
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

/// Fewest rows per partner or dirty row at which [`merge_sparse_in_place`] beats the per-row merge.
const SPARSE_MERGE_SEGMENT: usize = 8;

/// [`merge_backward_in_place`] for a bucket whose rows outside `dirty` keep their coefficients and already pass `keep_term`: the clean rows between consecutive events move up as blocks, and only partners and dirty rows take the per-row step; rows below the lowest event are not touched.
pub(super) fn merge_sparse_in_place<const W: usize, C: Coefficient, T>(
    bucket: &mut BucketColumns<W, C>,
    partners: ColumnsRef<'_, W, C>,
    dirty: &[u32],
    policy: &T,
    total: usize,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let ColumnsRef { x: p_x, z: p_z, coeff: p_c } = partners;
    let (len, extra) = (bucket.len(), p_c.len());
    debug_assert_eq!(total, len + extra);
    debug_assert!(bucket.x.capacity() >= total);
    debug_assert!(bucket.z.capacity() >= total);
    debug_assert!(bucket.coeff.capacity() >= total);
    debug_assert!(dirty.windows(2).all(|pair| pair[0] >> 4 < pair[1] >> 4));
    let x = bucket.x.as_mut_ptr();
    let z = bucket.z.as_mut_ptr();
    let c = bucket.coeff.as_mut_ptr();
    // As in `merge_backward_in_place`; `d` counts the unread dirty rows, each below `i`, and the gap `w − i` never falls below the unread partners `j`.
    let (mut i, mut j, mut d, mut w) = (len, extra, dirty.len(), total);
    // SAFETY: every read is below `i ≤ len`; every write and block destination lies in `[i, total)`, and `ptr::copy` allows the overlap of a block with its destination.
    unsafe {
        let key = |row: usize| (*x.add(row), *z.add(row));
        let emit = |w: &mut usize, key_x: [u64; W], key_z: [u64; W], sum: C| {
            if sum != C::ZERO && policy.keep_term(&key_x, &key_z, sum) {
                *w -= 1;
                x.add(*w).write(key_x);
                z.add(*w).write(key_z);
                c.add(*w).write(sum);
            }
        };
        let dirty_row = |d: usize| (dirty[d - 1] >> 4) as usize;
        while j > 0 || d > 0 {
            // Rows above the next dirty row and above the next partner's key are clean and already ordered.
            let mut low = if d > 0 { dirty_row(d) + 1 } else { 0 };
            if j > 0 {
                let target = (p_x[j - 1], p_z[j - 1]);
                let mut high = i;
                while low < high {
                    let middle = low + (high - low) / 2;
                    if key(middle) <= target {
                        low = middle + 1;
                    } else {
                        high = middle;
                    }
                }
            }
            let block = i - low;
            if block > 0 && w != i {
                std::ptr::copy(x.add(low), x.add(w - block), block);
                std::ptr::copy(z.add(low), z.add(w - block), block);
                std::ptr::copy(c.add(low), c.add(w - block), block);
            }
            w -= block;
            i = low;
            let order = match (i > 0, j > 0) {
                (true, true) => key(i - 1).cmp(&(p_x[j - 1], p_z[j - 1])),
                (true, false) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Less,
            };
            match order {
                std::cmp::Ordering::Less => {
                    j -= 1;
                    emit(&mut w, p_x[j], p_z[j], p_c[j]);
                }
                std::cmp::Ordering::Equal => {
                    i -= 1;
                    j -= 1;
                    d -= usize::from(d > 0 && dirty_row(d) == i);
                    let (key_x, key_z) = key(i);
                    emit(&mut w, key_x, key_z, *c.add(i) + p_c[j]);
                }
                std::cmp::Ordering::Greater => {
                    debug_assert!(d > 0 && dirty_row(d) == i - 1);
                    i -= 1;
                    d -= 1;
                    let (key_x, key_z) = key(i);
                    emit(&mut w, key_x, key_z, *c.add(i));
                }
            }
        }
        let out = total - w;
        if w != i {
            std::ptr::copy(x.add(w), x.add(i), out);
            std::ptr::copy(z.add(w), z.add(i), out);
            std::ptr::copy(c.add(w), c.add(i), out);
        }
        bucket.x.set_len(i + out);
        bucket.z.set_len(i + out);
        bucket.coeff.set_len(i + out);
    }
}

/// Keys `partners` shares with `bucket`, both strictly ascending.
fn count_hits<const W: usize, C: Coefficient>(
    bucket: ColumnsRef<'_, W, C>,
    partners: ColumnsRef<'_, W, C>,
) -> usize {
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

/// [`merge_partners`] in the bucket's own buffer: written from the back of `total`, at least the length before drops, then shifted to the front over the slots hits and drops left.
pub(super) fn merge_backward_in_place<const W: usize, C: Coefficient, T>(
    bucket: &mut BucketColumns<W, C>,
    partners: ColumnsRef<'_, W, C>,
    policy: &T,
    total: usize,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let ColumnsRef { x: p_x, z: p_z, coeff: p_c } = partners;
    let (len, extra) = (bucket.len(), p_c.len());
    debug_assert!(total >= len + extra - count_hits(bucket.view(), partners));
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
        let emit = |w: &mut usize, key_x: [u64; W], key_z: [u64; W], sum: C| {
            if sum != C::ZERO && policy.keep_term(&key_x, &key_z, sum) {
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
            if sum != C::ZERO && policy.keep_term(&key_x, &key_z, sum) {
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
