//! GF(2) spans and coset enumeration over the bucket index space.
//!
//! `Gf2Span` is the engine's coset index algebra (ARCHITECTURE.md §Engine): `engine::bucketed` uses it to enumerate `span(h(D))`'s cosets, the parallel unit each layer gathers, sorts and merges in place.

use crate::pauli_sum::hash::B_MAX_BITS;

/// Highest set bit of a nonzero word.
#[inline]
fn highest_bit(v: u32) -> u32 {
    debug_assert!(v != 0, "highest_bit: zero has no highest set bit");
    31 - v.leading_zeros()
}

/// Software `pext`: gather the bits of `value` at the set positions of `mask` into the low bits of the result, preserving ascending bit order.
///
/// Not `core::arch::x86_64::_pext_u32` — that is BMI2-only and this runs once per bucket index, off any inner loop.
#[inline]
fn pext(value: u32, mask: u32) -> u32 {
    let mut out = 0u32;
    let mut k = 0u32;
    let mut m = mask;
    while m != 0 {
        let bit = m & m.wrapping_neg();
        if value & bit != 0 {
            out |= 1 << k;
        }
        k += 1;
        m ^= bit;
    }
    out
}

/// The GF(2)-linear span of a channel's bucket deltas, with its cosets enumerated.
///
/// # Why a span and not the delta set itself
///
/// A channel's bucket-delta set `h(D)` is *usually* already a subspace — for every built-in channel it is, because `D` is a subspace of key deltas and `h` is linear.
/// But [`Channel`](crate::channel::Channel) is an open trait: a research channel may declare any delta set it likes, and nothing forces it to be XOR-closed.
/// Output bucket `β'` reads inputs `β' ⊕ δ` for `δ ∈ h(D)`, and those read sets only partition the bucket space when the delta set is a subspace.
/// So this type takes the span: `span(h(D)) ⊇ h(D)` is always a subspace, its cosets always partition, and reading a few buckets that happen to contribute nothing costs time, never correctness.
///
/// # The coset picture
///
/// Cosets of `span(h(D))` partition the `2^bits` bucket indices into `2^bits / 2^r` classes of `2^r` each, and every output bucket in a coset reads exactly that same coset — so one coset is a closed, independent unit of work that can be gathered once and merged back in place.
/// This type maps between a bucket index `β` and its `(coset, member)` coordinates: [`rep_of`](Self::rep_of) names the coset, [`coord_of`](Self::coord_of) names the member within it, and [`perm_index`](Self::perm_index) packs the pair into a contiguous index.
///
/// # Basis convention
///
/// `basis` is a **reduced row echelon** basis, stored in **ascending pivot significance**: `basis[0]` carries the least-significant pivot, `basis[r-1]` the most.
/// Each vector's pivot is its own highest set bit, and a pivot bit is set in exactly one basis vector.
/// Combination indices follow the same order: bit `j` of an index `i` selects `basis[j]`, so `member(rep, i) = rep ⊕ ⨁_{j ∈ bits(i)} basis[j]`.
#[derive(Clone, Debug)]
pub(crate) struct Gf2Span {
    /// Reduced echelon basis, ascending by pivot bit.
    basis: Vec<u32>,
    /// OR of the pivot bits — one bit per basis vector.
    pivot_mask: u32,
    /// Bucket-index width: indices live in `0..2^bits`.
    bits: u8,
    /// `(1 << bits) - 1`.
    space_mask: u32,
    /// `space_mask & !pivot_mask`: the free bits a representative ranges over.
    nonpivot_mask: u32,
}

impl Gf2Span {
    /// Build the span of `deltas` inside a `bits`-wide bucket index space.
    ///
    /// `deltas` is a [`Prepared::bucket_deltas`](crate::channel::prepared::Prepared::bucket_deltas) result — sorted, deduplicated, always containing `0` — but it is *not* required to be XOR-closed; that is the whole reason this computes a span.
    ///
    /// # Panics
    ///
    /// Panics if `bits > B_MAX_BITS`, or if any delta has a bit set outside the `bits`-wide index space.
    /// Both are contract violations by the caller, not input-dependent conditions, so they are checked in release too — the check runs once per layer, not per term.
    pub(crate) fn new(deltas: &[u32], bits: u8) -> Self {
        assert!(
            bits <= B_MAX_BITS,
            "Gf2Span: bits {bits} exceeds B_MAX_BITS {B_MAX_BITS}"
        );
        let space_mask = if bits == 0 { 0 } else { (1u32 << bits) - 1 };

        let mut basis: Vec<u32> = Vec::new();
        let mut pivot_mask = 0u32;

        for &d in deltas {
            assert!(
                d & !space_mask == 0,
                "Gf2Span: delta {d} has bits outside the {bits}-bit bucket space"
            );
            // Reduce by the existing basis. The basis is kept reduced, so pivot bit `p_j` is set in `basis[j]` alone: XORing one basis vector clears its own pivot bit and disturbs no other pivot bit.
            // Any iteration order therefore clears every pivot bit in one pass.
            let mut v = d;
            for &b in &basis {
                if v & (1 << highest_bit(b)) != 0 {
                    v ^= b;
                }
            }
            if v == 0 {
                // Dependent on what we already have; the span is unchanged.
                continue;
            }
            // `v` has no pivot bit set, so its highest set bit is a fresh pivot.
            let p = highest_bit(v);
            // Back-substitute so the new pivot is again unique to one vector.
            // Every `b` has highest bit `p_b > p` (a `b` with `p_b < p` cannot carry bit `p` at all), so this does not move any pivot.
            for b in basis.iter_mut() {
                if *b & (1 << p) != 0 {
                    *b ^= v;
                }
            }
            let idx = basis.partition_point(|&b| highest_bit(b) < p);
            basis.insert(idx, v);
            pivot_mask |= 1 << p;
        }

        Self {
            basis,
            pivot_mask,
            bits,
            space_mask,
            nonpivot_mask: space_mask & !pivot_mask,
        }
    }

    /// Dimension `r` of the span.
    #[inline]
    pub(crate) fn r(&self) -> usize {
        self.basis.len()
    }

    /// Buckets per coset, `2^r`. This is the task's gather/scatter width.
    #[inline]
    pub(crate) fn coset_size(&self) -> usize {
        1usize << self.basis.len()
    }

    /// Number of cosets, `2^bits / 2^r`. This is the parallel task count.
    #[inline]
    pub(crate) fn num_cosets(&self) -> usize {
        (1usize << self.bits) >> self.basis.len()
    }

    /// Is `beta` the canonical representative of its coset?
    ///
    /// # The theorem
    ///
    /// With a *reduced* echelon basis, the coset member whose pivot bits are all clear is unique, and it is the coset's integer minimum.
    ///
    /// *Unique:* two members differ by a nonempty basis combination, and such a combination has at least one pivot bit set (the highest pivot in the combination is contributed by exactly one vector, reducedness), so two distinct members cannot both have all pivot bits clear.
    ///
    /// *Minimal:* let `p*` be the highest pivot in a nonempty combination.
    /// Every vector in the combination has its highest set bit at its own pivot `≤ p*`, so the combination has no bits above `p*` and does have bit `p*`.
    /// XORing it into a pivot-clear `rep` leaves everything above `p*` alone and flips bit `p*` from 0 to 1, which strictly increases the value.
    #[inline]
    pub(crate) fn is_rep(&self, beta: u32) -> bool {
        beta & self.pivot_mask == 0
    }

    /// The canonical representative of `beta`'s coset.
    ///
    /// This **reduces** `beta` by the basis; it is *not* `beta & !pivot_mask`.
    /// A reduced echelon basis vector still carries non-pivot bits below its pivot, so clearing a pivot bit by masking gives a value in a different coset.
    /// With `basis = {0b110}` and `beta = 0b100`, masking yields `0b000`, while the coset is `{0b100, 0b010}` and the representative is `0b010`.
    #[inline]
    pub(crate) fn rep_of(&self, beta: u32) -> u32 {
        debug_assert!(
            beta & !self.space_mask == 0,
            "Gf2Span::rep_of: beta outside the bucket space"
        );
        let mut v = beta;
        // Pivot bits are disjoint across basis vectors, so one pass in any order clears them all.
        for &b in &self.basis {
            if v & (1 << highest_bit(b)) != 0 {
                v ^= b;
            }
        }
        v
    }

    /// The unique index `i` with `member(0, i) == delta`, for `delta` in the span.
    ///
    /// Because the basis is reduced, coordinates are *read off* rather than solved for: pivot bit `p_j` is set in `basis[j]` and no other, so bit `j` of the coordinate vector is bit `p_j` of `delta`.
    /// Compressing `delta` over `pivot_mask` in ascending bit order therefore yields `i` directly.
    #[inline]
    pub(crate) fn coord_of(&self, delta: u32) -> u32 {
        debug_assert!(
            self.rep_of(delta) == 0,
            "Gf2Span::coord_of: delta {delta} is not in the span"
        );
        pext(delta, self.pivot_mask)
    }

    /// The position of representative `rep` among all representatives in ascending order.
    ///
    /// Representatives are exactly the indices with every pivot bit clear, i.e. the free choices of the `bits - r` non-pivot bits.
    /// Compressing `rep` over those positions is an order-preserving bijection onto `0..num_cosets()`.
    #[inline]
    pub(crate) fn rank_of_rep(&self, rep: u32) -> u32 {
        debug_assert!(
            self.is_rep(rep),
            "Gf2Span::rank_of_rep: {rep} is not a representative"
        );
        pext(rep, self.nonpivot_mask)
    }

    /// The bucket index `beta` renumbered so that a coset occupies a contiguous run: `p(β) = (rank_of_rep(rep(β)) << r) | coord_of(β ⊕ rep(β))`.
    ///
    /// A bijection on `0..2^bits`. Coset `c` owns `c << r .. (c + 1) << r`, and within a run the low `r` bits are the basis coordinates — which is what makes `member(rep, i) ⊕ δ == member(rep, i ⊕ coord_of(δ))` an O(1) scatter-target computation for the layer engine.
    #[inline]
    pub(crate) fn perm_index(&self, beta: u32) -> u32 {
        let rep = self.rep_of(beta);
        (self.rank_of_rep(rep) << self.basis.len()) | self.coord_of(beta ^ rep)
    }
}

#[cfg(test)]
mod tests;
