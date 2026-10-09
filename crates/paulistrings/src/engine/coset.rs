//! `Gf2Span`, the GF(2) span of a channel's bucket deltas and the coset index algebra over it (ARCHITECTURE.md §Engine).

use crate::pauli_sum::hash::B_MAX_BITS;

/// Highest set bit of a nonzero word.
fn highest_bit(v: u32) -> u32 {
    debug_assert!(v != 0, "highest_bit: zero has no highest set bit");
    31 - v.leading_zeros()
}

/// Software `pext`, since BMI2 is not assumed and this runs off any inner loop.
fn pext(value: u32, mask: u32) -> u32 {
    let mut out = 0u32;
    let mut out_bit = 0u32;
    let mut remaining = mask;
    while remaining != 0 {
        let bit = remaining & remaining.wrapping_neg();
        if value & bit != 0 {
            out |= 1 << out_bit;
        }
        out_bit += 1;
        remaining ^= bit;
    }
    out
}

/// The span of a channel's bucket deltas, which need not be XOR-closed, and the map between a bucket index and its `(coset, member)` coordinates.
// `basis` is reduced row echelon, ascending by pivot (its highest set bit), and each pivot bit is set in exactly one vector; bit `j` of a member index selects `basis[j]`.
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
    /// The span of `deltas` in a `bits`-wide index space; panics if `bits > B_MAX_BITS` or a delta lies outside it.
    pub(crate) fn new(deltas: &[u32], bits: u8) -> Self {
        assert!(
            bits <= B_MAX_BITS,
            "Gf2Span: bits {bits} exceeds B_MAX_BITS {B_MAX_BITS}"
        );
        let space_mask = if bits == 0 { 0 } else { (1u32 << bits) - 1 };

        let mut basis: Vec<u32> = Vec::new();
        let mut pivot_mask = 0u32;

        for &delta in deltas {
            assert!(
                delta & !space_mask == 0,
                "Gf2Span: delta {delta} has bits outside the {bits}-bit bucket space"
            );
            // Pivot bits are disjoint across basis vectors, so one pass in any order clears them all.
            let mut reduced = delta;
            for &vector in &basis {
                if reduced & (1 << highest_bit(vector)) != 0 {
                    reduced ^= vector;
                }
            }
            if reduced == 0 {
                continue;
            }
            let pivot = highest_bit(reduced);
            // Back-substitute so the new pivot is unique to one vector; only vectors with a higher pivot can carry bit `p`.
            for vector in basis.iter_mut() {
                if *vector & (1 << pivot) != 0 {
                    *vector ^= reduced;
                }
            }
            let position = basis.partition_point(|&vector| highest_bit(vector) < pivot);
            basis.insert(position, reduced);
            pivot_mask |= 1 << pivot;
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
    pub(crate) fn r(&self) -> usize {
        self.basis.len()
    }

    /// Buckets per coset, `2^r`.
    pub(crate) fn coset_size(&self) -> usize {
        1usize << self.basis.len()
    }

    /// Number of cosets, `2^bits / 2^r`.
    pub(crate) fn num_cosets(&self) -> usize {
        (1usize << self.bits) >> self.basis.len()
    }

    /// Whether `beta` is its coset's representative: the unique member with every pivot bit clear, which is also the coset's minimum.
    pub(crate) fn is_representative(&self, beta: u32) -> bool {
        beta & self.pivot_mask == 0
    }

    /// The representative of `beta`'s coset.
    // Not `beta & !pivot_mask`: basis vectors carry non-pivot bits too, so masking can leave the coset.
    pub(crate) fn representative_of(&self, beta: u32) -> u32 {
        debug_assert!(
            beta & !self.space_mask == 0,
            "Gf2Span::representative_of: beta outside the bucket space"
        );
        let mut reduced = beta;
        for &vector in &self.basis {
            if reduced & (1 << highest_bit(vector)) != 0 {
                reduced ^= vector;
            }
        }
        reduced
    }

    /// The member index of `delta` in the span, read off its pivot bits since the basis is reduced.
    pub(crate) fn coord_of(&self, delta: u32) -> u32 {
        debug_assert!(
            self.representative_of(delta) == 0,
            "Gf2Span::coord_of: delta {delta} is not in the span"
        );
        pext(delta, self.pivot_mask)
    }

    /// The position of `representative` among all representatives in ascending order.
    pub(crate) fn rank_of_representative(&self, representative: u32) -> u32 {
        debug_assert!(
            self.is_representative(representative),
            "Gf2Span::rank_of_representative: {representative} is not a representative"
        );
        pext(representative, self.nonpivot_mask)
    }

    /// `beta` renumbered so coset `c` owns `c << r .. (c + 1) << r`, with the member coordinate in the low `r` bits.
    pub(crate) fn permuted_index(&self, beta: u32) -> u32 {
        let representative = self.representative_of(beta);
        (self.rank_of_representative(representative) << self.basis.len())
            | self.coord_of(beta ^ representative)
    }
}

#[cfg(test)]
mod tests;
