//! `Gf2Span`, the GF(2) span of a channel's bucket deltas and the coset index algebra over it, and `CosetOrder`, the coset-contiguous bucket order a layer runs in.
//!
//! The bucket hash is linear, so output bucket `β′` reads only the input buckets `β′ ⊕ h(D)` for a channel's key deltas `D`.
//! A coset of the span of `h(D)` is therefore a closed task: its output buckets read only its own input buckets and no other coset writes them, which is why a layer needs no synchronization inside it.
//! The span rather than `h(D)` itself, since a custom channel's delta set need not be XOR-closed.
//! Where several deltas share a bucket delta the gather walks them all, so correctness never depends on the draw of the hash rows.

use rayon::prelude::*;

use crate::pauli_sum::hash::INDEX_BITS;

/// Positions from which [`CosetOrder::rebuild`] tabulates on the Rayon pool: the serial table costs tens of ns per position, a fork-join tens of µs.
pub(crate) const PARALLEL_REBUILD_MIN_POSITIONS: usize = 4096;

/// Highest set bit of a nonzero word.
fn highest_bit(v: u32) -> u32 {
    debug_assert!(v != 0, "highest_bit: zero has no highest set bit");
    31 - v.leading_zeros()
}

/// `pext`: the bits of `value` under `mask`, packed into the low bits.
fn pext(value: u32, mask: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if bmi2::is_available() {
        // SAFETY: the CPU reports BMI2.
        return unsafe { bmi2::pext(value, mask) };
    }
    pext_software(value, mask)
}

/// `pdep`, the inverse of [`pext`] on `mask`: the low bits of `value` scattered into the set bits of `mask`.
fn pdep(value: u32, mask: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if bmi2::is_available() {
        // SAFETY: the CPU reports BMI2.
        return unsafe { bmi2::pdep(value, mask) };
    }
    pdep_software(value, mask)
}

fn pext_software(value: u32, mask: u32) -> u32 {
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

fn pdep_software(value: u32, mask: u32) -> u32 {
    let mut out = 0u32;
    let mut in_bit = 0u32;
    let mut remaining = mask;
    while remaining != 0 {
        let bit = remaining & remaining.wrapping_neg();
        if value & (1 << in_bit) != 0 {
            out |= bit;
        }
        in_bit += 1;
        remaining ^= bit;
    }
    out
}

/// Hardware `pext`/`pdep`, chosen at run time since the build does not assume BMI2.
#[cfg(target_arch = "x86_64")]
mod bmi2 {
    use std::arch::x86_64::{_pdep_u32, _pext_u32};

    /// Whether the CPU reports BMI2; Zen 1 and Zen 2 microcode it slower than the software loop, which this accepts since the callers are per bucket, not per term.
    pub(super) fn is_available() -> bool {
        std::is_x86_feature_detected!("bmi2")
    }

    #[target_feature(enable = "bmi2")]
    pub(super) fn pext(value: u32, mask: u32) -> u32 {
        _pext_u32(value, mask)
    }

    #[target_feature(enable = "bmi2")]
    pub(super) fn pdep(value: u32, mask: u32) -> u32 {
        _pdep_u32(value, mask)
    }
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
    /// The bucket-space bits outside `pivot_mask`: the free bits a representative ranges over.
    nonpivot_mask: u32,
}

impl Gf2Span {
    /// The span of `deltas` in a `bits`-wide index space; panics if `bits` exceeds the index width or a delta lies outside it.
    pub(crate) fn new(deltas: &[u32], bits: u8) -> Self {
        assert!(bits <= INDEX_BITS, "Gf2Span: bits {bits} exceeds the {INDEX_BITS}-bit index");
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

        Self { basis, pivot_mask, bits, nonpivot_mask: space_mask & !pivot_mask }
    }

    /// Dimension `r` of the span.
    pub(crate) fn dimension(&self) -> usize {
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
    fn is_representative(&self, beta: u32) -> bool {
        beta & self.pivot_mask == 0
    }

    /// The representative of `beta`'s coset.
    // Not `beta & !pivot_mask`: basis vectors carry non-pivot bits too, so masking can leave the coset.
    pub(crate) fn representative_of(&self, beta: u32) -> u32 {
        debug_assert!(
            beta >> self.bits == 0,
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
    pub(crate) fn coordinate_of(&self, delta: u32) -> u32 {
        debug_assert!(
            self.representative_of(delta) == 0,
            "Gf2Span::coordinate_of: delta {delta} is not in the span"
        );
        pext(delta, self.pivot_mask)
    }

    /// The position of `representative` among all representatives in ascending order.
    fn rank_of_representative(&self, representative: u32) -> u32 {
        debug_assert!(
            self.is_representative(representative),
            "Gf2Span::rank_of_representative: {representative} is not a representative"
        );
        pext(representative, self.nonpivot_mask)
    }

    /// The representative of coset `coset`, the `coset`-th smallest, so its members are positions `coset << r ..` of [`Self::permuted_index`].
    pub(crate) fn representative_at(&self, coset: u32) -> u32 {
        debug_assert!(
            (coset as usize) < self.num_cosets(),
            "Gf2Span::representative_at: coset {coset} out of range"
        );
        pdep(coset, self.nonpivot_mask)
    }

    /// Member `i` of the coset with representative `rep`; bit `j` of `i` selects `basis[j]`.
    pub(crate) fn member(&self, rep: u32, i: u32) -> u32 {
        debug_assert!(
            (i as usize) < self.coset_size(),
            "Gf2Span::member: index {i} beyond coset size"
        );
        let mut v = rep;
        for (j, &b) in self.basis.iter().enumerate() {
            if (i >> j) & 1 == 1 {
                v ^= b;
            }
        }
        v
    }

    /// `beta` renumbered so coset `c` owns `c << r .. (c + 1) << r`, with the member coordinate in the low `r` bits.
    pub(crate) fn permuted_index(&self, beta: u32) -> u32 {
        let representative = self.representative_of(beta);
        (self.rank_of_representative(representative) << self.basis.len())
            | self.coordinate_of(beta ^ representative)
    }
}

/// A layer's coset-contiguous bucket order: coset `c` owns positions `c << r .. (c + 1) << r` ([`Gf2Span::permuted_index`]).
///
/// Built once per layer and shared by the coset loop, the export pass and the exchange's chunking; empty, i.e. the identity, when `r = 0`.
#[derive(Clone, Debug, Default)]
pub(crate) struct CosetOrder {
    /// `permutation[β]` is bucket `β`'s position.
    permutation: Vec<u32>,
    /// `inverse[p]` is the bucket at position `p`; built only on request.
    inverse: Vec<u32>,
    positions: usize,
    dimension: u32,
}

/// Refill `table` with `f(0..len)`, on the Rayon pool when `parallel`.
fn tabulate(
    table: &mut Vec<u32>,
    len: usize,
    parallel: bool,
    f: impl Fn(u32) -> u32 + Sync + Send,
) {
    if parallel {
        (0..len)
            .into_par_iter()
            .with_min_len(PARALLEL_REBUILD_MIN_POSITIONS / 8)
            .map(|index| f(index as u32))
            .collect_into_vec(table);
    } else {
        table.clear();
        table.extend((0..len).map(|index| f(index as u32)));
    }
}

impl CosetOrder {
    /// Re-aim the order at `span`, keeping the allocations; the inverse is built only when `with_inverse`.
    pub(crate) fn rebuild(&mut self, span: &Gf2Span, with_inverse: bool) {
        self.positions = 1usize << span.bits;
        self.dimension = span.dimension() as u32;
        self.permutation.clear();
        self.inverse.clear();
        if self.dimension == 0 {
            return;
        }
        let parallel = self.positions >= PARALLEL_REBUILD_MIN_POSITIONS;
        tabulate(&mut self.permutation, self.positions, parallel, |beta| span.permuted_index(beta));
        if with_inverse {
            // Read off the span rather than scattered from `permutation`, so the table fills in parallel.
            let r = self.dimension;
            let member_mask = (1u32 << r) - 1;
            tabulate(&mut self.inverse, self.positions, parallel, |position| {
                span.member(span.representative_at(position >> r), position & member_mask)
            });
        }
    }

    /// Bucket `beta`'s position.
    pub(crate) fn position_of(&self, beta: u32) -> u32 {
        if self.permutation.is_empty() {
            beta
        } else {
            self.permutation[beta as usize]
        }
    }

    /// The bucket at position `position`, the inverse of [`Self::position_of`]; the order must have been built `with_inverse`.
    pub(crate) fn bucket_at(&self, position: u32) -> u32 {
        debug_assert!(
            self.inverse.len() == self.permutation.len(),
            "CosetOrder::bucket_at: the inverse was not built",
        );
        if self.inverse.is_empty() {
            position
        } else {
            self.inverse[position as usize]
        }
    }

    /// Positions, i.e. buckets.
    pub(crate) fn positions(&self) -> usize {
        self.positions
    }

    /// `r`, the `log2` of the coset size.
    pub(crate) fn dimension(&self) -> usize {
        self.dimension as usize
    }

    /// Capacity of the permutation, for the footprint tests.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.permutation.capacity() + self.inverse.capacity()
    }
}

#[cfg(test)]
mod tests;
