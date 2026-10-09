use super::*;
use proptest::prelude::*;
use std::collections::HashSet;

/// `(bits, deltas)`: in range, sorted, deduplicated, containing `0`, but not necessarily XOR-closed.
fn span_input() -> impl Strategy<Value = (u8, Vec<u32>)> {
    (0u8..=8)
        .prop_flat_map(|bits| {
            let hi = 1u32 << bits;
            (Just(bits), prop::collection::vec(0u32..hi, 0..5))
        })
        .prop_map(|(bits, mut deltas)| {
            deltas.push(0);
            deltas.sort_unstable();
            deltas.dedup();
            (bits, deltas)
        })
}

#[test]
fn span_of_zero_is_trivial() {
    let span = Gf2Span::new(&[0], 4);
    assert_eq!(span.r(), 0);
    assert_eq!(span.coset_size(), 1);
    assert_eq!(span.num_cosets(), 16);
    for beta in 0u32..16 {
        assert!(span.is_representative(beta));
        assert_eq!(span.representative_of(beta), beta);
        assert_eq!(span.member(beta, 0), beta);
        assert_eq!(span.coord_of(0), 0);
        assert_eq!(span.rank_of_representative(beta), beta);
        assert_eq!(span.permuted_index(beta), beta);
    }
}

#[test]
fn echelon_basis_is_reduced() {
    // 0b01010 = 0b00110 ^ 0b01100, so the rank is 3, not 4.
    let span = Gf2Span::new(&[0, 0b00110, 0b01010, 0b01100, 0b10001], 5);
    assert_eq!(span.r(), 3);

    let basis = span.basis();
    let pivots: Vec<u32> = basis.iter().map(|&b| highest_bit(b)).collect();
    // Pivots ascend, matching the bit-to-basis-vector convention.
    assert!(pivots.windows(2).all(|w| w[0] < w[1]), "pivots {pivots:?}");
    // Each pivot bit is carried by exactly one basis vector.
    for &p in &pivots {
        let carriers = basis.iter().filter(|&&b| b & (1 << p) != 0).count();
        assert_eq!(carriers, 1, "pivot {p} in basis {basis:?}");
    }
    // The OR of the pivots is what `is_representative` tests against.
    let mask = pivots.iter().fold(0u32, |m, &p| m | (1 << p));
    for beta in 0u32..32 {
        assert_eq!(span.is_representative(beta), beta & mask == 0);
    }
}

#[test]
fn rep_of_reduces_rather_than_masks() {
    // basis = {0b110}: pivot bit 2, but bit 1 rides along below it.
    let span = Gf2Span::new(&[0, 0b110], 3);
    assert_eq!(span.basis(), &[0b110]);
    assert_eq!(span.r(), 1);

    // The coset of 0b100 is {0b100, 0b010}; the mask `beta & !0b100` would give 0b000, in a different coset.
    assert_eq!(span.representative_of(0b100), 0b010);
    assert_ne!(span.representative_of(0b100), 0b100 & !(1 << 2));
    assert!(span.is_representative(0b010));
    assert_eq!(span.member(0b010, 1), 0b100);
    // 0b000 is its own coset's rep, and that coset is {0b000, 0b110}.
    assert_eq!(span.representative_of(0b000), 0b000);
    assert_eq!(span.representative_of(0b110), 0b000);
}

#[test]
fn non_subspace_input_is_covered() {
    // {0, a, b} with a ^ b absent: the coset partition needs a ^ b in the span.
    let (a, b) = (0b0011u32, 0b0101u32);
    let span = Gf2Span::new(&[0, a, b], 4);
    assert_eq!(span.r(), 2);
    assert_eq!(span.coset_size(), 4);

    let ab = a ^ b;
    assert_eq!(
        span.representative_of(ab),
        0,
        "a ^ b = {ab:#b} must be in the span"
    );
    assert_eq!(span.member(0, span.coord_of(ab)), ab);

    // The partition is still a partition: 4 cosets of 4, covering 0..16.
    let mut seen: HashSet<u32> = HashSet::new();
    let reps: Vec<u32> = (0u32..16).filter(|&x| span.is_representative(x)).collect();
    assert_eq!(reps.len(), span.num_cosets());
    for &rep in &reps {
        for i in 0..span.coset_size() as u32 {
            assert!(seen.insert(span.member(rep, i)), "coset overlap");
        }
    }
    assert_eq!(seen.len(), 16);
}

proptest! {
    /// Cosets partition the bucket space, and `(rep, coord)` round-trips.
    #[test]
    fn reps_partition_the_bucket_space((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        let n = 1u32 << bits;

        let reps: Vec<u32> = (0..n).filter(|&x| span.is_representative(x)).collect();
        prop_assert_eq!(reps.len(), (n as usize) >> span.r());
        prop_assert_eq!(reps.len(), span.num_cosets());

        for beta in 0..n {
            let rep = span.representative_of(beta);
            prop_assert!(span.is_representative(rep));
            prop_assert!(rep < n);
            let i = span.coord_of(beta ^ rep);
            prop_assert!((i as usize) < span.coset_size());
            prop_assert_eq!(span.member(rep, i), beta);
        }

        // Every member of every coset is covered exactly once.
        let mut seen: HashSet<u32> = HashSet::new();
        for &rep in &reps {
            for i in 0..span.coset_size() as u32 {
                let m = span.member(rep, i);
                prop_assert!(m < n);
                prop_assert_eq!(span.representative_of(m), rep);
                prop_assert!(seen.insert(m));
            }
        }
        prop_assert_eq!(seen.len(), n as usize);
    }

    /// The pivot-clear representative is its coset's integer minimum.
    #[test]
    fn rep_is_the_integer_minimum_of_its_coset((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        for rep in (0..1u32 << bits).filter(|&x| span.is_representative(x)) {
            for i in 0..span.coset_size() as u32 {
                prop_assert!(span.member(rep, i) >= rep);
            }
        }
    }

    /// XORing a delta onto a coset member is an XOR on the member index.
    #[test]
    fn run_index_xor_identity((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        let size = span.coset_size() as u32;
        for &delta in &deltas {
            let c = span.coord_of(delta);
            for rep in (0..1u32 << bits).filter(|&x| span.is_representative(x)) {
                for i in 0..size {
                    prop_assert_eq!(
                        span.member(rep, i) ^ delta,
                        span.member(rep, i ^ c)
                    );
                }
            }
        }
    }

    /// `permuted_index` is a bijection that makes each coset a contiguous run of `2^r` slots.
    #[test]
    fn perm_index_is_a_bijection((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        let n = 1u32 << bits;
        let mut hit = vec![false; n as usize];
        for beta in 0..n {
            let p = span.permuted_index(beta);
            prop_assert!(p < n);
            prop_assert!(!hit[p as usize], "permuted_index collision at {}", beta);
            hit[p as usize] = true;
            // The run a bucket lands in is its coset's rank.
            prop_assert_eq!(
                p >> span.r(),
                span.rank_of_representative(span.representative_of(beta))
            );
        }
    }
}

// ---- the coset dimension a real dense-PTM channel actually gets ----

/// For a two-qubit channel `rank(h(D))` depends on the hash rows, so one Haar SU(4) gets different coset widths on different supports under one seed.
#[test]
fn coset_dimension_is_the_delta_span_rank_capped_by_the_bucket_bits() {
    use crate::channel::{Channel, GeneralUnitary2Q};
    use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
    use crate::pauli_sum::Gf2Hash;
    use crate::test_support::{haar_su4_matrix, support_delta_rank};

    let channel = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());

    for bits in 0..=9u8 {
        // W = 2: full rank from `bits = 7`, the engine's own floor.
        let h2 = Gf2Hash::<2>::new(128, bits, DEFAULT_HASH_SEED);
        let want2 = support_delta_rank(&h2, &[0, 1]).min(bits as usize);
        let prep2 = Channel::<2>::prepare(&channel, &h2, false).expect("prepare W=2");
        assert_eq!(
            Gf2Span::new(&prep2.bucket_deltas(), bits).r(),
            want2,
            "W=2 at {bits} bits"
        );

        // W = 1: the same rows in word 0, so the same rank.
        let h1 = Gf2Hash::<1>::new(64, bits, DEFAULT_HASH_SEED);
        let want1 = support_delta_rank(&h1, &[0, 1]).min(bits as usize);
        let prep1 = Channel::<1>::prepare(&channel, &h1, false).expect("prepare W=1");
        assert_eq!(
            Gf2Span::new(&prep1.bucket_deltas(), bits).r(),
            want1,
            "W=1 at {bits} bits"
        );
    }

    // At the default bucket floor the support `(0, 7)` gets 8 members against 16 elsewhere, at both widths.
    let h2 = Gf2Hash::<2>::new(128, 7, DEFAULT_HASH_SEED);
    let h1 = Gf2Hash::<1>::new(64, 7, DEFAULT_HASH_SEED);
    let deficient = GeneralUnitary2Q::from_matrix(0, 7, haar_su4_matrix());
    let s2 = Gf2Span::new(
        &Channel::<2>::prepare(&channel, &h2, false)
            .unwrap()
            .bucket_deltas(),
        7,
    );
    let s1 = Gf2Span::new(
        &Channel::<1>::prepare(&deficient, &h1, false)
            .unwrap()
            .bucket_deltas(),
        7,
    );
    assert_eq!((s2.coset_size(), s2.num_cosets()), (16, 8));
    assert_eq!((s1.coset_size(), s1.num_cosets()), (8, 16));
}

impl Gf2Span {
    /// The reduced echelon basis, ascending by pivot bit.
    #[inline]
    pub(crate) fn basis(&self) -> &[u32] {
        &self.basis
    }

    /// Member `i` of the coset with representative `rep`; bit `j` of `i` selects `basis[j]`.
    #[inline]
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
}
