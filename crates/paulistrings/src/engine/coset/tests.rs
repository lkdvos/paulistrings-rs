use super::*;
use proptest::prelude::*;
use std::collections::HashSet;

/// `(bits, deltas)` with `deltas` in the contract's shape: in range, sorted, deduplicated, containing `0`, but *not* necessarily XOR-closed.
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
        assert!(span.is_rep(beta));
        assert_eq!(span.rep_of(beta), beta);
        assert_eq!(span.member(beta, 0), beta);
        assert_eq!(span.coord_of(0), 0);
        assert_eq!(span.rank_of_rep(beta), beta);
        assert_eq!(span.perm_index(beta), beta);
    }
}

#[test]
fn echelon_basis_is_reduced() {
    // 0b01010 = 0b00110 ^ 0b01100, so the third input is dependent and the
    // rank is 3, not 4.
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
    // And the OR of the pivots is what `is_rep` tests against.
    let mask = pivots.iter().fold(0u32, |m, &p| m | (1 << p));
    for beta in 0u32..32 {
        assert_eq!(span.is_rep(beta), beta & mask == 0);
    }
}

#[test]
fn rep_of_reduces_rather_than_masks() {
    // basis = {0b110}: pivot bit 2, but bit 1 rides along below it.
    let span = Gf2Span::new(&[0, 0b110], 3);
    assert_eq!(span.basis(), &[0b110]);
    assert_eq!(span.r(), 1);

    // The coset of 0b100 is {0b100, 0b010}; the naive mask `beta & !0b100`
    // gives 0b000, which is in a *different* coset entirely.
    assert_eq!(span.rep_of(0b100), 0b010);
    assert_ne!(span.rep_of(0b100), 0b100 & !(1 << 2));
    assert!(span.is_rep(0b010));
    assert_eq!(span.member(0b010, 1), 0b100);
    // 0b000 is its own coset's rep, and that coset is {0b000, 0b110}.
    assert_eq!(span.rep_of(0b000), 0b000);
    assert_eq!(span.rep_of(0b110), 0b000);
}

#[test]
fn non_subspace_input_is_covered() {
    // {0, a, b} with a ^ b absent from the input: a custom channel may hand
    // us exactly this, and the coset partition needs a ^ b in the span.
    let (a, b) = (0b0011u32, 0b0101u32);
    let span = Gf2Span::new(&[0, a, b], 4);
    assert_eq!(span.r(), 2);
    assert_eq!(span.coset_size(), 4);

    let ab = a ^ b;
    assert_eq!(span.rep_of(ab), 0, "a ^ b = {ab:#b} must be in the span");
    assert_eq!(span.member(0, span.coord_of(ab)), ab);

    // The partition is still a partition: 4 cosets of 4, covering 0..16.
    let mut seen: HashSet<u32> = HashSet::new();
    let reps: Vec<u32> = (0u32..16).filter(|&x| span.is_rep(x)).collect();
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

        let reps: Vec<u32> = (0..n).filter(|&x| span.is_rep(x)).collect();
        prop_assert_eq!(reps.len(), (n as usize) >> span.r());
        prop_assert_eq!(reps.len(), span.num_cosets());

        for beta in 0..n {
            let rep = span.rep_of(beta);
            prop_assert!(span.is_rep(rep));
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
                prop_assert_eq!(span.rep_of(m), rep);
                prop_assert!(seen.insert(m));
            }
        }
        prop_assert_eq!(seen.len(), n as usize);
    }

    /// The pivot-clear representative is its coset's integer minimum.
    #[test]
    fn rep_is_the_integer_minimum_of_its_coset((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        for rep in (0..1u32 << bits).filter(|&x| span.is_rep(x)) {
            for i in 0..span.coset_size() as u32 {
                prop_assert!(span.member(rep, i) >= rep);
            }
        }
    }

    /// The O(1) scatter-target identity the coset engine will use: XORing a
    /// delta onto a coset member is an XOR on the member *index*.
    #[test]
    fn run_index_xor_identity((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        let size = span.coset_size() as u32;
        for &delta in &deltas {
            let c = span.coord_of(delta);
            for rep in (0..1u32 << bits).filter(|&x| span.is_rep(x)) {
                for i in 0..size {
                    prop_assert_eq!(
                        span.member(rep, i) ^ delta,
                        span.member(rep, i ^ c)
                    );
                }
            }
        }
    }

    /// `perm_index` renumbers the bucket space without losing or aliasing
    /// anything, so a coset is a contiguous run of `2^r` slots.
    #[test]
    fn perm_index_is_a_bijection((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        let n = 1u32 << bits;
        let mut hit = vec![false; n as usize];
        for beta in 0..n {
            let p = span.perm_index(beta);
            prop_assert!(p < n);
            prop_assert!(!hit[p as usize], "perm_index collision at {}", beta);
            hit[p as usize] = true;
            // The run a bucket lands in is its coset's rank.
            prop_assert_eq!(
                p >> span.r(),
                span.rank_of_rep(span.rep_of(beta))
            );
        }
    }
}

// ---- the coset dimension a real dense-PTM channel actually gets ----

/// `r` is `min(rank(h(D)), bits)`, and for a two-qubit channel `rank(h(D))` is a property of the *hash rows*, not of the channel — so the same Haar SU(4) block gets a 16-member coset on one support and an 8-member one on another under the same seed.
///
/// This is the link between `pauli_sum::hash`'s rank tests and the dense-PTM sort's cost (`engine::merge`): the per-run sort's comparison count sits at its `log2(fanout)` floor exactly when `r` is full, and `r` is what this pins.
/// See `research/FINDINGS.md`.
#[test]
fn coset_dimension_is_the_delta_span_rank_capped_by_the_bucket_bits() {
    use crate::channel::{Channel, GeneralUnitary2Q};
    use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
    use crate::pauli_sum::Gf2Hash;
    use crate::test_support::{haar_su4_matrix, support_delta_rank};

    let ch = GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix());

    for bits in 0..=9u8 {
        // W = 2: full rank from `bits = 7`, the engine's own floor.
        let h2 = Gf2Hash::<2>::new(128, bits, DEFAULT_HASH_SEED);
        let want2 = support_delta_rank(&h2, &[0, 1]).min(bits as usize);
        let prep2 = Channel::<2>::prepare(&ch, &h2, false).expect("prepare W=2");
        assert_eq!(
            Gf2Span::new(&prep2.bucket_deltas(), bits).r(),
            want2,
            "W=2 at {bits} bits"
        );

        // W = 1: the same rows in word 0, so the same rank.
        let h1 = Gf2Hash::<1>::new(64, bits, DEFAULT_HASH_SEED);
        let want1 = support_delta_rank(&h1, &[0, 1]).min(bits as usize);
        let prep1 = Channel::<1>::prepare(&ch, &h1, false).expect("prepare W=1");
        assert_eq!(
            Gf2Span::new(&prep1.bucket_deltas(), bits).r(),
            want1,
            "W=1 at {bits} bits"
        );
    }

    // At the default bucket-count floor two supports get different coset widths from the same gate — 16 members against 8.
    // A rank draw of this kind, not the key width, is behind the "W = 1 sort defect" observed elsewhere; see `research/FINDINGS.md`.
    // Regenerated for the splitmix64 row draw: the deficient support is now `(0, 7)` at both widths, where `(0, 1)` used to be deficient at `W = 1` only.
    let h2 = Gf2Hash::<2>::new(128, 7, DEFAULT_HASH_SEED);
    let h1 = Gf2Hash::<1>::new(64, 7, DEFAULT_HASH_SEED);
    let deficient = GeneralUnitary2Q::from_matrix(0, 7, haar_su4_matrix());
    let s2 = Gf2Span::new(
        &Channel::<2>::prepare(&ch, &h2, false)
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

    /// Member `i` of the coset with representative `rep`.
    ///
    /// Bit `j` of `i` selects `basis[j]` (ascending pivot significance), so `i = 0` is `rep` itself and `i` ranges over `0..coset_size()`.
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
