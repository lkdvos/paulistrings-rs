use super::*;
use crate::channel::Prepare;
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

proptest! {
    /// Cosets partition the bucket space, and `(rep, coord)` round-trips.
    #[test]
    fn reps_partition_the_bucket_space((bits, deltas) in span_input()) {
        let span = Gf2Span::new(&deltas, bits);
        let n = 1u32 << bits;

        let reps: Vec<u32> = (0..n).filter(|&x| span.is_representative(x)).collect();
        prop_assert_eq!(reps.len(), (n as usize) >> span.dimension());
        prop_assert_eq!(reps.len(), span.num_cosets());

        for beta in 0..n {
            let rep = span.representative_of(beta);
            prop_assert!(span.is_representative(rep));
            prop_assert!(rep < n);
            let i = span.coordinate_of(beta ^ rep);
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
                p >> span.dimension(),
                span.rank_of_representative(span.representative_of(beta))
            );
        }
    }
}

// ---- the coset dimension a real dense-PTM channel actually gets ----

/// For a two-qubit channel `rank(h(D))` depends on the hash rows, so one Haar SU(4) gets different coset widths on different supports under one seed.
#[test]
fn coset_dimension_is_the_delta_span_rank_capped_by_the_bucket_bits() {
    use crate::channel::PtmChannel;
    use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
    use crate::pauli_sum::Gf2Hash;
    use crate::test_support::{haar_su4_matrix, support_delta_rank};

    let channel = PtmChannel::unitary_2q(0, 1, haar_su4_matrix());

    for bits in 0..=9u8 {
        // W = 2: full rank from `bits = 7`, the engine's own floor.
        let h2 = Gf2Hash::<2>::new(128, bits, DEFAULT_HASH_SEED);
        let want2 = support_delta_rank(&h2, &[0, 1]).min(bits as usize);
        let prep2 = Prepare::<2>::prepare(&channel, &h2, false).expect("prepare W=2");
        assert_eq!(
            Gf2Span::new(&prep2.bucket_deltas(), bits).dimension(),
            want2,
            "W=2 at {bits} bits"
        );

        // W = 1: the same rows in word 0, so the same rank.
        let h1 = Gf2Hash::<1>::new(64, bits, DEFAULT_HASH_SEED);
        let want1 = support_delta_rank(&h1, &[0, 1]).min(bits as usize);
        let prep1 = Prepare::<1>::prepare(&channel, &h1, false).expect("prepare W=1");
        assert_eq!(
            Gf2Span::new(&prep1.bucket_deltas(), bits).dimension(),
            want1,
            "W=1 at {bits} bits"
        );
    }

    // At the default bucket floor the support `(0, 7)` gets 8 members against 16 elsewhere, at both widths.
    let h2 = Gf2Hash::<2>::new(128, 7, DEFAULT_HASH_SEED);
    let h1 = Gf2Hash::<1>::new(64, 7, DEFAULT_HASH_SEED);
    let deficient = PtmChannel::unitary_2q(0, 7, haar_su4_matrix());
    let s2 = Gf2Span::new(
        &Prepare::<2>::prepare(&channel, &h2, false)
            .unwrap()
            .bucket_deltas(),
        7,
    );
    let s1 = Gf2Span::new(
        &Prepare::<1>::prepare(&deficient, &h1, false)
            .unwrap()
            .bucket_deltas(),
        7,
    );
    assert_eq!((s2.coset_size(), s2.num_cosets()), (16, 8));
    assert_eq!((s1.coset_size(), s1.num_cosets()), (8, 16));
}

impl Gf2Span {
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
}
