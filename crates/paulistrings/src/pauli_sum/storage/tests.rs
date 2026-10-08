#[test]
fn bucketed_expectation_agrees_with_the_flat_version() {
    // Not bitwise: partials are combined in bucket order, not global sorted order, and float addition is not associative.
    // The tolerance below is ~1e5 times looser than the observed difference and ~1e5 times tighter than anything physically meaningful.
    for &weight in &[2usize, 4] {
        let sum = rand_low_weight_sum::<2>(20_000, 100, weight, 0xE1 + weight as u64);
        let want = sum.expectation_product_state(ProductState::XPlus);
        for bits in [0u8, 3, 7, 11] {
            let h = Gf2Hash::<2>::new(100, bits, 0xE2);
            let b = sum.clone().with_hash(h);
            let got = b.expectation_product_state(ProductState::XPlus);
            assert!(
                (got - want).norm() < 1e-9,
                "weight={weight} bits={bits}: {got} vs {want}",
            );
        }
    }
}

#[test]
fn bucketed_expectation_covers_all_three_states() {
    let sum = rand_sum::<1>(5000, 20, 0xE3);
    let h = Gf2Hash::<1>::new(20, 5, 0xE4);
    let b = sum.clone().with_hash(h);
    for state in [
        ProductState::XPlus,
        ProductState::YPlus,
        ProductState::ZPlus,
    ] {
        let got = b.expectation_product_state(state);
        let want = sum.expectation_product_state(state);
        assert!((got - want).norm() < 1e-9, "{state:?}: {got} vs {want}");
    }
}

#[test]
fn bucketed_expectation_of_an_empty_sum_is_zero() {
    let h = Gf2Hash::<1>::new(8, 3, 0xE5);
    let b = PauliSum::<1>::empty_with_hash(8, h);
    assert!(
        b.expectation_product_state(ProductState::XPlus)
            .norm()
            .abs()
            < 1e-15
    );
}

use super::*;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::PartitionRows;
use crate::phase::Phase;
use crate::readout::{PauliAxis, ProductBasis, ProductState};
// `Xs64` and `rand_sum` are the canonical fixtures from
// `crate::test_support` — this module's copies were byte-identical.
use crate::test_support::{low_weight_sum, rand_sum, rand_sum_real, Xs64};

/// Low-weight keys — the physically relevant regime, and the one where a badly chosen hash would collapse into one bucket.
fn rand_low_weight_sum<const W: usize>(
    n: usize,
    num_qubits: usize,
    weight: usize,
    seed: u64,
) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for _ in 0..n {
        let mut p = PauliString::<W> {
            x: [0u64; W],
            z: [0u64; W],
        };
        for _ in 0..weight {
            let q = (rng.next_u64() as usize) % num_qubits;
            let bit = 1u64 << (q % 64);
            match rng.next_u64() % 3 {
                0 => p.x[q / 64] |= bit,
                1 => p.z[q / 64] |= bit,
                _ => {
                    p.x[q / 64] |= bit;
                    p.z[q / 64] |= bit;
                }
            }
        }
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, 0.0));
    }
    acc.finalize()
}

/// Same multiset of terms, coefficients bitwise — partition forgotten.
fn assert_same_sum<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>) {
    assert_eq!(a.len(), b.len(), "length");
    assert_eq!(a.num_qubits(), b.num_qubits(), "num_qubits");
    let ta = {
        let mut v: Vec<([u64; W], [u64; W], Complex64)> =
            a.iter().map(|(x, z, c)| (*x, *z, c)).collect();
        v.sort_unstable_by_key(|&(x, z, _)| (x, z));
        v
    };
    let tb = {
        let mut v: Vec<([u64; W], [u64; W], Complex64)> =
            b.iter().map(|(x, z, c)| (*x, *z, c)).collect();
        v.sort_unstable_by_key(|&(x, z, _)| (x, z));
        v
    };
    assert_eq!(ta, tb, "terms");
}

// ---- round trip ----

#[test]
fn round_trip_is_bitwise_identical_w1() {
    let sum = rand_sum::<1>(5000, 64, 0xA1);
    let h = Gf2Hash::<1>::new(64, 7, 0xBEEF);
    let bucketed = sum.clone().with_hash(h);
    bucketed.assert_invariants();
    let back = bucketed;
    back.assert_invariants();
    assert_same_sum(&sum, &back);
}

#[test]
fn round_trip_is_bitwise_identical_w2() {
    let sum = rand_sum::<2>(5000, 128, 0xA2);
    let h = Gf2Hash::<2>::new(128, 9, 0xBEEF);
    let bucketed = sum.clone().with_hash(h);
    bucketed.assert_invariants();
    let back = bucketed;
    back.assert_invariants();
    assert_same_sum(&sum, &back);
}

#[test]
fn round_trip_on_low_weight_input() {
    let sum = rand_low_weight_sum::<2>(4000, 100, 4, 0xA3);
    let h = Gf2Hash::<2>::new(100, 8, 0xBEEF);
    let bucketed = sum.clone().with_hash(h);
    bucketed.assert_invariants();
    assert_same_sum(&sum, &bucketed);
}

#[test]
fn round_trip_at_a_mid_word_qubit_count() {
    // 70 qubits in W=2: word 1 is only partly live.
    let sum = rand_sum::<2>(2000, 70, 0xA4);
    let h = Gf2Hash::<2>::new(70, 6, 0xBEEF);
    let bucketed = sum.clone().with_hash(h);
    bucketed.assert_invariants();
    assert_same_sum(&sum, &bucketed);
}

#[test]
fn round_trip_empty_sum() {
    let sum = PauliSum::<1>::empty(64);
    let h = Gf2Hash::<1>::new(64, 5, 0x1);
    let bucketed = sum.clone().with_hash(h);
    bucketed.assert_invariants();
    assert!(bucketed.is_empty());
    assert_eq!(bucketed.len(), 0);
    assert_same_sum(&sum, &bucketed);
}

#[test]
fn round_trip_single_term() {
    let sum = rand_sum::<1>(1, 64, 0xA5);
    assert_eq!(sum.len(), 1);
    let h = Gf2Hash::<1>::new(64, 8, 0x2);
    let bucketed = sum.clone().with_hash(h);
    bucketed.assert_invariants();
    assert_same_sum(&sum, &bucketed);
}

#[test]
fn round_trip_with_a_single_bucket() {
    // bits = 0: everything in bucket 0, so the canonical order is plain lex and the scatter must already have produced a sorted bucket.
    let sum = rand_sum::<1>(2000, 64, 0xA6);
    let h = Gf2Hash::<1>::new(64, 0, 0x3);
    let bucketed = sum.clone().with_hash(h);
    assert_eq!(bucketed.num_buckets(), 1);
    bucketed.assert_invariants();
    assert_same_sum(&sum, &bucketed);
}

#[test]
fn round_trip_with_more_buckets_than_terms() {
    let sum = rand_sum::<1>(50, 64, 0xA7);
    let h = Gf2Hash::<1>::new(64, 12, 0x4);
    let bucketed = sum.clone().with_hash(h);
    assert_eq!(bucketed.num_buckets(), 4096);
    bucketed.assert_invariants();
    assert_same_sum(&sum, &bucketed);
}

#[test]
fn empty_constructor_matches_empty_sum() {
    let h = Gf2Hash::<2>::new(128, 6, 0x5);
    let b = PauliSum::<2>::empty_with_hash(128, h);
    b.assert_invariants();
    assert_eq!(b.num_buckets(), 64);
    assert_eq!(b.len(), 0);
}

// ---- refine / coarsen ----

#[test]
fn refine_doubles_buckets_and_preserves_content() {
    let sum = rand_sum::<2>(3000, 128, 0xB1);
    let h = Gf2Hash::<2>::new(128, 6, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    let before_len = b.len();

    b.refine();
    assert_eq!(b.num_buckets(), 128);
    assert_eq!(b.len(), before_len);
    // The invariant check is the real assertion: it verifies every term is in its new hash bucket and that each bucket is still sorted.
    b.assert_invariants();
    assert_same_sum(&sum, &b);
}

#[test]
fn refine_splits_each_bucket_into_the_pair_i_and_i_plus_b() {
    let sum = rand_sum::<1>(3000, 64, 0xB2);
    let h = Gf2Hash::<1>::new(64, 5, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    let old_nb = b.num_buckets();
    let old_lens: Vec<usize> = (0..old_nb).map(|i| b.bucket_len(i)).collect();

    b.refine();
    for (i, &old_len) in old_lens.iter().enumerate() {
        assert_eq!(
            b.bucket_len(i) + b.bucket_len(i + old_nb),
            old_len,
            "bucket {i} did not split into (i, i + B)",
        );
    }
}

#[test]
fn coarsen_halves_buckets_and_preserves_content() {
    let sum = rand_sum::<2>(3000, 128, 0xB3);
    let h = Gf2Hash::<2>::new(128, 7, 0xC0DE);
    let mut b = sum.clone().with_hash(h);

    b.coarsen();
    assert_eq!(b.num_buckets(), 64);
    b.assert_invariants();
    assert_same_sum(&sum, &b);
}

#[test]
fn coarsen_merges_the_pair_i_and_i_plus_new_b() {
    let sum = rand_sum::<1>(3000, 64, 0xB4);
    let h = Gf2Hash::<1>::new(64, 6, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    let new_nb = b.num_buckets() / 2;
    let expect: Vec<usize> = (0..new_nb)
        .map(|i| b.bucket_len(i) + b.bucket_len(i + new_nb))
        .collect();

    b.coarsen();
    for (i, &want) in expect.iter().enumerate() {
        assert_eq!(b.bucket_len(i), want, "bucket {i} merge size");
    }
}

#[test]
fn refine_then_coarsen_round_trips() {
    let sum = rand_sum::<2>(2500, 128, 0xB5);
    let h = Gf2Hash::<2>::new(128, 6, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    let lens: Vec<usize> = (0..b.num_buckets()).map(|i| b.bucket_len(i)).collect();

    b.refine();
    b.coarsen();

    assert_eq!(b.num_buckets(), 64);
    let after: Vec<usize> = (0..b.num_buckets()).map(|i| b.bucket_len(i)).collect();
    assert_eq!(lens, after);
    b.assert_invariants();
    assert_same_sum(&sum, &b);
}

#[test]
fn repeated_refine_stays_consistent() {
    let sum = rand_sum::<2>(4000, 128, 0xB6);
    let h = Gf2Hash::<2>::new(128, 2, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    for _ in 0..8 {
        b.refine();
        b.assert_invariants();
    }
    assert_eq!(b.num_buckets(), 1024);
    assert_same_sum(&sum, &b);
}

#[test]
fn refine_and_coarsen_take_the_parallel_path_above_the_threshold() {
    // Above DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK (8192) terms, refine/coarsen go parallel; check that branch produces the same invariants and content as the serial path.
    let n = 10_000;
    assert!(n >= DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK);
    let sum = rand_sum::<2>(n, 128, 0xB7);
    let h = Gf2Hash::<2>::new(128, 2, 0xC0DE);
    let mut b = sum.clone().with_hash(h);

    b.refine();
    b.assert_invariants();
    b.refine();
    b.assert_invariants();
    assert_eq!(b.num_buckets(), 16);
    assert_same_sum(&sum, &b);

    b.coarsen();
    b.assert_invariants();
    b.coarsen();
    b.assert_invariants();
    assert_eq!(b.num_buckets(), 4);
    assert_same_sum(&sum, &b);
}

#[test]
fn refine_and_coarsen_on_an_empty_sum() {
    let h = Gf2Hash::<1>::new(64, 3, 0x7);
    let mut b = PauliSum::<1>::empty_with_hash(64, h);
    b.refine();
    b.assert_invariants();
    b.coarsen();
    b.assert_invariants();
    assert_eq!(b.num_buckets(), 8);
    assert_eq!(b.len(), 0);
}

// ---- rebucket policy ----

#[test]
fn rebucket_grows_toward_the_target() {
    // 8000 terms at target 64 wants ~125 buckets, i.e. 128.
    let sum = rand_sum::<2>(8000, 128, 0xD1);
    let h = Gf2Hash::<2>::new(128, 0, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    b.rebucket(64, 1);
    b.assert_invariants();
    let mean = b.len() / b.num_buckets();
    assert!(
        mean <= 4 * 64,
        "mean {mean} still above the hysteresis band with {} buckets",
        b.num_buckets(),
    );
    assert_same_sum(&sum, &b);
}

#[test]
fn rebucket_never_shrinks() {
    // rebucket only ever grows: 200 terms at target 256 wants far fewer than 1024 buckets, but starting at 1024 must stay at 1024.
    let sum = rand_sum::<1>(200, 64, 0xD2);
    let h = Gf2Hash::<1>::new(64, 10, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    assert_eq!(b.num_buckets(), 1024);
    b.rebucket(256, 1);
    b.assert_invariants();
    assert_eq!(
        b.num_buckets(),
        1024,
        "rebucket shrank from 1024 to {} buckets",
        b.num_buckets(),
    );
    assert_same_sum(&sum, &b);
}

#[test]
fn rebucket_is_a_no_op_when_already_at_the_target() {
    // 6400 terms at target 100 wants exactly 64 buckets, so nothing moves; pins the no-hysteresis behavior (ARCHITECTURE.md §Bucket-Policy).
    let sum = rand_sum::<2>(6400, 128, 0xD3);
    let h = Gf2Hash::<2>::new(128, 6, 0xC0DE); // 64 buckets, mean 100
    let mut b = sum.clone().with_hash(h);
    let before = b.num_buckets();
    b.rebucket(100, 1);
    assert_eq!(
        b.num_buckets(),
        before,
        "rebucket moved when already on target"
    );
}

#[test]
fn rebucket_lands_on_desired_bits_or_stays_at_the_high_water_mark() {
    // `rebucket` only grows, so it converges on `desired_bits` only when the starting partition is at or below that.
    // Otherwise the starting bit count is the high-water mark and survives unchanged; both are `want.max(start)`.
    for &n in &[500usize, 6400, 60_000] {
        let sum = rand_sum::<2>(n, 128, 0xD9 + n as u64);
        let want = desired_bits(sum.len(), 256, 8);
        for start in [0u8, 3, 12] {
            let h = Gf2Hash::<2>::new(128, start, 0xC0DE);
            let mut b = sum.clone().with_hash(h);
            b.rebucket(256, 8);
            assert_eq!(
                b.hash().bits(),
                want.max(start),
                "n={n} start={start}: expected max(want={want}, start={start})",
            );
            b.assert_invariants();
        }
    }
}

#[test]
fn rebucket_keeps_the_high_water_mark_after_len_shrinks() {
    // Grow to a high bucket count from a large sum, then shrink the term count sharply and rebucket again: the grow-only policy says the bucket count is a high-water mark and must not follow the length back down.
    let sum = rand_sum::<2>(60_000, 128, 0xDA1);
    let h = Gf2Hash::<2>::new(128, 0, 0xC0DE);
    let mut b = sum.with_hash(h);
    b.rebucket(256, 8);
    let grown_bits = b.hash().bits();
    assert!(
        grown_bits > 0,
        "sanity: rebucket should have grown from 0 bits"
    );

    // Shrink the term count sharply, well below what would justify
    // `grown_bits` under `desired_bits`.
    b.retain(|_, _, c| c.re > 0.995);
    b.assert_invariants();
    assert!(
        b.len() < 512,
        "sanity: shrink did not reduce len enough ({})",
        b.len(),
    );

    b.rebucket(256, 8);
    assert_eq!(
        b.hash().bits(),
        grown_bits,
        "rebucket shrank the high-water mark from {grown_bits} to {}",
        b.hash().bits(),
    );
    b.assert_invariants();
}

#[test]
fn rebucket_respects_the_parallelism_floor() {
    let sum = rand_sum::<2>(4096, 128, 0xD4);
    let h = Gf2Hash::<2>::new(128, 0, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    // Target far above n, but the floor still demands 32 buckets.
    b.rebucket(1 << 20, 32);
    b.assert_invariants();
    assert!(
        b.num_buckets() >= 32,
        "floor not respected: {} buckets",
        b.num_buckets(),
    );
}

#[test]
fn rebucket_does_not_split_a_tiny_sum_to_hit_the_floor() {
    // With only 10 terms, splitting to 32 buckets is pure overhead; the floor is gated on there being enough work to spread.
    let sum = rand_sum::<1>(10, 64, 0xD5);
    let h = Gf2Hash::<1>::new(64, 0, 0xC0DE);
    let mut b = sum.clone().with_hash(h);
    b.rebucket(1024, 32);
    assert_eq!(b.num_buckets(), 1, "tiny sum was split anyway");
}

// ---- the canonical-order contract ----

#[test]
fn canonical_order_is_bucket_then_key() {
    let sum = rand_sum::<1>(3000, 64, 0xC0);
    for bits in [1u8, 4, 8] {
        let b = sum.clone().with_hash(Gf2Hash::<1>::new(64, bits, 0xC1));
        let h = b.hash().clone();
        let mut prev: Option<(u32, [u64; 1], [u64; 1])> = None;
        for (x, z, _) in b.iter() {
            let bucket = h.bucket_of(x, z);
            if let Some((pb, px, pz)) = prev {
                assert!(
                    (pb, (px, pz)) < (bucket, (*x, *z)),
                    "bits={bits}: (bucket, key) not strictly ascending",
                );
            }
            prev = Some((bucket, *x, *z));
        }
    }
}

#[test]
fn single_bucket_sum_is_plain_lex_sorted() {
    // Below the split threshold the canonical order is lex order — the property every small-sum positional expectation in the crate rests on.
    let sum = rand_sum::<1>(1000, 64, 0xC2);
    assert_eq!(sum.num_buckets(), 1);
    let (x, z, _) = sum.to_arrays();
    for i in 1..x.len() {
        assert!(
            (x[i - 1], z[i - 1]) < (x[i], z[i]),
            "single-bucket sum not lex-sorted at {i}",
        );
    }
}

// ---- S1: canonical iteration / export ----

/// The canonical order as a plain vector, read out of `bucket()` alone.
fn canonical_triples<const W: usize>(b: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut out = Vec::with_capacity(b.len());
    for i in 0..b.num_buckets() {
        let (x, z, c) = b.bucket(i);
        for k in 0..c.len() {
            out.push((x[k], z[k], c[k]));
        }
    }
    out
}

#[test]
fn iter_yields_bucket_then_key_order() {
    let sum = rand_sum::<2>(3000, 128, 0xF1);
    let h = Gf2Hash::<2>::new(128, 6, 0xF2);
    let b = sum.clone().with_hash(h);

    let got: Vec<([u64; 2], [u64; 2], Complex64)> = b.iter().map(|(x, z, c)| (*x, *z, c)).collect();
    assert_eq!(got.len(), b.len());
    assert_eq!(got, canonical_triples(&b));

    // Within a bucket the keys ascend; the boundaries are exactly the bucket lengths, so the concatenation is not globally sorted.
    let mut start = 0usize;
    for i in 0..b.num_buckets() {
        let n = b.bucket_len(i);
        for k in start + 1..start + n {
            assert!(
                (got[k - 1].0, got[k - 1].1) < (got[k].0, got[k].1),
                "bucket {i} not ascending at {k}",
            );
        }
        start += n;
    }
}

#[test]
fn to_arrays_concatenates_buckets_in_index_order() {
    let sum = rand_sum::<1>(2500, 64, 0xF3);
    let h = Gf2Hash::<1>::new(64, 5, 0xF4);
    let b = sum.clone().with_hash(h);

    let (x, z, c) = b.to_arrays();
    let want = canonical_triples(&b);
    assert_eq!(x.len(), want.len());
    for (k, (wx, wz, wc)) in want.iter().enumerate() {
        assert_eq!(x[k], *wx, "x at {k}");
        assert_eq!(z[k], *wz, "z at {k}");
        assert_eq!(c[k], *wc, "coeff at {k}");
    }
}

#[test]
fn single_bucket_to_arrays_is_the_key_sorted_order() {
    // bits = 0 collapses the bucket order onto the global sorted order, so the export and the merge must agree bit for bit.
    let sum = rand_sum::<2>(2000, 128, 0xF5);
    let h = Gf2Hash::<2>::new(128, 0, 0xF6);
    let b = sum.clone().with_hash(h);
    let (x, z, c) = b.to_arrays();
    let want = sorted_triples(&b);
    for (i, &(wx, wz, wc)) in want.iter().enumerate() {
        assert_eq!(x[i], wx, "x column at {i}");
        assert_eq!(z[i], wz, "z column at {i}");
        assert_eq!(c[i], wc, "coeff column at {i}");
    }
}

// ---- S2: keyed lookup ----

#[test]
fn get_hits_and_misses_across_bucket_counts() {
    let sum = rand_sum::<1>(2000, 64, 0xF7);
    // Keys that are definitely absent: rather than gamble, take misses from a disjoint second draw and skip any that happen to collide.
    let other = rand_sum::<1>(2000, 64, 0xF8);

    for bits in [0u8, 3, 7] {
        let h = Gf2Hash::<1>::new(64, bits, 0xF9);
        let b = sum.clone().with_hash(h);
        for (i, (x, z, c)) in sum.iter().enumerate() {
            assert_eq!(
                b.get(x, z),
                Some(c),
                "bits={bits}: miss on present term {i}"
            );
        }
        let mut misses = 0usize;
        for (i, (x, z, _)) in other.iter().enumerate() {
            if sum.get(x, z).is_some() {
                continue;
            }
            misses += 1;
            assert_eq!(b.get(x, z), None, "bits={bits}: hit on absent term {i}",);
        }
        assert!(misses > 1000, "bits={bits}: only {misses} absent probes");
    }
}

#[test]
fn get_w2_word_boundary() {
    // Keys live entirely in word 1, so a lookup that only compared word 0 would confuse them.
    let mut acc = BuildAccumulator::<2>::new(128);
    for q in [64u32, 65, 100, 127] {
        acc.add_term(
            PauliString::<2>::x(q),
            Phase::ONE,
            Complex64::new(q as f64, 0.0),
        );
        acc.add_term(
            PauliString::<2>::z(q),
            Phase::ONE,
            Complex64::new(0.0, q as f64),
        );
    }
    let sum = acc.finalize();
    for bits in [0u8, 4] {
        let h = Gf2Hash::<2>::new(128, bits, 0xFA);
        let b = sum.clone().with_hash(h);
        for q in [64u32, 65, 100, 127] {
            let px = PauliString::<2>::x(q);
            let pz = PauliString::<2>::z(q);
            assert_eq!(
                b.get(&px.x, &px.z),
                Some(Complex64::new(q as f64, 0.0)),
                "bits={bits} X{q}",
            );
            assert_eq!(
                b.get(&pz.x, &pz.z),
                Some(Complex64::new(0.0, q as f64)),
                "bits={bits} Z{q}",
            );
        }
        // X on qubit 63 is a distinct key in word 0 and is absent.
        let absent = PauliString::<2>::x(63);
        assert_eq!(b.get(&absent.x, &absent.z), None);
    }
}

#[test]
fn get_agrees_with_a_map_model() {
    use std::collections::BTreeMap;
    let sum = rand_low_weight_sum::<2>(3000, 100, 3, 0xFB);
    let probes = rand_low_weight_sum::<2>(3000, 100, 3, 0xFC);
    let h = Gf2Hash::<2>::new(100, 6, 0xFD);
    let b = sum.clone().with_hash(h);
    let model: BTreeMap<([u64; 2], [u64; 2]), Complex64> =
        sum.iter().map(|(x, z, c)| ((*x, *z), c)).collect();
    for (i, (x, z, _)) in probes.iter().enumerate() {
        assert_eq!(b.get(x, z), model.get(&(*x, *z)).copied(), "probe {i}");
    }
}

// ---- S3: per-bucket mutators ----

#[test]
fn scale_matches_flat_bitwise() {
    // Scaling is elementwise, so per-bucket and flat orders cannot diverge:
    // the comparison is exact, not toleranced.
    for bits in [0u8, 5] {
        let sum = rand_sum::<2>(3000, 128, 0x101);
        let h = Gf2Hash::<2>::new(128, bits, 0x102);
        let mut b = sum.clone().with_hash(h);
        let mut flat = sum.clone();
        let c = Complex64::new(-0.75, 1.25);
        b.scale(c);
        flat.scale(c);
        b.assert_invariants();
        assert_eq!(b.len(), flat.len());
        assert_same_sum(&flat, &b);
    }
}

#[test]
fn retain_filters_in_place_and_keeps_invariants() {
    for bits in [0u8, 6] {
        let sum = rand_sum::<2>(4000, 128, 0x106);
        let h = Gf2Hash::<2>::new(128, bits, 0x107);
        let mut b = sum.clone().with_hash(h);
        // A predicate that reads the key as well as the coefficient, so a key/coefficient column desync would show up.
        let keep = |x: &[u64; 2], _z: &[u64; 2], c: Complex64| x[0] & 1 == 0 && c.re > 0.0;
        b.retain(keep);
        b.assert_invariants();

        let mut want_x = Vec::new();
        let mut want_z = Vec::new();
        let mut want_c = Vec::new();
        for (x, z, c) in sorted_triples(&sum) {
            if keep(&x, &z, c) {
                want_x.push(x);
                want_z.push(z);
                want_c.push(c);
            }
        }
        assert!(
            !want_x.is_empty(),
            "predicate kept nothing; test is vacuous"
        );
        assert_eq!(b.len(), want_x.len(), "bits={bits} length");
        let got = sorted_triples(&b);
        for (i, t) in got.iter().enumerate() {
            assert_eq!(t.0, want_x[i], "bits={bits} x at {i}");
            assert_eq!(t.1, want_z[i], "bits={bits} z at {i}");
            assert_eq!(t.2, want_c[i], "bits={bits} coeff at {i}");
        }
    }
}

// ---- S4: overlap ----

/// Reference overlap: two-pointer over globally key-sorted triples — the accumulation order a single-bucket sum uses.
fn flat_overlap<const W: usize>(a: &PauliSum<W>, b: &PauliSum<W>) -> Complex64 {
    let ta = sorted_triples(a);
    let tb = sorted_triples(b);
    let mut acc = Complex64::new(0.0, 0.0);
    let (mut i, mut j) = (0usize, 0usize);
    while i < ta.len() && j < tb.len() {
        match (ta[i].0, ta[i].1).cmp(&(tb[j].0, tb[j].1)) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                acc += ta[i].2.conj() * tb[j].2;
                i += 1;
                j += 1;
            }
        }
    }
    acc
}

#[test]
fn overlap_single_bucket_is_bitwise_flat() {
    // One bucket means one two-pointer pass over globally sorted columns — the same additions in the same order as the flat version, so equality is exact.
    let a = rand_low_weight_sum::<2>(3000, 100, 3, 0x201);
    let b = rand_low_weight_sum::<2>(3000, 100, 3, 0x202);
    let h = Gf2Hash::<2>::new(100, 0, 0x203);
    let ba = a.clone().with_hash(h.clone());
    let bb = b.clone().with_hash(h);
    assert_eq!(ba.overlap(&bb), flat_overlap(&a, &b));
    // Self-overlap is the squared norm.
    assert_eq!(ba.overlap(&ba), flat_overlap(&a, &a));
}

#[test]
fn overlap_matches_flat_within_tolerance_across_bits() {
    // Not bitwise past one bucket: partials are combined in bucket order.
    let a = rand_low_weight_sum::<2>(4000, 100, 3, 0x204);
    let b = rand_low_weight_sum::<2>(4000, 100, 3, 0x205);
    let want = flat_overlap(&a, &b);
    for bits in [0u8, 3, 6] {
        let h = Gf2Hash::<2>::new(100, bits, 0x206);
        let ba = a.clone().with_hash(h.clone());
        let bb = b.clone().with_hash(h);
        let got = ba.overlap(&bb);
        if bits == 0 {
            assert_eq!(got, want, "bits=0 must be bitwise");
        }
        // Relative, not absolute: the reordering error is bounded by the accumulated magnitude, which is what a relative bound tracks.
        assert!(
            (got - want).norm() <= 1e-12 * want.norm(),
            "bits={bits}: {got} vs {want}",
        );
    }
    assert!(want.norm() > 0.0, "operands share no keys; test is vacuous");
}

#[test]
fn overlap_with_an_empty_operand_is_zero() {
    let a = rand_sum::<1>(500, 64, 0x207);
    let h = Gf2Hash::<1>::new(64, 4, 0x208);
    let ba = a.clone().with_hash(h.clone());
    let empty = PauliSum::<1>::empty_with_hash(64, h);
    assert_eq!(ba.overlap(&empty), Complex64::new(0.0, 0.0));
    assert_eq!(empty.overlap(&ba), Complex64::new(0.0, 0.0));
}

#[test]
#[should_panic(expected = "bucket count mismatch")]
fn overlap_rejects_a_different_bucket_count() {
    let a = rand_sum::<1>(200, 64, 0x209);
    let ba = a.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0x20A));
    let bb = a.clone().with_hash(Gf2Hash::<1>::new(64, 4, 0x20A));
    let _ = ba.overlap(&bb);
}

#[test]
#[should_panic(expected = "hash mismatch")]
fn overlap_rejects_a_different_hash() {
    let a = rand_sum::<1>(200, 64, 0x20B);
    let ba = a.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0x20C));
    let bb = a.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0x20D));
    let _ = ba.overlap(&bb);
}

// ---- S5: flatten / repartition ----

/// The canonical order, sorted — i.e. the multiset of terms, partition forgotten.
fn sorted_triples<const W: usize>(b: &PauliSum<W>) -> Vec<([u64; W], [u64; W], Complex64)> {
    let mut v = canonical_triples(b);
    v.sort_by(|p, q| (p.0, p.1).cmp(&(q.0, q.1)));
    v
}

#[test]
fn with_hash_round_trips_terms_bitwise() {
    let sum = rand_low_weight_sum::<2>(4000, 100, 3, 0x303);
    let ha = Gf2Hash::<2>::new(100, 5, 0x304);
    let hb = Gf2Hash::<2>::new(100, 8, 0x305);
    let a = sum.clone().with_hash(ha.clone());
    let before = canonical_triples(&a);

    let moved = a.with_hash(hb);
    moved.assert_invariants();
    let back = moved.with_hash(ha);
    back.assert_invariants();

    assert_eq!(canonical_triples(&back), before);
}

#[test]
fn with_hash_only_changes_partition() {
    let sum = rand_sum::<1>(3000, 64, 0x306);
    let a = sum.clone().with_hash(Gf2Hash::<1>::new(64, 4, 0x307));
    let want = sorted_triples(&a);

    for (bits, seed) in [(0u8, 0x308u64), (9, 0x309), (4, 0x30A)] {
        let moved = a.clone().with_hash(Gf2Hash::<1>::new(64, bits, seed));
        moved.assert_invariants();
        assert_eq!(moved.len(), a.len(), "bits={bits} seed={seed} length");
        assert_eq!(moved.num_buckets(), 1usize << bits);
        assert_eq!(
            sorted_triples(&moved),
            want,
            "bits={bits} seed={seed}: term multiset changed",
        );
    }
}

#[test]
fn with_hash_rejects_a_qubit_count_mismatch() {
    let sum = rand_sum::<2>(100, 100, 0x30B);
    let a = sum.clone().with_hash(Gf2Hash::<2>::new(100, 3, 0x30C));
    let err = std::panic::catch_unwind(move || a.with_hash(Gf2Hash::<2>::new(128, 3, 0x30C)));
    assert!(err.is_err(), "expected a num_qubits mismatch panic");
}

#[test]
fn align_same_rows_via_refine_coarsen() {
    // Same rows: aligning must land on exactly the partition a scatter would have built under the target hash, both up and down.
    let sum = rand_low_weight_sum::<2>(4000, 100, 4, 0x30D);
    let seed = 0x30E;
    for &(from, to) in &[(5u8, 9u8), (9, 5), (6, 6), (0, 7), (7, 0)] {
        let a = sum.clone().with_hash(Gf2Hash::<2>::new(100, from, seed));
        let target = Gf2Hash::<2>::new(100, to, seed);
        let got = a.align_to(&target);
        got.assert_invariants();
        assert_eq!(got.hash().bits(), to, "from={from} to={to}");
        let want = sum.clone().with_hash(target);
        assert_eq!(
            canonical_triples(&got),
            canonical_triples(&want),
            "from={from} to={to}: partition differs from a direct build",
        );
    }
}

#[test]
fn align_different_rows_goes_through_with_hash() {
    let sum = rand_low_weight_sum::<2>(3000, 100, 3, 0x30F);
    let a = sum.clone().with_hash(Gf2Hash::<2>::new(100, 6, 0x310));
    let target = Gf2Hash::<2>::new(100, 4, 0x311);
    let got = a.align_to(&target);
    got.assert_invariants();
    assert_eq!(got.hash().seed(), 0x311);
    assert_eq!(got.hash().bits(), 4);
    let want = sum.clone().with_hash(target);
    assert_eq!(canonical_triples(&got), canonical_triples(&want));
}

// ---- S6: add ----

/// Two sums over the same keyspace with heavy key overlap, so `add` sees merges and not just interleaving.
fn overlapping_pair<const W: usize>(
    n: usize,
    num_qubits: usize,
    weight: usize,
    seed: u64,
) -> (PauliSum<W>, PauliSum<W>) {
    (
        rand_low_weight_sum::<W>(n, num_qubits, weight, seed),
        rand_low_weight_sum::<W>(n, num_qubits, weight, seed ^ 0xFFFF),
    )
}

#[test]
fn add_same_hash_is_bitwise_flat_add() {
    // Every surviving coefficient is one `a + b`, computed in the same operand order as the flat merge, so equality is exact even though the partial sums live in different buckets.
    let (a, b) = overlapping_pair::<2>(4000, 100, 3, 0x401);
    let want = a.add(&b);
    for bits in [0u8, 4, 9] {
        let h = Gf2Hash::<2>::new(100, bits, 0x402);
        let ba = a.clone().with_hash(h.clone());
        let bb = b.clone().with_hash(h);
        let got = ba.add(&bb);
        got.assert_invariants();
        assert_eq!(got.len(), want.len(), "bits={bits} length");
        assert_same_sum(&want, &got);
    }
}

#[test]
fn add_mixed_bits_matches_flat_bitwise() {
    let (a, b) = overlapping_pair::<2>(3000, 100, 3, 0x403);
    let want = a.add(&b);
    for &(bits_a, bits_b) in &[(4u8, 8u8), (8, 4), (0, 7), (7, 0), (6, 6)] {
        let ba = a.clone().with_hash(Gf2Hash::<2>::new(100, bits_a, 0x404));
        let bb = b.clone().with_hash(Gf2Hash::<2>::new(100, bits_b, 0x404));
        let got = ba.add(&bb);
        got.assert_invariants();
        assert_same_sum(&want, &got);
    }
}

#[test]
fn add_mixed_seeds_matches_flat_bitwise() {
    let (a, b) = overlapping_pair::<2>(3000, 100, 3, 0x405);
    let want = a.add(&b);
    for &(bits_a, bits_b) in &[(5u8, 5u8), (3, 8)] {
        let ba = a.clone().with_hash(Gf2Hash::<2>::new(100, bits_a, 0x406));
        let bb = b.clone().with_hash(Gf2Hash::<2>::new(100, bits_b, 0x407));
        let got = ba.add(&bb);
        got.assert_invariants();
        assert_same_sum(&want, &got);
    }
}

#[test]
fn add_result_carries_left_hash() {
    let (a, b) = overlapping_pair::<1>(500, 40, 2, 0x408);
    let ba = a.clone().with_hash(Gf2Hash::<1>::new(40, 3, 0x409));
    let bb = b.clone().with_hash(Gf2Hash::<1>::new(40, 7, 0x40A));
    let got = ba.add(&bb);
    assert_eq!(got.hash().bits(), 3, "bits");
    assert_eq!(got.hash().seed(), 0x409, "seed");
    assert_eq!(got.num_buckets(), 8);
    // The operands are untouched.
    assert_eq!(ba.hash().bits(), 3);
    assert_eq!(bb.hash().bits(), 7);
    assert_eq!(bb.hash().seed(), 0x40A);
}

#[test]
fn add_cancels_to_nothing() {
    let a = rand_low_weight_sum::<1>(300, 40, 2, 0x40B);
    let mut neg = a.clone();
    neg.scale(Complex64::new(-1.0, 0.0));
    let ba = a.clone().with_hash(Gf2Hash::<1>::new(40, 5, 0x40C));
    let bn = neg.clone().with_hash(Gf2Hash::<1>::new(40, 2, 0x40D));
    let got = ba.add(&bn);
    got.assert_invariants();
    assert!(
        got.is_empty(),
        "{} terms survived exact cancellation",
        got.len()
    );
}

#[test]
#[should_panic(expected = "num_qubits mismatch")]
fn add_rejects_a_qubit_count_mismatch() {
    let a = rand_sum::<2>(100, 100, 0x40E);
    let b = rand_sum::<2>(100, 128, 0x40F);
    let ba = a.clone().with_hash(Gf2Hash::<2>::new(100, 3, 0x410));
    let bb = b.clone().with_hash(Gf2Hash::<2>::new(128, 3, 0x410));
    let _ = ba.add(&bb);
}

// =====================================================================
// Hand-computed small-sum semantics.
//
// Everything above is differential or a bucket-count sweep: it pins agreement with the single-bucket path, not what either one computes. These pin the values themselves, on inputs small enough to work out by hand.
// =====================================================================

// ---- overlap / expectation ----

fn b10_build<const W: usize>(
    num_qubits: usize,
    terms: &[(PauliString<W>, Complex64)],
) -> PauliSum<W> {
    let mut acc = crate::pauli_sum::accumulator::BuildAccumulator::<W>::with_capacity(
        num_qubits,
        terms.len(),
    );
    for &(pp, c) in terms {
        acc.add_term(pp, crate::phase::Phase::ONE, c);
    }
    acc.finalize()
}

#[test]
fn overlap_with_self_is_the_squared_norm() {
    let a = b10_build::<1>(
        8,
        &[
            (PauliString::<1>::x(0), Complex64::new(2.0, 0.0)),
            (PauliString::<1>::z(3), Complex64::new(0.0, 3.0)),
        ],
    );
    assert!((a.overlap(&a) - Complex64::new(13.0, 0.0)).norm() < 1e-12);
}

#[test]
fn overlap_is_conjugate_symmetric() {
    let a = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(1.0, 2.0))]);
    let b = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(3.0, -1.0))]);
    let ab = a.overlap(&b);
    let ba = b.overlap(&a);
    assert!((ab - ba.conj()).norm() < 1e-12, "{ab} vs conj({ba})");
}

#[test]
fn overlap_of_disjoint_supports_is_zero() {
    let a = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(1.0, 0.0))]);
    let b = b10_build::<1>(8, &[(PauliString::<1>::z(5), Complex64::new(1.0, 0.0))]);
    assert!(a.overlap(&b).norm() < 1e-12);
}

#[test]
fn overlap_only_counts_shared_keys() {
    let a = b10_build::<1>(
        8,
        &[
            (PauliString::<1>::x(0), Complex64::new(2.0, 0.0)),
            (PauliString::<1>::y(1), Complex64::new(5.0, 0.0)),
        ],
    );
    let b = b10_build::<1>(
        8,
        &[
            (PauliString::<1>::x(0), Complex64::new(3.0, 0.0)),
            (PauliString::<1>::z(2), Complex64::new(7.0, 0.0)),
        ],
    );
    assert!((a.overlap(&b) - Complex64::new(6.0, 0.0)).norm() < 1e-12);
}

#[test]
fn overlap_across_a_word_boundary_w2() {
    let a = b10_build::<2>(
        128,
        &[
            (PauliString::<2>::x(3), Complex64::new(1.0, 0.0)),
            (PauliString::<2>::z(70), Complex64::new(2.0, 0.0)),
        ],
    );
    let b = b10_build::<2>(128, &[(PauliString::<2>::z(70), Complex64::new(4.0, 0.0))]);
    assert!((a.overlap(&b) - Complex64::new(8.0, 0.0)).norm() < 1e-12);
}

#[test]
fn identity_coefficient_picks_out_the_trace() {
    let a = b10_build::<1>(
        8,
        &[
            (PauliString::<1>::identity(), Complex64::new(1.5, 0.0)),
            (PauliString::<1>::x(0), Complex64::new(9.0, 0.0)),
        ],
    );
    assert!((a.identity_coefficient() - Complex64::new(1.5, 0.0)).norm() < 1e-12);
    let b = b10_build::<1>(8, &[(PauliString::<1>::x(0), Complex64::new(9.0, 0.0))]);
    assert!(b.identity_coefficient().norm() < 1e-12);
}

#[test]
fn expectation_of_single_paulis_in_each_product_state() {
    let cases = [
        (PauliString::<1>::identity(), 1.0, 1.0, 1.0),
        (PauliString::<1>::x(0), 1.0, 0.0, 0.0),
        (PauliString::<1>::y(0), 0.0, 1.0, 0.0),
        (PauliString::<1>::z(0), 0.0, 0.0, 1.0),
    ];
    for (pp, ex, ey, ez) in cases {
        let s = b10_build::<1>(8, &[(pp, Complex64::new(1.0, 0.0))]);
        assert!(
            (s.expectation_product_state(ProductState::XPlus).re - ex).abs() < 1e-12,
            "XPlus for {pp:?}",
        );
        assert!(
            (s.expectation_product_state(ProductState::YPlus).re - ey).abs() < 1e-12,
            "YPlus for {pp:?}",
        );
        assert!(
            (s.expectation_product_state(ProductState::ZPlus).re - ez).abs() < 1e-12,
            "ZPlus for {pp:?}",
        );
    }
}

#[test]
fn expectation_of_multi_qubit_products() {
    let mut xx = PauliString::<1>::x(0);
    xx.mul_assign(&PauliString::<1>::x(1));
    let mut xz = PauliString::<1>::x(0);
    xz.mul_assign(&PauliString::<1>::z(1));
    let mut yy = PauliString::<1>::y(0);
    yy.mul_assign(&PauliString::<1>::y(1));

    let s = b10_build::<1>(
        8,
        &[
            (xx, Complex64::new(1.0, 0.0)),
            (xz, Complex64::new(10.0, 0.0)),
            (yy, Complex64::new(100.0, 0.0)),
        ],
    );
    assert!((s.expectation_product_state(ProductState::XPlus).re - 1.0).abs() < 1e-12);
    assert!((s.expectation_product_state(ProductState::YPlus).re - 100.0).abs() < 1e-12);
    assert!(s.expectation_product_state(ProductState::ZPlus).re.abs() < 1e-12);
}

#[test]
fn expectation_is_linear_and_keeps_the_imaginary_part() {
    let s = b10_build::<1>(
        8,
        &[
            (PauliString::<1>::x(0), Complex64::new(1.0, 2.0)),
            (PauliString::<1>::x(1), Complex64::new(3.0, -5.0)),
        ],
    );
    let e = s.expectation_product_state(ProductState::XPlus);
    assert!((e - Complex64::new(4.0, -3.0)).norm() < 1e-12);
}

#[test]
fn expectation_across_a_word_boundary_w2() {
    let s = b10_build::<2>(
        128,
        &[
            (PauliString::<2>::x(70), Complex64::new(2.0, 0.0)),
            (PauliString::<2>::z(70), Complex64::new(9.0, 0.0)),
        ],
    );
    assert!((s.expectation_product_state(ProductState::XPlus).re - 2.0).abs() < 1e-12);
    assert!((s.expectation_product_state(ProductState::ZPlus).re - 9.0).abs() < 1e-12);
}

/// The new API must reproduce the observable
/// `examples/ising_2d_quench.rs` hand-rolled, which is why it exists.
#[test]
fn expectation_xplus_matches_the_hand_rolled_reference() {
    let mut rng = 0x2468u64 | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut acc = crate::pauli_sum::accumulator::BuildAccumulator::<1>::with_capacity(16, 500);
    for _ in 0..500 {
        let pp = PauliString::<1> {
            x: [next() & 0xFFFF],
            z: [next() & 0xFFFF],
        };
        let c = Complex64::new((next() as i64 as f64) / (i64::MAX as f64), 0.0);
        acc.add_term(pp, crate::phase::Phase::ONE, c);
    }
    let sum = acc.finalize();

    let mut want = 0.0f64;
    for i in 0..sum.len() {
        if sum.bucket(0).1[i] == [0u64] {
            want += sum.bucket(0).2[i].re;
        }
    }
    let got = sum.expectation_product_state(ProductState::XPlus).re;
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}

// --- non-uniform product states (ProductBasis) ---------------------
//
// Every expected value below is the product of single-qubit Bloch-vector
// components, hand-derived once here:
//
//   |0⟩ = Z+:  ⟨Z⟩ = +1,  ⟨X⟩ = ⟨Y⟩ = 0
//   |1⟩ = Z-:  ⟨Z⟩ = -1,  ⟨X⟩ = ⟨Y⟩ = 0
//   |+⟩ = X+:  ⟨X⟩ = +1,  ⟨Y⟩ = ⟨Z⟩ = 0
//   |-⟩ = X-:  ⟨X⟩ = -1,  ⟨Y⟩ = ⟨Z⟩ = 0
//   |r⟩ = Y+ = (|0⟩ + i|1⟩)/√2:  ⟨Y⟩ = +1,  ⟨X⟩ = ⟨Z⟩ = 0
//   |l⟩ = Y- = (|0⟩ - i|1⟩)/√2:  ⟨Y⟩ = -1,  ⟨X⟩ = ⟨Z⟩ = 0
//
// and ⟨I⟩ = 1 in every state. Off-axis components vanish because two distinct single-qubit Paulis anticommute.

/// Per-qubit label string → [`ProductBasis`], in the alphabet the Python binding accepts (qiskit `Statevector.from_label`): `0`/`1` = Z±, `+`/`-` = X±, `r`/`l` = Y±. Character `i` addresses qubit `i`.
/// Deliberately spelled out here rather than shared with the binding: the test's job is to encode the convention independently.
fn basis_from_labels<const W: usize>(labels: &str) -> ProductBasis<W> {
    ProductBasis::<W>::from_axes(labels.chars().map(|ch| match ch {
        '0' => (PauliAxis::Z, false),
        '1' => (PauliAxis::Z, true),
        '+' => (PauliAxis::X, false),
        '-' => (PauliAxis::X, true),
        'r' => (PauliAxis::Y, false),
        'l' => (PauliAxis::Y, true),
        other => panic!("unexpected label {other:?}"),
    }))
}

/// Differential oracle: `⟨ψ|O|ψ⟩` evaluated one qubit at a time straight from the Bloch table above, sharing no code with the masked scan.
fn naive_labelled_expectation<const W: usize>(sum: &PauliSum<W>, labels: &str) -> Complex64 {
    let mut total = Complex64::new(0.0, 0.0);
    for (x, z, c) in sum.iter() {
        let mut factor = 1.0f64;
        for (q, label) in labels.chars().enumerate() {
            let bx = (x[q / 64] >> (q % 64)) & 1 == 1;
            let bz = (z[q / 64] >> (q % 64)) & 1 == 1;
            factor *= match (bx, bz, label) {
                (false, false, _) => 1.0, // identity factor: no constraint
                (true, false, '+') => 1.0,
                (true, false, '-') => -1.0,
                (false, true, '0') => 1.0,
                (false, true, '1') => -1.0,
                (true, true, 'r') => 1.0,
                (true, true, 'l') => -1.0,
                _ => 0.0, // off-axis Pauli: zero overlap
            };
        }
        total += c * factor;
    }
    total
}

fn expect_close<const W: usize>(sum: &PauliSum<W>, labels: &str, want: f64) {
    let got = sum.expectation_product_basis(&basis_from_labels::<W>(labels));
    assert!(
        (got - Complex64::new(want, 0.0)).norm() < 1e-12,
        "state {labels:?}: got {got}, want {want}",
    );
}

fn single_qubit_labels_against_every_pauli<const W: usize>() {
    // (label, ⟨I⟩, ⟨X⟩, ⟨Y⟩, ⟨Z⟩) — the Bloch table above, transposed.
    let cases = [
        ('0', 1.0, 0.0, 0.0, 1.0),
        ('1', 1.0, 0.0, 0.0, -1.0),
        ('+', 1.0, 1.0, 0.0, 0.0),
        ('-', 1.0, -1.0, 0.0, 0.0),
        ('r', 1.0, 0.0, 1.0, 0.0),
        ('l', 1.0, 0.0, -1.0, 0.0),
    ];
    for (label, ei, ex, ey, ez) in cases {
        let labels = label.to_string();
        for (pauli, want) in [("I", ei), ("X", ex), ("Y", ey), ("Z", ez)] {
            let s = PauliSum::<W>::from_strings(&[(pauli, Complex64::new(1.0, 0.0))]);
            let got = s.expectation_product_basis(&basis_from_labels::<W>(&labels));
            assert!(
                (got - Complex64::new(want, 0.0)).norm() < 1e-12,
                "⟨{label}|{pauli}|{label}⟩ = {got}, want {want} (W={W})",
            );
        }
    }
}

#[test]
fn single_qubit_labels_against_every_pauli_w1() {
    single_qubit_labels_against_every_pauli::<1>();
}

#[test]
fn single_qubit_labels_against_every_pauli_w2() {
    single_qubit_labels_against_every_pauli::<2>();
}

fn multi_qubit_products_compose_per_qubit_signs<const W: usize>() {
    // ⟨01|Z⊗Z|01⟩ = ⟨0|Z|0⟩·⟨1|Z|1⟩ = (+1)(-1) = -1.
    let zz = PauliSum::<W>::from_strings(&[("ZZ", Complex64::new(1.0, 0.0))]);
    expect_close(&zz, "01", -1.0);
    expect_close(&zz, "10", -1.0);
    expect_close(&zz, "00", 1.0);
    expect_close(&zz, "11", 1.0); // (-1)(-1)

    // State |0⟩|+⟩|r⟩: axes Z, X, Y, every sign +1.
    let zxy = PauliSum::<W>::from_strings(&[("ZXY", Complex64::new(1.0, 0.0))]);
    expect_close(&zxy, "0+r", 1.0);
    // X on the Y-axis qubit is off-axis → the whole term drops.
    let zxx = PauliSum::<W>::from_strings(&[("ZXX", Complex64::new(1.0, 0.0))]);
    expect_close(&zxx, "0+r", 0.0);
    // Identity factors are ignored, whatever that qubit's label is.
    let ziy = PauliSum::<W>::from_strings(&[("ZIY", Complex64::new(1.0, 0.0))]);
    expect_close(&ziy, "0+r", 1.0);
    expect_close(&ziy, "0-r", 1.0);
    expect_close(&ziy, "01r", 1.0);

    // State |1⟩|-⟩|l⟩: the same axes with all three signs flipped, so a weight-3 term picks up (-1)^3 = -1.
    expect_close(&zxy, "1-l", -1.0);
    // Only the two non-identity sites' signs count: (-1)·(-1) = +1.
    expect_close(&ziy, "1-l", 1.0);
    // A single flipped site: -1.
    expect_close(&zxy, "1+r", -1.0);
    expect_close(&zxy, "0-r", -1.0);
    expect_close(&zxy, "0+l", -1.0);
    // Two flipped sites: +1.
    expect_close(&zxy, "1-r", 1.0);
}

#[test]
fn multi_qubit_products_compose_per_qubit_signs_w1() {
    multi_qubit_products_compose_per_qubit_signs::<1>();
}

#[test]
fn multi_qubit_products_compose_per_qubit_signs_w2() {
    multi_qubit_products_compose_per_qubit_signs::<2>();
}

fn an_off_axis_pauli_never_matches<const W: usize>() {
    // The subset-match trap: `X` on a Y-axis qubit must not contribute, even though the Y axis has its x-bit set — the match is an equality on both halves of the key, not `x & !ax_x == 0`. ⟨r|X|r⟩ = 0.
    let off_axis = [
        ('r', "X"),
        ('r', "Z"),
        ('l', "X"),
        ('l', "Z"),
        ('+', "Y"),
        ('+', "Z"),
        ('-', "Y"),
        ('0', "X"),
        ('0', "Y"),
        ('1', "X"),
        ('1', "Y"),
    ];
    for (label, pauli) in off_axis {
        let s = PauliSum::<W>::from_strings(&[(pauli, Complex64::new(3.0, -4.0))]);
        let got = s.expectation_product_basis(&basis_from_labels::<W>(&label.to_string()));
        assert!(
            got.norm() < 1e-12,
            "⟨{label}|{pauli}|{label}⟩ = {got}, want 0 (W={W})",
        );
    }
    // Mixed: one off-axis factor kills a term whose other factors match.
    let s = PauliSum::<W>::from_strings(&[("XXX", Complex64::new(1.0, 0.0))]);
    expect_close(&s, "++r", 0.0);
    expect_close(&s, "+++", 1.0);
}

#[test]
fn an_off_axis_pauli_never_matches_w1() {
    an_off_axis_pauli_never_matches::<1>();
}

#[test]
fn an_off_axis_pauli_never_matches_w2() {
    an_off_axis_pauli_never_matches::<2>();
}

#[test]
fn labelled_expectation_is_linear_and_keeps_the_imaginary_part() {
    // ⟨1|Z|1⟩ = -1 and ⟨1|I|1⟩ = +1, so this is -(1+2i) + (3-5i).
    let s = PauliSum::<1>::from_strings(&[
        ("Z", Complex64::new(1.0, 2.0)),
        ("I", Complex64::new(3.0, -5.0)),
    ]);
    let got = s.expectation_product_basis(&basis_from_labels::<1>("1"));
    assert!((got - Complex64::new(2.0, -7.0)).norm() < 1e-12, "{got}");
}

#[test]
fn labels_across_the_word_boundary_are_independent_w2() {
    // 128 qubits, |0…0⟩ except qubit 64, which is |1⟩ — its sign bit lives in word 1 of `neg`, so a word-0-only implementation would miss it.
    let mut labels: String = "0".repeat(128);
    labels.replace_range(64..65, "1");
    let basis = basis_from_labels::<2>(&labels);
    let cases = [
        (PauliString::<2>::z(0), 1.0),   // qubit 0 is |0⟩
        (PauliString::<2>::z(64), -1.0), // qubit 64 is |1⟩
        (PauliString::<2>::x(64), 0.0),  // off-axis on a Z qubit
    ];
    for (p, want) in cases {
        let s = b10_build::<2>(128, &[(p, Complex64::new(1.0, 0.0))]);
        let got = s.expectation_product_basis(&basis);
        assert!(
            (got - Complex64::new(want, 0.0)).norm() < 1e-12,
            "{p:?}: got {got}, want {want}",
        );
    }
    // Z on both sides of the boundary: (+1)·(-1) = -1.
    let mut z0z64 = PauliString::<2>::z(0);
    z0z64.mul_assign(&PauliString::<2>::z(64));
    let s = b10_build::<2>(128, &[(z0z64, Complex64::new(1.0, 0.0))]);
    let got = s.expectation_product_basis(&basis);
    assert!((got - Complex64::new(-1.0, 0.0)).norm() < 1e-12, "{got}");
}

#[test]
fn labelled_expectation_of_an_empty_sum_is_zero() {
    let h = Gf2Hash::<1>::new(8, 3, 0xE6);
    let b = PauliSum::<1>::empty_with_hash(8, h);
    let got = b.expectation_product_basis(&basis_from_labels::<1>("01+-rl01"));
    assert!(got.norm() < 1e-15, "{got}");
}

fn uniform_states_agree_with_their_label_spellings<const W: usize>() {
    let num_qubits = 50 * W;
    let sum = rand_sum::<W>(4000, num_qubits, 0xA40 + W as u64);
    for (state, label) in [
        (ProductState::XPlus, '+'),
        (ProductState::YPlus, 'r'),
        (ProductState::ZPlus, '0'),
    ] {
        let want = sum.expectation_product_state(state);
        let labels: String = std::iter::repeat_n(label, num_qubits).collect();
        let got = sum.expectation_product_basis(&basis_from_labels::<W>(&labels));
        assert!(
            (got - want).norm() < 1e-12,
            "{state:?} vs {label:?}: {got} vs {want} (W={W})",
        );
    }
}

#[test]
fn uniform_states_agree_with_their_label_spellings_w1() {
    uniform_states_agree_with_their_label_spellings::<1>();
}

#[test]
fn uniform_states_agree_with_their_label_spellings_w2() {
    uniform_states_agree_with_their_label_spellings::<2>();
}

fn labelled_expectation_agrees_with_the_naive_reference<const W: usize>() {
    // 33 qubits per word so W=2 straddles the boundary at 64.
    let num_qubits = 33 * W;
    let alphabet: Vec<char> = "01+-rl".chars().collect();
    let mut rng = Xs64::new(0xB40 + W as u64);
    let labels: String = (0..num_qubits)
        .map(|_| alphabet[(rng.next_u64() % 6) as usize])
        .collect();
    for &weight in &[1usize, 2, 3] {
        let sum = low_weight_sum::<W>(3000, num_qubits, weight, 0xB50 + weight as u64);
        let want = naive_labelled_expectation(&sum, &labels);
        let got = sum.expectation_product_basis(&basis_from_labels::<W>(&labels));
        assert!(
            (got - want).norm() < 1e-9,
            "W={W} weight={weight}: {got} vs {want}",
        );
    }
}

#[test]
fn labelled_expectation_agrees_with_the_naive_reference_w1() {
    labelled_expectation_agrees_with_the_naive_reference::<1>();
}

#[test]
fn labelled_expectation_agrees_with_the_naive_reference_w2() {
    labelled_expectation_agrees_with_the_naive_reference::<2>();
}

#[test]
fn bucketed_labelled_expectation_agrees_across_partitions() {
    // The sign parity is accumulated inside a bucket, so a partition change must not move the value (beyond float re-association).
    let alphabet: Vec<char> = "01+-rl".chars().collect();
    let mut rng = Xs64::new(0xB60);
    let labels: String = (0..100)
        .map(|_| alphabet[(rng.next_u64() % 6) as usize])
        .collect();
    let basis = basis_from_labels::<2>(&labels);
    let sum = low_weight_sum::<2>(20_000, 100, 3, 0xB61);
    let want = sum.expectation_product_basis(&basis);
    for bits in [0u8, 3, 7, 11] {
        let h = Gf2Hash::<2>::new(100, bits, 0xB62);
        let b = sum.clone().with_hash(h);
        let got = b.expectation_product_basis(&basis);
        assert!((got - want).norm() < 1e-9, "bits={bits}: {got} vs {want}");
    }
}

#[test]
fn assert_invariants_accepts_bits_within_num_qubits() {
    // num_qubits=50, single term with X on qubit 49 (in range).
    let sum = PauliSum::<1>::from_sorted_columns(
        vec![[1u64 << 49]],
        vec![[0u64; 1]],
        vec![Complex64::new(1.0, 0.0)],
        50,
    );
    sum.assert_invariants();
}

#[test]
#[should_panic(expected = "exceeds num_qubits")]
fn assert_invariants_rejects_bit_beyond_num_qubits() {
    // num_qubits=50, but X bit set at qubit 50 — must panic.
    let sum = PauliSum::<1>::from_sorted_columns(
        vec![[1u64 << 50]],
        vec![[0u64; 1]],
        vec![Complex64::new(1.0, 0.0)],
        50,
    );
    sum.assert_invariants();
}

#[test]
#[should_panic(expected = "exceeds num_qubits")]
fn assert_invariants_rejects_z_bit_beyond_num_qubits() {
    // Same as above but on the Z-part: invariant must check both parts.
    let sum = PauliSum::<1>::from_sorted_columns(
        vec![[0u64; 1]],
        vec![[1u64 << 60]],
        vec![Complex64::new(1.0, 0.0)],
        50,
    );
    sum.assert_invariants();
}

#[test]
#[should_panic(expected = "exceeds num_qubits")]
fn assert_invariants_rejects_bit_in_unused_word() {
    // num_qubits=64 (one full word), W=2. Bit on qubit 64 lives in word 1 and is therefore out of range.
    let sum = PauliSum::<2>::from_sorted_columns(
        vec![[0u64, 1u64]],
        vec![[0u64; 2]],
        vec![Complex64::new(1.0, 0.0)],
        64,
    );
    sum.assert_invariants();
}

// --- keyed lookup (get) -----------------------------------------------

/// Three-term `PauliSum<1>` with sorted, distinct keys `K0 < K1 < K2`.
fn three_term_sum_w1() -> PauliSum<1> {
    // K0 = (x=0, z=1), K1 = (x=1, z=0), K2 = (x=1, z=2). Sorted by lex on (x, z): K0 has smallest x; K1, K2 share x but K1 has smaller z.
    PauliSum::<1>::from_sorted_columns(
        vec![[0u64], [1u64], [1u64]],
        vec![[1u64], [0u64], [2u64]],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
        4,
    )
}

#[test]
fn get_on_empty_is_none() {
    let s = PauliSum::<1>::empty(4);
    assert_eq!(s.get(&[0u64], &[0u64]), None);
}

#[test]
fn get_hits_every_key_and_misses_between() {
    let s = three_term_sum_w1();
    assert_eq!(s.get(&[0u64], &[1u64]), Some(Complex64::new(1.0, 0.0)));
    assert_eq!(s.get(&[1u64], &[0u64]), Some(Complex64::new(2.0, 0.0)));
    assert_eq!(s.get(&[1u64], &[2u64]), Some(Complex64::new(3.0, 0.0)));
    // Below the smallest, in a gap, and above the largest key.
    assert_eq!(s.get(&[0u64], &[0u64]), None);
    assert_eq!(s.get(&[1u64], &[1u64]), None);
    assert_eq!(s.get(&[2u64], &[0u64]), None);
}

#[test]
fn canonical_order_is_lex_x_before_z_on_a_single_bucket() {
    // Two terms with K_a=(x=0, z=5) and K_b=(x=1, z=0). Despite z_a > z_b, x_a < x_b, so K_a < K_b in the canonical (lex) order of a single-bucket sum. A lex-on-x-only order would invert this.
    // `single_bucket_sum_is_plain_lex_sorted` above does not cover this: its keys are random, so two of them essentially never share an `x` and the `z` tiebreak is never exercised.
    let s = PauliSum::<1>::from_sorted_columns(
        vec![[0u64], [1u64]],
        vec![[5u64], [0u64]],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        4,
    );
    s.assert_invariants();
    let (x, z, _) = s.to_arrays();
    assert_eq!((x[0], z[0]), ([0u64], [5u64]));
    assert_eq!((x[1], z[1]), ([1u64], [0u64]));
}

// --- scale() ----------------------------------------------------------

#[test]
fn scale_by_zero_zeros_all_coeffs() {
    let mut s = three_term_sum_w1();
    s.scale(Complex64::new(0.0, 0.0));
    assert_eq!(s.len(), 3);
    for (_, _, c) in s.iter() {
        assert_eq!(c, Complex64::new(0.0, 0.0));
    }
    s.assert_invariants();
}

#[test]
fn scale_by_one_is_identity() {
    let mut s = three_term_sum_w1();
    let (_, _, before) = s.to_arrays();
    s.scale(Complex64::new(1.0, 0.0));
    assert_eq!(s.to_arrays().2, before);
}

#[test]
fn scale_by_i_rotates_phases() {
    let mut s = PauliSum::<1>::from_sorted_columns(
        vec![[0u64], [1u64]],
        vec![[1u64], [0u64]],
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, -3.0)],
        4,
    );
    s.scale(Complex64::new(0.0, 1.0));
    // (2 + 0i) * i = 0 + 2i; (0 - 3i) * i = 3 + 0i.
    assert_eq!(s.bucket(0).2[0], Complex64::new(0.0, 2.0));
    assert_eq!(s.bucket(0).2[1], Complex64::new(3.0, 0.0));
}

// --- add() ------------------------------------------------------------

#[test]
fn add_empty_left_is_other() {
    let a = PauliSum::<1>::empty(4);
    let b = three_term_sum_w1();
    let r = a.add(&b);
    assert_eq!(r.len(), 3);
    assert_eq!(r.to_arrays(), b.to_arrays());
    r.assert_invariants();
}

#[test]
fn add_empty_right_is_self() {
    let a = three_term_sum_w1();
    let b = PauliSum::<1>::empty(4);
    let r = a.add(&b);
    assert_eq!(r.len(), 3);
    assert_eq!(r.to_arrays(), a.to_arrays());
    r.assert_invariants();
}

#[test]
fn add_disjoint_keys_interleaves_in_sort_order() {
    // a has K0=(0,1), K2=(1,2); b has K1=(1,0), K3=(2,0).
    // Lex sort across the union: (0,1) < (1,0) < (1,2) < (2,0).
    let a = PauliSum::<1>::from_sorted_columns(
        vec![[0u64], [1u64]],
        vec![[1u64], [2u64]],
        vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        4,
    );
    let b = PauliSum::<1>::from_sorted_columns(
        vec![[1u64], [2u64]],
        vec![[0u64], [0u64]],
        vec![Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
        4,
    );
    let r = a.add(&b);
    assert_eq!(r.len(), 4);
    let (rx, rz, rc) = r.to_arrays();
    assert_eq!(rx, vec![[0u64], [1u64], [1u64], [2u64]]);
    assert_eq!(rz, vec![[1u64], [0u64], [2u64], [0u64]]);
    assert_eq!(
        rc,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]
    );
    r.assert_invariants();
}

#[test]
fn add_equal_keys_sum_coeffs() {
    let a = three_term_sum_w1();
    let r = a.add(&a);
    assert_eq!(r.len(), 3);
    assert_eq!(r.to_arrays().0, a.to_arrays().0);
    assert_eq!(r.to_arrays().1, a.to_arrays().1);
    for k in 0..3 {
        assert_eq!(
            r.bucket(0).2[k],
            a.bucket(0).2[k] * Complex64::new(2.0, 0.0)
        );
    }
    r.assert_invariants();
}

#[test]
fn add_mixed_cancellation_and_merge() {
    // a = {K1: 1, K2: 2, K3: 3}, b = {K1: -1, K2: 0.5, K4: 4}
    // K1 cancels, K2 sums to 2.5, K3 unique to a, K4 unique to b.
    let a = PauliSum::<1>::from_sorted_columns(
        vec![[0u64], [1u64], [2u64]],
        vec![[0u64], [0u64], [0u64]],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
        4,
    );
    let b = PauliSum::<1>::from_sorted_columns(
        vec![[0u64], [1u64], [3u64]],
        vec![[0u64], [0u64], [0u64]],
        vec![
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(4.0, 0.0),
        ],
        4,
    );
    let r = a.add(&b);
    assert_eq!(r.len(), 3);
    let (rx, rz, rc) = r.to_arrays();
    assert_eq!(rx, vec![[1u64], [2u64], [3u64]]);
    assert_eq!(rz, vec![[0u64], [0u64], [0u64]]);
    assert_eq!(
        rc,
        vec![
            Complex64::new(2.5, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]
    );
    r.assert_invariants();
}

#[test]
fn add_w2_across_word_boundary() {
    let a = PauliSum::<2>::from_sorted_columns(
        vec![[0u64, 1u64], [0u64, 2u64]],
        vec![[0u64, 0u64], [0u64, 0u64]],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        128,
    );
    let b = PauliSum::<2>::from_sorted_columns(
        vec![[0u64, 1u64], [0u64, 4u64]],
        vec![[0u64, 0u64], [0u64, 0u64]],
        vec![Complex64::new(0.5, 0.0), Complex64::new(7.0, 0.0)],
        128,
    );
    let r = a.add(&b);
    assert_eq!(r.len(), 3);
    assert_eq!(
        r.to_arrays().0,
        vec![[0u64, 1u64], [0u64, 2u64], [0u64, 4u64]]
    );
    assert_eq!(r.bucket(0).2[0], Complex64::new(1.5, 0.0));
    assert_eq!(r.bucket(0).2[1], Complex64::new(2.0, 0.0));
    assert_eq!(r.bucket(0).2[2], Complex64::new(7.0, 0.0));
    r.assert_invariants();
}

// --- PauliSum::from_strings test helper ----------------------------
//
// `from_strings` itself is a `#[cfg(test)]` inherent impl over in
// `pauli_sum.rs`; only its tests moved here.

#[test]
fn from_strings_single_x_term() {
    let s = PauliSum::<1>::from_strings(&[("XII", Complex64::new(1.0, 0.0))]);
    assert_eq!(s.len(), 1);
    assert_eq!(s.num_qubits(), 3);
    assert_eq!(s.bucket(0).0[0], [0b001u64]);
    assert_eq!(s.bucket(0).1[0], [0u64]);
    assert_eq!(s.bucket(0).2[0], Complex64::new(1.0, 0.0));
    s.assert_invariants();
}

#[test]
fn from_strings_x_z_combined() {
    // "XZI": X on qubit 0, Z on qubit 1, I on qubit 2.
    let s = PauliSum::<1>::from_strings(&[("XZI", Complex64::new(1.0, 0.0))]);
    assert_eq!(s.bucket(0).0[0], [0b001u64]);
    assert_eq!(s.bucket(0).1[0], [0b010u64]);
    s.assert_invariants();
}

#[test]
fn from_strings_y_is_hermitian() {
    // Coefficients multiply the literal Hermitian Pauli string: "Y" maps to the symplectic key (x=1, z=1) with no phase factor, matching PauliString::y and expectation_product_state.
    let s = PauliSum::<1>::from_strings(&[("Y", Complex64::new(1.0, 0.0))]);
    assert_eq!(s.bucket(0).0[0], [1u64]);
    assert_eq!(s.bucket(0).1[0], [1u64]);
    assert_eq!(s.bucket(0).2[0], Complex64::new(1.0, 0.0));
}

#[test]
fn from_strings_real_coeffs_stay_real_for_any_y_count() {
    // A Hermitian observable keeps real coefficients regardless of how many Y characters a term contains — no per-Y phase is folded.
    for s in ["Y", "YY", "YYY", "YYYY"] {
        let padded: String = format!("{s:I<4}");
        let sum = PauliSum::<1>::from_strings(&[(&padded, Complex64::new(2.5, 0.0))]);
        assert_eq!(sum.bucket(0).2[0], Complex64::new(2.5, 0.0), "{s}");
    }
}

#[test]
fn from_strings_dedup_sums_coeffs() {
    let s = PauliSum::<1>::from_strings(&[
        ("XI", Complex64::new(1.0, 0.0)),
        ("XI", Complex64::new(0.5, -0.25)),
    ]);
    assert_eq!(s.len(), 1);
    assert_eq!(s.bucket(0).2[0], Complex64::new(1.5, -0.25));
    s.assert_invariants();
}

#[test]
fn from_strings_cancellation_drops_term() {
    let s = PauliSum::<1>::from_strings(&[
        ("XI", Complex64::new(1.0, 0.0)),
        ("XI", Complex64::new(-1.0, 0.0)),
        ("ZI", Complex64::new(2.0, 0.0)),
    ]);
    assert_eq!(s.len(), 1);
    assert_eq!(s.bucket(0).0[0], [0u64]);
    assert_eq!(s.bucket(0).1[0], [1u64]);
    assert_eq!(s.bucket(0).2[0], Complex64::new(2.0, 0.0));
    s.assert_invariants();
}

#[test]
fn from_strings_sorts_lex_keys() {
    // Insert out of order: ZI=(0,1), XI=(1,0), YI=(1,1) — lex sorted is
    // ZI < XI < YI.
    let s = PauliSum::<1>::from_strings(&[
        ("YI", Complex64::new(1.0, 0.0)),
        ("ZI", Complex64::new(2.0, 0.0)),
        ("XI", Complex64::new(3.0, 0.0)),
    ]);
    assert_eq!(s.len(), 3);
    assert_eq!((s.bucket(0).0[0], s.bucket(0).1[0]), ([0u64], [1u64])); // ZI
    assert_eq!((s.bucket(0).0[1], s.bucket(0).1[1]), ([1u64], [0u64])); // XI
    assert_eq!((s.bucket(0).0[2], s.bucket(0).1[2]), ([1u64], [1u64])); // YI
    assert_eq!(s.bucket(0).2[0], Complex64::new(2.0, 0.0));
    assert_eq!(s.bucket(0).2[1], Complex64::new(3.0, 0.0));
    assert_eq!(s.bucket(0).2[2], Complex64::new(1.0, 0.0));
    s.assert_invariants();
}

#[test]
fn from_strings_w2_qubit_64() {
    // 65-character string: X at index 64 lands in word 1.
    let mut s_chars: String = "I".repeat(65);
    // Replace index 64 with 'X'.
    unsafe {
        let bytes = s_chars.as_bytes_mut();
        bytes[64] = b'X';
    }
    let s = PauliSum::<2>::from_strings(&[(s_chars.as_str(), Complex64::new(1.0, 0.0))]);
    assert_eq!(s.num_qubits(), 65);
    assert_eq!(s.bucket(0).0[0], [0u64, 1u64]);
    assert_eq!(s.bucket(0).1[0], [0u64, 0u64]);
    s.assert_invariants();
}

#[test]
#[should_panic(expected = "unexpected Pauli char")]
fn from_strings_panics_on_invalid_char() {
    let _ = PauliSum::<1>::from_strings(&[("AB", Complex64::new(1.0, 0.0))]);
}

#[test]
#[should_panic(expected = "all pauli strings must have the same length")]
fn from_strings_panics_on_length_mismatch() {
    let _ = PauliSum::<1>::from_strings(&[
        ("XI", Complex64::new(1.0, 0.0)),
        ("XII", Complex64::new(1.0, 0.0)),
    ]);
}

// ---- partition scatter / gather ----

/// Bitwise, order-included comparison of two sums' canonical columns.
fn assert_same_columns<const W: usize>(got: &PauliSum<W>, want: &PauliSum<W>, what: &str) {
    let (gx, gz, gc) = got.to_arrays();
    let (wx, wz, wc) = want.to_arrays();
    assert_eq!(gx.len(), wx.len(), "{what}: term count");
    for i in 0..wx.len() {
        assert_eq!(
            (gx[i], gz[i], gc[i]),
            (wx[i], wz[i], wc[i]),
            "{what}: term {i} differs",
        );
    }
}

/// Scatter `sum` (rehashed to `bits`) into `1 << pbits` partitions, check each part is a well-formed single-rank sum on the same partition, and gather it back — which must reproduce the input bit for bit, in the same canonical order.
fn check_split_merge<const W: usize>(
    sum: &PauliSum<W>,
    num_qubits: usize,
    bits: u8,
    pbits: u8,
    what: &str,
) {
    let s = sum
        .clone()
        .with_hash(Gf2Hash::<W>::new(num_qubits, bits, 0xBEEF));
    let rows = PartitionRows::<W>::from_seed(num_qubits, pbits, 0x5EED);
    let p = rows.num_partitions();

    let mut parts: Vec<PauliSum<W>> = Vec::with_capacity(p);
    let mut total = 0usize;
    for r in 0..p as u32 {
        let part = s.filter_partition(&rows, r);
        part.assert_invariants();
        assert_eq!(
            part.num_buckets(),
            s.num_buckets(),
            "{what}: part {r} bucket count",
        );
        assert!(
            part.hash().same_rows_as(s.hash()),
            "{what}: part {r} hash rows",
        );
        assert_eq!(
            part.hash().bits(),
            s.hash().bits(),
            "{what}: part {r} hash bits",
        );
        assert_eq!(
            part.num_qubits(),
            s.num_qubits(),
            "{what}: part {r} num_qubits",
        );
        if !part.is_empty() {
            assert_eq!(
                part.partition_rank_of_all(&rows),
                Some(r),
                "{what}: part {r} is not single-rank",
            );
        }
        total += part.len();
        parts.push(part);
    }
    assert_eq!(total, s.len(), "{what}: partition lengths must add up");

    let back = PauliSum::<W>::merge_partitions(parts);
    back.assert_invariants();
    assert_eq!(
        back.num_buckets(),
        s.num_buckets(),
        "{what}: merged bucket count",
    );
    assert_same_columns(&back, &s, what);
}

#[test]
fn split_merge_round_trip_w1() {
    let dense = rand_sum_real::<1>(9000, 64, 0xD1);
    let sparse = low_weight_sum::<1>(3000, 64, 4, 0xD2);
    for pbits in [0u8, 1, 2] {
        for bits in [0u8, 3, 7] {
            check_split_merge(&dense, 64, bits, pbits, "w1 dense");
            check_split_merge(&sparse, 64, bits, pbits, "w1 low-weight");
        }
    }
}

#[test]
fn split_merge_round_trip_w2() {
    let dense = rand_sum_real::<2>(9000, 128, 0xD3);
    let sparse = low_weight_sum::<2>(3000, 100, 4, 0xD4);
    for pbits in [0u8, 1, 2] {
        for bits in [0u8, 3, 7] {
            check_split_merge(&dense, 128, bits, pbits, "w2 dense");
            check_split_merge(&sparse, 100, bits, pbits, "w2 low-weight");
        }
    }
}

#[test]
fn an_empty_partition_is_a_valid_empty_sum() {
    let s = PauliSum::<1>::from_strings(&[
        ("XII", Complex64::new(1.0, 0.0)),
        ("IYI", Complex64::new(2.0, 0.0)),
        ("IIZ", Complex64::new(3.0, 0.0)),
    ]);
    let rows = PartitionRows::<1>::from_seed(3, 2, 0x5EED);
    assert_eq!(rows.num_partitions(), 4);

    let parts: Vec<PauliSum<1>> = (0..4u32).map(|r| s.filter_partition(&rows, r)).collect();
    // Three terms across four partitions: at least one part is empty.
    let empties = parts.iter().filter(|p| p.is_empty()).count();
    assert!(empties >= 1, "expected at least one empty partition");

    for (r, part) in parts.iter().enumerate() {
        part.assert_invariants();
        assert_eq!(part.num_buckets(), s.num_buckets(), "part {r} bucket count");
        assert!(part.hash().same_rows_as(s.hash()), "part {r} hash rows");
        assert_eq!(part.num_qubits(), 3, "part {r} num_qubits");
        if part.is_empty() {
            assert_eq!(part.len(), 0, "part {r} len");
            assert_eq!(part.partition_rank_of_all(&rows), None, "part {r} rank");
        }
    }

    let back = PauliSum::<1>::merge_partitions(parts);
    back.assert_invariants();
    assert_same_columns(&back, &s, "3-term round trip through 4 partitions");
}

#[test]
fn merge_partitions_of_a_single_input_returns_it_unchanged() {
    let s = rand_sum_real::<2>(500, 128, 0xD6).with_hash(Gf2Hash::<2>::new(128, 3, 0xBEEF));
    let back = PauliSum::<2>::merge_partitions(vec![s.clone()]);
    back.assert_invariants();
    assert_eq!(back.num_buckets(), s.num_buckets());
    assert_same_columns(&back, &s, "single-input merge");
}

#[test]
#[should_panic(expected = "bucket count mismatch")]
fn merge_partitions_panics_on_mismatched_bits() {
    let s = rand_sum_real::<1>(200, 64, 0xD7);
    let a = s.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0xBEEF));
    let b = s.with_hash(Gf2Hash::<1>::new(64, 4, 0xBEEF));
    let _ = PauliSum::<1>::merge_partitions(vec![a, b]);
}

#[test]
#[should_panic(expected = "hash mismatch")]
fn merge_partitions_panics_on_mismatched_hash_rows() {
    let s = rand_sum_real::<1>(200, 64, 0xD8);
    let a = s.clone().with_hash(Gf2Hash::<1>::new(64, 3, 0xBEEF));
    let b = s.with_hash(Gf2Hash::<1>::new(64, 3, 0xBEEE));
    let _ = PauliSum::<1>::merge_partitions(vec![a, b]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "duplicate key")]
fn merge_partitions_debug_asserts_on_a_key_shared_across_inputs() {
    let s = PauliSum::<1>::from_strings(&[
        ("XII", Complex64::new(1.0, 0.0)),
        ("IIZ", Complex64::new(2.0, 0.0)),
    ]);
    let _ = PauliSum::<1>::merge_partitions(vec![s.clone(), s]);
}

#[test]
fn coarsen_to_matches_repeated_coarsen() {
    let sum = rand_sum_real::<2>(4000, 128, 0xD9);
    let base = sum.with_hash(Gf2Hash::<2>::new(128, 5, 0xC0DE));
    let mut a = base.clone();
    a.coarsen_to(2);
    let mut b = base.clone();
    b.coarsen();
    b.coarsen();
    b.coarsen();

    assert_eq!(a.hash().bits(), 2);
    assert_eq!(a.num_buckets(), 4);
    a.assert_invariants();
    for i in 0..a.num_buckets() {
        assert_eq!(a.bucket_len(i), b.bucket_len(i), "bucket {i} length");
    }
    assert_same_columns(&a, &b, "coarsen_to(2) vs three coarsen()");
}

#[test]
fn coarsen_to_the_current_bits_is_a_no_op() {
    let base = rand_sum_real::<1>(2000, 64, 0xDA).with_hash(Gf2Hash::<1>::new(64, 4, 0xC0DE));
    let mut a = base.clone();
    a.coarsen_to(4);
    assert_eq!(a.hash().bits(), 4);
    assert_eq!(a.num_buckets(), 16);
    a.assert_invariants();
    assert_same_columns(&a, &base, "coarsen_to(current)");
}

#[test]
#[should_panic(expected = "coarsen_to")]
fn coarsen_to_above_the_current_bits_panics() {
    let mut a = rand_sum_real::<1>(2000, 64, 0xDB).with_hash(Gf2Hash::<1>::new(64, 4, 0xC0DE));
    a.coarsen_to(5);
}

#[test]
fn partition_rank_of_all_distinguishes_empty_mixed_and_filtered() {
    let rows = PartitionRows::<1>::from_seed(64, 2, 0x5EED);

    let empty = PauliSum::<1>::empty(64);
    assert_eq!(empty.partition_rank_of_all(&rows), None, "empty sum");

    let mixed = rand_sum_real::<1>(2000, 64, 0xDC);
    assert_eq!(mixed.partition_rank_of_all(&rows), None, "mixed sum");

    for r in 0..rows.num_partitions() as u32 {
        let part = mixed.filter_partition(&rows, r);
        assert!(!part.is_empty(), "part {r} unexpectedly empty");
        assert_eq!(part.partition_rank_of_all(&rows), Some(r), "part {r}");
    }
}

mod props {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeMap;

    const NQ: usize = 6;

    fn build(terms: &[(u64, u64, i32, i32)]) -> PauliSum<1> {
        let mut acc = BuildAccumulator::<1>::new(NQ);
        for &(x, z, re, im) in terms {
            acc.add_term(
                PauliString::<1> { x: [x], z: [z] },
                Phase::ONE,
                Complex64::new(re as f64, im as f64),
            );
        }
        acc.finalize()
    }

    proptest! {
        /// Scatter is a tiling: the parts have disjoint ranks, every term of part `r` really hashes to partition `r`, and their lengths add back up to the whole.
        #[test]
        fn filter_partition_tiles_the_sum(
            terms in prop::collection::vec((0u64..64, 0u64..64, -4i32..4, -4i32..4), 0..16),
            pbits in 0u8..=2,
            bits in 0u8..=3,
            seed in any::<u64>(),
        ) {
            let s = build(&terms).with_hash(Gf2Hash::<1>::new(NQ, bits, 0xB0));
            let rows = PartitionRows::<1>::from_seed(NQ, pbits, seed);
            let mut total = 0usize;
            for r in 0..rows.num_partitions() as u32 {
                let part = s.filter_partition(&rows, r);
                part.assert_invariants();
                total += part.len();
                for (x, z, _) in part.iter() {
                    prop_assert_eq!(rows.partition_of(x, z), r);
                }
            }
            prop_assert_eq!(total, s.len());
        }
    }

    proptest! {
        /// `add` against an independent model: a `BTreeMap` keyed by
        /// `(x, z)`, summed then stripped of exact zeros.
        ///
        /// Coefficients are small integers so exact cancellation actually happens, and the keyspace is 6 qubits so the two operands share keys often. Every surviving coefficient is a single `a + b`, so the comparison is bitwise rather than toleranced.
        #[test]
        fn bucketed_add_matches_btreemap_model(
            terms_a in prop::collection::vec(
                (0u64..64, 0u64..64, -4i32..=4, -4i32..=4), 0..60),
            terms_b in prop::collection::vec(
                (0u64..64, 0u64..64, -4i32..=4, -4i32..=4), 0..60),
            bits_a in 0u8..=6,
            bits_b in 0u8..=6,
            seed_shift in 0u64..=1,
        ) {
            let a = build(&terms_a);
            let b = build(&terms_b);

            let seed_a = 0x5EEDu64;
            let ba = a.clone().with_hash(Gf2Hash::<1>::new(NQ, bits_a, seed_a));
            let bb = b
                .clone()
                .with_hash(Gf2Hash::<1>::new(NQ, bits_b, seed_a + seed_shift));

            let got = ba.add(&bb);
            got.assert_invariants();
            prop_assert_eq!(got.hash().bits(), bits_a, "left partition must win");
            prop_assert_eq!(got.hash().seed(), seed_a, "left partition must win");

            let mut model: BTreeMap<([u64; 1], [u64; 1]), Complex64> = BTreeMap::new();
            for (x, z, c) in a.iter() {
                model.insert((*x, *z), c);
            }
            for (x, z, c) in b.iter() {
                model
                    .entry((*x, *z))
                    .and_modify(|acc| *acc += c)
                    .or_insert(c);
            }
            let zero = Complex64::new(0.0, 0.0);
            model.retain(|_, c| *c != zero);

            prop_assert_eq!(got.len(), model.len());
            let triples = sorted_triples(&got);
            for (i, (&(mx, mz), &mc)) in model.iter().enumerate() {
                prop_assert_eq!(triples[i].0, mx);
                prop_assert_eq!(triples[i].1, mz);
                prop_assert_eq!(triples[i].2, mc);
            }
        }
    }

    /// Build a sorted, deduplicated `PauliSum<2>` from random `(x, z, coeff)` triples, via a `BTreeMap` keyed on `(x, z)` to enforce the sorted/unique invariant before SoA materialization.
    /// Coefficients are kept small (`re, im ∈ [-4, 4]`) to avoid FP cancellation noise; length capped at 8 to bound shrinking time.
    fn arb_pauli_sum_w2() -> impl Strategy<Value = PauliSum<2>> {
        prop::collection::vec(
            (
                any::<u64>(),
                any::<u64>(),
                any::<u64>(),
                any::<u64>(),
                -4.0f64..4.0,
                -4.0f64..4.0,
            ),
            0..8,
        )
        .prop_map(|entries| {
            let mut map: BTreeMap<([u64; 2], [u64; 2]), Complex64> = BTreeMap::new();
            for (x0, x1, z0, z1, re, im) in entries {
                map.insert(([x0, x1], [z0, z1]), Complex64::new(re, im));
            }
            let mut x = Vec::with_capacity(map.len());
            let mut z = Vec::with_capacity(map.len());
            let mut coeff = Vec::with_capacity(map.len());
            for ((kx, kz), c) in map {
                x.push(kx);
                z.push(kz);
                coeff.push(c);
            }
            PauliSum::<2>::from_sorted_columns(x, z, coeff, 128)
        })
    }

    proptest! {
        #[test]
        fn add_is_associative(
            a in arb_pauli_sum_w2(),
            b in arb_pauli_sum_w2(),
            c in arb_pauli_sum_w2(),
        ) {
            let left = a.add(&b).add(&c);
            let right = a.add(&b.add(&c));
            left.assert_invariants();
            right.assert_invariants();
            let (lx, lz, lc) = left.to_arrays();
            let (rx, rz, rc) = right.to_arrays();
            prop_assert_eq!(lx, rx);
            prop_assert_eq!(lz, rz);
            prop_assert_eq!(lc.len(), rc.len());
            for k in 0..lc.len() {
                let diff = lc[k] - rc[k];
                prop_assert!(
                    diff.norm() <= 1e-12,
                    "coeff mismatch at idx {}: lhs={:?} rhs={:?}",
                    k, lc[k], rc[k]
                );
            }
        }
    }
}
