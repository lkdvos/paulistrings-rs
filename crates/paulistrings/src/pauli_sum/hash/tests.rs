use super::*;
use crate::test_support::Xs64;

/// XOR two Pauli keys — the group operation on the key space.
fn xor<const W: usize>(a: &PauliString<W>, b: &PauliString<W>) -> PauliString<W> {
    let mut out = *a;
    for w in 0..W {
        out.x[w] ^= b.x[w];
        out.z[w] ^= b.z[w];
    }
    out
}

fn rand_key<const W: usize>(rng: &mut Xs64, num_qubits: usize) -> PauliString<W> {
    let mut p = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    for w in 0..W {
        let mask = word_mask(num_qubits, w);
        p.x[w] = rng.next_u64() & mask;
        p.z[w] = rng.next_u64() & mask;
    }
    p
}

fn low_weight_key<const W: usize>(
    rng: &mut Xs64,
    num_qubits: usize,
    weight: usize,
) -> PauliString<W> {
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
    p
}

#[test]
fn bucket_is_within_range_w1() {
    let h = Gf2Hash::<1>::new(64, 7, 0xABCDEF);
    let mut rng = Xs64::new(1);
    for _ in 0..2000 {
        let p = rand_key::<1>(&mut rng, 64);
        assert!((h.bucket_of_pauli(&p) as usize) < h.num_buckets());
    }
}

#[test]
fn bucket_is_within_range_w2() {
    let h = Gf2Hash::<2>::new(128, 11, 0xABCDEF);
    let mut rng = Xs64::new(2);
    for _ in 0..2000 {
        let p = rand_key::<2>(&mut rng, 128);
        assert!((h.bucket_of_pauli(&p) as usize) < h.num_buckets());
    }
}

#[test]
fn identity_key_maps_to_bucket_zero() {
    // h(0) = 0 for any linear h, so the identity string always lands in bucket 0.
    let h = Gf2Hash::<2>::new(128, 10, 0x1234);
    assert_eq!(h.bucket_of(&[0, 0], &[0, 0]), 0);
}

#[test]
fn zero_bits_is_a_single_bucket() {
    let h = Gf2Hash::<1>::new(64, 0, 0x55);
    assert_eq!(h.num_buckets(), 1);
    let mut rng = Xs64::new(3);
    for _ in 0..100 {
        assert_eq!(h.bucket_of_pauli(&rand_key::<1>(&mut rng, 64)), 0);
    }
}

#[test]
fn linearity_hand_checked_w1() {
    let h = Gf2Hash::<1>::new(64, 8, 0xFEED);
    let v = PauliString::<1>::x(3);
    let w = PauliString::<1>::z(11);
    assert_eq!(
        h.bucket_of_pauli(&xor(&v, &w)),
        h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w)
    );
}

#[test]
fn linearity_random_w1() {
    let h = Gf2Hash::<1>::new(64, 9, 0xFEED);
    let mut rng = Xs64::new(11);
    for _ in 0..2000 {
        let v = rand_key::<1>(&mut rng, 64);
        let w = rand_key::<1>(&mut rng, 64);
        assert_eq!(
            h.bucket_of_pauli(&xor(&v, &w)),
            h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w),
        );
    }
}

#[test]
fn linearity_random_w2_crosses_word_boundary() {
    let h = Gf2Hash::<2>::new(128, 12, 0xFEED);
    let mut rng = Xs64::new(12);
    for _ in 0..2000 {
        let v = rand_key::<2>(&mut rng, 128);
        let w = rand_key::<2>(&mut rng, 128);
        assert_eq!(
            h.bucket_of_pauli(&xor(&v, &w)),
            h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w),
        );
    }
}

#[test]
fn bits_beyond_num_qubits_do_not_affect_the_bucket() {
    // 100 qubits in W=2: setting the dead bits 100..128 must not move a term.
    let h = Gf2Hash::<2>::new(100, 10, 0x99);
    let mut rng = Xs64::new(21);
    for _ in 0..500 {
        let p = rand_key::<2>(&mut rng, 100);
        let mut polluted = p;
        // Set every dead bit in word 1 (qubits 100..128).
        let dead = !((1u64 << (100 - 64)) - 1);
        polluted.x[1] |= dead;
        polluted.z[1] |= dead;
        assert_eq!(h.bucket_of_pauli(&p), h.bucket_of_pauli(&polluted));
    }
}

#[test]
fn rows_are_masked_at_a_mid_word_boundary() {
    // Directly: a key that is *only* out-of-range bits hashes to 0.
    let h = Gf2Hash::<2>::new(70, 12, 0x7A);
    let dead = !((1u64 << (70 - 64)) - 1);
    assert_eq!(h.bucket_of(&[0, dead], &[0, dead]), 0);
}

/// The XOR-fold in `row_parity` / `partition_of` agrees with a naive one-popcount-per-word oracle.
#[test]
fn xor_fold_parity_matches_per_word_popcount() {
    fn naive<const W: usize>(x: &[u64; W], z: &[u64; W], rx: &[u64; W], rz: &[u64; W]) -> u32 {
        let mut parity: u32 = 0;
        for w in 0..W {
            parity ^= (x[w] & rx[w]).count_ones();
            parity ^= (z[w] & rz[w]).count_ones();
        }
        parity & 1
    }

    fn check<const W: usize>(num_qubits: usize, seed: u64) {
        let h = Gf2Hash::<W>::new(num_qubits, B_MAX_BITS, seed);
        let mut rng = Xs64::new(seed ^ 0x5EED);
        for _ in 0..500 {
            let p = rand_key::<W>(&mut rng, num_qubits);
            for row in 0..B_MAX_BITS {
                assert_eq!(
                    h.row_parity(&p.x, &p.z, row),
                    naive(&p.x, &p.z, &h.rows_x[row as usize], &h.rows_z[row as usize]),
                    "W={W} row={row}: XOR-fold disagrees with per-word popcount"
                );
            }
        }
    }

    check::<1>(64, 0xA11CE);
    check::<2>(128, 0xB0B);
    check::<4>(256, 0xC0FFEE);
    check::<8>(512, 0xD00D);
}

#[test]
fn row_parity_matches_bucket_of_bit_extraction() {
    let h = Gf2Hash::<2>::new(128, B_MAX_BITS, 0xF00D);
    let mut rng = Xs64::new(81);
    for _ in 0..500 {
        let p = rand_key::<2>(&mut rng, 128);
        let full = h.bucket_of_pauli(&p);
        for row in 0..B_MAX_BITS {
            let bit = h.row_parity(&p.x, &p.z, row);
            assert!(bit == 0 || bit == 1, "row_parity must return 0 or 1");
            assert_eq!(
                bit,
                (full >> row) & 1,
                "row {row} disagrees with bucket_of's bit extraction",
            );
        }
    }
}

#[test]
fn refine_preserves_the_low_bits() {
    let mut h = Gf2Hash::<2>::new(128, 6, 0xB0B);
    let mut rng = Xs64::new(31);
    let keys: Vec<PauliString<2>> = (0..500).map(|_| rand_key::<2>(&mut rng, 128)).collect();
    let before: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();

    h.refine();
    assert_eq!(h.num_buckets(), 128);
    let mask = (1u32 << 6) - 1;
    for (k, &b) in keys.iter().zip(before.iter()) {
        // The refined index agrees with the old one on the low `bits` bits.
        assert_eq!(h.bucket_of_pauli(k) & mask, b);
    }
}

#[test]
fn coarsen_inverts_refine() {
    let mut h = Gf2Hash::<1>::new(64, 8, 0xCAFE);
    let mut rng = Xs64::new(41);
    let keys: Vec<PauliString<1>> = (0..500).map(|_| rand_key::<1>(&mut rng, 64)).collect();
    let before: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();

    h.refine();
    h.coarsen();
    assert_eq!(h.bits(), 8);
    let after: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();
    assert_eq!(before, after);
}

#[test]
fn coarsen_merges_bucket_pairs() {
    let mut h = Gf2Hash::<1>::new(64, 8, 0xDEAD);
    let mut rng = Xs64::new(51);
    let keys: Vec<PauliString<1>> = (0..500).map(|_| rand_key::<1>(&mut rng, 64)).collect();
    let fine: Vec<u32> = keys.iter().map(|k| h.bucket_of_pauli(k)).collect();

    h.coarsen();
    for (k, &f) in keys.iter().zip(fine.iter()) {
        // Dropping the top bit merges (b, b + B/2).
        assert_eq!(h.bucket_of_pauli(k), f & ((1 << 7) - 1));
    }
}

#[test]
#[should_panic(expected = "already at B_MAX_BITS")]
fn refine_past_the_maximum_panics() {
    let mut h = Gf2Hash::<1>::new(64, B_MAX_BITS, 0x1);
    h.refine();
}

#[test]
#[should_panic(expected = "already at a single bucket")]
fn coarsen_below_one_bucket_panics() {
    let mut h = Gf2Hash::<1>::new(64, 0, 0x1);
    h.coarsen();
}

#[test]
#[should_panic(expected = "exceeds B_MAX_BITS")]
fn constructing_past_the_maximum_panics() {
    let _ = Gf2Hash::<1>::new(64, B_MAX_BITS + 1, 0x1);
}

#[test]
fn same_seed_gives_the_same_hash() {
    let a = Gf2Hash::<2>::new(128, 10, 0x5EED);
    let b = Gf2Hash::<2>::new(128, 10, 0x5EED);
    assert!(a.same_rows_as(&b));
    let mut rng = Xs64::new(61);
    for _ in 0..500 {
        let p = rand_key::<2>(&mut rng, 128);
        assert_eq!(a.bucket_of_pauli(&p), b.bucket_of_pauli(&p));
    }
}

#[test]
fn different_seeds_give_different_hashes() {
    let a = Gf2Hash::<2>::new(128, 10, 0x5EED);
    let b = Gf2Hash::<2>::new(128, 10, 0x5EEE);
    assert!(!a.same_rows_as(&b));
    let mut rng = Xs64::new(71);
    let differs = (0..500)
        .map(|_| rand_key::<2>(&mut rng, 128))
        .filter(|p| a.bucket_of_pauli(p) != b.bucket_of_pauli(p))
        .count();
    // Two independent hashes agree on a given key with probability 2^-10.
    assert!(
        differs > 400,
        "expected most keys to hash differently, got {differs}/500"
    );
}

#[test]
fn no_row_masks_to_zero_even_at_one_qubit() {
    // At num_qubits = 1 a naive draw gives an all-zero row 1/4 of the time.
    let h = Gf2Hash::<1>::new(1, 2, 0x1);
    for i in 0..B_MAX_BITS as usize {
        assert!(
            (h.rows_x[i][0] | h.rows_z[i][0]) != 0,
            "row {i} masked to zero",
        );
    }
}

#[test]
fn zero_qubits_is_degenerate_but_terminates() {
    // Every row is legitimately zero; construction must not spin forever.
    let h = Gf2Hash::<1>::new(0, 3, 0x1);
    assert_eq!(h.bucket_of(&[0], &[0]), 0);
}

/// One step of xorshift64 (13, 7, 17), a GF(2)-linear map `M` on `u64`.
fn xorshift64_step(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Row `j` of `M` as a mask: bit `j` of `M·v` is the parity of `xorshift64_row(j) & v`.
fn xorshift64_row(j: u32) -> u64 {
    (0..64).fold(0u64, |row, k| {
        row | (((xorshift64_step(1u64 << k) >> j) & 1) << k)
    })
}

/// Consecutive outputs of a linear generator satisfy `rows_z = M·rows_x` word for word under every seed.
#[test]
fn row_words_are_not_one_xorshift_step_apart() {
    for seed in [0x1u64, 0x5EED, crate::pauli_sum::storage::DEFAULT_HASH_SEED] {
        let h = Gf2Hash::<2>::new(128, B_MAX_BITS, seed);
        let p = PartitionRows::<2>::from_seed(128, P_MAX_BITS, seed);
        let (px, pz) = p.rows();
        let linked = h
            .rows_x
            .iter()
            .zip(&h.rows_z)
            .chain(px.iter().zip(pz))
            .flat_map(|(rx, rz)| (0..2).map(move |w| xorshift64_step(rx[w]) == rz[w]))
            .filter(|&linked| linked)
            .count();
        assert_eq!(linked, 0, "seed {seed:#x}: {linked} row words are M·x");
    }
}

/// `rows_z = M·rows_x` puts `d_j = (x = row_j(M), z = e_j)`, Pauli weight ≈ 6, in the kernel of every row, so `u` and `u ⊕ d_j` always share a bucket.
/// A dense random `H` sends each to 0 with probability `2^-20` at 20 bits.
#[test]
fn xorshift_kernel_deltas_do_not_share_bucket_zero() {
    fn zeros<const W: usize>(seed: u64) -> usize {
        let h = Gf2Hash::<W>::new(64 * W, 20, seed);
        let mut n = 0;
        for w in 0..W {
            for j in 0..64u32 {
                let mut x = [0u64; W];
                let mut z = [0u64; W];
                x[w] = xorshift64_row(j);
                z[w] = 1u64 << j;
                n += (h.bucket_of(&x, &z) == 0) as usize;
            }
        }
        n
    }
    let seeds = [0x1u64, 0x5EED, crate::pauli_sum::storage::DEFAULT_HASH_SEED];
    let w1: usize = seeds.iter().map(|&s| zeros::<1>(s)).sum();
    let w2: usize = seeds.iter().map(|&s| zeros::<2>(s)).sum();
    assert!(w1 <= 1, "{w1}/192 kernel deltas hash to 0 at W=1");
    assert!(w2 <= 1, "{w2}/384 kernel deltas hash to 0 at W=2");
}

/// 64 rows over 64 qubits as a fingerprint must separate the 18 337 keys of weight ≤ 2; a random linear map collides on some pair with probability ~2^-37.
#[test]
fn a_64_row_fingerprint_is_injective_on_weight_two_keys() {
    let (rx, rz) = draw_rows::<1>(
        64,
        64,
        crate::pauli_sum::storage::DEFAULT_HASH_SEED,
        &[0],
        &[0],
    );
    let image = |x: u64, z: u64| {
        (0..64).fold(0u64, |bits, i| {
            bits | ((((x & rx[i][0]) ^ (z & rz[i][0])).count_ones() as u64 & 1) << i)
        })
    };
    // (x, z) bits of X, Z and Y on one qubit.
    let paulis = [(1u64, 0u64), (0, 1), (1, 1)];
    let mut keys = vec![(0u64, 0u64)];
    for q in 0..64 {
        for (a, b) in paulis {
            keys.push((a << q, b << q));
        }
    }
    for q in 0..64 {
        for r in (q + 1)..64 {
            for (a, b) in paulis {
                for (c, d) in paulis {
                    keys.push(((a << q) | (c << r), (b << q) | (d << r)));
                }
            }
        }
    }
    assert_eq!(keys.len(), 1 + 64 * 3 + 2016 * 9);
    let images: std::collections::HashSet<u64> = keys.iter().map(|&(x, z)| image(x, z)).collect();
    assert_eq!(images.len(), keys.len(), "fingerprint collisions");
}

/// Row word `w` depends on `(seed, row, w, x-or-z)` alone, so the same `(num_qubits, seed)` gives the same rows at every width.
#[test]
fn rows_do_not_depend_on_the_width() {
    let seed = crate::pauli_sum::storage::DEFAULT_HASH_SEED;
    for n in [1usize, 5, 64] {
        let h1 = Gf2Hash::<1>::new(n, 8, seed);
        let h2 = Gf2Hash::<2>::new(n, 8, seed);
        for i in 0..B_MAX_BITS as usize {
            let (x1, z1) = h1.row(i);
            assert_eq!(h2.row(i), ([x1[0], 0], [z1[0], 0]), "n={n} row {i}");
        }
        let p1 = PartitionRows::<1>::from_seed(n, P_MAX_BITS, seed);
        let p2 = PartitionRows::<2>::from_seed(n, P_MAX_BITS, seed);
        let widened = |rows: &[[u64; 1]]| rows.iter().map(|r| [r[0], 0]).collect::<Vec<_>>();
        assert_eq!(p2.rows().0, widened(p1.rows().0), "n={n} partition x-rows");
        assert_eq!(p2.rows().1, widened(p1.rows().1), "n={n} partition z-rows");
    }
    let h2 = Gf2Hash::<2>::new(128, 8, seed);
    let h4 = Gf2Hash::<4>::new(128, 8, seed);
    for i in 0..B_MAX_BITS as usize {
        let (x2, z2) = h2.row(i);
        let (x4, z4) = h4.row(i);
        assert_eq!(x4, [x2[0], x2[1], 0, 0], "row {i}");
        assert_eq!(z4, [z2[0], z2[1], 0, 0], "row {i}");
    }
}

/// Bucket occupancy on low-weight keys, the physically relevant regime.
/// A coordinate-projection `H` would fail this, sending nearly every weight-4 string to bucket 0.
#[test]
fn occupancy_is_balanced_on_low_weight_keys() {
    let num_qubits = 64;
    let h = Gf2Hash::<1>::new(num_qubits, 6, 0x0CC1);
    let b = h.num_buckets();

    let mut rng = Xs64::new(0xBA1);
    let mut seen = std::collections::HashSet::new();
    let mut counts = vec![0usize; b];
    let target = 8192usize;
    while seen.len() < target {
        let p = low_weight_key::<1>(&mut rng, num_qubits, 4);
        if seen.insert((p.x, p.z)) {
            counts[h.bucket_of_pauli(&p) as usize] += 1;
        }
    }

    let mean = target / b; // 128
    let max = *counts.iter().max().unwrap();
    let min = *counts.iter().min().unwrap();
    // Deterministic given the seeds, so these bounds are not flaky.
    assert!(max < 2 * mean, "max load {max} vs mean {mean}");
    assert!(min > mean / 2, "min load {min} vs mean {mean}");
}

/// Occupancy balance is not the whole story: a dense random `H` can still fail to separate a two-qubit channel's four delta generators, so two distinct local deltas share one bucket delta.
#[test]
fn support_delta_rank_is_usually_full_but_not_always() {
    // Deterministic given the seed, so these counts are not flaky.
    let h = Gf2Hash::<2>::new(128, 7, crate::pauli_sum::storage::DEFAULT_HASH_SEED);
    let mut deficient = 0usize;
    let mut total = 0usize;
    for i in 0..128 {
        for j in (i + 1)..128 {
            total += 1;
            if crate::test_support::support_delta_rank(&h, &[i, j]) < 4 {
                deficient += 1;
            }
        }
    }
    // About 11% in expectation over seeds; the bound pins only the order of magnitude.
    assert_eq!(total, 8128);
    assert!(
        (200..2000).contains(&deficient),
        "expected O(10%) rank-deficient support pairs at 7 bucket bits, got {deficient}/{total}"
    );
}

/// Rank is monotone in the number of active bucket bits, since the active hash is a prefix of one fixed matrix: refining can only separate deltas that were colliding, never merge separated ones.
#[test]
fn support_delta_rank_is_monotone_in_bits() {
    for seed in [0x1u64, 0xBEEF, crate::pauli_sum::storage::DEFAULT_HASH_SEED] {
        let mut last = 0usize;
        for bits in 0..=12u8 {
            let h = Gf2Hash::<2>::new(128, bits, seed);
            let r = crate::test_support::support_delta_rank(&h, &[0, 1]);
            assert!(
                r >= last && r <= 4,
                "seed {seed:#x}: rank went {last} -> {r} at bits {bits}"
            );
            assert!(r <= bits as usize, "rank {r} exceeds bits {bits}");
            last = r;
        }
    }
}

/// Rows do not depend on `W` (`rows_do_not_depend_on_the_width`), so neither does a support's delta-span rank.
/// At the default seed support `(0, 1)` is full-rank at 7..=9 bucket bits and `(0, 7)` one rank short at 7, at both widths.
#[test]
fn support_delta_rank_is_width_independent_at_the_default_seed() {
    use crate::test_support::support_delta_rank as rank;
    let seed = crate::pauli_sum::storage::DEFAULT_HASH_SEED;
    for bits in 7..=9u8 {
        let w1 = Gf2Hash::<1>::new(64, bits, seed);
        let w2 = Gf2Hash::<2>::new(65, bits, seed);
        let w2_wide = Gf2Hash::<2>::new(128, bits, seed);
        assert_eq!(rank(&w1, &[0, 1]), 4, "W=1/q=64 at {bits} bits");
        assert_eq!(rank(&w2, &[0, 1]), 4, "W=2/q=65 at {bits} bits");
        assert_eq!(rank(&w2_wide, &[0, 1]), 4, "W=2/q=128 at {bits} bits");
    }
    assert_eq!(rank(&Gf2Hash::<1>::new(64, 7, seed), &[0, 7]), 3);
    assert_eq!(rank(&Gf2Hash::<2>::new(128, 7, seed), &[0, 7]), 3);
}

/// A support delta cannot reorder a bucket's key column exactly when `h` separates the support's delta space.
#[test]
fn support_delta_preserves_bucket_order_iff_the_delta_span_is_full_rank() {
    assert!(!order_broken_by_some_delta::<2>(128, 7, &[0, 1]));
    assert!(order_broken_by_some_delta::<1>(64, 7, &[0, 7]));
}

/// Partition a closed key set under `h`, then check every non-identity support delta against every bucket's ascending key column; returns `true` if any delta reorders any bucket.
/// The key set is closed (every off-support pattern with all `2^(2k)` local patterns), as a repeated dense-PTM layer leaves it.
fn order_broken_by_some_delta<const W: usize>(
    num_qubits: usize,
    bits: u8,
    support: &[usize],
) -> bool {
    let h = Gf2Hash::<W>::new(
        num_qubits,
        bits,
        crate::pauli_sum::storage::DEFAULT_HASH_SEED,
    );
    // Enumerate the support's delta space: one bit per (qubit, x-or-z).
    let gens: Vec<PauliString<W>> = support
        .iter()
        .flat_map(|&q| [PauliString::<W>::x(q), PauliString::<W>::z(q)])
        .collect();
    let local = |combo: usize| -> PauliString<W> {
        let mut d = PauliString::<W> {
            x: [0u64; W],
            z: [0u64; W],
        };
        for (g, gen) in gens.iter().enumerate() {
            if combo >> g & 1 == 1 {
                d = xor(&d, gen);
            }
        }
        d
    };
    let mut rng = Xs64::new(0xD17A);
    let mut buckets: Vec<Vec<([u64; W], [u64; W])>> = vec![Vec::new(); h.num_buckets()];
    for _ in 0..2_000 {
        // A random off-support pattern, then its whole local orbit.
        let mut rest = rand_key::<W>(&mut rng, num_qubits);
        for &q in support {
            let (w, bit) = (q / 64, 1u64 << (q % 64));
            rest.x[w] &= !bit;
            rest.z[w] &= !bit;
        }
        for combo in 0..(1usize << gens.len()) {
            let p = xor(&rest, &local(combo));
            buckets[h.bucket_of_pauli(&p) as usize].push((p.x, p.z));
        }
    }
    for columns in buckets.iter_mut() {
        columns.sort_unstable();
        columns.dedup();
        for combo in 1..(1usize << gens.len()) {
            let d = local(combo);
            let translated: Vec<([u64; W], [u64; W])> = columns
                .iter()
                .map(|(x, z)| {
                    let mut kx = *x;
                    let mut kz = *z;
                    for w in 0..W {
                        kx[w] ^= d.x[w];
                        kz[w] ^= d.z[w];
                    }
                    (kx, kz)
                })
                .collect();
            if translated.windows(2).any(|w| w[1] < w[0]) {
                return true;
            }
        }
    }
    false
}

#[test]
fn occupancy_is_balanced_on_dense_keys() {
    let h = Gf2Hash::<2>::new(128, 8, 0x0CC2);
    let b = h.num_buckets();
    let mut rng = Xs64::new(0xBA2);
    let mut counts = vec![0usize; b];
    let target = 32768usize;
    for _ in 0..target {
        counts[h.bucket_of_pauli(&rand_key::<2>(&mut rng, 128)) as usize] += 1;
    }
    let mean = target / b; // 128
    let max = *counts.iter().max().unwrap();
    let min = *counts.iter().min().unwrap();
    assert!(max < 2 * mean, "max load {max} vs mean {mean}");
    assert!(min > mean / 2, "min load {min} vs mean {mean}");
}

#[test]
fn partition_of_hand_checked_w1() {
    // Two explicit rows over 8 qubits:
    //   row 0: x-mask 0b011, z-mask 0
    //   row 1: x-mask 0,     z-mask 0b101
    let p = PartitionRows::<1>::from_rows(8, vec![[0b11], [0]], vec![[0], [0b101]]);
    assert_eq!(p.bits(), 2);
    assert_eq!(p.num_partitions(), 4);
    assert_eq!(p.num_qubits(), 8);
    // X_0: bit 0 = parity(0b001 & 0b011) = 1, bit 1 = 0.
    assert_eq!(p.partition_of_pauli(&PauliString::<1>::x(0)), 1);
    // Z_0: bit 0 = 0, bit 1 = parity(0b001 & 0b101) = 1.
    assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(0)), 2);
    // Y_0 = X_0 ⊕ Z_0.
    assert_eq!(p.partition_of(&[1], &[1]), 3);
    // X_0 X_1: parity(0b011 & 0b011) = 0.
    assert_eq!(p.partition_of(&[0b11], &[0]), 0);
    // Z_1: parity(0b010 & 0b101) = 0.
    assert_eq!(p.partition_of(&[0], &[0b10]), 0);
    // Z_2: parity(0b100 & 0b101) = 1.
    assert_eq!(p.partition_of(&[0], &[0b100]), 2);
}

#[test]
fn partition_is_within_range_and_the_identity_key_is_partition_zero() {
    let p = PartitionRows::<2>::from_seed(128, P_MAX_BITS, 0xABCDEF);
    assert_eq!(p.num_partitions(), 1usize << P_MAX_BITS);
    // p(0) = 0 for any linear map — the same documented wart as `h`.
    assert_eq!(p.partition_of(&[0, 0], &[0, 0]), 0);
    let mut rng = Xs64::new(101);
    for _ in 0..2000 {
        let k = rand_key::<2>(&mut rng, 128);
        assert!((p.partition_of_pauli(&k) as usize) < p.num_partitions());
    }
}

#[test]
fn zero_partition_bits_is_a_single_partition() {
    let p = PartitionRows::<1>::none(64);
    assert_eq!(p.bits(), 0);
    assert_eq!(p.num_partitions(), 1);
    assert_eq!(p.num_qubits(), 64);
    let (rx, rz) = p.rows();
    assert!(rx.is_empty() && rz.is_empty());
    let mut rng = Xs64::new(102);
    for _ in 0..200 {
        assert_eq!(p.partition_of_pauli(&rand_key::<1>(&mut rng, 64)), 0);
    }
    // `from_seed` at zero bits is the same object.
    assert_eq!(PartitionRows::<1>::from_seed(64, 0, 0x1234), p);
}

#[test]
fn partition_bits_beyond_num_qubits_do_not_affect_the_partition() {
    let p = PartitionRows::<2>::from_seed(100, 3, 0x99);
    let mut rng = Xs64::new(103);
    let dead = !((1u64 << (100 - 64)) - 1);
    for _ in 0..500 {
        let k = rand_key::<2>(&mut rng, 100);
        let mut polluted = k;
        polluted.x[1] |= dead;
        polluted.z[1] |= dead;
        assert_eq!(p.partition_of_pauli(&k), p.partition_of_pauli(&polluted));
    }
}

#[test]
fn from_seed_is_reproducible_and_seed_dependent() {
    let a = PartitionRows::<2>::from_seed(128, 4, 0x5EED);
    let b = PartitionRows::<2>::from_seed(128, 4, 0x5EED);
    assert_eq!(a, b);
    assert_ne!(a, PartitionRows::<2>::from_seed(128, 4, 0x5EEE));
}

#[test]
fn excluding_nothing_is_from_seed_and_excluding_avoids() {
    for seed in [0x1u64, 0x5EED] {
        let plain = PartitionRows::<2>::from_seed(100, 4, seed);
        assert_eq!(
            PartitionRows::<2>::from_seed_excluding(100, 4, seed, &[0; 2], &[0; 2]),
            plain
        );
        let (mx, mz) = ([0xFFFF_0000_0000_FFFF, 0b1011], [!0u64, 0]);
        assert!(!plain.avoids(&mx, &mz), "random rows read these columns");
        let rows = PartitionRows::<2>::from_seed_excluding(100, 4, seed, &mx, &mz);
        assert!(rows.avoids(&mx, &mz));
        assert_eq!(rows.bits(), 4);
        // Keys differing only in excluded coordinates share a partition.
        assert_eq!(
            rows.partition_of(&[0x3, 1 << 30], &[0x5, 0]),
            rows.partition_of(&[0x3 ^ (1 << 63), (1 << 30) ^ 0b1000], &[0x5 ^ 0xABCD, 0]),
        );
    }
}

#[test]
fn avoids_reads_both_halves() {
    let rows = PartitionRows::<1>::from_rows(8, vec![[0b0100]], vec![[0b0001]]);
    assert!(rows.avoids(&[0b1011], &[0b1110]));
    assert!(!rows.avoids(&[0b0100], &[0]));
    assert!(!rows.avoids(&[0], &[0b0001]));
    assert!(PartitionRows::<1>::none(8).avoids(&[!0], &[!0]));
}

#[test]
#[should_panic(expected = "covers every column")]
fn excluding_every_column_is_rejected() {
    PartitionRows::<1>::from_seed_excluding(8, 1, 7, &[0xFF], &[0xFF]);
}

#[test]
fn partition_rows_are_salted_away_from_the_hash_rows() {
    // Under one seed the partition rows must not be the hash's first rows.
    for seed in [0x1u64, 0x5EED, crate::pauli_sum::storage::DEFAULT_HASH_SEED] {
        let p = PartitionRows::<2>::from_seed(128, P_MAX_BITS, seed);
        let h = Gf2Hash::<2>::new(128, P_MAX_BITS, seed);
        let (px, pz) = p.rows();
        for i in 0..P_MAX_BITS as usize {
            let (hx, hz) = h.row(i);
            assert!(
                px[i] != hx || pz[i] != hz,
                "seed {seed:#x}: partition row {i} equals hash row {i}"
            );
        }
    }
}

#[test]
fn from_rows_round_trips_after_masking() {
    let rows_x = vec![[!0u64, !0u64], [0x1, 0x0]];
    let rows_z = vec![[0x0u64, 0x3], [0xF, 0x0]];
    let p = PartitionRows::<2>::from_rows(70, rows_x, rows_z);
    let live = (1u64 << (70 - 64)) - 1;
    let (rx, rz) = p.rows();
    assert_eq!(rx, [[!0u64, live], [0x1, 0x0]]);
    assert_eq!(rz, [[0x0u64, 0x3], [0xF, 0x0]]);
    assert_eq!(p.bits(), 2);
}

/// A distributed run is one partition per rank, so `P_MAX_BITS` must allow 64 ranks.
#[test]
fn partition_row_ceiling_covers_64_ranks() {
    // `from_seed` panics with "exceeds P_MAX_BITS" if the constant is still below 6.
    let p = PartitionRows::<2>::from_seed(127, 6, 0xFEED_1234);
    assert_eq!(p.num_partitions(), 64);
}

#[test]
#[should_panic(expected = "exceeds P_MAX_BITS")]
fn partition_from_seed_past_the_maximum_panics() {
    let _ = PartitionRows::<1>::from_seed(64, P_MAX_BITS + 1, 0x1);
}

#[test]
#[should_panic(expected = "exceeds P_MAX_BITS")]
fn partition_from_rows_past_the_maximum_panics() {
    let rows: Vec<[u64; 1]> = (0..=P_MAX_BITS as u64).map(|i| [i + 1]).collect();
    let _ = PartitionRows::<1>::from_rows(64, rows.clone(), rows);
}

#[test]
#[should_panic(expected = "row count mismatch")]
fn partition_from_rows_length_mismatch_panics() {
    let _ = PartitionRows::<1>::from_rows(64, vec![[0x1], [0x2]], vec![[0x1]]);
}

#[test]
#[should_panic(expected = "row 1 masks to zero")]
fn partition_from_rows_all_zero_row_panics() {
    // Row 1 is nonzero only outside the 8 live qubit columns.
    let _ = PartitionRows::<1>::from_rows(8, vec![[0x1], [1 << 20]], vec![[0x0], [1 << 30]]);
}

#[test]
fn partition_from_rows_all_zero_row_is_fine_at_zero_qubits() {
    // Degenerate but legal: with no live columns every row is zero.
    let p = PartitionRows::<1>::from_rows(0, vec![[0x0]], vec![[0x0]]);
    assert_eq!(p.partition_of(&[0], &[0]), 0);
}

#[test]
fn cut_two_blocks_is_one_z_row_over_the_second_block() {
    // 4 qubits, blocks {0,1} | {2,3}. One row, z-only, set on block 1.
    let p = PartitionRows::<1>::cut(4, &[vec![0, 1], vec![2, 3]]);
    assert_eq!(p.bits(), 1);
    assert_eq!(p.num_partitions(), 2);
    let (rx, rz) = p.rows();
    assert_eq!(rx, [[0u64]]);
    assert_eq!(rz, [[0b1100u64]]);

    // A term's label is the XOR of the labels of the blocks it has odd z-weight in. Z0 sits in block 0, label 0; Z2 in block 1, label 1.
    assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(0)), 0);
    assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(2)), 1);
    // X rotation generators are x-only, so a cut row never reads them.
    assert_eq!(p.partition_of_pauli(&PauliString::<1>::x(2)), 0);
    // Bond generators: ZZ(0,1) is inside block 0, ZZ(1,2) crosses the cut,
    // ZZ(2,3) is inside block 1 and so has even z-weight there.
    assert_eq!(p.partition_of(&[0], &[0b0011]), 0);
    assert_eq!(p.partition_of(&[0], &[0b0110]), 1);
    assert_eq!(p.partition_of(&[0], &[0b1100]), 0);
}

#[test]
fn cut_four_blocks_labels_each_block_by_its_index() {
    // 8 qubits in four pairs; row `i` is set on the blocks whose index has bit `i` set, so a single-Z term lands on its own block's label.
    let p = PartitionRows::<1>::cut(8, &[vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
    assert_eq!(p.bits(), 2);
    let (rx, rz) = p.rows();
    assert_eq!(rx, [[0u64], [0u64]]);
    assert_eq!(rz, [[0b1100_1100u64], [0b1111_0000u64]]);
    for (q, want) in [
        (0usize, 0u32),
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 2),
        (5, 2),
        (6, 3),
        (7, 3),
    ] {
        assert_eq!(
            p.partition_of_pauli(&PauliString::<1>::z(q)),
            want,
            "qubit {q}",
        );
    }
    // Two odd blocks XOR their labels: Z2·Z4 -> 1 ^ 2 = 3.
    assert_eq!(p.partition_of(&[0], &[0b0001_0100]), 3);
    // Even z-weight inside one block contributes nothing.
    assert_eq!(p.partition_of(&[0], &[0b0000_1100]), 0);
}

#[test]
fn cut_leaves_uncovered_qubits_in_the_zero_label() {
    let p = PartitionRows::<1>::cut(4, &[vec![0], vec![1]]);
    assert_eq!(p.rows().1, [[0b0010u64]]);
    assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(2)), 0);
    assert_eq!(p.partition_of_pauli(&PauliString::<1>::z(3)), 0);
}

#[test]
fn cut_of_one_block_is_the_trivial_partitioning() {
    assert_eq!(
        PartitionRows::<1>::cut(4, &[vec![0, 1, 2, 3]]),
        PartitionRows::<1>::none(4),
    );
}

#[test]
fn cut_rows_round_trip_across_the_word_boundary() {
    let lo: Vec<usize> = (0..64).collect();
    let hi: Vec<usize> = (64..70).collect();
    let p = PartitionRows::<2>::cut(70, &[lo, hi]);
    assert_eq!(p.rows().0, [[0u64, 0]]);
    assert_eq!(p.rows().1, [[0u64, 0b11_1111]]);
    assert_eq!(p.partition_of_pauli(&PauliString::<2>::z(63)), 0);
    assert_eq!(p.partition_of_pauli(&PauliString::<2>::z(64)), 1);
}

#[test]
#[should_panic(expected = "power of two")]
fn cut_with_three_blocks_panics() {
    let _ = PartitionRows::<1>::cut(4, &[vec![0], vec![1], vec![2]]);
}

#[test]
#[should_panic(expected = "blocks must be disjoint")]
fn cut_with_overlapping_blocks_panics() {
    let _ = PartitionRows::<1>::cut(4, &[vec![0, 1], vec![1, 2]]);
}

#[test]
#[should_panic(expected = "outside 0..4")]
fn cut_with_an_out_of_range_qubit_panics() {
    let _ = PartitionRows::<1>::cut(4, &[vec![0], vec![9]]);
}

#[test]
#[should_panic(expected = "no qubit")]
fn cut_with_an_empty_labelled_block_panics() {
    let _ = PartitionRows::<1>::cut(4, &[vec![0, 1, 2, 3], vec![]]);
}

#[test]
fn partition_occupancy_is_balanced_on_low_weight_keys() {
    // The guard of `occupancy_is_balanced_on_low_weight_keys`, for partition rows.
    let num_qubits = 128;
    let p = PartitionRows::<2>::from_seed(num_qubits, 2, 0x0CC3);
    let mut rng = Xs64::new(0xBA3);
    let mut counts = vec![0usize; p.num_partitions()];
    let target = 4000usize;
    for _ in 0..target {
        let weight = 1 + (rng.next_u64() % 3) as usize;
        let k = low_weight_key::<2>(&mut rng, num_qubits, weight);
        counts[p.partition_of_pauli(&k) as usize] += 1;
    }
    let mean = target / p.num_partitions(); // 1000
    let max = *counts.iter().max().unwrap();
    let min = *counts.iter().min().unwrap();
    assert!(
        max < 2 * mean,
        "max load {max} vs mean {mean}, counts {counts:?}"
    );
    assert!(
        min > mean / 2,
        "min load {min} vs mean {mean}, counts {counts:?}"
    );
}

#[test]
fn seeded_partition_rows_are_independent_of_the_hash() {
    for seed in [
        0x1u64,
        0x5EED,
        0xBEEF,
        crate::pauli_sum::storage::DEFAULT_HASH_SEED,
    ] {
        let h = Gf2Hash::<2>::new(128, 7, seed);
        let p = PartitionRows::<2>::from_seed(128, 3, seed);
        assert!(p.is_independent_of(&h), "seed {seed:#x}");
    }
}

#[test]
fn a_partition_row_copied_from_the_hash_is_not_independent() {
    let h = Gf2Hash::<2>::new(128, 7, 0x5EED);
    let (hx, hz) = h.row(0);
    let seeded = PartitionRows::<2>::from_seed(128, 2, 0x5EED);
    let (sx, sz) = seeded.rows();
    let p = PartitionRows::<2>::from_rows(128, vec![sx[0], hx], vec![sz[0], hz]);
    assert!(!p.is_independent_of(&h));
    // Only the active rows count: at zero bucket bits there is nothing to be dependent on, and the two partition rows are independent among themselves.
    let h0 = Gf2Hash::<2>::new(128, 0, 0x5EED);
    assert!(p.is_independent_of(&h0));
}

mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Linearity over arbitrary keys.
        #[test]
        fn hash_is_gf2_linear_w2(
            ax in any::<[u64; 2]>(), az in any::<[u64; 2]>(),
            bx in any::<[u64; 2]>(), bz in any::<[u64; 2]>(),
            bits in 0u8..=13u8,
            seed in any::<u64>(),
        ) {
            let h = Gf2Hash::<2>::new(128, bits, seed);
            let cx = [ax[0] ^ bx[0], ax[1] ^ bx[1]];
            let cz = [az[0] ^ bz[0], az[1] ^ bz[1]];
            prop_assert_eq!(
                h.bucket_of(&cx, &cz),
                h.bucket_of(&ax, &az) ^ h.bucket_of(&bx, &bz)
            );
        }

        /// Refining keeps the low bits, so a bucket only ever splits.
        #[test]
        fn refine_is_a_prefix_extension_w1(
            x in any::<[u64; 1]>(), z in any::<[u64; 1]>(),
            bits in 0u8..=12u8,
            seed in any::<u64>(),
        ) {
            let mut h = Gf2Hash::<1>::new(64, bits, seed);
            let before = h.bucket_of(&x, &z);
            h.refine();
            let after = h.bucket_of(&x, &z);
            let mask = (1u32 << bits) - 1;
            prop_assert_eq!(after & mask, before);
        }

        /// The partition map is GF(2)-linear.
        #[test]
        fn partition_of_is_gf2_linear_w1(
            ax in any::<[u64; 1]>(), az in any::<[u64; 1]>(),
            bx in any::<[u64; 1]>(), bz in any::<[u64; 1]>(),
            bits in 0u8..=P_MAX_BITS,
            seed in any::<u64>(),
        ) {
            let p = PartitionRows::<1>::from_seed(64, bits, seed);
            let cx = [ax[0] ^ bx[0]];
            let cz = [az[0] ^ bz[0]];
            prop_assert_eq!(
                p.partition_of(&cx, &cz),
                p.partition_of(&ax, &az) ^ p.partition_of(&bx, &bz)
            );
        }

        #[test]
        fn partition_of_is_gf2_linear_w2(
            ax in any::<[u64; 2]>(), az in any::<[u64; 2]>(),
            bx in any::<[u64; 2]>(), bz in any::<[u64; 2]>(),
            bits in 0u8..=P_MAX_BITS,
            seed in any::<u64>(),
        ) {
            let p = PartitionRows::<2>::from_seed(128, bits, seed);
            let cx = [ax[0] ^ bx[0], ax[1] ^ bx[1]];
            let cz = [az[0] ^ bz[0], az[1] ^ bz[1]];
            prop_assert_eq!(
                p.partition_of(&cx, &cz),
                p.partition_of(&ax, &az) ^ p.partition_of(&bx, &bz)
            );
        }
    }
}
