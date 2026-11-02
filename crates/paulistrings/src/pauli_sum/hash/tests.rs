use super::*;
use crate::pauli_string::PauliString;
use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
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
    let mut p = PauliString::<W> { x: [0u64; W], z: [0u64; W] };
    for w in 0..W {
        let mask = word_mask(num_qubits, w);
        p.x[w] = rng.next_u64() & mask;
        p.z[w] = rng.next_u64() & mask;
    }
    p
}

fn bucket<const W: usize>(hash: &Gf2Hash<W>, p: &PauliString<W>) -> u32 {
    hash.bucket_of(&p.x, &p.z)
}

fn partition<const W: usize>(rows: &PartitionRows<W>, p: &PauliString<W>) -> u32 {
    rows.partition_of(&p.x, &p.z)
}

fn low_weight_key<const W: usize>(
    rng: &mut Xs64,
    num_qubits: usize,
    weight: usize,
) -> PauliString<W> {
    let mut p = PauliString::<W> { x: [0u64; W], z: [0u64; W] };
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

/// The `(k, 2 · num_qubits)` flag matrix whose row `i` reads `coordinates[i]`.
fn matrix(num_qubits: usize, coordinates: &[Vec<usize>]) -> Vec<Vec<bool>> {
    coordinates.iter().map(|row| (0..2 * num_qubits).map(|c| row.contains(&c)).collect()).collect()
}

/// The coordinates a row sets: `q` for an x-bit on qubit `q`, `num_qubits + q` for a z-bit.
fn coordinates<const W: usize>(row: ([u64; W], [u64; W]), num_qubits: usize) -> Vec<usize> {
    (0..2 * num_qubits)
        .filter(|&c| {
            let (half, q) = if c < num_qubits { (row.0, c) } else { (row.1, c - num_qubits) };
            (half[q / 64] >> (q % 64)) & 1 == 1
        })
        .collect()
}

#[test]
fn max_hash_bits_is_one_per_coordinate_up_to_the_index_width() {
    assert_eq!(max_hash_bits(0), 0);
    assert_eq!(max_hash_bits(1), 2);
    assert_eq!(max_hash_bits(15), 30);
    assert_eq!(max_hash_bits(16), 31);
    assert_eq!(max_hash_bits(1024), 31);
    assert_eq!(Gf2Hash::<2>::new(100, 3, 1).max_bits(), 31);
    assert_eq!(Gf2Hash::<1>::new(5, 3, 1).max_bits(), 10);
}

#[test]
fn bits_beyond_num_qubits_do_not_affect_the_bucket() {
    // 100 qubits in W=2: setting the dead bits 100..128 must not move a term.
    let h = Gf2Hash::<2>::new(100, 10, 0x99);
    let mut rng = Xs64::new(21);
    for _ in 0..500 {
        let p = rand_key::<2>(&mut rng, 100);
        let mut polluted = p;
        let dead = !((1u64 << (100 - 64)) - 1);
        polluted.x[1] |= dead;
        polluted.z[1] |= dead;
        assert_eq!(bucket(&h, &p), bucket(&h, &polluted));
    }
    // Directly: a key that is *only* out-of-range bits hashes to 0.
    let h = Gf2Hash::<2>::new(70, 12, 0x7A);
    let dead = !((1u64 << (70 - 64)) - 1);
    assert_eq!(h.bucket_of(&[0, dead], &[0, dead]), 0);
}

#[test]
fn the_hash_is_linear() {
    let h = Gf2Hash::<2>::new(100, 12, 0x5EED);
    let mut rng = Xs64::new(0x11E);
    for _ in 0..200 {
        let p = rand_key::<2>(&mut rng, 100);
        let q = rand_key::<2>(&mut rng, 100);
        assert_eq!(bucket(&h, &xor(&p, &q)), bucket(&h, &p) ^ bucket(&h, &q));
    }
}

/// The XOR-fold in `row_parity` / `partition_of` agrees with a naive one-popcount-per-word oracle, and `bucket_of` packs row `i` at bit `i`.
#[test]
fn row_parity_matches_a_per_word_popcount_and_bucket_of_packs_it() {
    fn naive<const W: usize>(x: &[u64; W], z: &[u64; W], rx: &[u64; W], rz: &[u64; W]) -> u32 {
        let mut parity: u32 = 0;
        for w in 0..W {
            parity ^= (x[w] & rx[w]).count_ones();
            parity ^= (z[w] & rz[w]).count_ones();
        }
        parity & 1
    }

    fn check<const W: usize>(num_qubits: usize, seed: u64) {
        let h = Gf2Hash::<W>::new(num_qubits, max_hash_bits(num_qubits), seed);
        let mut rng = Xs64::new(seed ^ 0x5EED);
        for _ in 0..200 {
            let p = rand_key::<W>(&mut rng, num_qubits);
            let full = bucket(&h, &p);
            for row in 0..h.max_bits() {
                let bit = h.row_parity(&p.x, &p.z, row);
                let (rx, rz) = h.row(row as usize);
                assert_eq!(bit, naive(&p.x, &p.z, &rx, &rz), "W={W} row={row}");
                assert_eq!(bit, (full >> row) & 1, "W={W} row={row}");
            }
        }
    }

    check::<1>(64, 0xA11CE);
    check::<2>(128, 0xB0B);
    check::<8>(512, 0xD00D);
}

#[test]
fn refine_preserves_the_low_bits_and_coarsen_inverts_it() {
    let mut h = Gf2Hash::<2>::new(128, 6, 0xB0B);
    let mut rng = Xs64::new(31);
    let keys: Vec<PauliString<2>> = (0..500).map(|_| rand_key::<2>(&mut rng, 128)).collect();
    let before: Vec<u32> = keys.iter().map(|k| bucket(&h, k)).collect();

    h.refine();
    assert_eq!(h.num_buckets(), 128);
    let mask = (1u32 << 6) - 1;
    for (k, &b) in keys.iter().zip(before.iter()) {
        assert_eq!(bucket(&h, k) & mask, b);
    }
    h.coarsen();
    assert_eq!(h.bits(), 6);
    let after: Vec<u32> = keys.iter().map(|k| bucket(&h, k)).collect();
    assert_eq!(before, after);
}

#[test]
#[should_panic(expected = "already at the 31 rows")]
fn refine_past_the_rows_panics() {
    let mut h = Gf2Hash::<1>::new(64, 31, 0x1);
    h.refine();
}

#[test]
#[should_panic(expected = "already at a single bucket")]
fn coarsen_below_one_bucket_panics() {
    let mut h = Gf2Hash::<1>::new(64, 0, 0x1);
    h.coarsen();
}

#[test]
#[should_panic(expected = "bits 11 exceeds the 10 rows of 5 qubits")]
fn constructing_past_the_rows_panics() {
    let _ = Gf2Hash::<1>::new(5, 11, 0x1);
}

#[test]
fn different_seeds_give_different_hashes() {
    let a = Gf2Hash::<2>::new(128, 10, 0x5EED);
    let b = Gf2Hash::<2>::new(128, 10, 0x5EEE);
    assert!(!a.same_rows_as(&b));
    let mut rng = Xs64::new(71);
    let differs = (0..500)
        .map(|_| rand_key::<2>(&mut rng, 128))
        .filter(|p| bucket(&a, p) != bucket(&b, p))
        .count();
    // Two independent hashes agree on a given key with probability 2^-10.
    assert!(differs > 400, "expected most keys to hash differently, got {differs}/500");
}

/// Every drawn hash is full rank, down to the one-qubit register where a plain draw gives a zero or repeated row most of the time; zero qubits is legitimately all zero.
#[test]
fn drawn_rows_are_full_rank_at_every_qubit_count() {
    for n in [1usize, 3, 5, 11, 16, 17, 64] {
        let h = Gf2Hash::<1>::new(n, 0, 0x1);
        assert_eq!(h.independent_rows(), max_hash_bits(n) as usize, "{n} qubits");
        assert_eq!(h.rank_deficiency(), None, "{n} qubits");
    }
    let h = Gf2Hash::<1>::new(0, 0, 0x1);
    assert_eq!(h.max_bits(), 0);
    assert_eq!(h.bucket_of(&[0], &[0]), 0);
}

/// Independence only redraws a dependent row, which over 127 qubits never happens: the rows are the plain splitmix64 draw.
#[test]
fn the_rows_are_the_plain_draw_where_that_is_already_full_rank() {
    let h = Gf2Hash::<2>::new(127, 4, DEFAULT_HASH_SEED);
    for i in 0..h.max_bits() as usize {
        let mut plain = ([0u64; 2], [0u64; 2]);
        for w in 0..2 {
            plain.0[w] = row_word(DEFAULT_HASH_SEED, i, 0, w, 0) & word_mask(127, w);
            plain.1[w] = row_word(DEFAULT_HASH_SEED, i, 0, w, 1) & word_mask(127, w);
        }
        assert_eq!(h.row(i), plain, "row {i}");
    }
}

/// One step of xorshift64 (13, 7, 17), a GF(2)-linear map `M` on `u64`.
fn xorshift64_step(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Consecutive outputs of a linear generator satisfy `rows_z = M·rows_x` word for word under every seed.
#[test]
fn row_words_are_not_one_xorshift_step_apart() {
    for seed in [0x1u64, 0x5EED, DEFAULT_HASH_SEED] {
        let h = Gf2Hash::<2>::new(128, 31, seed);
        let p = PartitionRows::<2>::from_seed(128, 6, seed);
        let (px, pz) = p.rows();
        let (hx, hz) = h.matrix.rows();
        let linked = hx
            .iter()
            .zip(hz)
            .chain(px.iter().zip(pz))
            .flat_map(|(rx, rz)| (0..2).map(move |w| xorshift64_step(rx[w]) == rz[w]))
            .filter(|&linked| linked)
            .count();
        assert_eq!(linked, 0, "seed {seed:#x}: {linked} row words are M·x");
    }
}

/// Row word `w` depends on `(seed, row, w, x-or-z)` alone, so the same `(num_qubits, seed)` gives the same rows at every width.
#[test]
fn rows_do_not_depend_on_the_width() {
    let seed = DEFAULT_HASH_SEED;
    for n in [1usize, 5, 64] {
        let h1 = Gf2Hash::<1>::new(n, 0, seed);
        let h2 = Gf2Hash::<2>::new(n, 0, seed);
        for i in 0..h1.max_bits() as usize {
            let (x1, z1) = h1.row(i);
            assert_eq!(h2.row(i), ([x1[0], 0], [z1[0], 0]), "n={n} row {i}");
        }
        let p1 = PartitionRows::<1>::from_seed(n, 2, seed);
        let p2 = PartitionRows::<2>::from_seed(n, 2, seed);
        let widened = |rows: &[[u64; 1]]| rows.iter().map(|r| [r[0], 0]).collect::<Vec<_>>();
        assert_eq!(p2.rows().0, widened(p1.rows().0), "n={n} partition x-rows");
        assert_eq!(p2.rows().1, widened(p1.rows().1), "n={n} partition z-rows");
    }
    let h2 = Gf2Hash::<2>::new(128, 8, seed);
    let h4 = Gf2Hash::<4>::new(128, 8, seed);
    for i in 0..h2.max_bits() as usize {
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
            counts[bucket(&h, &p) as usize] += 1;
        }
    }

    let mean = target / b; // 128
    let max = *counts.iter().max().unwrap();
    let min = *counts.iter().min().unwrap();
    // Deterministic given the seeds, so these bounds are not flaky.
    assert!(max < 2 * mean, "max load {max} vs mean {mean}");
    assert!(min > mean / 2, "min load {min} vs mean {mean}");
}

/// Rank is monotone in the number of active bucket bits, since the active hash is a prefix of one fixed matrix: refining can only separate deltas that were colliding, never merge separated ones.
#[test]
fn support_delta_rank_is_monotone_in_bits() {
    for seed in [0x1u64, 0xBEEF, DEFAULT_HASH_SEED] {
        let mut last = 0usize;
        for bits in 0..=12u8 {
            let h = Gf2Hash::<2>::new(128, bits, seed);
            let r = crate::test_support::support_delta_rank(&h, &[0, 1]);
            assert!(r >= last && r <= 4, "seed {seed:#x}: rank went {last} -> {r} at bits {bits}");
            assert!(r <= bits as usize, "rank {r} exceeds bits {bits}");
            last = r;
        }
    }
}

/// A support delta cannot reorder a bucket's key column exactly when `h` separates the support's delta space; at the default seed `(0, 1)` is full rank at 7 bits and `(0, 7)` one short.
#[test]
fn support_delta_preserves_bucket_order_iff_the_delta_span_is_full_rank() {
    use crate::test_support::support_delta_rank as rank;
    assert_eq!(rank(&Gf2Hash::<2>::new(128, 7, DEFAULT_HASH_SEED), &[0, 1]), 4);
    assert_eq!(rank(&Gf2Hash::<1>::new(64, 7, DEFAULT_HASH_SEED), &[0, 7]), 3);
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
    let h = Gf2Hash::<W>::new(num_qubits, bits, DEFAULT_HASH_SEED);
    // Enumerate the support's delta space: one bit per (qubit, x-or-z).
    let gens: Vec<PauliString<W>> =
        support.iter().flat_map(|&q| [PauliString::<W>::x(q), PauliString::<W>::z(q)]).collect();
    let local = |combo: usize| -> PauliString<W> {
        let mut d = PauliString::<W> { x: [0u64; W], z: [0u64; W] };
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
            buckets[bucket(&h, &p) as usize].push((p.x, p.z));
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
fn given_rows_lead_and_drawn_rows_complete_the_hash_to_full_rank() {
    let given = matrix(127, &[vec![0, 127], vec![5], vec![3, 64, 200]]);
    let h = Gf2Hash::<2>::from_matrix(127, 6, 0x5EED, &given);
    for (i, row) in given.iter().enumerate() {
        assert_eq!(&coordinates(h.row(i), 127), &(0..254).filter(|&c| row[c]).collect::<Vec<_>>());
    }
    assert_eq!(h.max_bits(), 31);
    assert_eq!(h.rank_deficiency(), None);
    let dense = Gf2Hash::<2>::new(127, 6, 7);
    let (rows_x, rows_z) = dense.rows();
    let copied = Gf2Hash::<2>::from_rows(127, 6, 99, rows_x.to_vec(), rows_z.to_vec());
    assert!(copied.same_rows_as(&dense), "rows decide sameness, not how they were made");
    assert_eq!(copied.seed(), 99);
}

#[test]
fn dependent_given_rows_are_accepted_and_reported() {
    let h = Gf2Hash::<1>::from_matrix(40, 2, 1, &matrix(40, &[vec![3], vec![3], vec![]]));
    // A duplicate and an empty row add nothing, so the three given rows have rank 1 and the 28 drawn ones complete it to 29.
    assert_eq!(h.independent_rows(), 29);
    assert!(h.rank_deficiency().unwrap().contains("31 rows have rank 29"));
    assert_eq!(h.bucket_of(&[1 << 3], &[0]), 0b011, "both copies of the row read qubit 3's x-bit");
    // Given rows are masked to the live columns before the completion is drawn against them.
    let h = Gf2Hash::<1>::from_rows(8, 1, 1, vec![[1 << 20]], vec![[0]]);
    assert_eq!(h.independent_rows(), 15);
}

#[test]
#[should_panic(expected = "row 0 has 253 entries, not the 254 coordinates")]
fn a_matrix_row_of_the_wrong_width_panics() {
    Gf2Hash::<2>::from_matrix(127, 2, 1, &[vec![false; 253]]);
}

#[test]
#[should_panic(expected = "11 rows exceed the 10 rows of 5 qubits")]
fn more_given_rows_than_the_register_holds_panics() {
    Gf2Hash::<1>::from_matrix(5, 2, 1, &vec![vec![false; 10]; 11]);
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
    assert_eq!(partition(&p, &PauliString::<1>::x(0)), 1);
    // Z_0: bit 0 = 0, bit 1 = parity(0b001 & 0b101) = 1.
    assert_eq!(partition(&p, &PauliString::<1>::z(0)), 2);
    // Y_0 = X_0 ⊕ Z_0.
    assert_eq!(p.partition_of(&[1], &[1]), 3);
    // X_0 X_1: parity(0b011 & 0b011) = 0.
    assert_eq!(p.partition_of(&[0b11], &[0]), 0);
    // Z_2: parity(0b100 & 0b101) = 1.
    assert_eq!(p.partition_of(&[0], &[0b100]), 2);
    // The same rows as a flag matrix, x-bits first.
    assert_eq!(PartitionRows::<1>::from_matrix(8, &matrix(8, &[vec![0, 1], vec![8, 10]])), p);
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
        assert_eq!(PartitionRows::<2>::from_seed_excluding(100, 4, seed, &[0; 2], &[0; 2]), plain);
        let (mx, mz) = ([0xFFFF_0000_0000_FFFF, 0b1011], [!0u64, 0]);
        assert!(!plain.avoids(&mx, &mz), "random rows read these columns");
        let rows = PartitionRows::<2>::from_seed_excluding(100, 4, seed, &mx, &mz);
        assert!(rows.avoids(&mx, &mz));
        assert_eq!(rows.bits(), 4);
        assert_eq!(rows.independent_rows(), 4);
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

/// An exclusion that leaves nothing readable builds zero rows and says so, like every other rank shortfall.
#[test]
fn excluding_every_column_gives_zero_rows_and_a_report() {
    let rows = PartitionRows::<1>::from_seed_excluding(8, 1, 7, &[0xFF], &[0xFF]);
    assert_eq!(rows.rows(), (&[[0u64]][..], &[[0u64]][..]));
    assert!(rows.rank_deficiency().unwrap().contains("only 1 of the 2 partitions"));
    // One readable coordinate: the one row reads it, and a second row can only repeat it.
    let rows = PartitionRows::<1>::from_seed_excluding(8, 2, 7, &[0xFE], &[0xFF]);
    assert_eq!(rows.rows().0, [[1u64], [1u64]]);
    assert_eq!(rows.independent_rows(), 1);
}

#[test]
fn partition_rows_are_salted_away_from_the_hash_rows() {
    // Under one seed the partition rows must not be the hash's first rows.
    for seed in [0x1u64, 0x5EED, DEFAULT_HASH_SEED] {
        let p = PartitionRows::<2>::from_seed(128, 6, seed);
        let h = Gf2Hash::<2>::new(128, 6, seed);
        let (px, pz) = p.rows();
        for i in 0..6 {
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

#[test]
#[should_panic(expected = "7 rows exceed the 6 rows of 3 qubits")]
fn partition_rows_past_the_register_panic() {
    let _ = PartitionRows::<1>::from_seed(3, 7, 0x1);
}

#[test]
#[should_panic(expected = "row count mismatch")]
fn partition_from_rows_length_mismatch_panics() {
    let _ = PartitionRows::<1>::from_rows(64, vec![[0x1], [0x2]], vec![[0x1]]);
}

#[test]
fn partition_rank_deficiency_names_dependent_rows() {
    // Row 1 is nonzero only outside the 8 live qubit columns.
    let zero = PartitionRows::<1>::from_rows(8, vec![[0x1], [1 << 20]], vec![[0x0], [1 << 30]]);
    assert_eq!(zero.independent_rows(), 1);
    let duplicate = PartitionRows::<1>::from_matrix(40, &matrix(40, &[vec![7], vec![7]]));
    for rows in [zero, duplicate] {
        assert!(rows.rank_deficiency().unwrap().contains("only 2 of the 4 partitions"));
    }
    assert_eq!(PartitionRows::<1>::from_seed(40, 2, 11).rank_deficiency(), None);
}

#[test]
fn overlap_onset_is_the_bucket_bit_count_where_the_rows_meet() {
    let hash = Gf2Hash::<1>::new(40, 4, 3);
    let (x, z) = hash.row(2);
    assert_eq!(PartitionRows::<1>::from_rows(40, vec![x], vec![z]).overlap_onset(&hash), Some(3));
    let (x0, z0) = hash.row(0);
    let (x1, z1) = hash.row(1);
    let sum = PartitionRows::<1>::from_rows(40, vec![[x0[0] ^ x1[0]]], vec![[z0[0] ^ z1[0]]]);
    assert_eq!(sum.overlap_onset(&hash), Some(2), "a combination of rows meets at the later one");
    assert_eq!(PartitionRows::<1>::from_seed(40, 2, 11).overlap_onset(&hash), None);
    // Four qubits: eight coordinates, so once the joint span is full any further overlap is forced and not reported.
    let small = Gf2Hash::<1>::new(4, 2, 5);
    assert_eq!(PartitionRows::<1>::from_seed(4, 1, 6).overlap_onset(&small), None);
}

#[test]
fn seeded_partition_rows_are_independent_of_the_hash() {
    for seed in [0x1u64, 0x5EED, 0xBEEF, DEFAULT_HASH_SEED] {
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
    assert_eq!(partition(&p, &PauliString::<1>::z(0)), 0);
    assert_eq!(partition(&p, &PauliString::<1>::z(2)), 1);
    // X rotation generators are x-only, so a cut row never reads them.
    assert_eq!(partition(&p, &PauliString::<1>::x(2)), 0);
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
    for (q, want) in [(0usize, 0u32), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3)] {
        assert_eq!(partition(&p, &PauliString::<1>::z(q)), want, "qubit {q}",);
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
    assert_eq!(partition(&p, &PauliString::<1>::z(2)), 0);
    assert_eq!(partition(&p, &PauliString::<1>::z(3)), 0);
    assert_eq!(PartitionRows::<1>::cut(4, &[vec![0, 1, 2, 3]]), PartitionRows::<1>::none(4));
}

#[test]
fn cut_rows_round_trip_across_the_word_boundary() {
    let lo: Vec<usize> = (0..64).collect();
    let hi: Vec<usize> = (64..70).collect();
    let p = PartitionRows::<2>::cut(70, &[lo, hi]);
    assert_eq!(p.rows().0, [[0u64, 0]]);
    assert_eq!(p.rows().1, [[0u64, 0b11_1111]]);
    assert_eq!(partition(&p, &PauliString::<2>::z(63)), 0);
    assert_eq!(partition(&p, &PauliString::<2>::z(64)), 1);
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

mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
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
    }
}
