//! The GF(2)-linear bucket function `h(v) = H·v`. See ARCHITECTURE.md §Hash.

use crate::pauli_string::PauliString;

/// Maximum number of bucket bits, i.e. `B ≤ 2^20 = 1_048_576` buckets.
///
/// Rows for all `B_MAX_BITS` bits are generated up front so that
/// [`Gf2Hash::refine`] is free: the active hash is always a *prefix* of the same
/// fixed matrix, which is what makes bucket refinement a single parity pass
/// rather than a re-hash.
pub const B_MAX_BITS: u8 = 20;

/// Xorshift64 — deterministic row generation without pulling in an RNG crate.
///
/// Matches the generator used for reproducible benchmark input, so `H` is
/// reproducible from `(num_qubits, seed)` alone on any machine.
struct Xs64(u64);

impl Xs64 {
    fn new(seed: u64) -> Self {
        // Avoid the degenerate all-zero state.
        Self(seed | 1)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Mask of the live qubit bits in word `word`, given `num_qubits` total.
///
/// Same construction as [`PauliString::is_within`]; kept separate because that
/// method folds the words together and we need them individually.
#[inline]
fn word_mask(num_qubits: usize, word: usize) -> u64 {
    let lo = 64 * word;
    if num_qubits >= lo + 64 {
        !0u64
    } else if num_qubits <= lo {
        0
    } else {
        (1u64 << (num_qubits - lo)) - 1
    }
}

/// A GF(2)-linear hash from Pauli keys to bucket indices.
///
/// `h(v) = H·v` for a fixed dense random `H ∈ GF(2)^{b × 2n}`, where `v = (x, z)`
/// is the symplectic key of a [`PauliString`]. Bit `i` of the result is the
/// parity of `(x & rows_x[i]) ^ (z & rows_z[i])`.
///
/// # Why dense and random
///
/// A coordinate projection (bucket = some chosen key bits) is GF(2)-linear too,
/// and has the appealing property that a gate acting away from the chosen
/// coordinates leaves every term in its own bucket. It is nevertheless wrong
/// here: `WeightCutoff` truncation keeps sums *low-weight*, so the chosen
/// coordinates are almost always zero and essentially everything lands in bucket
/// 0. A dense random `H` is a universal hash family on `GF(2)^{2n}` — for `m`
/// distinct keys the maximum bucket load is `m/B + O(√(m log B / B))` with high
/// probability, *independent of the input's structure*. See
/// ARCHITECTURE.md §Hash, and the `occupancy_*` tests below, which pin the
/// property.
///
/// Cost is `b × 2W` AND + popcount-parity operations, evaluated only at
/// ingestion and at rehash — **never in the propagation loop**, where buckets
/// are tracked structurally by XOR instead.
///
/// # Examples
///
/// ```
/// use paulistrings::bucket::Gf2Hash;
/// use paulistrings::PauliString;
///
/// let h = Gf2Hash::<1>::new(64, 6, 0xC0FFEE);
/// assert_eq!(h.num_buckets(), 64);
///
/// // Linearity: h(v ^ w) == h(v) ^ h(w).
/// let v = PauliString::<1>::x(3);
/// let w = PauliString::<1>::z(11);
/// let xor = PauliString::<1> { x: [v.x[0] ^ w.x[0]], z: [v.z[0] ^ w.z[0]] };
/// assert_eq!(h.bucket_of_pauli(&xor), h.bucket_of_pauli(&v) ^ h.bucket_of_pauli(&w));
/// ```
#[derive(Clone, Debug)]
pub struct Gf2Hash<const W: usize> {
    /// X-part of each row of `H`, masked to the live qubit columns.
    rows_x: Vec<[u64; W]>,
    /// Z-part of each row of `H`, masked to the live qubit columns.
    rows_z: Vec<[u64; W]>,
    /// Active prefix length: `B = 1 << bits` buckets. `0 ≤ bits ≤ B_MAX_BITS`.
    bits: u8,
    /// Seed the rows were generated from. Kept so the hash is reproducible and
    /// so two sums can be checked for compatibility.
    seed: u64,
    /// Qubit count the rows were masked against.
    num_qubits: usize,
}

impl<const W: usize> Gf2Hash<W> {
    /// Build a hash over `num_qubits` qubits with `bits` active bucket bits.
    ///
    /// Rows are generated deterministically from `seed`, so two `Gf2Hash`
    /// values with the same `(num_qubits, seed)` are identical and their sums
    /// are combinable.
    ///
    /// # Panics
    ///
    /// Panics if `bits > B_MAX_BITS`, or in debug builds if
    /// `num_qubits > 64 · W`.
    pub fn new(num_qubits: usize, bits: u8, seed: u64) -> Self {
        assert!(
            bits <= B_MAX_BITS,
            "Gf2Hash: bits {bits} exceeds B_MAX_BITS {B_MAX_BITS}",
        );
        debug_assert!(num_qubits <= 64 * W);

        let mut rng = Xs64::new(seed);
        let n_rows = B_MAX_BITS as usize;
        let mut rows_x: Vec<[u64; W]> = Vec::with_capacity(n_rows);
        let mut rows_z: Vec<[u64; W]> = Vec::with_capacity(n_rows);

        // `num_qubits == 0` has a single key (the identity), so every row is
        // legitimately zero and the retry below must not spin.
        let has_live_columns = num_qubits > 0;

        for _ in 0..n_rows {
            // A row that masks to all-zero would contribute a constant 0 bit,
            // wasting a bucket bit. Vanishingly unlikely for a reasonable qubit
            // count, but at `num_qubits = 1` there are only 2 live columns and
            // the chance is 1/4 per row, so retry rather than silently degrade.
            let (rx, rz) = loop {
                let mut rx = [0u64; W];
                let mut rz = [0u64; W];
                let mut any = false;
                for w in 0..W {
                    let mask = word_mask(num_qubits, w);
                    rx[w] = rng.next_u64() & mask;
                    rz[w] = rng.next_u64() & mask;
                    any |= (rx[w] | rz[w]) != 0;
                }
                if any || !has_live_columns {
                    break (rx, rz);
                }
            };
            rows_x.push(rx);
            rows_z.push(rz);
        }

        Self {
            rows_x,
            rows_z,
            bits,
            seed,
            num_qubits,
        }
    }

    /// Number of active bucket bits.
    #[inline]
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Number of buckets, `1 << bits()`.
    #[inline]
    pub fn num_buckets(&self) -> usize {
        1usize << self.bits
    }

    /// The seed the rows were generated from.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The qubit count the rows were masked against.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// `h(v)` for a key given as separate `x` and `z` words.
    ///
    /// This is the SoA-friendly entry point: the engine and the bucketed sum
    /// hold parallel `x`/`z` columns, not [`PauliString`] values.
    #[inline]
    pub fn bucket_of(&self, x: &[u64; W], z: &[u64; W]) -> u32 {
        let mut acc: u32 = 0;
        for i in 0..self.bits as usize {
            acc |= self.row_parity(x, z, i as u8) << i;
        }
        acc
    }

    /// One row of `H·v`: the parity of `(x & rows_x[row]) ^ (z & rows_z[row])`,
    /// as `0` or `1`.
    ///
    /// [`Self::bucket_of`] is this evaluated for every row `0..bits` and
    /// assembled into one `u32`; this is the single place that body lives, so
    /// a caller that needs only the *new* bit a [`Self::refine`] just
    /// introduced — [`crate::bucket::sum::PauliSum::refine`] — can get it in
    /// `O(2W)` instead of paying `O(bits · 2W)` for the whole prefix.
    #[inline]
    pub(crate) fn row_parity(&self, x: &[u64; W], z: &[u64; W], row: u8) -> u32 {
        let rx = &self.rows_x[row as usize];
        let rz = &self.rows_z[row as usize];
        let mut parity: u32 = 0;
        for w in 0..W {
            parity ^= (x[w] & rx[w]).count_ones();
            parity ^= (z[w] & rz[w]).count_ones();
        }
        parity & 1
    }

    /// `h(v)` for a [`PauliString`]. Convenience wrapper over [`Self::bucket_of`].
    ///
    /// Also the form used for a channel's delta vectors at prepare time, where
    /// `h(d)` is computed once per layer rather than per term.
    #[inline]
    pub fn bucket_of_pauli(&self, p: &PauliString<W>) -> u32 {
        self.bucket_of(&p.x, &p.z)
    }

    /// Double the bucket count: `B → 2B`.
    ///
    /// Because the active hash is a *prefix* of a fixed matrix, refining splits
    /// each existing bucket in two and the within-bucket order is inherited by
    /// both halves — an `O(n)` parity pass with no re-sorting.
    ///
    /// # Panics
    ///
    /// Panics if already at [`B_MAX_BITS`].
    #[inline]
    pub fn refine(&mut self) {
        assert!(
            self.bits < B_MAX_BITS,
            "Gf2Hash::refine: already at B_MAX_BITS {B_MAX_BITS}",
        );
        self.bits += 1;
    }

    /// Halve the bucket count: `B → B/2`. Merges bucket pairs `(2i, 2i+1)`.
    ///
    /// # Panics
    ///
    /// Panics if already at a single bucket.
    #[inline]
    pub fn coarsen(&mut self) {
        assert!(
            self.bits > 0,
            "Gf2Hash::coarsen: already at a single bucket"
        );
        self.bits -= 1;
    }

    /// `true` if `other` was generated with the same rows, so sums partitioned
    /// by the two can be combined (after matching `bits`).
    #[inline]
    pub fn same_rows_as(&self, other: &Self) -> bool {
        self.seed == other.seed && self.num_qubits == other.num_qubits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ---- range and the identity key ----

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
        // h(0) = 0 for any linear h. Documented wart: the identity string always
        // lands in bucket 0.
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

    // ---- linearity: the property everything else rests on ----

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

    // ---- column masking ----

    #[test]
    fn bits_beyond_num_qubits_do_not_affect_the_bucket() {
        // 100 qubits in W=2: bits 100..128 are dead. Setting them must not move
        // a term, or `PauliSum`'s `is_within` contract and the hash would
        // disagree about which keys are distinguishable.
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

    // ---- row_parity ----

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

    // ---- refine / coarsen prefix consistency ----

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
            // Refining splits each bucket in two: the new index agrees with the
            // old one on the low `bits` bits, so within-bucket order is
            // inherited by both halves.
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
            // Dropping the top bit merges (b, b + B/2) — the pair that differs
            // only in the bit being dropped.
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

    // ---- reproducibility ----

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
        // At num_qubits = 1 there are only 2 live columns, so a naive generator
        // produces an all-zero (and therefore useless) row 1/4 of the time.
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

    // ---- occupancy: what guards the choice of a dense random H ----

    /// Bucket occupancy on **low-weight** keys, the physically relevant regime.
    ///
    /// This is the test that fails for a coordinate-projection `H`: weight-4
    /// strings over 64 qubits leave any fixed handful of key coordinates zero
    /// almost always, so projection dumps nearly everything into bucket 0.
    /// A dense random `H` spreads them.
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
        // Deterministic given the seeds, so these bounds are not flaky. A
        // projection hash would put >99% of the mass in one bucket and blow the
        // upper bound by two orders of magnitude.
        assert!(max < 2 * mean, "max load {max} vs mean {mean}");
        assert!(min > mean / 2, "min load {min} vs mean {mean}");
    }

    // ---- rank of `h` on a channel's delta space ----
    //
    // These pin the mechanism in `research/notes/2026-09-01-bucket-cliff.md`.
    // A channel supported on qubits `{i, j}` has a 4-dimensional key-delta
    // space `span{X_i, Z_i, X_j, Z_j}`; the engine's coset dimension is
    // `r = rank(h(D))` (`engine::coset::Gf2Span::r`), and the per-run sort's
    // comparison count collapses to its `log2(fanout)` floor exactly when
    // `r` is full (4). `r` is *not* a property of the channel alone: it
    // depends on which rows `H` happens to have, so it moves with the hash
    // seed and — because `Gf2Hash::new` draws `2W` words per row — with `W`.

    /// Occupancy balance is not the whole story: a dense random `H` can still
    /// fail to *separate* a two-qubit channel's four delta generators, and
    /// then two distinct local deltas share one bucket delta.
    #[test]
    fn support_delta_rank_is_usually_full_but_not_always() {
        // Deterministic given the seed, so these counts are not flaky.
        let h = Gf2Hash::<2>::new(128, 7, crate::bucket::sum::DEFAULT_HASH_SEED);
        let mut deficient = 0usize;
        let mut total = 0usize;
        for i in 0..128u32 {
            for j in (i + 1)..128u32 {
                total += 1;
                if crate::test_support::support_delta_rank(&h, &[i, j]) < 4 {
                    deficient += 1;
                }
            }
        }
        // ~10% of placements at the default bucket-count floor (B = 128).
        // The bound is loose on purpose: it pins the order of magnitude, which
        // is the load-bearing fact, not the exact draw.
        assert_eq!(total, 8128);
        assert!(
            (200..2000).contains(&deficient),
            "expected O(10%) rank-deficient support pairs at 7 bucket bits, got {deficient}/{total}"
        );
    }

    /// Rank is monotone in the number of active bucket bits, since the active
    /// hash is a *prefix* of one fixed matrix: refining can only separate
    /// deltas that were colliding, never merge separated ones.
    #[test]
    fn support_delta_rank_is_monotone_in_bits() {
        for seed in [0x1u64, 0xBEEF, crate::bucket::sum::DEFAULT_HASH_SEED] {
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

    /// The `q = 64 -> q = 65` flip the Phase-1 fact sheet found, pinned at its
    /// root: it is not the near-empty second word, it is that `Gf2Hash::new`
    /// draws `2W` words per row, so `W = 1` and `W = 2` get *unrelated* row
    /// bit patterns in word 0 and their delta-span ranks are independent
    /// draws. At the default seed, the su4 probe's support `(0, 1)` happens to
    /// be rank-deficient at `W = 1` and full-rank at `W = 2`, for every bucket
    /// count the engine's own policy reaches.
    #[test]
    fn support_delta_rank_differs_across_the_word_boundary_at_the_default_seed() {
        let seed = crate::bucket::sum::DEFAULT_HASH_SEED;
        for bits in 7..=9u8 {
            let w1 = Gf2Hash::<1>::new(64, bits, seed);
            let w2 = Gf2Hash::<2>::new(65, bits, seed);
            assert_eq!(
                crate::test_support::support_delta_rank(&w1, &[0, 1]),
                3,
                "W=1/q=64 at {bits} bits: expected the deficient rank the fact sheet measured"
            );
            assert_eq!(
                crate::test_support::support_delta_rank(&w2, &[0, 1]),
                4,
                "W=2/q=65 at {bits} bits: expected full rank"
            );
        }
        // The masking is not what does it: at `W = 2` only 130 of 256 columns
        // are live at q = 65, and the rank is full anyway.
        let w2_wide = Gf2Hash::<2>::new(128, 7, seed);
        assert_eq!(
            crate::test_support::support_delta_rank(&w2_wide, &[0, 1]),
            4
        );
    }

    /// **The mechanism.** A support delta cannot reorder a bucket's key column
    /// exactly when `h` separates the support's delta space.
    ///
    /// Why it matters: the engine's per-run "rest" stream is a concatenation of
    /// blocks `{v ⊕ d : v ∈ bucket}`, one per non-identity delta `d`, and
    /// `merge::sort_rows_with_scratch` is a *stable* sort chosen for its run
    /// adaptivity. At full delta rank each bucket holds at most one of the
    /// `2^(2k)` local variants of any off-support pattern, so no two of its
    /// keys differ only inside the support, so XOR-by-`d` preserves the
    /// column's order and every block arrives already ascending. One rank short
    /// and each bucket holds *two* such variants — adjacent in key order,
    /// differing only in the support — and half the deltas invert every such
    /// adjacent pair, shattering the block into runs of ~2.
    #[test]
    fn support_delta_preserves_bucket_order_iff_the_delta_span_is_full_rank() {
        assert!(!order_broken_by_some_delta::<2>(128, 7, &[0, 1]));
        assert!(order_broken_by_some_delta::<1>(64, 7, &[0, 1]));
    }

    /// Partition a *closed* key set under `h`, then check every non-identity
    /// support delta against every bucket's (ascending) key column. Returns
    /// `true` if any delta reorders any bucket.
    ///
    /// The key set has to be the closed one — every off-support pattern paired
    /// with all `2^(2k)` local patterns — because that is the fixed point a
    /// repeated dense-PTM layer drives the sum to, and it is precisely the
    /// structure that puts local variants of one pattern in the same bucket
    /// when the rank is short. Keys drawn uniformly at random over the whole
    /// space would essentially never contain such a pair and the effect would
    /// be invisible.
    fn order_broken_by_some_delta<const W: usize>(
        num_qubits: usize,
        bits: u8,
        support: &[u32],
    ) -> bool {
        let h = Gf2Hash::<W>::new(num_qubits, bits, crate::bucket::sum::DEFAULT_HASH_SEED);
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
                let (w, bit) = ((q / 64) as usize, 1u64 << (q % 64));
                rest.x[w] &= !bit;
                rest.z[w] &= !bit;
            }
            for combo in 0..(1usize << gens.len()) {
                let p = xor(&rest, &local(combo));
                buckets[h.bucket_of_pauli(&p) as usize].push((p.x, p.z));
            }
        }
        for cols in buckets.iter_mut() {
            cols.sort_unstable();
            cols.dedup();
            for combo in 1..(1usize << gens.len()) {
                let d = local(combo);
                let translated: Vec<([u64; W], [u64; W])> = cols
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
}

#[cfg(test)]
mod props {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Linearity over arbitrary keys — the load-bearing algebraic property.
        /// Everything about bucket prediction follows from it.
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
    }
}
