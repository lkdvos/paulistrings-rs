//! The per-term GF(2)-linear fingerprint `g(v) = G·v` and its host oracle for `kernels/fingerprint.cu`.
//!
//! `g` only orders records inside a device bucket; identity is always decided on the full key.

/// Mixed into the hash seed before drawing the fingerprint rows, so `G` is unrelated to the `Gf2Hash` and `PartitionRows` rows of the same seed.
pub(crate) const FINGERPRINT_SALT: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// Rows of `G`, one bit of `g` each.
pub(crate) const FP_ROWS: usize = 64;

/// The 64 rows of `G` as `(x-mask, z-mask)` pairs.
#[derive(Clone, Debug)]
pub(crate) struct FingerprintRows<const W: usize> {
    rows_x: Vec<[u64; W]>,
    rows_z: Vec<[u64; W]>,
}

impl<const W: usize> FingerprintRows<W> {
    /// Rows drawn by splitmix64 from `hash_seed ^ FINGERPRINT_SALT`.
    /// Not the crate's xorshift: its consecutive outputs are GF(2)-linear in one state, so `rows_z = M·rows_x` and low-weight keys collide.
    pub(crate) fn new(hash_seed: u64) -> Self {
        let mut state = hash_seed ^ FINGERPRINT_SALT;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut rows_x = Vec::with_capacity(FP_ROWS);
        let mut rows_z = Vec::with_capacity(FP_ROWS);
        for _ in 0..FP_ROWS {
            rows_x.push(std::array::from_fn::<u64, W, _>(|_| next()));
            rows_z.push(std::array::from_fn::<u64, W, _>(|_| next()));
        }
        Self { rows_x, rows_z }
    }

    /// The unmasked 64-bit `g(x, z)`; the device applies its compile-time `FP_BITS` mask on top.
    pub(crate) fn fingerprint(&self, x: &[u64; W], z: &[u64; W]) -> u64 {
        let mut out = 0u64;
        for r in 0..FP_ROWS {
            let mut acc = 0u64;
            for w in 0..W {
                acc ^= (x[w] & self.rows_x[r][w]) ^ (z[w] & self.rows_z[r][w]);
            }
            out |= u64::from(acc.count_ones() & 1) << r;
        }
        out
    }

    /// The rows in the prelude's device layout.
    pub(crate) fn flat(&self) -> Vec<u64> {
        let mut v = Vec::with_capacity(FP_ROWS * 2 * W);
        for r in 0..FP_ROWS {
            v.extend_from_slice(&self.rows_x[r]);
            v.extend_from_slice(&self.rows_z[r]);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bucket::sum::DEFAULT_HASH_SEED;
    use crate::test_support::Xs64;

    fn rand_key<const W: usize>(rng: &mut Xs64) -> ([u64; W], [u64; W]) {
        (rng.next_array::<W>(), rng.next_array::<W>())
    }

    fn check_linear<const W: usize>(seed: u64) {
        let fp = FingerprintRows::<W>::new(seed);
        let mut rng = Xs64::new(seed ^ 0x5EED);
        for _ in 0..1000 {
            let (ax, az) = rand_key::<W>(&mut rng);
            let (bx, bz) = rand_key::<W>(&mut rng);
            let cx: [u64; W] = std::array::from_fn(|w| ax[w] ^ bx[w]);
            let cz: [u64; W] = std::array::from_fn(|w| az[w] ^ bz[w]);
            assert_eq!(
                fp.fingerprint(&cx, &cz),
                fp.fingerprint(&ax, &az) ^ fp.fingerprint(&bx, &bz),
                "W={W}"
            );
        }
    }

    #[test]
    fn fingerprint_is_gf2_linear_on_random_pairs() {
        check_linear::<1>(DEFAULT_HASH_SEED);
        check_linear::<2>(DEFAULT_HASH_SEED);
        check_linear::<2>(0xF00D);
    }

    #[test]
    fn a_single_bit_key_reads_one_column_of_g() {
        let fp = FingerprintRows::<2>::new(DEFAULT_HASH_SEED);
        assert_eq!(fp.fingerprint(&[0, 0], &[0, 0]), 0);
        for q in [0usize, 5, 63, 64, 100, 127] {
            let (w, b) = (q / 64, q % 64);
            let mut x = [0u64; 2];
            x[w] = 1 << b;
            let want_x: u64 = (0..FP_ROWS)
                .map(|r| ((fp.rows_x[r][w] >> b) & 1) << r)
                .sum();
            assert_eq!(fp.fingerprint(&x, &[0, 0]), want_x, "x on qubit {q}");
            let want_z: u64 = (0..FP_ROWS)
                .map(|r| ((fp.rows_z[r][w] >> b) & 1) << r)
                .sum();
            assert_eq!(fp.fingerprint(&[0, 0], &x), want_z, "z on qubit {q}");
        }
    }

    /// Every distinct key of weight at most two on 64 qubits gets a distinct 64-bit fingerprint.
    #[test]
    fn fingerprint_is_injective_on_weight_two_keys_at_64_qubits() {
        let single: Vec<([u64; 1], [u64; 1])> = (0..64)
            .flat_map(|q| {
                let b = 1u64 << q;
                [([b], [0]), ([0], [b]), ([b], [b])]
            })
            .collect();
        let mut keys: Vec<([u64; 1], [u64; 1])> = vec![([0], [0])];
        keys.extend(single.iter().copied());
        for (i, a) in single.iter().enumerate() {
            for b in &single[i + 1..] {
                if (a.0[0] | a.1[0]) & (b.0[0] | b.1[0]) == 0 {
                    keys.push(([a.0[0] | b.0[0]], [a.1[0] | b.1[0]]));
                }
            }
        }
        assert_eq!(keys.len(), 1 + 64 * 3 + 2016 * 9);
        for seed in [DEFAULT_HASH_SEED, 0, 1, 0xF00D, 0xC0FFEE] {
            let fp = FingerprintRows::<1>::new(seed);
            let mut g: Vec<u64> = keys.iter().map(|(x, z)| fp.fingerprint(x, z)).collect();
            g.sort_unstable();
            g.dedup();
            assert_eq!(g.len(), keys.len(), "seed {seed:#x}: fingerprint collision");
        }
    }
}
