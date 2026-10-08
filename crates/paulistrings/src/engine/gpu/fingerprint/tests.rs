use super::*;
use crate::pauli_sum::storage::DEFAULT_HASH_SEED;
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
