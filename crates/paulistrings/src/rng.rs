//! Key-addressed pseudo-random streams for the sampling truncation policies: splitmix64 seeding plus xoshiro256++.
//! A stream is a pure function of its key words, so a draw never depends on which thread ran it or in what order.

/// The splitmix64 increment, `2^64 / φ`.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// One splitmix64 step: advance `state` and return its mixed output.
#[inline]
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A xoshiro256++ generator.
#[derive(Clone, Debug)]
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// The stream named by `key`, typically `[seed, call, rank, ...]`.
    /// Every word, and the key length, is absorbed through a splitmix64 output, so keys differing in any word or in length name unrelated streams.
    pub(crate) fn from_key(key: &[u64]) -> Self {
        let mut absorbed = key.len() as u64;
        for &word in key {
            let mut t = absorbed ^ word;
            absorbed = splitmix64(&mut t);
        }
        let mut s = [0u64; 4];
        for word in s.iter_mut() {
            *word = splitmix64(&mut absorbed);
        }
        // xoshiro's one forbidden state; four consecutive splitmix outputs are never all zero in practice.
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    #[inline]
    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let out = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        out
    }

    /// A uniform `f64` in `[0, 1)` on the 2^-53 grid.
    #[inline]
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first output of splitmix64 from state 0 is the published reference value.
    #[test]
    fn splitmix64_matches_the_reference_value() {
        let mut state = 0u64;
        assert_eq!(splitmix64(&mut state), 0xE220_A839_7B1D_CDAF);
    }

    /// From state `[1, 2, 3, 4]`, worked by hand: `rotl(1 + 4, 23) + 1 = 41943041`, and after one update the state is `[7, 0, 262146, 6·2^45]`, giving `rotl(7 + 6·2^45, 23) + 7 = 58720359`.
    #[test]
    fn xoshiro256pp_matches_hand_computed_outputs() {
        let mut rng = Rng { s: [1, 2, 3, 4] };
        assert_eq!(rng.next_u64(), 41_943_041);
        assert_eq!(rng.next_u64(), 58_720_359);
    }

    #[test]
    fn a_stream_is_a_function_of_its_key() {
        let draw = |key: &[u64]| {
            let mut rng = Rng::from_key(key);
            [rng.next_u64(), rng.next_u64()]
        };
        assert_eq!(draw(&[7, 1, 2]), draw(&[7, 1, 2]));
        assert_ne!(draw(&[7, 1, 2]), draw(&[7, 1, 3]));
        assert_ne!(draw(&[7, 1, 2]), draw(&[7, 2, 1]));
        assert_ne!(draw(&[7, 1]), draw(&[7, 1, 0]));
    }

    /// Mean `1/2` and variance `1/12` within 5σ of their sampling error over 200k draws, and every draw in `[0, 1)`.
    #[test]
    fn uniform_has_the_first_two_moments_of_u01() {
        let n = 200_000;
        let mut rng = Rng::from_key(&[0xABCD]);
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for _ in 0..n {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
            sum += u;
            sum_sq += u * u;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        let mean_sigma = (1.0f64 / 12.0 / n as f64).sqrt();
        assert!((mean - 0.5).abs() < 5.0 * mean_sigma, "mean {mean}");
        // Var of (U - 1/2)² is 1/180, so the variance estimate's σ is sqrt(1/180/n).
        let var_sigma = (1.0f64 / 180.0 / n as f64).sqrt();
        assert!((var - 1.0 / 12.0).abs() < 5.0 * var_sigma, "var {var}");
    }
}
