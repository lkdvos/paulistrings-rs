//! Key-addressed pseudo-random streams for the sampling truncation policies: splitmix64 seeding plus xoshiro256++.
//! A stream is a pure function of its key words, so a draw never depends on which thread ran it or in what order.

/// The splitmix64 increment, the odd constant `⌊2^64/φ⌋`.
pub(crate) const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64's output finalizer: a bijection on `u64` with full avalanche.
#[inline]
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One splitmix64 step: advance `state` and return its mixed output.
#[inline]
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_GAMMA);
    mix64(*state)
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

    /// Every draw in `[0, 1)`, with mean `1/2` within 5σ over 100k draws.
    #[test]
    fn uniform_is_in_the_unit_interval_with_mean_one_half() {
        let n = 100_000;
        let mut rng = Rng::from_key(&[0xABCD]);
        let draws: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        assert!(draws.iter().all(|u| (0.0..1.0).contains(u)));
        let mean = draws.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 0.5).abs() < 5.0 * (1.0 / 12.0 / n as f64).sqrt(),
            "mean {mean}"
        );
    }
}
