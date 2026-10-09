//! Key-addressed pseudo-random streams for the sampling truncation policies: splitmix64 seeding plus xoshiro256++.
//! A stream is a pure function of its key words, so a draw never depends on which thread ran it or in what order.

/// The splitmix64 increment, the odd constant `⌊2^64/φ⌋`.
pub(crate) const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64's output finalizer: a bijection on `u64` with full avalanche.
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One splitmix64 step: advance `state` and return its mixed output.
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
    /// The stream named by `key`; keys differing in any word or in length name unrelated streams.
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
        // The all-zero state is xoshiro's one fixed point.
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

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
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests;
