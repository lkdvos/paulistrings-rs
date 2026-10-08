use super::PauliSum;
use num_complex::Complex64;

use crate::pauli_string::PauliString;

impl<const W: usize> PauliSum<W> {
    /// Test-only helper: build a `PauliSum<W>` from `(pauli_str, coeff)` pairs, index `i` of the string being qubit `i` (Hermitian convention, `Y` = no phase).
    /// `num_qubits` comes from the first string's length; routes through `BuildAccumulator`, so duplicate keys sum and exact zeros drop.
    pub(crate) fn from_strings(terms: &[(&str, Complex64)]) -> Self {
        use crate::phase::Phase;
        assert!(!terms.is_empty(), "from_strings requires at least one term");
        let num_qubits = terms[0].0.len();
        assert!(num_qubits <= 64 * W, "num_qubits must fit in W*64 bits");
        let mut acc = crate::pauli_sum::accumulator::BuildAccumulator::<W>::new(num_qubits);
        for (s, c) in terms {
            assert_eq!(
                s.len(),
                num_qubits,
                "all pauli strings must have the same length",
            );
            let mut x = [0u64; W];
            let mut z = [0u64; W];
            for (i, ch) in s.chars().enumerate() {
                let word = i / 64;
                let bit = 1u64 << (i % 64);
                match ch {
                    'I' => {}
                    'X' => x[word] |= bit,
                    'Z' => z[word] |= bit,
                    'Y' => {
                        x[word] |= bit;
                        z[word] |= bit;
                    }
                    other => panic!("unexpected Pauli char {:?} (expected I/X/Y/Z)", other),
                }
            }
            let p = PauliString::<W> { x, z };
            acc.add_term(p, Phase::ONE, *c);
        }
        acc.finalize()
    }
}
