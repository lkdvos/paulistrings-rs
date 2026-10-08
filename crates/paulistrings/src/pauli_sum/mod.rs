//! [`PauliSum<W>`] — weighted sum of Pauli strings in structure-of-arrays form. See ARCHITECTURE.md §Data-Model.
//!
//! # Canonical order
//!
//! Terms are ordered by bucket index `h(x, z)` ascending, then lexicographic `(x, z)` key within a bucket; [`PauliSum::iter`] and [`PauliSum::to_arrays`] produce exactly this order, and no two entries share a key.
//! A single-bucket sum's order is plain lexicographic `(x, z)`; sums of at most [`DEFAULT_TARGET_BUCKET_LEN`] terms built through [`BuildAccumulator`] are single-bucket, so small sums come out lex-sorted. Larger sums interleave buckets in an `H`-dependent order — compare by key ([`PauliSum::get`], [`PauliSum::iter`]), not by position.
//!
//! Build a [`PauliSum`] from unsorted inputs via [`BuildAccumulator`]; once built, combine sums with [`PauliSum::add`] or scale with [`PauliSum::scale`].
//!
//! # Examples
//!
//! Construct the observable `Z₀ + 0.5·X₁` on two qubits via
//! [`BuildAccumulator`], then merge in a second sum.
//!
//! ```
//! use paulistrings::{BuildAccumulator, PauliString, PauliSum, Phase};
//! use num_complex::Complex64;
//!
//! let mut acc = BuildAccumulator::<1>::new(2);
//! acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
//! acc.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(0.5, 0.0));
//! let a = acc.finalize();
//! assert_eq!(a.len(), 2);
//!
//! let mut acc2 = BuildAccumulator::<1>::new(2);
//! acc2.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(-0.25, 0.0));
//! let b = acc2.finalize();
//!
//! let merged = a.add(&b);
//! assert_eq!(merged.len(), 2); // Z₀ + 0.25·X₁
//! ```
//!
//! # Bucketing
//!
//! GF(2)-linear bucket partitioning of a Pauli sum. See ARCHITECTURE.md §Bucketing.
//!
//! The propagation engine does not maintain one global sorted order. Instead
//! the sum is partitioned by a GF(2)-linear hash `h(v) = H·v` of the Pauli key.
//! Two properties make that partition useful, and both follow from linearity:
//!
//! * A channel maps an input key to `v ⊕ d` for `d` in a small **delta set**, so
//!   `h(v ⊕ d) = h(v) ⊕ h(d)` — output buckets are predictable from input
//!   buckets, and because `⊕` is an involution the relation inverts: each
//!   *output* bucket gathers from a statically-known handful of input buckets
//!   (1, 2, 4 or 16 for the built-in channels).
//! * `h` is a function, so equal keys always land in the same bucket.
//!   Deduplication is therefore bucket-local, and **there is no global sort**.
//!
//! [`hash`] defines the hash `Gf2Hash<W>` itself — the linear map, its delta
//! set, and the bit-count policy (`desired_bits`) that ties bucket count to
//! term count. [`storage`] holds the bucketed storage: `PauliSum<W>`'s
//! column layout and the `refine`/`coarsen`/`rebucket` operations that keep
//! the bucket count matched to the sum as it grows or shrinks. There is one
//! `PauliSum` type, not a separate flat and bucketed form — a sum small
//! enough to live in a single bucket is, by construction, plain lex-sorted.
//!
//! [`BuildAccumulator`]: crate::BuildAccumulator

pub mod accumulator;
pub mod hash;
pub mod storage;

pub use hash::{Gf2Hash, PartitionRows, B_MAX_BITS, P_MAX_BITS};
pub use storage::{
    desired_bits, PauliSum, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
    MIN_TERMS_PER_TASK,
};

#[cfg(test)]
use num_complex::Complex64;

#[cfg(test)]
use crate::pauli_string::PauliString;

#[cfg(test)]
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
