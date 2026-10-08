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
mod partition;
pub mod storage;

pub use hash::{Gf2Hash, PartitionRows, B_MAX_BITS, P_MAX_BITS};
pub use storage::{
    desired_bits, PauliSum, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
    MIN_TERMS_PER_TASK,
};

#[cfg(test)]
mod tests;
