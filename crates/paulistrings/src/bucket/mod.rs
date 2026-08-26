//! GF(2)-linear bucket partitioning of a Pauli sum. See
//! `research/plans/2026-08-26-v0.2-gf2-bucketing.md` (v0.2 §2-§4), which
//! supersedes v0.1 scope §5 and §9.
//!
//! The propagation engine no longer maintains one global sorted order. Instead
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

pub mod hash;
pub mod sum;

pub use hash::{Gf2Hash, B_MAX_BITS};
pub use sum::{BucketedSum, DEFAULT_TARGET_BUCKET_LEN, MIN_TERMS_PER_TASK};
