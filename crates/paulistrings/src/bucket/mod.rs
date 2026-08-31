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
//! term count. [`sum`] holds the bucketed storage: `PauliSum<W>`'s
//! column layout and the `refine`/`coarsen`/`rebucket` operations that keep
//! the bucket count matched to the sum as it grows or shrinks. There is one
//! `PauliSum` type, not a separate flat and bucketed form — a sum small
//! enough to live in a single bucket is, by construction, plain lex-sorted.

pub mod hash;
pub mod sum;

pub use hash::{Gf2Hash, B_MAX_BITS};
pub use sum::{
    desired_bits, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
    MIN_TERMS_PER_TASK,
};
