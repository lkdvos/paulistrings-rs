//! Scatter and gather of a [`PauliSum`] across the partitions of the partitioned engine. See ARCHITECTURE.md §Partitioning.

use num_complex::Complex64;
use rayon::prelude::*;

use super::hash::{Gf2Hash, PartitionRows};
use super::storage::{merge_two, BucketCols, PauliSum, DEFAULT_MIN_BUCKETS, MIN_TERMS_PER_TASK};

/// Copy the terms of one bucket whose partition rank is `rank`.
/// Shared by `PauliSum::filter_partition`'s serial and parallel branches. A subsequence of a strictly ascending run is strictly ascending, so the output needs no sort.
/// Plain pushes, no `reserve`: the surviving count is not known without a second pass, and a speculative reservation would inflate peak RSS across all `P` parts for no measured gain (see `research/FINDINGS.md`).
fn filter_bucket<const W: usize>(
    cols: &BucketCols<W>,
    rows: &PartitionRows<W>,
    rank: u32,
) -> BucketCols<W> {
    let mut out = BucketCols::<W>::new();
    for i in 0..cols.len() {
        if rows.partition_of(&cols.x[i], &cols.z[i]) == rank {
            out.push(cols.x[i], cols.z[i], cols.coeff[i]);
        }
    }
    out
}

/// Merge `k` disjoint sorted runs into one, in `ceil(log2 k)` pairwise rounds — the gather half of the partition scatter/gather.
/// The same tree as [`merge_runs`], but serial and clone-free: the caller is already parallel over buckets and `k = P <= 16`, so a nested Rayon split would be pure overhead. Runs come from distinct partitions of one sum, so no key can appear twice; debug builds check that by re-verifying the strict ascent of the result.
fn merge_disjoint_runs<const W: usize>(mut runs: Vec<BucketCols<W>>) -> BucketCols<W> {
    while runs.len() > 1 {
        let mut next: Vec<BucketCols<W>> = Vec::with_capacity(runs.len().div_ceil(2));
        let mut it = runs.into_iter();
        while let Some(a) = it.next() {
            match it.next() {
                Some(b) => next.push(merge_two(&a, &b)),
                None => next.push(a),
            }
        }
        runs = next;
    }
    let out = runs.pop().unwrap_or_default();
    #[cfg(debug_assertions)]
    {
        for i in 1..out.len() {
            debug_assert!(
                (&out.x[i - 1], &out.z[i - 1]) < (&out.x[i], &out.z[i]),
                "PauliSum::merge_partitions: duplicate key across partitions",
            );
        }
    }
    out
}

impl<const W: usize> PauliSum<W> {
    /// Coarsen until the bucket count is `1 << bits`; a no-op if it is already at or below that.
    /// The bulk form of [`Self::coarsen`], used to bring the partitions of a partitioned run onto one agreed bucket count before they are gathered (ARCHITECTURE.md §Partitioning).
    ///
    /// # Panics
    ///
    /// Panics if `bits` exceeds the current bucket bits — growing is [`Self::refine`]'s job.
    pub(crate) fn coarsen_to(&mut self, bits: u8) {
        assert!(
            bits <= self.hash.bits(),
            "PauliSum::coarsen_to: {bits} exceeds the current {} bucket bits (use refine to grow)",
            self.hash.bits(),
        );
        while self.hash.bits() > bits {
            self.coarsen();
        }
    }

    /// The terms `rows` assigns to partition `rank`, as a sum under the same hash (cloned) and bucket count.
    /// The scatter half of the partition scatter/gather (ARCHITECTURE.md §Partitioning). Each output bucket is the subsequence of the input bucket that survives the rank test, so it inherits both the sort and the uniqueness — one `part(v)` evaluation per term, no sort, no coefficient arithmetic.
    /// Same threshold and rationale as [`Self::refine`].
    pub(crate) fn filter_partition(&self, rows: &PartitionRows<W>, rank: u32) -> Self {
        debug_assert_eq!(
            self.num_qubits,
            rows.num_qubits(),
            "PauliSum::filter_partition: num_qubits mismatch",
        );
        debug_assert!(
            (rank as usize) < rows.num_partitions(),
            "PauliSum::filter_partition: rank {rank} out of range",
        );
        let buckets: Vec<BucketCols<W>> = if self.len < DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK {
            self.buckets
                .iter()
                .map(|cols| filter_bucket(cols, rows, rank))
                .collect()
        } else {
            self.buckets
                .par_iter()
                .map(|cols| filter_bucket(cols, rows, rank))
                .collect()
        };
        let len = buckets.iter().map(|c| c.len()).sum();
        Self {
            buckets,
            hash: self.hash.clone(),
            num_qubits: self.num_qubits,
            len,
        }
    }

    /// Gather partition-disjoint sums that share a partition back into one.
    /// The inverse of [`Self::filter_partition`] (ARCHITECTURE.md §Partitioning). Bucket `b` of the result is a `P`-way merge of bucket `b` of each input — sorted runs in, one sorted run out, no coefficient arithmetic, so a `filter_partition`/`merge_partitions` round trip is bitwise. Debug builds assert the key sets are disjoint.
    /// Same threshold and rationale as [`Self::refine`]. A single input is returned unchanged.
    ///
    /// # Panics
    ///
    /// Panics if `parts` is empty, or if the inputs disagree on their hash rows or bucket count — align them with [`Self::coarsen_to`] or [`Self::align_to`] first.
    pub(crate) fn merge_partitions(parts: Vec<Self>) -> Self {
        let first = parts
            .first()
            .expect("PauliSum::merge_partitions: no inputs to merge");
        let hash = first.hash.clone();
        let num_qubits = first.num_qubits;
        for p in parts.iter().skip(1) {
            assert!(
                p.hash.same_rows_as(&hash),
                "PauliSum::merge_partitions: hash mismatch (seed or num_qubits differs)",
            );
            assert_eq!(
                p.hash.bits(),
                hash.bits(),
                "PauliSum::merge_partitions: bucket count mismatch",
            );
        }

        let nb = hash.num_buckets();
        let k = parts.len();
        let len: usize = parts.iter().map(|p| p.len).sum();

        // Transpose parts×buckets into buckets×parts, moving the columns so the merge below never copies a run it does not have to.
        let mut runs: Vec<Vec<BucketCols<W>>> = (0..nb).map(|_| Vec::with_capacity(k)).collect();
        for p in parts {
            for (b, cols) in p.buckets.into_iter().enumerate() {
                runs[b].push(cols);
            }
        }

        let buckets: Vec<BucketCols<W>> = if len < DEFAULT_MIN_BUCKETS * MIN_TERMS_PER_TASK {
            runs.into_iter().map(merge_disjoint_runs).collect()
        } else {
            runs.into_par_iter().map(merge_disjoint_runs).collect()
        };

        Self {
            buckets,
            hash,
            num_qubits,
            len,
        }
    }

    /// Rebuild a sum from the columns [`Self::to_arrays`] produced, cut back into buckets by `lens`.
    /// The receive side of the distributed gather (`engine::partitioned`): a rank ships its bucket lengths and concatenated columns, and the root reassembles the rank's `PauliSum` before handing it to [`Self::merge_partitions`]. Nothing is sorted or deduplicated — the columns already satisfy `PauliSum`'s invariants bucket by bucket.
    ///
    /// # Panics
    ///
    /// If `lens` is not one entry per bucket of `hash`, if the columns are not parallel, or if the lengths do not sum to the column length.
    pub(crate) fn from_bucket_columns(
        lens: &[usize],
        x: Vec<[u64; W]>,
        z: Vec<[u64; W]>,
        coeff: Vec<Complex64>,
        hash: Gf2Hash<W>,
        num_qubits: usize,
    ) -> Self {
        assert_eq!(
            lens.len(),
            hash.num_buckets(),
            "PauliSum::from_bucket_columns: {} bucket lengths for a hash with {} buckets",
            lens.len(),
            hash.num_buckets(),
        );
        assert!(
            x.len() == coeff.len() && z.len() == coeff.len(),
            "PauliSum::from_bucket_columns: columns are not parallel ({}, {}, {})",
            x.len(),
            z.len(),
            coeff.len(),
        );
        let len: usize = lens.iter().sum();
        assert_eq!(
            len,
            coeff.len(),
            "PauliSum::from_bucket_columns: bucket lengths sum to {len}, columns hold {}",
            coeff.len(),
        );

        let mut x = x.into_iter();
        let mut z = z.into_iter();
        let mut coeff = coeff.into_iter();
        let buckets = lens
            .iter()
            .map(|&n| BucketCols {
                x: x.by_ref().take(n).collect(),
                z: z.by_ref().take(n).collect(),
                coeff: coeff.by_ref().take(n).collect(),
            })
            .collect();

        Self {
            buckets,
            hash,
            num_qubits,
            len,
        }
    }

    /// Wrap per-bucket columns that already satisfy the invariant, one entry per bucket of `hash`.
    /// The device download builds its buckets in parallel, which the single flat stream of [`Self::from_bucket_columns`] cannot.
    ///
    /// # Panics
    ///
    /// If `buckets` is not one entry per bucket of `hash`.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_buckets(
        buckets: Vec<BucketCols<W>>,
        hash: Gf2Hash<W>,
        num_qubits: usize,
    ) -> Self {
        assert_eq!(
            buckets.len(),
            hash.num_buckets(),
            "PauliSum::from_buckets: {} buckets for a hash with {} buckets",
            buckets.len(),
            hash.num_buckets(),
        );
        let len = buckets.iter().map(BucketCols::len).sum();
        Self {
            buckets,
            hash,
            num_qubits,
            len,
        }
    }

    /// `Some(r)` if every term lies in partition `r`, `None` on an empty or a mixed sum.
    /// Debug and test predicate for "this sum is one partition's share"; it scans every term, so it is not for the propagation loop.
    pub(crate) fn partition_rank_of_all(&self, rows: &PartitionRows<W>) -> Option<u32> {
        let mut seen: Option<u32> = None;
        for (x, z, _) in self.iter() {
            let r = rows.partition_of(x, z);
            match seen {
                None => seen = Some(r),
                Some(s) if s == r => {}
                Some(_) => return None,
            }
        }
        seen
    }
}
