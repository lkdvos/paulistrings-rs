//! Scatter and gather of a [`PauliSum`] across the partitions of the partitioned engine (ARCHITECTURE.md §Partitioning).

use num_complex::Complex64;
use rayon::prelude::*;

use super::hash::{Gf2Hash, PartitionRows};
use super::storage::{merge_two, BucketCols, PauliSum, DEFAULT_MIN_BUCKETS, MIN_TERMS_PER_TASK};

/// Copy the terms of one bucket whose partition rank is `rank`.
// Not reserved up front: research/FINDINGS.md §Reserving a safe upper bound in the merge.
fn filter_bucket<const W: usize>(
    columns: &BucketCols<W>,
    rows: &PartitionRows<W>,
    rank: u32,
) -> BucketCols<W> {
    let mut out = BucketCols::<W>::new();
    for i in 0..columns.len() {
        if rows.partition_of(&columns.x[i], &columns.z[i]) == rank {
            out.push(columns.x[i], columns.z[i], columns.coeff[i]);
        }
    }
    out
}

/// Merge key-disjoint sorted runs into one by serial pairwise rounds; the caller is already parallel over buckets.
fn merge_disjoint_runs<const W: usize>(mut runs: Vec<BucketCols<W>>) -> BucketCols<W> {
    while runs.len() > 1 {
        let mut next: Vec<BucketCols<W>> = Vec::with_capacity(runs.len().div_ceil(2));
        let mut remaining = runs.into_iter();
        while let Some(a) = remaining.next() {
            match remaining.next() {
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
    /// Coarsen until the bucket count is `1 << bits`; panics if that would grow it.
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

    /// The terms `rows` assigns to partition `rank`, under the same hash and bucket count.
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
                .map(|columns| filter_bucket(columns, rows, rank))
                .collect()
        } else {
            self.buckets
                .par_iter()
                .map(|columns| filter_bucket(columns, rows, rank))
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

    /// Gather key-disjoint sums sharing hash rows and bucket count back into one; panics on an empty or misaligned input.
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

        let num_buckets = hash.num_buckets();
        let num_parts = parts.len();
        let len: usize = parts.iter().map(|p| p.len).sum();

        let mut runs: Vec<Vec<BucketCols<W>>> = (0..num_buckets)
            .map(|_| Vec::with_capacity(num_parts))
            .collect();
        for p in parts {
            for (b, columns) in p.buckets.into_iter().enumerate() {
                runs[b].push(columns);
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

    /// Rebuild a sum from the columns [`Self::to_arrays`] produced, cut back into buckets by `lens`; nothing is sorted.
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
