//! The bucket-by-bucket structural diff of a sum before and after one layer.

use num_complex::Complex64;
use rayon::prelude::*;

use crate::pauli_sum::PauliSum;

/// What one layer changed in a sum's storage, from [`structural_diff`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StructuralDiff {
    /// Terms before.
    pub terms: usize,
    /// Terms after.
    pub terms_after: usize,
    /// Keys in both sums whose coefficients differ bitwise.
    pub changed: usize,
    /// Keys only after.
    pub inserted: usize,
    /// Keys only before.
    pub deleted: usize,
    /// Strings after, summed over buckets, at or past the bucket's first index where the two key sequences differ: what a backward in-place merge would move.
    pub shift: usize,
}

impl std::ops::Add for StructuralDiff {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            terms: self.terms + other.terms,
            terms_after: self.terms_after + other.terms_after,
            changed: self.changed + other.changed,
            inserted: self.inserted + other.inserted,
            deleted: self.deleted + other.deleted,
            shift: self.shift + other.shift,
        }
    }
}

/// [`StructuralDiff`] of `after` against `before`, bucket by bucket on their sorted keys.
///
/// # Panics
///
/// Unless both sums share one bucket hash.
pub fn structural_diff<const W: usize>(
    before: &PauliSum<W>,
    after: &PauliSum<W>,
) -> StructuralDiff {
    let (hash, hash_after) = (before.hash(), after.hash());
    assert!(
        hash.bits() == hash_after.bits() && hash.rows() == hash_after.rows(),
        "structural_diff: the sums must share one bucket hash, got {} and {} bits",
        hash.bits(),
        hash_after.bits(),
    );
    (0..before.num_buckets())
        .into_par_iter()
        .map(|b| bucket_diff(before.bucket(b), after.bucket(b)))
        .reduce(StructuralDiff::default, |a, b| a + b)
}

type Columns<'a, const W: usize> = (&'a [[u64; W]], &'a [[u64; W]], &'a [Complex64]);

fn bucket_diff<const W: usize>(before: Columns<'_, W>, after: Columns<'_, W>) -> StructuralDiff {
    let (n, m) = (before.0.len(), after.0.len());
    let mut diff = StructuralDiff { terms: n, terms_after: m, ..StructuralDiff::default() };
    let (mut i, mut j) = (0, 0);
    // The index into `after` of the first insert or delete, if any.
    let mut first_difference = None;
    while i < n && j < m {
        match (before.0[i], before.1[i]).cmp(&(after.0[j], after.1[j])) {
            std::cmp::Ordering::Equal => {
                diff.changed += usize::from(before.2[i] != after.2[j]);
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                first_difference.get_or_insert(j);
                diff.deleted += 1;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                first_difference.get_or_insert(j);
                diff.inserted += 1;
                j += 1;
            }
        }
    }
    if i < n || j < m {
        first_difference.get_or_insert(j);
    }
    diff.deleted += n - i;
    diff.inserted += m - j;
    diff.shift = first_difference.map_or(0, |first| m - first);
    diff
}
