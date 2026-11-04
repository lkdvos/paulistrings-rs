//! Linear algebra on a [`PartitionedSum`] held in host memory, the steps of a Lanczos recursion besides the operator products; no call communicates, since equal rows put a key in the same partition of both sums.

use super::PartitionedSum;
use crate::coefficient::Coefficient;
use crate::engine::backend::HostPartition;
use std::marker::PhantomData;

#[cfg(feature = "phase-timing")]
use super::super::driver::DriverLaps;

impl<const W: usize, C: Coefficient> Clone for PartitionedSum<W, C> {
    /// Partition by partition; the copy records no trace.
    fn clone(&self) -> Self {
        Self {
            parts: self.parts.iter().map(|part| HostPartition::new(part.sum.clone())).collect(),
            coefficient: PhantomData,
            rows: self.rows.clone(),
            runtime: self.runtime.clone(),
            trace: None,
            #[cfg(feature = "phase-timing")]
            laps: DriverLaps::new(0),
        }
    }
}

impl<const W: usize, C: Coefficient> PartitionedSum<W, C> {
    /// Panics unless `other` is split by the same rows.
    fn assert_aligned(&self, other: &Self, what: &str) {
        assert_eq!(
            self.rows, other.rows,
            "PartitionedSum::{what}: the sums are split by different rows"
        );
    }

    /// Multiply every coefficient by `c`, in place.
    pub fn scale(&mut self, c: C) {
        for part in &mut self.parts {
            part.sum.scale(c);
        }
    }

    /// `self += c · other`, in place, partition by partition.
    ///
    /// # Panics
    ///
    /// If the two sums are split by different rows.
    pub fn add_assign_scaled(&mut self, other: &Self, c: C) {
        self.assert_aligned(other, "add_assign_scaled");
        for (part, theirs) in self.parts.iter_mut().zip(&other.parts) {
            let mut scaled = theirs.sum.clone();
            scaled.scale(c);
            part.sum = part.sum.add(&scaled);
        }
    }

    /// The Hilbert-Schmidt overlap `tr(self† · other) / 2ⁿ`, the partitions' own values added in rank order.
    ///
    /// # Panics
    ///
    /// As [`add_assign_scaled`](Self::add_assign_scaled).
    pub fn overlap(&self, other: &Self) -> C {
        self.assert_aligned(other, "overlap");
        self.parts.iter().zip(&other.parts).map(|(a, b)| a.sum.overlap(&b.sum)).sum()
    }
}
