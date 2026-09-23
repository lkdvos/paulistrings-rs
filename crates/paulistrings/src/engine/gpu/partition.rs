//! [`DevicePartition`], one partition of the layer loop held on a CUDA device. See ARCHITECTURE.md §Partitioning.

use super::error::GpuError;
use super::finalize::approx_top_n_device;
use super::layer::{apply_layer_device, GpuLayerCounters, GpuLayerOptions, LayerScratch};
use super::sum::GpuSum;
use super::truncation::{layer_pass_leaves, DevicePolicy};
use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::channel::prepared::Prepared;
use crate::engine::partitioned::backend::{PartitionBackend, PartitionStorage};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{Collectives, Transport};
use crate::engine::partitioned::truncation::PartitionedTruncation;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;
use crate::truncation::BuiltinTruncation;

/// A partition whose sum lives on a device.
///
/// The seam's methods cannot fail, so a device error is recorded in [`Self::error`] and every later layer is skipped; the driver surfaces it after the loop (ARCHITECTURE.md §GPU-Readiness).
/// `hash` is the driver's view of the bucket count: `refine` advances it alone, `apply_layer` brings the device up to it, and [`Self::take_error`] re-syncs it to the device.
/// `policy` is the lowered form of the run's policy, which the layer and the layer pass read instead of the generic `T`.
pub(crate) struct DevicePartition<const W: usize> {
    sum: Option<GpuSum<W>>,
    scratch: Option<LayerScratch<W>>,
    hash: Gf2Hash<W>,
    pub(crate) policy: DevicePolicy,
    pub(crate) error: Option<GpuError>,
    #[cfg(feature = "phase-timing")]
    stats: PhaseStats,
}

impl<const W: usize> DevicePartition<W> {
    pub(crate) fn new(sum: GpuSum<W>, options: GpuLayerOptions) -> Result<Self, GpuError> {
        let scratch = LayerScratch::new(&sum, options)?;
        Ok(Self {
            hash: sum.hash().clone(),
            sum: Some(sum),
            scratch: Some(scratch),
            policy: DevicePolicy::keep_all(),
            error: None,
            #[cfg(feature = "phase-timing")]
            stats: PhaseStats::default(),
        })
    }

    pub(crate) fn sum(&self) -> &GpuSum<W> {
        self.sum.as_ref().expect("DevicePartition: detached")
    }

    pub(crate) fn scratch(&self) -> &LayerScratch<W> {
        self.scratch.as_ref().expect("DevicePartition: detached")
    }

    pub(crate) fn scratch_mut(&mut self) -> &mut LayerScratch<W> {
        self.scratch.as_mut().expect("DevicePartition: detached")
    }

    pub(crate) fn counters(&self) -> GpuLayerCounters {
        self.scratch().counters
    }

    /// The recorded error, if any, after re-syncing the hash mirror to the device sum.
    pub(crate) fn take_error(&mut self) -> Result<(), GpuError> {
        if let Some(sum) = self.sum.as_ref() {
            self.hash = sum.hash().clone();
        }
        match self.error.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn record(&mut self, r: Result<(), GpuError>) {
        if let Err(e) = r {
            if self.error.is_none() {
                self.error = Some(e);
            }
        }
    }
}

impl<const W: usize> PartitionStorage<W> for DevicePartition<W> {
    fn len(&self) -> usize {
        self.sum.as_ref().map_or(0, GpuSum::len)
    }

    fn hash(&self) -> &Gf2Hash<W> {
        &self.hash
    }

    fn refine(&mut self) {
        // Only the mirror moves; the layer refines the device to the mirror and the bucket policy in one pass.
        self.hash.refine();
    }

    fn detach(&mut self) -> Self {
        Self {
            sum: self.sum.take(),
            scratch: self.scratch.take(),
            hash: self.hash.clone(),
            policy: self.policy.clone(),
            error: self.error.take(),
            #[cfg(feature = "phase-timing")]
            stats: std::mem::take(&mut self.stats),
        }
    }

    #[cfg(feature = "phase-timing")]
    fn stats(&mut self) -> &mut PhaseStats {
        &mut self.stats
    }
}

impl<const W: usize, T> PartitionBackend<W, T> for DevicePartition<W>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    fn apply_layer<X: Transport>(
        &mut self,
        prep: &Prepared<W>,
        plan: &PartitionPlan,
        _rows: &PartitionRows<W>,
        _policy: &T,
        transport: &X,
    ) -> LayerExchangeCounts {
        assert!(
            !plan.has_remote(),
            "DevicePartition: a layer with remote deltas needs the device exchange, which this backend does not implement; run with one partition"
        );
        let size = transport.size();
        if self.error.is_some() {
            return LayerExchangeCounts::none(size);
        }
        let (Some(sum), Some(scratch)) = (self.sum.as_mut(), self.scratch.as_mut()) else {
            panic!("DevicePartition: apply_layer on a detached placeholder");
        };
        let r = apply_layer_device(sum, prep, &self.policy.keep, scratch, self.hash.bits());
        self.hash = sum.hash().clone();
        self.record(r);
        LayerExchangeCounts::none(size)
    }

    /// The lowered tree's layer pass in the host's order, so the collectives issued equal the host impl's for the same policy.
    fn finalize_layer(&mut self, _policy: &T, coll: &dyn Collectives) {
        let tree = self.policy.tree.clone();
        layer_pass_leaves(&tree, &mut |leaf| {
            let r = match leaf {
                BuiltinTruncation::ApproxTopN(n) => {
                    let healthy = self.error.is_none();
                    let parts = match (self.sum.as_mut(), self.scratch.as_mut()) {
                        (Some(sum), Some(scratch)) if healthy => Some((sum, scratch)),
                        _ => None,
                    };
                    approx_top_n_device(parts, *n, coll)
                }
                _ => Err(GpuError::Unsupported("exact TopN on device")),
            };
            self.record(r);
        });
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::test_support::rand_sum_real;
    use crate::truncation::BuiltinTruncation as T;

    /// A one-partition group that counts its `allreduce_sum_u64` calls.
    #[derive(Default)]
    struct CountingGroup(AtomicU32);

    impl Collectives for CountingGroup {
        fn rank(&self) -> u32 {
            0
        }
        fn size(&self) -> u32 {
            1
        }
        fn allreduce_max_u8(&self, v: u8) -> u8 {
            v
        }
        fn allreduce_sum_u64(&self, _buf: &mut [u64]) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        fn barrier(&self) {}
    }

    fn and(a: T, b: T) -> T {
        T::And(Box::new(a), Box::new(b))
    }

    /// A device layer pass issues as many reductions as the host's `PartitionedTruncation` does for the same tree, and keeps the same terms.
    #[test]
    fn a_layer_pass_issues_the_hosts_collectives() {
        crate::require_cuda!();
        let input = rand_sum_real::<1>(4000, 32, 0xC011);
        let cases = [
            (T::ApproxTopN(1000), 1),
            (and(T::Coeff(1e-3), T::ApproxTopN(1000)), 1),
            (and(T::ApproxTopN(2000), T::ApproxTopN(500)), 2),
            (
                T::Or(Box::new(T::ApproxTopN(10)), Box::new(T::Coeff(1e-3))),
                0,
            ),
        ];
        for (tree, want) in cases {
            let device = CountingGroup::default();
            let mut part = DevicePartition::new(
                GpuSum::from_host(&input, 0).expect("upload"),
                GpuLayerOptions::default(),
            )
            .expect("partition");
            part.policy = DevicePolicy::lower(tree.clone()).expect("lower");
            <DevicePartition<1> as PartitionBackend<1, T>>::finalize_layer(
                &mut part, &tree, &device,
            );
            part.take_error().expect("device layer pass");
            let host = CountingGroup::default();
            let mut want_sum = input.clone();
            <T as PartitionedTruncation<1>>::finalize_layer_partitioned(
                &tree,
                &mut want_sum,
                &host,
            );
            assert_eq!(device.0.load(Ordering::Relaxed), want, "{tree:?}: device");
            assert_eq!(host.0.load(Ordering::Relaxed), want, "{tree:?}: host");
            assert_eq!(part.len(), want_sum.len(), "{tree:?}: len");
            assert_eq!(
                part.sum().to_host().unwrap().to_arrays(),
                want_sum.to_arrays(),
                "{tree:?}: terms"
            );
        }
    }

    /// A partition that already failed still enters every reduction, so a group cannot fall out of step on one device's error.
    #[test]
    fn a_failed_partition_still_enters_the_reduction() {
        crate::require_cuda!();
        let input = rand_sum_real::<1>(500, 32, 0xFA11);
        let tree = and(T::ApproxTopN(100), T::ApproxTopN(50));
        let mut part = DevicePartition::new(
            GpuSum::from_host(&input, 0).expect("upload"),
            GpuLayerOptions::default(),
        )
        .expect("partition");
        part.policy = DevicePolicy::lower(tree.clone()).expect("lower");
        part.error = Some(GpuError::Unsupported("injected"));
        let group = CountingGroup::default();
        <DevicePartition<1> as PartitionBackend<1, T>>::finalize_layer(&mut part, &tree, &group);
        assert_eq!(group.0.load(Ordering::Relaxed), 2);
        assert!(matches!(
            part.take_error(),
            Err(GpuError::Unsupported("injected"))
        ));
        assert_eq!(
            part.len(),
            input.len(),
            "a failed partition keeps its terms"
        );
    }
}
