//! [`DevicePartition`], one partition of the layer loop held on a CUDA device. See ARCHITECTURE.md §Partitioning.

use super::error::GpuError;
use super::layer::{apply_layer_device, GpuLayerCounters, GpuLayerOptions, LayerScratch};
use super::sum::GpuSum;
use super::truncation::DeviceKeep;
use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::channel::prepared::Prepared;
use crate::engine::partitioned::backend::{PartitionBackend, PartitionStorage};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{Collectives, Transport};
use crate::engine::partitioned::truncation::PartitionedTruncation;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;

/// A partition whose sum lives on a device.
///
/// The seam's methods cannot fail, so a device error is recorded in [`Self::error`] and every later layer is skipped; the driver surfaces it after the loop (ARCHITECTURE.md §GPU-Readiness).
/// `hash` is the driver's view of the bucket count: `refine` advances it alone, `apply_layer` brings the device up to it, and [`Self::take_error`] re-syncs it to the device.
pub(crate) struct DevicePartition<const W: usize> {
    sum: Option<GpuSum<W>>,
    scratch: Option<LayerScratch<W>>,
    hash: Gf2Hash<W>,
    pub(crate) keep: DeviceKeep,
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
            keep: DeviceKeep::Keep,
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
            keep: self.keep,
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
        let keep = self.keep;
        let (Some(sum), Some(scratch)) = (self.sum.as_mut(), self.scratch.as_mut()) else {
            panic!("DevicePartition: apply_layer on a detached placeholder");
        };
        let r = apply_layer_device(sum, prep, keep, scratch, self.hash.bits());
        self.hash = sum.hash().clone();
        self.record(r);
        LayerExchangeCounts::none(size)
    }

    fn finalize_layer(&mut self, _policy: &T, _coll: &dyn Collectives) {
        unreachable!("DevicePartition: finalizing policies are rejected when the policy is lowered")
    }
}
