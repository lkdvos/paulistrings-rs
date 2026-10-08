//! [`DevicePartition`], one partition of the layer loop held on a CUDA device. See ARCHITECTURE.md §Partitioning.

use super::error::GpuError;
use super::finalize::{approx_top_n_device, top_n_device};
use super::layer::{
    apply_layer_device, gpu_desired_bits, prepared_fanout, GpuLayerCounters, GpuLayerOptions,
    LayerScratch,
};
use super::sum::GpuSum;
use super::truncation::{layer_pass_leaves, KeepProgram};
use crate::channel::prepared::Prepared;
use crate::engine::partitioned::backend::{PartitionBackend, PartitionStorage};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{Collectives, Transport};
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows, B_MAX_BITS};
use crate::pauli_sum::storage::desired_bits;
use crate::truncation::BuiltinTruncation;

/// A partition whose sum lives on a device.
///
/// The seam's methods cannot fail, so a device error is recorded in [`Self::error`] and every later layer is skipped; the driver surfaces it after the loop (ARCHITECTURE.md §GPU-Readiness).
/// `hash` is the driver's view of the bucket count: `refine` advances it alone, `apply_layer` brings the device up to it, and [`Self::take_error`] re-syncs it to the device.
/// `keep` is the per-term half of the run's policy, lowered once per call.
/// Nominally `pub`, like `HostPartition`, so it can be the drivers' device backend; the module is crate-private.
pub struct DevicePartition<const W: usize> {
    sum: Option<GpuSum<W>>,
    scratch: Option<LayerScratch<W>>,
    hash: Gf2Hash<W>,
    pub(crate) keep: KeepProgram,
    pub(crate) error: Option<GpuError>,
    /// `(rank, layer)` of the group's first failure; set on every partition at once, and refuses every later call.
    pub(crate) poison: Option<(usize, usize)>,
    /// Partitions in the group this partition runs in; above one, the layer never refines off-schedule and the proposal carries growth headroom.
    pub(crate) group_size: u32,
    /// Layers `apply_layer` was asked for since construction.
    layers_applied: usize,
    /// The layer index (in `layers_applied` terms) at which the first error was recorded.
    pub(crate) failed_layer: Option<usize>,
    /// Test hook: fail before the exchange on this layer index.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fail_at_layer: Option<usize>,
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
            keep: KeepProgram::KEEP,
            error: None,
            poison: None,
            group_size: 1,
            layers_applied: 0,
            failed_layer: None,
            #[cfg(any(test, feature = "test-utils"))]
            fail_at_layer: None,
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

    /// The recorded error and its group code, `0` if healthy and the failed layer plus one otherwise, for [`first_failure`](super::first_failure).
    pub(crate) fn take_failure(&mut self) -> (Result<(), GpuError>, u64) {
        let layer = self.failed_layer.take();
        let own = self.take_error();
        let code = if own.is_err() {
            layer.unwrap_or(0) as u64 + 1
        } else {
            0
        };
        (own, code)
    }

    /// Refuse every later call after the group's first failure, `(rank, layer)`, dropping the wire now rather than at a finalize against failed peers.
    pub(crate) fn poison(&mut self, rank: usize, layer: usize) {
        self.poison = Some((rank, layer));
        if let Some(wire) = self
            .scratch
            .as_ref()
            .and_then(|scratch| scratch.export.wire.as_ref())
        {
            wire.abort();
        }
    }

    pub(crate) fn check_poison(&self) -> Result<(), GpuError> {
        match self.poison {
            Some((rank, layer)) => Err(GpuError::Poisoned { rank, layer }),
            None => Ok(()),
        }
    }

    fn record(&mut self, result: Result<(), GpuError>) {
        if let Err(e) = result {
            if self.error.is_none() {
                self.error = Some(e);
                self.failed_layer = Some(self.layers_applied.saturating_sub(1));
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

    /// The host formula raised to the device bucket policy's wish, so a remote layer, which runs at the agreed count and cannot refine, still gets blocks the fused kernel fits.
    fn proposed_bits(
        &self,
        prepared: &Prepared<W>,
        target_bucket_len: usize,
        min_buckets: usize,
    ) -> u8 {
        let host = desired_bits(self.len(), target_bucket_len, min_buckets).max(self.hash.bits());
        let Some(scratch) = self.scratch.as_ref() else {
            return host;
        };
        // Between agreements a group member cannot refine, so its proposal plans for twice the load a lone device would.
        let policy = match scratch.options.bucket_policy {
            policy if self.group_size == 1 => policy,
            super::layer::GpuBucketPolicy::RecordsPerBlock(target) => {
                super::layer::GpuBucketPolicy::RecordsPerBlock((target / 2).max(1))
            }
            super::layer::GpuBucketPolicy::TermsPerBucket(target) => {
                super::layer::GpuBucketPolicy::TermsPerBucket((target / 2).max(1))
            }
        };
        let device = gpu_desired_bits(
            self.len(),
            prepared_fanout(prepared),
            policy,
            self.hash.bits(),
        )
        .min(scratch.options.max_bits.min(B_MAX_BITS));
        host.max(device)
    }

    fn detach(&mut self) -> Self {
        Self {
            sum: self.sum.take(),
            scratch: self.scratch.take(),
            hash: self.hash.clone(),
            keep: self.keep,
            error: self.error.take(),
            poison: self.poison,
            group_size: self.group_size,
            layers_applied: self.layers_applied,
            failed_layer: self.failed_layer.take(),
            #[cfg(any(test, feature = "test-utils"))]
            fail_at_layer: self.fail_at_layer.take(),
            #[cfg(feature = "phase-timing")]
            stats: std::mem::take(&mut self.stats),
        }
    }

    #[cfg(feature = "phase-timing")]
    fn stats(&mut self) -> &mut PhaseStats {
        &mut self.stats
    }
}

impl<const W: usize> PartitionBackend<W, BuiltinTruncation> for DevicePartition<W> {
    fn apply_layer<X: Transport>(
        &mut self,
        prepared: &Prepared<W>,
        plan: &PartitionPlan,
        _rows: &PartitionRows<W>,
        _policy: &BuiltinTruncation,
        transport: &X,
    ) -> LayerExchangeCounts {
        let size = transport.size();
        self.layers_applied += 1;
        #[cfg(any(test, feature = "test-utils"))]
        if self.fail_at_layer == Some(self.layers_applied - 1) {
            self.record(Err(GpuError::Unsupported("injected before the exchange")));
        }
        let bits = self.hash.bits();
        let (Some(sum), Some(scratch)) = (self.sum.as_mut(), self.scratch.as_mut()) else {
            panic!("DevicePartition: apply_layer on a detached placeholder");
        };
        if self.error.is_some() {
            // A failed partition still pairs its partners' exchange with empty blocks, or the group hangs on it.
            if plan.has_remote() {
                super::export::pair_empty_exchange::<W, X>(
                    transport,
                    plan,
                    bits,
                    &mut scratch.export,
                );
            }
            return LayerExchangeCounts::none(size);
        }
        #[cfg(feature = "phase-timing")]
        let (started, before) = (std::time::Instant::now(), scratch.kernel_ms);
        let result = apply_layer_device(
            sum,
            prepared,
            plan,
            &self.keep,
            scratch,
            self.hash.bits(),
            transport,
        );
        #[cfg(feature = "phase-timing")]
        fold_layer_stats(
            &mut self.stats,
            scratch,
            before,
            started.elapsed().as_nanos() as u64,
        );
        self.hash = sum.hash().clone();
        match result {
            Ok(counts) => counts,
            Err(e) => {
                self.record(Err(e));
                LayerExchangeCounts::none(size)
            }
        }
    }

    /// The lowered tree's layer pass in the host's order, so the collectives issued equal the host's; a group member (`group_size > 1`) reports exact `TopN` `Unsupported`, and every member reports `CollapseSample` so.
    fn finalize_layer(&mut self, policy: &BuiltinTruncation, collectives: &dyn Collectives) {
        let single = self.group_size == 1;
        layer_pass_leaves(policy, &mut |leaf| {
            let result = match leaf {
                BuiltinTruncation::ApproxTopN(n) => {
                    let healthy = self.error.is_none();
                    let parts = match (self.sum.as_mut(), self.scratch.as_mut()) {
                        (Some(sum), Some(scratch)) if healthy => Some((sum, scratch)),
                        _ => None,
                    };
                    approx_top_n_device(parts, *n, collectives)
                }
                BuiltinTruncation::TopN(n) if single => {
                    match (self.sum.as_mut(), self.scratch.as_mut()) {
                        (Some(sum), Some(scratch)) if self.error.is_none() => {
                            top_n_device(sum, scratch, *n)
                        }
                        _ => Ok(()),
                    }
                }
                BuiltinTruncation::CollapseSample(_) => {
                    Err(GpuError::Unsupported("CollapseSample on device"))
                }
                _ => Err(GpuError::Unsupported("exact TopN on device")),
            };
            self.record(result);
        });
        // The driver's `finalize_ns` lap is the wall of this pass; the events only need reading so the kernel counters stay complete.
        if let (Some(sum), Some(scratch)) = (self.sum.as_ref(), self.scratch.as_mut()) {
            let result = scratch.resolve(sum);
            self.record(result);
        }
    }
}

/// One device layer's counters into the partition's [`PhaseStats`], mapped as [`GpuPartitionedSum::take_stats`](super::GpuPartitionedSum::take_stats) documents.
#[cfg(feature = "phase-timing")]
fn fold_layer_stats<const W: usize>(
    stats: &mut PhaseStats,
    scratch: &mut LayerScratch<W>,
    before: super::layer::GpuKernelMs,
    wall_ns: u64,
) {
    let after = scratch.kernel_ms;
    let ns = |ms: f64| (ms.max(0.0) * 1e6) as u64;
    let refine = ns(after.refine - before.refine);
    let rescale = ns(after.rescale - before.rescale);
    let laps = std::mem::take(&mut scratch.laps);
    stats.rebucket_ns += refine;
    stats.rescale_ns += rescale;
    stats.export_ns += laps.export_ns;
    stats.exchange_ns += laps.exchange_ns;
    stats.rows_exported += laps.rows_exported;
    stats.recv_rows += laps.recv_rows;
    stats.coset_loop_ns +=
        wall_ns.saturating_sub(refine + rescale + laps.export_ns + laps.exchange_ns);
    stats.gather_ns += ns(after.count - before.count) + ns(after.sizes - before.sizes);
    stats.merge_ns += ns(after.layer - before.layer) + ns(after.permute - before.permute);
    stats.compact_ns += ns(after.compact - before.compact);
    let [h2d, d2h] = std::mem::take(&mut scratch.xfer_ns);
    stats.h2d_ns += h2d;
    stats.d2h_ns += d2h;
    let counters = scratch.counters;
    if counters.permuted {
        stats.cosets += 1;
        stats.runs += 1u64 << counters.bits;
        stats.rows_gathered += counters.records;
    } else if !counters.rescaled {
        // Every pre-dedup record is gathered and sorted on the device; nothing takes the identity stream's shortcut.
        stats.cosets += u64::from(counters.batches);
        stats.runs += 1u64 << counters.bits;
        stats.rows_gathered += counters.records;
        stats.rows_sorted += counters.records;
    }
}

#[cfg(test)]
mod tests;
