//! [`DevicePartition`], one partition of the layer loop held on a CUDA device.

use super::collapse::collapse_sample_device;
use super::error::GpuError;
use super::finalize::{approx_top_n_device, top_n_device};
use super::layer::{
    apply_layer_device, gpu_desired_bits, permutation_run_device, prepared_fanout,
    rotation_run_device, DeviceLayer, GpuLayerOptions, LayerScratch,
};
use super::sum::{GpuSum, GPU_MAX_BITS};
use super::truncation::{layer_pass_leaves, KeepProgram};
use crate::channel::prepared::Prepared;
use crate::coefficient::Coefficient;
use crate::collectives::Collectives;
use crate::engine::backend::{PartitionBackend, PartitionStorage};
use crate::engine::bucketed::{RowEmitter, Run, MAX_ROTATION_RUN_LAYERS};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::plan::PartitionPlan;
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;
use crate::engine::timing::timed;
use crate::engine::trace::{DeviceLayerRecord, PartitionLayerRow};
use crate::engine::transport::{SoloTransport, Transport};
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::storage::desired_bits;
use crate::truncation::BuiltinTruncation;
use num_complex::Complex64;

/// Faults a test arms on one device partition, each firing once (`test_support::set_fault_plan`).
#[cfg(any(test, feature = "test-utils"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FaultPlan {
    /// Fail before the exchange of this layer, counted over every layer the partition has applied since its scatter.
    pub at_layer: Option<usize>,
    /// The next remote layer's receive growth fails as out of memory, after the skeletons and before the vote.
    pub receive_growth: bool,
    /// The next chunked receive fails as out of memory once this chunk has moved, after the vote and mid-layer.
    pub after_chunk: Option<usize>,
    /// The partition's device wire fails in its next post or wait.
    pub wire: Option<super::wire::WireFault>,
}

/// A partition whose sum lives on a device.
///
/// The seam's methods cannot fail, so a device error is recorded in [`Self::error`] and every later layer is skipped; the driver returns it after the loop, agreed over the group.
/// A lone partition keeps the last completed layer's output and resumes from it; a group of more than one is poisoned until the caller scatters again.
/// `hash` is the driver's view of the bucket count: `refine` advances it alone, `apply_layer` brings the device up to it, and [`Self::take_error`] re-syncs it to the device.
/// `keep` is the per-term half of the run's policy, lowered once per call.
// `pub` only because it is the backend parameter of the `pub` device driver aliases, like `HostPartition`.
#[doc(hidden)]
pub struct DevicePartition<const W: usize, C: Coefficient = Complex64> {
    sum: Option<GpuSum<W, C>>,
    scratch: Option<LayerScratch<W, C>>,
    hash: Gf2Hash<W>,
    pub(crate) keep: KeepProgram,
    error: Option<GpuError>,
    /// `(rank, layer)` of the group's first failure; set on every partition at once, and refuses every later call.
    poison: Option<(usize, usize)>,
    /// Partitions in the group this partition runs in; above one, the layer never refines off-schedule and the proposal carries growth headroom.
    pub(crate) group_size: u32,
    /// Layers `apply_layer` was asked for since construction.
    layers_applied: usize,
    /// The layer index (in `layers_applied` terms) at which the first error was recorded.
    failed_layer: Option<usize>,
    /// Test hook: fail before the exchange on this layer index.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fail_at_layer: Option<usize>,
}

impl<const W: usize, C: Coefficient> DevicePartition<W, C> {
    pub(crate) fn new(sum: GpuSum<W, C>, options: GpuLayerOptions) -> Result<Self, GpuError> {
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
        })
    }

    pub(crate) fn sum(&self) -> &GpuSum<W, C> {
        self.sum.as_ref().expect("DevicePartition: detached")
    }

    pub(crate) fn scratch_mut(&mut self) -> &mut LayerScratch<W, C> {
        self.scratch.as_mut().expect("DevicePartition: detached")
    }

    /// Arm `plan`; a wire fault needs the partition's wire, which a one-partition split has none of.
    #[cfg(any(test, feature = "test-utils"))]
    pub(crate) fn set_fault_plan(&mut self, plan: FaultPlan) {
        self.fail_at_layer = plan.at_layer;
        let export = &mut self.scratch_mut().export;
        export.fail_receive_growth = plan.receive_growth;
        export.fail_after_chunk = plan.after_chunk;
        if let Some(fault) = plan.wire {
            export
                .wire
                .as_ref()
                .expect("a wire fault on a partition without a device wire")
                .inject(fault);
        }
    }

    /// Re-sync the hash mirror to the device sum.
    pub(crate) fn resync_hash(&mut self) {
        if let Some(sum) = self.sum.as_ref() {
            self.hash = sum.hash().clone();
        }
    }

    /// The recorded error, if any, after re-syncing the hash mirror to the device sum.
    pub(crate) fn take_error(&mut self) -> Result<(), GpuError> {
        self.resync_hash();
        match self.error.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The recorded error and its group code, `0` if healthy and the failed layer plus one otherwise, for [`first_failure`](super::first_failure).
    pub(crate) fn take_failure(&mut self) -> (Result<(), GpuError>, u64) {
        let layer = self.failed_layer.take();
        let own = self.take_error();
        let code = if own.is_err() { layer.unwrap_or(0) as u64 + 1 } else { 0 };
        (own, code)
    }

    /// Refuse every later call after the group's first failure, `(rank, layer)`, dropping the wire now rather than at a finalize against failed peers.
    pub(crate) fn poison(&mut self, rank: usize, layer: usize) {
        self.poison = Some((rank, layer));
        if let Some(wire) = self.scratch.as_ref().and_then(|scratch| scratch.export.wire.as_ref()) {
            wire.abort();
        }
    }

    pub(crate) fn check_poison(&self) -> Result<(), GpuError> {
        match self.poison {
            Some((rank, layer)) => Err(GpuError::Poisoned { rank, layer }),
            None => Ok(()),
        }
    }

    pub(crate) fn record(&mut self, result: Result<(), GpuError>) {
        if let Err(e) = result {
            if self.error.is_none() {
                self.error = Some(e);
                self.failed_layer = Some(self.layers_applied.saturating_sub(1));
            }
        }
    }

    /// The sum and its scratch, for a test driving a device pass directly.
    #[cfg(test)]
    pub(crate) fn parts_mut(&mut self) -> (&mut GpuSum<W, C>, &mut LayerScratch<W, C>) {
        self.healthy_parts().expect("DevicePartition: failed or detached")
    }

    /// The sum and its scratch, unless the partition has failed.
    fn healthy_parts(&mut self) -> Option<(&mut GpuSum<W, C>, &mut LayerScratch<W, C>)> {
        match (self.sum.as_mut(), self.scratch.as_mut()) {
            (Some(sum), Some(scratch)) if self.error.is_none() => Some((sum, scratch)),
            _ => None,
        }
    }

    /// `tree`'s layer pass over `collectives`; a group above one partition reports exact `TopN` `Unsupported`, and a `PerPartition` runs as a group of one.
    fn layer_pass(&mut self, tree: &BuiltinTruncation, collectives: &dyn Collectives) {
        layer_pass_leaves(tree, &mut |leaf| {
            let result = match leaf {
                BuiltinTruncation::ApproxTopN(n) => {
                    approx_top_n_device(self.healthy_parts(), *n, collectives)
                }
                BuiltinTruncation::TopN(n) if collectives.size() == 1 => self
                    .healthy_parts()
                    .map_or(Ok(()), |(sum, scratch)| top_n_device(sum, scratch, *n)),
                BuiltinTruncation::PerPartition(inner) => {
                    self.layer_pass(inner, &SoloTransport);
                    Ok(())
                }
                BuiltinTruncation::CollapseSample(policy) => {
                    collapse_sample_device(self.healthy_parts(), policy, collectives)
                }
                _ => Err(GpuError::Unsupported("exact TopN on device")),
            };
            self.record(result);
        });
    }
}

impl<const W: usize, C: Coefficient> PartitionStorage<W, C> for DevicePartition<W, C> {
    /// An upper bound on the terms mid-call, as [`GpuSum::len`], so the driver's run admission stays safe; exact once the call settles.
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

    fn coarsen_to(&mut self, bits: u8) {
        unreachable!("a device partition proposes at least its own {} bucket bits, so its group never agrees {bits}", self.hash.bits());
    }

    /// The grow-only host formula raised to the device bucket policy's wish, since the device has no coarsen pass, so a remote layer, which runs at the agreed count and cannot refine, still gets blocks the fused kernel fits.
    /// Both read [`GpuSum::len_estimate`] rather than the bound, so a bucket count lags its sum by a layer instead of overshooting it.
    fn proposed_bits(
        &self,
        prepared: &Prepared<W, C>,
        target_bucket_len: usize,
        min_buckets: usize,
    ) -> u8 {
        let estimate = self.sum.as_ref().map_or(0, GpuSum::len_estimate);
        let host = desired_bits(estimate, target_bucket_len, min_buckets).max(self.hash.bits());
        let Some(scratch) = self.scratch.as_ref() else {
            return host;
        };
        // Between agreements a group member cannot refine, so its proposal plans for twice the load a lone device would.
        let records_per_block = match scratch.options.records_per_block {
            records if self.group_size == 1 => records,
            records => (records / 2).max(1),
        };
        let device = gpu_desired_bits(
            estimate,
            prepared_fanout(prepared),
            records_per_block,
            self.hash.bits(),
        )
        .min(scratch.options.max_bits)
        .min(self.hash.max_bits().min(GPU_MAX_BITS));
        host.max(device)
    }

    /// Carry the device to the mirror, which the call's closing agreement may have refined with no layer after it to do so, and make the length exact.
    fn settle(&mut self) {
        let bits = self.hash.bits();
        if let Some((sum, _)) = self.healthy_parts() {
            let result = match sum.hash.bits() < bits {
                true => sum.refine_to(bits),
                false => Ok(()),
            };
            let result = result.and_then(|()| sum.settle_lengths());
            self.record(result);
        }
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
        }
    }

    #[cfg(feature = "phase-timing")]
    fn stats(&mut self) -> &mut PhaseStats {
        &mut self.scratch_mut().stats
    }
}

impl<const W: usize, C: Coefficient> PartitionBackend<W, C, BuiltinTruncation>
    for DevicePartition<W, C>
{
    fn apply_layer<X: Transport>(
        &mut self,
        prepared: &Prepared<W, C>,
        plan: &PartitionPlan,
        _rows: &PartitionRows<W>,
        policy: &BuiltinTruncation,
        transport: &X,
        _clocked: bool,
        _ahead: Option<&RowEmitter<'_, W, C>>,
    ) -> LayerExchangeCounts {
        let size = transport.size();
        self.layers_applied += 1;
        if let Some(scratch) = self.scratch.as_mut() {
            scratch.record = DeviceLayerRecord::default();
            scratch.refine_passes = 0;
            scratch.receive_chunks = 0;
            scratch.ceiling = input_ceiling::<W, C>(policy);
        }
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
                super::export::pair_empty_exchange::<W, C, X>(
                    transport,
                    plan,
                    bits,
                    &mut scratch.export,
                );
            }
            return failed_counts(plan, size);
        }
        timed!(let (started, before) = (std::time::Instant::now(), scratch.stats));
        let layer = DeviceLayer { prepared, plan, keep: &self.keep, target_bits: self.hash.bits() };
        let result = apply_layer_device(sum, &layer, scratch, transport);
        timed!(fold_layer_stats(
            scratch,
            &before,
            started.elapsed().as_nanos() as u64,
            sum.hash().bits(),
        ));
        match result {
            Ok(counts) => {
                self.hash = sum.hash().clone();
                counts
            }
            Err(e) => {
                // The mirror keeps the agreed bucket count, so the group's trace rows still agree; `take_error` re-syncs it.
                self.record(Err(e));
                failed_counts(plan, size)
            }
        }
    }

    /// Permutation runs whenever the scatter path is on.
    fn takes_runs(&self) -> bool {
        self.scratch.as_ref().is_some_and(|scratch| scratch.options.clifford)
    }

    /// A lone partition's rotation runs, of any span, since the device holds no coset in cache.
    fn rotation_run_dimension(&self) -> Option<usize> {
        (self.group_size == 1).then_some(MAX_ROTATION_RUN_LAYERS)
    }

    fn apply_run(&mut self, run: Run<'_, W, C>, policy: &BuiltinTruncation) {
        let layers = match run {
            Run::Rotations(layers) => layers.len(),
            Run::Permutations(steps) => steps.len(),
        };
        let first = self.layers_applied;
        self.layers_applied += layers;
        if let Some(scratch) = self.scratch.as_mut() {
            scratch.record = DeviceLayerRecord::default();
            scratch.refine_passes = 0;
            scratch.receive_chunks = 0;
            scratch.ceiling = input_ceiling::<W, C>(policy);
        }
        // An injected failure names its own layer of the run, as the one-by-one path would.
        #[cfg(any(test, feature = "test-utils"))]
        if let Some(layer) =
            self.fail_at_layer.filter(|layer| (first..first + layers).contains(layer))
        {
            self.layers_applied = layer + 1;
            self.record(Err(GpuError::Unsupported("injected before the exchange")));
            self.layers_applied = first + layers;
        }
        let _ = first;
        let (keep, target_bits) = (self.keep, self.hash.bits());
        let Some((sum, scratch)) = self.healthy_parts() else {
            return;
        };
        timed!(let (started, before) = (std::time::Instant::now(), scratch.stats));
        let result = match run {
            Run::Permutations(steps) => {
                permutation_run_device(sum, steps, &keep, target_bits, scratch)
            }
            Run::Rotations(layers) => rotation_run_device(sum, layers, &keep, target_bits, scratch),
        };
        timed!(fold_layer_stats(
            scratch,
            &before,
            started.elapsed().as_nanos() as u64,
            sum.hash().bits(),
        ));
        match result {
            Ok(()) => self.resync_hash(),
            Err(e) => self.record(Err(e)),
        }
    }

    fn trace_layer(&self, row: &mut PartitionLayerRow) {
        if let Some(scratch) = self.scratch.as_ref() {
            row.refine_passes = scratch.refine_passes;
            row.receive_chunks = scratch.receive_chunks;
            row.device = Some(scratch.record);
        }
    }

    /// The lowered tree's layer pass in the host's order, so the collectives issued equal the host's.
    fn finalize_layer(&mut self, policy: &BuiltinTruncation, collectives: &dyn Collectives) {
        self.layer_pass(policy, collectives);
        // The driver's `finalize_ns` lap is the wall of this pass; the events only need reading so the kernel counters stay complete.
        if let (Some(sum), Some(scratch)) = (self.sum.as_ref(), self.scratch.as_mut()) {
            let result = scratch.resolve(sum);
            self.record(result);
        }
    }
}

/// The policy's ceiling on a layer's input, its run length limit, which every layer of a run stays within; `None` when it sets none.
fn input_ceiling<const W: usize, C: Coefficient>(policy: &BuiltinTruncation) -> Option<usize> {
    let limit = <BuiltinTruncation as crate::TruncationPolicy<W, C>>::run_length_limit(policy);
    (limit != 0 && limit != usize::MAX).then_some(limit)
}

/// A failed or skipped layer's counts: nothing moved, but the remote-delta count the group agrees on.
fn failed_counts(plan: &PartitionPlan, size: u32) -> LayerExchangeCounts {
    LayerExchangeCounts { remote_deltas: plan.remote.len(), ..LayerExchangeCounts::none(size) }
}

/// The rest of one device layer's [`PhaseStats`], its kernel times and laps having landed directly: the driving thread's wall over the fused path, and the host counters a device layer stands in for (`benchmarks/PROFILING.md` contract (a)).
#[cfg(feature = "phase-timing")]
fn fold_layer_stats<const W: usize, C: Coefficient>(
    scratch: &mut LayerScratch<W, C>,
    before: &PhaseStats,
    wall_ns: u64,
    bits: u8,
) {
    let record = scratch.record;
    let stats = &mut scratch.stats;
    let outside = (stats.rebucket_ns - before.rebucket_ns)
        + (stats.rescale_ns - before.rescale_ns)
        + (stats.export_ns - before.export_ns)
        + (stats.exchange_ns - before.exchange_ns);
    stats.coset_loop_ns += wall_ns.saturating_sub(outside);
    if record.permuted {
        stats.cosets += 1;
        stats.runs += 1u64 << bits;
        stats.rows_gathered += record.records;
    } else if !record.rescaled {
        // Every pre-dedup record is gathered and sorted on the device; nothing takes the identity stream's shortcut.
        stats.cosets += u64::from(record.batches);
        stats.runs += 1u64 << bits;
        stats.rows_gathered += record.records;
        stats.rows_sorted += record.records;
    }
}

#[cfg(test)]
mod tests;
