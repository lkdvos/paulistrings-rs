//! [`DevicePartition`], one partition of the layer loop held on a CUDA device. See ARCHITECTURE.md §Partitioning.

use super::error::GpuError;
use super::finalize::{approx_top_n_device, top_n_device};
use super::layer::{
    apply_layer_device, gpu_desired_bits, prepared_fanout, GpuLayerCounters, GpuLayerOptions,
    LayerScratch,
};
use super::sum::GpuSum;
use super::truncation::{layer_pass_leaves, KeepProgram};
use crate::bucket::hash::{Gf2Hash, PartitionRows, B_MAX_BITS};
use crate::bucket::sum::desired_bits;
use crate::channel::prepared::Prepared;
use crate::engine::partitioned::backend::{PartitionBackend, PartitionStorage};
use crate::engine::partitioned::layer::LayerExchangeCounts;
use crate::engine::partitioned::plan::PartitionPlan;
use crate::engine::partitioned::transport::{Collectives, Transport};
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;
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
        if let Some(wire) = self.scratch.as_ref().and_then(|s| s.export.wire.as_ref()) {
            wire.abort();
        }
    }

    pub(crate) fn check_poison(&self) -> Result<(), GpuError> {
        match self.poison {
            Some((rank, layer)) => Err(GpuError::Poisoned { rank, layer }),
            None => Ok(()),
        }
    }

    fn record(&mut self, r: Result<(), GpuError>) {
        if let Err(e) = r {
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
        prep: &Prepared<W>,
        target_bucket_len: usize,
        min_buckets: usize,
    ) -> u8 {
        let host = desired_bits(self.len(), target_bucket_len, min_buckets).max(self.hash.bits());
        let Some(scratch) = self.scratch.as_ref() else {
            return host;
        };
        // Between agreements a group member cannot refine, so its proposal plans for twice the load a lone device would.
        let policy = match scratch.options.bucket_policy {
            p if self.group_size == 1 => p,
            super::layer::GpuBucketPolicy::RecordsPerBlock(t) => {
                super::layer::GpuBucketPolicy::RecordsPerBlock((t / 2).max(1))
            }
            super::layer::GpuBucketPolicy::TermsPerBucket(t) => {
                super::layer::GpuBucketPolicy::TermsPerBucket((t / 2).max(1))
            }
        };
        let device = gpu_desired_bits(self.len(), prepared_fanout(prep), policy, self.hash.bits())
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
        prep: &Prepared<W>,
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
        let (t0, before) = (std::time::Instant::now(), scratch.kernel_ms);
        let r = apply_layer_device(
            sum,
            prep,
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
            t0.elapsed().as_nanos() as u64,
        );
        self.hash = sum.hash().clone();
        match r {
            Ok(counts) => counts,
            Err(e) => {
                self.record(Err(e));
                LayerExchangeCounts::none(size)
            }
        }
    }

    /// The lowered tree's layer pass in the host's order, so the collectives issued equal the host impl's for the same policy.
    ///
    /// Exact `TopN` runs only alone on the device (`group_size == 1`): the `n`-th largest magnitude of a group's sum has no collective form (`PartitionedTruncation`'s docs), so a group member reports `Unsupported` rather than issue a wrong local selection.
    fn finalize_layer(&mut self, policy: &BuiltinTruncation, coll: &dyn Collectives) {
        let single = self.group_size == 1;
        layer_pass_leaves(policy, &mut |leaf| {
            let r = match leaf {
                BuiltinTruncation::ApproxTopN(n) => {
                    let healthy = self.error.is_none();
                    let parts = match (self.sum.as_mut(), self.scratch.as_mut()) {
                        (Some(sum), Some(scratch)) if healthy => Some((sum, scratch)),
                        _ => None,
                    };
                    approx_top_n_device(parts, *n, coll)
                }
                BuiltinTruncation::TopN(n) if single => {
                    match (self.sum.as_mut(), self.scratch.as_mut()) {
                        (Some(sum), Some(scratch)) if self.error.is_none() => {
                            top_n_device(sum, scratch, *n)
                        }
                        _ => Ok(()),
                    }
                }
                _ => Err(GpuError::Unsupported("exact TopN on device")),
            };
            self.record(r);
        });
        // The driver's `finalize_ns` lap is the wall of this pass; the events only need reading so the kernel counters stay complete.
        if let (Some(sum), Some(scratch)) = (self.sum.as_ref(), self.scratch.as_mut()) {
            let r = scratch.resolve(sum);
            self.record(r);
        }
    }
}

/// One device layer's counters into the partition's [`PhaseStats`]: kernel families onto the host phases they replace, the driving thread's wall minus the refine, rescale, export and exchange as the coset loop.
/// `sort_ns` stays zero: the sort is inside the fused layer, so it is part of `merge_ns` on a device row.
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
    let c = scratch.counters;
    if c.permuted {
        // One scatter over every input row, nothing sorted.
        stats.cosets += 1;
        stats.runs += 1u64 << c.bits;
        stats.rows_gathered += c.records;
    } else if !c.rescaled {
        // Every pre-dedup record is gathered and sorted on the device; nothing takes the identity stream's shortcut.
        stats.cosets += u64::from(c.batches);
        stats.runs += 1u64 << c.bits;
        stats.rows_gathered += c.records;
        stats.rows_sorted += c.records;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::engine::partitioned::truncation::PartitionedTruncation;
    use crate::test_support::rand_sum_real;
    use crate::truncation::BuiltinTruncation as T;
    use crate::TruncationPolicy;

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
            part.keep = KeepProgram::lower(&tree).expect("lower");
            PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &device);
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

    /// Exact `TopN` on a lone device partition (`group_size == 1`) issues no collective at all, unlike `ApproxTopN`, and keeps the host's terms exactly — alone and composed with `And`/`Or`.
    #[test]
    fn a_lone_partition_runs_exact_topn_with_no_collective() {
        crate::require_cuda!();
        use crate::truncation::TopN;
        let input = rand_sum_real::<1>(4000, 32, 0xC012);
        let cases = [
            T::TopN(1000),
            and(T::Coeff(1e-3), T::TopN(1000)),
            and(T::TopN(2000), T::Weight(20)),
            T::Or(Box::new(T::TopN(10)), Box::new(T::Coeff(1e-3))),
        ];
        for tree in cases {
            let device = CountingGroup::default();
            let mut part = DevicePartition::new(
                GpuSum::from_host(&input, 0).expect("upload"),
                GpuLayerOptions::default(),
            )
            .expect("partition");
            part.keep = KeepProgram::lower(&tree).expect("lower");
            PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &device);
            part.take_error().expect("device layer pass");
            assert_eq!(
                device.0.load(Ordering::Relaxed),
                0,
                "{tree:?}: exact TopN has no collective form"
            );
            let mut want_sum = input.clone();
            <T as TruncationPolicy<1>>::finalize_layer(&tree, &mut want_sum);
            assert_eq!(part.len(), want_sum.len(), "{tree:?}: len");
            assert_eq!(
                part.sum().to_host().unwrap().to_arrays(),
                want_sum.to_arrays(),
                "{tree:?}: terms"
            );
        }
        // Sanity: `TopN` alone actually truncates against this input.
        let mut sanity = input.clone();
        TopN(1000).finalize_layer(&mut sanity);
        assert_eq!(sanity.len(), 1000);
    }

    /// A group member (`group_size > 1`) reports `Unsupported` on an exact `TopN` rather than run a wrong local selection.
    #[test]
    fn a_group_member_rejects_exact_topn() {
        crate::require_cuda!();
        let input = rand_sum_real::<1>(500, 32, 0xC013);
        let tree = T::TopN(100);
        let mut part = DevicePartition::new(
            GpuSum::from_host(&input, 0).expect("upload"),
            GpuLayerOptions::default(),
        )
        .expect("partition");
        part.group_size = 2;
        part.keep = KeepProgram::lower(&tree).expect("lower");
        let group = CountingGroup::default();
        PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &group);
        assert!(matches!(part.take_error(), Err(GpuError::Unsupported(_))));
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
        part.keep = KeepProgram::lower(&tree).expect("lower");
        part.error = Some(GpuError::Unsupported("injected"));
        let group = CountingGroup::default();
        PartitionBackend::<1, T>::finalize_layer(&mut part, &tree, &group);
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
