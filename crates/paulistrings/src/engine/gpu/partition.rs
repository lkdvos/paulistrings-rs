//! [`DevicePartition`], one partition of the layer loop held on a CUDA device. See ARCHITECTURE.md §Partitioning.

use super::error::GpuError;
use super::finalize::approx_top_n_device;
use super::layer::{
    apply_layer_device, gpu_desired_bits, prepared_fanout, GpuLayerCounters, GpuLayerOptions,
    LayerScratch,
};
use super::sum::GpuSum;
use super::truncation::{layer_pass_leaves, DevicePolicy};
use crate::bucket::hash::{Gf2Hash, PartitionRows, B_MAX_BITS};
use crate::bucket::sum::desired_bits;
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
            policy: DevicePolicy::keep_all(),
            error: None,
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
            policy: self.policy.clone(),
            error: self.error.take(),
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

impl<const W: usize, T> PartitionBackend<W, T> for DevicePartition<W>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    fn apply_layer<X: Transport>(
        &mut self,
        prep: &Prepared<W>,
        plan: &PartitionPlan,
        rows: &PartitionRows<W>,
        _policy: &T,
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
            rows,
            &self.policy.keep,
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
    stats.chunk_wait_ns += laps.chunk_wait_ns;
    stats.rows_exported += laps.rows_exported;
    stats.recv_rows += laps.recv_rows;
    stats.coset_loop_ns +=
        wall_ns.saturating_sub(refine + rescale + laps.export_ns + laps.exchange_ns);
    stats.gather_ns += ns(after.count - before.count) + ns(after.sizes - before.sizes);
    stats.merge_ns += ns(after.layer - before.layer);
    stats.compact_ns += ns(after.compact - before.compact);
    let [h2d, d2h] = std::mem::take(&mut scratch.xfer_ns);
    stats.h2d_ns += h2d;
    stats.d2h_ns += d2h;
    let c = scratch.counters;
    if !c.rescaled {
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

    /// How one mixed run is set up: the rows, an optional common bucket count replacing the scatter's, and a failure to inject on the device rank.
    struct Mixed<const W: usize> {
        rows: PartitionRows<W>,
        bits: Option<u8>,
        options: crate::PropagateOptions,
        device_options: GpuLayerOptions,
        fail_at: Option<usize>,
    }

    impl<const W: usize> Mixed<W> {
        fn seeded(nq: usize) -> Self {
            Self {
                rows: PartitionRows::<W>::from_seed(nq, 1, 0x3D1),
                bits: None,
                options: crate::PropagateOptions::default(),
                device_options: GpuLayerOptions::default(),
                fail_at: None,
            }
        }
    }

    /// Rank 0 on the host and rank 1 on the device over one in-process group; returns the host share and the device rank's outcome.
    fn mixed_run<const W: usize, P>(
        circuit: &crate::Circuit<W>,
        input: &crate::PauliSum<W>,
        policy: &P,
        direction: crate::Direction,
        m: &Mixed<W>,
    ) -> (crate::PauliSum<W>, Result<crate::PauliSum<W>, GpuError>)
    where
        P: PartitionedTruncation<W> + Sync,
    {
        use crate::engine::partitioned::backend::HostPartition;
        use crate::engine::partitioned::driver::{
            run_layers, scatter_local, PartitionCtx, PartitionWork,
        };
        use crate::engine::partitioned::transport::InProcessTransport;
        let nq = input.num_qubits();
        let n = circuit.channels.len();
        let mut group =
            InProcessTransport::group_with_timeout(2, std::time::Duration::from_secs(120))
                .into_iter();
        let (t0, t1) = (group.next().unwrap(), group.next().unwrap());
        let share =
            |rank: u32, coll: &dyn crate::engine::partitioned::Collectives| match m.bits {
                Some(bits) => input.filter_partition(&m.rows, rank).with_hash(
                    crate::bucket::hash::Gf2Hash::new(nq, bits, input.hash().seed()),
                ),
                None => scatter_local(input, &m.rows, rank, coll),
            };
        std::thread::scope(|s| {
            let rows = &m.rows;
            let share = &share;
            let h = s.spawn(move || {
                let mut part = HostPartition::new(share(0, &t0));
                let mut work = PartitionWork::take(&mut part, n, false);
                let ctx = PartitionCtx {
                    rows,
                    rank: 0,
                    size: 2,
                    tracing: false,
                };
                run_layers(circuit, policy, direction, m.options, ctx, &mut work, &t0);
                work.local.sum
            });
            let d = s.spawn(move || -> Result<crate::PauliSum<W>, GpuError> {
                let local = share(1, &t1);
                let mut part =
                    DevicePartition::new(GpuSum::from_host(&local, 0)?, m.device_options)?;
                part.group_size = 2;
                part.fail_at_layer = m.fail_at;
                part.policy = DevicePolicy::lower(policy.device_policy().unwrap())?;
                let mut work = PartitionWork::take(&mut part, n, false);
                let ctx = PartitionCtx {
                    rows,
                    rank: 1,
                    size: 2,
                    tracing: false,
                };
                run_layers(circuit, policy, direction, m.options, ctx, &mut work, &t1);
                work.local.take_error()?;
                work.local.sum().to_host()
            });
            (h.join().unwrap(), d.join().unwrap())
        })
    }

    /// A mixed run gathered against `propagate`: the wire format is shared.
    fn mixed_group<const W: usize, P>(
        circuit: &crate::Circuit<W>,
        input: &crate::PauliSum<W>,
        policy: &P,
        direction: crate::Direction,
        m: &Mixed<W>,
        what: &str,
    ) where
        P: PartitionedTruncation<W> + Sync,
    {
        let (host, device) = mixed_run(circuit, input, policy, direction, m);
        let device = device.expect("device rank");
        let got = crate::PauliSum::merge_partitions(vec![host, device]);
        let want = crate::propagate(circuit, input.clone(), policy, direction);
        assert_eq!(got.len(), want.len(), "{what}: term count");
        crate::test_support::assert_terms_close(&got, &want, 1e-11, what);
    }

    #[test]
    fn a_host_and_a_device_partition_interoperate() {
        crate::require_cuda!();
        use crate::test_support::{rand_sum, random_circuit, trotter_circuit, KeepAll};
        use crate::truncation::ApproxTopN;
        let trotter = trotter_circuit::<1>(24, 0.1);
        let input = rand_sum_real::<1>(1200, 24, 0x3D2);
        let m = Mixed::<1>::seeded(24);
        for direction in [crate::Direction::Forward, crate::Direction::Heisenberg] {
            mixed_group(
                &trotter,
                &input,
                &ApproxTopN(2000),
                direction,
                &m,
                &format!("mixed trotter {direction:?}"),
            );
        }
        let dense = random_circuit::<2>(70, 10, 0x3D3, true);
        let input2 = rand_sum::<2>(800, 70, 0x3D4);
        let m2 = Mixed::<2>::seeded(70);
        mixed_group(
            &dense,
            &input2,
            &KeepAll,
            crate::Direction::Forward,
            &m2,
            "mixed dense keep",
        );
        mixed_group(
            &dense,
            &input2,
            &ApproxTopN(3000),
            crate::Direction::Heisenberg,
            &m2,
            "mixed dense approx",
        );
    }

    /// `n` distinct terms carrying `X₀` and the identity on qubit 63: all on rank 0 under a row reading `Z₆₃`, and every one anticommutes with `Z₀Z₆₃`, so a `ZZ(0, 63)` layer exports all `n` to rank 1.
    fn x0_terms_identity_on_q63(n: usize, seed: u64) -> crate::PauliSum<1> {
        let base = crate::test_support::rand_sum::<1>(n, 64, seed);
        let mut acc = crate::accumulator::BuildAccumulator::<1>::new(64);
        for (x, z, c) in base.iter() {
            let p = crate::pauli_string::PauliString::<1> {
                x: [(x[0] | 1) & !(1u64 << 63)],
                z: [z[0] & !(1u64 << 63)],
            };
            acc.add_term(p, crate::phase::Phase::ONE, c);
        }
        let out = acc.finalize();
        assert_eq!(out.len(), n, "fixture: the terms must stay distinct");
        let zz = crate::pauli_string::PauliString::<1> {
            x: [0],
            z: [1 | (1u64 << 63)],
        };
        assert!(
            out.iter().all(|(x, z, _)| {
                !crate::pauli_string::PauliString::<1> { x: *x, z: *z }.commutes_with(&zz)
            }),
            "fixture: every term must anticommute with Z₀Z₆₃ or the layer exports fewer than {n} rows"
        );
        out
    }

    /// Rows reading `Z₆₃`: `ZZ(0, 63)` is remote and every `x0_terms_identity_on_q63` term sits on rank 0.
    fn rows_reading_z63() -> PartitionRows<1> {
        PartitionRows::<1>::from_rows(64, vec![[0u64]], vec![[1u64 << 63]])
    }

    /// The device rank is empty, so it ships empty blocks the host receives; the host ships rows the device merges.
    #[test]
    fn an_empty_device_partition_ships_empty_blocks_to_the_host() {
        crate::require_cuda!();
        use crate::test_support::{zz_rotation, KeepAll};
        let input = x0_terms_identity_on_q63(500, 0xE0);
        let mut c = crate::Circuit::<1>::new(64);
        c.push(zz_rotation::<1>(0, 63, 0.3));
        c.push(crate::channel::clifford::Clifford1Q::h(5));
        c.push(crate::channel::clifford::Clifford2Q::cnot(2, 63));
        c.push(zz_rotation::<1>(7, 63, 0.4));
        let m = Mixed::<1> {
            rows: rows_reading_z63(),
            ..Mixed::seeded(64)
        };
        mixed_group(
            &c,
            &input,
            &KeepAll,
            crate::Direction::Forward,
            &m,
            "empty device rank",
        );
    }

    /// A received segment of exactly `MAX_BUCKET_LEN` rows is merged; one more is `Unsupported` on the device while the host finishes.
    #[test]
    fn a_received_segment_at_the_tag_limit_is_accepted_and_one_more_is_unsupported() {
        crate::require_cuda!();
        use super::super::module::MAX_BUCKET_LEN;
        use crate::test_support::{zz_rotation, KeepAll};
        let mut c = crate::Circuit::<1>::new(64);
        c.push(zz_rotation::<1>(0, 63, 0.3));
        let m = Mixed::<1> {
            rows: rows_reading_z63(),
            bits: Some(0),
            options: crate::PropagateOptions {
                target_bucket_len: 1 << 20,
                min_buckets: 1,
                ..crate::PropagateOptions::default()
            },
            device_options: GpuLayerOptions {
                bucket_policy: super::super::layer::GpuBucketPolicy::TermsPerBucket(1 << 20),
                ..GpuLayerOptions::default()
            },
            fail_at: None,
        };
        let fits = x0_terms_identity_on_q63(MAX_BUCKET_LEN, 0xF1);
        mixed_group(
            &c,
            &fits,
            &KeepAll,
            crate::Direction::Forward,
            &m,
            "segment of 4096 rows",
        );
        let over = x0_terms_identity_on_q63(MAX_BUCKET_LEN + 1, 0xF2);
        let (host, device) = mixed_run(&c, &over, &KeepAll, crate::Direction::Forward, &m);
        assert!(
            matches!(device, Err(GpuError::Unsupported(_))),
            "device: {:?}, host {} terms",
            device.as_ref().map(|s| s.len()),
            host.len()
        );
        assert_eq!(host.len(), over.len(), "the host rank finished its layer");
    }

    /// The tag limit is per segment, not per block: a received block of well over `MAX_BUCKET_LEN` rows split across two positions is merged in full.
    #[test]
    fn a_received_block_above_the_tag_limit_is_merged_when_every_segment_fits() {
        crate::require_cuda!();
        use super::super::module::MAX_BUCKET_LEN;
        use crate::test_support::{zz_rotation, KeepAll};
        let mut c = crate::Circuit::<1>::new(64);
        c.push(zz_rotation::<1>(0, 63, 0.3));
        let m = Mixed::<1> {
            rows: rows_reading_z63(),
            bits: Some(1),
            options: crate::PropagateOptions {
                target_bucket_len: 1 << 20,
                min_buckets: 1,
                ..crate::PropagateOptions::default()
            },
            device_options: GpuLayerOptions {
                bucket_policy: super::super::layer::GpuBucketPolicy::TermsPerBucket(1 << 20),
                ..GpuLayerOptions::default()
            },
            fail_at: None,
        };
        let input = x0_terms_identity_on_q63(MAX_BUCKET_LEN + MAX_BUCKET_LEN / 2, 0xF3);
        mixed_group(
            &c,
            &input,
            &KeepAll,
            crate::Direction::Forward,
            &m,
            "block of 6144 rows over two segments",
        );
    }

    /// A device rank failing before its exchange leaves the host rank a finished run and returns the injected error.
    #[test]
    fn an_injected_failure_before_the_exchange_lets_the_host_partner_finish() {
        crate::require_cuda!();
        use crate::test_support::{rand_sum, random_circuit, KeepAll};
        let dense = random_circuit::<1>(8, 6, 0x3D5, true);
        let input = rand_sum::<1>(300, 8, 0x3D6);
        let m = Mixed::<1> {
            fail_at: Some(1),
            ..Mixed::seeded(8)
        };
        let (host, device) = mixed_run(&dense, &input, &KeepAll, crate::Direction::Forward, &m);
        assert!(
            matches!(
                device,
                Err(GpuError::Unsupported("injected before the exchange"))
            ),
            "{device:?}"
        );
        assert!(!host.is_empty(), "the host rank finished all six layers");
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
