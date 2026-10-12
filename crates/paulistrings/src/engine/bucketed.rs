//! One layer of the bucketed engine, applied coset by coset, and the reusable `LayerScratch`.
//!
//! See ARCHITECTURE.md §Engine.

use std::sync::Mutex;

use num_complex::Complex64;
use rayon::prelude::*;

use super::coset::{CosetOrder, Gf2Span};
use super::merge::{rest_rows_per_key, SortKernel};
use super::partitioned::trace::PartitionTrace;
use crate::channel::prepared::{LocalPtm, Prepared, PreparedRotation};
use crate::pauli_sum::storage::{BucketColumns, PauliSum};
use crate::truncation::TruncationPolicy;

#[cfg(feature = "phase-timing")]
use super::stats::{PartitionPhaseStats, PhaseStats};
use super::timing::{lap, stamp};

mod coset_fill;

pub(super) use coset_fill::{fill_coset, CosetScratch, MIN_COSETS_FOR_PARALLEL};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Reusable layer working set for [`propagate_with`](crate::propagate_with), which keeps its capacity across layers and calls and carries the opt-in traces.
///
/// A task's output cannot depend on which worker slot it drew: every write destination is cleared before use.
#[derive(Debug, Default)]
pub struct LayerScratch<const W: usize> {
    /// The serial path's coset working set.
    pub(super) task: CosetScratch<W>,
    /// The layer's coset order, with its inverse only under [`ExtraRows::NEEDS_BETA`]; a partitioned layer builds it before the export and passes it back in [`LayerKnobs::order`].
    pub(super) order: CosetOrder,
    /// Bucket handles in coset-contiguous order while a layer runs.
    pub(super) staging: Vec<BucketColumns<W>>,
    /// One coset working set per Rayon worker, indexed by `rayon::current_thread_index()`, so each mutex is uncontended.
    pub(super) workers: Vec<Mutex<CosetScratch<W>>>,
    /// Layer-level wall-clock counters; per-coset busy time lives in each `CosetScratch`.
    #[cfg(feature = "phase-timing")]
    pub(crate) stats: PhaseStats,
    /// `None` unless [`Self::enable_trace`] was called.
    pub(crate) trace: Option<PartitionTrace>,
}

impl<const W: usize> LayerScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain the phase counters since the last drain, zeroing them; `per_partition` has one entry.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        let stats = self.take_phase_stats();
        PartitionPhaseStats {
            layers: stats.layers,
            per_partition: vec![stats],
            scatter_ns: 0,
            gather_ns: 0,
        }
    }

    /// Drain this scratch's own counters, one partition's entry of a driver's [`PartitionPhaseStats`].
    #[cfg(feature = "phase-timing")]
    pub(crate) fn take_phase_stats(&mut self) -> PhaseStats {
        let mut total = std::mem::take(&mut self.stats);
        total.absorb_coset(&std::mem::take(&mut self.task.stats));
        for slot in &self.workers {
            let mut worker_scratch = slot.lock().unwrap();
            total.absorb_coset(&std::mem::take(&mut worker_scratch.stats));
        }
        total
    }

    /// Start recording a one-partition [`PartitionTrace`] on every later propagation driven by this scratch; idempotent.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain the per-layer records, or `None` if tracing was never enabled; tracing stays enabled.
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.trace.as_mut().map(std::mem::take)
    }
}

/// A prepared channel's delta set annotated with each entry's coset coordinate, computed once per layer.
pub(super) enum DeltaPlan<'p, const W: usize> {
    /// Tabulated deltas; `coordinates[e]` pairs with `ptm.deltas()[e]`.
    Local {
        ptm: &'p LocalPtm<W>,
        coordinates: Vec<u32>,
        /// Whether `deltas()[0]` is the identity delta, whose stream goes to the run's pre-sorted `id` columns.
        has_identity: bool,
        /// Whether the identity amplitude is nonzero on every active support pattern, so the merge borrows the id keys from the source bucket (ARCHITECTURE.md §Engine).
        dense_identity: bool,
        /// This layer's per-run sort kernel.
        sort_kernel: SortKernel,
    },
    /// Wide rotation: two implicit entries, the identity pass and the generator pass.
    Rotation {
        rotation: &'p PreparedRotation<W>,
        coordinate_identity: u32,
        generator_coordinate: u32,
        /// Whether the generator pass emits here ([`LayerKnobs::generator_local`]); `generator_coordinate` is 0 when not.
        generator_local: bool,
    },
}

/// Per-layer overrides the partitioned engine hands the coset loop; [`Default`] is the unpartitioned answer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerKnobs<'k> {
    /// Bucket deltas to build the coset span from, instead of [`Prepared::bucket_deltas`].
    pub bucket_deltas: Option<&'k [u32]>,
    /// The channel's total rest-stream count, so every partition picks the sort kernel the unpartitioned run would.
    pub rest_streams: Option<usize>,
    /// The unrestricted channel's [`rest_rows_per_key`], since a PTM cut down to local entries can look more disjoint than the channel is.
    pub rows_per_key: Option<f64>,
    /// Whether a wide rotation's generator pass emits here; false when every generator row belongs to a partner.
    pub generator_local: bool,
    /// The layer's coset order, already built `with_inverse` over the span of `bucket_deltas`; built here when `None`.
    pub order: Option<&'k CosetOrder>,
}

impl Default for LayerKnobs<'_> {
    fn default() -> Self {
        Self {
            bucket_deltas: None,
            rest_streams: None,
            rows_per_key: None,
            generator_local: true,
            order: None,
        }
    }
}

impl<'p, const W: usize> DeltaPlan<'p, W> {
    pub(super) fn new(prepared: &'p Prepared<W>, span: &Gf2Span, knobs: LayerKnobs<'_>) -> Self {
        match prepared {
            Prepared::Local(ptm) => {
                let coordinates: Vec<u32> = ptm
                    .deltas()
                    .iter()
                    .map(|d| span.coordinate_of(d.bucket_delta))
                    .collect();
                let has_identity = ptm.has_identity();
                debug_assert!(!has_identity || coordinates[0] == 0);
                let dense_identity = has_identity && ptm.entry_is_dense(0);
                let rest_streams = knobs.rest_streams.unwrap_or(ptm.rest_streams());
                let sort_kernel = SortKernel::for_layer(rest_streams, || {
                    knobs.rows_per_key.unwrap_or_else(|| rest_rows_per_key(ptm))
                });
                DeltaPlan::Local {
                    ptm,
                    coordinates,
                    has_identity,
                    dense_identity,
                    sort_kernel,
                }
            }
            Prepared::Rotation(rotation) => DeltaPlan::Rotation {
                rotation,
                coordinate_identity: span.coordinate_of(rotation.bucket_delta_identity),
                // `coordinate_of` demands its argument be in the span, and a remote generator's bucket delta is not.
                generator_coordinate: if knobs.generator_local {
                    span.coordinate_of(rotation.bucket_delta_generator)
                } else {
                    0
                },
                generator_local: knobs.generator_local,
            },
        }
    }
}

/// Rows received from other partitions, appended to an output bucket's rest stream before the sort so `keep_term` sees the complete sum (ARCHITECTURE.md §Truncation).
/// An implementation that can return rows must set `NEEDS_BETA = true`; under `false` every call site compiles away.
pub(crate) trait ExtraRows<const W: usize> {
    /// Whether the engine recovers original bucket indices and calls this source at all.
    const NEEDS_BETA: bool = false;

    /// Rows destined for original bucket `beta`; must equal what [`append_into`](Self::append_into) pushes.
    fn count(&self, beta: u32) -> usize {
        let _ = beta;
        0
    }

    /// Append original bucket `beta`'s rows onto the run's rest columns.
    fn append_into(&self, beta: u32, rest: &mut BucketColumns<W>) {
        let _ = (beta, rest);
    }
}

/// The unpartitioned engine's [`ExtraRows`]: no rows.
pub(crate) struct NoExtra;

impl<const W: usize> ExtraRows<W> for NoExtra {}

/// Apply one prepared channel to a bucketed sum, folding `keep_term` into the merge; the caller runs `finalize_layer`.
pub fn apply_layer_bucketed<const W: usize, T>(
    sum: &mut PauliSum<W>,
    prepared: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    apply_layer_bucketed_with(
        sum,
        prepared,
        policy,
        scratch,
        &NoExtra,
        LayerKnobs::default(),
    )
}

/// [`apply_layer_bucketed`] with the partitioned engine's [`ExtraRows`] source and [`LayerKnobs`].
pub(crate) fn apply_layer_bucketed_with<const W: usize, T, X>(
    sum: &mut PauliSum<W>,
    prepared: &Prepared<W>,
    policy: &T,
    scratch: &mut LayerScratch<W>,
    extra: &X,
    knobs: LayerKnobs<'_>,
) where
    T: TruncationPolicy<W> + ?Sized,
    X: ExtraRows<W> + Sync,
{
    stamp!(stamp);

    // A partitioned layer's local table can look key-preserving while received rows still need merging.
    if let Prepared::Local(ptm) = prepared {
        if !X::NEEDS_BETA && ptm.is_key_preserving() {
            rescale_in_place(sum, ptm, policy);
            lap!(stamp, scratch.stats.rescale_ns);
            return;
        }
    }

    let own_deltas;
    let deltas: &[u32] = match knobs.bucket_deltas {
        Some(deltas) => deltas,
        None => {
            own_deltas = prepared.bucket_deltas();
            &own_deltas
        }
    };
    let span = Gf2Span::new(deltas, sum.hash().bits());
    let plan = DeltaPlan::new(prepared, &span, knobs);
    let coset_size = span.coset_size();
    let num_cosets = span.num_cosets();
    lap!(stamp, scratch.stats.span_plan_ns);

    // Coset `c` owns `staging[c·2^r .. (c+1)·2^r]`; at `r = 0` the order is the identity and the handle passes are skipped.
    let order: &CosetOrder = match knobs.order {
        Some(order) => order,
        None => {
            scratch.order.rebuild(&span, X::NEEDS_BETA);
            &scratch.order
        }
    };
    debug_assert_eq!(order.positions(), sum.num_buckets());
    debug_assert_eq!(order.dimension(), span.dimension());
    let identity_order = order.is_identity();
    if !identity_order {
        let buckets = sum.buckets_mut();
        scratch
            .staging
            .resize_with(buckets.len(), BucketColumns::default);
        for (beta, columns) in buckets.iter_mut().enumerate() {
            scratch.staging[order.position_of(beta as u32) as usize] = std::mem::take(columns);
        }
    }
    lap!(stamp, scratch.stats.permute_ns);

    // Each coset reads and writes only its own chunk (ARCHITECTURE.md §Parallelism).
    {
        if num_cosets >= MIN_COSETS_FOR_PARALLEL {
            let pool = rayon::current_num_threads().max(1);
            if scratch.workers.len() < pool {
                scratch.workers.resize_with(pool, Mutex::default);
            }
        }
        let workers = &scratch.workers;
        let chunks: &mut [BucketColumns<W>] = if identity_order {
            sum.buckets_mut()
        } else {
            scratch.staging.as_mut_slice()
        };
        if num_cosets < MIN_COSETS_FOR_PARALLEL {
            for (coset_index, chunk) in chunks.chunks_mut(coset_size).enumerate() {
                let base = coset_index * coset_size;
                fill_coset::<W, T, X>(chunk, &plan, policy, &mut scratch.task, extra, base, order);
            }
        } else {
            chunks
                .par_chunks_mut(coset_size)
                .enumerate()
                .for_each(|(coset_index, chunk)| {
                    let base = coset_index * coset_size;
                    // `workers` was sized to `current_num_threads()` above, and this closure runs on that pool.
                    let worker = rayon::current_thread_index()
                        .expect("the coset loop runs on a Rayon pool worker");
                    let mut worker_scratch = workers[worker].lock().unwrap();
                    fill_coset::<W, T, X>(
                        chunk,
                        &plan,
                        policy,
                        &mut worker_scratch,
                        extra,
                        base,
                        order,
                    );
                });
        }
    }
    lap!(stamp, scratch.stats.coset_loop_ns);

    if !identity_order {
        let buckets = sum.buckets_mut();
        for (beta, columns) in buckets.iter_mut().enumerate() {
            *columns =
                std::mem::take(&mut scratch.staging[order.position_of(beta as u32) as usize]);
        }
    }
    lap!(stamp, scratch.stats.unpermute_ns);
    sum.recount();
    lap!(stamp, scratch.stats.recount_ns);

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

/// In-place coefficient rescale for a key-preserving channel, with the general path's zero drop and `keep_term`.
fn rescale_in_place<const W: usize, T>(sum: &mut PauliSum<W>, ptm: &LocalPtm<W>, policy: &T)
where
    T: TruncationPolicy<W> + ?Sized,
{
    let amplitude = &ptm.deltas()[0].amplitude;
    sum.update_per_bucket(|_| {
        |x: &[u64; W], z: &[u64; W], c: Complex64| {
            let coeff = c * amplitude[ptm.support_bits(x, z)];
            (coeff != ZERO && policy.keep_term(x, z, coeff)).then_some(coeff)
        }
    });

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

#[cfg(test)]
mod tests;
