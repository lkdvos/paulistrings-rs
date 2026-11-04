//! One layer of the bucketed engine, applied coset by coset, and the reusable `LayerScratch`.
//!
//! A coset task swaps its bucket columns into worker scratch and writes its outputs back into the emptied columns, so the layer runs in place.
//! Identity-delta rows keep their keys, so a run's identity stream inherits the source bucket's order and is never sorted; every other row, received rows included, joins the rest stream.
//! A table shaped like a rotation skips both streams and runs `rotation_layer` on each bucket in place.

use std::sync::Mutex;

use crate::coefficient::Coefficient;
use rayon::prelude::*;

use super::coset::{CosetOrder, Gf2Span};
use super::merge::{rest_rows_per_key, SortKernel};
use super::trace::PartitionTrace;
use crate::channel::prepared::{LocalPtm, Prepared, PreparedRotation};
use crate::collectives::Collectives;
use crate::pauli_sum::storage::{BucketColumns, PauliSum};
use crate::truncation::builtin::{approx_top_n_pass, EdgeDecision};
use crate::truncation::octave_window::{finalize_from_window, BucketOctaves, OctaveWindow};
use crate::truncation::TruncationPolicy;
use rotation_layer::{
    fill_received_rotation_bucket, fill_rotation_coset, ReceivedRotation, RotationLayer,
};

#[cfg(feature = "phase-timing")]
use super::stats::{PartitionPhaseStats, PhaseStats};
use super::timing::{lap, stamp};

mod coset_fill;
mod partner_order;
mod rotation_layer;

pub(super) use coset_fill::{fill_coset, CosetScratch, MIN_COSETS_FOR_PARALLEL};

/// Reusable layer working set for [`propagate_with`](crate::propagate_with), which keeps its capacity across layers and calls and carries the opt-in traces.
///
/// A task's output cannot depend on which worker slot it drew: every write destination is cleared before use.
#[derive(Debug, Default)]
pub struct LayerScratch<const W: usize, C: Coefficient = num_complex::Complex64> {
    /// The serial path's coset working set.
    pub(super) task: CosetScratch<W, C>,
    /// The partitioned layer's coset order, kept here for its allocations; the coset loop itself enumerates cosets from the span.
    pub(super) order: CosetOrder,
    /// One coset working set per Rayon worker, indexed by `rayon::current_thread_index()`, so each mutex is uncontended.
    pub(super) workers: Vec<Mutex<CosetScratch<W, C>>>,
    /// Layer-level wall-clock counters; per-coset busy time lives in each `CosetScratch`.
    #[cfg(feature = "phase-timing")]
    pub(crate) stats: PhaseStats,
    /// `None` unless [`Self::enable_trace`] was called.
    pub(crate) trace: Option<PartitionTrace>,
    /// The octave counts an `ApproxTopN` layer pass reads instead of the sum.
    pub(crate) octaves: LayerOctaves,
}

/// Per-bucket octave counts, taken by the coset loop as it writes each bucket, for [`finalize_from_window`].
///
/// Every partition's window is set from the agreed boundary octave, so all count alike.
#[derive(Debug, Default)]
pub(crate) struct LayerOctaves {
    /// The window the next layer counts, centred on the last boundary octave; `None` before the first.
    window: Option<OctaveWindow>,
    per_bucket: Vec<BucketOctaves>,
    /// Whether `per_bucket` counts the sum as the last layer left it, under `window`.
    counted: bool,
}

impl LayerOctaves {
    /// `ApproxTopN(n)`'s layer pass over `sum`, from the counts when the last layer took them.
    pub(crate) fn finalize<const W: usize, C: Coefficient>(
        &mut self,
        sum: &mut PauliSum<W, C>,
        collectives: &dyn Collectives,
        n: usize,
    ) {
        let edge = match self.window {
            Some(window) if std::mem::take(&mut self.counted) => {
                finalize_from_window(sum, collectives, n, window, &self.per_bucket)
            }
            _ => approx_top_n_pass(sum, collectives, n),
        };
        if let EdgeDecision::Boundary { bin, .. } = edge {
            self.window = Some(OctaveWindow::around(bin));
        }
    }
}

impl<const W: usize, C: Coefficient> LayerScratch<W, C> {
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

    /// The counters [`Self::take_phase_stats`] would drain, left in place.
    #[cfg(feature = "phase-timing")]
    pub(crate) fn peek_phase_stats(&self) -> PhaseStats {
        let mut total = self.stats;
        total.absorb_coset(&self.task.stats);
        for slot in &self.workers {
            total.absorb_coset(&slot.lock().unwrap().stats);
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
pub(super) enum DeltaPlan<'p, const W: usize, C: Coefficient = num_complex::Complex64> {
    /// Tabulated deltas; `coordinates[e]` pairs with `ptm.deltas()[e]`.
    Local {
        ptm: &'p LocalPtm<W, C>,
        coordinates: Vec<u32>,
        /// Whether `deltas()[0]` is the identity delta, whose stream goes to the run's pre-sorted `id` columns.
        has_identity: bool,
        /// Whether the identity amplitude is nonzero on every active support pattern, so the identity stream is one-to-one with the source bucket and the merge borrows its key columns.
        dense_identity: bool,
        /// This layer's per-run sort kernel.
        sort_kernel: SortKernel,
    },
    /// Wide rotation: two implicit entries, the identity pass and the generator pass.
    Rotation {
        rotation: &'p PreparedRotation<W, C::Real>,
        coordinate_identity: u32,
        generator_coordinate: u32,
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
    /// Whether a rotation's generator pass emits here; false when the generator crosses to a partner, which sends the layer, tabulated or wide, to `fill_received_rotation_bucket`.
    pub generator_local: bool,
    /// The coset order a partitioned layer exported and chunks by, whose coset `c` must be the loop's coset `c` ([`Gf2Span::representative_at`]); checked in debug builds.
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

impl<'p, const W: usize, C: Coefficient> DeltaPlan<'p, W, C> {
    pub(super) fn new(prepared: &'p Prepared<W, C>, span: &Gf2Span, knobs: LayerKnobs<'_>) -> Self {
        match prepared {
            Prepared::Local(ptm) => {
                let coordinates: Vec<u32> =
                    ptm.deltas().iter().map(|d| span.coordinate_of(d.bucket_delta)).collect();
                let has_identity = ptm.has_identity();
                debug_assert!(!has_identity || coordinates[0] == 0);
                let dense_identity = has_identity && ptm.entry_is_dense(0);
                let rest_streams = knobs.rest_streams.unwrap_or(ptm.rest_streams());
                let sort_kernel = SortKernel::for_layer(rest_streams, || {
                    knobs.rows_per_key.unwrap_or_else(|| rest_rows_per_key(ptm))
                });
                DeltaPlan::Local { ptm, coordinates, has_identity, dense_identity, sort_kernel }
            }
            Prepared::Rotation(rotation) => {
                debug_assert!(
                    knobs.generator_local,
                    "a remote generator runs on `ReceivedRotation`"
                );
                DeltaPlan::Rotation {
                    rotation,
                    coordinate_identity: span.coordinate_of(rotation.bucket_delta_identity),
                    generator_coordinate: span.coordinate_of(rotation.bucket_delta_generator),
                }
            }
        }
    }
}

/// Rows received from other partitions, appended to an output bucket's rest stream before the sort so `keep_term` sees the complete sum; never the identity stream, which the merge requires strictly ascending.
/// An implementation that can return rows must set `NEEDS_BETA = true`; under `false` every call site compiles away.
pub(crate) trait ExtraRows<const W: usize, C: Coefficient = num_complex::Complex64> {
    /// Whether the engine recovers original bucket indices and calls this source at all.
    const NEEDS_BETA: bool = false;

    /// Rows destined for original bucket `beta`; must equal what [`append_into`](Self::append_into) pushes.
    fn count(&self, beta: u32) -> usize {
        let _ = beta;
        0
    }

    /// Append original bucket `beta`'s rows onto the run's rest columns.
    fn append_into(&self, beta: u32, rest: &mut BucketColumns<W, C>) {
        let _ = (beta, rest);
    }
}

/// The unpartitioned engine's [`ExtraRows`]: no rows.
pub(crate) struct NoExtra;

impl<const W: usize, C: Coefficient> ExtraRows<W, C> for NoExtra {}

/// Apply one prepared channel to a bucketed sum, folding `keep_term` into the merge; the caller runs `finalize_layer`.
pub fn apply_layer_bucketed<const W: usize, C: Coefficient, T>(
    sum: &mut PauliSum<W, C>,
    prepared: &Prepared<W, C>,
    policy: &T,
    scratch: &mut LayerScratch<W, C>,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    apply_layer_bucketed_with(sum, prepared, policy, scratch, &NoExtra, LayerKnobs::default())
}

/// [`apply_layer_bucketed`] with the partitioned engine's [`ExtraRows`] source and [`LayerKnobs`].
pub(crate) fn apply_layer_bucketed_with<const W: usize, C: Coefficient, T, X>(
    sum: &mut PauliSum<W, C>,
    prepared: &Prepared<W, C>,
    policy: &T,
    scratch: &mut LayerScratch<W, C>,
    extra: &X,
    knobs: LayerKnobs<'_>,
) where
    T: TruncationPolicy<W, C> + ?Sized,
    X: ExtraRows<W, C> + Sync,
{
    stamp!(stamp);
    scratch.octaves.counted = false;

    // A partitioned layer's local table can look key-preserving while received rows still need merging.
    let rotation = match prepared {
        Prepared::Local(ptm) if !X::NEEDS_BETA => {
            if ptm.is_key_preserving() {
                rescale_in_place(sum, ptm, policy);
                lap!(stamp, scratch.stats.rescale_ns);
                return;
            }
            RotationLayer::of(ptm)
        }
        _ => None,
    };
    // A partitioned rotation whose generator pass is off exported every partner.
    let received_rotation = match prepared {
        _ if !X::NEEDS_BETA || knobs.generator_local => None,
        Prepared::Local(ptm) => Some(ReceivedRotation::Table(
            RotationLayer::of(ptm).expect("only a rotation table runs with its generator pass off"),
        )),
        Prepared::Rotation(rotation) => Some(ReceivedRotation::wide(rotation)),
    };

    let own_deltas;
    let deltas: &[u32] = match knobs.bucket_deltas {
        Some(deltas) => deltas,
        None => {
            own_deltas = prepared.bucket_deltas();
            &own_deltas
        }
    };
    let span = Gf2Span::new(deltas, sum.hash().bits());
    debug_assert!(knobs.order.is_none_or(|order| order.dimension() == span.dimension()));

    let window = scratch.octaves.window.filter(|_| policy.sole_approx_top_n().is_some());
    let mut per_bucket = std::mem::take(&mut scratch.octaves.per_bucket);
    if window.is_some() {
        per_bucket.resize(sum.num_buckets(), [0; _]);
    }
    let octaves = window.map(|window| (window, DisjointSlots::new(&mut per_bucket)));
    let octaves = octaves.as_ref().map(|(window, slots)| (*window, slots));
    let len = match (received_rotation, rotation) {
        (Some(layer), _) => {
            debug_assert_eq!(span.dimension(), 0, "only the identity delta is local");
            lap!(stamp, scratch.stats.span_plan_ns);
            coset_loop(sum, &span, scratch, octaves, |members, betas, worker_scratch| {
                fill_received_rotation_bucket(members, betas, &layer, policy, worker_scratch, extra)
            })
        }
        (None, Some(layer)) => {
            debug_assert!(knobs.order.is_none());
            lap!(stamp, scratch.stats.span_plan_ns);
            coset_loop(sum, &span, scratch, octaves, |members, _, worker_scratch| {
                fill_rotation_coset(members, &layer, policy, worker_scratch)
            })
        }
        (None, None) => {
            let plan = DeltaPlan::new(prepared, &span, knobs);
            lap!(stamp, scratch.stats.span_plan_ns);
            coset_loop(sum, &span, scratch, octaves, |members, betas, worker_scratch| {
                fill_coset::<W, C, T, X>(members, &plan, policy, worker_scratch, extra, betas)
            })
        }
    };
    scratch.octaves.per_bucket = per_bucket;
    scratch.octaves.counted = window.is_some();
    lap!(stamp, scratch.stats.coset_loop_ns);
    sum.set_len_after_layer(len);
    lap!(stamp, scratch.stats.recount_ns);

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

/// Each bucket's slot in a layer's octave counts, and the window they count.
type OctaveSlots<'s, 'a> = Option<(OctaveWindow, &'s DisjointSlots<'a, BucketOctaves>)>;

/// Run `fill(members, betas, scratch)` on every coset of `span`, serially or on the Rayon pool, each over its swapped-out bucket columns, counting each output bucket into `octaves`; returns the output terms.
fn coset_loop<const W: usize, C: Coefficient, F>(
    sum: &mut PauliSum<W, C>,
    span: &Gf2Span,
    scratch: &mut LayerScratch<W, C>,
    octaves: OctaveSlots<'_, '_>,
    fill: F,
) -> usize
where
    F: Fn(&mut [BucketColumns<W, C>], &[u32], &mut CosetScratch<W, C>) + Sync,
{
    let num_cosets = span.num_cosets();
    // Each coset reads and writes only its own buckets (engine/coset.rs).
    if num_cosets < MIN_COSETS_FOR_PARALLEL {
        let buckets = DisjointSlots::new(sum.buckets_mut());
        (0..num_cosets as u32)
            .map(|coset| run_coset(coset, span, &buckets, octaves, &mut scratch.task, &fill))
            .sum()
    } else {
        let pool = rayon::current_num_threads().max(1);
        if scratch.workers.len() < pool {
            scratch.workers.resize_with(pool, Mutex::default);
        }
        let workers = &scratch.workers;
        let buckets = DisjointSlots::new(sum.buckets_mut());
        (0..num_cosets as u32)
            .into_par_iter()
            .map(|coset| {
                // `workers` was sized to `current_num_threads()` above, and this closure runs on that pool.
                let worker = rayon::current_thread_index()
                    .expect("the coset loop runs on a Rayon pool worker");
                let mut worker_scratch = workers[worker].lock().unwrap();
                run_coset(coset, span, &buckets, octaves, &mut worker_scratch, &fill)
            })
            .sum()
    }
}

/// Swap coset `coset`'s buckets into the task's member slots, fill them, count them while they are in cache, and swap the outputs back; returns the coset's output terms.
fn run_coset<const W: usize, C: Coefficient, F>(
    coset: u32,
    span: &Gf2Span,
    buckets: &DisjointSlots<'_, BucketColumns<W, C>>,
    octaves: OctaveSlots<'_, '_>,
    scratch: &mut CosetScratch<W, C>,
    fill: &F,
) -> usize
where
    F: Fn(&mut [BucketColumns<W, C>], &[u32], &mut CosetScratch<W, C>),
{
    let representative = span.representative_at(coset);
    let mut members = std::mem::take(&mut scratch.members);
    let mut betas = std::mem::take(&mut scratch.betas);
    betas.clear();
    betas.extend((0..span.coset_size() as u32).map(|j| span.member(representative, j)));
    members.resize_with(betas.len(), BucketColumns::default);
    for (slot, &beta) in members.iter_mut().zip(&betas) {
        // SAFETY: the cosets partition the buckets, so no other task reaches bucket `beta`.
        std::mem::swap(slot, unsafe { buckets.get(beta as usize) });
    }
    fill(&mut members, &betas, scratch);
    let mut len = 0;
    for (slot, &beta) in members.iter_mut().zip(&betas) {
        len += slot.len();
        if let Some((window, counts)) = octaves {
            // SAFETY: as for the buckets, which the counts are indexed alike with.
            window.count(&slot.coeff, unsafe { counts.get(beta as usize) });
        }
        // SAFETY: as above.
        std::mem::swap(slot, unsafe { buckets.get(beta as usize) });
    }
    scratch.members = members;
    scratch.betas = betas;
    len
}

/// A slice shared by tasks that each touch pairwise distinct elements.
struct DisjointSlots<'a, T> {
    base: *mut T,
    len: usize,
    _slice: std::marker::PhantomData<&'a mut [T]>,
}

// SAFETY: `get`'s contract keeps every element reached by at most one task at a time, so sharing the wrapper shares no element.
unsafe impl<T: Send> Sync for DisjointSlots<'_, T> {}

impl<'a, T> DisjointSlots<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        Self { base: slice.as_mut_ptr(), len: slice.len(), _slice: std::marker::PhantomData }
    }

    /// Element `index`.
    ///
    /// # Safety
    ///
    /// No other reference to element `index` may be live while the returned one is.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get(&self, index: usize) -> &mut T {
        assert!(index < self.len, "DisjointSlots: index {index} out of {}", self.len);
        // SAFETY: in bounds, and exclusive by the caller's contract.
        unsafe { &mut *self.base.add(index) }
    }
}

/// Whether `ptm` has a rotation's shape, which a partitioned layer keeps whole when its generator crosses.
pub(crate) fn is_rotation_table<const W: usize, C: Coefficient>(ptm: &LocalPtm<W, C>) -> bool {
    RotationLayer::of(ptm).is_some()
}

/// In-place coefficient rescale for a key-preserving channel, with the general path's zero drop and `keep_term`.
fn rescale_in_place<const W: usize, C: Coefficient, T>(
    sum: &mut PauliSum<W, C>,
    ptm: &LocalPtm<W, C>,
    policy: &T,
) where
    T: TruncationPolicy<W, C> + ?Sized,
{
    let amplitude = &ptm.deltas()[0].amplitude;
    sum.update_per_bucket(|_| {
        |x: &[u64; W], z: &[u64; W], c: C| {
            let coeff = c * amplitude[ptm.support_bits(x, z)];
            (coeff != C::ZERO && policy.keep_term(x, z, coeff)).then_some(coeff)
        }
    });

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

#[cfg(test)]
mod tests;
