//! One layer of the bucketed engine, applied coset by coset, and the reusable `LayerScratch`.
//!
//! See ARCHITECTURE.md §Engine.

use std::sync::Mutex;

use num_complex::Complex64;
use rayon::prelude::*;

use super::coset::Gf2Span;
use super::merge::{
    RADIX_MAX_REST_ROWS_PER_KEY, RADIX_MIN_DISJOINT_STREAMS, RADIX_MIN_REST_STREAMS,
};
use crate::channel::prepared::{LocalPtm, Prepared, PreparedRotation};
use crate::pauli_sum::storage::{BucketColumns, PauliSum};
use crate::truncation::TruncationPolicy;

#[cfg(feature = "phase-timing")]
use super::stats::{PhaseStats, Stamp};

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
    /// The layer's handle permutation, `permutation[β] = span.permuted_index(β)`.
    pub(super) permutation: Vec<u32>,
    /// The inverse of [`Self::permutation`], filled only under [`ExtraRows::NEEDS_BETA`]; empty means the identity.
    pub(super) inverse_permutation: Vec<u32>,
    /// Bucket handles in coset-contiguous order while a layer runs.
    pub(super) staging: Vec<BucketColumns<W>>,
    /// One coset working set per Rayon worker, indexed by `rayon::current_thread_index()`, so each mutex is uncontended.
    pub(super) workers: Vec<Mutex<CosetScratch<W>>>,
    /// Layer-level wall-clock counters; per-coset busy time lives in each `CosetScratch`.
    #[cfg(feature = "phase-timing")]
    pub(crate) stats: PhaseStats,
    /// `None` unless [`Self::enable_term_trace`] was called.
    pub(crate) term_trace: Option<TermTrace>,
    /// `None` unless [`Self::enable_gate_trace`] was called.
    pub(crate) gate_trace: Option<GateTrace>,
}

impl<const W: usize> LayerScratch<W> {
    /// An empty scratch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain the phase counters accumulated since the last drain, across layers and calls.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PhaseStats {
        let mut total = std::mem::take(&mut self.stats);
        total.absorb_coset(&std::mem::take(&mut self.task.stats));
        for slot in &self.workers {
            let mut worker_scratch = slot.lock().unwrap();
            total.absorb_coset(&std::mem::take(&mut worker_scratch.stats));
        }
        total
    }

    /// Record a [`TermTrace`] on every later call driven by this scratch; idempotent.
    pub fn enable_term_trace(&mut self) {
        self.term_trace.get_or_insert_with(TermTrace::default);
    }

    /// Drain the term trace, or `None` if it was never enabled; tracing stays enabled.
    pub fn take_term_trace(&mut self) -> Option<TermTrace> {
        self.term_trace.as_mut().map(std::mem::take)
    }

    /// Record a [`GateTrace`] on every later call driven by this scratch; idempotent.
    pub fn enable_gate_trace(&mut self) {
        self.gate_trace.get_or_insert_with(GateTrace::default);
    }

    /// Drain the gate trace, or `None` if it was never enabled; tracing stays enabled.
    pub fn take_gate_trace(&mut self) -> Option<GateTrace> {
        self.gate_trace.as_mut().map(std::mem::take)
    }
}

/// Resident term counts between layers, one entry per layer in application order.
///
/// Counts are post-truncation; the transient expansion inside a layer is not captured.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TermTrace {
    /// Resident term count before each layer.
    pub terms_in: Vec<usize>,
    /// Resident term count after each layer's `finalize_layer`.
    pub terms_out: Vec<usize>,
}

impl TermTrace {
    /// Peak resident term count between layers, or `None` for a zero-layer trace.
    pub fn peak_terms(&self) -> Option<usize> {
        self.terms_in
            .first()
            .copied()
            .into_iter()
            .chain(self.terms_out.iter().copied())
            .max()
    }
}

/// Per-layer gate records, one entry per layer in application order (reverse circuit order under [`Direction::Heisenberg`](crate::Direction::Heisenberg)).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GateTrace {
    /// This layer's position in [`Circuit::channels`](crate::Circuit), independent of propagation direction.
    pub circuit_index: Vec<u32>,
    /// This layer's position in application order.
    pub application_index: Vec<u32>,
    /// [`Channel::debug_name`](crate::Channel::debug_name) of the applied channel.
    pub gate_name: Vec<&'static str>,
    /// Resident term count before the layer.
    pub terms_in: Vec<usize>,
    /// Resident term count after the layer, i.e. after `finalize_layer`.
    pub terms_out: Vec<usize>,
    /// Wall-clock nanoseconds for the whole layer, from `rebucket` through `finalize_layer`.
    pub nanos: Vec<u64>,
}

/// A prepared channel's delta set annotated with each entry's coset coordinate, computed once per layer.
pub(super) enum DeltaPlan<'p, const W: usize> {
    /// Tabulated deltas; `coords[e]` pairs with `ptm.deltas()[e]`.
    Local {
        ptm: &'p LocalPtm<W>,
        coords: Vec<u32>,
        /// Whether `deltas()[0]` is the identity delta, whose stream goes to the run's pre-sorted `id` columns.
        has_identity: bool,
        /// Whether the identity amplitude is nonzero on every active support pattern, so the merge borrows the id keys from the source bucket (ARCHITECTURE.md §Engine).
        dense_identity: bool,
        /// Whether this layer's runs use the radix sort kernel, decided once per layer (see `RADIX_MIN_REST_STREAMS`).
        radix_sort: bool,
    },
    /// Wide rotation: two implicit entries, the identity pass and the generator pass.
    Rotation {
        rotation: &'p PreparedRotation<W>,
        coord_identity: u32,
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
}

impl Default for LayerKnobs<'_> {
    fn default() -> Self {
        Self {
            bucket_deltas: None,
            rest_streams: None,
            rows_per_key: None,
            generator_local: true,
        }
    }
}

/// Mean rest rows per output key that gets any, the radix gate's second arm; ask it of the channel's own PTM, never a restricted one.
pub(super) fn rest_rows_per_key<const W: usize>(ptm: &LocalPtm<W>) -> f64 {
    let rest_start = usize::from(ptm.deltas().first().is_some_and(|d| d.local_delta == 0));
    let dim = 1usize << (2 * ptm.k());
    let (mut rows, mut keys) = (0u32, 0u32);
    for output in 0..dim {
        let rows_here = ptm.deltas()[rest_start..]
            .iter()
            .filter(|delta| delta.amplitude[output ^ delta.local_delta as usize] != ZERO)
            .count() as u32;
        rows += rows_here;
        keys += u32::from(rows_here > 0);
    }
    if keys == 0 {
        0.0
    } else {
        f64::from(rows) / f64::from(keys)
    }
}

impl<'p, const W: usize> DeltaPlan<'p, W> {
    pub(super) fn new(prepared: &'p Prepared<W>, span: &Gf2Span, knobs: LayerKnobs<'_>) -> Self {
        match prepared {
            Prepared::Local(ptm) => {
                let coords: Vec<u32> = ptm
                    .deltas()
                    .iter()
                    .map(|d| span.coord_of(d.bucket_delta))
                    .collect();
                let has_identity = ptm.deltas().first().is_some_and(|d| d.local_delta == 0);
                debug_assert!(!has_identity || coords[0] == 0);
                // `amplitude` is sized LOCAL_DIM but only `4^k` entries are populated.
                let dim = 1usize << (2 * ptm.k());
                let dense_identity =
                    has_identity && ptm.deltas()[0].amplitude[..dim].iter().all(|a| *a != ZERO);
                let rest_streams = knobs
                    .rest_streams
                    .unwrap_or(ptm.deltas().len() - has_identity as usize);
                let radix_sort = rest_streams >= RADIX_MIN_REST_STREAMS
                    || (rest_streams >= RADIX_MIN_DISJOINT_STREAMS
                        && knobs.rows_per_key.unwrap_or_else(|| rest_rows_per_key(ptm))
                            < RADIX_MAX_REST_ROWS_PER_KEY);
                DeltaPlan::Local {
                    ptm,
                    coords,
                    has_identity,
                    dense_identity,
                    radix_sort,
                }
            }
            Prepared::Rotation(rotation) => DeltaPlan::Rotation {
                rotation,
                coord_identity: span.coord_of(rotation.bucket_delta_identity),
                // `coord_of` demands its argument be in the span, and a remote generator's bucket delta is not.
                generator_coordinate: if knobs.generator_local {
                    span.coord_of(rotation.bucket_delta_generator)
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
    #[inline]
    fn count(&self, beta: u32) -> usize {
        let _ = beta;
        0
    }

    /// Append original bucket `beta`'s rows onto the run's rest columns.
    #[inline]
    fn append_into(
        &self,
        beta: u32,
        x: &mut Vec<[u64; W]>,
        z: &mut Vec<[u64; W]>,
        c: &mut Vec<Complex64>,
    ) {
        let _ = (beta, x, z, c);
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
    #[cfg(feature = "phase-timing")]
    let mut stamp = Stamp::now();

    // A partitioned layer's local table can look key-preserving while received rows still need merging.
    if let Prepared::Local(ptm) = prepared {
        if !X::NEEDS_BETA && ptm.is_key_preserving() {
            rescale_in_place(sum, ptm, policy);
            #[cfg(feature = "phase-timing")]
            stamp.lap(&mut scratch.stats.rescale_ns);
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
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut scratch.stats.span_plan_ns);

    // Coset `c` owns `staging[c·2^r .. (c+1)·2^r]`; at `r = 0` the permutation is the identity and the handle passes are skipped.
    let identity_perm = span.r() == 0;
    if !identity_perm {
        let buckets = sum.buckets_mut();
        scratch.permutation.clear();
        scratch
            .permutation
            .extend((0..buckets.len() as u32).map(|beta| span.permuted_index(beta)));
        scratch
            .staging
            .resize_with(buckets.len(), BucketColumns::default);
        for (beta, columns) in buckets.iter_mut().enumerate() {
            scratch.staging[scratch.permutation[beta] as usize] = std::mem::take(columns);
        }
    }
    if X::NEEDS_BETA {
        scratch.inverse_permutation.clear();
        if !identity_perm {
            scratch
                .inverse_permutation
                .resize(scratch.permutation.len(), 0);
            for (beta, &position) in scratch.permutation.iter().enumerate() {
                scratch.inverse_permutation[position as usize] = beta as u32;
            }
        }
    }
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut scratch.stats.permute_ns);

    // Each coset reads and writes only its own chunk (ARCHITECTURE.md §Parallelism).
    {
        if num_cosets >= MIN_COSETS_FOR_PARALLEL {
            let pool = rayon::current_num_threads().max(1);
            if scratch.workers.len() < pool {
                scratch.workers.resize_with(pool, Mutex::default);
            }
        }
        let workers = &scratch.workers;
        let inverse_permutation: &[u32] = &scratch.inverse_permutation;
        let chunks: &mut [BucketColumns<W>] = if identity_perm {
            sum.buckets_mut()
        } else {
            scratch.staging.as_mut_slice()
        };
        if num_cosets < MIN_COSETS_FOR_PARALLEL {
            for (coset_index, chunk) in chunks.chunks_mut(coset_size).enumerate() {
                let base = coset_index * coset_size;
                fill_coset::<W, T, X>(
                    chunk,
                    &plan,
                    policy,
                    &mut scratch.task,
                    extra,
                    base,
                    inverse_permutation,
                );
            }
        } else {
            chunks
                .par_chunks_mut(coset_size)
                .enumerate()
                .for_each(|(coset_index, chunk)| {
                    let base = coset_index * coset_size;
                    // The fresh-scratch arm is a defensive fallback; a pool worker always has an index.
                    match rayon::current_thread_index() {
                        Some(worker) if worker < workers.len() => {
                            let mut worker_scratch = workers[worker].lock().unwrap();
                            fill_coset::<W, T, X>(
                                chunk,
                                &plan,
                                policy,
                                &mut worker_scratch,
                                extra,
                                base,
                                inverse_permutation,
                            );
                        }
                        _ => {
                            let mut worker_scratch = CosetScratch::<W>::default();
                            fill_coset::<W, T, X>(
                                chunk,
                                &plan,
                                policy,
                                &mut worker_scratch,
                                extra,
                                base,
                                inverse_permutation,
                            );
                        }
                    }
                });
        }
    }
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut scratch.stats.coset_loop_ns);

    if !identity_perm {
        let buckets = sum.buckets_mut();
        for (beta, columns) in buckets.iter_mut().enumerate() {
            *columns = std::mem::take(&mut scratch.staging[scratch.permutation[beta] as usize]);
        }
    }
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut scratch.stats.unpermute_ns);
    sum.recount();
    #[cfg(feature = "phase-timing")]
    stamp.lap(&mut scratch.stats.recount_ns);

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

/// In-place coefficient rescale for a key-preserving channel, with the general path's zero drop and `keep_term`.
fn rescale_in_place<const W: usize, T>(sum: &mut PauliSum<W>, ptm: &LocalPtm<W>, policy: &T)
where
    T: TruncationPolicy<W> + ?Sized,
{
    let amplitude = &ptm.deltas()[0].amplitude;
    sum.buckets_mut().par_iter_mut().for_each(|columns| {
        let len = columns.len();
        let mut keep = 0usize;
        for i in 0..len {
            let pattern = ptm.support_bits(&columns.x[i], &columns.z[i]);
            let coeff = columns.coeff[i] * amplitude[pattern];
            if coeff == ZERO || !policy.keep_term(&columns.x[i], &columns.z[i], coeff) {
                continue;
            }
            columns.x[keep] = columns.x[i];
            columns.z[keep] = columns.z[i];
            columns.coeff[keep] = coeff;
            keep += 1;
        }
        columns.x.truncate(keep);
        columns.z.truncate(keep);
        columns.coeff.truncate(keep);
    });
    sum.recount();

    #[cfg(debug_assertions)]
    sum.assert_invariants();
}

#[cfg(test)]
mod tests;
