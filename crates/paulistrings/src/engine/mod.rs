//! The propagation engine: front door [`propagate`], layer loop in
//! [`bucketed`].
//!
//! One layer is a coset walk over the GF(2)-linear bucket partition (the
//! crate-private `coset` module), with the per-run sort and fused merge kernels
//! in the crate-private `merge` module. See ARCHITECTURE.md §Engine for the
//! layer and this module's propagation loop.

pub mod bucketed;
pub(crate) mod coset;
pub(crate) mod merge;
#[cfg(feature = "phase-timing")]
pub mod stats;

use crate::bucket::sum::DEFAULT_TARGET_BUCKET_LEN;
use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;
use bucketed::{apply_layer_bucketed, LayerScratch};

/// `log` target for the engine's progress events, so a consumer can filter
/// them without touching the rest of the crate's (currently empty) logging.
/// See the "Progress logging" section on [`propagate`].
const LOG_TARGET: &str = "paulistrings::propagate";

/// Propagation direction.
///
/// [`Direction::Forward`] applies channels in order; [`Direction::Heisenberg`]
/// iterates in reverse and applies adjoints (for backpropagating
/// observables).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Apply channels in the order they were pushed onto the [`Circuit`].
    Forward,
    /// Apply channels in reverse order, using each channel's
    /// [`Channel::apply_adjoint`].
    Heisenberg,
}

/// Propagate `initial` through `circuit` under `policy`.
///
/// Iterates the circuit's channels — in order for [`Direction::Forward`], in
/// reverse for [`Direction::Heisenberg`], calling [`Channel::apply_adjoint`]
/// in the latter case (default = self-adjoint; overridden on
/// [`PauliRotation`](crate::channel::PauliRotation) and
/// [`Clifford1Q`](crate::channel::Clifford1Q)).
///
/// The sum is propagated in its bucketed form throughout — there is no
/// conversion at either end, so calling this repeatedly on the same sum (a
/// Trotter driver stepping an observable) costs nothing beyond the layers
/// themselves; per-bucket storage capacity is retained inside the returned
/// sum across calls. The bucket count is re-normalized against
/// [`desired_bits`](crate::bucket::desired_bits) before every layer.
///
/// # Progress logging
///
/// Progress is reported through the [`log`] facade under the target
/// `paulistrings::propagate`: one `INFO` line on entry (term count, channel
/// count, direction), one `INFO` line on exit (layers applied, terms in → out,
/// elapsed seconds), and one `DEBUG` line per layer (`layer k/n [name]:
/// before -> after terms, ms`). Per-layer is `DEBUG` rather than `INFO`
/// because a Trotter driver calls this hundreds of times. The layer name comes
/// from [`Channel::debug_name`].
///
/// With no logger installed each site costs one relaxed atomic load and
/// allocates nothing — in particular the per-layer clock read is itself behind
/// the level check. To see the lines, install any `log` implementation, e.g.
/// `env_logger::init()` in `main` plus
/// `RUST_LOG=paulistrings=debug` in the environment.
///
/// # Examples
///
/// ```
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, PauliString, Phase, TruncationPolicy,
///     channel::Clifford1Q, propagate,
/// };
/// use num_complex::Complex64;
///
/// let mut acc = BuildAccumulator::<1>::new(1);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
/// let observable = acc.finalize();
///
/// let mut circuit = Circuit::<1>::new(1);
/// circuit.push(Clifford1Q::h(0));
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {}
///
/// // H conjugates Z to X, so propagating Z₀ through H gives X₀.
/// let evolved = propagate(&circuit, observable, &KeepAll, Direction::Heisenberg);
/// assert_eq!(evolved.len(), 1);
/// let (x, z, _c) = evolved.bucket(0);
/// assert_eq!(x[0], [1]);
/// assert_eq!(z[0], [0]);
/// ```
pub fn propagate<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let mut scratch = LayerScratch::<W>::new();
    propagate_with_scratch(circuit, sum, policy, direction, &mut scratch)
}

/// [`propagate`] with a caller-held [`LayerScratch`].
///
/// Behaves identically to [`propagate`] — it *is* the implementation — but
/// lets the caller retain the scratch's high-water buffer capacity across
/// calls (a Trotter driver stepping an observable), and, under the
/// `phase-timing` feature, read the per-phase counters afterwards through
/// `LayerScratch::take_stats` (the method — and so a resolvable doc link to
/// it — exists only when that feature is enabled).
///
/// It is also the only entry point that records a
/// [`TermTrace`](bucketed::TermTrace) — the always-compiled, opt-in per-layer
/// term counts. Call [`LayerScratch::enable_term_trace`] before propagating
/// and [`LayerScratch::take_term_trace`] afterwards; [`propagate`], which owns
/// its scratch, never enables it.
///
/// # Progress logging
///
/// This is where the events described under [`propagate`] are emitted: target
/// `paulistrings::propagate`, `INFO` on entry and exit, `DEBUG` per layer.
/// Every event is emitted on the calling thread, outside the engine's Rayon
/// region, so a logger implementation never runs inside a parallel layer.
pub fn propagate_with_scratch<const W: usize, T>(
    circuit: &Circuit<W>,
    mut sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut LayerScratch<W>,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let n = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    // The bucket count `B` is a deterministic function of the *history* of
    // term counts (`rebucket` is grow-only, so `B` is the running max of
    // `desired_bits` over every layer so far, not the instantaneous length),
    // which is itself fully determined by the circuit and the starting sum.
    // Bitwise independence of the engine's output from `B` is tested, but
    // agreement is only required to floating-point tolerance
    // (ARCHITECTURE.md §Determinism).
    let min_buckets = default_min_buckets();

    // Entry/exit INFO pair. One unconditional `Instant` pair per `propagate`
    // call is negligible next to a single layer; the *per-layer* clock reads
    // below are the ones that have to be gated.
    let terms_in = sum.len();
    let started = std::time::Instant::now();
    log::info!(
        target: LOG_TARGET,
        "propagate: {terms_in} terms through {n} channels ({direction:?})",
    );

    // Hoisted out of the layer loop: nothing inside it can enable or disable
    // the trace, so the per-layer test is on a register rather than a load
    // through `scratch`.
    let tracing = scratch.term_trace.is_some();

    for k in 0..n {
        let idx = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();

        // Per-layer DEBUG progress. `log_enabled!` is a relaxed atomic load
        // plus a compare, so a disabled logger costs one branch per layer and
        // never reads the clock; `terms_before` is the cached length field.
        let layer_t0 =
            log::log_enabled!(target: LOG_TARGET, log::Level::Debug).then(std::time::Instant::now);
        let terms_before = sum.len();

        #[cfg(feature = "phase-timing")]
        let mut st = stats::Stamp::now();
        #[cfg(feature = "phase-timing")]
        {
            scratch.stats.layers += 1;
            scratch.stats.terms_in += sum.len() as u64;
        }

        sum.rebucket(DEFAULT_TARGET_BUCKET_LEN, min_buckets);
        #[cfg(feature = "phase-timing")]
        st.lap(&mut scratch.stats.rebucket_ns);

        let prep = ch.prepare(sum.hash(), adjoint);
        #[cfg(feature = "phase-timing")]
        st.lap(&mut scratch.stats.prepare_ns);

        match prep {
            Some(prep) => {
                apply_layer_bucketed(&mut sum, &prep, policy, scratch);
                #[cfg(feature = "phase-timing")]
                st.rearm();
            }
            None => {
                // No built-in channel can reach this arm: every one of them has
                // support ≤ MAX_LOCAL_SUPPORT except `PauliRotation`, which
                // overrides `prepare` and returns `Prepared::Rotation` at any
                // generator weight, and none writes outside its declared
                // support. So this is only reachable from a user-supplied
                // `Channel` impl, and it is a hard error rather than a slow
                // path: there is no whole-sum fallback to absorb it.
                let weight: u32 = ch.support().iter().map(|w| w.count_ones()).sum();
                panic!(
                    "layer {idx}: Channel::prepare declined, so this channel cannot be \
                     propagated. The engine tabulates channels of support ≤ \
                     {MAX_LOCAL_SUPPORT} qubits (this one declares {weight}), and a \
                     channel must not write outside its declared support. See \
                     research/notes/2026-08-31-local-ptm-generalization.md",
                );
            }
        }

        policy.finalize_layer(&mut sum);
        #[cfg(feature = "phase-timing")]
        {
            st.lap(&mut scratch.stats.finalize_ns);
            scratch.stats.terms_out += sum.len() as u64;
        }
        // Opt-in term trace (`LayerScratch::enable_term_trace`), both counts
        // in one call: `terms_before` from the read above, `sum.len()`
        // post-truncation, so the pair says what the layer actually left
        // resident. Placed last so it is outside every measured phase, and
        // behind a hoisted flag + a `#[cold]` callee so the untraced layer
        // body grows by a register test and a not-taken branch — this loop
        // inlines `apply_layer_bucketed`, whose merge kernels are sensitive
        // to a few bytes of code motion (CLAUDE.md §Performance discipline).
        if tracing {
            record_layer_terms(scratch, terms_before, sum.len());
        }

        if let Some(t0) = layer_t0 {
            log::debug!(
                target: LOG_TARGET,
                "layer {}/{} [{}]: {} -> {} terms, {:.1} ms",
                k + 1,
                n,
                ch.debug_name(),
                terms_before,
                sum.len(),
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
    }

    log::info!(
        target: LOG_TARGET,
        "propagate: {} layers applied, {} -> {} terms, {:.3} s",
        n,
        terms_in,
        sum.len(),
        started.elapsed().as_secs_f64(),
    );

    sum
}

/// Append one layer's `(terms_in, terms_out)` to the scratch's trace.
///
/// Deliberately `#[cold]` + `#[inline(never)]`: the layer loop is the caller
/// that inlines [`apply_layer_bucketed`] and, through it, the merge kernels,
/// whose measured throughput moves by 6–34% under a few bytes of code motion
/// (CLAUDE.md §Performance discipline; the first A/B of this feature with the
/// two `Vec::push`es inline in the loop body measured a direction-consistent
/// +7% single-thread regression with `merge_ns` +20%). Keeping the pushes
/// out-of-line and marked cold puts the whole trace off the layer's code
/// path; the tracing caller pays a call, which is nothing next to a layer.
#[cold]
#[inline(never)]
fn record_layer_terms<const W: usize>(
    scratch: &mut LayerScratch<W>,
    terms_in: usize,
    terms_out: usize,
) {
    if let Some(trace) = scratch.term_trace.as_mut() {
        trace.terms_in.push(terms_in);
        trace.terms_out.push(terms_out);
    }
}

/// Bucket-count floor: enough buckets that Rayon has slack to load-balance.
///
/// Fixed, not derived from `rayon::current_num_threads`: see
/// [`crate::bucket::sum::DEFAULT_MIN_BUCKETS`] for why a thread-independent
/// floor is what we want here (ARCHITECTURE.md §Bucket-Policy). Combined
/// with the grow-only `rebucket` policy, this floor is a lower bound on `B`
/// throughout a `propagate` call, not just at the layer where it was first
/// crossed.
pub fn default_min_buckets() -> usize {
    crate::bucket::sum::DEFAULT_MIN_BUCKETS
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use crate::accumulator::BuildAccumulator;
    use crate::channel::{support_mask, Channel, OutputBuffer};
    use crate::circuit::Circuit;
    use crate::pauli_string::PauliString;
    use crate::phase::Phase;
    use crate::truncation::TruncationPolicy;

    use super::{propagate, Direction};

    struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    /// Support on three qubits, so `Prepared::derive_local` bails on the
    /// popcount check and `prepare` returns `None`. Cribbed from
    /// `channel::prepared::tests::derive_local_rejects_popcount_gt_2`.
    struct ThreeQubits;
    impl<const W: usize> Channel<W> for ThreeQubits {
        fn max_fanout(&self) -> usize {
            1
        }
        fn support(&self) -> [u64; W] {
            support_mask(&[0, 1, 2])
        }
        fn apply(
            &self,
            input_x: &[u64; W],
            input_z: &[u64; W],
            coeff: Complex64,
            out: &mut OutputBuffer<'_, W>,
        ) {
            out.push(*input_x, *input_z, coeff);
        }
    }

    /// A channel that declines `prepare` is a hard error: there is no
    /// whole-sum fallback to absorb it, so `propagate` panics with the
    /// support weight and a pointer to the generalization note.
    #[test]
    #[should_panic(expected = "Channel::prepare declined")]
    fn an_unpreparable_channel_panics() {
        let mut acc = BuildAccumulator::<1>::with_capacity(8, 1);
        acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
        let sum = acc.finalize();

        let mut circuit = Circuit::<1>::new(8);
        circuit.push(ThreeQubits);
        let _ = propagate(&circuit, sum, &AlwaysKeep, Direction::Forward);
    }
}
