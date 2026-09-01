//! The propagation engine: front door [`propagate`], layer loop in
//! [`bucketed`].
//!
//! One layer is a coset walk over the GF(2)-linear bucket partition (the
//! crate-private `coset` module), with the per-run sort and fused merge kernels
//! in the crate-private `merge` module. See ARCHITECTURE.md §Engine for the
//! layer and this module's propagation loop.
//!
//! The crate-private `direct` module is an **additive** second layer path for
//! sums small enough that the bucketed layer's per-layer fixed cost dominates:
//! off unless a caller asks for it through [`propagate_with_options`], never
//! reached by [`propagate`], and canonical for nothing. See
//! `research/notes/2026-09-01-small-m-path.md`.

pub mod bucketed;
pub(crate) mod coset;
pub(crate) mod direct;
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

/// Which layer engine [`propagate_with_options`] uses.
///
/// The bucketed sorting engine ([`bucketed`]) is canonical at every term count;
/// the alternative is a strictly additive small-sum path (`engine::direct`) that
/// applies a layer through [`Channel::apply`] into a hash map, skipping
/// [`Channel::prepare`] and the bucketed machinery. See
/// `research/notes/2026-09-01-small-m-path.md`.
///
/// The default is [`EngineSelection::SortedOnly`]: today's behaviour, unchanged,
/// for every caller that does not ask for anything else.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngineSelection {
    /// The bucketed sorting engine for every layer. The default.
    #[default]
    SortedOnly,
    /// The small-sum direct path while it is expected to be faster — the sum is
    /// at most [`PropagateOptions::small_sum_threshold`] terms *and* the policy
    /// reports no layer finalization
    /// ([`TruncationPolicy::finalizes_layer`]) — then the sorting engine for
    /// the rest of the circuit.
    ///
    /// The policy condition is a performance decision, not a correctness one: a
    /// finalizing policy would make the direct path pay a materialize →
    /// finalize → re-ingest round trip per layer, which is what the direct path
    /// exists to avoid. Use [`EngineSelection::SmallSumDirect`] to take the
    /// direct path anyway.
    Auto,
    /// The small-sum direct path whenever the sum is at most
    /// [`PropagateOptions::small_sum_threshold`] terms, whatever the policy.
    ///
    /// A policy with a layer pass still gets it, once per layer, on a
    /// materialized sum — the results match [`EngineSelection::SortedOnly`] to
    /// floating-point tolerance either way. Mainly an A/B knob: it is the only
    /// way to measure the direct path against a `TopN`-style policy.
    SmallSumDirect,
}

/// Default for [`PropagateOptions::small_sum_threshold`].
///
/// Two costs move in opposite directions with the term count. The direct path
/// saves the bucketed layer's **fixed** cost — 1.43 µs for a two-qubit rotation
/// at `W = 2`, 5.4 µs for a dense two-qubit PTM
/// (`research/notes/2026-09-01-large-m-phase-breakdown.md` §2) — and pays a
/// *per-term* cost that rises as its map leaves cache, where the sorting engine's
/// is flat in `m` to ±10% over three decades (same sheet, §1). So there is a
/// crossover, and it is workload-dependent: measured on the head-to-head study's
/// own circuits (`examples/small_m_ab.rs`) it is **≈ 1.5 × 10²** resident terms
/// for kicked-Ising and **≈ 2 × 10³** for XXZ — a 14× spread, in keeping with the
/// 4.4–21× spread of the study's own cross-engine crossovers. A threshold also
/// acts through a second channel: setting it above a workload's *peak* keeps the
/// whole run on one path, and being undivided is itself worth something.
///
/// 2048 is the largest value in a measured `{128, 512, 1024, 2048, 4096}` sweep
/// at which **no** configuration regresses — every cell is either a
/// sign-consistent win or a sign-inconsistent null — and the only one that also
/// keeps XXZ's 1 625-peak configuration on the direct path end to end, which is
/// worth **1.68×** there. 4096 is the cliff: two configurations turn into
/// sign-consistent regressions (kicked-Ising 2⁻⁸ **0.68×**, XXZ 1e-4 0.93×).
///
/// The two configurations where the study loses worst — kicked-Ising 2⁻⁴ (ratio
/// 0.323) and XXZ 1e-2 (0.460) — peak at 68 and 164 terms, so they are *entirely
/// insensitive* to this constant across that whole sweep (2.28–2.36× and
/// 1.48–1.58×). What the constant trades is the middle of the range, and the
/// trade is asymmetric: 2048 gives up ~6 points on kicked-Ising 2⁻⁶ (1.08× at
/// 512 becomes a 1.02× null) to gain 0.64 on XXZ 1e-3. Full table:
/// `research/notes/2026-09-01-small-m-path.md` §5.
///
/// It also sits below `desired_bits`'s `worth_splitting` floor
/// (`DEFAULT_MIN_BUCKETS × MIN_TERMS_PER_TASK = 8192`), so a sum on this path is
/// one the sorting engine would have run in few buckets anyway.
pub const DEFAULT_SMALL_SUM_THRESHOLD: usize = 2048;

/// Tuning knobs for [`propagate_with_options`].
///
/// [`Default`] is exactly today's behaviour — the sorting engine for every
/// layer — so `PropagateOptions::default()` and [`propagate`] agree bit for bit.
///
/// # Examples
///
/// ```
/// use paulistrings::{EngineSelection, PropagateOptions};
///
/// let opts = PropagateOptions {
///     engine: EngineSelection::Auto,
///     ..PropagateOptions::default()
/// };
/// assert_eq!(PropagateOptions::default().engine, EngineSelection::SortedOnly);
/// # let _ = opts;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropagateOptions {
    /// Which layer engine to use. Default [`EngineSelection::SortedOnly`].
    pub engine: EngineSelection,
    /// Resident term count up to which the small-sum direct path is used.
    /// Ignored under [`EngineSelection::SortedOnly`]. Default
    /// [`DEFAULT_SMALL_SUM_THRESHOLD`].
    pub small_sum_threshold: usize,
}

impl Default for PropagateOptions {
    fn default() -> Self {
        Self {
            engine: EngineSelection::SortedOnly,
            small_sum_threshold: DEFAULT_SMALL_SUM_THRESHOLD,
        }
    }
}

impl PropagateOptions {
    /// Whether a propagation starting at `len` terms under a policy that
    /// answers `policy_finalizes` to [`TruncationPolicy::finalizes_layer`]
    /// starts on the direct path.
    ///
    /// Evaluated once per [`propagate_with_options`] call, outside the layer
    /// loop. The direct path is entered only here: once the sum outgrows the
    /// threshold the run continues on the sorting engine and never comes back
    /// (see [`propagate_with_options`]).
    fn starts_direct(&self, len: usize, policy_finalizes: bool) -> bool {
        match self.engine {
            EngineSelection::SortedOnly => false,
            EngineSelection::Auto => len <= self.small_sum_threshold && !policy_finalizes,
            EngineSelection::SmallSumDirect => len <= self.small_sum_threshold,
        }
    }
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
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut LayerScratch<W>,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    propagate_with_scratch_and_options(
        circuit,
        sum,
        policy,
        direction,
        scratch,
        PropagateOptions::default(),
    )
}

/// [`propagate`] with [`PropagateOptions`].
///
/// `PropagateOptions::default()` is [`propagate`] exactly — same code path, same
/// bits — so this is only interesting for opting into a non-default
/// [`EngineSelection`].
///
/// # Examples
///
/// ```
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, EngineSelection, PauliString, Phase,
///     PropagateOptions, TruncationPolicy, channel::Clifford1Q, propagate_with_options,
/// };
/// use num_complex::Complex64;
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {
///     // The direct path skips the finalize round trip only for a policy that
///     // says it has no layer pass; the trait's default answer is the
///     // conservative `true`.
///     fn finalizes_layer(&self) -> bool { false }
/// }
///
/// let mut acc = BuildAccumulator::<1>::new(1);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
/// let mut circuit = Circuit::<1>::new(1);
/// circuit.push(Clifford1Q::h(0));
///
/// let opts = PropagateOptions {
///     engine: EngineSelection::Auto,
///     ..PropagateOptions::default()
/// };
/// let evolved = propagate_with_options(
///     &circuit, acc.finalize(), &KeepAll, Direction::Heisenberg, opts,
/// );
/// // H conjugates Z to X, whichever engine ran the layer.
/// assert_eq!(evolved.get(&[1], &[0]), Some(Complex64::new(1.0, 0.0)));
/// ```
pub fn propagate_with_options<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let mut scratch = LayerScratch::<W>::new();
    propagate_with_scratch_and_options(circuit, sum, policy, direction, &mut scratch, options)
}

/// [`propagate_with_scratch`] with [`PropagateOptions`] — the implementation
/// every other entry point delegates to.
///
/// # The small-sum path, when selected
///
/// Under [`EngineSelection::Auto`] or [`EngineSelection::SmallSumDirect`] and a
/// starting sum within [`PropagateOptions::small_sum_threshold`], the leading
/// layers run on the direct path (`engine::direct`): the sum is held in a hash
/// map across those layers, and materialized back into a [`PauliSum`] once —
/// when a layer leaves it above the threshold, or when the circuit ends. From
/// that point the sorting engine runs every remaining layer.
///
/// The transition is **one-way**. Re-entering the direct path when a later
/// truncation drops the sum back under the threshold is deliberately not done:
/// each crossing costs an `O(n)` ingest plus an `O(n log n)` materialize, a sum
/// oscillating around the threshold would pay them per layer, and the upside is
/// bounded by the fixed cost the direct path saves (1.43 µs/layer for a
/// two-qubit rotation). One crossing per call is also trivially reasoned about:
/// the per-layer term counts and the [`TermTrace`](bucketed::TermTrace) are the
/// same records in the same order regardless of where it happened.
///
/// Truncation is applied identically on both sides of the transition:
/// `keep_term` runs per layer on summed coefficients (where the merge phase runs
/// it), and `finalize_layer` runs per layer on a materialized sum whenever
/// [`TruncationPolicy::finalizes_layer`] says there is one. Progress logging and
/// the term trace emit the same records from either path.
///
/// # Channels wider than the sorting engine can prepare
///
/// The direct path calls only [`Channel::apply`], so it applies a channel of any
/// support width — including the `> MAX_LOCAL_SUPPORT` channels for which
/// `Channel::prepare` returns `None` and the sorting engine panics. It is a
/// wider path, not a narrower one, and that asymmetry is visible: a circuit
/// containing such a channel propagates while the sum is under the threshold and
/// panics on the layer after it grows past it, exactly as it panics today under
/// [`EngineSelection::SortedOnly`]. The generalization design for the sorting
/// engine is `research/notes/2026-08-31-local-ptm-generalization.md`.
pub fn propagate_with_scratch_and_options<const W: usize, T>(
    circuit: &Circuit<W>,
    mut sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut LayerScratch<W>,
    options: PropagateOptions,
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

    // The engine choice is made once, here, outside every loop: a layer's own
    // code path cannot change it, and under the default `SortedOnly` this is
    // one not-taken branch before the loop and nothing inside it. `n > 0`
    // keeps a zero-layer call off the direct path entirely, so it stays a
    // no-op rather than an ingest/materialize round trip.
    let mut start = 0usize;
    if n > 0 && options.starts_direct(terms_in, policy.finalizes_layer()) {
        let (out, applied) =
            direct::run_direct_prefix(circuit, sum, policy, direction, scratch, options);
        sum = out;
        start = applied;
    }

    for k in start..n {
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
