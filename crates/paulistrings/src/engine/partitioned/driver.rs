//! The partitioned propagation driver: [`PartitionedSum`] and the
//! [`propagate_partitioned`] front doors.
//!
//! A [`PartitionedSum`] is one [`PauliSum`] split across the partitions of a
//! [`PartitionRuntime`] — the terms with `rows.partition_of(v) == r` live in
//! partition `r` and nowhere else — plus the per-partition scratch that
//! survives between calls. The layer loop mirrors the unpartitioned one in
//! [`engine`](crate::engine): rebucket → prepare → layer → finalize, once per
//! channel, on every partition in lock-step.
//!
//! Three things differ from the unpartitioned loop, all of them collective:
//!
//! 1. **The bucket count is agreed, not computed.** Every partition proposes
//!    `desired_bits` for its own share (so `P` partitions of `n/P` terms have
//!    the same *total* bucket count as one partition of `n`), the group takes
//!    the maximum, and each partition refines to it. Equal bucket counts are
//!    what the exchange's CSR block index and its `β ^ bd` receive rule assume
//!    ([`transport`](super::transport)), and `refine` is grow-only, so the
//!    count never falls mid-run.
//! 2. **The layer may exchange rows.** [`apply_layer_partitioned`] does that
//!    itself, including the "no remote delta ⇒ no transport call" case; the
//!    bits agreement above is the one *unconditional* collective per layer.
//! 3. **The layer finalization is collective**, so the policy bound is
//!    [`PartitionedTruncation`] and its
//!    [`finalize_layer_partitioned`](PartitionedTruncation::finalize_layer_partitioned)
//!    runs on **every** layer, on every partition, whatever
//!    [`finalizes_layer`](crate::TruncationPolicy::finalizes_layer) says.
//!
//! [`PropagateOptions`] is reused unchanged, with one exception:
//! [`EngineSelection`](crate::EngineSelection) is **ignored**. The partitioned
//! path is always the bucketed layer — the small-sum direct path holds its
//! terms in a hash map with no bucket structure for an exchange to index, and
//! a sum small enough to want it is a sum too small to partition.

use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;

use super::layer::{apply_layer_partitioned, PartitionState};
use super::runtime::PartitionRuntime;
use super::topology::{PartitionConfig, TopologyError};
use super::transport::{Collectives, InProcessTransport};
use super::truncation::PartitionedTruncation;
use crate::bucket::hash::PartitionRows;
use crate::bucket::sum::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::channel::prepared::MAX_LOCAL_SUPPORT;
use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::{PauliSum, ProductState};

/// `log` target for the partitioned engine's progress events — the same target
/// the unpartitioned [`propagate`](crate::propagate) uses, so one filter
/// covers both.
const LOG_TARGET: &str = "paulistrings::propagate";

/// One partition's payload, moved into its thread for the duration of a call
/// and handed back.
struct PartitionWork<const W: usize> {
    /// This partition's share of the sum.
    local: PauliSum<W>,
    /// Its layer and export scratch, retained across calls.
    state: PartitionState<W>,
}

/// A [`PauliSum`] split across the partitions of a [`PartitionRuntime`].
///
/// Held across calls: the split, the partition rows, the pools and the
/// per-partition scratch all persist, so a driver stepping an observable
/// through many Trotter steps scatters once and gathers once
/// ([`PartitionedSum::propagate`] per step).
///
/// # Invariants
///
/// Between calls — and asserted by [`PartitionedSum::assert_invariants`] —
/// every partition's sum satisfies [`PauliSum`]'s own invariants, holds only
/// keys of its own partition, and shares one hash family and bucket count with
/// its peers.
///
/// # Examples
///
/// ```
/// use paulistrings::channel::Clifford1Q;
/// use paulistrings::engine::partitioned::{
///     PartitionConfig, PartitionRuntime, PartitionedSum, Placement,
/// };
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, PartitionedTruncation, PauliString, Phase,
///     TruncationPolicy,
/// };
/// use num_complex::Complex64;
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {
///     // `finalizes_layer`'s default is the conservative `true`; the
///     // `PartitionedTruncation` default body rejects that, since it cannot
///     // know a layer pass it would be skipping is a no-op.
///     fn finalizes_layer(&self) -> bool { false }
/// }
/// impl<const W: usize> PartitionedTruncation<W> for KeepAll {}
///
/// let config = PartitionConfig {
///     placement: Placement::Unpinned { partitions: 2, threads_per_partition: Some(1) },
///     bind_memory: false,
///     partition_row_seed: None,
/// };
/// let runtime = PartitionRuntime::new(&config).expect("topology resolves");
///
/// let mut acc = BuildAccumulator::<1>::new(2);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
///
/// let mut circuit = Circuit::<1>::new(2);
/// circuit.push(Clifford1Q::h(0));
///
/// let mut split = PartitionedSum::scatter(acc.finalize(), runtime, &config);
/// split.propagate(&circuit, &KeepAll, Direction::Heisenberg);
/// // H conjugates Z to X, wherever the term happened to live.
/// assert_eq!(split.gather().get(&[1], &[0]), Some(Complex64::new(1.0, 0.0)));
/// ```
pub struct PartitionedSum<const W: usize> {
    /// One sum per partition, in rank order.
    locals: Vec<PauliSum<W>>,
    /// The rows that decide which partition a key belongs to.
    rows: PartitionRows<W>,
    /// The placement and pools this sum runs on.
    runtime: Arc<PartitionRuntime>,
    /// Per-partition layer/export scratch, retained across calls.
    states: Vec<PartitionState<W>>,
}

impl<const W: usize> PartitionedSum<W> {
    /// Splits `sum` across `runtime`'s partitions, deriving the partition rows
    /// from `config`.
    ///
    /// The rows come from
    /// [`PartitionRows::from_seed`] with `config.partition_row_seed`, falling
    /// back to the sum's own hash seed — a different draw from the bucket
    /// hash's, so the two are independent with high probability.
    ///
    /// Each partition filters its own share **on its own pool**, so the columns
    /// are first-touched in the domain that will read them.
    ///
    /// # Panics
    ///
    /// If `runtime`'s partition count and the derived rows disagree (they
    /// cannot, both being `log2(P)` rows), and in debug builds if the partition
    /// rows are not independent of the sum's hash rows.
    pub fn scatter(
        sum: PauliSum<W>,
        runtime: Arc<PartitionRuntime>,
        config: &PartitionConfig,
    ) -> Self {
        let seed = config
            .partition_row_seed
            .unwrap_or_else(|| sum.hash().seed());
        let rows = PartitionRows::<W>::from_seed(sum.num_qubits(), runtime.partition_bits(), seed);
        Self::scatter_with_rows(sum, rows, runtime)
    }

    /// Splits `sum` across `runtime`'s partitions using caller-supplied rows.
    ///
    /// # Bucket counts
    ///
    /// Each partition takes the bucket count its *own* share wants
    /// ([`desired_bits`] on the default bucket policy, all-reduced to a maximum
    /// so the group agrees), but sheds at most `log2(P)` bits of the count the
    /// unpartitioned sum arrived with — so the bucket count summed over
    /// partitions is the one the sum already had, and at `P = 1` the scatter
    /// changes nothing at all. The layer loop then re-normalizes upward against
    /// the caller's own [`PropagateOptions`].
    ///
    /// # Panics
    ///
    /// If `rows.num_partitions()` is not the runtime's partition count. In
    /// debug builds, if the rows are not independent of the sum's hash rows —
    /// a partition row inside the hash's row space correlates partition with
    /// bucket, which costs load balance (not correctness). The check is made
    /// here only: the hash gains rows as the sum grows, and independence from
    /// *future* rows cannot be checked up front. Random rows stay independent
    /// with high probability.
    pub fn scatter_with_rows(
        sum: PauliSum<W>,
        rows: PartitionRows<W>,
        runtime: Arc<PartitionRuntime>,
    ) -> Self {
        let size = runtime.num_partitions();
        assert_eq!(
            rows.num_partitions(),
            size,
            "partition rows name {} partitions but the runtime has {size}",
            rows.num_partitions(),
        );
        assert_eq!(
            rows.num_qubits(),
            sum.num_qubits(),
            "partition rows are for {} qubits, the sum for {}",
            rows.num_qubits(),
            sum.num_qubits(),
        );
        debug_assert!(
            rows.is_independent_of(sum.hash()),
            "partition rows are dependent on the bucket hash rows — the split \
             will correlate with the bucket partition and load-balance badly",
        );

        let pbits = rows.bits();
        let started = Instant::now();
        let locals = {
            let sum = &sum;
            let rows = &rows;
            runtime.map_partitions((0..size).collect(), |rank, _, transport| {
                let mut local = sum.filter_partition(rows, rank as u32);
                let want =
                    desired_bits(local.len(), DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS);
                let want = transport.allreduce_max_u8(want);
                local.coarsen_to(scatter_bits(local.hash().bits(), pbits, want));
                local
            })
        };
        log::info!(
            target: LOG_TARGET,
            "scatter: {} terms over {size} partitions, {} bucket bits, {:.3} s",
            sum.len(),
            locals[0].hash().bits(),
            started.elapsed().as_secs_f64(),
        );

        let states = (0..size).map(|_| PartitionState::default()).collect();
        Self {
            locals,
            rows,
            runtime,
            states,
        }
    }

    /// Propagates through `circuit` under `policy`, in place.
    ///
    /// [`PropagateOptions::default()`] — see
    /// [`propagate_with_options`](Self::propagate_with_options) for the
    /// non-default knobs and for what the loop does per layer.
    pub fn propagate<T>(&mut self, circuit: &Circuit<W>, policy: &T, direction: Direction)
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with_options(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagates through `circuit` under `policy` with explicit
    /// [`PropagateOptions`].
    ///
    /// `direction` means what it means in [`propagate`](crate::propagate):
    /// [`Direction::Forward`] applies the channels in order,
    /// [`Direction::Heisenberg`] in reverse through
    /// [`Channel::apply_adjoint`](crate::Channel::apply_adjoint).
    /// [`EngineSelection`](crate::EngineSelection) is ignored — the partitioned
    /// path is always the bucketed layer (see the module docs).
    ///
    /// # Progress logging
    ///
    /// Target `paulistrings::propagate`, as in the unpartitioned engine: one
    /// `INFO` line on entry and exit, on the calling thread, and one `DEBUG`
    /// line per layer **per partition**, tagged `partition r/P`. Unlike
    /// [`propagate_with_scratch`](crate::propagate_with_scratch) the per-layer
    /// lines are *not* emitted on the calling thread — they come from the
    /// partition's own driving thread, between layers. That thread is inside
    /// its partition's pool (`ThreadPool::install`) but not inside a parallel
    /// region, so a logger implementation still never runs inside a layer;
    /// with `P` partitions it does run on `P` threads at once. Every site is
    /// behind `log_enabled!`, so a disabled logger reads no clock.
    ///
    /// # Panics
    ///
    /// If a channel's [`Channel::prepare`] declines (support wider than
    /// `MAX_LOCAL_SUPPORT`), exactly as the unpartitioned engine does — there
    /// is no fallback path. If `policy` reports
    /// [`finalizes_layer`](crate::TruncationPolicy::finalizes_layer) without
    /// overriding
    /// [`finalize_layer_partitioned`](PartitionedTruncation::finalize_layer_partitioned).
    /// A panic in any partition is re-raised on the calling thread.
    pub fn propagate_with_options<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
    {
        let n = circuit.channels.len();
        let size = self.num_partitions();
        let terms_in = self.len();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate_partitioned: {terms_in} terms through {n} channels ({direction:?}) \
             on {size} partitions [{}]",
            self.runtime.placement_summary(),
        );

        if n > 0 {
            let num_qubits = self.num_qubits();
            let items: Vec<PartitionWork<W>> = (0..size)
                .map(|rank| {
                    // The placeholder keeps `self` self-consistent (same rows,
                    // same hash, same bucket count on every partition) if a
                    // partition panics and the work is never handed back.
                    let hash = self.locals[rank].hash().clone();
                    PartitionWork {
                        local: std::mem::replace(
                            &mut self.locals[rank],
                            PauliSum::empty_with_hash(num_qubits, hash),
                        ),
                        state: std::mem::take(&mut self.states[rank]),
                    }
                })
                .collect();

            let runtime = Arc::clone(&self.runtime);
            let rows = &self.rows;
            let done = runtime.map_partitions(items, |rank, mut work, transport| {
                run_layers(
                    circuit, policy, direction, options, rows, rank, size, &mut work, transport,
                );
                work
            });
            for (rank, work) in done.into_iter().enumerate() {
                self.locals[rank] = work.local;
                self.states[rank] = work.state;
            }
        }

        log::info!(
            target: LOG_TARGET,
            "propagate_partitioned: {n} layers applied, {terms_in} -> {} terms, {:.3} s",
            self.len(),
            started.elapsed().as_secs_f64(),
        );
    }

    /// Merges the partitions back into one sum, leaving `self` intact.
    pub fn gather(&self) -> PauliSum<W> {
        PauliSum::merge_partitions(self.locals.clone())
    }

    /// [`gather`](Self::gather) by value — no clone of the parts.
    pub fn into_gathered(self) -> PauliSum<W> {
        PauliSum::merge_partitions(self.locals)
    }

    /// Terms in the whole sum, summed over partitions.
    pub fn len(&self) -> usize {
        self.locals.iter().map(PauliSum::len).sum()
    }

    /// Whether every partition is empty.
    pub fn is_empty(&self) -> bool {
        self.locals.iter().all(PauliSum::is_empty)
    }

    /// Partitions this sum is split across.
    pub fn num_partitions(&self) -> usize {
        self.locals.len()
    }

    /// The bucket bits every partition currently holds (they are equal by
    /// construction).
    pub fn bits(&self) -> u8 {
        self.locals[0].hash().bits()
    }

    /// Qubits the sum is over.
    pub fn num_qubits(&self) -> usize {
        self.locals[0].num_qubits()
    }

    /// Partition `r`'s share of the sum.
    ///
    /// # Panics
    ///
    /// If `r` is not a partition of this sum.
    pub fn partition(&self, r: usize) -> &PauliSum<W> {
        &self.locals[r]
    }

    /// The rows deciding which partition a key belongs to.
    pub fn rows(&self) -> &PartitionRows<W> {
        &self.rows
    }

    /// The runtime this sum runs on, for handing to another
    /// [`PartitionedSum`].
    pub fn runtime(&self) -> &Arc<PartitionRuntime> {
        &self.runtime
    }

    /// `⟨ψ|O|ψ⟩` in a uniform single-qubit product state — the sum of the
    /// partitions' own expectation values, since the partitions hold disjoint
    /// terms.
    ///
    /// Partitions are combined in rank order. As with
    /// [`PauliSum::expectation_product_state`], floating-point addition is not
    /// associative, so this need not agree bit for bit with the gathered sum's
    /// answer.
    pub fn expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.locals
            .iter()
            .map(|local| local.expectation_product_state(state))
            .sum()
    }

    /// Debug helper: checks every invariant a partitioned sum is supposed to
    /// hold between calls.
    ///
    /// `O(terms)` — a full scan per partition — so it belongs in a test or an
    /// assertion, not in a loop. The per-partition structural check is
    /// `PauliSum`'s own `assert_invariants`, which exists only in test and
    /// debug builds; in a release build this checks the cross-partition
    /// properties alone.
    ///
    /// # Panics
    ///
    /// If a partition's sum is internally inconsistent, holds a key belonging
    /// to another partition, or disagrees with partition 0 about the hash rows,
    /// the bucket count or the qubit count.
    pub fn assert_invariants(&self) {
        assert_eq!(
            self.locals.len(),
            self.rows.num_partitions(),
            "partition count disagrees with the rows",
        );
        let head = &self.locals[0];
        for (rank, local) in self.locals.iter().enumerate() {
            // `PauliSum::assert_invariants` itself exists only in test and
            // debug builds; the cross-partition checks below are always
            // compiled, so this method's signature does not change with the
            // profile.
            #[cfg(any(test, debug_assertions))]
            local.assert_invariants();
            assert_eq!(
                local.num_qubits(),
                head.num_qubits(),
                "partition {rank}: qubit count differs from partition 0",
            );
            assert!(
                local.hash().same_rows_as(head.hash()),
                "partition {rank}: hash rows differ from partition 0",
            );
            assert_eq!(
                local.hash().bits(),
                head.hash().bits(),
                "partition {rank}: bucket bits differ from partition 0",
            );
            let held = local.partition_rank_of_all(&self.rows);
            assert!(
                held.is_none() || held == Some(rank as u32),
                "partition {rank} holds keys of partition {held:?}",
            );
        }
    }
}

/// The bucket bits a partition takes on at scatter: at most `pbits` shed from
/// the count the unpartitioned sum arrived with, and never below what this
/// partition's own share wants.
///
/// `bits` is the incoming count, `pbits = log2(P)`, `want` the all-reduced
/// per-partition [`desired_bits`]. Shedding exactly `pbits` keeps the bucket
/// count *summed over partitions* equal to the unpartitioned one; the `want`
/// floor stops a sum that arrived under-bucketed from being coarsened at all
/// (and the layer loop grows it from there). At `P = 1`, `pbits = 0`, so this
/// is `bits` — the scatter is the identity and the partitioned run is the
/// unpartitioned one.
fn scatter_bits(bits: u8, pbits: u8, want: u8) -> u8 {
    want.max(bits.saturating_sub(pbits)).min(bits)
}

/// One partition's whole layer loop.
///
/// Runs on the partition's driving thread inside its own pool, in lock-step
/// with its peers: the same channels in the same order, the same collectives
/// per layer.
#[allow(clippy::too_many_arguments)]
fn run_layers<const W: usize, T>(
    circuit: &Circuit<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    rows: &PartitionRows<W>,
    rank: usize,
    size: usize,
    work: &mut PartitionWork<W>,
    transport: &InProcessTransport,
) where
    T: PartitionedTruncation<W> + ?Sized,
{
    let n = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    let local = &mut work.local;

    for k in 0..n {
        let idx = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();

        let layer_t0 = log::log_enabled!(target: LOG_TARGET, log::Level::Debug).then(Instant::now);
        let terms_before = local.len();

        // The one unconditional collective per layer: the bucket count. Each
        // partition proposes what its own share wants, clamped below by what it
        // already has (`refine` is grow-only), and the group takes the maximum
        // — equal bucket counts are what the exchange assumes.
        let want = desired_bits(local.len(), options.target_bucket_len, options.min_buckets)
            .max(local.hash().bits());
        let want = transport.allreduce_max_u8(want);
        while local.hash().bits() < want {
            local.refine();
        }

        let Some(prep) = ch.prepare(local.hash(), adjoint) else {
            // Same hard error as the unpartitioned engine: no whole-sum
            // fallback exists to absorb a channel the engine cannot tabulate.
            let weight: u32 = ch.support().iter().map(|w| w.count_ones()).sum();
            panic!(
                "partition {rank}, layer {idx}: Channel::prepare declined, so this channel \
                 cannot be propagated. The engine tabulates channels of support ≤ \
                 {MAX_LOCAL_SUPPORT} qubits (this one declares {weight}), and a channel must \
                 not write outside its declared support. See \
                 research/notes/2026-08-31-local-ptm-generalization.md",
            );
        };

        let _counts =
            apply_layer_partitioned(local, &prep, rows, policy, &mut work.state, transport);

        // Unconditional and collective, whatever `finalizes_layer` says: a
        // partition that skipped it would desynchronize the group (see
        // `PartitionedTruncation`).
        policy.finalize_layer_partitioned(local, transport);

        if let Some(t0) = layer_t0 {
            log::debug!(
                target: LOG_TARGET,
                "partition {}/{} layer {}/{} [{}]: {} -> {} terms, {:.1} ms",
                rank,
                size,
                k + 1,
                n,
                ch.debug_name(),
                terms_before,
                local.len(),
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
    }
}

/// Propagates `sum` through `circuit` on a partitioned engine built from
/// `config`, and gathers the result.
///
/// One-shot convenience: it builds a [`PartitionRuntime`], scatters, runs and
/// gathers. A caller propagating repeatedly (a Trotter driver stepping an
/// observable) should hold the runtime and a [`PartitionedSum`] instead, so the
/// pools, the split and the scratch survive between calls.
///
/// # Errors
///
/// [`TopologyError`] if `config` cannot be resolved into slots or a pool cannot
/// be built. Everything else is a panic, as in [`propagate`](crate::propagate).
///
/// # Examples
///
/// ```
/// use paulistrings::channel::Clifford1Q;
/// use paulistrings::engine::partitioned::{propagate_partitioned, PartitionConfig, Placement};
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, PartitionedTruncation, PauliString, Phase,
///     TruncationPolicy,
/// };
/// use num_complex::Complex64;
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {
///     // `finalizes_layer`'s default is the conservative `true`; the
///     // `PartitionedTruncation` default body rejects that, since it cannot
///     // know a layer pass it would be skipping is a no-op.
///     fn finalizes_layer(&self) -> bool { false }
/// }
/// impl<const W: usize> PartitionedTruncation<W> for KeepAll {}
///
/// let mut acc = BuildAccumulator::<1>::new(2);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
/// let mut circuit = Circuit::<1>::new(2);
/// circuit.push(Clifford1Q::h(0));
///
/// let config = PartitionConfig {
///     placement: Placement::Unpinned { partitions: 2, threads_per_partition: Some(1) },
///     bind_memory: false,
///     partition_row_seed: None,
/// };
/// let out = propagate_partitioned(
///     &circuit, acc.finalize(), &KeepAll, Direction::Heisenberg, &config,
/// ).expect("topology resolves");
/// assert_eq!(out.get(&[1], &[0]), Some(Complex64::new(1.0, 0.0)));
/// ```
pub fn propagate_partitioned<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    config: &PartitionConfig,
) -> Result<PauliSum<W>, TopologyError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    propagate_partitioned_with_options(
        circuit,
        sum,
        policy,
        direction,
        config,
        PropagateOptions::default(),
    )
}

/// [`propagate_partitioned`] with explicit [`PropagateOptions`].
///
/// [`EngineSelection`](crate::EngineSelection) is ignored; see
/// [`PartitionedSum::propagate_with_options`].
///
/// # Errors
///
/// [`TopologyError`], as [`propagate_partitioned`].
pub fn propagate_partitioned_with_options<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
    config: &PartitionConfig,
    options: PropagateOptions,
) -> Result<PauliSum<W>, TopologyError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let runtime = PartitionRuntime::new(config)?;
    let mut split = PartitionedSum::scatter(sum, runtime, config);
    split.propagate_with_options(circuit, policy, direction, options);
    Ok(split.into_gathered())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `P = 1` sheds nothing, so a scatter cannot change the bucket count and
    /// the partitioned run starts exactly where `propagate` would.
    #[test]
    fn scatter_bits_is_the_identity_at_one_partition() {
        for bits in 0u8..12 {
            for want in 0u8..12 {
                assert_eq!(scatter_bits(bits, 0, want), bits, "bits={bits} want={want}");
            }
        }
    }

    /// With `P` partitions the count sheds `log2(P)` bits, so the bucket count
    /// summed over partitions is the unpartitioned one — unless a partition's
    /// own share wants more.
    #[test]
    fn scatter_bits_sheds_at_most_log2_p() {
        // Incoming 10 bits, 4 partitions each wanting 8: shed exactly 2.
        assert_eq!(scatter_bits(10, 2, 8), 8);
        // Wanting more than the split leaves: the want wins, capped by what is
        // there (the layer loop grows past it).
        assert_eq!(scatter_bits(10, 2, 9), 9);
        assert_eq!(scatter_bits(10, 2, 12), 10);
        // Wanting less than the split leaves: never shed more than log2(P).
        assert_eq!(scatter_bits(10, 2, 3), 8);
        // Fewer incoming bits than there are partitions: the floor saturates at
        // a single bucket per partition, so the per-share want decides.
        assert_eq!(scatter_bits(1, 2, 0), 0);
        assert_eq!(scatter_bits(1, 2, 1), 1);
        assert_eq!(scatter_bits(0, 4, 0), 0);
    }
}
