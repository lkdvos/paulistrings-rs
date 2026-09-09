//! [`DistributedSum`]: the partitioned driver with **one partition per
//! process**.
//!
//! [`PartitionedSum`](super::PartitionedSum) fans `P` partitions out over the
//! threads of one process; a `DistributedSum` is the other shape of the same
//! engine — this process *is* partition `transport.rank()`, and its peers are
//! other processes. Everything between the two is shared: the per-layer body is
//! literally the same function ([`run_layers`]), because a partition talks to
//! its peers only through [`Transport`] and nothing in the layer knows whether
//! the peer is a thread or a rank (ARCHITECTURE.md §Partitioning, *Transport
//! composition*).
//!
//! The type is generic over the transport, so the in-process one drives it too
//! — which is how the distributed shape is tested with no MPI in the picture.
//! With the `mpi` feature, [`paulistrings::mpi`](crate::engine::partitioned::mpi)
//! supplies `MpiTransport` and the [`propagate_mpi`] one-shot.
//!
//! [`propagate_mpi`]: crate::engine::partitioned::mpi::propagate_mpi
//!
//! # The contract
//!
//! - **Input is replicated.** Every rank calls [`DistributedSum::scatter`] with
//!   the *same* sum and keeps `filter_partition(rows, rank)` of it. The
//!   partition rows are drawn from one seed, so every rank derives the same
//!   ones without a collective. (Nothing checks the sums are equal — that is
//!   what would need a full comparison; [`check_consistency`](super::Collectives::check_consistency)
//!   catches the cheap half of the mistake at the first propagation.)
//! - **`D = 1`.** The runtime has exactly one partition. Placement comes from
//!   the launcher: `mpirun --map-by ppr:1:numa --bind-to numa` (or `srun
//!   --cpu-bind=ldoms`) leaves the process an affinity mask of one NUMA
//!   domain, and [`Placement::Auto`](super::Placement::Auto) over that mask
//!   resolves to a single slot covering it. There is no domains-per-rank
//!   hybrid yet.
//! - **Output is rank 0's.** [`gather`](DistributedSum::gather) returns
//!   `Some(sum)` on rank 0 and `None` elsewhere;
//!   [`local`](DistributedSum::local) is always this rank's share, itself a
//!   valid [`PauliSum`] under the group's shared hash.
//! - **The trace is per rank.** A [`PartitionTrace`] taken here has one entry
//!   in each of `terms_in`, `terms_out` and `rows_received` — the local one —
//!   while `rows_sent[0]` and `bytes_sent[0]` are indexed by destination rank
//!   over the whole group. Assembling the group's view is the caller's job (the
//!   records are small; gather them however the harness gathers everything
//!   else).

use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;

use super::driver::{run_layers, scatter_local, PartitionWork};
use super::layer::PartitionState;
use super::runtime::PartitionRuntime;
use super::topology::{PartitionConfig, TopologyError};
use super::trace::{assemble, PartitionTrace};
use super::transport::Transport;
use super::truncation::PartitionedTruncation;
use crate::bucket::hash::PartitionRows;
use crate::circuit::Circuit;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::{PauliSum, ProductState};

#[cfg(feature = "phase-timing")]
use super::driver::PartitionPhaseStats;

/// `log` target for the driver's progress events, as in
/// [`propagate`](crate::propagate).
const LOG_TARGET: &str = "paulistrings::propagate";

/// Parts the gather ships per rank: bucket lengths, then the three columns.
const GATHER_PARTS: usize = 4;

/// One process's partition of a sum split across a [`Transport`]'s group.
///
/// Held across calls — the split, the rows, the pool and the layer scratch all
/// persist — so a Trotter driver scatters once, steps many times, and gathers
/// once. See the module docs for the input/output contract.
///
/// # Examples
///
/// The in-process transport drives it as well as MPI does, which is what makes
/// the shape testable without a launcher. Two "ranks", one thread each:
///
/// ```
/// use std::sync::Arc;
/// use paulistrings::channel::Clifford1Q;
/// use paulistrings::engine::partitioned::{
///     Collectives, DistributedSum, InProcessTransport, PartitionConfig, Placement,
/// };
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, PartitionedTruncation, PauliString, Phase,
///     TruncationPolicy,
/// };
/// use num_complex::Complex64;
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {
///     fn finalizes_layer(&self) -> bool { false }
/// }
/// impl<const W: usize> PartitionedTruncation<W> for KeepAll {}
///
/// let config = PartitionConfig {
///     placement: Placement::Unpinned { partitions: 1, threads_per_partition: Some(1) },
///     bind_memory: false,
///     partition_row_seed: Some(7),
/// };
///
/// // The replicated input: every rank builds the identical sum.
/// let input = || {
///     let mut acc = BuildAccumulator::<1>::new(2);
///     acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
///     acc.finalize()
/// };
/// let mut circuit = Circuit::<1>::new(2);
/// circuit.push(Clifford1Q::h(0));
///
/// let gathered = std::thread::scope(|scope| {
///     let handles: Vec<_> = InProcessTransport::group(2)
///         .into_iter()
///         .map(|transport| {
///             let config = config.clone();
///             let circuit = &circuit;
///             scope.spawn(move || {
///                 let mut split = DistributedSum::scatter(input(), transport, &config)
///                     .expect("topology resolves");
///                 split.propagate(circuit, &KeepAll, Direction::Heisenberg);
///                 split.gather()
///             })
///         })
///         .collect();
///     handles.into_iter().filter_map(|h| h.join().unwrap()).next()
/// });
///
/// // H conjugates Z to X, on whichever rank the term happened to land.
/// let out = gathered.expect("rank 0 gathers");
/// assert_eq!(out.get(&[1], &[0]), Some(Complex64::new(1.0, 0.0)));
/// ```
pub struct DistributedSum<const W: usize, X: Transport> {
    /// This rank's share of the sum.
    local: PauliSum<W>,
    /// The rows deciding which rank a key belongs to.
    rows: PartitionRows<W>,
    /// The one-partition runtime this rank's work runs on.
    runtime: Arc<PartitionRuntime>,
    /// This rank's layer and export scratch, retained across calls.
    state: PartitionState<W>,
    /// This rank's endpoint in the group.
    transport: X,
    /// The opt-in per-layer trace, `None` unless
    /// [`enable_trace`](Self::enable_trace) was called. Local to this rank.
    trace: Option<PartitionTrace>,
    /// Scatter time, this rank's.
    #[cfg(feature = "phase-timing")]
    scatter_ns: u64,
    /// Gather time summed over [`gather`](Self::gather) calls. Atomic because
    /// `gather` takes `&self`; nothing contends for it.
    #[cfg(feature = "phase-timing")]
    gather_ns: std::sync::atomic::AtomicU64,
    /// Layers driven since the counters were drained.
    #[cfg(feature = "phase-timing")]
    layers: u64,
}

impl<const W: usize, X: Transport> DistributedSum<W, X> {
    /// Split the replicated `sum` across `transport`'s group, keeping this
    /// rank's share, and build the one-partition runtime `config` describes.
    ///
    /// The rows come from [`PartitionRows::from_seed`] with
    /// `config.partition_row_seed`, falling back to the sum's own hash seed —
    /// identical on every rank, since the input is, so no collective is needed
    /// to agree on them.
    ///
    /// # Errors
    ///
    /// [`TopologyError`] if `config` cannot be resolved or the pool cannot be
    /// built.
    ///
    /// # Panics
    ///
    /// If `config` asks for more than one partition (a rank *is* a partition;
    /// the hybrid shape does not exist yet), or if the group size is not a
    /// power of two.
    pub fn scatter(
        sum: PauliSum<W>,
        transport: X,
        config: &PartitionConfig,
    ) -> Result<Self, TopologyError> {
        let runtime = PartitionRuntime::new(config)?;
        Ok(Self::scatter_with_runtime(
            sum,
            transport,
            runtime,
            config.partition_row_seed,
        ))
    }

    /// [`scatter`](Self::scatter) onto a runtime the caller already built —
    /// the form a Trotter driver uses to keep one pinned pool across many
    /// sums.
    ///
    /// # Panics
    ///
    /// If `runtime` has more than one partition, or the group size is not a
    /// power of two.
    pub fn scatter_with_runtime(
        sum: PauliSum<W>,
        transport: X,
        runtime: Arc<PartitionRuntime>,
        partition_row_seed: Option<u64>,
    ) -> Self {
        let size = transport.size();
        assert!(
            size.is_power_of_two(),
            "a group of {size} ranks cannot be a partitioning: a partition is named by log2(P) \
             GF(2) rows, so the rank count must be a power of two",
        );
        let seed = partition_row_seed.unwrap_or_else(|| sum.hash().seed());
        let rows =
            PartitionRows::<W>::from_seed(sum.num_qubits(), size.trailing_zeros() as u8, seed);
        Self::scatter_with_rows(sum, transport, runtime, rows)
    }

    /// [`scatter`](Self::scatter) with caller-supplied partition rows.
    ///
    /// Every rank must pass the *same* rows; nothing checks it, and a
    /// disagreement misroutes an exchange rather than failing loudly.
    ///
    /// # Panics
    ///
    /// If `runtime` has more than one partition, if `rows` does not name one
    /// partition per rank, or if it is for a different qubit count than `sum`.
    /// In debug builds, if the rows are not independent of the sum's hash rows
    /// (which costs load balance, not correctness — see
    /// [`PartitionedSum::scatter_with_rows`](super::PartitionedSum::scatter_with_rows)).
    pub fn scatter_with_rows(
        sum: PauliSum<W>,
        transport: X,
        runtime: Arc<PartitionRuntime>,
        rows: PartitionRows<W>,
    ) -> Self {
        assert_eq!(
            runtime.num_partitions(),
            1,
            "a distributed run holds one partition per process, but the runtime has {}; \
             domains-per-rank hybrids are not implemented",
            runtime.num_partitions(),
        );
        assert_eq!(
            rows.num_partitions(),
            transport.size() as usize,
            "partition rows name {} partitions but the group has {} ranks",
            rows.num_partitions(),
            transport.size(),
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
            "partition rows are dependent on the bucket hash rows — the split will correlate \
             with the bucket partition and load-balance badly",
        );

        let started = Instant::now();
        let local = {
            let sum = &sum;
            let rows = &rows;
            let transport = &transport;
            runtime.install(move || scatter_local(sum, rows, transport.rank(), transport))
        };
        log::info!(
            target: LOG_TARGET,
            "scatter: rank {}/{}, {} terms in, {} kept locally, {} bucket bits, {:.3} s",
            transport.rank(),
            transport.size(),
            sum.len(),
            local.len(),
            local.hash().bits(),
            started.elapsed().as_secs_f64(),
        );

        Self {
            local,
            rows,
            runtime,
            state: PartitionState::default(),
            transport,
            trace: None,
            #[cfg(feature = "phase-timing")]
            scatter_ns: started.elapsed().as_nanos() as u64,
            #[cfg(feature = "phase-timing")]
            gather_ns: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "phase-timing")]
            layers: 0,
        }
    }

    /// Propagate through `circuit` under `policy`, in place, with
    /// [`PropagateOptions::default()`].
    pub fn propagate<T>(&mut self, circuit: &Circuit<W>, policy: &T, direction: Direction)
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with_options(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit` under `policy` with explicit
    /// [`PropagateOptions`].
    ///
    /// **Collective**: every rank must call it, with the same circuit, the same
    /// direction and the same options. One
    /// [`check_consistency`](super::Collectives::check_consistency) call up front turns the common way of
    /// getting that wrong — ranks driven through different circuits — into a
    /// message rather than a deadlock two layers in. It cannot see a difference
    /// in the *contents* of two channels, only in the shape of the run.
    ///
    /// [`EngineSelection`](crate::EngineSelection) is ignored, exactly as in
    /// [`PartitionedSum::propagate_with_options`](super::PartitionedSum::propagate_with_options).
    ///
    /// # Panics
    ///
    /// As the in-process driver: a channel whose `prepare` declines, or a
    /// policy that finalizes layers with no collective form. Plus
    /// [`check_consistency`](super::Collectives::check_consistency)'s own panic when the ranks disagree
    /// about the run.
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
        let rank = self.transport.rank() as usize;
        let size = self.transport.size() as usize;
        let terms_in = self.local.len();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate_distributed: rank {rank}/{size}, {terms_in} local terms through {n} \
             channels ({direction:?}) [{}]",
            self.runtime.placement_summary(),
        );

        self.transport.check_consistency(run_fingerprint(
            n,
            direction,
            options,
            self.num_qubits(),
            W,
        ));

        if n > 0 {
            let tracing = self.trace.is_some();
            let mut work = PartitionWork::take(&mut self.local, &mut self.state, n, tracing);

            {
                let runtime = Arc::clone(&self.runtime);
                let rows = &self.rows;
                let transport = &self.transport;
                let work = &mut work;
                runtime.install(move || {
                    run_layers(
                        circuit, policy, direction, options, rows, rank, size, tracing, work,
                        transport,
                    );
                });
            }

            self.local = work.local;
            self.state = work.state;
            if let Some(trace) = self.trace.as_mut() {
                assemble(trace, vec![work.rows]);
            }
            #[cfg(feature = "phase-timing")]
            {
                self.layers += n as u64;
            }
        }

        log::info!(
            target: LOG_TARGET,
            "propagate_distributed: rank {rank}/{size}, {n} layers applied, {terms_in} -> {} \
             local terms, {:.3} s",
            self.local.len(),
            started.elapsed().as_secs_f64(),
        );
    }

    /// Collect the whole sum on rank 0: `Some(sum)` there, `None` elsewhere.
    ///
    /// **Collective**: every rank must call it. `self` is left intact, so a
    /// driver can gather a checkpoint and keep stepping.
    ///
    /// Rank `r` ships its bucket lengths and the three columns
    /// [`PauliSum::to_arrays`] concatenates, and rank 0 rebuilds each rank's
    /// sum under the group's shared hash before
    /// `PauliSum::merge_partitions` merges the disjoint runs. Rank 0
    /// therefore holds the whole sum plus one rank's transient copy of it, and
    /// every other rank holds one transient copy of its own share — the
    /// send-side copy is `to_arrays`, and the alternative (one zero-copy part
    /// per bucket) would cost thousands of messages per rank.
    pub fn gather(&self) -> Option<PauliSum<W>> {
        #[cfg(feature = "phase-timing")]
        let started = Instant::now();

        let lens: Vec<u64> = (0..self.local.num_buckets())
            .map(|b| self.local.bucket_len(b) as u64)
            .collect();
        let (x, z, coeff) = self.local.to_arrays();
        let parts: Vec<&[u8]> = vec![
            bytemuck::cast_slice(&lens),
            bytemuck::cast_slice(x.as_flattened()),
            bytemuck::cast_slice(z.as_flattened()),
            bytemuck::cast_slice(&coeff),
        ];
        let all = self.transport.gather_to_root(parts);
        drop((x, z, coeff));

        let out = all.map(|all| {
            let hash = self.local.hash().clone();
            let num_qubits = self.local.num_qubits();
            let parts: Vec<PauliSum<W>> = all
                .into_iter()
                .enumerate()
                .map(|(rank, parts)| decode_rank(&parts, hash.clone(), num_qubits, rank))
                .collect();
            PauliSum::merge_partitions(parts)
        });

        #[cfg(feature = "phase-timing")]
        self.gather_ns.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        out
    }

    /// This rank's share of the sum — a valid [`PauliSum`] under the group's
    /// shared hash, holding exactly the keys of partition
    /// [`rank`](Self::rank).
    pub fn local(&self) -> &PauliSum<W> {
        &self.local
    }

    /// This rank's index in the group.
    pub fn rank(&self) -> u32 {
        self.transport.rank()
    }

    /// Ranks in the group, which is also the partition count.
    pub fn size(&self) -> u32 {
        self.transport.size()
    }

    /// This rank's endpoint, for a caller that needs a collective of its own
    /// (a reduction over per-rank measurements, say).
    pub fn transport(&self) -> &X {
        &self.transport
    }

    /// Terms this rank holds. Local, and cheap.
    pub fn len_local(&self) -> usize {
        self.local.len()
    }

    /// Whether this rank holds no terms. Local: the *group* may be non-empty.
    pub fn is_empty_local(&self) -> bool {
        self.local.is_empty()
    }

    /// Terms in the whole sum. **Collective** — one all-reduce, same answer on
    /// every rank.
    pub fn len(&self) -> usize {
        let mut buf = [self.local.len() as u64];
        self.transport.allreduce_sum_u64(&mut buf);
        buf[0] as usize
    }

    /// Whether the whole sum is empty. **Collective**, via [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bucket bits this rank holds. Equal on every rank by construction —
    /// the count is agreed collectively every layer.
    pub fn bits(&self) -> u8 {
        self.local.hash().bits()
    }

    /// Qubits the sum is over.
    pub fn num_qubits(&self) -> usize {
        self.local.num_qubits()
    }

    /// The rows deciding which rank a key belongs to.
    pub fn rows(&self) -> &PartitionRows<W> {
        &self.rows
    }

    /// The runtime this rank's work runs on, for handing to another
    /// [`DistributedSum`].
    pub fn runtime(&self) -> &Arc<PartitionRuntime> {
        &self.runtime
    }

    /// `⟨ψ|O|ψ⟩` in a uniform single-qubit product state, over the whole sum.
    ///
    /// Not collective — it cannot be, [`Collectives`](super::Collectives) reducing only integers —
    /// so it is this rank's contribution alone. Sum the ranks' answers however
    /// the application reduces its own scalars (`MPI_Allreduce` on two `f64`s,
    /// through the communicator the transport was built from).
    pub fn local_expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.local.expectation_product_state(state)
    }

    /// Start recording a [`PartitionTrace`] of **this rank's** layers on every
    /// subsequent propagation. Idempotent.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain and return this rank's records, or `None` if tracing was never
    /// enabled.
    ///
    /// Per rank: each record's `terms_in`, `terms_out` and `rows_received`
    /// have exactly one entry (this rank's), while `rows_sent[0]` and
    /// `bytes_sent[0]` are indexed by destination rank over the whole group.
    /// Draining leaves tracing enabled with no records.
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.trace.as_mut().map(std::mem::take)
    }

    /// Drain and return this rank's phase counters, in the same shape the
    /// in-process driver reports: `per_partition` has one entry.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        PartitionPhaseStats {
            per_partition: vec![self.state.layer.take_stats()],
            scatter_ns: std::mem::take(&mut self.scatter_ns),
            gather_ns: self.gather_ns.swap(0, std::sync::atomic::Ordering::Relaxed),
            layers: std::mem::take(&mut self.layers),
        }
    }

    /// Debug helper: check that this rank's share is a well-formed sum holding
    /// only its own keys.
    ///
    /// `O(terms)`, so it belongs in a test. The cross-rank half of
    /// [`PartitionedSum::assert_invariants`](super::PartitionedSum::assert_invariants)
    /// — equal bits and hash rows across partitions — is not checked here: it
    /// would need a collective, and the layer loop's bucket-count all-reduce
    /// establishes it every layer anyway.
    ///
    /// # Panics
    ///
    /// If this rank's sum is internally inconsistent or holds a key belonging
    /// to another rank.
    pub fn assert_invariants(&self) {
        #[cfg(any(test, debug_assertions))]
        self.local.assert_invariants();
        let held = self.local.partition_rank_of_all(&self.rows);
        assert!(
            held.is_none() || held == Some(self.rank()),
            "rank {} holds keys of partition {held:?}",
            self.rank(),
        );
    }
}

/// A fingerprint of everything the ranks must agree on before the first layer.
///
/// Deliberately cheap and shape-only: the channel count, the direction, the
/// bucket-policy knobs, the qubit count and `W`. It does not hash the channels
/// themselves — a `Circuit` is a list of trait objects with no canonical
/// encoding, and the failure it is there to catch (one rank handed a different
/// circuit, or a different `PropagateOptions`) shows up in these numbers in
/// practice.
fn run_fingerprint(
    channels: usize,
    direction: Direction,
    options: PropagateOptions,
    num_qubits: usize,
    w: usize,
) -> u64 {
    // FNV-1a over the fields, in a fixed order.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(channels as u64);
    mix(matches!(direction, Direction::Heisenberg) as u64);
    mix(options.target_bucket_len as u64);
    mix(options.min_buckets as u64);
    mix(num_qubits as u64);
    mix(w as u64);
    h
}

/// Copy `len` values of `T` out of a possibly unaligned byte view.
///
/// `bytemuck::cast_slice` would be free but requires the input to be aligned
/// for `T`, which a received buffer need not be. The layer's exchange avoids
/// the copy entirely (`Payload::recv_into` receives into the typed columns);
/// the gather runs once per run and does not bother.
fn decode_column<T: bytemuck::Pod>(bytes: &[u8], len: usize, what: &str) -> Vec<T> {
    let stride = std::mem::size_of::<T>();
    assert_eq!(
        bytes.len(),
        len * stride,
        "gather {what}: expected {} bytes for {len} entries, got {}",
        len * stride,
        bytes.len(),
    );
    bytes
        .chunks_exact(stride)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

/// [`decode_column`] for key words: `[u64; W]` at a generic `W` is not `Pod`
/// under the feature set this crate builds `bytemuck` with, so the words are
/// read individually and assembled.
fn decode_rows<const W: usize>(bytes: &[u8], rows: usize, what: &str) -> Vec<[u64; W]> {
    let stride = W * std::mem::size_of::<u64>();
    assert_eq!(
        bytes.len(),
        rows * stride,
        "gather {what}: expected {} bytes for {rows} rows, got {}",
        rows * stride,
        bytes.len(),
    );
    bytes
        .chunks_exact(stride)
        .map(|row| std::array::from_fn(|i| bytemuck::pod_read_unaligned(&row[i * 8..i * 8 + 8])))
        .collect()
}

/// Rebuild one rank's share from the [`GATHER_PARTS`] parts it shipped.
///
/// # Panics
///
/// If the parts are not the four the gather encodes, or their lengths
/// contradict each other — either means the ranks are not running the same
/// build.
fn decode_rank<const W: usize>(
    parts: &[Vec<u8>],
    hash: crate::bucket::hash::Gf2Hash<W>,
    num_qubits: usize,
    rank: usize,
) -> PauliSum<W> {
    assert_eq!(
        parts.len(),
        GATHER_PARTS,
        "gather: rank {rank} sent {} parts, expected {GATHER_PARTS}",
        parts.len(),
    );
    let num_buckets = hash.num_buckets();
    let lens: Vec<u64> = decode_column(&parts[0], num_buckets, "bucket lengths");
    let lens: Vec<usize> = lens.iter().map(|&l| l as usize).collect();
    let terms: usize = lens.iter().sum();
    let x = decode_rows::<W>(&parts[1], terms, "x column");
    let z = decode_rows::<W>(&parts[2], terms, "z column");
    let coeff = decode_column::<Complex64>(&parts[3], terms, "coeff column");
    PauliSum::from_bucket_columns(&lens, x, z, coeff, hash, num_qubits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fingerprint_separates_every_field_it_covers() {
        let base = run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 8, 1);
        assert_ne!(
            base,
            run_fingerprint(4, Direction::Forward, PropagateOptions::default(), 8, 1),
        );
        assert_ne!(
            base,
            run_fingerprint(3, Direction::Heisenberg, PropagateOptions::default(), 8, 1),
        );
        assert_ne!(
            base,
            run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 9, 1),
        );
        assert_ne!(
            base,
            run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 8, 2),
        );
        let mut options = PropagateOptions::default();
        options.target_bucket_len += 1;
        assert_ne!(base, run_fingerprint(3, Direction::Forward, options, 8, 1),);
        let mut options = PropagateOptions::default();
        options.min_buckets += 1;
        assert_ne!(base, run_fingerprint(3, Direction::Forward, options, 8, 1),);
        // And it is a function of its inputs, not of the call.
        assert_eq!(
            base,
            run_fingerprint(3, Direction::Forward, PropagateOptions::default(), 8, 1),
        );
    }
}
