//! [`DistributedSum`], the partitioned driver with one partition per process, and its scatter options.

use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;

use super::backend::{HostPartition, PartitionBackend, PartitionStorage};
use super::driver::{run_layers, scatter_local, PartitionCtx, PartitionWork};
use super::runtime::PartitionRuntime;
use super::topology::{PartitionConfig, TopologyError};
use super::trace::{assemble, PartitionTrace};
use super::transport::Transport;
use super::truncation::PartitionedTruncation;
use crate::circuit::Circuit;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::PauliSum;
use crate::readout::echo::{qubit_mask, RotationAxis};
use crate::readout::product_state::ProductState;

#[cfg(feature = "phase-timing")]
use super::sum::PartitionPhaseStats;

/// `log` target shared with [`propagate`](crate::propagate).
const LOG_TARGET: &str = "paulistrings::propagate";

/// Parts the gather ships per rank: bucket lengths, then the three columns.
const GATHER_PARTS: usize = 4;

/// Which partition rows a scatter splits by: a seeded GF(2)-random draw, or a qubit cut that keeps a channel local when its qubits share a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartitionRowPolicy {
    /// [`PartitionRows::from_seed`] with this seed, or the sum's own hash seed at `None`; the default.
    Seeded(Option<u64>),
    /// [`PartitionRows::cut`]: one disjoint qubit block per rank, in rank order; the rows are z-only.
    Cut(Vec<Vec<u32>>),
    /// [`PartitionRows::from_seed_excluding`]: a seeded draw whose rows read neither the x-bits of `exclude_x` nor the z-bits of `exclude_z`, as [`DistributedSum::rotated_overlap`] needs.
    SeededExcluding {
        /// As in [`Seeded`](Self::Seeded).
        seed: Option<u64>,
        /// Qubits whose x-bit no row reads.
        exclude_x: Vec<u32>,
        /// Qubits whose z-bit no row reads.
        exclude_z: Vec<u32>,
    },
}

impl PartitionRowPolicy {
    /// The rows for `2^bits` partitions of a `num_qubits` register, with `default_seed` standing in for a `None` seed.
    ///
    /// # Panics
    ///
    /// As [`PartitionRows::cut`] or [`PartitionRows::from_seed_excluding`], and if an excluded qubit is out of range.
    pub fn rows<const W: usize>(
        &self,
        num_qubits: usize,
        bits: u8,
        default_seed: u64,
    ) -> PartitionRows<W> {
        let mask = |qubits: &[u32]| qubit_mask(qubits.iter().map(|&q| q as usize), num_qubits);
        match self {
            Self::Seeded(seed) => {
                PartitionRows::from_seed(num_qubits, bits, seed.unwrap_or(default_seed))
            }
            Self::Cut(blocks) => PartitionRows::cut(num_qubits, blocks),
            Self::SeededExcluding {
                seed,
                exclude_x,
                exclude_z,
            } => PartitionRows::from_seed_excluding(
                num_qubits,
                bits,
                seed.unwrap_or(default_seed),
                &mask(exclude_x),
                &mask(exclude_z),
            ),
        }
    }
}

/// How [`DistributedSum::scatter_with`] runs this rank's partition and picks the rows the group splits by.
#[derive(Clone)]
pub struct ScatterOptions<const W: usize> {
    /// The one-partition runtime, reusable across sums.
    pub runtime: Arc<PartitionRuntime>,
    /// The partition rows the group splits by.
    pub rows: ScatterRows<W>,
}

/// The partition rows a [`DistributedSum::scatter_with`] splits by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScatterRows<const W: usize> {
    /// The rows a policy draws for the group's size, falling back to the sum's own hash seed.
    Policy(PartitionRowPolicy),
    /// Rows the caller built; every rank must pass the same ones.
    Explicit(PartitionRows<W>),
}

/// `log2(size)`, the partition bits of a group; panics unless `size` is a power of two.
pub(crate) fn group_bits(size: u32) -> u8 {
    assert!(
        size.is_power_of_two(),
        "a group of {size} ranks cannot be a partitioning: a partition is named by log2(P) \
         GF(2) rows, so the rank count must be a power of two",
    );
    size.trailing_zeros() as u8
}

/// One process's partition of a sum split across a [`Transport`]'s group, held across calls so a driver scatters once, steps many times and gathers once (ARCHITECTURE.md §Transport composition).
///
/// The input is replicated: every rank scatters the same sum and keeps its own partition.
/// [`gather`](Self::gather) returns `Some` on rank 0 only, and a [`PartitionTrace`] holds this rank's counts only.
/// `B` is where the partition lives: host memory, or a CUDA device under the `cuda` feature (`gpu::GpuDistributedSum`).
pub struct DistributedSum<const W: usize, X: Transport, B = HostPartition<W>> {
    local: B,
    rows: PartitionRows<W>,
    runtime: Arc<PartitionRuntime>,
    transport: X,
    trace: Option<PartitionTrace>,
    #[cfg(feature = "phase-timing")]
    scatter_ns: u64,
    /// Atomic because `gather` takes `&self`.
    #[cfg(feature = "phase-timing")]
    gather_ns: std::sync::atomic::AtomicU64,
    /// Layers driven since the counters were drained.
    #[cfg(feature = "phase-timing")]
    layers: u64,
}

impl<const W: usize, X: Transport, B> DistributedSum<W, X, B> {
    /// This rank's index in the group.
    pub fn rank(&self) -> u32 {
        self.transport.rank()
    }

    /// Ranks in the group, which is also the partition count.
    pub fn size(&self) -> u32 {
        self.transport.size()
    }

    /// This rank's endpoint, for a collective of the caller's own.
    pub fn transport(&self) -> &X {
        &self.transport
    }

    /// The rows deciding which rank a key belongs to.
    pub fn rows(&self) -> &PartitionRows<W> {
        &self.rows
    }

    /// The runtime this rank's work runs on, for handing to another [`DistributedSum`].
    pub fn runtime(&self) -> &Arc<PartitionRuntime> {
        &self.runtime
    }

    /// Start recording a [`PartitionTrace`] of this rank's layers; idempotent.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain this rank's records, or `None` if tracing was never enabled.
    ///
    /// `terms_in`, `terms_out` and `rows_received` have one entry, while `rows_sent[0]` and `bytes_sent[0]` are indexed by destination rank.
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.trace.as_mut().map(std::mem::take)
    }

    /// A split around a partition another backend already scattered, with `scatter_ns` its scatter time.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_backend(
        local: B,
        rows: PartitionRows<W>,
        runtime: Arc<PartitionRuntime>,
        transport: X,
        scatter_ns: u64,
    ) -> Self {
        #[cfg(not(feature = "phase-timing"))]
        let _ = scatter_ns;
        Self {
            local,
            rows,
            runtime,
            transport,
            trace: None,
            #[cfg(feature = "phase-timing")]
            scatter_ns,
            #[cfg(feature = "phase-timing")]
            gather_ns: std::sync::atomic::AtomicU64::new(0),
            #[cfg(feature = "phase-timing")]
            layers: 0,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn backend(&self) -> &B {
        &self.local
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn backend_mut(&mut self) -> &mut B {
        &mut self.local
    }

    /// Drain scatter time, gather time and layers driven.
    #[cfg(all(feature = "cuda", feature = "phase-timing"))]
    pub(crate) fn take_driver_laps(&mut self) -> (u64, u64, u64) {
        (
            std::mem::take(&mut self.scatter_ns),
            self.gather_ns.swap(0, std::sync::atomic::Ordering::Relaxed),
            std::mem::take(&mut self.layers),
        )
    }

    #[cfg(feature = "phase-timing")]
    pub(crate) fn lap_gather(&self, ns: u64) {
        self.gather_ns
            .fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }

    /// [`DistributedSum::propagate_with`] on any backend.
    pub(crate) fn propagate_on_backend<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
        B: PartitionBackend<W, T>,
    {
        let layer_count = circuit.channels.len();
        let rank = self.transport.rank() as usize;
        let size = self.transport.size() as usize;
        let terms_in = self.local.len();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate_distributed: rank {rank}/{size}, {terms_in} local terms through {layer_count} \
             channels ({direction:?}) [{}]",
            self.runtime.placement_summary(),
        );

        self.transport.check_consistency(run_fingerprint(
            layer_count,
            direction,
            options,
            self.rows.num_qubits(),
            W,
        ));

        if layer_count > 0 {
            let tracing = self.trace.is_some();
            let mut work = PartitionWork::take(&mut self.local, layer_count, tracing);

            {
                let runtime = Arc::clone(&self.runtime);
                let rows = &self.rows;
                let transport = &self.transport;
                let work = &mut work;
                let context = PartitionCtx {
                    rows,
                    rank,
                    size,
                    tracing,
                };
                runtime.install(move || {
                    run_layers(
                        circuit, policy, direction, options, context, work, transport,
                    );
                });
            }

            self.local = work.local;
            if let Some(trace) = self.trace.as_mut() {
                assemble(trace, vec![work.rows]);
            }
            #[cfg(feature = "phase-timing")]
            {
                self.layers += layer_count as u64;
            }
        }

        log::info!(
            target: LOG_TARGET,
            "propagate_distributed: rank {rank}/{size}, {layer_count} layers applied, {terms_in} -> {} \
             local terms, {:.3} s",
            self.local.len(),
            started.elapsed().as_secs_f64(),
        );
    }
}

impl<const W: usize, X: Transport, B: PartitionStorage<W>> DistributedSum<W, X, B> {
    /// Terms this rank holds.
    pub fn len_local(&self) -> usize {
        self.local.len()
    }

    /// Terms in the whole sum. **Collective.**
    pub fn len(&self) -> usize {
        let mut total = [self.local.len() as u64];
        self.transport.allreduce_sum_u64(&mut total);
        total[0] as usize
    }

    /// Whether the whole sum is empty. **Collective**, via [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The bucket bits, equal on every rank.
    pub fn bits(&self) -> u8 {
        self.local.hash().bits()
    }

    /// Qubits the sum is over.
    pub fn num_qubits(&self) -> usize {
        self.rows.num_qubits()
    }
}

impl<const W: usize, X: Transport> DistributedSum<W, X> {
    /// Split the replicated `sum` across `transport`'s group by seeded rows, keeping this rank's share on the one-partition runtime `config` describes.
    ///
    /// # Errors
    ///
    /// [`TopologyError`] if `config` cannot be resolved or the pool cannot be built.
    ///
    /// # Panics
    ///
    /// If `config` asks for more than one partition, or if the group size is not a power of two.
    pub fn scatter(
        sum: PauliSum<W>,
        transport: X,
        config: &PartitionConfig,
    ) -> Result<Self, TopologyError> {
        let options = ScatterOptions {
            runtime: PartitionRuntime::new(config)?,
            rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(config.partition_row_seed)),
        };
        Ok(Self::scatter_with(sum, transport, options))
    }

    /// [`scatter`](Self::scatter) onto a caller-built runtime, split by the rows `options` names.
    ///
    /// [`ScatterRows::Explicit`] rows must be the same on every rank; nothing checks it, and a disagreement misroutes an exchange.
    ///
    /// # Panics
    ///
    /// If the runtime has more than one partition, if the group size is not a power of two, if the rows do not name one partition per rank or are for a different qubit count than `sum`, or as [`PartitionRows::cut`].
    /// In debug builds, if the rows are not independent of the sum's hash rows.
    pub fn scatter_with(sum: PauliSum<W>, transport: X, options: ScatterOptions<W>) -> Self {
        let ScatterOptions { runtime, rows } = options;
        let rows = match rows {
            ScatterRows::Policy(policy) => policy.rows(
                sum.num_qubits(),
                group_bits(transport.size()),
                sum.hash().seed(),
            ),
            ScatterRows::Explicit(rows) => rows,
        };
        assert_eq!(
            runtime.num_partitions(),
            1,
            "a distributed run holds one partition per process, but the runtime has {}; \
             domains-per-rank hybrids are not implemented",
            runtime.num_partitions(),
        );
        rows.assert_splits(
            sum.hash(),
            sum.num_qubits(),
            1usize << group_bits(transport.size()),
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
            local: HostPartition::new(local),
            rows,
            runtime,
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

    /// [`propagate_with`](Self::propagate_with) under [`PropagateOptions::default()`].
    pub fn propagate<T>(&mut self, circuit: &Circuit<W>, policy: &T, direction: Direction)
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit` under `policy`, in place. **Collective**: every rank passes the same circuit, direction and options.
    ///
    /// A shape fingerprint of the run (channel count, direction, options, qubit count, `W`) is checked across ranks first; channel contents are not.
    /// [`EngineSelection`](crate::EngineSelection) is ignored.
    ///
    /// # Panics
    ///
    /// If a channel's `prepare` declines, or if the ranks disagree about the run's shape.
    pub fn propagate_with<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_on_backend(circuit, policy, direction, options);
    }

    /// Collect the whole sum on rank 0, `None` elsewhere, leaving `self` intact. **Collective.**
    pub fn gather(&self) -> Option<PauliSum<W>> {
        #[cfg(feature = "phase-timing")]
        let started = Instant::now();
        let out = gather_share(&self.local.sum, &self.transport);
        #[cfg(feature = "phase-timing")]
        self.lap_gather(started.elapsed().as_nanos() as u64);
        out
    }

    /// This rank's share, a valid [`PauliSum`] under the group's shared hash.
    pub fn local(&self) -> &PauliSum<W> {
        &self.local.sum
    }

    /// This rank's contribution to `⟨ψ|O|ψ⟩` in a uniform single-qubit product state; not collective, so sum the ranks' answers.
    pub fn local_expectation_product_state(&self, state: ProductState) -> Complex64 {
        self.local.sum.expectation_product_state(state)
    }

    /// [`PauliSum::anticommute_histogram`] of the whole sum. **Collective.**
    pub fn anticommute_histogram(&self, sites: &[usize], axis: RotationAxis) -> Vec<f64> {
        let local = &self.local.sum;
        let mut histogram = self
            .runtime
            .install(move || local.anticommute_histogram(sites, axis));
        self.transport.allreduce_sum_f64(&mut histogram);
        histogram
    }

    /// [`PauliSum::rotated_overlap`] of the whole sum without a gather. **Collective.**
    ///
    /// The partition rows must avoid the coordinates the rotation flips; scatter with [`PartitionRowPolicy::SeededExcluding`], or [`Cut`](PartitionRowPolicy::Cut) rows for an `X` axis.
    ///
    /// # Panics
    ///
    /// On every rank, before communicating, if the rows read a flipped coordinate, or as [`PauliSum::rotated_overlap`].
    pub fn rotated_overlap(&self, sites: &[usize], delta: f64, axis: RotationAxis) -> f64 {
        assert!(
            self.rows.keeps_flip_classes(sites, axis),
            "DistributedSum::rotated_overlap: the partition rows read coordinates the rotation \
             flips, so its classes span ranks; scatter with PartitionRowPolicy::SeededExcluding \
             over {axis:?}-axis sites {sites:?}",
        );
        let local = &self.local.sum;
        let mut value = [self
            .runtime
            .install(move || local.rotated_overlap(sites, delta, axis))];
        self.transport.allreduce_sum_f64(&mut value);
        value[0]
    }

    /// Drain this rank's phase counters; `per_partition` has one entry.
    #[cfg(feature = "phase-timing")]
    pub fn take_stats(&mut self) -> PartitionPhaseStats {
        PartitionPhaseStats {
            per_partition: vec![self.local.state.layer.take_stats()],
            scatter_ns: std::mem::take(&mut self.scatter_ns),
            gather_ns: self.gather_ns.swap(0, std::sync::atomic::Ordering::Relaxed),
            layers: std::mem::take(&mut self.layers),
        }
    }

    /// Panic unless this rank's share is well-formed and holds only its own keys; `O(terms)`, local only.
    pub fn assert_invariants(&self) {
        #[cfg(any(test, debug_assertions))]
        self.local.sum.assert_invariants();
        let held = self.local.sum.partition_rank_of_all(&self.rows);
        assert!(
            held.is_none() || held == Some(self.rank()),
            "rank {} holds keys of partition {held:?}",
            self.rank(),
        );
    }
}

/// [`DistributedSum::gather`] over any rank's host share. **Collective.**
pub(crate) fn gather_share<const W: usize, X: Transport>(
    local: &PauliSum<W>,
    transport: &X,
) -> Option<PauliSum<W>> {
    let lens: Vec<u64> = (0..local.num_buckets())
        .map(|b| local.bucket_len(b) as u64)
        .collect();
    let (x, z, coeff) = local.to_arrays();
    let parts: Vec<&[u8]> = vec![
        bytemuck::cast_slice(&lens),
        bytemuck::cast_slice(x.as_flattened()),
        bytemuck::cast_slice(z.as_flattened()),
        bytemuck::cast_slice(&coeff),
    ];
    let all = transport.gather_to_root(parts);
    drop((x, z, coeff));

    all.map(|all| {
        let hash = local.hash().clone();
        let num_qubits = local.num_qubits();
        let parts: Vec<PauliSum<W>> = all
            .into_iter()
            .enumerate()
            .map(|(rank, parts)| decode_rank(&parts, hash.clone(), num_qubits, rank))
            .collect();
        PauliSum::merge_partitions(parts)
    })
}

/// A shape-only fingerprint of the run, since a `Circuit` of trait objects has no canonical encoding.
fn run_fingerprint(
    channels: usize,
    direction: Direction,
    options: PropagateOptions,
    num_qubits: usize,
    w: usize,
) -> u64 {
    // FNV-1a over the fields, in a fixed order.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        hash ^= v;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(channels as u64);
    mix(matches!(direction, Direction::Heisenberg) as u64);
    mix(options.target_bucket_len as u64);
    mix(options.min_buckets as u64);
    mix(num_qubits as u64);
    mix(w as u64);
    hash
}

/// Copy `len` values of `T` out of a possibly unaligned byte view.
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

/// [`decode_column`] for `[u64; W]` rows, which are not `Pod` at a generic `W`.
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

/// Rebuild one rank's share from the [`GATHER_PARTS`] parts it shipped; panics on contradictory lengths.
fn decode_rank<const W: usize>(
    parts: &[Vec<u8>],
    hash: crate::pauli_sum::hash::Gf2Hash<W>,
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
mod tests;
