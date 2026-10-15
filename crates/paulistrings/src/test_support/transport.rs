//! Partitioned and distributed fixtures: the logging transport, partition rows, unpinned placements and the keep-everything policy.

use super::*;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

/// `a` and `b` as a [`BuiltinTruncation`](crate::truncation::BuiltinTruncation) `And` node.
pub fn and(
    a: crate::truncation::BuiltinTruncation,
    b: crate::truncation::BuiltinTruncation,
) -> crate::truncation::BuiltinTruncation {
    crate::truncation::BuiltinTruncation::And(Box::new(a), Box::new(b))
}

/// `a` or `b` as a [`BuiltinTruncation`](crate::truncation::BuiltinTruncation) `Or` node.
pub fn or(
    a: crate::truncation::BuiltinTruncation,
    b: crate::truncation::BuiltinTruncation,
) -> crate::truncation::BuiltinTruncation {
    crate::truncation::BuiltinTruncation::or(a, b)
}

/// Every transport call one rank issued, in order.
#[derive(Clone, Default)]
pub struct CallLog(std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>);

impl CallLog {
    fn push(&self, call: &'static str) {
        self.0.lock().unwrap().push(call);
    }

    /// The calls so far, clearing the log.
    pub fn take(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }

    /// How many of the calls so far were `call`.
    pub fn count(&self, call: &str) -> usize {
        self.0
            .lock()
            .unwrap()
            .iter()
            .filter(|&&c| c == call)
            .count()
    }
}

/// An [`InProcessTransport`](crate::engine::partitioned::in_process::InProcessTransport) rank that logs every collective and exchange it is asked for.
pub struct LoggingTransport {
    inner: crate::engine::partitioned::in_process::InProcessTransport,
    /// This rank's calls.
    pub log: CallLog,
}

impl LoggingTransport {
    /// A group of `size` ranks, each waiting up to two minutes on its partners.
    pub fn group(size: u32) -> Vec<Self> {
        crate::engine::partitioned::in_process::InProcessTransport::group_with_timeout(
            size,
            std::time::Duration::from_secs(120),
        )
        .into_iter()
        .map(|inner| Self {
            inner,
            log: CallLog::default(),
        })
        .collect()
    }
}

impl crate::collectives::sealed::Sealed for LoggingTransport {}

impl crate::collectives::Collectives for LoggingTransport {
    fn rank(&self) -> u32 {
        self.inner.rank()
    }
    fn size(&self) -> u32 {
        self.inner.size()
    }
    fn allreduce_max_u8(&self, v: u8) -> u8 {
        self.log.push("allreduce_max_u8");
        self.inner.allreduce_max_u8(v)
    }
    fn allreduce_sum_u64(&self, buf: &mut [u64]) {
        self.log.push("allreduce_sum_u64");
        self.inner.allreduce_sum_u64(buf);
    }
    fn allreduce_sum_f64(&self, buf: &mut [f64]) {
        self.log.push("allreduce_sum_f64");
        self.inner.allreduce_sum_f64(buf);
    }
    fn barrier(&self) {
        self.log.push("barrier");
        self.inner.barrier();
    }
}

impl crate::engine::transport::Transport for LoggingTransport {
    fn exchange_layer<P, F, R>(
        &self,
        send: Vec<Option<P>>,
        spare: &mut Vec<P>,
        map: &crate::engine::transport::ChunkMap,
        body: F,
    ) -> (Vec<Option<P>>, R)
    where
        P: crate::engine::transport::Payload,
        F: FnOnce(&[Option<P>], &dyn crate::engine::transport::ChunkWait) -> R,
    {
        self.log.push("exchange_layer");
        self.inner.exchange_layer(send, spare, map, body)
    }
    fn exchange<P: crate::engine::transport::Payload>(
        &self,
        send: Vec<Option<P>>,
        spare: &mut Vec<P>,
    ) -> Vec<Option<P>> {
        self.log.push("exchange");
        self.inner.exchange(send, spare)
    }
}

/// Keep every term, with no layer finalization at all.
///
/// [`TruncationPolicy::finalizes_layer`] defaults to `true`, which a partitioned run rejects ([`TruncationPolicy::supports_partitioned`]) — a policy with no layer pass has to say so explicitly.
#[derive(Clone, Copy, Debug)]
pub struct KeepAll;

impl<const W: usize> TruncationPolicy<W> for KeepAll {
    fn finalizes_layer(&self) -> bool {
        false
    }
}

impl From<KeepAll> for crate::truncation::BuiltinTruncation {
    fn from(_: KeepAll) -> Self {
        Self::Keep
    }
}

/// `n` distinct terms carrying `X₀` and the identity on qubit 63: all on rank 0 under [`rows_reading_z63`], and every one anticommutes with `Z₀Z₆₃`, so a `ZZ(0, 63)` layer exports all `n` to rank 1.
pub fn x0_terms_identity_on_q63(n: usize, seed: u64) -> PauliSum<1> {
    let base = rand_sum::<1>(n, 64, seed);
    let mut acc = BuildAccumulator::<1>::new(64);
    for (x, z, c) in base.iter() {
        let p = PauliString::<1> {
            x: [(x[0] | 1) & !(1u64 << 63)],
            z: [z[0] & !(1u64 << 63)],
        };
        acc.add_term(p, c);
    }
    let out = acc.finalize();
    assert_eq!(out.len(), n, "fixture: the terms must stay distinct");
    let zz = PauliString::<1> {
        x: [0],
        z: [1 | (1u64 << 63)],
    };
    assert!(
        out.iter()
            .all(|(x, z, _)| !PauliString::<1> { x: *x, z: *z }.commutes_with(&zz)),
        "fixture: every term must anticommute with Z₀Z₆₃ or the layer exports fewer than {n} rows"
    );
    out
}

/// [`PartitionRows`](crate::PartitionRows) from explicit rows, masked to the live columns, for tests that pin the split by hand.
///
/// # Panics
///
/// As the crate-private `PartitionRows::from_rows`: mismatched lengths, more than `P_MAX_BITS` rows, or a row that masks to zero.
pub fn partition_rows<const W: usize>(
    num_qubits: usize,
    rows_x: Vec<[u64; W]>,
    rows_z: Vec<[u64; W]>,
) -> crate::PartitionRows<W> {
    crate::PartitionRows::from_rows(num_qubits, rows_x, rows_z)
}

/// Partition rows reading `Z₆₃`: `ZZ(0, 63)` is remote and every [`x0_terms_identity_on_q63`] term sits on rank 0.
pub fn rows_reading_z63() -> crate::pauli_sum::hash::PartitionRows<1> {
    crate::pauli_sum::hash::PartitionRows::<1>::from_rows(64, vec![[0u64]], vec![[1u64 << 63]])
}

/// A partitioned placement with no placement: `partitions` unpinned pools of `threads` workers each.
/// Every partitioned test uses this rather than `Placement::Auto`, so the suite runs on a one-node box or a `taskset`ed CI container.
pub fn unpinned_partitions(
    partitions: usize,
    threads: usize,
) -> crate::engine::partitioned::PartitionConfig {
    crate::engine::partitioned::PartitionConfig {
        placement: crate::engine::partitioned::Placement::Unpinned { partitions },
        bind_memory: false,
        threads_per_partition: Some(threads),
    }
}

/// [`unpinned_partitions`] as `scatter_with` options, drawing partition rows from `row_seed`.
pub fn unpinned_scatter<const W: usize>(
    partitions: usize,
    threads: usize,
    row_seed: u64,
) -> crate::engine::partitioned::ScatterOptions<W> {
    use crate::engine::partitioned::{PartitionRowPolicy, PartitionRuntime, ScatterRows};
    crate::engine::partitioned::ScatterOptions {
        runtime: PartitionRuntime::new(&unpinned_partitions(partitions, threads))
            .expect("an unpinned placement resolves"),
        rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(Some(row_seed))),
    }
}
