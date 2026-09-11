//! [`PartitionTrace`]: the opt-in per-layer record of what a partitioned run did — term counts per partition, bucket bits, and the exchange volume.
//!
//! Always compiled, and off unless [`PartitionedSum::enable_trace`](super::PartitionedSum::enable_trace) is called, the same shape as [`TermTrace`](crate::TermTrace) for the same reason: everything recorded is already computed by the layer, so an untraced layer pays only one register test.
//! Per-partition rows are transposed into per-layer records after the join, so a record shows one layer *across* the group: which partition held how many terms (hence [`PartitionTrace::imbalance`]), and who sent how much to whom.

use super::layer::LayerExchangeCounts;

/// What one layer did, seen across the whole group.
///
/// The per-partition vectors are indexed by rank and are all `P` long.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionLayerRecord {
    /// Bucket bits every partition held for this layer — one number, because the count is agreed collectively before the layer runs.
    pub bits: u8,
    /// Remote deltas the layer had. Also one number: a delta is remote by its mask alone, so every partition reaches the same verdict.
    /// Zero means the layer was purely local and made no transport call.
    pub remote_deltas: u32,
    /// Collective calls the layer issued, not counting the exchange itself (`remote_deltas` reports that): the bucket-count all-reduce, when the schedule called for one, plus whatever the policy's collective finalization ran.
    /// One number like the two above — the schedule is a function of the layer index and the plan, both rank-independent, so every partition issues the same calls.
    pub collectives: u32,
    /// Terms each partition held before the layer.
    pub terms_in: Vec<usize>,
    /// Terms each partition held after the layer, i.e. after `keep_term` and the collective finalization.
    pub terms_out: Vec<usize>,
    /// Rows sent, `rows_sent[from][to]`. The diagonal is always zero — a partition never sends to itself.
    pub rows_sent: Vec<Vec<u64>>,
    /// Wire bytes sent, `bytes_sent[from][to]`, for the same rows.
    pub bytes_sent: Vec<Vec<u64>>,
    /// Rows each partition received, summed over its partners.
    pub rows_received: Vec<u64>,
}

/// One partitioned propagation's per-layer records, in application order (so *reverse* circuit order under [`Direction::Heisenberg`](crate::Direction)).
///
/// Counts accumulate across [`propagate`](super::PartitionedSum::propagate) calls until drained by [`take_trace`](super::PartitionedSum::take_trace).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionTrace {
    /// One record per layer applied.
    pub layers: Vec<PartitionLayerRecord>,
}

impl PartitionTrace {
    /// Rows moved across partitions over the whole trace.
    ///
    /// The traffic figure to divide by the layer count or the term count: each exchanged row is one key plus one coefficient written by the sender and read by the receiver's merge.
    pub fn total_rows_exchanged(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|layer| layer.rows_sent.iter())
            .flat_map(|row| row.iter())
            .sum()
    }

    /// Per layer, the load imbalance of the *input* term counts: the maximum over partitions divided by their mean.
    ///
    /// `1.0` is perfect balance and also the answer for a layer where every partition was empty; `P` is the worst case (one partition holds everything).
    pub fn imbalance(&self) -> Vec<f64> {
        self.layers
            .iter()
            .map(|layer| {
                let total: usize = layer.terms_in.iter().sum();
                let max = layer.terms_in.iter().copied().max().unwrap_or(0);
                if total == 0 {
                    return 1.0;
                }
                let mean = total as f64 / layer.terms_in.len() as f64;
                max as f64 / mean
            })
            .collect()
    }

    /// Layers that exchanged nothing — every delta stayed inside its partition, so the layer made no transport call at all.
    pub fn local_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.remote_deltas == 0)
            .count()
    }

    /// Layers with at least one remote delta, i.e. layers that exchanged.
    ///
    /// `local_layers() + remote_layers()` is the number of layers traced.
    pub fn remote_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.remote_deltas > 0)
            .count()
    }

    /// Collective calls over the whole trace — the figure the per-layer collective schedule exists to hold down (ARCHITECTURE.md §Partitioning).
    ///
    /// Excludes the exchanges themselves, which are point-to-point; [`remote_layers`](Self::remote_layers) counts those.
    pub fn total_collectives(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| u64::from(layer.collectives))
            .sum()
    }
}

/// One layer as a single partition saw it, before the transpose.
///
/// Recorded on the partition's own driving thread into its own `Vec`, so nothing is shared and nothing is synchronized; the group's view is assembled by [`assemble`] after the join.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PartitionLayerRow {
    pub bits: u8,
    pub remote_deltas: u32,
    /// Collectives this partition issued for the layer; see
    /// [`PartitionLayerRecord::collectives`].
    pub collectives: u32,
    pub terms_in: usize,
    pub terms_out: usize,
    /// Rows sent to each partner rank (`P` long, own slot zero).
    pub rows_sent: Vec<u64>,
    /// Wire bytes sent to each partner rank.
    pub bytes_sent: Vec<u64>,
    pub rows_received: u64,
}

/// Append one layer's record to this partition's rows.
///
/// `#[cold]` + `#[inline(never)]` for the same reason as `engine::record_layer_terms`: the layer loop inlines the bucketed layer and, through it, the merge kernels, whose throughput moves by 6-34% under a few bytes of code motion (CLAUDE.md §Performance discipline).
/// The counts arrive by value, so the two `Vec`s the exchange already allocated are moved rather than copied.
#[cold]
#[inline(never)]
pub(crate) fn record_layer_row(
    rows: &mut Vec<PartitionLayerRow>,
    bits: u8,
    collectives: u32,
    terms_in: usize,
    terms_out: usize,
    counts: LayerExchangeCounts,
) {
    rows.push(PartitionLayerRow {
        bits,
        remote_deltas: counts.remote_deltas as u32,
        collectives,
        terms_in,
        terms_out,
        rows_sent: counts.rows_sent,
        bytes_sent: counts.bytes_sent,
        rows_received: counts.rows_received,
    });
}

/// Transposes the partitions' rows into per-layer records and appends them to `trace`.
///
/// # Panics
///
/// If the partitions recorded different numbers of layers, or disagree about a layer's bucket bits or remote-delta count — both are collective decisions, so a disagreement is a driver bug rather than a data-dependent outcome.
pub(crate) fn assemble(trace: &mut PartitionTrace, per_partition: Vec<Vec<PartitionLayerRow>>) {
    let size = per_partition.len();
    let layers = per_partition[0].len();
    for (rank, rows) in per_partition.iter().enumerate() {
        assert_eq!(
            rows.len(),
            layers,
            "partition {rank} traced {} layers, partition 0 traced {layers}",
            rows.len(),
        );
    }

    trace.layers.reserve(layers);
    for k in 0..layers {
        let head = &per_partition[0][k];
        let mut record = PartitionLayerRecord {
            bits: head.bits,
            remote_deltas: head.remote_deltas,
            collectives: head.collectives,
            terms_in: Vec::with_capacity(size),
            terms_out: Vec::with_capacity(size),
            rows_sent: Vec::with_capacity(size),
            bytes_sent: Vec::with_capacity(size),
            rows_received: Vec::with_capacity(size),
        };
        for (rank, rows) in per_partition.iter().enumerate() {
            let row = &rows[k];
            assert_eq!(
                row.bits, head.bits,
                "layer {k}: partition {rank} has {} bucket bits, partition 0 has {}",
                row.bits, head.bits,
            );
            assert_eq!(
                row.remote_deltas, head.remote_deltas,
                "layer {k}: partition {rank} saw {} remote deltas, partition 0 saw {}",
                row.remote_deltas, head.remote_deltas,
            );
            assert_eq!(
                row.collectives, head.collectives,
                "layer {k}: partition {rank} issued {} collectives, partition 0 issued {}",
                row.collectives, head.collectives,
            );
            record.terms_in.push(row.terms_in);
            record.terms_out.push(row.terms_out);
            record.rows_sent.push(row.rows_sent.clone());
            record.bytes_sent.push(row.bytes_sent.clone());
            record.rows_received.push(row.rows_received);
        }
        trace.layers.push(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(bits: u8, remote: u32, terms_in: usize, sent: Vec<u64>) -> PartitionLayerRow {
        let bytes = sent.iter().map(|r| r * 48).collect();
        PartitionLayerRow {
            bits,
            remote_deltas: remote,
            collectives: 1,
            terms_in,
            terms_out: terms_in,
            rows_sent: sent,
            bytes_sent: bytes,
            rows_received: 0,
        }
    }

    fn two_layer_trace() -> PartitionTrace {
        let mut trace = PartitionTrace::default();
        assemble(
            &mut trace,
            vec![
                vec![row(3, 0, 100, vec![0, 0]), row(4, 2, 120, vec![0, 7])],
                vec![row(3, 0, 300, vec![0, 0]), row(4, 2, 280, vec![5, 0])],
            ],
        );
        trace
    }

    #[test]
    fn assemble_transposes_into_per_layer_records() {
        let trace = two_layer_trace();
        assert_eq!(trace.layers.len(), 2);
        assert_eq!(trace.layers[0].bits, 3);
        assert_eq!(trace.layers[0].terms_in, vec![100, 300]);
        assert_eq!(trace.layers[1].bits, 4);
        assert_eq!(trace.layers[1].rows_sent, vec![vec![0, 7], vec![5, 0]]);
        assert_eq!(trace.layers[1].bytes_sent, vec![vec![0, 336], vec![240, 0]]);
    }

    #[test]
    fn totals_and_layer_kinds() {
        let trace = two_layer_trace();
        assert_eq!(trace.total_rows_exchanged(), 12);
        assert_eq!(trace.local_layers(), 1);
        assert_eq!(trace.remote_layers(), 1);
        assert_eq!(
            trace.local_layers() + trace.remote_layers(),
            trace.layers.len()
        );
    }

    #[test]
    fn imbalance_is_max_over_mean() {
        let trace = two_layer_trace();
        let got = trace.imbalance();
        // Layer 0: 100 and 300, mean 200, max 300.
        assert!((got[0] - 1.5).abs() < 1e-12, "{got:?}");
        // Layer 1: 120 and 280, mean 200, max 280.
        assert!((got[1] - 1.4).abs() < 1e-12, "{got:?}");
    }

    #[test]
    fn an_all_empty_layer_is_perfectly_balanced() {
        let mut trace = PartitionTrace::default();
        assemble(
            &mut trace,
            vec![
                vec![row(0, 0, 0, vec![0, 0])],
                vec![row(0, 0, 0, vec![0, 0])],
            ],
        );
        assert_eq!(trace.imbalance(), vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "bucket bits")]
    fn assemble_rejects_disagreeing_bucket_bits() {
        let mut trace = PartitionTrace::default();
        assemble(
            &mut trace,
            vec![
                vec![row(3, 0, 1, vec![0, 0])],
                vec![row(4, 0, 1, vec![0, 0])],
            ],
        );
    }

    #[test]
    #[should_panic(expected = "traced")]
    fn assemble_rejects_a_short_partition() {
        let mut trace = PartitionTrace::default();
        assemble(
            &mut trace,
            vec![
                vec![row(3, 0, 1, vec![0, 0]), row(3, 0, 1, vec![0, 0])],
                vec![row(3, 0, 1, vec![0, 0])],
            ],
        );
    }
}
