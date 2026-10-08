//! [`PartitionTrace`], the opt-in per-layer record of a partitioned run, and the per-partition rows it is transposed from.

use super::layer::LayerExchangeCounts;

/// What one layer did across the whole group; the per-partition vectors are indexed by rank.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionLayerRecord {
    /// Bucket bits every partition held for this layer.
    pub bits: u8,
    /// Remote deltas the layer had; zero means it made no transport call.
    pub remote_deltas: u32,
    /// Collective calls the layer issued besides the exchange: the scheduled bucket-count all-reduce plus the policy's collective finalization.
    pub collectives: u32,
    /// This layer's position in the [`Circuit`](crate::Circuit), independent of direction.
    pub circuit_index: u32,
    /// This layer's position in the propagation loop.
    pub application_index: u32,
    /// [`Channel::debug_name`](crate::Channel::debug_name) of the applied channel.
    pub gate_name: &'static str,
    /// Terms each partition held before the layer.
    pub terms_in: Vec<usize>,
    /// Terms each partition held after the layer and its finalization.
    pub terms_out: Vec<usize>,
    /// Rows sent, `rows_sent[from][to]`.
    pub rows_sent: Vec<Vec<u64>>,
    /// Wire bytes sent, `bytes_sent[from][to]`, for the same rows.
    pub bytes_sent: Vec<Vec<u64>>,
    /// Rows each partition received, summed over its partners.
    pub rows_received: Vec<u64>,
    /// Each partition's own wall time for the layer; ranks are not synchronized mid-layer, so `max` is only a critical-rank proxy.
    pub nanos: Vec<u64>,
}

/// Per-layer records of partitioned propagations in application order, accumulated until drained by [`take_trace`](crate::PartitionedSum::take_trace).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartitionTrace {
    /// One record per layer applied.
    pub layers: Vec<PartitionLayerRecord>,
}

impl PartitionTrace {
    /// Rows moved across partitions over the whole trace.
    pub fn total_rows_exchanged(&self) -> u64 {
        self.layers
            .iter()
            .flat_map(|layer| layer.rows_sent.iter())
            .flat_map(|row| row.iter())
            .sum()
    }

    /// Per layer, the maximum input term count over partitions divided by the mean; `1.0` for a layer where every partition was empty.
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

    /// Layers that made no transport call.
    pub fn local_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.remote_deltas == 0)
            .count()
    }

    /// Layers that exchanged.
    pub fn remote_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.remote_deltas > 0)
            .count()
    }

    /// Collective calls over the whole trace, excluding the point-to-point exchanges.
    pub fn total_collectives(&self) -> u64 {
        self.layers
            .iter()
            .map(|layer| u64::from(layer.collectives))
            .sum()
    }
}

/// One layer as a single partition saw it, before [`assemble`] transposes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PartitionLayerRow {
    pub bits: u8,
    pub remote_deltas: u32,
    pub collectives: u32,
    pub circuit_index: u32,
    pub application_index: u32,
    pub gate_name: &'static str,
    pub terms_in: usize,
    pub terms_out: usize,
    pub rows_sent: Vec<u64>,
    pub bytes_sent: Vec<u64>,
    pub rows_received: u64,
    pub nanos: u64,
}

/// Append one layer's record; cold and out of line to keep it from perturbing the inlined merge kernels' layout.
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_layer_row(
    rows: &mut Vec<PartitionLayerRow>,
    bits: u8,
    collectives: u32,
    circuit_index: u32,
    application_index: u32,
    gate_name: &'static str,
    terms_in: usize,
    terms_out: usize,
    counts: LayerExchangeCounts,
    nanos: u64,
) {
    rows.push(PartitionLayerRow {
        bits,
        remote_deltas: counts.remote_deltas as u32,
        collectives,
        circuit_index,
        application_index,
        gate_name,
        terms_in,
        terms_out,
        rows_sent: counts.rows_sent,
        bytes_sent: counts.bytes_sent,
        rows_received: counts.rows_received,
        nanos,
    });
}

/// Transposes the partitions' rows into per-layer records appended to `trace`, panicking if the partitions disagree on a collective decision.
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
            circuit_index: head.circuit_index,
            application_index: head.application_index,
            gate_name: head.gate_name,
            terms_in: Vec::with_capacity(size),
            terms_out: Vec::with_capacity(size),
            rows_sent: Vec::with_capacity(size),
            bytes_sent: Vec::with_capacity(size),
            rows_received: Vec::with_capacity(size),
            nanos: Vec::with_capacity(size),
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
            assert_eq!(
                row.circuit_index, head.circuit_index,
                "layer {k}: partition {rank} ran circuit index {}, partition 0 ran {}",
                row.circuit_index, head.circuit_index,
            );
            assert_eq!(
                row.application_index, head.application_index,
                "layer {k}: partition {rank} ran application index {}, partition 0 ran {}",
                row.application_index, head.application_index,
            );
            assert_eq!(
                row.gate_name, head.gate_name,
                "layer {k}: partition {rank} ran gate {:?}, partition 0 ran {:?}",
                row.gate_name, head.gate_name,
            );
            record.terms_in.push(row.terms_in);
            record.terms_out.push(row.terms_out);
            record.rows_sent.push(row.rows_sent.clone());
            record.bytes_sent.push(row.bytes_sent.clone());
            record.rows_received.push(row.rows_received);
            record.nanos.push(row.nanos);
        }
        trace.layers.push(record);
    }
}

#[cfg(test)]
mod tests;
