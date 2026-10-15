use super::*;

fn row(bits: u8, remote: u32, terms_in: usize, sent: Vec<u64>) -> PartitionLayerRow {
    let bytes = sent.iter().map(|r| r * 48).collect();
    PartitionLayerRow {
        bits,
        remote_deltas: remote,
        collectives: 1,
        circuit_index: 0,
        application_index: 0,
        gate_name: "channel",
        terms_in,
        terms_out: terms_in,
        rows_sent: sent,
        bytes_sent: bytes,
        rows_received: 0,
        nanos: 0,
        refine_passes: 0,
        receive_chunks: 0,
        device: None,
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
        2,
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

/// The shared per-partition counters and the device sub-record transpose like the others.
#[test]
fn assemble_carries_the_refine_receive_and_device_columns() {
    let device = DeviceLayerRecord {
        records: 12,
        permuted: true,
        ..DeviceLayerRecord::default()
    };
    let host = PartitionLayerRow {
        refine_passes: 2,
        ..row(3, 1, 10, vec![0, 1])
    };
    let on_device = PartitionLayerRow {
        receive_chunks: 4,
        device: Some(device),
        ..row(3, 1, 10, vec![1, 0])
    };
    let mut trace = PartitionTrace::default();
    assemble(&mut trace, vec![vec![host], vec![on_device]], 2);
    let layer = &trace.layers[0];
    assert_eq!(layer.refine_passes, vec![2, 0]);
    assert_eq!(layer.receive_chunks, vec![0, 4]);
    assert_eq!(layer.device, vec![None, Some(device)]);
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
        2,
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
        2,
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
        2,
    );
}

#[test]
fn assemble_pads_a_local_layers_per_partner_vectors() {
    let mut trace = PartitionTrace::default();
    assemble(
        &mut trace,
        vec![vec![row(3, 0, 1, vec![])], vec![row(3, 0, 1, vec![])]],
        2,
    );
    assert_eq!(trace.layers[0].rows_sent, vec![vec![0, 0], vec![0, 0]]);
    assert_eq!(trace.layers[0].bytes_sent, vec![vec![0, 0], vec![0, 0]]);
}

fn counts(terms_in: Vec<usize>, terms_out: Vec<usize>) -> PartitionLayerRecord {
    PartitionLayerRecord {
        terms_in,
        terms_out,
        ..PartitionLayerRecord::default()
    }
}

/// `peak_terms` is the between-layer maximum of the summed counts, which includes the first layer's input.
#[test]
fn peak_terms_spans_the_first_input_and_every_output() {
    assert_eq!(PartitionTrace::default().peak_terms(), None);
    let shrinking = PartitionTrace {
        layers: vec![counts(vec![9], vec![4]), counts(vec![4], vec![6])],
    };
    assert_eq!(
        shrinking.peak_terms(),
        Some(9),
        "a shrinking first layer keeps the input as the peak"
    );
    let growing = PartitionTrace {
        layers: vec![counts(vec![1], vec![5]), counts(vec![5], vec![3])],
    };
    assert_eq!(growing.peak_terms(), Some(5));
    let split = PartitionTrace {
        layers: vec![
            counts(vec![3, 4], vec![5, 1]),
            counts(vec![5, 1], vec![2, 2]),
        ],
    };
    assert_eq!(split.peak_terms(), Some(7), "summed over partitions");
}
