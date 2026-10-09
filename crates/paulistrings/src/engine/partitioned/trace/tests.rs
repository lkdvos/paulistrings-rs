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
