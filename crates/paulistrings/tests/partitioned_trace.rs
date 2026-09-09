//! `PartitionTrace`: the opt-in per-layer record of a partitioned run.

use paulistrings::bucket::desired_bits;
use paulistrings::channel::{Clifford1Q, Clifford2Q, PauliRotation};
use paulistrings::engine::partitioned::{PartitionConfig, PartitionRuntime, PartitionedSum};
use paulistrings::test_support::{low_weight_sum, rand_sum, unpinned_partitions, KeepAll};
use paulistrings::{Circuit, Direction, PartitionRows, PauliString, PropagateOptions};

const NQ: usize = 16;

fn config(partitions: usize) -> PartitionConfig {
    unpinned_partitions(partitions, 2, 0x7_ACED_7ACE)
}

/// A mixed circuit: two Cliffords (whose non-identity delta may or may not
/// cross), a weight-2 rotation and a weight-4 one.
fn circuit() -> Circuit<1> {
    let mut c = Circuit::<1>::new(NQ);
    c.push(Clifford1Q::h(0));
    c.push(Clifford2Q::cnot(1, 2));
    c.push(PauliRotation::new(
        PauliString::<1> {
            x: [0],
            z: [0b100110],
        },
        0.31,
    ));
    c.push(PauliRotation::new(wide_gen(), 0.17));
    c
}

/// `X₀ Z₁ X₂ Z₃` — weight 4, so `prepare` takes the `Prepared::Rotation` arm.
fn wide_gen() -> PauliString<1> {
    PauliString::<1> {
        x: [0b0101],
        z: [0b1010],
    }
}

/// One row seeing the `x` bit of qubit 0, which [`wide_gen`] carries: the
/// generator pass is remote at every rank.
fn rows_seeing_qubit_0_x() -> PartitionRows<1> {
    PartitionRows::<1>::from_rows(NQ, vec![[1u64]], vec![[0u64]])
}

#[test]
fn tracing_is_off_by_default() {
    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter(rand_sum::<1>(200, NQ, 0x1), runtime, &config(2));
    ps.propagate(&circuit(), &KeepAll, Direction::Forward);
    assert!(ps.take_trace().is_none(), "no trace unless asked for");
}

/// Every layer gets a record, the per-partition vectors are `P` long, the
/// diagonal of `rows_sent` is zero, and draining leaves tracing on.
#[test]
fn a_trace_records_every_layer() {
    let runtime = PartitionRuntime::new(&config(4)).expect("topology resolves");
    let circuit = circuit();
    let mut ps = PartitionedSum::scatter(rand_sum::<1>(600, NQ, 0x2), runtime, &config(4));
    ps.enable_trace();
    ps.enable_trace(); // idempotent
    ps.propagate(&circuit, &KeepAll, Direction::Forward);

    let trace = ps.take_trace().expect("tracing is on");
    assert_eq!(trace.layers.len(), circuit.channels.len());
    assert_eq!(
        trace.local_layers() + trace.remote_layers(),
        trace.layers.len(),
    );
    assert!(
        trace.remote_layers() > 0,
        "these rows should make some delta cross: {trace:?}",
    );
    for (k, layer) in trace.layers.iter().enumerate() {
        assert_eq!(layer.terms_in.len(), 4, "layer {k}");
        assert_eq!(layer.terms_out.len(), 4, "layer {k}");
        assert_eq!(layer.rows_received.len(), 4, "layer {k}");
        assert_eq!(layer.rows_sent.len(), 4, "layer {k}");
        for (r, row) in layer.rows_sent.iter().enumerate() {
            assert_eq!(row.len(), 4, "layer {k} partition {r}");
            assert_eq!(row[r], 0, "layer {k}: partition {r} sent to itself");
            assert_eq!(layer.bytes_sent[r][r], 0, "layer {k} partition {r} bytes");
        }
        // Sent and received rows balance across the group.
        let sent: u64 = layer.rows_sent.iter().flat_map(|r| r.iter()).sum();
        let received: u64 = layer.rows_received.iter().sum();
        assert_eq!(
            sent, received,
            "layer {k}: sent {sent} but received {received}"
        );
        // A purely local layer moves nothing; a remote one moves something for
        // a sum this dense.
        if layer.remote_deltas == 0 {
            assert_eq!(sent, 0, "layer {k} is local but sent rows");
        }
    }

    // Draining leaves tracing enabled with an empty record set.
    assert_eq!(
        ps.take_trace(),
        Some(paulistrings::PartitionTrace::default()),
    );
    ps.propagate(&circuit, &KeepAll, Direction::Forward);
    assert_eq!(
        ps.take_trace().expect("still tracing").layers.len(),
        circuit.channels.len(),
    );
}

/// A single wide rotation whose generator crosses: exactly the terms that
/// anticommute with the generator are exported, one row each.
#[test]
fn total_rows_exchanged_counts_the_anticommuting_terms() {
    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let gen = wide_gen();
    let mut circuit = Circuit::<1>::new(NQ);
    circuit.push(PauliRotation::new(gen, 0.3));

    let sum = rand_sum::<1>(500, NQ, 0x3);
    let anticommuting = sum
        .iter()
        .filter(|(x, z, _)| !PauliString::<1> { x: **x, z: **z }.commutes_with(&gen))
        .count() as u64;
    assert!(anticommuting > 0 && anticommuting < sum.len() as u64);

    let mut ps = PartitionedSum::scatter_with_rows(sum, rows_seeing_qubit_0_x(), runtime);
    ps.enable_trace();
    ps.propagate(&circuit, &KeepAll, Direction::Forward);
    let trace = ps.take_trace().expect("tracing is on");

    assert_eq!(trace.layers.len(), 1);
    assert_eq!(trace.remote_layers(), 1);
    assert_eq!(trace.layers[0].remote_deltas, 1, "only the generator pass");
    assert_eq!(trace.total_rows_exchanged(), anticommuting);
}

/// A key-preserving layer crosses nothing and makes no transport call.
#[test]
fn a_key_preserving_layer_is_local() {
    use paulistrings::channel::Depolarizing;

    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let mut circuit = Circuit::<1>::new(NQ);
    circuit.push(Depolarizing {
        support: [3],
        p: 0.1,
    });
    let mut ps = PartitionedSum::scatter(rand_sum::<1>(200, NQ, 0x4), runtime, &config(2));
    ps.enable_trace();
    ps.propagate(&circuit, &KeepAll, Direction::Forward);
    let trace = ps.take_trace().expect("tracing is on");
    assert_eq!(trace.local_layers(), 1);
    assert_eq!(trace.remote_layers(), 0);
    assert_eq!(trace.total_rows_exchanged(), 0);
}

/// Random rows on a big dense sum balance the split to within a few percent.
#[test]
fn imbalance_is_near_one_for_random_rows() {
    let runtime = PartitionRuntime::new(&config(4)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter(rand_sum::<1>(20_000, NQ, 0x5), runtime, &config(4));
    ps.enable_trace();
    ps.propagate(&circuit(), &KeepAll, Direction::Forward);
    let trace = ps.take_trace().expect("tracing is on");
    for (k, imbalance) in trace.imbalance().iter().enumerate() {
        assert!(
            (1.0..1.15).contains(imbalance),
            "layer {k}: imbalance {imbalance} is not near 1 — {:?}",
            trace.layers[k].terms_in,
        );
    }
}

/// The bucket count is collective: equal on every partition (asserted by the
/// trace's own assembly), never falling, and always the maximum over
/// partitions of what each share wants.
#[test]
fn bits_are_uniform_and_grow_only() {
    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    // Lopsided on purpose: weight-2 keys over 16 qubits rarely carry `X` on
    // qubit 0, so partition 0 holds the bulk and the `max` is what decides.
    let sum = low_weight_sum::<1>(3_000, NQ, 2, 0x6);
    let mut ps = PartitionedSum::scatter_with_rows(sum, rows_seeing_qubit_0_x(), runtime);
    assert!(
        ps.partition(0).len() > 2 * ps.partition(1).len(),
        "fixture should be lopsided: {} vs {}",
        ps.partition(0).len(),
        ps.partition(1).len(),
    );

    // A fine bucket policy, so the count actually moves over four layers.
    let options = PropagateOptions {
        target_bucket_len: 32,
        min_buckets: 16,
        ..PropagateOptions::default()
    };
    let before = ps.bits();
    ps.enable_trace();
    ps.propagate_with_options(&circuit(), &KeepAll, Direction::Forward, options);
    let trace = ps.take_trace().expect("tracing is on");

    let mut prev = before;
    let mut grew = false;
    for (k, layer) in trace.layers.iter().enumerate() {
        let want = layer
            .terms_in
            .iter()
            .map(|&len| desired_bits(len, options.target_bucket_len, options.min_buckets))
            .max()
            .expect("at least one partition")
            .max(prev);
        assert_eq!(
            layer.bits, want,
            "layer {k}: bits {} but the group wanted {want} from {:?}",
            layer.bits, layer.terms_in,
        );
        assert!(layer.bits >= prev, "layer {k}: bits fell from {prev}");
        grew |= layer.bits > prev;
        prev = layer.bits;
    }
    assert!(grew, "the fixture should have grown the bucket count");
    assert_eq!(ps.bits(), prev);
}
