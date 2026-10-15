//! The opt-in per-layer trace: `LayerScratch::enable_trace` on the one-partition layer loop and `PartitionTrace` on a partitioned run.

/// A one-partition `PartitionTrace` written by `propagate_with`'s layer loop.
mod one_partition {
    use num_complex::Complex64;
    use paulistrings::test_support::assert_terms_close;
    use paulistrings::CoefficientThreshold;
    use paulistrings::{
        propagate, propagate_with, BuildAccumulator, Circuit, Direction, LayerScratch,
        PartitionTrace, PauliString, PauliSum, PropagateOptions, TruncationPolicy,
    };
    use paulistrings::{PauliRotation, PtmChannel};

    struct AlwaysKeep;
    impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

    const NUM_QUBITS: usize = 4;

    /// Each layer's input count; a plain run's records hold one partition.
    fn terms_in(trace: &PartitionTrace) -> Vec<usize> {
        trace
            .layers
            .iter()
            .map(|layer| one(&layer.terms_in))
            .collect()
    }

    /// Each layer's output count.
    fn terms_out(trace: &PartitionTrace) -> Vec<usize> {
        trace
            .layers
            .iter()
            .map(|layer| one(&layer.terms_out))
            .collect()
    }

    fn one<T: Copy>(per_partition: &[T]) -> T {
        assert_eq!(per_partition.len(), 1, "a plain run records one partition");
        per_partition[0]
    }
    /// cos 0.3 = 0.955336…, sin 0.3 = 0.295520… — far enough from each other and
    /// from zero that no count below depends on a cancellation.
    const THETA: f64 = 0.3;

    /// `{X₀: 1}`.
    fn x0_observable<const W: usize>() -> PauliSum<W> {
        let mut acc = BuildAccumulator::<W>::with_capacity(NUM_QUBITS, 1);
        acc.add_term(PauliString::<W>::x(0), Complex64::new(1.0, 0.0));
        acc.finalize()
    }

    /// Three layers whose forward term counts on `{X₀: 1}` are hand-computable:
    ///
    /// 1. `exp(-iθZ₀/2)`: `X₀` anticommutes with the generator, so it fans out —
    ///    `cos θ·X₀ − sin θ·Y₀` (sign per the rotation's `i^k` fold; the counts
    ///    do not depend on it). 1 → 2 terms.
    /// 2. `H₀`: `X₀ → Z₀`, `Y₀ → −Y₀`. A relabelling, 2 → 2 terms.
    /// 3. `exp(-iθZ₀/2)` again: `Z₀` commutes with the generator (stays one
    ///    term), `Y₀` anticommutes (fans out to `Y₀` and `X₀`). The three keys
    ///    `Z₀`, `Y₀`, `X₀` are distinct, so 2 → 3 terms.
    fn three_layer_circuit<const W: usize>() -> Circuit<W> {
        let mut c = Circuit::<W>::new(NUM_QUBITS);
        c.push(PauliRotation::new(PauliString::<W>::z(0), THETA));
        c.push(PtmChannel::hadamard(0));
        c.push(PauliRotation::new(PauliString::<W>::z(0), THETA));
        c
    }

    /// A fresh scratch traces nothing: `take_trace` is `None` after a
    /// propagation that never enabled it, so `None` ⟺ "tracing off".
    fn check_off_by_default<const W: usize>() {
        let mut scratch = LayerScratch::<W>::new();
        let out = propagate_with(
            &three_layer_circuit::<W>(),
            x0_observable::<W>(),
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
            PropagateOptions::default(),
        );
        assert_eq!(out.len(), 3);
        assert!(scratch.take_trace().is_none());
    }

    #[test]
    fn trace_is_off_by_default_w1() {
        check_off_by_default::<1>();
    }

    #[test]
    fn trace_is_off_by_default_w2() {
        check_off_by_default::<2>();
    }

    /// Hand-computed counts for the circuit above, and the "peak resident"
    /// definition: `max(terms_in[0], terms_out…)`.
    fn check_hand_computed_counts<const W: usize>() {
        let mut scratch = LayerScratch::<W>::new();
        scratch.enable_trace();
        let out = propagate_with(
            &three_layer_circuit::<W>(),
            x0_observable::<W>(),
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
            PropagateOptions::default(),
        );

        let trace = scratch.take_trace().expect("trace was enabled");
        assert_eq!(terms_in(&trace), vec![1, 2, 2]);
        assert_eq!(terms_out(&trace), vec![2, 2, 3]);
        assert_eq!(trace.peak_terms(), Some(3));
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn trace_records_hand_computed_counts_w1() {
        check_hand_computed_counts::<1>();
    }

    #[test]
    fn trace_records_hand_computed_counts_w2() {
        check_hand_computed_counts::<2>();
    }

    /// The counts are read *after* `policy.finalize_layer`, so they are
    /// post-truncation. Layer 3's third key is `X₀` with coefficient
    /// `sin²θ·cos θ… = 0.0873…` (the two rotations' `sin θ` factors times the
    /// intervening `cos θ`), which a threshold of 0.1 drops; the other two keys
    /// carry 0.9553… (`Z₀`) and 0.2823… (`Y₀`) and survive. So the final layer
    /// reports 2 out, not 3, and the peak drops with it.
    fn check_counts_are_post_truncation<const W: usize>() {
        let mut scratch = LayerScratch::<W>::new();
        scratch.enable_trace();
        let out = propagate_with(
            &three_layer_circuit::<W>(),
            x0_observable::<W>(),
            &CoefficientThreshold(0.1),
            Direction::Forward,
            &mut scratch,
            PropagateOptions::default(),
        );

        let trace = scratch.take_trace().expect("trace was enabled");
        assert_eq!(terms_in(&trace), vec![1, 2, 2]);
        assert_eq!(terms_out(&trace), vec![2, 2, 2]);
        assert_eq!(trace.peak_terms(), Some(2));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn trace_counts_are_post_truncation_w1() {
        check_counts_are_post_truncation::<1>();
    }

    #[test]
    fn trace_counts_are_post_truncation_w2() {
        check_counts_are_post_truncation::<2>();
    }

    /// A zero-layer circuit records nothing, and "peak" is undefined from the
    /// trace alone (the caller knows the resident count — it never changed).
    #[test]
    fn trace_of_an_empty_circuit_is_empty() {
        let mut scratch = LayerScratch::<1>::new();
        scratch.enable_trace();
        let out = propagate_with(
            &Circuit::<1>::new(NUM_QUBITS),
            x0_observable::<1>(),
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
            PropagateOptions::default(),
        );
        assert_eq!(out.len(), 1);

        let trace = scratch.take_trace().expect("trace was enabled");
        assert!(trace.layers.is_empty());
        assert_eq!(trace.peak_terms(), None);
    }

    /// Taking drains the counts but leaves tracing *on*, so a scratch reused
    /// across `propagate_with` calls (a Trotter driver) reports each
    /// call separately without re-enabling.
    #[test]
    fn taking_the_trace_drains_but_stays_enabled() {
        let circuit = three_layer_circuit::<1>();
        let mut scratch = LayerScratch::<1>::new();
        scratch.enable_trace();

        for _ in 0..2 {
            let _ = propagate_with(
                &circuit,
                x0_observable::<1>(),
                &AlwaysKeep,
                Direction::Forward,
                &mut scratch,
                PropagateOptions::default(),
            );
            let trace = scratch.take_trace().expect("still enabled");
            assert_eq!(terms_out(&trace), vec![2, 2, 3]);
        }
        // Nothing ran in between, so the third take sees an empty — but present —
        // trace.
        let trace = scratch.take_trace().expect("still enabled");
        assert!(trace.layers.is_empty());
    }

    /// Tracing is bookkeeping on the calling thread: the propagated sum is the
    /// same one `propagate` (which never enables the trace) produces.
    fn check_trace_does_not_change_the_result<const W: usize>() {
        let circuit = three_layer_circuit::<W>();
        let want = propagate(
            &circuit,
            x0_observable::<W>(),
            &AlwaysKeep,
            Direction::Forward,
        );

        let mut scratch = LayerScratch::<W>::new();
        scratch.enable_trace();
        let got = propagate_with(
            &circuit,
            x0_observable::<W>(),
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
            PropagateOptions::default(),
        );
        assert_terms_close(&got, &want, 1e-15, "traced vs untraced propagate");
    }

    #[test]
    fn trace_does_not_change_the_result_w1() {
        check_trace_does_not_change_the_result::<1>();
    }

    #[test]
    fn trace_does_not_change_the_result_w2() {
        check_trace_does_not_change_the_result::<2>();
    }

    /// Two single-qubit gates on `{Z₀: 1}`, each keeping one term.
    fn two_gate_circuit() -> Circuit<1> {
        let mut circuit = Circuit::<1>::new(1);
        circuit.push(PtmChannel::hadamard(0));
        circuit.push(PtmChannel::phase_gate(0));
        circuit
    }

    fn z0_observable() -> PauliSum<1> {
        let mut accumulator = BuildAccumulator::<1>::with_capacity(1, 1);
        accumulator.add_term(PauliString::<1>::z(0), Complex64::new(1.0, 0.0));
        accumulator.finalize()
    }

    #[test]
    fn trace_forward_indices_match_circuit_order() {
        let mut scratch = LayerScratch::<1>::new();
        scratch.enable_trace();
        let _ = propagate_with(
            &two_gate_circuit(),
            z0_observable(),
            &AlwaysKeep,
            Direction::Forward,
            &mut scratch,
            PropagateOptions::default(),
        );
        let trace = scratch.take_trace().unwrap();

        let field = |f: fn(&paulistrings::PartitionLayerRecord) -> u32| -> Vec<u32> {
            trace.layers.iter().map(f).collect()
        };
        assert_eq!(field(|layer| layer.application_index), vec![0, 1]);
        assert_eq!(field(|layer| layer.circuit_index), vec![0, 1]);
        assert_eq!(field(|layer| layer.remote_deltas), vec![0, 0]);
        assert_eq!(terms_in(&trace), vec![1, 1]);
        assert_eq!(terms_out(&trace), vec![1, 1]);
        for layer in &trace.layers {
            assert_eq!(layer.nanos.len(), 1);
            assert_eq!(layer.rows_sent, vec![vec![0]]);
            assert_eq!(layer.rows_received, vec![0]);
        }
    }

    #[test]
    fn trace_heisenberg_reverses_circuit_index_not_application_index() {
        let mut scratch = LayerScratch::<1>::new();
        scratch.enable_trace();
        let _ = propagate_with(
            &two_gate_circuit(),
            z0_observable(),
            &AlwaysKeep,
            Direction::Heisenberg,
            &mut scratch,
            PropagateOptions::default(),
        );
        let trace = scratch.take_trace().unwrap();

        // Applied in loop order 0, 1, but the circuit ran channel 1 (`s`) first.
        let application: Vec<u32> = trace.layers.iter().map(|l| l.application_index).collect();
        let circuit: Vec<u32> = trace.layers.iter().map(|l| l.circuit_index).collect();
        assert_eq!(application, vec![0, 1]);
        assert_eq!(circuit, vec![1, 0]);
    }
}

/// The record of a partitioned run.
mod partitioned {
    use paulistrings::test_support::desired_bits;
    use paulistrings::test_support::{
        low_weight_sum, partition_rows, rand_sum, unpinned_partitions, unpinned_scatter, KeepAll,
    };
    use paulistrings::{Circuit, Direction, PartitionRows, PauliString, PropagateOptions};
    use paulistrings::{PartitionConfig, PartitionRuntime, PartitionedSum};
    use paulistrings::{PauliRotation, PtmChannel};
    use paulistrings::{ScatterOptions, ScatterRows};

    const NQ: usize = 16;

    fn config(partitions: usize) -> PartitionConfig {
        unpinned_partitions(partitions, 2)
    }

    /// A mixed circuit: two Cliffords (whose non-identity delta may or may not
    /// cross), a weight-2 rotation and a weight-4 one.
    fn circuit() -> Circuit<1> {
        let mut c = Circuit::<1>::new(NQ);
        c.push(PtmChannel::hadamard(0));
        c.push(PtmChannel::cnot(1, 2));
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
        partition_rows::<1>(NQ, vec![[1u64]], vec![[0u64]])
    }

    #[test]
    fn tracing_is_off_by_default() {
        let mut ps = PartitionedSum::scatter_with(
            rand_sum::<1>(200, NQ, 0x1),
            unpinned_scatter(2, 2, 0x7_ACED_7ACE),
        );
        ps.propagate(&circuit(), &KeepAll, Direction::Forward);
        assert!(ps.take_trace().is_none(), "no trace unless asked for");
    }

    /// Every layer gets a record, the per-partition vectors are `P` long, the
    /// diagonal of `rows_sent` is zero, and draining leaves tracing on.
    #[test]
    fn a_trace_records_every_layer() {
        let circuit = circuit();
        let mut ps = PartitionedSum::scatter_with(
            rand_sum::<1>(600, NQ, 0x2),
            unpinned_scatter(4, 2, 0x7_ACED_7ACE),
        );
        ps.enable_trace();
        ps.enable_trace(); // idempotent
        ps.propagate(&circuit, &KeepAll, Direction::Forward);

        let trace = ps.take_trace().expect("tracing is on");
        assert_eq!(trace.layers.len(), circuit.len());
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
            // A host partition receives in chunks on a remote layer only and carries no device record.
            for r in 0..4 {
                assert_eq!(
                    layer.receive_chunks[r] > 0,
                    layer.remote_deltas > 0,
                    "layer {k} partition {r}"
                );
                assert_eq!(layer.device[r], None, "layer {k} partition {r}");
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
            circuit.len(),
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

        let mut ps = PartitionedSum::scatter_with(
            sum,
            ScatterOptions {
                runtime,
                rows: ScatterRows::Explicit(rows_seeing_qubit_0_x()),
            },
        );
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
        use paulistrings::PtmChannel;

        let mut circuit = Circuit::<1>::new(NQ);
        circuit.push(PtmChannel::depolarizing(3, 0.1));
        let mut ps = PartitionedSum::scatter_with(
            rand_sum::<1>(200, NQ, 0x4),
            unpinned_scatter(2, 2, 0x7_ACED_7ACE),
        );
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
        let mut ps = PartitionedSum::scatter_with(
            rand_sum::<1>(20_000, NQ, 0x5),
            unpinned_scatter(4, 2, 0x7_ACED_7ACE),
        );
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
        let mut ps = PartitionedSum::scatter_with(
            sum,
            ScatterOptions {
                runtime,
                rows: ScatterRows::Explicit(rows_seeing_qubit_0_x()),
            },
        );
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
        };
        let before = ps.bits();
        ps.enable_trace();
        ps.propagate_with(&circuit(), &KeepAll, Direction::Forward, options);
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
            // A host partition refines one doubling per pass.
            assert!(
                layer
                    .refine_passes
                    .iter()
                    .all(|&passes| passes == u32::from(layer.bits - prev)),
                "layer {k}: {:?} refine passes from {prev} to {} bits",
                layer.refine_passes,
                layer.bits
            );
            grew |= layer.bits > prev;
            prev = layer.bits;
        }
        assert!(grew, "the fixture should have grown the bucket count");
        assert_eq!(ps.bits(), prev);
    }

    /// The trace's gate-identity fields (`circuit_index`, `application_index`,
    /// `gate_name`) agree across the group by construction (every partition runs
    /// the same channel in lock-step), run `0..n` under `Direction::Forward`, and
    /// `nanos` has one rank-local timing per partition.
    #[test]
    fn gate_identity_and_timing_are_recorded_per_layer() {
        let circuit = circuit();
        let mut ps = PartitionedSum::scatter_with(
            rand_sum::<1>(600, NQ, 0x4),
            unpinned_scatter(4, 2, 0x7_ACED_7ACE),
        );
        ps.enable_trace();
        ps.propagate(&circuit, &KeepAll, Direction::Forward);
        let trace = ps.take_trace().expect("tracing is on");

        assert_eq!(trace.layers.len(), circuit.len());
        for (k, layer) in trace.layers.iter().enumerate() {
            assert_eq!(layer.circuit_index, k as u32, "layer {k}");
            assert_eq!(layer.application_index, k as u32, "layer {k}");
            assert!(!layer.gate_name.is_empty(), "layer {k}: empty gate name");
            assert_eq!(layer.nanos.len(), 4, "layer {k}: one timing per partition");
        }
    }

    /// Under `Direction::Heisenberg`, `application_index` still runs `0..n` in
    /// application order, but `circuit_index` is reversed — the same contract the
    /// unpartitioned engine's one-partition trace gives, since a distributed consumer
    /// needs it to avoid mislabeling a Heisenberg trace against the circuit as
    /// written.
    #[test]
    fn heisenberg_reverses_circuit_index_not_application_index() {
        let circuit = circuit();
        let n = circuit.len();
        let mut ps = PartitionedSum::scatter_with(
            rand_sum::<1>(200, NQ, 0x5),
            unpinned_scatter(2, 2, 0x7_ACED_7ACE),
        );
        ps.enable_trace();
        ps.propagate(&circuit, &KeepAll, Direction::Heisenberg);
        let trace = ps.take_trace().expect("tracing is on");

        for (k, layer) in trace.layers.iter().enumerate() {
            assert_eq!(layer.application_index, k as u32, "layer {k}");
            assert_eq!(layer.circuit_index, (n - 1 - k) as u32, "layer {k}");
        }
    }
}
