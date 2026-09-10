//! The partitioned engine's **per-layer collective schedule**: which layers
//! talk to the group at all.
//!
//! A layer's exchange is conditional already (no remote delta ⇒ no transport
//! call). These tests pin the other half of the contract, ARCHITECTURE.md
//! §Partitioning: the bucket-count agreement is on a *schedule*, so a run of
//! exchange-free layers is also collective-free, and an exchange is always
//! preceded by an agreement, so both sides index their blocks by the same
//! bucket count.
//!
//! Every fixture truncates by weight. `WeightCutoff` is a per-term filter — no
//! collective form, so the policy contributes nothing to the counts here — and
//! it bounds the key space outright, which keeps a 128-layer debug-build run
//! affordable.

use num_complex::Complex64;
use paulistrings::bucket::desired_bits;
use paulistrings::channel::PauliRotation;
use paulistrings::engine::partitioned::{
    PartitionConfig, PartitionRuntime, PartitionTrace, PartitionedSum, BITS_AGREE_EVERY,
};
use paulistrings::test_support::{
    assert_terms_close, rand_sum_real, unpinned_partitions, zz_rotation,
};
use paulistrings::truncation::WeightCutoff;
use paulistrings::{
    propagate, BuildAccumulator, Circuit, Direction, PartitionRows, PauliString, Phase,
    PropagateOptions,
};

const NQ: usize = 32;
const TOL: f64 = 1e-11;
/// Fine enough that the bucket count tracks the term count instead of sitting
/// on the `min_buckets` floor — which is what makes the *lag* observable.
fn fine() -> PropagateOptions {
    PropagateOptions {
        target_bucket_len: 32,
        min_buckets: 16,
        ..PropagateOptions::default()
    }
}

fn config(partitions: usize) -> PartitionConfig {
    unpinned_partitions(partitions, 2, 0x0C0F_FEE0_1234_5678)
}

/// The chain bisected: `[0, 16)` and `[16, 32)`. A cut row is z-only, so a
/// term's partition is the parity of its z-weight in the second block — every
/// transverse-field rotation is local, and a `ZZ` bond is remote exactly when
/// it crosses.
fn cut_rows() -> PartitionRows<1> {
    PartitionRows::<1>::cut(NQ, &[(0..16u32).collect::<Vec<_>>(), (16..32u32).collect()])
}

/// One TFIM Trotter step on a periodic chain: `NQ` `ZZ` bonds then `NQ`
/// transverse-field rotations, `2·NQ = 64` layers, of which the two bonds
/// `15–16` and `31–0` cross [`cut_rows`].
fn ring_step(circuit: &mut Circuit<1>, theta: f64) {
    for q in 0..NQ as u32 {
        circuit.push(zz_rotation::<1>(q, (q + 1) % NQ as u32, 2.0 * theta));
    }
    for q in 0..NQ as u32 {
        circuit.push(PauliRotation::new(PauliString::<1>::x(q), 2.0 * theta));
    }
}

fn steps(n: usize, theta: f64) -> Circuit<1> {
    let mut circuit = Circuit::<1>::new(NQ);
    for _ in 0..n {
        ring_step(&mut circuit, theta);
    }
    circuit
}

/// Collectives the schedule allows over `layers` layers of which `remote`
/// exchanged: the opening ramp, one per period after it, and one per remote
/// layer.
fn bound(layers: usize, remote: usize) -> usize {
    BITS_AGREE_EVERY + layers.div_ceil(BITS_AGREE_EVERY) + remote
}

fn total(trace: &PartitionTrace) -> usize {
    trace.total_collectives() as usize
}

/// **The point of the schedule**: a long run of exchange-free layers costs a
/// handful of collectives, not one per layer.
#[test]
fn exchange_free_layers_are_collective_free() {
    let circuit = steps(2, 0.1);
    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let mut ps =
        PartitionedSum::scatter_with_rows(rand_sum_real::<1>(600, NQ, 0xC01), cut_rows(), runtime);
    ps.enable_trace();
    ps.propagate_with_options(&circuit, &WeightCutoff(3), Direction::Forward, fine());
    let trace = ps.take_trace().expect("tracing is on");

    let layers = trace.layers.len();
    let remote = trace.remote_layers();
    assert_eq!(layers, 4 * NQ);
    assert_eq!(remote, 4, "two crossing bonds per step, not {remote}");

    let got = total(&trace);
    assert!(
        got <= bound(layers, remote),
        "{got} collectives over {layers} layers ({remote} remote); the schedule allows {}",
        bound(layers, remote),
    );
    // And the win is real, not marginal: fewer than one collective per four
    // layers, where the unconditional schedule spent one per layer.
    assert!(
        got * 4 < layers,
        "{got} collectives over {layers} layers is not a schedule",
    );

    // Every layer that did talk to the group either exchanged or was on the
    // fixed layer-index schedule — never "because my own share grew".
    for (k, layer) in trace.layers.iter().enumerate() {
        assert!(
            layer.collectives == 0
                || layer.remote_deltas > 0
                || k < BITS_AGREE_EVERY
                || k % BITS_AGREE_EVERY == 0,
            "layer {k} took a collective off the schedule",
        );
    }
}

/// `P = 1` has no group to agree with, so the driver takes no collective at
/// all — and keeps rebucketing every layer, which is what keeps the run
/// bitwise identical to `propagate` (`propagate_partitioned`'s own tripwire).
#[test]
fn one_partition_takes_no_collective_and_still_rebuckets() {
    let circuit = steps(2, 0.1);
    let runtime = PartitionRuntime::new(&config(1)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter_with_rows(
        rand_sum_real::<1>(400, NQ, 0xC02),
        PartitionRows::<1>::none(NQ),
        runtime,
    );
    let before = ps.bits();
    ps.enable_trace();
    ps.propagate_with_options(&circuit, &WeightCutoff(5), Direction::Forward, fine());
    let trace = ps.take_trace().expect("tracing is on");

    assert_eq!(total(&trace), 0, "P = 1 has nobody to reduce with");
    // Exactly `propagate`'s own rule, every layer: the running maximum of
    // `desired_bits`. No lag, because there is nothing to lag behind.
    let mut prev = before;
    let mut grew = false;
    for (k, layer) in trace.layers.iter().enumerate() {
        let want = desired_bits(
            layer.terms_in[0],
            fine().target_bucket_len,
            fine().min_buckets,
        )
        .max(prev);
        assert_eq!(
            layer.bits, want,
            "layer {k}: P = 1 must rebucket every layer ({} terms in)",
            layer.terms_in[0],
        );
        grew |= layer.bits > prev;
        prev = layer.bits;
    }
    assert!(grew, "the fixture should have grown the bucket count");
}

/// A start whose terms all live in the first block, plus a handful in the
/// second: the two partitions are lopsided, and local layers keep them that
/// way (a z-only cut row is blind to an `X` rotation, so no term ever changes
/// owner).
fn lopsided_start() -> paulistrings::PauliSum<1> {
    let mut acc = BuildAccumulator::<1>::with_capacity(NQ, 260);
    for q in 0..16u32 {
        for r in (q + 1)..16u32 {
            let mut p = PauliString::<1>::z(q);
            p.z[0] |= 1u64 << r;
            acc.add_term(p, Phase::ONE, Complex64::new(1.0 / (1 + q + r) as f64, 0.0));
        }
    }
    // Four terms on the far side of the cut: odd z-weight in `[16, 32)`.
    for q in 16..20u32 {
        acc.add_term(
            PauliString::<1>::z(q),
            Phase::ONE,
            Complex64::new(0.25, 0.0),
        );
    }
    acc.finalize()
}

/// Between two agreements the partitions' own `desired_bits` may diverge — and
/// neither is allowed to act on it. The layer that finally crosses agrees
/// first, so both sides index the blocks by the same bucket count; the
/// receiver's `num_buckets` debug assert is the tripwire, and a debug build
/// runs it.
#[test]
fn a_remote_layer_after_a_long_local_run_agrees_first() {
    // 40 transverse-field layers — all local under a z-only cut row, and long
    // enough to leave the opening ramp behind — then the one crossing bond.
    let mut circuit = Circuit::<1>::new(NQ);
    for k in 0..40u32 {
        circuit.push(PauliRotation::new(PauliString::<1>::x(k % NQ as u32), 0.37));
    }
    circuit.push(zz_rotation::<1>(15, 16, 0.41));

    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter_with_rows(lopsided_start(), cut_rows(), runtime);
    ps.enable_trace();
    ps.propagate_with_options(&circuit, &WeightCutoff(3), Direction::Forward, fine());
    let trace = ps.take_trace().expect("tracing is on");

    // The fixture has to actually exercise the lag: a layer past the ramp
    // where the two partitions want different bucket counts, and the count
    // they ran under is not what one of them wanted.
    let lagged = trace.layers.iter().enumerate().any(|(k, layer)| {
        let wants: Vec<u8> = layer
            .terms_in
            .iter()
            .map(|&n| desired_bits(n, fine().target_bucket_len, fine().min_buckets))
            .collect();
        k >= BITS_AGREE_EVERY && wants[0] != wants[1] && layer.collectives == 0
    });
    assert!(
        lagged,
        "fixture must reach an off-schedule layer where the partitions disagree about bits: {:?}",
        trace
            .layers
            .iter()
            .map(|l| l.terms_in.clone())
            .collect::<Vec<_>>(),
    );

    // The crossing layer is the last one, and it agreed before it exchanged.
    let last = trace.layers.last().expect("layers");
    assert!(last.remote_deltas > 0, "the last layer must cross");
    assert!(
        last.collectives > 0,
        "a remote layer must agree the bucket count before it exchanges",
    );
    assert!(last.rows_received.iter().sum::<u64>() > 0, "rows must move");
}

/// The lag changes bucket counts, and bucket counts are not the answer: the
/// gathered result still agrees with `propagate` to tolerance at every
/// partition count.
#[test]
fn the_lagged_schedule_still_matches_propagate() {
    let circuit = steps(2, 0.1);
    let sum = rand_sum_real::<1>(400, NQ, 0xC04);
    let policy = WeightCutoff(3);
    let quarters: Vec<Vec<u32>> = (0..4)
        .map(|b| (b * 8..(b + 1) * 8).collect::<Vec<u32>>())
        .collect();
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let want = propagate(&circuit, sum.clone(), &policy, direction);
        for &p in &[2usize, 4] {
            let rows = if p == 2 {
                cut_rows()
            } else {
                PartitionRows::<1>::cut(NQ, &quarters)
            };
            let runtime = PartitionRuntime::new(&config(p)).expect("topology resolves");
            let mut ps = PartitionedSum::scatter_with_rows(sum.clone(), rows, runtime);
            ps.propagate(&circuit, &policy, direction);
            let got = ps.into_gathered();
            let what = format!("P={p} {direction:?}");
            assert_terms_close(&got, &want, TOL, &what);
            assert_eq!(got.len(), want.len(), "{what}: term count");
        }
    }
}
