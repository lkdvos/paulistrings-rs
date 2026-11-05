//! The partitioned engine's collective schedule (`engine/driver.rs`): the bucket count is agreed within each remote layer's exchange group and once at the end of a call, never on a local layer.
//!
//! Every fixture truncates by weight, a per-term filter with no collective form, which also bounds the key space so 128-layer debug runs stay affordable.

use paulistrings::test_support::rebucketed_bits;
use paulistrings::test_support::{
    assert_terms_close, rand_sum_real, unpinned_partitions, zz_rotation,
};
use paulistrings::PauliRotation;
use paulistrings::WeightCutoff;
use paulistrings::{propagate, Circuit, Direction, PartitionRows, PauliString, PropagateOptions};
use paulistrings::{PartitionConfig, PartitionRuntime, PartitionedSum};
use paulistrings::{ScatterOptions, ScatterRows};

const NQ: usize = 32;
const TOL: f64 = 1e-11;
/// Fine enough that the bucket count tracks the term count instead of sitting on the `min_buckets` floor.
fn fine() -> PropagateOptions {
    PropagateOptions { target_bucket_len: 32, min_buckets: 16 }
}

fn config(partitions: usize) -> PartitionConfig {
    unpinned_partitions(partitions, 2)
}

/// The chain bisected: `[0, 16)` and `[16, 32)`. A cut row is z-only, so a
/// term's partition is the parity of its z-weight in the second block — every
/// transverse-field rotation is local, and a `ZZ` bond is remote exactly when
/// it crosses.
fn cut_rows() -> PartitionRows<1> {
    PartitionRows::<1>::cut(NQ, &[(0..16).collect::<Vec<_>>(), (16..32).collect()])
}

/// One TFIM Trotter step on a periodic chain: `NQ` `ZZ` bonds then `NQ`
/// transverse-field rotations, `2·NQ = 64` layers, of which the two bonds
/// `15–16` and `31–0` cross [`cut_rows`].
fn ring_step(circuit: &mut Circuit<1>, theta: f64) {
    for q in 0..NQ {
        circuit.push(zz_rotation::<1>(q, (q + 1) % NQ, 2.0 * theta));
    }
    for q in 0..NQ {
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

/// A run of exchanging and local layers agrees the bucket count on the exchanging ones alone.
#[test]
fn exchange_free_layers_are_collective_free() {
    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter_with(
        rand_sum_real::<1>(600, NQ, 0xC01),
        ScatterOptions { runtime, rows: ScatterRows::Explicit(cut_rows()) },
    );
    ps.enable_trace();
    ps.propagate_with(&steps(2, 0.1), &WeightCutoff(3), Direction::Forward, fine());
    let trace = ps.take_trace().expect("tracing is on");
    assert_eq!(trace.remote_layers(), 4, "two crossing bonds per step");
    for (k, layer) in trace.layers.iter().enumerate() {
        assert_eq!(layer.collectives, u32::from(layer.remote_deltas > 0), "layer {k}");
    }
}

/// `P = 1` has no group to agree with, so the driver takes no collective at
/// all — and keeps rebucketing every layer, as `propagate` does.
#[test]
fn one_partition_takes_no_collective_and_still_rebuckets() {
    let circuit = steps(2, 0.1);
    let runtime = PartitionRuntime::new(&config(1)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter_with(
        rand_sum_real::<1>(400, NQ, 0xC02),
        ScatterOptions { runtime, rows: ScatterRows::Explicit(PartitionRows::<1>::none(NQ)) },
    );
    let before = ps.bits();
    ps.enable_trace();
    ps.propagate_with(&circuit, &WeightCutoff(5), Direction::Forward, fine());
    let trace = ps.take_trace().expect("tracing is on");

    assert_eq!(trace.total_collectives(), 0, "P = 1 has nobody to reduce with");
    // Exactly `propagate`'s own rule, every layer, with no lag because there is nothing to lag behind.
    let mut prev = before;
    let mut grew = false;
    for (k, layer) in trace.layers.iter().enumerate() {
        let want =
            rebucketed_bits(layer.terms_in[0], prev, fine().target_bucket_len, fine().min_buckets);
        assert_eq!(
            layer.bits[0], want,
            "layer {k}: P = 1 must rebucket every layer ({} terms in)",
            layer.terms_in[0],
        );
        grew |= layer.bits[0] > prev;
        prev = layer.bits[0];
    }
    assert!(grew, "the fixture should have grown the bucket count");
}

/// A few low-weight terms buried in many heavy ones: the first layer's weight cutoff drops the heavy ones, so the sum shrinks far below its bucket count.
fn collapsing_sum() -> paulistrings::PauliSum<1> {
    let light = paulistrings::test_support::low_weight_sum::<1>(300, NQ, 2, 0xC0A);
    let heavy = rand_sum_real::<1>(20_000, NQ, 0xC0B);
    light.add(&heavy)
}

/// Once the sum shrinks, `P = 1` coarsens back to what its length wants, by the same rule every layer.
#[test]
fn one_partition_coarsens_once_the_sum_collapses() {
    let circuit = steps(1, 0.1);
    let runtime = PartitionRuntime::new(&config(1)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter_with(
        collapsing_sum(),
        ScatterOptions { runtime, rows: ScatterRows::Explicit(PartitionRows::<1>::none(NQ)) },
    );
    let mut prev = ps.bits();
    ps.enable_trace();
    ps.propagate_with(&circuit, &WeightCutoff(3), Direction::Forward, fine());
    let trace = ps.take_trace().expect("tracing is on");

    let mut coarsened = false;
    for (k, layer) in trace.layers.iter().enumerate() {
        let want =
            rebucketed_bits(layer.terms_in[0], prev, fine().target_bucket_len, fine().min_buckets);
        assert_eq!(layer.bits[0], want, "layer {k}: {} terms in", layer.terms_in[0]);
        coarsened |= layer.bits[0] < prev;
        prev = layer.bits[0];
    }
    assert!(coarsened, "the fixture should have coarsened the bucket count");
}

/// A group coarsens together, at an agreement, and the gathered result still matches `propagate`.
#[test]
fn a_group_coarsens_together_and_still_matches_propagate() {
    let circuit = steps(1, 0.1);
    let policy = WeightCutoff(3);
    let sum = collapsing_sum();
    let want = propagate(&circuit, sum.clone(), &policy, Direction::Forward);
    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let mut ps = PartitionedSum::scatter_with(
        sum,
        ScatterOptions { runtime, rows: ScatterRows::Explicit(cut_rows()) },
    );
    ps.enable_trace();
    ps.propagate_with(&circuit, &policy, Direction::Forward, fine());
    let trace = ps.take_trace().expect("tracing is on");
    let peak = trace.layers.iter().flat_map(|layer| layer.bits.clone()).max().expect("layers ran");
    assert!(ps.bits() < peak, "bits stayed at the peak {peak}");
    let got = ps.into_gathered();
    assert_terms_close(&got, &want, TOL, "P=2 after coarsening");
    assert_eq!(got.len(), want.len(), "term count");
}

/// Two calls on one split agree with `propagate` at every partition count and in both directions, each returning with one bucket count on every partition.
#[test]
fn two_calls_match_propagate_and_return_one_count() {
    let circuit = steps(2, 0.1);
    let sum = rand_sum_real::<1>(400, NQ, 0xC04);
    let policy = WeightCutoff(3);
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let once = propagate(&circuit, sum.clone(), &policy, direction);
        let twice = propagate(&circuit, once.clone(), &policy, direction);
        for &p in &[2usize, 4, 8] {
            let blocks: Vec<Vec<usize>> =
                (0..p).map(|b| (b * NQ / p..(b + 1) * NQ / p).collect()).collect();
            let runtime = PartitionRuntime::new(&config(p)).expect("topology resolves");
            let mut ps = PartitionedSum::scatter_with(
                sum.clone(),
                ScatterOptions {
                    runtime,
                    rows: ScatterRows::Explicit(PartitionRows::<1>::cut(NQ, &blocks)),
                },
            );
            for (call, want) in [&once, &twice].into_iter().enumerate() {
                let what = format!("P={p} {direction:?} call {call}");
                ps.propagate_with(&circuit, &policy, direction, fine());
                let bits: Vec<u8> = (0..p).map(|r| ps.partition(r).hash().bits()).collect();
                assert!(bits.iter().all(|&b| b == bits[0]), "{what}: bits {bits:?} after the call");
                let got = ps.gather();
                assert_terms_close(&got, want, TOL, &what);
                assert_eq!(got.len(), want.len(), "{what}: term count");
            }
        }
    }
}
