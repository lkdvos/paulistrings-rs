//! The persistent partitioned driver: `PartitionedSum` held across calls, its
//! accessors, and the runtime shared between sums.

use paulistrings::test_support::{
    approx_eq, assert_runs_agree, assert_terms_close, drop_tiny, low_weight_sum, partition_rows,
    propagate_discarding, rand_sum, spread_magnitudes, unpinned_partitions, KeepAll,
};
use paulistrings::{propagate, Circuit, Direction, PauliAxis, PauliString, PauliSum, ProductBasis};
use paulistrings::{ApproxTopN, CoefficientThreshold};
use paulistrings::{PartitionConfig, PartitionRuntime, PartitionedSum};
use paulistrings::{PartitionRowPolicy, ScatterOptions, ScatterRows};
use paulistrings::{PauliRotation, PtmChannel};

const TOL: f64 = 1e-11;
const NQ: usize = 16;

/// Partition-row seed of every seeded scatter here.
const ROW_SEED: u64 = 0x0BAD_F00D;

fn config(partitions: usize) -> PartitionConfig {
    unpinned_partitions(partitions, 2)
}

/// A short mixed circuit: a Clifford, a weight-2 rotation and a weight-4 one,
/// so both prepared arms and a crossing generator are exercised.
fn circuit() -> Circuit<1> {
    let mut c = Circuit::<1>::new(NQ);
    c.push(PtmChannel::hadamard(0));
    c.push(PtmChannel::cnot(1, 2));
    let mut zz = PauliString::<1> { x: [0], z: [0b110] };
    zz.z[0] |= 1 << 5;
    c.push(PauliRotation::new(zz, 0.31));
    let wide = PauliString::<1> { x: [0b1001], z: [0b0110] };
    c.push(PauliRotation::new(wide, 0.17));
    c.push(PtmChannel::phase_gate(3));
    c
}

/// Two `propagate` calls on a held `PartitionedSum` equal two calls on the
/// unpartitioned sum, and the accessors agree with the gathered sum in between.
#[test]
fn a_held_sum_propagates_across_calls() {
    let runtime = PartitionRuntime::new(&config(4)).expect("topology resolves");
    let circuit = circuit();
    let sum = rand_sum::<1>(400, NQ, 0xD00D);

    let once = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    let twice = propagate(&circuit, once.clone(), &KeepAll, Direction::Forward);

    let mut ps = PartitionedSum::scatter_with(
        sum,
        ScatterOptions {
            runtime: runtime.clone(),
            rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(Some(ROW_SEED))),
        },
    );
    assert_eq!(ps.num_partitions(), 4);
    ps.assert_invariants();

    ps.propagate(&circuit, &KeepAll, Direction::Forward);
    ps.assert_invariants();
    assert_eq!(ps.len(), once.len(), "after one call");
    assert_terms_close(&ps.gather(), &once, TOL, "one call");
    assert!(approx_eq(
        ps.expectation_product(&ProductBasis::uniform(PauliAxis::Z)),
        once.expectation_product(&ProductBasis::uniform(PauliAxis::Z)),
        1e-9,
    ));

    ps.propagate(&circuit, &KeepAll, Direction::Forward);
    ps.assert_invariants();
    assert_eq!(ps.len(), twice.len(), "after two calls");
    let gathered = ps.gather();
    assert_terms_close(&gathered, &twice, TOL, "two calls");
    assert_eq!(gathered.len(), ps.len());
    assert_eq!(gathered.num_qubits(), ps.num_qubits());

    // The parts tile the whole: every key sits in exactly its own partition,
    // and the lengths add up.
    let total: usize = (0..ps.num_partitions()).map(|r| ps.partition(r).len()).sum();
    assert_eq!(total, ps.len());
    assert_eq!(ps.rows().num_partitions(), 4);

    // `into_gathered` is `gather` without the clone.
    assert_eq!(ps.into_gathered().to_arrays(), gathered.to_arrays());
}

/// A second sum on the same `Arc<PartitionRuntime>` reuses the pools.
#[test]
fn a_runtime_serves_more_than_one_sum() {
    let cfg = config(2);
    let runtime = PartitionRuntime::new(&cfg).expect("topology resolves");
    assert_eq!(runtime.num_partitions(), 2);
    let circuit = circuit();

    for seed in [0x1u64, 0x2] {
        let sum = rand_sum::<1>(300, NQ, seed);
        let want =
            propagate(&circuit, sum.clone(), &CoefficientThreshold(1e-10), Direction::Heisenberg);
        let mut ps = PartitionedSum::scatter_with(
            sum,
            ScatterOptions {
                runtime: runtime.clone(),
                rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(Some(ROW_SEED))),
            },
        );
        ps.propagate(&circuit, &CoefficientThreshold(1e-10), Direction::Heisenberg);
        assert_terms_close(&ps.into_gathered(), &want, TOL, "reused runtime");
    }
}

/// Explicit partition rows, crafted so partition 0 holds most of the sum: the
/// driver must not assume balance, and every part must still hold only its own
/// keys.
#[test]
fn explicit_rows_may_be_lopsided() {
    let runtime = PartitionRuntime::new(&config(2)).expect("topology resolves");
    let circuit = circuit();
    // Weight-2 keys over 16 qubits, so few of them touch qubit 13 at all.
    let sum = spread_magnitudes(&low_weight_sum::<1>(400, NQ, 2, 0xBEEF), 40, 0xBEEF);
    let (want, discarded) =
        propagate_discarding(&circuit, sum.clone(), &ApproxTopN(200), Direction::Forward);

    // One row seeing only qubit 13's `x` bit: a term is in partition 1 only if
    // it carries `X` or `Y` there — roughly one term in twelve.
    let rows = partition_rows::<1>(NQ, vec![[1u64 << 13]], vec![[0u64]]);
    let mut ps = PartitionedSum::scatter_with(
        sum,
        ScatterOptions { runtime, rows: ScatterRows::Explicit(rows) },
    );
    ps.assert_invariants();
    assert!(
        ps.partition(0).len() > ps.partition(1).len(),
        "the fixture should be lopsided: {} vs {}",
        ps.partition(0).len(),
        ps.partition(1).len(),
    );

    ps.propagate(&circuit, &ApproxTopN(200), Direction::Forward);
    ps.assert_invariants();
    let got = ps.into_gathered();
    assert_runs_agree(ApproxTopN(200), discarded, &got, &want, TOL, "lopsided rows");
}

/// An empty sum scatters, propagates and gathers.
#[test]
fn an_empty_sum_is_a_valid_partitioned_sum() {
    let mut ps =
        PartitionedSum::scatter(PauliSum::<1>::empty(NQ), &config(4)).expect("topology resolves");
    assert!(ps.is_empty());
    assert_eq!(ps.len(), 0);
    ps.assert_invariants();
    ps.propagate(&circuit(), &KeepAll, Direction::Forward);
    ps.assert_invariants();
    assert!(ps.gather().is_empty());
}

#[test]
fn a_nested_commutator_stays_split_and_agrees_with_one_process() {
    let generator = low_weight_sum::<1>(24, NQ, 2, 0x51);
    let operand = low_weight_sum::<1>(3_000, NQ, 3, 0x52);
    let split =
        PartitionedSum::scatter(operand.clone(), &config(4)).expect("an unpinned config resolves");

    let product = split.product_with(&generator, &KeepAll);
    product.assert_invariants();
    assert_terms_close(
        &drop_tiny(&product.gather()),
        &drop_tiny(&generator.product(&operand)),
        TOL,
        "product",
    );
    let anticommutator = split.anticommutator_with(&generator, &KeepAll);
    assert_terms_close(
        &drop_tiny(&anticommutator.gather()),
        &drop_tiny(&generator.anticommutator(&operand)),
        TOL,
        "anticommutator",
    );

    let policy = CoefficientThreshold(1e-3);
    let (mut want, mut split) = (operand, split);
    for step in 0..3 {
        want = generator.liouvillian_with(&want, &policy);
        split = split.liouvillian_with(&generator, &policy);
        split.assert_invariants();
        assert_terms_close(&split.gather(), &want, TOL, &format!("step {step}"));
    }
}

/// The first `steps` Lanczos coefficients `b_n` of `i[h, ·]` from `seed`, by the recursion `A = L(O_n) − b_n·O_{n−1}`, `b_{n+1} = ‖A‖`; `a_n` vanishes for Hermitian `h` and `O`.
fn lanczos_one_process(h: &PauliSum<1, f64>, seed: &PauliSum<1, f64>, steps: usize) -> Vec<f64> {
    let policy = CoefficientThreshold(1e-10);
    let mut current = seed.clone();
    current.scale(1.0 / current.overlap(&current).sqrt());
    let mut previous = PauliSum::<1, f64>::empty(seed.num_qubits());
    let mut b = vec![0.0];
    for _ in 0..steps {
        let mut next = h.liouvillian_with(&current, &policy);
        let mut scaled = previous.clone();
        scaled.scale(-b[b.len() - 1]);
        next = next.add(&scaled);
        let norm = next.overlap(&next).sqrt();
        next.scale(1.0 / norm);
        b.push(norm);
        previous = std::mem::replace(&mut current, next);
    }
    b.split_off(1)
}

fn chain_and_seed() -> (PauliSum<1, f64>, PauliSum<1, f64>) {
    let mut h = paulistrings::BuildAccumulator::<1, f64>::new(NQ);
    for j in 0..NQ - 1 {
        for f in [PauliString::<1>::x, PauliString::<1>::y, PauliString::<1>::z] {
            let mut p = f(j);
            p.mul_assign(&f(j + 1));
            h.add_term(p, 1.0);
        }
        h.add_term(PauliString::<1>::z(j), 0.3);
    }
    let mut seed = paulistrings::BuildAccumulator::<1, f64>::new(NQ);
    seed.add_term(PauliString::<1>::z(NQ / 2), 1.0);
    (h.finalize(), seed.finalize())
}

#[test]
fn lanczos_coefficients_agree_split_in_process() {
    let (h, seed) = chain_and_seed();
    let want = lanczos_one_process(&h, &seed, 8);
    let policy = CoefficientThreshold(1e-10);
    let mut current =
        PartitionedSum::scatter(seed, &config(4)).expect("an unpinned config resolves");
    current.scale(1.0 / current.overlap(&current).sqrt());
    // Each step overwrites the older vector in place: `O_{n−1} ← L(O_n) − b_n·O_{n−1}`, so only two splits are ever held.
    let mut previous = current.clone();
    let mut got: Vec<f64> = Vec::new();
    for _ in 0..want.len() {
        let beta = -got.last().copied().unwrap_or(0.0);
        previous.add_liouvillian(&h, &current, 1.0, beta, &policy);
        previous.assert_invariants();
        let norm: f64 = previous.overlap(&previous).sqrt();
        previous.scale(1.0 / norm);
        got.push(norm);
        std::mem::swap(&mut previous, &mut current);
    }
    for (n, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-9 * w, "b_{}: {g} vs {w}", n + 1);
    }
}
