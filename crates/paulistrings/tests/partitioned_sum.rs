//! The persistent partitioned driver: `PartitionedSum` held across calls, its
//! accessors, and the runtime shared between sums.

use paulistrings::test_support::{
    approx_eq, assert_terms_close, low_weight_sum, rand_sum, unpinned_partitions, KeepAll,
};
use paulistrings::{
    propagate, Circuit, Direction, PartitionRows, PauliString, PauliSum, ProductState,
};
use paulistrings::{ApproxTopN, CoefficientThreshold};
use paulistrings::{Clifford1Q, Clifford2Q, PauliRotation};
use paulistrings::{PartitionConfig, PartitionRuntime, PartitionedSum};
use paulistrings::{PartitionRowPolicy, ScatterOptions, ScatterRows};

const TOL: f64 = 1e-11;
const NQ: usize = 16;

fn config(partitions: usize) -> PartitionConfig {
    unpinned_partitions(partitions, 2, 0x0BAD_F00D)
}

/// A short mixed circuit: a Clifford, a weight-2 rotation and a weight-4 one,
/// so both prepared arms and a crossing generator are exercised.
fn circuit() -> Circuit<1> {
    let mut c = Circuit::<1>::new(NQ);
    c.push(Clifford1Q::h(0));
    c.push(Clifford2Q::cnot(1, 2));
    let mut zz = PauliString::<1> { x: [0], z: [0b110] };
    zz.z[0] |= 1 << 5;
    c.push(PauliRotation::new(zz, 0.31));
    let wide = PauliString::<1> {
        x: [0b1001],
        z: [0b0110],
    };
    c.push(PauliRotation::new(wide, 0.17));
    c.push(Clifford1Q::s(3));
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
            rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(config(4).partition_row_seed)),
        },
    );
    assert_eq!(ps.num_partitions(), 4);
    ps.assert_invariants();

    ps.propagate(&circuit, &KeepAll, Direction::Forward);
    ps.assert_invariants();
    assert_eq!(ps.len(), once.len(), "after one call");
    assert_terms_close(&ps.gather(), &once, TOL, "one call");
    assert!(approx_eq(
        ps.expectation_product_state(ProductState::ZPlus),
        once.expectation_product_state(ProductState::ZPlus),
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
    let total: usize = (0..ps.num_partitions())
        .map(|r| ps.partition(r).len())
        .sum();
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
    assert_eq!(runtime.slots().len(), 2);
    let circuit = circuit();

    for seed in [0x1u64, 0x2] {
        let sum = rand_sum::<1>(300, NQ, seed);
        let want = propagate(
            &circuit,
            sum.clone(),
            &CoefficientThreshold(1e-10),
            Direction::Heisenberg,
        );
        let mut ps = PartitionedSum::scatter_with(
            sum,
            ScatterOptions {
                runtime: runtime.clone(),
                rows: ScatterRows::Policy(PartitionRowPolicy::Seeded(cfg.partition_row_seed)),
            },
        );
        ps.propagate(
            &circuit,
            &CoefficientThreshold(1e-10),
            Direction::Heisenberg,
        );
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
    let sum = low_weight_sum::<1>(400, NQ, 2, 0xBEEF);
    let want = propagate(&circuit, sum.clone(), &ApproxTopN(200), Direction::Forward);

    // One row seeing only qubit 13's `x` bit: a term is in partition 1 only if
    // it carries `X` or `Y` there — roughly one term in twelve.
    let rows = PartitionRows::<1>::from_rows(NQ, vec![[1u64 << 13]], vec![[0u64]]);
    let mut ps = PartitionedSum::scatter_with(
        sum,
        ScatterOptions {
            runtime,
            rows: ScatterRows::Explicit(rows),
        },
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
    assert_terms_close(&ps.into_gathered(), &want, TOL, "lopsided rows");
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

/// The echo read-outs over partitions: gather-free when the rows exclude the coordinates `V` flips, through a gather otherwise, and the same answer either way.
#[test]
fn echo_read_outs_agree_with_or_without_excluded_rows() {
    use paulistrings::test_support::rand_sum_on;
    use paulistrings::RotationAxis;

    let runtime = PartitionRuntime::new(&config(4)).expect("topology resolves");
    let sum = rand_sum_on::<1>(500, NQ, &[0, 1, 2, 3, 5, 6], 0xEC42);
    let evolved = propagate(&circuit(), sum.clone(), &KeepAll, Direction::Heisenberg);
    let sites = [1usize, 2, 5];
    for axis in [RotationAxis::Z, RotationAxis::X] {
        let want = evolved.rotated_overlap(&sites, 0.3, axis);
        let want_hist = evolved.anticommute_histogram(&sites, axis);
        let (mx, mz) = axis.flip_mask::<1>(&sites);
        for rows in [
            PartitionRows::<1>::from_seed_excluding(NQ, 2, 0xEC43, &mx, &mz),
            PartitionRows::<1>::from_seed(NQ, 2, 0xEC43),
        ] {
            let excluded = rows.avoids(&mx, &mz);
            let mut ps = PartitionedSum::scatter_with(
                sum.clone(),
                ScatterOptions {
                    runtime: runtime.clone(),
                    rows: ScatterRows::Explicit(rows),
                },
            );
            ps.propagate(&circuit(), &KeepAll, Direction::Heisenberg);
            let got = ps.rotated_overlap(&sites, 0.3, axis);
            assert!(
                (got - want).abs() < 1e-10,
                "{axis:?} excluded={excluded}: {got} vs {want}"
            );
            for (h, w) in ps
                .anticommute_histogram(&sites, axis)
                .iter()
                .zip(&want_hist)
            {
                assert!((h - w).abs() < 1e-10, "{axis:?} excluded={excluded}");
            }
        }
    }
}
