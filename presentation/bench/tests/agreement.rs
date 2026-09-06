//! Every baseline must agree with `paulistrings::propagate` to floating-point
//! tolerance (CLAUDE.md §Determinism: equal-key summation order is unspecified,
//! so `assert_terms_close`, never byte equality). Two Trotter steps keep the
//! debug-build runtime reasonable.

use paulistrings::test_support::assert_terms_close;
use paulistrings::truncation::CoefficientThreshold;
use paulistrings::{propagate, Direction, PauliSum};
use presentation_bench::{bucketed, mergesort, naive, threadmaps, workload};

const TOL: f64 = 1e-9;

fn reference(steps: usize, eps: f64) -> (paulistrings::Circuit<{ workload::W }>, PauliSum<{ workload::W }>, PauliSum<{ workload::W }>) {
    let circuit = workload::talk_circuit(&workload::default_edges_path(), steps);
    let obs = workload::z_observable(workload::QUBITS, workload::OBSERVABLE_QUBIT);
    let want = propagate(&circuit, obs.clone(), &CoefficientThreshold(eps), Direction::Heisenberg);
    (circuit, obs, want)
}

#[test]
fn workload_is_pinned() {
    // Committed constants from small_m_ab.rs / term_growth.jsonl: eps 2^-8, 5 steps.
    let (circuit, obs, want) = reference(5, 0.00390625);
    assert_eq!(circuit.len(), 1355);
    assert_eq!(want.len(), 5038, "final terms at eps 2^-8");
    let r = bucketed::run(&circuit, None, &obs, 0.00390625, bucketed::options(1024, 128));
    assert_eq!(r.peak_terms(), 6311, "peak terms at eps 2^-8");
}

#[test]
fn naive_agrees_with_propagate() {
    for eps in [0.015625, 0.00390625] {
        let (circuit, obs, want) = reference(2, eps);
        let got = naive::run(&circuit, &obs, eps).sum;
        assert_terms_close(&got, &want, TOL, "naive vs propagate");
    }
}

#[test]
fn threadmaps_agrees_with_propagate() {
    for eps in [0.015625, 0.00390625] {
        let (circuit, obs, want) = reference(2, eps);
        for t in [1usize, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
            let got = pool.install(|| threadmaps::run(&circuit, &obs, eps, t)).sum;
            assert_terms_close(&got, &want, TOL, "threadmaps vs propagate");
        }
    }
}

#[test]
fn mergesort_agrees_with_propagate() {
    for eps in [0.015625, 0.00390625] {
        let (circuit, obs, want) = reference(2, eps);
        for t in [1usize, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
            let got = pool.install(|| mergesort::run(&circuit, &obs, eps, t)).sum;
            assert_terms_close(&got, &want, TOL, "mergesort vs propagate");
        }
    }
}

#[test]
fn coarse_buckets_agree_and_are_coarser() {
    let eps = 0.0009765625; // 2^-10: 79k peak terms, above the 8192-term bucket floor
    let (circuit, obs, want) = reference(5, eps);
    let fine = bucketed::run(&circuit, None, &obs, eps, bucketed::options(1024, 128));
    let coarse = bucketed::run(&circuit, None, &obs, eps, bucketed::options(16384, 16));
    assert_terms_close(&coarse.sum, &want, TOL, "coarse buckets vs propagate");
    assert!(coarse.buckets.unwrap() < fine.buckets.unwrap(), "coarse {:?} vs fine {:?}", coarse.buckets, fine.buckets);
    let (pf, pc) = (fine.phase.unwrap(), coarse.phase.unwrap());
    assert!(pc.cosets < pf.cosets, "cosets coarse {} vs fine {}", pc.cosets, pf.cosets);
}

#[test]
fn layer_times_match_whole_circuit() {
    let eps = 0.00390625;
    let (circuit, obs, want) = reference(2, eps);
    let layers = workload::talk_layers(&workload::default_edges_path(), 2);
    let r = bucketed::run(&circuit, Some(&layers), &obs, eps, bucketed::options(1024, 128));
    assert_terms_close(&r.sum, &want, TOL, "per-layer drive vs propagate");
    assert_eq!(r.layer_wall_ns.unwrap().len(), circuit.len());
}
