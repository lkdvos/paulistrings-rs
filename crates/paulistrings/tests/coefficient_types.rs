//! Propagation in every coefficient type against the `Complex64` reference: real and single-precision sums agree with it to the type's tolerance, and a real sum refuses a channel that does not preserve Hermiticity before any layer runs.

use num_complex::{Complex32, Complex64};
use paulistrings::test_support::{
    assert_terms_close, coefficient_tolerance, non_hermitian_circuit, rand_sum, rand_sum_real,
    random_circuit, trotter_circuit, unpinned_partitions, KeepAll,
};
use paulistrings::{
    propagate, propagate_partitioned, BuildAccumulator, Circuit, Coefficient, Direction, PauliAxis,
    PauliString, PauliSum, ProductBasis, PropagateOptions,
};

fn agrees_with_reference<const W: usize, C: Coefficient>(circuit: &Circuit<W>, sum: &PauliSum<W>) {
    for direction in [Direction::Heisenberg, Direction::Forward] {
        let want = propagate(circuit, sum.clone(), &KeepAll, direction);
        let narrowed = sum.cast::<C>(0.0).expect("a real input sum");
        let got = propagate(circuit, narrowed, &KeepAll, direction).cast(0.0).unwrap();
        assert_terms_close(&got, &want, coefficient_tolerance::<C>(), C::NAME);
    }
}

fn every_type_agrees<const W: usize>(circuit: &Circuit<W>, sum: &PauliSum<W>) {
    assert!(circuit.preserves_hermiticity());
    agrees_with_reference::<W, f64>(circuit, sum);
    agrees_with_reference::<W, f32>(circuit, sum);
    agrees_with_reference::<W, Complex64>(circuit, sum);
    agrees_with_reference::<W, Complex32>(circuit, sum);
}

#[test]
fn every_channel_class_agrees_in_every_type() {
    for seed in 0..4 {
        every_type_agrees(
            &random_circuit::<1>(6, 40, seed, true),
            &rand_sum_real::<1>(100, 6, seed),
        );
    }
    every_type_agrees(&random_circuit::<2>(100, 25, 0, false), &rand_sum_real::<2>(400, 100, 0));
    // Past one bucket, so the refine and the coset loop run in every type.
    every_type_agrees(&trotter_circuit::<1>(8, 0.3), &rand_sum_real::<1>(50, 8, 7));
}

#[test]
fn a_partitioned_single_precision_run_agrees_with_the_complex_reference() {
    let circuit = random_circuit::<1>(7, 60, 11, true);
    let sum = rand_sum_real::<1>(300, 7, 11);
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Heisenberg);
    for partitions in [2, 4] {
        let config = unpinned_partitions(partitions, 2);
        let narrowed = sum.cast::<f32>(0.0).unwrap();
        let options = PropagateOptions::default();
        let got = propagate_partitioned(
            &circuit,
            narrowed,
            &KeepAll,
            Direction::Heisenberg,
            &config,
            options,
        );
        assert_terms_close(&got.unwrap().cast(0.0).unwrap(), &want, 2e-5, "f32 partitioned");
    }
}

#[test]
fn readouts_and_casts_keep_the_coefficient_type() {
    // ⟨0|(0.5·Z₀ − 0.25·X₁ + 2·I)|0⟩ = 0.5 + 2 = 2.5.
    let mut accumulator = BuildAccumulator::<1, f32>::new(2);
    accumulator.add_term(PauliString::z(0), 0.5);
    accumulator.add_term(PauliString::x(1), -0.25);
    accumulator.add_term(PauliString::identity(), 2.0);
    let sum = accumulator.finalize();
    assert_eq!(sum.expectation_product(&ProductBasis::uniform(PauliAxis::Z)), 2.5f32);
    assert_eq!(sum.identity_coefficient(), 2.0f32);

    let complex = rand_sum::<1>(50, 8, 1);
    assert!(complex.cast::<f64>(1e-12).is_none());
    let back = complex.cast::<Complex32>(0.0).unwrap().cast::<Complex64>(0.0).unwrap();
    assert_terms_close(&back, &complex, 1e-6, "Complex32 round trip");
}

fn x1<C: Coefficient>(c: C) -> PauliSum<1, C> {
    let mut accumulator = BuildAccumulator::<1, C>::new(4);
    accumulator.add_term(PauliString::x(1), c);
    accumulator.finalize()
}

#[test]
fn a_complex_sum_carries_a_non_hermitian_channel() {
    assert!(!non_hermitian_circuit().preserves_hermiticity());
    let out = propagate(&non_hermitian_circuit(), x1(Complex64::ONE), &KeepAll, Direction::Forward);
    assert_eq!(out.get(&PauliString::<1>::x(1).x, &[0]), Some(Complex64::i()));
}

#[test]
#[should_panic(expected = "needs every channel to preserve Hermiticity")]
fn a_real_sum_refuses_a_non_hermitian_channel() {
    let _ = propagate(&non_hermitian_circuit(), x1(1.0f64), &KeepAll, Direction::Forward);
}
