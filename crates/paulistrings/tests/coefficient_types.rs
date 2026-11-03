//! Propagation in every coefficient type against the `Complex64` reference: real and single-precision sums agree with it to the type's tolerance, and a real sum refuses a channel that does not preserve Hermiticity before any layer runs.

use num_complex::{Complex32, Complex64};
use paulistrings::test_support::{
    assert_terms_close, rand_sum, rand_sum_real, random_circuit, trotter_circuit,
    unpinned_partitions, KeepAll,
};
use paulistrings::{
    propagate, propagate_partitioned, BuildAccumulator, Channel, Circuit, Coefficient,
    CoefficientThreshold, Direction, OutputBuffer, PauliAxis, PauliString, PauliSum, ProductBasis,
    PropagateOptions, TruncationPolicy,
};

/// `f64` and `Complex64` are exact up to summation order; `f32` and `Complex32` round every product.
fn tolerance<C: Coefficient>() -> f64 {
    if C::NAME == "f64" || C::NAME == "c128" {
        1e-12
    } else {
        2e-5
    }
}

/// `circuit` on `sum` in coefficient type `C`, widened back for comparison.
fn propagate_as<const W: usize, C, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W>
where
    C: Coefficient,
    T: TruncationPolicy<W, C> + TruncationPolicy<W>,
{
    let narrowed = sum.cast::<C>(0.0).expect("a real input sum");
    let out = propagate(circuit, narrowed, policy, direction);
    out.cast::<Complex64>(0.0).unwrap()
}

fn agrees_with_reference<const W: usize, C: Coefficient>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    direction: Direction,
) {
    let want = propagate(circuit, sum.clone(), &KeepAll, direction);
    let got = propagate_as::<W, C, _>(circuit, sum, &KeepAll, direction);
    assert_terms_close(&got, &want, tolerance::<C>(), C::NAME);
}

fn every_type_agrees<const W: usize>(circuit: &Circuit<W>, sum: &PauliSum<W>) {
    for direction in [Direction::Heisenberg, Direction::Forward] {
        agrees_with_reference::<W, f64>(circuit, sum, direction);
        agrees_with_reference::<W, f32>(circuit, sum, direction);
        agrees_with_reference::<W, Complex64>(circuit, sum, direction);
        agrees_with_reference::<W, Complex32>(circuit, sum, direction);
    }
}

#[test]
fn every_built_in_channel_class_agrees_in_every_type_w1() {
    for seed in 0..4 {
        let circuit = random_circuit::<1>(6, 40, seed, true);
        assert!(circuit.preserves_hermiticity());
        every_type_agrees(&circuit, &rand_sum_real::<1>(100, 6, seed));
    }
}

#[test]
fn every_built_in_channel_class_agrees_in_every_type_w2() {
    for seed in 0..2 {
        let circuit = random_circuit::<2>(100, 25, seed, false);
        every_type_agrees(&circuit, &rand_sum_real::<2>(400, 100, seed));
    }
}

#[test]
fn a_trotter_run_past_one_bucket_agrees_in_every_type() {
    let circuit = trotter_circuit::<1>(8, 0.3);
    let sum = rand_sum_real::<1>(50, 8, 7);
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Heisenberg);
    assert!(want.num_buckets() > 1, "the fixture must reach several buckets");
    every_type_agrees(&circuit, &sum);
}

#[test]
fn a_truncated_run_agrees_in_every_real_type() {
    let circuit = trotter_circuit::<1>(8, 0.3);
    let sum = rand_sum_real::<1>(50, 8, 3);
    let policy = CoefficientThreshold(1e-3);
    let want = propagate(&circuit, sum.clone(), &policy, Direction::Heisenberg);
    let got = propagate_as::<1, f64, _>(&circuit, &sum, &policy, Direction::Heisenberg);
    assert_terms_close(&got, &want, 1e-12, "f64");
}

#[test]
fn a_partitioned_real_run_agrees_with_the_complex_reference() {
    let circuit = random_circuit::<1>(7, 60, 11, true);
    let sum = rand_sum_real::<1>(300, 7, 11);
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Heisenberg);
    for partitions in [2, 4] {
        let got = propagate_partitioned(
            &circuit,
            sum.cast::<f32>(0.0).unwrap(),
            &KeepAll,
            Direction::Heisenberg,
            &unpinned_partitions(partitions, 2),
            PropagateOptions::default(),
        )
        .unwrap();
        assert_terms_close(&got.cast::<Complex64>(0.0).unwrap(), &want, 2e-5, "f32 partitioned");
    }
}

#[test]
fn readouts_return_the_coefficient_type() {
    // ⟨0|(0.5·Z₀ − 0.25·X₁ + 2·I)|0⟩ = 0.5 + 2 = 2.5.
    let mut accumulator = BuildAccumulator::<1, f32>::new(2);
    accumulator.add_term(PauliString::z(0), 0.5);
    accumulator.add_term(PauliString::x(1), -0.25);
    accumulator.add_term(PauliString::identity(), 2.0);
    let sum = accumulator.finalize();
    let value: f32 = sum.expectation_product(&ProductBasis::uniform(PauliAxis::Z));
    assert_eq!(value, 2.5);
    assert_eq!(sum.identity_coefficient(), 2.0f32);
}

#[test]
fn cast_refuses_an_imaginary_coefficient_for_a_real_type() {
    let sum = rand_sum::<1>(50, 8, 1);
    assert!(sum.cast::<f64>(1e-12).is_none());
    let back = sum.cast::<Complex32>(0.0).unwrap().cast::<Complex64>(0.0).unwrap();
    assert_terms_close(&back, &sum, 1e-6, "Complex32 round trip");
}

/// `X ↦ i·X`: maps the Hermitian `X` to an anti-Hermitian operator, so its transfer table is imaginary.
struct ImaginaryPhase(usize);

impl Channel<1> for ImaginaryPhase {
    fn support(&self) -> [u64; 1] {
        paulistrings::support_mask(&[self.0])
    }

    fn apply(&self, x: &[u64; 1], z: &[u64; 1], coeff: Complex64, out: &mut OutputBuffer<1>) {
        let on_x = x[0] >> self.0 & 1 == 1 && z[0] >> self.0 & 1 == 0;
        out.push(*x, *z, if on_x { coeff * Complex64::i() } else { coeff });
    }

    fn apply_adjoint(
        &self,
        x: &[u64; 1],
        z: &[u64; 1],
        coeff: Complex64,
        out: &mut OutputBuffer<1>,
    ) {
        let on_x = x[0] >> self.0 & 1 == 1 && z[0] >> self.0 & 1 == 0;
        out.push(*x, *z, if on_x { -coeff * Complex64::i() } else { coeff });
    }
}

fn single<C: Coefficient>(p: PauliString<1>, c: C) -> PauliSum<1, C> {
    let mut accumulator = BuildAccumulator::<1, C>::new(4);
    accumulator.add_term(p, c);
    accumulator.finalize()
}

fn non_hermitian_circuit() -> Circuit<1> {
    let mut circuit = Circuit::<1>::new(4);
    circuit.push(paulistrings::PtmChannel::hadamard(0));
    circuit.push(ImaginaryPhase(1));
    circuit
}

#[test]
fn preserves_hermiticity_spots_an_imaginary_table() {
    assert!(!non_hermitian_circuit().preserves_hermiticity());
}

#[test]
fn a_complex_sum_carries_a_non_hermitian_channel() {
    let sum = single::<Complex64>(PauliString::x(1), Complex64::new(1.0, 0.0));
    let out = propagate(&non_hermitian_circuit(), sum, &KeepAll, Direction::Forward);
    assert_eq!(out.get(&PauliString::<1>::x(1).x, &[0]), Some(Complex64::new(0.0, 1.0)));
}

#[test]
#[should_panic(expected = "needs every channel to preserve Hermiticity")]
fn a_real_sum_refuses_a_non_hermitian_channel() {
    let sum = single::<f64>(PauliString::x(1), 1.0);
    let _ = propagate(&non_hermitian_circuit(), sum, &KeepAll, Direction::Forward);
}
