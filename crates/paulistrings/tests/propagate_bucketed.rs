//! `propagate` on the v0.2 bucketed engine, through the public API only.
//!
//! `tests/propagate.rs` predates the rewrite and still passes unmodified, but
//! most of its cases call `apply_layer` directly and therefore exercise the v0.1
//! whole-sum pipeline. This file drives equivalent behaviour through
//! `propagate`, which now runs bucketed, plus the properties that are specific
//! to the new engine: agreement with the v0.1 pipeline over a whole circuit, and
//! byte-identical output across thread counts.

use num_complex::Complex64;
use paulistrings::channel::{
    AmplitudeDamping, Channel, Clifford1Q, Clifford2Q, Dephasing, Depolarizing, IdentityChannel,
    PauliRotation,
};
use paulistrings::engine::sort_merge::{apply_layer, apply_layer_adjoint};
use paulistrings::truncation::{And, CoefficientThreshold, TopN, WeightCutoff};
use paulistrings::{
    BuildAccumulator, Circuit, Direction, PauliString, PauliSum, Phase, TruncationPolicy,
};

const TOL: f64 = 1e-11;

struct NoTruncation;
impl<const W: usize> TruncationPolicy<W> for NoTruncation {}

struct Xs64(u64);
impl Xs64 {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn word_mask(num_qubits: usize, word: usize) -> u64 {
    let lo = 64 * word;
    if num_qubits >= lo + 64 {
        !0u64
    } else if num_qubits <= lo {
        0
    } else {
        (1u64 << (num_qubits - lo)) - 1
    }
}

fn rand_sum<const W: usize>(n: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n);
    for _ in 0..n {
        let mut p = PauliString::<W> {
            x: [0u64; W],
            z: [0u64; W],
        };
        for w in 0..W {
            let m = word_mask(num_qubits, w);
            p.x[w] = rng.next_u64() & m;
            p.z[w] = rng.next_u64() & m;
        }
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, im));
    }
    acc.finalize()
}

fn one_term<const W: usize>(p: PauliString<W>, num_qubits: usize, c: Complex64) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, 1);
    acc.add_term(p, Phase::ONE, c);
    acc.finalize()
}

fn single<const W: usize, C: Channel<W> + 'static>(num_qubits: usize, ch: C) -> Circuit<W> {
    let mut c = Circuit::<W>::new(num_qubits);
    c.push(ch);
    c
}

// ---- single-channel circuits: the algebra, through the bucketed path ----

#[test]
fn h_conjugates_z_to_x() {
    let input = one_term(PauliString::<1>::z(0), 4, Complex64::new(1.0, 0.0));
    let out = paulistrings::propagate(
        &single(4, Clifford1Q::h(0)),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out.x()[0], [1]);
    assert_eq!(out.z()[0], [0]);
    assert!((out.coeff()[0] - Complex64::new(1.0, 0.0)).norm() < TOL);
}

#[test]
fn s_conjugates_x_to_y_with_phase_plus_one() {
    let input = one_term(PauliString::<1>::x(0), 4, Complex64::new(1.0, 0.0));
    let out = paulistrings::propagate(
        &single(4, Clifford1Q::s(0)),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out.x()[0], [1]);
    assert_eq!(out.z()[0], [1]);
    assert!((out.coeff()[0] - Complex64::new(1.0, 0.0)).norm() < TOL);
}

#[test]
fn cnot_propagates_z_on_the_control() {
    let input = one_term(PauliString::<1>::z(1), 4, Complex64::new(1.0, 0.0));
    let out = paulistrings::propagate(
        &single(4, Clifford2Q::cnot(0, 1)),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out.z()[0], [0b11]);
    assert_eq!(out.x()[0], [0]);
}

#[test]
fn cnot_propagates_x_on_the_target() {
    let input = one_term(PauliString::<1>::x(0), 4, Complex64::new(1.0, 0.0));
    let out = paulistrings::propagate(
        &single(4, Clifford2Q::cnot(0, 1)),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out.x()[0], [0b11]);
    assert_eq!(out.z()[0], [0]);
}

#[test]
fn identity_channel_passes_the_sum_through_unchanged() {
    let input = rand_sum::<1>(500, 8, 0x1D);
    let expect = input.clone();
    let out = paulistrings::propagate(
        &single(8, IdentityChannel::new()),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    // Whole-slice equality: the bucketed round trip must be exact, not merely
    // order-preserving.
    assert_eq!(out.x(), expect.x());
    assert_eq!(out.z(), expect.z());
    assert_eq!(out.coeff(), expect.coeff());
}

#[test]
fn word_boundary_qubit_64_w2() {
    let input = one_term(PauliString::<2>::z(64), 65, Complex64::new(1.0, 0.0));
    let out = paulistrings::propagate(
        &single(65, Clifford1Q::h(64)),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(out.len(), 1);
    assert_eq!(out.x()[0], [0, 1]);
    assert_eq!(out.z()[0], [0, 0]);
}

#[test]
fn inputs_that_collide_under_the_channel_are_combined_and_resorted() {
    // X(3.0) and Y(2.0) under S: the channel maps X -> Y and Y -> -X, so the
    // outputs are (Y, +3) and (X, -2) in that emission order. The result must
    // come back in lex (x, z) order, i.e. X before Y.
    let mut acc = BuildAccumulator::<1>::with_capacity(4, 2);
    acc.add_term(PauliString::<1>::x(0), Phase::ONE, Complex64::new(3.0, 0.0));
    acc.add_term(PauliString::<1>::y(0), Phase::ONE, Complex64::new(2.0, 0.0));
    let input = acc.finalize();

    let out = paulistrings::propagate(
        &single(4, Clifford1Q::s(0)),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(out.len(), 2);
    assert_eq!(out.x()[0], [1]);
    assert_eq!(out.z()[0], [0]); // X
    assert!((out.coeff()[0] - Complex64::new(-2.0, 0.0)).norm() < TOL);
    assert_eq!(out.x()[1], [1]);
    assert_eq!(out.z()[1], [1]); // Y
    assert!((out.coeff()[1] - Complex64::new(3.0, 0.0)).norm() < TOL);
}

#[test]
fn weight_cutoff_drops_high_weight_terms() {
    let input = rand_sum::<1>(1000, 8, 0x2D);
    let out = paulistrings::propagate(
        &single(8, Clifford1Q::h(0)),
        input.clone(),
        &WeightCutoff(2),
        Direction::Forward,
    );
    assert!(out.len() < input.len(), "nothing was dropped");
    for i in 0..out.len() {
        let p = PauliString::<1> {
            x: out.x()[i],
            z: out.z()[i],
        };
        assert!(p.weight() <= 2, "kept a weight-{} term", p.weight());
    }
}

#[test]
fn top_n_truncates_each_layer() {
    let input = rand_sum::<1>(2000, 8, 0x3D);
    let mut circuit = Circuit::<1>::new(8);
    circuit.push(PauliRotation::new(PauliString::<1>::z(2), 0.4));
    circuit.push(Clifford1Q::h(0));
    let out = paulistrings::propagate(&circuit, input, &TopN(100), Direction::Forward);
    assert_eq!(out.len(), 100);
}

// ---- Heisenberg direction ----

#[test]
fn rotation_round_trips_via_heisenberg_on_a_single_term() {
    // A single input term cancels exactly, so no threshold is needed.
    let circuit = single(8, PauliRotation::new(PauliString::<1>::z(2), 0.7));
    let input = one_term(PauliString::<1>::x(2), 8, Complex64::new(1.0, 0.0));
    let fwd = paulistrings::propagate(&circuit, input, &NoTruncation, Direction::Forward);
    assert_eq!(fwd.len(), 2, "rotation should fan out to two terms");
    let back = paulistrings::propagate(&circuit, fwd, &NoTruncation, Direction::Heisenberg);
    // Back to X with coefficient 1; the Y component cancels.
    let xs: Vec<usize> = (0..back.len())
        .filter(|&i| back.coeff()[i].norm() > 1e-9)
        .collect();
    assert_eq!(xs.len(), 1, "expected one surviving term, got {back:?}");
    let i = xs[0];
    assert_eq!((back.x()[i], back.z()[i]), ([0b100], [0]), "should be X(2)");
    assert!((back.coeff()[i] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
}

#[test]
fn rotation_round_trips_via_heisenberg_on_a_full_sum() {
    // On a many-term sum the cancellations are not bit-exact, so `U` then `U†`
    // leaves residual terms at ~1e-17. `merge_into` drops only *exact* zeros --
    // deliberately, matching v0.1 -- so a threshold is what makes the round trip
    // clean. Without one the sum grows (396 terms in, 517 out here), which is
    // correct behaviour rather than a defect.
    let input = rand_sum::<1>(400, 8, 0x4D);
    let circuit = single(8, PauliRotation::new(PauliString::<1>::z(2), 0.7));
    let policy = CoefficientThreshold(1e-12);
    let fwd = paulistrings::propagate(&circuit, input.clone(), &policy, Direction::Forward);
    let back = paulistrings::propagate(&circuit, fwd, &policy, Direction::Heisenberg);
    assert_eq!(back.len(), input.len(), "round trip changed the term count");
    assert_eq!(back.x(), input.x());
    assert_eq!(back.z(), input.z());
    for i in 0..back.len() {
        assert!(
            (back.coeff()[i] - input.coeff()[i]).norm() < 1e-10,
            "coeff[{i}] did not round trip",
        );
    }
}

#[test]
fn residual_growth_without_a_threshold_matches_v0_1() {
    // The same no-threshold round trip, checked against the v0.1 pipeline: both
    // engines must keep exactly the same residual terms.
    let input = rand_sum::<1>(400, 8, 0x4D);
    let chans: Vec<Box<dyn Channel<1>>> =
        vec![Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.7))];
    let circuit = single(8, PauliRotation::new(PauliString::<1>::z(2), 0.7));

    let fwd = paulistrings::propagate(&circuit, input.clone(), &NoTruncation, Direction::Forward);
    let back = paulistrings::propagate(&circuit, fwd, &NoTruncation, Direction::Heisenberg);

    let want_fwd = replay_v01(&input, &chans, &NoTruncation, Direction::Forward);
    let want_back = replay_v01(&want_fwd, &chans, &NoTruncation, Direction::Heisenberg);

    assert_close(&back, &want_back, "no-threshold round trip");
}

#[test]
fn heisenberg_reverses_the_channel_order() {
    // [H, S] forward on Z gives Y; Heisenberg on Z gives X.
    let mut circuit = Circuit::<1>::new(4);
    circuit.push(Clifford1Q::h(0));
    circuit.push(Clifford1Q::s(0));

    let fwd = paulistrings::propagate(
        &circuit,
        one_term(PauliString::<1>::z(0), 4, Complex64::new(1.0, 0.0)),
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(
        (fwd.x()[0], fwd.z()[0]),
        ([1], [1]),
        "forward should give Y"
    );

    let back = paulistrings::propagate(
        &circuit,
        one_term(PauliString::<1>::z(0), 4, Complex64::new(1.0, 0.0)),
        &NoTruncation,
        Direction::Heisenberg,
    );
    assert_eq!(
        (back.x()[0], back.z()[0]),
        ([1], [0]),
        "heisenberg should give X",
    );
}

#[test]
fn an_empty_circuit_returns_the_input_bit_for_bit() {
    let input = rand_sum::<2>(300, 128, 0x5D);
    let expect = input.clone();
    let out = paulistrings::propagate(
        &Circuit::<2>::new(128),
        input,
        &NoTruncation,
        Direction::Forward,
    );
    assert_eq!(out.x(), expect.x());
    assert_eq!(out.z(), expect.z());
    assert_eq!(out.coeff(), expect.coeff());
}

// ---- whole-circuit agreement with the v0.1 pipeline ----

/// Build a Trotter-shaped circuit plus the same channels as a flat list, so the
/// v0.1 loop can be replayed by hand.
fn mixed_channels() -> Vec<Box<dyn Channel<1>>> {
    let mut zz = PauliString::<1>::z(1);
    zz.mul_assign(&PauliString::<1>::z(4));
    let mut wide = PauliString::<1>::z(0);
    for q in [2u32, 3, 6] {
        wide.mul_assign(&PauliString::<1>::x(q));
    }
    vec![
        Box::new(Clifford1Q::h(0)),
        Box::new(PauliRotation::new(PauliString::<1>::z(2), 0.31)),
        Box::new(Clifford2Q::cnot(1, 5)),
        Box::new(Depolarizing {
            support: [3],
            p: 0.05,
        }),
        Box::new(PauliRotation::new(zz, 0.23)),
        Box::new(Clifford1Q::s(6)),
        Box::new(Dephasing {
            support: [7],
            p: 0.04,
        }),
        Box::new(Clifford2Q::cz(0, 7)),
        Box::new(AmplitudeDamping {
            support: [4],
            gamma: 0.2,
        }),
        // Weight 4: takes the functional rotation path.
        Box::new(PauliRotation::new(wide, 0.17)),
        Box::new(Clifford2Q::swap(2, 6)),
    ]
}

fn replay_v01<T: TruncationPolicy<1>>(
    input: &PauliSum<1>,
    chans: &[Box<dyn Channel<1>>],
    policy: &T,
    direction: Direction,
) -> PauliSum<1> {
    let mut sum = input.clone();
    let n = chans.len();
    for k in 0..n {
        let idx = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let ch = chans[idx].as_ref();
        sum = match direction {
            Direction::Forward => apply_layer(&sum, ch, policy),
            Direction::Heisenberg => apply_layer_adjoint(&sum, ch, policy),
        };
        policy.finalize_layer(&mut sum);
    }
    sum
}

fn assert_close(got: &PauliSum<1>, want: &PauliSum<1>, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: term count");
    assert_eq!(got.x(), want.x(), "{what}: x keys");
    assert_eq!(got.z(), want.z(), "{what}: z keys");
    for i in 0..got.len() {
        let d = (got.coeff()[i] - want.coeff()[i]).norm();
        assert!(d < 1e-10, "{what}: coeff[{i}] delta {d:e}");
    }
}

#[test]
fn eleven_channel_circuit_matches_the_v0_1_pipeline() {
    let input = rand_sum::<1>(1500, 8, 0x6D);
    let chans = mixed_channels();
    let mut circuit = Circuit::<1>::new(8);
    for ch in mixed_channels() {
        circuit.channels.push(ch);
    }

    for direction in [Direction::Forward, Direction::Heisenberg] {
        let want = replay_v01(&input, &chans, &NoTruncation, direction);
        let got = paulistrings::propagate(&circuit, input.clone(), &NoTruncation, direction);
        assert_close(&got, &want, &format!("{direction:?}"));
    }
}

#[test]
fn eleven_channel_circuit_matches_v0_1_under_truncation() {
    let input = rand_sum::<1>(1500, 8, 0x7D);
    let chans = mixed_channels();
    let mut circuit = Circuit::<1>::new(8);
    for ch in mixed_channels() {
        circuit.channels.push(ch);
    }
    let policy = And(CoefficientThreshold(1e-9), TopN(400));
    let want = replay_v01(&input, &chans, &policy, Direction::Heisenberg);
    let got = paulistrings::propagate(&circuit, input, &policy, Direction::Heisenberg);
    assert_close(&got, &want, "truncated heisenberg");
}

// ---- determinism ----

#[test]
fn output_is_byte_identical_across_thread_counts() {
    // The v0.2 counterpart of sort_merge's `scan_determinism_across_thread_counts`.
    // Stronger than it looks: `propagate` derives its bucket count from
    // `rayon::current_num_threads()`, so this also asserts that results are
    // independent of the partition itself.
    let input = rand_sum::<1>(2000, 8, 0x8D);
    let mut circuit = Circuit::<1>::new(8);
    for ch in mixed_channels() {
        circuit.channels.push(ch);
    }
    let policy = And(CoefficientThreshold(1e-12), TopN(900));

    let run = |threads: usize| -> PauliSum<1> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                paulistrings::propagate(&circuit, input.clone(), &policy, Direction::Heisenberg)
            })
    };

    let reference = run(1);
    for threads in [2usize, 4, 8, 16, 32] {
        let got = run(threads);
        assert_eq!(got.len(), reference.len(), "threads={threads}: length");
        assert_eq!(got.x(), reference.x(), "threads={threads}: x keys");
        assert_eq!(got.z(), reference.z(), "threads={threads}: z keys");
        assert_eq!(
            got.coeff(),
            reference.coeff(),
            "threads={threads}: coefficients are not byte-identical",
        );
    }
}

// ---- a realistic 2D Ising quench, both engines ----

/// One first-order Trotter step of the 2D transverse-field Ising model on an
/// `lx × ly` periodic lattice: ZZ bond rotations, then single-site X rotations.
/// Mirrors `examples/ising_2d_quench.rs`.
fn ising_step_channels(lx: usize, ly: usize, dt: f64) -> Vec<Box<dyn Channel<1>>> {
    let idx = |x: usize, y: usize| (y * lx + x) as u32;
    let mut chans: Vec<Box<dyn Channel<1>>> = Vec::new();
    for y in 0..ly {
        for x in 0..lx {
            let i = idx(x, y);
            for partner in [idx((x + 1) % lx, y), idx(x, (y + 1) % ly)] {
                if partner == i {
                    continue; // degenerate wrap on a length-1 axis
                }
                let mut gen = PauliString::<1>::z(i);
                gen.mul_assign(&PauliString::<1>::z(partner));
                chans.push(Box::new(PauliRotation::new(gen, 2.0 * dt)));
            }
        }
    }
    for y in 0..ly {
        for x in 0..lx {
            chans.push(Box::new(PauliRotation::new(
                PauliString::<1>::x(idx(x, y)),
                2.0 * dt,
            )));
        }
    }
    chans
}

/// `⟨+…+|O|+…+⟩` — the observable the Ising example tracks. Sum of `Re(coeff)`
/// over terms whose Z-part is empty.
fn expectation_plus(sum: &PauliSum<1>) -> f64 {
    let mut total = 0.0;
    for i in 0..sum.len() {
        if sum.z()[i] == [0u64] {
            total += sum.coeff()[i].re;
        }
    }
    total
}

/// A full multi-step 2D Ising quench must give the same trajectory on both
/// engines, step by step.
///
/// This is the shape that exposed the `TopN` tie-breaking bug: a symmetric
/// Hamiltonian on a periodic lattice produces many terms with *exactly* equal
/// coefficients, which random-coefficient tests never generate. Run here without
/// `TopN` so that any divergence is attributable to the engine rather than to a
/// truncation choice.
#[test]
fn ising_quench_trajectory_matches_the_v0_1_pipeline() {
    let (lx, ly) = (2usize, 3usize); // 6 qubits: the sum saturates 4^6 = 4096 terms
    let n = lx * ly;
    let dt = 0.15;
    let policy = CoefficientThreshold(1e-13);

    // Observable: uniform X magnetization.
    let mut acc = BuildAccumulator::<1>::with_capacity(n, n);
    for q in 0..n as u32 {
        acc.add_term(
            PauliString::<1>::x(q),
            Phase::ONE,
            Complex64::new(1.0 / n as f64, 0.0),
        );
    }
    let initial = acc.finalize();

    let mut circuit = Circuit::<1>::new(n);
    for ch in ising_step_channels(lx, ly, dt) {
        circuit.channels.push(ch);
    }
    let chans = ising_step_channels(lx, ly, dt);

    let mut bucketed = initial.clone();
    let mut reference = initial;

    for step in 1..=10 {
        bucketed = paulistrings::propagate(&circuit, bucketed, &policy, Direction::Heisenberg);
        reference = replay_v01(&reference, &chans, &policy, Direction::Heisenberg);

        assert_eq!(
            bucketed.len(),
            reference.len(),
            "step {step}: term count diverged",
        );
        assert_eq!(bucketed.x(), reference.x(), "step {step}: x keys diverged");
        assert_eq!(bucketed.z(), reference.z(), "step {step}: z keys diverged");

        let a = expectation_plus(&bucketed);
        let b = expectation_plus(&reference);
        assert!(
            (a - b).abs() < 1e-11,
            "step {step}: m_x {a} vs {b} (delta {:e})",
            (a - b).abs(),
        );
    }
}

/// A 3x3 periodic lattice with a hard-binding `TopN`, which is the closest
/// in-test analogue of what `examples/ising_2d_quench.rs` actually runs: real 2D
/// topology, 27 channels per step, and truncation active from the first step.
///
/// If the two engines agree here they agree in the example, so a difference
/// between the example's output and a previously committed reference must come
/// from a change in truncation *semantics*, not from the engine.
#[test]
fn ising_3x3_with_binding_top_n_matches_the_v0_1_pipeline() {
    let (lx, ly) = (3usize, 3usize);
    let n = lx * ly;
    let dt = 0.05;
    let policy = And(CoefficientThreshold(1e-12), TopN(1500));

    let mut acc = BuildAccumulator::<1>::with_capacity(n, n);
    for q in 0..n as u32 {
        acc.add_term(
            PauliString::<1>::x(q),
            Phase::ONE,
            Complex64::new(1.0 / n as f64, 0.0),
        );
    }
    let initial = acc.finalize();

    let mut circuit = Circuit::<1>::new(n);
    for ch in ising_step_channels(lx, ly, dt) {
        circuit.channels.push(ch);
    }
    let chans = ising_step_channels(lx, ly, dt);

    let mut bucketed = initial.clone();
    let mut reference = initial;
    let mut binding = false;

    for step in 1..=8 {
        bucketed = paulistrings::propagate(&circuit, bucketed, &policy, Direction::Heisenberg);
        reference = replay_v01(&reference, &chans, &policy, Direction::Heisenberg);
        if bucketed.len() == 1500 {
            binding = true;
        }
        assert_eq!(bucketed.len(), reference.len(), "step {step}: term count");
        assert_eq!(
            bucketed.x(),
            reference.x(),
            "step {step}: kept different terms",
        );
        assert_eq!(bucketed.z(), reference.z(), "step {step}: z keys");
        let a = expectation_plus(&bucketed);
        let b = expectation_plus(&reference);
        assert!((a - b).abs() < 1e-11, "step {step}: m_x {a} vs {b}");
    }
    assert!(
        binding,
        "TopN never bound; the test would not prove anything"
    );
}

/// The same quench *with* `TopN` active, which is what the shipped example does.
/// Both engines must agree exactly, including on which tied terms `TopN` keeps.
#[test]
fn ising_quench_with_top_n_matches_the_v0_1_pipeline() {
    let (lx, ly) = (2usize, 3usize);
    let n = lx * ly;
    let dt = 0.15;
    let policy = And(CoefficientThreshold(1e-13), TopN(300));

    let mut acc = BuildAccumulator::<1>::with_capacity(n, n);
    for q in 0..n as u32 {
        acc.add_term(
            PauliString::<1>::x(q),
            Phase::ONE,
            Complex64::new(1.0 / n as f64, 0.0),
        );
    }
    let initial = acc.finalize();

    let mut circuit = Circuit::<1>::new(n);
    for ch in ising_step_channels(lx, ly, dt) {
        circuit.channels.push(ch);
    }
    let chans = ising_step_channels(lx, ly, dt);

    let mut bucketed = initial.clone();
    let mut reference = initial;

    for step in 1..=10 {
        bucketed = paulistrings::propagate(&circuit, bucketed, &policy, Direction::Heisenberg);
        reference = replay_v01(&reference, &chans, &policy, Direction::Heisenberg);
        assert_eq!(bucketed.len(), reference.len(), "step {step}: term count");
        assert_eq!(
            bucketed.x(),
            reference.x(),
            "step {step}: TopN kept different terms",
        );
        let a = expectation_plus(&bucketed);
        let b = expectation_plus(&reference);
        assert!((a - b).abs() < 1e-11, "step {step}: m_x {a} vs {b}");
    }
}
