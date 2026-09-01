//! The direct-apply small-sum path against the canonical sorting engine.
//!
//! Cross-module by construction: the claim under test is that
//! `EngineSelection::Auto` and `EngineSelection::SortedOnly` are the same
//! computation — same terms to floating-point tolerance, same per-layer term
//! counts, same records — whichever side of the transition a layer falls on.
//! Design: `research/notes/2026-09-01-small-m-path.md`.
//!
//! The sorting engine is the reference here, not `naive_apply_layer`: the
//! oracle and the direct path share an algorithm (a `Channel::apply` loop into a
//! hash map), so only the bucketed engine is an independent check of the
//! algebra. `engine::direct`'s own unit tests cover the oracle comparison.

use num_complex::Complex64;
use proptest::prelude::*;

use paulistrings::channel::{
    AmplitudeDamping, Channel, Clifford1Q, Clifford2Q, Depolarizing, Depolarizing2Q,
    GeneralUnitary2Q, PauliRotation,
};
use paulistrings::test_support::{assert_same_terms, assert_terms_close, rand_sum};
use paulistrings::truncation::{And, CoefficientThreshold, TopN, WeightCutoff};
use paulistrings::{
    propagate, propagate_with_options, propagate_with_scratch_and_options, Circuit, Direction,
    EngineSelection, LayerScratch, PauliString, PauliSum, PropagateOptions, TruncationPolicy,
    DEFAULT_SMALL_SUM_THRESHOLD,
};

/// Keeps everything, and declares no layer pass — so `Auto` will actually take
/// the direct path. The trait's default answer is the conservative `true`.
struct KeepAll;
impl<const W: usize> TruncationPolicy<W> for KeepAll {
    fn finalizes_layer(&self) -> bool {
        false
    }
}

fn auto(threshold: usize) -> PropagateOptions {
    PropagateOptions {
        engine: EngineSelection::Auto,
        small_sum_threshold: threshold,
    }
}

fn forced(threshold: usize) -> PropagateOptions {
    PropagateOptions {
        engine: EngineSelection::SmallSumDirect,
        small_sum_threshold: threshold,
    }
}

const SORTED: PropagateOptions = PropagateOptions {
    engine: EngineSelection::SortedOnly,
    small_sum_threshold: DEFAULT_SMALL_SUM_THRESHOLD,
};

/// Size of the channel zoo — every built-in class the engine has a different
/// code path for, plus a rotation above `MAX_LOCAL_SUPPORT` (the
/// `prepare`-overriding case) and a dense two-qubit PTM (the `su4`-shaped case).
const ZOO: usize = 12;

/// Zoo member `i % ZOO`, freshly built. Needs `num_qubits >= 8`.
///
/// A factory rather than a `Vec<Box<dyn Channel>>` because `Box<dyn Channel>` is
/// not `Clone`, and a circuit wants its own copy of each channel.
fn make_channel<const W: usize>(i: usize) -> Box<dyn Channel<W>> {
    match i % ZOO {
        0 => Box::new(Clifford1Q::h(0)),
        1 => Box::new(Clifford1Q::s(1)),
        2 => Box::new(Clifford2Q::cnot(0, 3)),
        3 => Box::new(Clifford2Q::cz(1, 4)),
        4 => Box::new(Clifford2Q::swap(2, 5)),
        5 => Box::new(PauliRotation::new(PauliString::<W>::z(0), 0.31)),
        // Weight-2 generator Z₁Z₂ — the workhorse of every rotation workload.
        6 => Box::new(PauliRotation::new(
            {
                let mut g = PauliString::<W>::z(1);
                g.z[0] |= 0b100;
                g
            },
            0.47,
        )),
        // Weight-4 generator — above MAX_LOCAL_SUPPORT = 2, so
        // `Prepared::derive_local` declines and `PauliRotation::prepare` is
        // what keeps the sorting engine able to run it at all.
        7 => Box::new(PauliRotation::new(
            {
                let mut g = PauliString::<W>::z(0);
                g.x[0] |= 0b0110;
                g.z[0] |= 0b1000;
                g
            },
            0.23,
        )),
        8 => Box::new(Depolarizing {
            support: [2],
            p: 0.13,
        }),
        9 => Box::new(Depolarizing2Q {
            support: [4, 6],
            p: 0.07,
        }),
        10 => Box::new(AmplitudeDamping {
            support: [3],
            gamma: 0.21,
        }),
        // A real 4x4 rotation in the (0,1) block: unitary, and dense in the PTM.
        _ => {
            let mut u = [[Complex64::new(0.0, 0.0); 4]; 4];
            for (r, row) in u.iter_mut().enumerate() {
                row[r] = Complex64::new(1.0, 0.0);
            }
            let (c, s) = (0.6f64, 0.8f64);
            u[0][0] = Complex64::new(c, 0.0);
            u[0][1] = Complex64::new(-s, 0.0);
            u[1][0] = Complex64::new(s, 0.0);
            u[1][1] = Complex64::new(c, 0.0);
            Box::new(GeneralUnitary2Q::from_matrix(6, 7, u))
        }
    }
}

/// A circuit picking channels out of the zoo by index.
fn circuit_from<const W: usize>(num_qubits: usize, order: &[usize]) -> Circuit<W> {
    let mut c = Circuit::<W>::new(num_qubits);
    for &i in order {
        // `Circuit::push` is generic over a *sized* channel; `channels` is a
        // public field, which is how a boxed one gets in.
        c.channels.push(make_channel::<W>(i));
    }
    c
}

/// Compare the two engines on one configuration.
fn assert_engines_agree<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    options: PropagateOptions,
    what: &str,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let mut s1 = LayerScratch::<W>::new();
    s1.enable_term_trace();
    let want = propagate_with_scratch_and_options(
        circuit,
        sum.clone(),
        policy,
        direction,
        &mut s1,
        SORTED,
    );
    let want_trace = s1.take_term_trace().expect("tracing enabled");

    let mut s2 = LayerScratch::<W>::new();
    s2.enable_term_trace();
    let got = propagate_with_scratch_and_options(
        circuit,
        sum.clone(),
        policy,
        direction,
        &mut s2,
        options,
    );
    let got_trace = s2.take_term_trace().expect("tracing enabled");

    assert_eq!(
        got_trace.terms_in, want_trace.terms_in,
        "{what}: per-layer terms_in",
    );
    assert_eq!(
        got_trace.terms_out, want_trace.terms_out,
        "{what}: per-layer terms_out",
    );
    assert_terms_close(&got, &want, 1e-9, what);
    // `assert_invariants` is debug-only and this file is also built by
    // `cargo test --release`, so the call has to be gated (see
    // `tests/propagate.rs`).
    #[cfg(debug_assertions)]
    got.assert_invariants();
}

// ---- defaults are untouched ----

/// `PropagateOptions::default()` is `propagate`, bit for bit: same engine, same
/// summation order, same output. This is the non-perturbation claim at the
/// behavioural level (the timing claim is `scripts/ab-compare.sh`'s).
#[test]
fn default_options_match_propagate_bitwise() {
    let sum = rand_sum::<2>(3000, 96, 0x51D);
    let circuit = circuit_from::<2>(96, &[0, 5, 2, 8, 11, 6, 9, 10]);
    let policy = CoefficientThreshold(1e-6);

    let want = propagate(&circuit, sum.clone(), &policy, Direction::Heisenberg);
    let got = propagate_with_options(
        &circuit,
        sum,
        &policy,
        Direction::Heisenberg,
        PropagateOptions::default(),
    );
    assert_same_terms(&got, &want, "default options vs propagate");
}

/// A sum already past the threshold never enters the direct path, so the run is
/// the sorting engine's from the first layer — including under
/// `SmallSumDirect`.
#[test]
fn a_large_starting_sum_stays_on_the_sorting_engine() {
    let sum = rand_sum::<1>(400, 20, 0x1A26);
    let circuit = circuit_from::<1>(20, &[5, 0, 6, 2]);
    let want = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    for options in [auto(100), forced(100)] {
        let got =
            propagate_with_options(&circuit, sum.clone(), &KeepAll, Direction::Forward, options);
        assert_same_terms(&got, &want, "above-threshold start");
    }
}

// ---- agreement across the zoo ----

fn zoo_agrees<const W: usize>(num_qubits: usize, seed: u64) {
    let order: Vec<usize> = (0..ZOO).collect();
    for &n_terms in &[1usize, 7, 64, 300] {
        let sum = rand_sum::<W>(n_terms, num_qubits, seed ^ n_terms as u64);
        let circuit = circuit_from::<W>(num_qubits, &order);
        for direction in [Direction::Forward, Direction::Heisenberg] {
            for &policy_kind in &[0u8, 1, 2] {
                let what = format!("W={W} n={n_terms} {direction:?} policy={policy_kind}");
                // A threshold well above anything these circuits reach, so the
                // whole circuit runs on the direct path.
                match policy_kind {
                    0 => assert_engines_agree(
                        &circuit,
                        &sum,
                        &KeepAll,
                        direction,
                        auto(1 << 20),
                        &what,
                    ),
                    1 => assert_engines_agree(
                        &circuit,
                        &sum,
                        &CoefficientThreshold(1e-3),
                        direction,
                        auto(1 << 20),
                        &what,
                    ),
                    _ => assert_engines_agree(
                        &circuit,
                        &sum,
                        &And(CoefficientThreshold(1e-6), WeightCutoff(6)),
                        direction,
                        auto(1 << 20),
                        &what,
                    ),
                }
            }
        }
    }
}

#[test]
fn zoo_agrees_w1() {
    zoo_agrees::<1>(24, 0xB01);
}

#[test]
fn zoo_agrees_w2() {
    zoo_agrees::<2>(96, 0xB02);
}

// ---- the transition ----

/// A propagation that crosses the threshold mid-circuit: the leading layers run
/// direct, the sum is rebuilt into buckets, and the rest runs sorted. Every
/// threshold in the sweep puts the crossing at a different layer, and the
/// per-layer counts and the result must not notice.
#[test]
fn crossing_the_threshold_mid_circuit_agrees() {
    // Rotations at a generic angle grow the sum every layer, so the crossing
    // lands wherever the threshold is put.
    let order: Vec<usize> = vec![5, 6, 5, 6, 7, 5, 6, 7, 5, 6];
    for &num_qubits in &[24usize, 96] {
        for &threshold in &[0usize, 1, 2, 3, 8, 33, 200, 1 << 20] {
            for direction in [Direction::Forward, Direction::Heisenberg] {
                let what = format!("q={num_qubits} threshold={threshold} {direction:?}");
                if num_qubits <= 64 {
                    let sum = rand_sum::<1>(2, num_qubits, 0xC1);
                    let circuit = circuit_from::<1>(num_qubits, &order);
                    assert_engines_agree(
                        &circuit,
                        &sum,
                        &KeepAll,
                        direction,
                        auto(threshold),
                        &what,
                    );
                } else {
                    let sum = rand_sum::<2>(2, num_qubits, 0xC2);
                    let circuit = circuit_from::<2>(num_qubits, &order);
                    assert_engines_agree(
                        &circuit,
                        &sum,
                        &KeepAll,
                        direction,
                        auto(threshold),
                        &what,
                    );
                }
            }
        }
    }
}

/// An empty sum and a zero-layer circuit are the two degenerate entries.
#[test]
fn degenerate_inputs_agree() {
    let empty = PauliSum::<1>::empty(8);
    let circuit = circuit_from::<1>(8, &[5, 0, 6]);
    assert_engines_agree(
        &circuit,
        &empty,
        &KeepAll,
        Direction::Forward,
        auto(64),
        "empty sum",
    );

    let sum = rand_sum::<1>(10, 8, 0xE0);
    let no_layers = Circuit::<1>::new(8);
    assert_engines_agree(
        &no_layers,
        &sum,
        &KeepAll,
        Direction::Forward,
        auto(64),
        "zero-layer circuit",
    );
}

// ---- truncation parity ----

/// `TopN` runs in `finalize_layer`, so the direct path must pay the
/// materialize → finalize → re-ingest round trip. `SmallSumDirect` is the
/// selection that makes it do so; the result and the per-layer counts must
/// match the sorting engine's exactly.
#[test]
fn topn_matches_on_the_direct_path() {
    for &n in &[1usize, 5, 40] {
        let sum = rand_sum::<2>(60, 96, 0x70D + n as u64);
        let circuit = circuit_from::<2>(96, &[5, 6, 0, 7, 11, 6]);
        assert_engines_agree(
            &circuit,
            &sum,
            &TopN(n),
            Direction::Heisenberg,
            forced(1 << 20),
            &format!("TopN({n}) forced direct"),
        );
        assert_engines_agree(
            &circuit,
            &sum,
            &And(CoefficientThreshold(1e-9), TopN(n)),
            Direction::Forward,
            forced(1 << 20),
            &format!("And(coeff, TopN({n})) forced direct"),
        );
    }
}

/// `TopN` with tie groups is the case where the selection rule (whole groups,
/// exact `f64` equality) is load-bearing, and it must come out the same on both
/// paths.
#[test]
fn topn_tie_groups_match_on_the_direct_path() {
    use paulistrings::test_support::tie_heavy_sum;
    let sum = tie_heavy_sum::<1>(80, 20, 0x7135);
    let circuit = circuit_from::<1>(20, &[0, 2, 4]);
    for &n in &[7usize, 20, 41] {
        assert_engines_agree(
            &circuit,
            &sum,
            &TopN(n),
            Direction::Forward,
            forced(1 << 20),
            &format!("tie-heavy TopN({n})"),
        );
    }
}

// ---- channels the sorting engine cannot prepare ----

/// Support on three qubits, so `Channel::prepare` declines: the sorting engine
/// panics, the direct path applies it.
struct ThreeQubitShift;
impl<const W: usize> Channel<W> for ThreeQubitShift {
    fn max_fanout(&self) -> usize {
        1
    }
    fn support(&self) -> [u64; W] {
        paulistrings::channel::support_mask(&[0, 1, 2])
    }
    fn apply(
        &self,
        input_x: &[u64; W],
        input_z: &[u64; W],
        coeff: Complex64,
        out: &mut paulistrings::OutputBuffer<'_, W>,
    ) {
        let mut x = *input_x;
        let low = input_x[0] & 0b111;
        x[0] = (input_x[0] & !0b111) | ((low << 1) & 0b111) | (low >> 2);
        out.push(x, *input_z, coeff);
    }
}

/// Below the threshold the direct path runs a channel the sorting engine
/// refuses — it needs only `Channel::apply`. Capability, not a divergence: the
/// same circuit under `SortedOnly` panics, which the next test pins.
#[test]
fn the_direct_path_runs_a_channel_the_sorting_engine_refuses() {
    let sum = rand_sum::<1>(16, 8, 0x3B1);
    let mut circuit = Circuit::<1>::new(8);
    circuit.push(ThreeQubitShift);
    circuit.push(ThreeQubitShift);
    circuit.push(ThreeQubitShift);
    // Three shifts of a 3-cycle are the identity on the keys.
    let out = propagate_with_options(
        &circuit,
        sum.clone(),
        &KeepAll,
        Direction::Forward,
        auto(1024),
    );
    assert_same_terms(&out, &sum, "three-cycle returns to the identity");
}

/// The asymmetry is not silent: once the sum outgrows the threshold the run
/// moves to the sorting engine, which panics on that channel exactly as it does
/// today under the default selection.
#[test]
#[should_panic(expected = "Channel::prepare declined")]
fn a_channel_the_sorting_engine_refuses_panics_after_the_transition() {
    let sum = rand_sum::<1>(16, 8, 0x3B2);
    let mut circuit = Circuit::<1>::new(8);
    circuit.push(ThreeQubitShift);
    circuit.push(ThreeQubitShift);
    // Threshold 1: the first layer leaves 16 terms, above it, so layer two is
    // the sorting engine's.
    let _ = propagate_with_options(&circuit, sum, &KeepAll, Direction::Forward, auto(1));
}

/// Under `Auto` a finalizing policy keeps the run on the sorting engine — which
/// is observable exactly here, because that engine cannot prepare this channel.
/// `SmallSumDirect` takes the direct path anyway and succeeds.
#[test]
#[should_panic(expected = "Channel::prepare declined")]
fn auto_declines_a_finalizing_policy() {
    let sum = rand_sum::<1>(16, 8, 0x3B3);
    let mut circuit = Circuit::<1>::new(8);
    circuit.push(ThreeQubitShift);
    let _ = propagate_with_options(&circuit, sum, &TopN(1000), Direction::Forward, auto(1024));
}

#[test]
fn small_sum_direct_takes_a_finalizing_policy_anyway() {
    let sum = rand_sum::<1>(16, 8, 0x3B3);
    let mut circuit = Circuit::<1>::new(8);
    circuit.push(ThreeQubitShift);
    let out = propagate_with_options(
        &circuit,
        sum.clone(),
        &TopN(1000),
        Direction::Forward,
        forced(1024),
    );
    assert_eq!(out.len(), sum.len());
}

// ---- properties ----

proptest! {
    /// Random sum, random channel sequence, random threshold: the two engines
    /// agree on the result and on every per-layer term count, wherever the
    /// transition lands.
    #[test]
    fn engines_agree_on_random_configurations_w1(
        seed in any::<u64>(),
        n_terms in 1usize..120,
        threshold in 0usize..400,
        order in prop::collection::vec(0usize..12, 1..10),
        heisenberg in any::<bool>(),
    ) {
        let direction = if heisenberg { Direction::Heisenberg } else { Direction::Forward };
        let sum = rand_sum::<1>(n_terms, 24, seed);
        let circuit = circuit_from::<1>(24, &order);
        assert_engines_agree(
            &circuit, &sum, &CoefficientThreshold(1e-9), direction, auto(threshold), "proptest w1",
        );
    }

    /// The same at `W = 2`, where the bucket partition is over two words.
    #[test]
    fn engines_agree_on_random_configurations_w2(
        seed in any::<u64>(),
        n_terms in 1usize..120,
        threshold in 0usize..400,
        order in prop::collection::vec(0usize..12, 1..10),
        heisenberg in any::<bool>(),
    ) {
        let direction = if heisenberg { Direction::Heisenberg } else { Direction::Forward };
        let sum = rand_sum::<2>(n_terms, 96, seed);
        let circuit = circuit_from::<2>(96, &order);
        assert_engines_agree(
            &circuit, &sum, &CoefficientThreshold(1e-9), direction, auto(threshold), "proptest w2",
        );
    }
}
