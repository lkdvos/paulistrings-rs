//! Behavior of the `phase-timing` counters themselves. The *non-perturbation*
//! guarantee is not tested here — it is the whole existing suite (fingerprint
//! net, thread-count/bucket-count bitwise identity, capacity stability)
//! passing under `--features phase-timing`.
#![cfg(feature = "phase-timing")]

use num_complex::Complex64;
use paulistrings::channel::{Depolarizing, PauliRotation};
use paulistrings::{
    propagate_with_scratch, BuildAccumulator, Circuit, Direction, LayerScratch, PauliString,
    PauliSum, Phase, PhaseStats, TruncationPolicy,
};

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

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

fn random_sum(n_terms: usize, num_qubits: usize, seed: u64) -> PauliSum<1> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<1>::with_capacity(num_qubits, n_terms);
    let mask = (1u64 << num_qubits) - 1;
    for _ in 0..n_terms {
        let p = PauliString::<1> {
            x: [rng.next_u64() & mask],
            z: [rng.next_u64() & mask],
        };
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, 0.0));
    }
    acc.finalize()
}

fn zz_rotation(q0: u32, q1: u32, theta: f64) -> PauliRotation<1> {
    let gen = PauliString::<1> {
        x: [0],
        z: [(1 << q0) | (1 << q1)],
    };
    PauliRotation::new(gen, theta)
}

#[test]
fn stats_sum_approximates_total() {
    // 20k dense terms over 32 qubits → well past the single-bucket regime, so
    // the coset machinery actually runs. One-thread pool so worker busy time
    // and coset-loop wall time are the same clock domain.
    let mut circuit = Circuit::<1>::new(32);
    for q in 0..4 {
        circuit.push(zz_rotation(q, q + 1, 0.13 + q as f64 * 0.05));
    }
    let sum = random_sum(20_000, 32, 0x51A75);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    let mut scratch = LayerScratch::<1>::new();
    let out = pool.install(|| {
        propagate_with_scratch(
            &circuit,
            sum,
            &AlwaysKeep,
            Direction::Heisenberg,
            &mut scratch,
        )
    });
    assert!(!out.is_empty());

    let stats = scratch.take_stats();
    assert_eq!(stats.layers, 4);
    assert!(stats.cosets > 0, "coset tasks must be counted: {stats:?}");
    assert!(stats.runs >= stats.cosets);
    assert!(stats.terms_in > 0 && stats.terms_out > 0);
    assert_eq!(
        stats.rescale_ns, 0,
        "no key-preserving layer in this circuit"
    );
    assert!(stats.coset_loop_ns > 0);

    // Busy time is measured strictly inside the coset-loop wall interval, so
    // on one thread it is bounded by it; and the gap (loop dispatch, scratch
    // lookup) should be small. 0.5 is deliberately loose to avoid flakes.
    let busy = stats.busy_total_ns();
    assert!(
        busy <= stats.coset_loop_ns,
        "busy {} > wall {}",
        busy,
        stats.coset_loop_ns
    );
    assert!(
        busy * 2 >= stats.coset_loop_ns,
        "busy {} < half of wall {}",
        busy,
        stats.coset_loop_ns
    );
    assert!(stats.gather_ns > 0 && stats.sort_ns > 0 && stats.merge_ns > 0);
    assert!(stats.timer_reads() > 0);
}

#[test]
fn take_stats_drains() {
    let mut circuit = Circuit::<1>::new(16);
    circuit.push(zz_rotation(0, 1, 0.4));
    let sum = random_sum(5_000, 16, 0xD1CE);

    let mut scratch = LayerScratch::<1>::new();
    let _ = propagate_with_scratch(&circuit, sum, &AlwaysKeep, Direction::Forward, &mut scratch);

    let first = scratch.take_stats();
    assert!(first.layers == 1 && first.coset_loop_ns > 0);
    let second = scratch.take_stats();
    assert_eq!(second, PhaseStats::default());
}

#[test]
fn rescale_path_is_attributed() {
    let mut circuit = Circuit::<1>::new(16);
    circuit.push(Depolarizing {
        support: [2],
        p: 0.05,
    });
    let sum = random_sum(5_000, 16, 0xACE);

    let mut scratch = LayerScratch::<1>::new();
    let _ = propagate_with_scratch(&circuit, sum, &AlwaysKeep, Direction::Forward, &mut scratch);

    let stats = scratch.take_stats();
    assert!(stats.rescale_ns > 0, "{stats:?}");
    assert_eq!(stats.coset_loop_ns, 0, "{stats:?}");
    assert_eq!(stats.cosets, 0);
    assert_eq!(stats.busy_total_ns(), 0);
}

#[test]
fn add_accumulates() {
    let mut a = PhaseStats::default();
    let mut b = PhaseStats::default();
    a.layers = 2;
    a.gather_ns = 10;
    b.layers = 3;
    b.gather_ns = 5;
    a.add(&b);
    assert_eq!(a.layers, 5);
    assert_eq!(a.gather_ns, 15);
}
