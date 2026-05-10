//! Criterion microbenches for the hot ops on the propagation path.
//!
//! Slice 11.1 in `research/plans/2026-04-30-v0.1-tdd-slices.md`. The goal is
//! to lock in baseline numbers for the three operations that dominate a layer
//! of propagation:
//!
//!   * `PauliString::mul_assign` — single-term multiplication (W ∈ {1, 2}).
//!   * `PauliSum::add` — sorted merge of two SoA sums (N ∈ {10⁴, 10⁶}).
//!   * `apply_layer` — full scan → sort → merge engine pass under a
//!     fanout-1 Clifford (N ∈ {10⁴, 10⁶}).
//!
//! Inputs are built with a seeded `Xs64` xorshift so timings are reproducible
//! across machines. Setup runs outside the timed region via
//! `bench_with_input` / cached owned inputs; the bench body either reads the
//! pre-built data (`add`, `apply_layer`) or clones a fresh mutable copy
//! (`mul_assign`) via `iter_batched`.

use criterion::{
    criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, Throughput,
};
use num_complex::Complex64;
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::channel::Clifford1Q;
use paulistrings::engine::sort_merge::apply_layer;
use paulistrings::pauli_string::PauliString;
use paulistrings::pauli_sum::PauliSum;
use paulistrings::phase::Phase;
use paulistrings::truncation::TruncationPolicy;
use std::hint::black_box;

/// Xorshift64* — small, deterministic, no dev-dep.
struct Xs64(u64);

impl Xs64 {
    fn new(seed: u64) -> Self {
        // Avoid the degenerate all-zero state.
        Self(seed | 1)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline]
    fn next_array<const W: usize>(&mut self) -> [u64; W] {
        let mut a = [0u64; W];
        for slot in a.iter_mut() {
            *slot = self.next_u64();
        }
        a
    }
}

/// A truncation policy that never drops anything. Mirrors the `AlwaysKeep`
/// helper in the engine tests; the trait default does what we want.
struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

fn random_pauli<const W: usize>(rng: &mut Xs64) -> PauliString<W> {
    PauliString::<W> {
        x: rng.next_array::<W>(),
        z: rng.next_array::<W>(),
    }
}

/// Build a sorted/deduplicated `PauliSum<W>` of length close to `n_terms`
/// from random Pauli keys. Duplicates are unlikely at these widths but the
/// accumulator handles them transparently; the resulting length may be
/// slightly less than `n_terms`.
fn random_sum<const W: usize>(n_terms: usize, num_qubits: usize, seed: u64) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n_terms);
    for _ in 0..n_terms {
        let p = random_pauli::<W>(&mut rng);
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, im));
    }
    acc.finalize()
}

/// PauliString mul_assign — the inner hot loop of every channel that does a
/// runtime Pauli multiplication (rotation, general unitary). Cliffords avoid
/// it via lookup tables, so this bench targets the rotation/unitary path.
fn bench_mul_assign(c: &mut Criterion) {
    let mut group = c.benchmark_group("pauli_string_mul_assign");
    group.throughput(Throughput::Elements(1));

    {
        let mut rng = Xs64::new(0xA11CE);
        let a: PauliString<1> = random_pauli(&mut rng);
        let b: PauliString<1> = random_pauli(&mut rng);
        group.bench_function("W=1", |bencher| {
            bencher.iter_batched(
                || a,
                |mut lhs| {
                    let phase = lhs.mul_assign(black_box(&b));
                    black_box((lhs, phase))
                },
                BatchSize::SmallInput,
            )
        });
    }
    {
        let mut rng = Xs64::new(0xB0B);
        let a: PauliString<2> = random_pauli(&mut rng);
        let b: PauliString<2> = random_pauli(&mut rng);
        group.bench_function("W=2", |bencher| {
            bencher.iter_batched(
                || a,
                |mut lhs| {
                    let phase = lhs.mul_assign(black_box(&b));
                    black_box((lhs, phase))
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

/// `PauliSum::add` — sorted-merge of two SoA sums. Linear in the union size.
///
/// We benchmark two regimes: 10⁴ terms (fits comfortably in L1/L2) and 10⁶
/// terms (DRAM-bound). Both runs use disjoint random key streams so the
/// merge has to interleave rather than degenerate into "left then right".
fn bench_pauli_sum_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("pauli_sum_add");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &n in &[10_000usize, 1_000_000usize] {
        let a: PauliSum<2> = random_sum::<2>(n, 128, 0xADDA);
        let b: PauliSum<2> = random_sum::<2>(n, 128, 0xADDB);
        let union_estimate = (a.len() + b.len()) as u64;
        group.throughput(Throughput::Elements(union_estimate));
        // The 10⁶ case runs ~tens of ms per iter; trim sample count so the
        // bench wall-clock stays in the 10s of seconds range.
        let sample_size = if n >= 1_000_000 { 20 } else { 100 };
        group.sample_size(sample_size);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| black_box(a.add(black_box(&b))))
        });
    }

    group.finish();
}

/// Full scan → sort → merge layer under a fanout-1 Clifford gate.
///
/// `Clifford1Q::h(0)` has `max_fanout = 1` and a fixed 4-entry conjugation
/// table, so the bench measures engine overhead (sort + segmented reduction)
/// rather than channel inner-loop cost. The output cardinality equals the
/// input cardinality (no duplicates from a permutation), which makes the
/// per-element work easy to reason about.
fn bench_apply_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_layer_clifford1q_h");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let h = Clifford1Q::h(0);
    let policy = AlwaysKeep;

    for &n in &[10_000usize, 1_000_000usize] {
        let input: PauliSum<2> = random_sum::<2>(n, 128, 0xC0FFEE);
        group.throughput(Throughput::Elements(input.len() as u64));
        let sample_size = if n >= 1_000_000 { 10 } else { 30 };
        group.sample_size(sample_size);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| black_box(apply_layer(black_box(&input), &h, &policy)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mul_assign, bench_pauli_sum_add, bench_apply_layer);
criterion_main!(benches);
