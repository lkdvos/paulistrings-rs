//! Criterion microbenches for the hot ops on the propagation path.
//!
//! Baseline surface for the v0.2 engine rewrite
//! (`research/plans/2026-08-26-v0.2-tdd-slices.md`, slice A.2). v0.1 slice 11.1
//! established only three benches; that covered exactly one point of the layer
//! cost surface — `Clifford1Q::h`, which is fanout-1 and key-bijective, i.e. the
//! *worst* case for the sort phase and the *best* case for the merge phase.
//!
//! What is measured here:
//!
//!   * `PauliString::mul_assign` — single-term multiplication (W ∈ {1, 2}).
//!   * `PauliSum::add` — sorted merge of two SoA sums (N ∈ {10⁴, 10⁶}). This is
//!     the useful reference point for `apply_layer`: a full two-pointer merge of
//!     the same payload, with allocation, and no sort.
//!   * `apply_layer` across the four structurally distinct channel classes
//!     (see `apply_layer` group below), each in both the monomorphized and the
//!     `dyn`-erased calling convention.
//!   * `apply_layer` on dense vs low-weight inputs — the two occupancy regimes,
//!     which matter for any support- or hash-derived bucketing.
//!   * `propagate` over a multi-channel Trotter-shaped circuit, which is the
//!     shape real workloads have (`examples/ising_2d_quench.rs` is 108 channels
//!     per step).
//!   * Thread scaling of one layer at 1/2/4/8/16/32 threads. v0.1 §9 claims
//!     "near-linear scaling up to the memory bandwidth limit" and nothing in the
//!     repo ever tested it.
//!
//! Inputs are built with a seeded `Xs64` xorshift so timings are reproducible
//! across machines. Setup runs outside the timed region via
//! `bench_with_input` / cached owned inputs; the bench body either reads the
//! pre-built data or clones a fresh mutable copy via `iter_batched`.

use criterion::measurement::WallTime;
use criterion::{
    criterion_group, criterion_main, AxisScale, BatchSize, BenchmarkGroup, BenchmarkId, Criterion,
    PlotConfiguration, Throughput,
};
use num_complex::Complex64;
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::bucket::{BucketedSum, Gf2Hash, DEFAULT_TARGET_BUCKET_LEN};
use paulistrings::channel::{Channel, Clifford1Q, Clifford2Q, Depolarizing, PauliRotation};
use paulistrings::circuit::Circuit;
use paulistrings::engine::bucketed::{apply_layer_bucketed, LayerScratch};
use paulistrings::engine::sort_merge::apply_layer;
use paulistrings::engine::{propagate, Direction};
use paulistrings::pauli_string::PauliString;
use paulistrings::pauli_sum::PauliSum;
use paulistrings::phase::Phase;
use paulistrings::truncation::TruncationPolicy;
use std::hint::black_box;
use std::time::Duration;

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

/// A Pauli string of Hamming weight `weight` over `num_qubits` qubits.
///
/// This is the *realistic* occupancy regime: physical Hamiltonians are
/// low-weight, and `WeightCutoff` truncation keeps them that way. The dense
/// `random_pauli` above is the opposite extreme. Any bucketing scheme derived
/// from key bits behaves very differently on the two, so both are benched.
fn low_weight_pauli<const W: usize>(
    rng: &mut Xs64,
    num_qubits: usize,
    weight: usize,
) -> PauliString<W> {
    let mut p = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    for _ in 0..weight {
        let q = (rng.next_u64() as usize) % num_qubits;
        let word = q / 64;
        let bit = 1u64 << (q % 64);
        // Pick one of X, Z, Y (never I, or the weight would not be `weight`).
        match rng.next_u64() % 3 {
            0 => p.x[word] |= bit,
            1 => p.z[word] |= bit,
            _ => {
                p.x[word] |= bit;
                p.z[word] |= bit;
            }
        }
    }
    p
}

/// Build a sorted/deduplicated `PauliSum<W>` of length close to `n_terms`
/// from random dense Pauli keys. Duplicates are unlikely at these widths but
/// the accumulator handles them transparently; the resulting length may be
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

/// As `random_sum`, but with low-weight keys. Collisions are far more likely
/// here, so the realized length can be noticeably below `n_terms`.
fn low_weight_sum<const W: usize>(
    n_terms: usize,
    num_qubits: usize,
    weight: usize,
    seed: u64,
) -> PauliSum<W> {
    let mut rng = Xs64::new(seed);
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, n_terms);
    for _ in 0..n_terms {
        let p = low_weight_pauli::<W>(&mut rng, num_qubits, weight);
        let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
        acc.add_term(p, Phase::ONE, Complex64::new(re, im));
    }
    acc.finalize()
}

/// A weight-2 `ZZ` rotation on qubits `(q0, q1)` — the bond term of a
/// transverse-field Ising Trotter step, and the single most common channel in
/// real workloads. Fanout is data-dependent (1 when the input commutes with the
/// generator, 2 when it does not), so on random input the realized fanout is
/// ~1.5 and the merge phase actually has duplicates to combine.
fn zz_rotation<const W: usize>(q0: u32, q1: u32, theta: f64) -> PauliRotation<W> {
    let mut gen = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    gen.z[(q0 as usize) / 64] |= 1u64 << (q0 % 64);
    gen.z[(q1 as usize) / 64] |= 1u64 << (q1 % 64);
    PauliRotation::new(gen, theta)
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

/// One `apply_layer` case. `?Sized` on `C` so the same helper measures both the
/// monomorphized call and the `dyn Channel<W>` call.
fn layer_case<const W: usize, C>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: String,
    input: &PauliSum<W>,
    ch: &C,
) where
    C: Channel<W> + ?Sized,
{
    let policy = AlwaysKeep;
    group.throughput(Throughput::Elements(input.len() as u64));
    group.bench_function(label, |bencher| {
        bencher.iter(|| black_box(apply_layer(black_box(input), ch, &policy)))
    });
}

/// Full scan → sort → merge layer, across the four structurally distinct
/// channel classes. These differ in ways the engine is blind to today but that
/// the v0.2 design keys off directly (v0.2 §2.3):
///
///   * `depolarizing` — keys bitwise **unchanged**, coefficients rescaled. The
///     output is already sorted and duplicate-free, so the entire sort and merge
///     is wasted work. Delta set `{0}`; v0.2 reads 1 input bucket per output.
///   * `clifford1q_h` — key **bijection**, fanout 1. The merge is provably a
///     no-op. Delta set is 1-dimensional; v0.2 reads 2 buckets.
///   * `clifford2q_cnot` — key bijection on 2 qubits, fanout 1. Delta set is
///     2-dimensional; v0.2 reads 4 buckets.
///   * `rotation_zz` — fanout 2, **non-injective**: this is the only case where
///     the merge phase has real work to do. Delta set `{0, gen}`, so v0.2 reads
///     2 buckets regardless of generator weight.
///
/// Each is measured both monomorphized and `dyn`-erased. The `dyn` figure is the
/// one that matters: `propagate` erases to `&dyn Channel<W>`
/// (`engine/mod.rs:78`), so every real layer pays a vtable call *per input
/// term*, which the monomorphized bench hides.
fn bench_apply_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_layer");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let h = Clifford1Q::h(0);
    let cnot = Clifford2Q::cnot(0, 1);
    let depol = Depolarizing {
        support: [0],
        p: 0.05,
    };
    let rot = zz_rotation::<2>(0, 1, 0.1);

    for &n in &[10_000usize, 1_000_000usize] {
        let input: PauliSum<2> = random_sum::<2>(n, 128, 0xC0FFEE);
        if n >= 1_000_000 {
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(500));
        } else {
            group.sample_size(30);
            group.warm_up_time(Duration::from_secs(1));
        }

        layer_case(&mut group, format!("depolarizing/{n}"), &input, &depol);
        layer_case(&mut group, format!("clifford1q_h/{n}"), &input, &h);
        layer_case(&mut group, format!("clifford2q_cnot/{n}"), &input, &cnot);
        layer_case(&mut group, format!("rotation_zz/{n}"), &input, &rot);

        // Same four, through the calling convention `propagate` actually uses.
        let dyn_h: &dyn Channel<2> = &h;
        let dyn_cnot: &dyn Channel<2> = &cnot;
        let dyn_depol: &dyn Channel<2> = &depol;
        let dyn_rot: &dyn Channel<2> = &rot;
        layer_case(
            &mut group,
            format!("dyn_depolarizing/{n}"),
            &input,
            dyn_depol,
        );
        layer_case(&mut group, format!("dyn_clifford1q_h/{n}"), &input, dyn_h);
        layer_case(
            &mut group,
            format!("dyn_clifford2q_cnot/{n}"),
            &input,
            dyn_cnot,
        );
        layer_case(&mut group, format!("dyn_rotation_zz/{n}"), &input, dyn_rot);
    }

    group.finish();
}

/// Dense vs low-weight input at fixed `n`.
///
/// Physical Hamiltonians are low-weight and `WeightCutoff` keeps them that way,
/// so `low_weight` is the realistic regime while `dense` is what every existing
/// bench used. The two have very different key distributions, which is exactly
/// what decides whether a bucketing scheme balances (v0.2 §3.2) — a
/// coordinate-projection hash collapses on the low-weight case and looks fine on
/// the dense one.
fn bench_apply_layer_occupancy(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_layer_occupancy");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let rot = zz_rotation::<2>(0, 1, 0.1);

    let dense: PauliSum<2> = random_sum::<2>(n, 128, 0xDE_11_5E);
    layer_case(&mut group, format!("dense/{}", dense.len()), &dense, &rot);

    for &w in &[2usize, 4, 8] {
        let sparse: PauliSum<2> =
            low_weight_sum::<2>(n, 128, w, 0x5A_12_5E_u64.wrapping_add(w as u64));
        layer_case(
            &mut group,
            format!("weight{w}/{}", sparse.len()),
            &sparse,
            &rot,
        );
    }

    group.finish();
}

/// `propagate` over a Trotter-shaped circuit — the realistic call shape.
///
/// One first-order Trotter step on a 1-D transverse-field Ising chain of
/// `num_qubits` sites: `num_qubits` ZZ bond rotations followed by `num_qubits`
/// single-site X rotations. `examples/ising_2d_quench.rs` builds 108 channels
/// per step for a 6×6 lattice, so per-layer fixed costs are multiplied by a
/// large constant in any real run — which no existing bench captured.
fn bench_propagate_trotter(c: &mut Criterion) {
    let mut group = c.benchmark_group("propagate_trotter_step");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));

    let num_qubits = 32usize;
    let theta = 0.1;

    let mut circuit = Circuit::<1>::new(num_qubits);
    for q in 0..num_qubits {
        let q0 = q as u32;
        let q1 = ((q + 1) % num_qubits) as u32;
        circuit.push(zz_rotation::<1>(q0, q1, 2.0 * theta));
    }
    for q in 0..num_qubits {
        let qq = q as u32;
        let gen = PauliString::<1>::x(qq);
        circuit.push(PauliRotation::new(gen, 2.0 * theta));
    }

    // Kept small: a 64-channel circuit with fanout-2 channels and no truncation
    // grows the sum by up to 2^64, so `n` here is the *starting* size and the
    // bench measures early-growth behaviour, not steady state.
    for &n in &[1_000usize, 10_000usize] {
        let input: PauliSum<1> = low_weight_sum::<1>(n, num_qubits, 3, 0x77077 + n as u64);
        group.throughput(Throughput::Elements(input.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                black_box(propagate(
                    black_box(&circuit),
                    input.clone(),
                    &AlwaysKeep,
                    Direction::Heisenberg,
                ))
            })
        });
    }

    group.finish();
}

/// Thread scaling of a single layer.
///
/// v0.1 §9 claims "near-linear scaling up to the memory bandwidth limit" and
/// nothing in the repo has ever measured it. The expectation for the current
/// engine is *poor* scaling, because `sort_phase` is sequential
/// (`sort_merge.rs:215`) — Amdahl bounds the whole layer by whatever fraction
/// the sort takes. That fraction is precisely what v0.2 removes, so this group
/// is the headline before/after comparison.
///
/// Uses a fanout-2 rotation so both the scan and the merge have real work.
fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling_rotation_1e6");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let input: PauliSum<2> = random_sum::<2>(n, 128, 0x74_12_EA_D5_u64);
    let rot = zz_rotation::<2>(0, 1, 0.1);
    let policy = AlwaysKeep;
    group.throughput(Throughput::Elements(input.len() as u64));

    for &t in &[1usize, 2, 4, 8, 16, 32] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("failed to build rayon pool");
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |bencher, _| {
            bencher
                .iter(|| pool.install(|| black_box(apply_layer(black_box(&input), &rot, &policy))))
        });
    }

    group.finish();
}

/// Bucket count the engine would pick for `n` terms at the given thread count.
fn bits_for(n: usize, threads: usize) -> u8 {
    paulistrings::bucket::desired_bits(n, DEFAULT_TARGET_BUCKET_LEN, 4 * threads)
}

/// One `apply_layer_bucketed` case, on an already-bucketed sum.
///
/// The layer is applied **in place, repeatedly**, with no per-iteration reset.
/// That is sound rather than sloppy: every channel's delta set `D` is a subspace,
/// so after one layer the key set is closed under `D` and the term count
/// stabilizes — a rotation, for instance, produces `S ∪ (S ⊕ gen)` and then stops
/// growing, because that set is already closed under `⊕ gen`. Criterion's warm-up
/// reaches the fixed point before timing starts. Resetting instead would mean
/// cloning or re-scattering a 10⁶-term sum per iteration, which would dominate
/// the measurement.
fn bucketed_layer_case<const W: usize, C>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: String,
    input: &PauliSum<W>,
    ch: &C,
    threads: usize,
) where
    C: Channel<W> + ?Sized,
{
    let policy = AlwaysKeep;
    let hash = Gf2Hash::<W>::new(input.num_qubits(), bits_for(input.len(), threads), 0xBEEF);
    let mut sum = BucketedSum::from_sum(input, hash);
    let prep = ch
        .prepare(sum.hash(), false)
        .expect("channel could not be prepared");
    let mut scratch = LayerScratch::<W>::new();

    group.throughput(Throughput::Elements(input.len() as u64));
    group.bench_function(label, |bencher| {
        bencher.iter(|| {
            apply_layer_bucketed(&mut sum, black_box(&prep), &policy, &mut scratch);
            black_box(sum.len())
        })
    });
}

/// The v0.2 bucketed engine on the same four channel classes as
/// `bench_apply_layer`, so the two groups are directly comparable.
fn bench_apply_layer_bucketed(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_layer_bucketed");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let h = Clifford1Q::h(0);
    let cnot = Clifford2Q::cnot(0, 1);
    let depol = Depolarizing {
        support: [0],
        p: 0.05,
    };
    let rot = zz_rotation::<2>(0, 1, 0.1);
    let threads = rayon::current_num_threads().max(1);

    for &n in &[10_000usize, 1_000_000usize] {
        let input: PauliSum<2> = random_sum::<2>(n, 128, 0xC0FFEE);
        if n >= 1_000_000 {
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(500));
        } else {
            group.sample_size(30);
            group.warm_up_time(Duration::from_secs(1));
        }
        bucketed_layer_case(
            &mut group,
            format!("depolarizing/{n}"),
            &input,
            &depol,
            threads,
        );
        bucketed_layer_case(&mut group, format!("clifford1q_h/{n}"), &input, &h, threads);
        bucketed_layer_case(
            &mut group,
            format!("clifford2q_cnot/{n}"),
            &input,
            &cnot,
            threads,
        );
        bucketed_layer_case(
            &mut group,
            format!("rotation_zz/{n}"),
            &input,
            &rot,
            threads,
        );
    }

    group.finish();
}

/// Thread scaling of the bucketed engine, against `bench_thread_scaling`'s
/// v0.1 numbers on the same input and channel.
fn bench_thread_scaling_bucketed(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling_bucketed_rotation_1e6");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let input: PauliSum<2> = random_sum::<2>(n, 128, 0x74_12_EA_D5_u64);
    let rot = zz_rotation::<2>(0, 1, 0.1);
    let policy = AlwaysKeep;
    group.throughput(Throughput::Elements(input.len() as u64));

    for &t in &[1usize, 2, 4, 8, 16, 32] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("failed to build rayon pool");
        // Bucket count is fixed per thread count exactly as `propagate` would
        // choose it, so this measures the configuration users actually get.
        let hash = Gf2Hash::<2>::new(128, bits_for(input.len(), t), 0xBEEF);
        let mut sum = BucketedSum::from_sum(&input, hash);
        let prep = Channel::<2>::prepare(&rot, sum.hash(), false).unwrap();
        let mut scratch = LayerScratch::<2>::new();
        group.bench_with_input(BenchmarkId::from_parameter(t), &t, |bencher, _| {
            bencher.iter(|| {
                pool.install(|| {
                    apply_layer_bucketed(&mut sum, black_box(&prep), &policy, &mut scratch);
                    black_box(sum.len())
                })
            })
        });
    }

    group.finish();
}

/// The conversion cost users pay once per `propagate` call, amortized over every
/// layer in the circuit (4320 of them for one 6x6 Ising quench).
fn bench_bucket_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("bucket_conversion_1e6");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let input: PauliSum<2> = random_sum::<2>(n, 128, 0xC0FFEE);
    let threads = rayon::current_num_threads().max(1);
    let bits = bits_for(input.len(), threads);
    group.throughput(Throughput::Elements(input.len() as u64));

    group.bench_function("from_sum", |bencher| {
        bencher.iter(|| {
            let hash = Gf2Hash::<2>::new(128, bits, 0xBEEF);
            black_box(BucketedSum::from_sum(black_box(&input), hash))
        })
    });

    let hash = Gf2Hash::<2>::new(128, bits, 0xBEEF);
    let bucketed = BucketedSum::from_sum(&input, hash);
    group.bench_function("to_sum", |bencher| {
        bencher.iter(|| black_box(bucketed.to_sum()))
    });

    group.finish();
}

/// `TopN::finalize_layer_bucketed`, which `propagate` runs after **every**
/// channel — 4320 times for one 6x6 Ising quench — so its per-call cost is
/// multiplied by the layer count just as a layer's is.
fn bench_finalize_top_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("finalize_top_n");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let input: PauliSum<2> = random_sum::<2>(n, 128, 0x70_9E);
    let threads = rayon::current_num_threads().max(1);
    let hash = Gf2Hash::<2>::new(128, bits_for(input.len(), threads), 0xBEEF);
    group.throughput(Throughput::Elements(input.len() as u64));

    // Keep 80%: enough that the cut is real but the sum does not collapse, so
    // repeated application is stable and the measurement is of the selection
    // rather than of a shrinking input.
    let keep = (input.len() * 4) / 5;
    group.bench_function("bucketed/keep80pct", |bencher| {
        bencher.iter_batched_ref(
            || BucketedSum::from_sum(&input, hash.clone()),
            |sum| {
                paulistrings::truncation::TopN(keep).finalize_layer_bucketed(sum);
                black_box(sum.len())
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("flat/keep80pct", |bencher| {
        bencher.iter_batched_ref(
            || input.clone(),
            |sum| {
                paulistrings::truncation::TopN(keep).finalize_layer(sum);
                black_box(sum.len())
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

/// Sweep the terms-per-bucket target, which sets the per-bucket sort size.
///
/// `DEFAULT_TARGET_BUCKET_LEN = 1024` was chosen from cache arithmetic alone (a
/// `W=2` term is 48 B, so 1024 terms is ~48 KB against 1 MiB of L2 per core).
/// This measures the curve instead of trusting that. Two effects pull against
/// each other: larger buckets mean `O(m log m)` grows per element, smaller
/// buckets mean more per-bucket fixed cost and more Rayon tasks.
fn bench_bucket_size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("bucket_size_sweep_rotation_1e6");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let input: PauliSum<2> = random_sum::<2>(n, 128, 0x5_1EE0);
    let rot = zz_rotation::<2>(0, 1, 0.1);
    let policy = AlwaysKeep;
    group.throughput(Throughput::Elements(input.len() as u64));

    // bits chosen directly so the sweep is over bucket size, not over policy.
    for &bits in &[4u8, 6, 8, 10, 12, 14] {
        let per_bucket = n >> bits;
        let hash = Gf2Hash::<2>::new(128, bits, 0xBEEF);
        let mut sum = BucketedSum::from_sum(&input, hash);
        let prep = Channel::<2>::prepare(&rot, sum.hash(), false).unwrap();
        let mut scratch = LayerScratch::<2>::new();
        group.bench_function(format!("{per_bucket}_per_bucket"), |bencher| {
            bencher.iter(|| {
                apply_layer_bucketed(&mut sum, black_box(&prep), &policy, &mut scratch);
                black_box(sum.len())
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mul_assign,
    bench_pauli_sum_add,
    bench_apply_layer,
    bench_apply_layer_bucketed,
    bench_apply_layer_occupancy,
    bench_propagate_trotter,
    bench_thread_scaling,
    bench_thread_scaling_bucketed,
    bench_bucket_conversion,
    bench_finalize_top_n,
    bench_bucket_size_sweep
);
criterion_main!(benches);
