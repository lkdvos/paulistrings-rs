//! Criterion microbenches for the hot ops on the propagation path.
//!
//! `Clifford1Q::h`, fanout-1 and key-bijective, is the *worst* case for the
//! sort phase and the *best* case for the merge phase — one point on the
//! layer cost surface, not representative on its own.
//!
//! What is measured here:
//!
//!   * `PauliString::mul_assign` — single-term multiplication (W ∈ {1, 2}).
//!   * `PauliSum::add` — sorted merge of two SoA sums (N ∈ {10⁴, 10⁶}). This is
//!     the useful reference point for one bucketed layer: a full two-pointer
//!     merge of the same payload, with allocation, and no sort.
//!   * `apply_layer_bucketed` across the structurally distinct channel classes
//!     (see the group's own doc), plus the two extreme fan-in shapes.
//!   * `propagate` over a multi-channel Trotter-shaped circuit, which is the
//!     shape real workloads have (`examples/ising_2d_quench.rs` is 108 channels
//!     per step).
//!   * Thread scaling of one bucketed layer at 1/2/4/8/16/32 threads, against
//!     the memory-bandwidth ceiling in ARCHITECTURE.md §Performance-Model.
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
use paulistrings::bucket::{Gf2Hash, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use paulistrings::channel::{
    Channel, Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation,
};
use paulistrings::circuit::Circuit;
use paulistrings::engine::bucketed::{apply_layer_bucketed, LayerScratch};
use paulistrings::engine::{propagate, Direction};
use paulistrings::pauli_string::PauliString;
use paulistrings::pauli_sum::PauliSum;
use paulistrings::phase::Phase;
// The input generators are the shared fixtures. `rand_sum_unmasked` /
// `tie_heavy_sum_unmasked` are the *unmasked, word-major* variants this file
// has always used — a different draw order from `rand_sum`, so the committed
// criterion baselines are pinned to them specifically.
use paulistrings::test_support::{
    low_weight_sum, rand_pauli, rand_sum_unmasked, tie_heavy_sum_unmasked, Xs64,
};
use paulistrings::truncation::TruncationPolicy;
use std::hint::black_box;
use std::time::Duration;

/// A truncation policy that never drops anything. Mirrors the `AlwaysKeep`
/// helper in the engine tests; the trait default does what we want.
struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

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

/// A Pauli generator of weight 4 on `qubits`, for a rotation whose support
/// exceeds `MAX_LOCAL_SUPPORT` and therefore cannot be tabulated as a local PTM.
///
/// The delta set is still `{0, gen}` — 2 input buckets per output — but the
/// `i^k` phase is computed per term rather than looked up, so this is the other
/// half of the rotation cost surface from `zz_rotation`.
fn wide_rotation<const W: usize>(qubits: [u32; 4], theta: f64) -> PauliRotation<W> {
    let mut gen = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    // Mixed X/Z letters so the generator is not a product of commuting Zs.
    for (i, &q) in qubits.iter().enumerate() {
        let word = (q as usize) / 64;
        let bit = 1u64 << (q % 64);
        if i % 2 == 0 {
            gen.z[word] |= bit;
        } else {
            gen.x[word] |= bit;
        }
    }
    PauliRotation::new(gen, theta)
}

/// sqrt(SWAP) on `(q0, q1)` — a fixed non-Clifford two-qubit unitary.
///
/// Non-Clifford, so its Pauli-expansion table does not collapse to a
/// permutation: the prepared delta set is wide (up to 16), which is the maximum
/// bucket fan-in the engine ever sees.
fn sqrt_swap(q0: u32, q1: u32) -> GeneralUnitary2Q {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    GeneralUnitary2Q::from_matrix(
        q0,
        q1,
        [
            [one, zero, zero, zero],
            [zero, h, hc, zero],
            [zero, hc, h, zero],
            [zero, zero, zero, one],
        ],
    )
}

/// PauliString mul_assign — the inner hot loop of every channel that does a
/// runtime Pauli multiplication (rotation, general unitary). Cliffords avoid
/// it via lookup tables, so this bench targets the rotation/unitary path.
fn bench_mul_assign(c: &mut Criterion) {
    let mut group = c.benchmark_group("pauli_string_mul_assign");
    group.throughput(Throughput::Elements(1));

    {
        let mut rng = Xs64::new(0xA11CE);
        let a: PauliString<1> = rand_pauli(&mut rng);
        let b: PauliString<1> = rand_pauli(&mut rng);
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
        let a: PauliString<2> = rand_pauli(&mut rng);
        let b: PauliString<2> = rand_pauli(&mut rng);
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
        let a: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0xADDA);
        let b: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0xADDB);
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

/// Bucket count the engine would pick for `n` terms.
fn bits_for(n: usize) -> u8 {
    paulistrings::bucket::desired_bits(n, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS)
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
///
/// Because of that closure, `sum.len()` after warm-up can be up to ~2x the
/// initial `input.len()` (e.g. a rotation's `S ∪ (S ⊕ gen)`). Elements =
/// steady-state input terms per layer application, so the layer is applied a
/// few times up front to reach the fixed point *before* throughput is read,
/// rather than using the pre-closure `input.len()` as the denominator.
fn bucketed_layer_case<const W: usize, C>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    label: String,
    input: &PauliSum<W>,
    ch: &C,
) where
    C: Channel<W> + ?Sized,
{
    let policy = AlwaysKeep;
    let hash = Gf2Hash::<W>::new(input.num_qubits(), bits_for(input.len()), 0xBEEF);
    let mut sum = input.clone().with_hash(hash);
    let prep = ch
        .prepare(sum.hash(), false)
        .expect("channel could not be prepared");
    let mut scratch = LayerScratch::<W>::new();

    // Warm to the fixed point before reading throughput.
    for _ in 0..3 {
        apply_layer_bucketed(&mut sum, &prep, &policy, &mut scratch);
    }

    // Elements = steady-state input terms per layer application.
    group.throughput(Throughput::Elements(sum.len() as u64));
    group.bench_function(label, |bencher| {
        bencher.iter(|| {
            apply_layer_bucketed(&mut sum, black_box(&prep), &policy, &mut scratch);
            black_box(sum.len())
        })
    });
}

/// The bucketed engine on the four structurally distinct channel classes,
/// plus the two extreme fan-in shapes.
///
///   * `depolarizing` — keys bitwise **unchanged**, coefficients rescaled;
///     delta set `{0}`, so one input bucket per output.
///   * `clifford1q_h` — key **bijection**, fanout 1; the delta set is
///     1-dimensional, so two input buckets per output.
///   * `clifford2q_cnot` — key bijection on 2 qubits, fanout 1; 2-dimensional
///     delta set, four buckets.
///   * `rotation_zz` — fanout 2, **non-injective**: the only case where the
///     merge has real duplicate keys to combine. Delta set `{0, gen}`, so two
///     buckets regardless of generator weight.
///
///   * `general_unitary2q` — up to 16 deltas, so each output bucket gathers
///     from 16 input buckets.
///   * `rotation_w4` — 2 deltas, but a per-term phase computation instead of a
///     table lookup.
///
/// Those two bracket the read amplification from fan-in
/// (ARCHITECTURE.md §Bucketing — up to 16 input buckets per output bucket
/// for a dense two-qubit unitary). Measured at 10⁶ only: at 10⁴ the whole
/// sum is in cache and fan-in costs nothing.
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
    let gu2q = sqrt_swap(0, 1);
    let rot_w4 = wide_rotation::<2>([0, 1, 2, 3], 0.1);

    for &n in &[10_000usize, 1_000_000usize] {
        let input: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0xC0FFEE);
        if n >= 1_000_000 {
            group.sample_size(10);
            group.warm_up_time(Duration::from_millis(500));
        } else {
            group.sample_size(30);
            group.warm_up_time(Duration::from_secs(1));
        }
        bucketed_layer_case(&mut group, format!("depolarizing/{n}"), &input, &depol);
        bucketed_layer_case(&mut group, format!("clifford1q_h/{n}"), &input, &h);
        bucketed_layer_case(&mut group, format!("clifford2q_cnot/{n}"), &input, &cnot);
        bucketed_layer_case(&mut group, format!("rotation_zz/{n}"), &input, &rot);
        if n >= 1_000_000 {
            bucketed_layer_case(&mut group, format!("general_unitary2q/{n}"), &input, &gu2q);
            bucketed_layer_case(&mut group, format!("rotation_w4/{n}"), &input, &rot_w4);
        }
    }

    group.finish();
}

/// Thread scaling of the bucketed engine on a single rotation layer.
///
/// As in `bucketed_layer_case`, the layer closes the key set under the
/// rotation's delta span, so `sum.len()` after warm-up can be up to ~2x the
/// initial `input.len()`. `input` is warmed to that fixed point once, before
/// it is cloned per thread count, so every thread count starts from (and
/// stays at) the same steady state, and elements = steady-state input terms
/// per layer application.
fn bench_thread_scaling_bucketed(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling_bucketed_rotation_1e6");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let mut input: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0x74_12_EA_D5_u64);
    let rot = zz_rotation::<2>(0, 1, 0.1);
    let policy = AlwaysKeep;

    // Warm `input` itself to the fixed point once, before the per-thread-count
    // clones below, so every thread count clones the same steady state. This
    // also means `bits_for` in the loop below sees the *warmed* length,
    // exactly as `propagate`'s per-layer rebucket would.
    {
        let hash = Gf2Hash::<2>::new(128, bits_for(input.len()), 0xBEEF);
        let mut warm = input.clone().with_hash(hash);
        let prep = Channel::<2>::prepare(&rot, warm.hash(), false).unwrap();
        let mut scratch = LayerScratch::<2>::new();
        for _ in 0..3 {
            apply_layer_bucketed(&mut warm, &prep, &policy, &mut scratch);
        }
        input = warm;
    }

    // Elements = steady-state input terms per layer application.
    group.throughput(Throughput::Elements(input.len() as u64));

    for &t in &[1usize, 2, 4, 8, 16, 32] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("failed to build rayon pool");
        // Bucket count does not depend on the thread count
        // (ARCHITECTURE.md §Bucket-Policy), so a fixed bit count across
        // every thread count in this loop IS the configuration users
        // actually get, which makes the scaling measurement purer than
        // varying it per thread count would.
        let hash = Gf2Hash::<2>::new(128, bits_for(input.len()), 0xBEEF);
        let mut sum = input.clone().with_hash(hash);
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

/// Thread scaling on the widest delta set the engine has.
///
/// `bench_thread_scaling_bucketed` uses a 2-delta rotation, where each output
/// bucket reads 2 input buckets. `GeneralUnitary2Q` reads 16, so the same input
/// is streamed 16 times per layer — read amplification the engine bounds by
/// walking cosets of `h(D)` rather than gathering globally
/// (ARCHITECTURE.md §Bucketing). If that amplification is bandwidth-bound
/// then this group flattens earlier than the rotation one, and the gap
/// between them is the
/// size of the prize.
///
/// Threads are {1, 8, 32} rather than the full sweep: three points fix the
/// curve's ends and its knee, and a 16-fold gather at 10⁶ terms is expensive
/// enough that the full sweep would not pay for itself.
///
/// As in `bench_thread_scaling_bucketed`, `input` is warmed to the fixed
/// point once, before it is cloned per thread count, so elements =
/// steady-state input terms per layer application.
fn bench_thread_scaling_bucketed_gu2q(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling_bucketed_gu2q");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let mut input: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0x74_12_EA_D5_u64);
    let gu2q = sqrt_swap(0, 1);
    let policy = AlwaysKeep;

    // Warm `input` itself to the fixed point once, before the per-thread-count
    // clones below, so every thread count clones the same steady state. This
    // also means `bits_for` in the loop below sees the *warmed* length,
    // exactly as `propagate`'s per-layer rebucket would.
    {
        let hash = Gf2Hash::<2>::new(128, bits_for(input.len()), 0xBEEF);
        let mut warm = input.clone().with_hash(hash);
        let prep = Channel::<2>::prepare(&gu2q, warm.hash(), false).unwrap();
        let mut scratch = LayerScratch::<2>::new();
        for _ in 0..3 {
            apply_layer_bucketed(&mut warm, &prep, &policy, &mut scratch);
        }
        input = warm;
    }

    // Elements = steady-state input terms per layer application.
    group.throughput(Throughput::Elements(input.len() as u64));

    for &t in &[1usize, 8, 32] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("failed to build rayon pool");
        // Same fixed bit count at every thread count, for the reason given in
        // `bench_thread_scaling_bucketed`.
        let hash = Gf2Hash::<2>::new(128, bits_for(input.len()), 0xBEEF);
        let mut sum = input.clone().with_hash(hash);
        let prep = Channel::<2>::prepare(&gu2q, sum.hash(), false).unwrap();
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

/// Partition maintenance cost: a full repartition under a different hash
/// (`with_hash`, the worst case — flatten plus rescatter).
/// Cost of ingestion: `BuildAccumulator::finalize` picks the hash and scatters
/// terms straight into their buckets, so there is no separate "convert a flat
/// sum into a bucketed one" step to measure.
fn bench_ingest_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_finalize_1e6");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let num_qubits = 128;
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("finalize", |bencher| {
        bencher.iter_batched(
            || {
                let mut rng = Xs64::new(0xC0FFEE);
                let mut acc = BuildAccumulator::<2>::with_capacity(num_qubits, n);
                for _ in 0..n {
                    let p = rand_pauli::<2>(&mut rng);
                    let re = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
                    let im = (rng.next_u64() as i64 as f64) / (i64::MAX as f64);
                    acc.add_term(p, Phase::ONE, Complex64::new(re, im));
                }
                acc
            },
            |acc| black_box(acc.finalize()),
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

/// Cost of partition maintenance: a `refine` (double the bucket count) then a
/// `coarsen` (halve it back) round trip, the remaining O(n) work a layer pays
/// to keep the bucket count matched to the live term count.
fn bench_rebucket(c: &mut Criterion) {
    let mut group = c.benchmark_group("rebucket_1e6");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let input: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0xC0FFEE);
    group.throughput(Throughput::Elements(input.len() as u64));

    group.bench_function("refine_coarsen", |bencher| {
        bencher.iter_batched(
            || input.clone(),
            |mut sum| {
                sum.refine();
                sum.coarsen();
                black_box(sum.len())
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

/// `TopN::finalize_layer`, which `propagate` runs after **every** channel —
/// 4320 times for one 6x6 Ising quench — so its per-call cost is multiplied by
/// the layer count just as a layer's is.
fn bench_finalize_top_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("finalize_top_n");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));

    let n = 1_000_000usize;
    let input: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0x70_9E);
    let hash = Gf2Hash::<2>::new(128, bits_for(input.len()), 0xBEEF);
    group.throughput(Throughput::Elements(input.len() as u64));

    // Keep 80%: enough that the cut is real but the sum does not collapse, so
    // repeated application is stable and the measurement is of the selection
    // rather than of a shrinking input.
    let keep = (input.len() * 4) / 5;
    group.bench_function("bucketed/keep80pct", |bencher| {
        bencher.iter_batched_ref(
            || input.clone().with_hash(hash.clone()),
            |sum| {
                paulistrings::truncation::TopN(keep).finalize_layer(sum);
                black_box(sum.len())
            },
            BatchSize::LargeInput,
        )
    });

    // The same selection on the accumulator's own default partition — what a
    // `propagate` caller actually gets at this size.
    group.bench_function("default_partition/keep80pct", |bencher| {
        bencher.iter_batched_ref(
            || input.clone(),
            |sum| {
                paulistrings::truncation::TopN(keep).finalize_layer(sum);
                black_box(sum.len())
            },
            BatchSize::LargeInput,
        )
    });

    // Tie-dense: the same size and the same 80% cut, but only four distinct
    // magnitudes, so the tie group at the threshold is ~25% of the sum. This is
    // the shape a symmetric Hamiltonian produces, and it is the only shape that
    // exercises the group-detection pass (ARCHITECTURE.md §Truncation) on a
    // non-trivial group — random coefficients give `count_eq == 1` and the
    // counting reduce sees a degenerate case. The `rand_sum_unmasked` cases
    // above stay as the contrast.
    let tied: PauliSum<2> = tie_heavy_sum_unmasked::<2>(n, 128, 0x70_9F);
    let tied_hash = Gf2Hash::<2>::new(128, bits_for(tied.len()), 0xBEEF);
    let tied_keep = (tied.len() * 4) / 5;
    group.bench_function("bucketed/tie_heavy_keep80pct", |bencher| {
        bencher.iter_batched_ref(
            || tied.clone().with_hash(tied_hash.clone()),
            |sum| {
                paulistrings::truncation::TopN(tied_keep).finalize_layer(sum);
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
    let input: PauliSum<2> = rand_sum_unmasked::<2>(n, 128, 0x5_1EE0);
    let rot = zz_rotation::<2>(0, 1, 0.1);
    let policy = AlwaysKeep;
    group.throughput(Throughput::Elements(input.len() as u64));

    // bits chosen directly so the sweep is over bucket size, not over policy.
    for &bits in &[4u8, 6, 8, 10, 12, 14] {
        let per_bucket = n >> bits;
        let hash = Gf2Hash::<2>::new(128, bits, 0xBEEF);
        let mut sum = input.clone().with_hash(hash);
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
    bench_apply_layer_bucketed,
    bench_propagate_trotter,
    bench_thread_scaling_bucketed,
    bench_thread_scaling_bucketed_gu2q,
    bench_ingest_finalize,
    bench_rebucket,
    bench_finalize_top_n,
    bench_bucket_size_sweep
);
criterion_main!(benches);
