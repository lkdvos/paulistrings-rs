use super::*;
use crate::engine::backend::PartitionBackend;
use crate::engine::gpu::driver::GpuPartitionedSum;
use crate::engine::gpu::layer::GpuLayerOptions;
use crate::engine::gpu::partition::DevicePartition;
use crate::engine::partitioned::in_process::InProcessTransport;
use crate::engine::transport::SoloTransport;
use crate::engine::Direction;
use crate::pauli_sum::{Gf2Hash, PauliSum};
use crate::test_support::{
    assert_systematic_resample, assert_terms_close, coefficient_tolerance, collapsing_circuit,
    ordered_sum, rand_sum, rand_sum_real, z0_sum,
};
use crate::truncation::{And, BuiltinTruncation as T, CoefficientThreshold, WeightCutoff};
use crate::{Circuit, TruncationPolicy};
use num_complex::{Complex32, Complex64};

fn partition<const W: usize, C: Coefficient>(input: &PauliSum<W, C>) -> DevicePartition<W, C> {
    DevicePartition::new(GpuSum::from_host(input, 0).expect("upload"), GpuLayerOptions::default())
        .expect("partition")
}

fn widened<const W: usize, C: Coefficient>(sum: &PauliSum<W, C>) -> PauliSum<W> {
    sum.cast::<Complex64>(0.0).expect("every coefficient widens")
}

/// `sum`'s terms in device storage order, the order a device collapse lays its stretches in: buckets by index, each as stored.
fn storage_order<const W: usize, C: Coefficient>(
    sum: &GpuSum<W, C>,
) -> Vec<([u64; W], [u64; W], C)> {
    let b = sum.hash.num_buckets();
    let stream = &sum.stream;
    let start = stream.clone_dtoh(&sum.columns.start.slice(0..b)).unwrap();
    let lengths = stream.clone_dtoh(&sum.columns.lengths.slice(0..b)).unwrap();
    let x = stream.clone_dtoh(&sum.columns.x).unwrap();
    let z = stream.clone_dtoh(&sum.columns.z).unwrap();
    let coeff = stream.clone_dtoh(&sum.columns.coeff).unwrap();
    stream.synchronize().unwrap();
    let coeff: &[C] = bytemuck::cast_slice(&coeff[..coeff.len() / size_of::<C>() * size_of::<C>()]);
    let key =
        |column: &[u64], i: usize| -> [u64; W] { column[i * W..(i + 1) * W].try_into().unwrap() };
    (0..b)
        .flat_map(|bucket| {
            let first = start[bucket] as usize;
            first..first + lengths[bucket] as usize
        })
        .map(|i| (key(&x, i), key(&z, i), coeff[i]))
        .collect()
}

/// `|c|`-weighted coefficients of `C` over many buckets of a 16-bit hash.
fn fixture<const W: usize, C: Coefficient>(seed: u64) -> PauliSum<W, C> {
    let complex =
        if C::IS_REAL { rand_sum_real::<W>(3000, 16, seed) } else { rand_sum::<W>(3000, 16, seed) };
    let narrowed = complex.cast::<C>(1e-12).expect("the fixture fits the coefficient type");
    narrowed.with_hash(Gf2Hash::new(16, 5, 0x1234))
}

/// On a sum in the host's order the device draws the host's terms with the host's multiplicities, for every coefficient type and a cache that holds or does not.
fn check_one_device_against_the_host<const W: usize, C: Coefficient>() {
    let input = fixture::<W, C>(0xC0 + W as u64);
    let len = input.len();
    let terms: Vec<_> = input.iter().map(|(x, z, c)| (*x, *z, c)).collect();
    let total: f64 = terms.iter().map(|term| term.2.magnitude()).sum();
    for (seed, samples) in [(1u64, 1usize), (2, 7), (3, 300), (4, 2000), (5, 5000)] {
        let what = format!("{} W={W} samples={samples}", C::NAME);
        let host = T::from(CollapseSample::new(len - 1, samples, seed));
        let device = T::from(CollapseSample::new(len - 1, samples, seed));
        let mut want = input.clone();
        <T as TruncationPolicy<W, C>>::finalize_layer(&host, &mut want);
        let mut part = partition(&input);
        PartitionBackend::<W, C, T>::finalize_layer(&mut part, &device, &SoloTransport);
        part.take_error().expect("device collapse");
        part.sum().assert_invariants_device().unwrap_or_else(|e| panic!("{what}: {e}"));
        let got = part.sum().to_host().expect("download");
        let step = total / samples as f64;
        let tolerance = 4.0 * coefficient_tolerance::<C>() * step;
        assert_terms_close(&widened(&got), &widened(&want), tolerance, &what);
        let T::CollapseSample(sampler) = &device else { unreachable!() };
        let u = sampler.uniform(0);
        assert_systematic_resample(&got, &terms, u, samples, coefficient_tolerance::<C>(), &what);
        assert_eq!(sampler.collapses(), 1, "{what}: collapses");
    }
}

#[test]
fn a_device_collapse_draws_the_hosts_terms() {
    crate::require_cuda!();
    check_one_device_against_the_host::<1, f64>();
    check_one_device_against_the_host::<2, f64>();
    check_one_device_against_the_host::<1, f32>();
    check_one_device_against_the_host::<2, f32>();
    check_one_device_against_the_host::<1, Complex64>();
    check_one_device_against_the_host::<2, Complex64>();
    check_one_device_against_the_host::<1, Complex32>();
}

/// The resample of `coefficients`, laid out in one bucket in order, at offset `u` with `samples` points, by `x[0]`.
fn device_resampled<const W: usize>(
    coefficients: &[f64],
    u: f64,
    samples: usize,
) -> Vec<(u64, f64)> {
    let mut part = partition(&ordered_sum::<W, f64>(coefficients));
    let (sum, scratch) = part.parts_mut();
    resample_device(sum, scratch, u, samples).expect("device resample");
    let mut out: Vec<(u64, f64)> =
        sum.to_host().expect("download").iter().map(|(x, _, c)| (x[0], c)).collect();
    out.sort_by_key(|&(i, _)| i);
    out
}

/// The host's hand-worked draws: boundary points go to the later stretch, an empty stretch draws nothing, and a point rounded onto the total goes to the last nonzero stretch.
fn check_hand_picks<const W: usize>() {
    let signed = [1.0, 0.0, -2.5, 0.5];
    assert_eq!(device_resampled::<W>(&signed, 0.25, 4), [(1, 1.0), (3, -3.0)]);
    assert_eq!(device_resampled::<W>(&signed, 0.0, 4), [(1, 1.0), (3, -3.0)]);
    assert_eq!(device_resampled::<W>(&signed, 0.5, 4), [(1, 1.0), (3, -2.0), (4, 1.0)]);
    assert_eq!(device_resampled::<W>(&signed, 0.0, 2), [(1, 2.0), (3, -2.0)]);
    assert_eq!(device_resampled::<W>(&signed, 0.5, 8), [(1, 1.0), (3, -2.5), (4, 0.5)]);
    let last_uniform = 1.0 - 1.0 / (1u64 << 53) as f64;
    assert_eq!(device_resampled::<W>(&[1.0, -2.0, 0.0], last_uniform, 3), [(1, 1.0), (2, -2.0)]);
}

#[test]
fn a_device_collapse_matches_the_hand_picks() {
    crate::require_cuda!();
    check_hand_picks::<1>();
    check_hand_picks::<2>();
}

/// After device layers, whose buckets hold their terms in no particular order, the collapse is the systematic draw over the storage order.
fn check_after_layers<const W: usize>() {
    let circuit = crate::test_support::trotter_circuit::<W>(8, 0.4);
    let input = rand_sum::<W>(200, 8, 0xA7 + W as u64);
    let mut resident = GpuPartitionedSum::<W>::from_host(&input, 0).expect("upload");
    resident.propagate(&circuit, T::Keep, Direction::Heisenberg).expect("layers");
    let (sum, scratch) = resident.backends_mut()[0].parts_mut();
    let terms = storage_order(sum);
    assert!(terms.len() > 500, "the layers must grow the sum, {} terms", terms.len());
    resample_device(sum, scratch, 0.6, 300).expect("device resample");
    let got = sum.to_host().expect("download");
    assert_systematic_resample(&got, &terms, 0.6, 300, 1e-12, &format!("W={W} after layers"));
}

/// The device counts draws in 32 bits, so more than `u32::MAX` is `Unsupported` and leaves the sum as it was.
#[test]
fn more_draws_than_u32_is_unsupported_on_the_device() {
    crate::require_cuda!();
    let input = rand_sum::<1>(200, 8, 0xD4A);
    let mut part = partition(&input);
    let (sum, scratch) = part.parts_mut();
    let too_many = u32::MAX as usize + 1;
    assert!(matches!(resample_device(sum, scratch, 0.5, too_many), Err(GpuError::Unsupported(_))));
    assert_terms_close(&sum.to_host().expect("download"), &input, 1e-300, "untouched");
}

#[test]
fn a_collapse_after_device_layers_draws_in_storage_order() {
    crate::require_cuda!();
    check_after_layers::<1>();
    check_after_layers::<2>();
}

/// A sum at or below the cache is left exactly as it was, and the pass still counts.
#[test]
fn a_sum_within_the_cache_is_untouched() {
    crate::require_cuda!();
    let input = fixture::<1, Complex64>(0xCA);
    let policy = T::from(CollapseSample::new(input.len(), 1, 3));
    let mut part = partition(&input);
    PartitionBackend::<1, Complex64, T>::finalize_layer(&mut part, &policy, &SoloTransport);
    part.take_error().expect("device pass");
    assert_eq!(part.sum().to_host().unwrap().to_arrays(), input.to_arrays());
    let T::CollapseSample(sampler) = &policy else { unreachable!() };
    assert_eq!(sampler.collapses(), 0);
    assert_eq!(sampler.next_call(), 1, "the pass was counted");

    let circuit = collapsing_circuit();
    let roomy = CollapseSample::new(usize::MAX, 1, 3);
    let mut run = GpuPartitionedSum::from_host(&z0_sum(), 0).unwrap();
    let roomy = std::sync::Arc::new(roomy);
    run.propagate(&circuit, roomy.clone(), Direction::Heisenberg).unwrap();
    let mut keep = GpuPartitionedSum::from_host(&z0_sum(), 0).unwrap();
    keep.propagate(&circuit, T::Keep, Direction::Heisenberg).unwrap();
    assert_terms_close(&run.gather().unwrap(), &keep.gather().unwrap(), 1e-12, "roomy cache");
    assert_eq!(roomy.collapses(), 0);
}

/// `input`'s terms whose `x[0]` is `rank` modulo `size`, in the same buckets and order.
fn share<const W: usize, C: Coefficient>(
    input: &PauliSum<W, C>,
    rank: usize,
    size: usize,
) -> PauliSum<W, C> {
    let mut out = input.clone();
    out.update_per_bucket(|_| {
        move |x: &[u64; W], _: &[u64; W], c: C| (x[0] as usize % size == rank).then_some(c)
    });
    out
}

/// Each rank's share after one partitioned pass of `policy` over `shares`, the host's or the device's.
fn group_pass<const W: usize, C: Coefficient>(
    shares: &[PauliSum<W, C>],
    policy: &T,
    device: bool,
) -> Vec<PauliSum<W, C>> {
    let transports = InProcessTransport::group(shares.len() as u32);
    std::thread::scope(|scope| {
        let ranks: Vec<_> = shares
            .iter()
            .zip(transports)
            .map(|(share, transport)| {
                scope.spawn(move || {
                    if device {
                        let mut part = partition(share);
                        PartitionBackend::<W, C, T>::finalize_layer(&mut part, policy, &transport);
                        part.take_error().expect("device group collapse");
                        part.sum().to_host().expect("download")
                    } else {
                        let mut out = share.clone();
                        <T as TruncationPolicy<W, C>>::finalize_layer_partitioned(
                            policy, &mut out, &transport,
                        );
                        out
                    }
                })
            })
            .collect();
        ranks.into_iter().map(|rank| rank.join().expect("rank thread")).collect()
    })
}

/// Over a group the device draws the host's partitioned collapse rank by rank: the global prefix places each partition's points, one partition may hold none.
fn check_group<const W: usize, C: Coefficient>(size: usize) {
    let input = fixture::<W, C>(0x6A + size as u64);
    let total: f64 = input.iter().map(|(_, _, c)| c.magnitude()).sum();
    let mut shares: Vec<_> = (0..size).map(|rank| share(&input, rank, size)).collect();
    shares.push(PauliSum::<W, C>::empty(16).with_hash(input.hash().clone()));
    for (seed, samples) in [(7u64, 3usize), (8, 900)] {
        let what = format!("{} W={W} {} ranks samples={samples}", C::NAME, shares.len());
        let host = T::from(CollapseSample::new(100, samples, seed));
        let device = T::from(CollapseSample::new(100, samples, seed));
        let want = group_pass(&shares, &host, false);
        let got = group_pass(&shares, &device, true);
        let tolerance = 4.0 * coefficient_tolerance::<C>() * total / samples as f64;
        let kept: usize = got.iter().map(PauliSum::len).sum();
        assert!(kept <= samples, "{what}: {kept} terms");
        for (rank, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_terms_close(
                &widened(got),
                &widened(want),
                tolerance,
                &format!("{what} rank {rank}"),
            );
        }
        let (T::CollapseSample(h), T::CollapseSample(d)) = (&host, &device) else { unreachable!() };
        assert_eq!(
            (d.collapses(), d.next_call()),
            (h.collapses(), h.next_call()),
            "{what}: counters"
        );
    }
}

#[test]
fn a_device_group_collapse_is_the_hosts_partitioned_collapse() {
    crate::require_cuda!();
    check_group::<1, Complex64>(2);
    check_group::<2, f64>(3);
    check_group::<1, f32>(2);
}

/// `circuit`'s channels cut into the segments a Heisenberg run applies between its layer passes, `passes_after[i]` marking the application-order layer a pass follows; each segment is a circuit whose own Heisenberg run applies its layers in that order.
fn heisenberg_segments(mut circuit: Circuit<1>, passes_after: &[bool]) -> Vec<Circuit<1>> {
    let n = circuit.num_qubits;
    let mut segments = Vec::new();
    let mut segment = Vec::new();
    for (channel, &pass) in circuit.channels.drain(..).rev().zip(passes_after) {
        segment.push(channel);
        if pass {
            let mut layers = Circuit::new(n);
            for channel in segment.drain(..).rev() {
                layers.push_boxed(channel);
            }
            segments.push(layers);
        }
    }
    segments
}

/// A run's policy around its sampler.
type PolicyOf = fn(CollapseSample) -> T;

/// A device run with `CollapseSample` alone or after a per-term filter is its replay segment by segment: the filter over each layer or run the trace shows, then a collapse at `u` of pass `k` whenever the sum outgrew the cache.
#[test]
fn a_device_run_collapses_as_its_layer_by_layer_replay() {
    crate::require_cuda!();
    const CACHE: usize = 6;
    const SAMPLES: usize = 4;
    let filters: [(&str, PolicyOf, T); 3] = [
        ("alone", |c| T::from(c), T::Keep),
        (
            "after a threshold",
            |c| T::from(And(CoefficientThreshold(0.05), c)),
            T::from(CoefficientThreshold(0.05)),
        ),
        ("after a weight cutoff", |c| T::from(And(WeightCutoff(3), c)), T::from(WeightCutoff(3))),
    ];
    for (name, policy_of, filter) in filters {
        for seed in 0..4u64 {
            let what = format!("{name}, seed {seed}");
            let policy = policy_of(CollapseSample::new(CACHE, SAMPLES, seed));
            let mut run = GpuPartitionedSum::from_host(&z0_sum(), 0).unwrap();
            run.enable_trace();
            run.propagate(&collapsing_circuit(), policy.clone(), Direction::Heisenberg).unwrap();
            // A layer inside a run past its first records no time of its own, and the pass follows the run's last layer.
            let rows = run.take_trace().unwrap().layers;
            let passes_after: Vec<bool> = (0..rows.len())
                .map(|i| rows.get(i + 1).is_none_or(|next| next.nanos[0] != 0))
                .collect();
            assert!(passes_after.contains(&false), "{what}: the run took no run");
            let run = run.gather().unwrap();

            let reference = CollapseSample::new(CACHE, SAMPLES, seed);
            let mut replay = GpuPartitionedSum::from_host(&z0_sum(), 0).unwrap();
            let (mut collapses, mut call) = (0, 0u64);
            // Each segment repeats the run's own joins and walks, so the sum matches the run's storage order the collapse draws in.
            for segment in heisenberg_segments(collapsing_circuit(), &passes_after) {
                replay.propagate(&segment, filter.clone(), Direction::Heisenberg).unwrap();
                let (sum, scratch) = replay.backends_mut()[0].parts_mut();
                if sum.len() > CACHE {
                    resample_device(sum, scratch, reference.uniform(call), SAMPLES).unwrap();
                    collapses += 1;
                }
                call += 1;
            }
            assert!(collapses >= 3, "{what}: {collapses} collapses");
            assert!(run.len() <= CACHE, "{what}: {} terms", run.len());
            assert_terms_close(&run, &replay.gather().unwrap(), 1e-12, &what);
            let mut sampler = None;
            crate::engine::gpu::truncation::layer_pass_leaves(&policy, &mut |leaf| {
                if let T::CollapseSample(s) = leaf {
                    sampler = Some(s.clone());
                }
            });
            assert_eq!(sampler.unwrap().collapses(), collapses, "{what}: collapses");
        }
    }
}

/// The `⟨0…0|O|0…0⟩` readout of device trajectories that collapse at nearly every layer averages to the untruncated value within four standard errors.
#[test]
fn device_collapse_readout_is_unbiased() {
    crate::require_cuda!();
    use crate::{PauliAxis, ProductBasis};
    let (circuit, input) = (collapsing_circuit(), z0_sum());
    let zero = ProductBasis::uniform(PauliAxis::Z);
    let mut exact = GpuPartitionedSum::from_host(&input, 0).unwrap();
    exact.propagate(&circuit, T::Keep, Direction::Heisenberg).unwrap();
    let exact = exact.gather().unwrap().expectation_product(&zero).re;
    let seeds = 400u64;
    let (mut sum, mut squares, mut collapses) = (0.0f64, 0.0f64, 0u64);
    for seed in 0..seeds {
        let policy = std::sync::Arc::new(CollapseSample::new(6, 4, seed));
        let mut run = GpuPartitionedSum::from_host(&input, 0).unwrap();
        run.propagate(&circuit, policy.clone(), Direction::Heisenberg).unwrap();
        let readout = run.gather().unwrap().expectation_product(&zero).re;
        sum += readout;
        squares += readout * readout;
        collapses += policy.collapses();
    }
    let n = seeds as f64;
    let mean = sum / n;
    let error = ((squares / n - mean * mean) / (n - 1.0)).sqrt();
    assert!(collapses >= 5 * seeds, "{collapses} collapses over {seeds} seeds");
    assert!(
        (mean - exact).abs() < 4.0 * error,
        "mean {mean} vs exact {exact}, standard error {error}"
    );
}
