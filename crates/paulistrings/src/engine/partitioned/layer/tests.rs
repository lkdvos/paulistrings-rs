use super::*;
use crate::channel::clifford::Clifford1Q;
use crate::channel::rotation::PauliRotation;
use crate::channel::Channel;
use crate::engine::partitioned::transport::InProcessTransport;
use crate::pauli_string::PauliString;
use crate::pauli_sum::accumulator::BuildAccumulator;
use crate::pauli_sum::hash::Gf2Hash;
use crate::phase::Phase;
use crate::test_support::{
    assert_terms_close, differential_channels_w1, differential_channels_w2, naive_apply_layer,
    rand_sum, rand_sum_real,
};
use crate::truncation::builtin::CoefficientThreshold;
use num_complex::Complex64;

const TOL: f64 = 1e-11;
const ZERO: Complex64 = Complex64::new(0.0, 0.0);

struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// A [`ChunkWait`] that records what it was asked for.
struct WaitSpy(std::sync::Mutex<Vec<usize>>);

impl super::ChunkWait for WaitSpy {
    fn wait_chunk(&self, k: usize) {
        self.0.lock().expect("spy").push(k);
    }
}

/// `count` and `append_into` read the block at the destination position, and each waits for exactly the chunk that position falls in.
#[test]
fn received_rows_are_read_by_position_and_wait_for_their_own_chunk() {
    const BITS: u8 = 5;
    let n = 1u32 << BITS;
    for deltas in [vec![0u32], vec![0, 4], vec![0, 1, 2, 3]] {
        let span = crate::engine::coset::Gf2Span::new(&deltas, BITS);
        for chunks in [1usize, 2, 4, 8, 64] {
            let mut map = ChunkMap::default();
            map.rebuild(&span, n as usize, chunks);

            // One row per position, coefficient = the position index.
            let counts: Vec<u32> = vec![1; n as usize];
            let mut block = ExchangeBlock::<1>::with_counts(1, &counts);
            block.x = (0..n).map(|p| [u64::from(p)]).collect();
            block.z = (0..n).map(|_| [0u64]).collect();
            block.coeff = (0..n).map(|p| Complex64::new(f64::from(p), 0.0)).collect();

            let payload = PartnerPayload::<1> {
                blocks: vec![block],
            };
            let recv = vec![None, Some(payload)];
            let mut plan_deltas = deltas.clone();
            plan_deltas.sort_unstable();
            let plan = PartitionPlan {
                local_entries: vec![true, false],
                local_bucket_deltas: plan_deltas,
                remote: vec![super::super::plan::RemoteDelta {
                    entry: 1,
                    partner: 1,
                    bucket_delta: 0,
                    partition_delta: 1,
                }],
                rest_streams_total: 1,
            };

            let spy = WaitSpy(std::sync::Mutex::new(Vec::new()));
            let rows = RecvRows::new(&plan, &recv, &map, &spy);
            for beta in 0..n {
                let p = map.position_of(beta);
                assert_eq!(rows.count(beta), 1, "one row per bucket");
                let (mut x, mut z, mut c) = (Vec::new(), Vec::new(), Vec::new());
                rows.append_into(beta, &mut x, &mut z, &mut c);
                assert_eq!(x, vec![[u64::from(p)]], "bucket {beta} read position {p}");
                assert_eq!(z, vec![[0u64]]);
                assert_eq!(c, vec![Complex64::new(f64::from(p), 0.0)]);
            }
            let waited = spy.0.into_inner().expect("spy");
            assert_eq!(waited.len(), n as usize, "one wait per append_into");
            for (beta, &k) in waited.iter().enumerate() {
                assert_eq!(
                    k,
                    map.chunk_of_position(map.position_of(beta as u32)),
                    "bucket {beta} waited for the wrong chunk at {chunks} chunks",
                );
                assert!(k < map.chunks(), "chunk {k} is outside the map");
            }
        }
    }
}

/// Run one layer on every partition of `rows` on its own thread (inside a `threads`-wide pool if given), returning the checked parts and counts in rank order.
fn run_parts<const W: usize, T>(
    whole: &PauliSum<W>,
    prepared: &Prepared<W>,
    rows: &PartitionRows<W>,
    policy: &T,
    threads: Option<usize>,
) -> (Vec<PauliSum<W>>, Vec<LayerExchangeCounts>)
where
    T: TruncationPolicy<W> + ?Sized,
{
    let size = rows.num_partitions() as u32;
    let transports = InProcessTransport::group(size);
    let parts: Vec<PauliSum<W>> = (0..size).map(|r| whole.filter_partition(rows, r)).collect();
    let results: Vec<(PauliSum<W>, LayerExchangeCounts)> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .into_iter()
            .zip(transports)
            .map(|(mut local, transport)| {
                scope.spawn(move || {
                    let mut state = PartitionState::<W>::default();
                    let mut run = || {
                        apply_layer_partitioned(
                            &mut local, prepared, rows, policy, &mut state, &transport,
                        )
                    };
                    let counts = match threads {
                        Some(n) => rayon::ThreadPoolBuilder::new()
                            .num_threads(n)
                            .build()
                            .expect("pool")
                            .install(run),
                        None => run(),
                    };
                    (local, counts)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a partition panicked"))
            .collect()
    });

    let mut sums = Vec::with_capacity(results.len());
    let mut counts = Vec::with_capacity(results.len());
    for (rank, (part, count)) in results.into_iter().enumerate() {
        part.assert_invariants();
        let held = part.partition_rank_of_all(rows);
        assert!(
            held.is_none() || held == Some(rank as u32),
            "part {rank} holds keys of partition {held:?}",
        );
        sums.push(part);
        counts.push(count);
    }
    (sums, counts)
}

/// The unpartitioned layer on the whole sum, for the same hash.
fn unsplit_layer<const W: usize, T>(
    whole: &PauliSum<W>,
    prepared: &Prepared<W>,
    policy: &T,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let mut sum = whole.clone();
    let mut scratch = LayerScratch::<W>::new();
    apply_layer_bucketed(&mut sum, prepared, policy, &mut scratch);
    sum
}

/// `Z₀X₂X₄X₆` — weight 4, so `prepare` gives the `Prepared::Rotation` arm.
fn wide_gen() -> PauliString<1> {
    let mut g = PauliString::<1>::z(0);
    for q in [2u32, 4, 6] {
        g.mul_assign(&PauliString::<1>::x(q));
    }
    g
}

/// A partitioning that sees exactly the `x` bit of qubit 2, so [`wide_gen`]'s generator pass is remote.
fn rows_seeing_qubit_2_x() -> PartitionRows<1> {
    PartitionRows::<1>::from_rows(8, vec![[1u64 << 2]], vec![[0u64]])
}

/// Merged output against the naive oracle and the unpartitioned engine, over both prepared arms, both directions, several bucket counts, `P` and row draws.
#[test]
fn partitioned_layer_matches_naive_oracle_w1() {
    let input = rand_sum::<1>(600, 8, 0x9D0);
    for (name, channel) in &differential_channels_w1() {
        let channel_ref: &dyn Channel<1> = channel.as_ref();
        for &adjoint in &[false, true] {
            let want = naive_apply_layer(&input, channel_ref, &AlwaysKeep, adjoint);
            for &bits in &[0u8, 1, 3, 6] {
                let hash = Gf2Hash::<1>::new(8, bits, 0xABCD);
                let whole = input.clone().with_hash(hash);
                let prepared = channel_ref.prepare(whole.hash(), adjoint).expect("prepare");
                let unsplit = unsplit_layer(&whole, &prepared, &AlwaysKeep);
                for &pbits in &[0u8, 1, 2] {
                    for &pseed in &[0x1357u64, 0x2468] {
                        let rows = PartitionRows::<1>::from_seed(8, pbits, pseed);
                        let (parts, _) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, None);
                        let got = PauliSum::merge_partitions(parts);
                        let what = format!(
                            "{name} adjoint={adjoint} bits={bits} P={} pseed={pseed:x}",
                            rows.num_partitions(),
                        );
                        assert_terms_close(&got, &want, TOL, &what);
                        assert_terms_close(&got, &unsplit, TOL, &format!("{what} vs unsplit"));
                    }
                }
            }
        }
    }
}

/// The same at `W = 2`: wide keys, supports straddling the word boundary.
#[test]
fn partitioned_layer_matches_naive_oracle_w2() {
    let input = rand_sum_real::<2>(700, 128, 0x9D1);
    for (name, channel) in &differential_channels_w2() {
        let channel_ref: &dyn Channel<2> = channel.as_ref();
        for &adjoint in &[false, true] {
            let want = naive_apply_layer(&input, channel_ref, &AlwaysKeep, adjoint);
            for &bits in &[2u8, 5] {
                let hash = Gf2Hash::<2>::new(128, bits, 0xABCD);
                let whole = input.clone().with_hash(hash);
                let prepared = channel_ref.prepare(whole.hash(), adjoint).expect("prepare");
                let unsplit = unsplit_layer(&whole, &prepared, &AlwaysKeep);
                for &pbits in &[0u8, 1, 2] {
                    for &pseed in &[0x1357u64, 0x2468] {
                        let rows = PartitionRows::<2>::from_seed(128, pbits, pseed);
                        let (parts, _) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, None);
                        let got = PauliSum::merge_partitions(parts);
                        let what = format!(
                            "{name} adjoint={adjoint} bits={bits} P={} pseed={pseed:x}",
                            rows.num_partitions(),
                        );
                        assert_terms_close(&got, &want, TOL, &what);
                        assert_terms_close(&got, &unsplit, TOL, &format!("{what} vs unsplit"));
                    }
                }
            }
        }
    }
}

/// A layer whose every delta stays inside its partition issues no transport call, not an empty one.
#[test]
fn no_remote_deltas_means_no_exchange() {
    let input = rand_sum::<1>(300, 8, 0x9D2);
    // `h(3)`'s only non-identity delta is the mask `x₃ z₃`; a partition row reading qubit 0 alone cannot see it.
    let rows = PartitionRows::<1>::from_rows(8, vec![[1u64]], vec![[0u64]]);
    let h = Clifford1Q::h(3);
    let hash = Gf2Hash::<1>::new(8, 4, 0x9D);
    let whole = input.clone().with_hash(hash);
    let prepared = Channel::<1>::prepare(&h, whole.hash(), false).expect("prepare");
    assert!(
        !PartitionPlan::new(&prepared, &rows, 0).has_remote(),
        "fixture must have no remote delta",
    );

    let transports = crate::test_support::LoggingTransport::group(2);
    let parts: Vec<PauliSum<1>> = (0..2).map(|r| whole.filter_partition(&rows, r)).collect();
    let prepared = &prepared;
    let rows = &rows;
    let results: Vec<(PauliSum<1>, LayerExchangeCounts, usize)> = std::thread::scope(|scope| {
        let handles: Vec<_> = parts
            .into_iter()
            .zip(transports)
            .map(|(mut local, transport)| {
                scope.spawn(move || {
                    let mut state = PartitionState::<1>::default();
                    let counts = apply_layer_partitioned(
                        &mut local,
                        prepared,
                        rows,
                        &AlwaysKeep,
                        &mut state,
                        &transport,
                    );
                    let seen = transport.log.count("exchange_layer");
                    (local, counts, seen)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut got_parts = Vec::new();
    for (part, counts, exchanges) in results {
        assert_eq!(exchanges, 0, "a local-only layer called the transport");
        assert_eq!(counts, LayerExchangeCounts::none(2));
        got_parts.push(part);
    }
    let got = PauliSum::merge_partitions(got_parts);
    let want = naive_apply_layer(&input, &h, &AlwaysKeep, false);
    assert_terms_close(&got, &want, TOL, "local-only layer");
}

/// A wide rotation whose generator crosses: every generator row is exported, and the exported total is exactly the number of terms that anticommute with the generator.
#[test]
fn all_remote_rotation_generator() {
    let gen = wide_gen();
    let rot = PauliRotation::new(gen, 0.41);
    let rows = rows_seeing_qubit_2_x();
    assert_eq!(rows.partition_of(&gen.x, &gen.z), 1);
    let input = rand_sum::<1>(500, 8, 0x9D3);
    let hash = Gf2Hash::<1>::new(8, 3, 0x9E);
    let whole = input.clone().with_hash(hash);
    let prepared = Channel::<1>::prepare(&rot, whole.hash(), false).expect("prepare");
    let plan = PartitionPlan::new(&prepared, &rows, 0);
    assert!(plan.has_remote(), "the generator must be remote");
    assert!(!plan.local_entries[1], "entry 1 is the generator pass");
    assert_eq!(plan.local_bucket_deltas, vec![0], "only the identity stays");

    let anticommuting = whole
        .iter()
        .filter(|(x, z, _)| !PauliString::<1> { x: **x, z: **z }.commutes_with(&gen))
        .count() as u64;
    assert!(anticommuting > 0, "fixture must have anticommuting terms");

    let (parts, counts) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, None);
    let sent: u64 = counts.iter().map(|c| c.rows_sent.iter().sum::<u64>()).sum();
    let received: u64 = counts.iter().map(|c| c.rows_received).sum();
    assert_eq!(
        sent, anticommuting,
        "one exported row per anticommuting term"
    );
    assert_eq!(received, anticommuting, "everything sent is received");
    for c in &counts {
        assert_eq!(c.remote_deltas, 1);
    }

    let got = PauliSum::merge_partitions(parts);
    let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
    assert_terms_close(&got, &want, TOL, "all-remote generator");
}

/// A key `w` and its generator image `u` on two partitions, with the oracle's amplitudes `alpha` (`w → w`) and `beta` (`u → w`).
fn cancelling_pair(
    rot: &PauliRotation<1>,
    rows: &PartitionRows<1>,
) -> (PauliString<1>, PauliString<1>, Complex64, Complex64) {
    let gen = wide_gen();
    let w = PauliString::<1>::x(0);
    assert!(!w.commutes_with(&gen));
    let mut u = w;
    u.mul_assign(&gen);
    assert_ne!(
        rows.partition_of(&w.x, &w.z),
        rows.partition_of(&u.x, &u.z),
        "the pair must straddle the partition boundary",
    );

    let probe = |term: PauliString<1>| -> Complex64 {
        let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 1);
        accumulator.add_term(term, Phase::ONE, Complex64::new(1.0, 0.0));
        naive_apply_layer(&accumulator.finalize(), rot, &AlwaysKeep, false)
            .get(&w.x, &w.z)
            .unwrap_or(ZERO)
    };
    let alpha = probe(w);
    let beta = probe(u);
    assert!(alpha.norm() > 0.1 && beta.norm() > 0.1, "degenerate probe");
    (w, u, alpha, beta)
}

/// `w` with coefficient `beta` and `u` with `-alpha · scale`, so `w`'s output is `beta·alpha − scale · alpha·beta`.
fn cancelling_input(
    w: PauliString<1>,
    u: PauliString<1>,
    alpha: Complex64,
    beta: Complex64,
    scale: f64,
) -> PauliSum<1> {
    let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 2);
    accumulator.add_term(w, Phase::ONE, beta);
    accumulator.add_term(u, Phase::ONE, -alpha * scale);
    accumulator.finalize()
}

/// `keep_term` sees the sum across the partition boundary: two contributions far above the threshold cancel below it, and the term is dropped.
#[test]
fn keep_term_sees_the_sum_across_partitions() {
    let rot = PauliRotation::new(wide_gen(), 0.41);
    let rows = rows_seeing_qubit_2_x();
    let (w, u, alpha, beta) = cancelling_pair(&rot, &rows);
    let input = cancelling_input(w, u, alpha, beta, 1.0 - 1e-7);
    let policy = CoefficientThreshold(1e-6);

    for bits in [0u8, 3] {
        let hash = Gf2Hash::<1>::new(8, bits, 0x9F);
        let whole = input.clone().with_hash(hash);
        let prepared = Channel::<1>::prepare(&rot, whole.hash(), false).expect("prepare");
        let (parts, _) = run_parts(&whole, &prepared, &rows, &policy, None);
        let got = PauliSum::merge_partitions(parts);
        assert!(
            got.get(&w.x, &w.z).is_none(),
            "bits={bits}: a term that only survives unsummed was kept ({:?})",
            got.get(&w.x, &w.z),
        );
        let want = naive_apply_layer(&input, &rot, &policy, false);
        assert_terms_close(&got, &want, TOL, &format!("threshold bits={bits}"));

        // Without the threshold the residue is there, and it is the sum.
        let (parts, _) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, None);
        let kept = PauliSum::merge_partitions(parts);
        let residue = kept.get(&w.x, &w.z).expect("the residue survives");
        assert!(
            residue.norm() < 1e-7 && residue.norm() > 1e-9,
            "bits={bits}: residue {residue} is not the near-cancellation",
        );
        let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
        assert_terms_close(&kept, &want, TOL, &format!("no threshold bits={bits}"));
    }
}

/// Contributions from two partitions that cancel exactly leave no term.
#[test]
fn exact_zero_sum_across_partitions_is_dropped() {
    let rot = PauliRotation::new(wide_gen(), 0.41);
    let rows = rows_seeing_qubit_2_x();
    let (w, u, alpha, beta) = cancelling_pair(&rot, &rows);
    let input = cancelling_input(w, u, alpha, beta, 1.0);

    for bits in [0u8, 3] {
        let hash = Gf2Hash::<1>::new(8, bits, 0x9F);
        let whole = input.clone().with_hash(hash);
        let prepared = Channel::<1>::prepare(&rot, whole.hash(), false).expect("prepare");
        let (parts, _) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, None);
        let got = PauliSum::merge_partitions(parts);
        assert!(
            got.get(&w.x, &w.z).is_none(),
            "bits={bits}: an exactly cancelled term survived ({:?})",
            got.get(&w.x, &w.z),
        );
        let want = naive_apply_layer(&input, &rot, &AlwaysKeep, false);
        assert!(
            want.get(&w.x, &w.z).is_none(),
            "bits={bits}: the fixture does not cancel exactly in the oracle either",
        );
        assert_terms_close(&got, &want, TOL, &format!("exact zero bits={bits}"));
    }
}

/// Partitions with no input still export empty blocks, receive real ones, and produce their share of the output.
#[test]
fn empty_partition_participates() {
    let channels = differential_channels_w1();
    let (_, su4) = channels
        .iter()
        .find(|(n, _)| *n == "haar_su4")
        .expect("the dense SU(4) cell");
    // Two terms over four partitions: at least two parts are empty.
    let mut accumulator = BuildAccumulator::<1>::with_capacity(8, 2);
    accumulator.add_term(PauliString::<1>::x(1), Phase::ONE, Complex64::new(1.0, 0.0));
    accumulator.add_term(
        PauliString::<1>::z(5),
        Phase::ONE,
        Complex64::new(-0.5, 0.25),
    );
    let input = accumulator.finalize();
    let rows = PartitionRows::<1>::from_seed(8, 2, 0x9D4);
    assert_eq!(rows.num_partitions(), 4);

    let hash = Gf2Hash::<1>::new(8, 2, 0x9D5);
    let whole = input.clone().with_hash(hash);
    let prepared = su4.prepare(whole.hash(), false).expect("prepare");
    assert!(PartitionPlan::new(&prepared, &rows, 0).has_remote());

    assert!(
        (0..rows.num_partitions() as u32).any(|r| whole.filter_partition(&rows, r).is_empty()),
        "the fixture must leave a partition's input empty",
    );
    let (parts, counts) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, None);
    assert_eq!(counts.len(), 4);
    let got = PauliSum::merge_partitions(parts);
    let want = naive_apply_layer(&input, su4.as_ref(), &AlwaysKeep, false);
    assert_terms_close(&got, &want, TOL, "empty partition");
}

/// Each partition's output is byte-identical across Rayon pool sizes (ARCHITECTURE.md §Determinism).
#[test]
fn partitioned_output_is_byte_identical_across_pool_sizes() {
    let input = rand_sum::<1>(1500, 8, 0x9D6);
    let channels = differential_channels_w1();
    let rot = PauliRotation::new(wide_gen(), 0.41);
    let (_, su4) = channels
        .iter()
        .find(|(n, _)| *n == "haar_su4")
        .expect("the dense SU(4) cell");
    // The wide rotation needs the row that sees its generator, or nothing crosses.
    let cases: [(&str, &dyn Channel<1>, PartitionRows<1>); 2] = [
        (
            "su4",
            su4.as_ref(),
            PartitionRows::<1>::from_seed(8, 1, 0x9D8),
        ),
        ("rot_wide", &rot, rows_seeing_qubit_2_x()),
    ];

    for (name, channel, rows) in cases {
        // 64 buckets: comfortably above MIN_COSETS_FOR_PARALLEL.
        let hash = Gf2Hash::<1>::new(8, 6, 0x9D7);
        let whole = input.clone().with_hash(hash);
        let prepared = channel.prepare(whole.hash(), false).expect("prepare");
        assert!(
            PartitionPlan::new(&prepared, &rows, 0).has_remote(),
            "{name}"
        );

        let (one, counts_one) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, Some(1));
        let (four, counts_four) = run_parts(&whole, &prepared, &rows, &AlwaysKeep, Some(4));
        assert_eq!(counts_one, counts_four, "{name}: counts");
        for (rank, (a, b)) in one.iter().zip(&four).enumerate() {
            assert_eq!(
                a.to_arrays(),
                b.to_arrays(),
                "{name}: partition {rank} is not byte-identical across pool sizes",
            );
        }
    }
}

/// [`apply_layer_partitioned_with_plan`], classifying `prepared`'s deltas itself.
fn apply_layer_partitioned<const W: usize, T, X>(
    local: &mut PauliSum<W>,
    prepared: &Prepared<W>,
    rows: &PartitionRows<W>,
    policy: &T,
    state: &mut PartitionState<W>,
    transport: &X,
) -> LayerExchangeCounts
where
    T: TruncationPolicy<W> + ?Sized,
    X: Transport,
{
    let plan = PartitionPlan::new(prepared, rows, transport.rank());
    apply_layer_partitioned_with_plan(local, prepared, &plan, rows, policy, state, transport)
}
