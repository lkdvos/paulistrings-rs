//! `DistributedSum` against the unpartitioned `propagate`.
//!
//! The distributed driver is the partitioned engine with one partition per
//! *process*. Its contract is the same as the in-process one's — the gathered
//! output agrees with `propagate` to floating-point tolerance — but the code
//! path differs: the layer loop runs on one thread
//! per rank rather than fanned out from a single driver, the scatter is
//! replicated rather than split from one sum, and the gather goes through the
//! transport's byte framing instead of a local merge.
//!
//! All of that is transport-independent, so this file drives it with
//! `InProcessTransport`: one thread per "rank", no MPI, no launcher. The MPI
//! transport's own net is `tests/mpi_ranks.rs`, run under `mpirun`.

use num_complex::Complex64;
use paulistrings::test_support::InProcessTransport;
use paulistrings::test_support::{
    assert_runs_agree, assert_runs_agree_count_only, assert_terms_close, collapsing_circuit,
    haar_su4_matrix, heavy_cut, kicked_ising, propagate_discarding, rand_sum, rand_sum_real,
    single_cuts, site_and_magnetization, spread_magnitudes, square_lattice, trotter_circuit,
    truncated_liouvillian, unpinned_partitions, unpinned_scatter, z0_sum, zz_rotation, KeepAll,
};
use paulistrings::Collectives;
use paulistrings::PtmChannel;
use paulistrings::{
    propagate, BuildAccumulator, BuiltinTruncation, Circuit, Direction, PauliString, PauliSum,
    TruncationPolicy,
};
use paulistrings::{
    repartition_share, DistributedSum, Gf2Hash, PartitionConfig, PartitionRowPolicy, PartitionRows,
    PartitionRuntime, PartitionedSum, ScatterOptions, ScatterRows,
};
use paulistrings::{And, ApproxTopN, CoefficientThreshold, CollapseSample, WeightCutoff};

const TOL: f64 = 1e-11;
/// The Trotter angle every `trotter_circuit` fixture here rotates by. Long
/// enough a circuit that the bucket count grows mid-run, so the per-layer bits
/// all-reduce actually changes value.
const THETA: f64 = 0.01;

/// Partition-row seed of every seeded scatter here.
const ROW_SEED: u64 = 0x5EED_C0FF_EE00_4321;

/// One unpinned partition of two threads: what a rank's runtime looks like when
/// the launcher, not the engine, did the placement.
fn config() -> PartitionConfig {
    unpinned_partitions(1, 2)
}

/// This rank's share of `sum`, split by the rows `policy` draws, on a runtime built from [`config`].
fn scatter_by_policy<const W: usize>(
    sum: PauliSum<W>,
    transport: InProcessTransport,
    policy: PartitionRowPolicy,
) -> DistributedSum<W, InProcessTransport> {
    let options = ScatterOptions {
        runtime: PartitionRuntime::new(&config()).expect("topology resolves"),
        rows: ScatterRows::Policy(policy),
    };
    DistributedSum::scatter_with(sum, transport, options)
}

/// A short circuit mixing the layer shapes that matter: a local-ish Clifford, a
/// dense Haar SU(4) (all 15 deltas, worst fan-out), a rotation, and a noise
/// channel (which is key-preserving, so it never exchanges).
fn mixed_circuit<const W: usize>(num_qubits: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    let n = num_qubits;
    for q in 0..n {
        circuit.push(PtmChannel::cnot(q, (q + 1) % n));
    }
    circuit.push(PtmChannel::hadamard(0));
    circuit.push(PtmChannel::unitary_2q(0, 1, haar_su4_matrix()));
    circuit.push(zz_rotation::<W>(0, n / 2, 0.31));
    for q in 0..n {
        circuit.push(PtmChannel::depolarizing(q, 0.04));
    }
    circuit.push(PtmChannel::unitary_2q(1, 2, haar_su4_matrix()));
    circuit
}

/// `f` on every rank of a `size`-rank in-process group, one thread each, the results in rank order.
fn on_ranks<T: Send>(size: u32, f: impl Fn(InProcessTransport) -> T + Sync) -> Vec<T> {
    std::thread::scope(|scope| {
        let f = &f;
        let handles: Vec<_> = InProcessTransport::group(size)
            .into_iter()
            .map(|transport| scope.spawn(move || f(transport)))
            .collect();
        handles.into_iter().map(|h| h.join().expect("rank thread panicked")).collect()
    })
}

/// Run `circuit` on `size` in-process "ranks", each with its own replicated
/// copy of `sum`, and return rank 0's gathered output.
///
/// This is the shape a launcher produces, minus the processes: every rank
/// builds the identical input, scatters, propagates and gathers, and exactly
/// one of them comes back with `Some`.
fn distributed<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    size: u32,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + Sync,
{
    let gathered: Vec<Option<PauliSum<W>>> = on_ranks(size, |transport| {
        let mut split =
            DistributedSum::scatter_with(sum.clone(), transport, unpinned_scatter(1, 2, ROW_SEED));
        split.assert_invariants();
        split.propagate(circuit, policy, direction);
        split.assert_invariants();
        let out = split.gather();
        // `local` is always this rank's share, whether or not the
        // rank gathers.
        assert!(split.local().len() <= split.len());
        out
    });

    assert!(gathered[0].is_some(), "rank 0 gathers");
    for (rank, out) in gathered.iter().enumerate().skip(1) {
        assert!(out.is_none(), "rank {rank} must not gather");
    }
    gathered.into_iter().next().unwrap().unwrap()
}

/// The unpartitioned engine is the oracle for every group size and direction.
fn check<const W: usize, T>(circuit: &Circuit<W>, sum: &PauliSum<W>, policy: &T, name: &str)
where
    T: TruncationPolicy<W> + Sync + Clone + Into<BuiltinTruncation>,
{
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let (want, discarded) = propagate_discarding(circuit, sum.clone(), policy, direction);
        for size in [1u32, 2, 4] {
            let got = distributed(circuit, sum, policy, direction, size);
            let what = format!("{name} ranks={size} {direction:?}");
            assert_runs_agree(policy, discarded, &got, &want, TOL, &what);
        }
    }
}

/// One cut that splits a boundary group, dropping little weight.
#[test]
fn single_cuts_match_propagate() {
    for (name, circuit, sum, n) in single_cuts::<1>(16, 0x51C3) {
        check(&circuit, &sum, &ApproxTopN(n), name);
    }
    for (name, circuit, sum, n) in single_cuts::<2>(70, 0x51C4) {
        check(&circuit, &sum, &ApproxTopN(n), name);
    }
}

#[test]
fn trotter_matches_propagate_w1() {
    let circuit = trotter_circuit::<1>(24, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<1>(1_200, 24, 0x0D15), 40, 0x0D15);
    check(&circuit, &sum, &ApproxTopN(2_000), "trotter approx w1");
    check(&circuit, &sum, &And(CoefficientThreshold(1e-9), ApproxTopN(2_000)), "trotter and w1");
}

#[test]
fn trotter_matches_propagate_w2() {
    let circuit = trotter_circuit::<2>(24, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<2>(900, 24, 0x0D16), 40, 0x0D16);
    check(&circuit, &sum, &ApproxTopN(1_500), "trotter w2");
}

/// Every layer of [`heavy_cut`] cuts through ties or an octave, too heavily for the operator check (`assert_runs_agree_count_only` asserts so): the counts must still match, `ApproxTopN` on 2 and 4 ranks and exact `TopN` on one.
#[test]
fn heavy_cuts_keep_the_reference_count() {
    use paulistrings::TopN;
    let (circuit, sum) = heavy_cut::<1>(12, 1, 800, 0x4EA2);
    for direction in [Direction::Forward, Direction::Heisenberg] {
        let policies: [(BuiltinTruncation, &[u32]); 2] =
            [(ApproxTopN(400).into(), &[2, 4]), (TopN(400).into(), &[1])];
        for (policy, sizes) in policies {
            let (want, discarded) = propagate_discarding(&circuit, sum.clone(), &policy, direction);
            for &size in sizes {
                let got = distributed(&circuit, &sum, &policy, direction, size);
                let what = format!("heavy {policy:?} ranks={size} {direction:?}");
                assert_runs_agree_count_only(policy.clone(), discarded, &got, &want, &what);
            }
        }
    }
}

/// Exact `TopN` runs on one rank and panics on every rank of two before the first layer.
#[test]
fn exact_top_n_runs_on_one_rank_only() {
    use paulistrings::TopN;
    let circuit = trotter_circuit::<1>(24, THETA);
    let sum = rand_sum_real::<1>(1_200, 24, 0x70B3);
    let want = propagate(&circuit, sum.clone(), &TopN(800), Direction::Forward);
    let got = distributed(&circuit, &sum, &TopN(800), Direction::Forward, 1);
    assert_terms_close(&got, &want, TOL, "TopN on one rank");

    let messages = on_ranks(2, |transport| {
        let mut split =
            DistributedSum::scatter_with(sum.clone(), transport, unpinned_scatter(1, 2, ROW_SEED));
        let terms = split.local().len();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            split.propagate(&circuit, &TopN(800), Direction::Forward)
        }))
        .expect_err("TopN on two ranks must panic");
        assert_eq!(split.local().len(), terms, "no layer may run");
        panic.downcast_ref::<String>().cloned().unwrap_or_default()
    });
    for message in messages {
        assert!(message.contains("not yet supported"), "{message}");
    }
}

/// `BuiltinTruncation`'s collective layer pass, per rank.
#[test]
fn builtin_truncation_tree_matches_propagate() {
    use paulistrings::BuiltinTruncation as T;
    let circuit = trotter_circuit::<1>(24, THETA);
    let sum = spread_magnitudes(&rand_sum_real::<1>(1_200, 24, 0x0D17), 40, 0x0D17);
    let tree = T::And(Box::new(T::Coefficient(1e-9)), Box::new(T::ApproxTopN(2_000)));
    check(&circuit, &sum, &tree, "trotter tree w1");
}

/// Dense two-qubit blocks and a noise channel, at both widths: the layer
/// shapes with the widest fan-out, and the one that must issue no exchange at
/// all.
#[test]
fn mixed_channels_match_propagate() {
    let circuit = mixed_circuit::<1>(6);
    let sum = spread_magnitudes(&rand_sum::<1>(200, 6, 0x0D17), 40, 0x0D17);
    check(&circuit, &sum, &KeepAll, "mixed keep w1");
    check(&circuit, &sum, &CoefficientThreshold(1e-9), "mixed eps w1");
    check(&circuit, &sum, &WeightCutoff(4), "mixed weight w1");
    check(&circuit, &sum, &ApproxTopN(1_000), "mixed approx w1");

    let circuit = mixed_circuit::<2>(6);
    let sum = spread_magnitudes(&rand_sum::<2>(200, 6, 0x0D18), 40, 0x0D18);
    check(&circuit, &sum, &KeepAll, "mixed keep w2");
    check(&circuit, &sum, &ApproxTopN(1_000), "mixed approx w2");
}

/// A sum small enough that some ranks hold nothing: the empty-partition path
/// through scatter, the layer, the gather framing and the merge.
#[test]
fn a_two_term_sum_survives_four_ranks() {
    let mut circuit = Circuit::<1>::new(4);
    circuit.push(PtmChannel::hadamard(0));
    circuit.push(PtmChannel::cnot(0, 1));

    let sum = rand_sum::<1>(2, 4, 0x0D19);
    check(&circuit, &sum, &KeepAll, "two terms");
}

/// An empty circuit still has to run its one consistency collective and gather
/// the scattered input back unchanged.
#[test]
fn an_empty_circuit_round_trips_the_input() {
    let circuit = Circuit::<1>::new(6);
    let sum = rand_sum::<1>(150, 6, 0x0D1A);
    let got = distributed(&circuit, &sum, &KeepAll, Direction::Forward, 4);
    assert_terms_close(&got, &sum, TOL, "empty circuit");
    assert_eq!(got.len(), sum.len());
}

/// `gather` leaves the sum intact, so a driver can checkpoint and keep
/// stepping. Two gathers around one more propagation.
#[test]
fn gather_is_repeatable_and_non_destructive() {
    let circuit = mixed_circuit::<1>(6);
    let sum = rand_sum::<1>(200, 6, 0x0D1B);

    let want_once = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    let want_twice = propagate(&circuit, want_once.clone(), &KeepAll, Direction::Forward);

    let (first, second) = on_ranks(2, |transport| {
        let mut split =
            DistributedSum::scatter_with(sum.clone(), transport, unpinned_scatter(1, 2, ROW_SEED));
        split.propagate(&circuit, &KeepAll, Direction::Forward);
        let first = split.gather();
        split.propagate(&circuit, &KeepAll, Direction::Forward);
        (first, split.gather())
    })
    .remove(0);

    assert_terms_close(&first.expect("rank 0"), &want_once, TOL, "first gather");
    assert_terms_close(&second.expect("rank 0"), &want_twice, TOL, "second gather");
}

/// The per-rank trace: one layer record per layer, `terms_in`/`terms_out` a
/// single (local) entry, `rows_sent` indexed by destination rank.
#[test]
fn the_trace_is_this_ranks_view_of_every_layer() {
    let circuit = mixed_circuit::<1>(6);
    let sum = rand_sum::<1>(200, 6, 0x0D1C);
    let layers = circuit.len();

    let traces = on_ranks(2, |transport| {
        let mut split =
            DistributedSum::scatter_with(sum.clone(), transport, unpinned_scatter(1, 2, ROW_SEED));
        split.enable_trace();
        split.propagate(&circuit, &KeepAll, Direction::Forward);
        (split.rank(), split.take_trace().expect("tracing is on"))
    });

    let mut any_exchange = false;
    for (rank, trace) in &traces {
        assert_eq!(trace.layers.len(), layers, "rank {rank}: layer count");
        for (k, record) in trace.layers.iter().enumerate() {
            assert_eq!(record.terms_in.len(), 1, "rank {rank} layer {k}: per-rank");
            assert_eq!(record.terms_out.len(), 1);
            assert_eq!(record.rows_received.len(), 1);
            // One row of the [from][to] matrix — this rank's — over the group.
            assert_eq!(record.rows_sent.len(), 1);
            assert_eq!(record.rows_sent[0].len(), 2, "indexed by destination rank");
            any_exchange |= record.remote_deltas > 0;
        }
    }
    assert!(any_exchange, "the mixed circuit must cross a boundary");

    // The depolarizing layers are key-preserving, so no rank exchanges there.
    for (rank, trace) in &traces {
        assert!(trace.local_layers() > 0, "rank {rank}: the noise layers exchange nothing",);
    }
}

/// Ranks driven through different circuits are named, not left to deadlock.
#[test]
#[should_panic(expected = "disagree about the run")]
fn ranks_driven_through_different_circuits_are_caught() {
    let sum = rand_sum::<1>(50, 4, 0x0D1D);
    let short = Circuit::<1>::new(4);
    let mut long = Circuit::<1>::new(4);
    long.push(PtmChannel::hadamard(0));

    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    std::thread::scope(|scope| {
        let sum = &sum;
        let long = &long;
        scope.spawn(move || {
            // Rank 1 sees the mismatch too; swallow it so only rank 0's panic
            // reaches the harness.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut split = DistributedSum::scatter_with(
                    sum.clone(),
                    one,
                    unpinned_scatter(1, 2, ROW_SEED),
                );
                split.propagate(long, &KeepAll, Direction::Forward);
            }));
        });
        let mut split =
            DistributedSum::scatter_with(sum.clone(), zero, unpinned_scatter(1, 2, ROW_SEED));
        split.propagate(&short, &KeepAll, Direction::Forward);
    });
}

/// Ranks splitting by different explicit partition rows are named too: the run fingerprint folds the rows in.
#[test]
#[should_panic(expected = "disagree about the run")]
fn ranks_splitting_by_different_partition_rows_are_caught() {
    let sum = rand_sum::<1>(50, 4, 0x0D1D);
    let circuit = Circuit::<1>::new(4);
    let scatter = |rows: &[&[usize]]| ScatterOptions {
        runtime: PartitionRuntime::new(&unpinned_partitions(1, 2)).expect("unpinned"),
        rows: ScatterRows::Policy(PartitionRowPolicy::Matrix(
            rows.iter().map(|row| (0..8).map(|c| row.contains(&c)).collect()).collect(),
        )),
    };

    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");

    std::thread::scope(|scope| {
        let sum = &sum;
        let circuit = &circuit;
        scope.spawn(move || {
            // Rank 1 sees the mismatch too; swallow it so only rank 0's panic reaches the harness.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut split = DistributedSum::scatter_with(sum.clone(), one, scatter(&[&[0, 1]]));
                split.propagate(circuit, &KeepAll, Direction::Forward);
            }));
        });
        let mut split = DistributedSum::scatter_with(sum.clone(), zero, scatter(&[&[4, 5]]));
        split.propagate(circuit, &KeepAll, Direction::Forward);
    });
}

/// A group size that is not a power of two cannot name a partition.
#[test]
#[should_panic(expected = "power of two")]
fn a_group_that_is_not_a_power_of_two_is_rejected() {
    let sum = rand_sum::<1>(10, 4, 0x0D1E);
    let mut group = InProcessTransport::group(3);
    let zero = group.remove(0);
    let _ = DistributedSum::scatter(sum, zero, &config());
}

/// The runtime is one partition per process; a multi-partition one is the
/// unimplemented hybrid.
#[test]
#[should_panic(expected = "one partition per process")]
fn a_multi_partition_runtime_is_rejected() {
    let sum = rand_sum::<1>(10, 4, 0x0D1F);
    let mut group = InProcessTransport::group(1);
    let zero = group.remove(0);
    let _ = DistributedSum::scatter(sum, zero, &unpinned_partitions(2, 1));
}

/// Every rank's `len()` is the whole sum's, and the local shares add up to it.
#[test]
fn len_is_collective_and_the_shares_add_up() {
    let sum = rand_sum::<1>(500, 6, 0x0D20);
    let want = sum.len();

    let seen = on_ranks(4, |transport| {
        let split =
            DistributedSum::scatter_with(sum.clone(), transport, unpinned_scatter(1, 2, ROW_SEED));
        (split.len(), split.local().len())
    });

    for (rank, &(total, _)) in seen.iter().enumerate() {
        assert_eq!(total, want, "rank {rank}: collective len");
    }
    assert_eq!(seen.iter().map(|&(_, local)| local).sum::<usize>(), want);
    // A 500-term sum over four ranks: nobody holds all of it.
    assert!(seen.iter().all(|&(_, local)| local < want));
}

/// One `Z` per qubit, so a cut makes every term's rank hand-computable: a
/// single-`Z` key has odd z-weight in exactly the block holding that qubit.
fn single_z_sum<const W: usize>(num_qubits: usize) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::new(num_qubits);
    for q in 0..num_qubits {
        acc.add_term(PauliString::<W>::z(q), Complex64::new(1.0 + q as f64, 0.0));
    }
    acc.finalize()
}

/// The qubits this rank's share names, one per single-`Z` term, ascending.
fn local_z_qubits<const W: usize, X: paulistrings::Transport>(
    split: &DistributedSum<W, X>,
) -> Vec<usize> {
    let (_, z, _) = split.local().to_arrays();
    let mut qubits: Vec<usize> = z.iter().map(|row| row[0].trailing_zeros() as usize).collect();
    qubits.sort_unstable();
    qubits
}

/// An explicit cut is the whole point of the policy: the caller, not a random
/// draw, decides which rank a block of qubits belongs to.
#[test]
fn a_cut_row_policy_lands_each_block_on_its_own_rank() {
    const NQ: usize = 8;
    let sum = single_z_sum::<1>(NQ);
    let policy = PartitionRowPolicy::Cut(vec![(0..4).collect(), (4..8).collect()]);

    let held: Vec<Vec<usize>> = on_ranks(2, |transport| {
        let split = scatter_by_policy(sum.clone(), transport, policy.clone());
        split.assert_invariants();
        local_z_qubits(&split)
    });

    assert_eq!(held[0], vec![0, 1, 2, 3], "block 0 is rank 0's");
    assert_eq!(held[1], vec![4, 5, 6, 7], "block 1 is rank 1's");
}

/// `DistributedSum::scatter` splits by the default rows: the seeded draw at the sum's own hash seed.
#[test]
fn scatter_splits_by_the_sums_own_seed() {
    let sum = rand_sum::<1>(300, 8, 0x0D22);
    let seed = sum.hash().seed();

    let seen: Vec<(usize, usize)> = std::thread::scope(|scope| {
        let sum = &sum;
        let handles: Vec<_> = InProcessTransport::group(2)
            .into_iter()
            .zip(InProcessTransport::group(2))
            .map(|(a, b)| {
                scope.spawn(move || {
                    let want = DistributedSum::scatter(sum.clone(), a, &config())
                        .expect("topology resolves");
                    let got =
                        scatter_by_policy(sum.clone(), b, PartitionRowPolicy::Seeded(Some(seed)));
                    assert_eq!(want.rows(), got.rows());
                    (want.local().len(), got.local().len())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("rank thread panicked")).collect()
    });

    for (rank, &(want, got)) in seen.iter().enumerate() {
        assert_eq!(want, got, "rank {rank}: terms held");
    }
    assert_eq!(seen.iter().map(|&(_, got)| got).sum::<usize>(), sum.len());
}

/// `local().expectation_product` is per rank, and the ranks' answers sum to
/// the whole sum's.
#[test]
fn local_expectations_sum_to_the_whole_sums() {
    use paulistrings::{PauliAxis, ProductBasis};

    let sum = rand_sum_real::<1>(300, 6, 0x0D21);
    let state = PauliAxis::Z;
    let want = sum.expectation_product(&ProductBasis::uniform(state));

    let parts: Vec<Complex64> = on_ranks(4, |transport| {
        DistributedSum::scatter_with(sum.clone(), transport, unpinned_scatter(1, 2, ROW_SEED))
            .local()
            .expectation_product(&ProductBasis::uniform(state))
    });

    let got: Complex64 = parts.iter().sum();
    assert!((got - want).norm() < TOL, "{got} vs {want}");
}

// ---- CollapseSample ---------------------------------------------------------

/// One rank is the unpartitioned trajectory; more ranks collapse the same way in kind: a sum bounded by the cache and a nonzero collapse count on the shared policy.
#[test]
fn collapse_sample_trajectories_over_ranks() {
    const CACHE: usize = 6;
    let (circuit, input) = (collapsing_circuit(), z0_sum());
    for seed in 0..5u64 {
        let want = propagate(
            &circuit,
            input.clone(),
            &CollapseSample::new(CACHE, 4, seed),
            Direction::Heisenberg,
        );
        let one = distributed(
            &circuit,
            &input,
            &CollapseSample::new(CACHE, 4, seed),
            Direction::Heisenberg,
            1,
        );
        assert_terms_close(&one, &want, 1e-12, &format!("seed {seed} one rank"));

        for size in [2u32, 4] {
            let policy = CollapseSample::new(CACHE, 4, seed);
            let got = distributed(&circuit, &input, &policy, Direction::Heisenberg, size);
            assert!(policy.collapses() >= 1, "seed {seed} ranks={size}");
            assert!(got.len() <= CACHE, "seed {seed} ranks={size}: {} terms", got.len());
        }
    }
}

/// Over `InProcessTransport` ranks one collapse draws exactly `samples` points across the group: every term holds `⌊samples·|c|/Σ|c|⌋` or `⌈…⌉` of them, each worth `Σ|c|/samples`.
#[test]
fn collapse_sample_distributed_draws_every_point_once() {
    use paulistrings::test_support::{
        assert_draws_tile_l1, four_term_keys, weighted_four_term_sum,
    };
    use paulistrings::PartitionRuntime;
    let keys = four_term_keys::<1>();
    let input = weighted_four_term_sum::<1>(8);
    let l1: f64 = input.iter().map(|(_, _, c)| c.norm()).sum();
    let mut circuit = Circuit::<1>::new(8);
    circuit.push(PtmChannel::depolarizing(0, 0.0));

    let size = 2u32;
    let rows =
        paulistrings::PartitionRows::<1>::from_seed(8, size.trailing_zeros() as u8, ROW_SEED);
    let owners: std::collections::HashSet<u32> =
        keys.iter().map(|p| rows.partition_of(&p.x, &p.z)).collect();
    assert!(owners.len() > 1, "ranks={size}: the fixture must span ranks");

    let runtimes: Vec<_> =
        (0..size).map(|_| PartitionRuntime::new(&unpinned_partitions(1, 1)).unwrap()).collect();
    for samples in [1usize, 5] {
        let step = l1 / samples as f64;
        for seed in 0..40u64 {
            let policy = CollapseSample::new(3, samples, seed);
            let gathered = on_ranks(size, |transport| {
                use paulistrings::Collectives;
                let runtime = runtimes[transport.rank() as usize].clone();
                let options = ScatterOptions { runtime, rows: ScatterRows::Explicit(rows.clone()) };
                let mut split = DistributedSum::scatter_with(input.clone(), transport, options);
                split.propagate(&circuit, &policy, Direction::Forward);
                assert!(split.local().len() <= samples);
                split.gather()
            });
            let got = gathered.into_iter().next().unwrap().unwrap();
            assert_draws_tile_l1(&got, l1, samples, &format!("seed {seed}"));
            for (x, z, c) in input.iter() {
                let points = got.get(x, z).map_or(0.0, |found| found.norm()) / step;
                let share = c.norm() / step;
                assert!(share.floor() <= points.round() && points.round() <= share.ceil());
            }
            assert_eq!(policy.collapses(), 1);
        }
    }
}

/// A run repartitioned mid-way by other rows, and its buckets rehashed by explicit rows, continues exactly as the unpartitioned run: no term is lost or duplicated, and every rank holds only its new partition's keys.
#[test]
fn repartitioning_and_rehashing_mid_run_match_propagate() {
    let n = 12;
    let sum = rand_sum::<1>(600, n, 0x2E9A);
    let circuit = mixed_circuit::<1>(n);
    let half = propagate(&circuit, sum.clone(), &KeepAll, Direction::Forward);
    let want = propagate(&circuit, half.clone(), &KeepAll, Direction::Forward);
    for size in [2u32, 4] {
        let bits = size.trailing_zeros() as u8;
        let gathered = on_ranks(size, |transport| {
            let mut split = DistributedSum::scatter_with(
                sum.clone(),
                transport,
                unpinned_scatter(1, 2, ROW_SEED),
            );
            split.propagate(&circuit, &KeepAll, Direction::Forward);
            let held = split.len();
            split.repartition(PartitionRows::from_seed(n, bits, ROW_SEED ^ 0xFFFF));
            split.assert_invariants();
            assert_eq!(split.len(), held, "repartitioning keeps every term");
            // A projection onto three coordinates leads; dense rows from the seed fill the rest.
            let hash =
                Gf2Hash::from_rows(n, 4, 0xB0C, vec![[0b1], [0], [0b10]], vec![[0], [0b100], [0]]);
            split.rehash_buckets(hash.clone());
            split.assert_invariants();
            assert_eq!(split.local().hash().rows(), hash.rows());
            let midway = split.gather();
            split.propagate(&circuit, &KeepAll, Direction::Forward);
            (midway, split.gather())
        });
        let (midway, out) = &gathered[0];
        assert_terms_close(
            midway.as_ref().expect("rank 0 gathers"),
            &half,
            TOL,
            &format!("midway ranks={size}"),
        );
        assert_terms_close(
            out.as_ref().expect("rank 0 gathers"),
            &want,
            TOL,
            &format!("ranks={size}"),
        );
    }
}

/// `repartition_share` moves shares that were split by other rows, the form `result="local"` chaining hands from one call to the next.
#[test]
fn repartition_share_moves_every_term_to_its_new_rank() {
    let n = 10;
    let sum = rand_sum::<1>(500, n, 0x5A4E);
    let old = PartitionRows::<1>::from_seed(n, 2, 1);
    let new = PartitionRows::<1>::from_seed(n, 2, 2);
    let shares = on_ranks(4, |transport| {
        let share = PartitionedSum::scatter_with(
            sum.clone(),
            ScatterOptions {
                runtime: PartitionRuntime::new(&unpinned_partitions(4, 1))
                    .expect("an unpinned placement resolves"),
                rows: ScatterRows::Explicit(old.clone()),
            },
        )
        .partition(transport.rank() as usize)
        .clone();
        repartition_share(share, &new, &transport)
    });
    let mut total = 0;
    for (rank, share) in shares.iter().enumerate() {
        for (x, z, _) in share.iter() {
            assert_eq!(new.partition_of(x, z), rank as u32);
        }
        total += share.len();
    }
    assert_eq!(total, sum.len());
}

/// Ranks that disagree about the new rows would misroute terms, so the repartition refuses.
#[test]
#[should_panic(expected = "disagree")]
fn ranks_repartitioning_by_different_rows_are_caught() {
    let n = 8;
    let sum = rand_sum::<1>(50, n, 0x0D1F);
    let mut group = InProcessTransport::group(2);
    let one = group.pop().expect("rank 1");
    let zero = group.pop().expect("rank 0");
    std::thread::scope(|scope| {
        let sum = &sum;
        scope.spawn(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut split = DistributedSum::scatter_with(
                    sum.clone(),
                    one,
                    unpinned_scatter(1, 2, ROW_SEED),
                );
                split.repartition(PartitionRows::from_seed(n, 1, 7));
            }));
        });
        let mut split =
            DistributedSum::scatter_with(sum.clone(), zero, unpinned_scatter(1, 2, ROW_SEED));
        split.repartition(PartitionRows::from_seed(n, 1, 8));
    });
}

/// A Heisenberg chain with a `Z` field and `Z` on the middle qubit, both real.
fn chain_and_seed(num_qubits: usize) -> (PauliSum<1, f64>, PauliSum<1, f64>) {
    let mut h = BuildAccumulator::<1, f64>::new(num_qubits);
    for j in 0..num_qubits - 1 {
        for f in [PauliString::<1>::x, PauliString::<1>::y, PauliString::<1>::z] {
            let mut p = f(j);
            p.mul_assign(&f(j + 1));
            h.add_term(p, 1.0);
        }
        h.add_term(PauliString::<1>::z(j), 0.3);
    }
    let mut seed = BuildAccumulator::<1, f64>::new(num_qubits);
    seed.add_term(PauliString::<1>::z(num_qubits / 2), 1.0);
    (h.finalize(), seed.finalize())
}

#[test]
fn lanczos_coefficients_agree_across_ranks() {
    let (num_qubits, steps, size) = (16, 8, 4u32);
    let (h, seed) = chain_and_seed(num_qubits);
    let policy = CoefficientThreshold(1e-10);

    let mut want = Vec::new();
    let (mut previous, mut current) = (PauliSum::<1, f64>::empty(num_qubits), seed.clone());
    for _ in 0..steps {
        let mut next = truncated_liouvillian(&h, &current, &policy);
        let mut scaled = previous.clone();
        scaled.scale(-want.last().copied().unwrap_or(0.0));
        next = next.add(&scaled);
        let norm = next.overlap(&next).sqrt();
        next.scale(1.0 / norm);
        want.push(norm);
        previous = std::mem::replace(&mut current, next);
    }

    let got: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = InProcessTransport::group(size)
            .into_iter()
            .zip(InProcessTransport::group(size))
            .map(|(transport, spare)| {
                let (h, seed, policy) = (&h, &seed, &policy);
                scope.spawn(move || {
                    let mut current = DistributedSum::scatter_with(
                        seed.clone(),
                        transport,
                        unpinned_scatter(1, 2, 0x5EED),
                    );
                    // `O_{n−1} ← L(O_n) − b_n·O_{n−1}` in place, so two sums and their two endpoints serve every step.
                    let mut previous = current.duplicate(spare);
                    let mut b: Vec<f64> = Vec::new();
                    for _ in 0..steps {
                        let beta = -b.last().copied().unwrap_or(0.0);
                        previous.add_liouvillian(h, &current, 1.0, beta, policy);
                        let norm = previous.overlap(&previous).sqrt();
                        previous.scale(1.0 / norm);
                        b.push(norm);
                        std::mem::swap(&mut previous, &mut current);
                    }
                    b
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("rank thread panicked")).collect()
    });
    for (rank, b) in got.iter().enumerate() {
        assert_eq!(b, &got[0], "rank {rank} disagrees with rank 0");
        for (n, (g, w)) in b.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-9 * w, "rank {rank} b_{}: {g} vs {w}", n + 1);
        }
    }
}

/// Rank groups split by `PartitionRowPolicy::for_run`, parity rows for a site and block rows for the magnetization, agree with one process.
#[test]
fn for_run_rows_match_propagate() {
    let (lx, ly) = (4, 4);
    let circuit = kicked_ising::<1>(lx * ly, &square_lattice(lx, ly), 0.4, 0.9, 2);
    let (site, magnetization) = site_and_magnetization::<1>(lx * ly, 9);
    let truncation = CoefficientThreshold(1e-7);
    for (name, observable) in [("site", site), ("magnetization", magnetization)] {
        let rows = PartitionRowPolicy::for_run(&observable, &circuit);
        let (want, discarded) =
            propagate_discarding(&circuit, observable.clone(), &truncation, Direction::Heisenberg);
        for size in [2u32, 4] {
            let gathered = on_ranks(size, |transport| {
                let mut split = scatter_by_policy(observable.clone(), transport, rows.clone());
                split.propagate(&circuit, &truncation, Direction::Heisenberg);
                split.gather()
            });
            let got = gathered.into_iter().next().unwrap().expect("rank 0 gathers");
            let what = format!("{name} ranks={size}");
            assert_runs_agree(truncation, discarded, &got, &want, TOL, &what);
        }
    }
}
