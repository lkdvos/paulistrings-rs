//! `DistributedSum` against the unpartitioned `propagate`.
//!
//! The distributed driver is the partitioned engine with one partition per
//! *process*. Its contract is the same as the in-process one's — the gathered
//! output agrees with `propagate` to floating-point tolerance (ARCHITECTURE.md
//! §Determinism) — but the code path differs: the layer loop runs on one thread
//! per rank rather than fanned out from a single driver, the scatter is
//! replicated rather than split from one sum, and the gather goes through the
//! transport's byte framing instead of a local merge.
//!
//! All of that is transport-independent, so this file drives it with
//! `InProcessTransport`: one thread per "rank", no MPI, no launcher. The MPI
//! transport's own net is `tests/mpi_ranks.rs`, run under `mpirun`.

use num_complex::Complex64;
use paulistrings::channel::{Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q};
use paulistrings::engine::partitioned::{DistributedSum, InProcessTransport, PartitionConfig};
use paulistrings::test_support::{
    assert_terms_close, haar_su4_matrix, rand_sum, rand_sum_real, trotter_circuit,
    unpinned_partitions, zz_rotation, KeepAll,
};
use paulistrings::truncation::{And, ApproxTopN, CoefficientThreshold, WeightCutoff};
use paulistrings::{propagate, Circuit, Direction, PartitionedTruncation, PauliSum};

const TOL: f64 = 1e-11;
/// The Trotter angle every `trotter_circuit` fixture here rotates by. Long
/// enough a circuit that the bucket count grows mid-run, so the per-layer bits
/// all-reduce actually changes value.
const THETA: f64 = 0.1;

/// One unpinned partition of two threads: what a rank's runtime looks like when
/// the launcher, not the engine, did the placement.
fn config() -> PartitionConfig {
    unpinned_partitions(1, 2, 0x5EED_C0FF_EE00_4321)
}

/// A short circuit mixing the layer shapes that matter: a local-ish Clifford, a
/// dense Haar SU(4) (all 15 deltas, worst fan-out), a rotation, and a noise
/// channel (which is key-preserving, so it never exchanges).
fn mixed_circuit<const W: usize>(num_qubits: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(num_qubits);
    let n = num_qubits as u32;
    for q in 0..n {
        circuit.push(Clifford2Q::cnot(q, (q + 1) % n));
    }
    circuit.push(Clifford1Q::h(0));
    circuit.push(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix()));
    circuit.push(zz_rotation::<W>(0, n / 2, 0.31));
    for q in 0..n {
        circuit.push(Depolarizing {
            support: [q],
            p: 0.04,
        });
    }
    circuit.push(GeneralUnitary2Q::from_matrix(1, 2, haar_su4_matrix()));
    circuit
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
    T: PartitionedTruncation<W> + Sync,
{
    let gathered: Vec<Option<PauliSum<W>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = InProcessTransport::group(size)
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let mut split = DistributedSum::scatter(sum.clone(), transport, &config())
                        .expect("topology resolves");
                    split.assert_invariants();
                    split.propagate(circuit, policy, direction);
                    split.assert_invariants();
                    let out = split.gather();
                    // `local` is always this rank's share, whether or not the
                    // rank gathers.
                    assert!(split.local().len() <= split.len());
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect()
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
    T: PartitionedTruncation<W> + Sync,
{
    for &direction in &[Direction::Forward, Direction::Heisenberg] {
        let want = propagate(circuit, sum.clone(), policy, direction);
        for size in [1u32, 2, 4] {
            let got = distributed(circuit, sum, policy, direction, size);
            let what = format!("{name} ranks={size} {direction:?}");
            assert_terms_close(&got, &want, TOL, &what);
            assert_eq!(got.len(), want.len(), "{what}: term count");
        }
    }
}

#[test]
fn trotter_matches_propagate_w1() {
    let circuit = trotter_circuit::<1>(24, THETA);
    let sum = rand_sum_real::<1>(1_200, 24, 0x0D15);
    check(&circuit, &sum, &ApproxTopN(2_000), "trotter approx w1");
    check(
        &circuit,
        &sum,
        &And(CoefficientThreshold(1e-9), ApproxTopN(2_000)),
        "trotter and w1",
    );
}

#[test]
fn trotter_matches_propagate_w2() {
    let circuit = trotter_circuit::<2>(24, THETA);
    let sum = rand_sum_real::<2>(900, 24, 0x0D16);
    check(&circuit, &sum, &ApproxTopN(1_500), "trotter w2");
}

/// Dense two-qubit blocks and a noise channel, at both widths: the layer
/// shapes with the widest fan-out, and the one that must issue no exchange at
/// all.
#[test]
fn mixed_channels_match_propagate() {
    let circuit = mixed_circuit::<1>(6);
    let sum = rand_sum::<1>(200, 6, 0x0D17);
    check(&circuit, &sum, &KeepAll, "mixed keep w1");
    check(&circuit, &sum, &CoefficientThreshold(1e-9), "mixed eps w1");
    check(&circuit, &sum, &WeightCutoff(4), "mixed weight w1");
    check(&circuit, &sum, &ApproxTopN(300), "mixed approx w1");

    let circuit = mixed_circuit::<2>(6);
    let sum = rand_sum::<2>(200, 6, 0x0D18);
    check(&circuit, &sum, &KeepAll, "mixed keep w2");
    check(&circuit, &sum, &ApproxTopN(300), "mixed approx w2");
}

/// A sum small enough that some ranks hold nothing: the empty-partition path
/// through scatter, the layer, the gather framing and the merge.
#[test]
fn a_two_term_sum_survives_four_ranks() {
    let mut circuit = Circuit::<1>::new(4);
    circuit.push(Clifford1Q::h(0));
    circuit.push(Clifford2Q::cnot(0, 1));

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

    let (first, second) = std::thread::scope(|scope| {
        let circuit = &circuit;
        let sum = &sum;
        let handles: Vec<_> = InProcessTransport::group(2)
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let mut split = DistributedSum::scatter(sum.clone(), transport, &config())
                        .expect("topology resolves");
                    split.propagate(circuit, &KeepAll, Direction::Forward);
                    let first = split.gather();
                    split.propagate(circuit, &KeepAll, Direction::Forward);
                    (first, split.gather())
                })
            })
            .collect();
        let mut out: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect();
        out.remove(0)
    });

    assert_terms_close(&first.expect("rank 0"), &want_once, TOL, "first gather");
    assert_terms_close(&second.expect("rank 0"), &want_twice, TOL, "second gather");
}

/// The per-rank trace: one layer record per layer, `terms_in`/`terms_out` a
/// single (local) entry, `rows_sent` indexed by destination rank.
#[test]
fn the_trace_is_this_ranks_view_of_every_layer() {
    let circuit = mixed_circuit::<1>(6);
    let sum = rand_sum::<1>(200, 6, 0x0D1C);
    let layers = circuit.channels.len();

    let traces = std::thread::scope(|scope| {
        let circuit = &circuit;
        let sum = &sum;
        let handles: Vec<_> = InProcessTransport::group(2)
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let mut split = DistributedSum::scatter(sum.clone(), transport, &config())
                        .expect("topology resolves");
                    split.enable_trace();
                    split.propagate(circuit, &KeepAll, Direction::Forward);
                    (split.rank(), split.take_trace().expect("tracing is on"))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect::<Vec<_>>()
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
        assert!(
            trace.local_layers() > 0,
            "rank {rank}: the noise layers exchange nothing",
        );
    }
}

/// Ranks driven through different circuits are named, not left to deadlock.
#[test]
#[should_panic(expected = "disagree about the run")]
fn ranks_driven_through_different_circuits_are_caught() {
    let sum = rand_sum::<1>(50, 4, 0x0D1D);
    let short = Circuit::<1>::new(4);
    let mut long = Circuit::<1>::new(4);
    long.push(Clifford1Q::h(0));

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
                let mut split =
                    DistributedSum::scatter(sum.clone(), one, &config()).expect("topology");
                split.propagate(long, &KeepAll, Direction::Forward);
            }));
        });
        let mut split = DistributedSum::scatter(sum.clone(), zero, &config()).expect("topology");
        split.propagate(&short, &KeepAll, Direction::Forward);
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
    let _ = DistributedSum::scatter(sum, zero, &unpinned_partitions(2, 1, 3));
}

/// Every rank's `len()` is the whole sum's, and the local shares add up to it.
#[test]
fn len_is_collective_and_the_shares_add_up() {
    let sum = rand_sum::<1>(500, 6, 0x0D20);
    let want = sum.len();

    let seen = std::thread::scope(|scope| {
        let sum = &sum;
        let handles: Vec<_> = InProcessTransport::group(4)
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    let split = DistributedSum::scatter(sum.clone(), transport, &config())
                        .expect("topology resolves");
                    (split.len(), split.len_local())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect::<Vec<_>>()
    });

    for (rank, &(total, _)) in seen.iter().enumerate() {
        assert_eq!(total, want, "rank {rank}: collective len");
    }
    assert_eq!(seen.iter().map(|&(_, local)| local).sum::<usize>(), want);
    // A 500-term sum over four ranks: nobody holds all of it.
    assert!(seen.iter().all(|&(_, local)| local < want));
}

/// `local_expectation_product_state` is per rank, and the ranks' answers sum to
/// the whole sum's.
#[test]
fn local_expectations_sum_to_the_whole_sums() {
    use paulistrings::ProductState;

    let sum = rand_sum_real::<1>(300, 6, 0x0D21);
    let state = ProductState::ZPlus;
    let want = sum.expectation_product_state(state);

    let parts: Vec<Complex64> = std::thread::scope(|scope| {
        let sum = &sum;
        let handles: Vec<_> = InProcessTransport::group(4)
            .into_iter()
            .map(|transport| {
                scope.spawn(move || {
                    DistributedSum::scatter(sum.clone(), transport, &config())
                        .expect("topology resolves")
                        .local_expectation_product_state(state)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread panicked"))
            .collect()
    });

    let got: Complex64 = parts.iter().sum();
    assert!((got - want).norm() < TOL, "{got} vs {want}");
}
