//! The MPI transport's differential net, run at whatever rank count it finds.
//!
//! `harness = false`: the cases are **collective**, so they must run in the
//! same order on every rank, one at a time, with nothing else in flight. The
//! libtest harness offers none of that — it may run tests in parallel, in an
//! unspecified order, and skip some — so `main` drives the matrix itself,
//! all-reduces each case's verdict and exits with the same status on every
//! rank.
//!
//! Two ways in:
//!
//! ```text
//! cargo test -p paulistrings --features mpi --test mpi_ranks   # a singleton world of one rank
//! scripts/mpi-test.sh --ranks 2,4                              # under mpirun
//! ```
//!
//! The oracle is always the unpartitioned single-process [`propagate`]: rank 0
//! runs it on the same replicated input and compares the gathered result with
//! [`assert_terms_close`], plus exact `len()` equality (the truncation policies
//! are partition-exact, so the term count is not a tolerance question).
//!
//! Threading is [`Threading::Serialized`], not `Funneled`: the layer loop runs
//! inside a Rayon pool, so MPI calls come off a pool worker rather than the
//! process's main thread. Only ever one at a time, which is what `SERIALIZED`
//! promises.

use std::panic::AssertUnwindSafe;

use num_complex::Complex64;
use paulistrings::channel::{
    Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation,
};
use paulistrings::engine::partitioned::{
    count_remote_deltas, DistributedSum, PartitionConfig, PartitionRuntime, BITS_AGREE_EVERY,
};
use paulistrings::mpi::{propagate_mpi, rsmpi, MpiTransport};
use paulistrings::test_support::{
    assert_terms_close, haar_su4_matrix, rand_sum, rand_sum_real, trotter_circuit,
    unpinned_partitions, zz_rotation, KeepAll,
};
use paulistrings::truncation::{And, ApproxTopN, CoefficientThreshold, WeightCutoff};
use paulistrings::{
    propagate, Circuit, Direction, PartitionRows, PartitionedTruncation, PauliString, PauliSum,
    PropagateOptions,
};
use rsmpi::collective::{CommunicatorCollectives, SystemOperation};
use rsmpi::topology::{Communicator, SimpleCommunicator};
use rsmpi::Threading;

const TOL: f64 = 1e-11;
/// The Trotter angle every `trotter_circuit` fixture here rotates by.
const THETA: f64 = 0.1;

// ---- circuits -----------------------------------------------------------

/// One weight-2 `ZZ` rotation: the smallest layer that can cross a boundary.
fn single_rotation<const W: usize>(nq: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(nq);
    circuit.push(zz_rotation::<W>(0, (nq / 2) as u32, 0.37));
    circuit
}

/// A ring of CNOTs: key-permuting, fanout 1, so every row an exchange moves is
/// a pure relabelling.
fn cnot_ring<const W: usize>(nq: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(nq);
    for q in 0..nq as u32 {
        circuit.push(Clifford2Q::cnot(q, (q + 1) % nq as u32));
    }
    circuit
}

/// Haar SU(4) blocks: all 15 deltas, the worst fan-out a two-qubit layer has.
fn haar_su4<const W: usize>(nq: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(nq);
    circuit.push(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix()));
    circuit.push(GeneralUnitary2Q::from_matrix(1, 2, haar_su4_matrix()));
    circuit.push(GeneralUnitary2Q::from_matrix(2, 3, haar_su4_matrix()));
    circuit
}

/// A long run of layers that cannot cross a **z-only cut row** — `X`
/// rotations, whose generator carries no `z` bits — with the one bond that
/// does cross at the end.
///
/// Longer than [`BITS_AGREE_EVERY`], so the bucket-count schedule has to skip
/// layers, and the closing bond is the layer that must agree before it
/// exchanges.
fn local_run_then_one_crossing<const W: usize>(nq: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(nq);
    for round in 0..3 {
        for q in 0..nq as u32 {
            circuit.push(PauliRotation::new(
                PauliString::<W>::x(q),
                0.21 + 0.01 * f64::from(round),
            ));
        }
    }
    circuit.push(zz_rotation::<W>(nq as u32 / 2 - 1, nq as u32 / 2, 0.33));
    circuit
}

/// `size` equal blocks of `nq` qubits, as a z-only cut: the rows that make
/// [`local_run_then_one_crossing`] one remote layer at any rank count.
fn block_cut<const W: usize>(nq: usize, size: usize) -> PartitionRows<W> {
    let per = nq / size;
    let blocks: Vec<Vec<u32>> = (0..size)
        .map(|b| ((b * per) as u32..((b + 1) * per) as u32).collect())
        .collect();
    PartitionRows::<W>::cut(nq, &blocks)
}

/// Depolarizing noise on every qubit: key-preserving, so the layer must issue
/// **no** transport call at all.
fn depolarizing_only<const W: usize>(nq: usize) -> Circuit<W> {
    let mut circuit = Circuit::<W>::new(nq);
    for q in 0..nq as u32 {
        circuit.push(Depolarizing {
            support: [q],
            p: 0.05,
        });
    }
    circuit
}

// ---- the harness --------------------------------------------------------

/// A rank's view of the run: its endpoint, and the group it is in.
struct Runner<'a> {
    world: &'a SimpleCommunicator,
    rank: u32,
    size: u32,
    failures: u64,
    cases: u64,
}

impl Runner<'_> {
    /// Placement for a rank's single partition. `Unpinned` rather than the
    /// production `Auto`: a CI container or an oversubscribed workstation may
    /// hand several ranks the same CPUs, and pinning them all to it would
    /// serialize the run. Placement itself is covered by `topology`'s tests.
    fn config(&self, seed: u64) -> PartitionConfig {
        unpinned_partitions(1, 2, seed)
    }

    /// Run one collective case, all-reduce its verdict, and report it on rank
    /// 0.
    ///
    /// A panic on any rank is caught, so the group reaches the reduction rather
    /// than leaving the survivors blocked in a collective; but a rank that died
    /// mid-exchange has left unmatched messages behind, so the first failure
    /// aborts the whole job instead of trying the next case.
    fn case(&mut self, name: &str, body: impl FnOnce(&mut Self)) {
        self.cases += 1;
        let failed = std::panic::catch_unwind(AssertUnwindSafe(|| body(self))).is_err();

        let send = [u64::from(failed)];
        let mut total = [0u64];
        self.world
            .all_reduce_into(&send[..], &mut total[..], SystemOperation::sum());
        if self.rank == 0 {
            if total[0] == 0 {
                println!("ok       {name}");
            } else {
                println!("FAILED   {name}  ({} of {} ranks)", total[0], self.size);
            }
        }
        if total[0] > 0 {
            self.failures += 1;
            // The panicking rank may have abandoned an exchange; anything after
            // this would deadlock or cross messages.
            if self.rank == 0 {
                println!("aborting: a failed collective case leaves the group out of step");
            }
            self.world.abort(1);
        }
    }

    /// Scatter `sum` over the group, propagate, gather on rank 0, and compare
    /// against the single-process oracle.
    // Seven knobs, and every one of them varies across the matrix; bundling
    // them into a struct would only move the argument list.
    #[allow(clippy::too_many_arguments)]
    fn differential<const W: usize, T>(
        &self,
        circuit: &Circuit<W>,
        sum: &PauliSum<W>,
        policy: &T,
        direction: Direction,
        seed: u64,
        chunk_bytes: Option<usize>,
        what: &str,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
    {
        let mut transport = MpiTransport::from_communicator(self.world);
        if let Some(bytes) = chunk_bytes {
            transport = transport.with_chunk_bytes(bytes);
        }
        let config = self.config(seed);
        let mut split =
            DistributedSum::scatter(sum.clone(), transport, &config).expect("topology resolves");
        split.assert_invariants();
        split.propagate(circuit, policy, direction);
        split.assert_invariants();
        let got = split.gather();

        match (self.rank, got) {
            (0, Some(got)) => {
                let want = propagate(circuit, sum.clone(), policy, direction);
                let what = format!("{what} ranks={} {direction:?}", self.size);
                assert_terms_close(&got, &want, TOL, &what);
                assert_eq!(got.len(), want.len(), "{what}: term count");
            }
            (0, None) => panic!("{what}: rank 0 did not gather"),
            (r, Some(_)) => panic!("{what}: rank {r} gathered but is not the root"),
            (_, None) => {}
        }
    }

    /// [`differential`](Self::differential) in both directions.
    fn both_directions<const W: usize, T>(
        &self,
        circuit: &Circuit<W>,
        sum: &PauliSum<W>,
        policy: &T,
        seed: u64,
        what: &str,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
    {
        for direction in [Direction::Forward, Direction::Heisenberg] {
            self.differential(circuit, sum, policy, direction, seed, None, what);
        }
    }
}

/// The partition-row seed the fixtures use unless a case needs a specific
/// draw.
const SEED: u64 = 0x5EED_C0FF_EE00_9001;

fn main() {
    let Some((universe, threading)) = rsmpi::initialize_with_threading(Threading::Serialized)
    else {
        eprintln!("MPI is already initialized in this process; mpi_ranks must own the universe");
        std::process::exit(2);
    };
    let world = universe.world();
    let rank = world.rank() as u32;
    let size = world.size() as u32;

    if rank == 0 {
        println!(
            "mpi_ranks: {size} rank(s), thread support {threading:?}, exchange chunks {}, \
             library built against {}",
            std::env::var("PAULISTRINGS_EXCHANGE_CHUNKS").unwrap_or_else(|_| "default".to_string()),
            MpiTransport::build_library_version(),
        );
        if threading < Threading::Serialized {
            println!(
                "warning: MPI provided {threading:?}, below the SERIALIZED the engine needs \
                 (its MPI calls come off a Rayon pool worker)",
            );
        }
    }
    if !size.is_power_of_two() {
        if rank == 0 {
            eprintln!("mpi_ranks needs a power-of-two rank count, got {size}");
        }
        drop(universe);
        std::process::exit(2);
    }

    let mut r = Runner {
        world: &world,
        rank,
        size,
        failures: 0,
        cases: 0,
    };
    run_matrix(&mut r);

    let (cases, failures) = (r.cases, r.failures);
    if rank == 0 {
        if failures == 0 {
            println!("mpi_ranks: {cases} case(s) ok on {size} rank(s)");
        } else {
            println!("mpi_ranks: {failures} of {cases} case(s) FAILED on {size} rank(s)");
        }
    }
    drop(universe);
    if failures > 0 {
        std::process::exit(1);
    }
}

fn run_matrix(r: &mut Runner) {
    // ---- W = 1, the layer shapes, both directions ----------------------
    r.case("w1 single zz rotation", |r| {
        let circuit = single_rotation::<1>(16);
        let sum = rand_sum_real::<1>(400, 16, 0xA001);
        r.both_directions(&circuit, &sum, &KeepAll, SEED, "w1 rotation");
    });

    r.case("w1 cnot ring", |r| {
        let circuit = cnot_ring::<1>(12);
        let sum = rand_sum::<1>(400, 12, 0xA002);
        r.both_directions(&circuit, &sum, &KeepAll, SEED, "w1 cnot");
    });

    r.case("w1 haar su(4)", |r| {
        let circuit = haar_su4::<1>(8);
        let sum = rand_sum::<1>(300, 8, 0xA003);
        r.both_directions(&circuit, &sum, &KeepAll, SEED, "w1 su4");
    });

    r.case("w1 trotter, 32 layers", |r| {
        let circuit = trotter_circuit::<1>(16, THETA);
        assert!(
            circuit.channels.len() >= 20,
            "the bits collective must move"
        );
        let sum = rand_sum_real::<1>(800, 16, 0xA004);
        r.both_directions(&circuit, &sum, &ApproxTopN(1_500), SEED, "w1 trotter");
    });

    // ---- W = 2: two-word keys through the same shapes -------------------
    r.case("w2 haar su(4)", |r| {
        let circuit = haar_su4::<2>(8);
        let sum = rand_sum::<2>(300, 8, 0xA005);
        r.both_directions(&circuit, &sum, &KeepAll, SEED, "w2 su4");
    });

    r.case("w2 trotter, 32 layers", |r| {
        let circuit = trotter_circuit::<2>(16, THETA);
        let sum = rand_sum_real::<2>(600, 16, 0xA006);
        r.both_directions(&circuit, &sum, &ApproxTopN(1_200), SEED, "w2 trotter");
    });

    // ---- policies -------------------------------------------------------
    r.case("policies on a mixed circuit", |r| {
        let mut circuit = cnot_ring::<1>(8);
        circuit.push(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix()));
        circuit.push(zz_rotation::<1>(0, 4, 0.31));
        circuit.push(GeneralUnitary2Q::from_matrix(2, 3, haar_su4_matrix()));
        let sum = rand_sum::<1>(250, 8, 0xA007);

        r.both_directions(&circuit, &sum, &KeepAll, SEED, "policy none");
        r.both_directions(
            &circuit,
            &sum,
            &CoefficientThreshold(1e-9),
            SEED,
            "policy eps",
        );
        r.both_directions(&circuit, &sum, &WeightCutoff(4), SEED, "policy weight");
        r.both_directions(&circuit, &sum, &ApproxTopN(400), SEED, "policy approx");
        r.both_directions(
            &circuit,
            &sum,
            &And(CoefficientThreshold(1e-12), ApproxTopN(400)),
            SEED,
            "policy and",
        );
    });

    /// A budget small enough that the collective octave selection fires on
    /// several layers, so the exact `len()` equality the differential asserts
    /// is a real test of partition-exactness.
    const TIGHT: usize = 200;

    r.case("approx-topn firing every layer", |r| {
        let circuit = trotter_circuit::<1>(14, THETA);
        let sum = rand_sum_real::<1>(600, 14, 0xA008);
        let want = propagate(
            &circuit,
            sum.clone(),
            &ApproxTopN(TIGHT),
            Direction::Forward,
        );
        if r.rank == 0 {
            assert!(
                want.len() <= 4 * TIGHT,
                "the budget must actually bind: {} terms out",
                want.len(),
            );
        }
        r.both_directions(&circuit, &sum, &ApproxTopN(TIGHT), SEED, "approx tight");
    });

    // ---- degenerate inputs ----------------------------------------------
    r.case("a three-term sum leaves ranks empty", |r| {
        let mut circuit = Circuit::<1>::new(8);
        circuit.push(Clifford1Q::h(0));
        circuit.push(Clifford2Q::cnot(0, 1));
        for n in 1..=3 {
            let sum = rand_sum::<1>(n, 8, 0xA009 + n as u64);
            r.both_directions(&circuit, &sum, &KeepAll, SEED, "tiny sum");
        }
    });

    r.case("an empty circuit round-trips the input", |r| {
        let circuit = Circuit::<1>::new(8);
        let sum = rand_sum::<1>(120, 8, 0xA00D);
        r.differential(
            &circuit,
            &sum,
            &KeepAll,
            Direction::Forward,
            SEED,
            None,
            "empty circuit",
        );
    });

    // ---- a layer that must not exchange ---------------------------------
    r.case("depolarizing issues no exchange", |r| {
        let circuit = depolarizing_only::<1>(10);
        let sum = rand_sum::<1>(300, 10, 0xA010);

        let transport = MpiTransport::from_communicator(r.world);
        let config = r.config(SEED);
        let mut split =
            DistributedSum::scatter(sum.clone(), transport, &config).expect("topology resolves");
        split.enable_trace();
        split.propagate(&circuit, &KeepAll, Direction::Forward);
        let trace = split.take_trace().expect("tracing is on");

        assert_eq!(trace.layers.len(), circuit.channels.len());
        assert_eq!(
            trace.remote_layers(),
            0,
            "a key-preserving channel must issue no transport call",
        );
        assert_eq!(trace.total_rows_exchanged(), 0);
        // The per-rank trace shape: one local entry, a group-wide send row.
        for record in &trace.layers {
            assert_eq!(record.terms_in.len(), 1);
            assert_eq!(record.rows_sent[0].len(), r.size as usize);
        }

        if let Some(got) = split.gather() {
            let want = propagate(&circuit, sum, &KeepAll, Direction::Forward);
            assert_terms_close(&got, &want, TOL, "depolarizing");
            assert_eq!(got.len(), want.len());
        }
    });

    // ---- the collective schedule ----------------------------------------
    r.case("a long local run skips the bucket-count collective", |r| {
        const NQ: usize = 16;
        let circuit = local_run_then_one_crossing::<1>(NQ);
        let layers = circuit.channels.len();
        assert!(
            layers > 2 * BITS_AGREE_EVERY,
            "the schedule must have something to skip: {layers} layers",
        );
        let rows = block_cut::<1>(NQ, r.size as usize);
        let sum = rand_sum_real::<1>(600, NQ, 0xA021);

        let config = r.config(SEED);
        let runtime = PartitionRuntime::new(&config).expect("topology resolves");
        let transport = MpiTransport::from_communicator(r.world);
        let mut split = DistributedSum::scatter_with_rows(sum.clone(), transport, runtime, rows);
        split.enable_trace();
        split.propagate(&circuit, &WeightCutoff(4), Direction::Forward);
        let trace = split.take_trace().expect("tracing is on");

        assert_eq!(trace.layers.len(), layers);
        assert_eq!(
            trace.remote_layers(),
            usize::from(r.size > 1),
            "only the closing bond crosses, and only when there is a group",
        );
        // The opening ramp, one per period after it, and the remote layer. At
        // one rank there is no group at all and the count is zero.
        let allowed = if r.size == 1 {
            0
        } else {
            (BITS_AGREE_EVERY + layers.div_ceil(BITS_AGREE_EVERY) + 1) as u64
        };
        assert!(
            trace.total_collectives() <= allowed,
            "{} collectives over {layers} layers, the schedule allows {allowed}",
            trace.total_collectives(),
        );
        // The crossing layer agreed first — otherwise a receiver would index a
        // partner's blocks by the wrong bucket count.
        let last = trace.layers.last().expect("layers");
        assert!(
            r.size == 1 || (last.remote_deltas > 0 && last.collectives > 0),
            "a remote layer must agree the bucket count before it exchanges",
        );

        if let Some(got) = split.gather() {
            let want = propagate(&circuit, sum, &WeightCutoff(4), Direction::Forward);
            assert_terms_close(&got, &want, TOL, "long local run");
            assert_eq!(got.len(), want.len());
        }
    });

    // ---- the one-shot front door ----------------------------------------
    r.case("propagate_mpi is the persistent driver in one call", |r| {
        let circuit = cnot_ring::<1>(10);
        let sum = rand_sum::<1>(300, 10, 0xA015);
        // Everything the persistent path does — duplicate, place, scatter, run,
        // gather — behind one call, under the production `default_config`
        // placement rather than this file's unpinned one.
        let got = propagate_mpi(
            &circuit,
            sum.clone(),
            &KeepAll,
            Direction::Forward,
            PropagateOptions::default(),
            r.world,
        );
        match (r.rank, got) {
            (0, Some(got)) => {
                let want = propagate(&circuit, sum, &KeepAll, Direction::Forward);
                assert_terms_close(&got, &want, TOL, "propagate_mpi");
                assert_eq!(got.len(), want.len());
            }
            (0, None) => panic!("propagate_mpi: rank 0 did not gather"),
            (rank, Some(_)) => panic!("propagate_mpi: rank {rank} gathered but is not the root"),
            (_, None) => {}
        }
    });

    // ---- a layer whose every row crosses --------------------------------
    r.case("an all-remote rotation", |r| {
        let nq = 12;
        let sum = rand_sum_real::<1>(400, nq, 0xA020);
        let circuit = single_rotation::<1>(nq);

        if r.size == 1 {
            // A group of one has no boundary to cross; the case degenerates to
            // the plain differential, which is still worth running.
            r.both_directions(&circuit, &sum, &KeepAll, SEED, "all-remote (P=1)");
            return;
        }

        // Search for a row draw under which the rotation's generator pass is
        // remote — every anticommuting term then leaves the rank. Deterministic,
        // and every rank searches the same sequence, so they agree without a
        // collective.
        let pbits = r.size.trailing_zeros() as u8;
        let seed = (0u64..4096)
            .find(|&s| {
                let rows = PartitionRows::<1>::from_seed(nq, pbits, SEED ^ s);
                count_remote_deltas(&circuit, sum.hash(), &rows, false)
                    .iter()
                    .all(|&(_, remote)| remote > 0)
            })
            .map(|s| SEED ^ s)
            .expect("some row draw sends the generator across");

        let rows = PartitionRows::<1>::from_seed(nq, pbits, seed);
        let counts = count_remote_deltas(&circuit, sum.hash(), &rows, false);
        assert_eq!(counts.len(), 1);
        assert!(counts[0].1 > 0, "the layer must be remote: {counts:?}");

        r.both_directions(&circuit, &sum, &KeepAll, seed, "all-remote");
    });

    // ---- the chunked send path ------------------------------------------
    r.case("multi-chunk parts at 1 KiB", |r| {
        let circuit = haar_su4::<1>(8);
        let sum = rand_sum::<1>(1_200, 8, 0xA030);
        // A 1200-term sum through three SU(4) blocks moves far more than 1 KiB
        // per column, so every part is chunked several times over.
        for direction in [Direction::Forward, Direction::Heisenberg] {
            r.differential(
                &circuit,
                &sum,
                &KeepAll,
                direction,
                SEED,
                Some(1024),
                "chunked",
            );
        }
    });

    r.case("a one-byte chunk size still delivers", |r| {
        // The pathological end of the same path: every byte its own message.
        // Small input, or the message count explodes.
        let mut circuit = Circuit::<1>::new(6);
        circuit.push(Clifford2Q::cnot(0, 3));
        let sum = rand_sum::<1>(24, 6, 0xA031);
        r.differential(
            &circuit,
            &sum,
            &KeepAll,
            Direction::Forward,
            SEED,
            Some(1),
            "one-byte chunks",
        );
    });

    // ---- the consistency check -------------------------------------------
    r.case("ranks driven through different circuits are caught", |r| {
        if r.size == 1 {
            // Nothing to disagree with.
            return;
        }
        let sum = rand_sum::<1>(60, 8, 0xA040);
        // Rank 1 is handed one extra layer. The check must panic on *every*
        // rank (the reduction is symmetric), and it must do so before the first
        // exchange, so the group is still in step afterwards.
        let mut circuit = Circuit::<1>::new(8);
        circuit.push(Clifford1Q::h(0));
        if r.rank == 1 {
            circuit.push(Clifford1Q::h(1));
        }

        let transport = MpiTransport::from_communicator(r.world);
        let config = r.config(SEED);
        let mut split =
            DistributedSum::scatter(sum, transport, &config).expect("topology resolves");
        let out = std::panic::catch_unwind(AssertUnwindSafe(|| {
            split.propagate(&circuit, &KeepAll, Direction::Forward);
        }));
        assert!(
            out.is_err(),
            "rank {} accepted a circuit its peers do not have",
            r.rank,
        );
    });

    // ---- expectation values, read out without a gather --------------------
    r.case("local expectations sum to the whole sum's", |r| {
        use paulistrings::ProductState;

        let circuit = cnot_ring::<1>(10);
        let sum = rand_sum_real::<1>(500, 10, 0xA050);
        let transport = MpiTransport::from_communicator(r.world);
        let config = r.config(SEED);
        let mut split =
            DistributedSum::scatter(sum.clone(), transport, &config).expect("topology resolves");
        split.propagate(&circuit, &KeepAll, Direction::Forward);

        let mine = split.local_expectation_product_state(ProductState::ZPlus);
        // `Collectives` reduces integers only, so the scalar goes through the
        // communicator directly — the pattern a caller uses for its own
        // observables.
        let send = [mine.re, mine.im];
        let mut total = [0.0f64; 2];
        r.world
            .all_reduce_into(&send[..], &mut total[..], SystemOperation::sum());
        let got = Complex64::new(total[0], total[1]);

        let oracle = propagate(&circuit, sum, &KeepAll, Direction::Forward);
        let want = oracle.expectation_product_state(ProductState::ZPlus);
        assert!(
            (got - want).norm() < 1e-9,
            "rank {}: expectation {got} vs {want}",
            r.rank,
        );

        // And the collective term count agrees with the oracle's, on every
        // rank — `len` is the whole sum's, `len_local` this rank's share.
        assert_eq!(split.len(), oracle.len(), "rank {}: collective len", r.rank);
        assert!(split.len_local() <= split.len());
    });
}
