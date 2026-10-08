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
//! Built with `--features cuda` as well, the matrix repeats its layer shapes with one CUDA device per rank (`MpiGpuSum`), ranks sharing a device when there are fewer devices than ranks; those cases are skipped on every rank unless every rank sees a device.
//!
//! Threading is [`Threading::Serialized`], not `Funneled`: the layer loop runs
//! inside a Rayon pool, so MPI calls come off a pool worker rather than the
//! process's main thread. Only ever one at a time, which is what `SERIALIZED`
//! promises.

use std::panic::AssertUnwindSafe;

use num_complex::Complex64;
use paulistrings::mpi::{propagate_mpi, rsmpi, MpiTransport};
use paulistrings::test_support::{
    assert_terms_close, haar_su4_matrix, rand_sum, rand_sum_real, trotter_circuit,
    unpinned_partitions, zz_rotation, KeepAll,
};
use paulistrings::test_support::{count_remote_deltas, BITS_AGREE_EVERY};
use paulistrings::{
    propagate, BuildAccumulator, Circuit, Direction, PartitionRows, PartitionedTruncation,
    PauliString, PauliSum, Phase, PropagateOptions,
};
use paulistrings::{And, ApproxTopN, BuiltinTruncation, CoefficientThreshold, WeightCutoff};
use paulistrings::{Clifford1Q, Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation};
use paulistrings::{
    Collectives, DistributedSum, PartitionConfig, PartitionRowPolicy, PartitionRuntime,
    ScatterOptions, ScatterRows,
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
            abort_world(self.world);
        }
    }
}

/// How long `MPI_Abort` gets to end the process before the watchdog does.
const ABORT_GRACE: std::time::Duration = std::time::Duration::from_secs(15);

/// `MPI_Abort`, with a watchdog that ends the process itself if the launcher has not: an abort that blocks in the runtime, or a teardown stuck behind a hung device, would otherwise hold the whole allocation.
fn abort_world(world: &SimpleCommunicator) -> ! {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    let rank = world.rank();
    std::thread::spawn(move || {
        std::thread::sleep(ABORT_GRACE);
        eprintln!(
            "mpi_ranks: rank {rank}: MPI_Abort did not end the process within {ABORT_GRACE:?}; aborting it"
        );
        std::process::abort();
    });
    world.abort(1)
}

/// Where a case's partitions live: host memory, or one CUDA device per rank (`MpiGpuSum`, feature `cuda`).
#[derive(Clone, Copy, PartialEq)]
enum Backend {
    Host,
    #[cfg(feature = "cuda")]
    Device,
}

impl Runner<'_> {
    /// Scatter `sum` over the group on `backend`, propagate, gather on rank 0, and compare against the single-process oracle.
    // Every argument varies across the matrix; bundling them would only move the list.
    #[allow(clippy::too_many_arguments)]
    fn differential<const W: usize, T>(
        &self,
        backend: Backend,
        circuit: &Circuit<W>,
        sum: &PauliSum<W>,
        policy: &T,
        direction: Direction,
        seed: u64,
        chunk_bytes: Option<usize>,
        what: &str,
    ) where
        T: PartitionedTruncation<W> + Clone + Into<BuiltinTruncation>,
    {
        let mut transport = MpiTransport::from_communicator(self.world);
        if let Some(bytes) = chunk_bytes {
            transport = transport.with_chunk_bytes(bytes);
        }
        let got = match backend {
            Backend::Host => {
                let mut split = DistributedSum::scatter(sum.clone(), transport, &self.config(seed))
                    .expect("topology resolves");
                split.assert_invariants();
                split.propagate(circuit, policy, direction);
                split.assert_invariants();
                split.gather()
            }
            #[cfg(feature = "cuda")]
            Backend::Device => {
                use paulistrings::gpu::{local_device_for_comm, MpiGpuSum};
                let device = local_device_for_comm(self.world).expect("a device on every rank");
                let mut split = MpiGpuSum::scatter_to_device(
                    sum,
                    transport,
                    device,
                    &PartitionRowPolicy::Seeded(Some(seed)),
                )
                .expect("device scatter");
                assert_eq!(split.device(), device);
                split
                    .propagate(circuit, policy, direction)
                    .expect("device propagate");
                split.gather().expect("device gather")
            }
        };
        self.compare(circuit, sum, policy, direction, got, what);
    }

    /// Rank 0's gathered result against [`propagate`], term count included; every other rank must hold `None`.
    fn compare<const W: usize, T>(
        &self,
        circuit: &Circuit<W>,
        sum: &PauliSum<W>,
        policy: &T,
        direction: Direction,
        got: Option<PauliSum<W>>,
        what: &str,
    ) where
        T: PartitionedTruncation<W> + ?Sized,
    {
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
        backend: Backend,
        circuit: &Circuit<W>,
        sum: &PauliSum<W>,
        policy: &T,
        seed: u64,
        what: &str,
    ) where
        T: PartitionedTruncation<W> + Clone + Into<BuiltinTruncation>,
    {
        for direction in [Direction::Forward, Direction::Heisenberg] {
            self.differential(backend, circuit, sum, policy, direction, seed, None, what);
        }
    }
}

/// A partition-row seed under which every layer of `circuit` over `sum` is remote, found identically on every rank.
fn all_remote_seed(circuit: &Circuit<1>, sum: &PauliSum<1>, size: u32) -> u64 {
    let nq = sum.num_qubits();
    let pbits = size.trailing_zeros() as u8;
    (0u64..4096)
        .map(|s| SEED ^ s)
        .find(|&seed| {
            let rows = PartitionRows::<1>::from_seed(nq, pbits, seed);
            count_remote_deltas(circuit, sum.hash(), &rows, false)
                .iter()
                .all(|&(_, remote)| remote > 0)
        })
        .expect("some row draw sends the generator across")
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
    run_collectives(&mut r);
    run_matrix(&mut r, Backend::Host);
    run_host_cases(&mut r);
    #[cfg(feature = "cuda")]
    device::run_device_matrix(&mut r);

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

/// The collectives every backend shares, once per run.
fn run_collectives(r: &mut Runner) {
    r.case("allreduce_sum_f64 agrees bitwise on every rank", |r| {
        let transport = MpiTransport::from_communicator(r.world);
        let (rank, size) = (r.rank as usize, r.size as usize);
        // Slot 0 is order-sensitive, slot `1 + rank` is filled by this rank alone.
        let mut buf = vec![0.0f64; 1 + size];
        buf[0] = 0.1 * (rank + 1) as f64;
        buf[1 + rank] = rank as f64 + 0.5;
        transport.allreduce_sum_f64(&mut buf);

        let want0 = 0.1 * (size * (size + 1) / 2) as f64;
        assert!(
            (buf[0] - want0).abs() < 1e-12,
            "slot 0: {} vs {want0}",
            buf[0]
        );
        for (q, v) in buf[1..].iter().enumerate() {
            assert_eq!(*v, q as f64 + 0.5, "slot {q} is one rank's and exact");
        }
        // Equal bits everywhere iff each word's sum is `size` copies of this rank's.
        let mut bits: Vec<u64> = buf.iter().map(|v| v.to_bits()).collect();
        let mine = bits.clone();
        transport.allreduce_sum_u64(&mut bits);
        for (got, own) in bits.iter().zip(&mine) {
            assert_eq!(
                *got,
                own.wrapping_mul(size as u64),
                "ranks disagree on the bits"
            );
        }
    });
}

/// The layer shapes every backend runs, both directions; case names carry `tag`.
fn run_matrix(r: &mut Runner, backend: Backend) {
    let tag = match backend {
        Backend::Host => "",
        #[cfg(feature = "cuda")]
        Backend::Device => "device ",
    };
    r.case(&format!("{tag}w1 single zz rotation"), |r| {
        let circuit = single_rotation::<1>(16);
        let sum = rand_sum_real::<1>(400, 16, 0xA001);
        r.both_directions(backend, &circuit, &sum, &KeepAll, SEED, "w1 rotation");
    });

    r.case(&format!("{tag}w1 cnot ring"), |r| {
        let circuit = cnot_ring::<1>(12);
        let sum = rand_sum::<1>(400, 12, 0xA002);
        r.both_directions(backend, &circuit, &sum, &KeepAll, SEED, "w1 cnot");
    });

    r.case(&format!("{tag}w1 haar su(4)"), |r| {
        let circuit = haar_su4::<1>(8);
        let sum = rand_sum::<1>(300, 8, 0xA003);
        r.both_directions(backend, &circuit, &sum, &KeepAll, SEED, "w1 su4");
    });

    r.case(&format!("{tag}w1 trotter, 32 layers"), |r| {
        let circuit = trotter_circuit::<1>(16, THETA);
        assert!(
            circuit.channels.len() >= 20,
            "the bits collective must move"
        );
        let sum = rand_sum_real::<1>(800, 16, 0xA004);
        r.both_directions(
            backend,
            &circuit,
            &sum,
            &ApproxTopN(1_500),
            SEED,
            "w1 trotter",
        );
    });

    r.case(&format!("{tag}w2 haar su(4)"), |r| {
        let circuit = haar_su4::<2>(8);
        let sum = rand_sum::<2>(300, 8, 0xA005);
        r.both_directions(backend, &circuit, &sum, &KeepAll, SEED, "w2 su4");
    });

    r.case(&format!("{tag}w2 trotter, 32 layers"), |r| {
        let circuit = trotter_circuit::<2>(16, THETA);
        let sum = rand_sum_real::<2>(600, 16, 0xA006);
        r.both_directions(
            backend,
            &circuit,
            &sum,
            &ApproxTopN(1_200),
            SEED,
            "w2 trotter",
        );
    });

    /// A budget small enough that the collective octave selection fires on several layers, so the exact `len()` equality is a real test of partition-exactness.
    const TIGHT: usize = 200;

    r.case(&format!("{tag}approx-topn firing every layer"), |r| {
        let circuit = trotter_circuit::<1>(14, THETA);
        let sum = rand_sum_real::<1>(600, 14, 0xA008);
        let want = propagate(
            &circuit,
            sum.clone(),
            &ApproxTopN(TIGHT),
            Direction::Forward,
        );
        assert!(
            want.len() <= 4 * TIGHT,
            "the budget must actually bind: {} terms out",
            want.len(),
        );
        r.both_directions(
            backend,
            &circuit,
            &sum,
            &ApproxTopN(TIGHT),
            SEED,
            "approx tight",
        );
    });

    r.case(&format!("{tag}an all-remote rotation"), |r| {
        let sum = rand_sum_real::<1>(400, 12, 0xA020);
        let circuit = single_rotation::<1>(12);
        // A group of one has no boundary to cross; the plain differential still runs.
        let seed = if r.size == 1 {
            SEED
        } else {
            all_remote_seed(&circuit, &sum, r.size)
        };
        r.both_directions(backend, &circuit, &sum, &KeepAll, seed, "all-remote");
    });

    r.case(&format!("{tag}multi-chunk parts at 1 KiB"), |r| {
        let circuit = haar_su4::<1>(8);
        let sum = rand_sum::<1>(1_200, 8, 0xA030);
        // Far more than 1 KiB per column, so every part is chunked several times over.
        for direction in [Direction::Forward, Direction::Heisenberg] {
            r.differential(
                backend,
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
}

/// The host-only cases.
fn run_host_cases(r: &mut Runner) {
    let backend = Backend::Host;
    // ---- policies -------------------------------------------------------
    r.case("policies on a mixed circuit", |r| {
        let mut circuit = cnot_ring::<1>(8);
        circuit.push(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix()));
        circuit.push(zz_rotation::<1>(0, 4, 0.31));
        circuit.push(GeneralUnitary2Q::from_matrix(2, 3, haar_su4_matrix()));
        let sum = rand_sum::<1>(250, 8, 0xA007);

        r.both_directions(backend, &circuit, &sum, &KeepAll, SEED, "policy none");
        r.both_directions(
            backend,
            &circuit,
            &sum,
            &CoefficientThreshold(1e-9),
            SEED,
            "policy eps",
        );
        r.both_directions(
            backend,
            &circuit,
            &sum,
            &WeightCutoff(4),
            SEED,
            "policy weight",
        );
        r.both_directions(
            backend,
            &circuit,
            &sum,
            &ApproxTopN(400),
            SEED,
            "policy approx",
        );
        r.both_directions(
            backend,
            &circuit,
            &sum,
            &And(CoefficientThreshold(1e-12), ApproxTopN(400)),
            SEED,
            "policy and",
        );
    });

    // ---- degenerate inputs ----------------------------------------------
    r.case("a three-term sum leaves ranks empty", |r| {
        let mut circuit = Circuit::<1>::new(8);
        circuit.push(Clifford1Q::h(0));
        circuit.push(Clifford2Q::cnot(0, 1));
        for n in 1..=3 {
            let sum = rand_sum::<1>(n, 8, 0xA009 + n as u64);
            r.both_directions(backend, &circuit, &sum, &KeepAll, SEED, "tiny sum");
        }
    });

    r.case("an empty circuit round-trips the input", |r| {
        let circuit = Circuit::<1>::new(8);
        let sum = rand_sum::<1>(120, 8, 0xA00D);
        r.differential(
            backend,
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
        let options = ScatterOptions {
            runtime,
            rows: ScatterRows::Explicit(rows),
        };
        let mut split = DistributedSum::scatter_with(sum.clone(), transport, options);
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

    // ---- the row policy -------------------------------------------------
    r.case("a cut row policy puts each block on its own rank", |r| {
        const NQ: usize = 16;
        let size = r.size as usize;
        let per = NQ / size;
        let blocks: Vec<Vec<u32>> = (0..size)
            .map(|b| ((b * per) as u32..((b + 1) * per) as u32).collect())
            .collect();

        // One `Z` per qubit: a single-`Z` key has odd z-weight in exactly the
        // block holding that qubit, so its rank is that block's index.
        let mut acc = BuildAccumulator::<1>::new(NQ);
        for q in 0..NQ as u32 {
            acc.add_term(
                PauliString::<1>::z(q),
                Phase::ONE,
                Complex64::new(1.0 + f64::from(q), 0.0),
            );
        }
        let sum = acc.finalize();

        let transport = MpiTransport::from_communicator(r.world);
        let options = ScatterOptions {
            runtime: PartitionRuntime::new(&r.config(SEED)).expect("topology resolves"),
            rows: ScatterRows::Policy(PartitionRowPolicy::Cut(blocks.clone())),
        };
        let mut split = DistributedSum::scatter_with(sum.clone(), transport, options);
        split.assert_invariants();

        let (_, z, _) = split.local().to_arrays();
        let mut held: Vec<u32> = z.iter().map(|row| row[0].trailing_zeros()).collect();
        held.sort_unstable();
        assert_eq!(held, blocks[r.rank as usize], "rank {}'s block", r.rank);

        // A caller's rows are a placement, not a semantics: the gathered answer
        // is still the serial one.
        let circuit = local_run_then_one_crossing::<1>(NQ);
        split.propagate(&circuit, &WeightCutoff(4), Direction::Forward);
        if let Some(got) = split.gather() {
            let want = propagate(&circuit, sum, &WeightCutoff(4), Direction::Forward);
            assert_terms_close(&got, &want, TOL, "cut row policy");
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
        r.compare(
            &circuit,
            &sum,
            &KeepAll,
            Direction::Forward,
            got,
            "propagate_mpi",
        );
    });

    r.case("a one-byte chunk size still delivers", |r| {
        // The pathological end of the same path: every byte its own message.
        // Small input, or the message count explodes.
        let mut circuit = Circuit::<1>::new(6);
        circuit.push(Clifford2Q::cnot(0, 3));
        let sum = rand_sum::<1>(24, 6, 0xA031);
        r.differential(
            backend,
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

    // ---- sampling truncation ------------------------------------------------
    r.case(
        "collapse sample keeps one bounded unit-norm trajectory",
        |r| {
            use paulistrings::test_support::{collapsing_circuit, z0_sum};
            use paulistrings::CollapseSample;
            const CACHE: usize = 6;
            let (circuit, input) = (collapsing_circuit(), z0_sum());

            for seed in 0..4u64 {
                // One policy per process, as a launcher gives each rank its own.
                let policy = CollapseSample::new(CACHE, seed);
                let transport = MpiTransport::from_communicator(r.world);
                let mut split = DistributedSum::scatter(input.clone(), transport, &r.config(SEED))
                    .expect("topology resolves");
                split.propagate(&circuit, &policy, Direction::Heisenberg);
                assert!(split.len() <= CACHE, "seed {seed}: {} terms", split.len());
                let got = split.gather();
                if r.rank == 0 {
                    assert!(
                        policy.collapses() >= 1,
                        "seed {seed}: rank 0 counts collapses"
                    );
                    let got = got.expect("rank 0 gathers");
                    let norm: f64 = got.iter().map(|(_, _, c)| c.norm_sqr()).sum();
                    assert!((norm - 1.0).abs() < 1e-12, "seed {seed}: {norm}");
                    if r.size == 1 {
                        let want = propagate(
                            &circuit,
                            input.clone(),
                            &CollapseSample::new(CACHE, seed),
                            Direction::Heisenberg,
                        );
                        assert_terms_close(&got, &want, 1e-12, "one rank is propagate");
                    }
                } else {
                    assert_eq!(policy.collapses(), 0, "only rank 0 counts");
                }
            }
        },
    );

    // ---- echo read-outs --------------------------------------------------
    r.case("echo read-outs over excluded rows match the oracle", |r| {
        use paulistrings::test_support::rand_sum_on;
        use paulistrings::RotationAxis;

        let circuit = cnot_ring::<1>(10);
        let sum = rand_sum_on::<1>(400, 10, &[0, 2, 3, 5, 6, 9], 0xA060);
        let oracle = propagate(&circuit, sum.clone(), &KeepAll, Direction::Heisenberg);
        let sites = [2usize, 3, 6];
        for axis in [RotationAxis::Z, RotationAxis::X] {
            let policy = PartitionRowPolicy::SeededExcluding {
                seed: Some(SEED),
                exclude_x: vec![2, 3, 6],
                exclude_z: vec![2, 3, 6],
            };
            let transport = MpiTransport::from_communicator(r.world);
            let options = ScatterOptions {
                runtime: PartitionRuntime::new(&r.config(SEED)).expect("topology resolves"),
                rows: ScatterRows::Policy(policy.clone()),
            };
            let mut split = DistributedSum::scatter_with(sum.clone(), transport, options);
            split.propagate(&circuit, &KeepAll, Direction::Heisenberg);
            let got = split.rotated_overlap(&sites, 0.3, axis);
            let want = oracle.rotated_overlap(&sites, 0.3, axis);
            assert!((got - want).abs() < 1e-10, "{axis:?}: {got} vs {want}");
            let hist = split.anticommute_histogram(&sites, axis);
            for (h, w) in hist.iter().zip(oracle.anticommute_histogram(&sites, axis)) {
                assert!((h - w).abs() < 1e-10, "{axis:?}: histogram");
            }
        }
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
        // The scalar goes through the communicator directly — the pattern a
        // caller uses for its own observables.
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

/// The device half: the shared matrix through `MpiGpuSum`, each rank on `local_device_for_comm`, and the device-only cases.
#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use paulistrings::gpu::{
        cuda_available, local_device_for_comm, propagate_mpi_gpu, GpuError, MpiGpuSum,
    };

    /// The number of ranks for which `flag` holds. **Collective.**
    fn count(r: &Runner, flag: bool) -> u64 {
        let mut total = [0u64];
        r.world.all_reduce_into(
            &[u64::from(flag)][..],
            &mut total[..],
            SystemOperation::sum(),
        );
        total[0]
    }

    pub(super) fn run_device_matrix(r: &mut Runner) {
        // Every rank must take the same branch, or the cases below are collectives some ranks never enter.
        let without = count(r, !cuda_available());
        if without > 0 {
            if r.rank == 0 {
                println!(
                    "skipped  device cases ({without} of {} ranks see no CUDA device)",
                    r.size
                );
            }
            return;
        }

        // A group of more than one rank needs NCCL on distinct devices; where the ranks share one, or cannot load NCCL, every rank must fail the scatter alike.
        let device = local_device_for_comm(r.world).expect("a device on every rank");
        let started = MpiGpuSum::<1>::scatter_to_device(
            &rand_sum::<1>(50, 8, 0xB0FF),
            MpiTransport::from_communicator(r.world),
            device,
            &PartitionRowPolicy::Seeded(Some(SEED)),
        )
        .map(|_| ());
        let failed = count(r, started.is_err());
        if failed > 0 {
            r.case("a device group that cannot start NCCL fails on every rank", |r| {
                assert_eq!(failed, u64::from(r.size), "some rank started: {started:?}");
                assert!(
                    matches!(&started, Err(GpuError::Unsupported(m)) if m.contains("share a device") || m.contains("libnccl")),
                    "{started:?}"
                );
            });
            if r.rank == 0 {
                println!("skipped  device cases (the ranks cannot start NCCL: {started:?})");
            }
            return;
        }

        run_matrix(r, Backend::Device);

        r.case(
            "propagate_mpi_gpu is the persistent device driver in one call",
            |r| {
                let circuit = cnot_ring::<1>(10);
                let sum = rand_sum::<1>(300, 10, 0xB015);
                let got = propagate_mpi_gpu(
                    &circuit,
                    &sum,
                    KeepAll,
                    Direction::Forward,
                    PropagateOptions::default(),
                    r.world,
                    None,
                )
                .expect("device run");
                r.compare(
                    &circuit,
                    &sum,
                    &KeepAll,
                    Direction::Forward,
                    got,
                    "propagate_mpi_gpu",
                );
            },
        );

        // Four disjoint dense two-qubit layers: a rank sends each partner several blocks whose row counts differ across layers, so an out-of-order per-partner match would not silently agree.
        r.case("device several distinct-size blocks per partner", |r| {
            let mut circuit = Circuit::<1>::new(12);
            for (q0, q1) in [(0u32, 1u32), (2, 5), (3, 9), (4, 11)] {
                circuit.push(GeneralUnitary2Q::from_matrix(q0, q1, haar_su4_matrix()));
            }
            let sum = rand_sum::<1>(2_000, 12, 0xB031);
            r.both_directions(
                Backend::Device,
                &circuit,
                &sum,
                &KeepAll,
                SEED,
                "device distinct-size blocks",
            );
        });
    }
}
