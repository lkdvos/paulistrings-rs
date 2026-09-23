//! Behavior of the `phase-timing` counters themselves. The *non-perturbation*
//! guarantee is not tested here — it is the whole existing suite (fingerprint
//! net, thread-count/bucket-count bitwise identity, capacity stability)
//! passing under `--features phase-timing`.
#![cfg(feature = "phase-timing")]

use paulistrings::channel::{Depolarizing, PauliRotation};
use paulistrings::engine::partitioned::{
    DistributedSum, InProcessTransport, PartitionRuntime, PartitionedSum,
};
// `rand_sum_real::<1>` — at `W = 1` its per-word masking loop reduces to the
// single `(1 << num_qubits) - 1` mask, and the draw order (`x`, `z`, `re`)
// matches the other propagation test files' fixtures.
use paulistrings::test_support::{rand_sum_real, unpinned_partitions, zz_rotation, KeepAll};
use paulistrings::{
    propagate_with_scratch, Circuit, Direction, LayerScratch, PartitionRows, PauliString,
    PhaseStats,
};

#[test]
fn stats_sum_approximates_total() {
    // 20k dense terms over 32 qubits → well past the single-bucket regime, so
    // the coset machinery actually runs. One-thread pool so worker busy time
    // and coset-loop wall time are the same clock domain.
    let mut circuit = Circuit::<1>::new(32);
    for q in 0..4 {
        circuit.push(zz_rotation::<1>(q, q + 1, 0.13 + q as f64 * 0.05));
    }
    let sum = rand_sum_real::<1>(20_000, 32, 0x51A75);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    let mut scratch = LayerScratch::<1>::new();
    let out = pool.install(|| {
        propagate_with_scratch(&circuit, sum, &KeepAll, Direction::Heisenberg, &mut scratch)
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
    circuit.push(zz_rotation::<1>(0, 1, 0.4));
    let sum = rand_sum_real::<1>(5_000, 16, 0xD1CE);

    let mut scratch = LayerScratch::<1>::new();
    let _ = propagate_with_scratch(&circuit, sum, &KeepAll, Direction::Forward, &mut scratch);

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
    let sum = rand_sum_real::<1>(5_000, 16, 0xACE);

    let mut scratch = LayerScratch::<1>::new();
    let _ = propagate_with_scratch(&circuit, sum, &KeepAll, Direction::Forward, &mut scratch);

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

/// The partitioned driver's counters: one breakdown per partition with the
/// collective, export and exchange phases attributed, plus the driver's own
/// scatter and gather.
#[test]
fn partitioned_stats_are_attributed() {
    // A weight-4 rotation whose generator carries `X` on qubit 0, and one
    // partition row seeing exactly that bit — so the generator pass is remote
    // and every layer exports.
    let gen = PauliString::<1> {
        x: [0b0101],
        z: [0b1010],
    };
    let mut circuit = Circuit::<1>::new(16);
    for _ in 0..3 {
        circuit.push(PauliRotation::new(gen, 0.3));
    }
    let sum = rand_sum_real::<1>(20_000, 16, 0x9A17);

    let config = unpinned_partitions(2, 1, 0x51A75);
    let runtime = PartitionRuntime::new(&config).expect("topology resolves");
    let rows = PartitionRows::<1>::from_rows(16, vec![[1u64]], vec![[0u64]]);
    let mut split = PartitionedSum::scatter_with_rows(sum, rows, runtime);

    let started = std::time::Instant::now();
    split.propagate(&circuit, &KeepAll, Direction::Forward);
    let wall = started.elapsed().as_nanos() as u64;

    let stats = split.take_stats();
    assert_eq!(stats.layers, 3);
    assert_eq!(stats.per_partition.len(), 2);
    assert!(stats.scatter_ns > 0, "{stats:?}");
    assert_eq!(stats.gather_ns, 0, "nothing gathered yet: {stats:?}");

    for (rank, p) in stats.per_partition.iter().enumerate() {
        assert_eq!(p.layers, 3, "partition {rank}: {p:?}");
        assert!(p.collective_ns > 0, "partition {rank}: {p:?}");
        assert!(p.export_ns > 0, "partition {rank}: {p:?}");
        assert!(p.exchange_ns > 0, "partition {rank}: {p:?}");
        assert!(p.rows_exported > 0, "partition {rank}: {p:?}");
        assert!(p.recv_rows > 0, "partition {rank}: {p:?}");
        assert!(p.coset_loop_ns > 0, "partition {rank}: {p:?}");
        // The wall-clock phases are measured inside the call, on this
        // partition's driving thread, so they are bounded by the call itself —
        // and, with real work in every layer, they account for most of it.
        assert!(
            p.wall_total_ns() <= wall,
            "partition {rank}: phases {} exceed the call's {wall} ns",
            p.wall_total_ns(),
        );
        assert!(
            p.wall_total_ns() * 4 >= wall,
            "partition {rank}: phases {} are less than a quarter of the call's {wall} ns",
            p.wall_total_ns(),
        );
        // And the exchange's own phases fit inside them.
        assert!(p.export_ns + p.exchange_ns + p.coset_loop_ns <= p.wall_total_ns());
    }

    // Drained, and `gather` is timed too.
    let drained = split.take_stats();
    assert_eq!(drained.layers, 0);
    assert_eq!(drained.per_partition.len(), 2);
    assert_eq!(drained.per_partition[0], PhaseStats::default());
    let _ = split.gather();
    assert!(split.take_stats().gather_ns > 0);
}

/// The distributed driver's counters, over the in-process transport: one breakdown per rank with the collective, export and exchange phases attributed, plus the rank's own scatter and gather.
#[test]
fn distributed_stats_are_attributed() {
    // The same remote-generator circuit and partition row as `partitioned_stats_are_attributed`.
    let gen = PauliString::<1> {
        x: [0b0101],
        z: [0b1010],
    };
    let mut circuit = Circuit::<1>::new(16);
    for _ in 0..3 {
        circuit.push(PauliRotation::new(gen, 0.3));
    }
    let sum = rand_sum_real::<1>(20_000, 16, 0x9A17);

    let per_rank = std::thread::scope(|scope| {
        let handles: Vec<_> = InProcessTransport::group(2)
            .into_iter()
            .map(|transport| {
                let (circuit, sum) = (&circuit, sum.clone());
                scope.spawn(move || {
                    let config = unpinned_partitions(1, 1, 0x51A75);
                    let runtime = PartitionRuntime::new(&config).expect("topology resolves");
                    let rows = PartitionRows::<1>::from_rows(16, vec![[1u64]], vec![[0u64]]);
                    let mut split =
                        DistributedSum::scatter_with_rows(sum, transport, runtime, rows);

                    let started = std::time::Instant::now();
                    split.propagate(circuit, &KeepAll, Direction::Forward);
                    let wall = started.elapsed().as_nanos() as u64;
                    let stats = split.take_stats();

                    let drained = split.take_stats();
                    let _ = split.gather();
                    let gathered = split.take_stats();
                    (split.rank(), wall, stats, drained, gathered)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("rank thread"))
            .collect::<Vec<_>>()
    });

    assert_eq!(per_rank.len(), 2);
    for (rank, wall, stats, drained, gathered) in per_rank {
        assert_eq!(stats.layers, 3, "rank {rank}: {stats:?}");
        assert_eq!(stats.per_partition.len(), 1, "rank {rank}: {stats:?}");
        assert!(stats.scatter_ns > 0, "rank {rank}: {stats:?}");
        assert_eq!(
            stats.gather_ns, 0,
            "rank {rank}: nothing gathered yet: {stats:?}"
        );

        let p = &stats.per_partition[0];
        assert_eq!(p.layers, 3, "rank {rank}: {p:?}");
        assert!(p.collective_ns > 0, "rank {rank}: {p:?}");
        assert!(p.export_ns > 0, "rank {rank}: {p:?}");
        assert!(p.exchange_ns > 0, "rank {rank}: {p:?}");
        assert!(p.rows_exported > 0, "rank {rank}: {p:?}");
        assert!(p.recv_rows > 0, "rank {rank}: {p:?}");
        assert!(p.coset_loop_ns > 0, "rank {rank}: {p:?}");
        assert!(p.terms_in > 0 && p.terms_out > 0, "rank {rank}: {p:?}");
        assert!(
            p.wall_total_ns() <= wall,
            "rank {rank}: phases {} exceed the call's {wall} ns",
            p.wall_total_ns(),
        );

        assert_eq!(drained.layers, 0, "rank {rank}: {drained:?}");
        assert_eq!(drained.per_partition, vec![PhaseStats::default()]);
        assert_eq!(drained.scatter_ns, 0, "rank {rank}: {drained:?}");
        assert!(gathered.gather_ns > 0, "rank {rank}: {gathered:?}");
    }
}

/// The unpartitioned engine leaves the partitioned counters at zero.
#[test]
fn the_unpartitioned_engine_reports_no_exchange() {
    let mut circuit = Circuit::<1>::new(16);
    circuit.push(zz_rotation::<1>(0, 1, 0.4));
    let sum = rand_sum_real::<1>(5_000, 16, 0xD1CE);

    let mut scratch = LayerScratch::<1>::new();
    let _ = propagate_with_scratch(&circuit, sum, &KeepAll, Direction::Forward, &mut scratch);

    let stats = scratch.take_stats();
    assert_eq!(stats.collective_ns, 0, "{stats:?}");
    assert_eq!(stats.export_ns, 0, "{stats:?}");
    assert_eq!(stats.exchange_ns, 0, "{stats:?}");
    assert_eq!(stats.rows_exported, 0, "{stats:?}");
    assert_eq!(stats.recv_rows, 0, "{stats:?}");
}
