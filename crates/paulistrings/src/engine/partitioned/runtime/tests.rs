use super::*;
use crate::engine::partitioned::topology::{allowed_cpus, Placement};
use crate::engine::partitioned::transport::Collectives;

fn config(partitions: usize) -> PartitionConfig {
    PartitionConfig {
        placement: Placement::Unpinned {
            partitions,
            threads_per_partition: Some(2),
        },
        bind_memory: false,
        partition_row_seed: None,
    }
}

#[test]
fn partition_bits_is_log2_of_the_count() {
    for (p, bits) in [(1usize, 0u8), (2, 1), (4, 2), (8, 3)] {
        let runtime = PartitionRuntime::new(&config(p)).expect("resolve");
        assert_eq!(runtime.num_partitions(), p);
        assert_eq!(runtime.partition_bits(), bits);
    }
}

/// The pool-width override replaces the resolved worker count on every slot and leaves the CPU sets alone.
#[test]
fn with_threads_per_partition_overrides_the_resolved_width() {
    let cpus = allowed_cpus();
    let placement = Placement::Explicit(vec![cpus.clone(), cpus.clone()]);
    let config = PartitionConfig {
        placement,
        bind_memory: false,
        partition_row_seed: None,
    };

    let resolved = PartitionRuntime::new(&config).expect("resolve");
    for slot in resolved.slots() {
        assert_eq!(slot.threads, cpus.len());
    }

    let narrowed = PartitionRuntime::with_threads_per_partition(&config, Some(1))
        .expect("resolve with an explicit width");
    for slot in narrowed.slots() {
        assert_eq!(slot.threads, 1);
        assert_eq!(slot.cpus.as_ref(), Some(&cpus), "the CPU set is untouched");
    }
}

#[test]
fn map_partitions_returns_results_in_rank_order() {
    let runtime = PartitionRuntime::new(&config(4)).expect("resolve");
    let got = runtime.map_partitions(vec![10usize, 20, 30, 40], |rank, item, transport| {
        assert_eq!(transport.rank() as usize, rank);
        assert_eq!(transport.size(), 4);
        // Inside the partition's own pool.
        assert!(rayon::current_thread_index().is_some());
        (rank, item)
    });
    assert_eq!(got, vec![(0, 10), (1, 20), (2, 30), (3, 40)]);
}

#[test]
fn map_partitions_runs_collectives_between_partitions() {
    let runtime = PartitionRuntime::new(&config(4)).expect("resolve");
    let got = runtime.map_partitions(vec![3u8, 7, 1, 5], |_, item, transport| {
        transport.allreduce_max_u8(item)
    });
    assert_eq!(got, vec![7, 7, 7, 7]);
}

/// A partition that panics while its partners are inside a collective surfaces as a panic (which one is a race), not a hang.
#[test]
#[should_panic]
fn a_panicking_partition_does_not_hang_the_group() {
    let runtime = PartitionRuntime::new(&config(4)).expect("resolve");
    runtime.map_partitions(vec![0usize; 4], |rank, _, transport| {
        if rank == 2 {
            panic!("rank 2 fell over");
        }
        for _ in 0..4 {
            transport.allreduce_max_u8(rank as u8);
        }
    });
}

#[test]
fn placement_summary_names_every_slot() {
    let runtime = PartitionRuntime::new(&config(2)).expect("resolve");
    let summary = runtime.placement_summary();
    assert!(summary.contains("0:"), "{summary}");
    assert!(summary.contains("1:"), "{summary}");
}
