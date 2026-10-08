use super::*;

#[test]
fn parse_accepts_ranges_and_singletons() {
    assert_eq!(CpuSet::parse("0-7,16-23").unwrap().0, {
        let mut v: Vec<usize> = (0..8).collect();
        v.extend(16..24);
        v
    });
    assert_eq!(CpuSet::parse("3").unwrap().0, vec![3]);
    assert_eq!(CpuSet::parse("5-5").unwrap().0, vec![5]);
    assert_eq!(CpuSet::parse(" 0-1,\n").unwrap().0, vec![0, 1]);
}

#[test]
fn parse_rejects_malformed_lists() {
    assert!(matches!(CpuSet::parse(""), Err(TopologyError::EmptySet)));
    assert!(matches!(
        CpuSet::parse("a"),
        Err(TopologyError::InvalidCpuList(_))
    ));
    assert!(matches!(
        CpuSet::parse("7-3"),
        Err(TopologyError::InvalidCpuList(_))
    ));
}

#[test]
fn display_compresses_ranges_and_round_trips() {
    let set = CpuSet::parse("0-7,16-23").unwrap();
    assert_eq!(set.to_string(), "0-7,16-23");
    assert_eq!(CpuSet::parse(&set.to_string()).unwrap(), set);
    assert_eq!(CpuSet::parse("3,1,2,2").unwrap().to_string(), "1-3");
    assert_eq!(CpuSet::parse("1,3,5-6").unwrap().to_string(), "1,3,5-6");
}

#[test]
fn intersect_keeps_common_cpus() {
    let a = CpuSet::parse("0-7,16-23").unwrap();
    let b = CpuSet::parse("4-18").unwrap();
    assert_eq!(a.intersect(&b).to_string(), "4-7,16-18");
    assert!(a.intersect(&CpuSet::parse("100").unwrap()).is_empty());
    assert_eq!(a.len(), 16);
}

#[test]
fn numa_nodes_are_disjoint_subsets_of_the_affinity_mask() {
    let allowed = allowed_cpus();
    let nodes = numa_nodes();
    assert!(!nodes.is_empty());
    for (i, (_, set)) in nodes.iter().enumerate() {
        assert!(!set.is_empty());
        assert_eq!(set.intersect(&allowed), *set, "node cpus outside affinity");
        for (_, other) in nodes.iter().skip(i + 1) {
            assert!(set.intersect(other).is_empty(), "nodes overlap");
        }
    }
    let ids: Vec<usize> = nodes.iter().map(|(id, _)| *id).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(ids, sorted);
}

#[test]
fn unpinned_placement_requires_a_power_of_two() {
    let cfg = PartitionConfig {
        placement: Placement::Unpinned {
            partitions: 3,
            threads_per_partition: None,
        },
        ..PartitionConfig::default()
    };
    assert!(matches!(
        cfg.resolve(),
        Err(TopologyError::NotPowerOfTwo(3))
    ));
}

#[test]
fn unpinned_placement_resolves_to_unpinned_slots() {
    let cfg = PartitionConfig {
        placement: Placement::Unpinned {
            partitions: 4,
            threads_per_partition: Some(2),
        },
        ..PartitionConfig::default()
    };
    let slots = cfg.resolve().unwrap();
    assert_eq!(slots.len(), 4);
    for slot in &slots {
        assert_eq!(slot.threads, 2);
        assert!(slot.cpus.is_none());
        assert!(slot.node.is_none());
    }
}

#[test]
fn explicit_placement_validates_its_sets() {
    let one = CpuSet(vec![allowed_cpus().0[0]]);
    let cfg = |sets: Vec<CpuSet>| PartitionConfig {
        placement: Placement::Explicit(sets),
        ..PartitionConfig::default()
    };
    assert!(matches!(
        cfg(vec![one.clone(), one.clone(), one.clone()]).resolve(),
        Err(TopologyError::NotPowerOfTwo(3))
    ));
    assert!(matches!(
        cfg(vec![one.clone(), CpuSet(vec![])]).resolve(),
        Err(TopologyError::EmptySet)
    ));
    assert!(matches!(
        cfg(vec![one, CpuSet(vec![1_000_000])]).resolve(),
        Err(TopologyError::CpuNotAllowed(1_000_000))
    ));
}

#[test]
fn explicit_placement_resolves_disjoint_singletons() {
    let allowed = allowed_cpus();
    if allowed.len() < 2 {
        eprintln!("skipping: fewer than two allowed CPUs ({allowed})");
        return;
    }
    let cfg = PartitionConfig {
        placement: Placement::Explicit(vec![
            CpuSet(vec![allowed.0[0]]),
            CpuSet(vec![allowed.0[1]]),
        ]),
        ..PartitionConfig::default()
    };
    let slots = cfg.resolve().unwrap();
    assert_eq!(slots.len(), 2);
    assert_eq!(slots[0].cpus.as_ref().unwrap().0, vec![allowed.0[0]]);
    assert_eq!(slots[1].cpus.as_ref().unwrap().0, vec![allowed.0[1]]);
    for slot in &slots {
        assert_eq!(slot.threads, 1);
    }
}

#[test]
fn auto_placement_is_a_power_of_two_of_disjoint_sets() {
    let slots = PartitionConfig::default().resolve().unwrap();
    assert!(!slots.is_empty());
    assert!(slots.len().is_power_of_two());
    for (i, slot) in slots.iter().enumerate() {
        let cpus = slot.cpus.as_ref().expect("auto slots are pinned");
        assert!(!cpus.is_empty());
        assert_eq!(slot.threads, cpus.len());
        for other in slots.iter().skip(i + 1) {
            let other = other.cpus.as_ref().unwrap();
            assert!(cpus.intersect(other).is_empty(), "partitions overlap");
        }
    }
}

#[test]
fn auto_placement_honours_max_partitions() {
    let cfg = PartitionConfig {
        placement: Placement::Auto {
            max_partitions: Some(1),
        },
        ..PartitionConfig::default()
    };
    assert_eq!(cfg.resolve().unwrap().len(), 1);
}

#[test]
fn a_pinned_pool_runs_every_worker_on_the_pinned_cpu() {
    let allowed = allowed_cpus();
    if allowed.len() < 2 {
        eprintln!("skipping: fewer than two allowed CPUs ({allowed})");
        return;
    }
    let cpu = allowed.0[0];
    let slot = PartitionSlot {
        cpus: Some(CpuSet(vec![cpu])),
        node: None,
        threads: 2,
        device: None,
    };
    let pool = build_pool(&slot, false, "test-pinned").unwrap();
    let seen: Vec<Option<usize>> = pool.broadcast(|_| current_cpu());
    assert_eq!(seen.len(), 2);
    if cfg!(target_os = "linux") {
        for got in seen {
            assert_eq!(got, Some(cpu));
        }
    }
}

#[test]
fn an_unpinned_pool_builds_and_runs() {
    let slot = PartitionSlot {
        cpus: None,
        node: None,
        threads: 2,
        device: None,
    };
    let pool = build_pool(&slot, false, "test-unpinned").unwrap();
    assert_eq!(pool.install(|| (0..100).sum::<usize>()), 4950);
}

#[test]
fn memory_binding_round_trips_on_a_scratch_thread() {
    // On a scratch thread, leaving the test runner's memory policy alone.
    let node = numa_nodes()[0].0;
    std::thread::scope(|scope| {
        scope.spawn(|| match bind_current_thread_memory(Some(node)) {
            Ok(()) => bind_current_thread_memory(None).unwrap(),
            // A kernel built without NUMA has no `set_mempolicy`.
            Err(err) if err.kind() == io::ErrorKind::Unsupported => {
                eprintln!("skipping: set_mempolicy unavailable ({err})");
            }
            Err(err) => panic!("binding to node {node} failed: {err}"),
        });
    });
}

/// The CPU the calling thread is running on, `None` on non-Linux targets.
fn current_cpu() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `sched_getcpu` takes no arguments and cannot fail beyond returning a negative value.
        let cpu = unsafe { libc::sched_getcpu() };
        usize::try_from(cpu).ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
