use super::*;

const N0: u64 = 1;
const N1: u64 = 2;
/// The 4xA100 node: devices 0-1 on NUMA 0, 2-3 on NUMA 1.
const A100: [Option<u32>; 4] = [Some(0), Some(0), Some(1), Some(1)];

#[test]
fn ranks_bound_cyclically_across_sockets_get_a_distinct_local_device() {
    assert_eq!(assign_devices(&[N0, N1, N0, N1], &A100), [0, 2, 1, 3]);
}

#[test]
fn ranks_bound_in_blocks_get_the_modulo_mapping() {
    assert_eq!(assign_devices(&[N0, N0, N1, N1], &A100), [0, 1, 2, 3]);
}

#[test]
fn ranks_all_on_one_socket_fill_its_devices_then_take_distinct_remote_ones() {
    assert_eq!(assign_devices(&[N1, N1, N1, N1], &A100), [2, 3, 0, 1]);
    assert_eq!(assign_devices(&[N0, N0], &A100), [0, 1]);
}

#[test]
fn more_ranks_than_devices_share_evenly_and_locally() {
    assert_eq!(
        assign_devices(&[N0, N1, N0, N1, N0, N1, N0, N1], &A100),
        [0, 2, 1, 3, 0, 2, 1, 3]
    );
    assert_eq!(
        assign_devices(&[N0, N0, N0], &[Some(0), Some(1)]),
        [0, 0, 1]
    );
}

#[test]
fn unknown_numa_is_the_local_rank_modulo_the_devices() {
    assert_eq!(assign_devices(&[0, 0, 0], &[None, None]), [0, 1, 0]);
    assert_eq!(assign_devices(&[N0, N1, N0], &[None, None]), [0, 1, 0]);
    assert_eq!(assign_devices(&[0, 0, 0, 0], &A100), [0, 1, 2, 3]);
}

#[test]
fn one_device_is_shared_by_every_rank() {
    assert_eq!(assign_devices(&[N0], &[Some(1)]), [0]);
    assert_eq!(assign_devices(&[N0, N1], &[Some(0)]), [0, 0]);
    assert_eq!(assign_devices(&[0, 0], &[None]), [0, 0]);
}

#[test]
fn an_unbound_rank_yields_its_device_to_a_bound_one() {
    assert_eq!(assign_devices(&[N0 | N1, N0], &[Some(0), Some(1)]), [1, 0]);
}

fn report(cpu_numa: u64, pci: &[u64], numa: &[Option<u32>]) -> Report {
    Report {
        cpu_numa,
        count: pci.len(),
        pci: pci.iter().map(|&k| Some(k)).collect(),
        numa: numa.to_vec(),
    }
}

#[test]
fn a_report_round_trips_through_its_record() {
    let r = report(N1, &[0x18 << 8, 0x3b << 8], &[Some(0), None]);
    assert_eq!(Report::decode(&r.encode()), r);
}

#[test]
fn ranks_seeing_the_same_devices_are_matched_by_numa() {
    let pci = [0x07 << 8, 0x0b << 8, 0x48 << 8, 0x4c << 8];
    let node: Vec<Report> = [N0, N1, N0, N1]
        .iter()
        .map(|&m| report(m, &pci, &A100))
        .collect();
    let picks: Vec<u32> = (0..4).map(|l| pick_for(l, &node)).collect();
    assert_eq!(picks, [0, 2, 1, 3]);
}

#[test]
fn ranks_seeing_different_devices_take_the_modulo_of_their_own() {
    let node = [
        report(N1, &[0x07 << 8], &[Some(0)]),
        report(N0, &[0x48 << 8], &[Some(1)]),
    ];
    assert_eq!(pick_for(0, &node), 0);
    assert_eq!(pick_for(1, &node), 0);
}

#[test]
fn the_pci_name_is_the_sysfs_form() {
    assert_eq!(pci_sysfs_name(0, 0x18, 0), "0000:18:00.0");
    assert_eq!(pci_sysfs_name(0x10000, 0xc1, 0x1f), "10000:c1:1f.0");
}

#[test]
fn a_numa_node_of_minus_one_is_unknown() {
    assert_eq!(parse_numa_node("1\n"), Some(1));
    assert_eq!(parse_numa_node("-1\n"), None);
    assert_eq!(parse_numa_node(""), None);
}

#[test]
fn a_missing_pci_device_has_no_numa_node() {
    assert_eq!(pci_numa_node("ffff:ff:1f.0"), None);
}

/// Every visible device resolves to a PCI address, and its sysfs NUMA node is either a node of this host or unknown.
#[test]
fn the_visible_devices_have_a_pci_address_and_a_valid_numa_node() {
    crate::require_cuda!();
    let nodes: Vec<u32> = sysfs_numa_nodes(&allowed_cpus())
        .iter()
        .map(|(n, _)| *n as u32)
        .collect();
    let r = own_report(device_count());
    assert_eq!(r.pci.len(), device_count().min(MAX_DEVICES));
    for (pci, numa) in r.pci.iter().zip(&r.numa) {
        assert!(pci.is_some(), "a visible device has a PCI address");
        if let Some(n) = numa {
            assert!(
                nodes.is_empty() || nodes.contains(n),
                "device NUMA {n} is not in {nodes:?}"
            );
        }
    }
}
