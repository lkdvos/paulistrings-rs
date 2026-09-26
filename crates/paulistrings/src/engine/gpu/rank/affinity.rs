//! The collective device pick by CPU locality, [`local_device_for_comm`], over a pure assignment of node-local ranks to devices.

use cudarc::driver::{result, sys};

use super::super::device::device_count;
use super::super::error::GpuError;
use super::LOG_TARGET;
use crate::engine::partitioned::mpi::rsmpi::collective::CommunicatorCollectives;
use crate::engine::partitioned::mpi::rsmpi::topology::Communicator;
use crate::engine::partitioned::topology::{allowed_cpus, sysfs_numa_nodes};

/// Most devices a rank reports; a rank seeing more is matched by the local-rank modulo only.
const MAX_DEVICES: usize = 16;

/// One rank's gathered record: CPU NUMA mask, device count, then `MAX_DEVICES` PCI keys and `MAX_DEVICES` NUMA nodes (`0` unknown, else the value `+ 1`).
const RECORD: usize = 2 + 2 * MAX_DEVICES;

/// The CUDA device this rank should drive, chosen by CPU locality among the ranks of `comm` that share its node.
///
/// **Collective** over `comm`: one `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` and one all-gather over the node.
/// Each rank reports the NUMA nodes its CPU affinity mask touches and the NUMA node of every device it sees (from sysfs by PCI address), and every rank of the node computes the same assignment from the gathered reports.
/// When every rank of the node sees the same devices, each gets a distinct device on one of its NUMA nodes where the matching allows, else a distinct device on another node, and ranks share devices, evenly, only when they outnumber them.
/// When the ranks see different devices (`srun --gpus-per-task`), or the NUMA facts are unreadable (a non-Linux host, no sysfs, `numa_node` of `-1`), the pick is the node-local rank modulo the visible devices, as for [`local_device_for_rank`](super::local_device_for_rank).
/// Logs each rank's pick at INFO, and at WARN when the device sits on a NUMA node the rank's CPUs are not on.
///
/// # Errors
///
/// [`GpuError::NoDevice`] if this rank sees no device, after the collective, so its peers do not hang.
pub fn local_device_for_comm(comm: &impl Communicator) -> Result<u32, GpuError> {
    let node = comm.split_shared(comm.rank());
    let local = node.rank() as usize;
    let count = device_count();
    let mine = own_report(count).encode();
    let mut all = vec![0u64; RECORD * node.size() as usize];
    node.all_gather_into(&mine[..], &mut all[..]);
    if count == 0 {
        return Err(GpuError::NoDevice);
    }
    let reports: Vec<Report> = all.chunks_exact(RECORD).map(Report::decode).collect();
    let pick = pick_for(local, &reports);
    let me = &reports[local];
    let device_numa = me.numa.get(pick as usize).copied().flatten();
    let where_ = |numa: Option<u32>| numa.map_or_else(|| "unknown".to_string(), |n| n.to_string());
    log::info!(
        target: LOG_TARGET,
        "gpu device pick: rank {} (node-local {local} of {}, CPU NUMA {}) drives device {pick} (NUMA {})",
        comm.rank(),
        node.size(),
        mask_list(me.cpu_numa),
        where_(device_numa),
    );
    if device_numa.is_some_and(|n| me.cpu_numa != 0 && !on_node(me.cpu_numa, n)) {
        log::warn!(
            target: LOG_TARGET,
            "gpu device pick: rank {} drives device {pick} on NUMA node {}, but its CPUs are on NUMA {}, so its host staging crosses sockets",
            comm.rank(),
            where_(device_numa),
            mask_list(me.cpu_numa),
        );
    }
    Ok(pick)
}

/// What one rank knows about its CPUs and devices.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Report {
    /// Bit `n` set when the affinity mask meets NUMA node `n`; `0` unknown.
    cpu_numa: u64,
    /// Visible device count.
    count: usize,
    /// Each device's PCI key (`domain << 16 | bus << 8 | device`), `None` unknown; at most `MAX_DEVICES` entries.
    pci: Vec<Option<u64>>,
    /// Each device's NUMA node, `None` unknown; as long as `pci`.
    numa: Vec<Option<u32>>,
}

impl Report {
    fn encode(&self) -> [u64; RECORD] {
        let mut out = [0u64; RECORD];
        out[0] = self.cpu_numa;
        out[1] = self.count as u64;
        for (i, (pci, numa)) in self.pci.iter().zip(&self.numa).enumerate() {
            out[2 + i] = pci.map_or(0, |k| k + 1);
            out[2 + MAX_DEVICES + i] = numa.map_or(0, |n| u64::from(n) + 1);
        }
        out
    }

    fn decode(record: &[u64]) -> Self {
        let count = record[1] as usize;
        let shown = count.min(MAX_DEVICES);
        let opt = |v: u64| v.checked_sub(1);
        Self {
            cpu_numa: record[0],
            count,
            pci: record[2..2 + shown].iter().map(|&v| opt(v)).collect(),
            numa: record[2 + MAX_DEVICES..2 + MAX_DEVICES + shown]
                .iter()
                .map(|&v| opt(v).and_then(|n| u32::try_from(n).ok()))
                .collect(),
        }
    }

    /// The device list, when it identifies every device.
    fn devices(&self) -> Option<&[Option<u64>]> {
        (self.count <= MAX_DEVICES && self.pci.iter().all(Option::is_some)).then_some(&self.pci)
    }
}

/// This rank's report over its `count` visible devices.
fn own_report(count: usize) -> Report {
    let cpu_numa = sysfs_numa_nodes(&allowed_cpus())
        .iter()
        .try_fold(0u64, |mask, (node, _)| {
            (*node < 64).then(|| mask | 1 << node)
        })
        .unwrap_or(0);
    let shown = count.min(MAX_DEVICES);
    let pci: Vec<Option<(u32, u32, u32)>> = (0..shown as u32).map(device_pci).collect();
    Report {
        cpu_numa,
        count,
        pci: pci
            .iter()
            .map(|p| p.map(|(d, b, s)| u64::from(d) << 16 | u64::from(b) << 8 | u64::from(s)))
            .collect(),
        numa: pci
            .iter()
            .map(|p| p.and_then(|(d, b, s)| pci_numa_node(&pci_sysfs_name(d, b, s))))
            .collect(),
    }
}

/// Local rank `local`'s device from the node's gathered reports.
fn pick_for(local: usize, reports: &[Report]) -> u32 {
    let me = &reports[local];
    let shared = me
        .devices()
        .is_some_and(|mine| reports.iter().all(|r| r.devices() == Some(mine)));
    if shared {
        let masks: Vec<u64> = reports.iter().map(|r| r.cpu_numa).collect();
        assign_devices(&masks, &me.numa)[local]
    } else {
        (local % me.count) as u32
    }
}

/// Whether mask `mask` has NUMA node `node`.
fn on_node(mask: u64, node: u32) -> bool {
    node < 64 && mask >> node & 1 == 1
}

/// `mask`'s nodes as a comma list, `unknown` for `0`.
fn mask_list(mask: u64) -> String {
    if mask == 0 {
        return "unknown".to_string();
    }
    (0..64)
        .filter(|&n| on_node(mask, n))
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Each rank's device, given each rank's CPU NUMA mask (`0` unknown) in node-local rank order and each device's NUMA node.
///
/// Every device offers `ceil(ranks / devices)` slots, filled round by round.
/// A maximum matching (augmenting paths, ranks in order) puts as many ranks as possible on a slot of a device on one of their NUMA nodes; every other rank takes the least-loaded device with a free slot, lowest ordinal first.
/// With no NUMA facts at all this is local rank `i` on device `i % devices`.
fn assign_devices(ranks: &[u64], devices: &[Option<u32>]) -> Vec<u32> {
    let k = devices.len();
    assert!(k > 0, "assign_devices needs a device");
    let slots = k * ranks.len().div_ceil(k).max(1);
    let local =
        |rank: usize, slot: usize| devices[slot % k].is_some_and(|n| on_node(ranks[rank], n));
    let mut owner: Vec<Option<usize>> = vec![None; slots];

    fn augment(
        rank: usize,
        seen: &mut [bool],
        owner: &mut [Option<usize>],
        local: &dyn Fn(usize, usize) -> bool,
    ) -> bool {
        for slot in 0..owner.len() {
            if seen[slot] || !local(rank, slot) {
                continue;
            }
            seen[slot] = true;
            if owner[slot].is_none_or(|other| augment(other, seen, owner, local)) {
                owner[slot] = Some(rank);
                return true;
            }
        }
        false
    }

    for rank in 0..ranks.len() {
        if let Some(slot) = (0..slots).find(|&s| owner[s].is_none() && local(rank, s)) {
            owner[slot] = Some(rank);
        } else {
            augment(rank, &mut vec![false; slots], &mut owner, &local);
        }
    }

    let mut pick: Vec<Option<u32>> = vec![None; ranks.len()];
    for (slot, rank) in owner.iter().enumerate() {
        if let Some(rank) = *rank {
            pick[rank] = Some((slot % k) as u32);
        }
    }
    let cap = slots / k;
    let mut load = vec![0usize; k];
    for &d in pick.iter().flatten() {
        load[d as usize] += 1;
    }
    for p in pick.iter_mut().filter(|p| p.is_none()) {
        let d = (0..k)
            .filter(|&d| load[d] < cap)
            .min_by_key(|&d| load[d])
            .expect("the slots cover every rank");
        load[d] += 1;
        *p = Some(d as u32);
    }
    pick.into_iter()
        .map(|p| p.expect("every rank placed"))
        .collect()
}

/// Device `ordinal`'s PCI `(domain, bus, device)`, `None` when the driver cannot say.
fn device_pci(ordinal: u32) -> Option<(u32, u32, u32)> {
    use sys::CUdevice_attribute_enum as A;
    // SAFETY: `is_culib_present` only probes the library; `result::*` is reached only when it is present.
    if !unsafe { sys::is_culib_present() } {
        return None;
    }
    result::init().ok()?;
    let dev = result::device::get(i32::try_from(ordinal).ok()?).ok()?;
    // SAFETY: `dev` came from `result::device::get`.
    let attr = |a| {
        unsafe { result::device::get_attribute(dev, a) }
            .ok()
            .and_then(|v| u32::try_from(v).ok())
    };
    Some((
        attr(A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
        attr(A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
        attr(A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?,
    ))
}

/// The sysfs name of a PCI function-0 device, `0000:18:00.0`.
fn pci_sysfs_name(domain: u32, bus: u32, device: u32) -> String {
    format!("{domain:04x}:{bus:02x}:{device:02x}.0")
}

/// The NUMA node sysfs gives the PCI device `name`, `None` when unreadable or `-1`.
fn pci_numa_node(name: &str) -> Option<u32> {
    let text = std::fs::read_to_string(format!("/sys/bus/pci/devices/{name}/numa_node")).ok()?;
    parse_numa_node(&text)
}

fn parse_numa_node(text: &str) -> Option<u32> {
    text.trim()
        .parse::<i64>()
        .ok()
        .and_then(|n| u32::try_from(n).ok())
}

#[cfg(test)]
mod tests {
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
}
