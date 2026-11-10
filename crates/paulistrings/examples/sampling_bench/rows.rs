//! The partition rows a split run scatters by, and how many channels of one step they leave remote.

use std::collections::VecDeque;

use paulistrings::{
    Circuit, Coefficient, PartitionRowPolicy, PartitionRows, PauliAxis, PauliSum, ScatterRows,
};

use crate::cli::Rows;
use crate::models::{IsingQuench, Observable};

/// What a split run scatters by: `label` names the rows (`auto(...)` with the choice it resolved to), `remote_channels` counts the channels of one step that cross them (`None` under seeded rows or at one partition).
pub(crate) struct RowsSummary {
    pub(crate) label: String,
    pub(crate) remote_channels: Option<usize>,
}

/// The scatter rows of a split run of `step` (one Trotter step) on `observable` over `partitions`.
pub(crate) fn scatter_rows<const W: usize, C: Coefficient>(
    rows: Rows,
    quench: &IsingQuench,
    observable: &PauliSum<W, C>,
    step: &Circuit<W>,
    partitions: usize,
) -> (ScatterRows<W>, RowsSummary) {
    let bits = partitions.trailing_zeros() as usize;
    let n = quench.num_qubits;
    let around_start = |axis| {
        PartitionRowPolicy::parity_around(
            axis,
            n,
            &quench.bonds,
            &[start_qubit(quench)],
            rows.group(),
        )
    };
    let policy = match rows {
        Rows::Auto => PartitionRowPolicy::for_run(observable, step),
        Rows::Seeded => PartitionRowPolicy::Seeded(None),
        Rows::Cut => PartitionRowPolicy::blocks(PauliAxis::Z, (0..n).collect()),
        Rows::Quadrant => {
            assert_eq!(partitions, 4, "--rows quadrant splits into exactly four partitions");
            // Labels 0..3 as (x ≥ x0) + 2 (y ≥ y0): each row, and their sum, halves the plane through the site, so the conserved total z-parity fixes none of them.
            let (y0, x0) = quench.positions[start_qubit(quench)];
            let block_of = |q: usize| {
                let (y, x) = quench.positions[q];
                usize::from(x >= x0) + 2 * usize::from(y >= y0)
            };
            PartitionRowPolicy::Cut(
                (0..partitions).map(|b| (0..n).filter(|&q| block_of(q) == b).collect()).collect(),
            )
        }
        Rows::XParity(_) => around_start(PauliAxis::X),
        Rows::ZParity(_) => around_start(PauliAxis::Z),
    };
    let resolved = match (&policy, rows) {
        (PartitionRowPolicy::Seeded(_), _) => "seeded".to_string(),
        (PartitionRowPolicy::Parity { axis, group, .. }, _) => {
            format!("{}parity:{group}", axis_name(*axis))
        }
        (PartitionRowPolicy::Blocks { axis: PauliAxis::Z, .. }, _) => "cut".to_string(),
        (PartitionRowPolicy::Blocks { axis, .. }, _) => format!("{}blocks", axis_name(*axis)),
        (_, rows) => rows.name(),
    };
    let label = if rows == Rows::Auto { format!("auto({resolved})") } else { resolved };
    let remote_channels = (partitions > 1 && !matches!(policy, PartitionRowPolicy::Seeded(_)))
        .then(|| remote_channels_per_step(&policy.rows::<W>(n, bits as u8, 0), quench));
    (ScatterRows::Policy(policy), RowsSummary { label, remote_channels })
}

fn axis_name(axis: PauliAxis) -> &'static str {
    match axis {
        PauliAxis::X => "x",
        PauliAxis::Y => "y",
        PauliAxis::Z => "z",
    }
}

/// The channels of one Trotter step whose generator some row reads an odd number of bits of.
fn remote_channels_per_step<const W: usize>(
    rows: &PartitionRows<W>,
    quench: &IsingQuench,
) -> usize {
    let (rows_x, rows_z) = rows.rows();
    let bit = |mask: &[u64; W], q: usize| (mask[q / 64] >> (q % 64)) & 1 == 1;
    let kicks = (0..quench.num_qubits).filter(|&q| rows_x.iter().any(|r| bit(r, q))).count();
    let bonds = quench
        .bonds
        .iter()
        .filter(|&&(a, b)| rows_z.iter().any(|r| bit(r, a) != bit(r, b)))
        .count();
    kicks * usize::from(quench.order) + bonds
}

/// The observable's site, or the lattice's centre (least eccentricity, lowest index first) for a sum over all sites.
fn start_qubit(quench: &IsingQuench) -> usize {
    match quench.observable {
        Observable::Site(q) => q,
        Observable::Magnetization => (0..quench.num_qubits)
            .min_by_key(|&q| eccentricity(quench, q))
            .expect("a lattice has qubits"),
    }
}

/// The largest breadth-first distance from `start` over the bonds.
fn eccentricity(quench: &IsingQuench, start: usize) -> usize {
    let mut adjacent = vec![Vec::new(); quench.num_qubits];
    for &(a, b) in &quench.bonds {
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let mut depth = vec![usize::MAX; quench.num_qubits];
    let mut queue = VecDeque::from([start]);
    depth[start] = 0;
    while let Some(q) = queue.pop_front() {
        for &next in &adjacent[q] {
            if depth[next] == usize::MAX {
                depth[next] = depth[q] + 1;
                queue.push_back(next);
            }
        }
    }
    depth.into_iter().filter(|&d| d != usize::MAX).max().unwrap_or(0)
}
