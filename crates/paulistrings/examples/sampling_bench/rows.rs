//! The partition rows a split run scatters by, and how many channels of one step they leave remote.

use std::collections::VecDeque;

use paulistrings::{PartitionRowPolicy, PartitionRows, ScatterRows};

use crate::cli::Rows;
use crate::models::{IsingQuench, Observable};

/// The scatter rows of a split run over `partitions`, and the channels of one step that cross them (`None` under seeded rows or at one partition).
pub(crate) fn scatter_rows<const W: usize>(
    rows: Rows,
    quench: &IsingQuench,
    partitions: usize,
) -> (ScatterRows<W>, Option<usize>) {
    let bits = partitions.trailing_zeros() as usize;
    let n = quench.num_qubits;
    if partitions == 1 || rows == Rows::Seeded {
        return (ScatterRows::Policy(PartitionRowPolicy::Seeded(None)), None);
    }
    let policy = match rows {
        Rows::Seeded => unreachable!(),
        Rows::Cut => {
            let block_of = |q: usize| q * partitions / n;
            PartitionRowPolicy::Cut(
                (0..partitions).map(|b| (0..n).filter(|&q| block_of(q) == b).collect()).collect(),
            )
        }
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
        Rows::XParity(k) | Rows::ZParity(k) => {
            assert!(bits * k <= n, "--rows {}: {bits} rows of {k} qubits exceed {n}", rows.name());
            let offset = if matches!(rows, Rows::XParity(_)) { 0 } else { n };
            let order = breadth_first(quench, start_qubit(quench));
            PartitionRowPolicy::Matrix(
                order
                    .chunks(k)
                    .take(bits)
                    .map(|group| {
                        let mut row = vec![false; 2 * n];
                        for &q in group {
                            row[offset + q] = true;
                        }
                        row
                    })
                    .collect(),
            )
        }
    };
    let remote = remote_channels_per_step(&policy.rows::<W>(n, bits as u8, 0), quench);
    (ScatterRows::Policy(policy), Some(remote))
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
            .min_by_key(|&q| depths(quench, q).into_iter().max().unwrap_or(0))
            .expect("a lattice has qubits"),
    }
}

/// Every qubit's breadth-first distance from `start`.
fn depths(quench: &IsingQuench, start: usize) -> Vec<usize> {
    breadth_first_with_depth(quench, start).1
}

/// Every qubit in breadth-first order from `start`, neighbours in index order.
fn breadth_first(quench: &IsingQuench, start: usize) -> Vec<usize> {
    breadth_first_with_depth(quench, start).0
}

fn breadth_first_with_depth(quench: &IsingQuench, start: usize) -> (Vec<usize>, Vec<usize>) {
    let mut adjacent = vec![Vec::new(); quench.num_qubits];
    for &(a, b) in &quench.bonds {
        adjacent[a].push(b);
        adjacent[b].push(a);
    }
    let mut depth = vec![usize::MAX; quench.num_qubits];
    let (mut order, mut queue) = (Vec::new(), VecDeque::from([start]));
    depth[start] = 0;
    while let Some(q) = queue.pop_front() {
        order.push(q);
        adjacent[q].sort_unstable();
        for &next in &adjacent[q] {
            if depth[next] == usize::MAX {
                depth[next] = depth[q] + 1;
                queue.push_back(next);
            }
        }
    }
    (order, depth)
}
