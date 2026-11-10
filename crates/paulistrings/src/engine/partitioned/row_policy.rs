//! [`PartitionRowPolicy`], how a scatter picks the partition rows it splits a sum by, and [`PartitionRowPolicy::for_run`], the choice a circuit and its observable suggest.

use std::collections::VecDeque;

use crate::circuit::Circuit;
use crate::coefficient::Coefficient;
use crate::pauli_string::qubit_mask;
use crate::pauli_sum::hash::PartitionRows;
use crate::pauli_sum::PauliSum;
use crate::readout::PauliAxis;

/// The qubits per row [`PartitionRowPolicy::for_run`] gives parity rows, the best balance on four Blackwell GPUs (`research/FINDINGS.md` §X-parity rows on device groups).
pub const FOR_RUN_PARITY_GROUP: usize = 2;

/// Which partition rows a scatter splits by: a seeded GF(2)-random draw, a qubit cut that keeps a channel local when its qubits share a block, parity rows near a site, or explicit rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartitionRowPolicy {
    /// [`PartitionRows::from_seed`] with this seed, or the sum's own hash seed at `None`; the default.
    Seeded(Option<u64>),
    /// [`PartitionRows::cut`]: one disjoint qubit block per rank, in rank order; the rows are z-only.
    Cut(Vec<Vec<usize>>),
    /// [`PartitionRows::from_seed_excluding`]: a seeded draw whose rows read neither the x-bits of `exclude_x` nor the z-bits of `exclude_z`, so keys differing only there share a partition.
    SeededExcluding {
        /// As in [`Seeded`](Self::Seeded).
        seed: Option<u64>,
        /// Qubits whose x-bit no row reads.
        exclude_x: Vec<usize>,
        /// Qubits whose z-bit no row reads.
        exclude_z: Vec<usize>,
    },
    /// [`PartitionRows::from_matrix`]: one row per partition bit as a flag matrix over the `2 · num_qubits` coordinates, x-bits first.
    Matrix(Vec<Vec<bool>>),
    /// [`parity`](Self::parity): row `i` reads the `axis` bits of the qubits `order[i · group..(i + 1) · group]`.
    Parity {
        /// Which bits a row reads: x-bits for `X`, z-bits for `Z`, both for `Y`.
        axis: PauliAxis,
        /// The qubits, nearest the observable first; the rows read its first `log2(P) · group`.
        order: Vec<usize>,
        /// Qubits per row.
        group: usize,
    },
    /// [`blocks`](Self::blocks): `order` cut into `P` contiguous runs, position `i` in run `⌊i · P / len⌋`, labelled as [`Cut`](Self::Cut) labels its blocks, the rows reading `axis` bits.
    Blocks {
        /// Which bits a row reads, as in [`Parity`](Self::Parity).
        axis: PauliAxis,
        /// Every qubit at most once; run `b` is partition `b`'s block.
        order: Vec<usize>,
    },
}

impl PartitionRowPolicy {
    /// Parity rows: row `i` is the parity of the `axis` bits of the `i`-th `group` qubits of `order`, so a channel moves a term between partitions exactly when one of its key shifts sets an odd number of those bits in some row's group.
    ///
    /// `X` rows read x-bits only, so a channel whose shifts set no x-bit (`R_Z`, `R_ZZ`, any diagonal gate or dephasing) is local whatever its qubits, and only a channel setting x-bits on the rows' `log2(P) · group` qubits is remote; `Z` rows mirror that with `R_X` and `R_XX` local; `Y` rows read both bits, which keeps exactly the `Y`-type shifts local.
    /// They balance an observable on few qubits whose terms grow outward from `order`'s head, since every term then has content on the rows' qubits; an observable spread over the register leaves most terms with none there and in partition 0 (`research/FINDINGS.md` §Parity rows balance a local observable, block rows an extensive one).
    ///
    /// # Panics
    ///
    /// When resolved, if `group` is zero, if the rows need more qubits than `order` holds, or if `order` repeats a qubit or names one outside the register.
    pub fn parity(axis: PauliAxis, order: Vec<usize>, group: usize) -> Self {
        Self::Parity { axis, order, group }
    }

    /// [`parity`](Self::parity) rows over the qubits nearest `sites` on the interaction graph `edges`, in [`nearest_first`](Self::nearest_first) order.
    ///
    /// # Panics
    ///
    /// As [`nearest_first`](Self::nearest_first), and as [`parity`](Self::parity) when resolved.
    pub fn parity_around(
        axis: PauliAxis,
        num_qubits: usize,
        edges: &[(usize, usize)],
        sites: &[usize],
        group: usize,
    ) -> Self {
        Self::parity(axis, Self::nearest_first(num_qubits, edges, sites), group)
    }

    /// Block rows: `order` cut into `P` contiguous runs, position `i` in run `⌊i · P / len⌋`, run `b` labelled `b` as [`PartitionRows::cut`] labels its blocks, every row reading `axis` bits.
    ///
    /// A term's label is the XOR of the labels of the runs it has odd `axis`-weight in, so under `Z` rows every `R_X` is local and a `ZZ` bond is remote exactly when it crosses runs; `X` rows mirror that.
    /// They balance an observable spread over the register, whose terms occupy every run; a circuit that conserves the total `axis` parity starves half the partitions once the terms occupy few runs, as a local observable's do (`research/FINDINGS.md` §Row choice on heavy-hex: x-parity rows beat the z-only cut).
    ///
    /// # Panics
    ///
    /// When resolved, as [`PartitionRows::cut`] on the runs, and if `order` repeats a qubit.
    pub fn blocks(axis: PauliAxis, order: Vec<usize>) -> Self {
        Self::Blocks { axis, order }
    }

    /// Every qubit of a `num_qubits` register in breadth-first order over `edges` from all of `sites` at once, `sites` first in ascending order, each qubit's neighbours in ascending order, and the qubits `edges` never reach last in ascending order.
    ///
    /// # Panics
    ///
    /// If a site or an edge's qubit is outside the register.
    pub fn nearest_first(
        num_qubits: usize,
        edges: &[(usize, usize)],
        sites: &[usize],
    ) -> Vec<usize> {
        let mut adjacent = vec![Vec::new(); num_qubits];
        for &(a, b) in edges {
            assert!(
                a < num_qubits && b < num_qubits,
                "edge ({a}, {b}) is outside the {num_qubits}-qubit register"
            );
            adjacent[a].push(b);
            adjacent[b].push(a);
        }
        let mut seen = vec![false; num_qubits];
        let mut starts = sites.to_vec();
        starts.sort_unstable();
        starts.dedup();
        let mut queue = VecDeque::new();
        for &site in &starts {
            assert!(site < num_qubits, "site {site} is outside the {num_qubits}-qubit register");
            seen[site] = true;
            queue.push_back(site);
        }
        let mut order = Vec::with_capacity(num_qubits);
        while let Some(q) = queue.pop_front() {
            order.push(q);
            adjacent[q].sort_unstable();
            for &next in &adjacent[q] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
        order.extend((0..num_qubits).filter(|&q| !seen[q]));
        order
    }

    /// The rows `circuit` and its `observable` suggest, from which bits the circuit's entanglers set and how far the observable spreads.
    ///
    /// When every channel on two or more qubits shifts only z-bits (`ZZ`-type entanglers) the parity axis is `X`, when only x-bits it is `Z`; any other circuit gets [`Seeded`](Self::Seeded) rows.
    /// An observable on at most [`FOR_RUN_PARITY_GROUP`] qubits, under a circuit some channel of which sets parity-axis bits, gets [`parity_around`](Self::parity_around) rows at that axis from its support over the entanglers' graph, which keep every entangler local.
    /// Any other observable gets [`blocks`](Self::blocks) over the qubits in index order, reading the bits the entanglers set, which keeps every single-qubit rotation about the parity axis local.
    ///
    /// Partitioned results agree with one partition's to rounding under any rows for an exact or threshold truncation; a partitioned `ApproxTopN` hands its boundary quota out in rank order and a collapse draws over the partitions in rank order, so their results depend on the rows (`research/FINDINGS.md` §The rank-order boundary quota biases cut-row splits), which is why this is not the scatters' default.
    ///
    /// # Panics
    ///
    /// When resolved, as [`parity`](Self::parity) or [`blocks`](Self::blocks): parity rows for `P` partitions need `2 · log2(P)` qubits.
    pub fn for_run<const W: usize, C: Coefficient>(
        observable: &PauliSum<W, C>,
        circuit: &Circuit<W>,
    ) -> Self {
        let num_qubits = circuit.num_qubits();
        let mut edges = Vec::new();
        let (mut entangler_x, mut entangler_z, mut sets_x, mut sets_z) =
            (false, false, false, false);
        for (channel, shifts) in circuit.channels().iter().zip(circuit.deltas()) {
            let Some(shifts) = shifts else { continue };
            let shifts_x = shifts.iter().any(|(x, _)| x.iter().any(|&w| w != 0));
            let shifts_z = shifts.iter().any(|(_, z)| z.iter().any(|&w| w != 0));
            sets_x |= shifts_x;
            sets_z |= shifts_z;
            let qubits = mask_qubits(&channel.support());
            if qubits.len() >= 2 {
                entangler_x |= shifts_x;
                entangler_z |= shifts_z;
                edges.extend(qubits.iter().zip(&qubits[1..]).map(|(&a, &b)| (a, b)));
            }
        }
        let (parity_axis, block_axis, kicked) = match (entangler_x, entangler_z) {
            (false, true) => (PauliAxis::X, PauliAxis::Z, sets_x),
            (true, false) => (PauliAxis::Z, PauliAxis::X, sets_z),
            _ => return Self::Seeded(None),
        };
        let mut support = [0u64; W];
        for (x, z, _) in observable.iter() {
            for w in 0..W {
                support[w] |= x[w] | z[w];
            }
        }
        let sites = mask_qubits(&support);
        if kicked && sites.len() <= FOR_RUN_PARITY_GROUP {
            Self::parity_around(parity_axis, num_qubits, &edges, &sites, FOR_RUN_PARITY_GROUP)
        } else {
            Self::blocks(block_axis, (0..num_qubits).collect())
        }
    }

    /// The rows for `2^bits` partitions of a `num_qubits` register, with `default_seed` standing in for a `None` seed.
    ///
    /// # Panics
    ///
    /// As the [`PartitionRows`] constructor each variant names, and if an excluded qubit is out of range.
    pub fn rows<const W: usize>(
        &self,
        num_qubits: usize,
        bits: u8,
        default_seed: u64,
    ) -> PartitionRows<W> {
        let mask = |qubits: &[usize]| qubit_mask(qubits.iter().copied(), num_qubits);
        match self {
            Self::Seeded(seed) => {
                PartitionRows::from_seed(num_qubits, bits, seed.unwrap_or(default_seed))
            }
            Self::Cut(blocks) => PartitionRows::cut(num_qubits, blocks),
            Self::SeededExcluding { seed, exclude_x, exclude_z } => {
                PartitionRows::from_seed_excluding(
                    num_qubits,
                    bits,
                    seed.unwrap_or(default_seed),
                    &mask(exclude_x),
                    &mask(exclude_z),
                )
            }
            Self::Matrix(rows) => PartitionRows::from_matrix(num_qubits, rows),
            Self::Parity { axis, order, group } => {
                assert!(*group > 0, "parity rows need at least one qubit per row");
                assert_distinct(order, num_qubits);
                let needed = bits as usize * group;
                assert!(
                    needed <= order.len(),
                    "{bits} parity rows of {group} qubits need {needed} qubits, the order holds {}",
                    order.len()
                );
                let masks = order.chunks(*group).take(bits as usize).map(mask);
                on_axis(num_qubits, *axis, masks.collect())
            }
            Self::Blocks { axis, order } => {
                assert_distinct(order, num_qubits);
                let partitions = 1usize << bits;
                let mut blocks = vec![Vec::new(); partitions];
                for (i, &q) in order.iter().enumerate() {
                    blocks[i * partitions / order.len()].push(q);
                }
                let cut = PartitionRows::<W>::cut(num_qubits, &blocks);
                on_axis(num_qubits, *axis, cut.rows().1.to_vec())
            }
        }
    }
}

/// Rows reading the `axis` bits of each mask.
fn on_axis<const W: usize>(
    num_qubits: usize,
    axis: PauliAxis,
    masks: Vec<[u64; W]>,
) -> PartitionRows<W> {
    let zero = vec![[0u64; W]; masks.len()];
    match axis {
        PauliAxis::X => PartitionRows::from_rows(num_qubits, masks, zero),
        PauliAxis::Y => PartitionRows::from_rows(num_qubits, masks.clone(), masks),
        PauliAxis::Z => PartitionRows::from_rows(num_qubits, zero, masks),
    }
}

/// Panics if `order` repeats a qubit or names one outside the register.
fn assert_distinct(order: &[usize], num_qubits: usize) {
    let mut seen = vec![false; num_qubits];
    for &q in order {
        assert!(q < num_qubits, "qubit {q} is outside the {num_qubits}-qubit register");
        assert!(!seen[q], "the row order repeats qubit {q}");
        seen[q] = true;
    }
}

/// The qubits a mask sets, ascending.
fn mask_qubits<const W: usize>(mask: &[u64; W]) -> Vec<usize> {
    (0..64 * W).filter(|&q| (mask[q / 64] >> (q % 64)) & 1 == 1).collect()
}

#[cfg(test)]
mod tests;
