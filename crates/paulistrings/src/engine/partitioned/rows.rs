//! The circuit's generator masks and the per-layer locality a row set gives
//! them.
//!
//! A partitioning splits keys by `part(v) = R·v` over GF(2), one bit per row
//! `r = (rx, rz)` of `R` (ARCHITECTURE.md §Partitioning). A prepared channel
//! moves a term by a key delta `d`, so the layer is local for that delta iff
//!
//! ```text
//! ⟨r, d⟩ = parity(rx & dx) ^ parity(rz & dz) = 0     for every row r,
//! ```
//!
//! which is [`PartitionRows::partition_of`] returning 0. Remoteness is a
//! property of the mask alone, so a row set is judged by which of the circuit's
//! delta masks it is orthogonal to. [`circuit_generators`] lists those masks
//! with the number of layers carrying each; [`layer_locality`] is the
//! per-layer verdict for a given row set.
//!
//! For 1- and 2-local generators a good row set is a graph cut: a z-only row
//! makes every single-qubit `X` rotation local (its mask has no z-bits), and a
//! bond `ZZ(i, j)` is remote exactly when the edge crosses the cut. That row
//! set is [`PartitionRows::cut`]. A row orthogonal to *every* mask is a
//! conserved quantity — no term ever changes partition and the other
//! partitions stay empty — so a useful row set always leaves some generator
//! remote. (A greedy weighted MAX-XOR-SAT selector for circuits without a known
//! geometry lived here until commit da86546; on the heavy-hex step it halved
//! the remote layers but doubled the imbalance, see
//! `research/notes/2026-09-09-partition-row-tuning-results.md`.)

use std::collections::HashMap;

use crate::bucket::hash::{Gf2Hash, PartitionRows};
use crate::channel::prepared::Prepared;
use crate::circuit::Circuit;

use super::plan::count_remote_deltas;

/// One key delta mask the circuit produces, with how much of the circuit
/// carries it.
///
/// A row `r` leaves this mask local iff `⟨r, mask⟩ = 0`; `weight` is how many
/// layers that buys (ARCHITECTURE.md §Partitioning).
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorWeight<const W: usize> {
    /// X-half of the key delta.
    pub mask_x: [u64; W],
    /// Z-half of the key delta.
    pub mask_z: [u64; W],
    /// Number of layers carrying this mask, as produced by
    /// [`circuit_generators`]. A caller building the list itself may weight it
    /// any way it likes — only the ordering and the sums are read.
    pub weight: f64,
    /// [`Channel::debug_name`](crate::Channel::debug_name) of the first layer
    /// that produced this mask. Diagnostics only.
    pub name: &'static str,
}

/// Every non-identity key delta mask `circuit` produces, with the number of
/// layers carrying it.
///
/// Layers are walked in **application order** — circuit order for
/// `adjoint == false`, reverse order for `adjoint == true`, matching
/// [`Direction::Heisenberg`](crate::Direction) — and each layer is prepared
/// against `hash` exactly as `propagate` prepares it, so the masks are the
/// engine's own. A [`Prepared::Rotation`] contributes its generator mask; a
/// [`Prepared::Local`] contributes each delta entry's mask. Duplicates across
/// layers are merged and their weights added, so a Trotter circuit of `k`
/// identical steps reports each mask once with weight `k`.
///
/// The identity delta (mask 0) is dropped: `part(0) = 0` unconditionally, so it
/// constrains nothing.
///
/// The result is in first-appearance order, which makes it deterministic but
/// carries no other meaning.
///
/// # Panics
///
/// Panics if any channel declines
/// [`Channel::prepare`](crate::Channel::prepare), the same condition on which
/// `propagate` panics.
pub fn circuit_generators<const W: usize>(
    circuit: &Circuit<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
) -> Vec<GeneratorWeight<W>> {
    let n = circuit.channels.len();
    let mut out: Vec<GeneratorWeight<W>> = Vec::new();
    let mut seen: HashMap<Mask<W>, usize> = HashMap::new();

    for k in 0..n {
        let idx = if adjoint { n - 1 - k } else { k };
        let ch = &circuit.channels[idx];
        let prep = ch
            .prepare(hash, adjoint)
            .unwrap_or_else(|| panic!("circuit_generators: layer {idx} declined Channel::prepare"));
        // One layer's masks are distinct by construction (a `LocalPtm`'s
        // entries are keyed by distinct `local_delta`s), so a mask seen twice
        // here is a mask seen in two layers.
        let masks: Vec<Mask<W>> = match &prep {
            Prepared::Local(ptm) => ptm.deltas().iter().map(|d| d.mask()).collect(),
            Prepared::Rotation(r) => vec![r.gen_mask()],
        };
        for m in masks {
            if mask_is_zero(&m) {
                continue;
            }
            match seen.get(&m) {
                Some(&i) => out[i].weight += 1.0,
                None => {
                    seen.insert(m, out.len());
                    out.push(GeneratorWeight {
                        mask_x: m.0,
                        mask_z: m.1,
                        weight: 1.0,
                        name: ch.debug_name(),
                    });
                }
            }
        }
    }
    out
}

/// Per layer, `true` if the layer moves nothing across a partition boundary
/// under `rows`.
///
/// The boolean form of [`count_remote_deltas`]: layers in application order
/// (circuit order forward, reverse order adjoint), one flag each, `true` when
/// the layer's every delta has `part(mask) == 0` and the layer therefore runs
/// with no exchange at all.
///
/// # Panics
///
/// Panics if any channel declines
/// [`Channel::prepare`](crate::Channel::prepare).
pub fn layer_locality<const W: usize>(
    circuit: &Circuit<W>,
    rows: &PartitionRows<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
) -> Vec<bool> {
    count_remote_deltas(circuit, hash, rows, adjoint)
        .into_iter()
        .map(|(_, remote)| remote == 0)
        .collect()
}

/// `(x, z)` halves of a key delta, as `PartitionRows` numbers the columns.
type Mask<const W: usize> = ([u64; W], [u64; W]);

#[inline]
fn mask_is_zero<const W: usize>(m: &Mask<W>) -> bool {
    m.0.iter().chain(m.1.iter()).all(|w| *w == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::channel::clifford::Clifford1Q;
    use crate::channel::rotation::PauliRotation;
    use crate::pauli_string::PauliString;
    use crate::test_support::zz_rotation;

    fn hash(num_qubits: usize) -> Gf2Hash<1> {
        Gf2Hash::<1>::new(num_qubits, 4, 0xB00C)
    }

    /// One TFIM Trotter step on an **open** chain: an `X` rotation on every
    /// qubit, then a `ZZ` bond rotation on every edge. The open chain has one
    /// fewer bond than qubits, which is what makes a single cut able to leave
    /// exactly one bond remote.
    fn tfim_open_chain(num_qubits: usize) -> Circuit<1> {
        let mut c = Circuit::<1>::new(num_qubits);
        for q in 0..num_qubits {
            c.push(PauliRotation::new(PauliString::<1>::x(q as u32), 0.3));
        }
        for q in 0..num_qubits - 1 {
            c.push(zz_rotation::<1>(q as u32, q as u32 + 1, 0.2));
        }
        c
    }

    #[test]
    fn circuit_generators_merges_duplicate_masks_and_counts_layers() {
        let mut c = Circuit::<1>::new(4);
        c.push(PauliRotation::new(PauliString::<1>::x(0), 0.3));
        c.push(zz_rotation::<1>(0, 1, 0.2));
        c.push(PauliRotation::new(PauliString::<1>::x(0), 0.7));
        let gens = circuit_generators(&c, &hash(4), false);

        // Two distinct masks: X_0 (twice) and Z_0·Z_1 (once). The identity
        // delta of every layer is dropped.
        assert_eq!(gens.len(), 2);
        assert_eq!(gens[0].mask_x, [0b0001]);
        assert_eq!(gens[0].mask_z, [0]);
        assert_eq!(gens[0].weight, 2.0);
        assert_eq!(gens[0].name, "PauliRotation");
        assert_eq!(gens[1].mask_x, [0]);
        assert_eq!(gens[1].mask_z, [0b0011]);
        assert_eq!(gens[1].weight, 1.0);
    }

    #[test]
    fn circuit_generators_walks_layers_in_application_order() {
        let mut c = Circuit::<1>::new(4);
        c.push(Clifford1Q::h(0)); // delta X_0 Z_0
        c.push(Clifford1Q::s(1)); // delta Z_1
        let forward = circuit_generators(&c, &hash(4), false);
        let adjoint = circuit_generators(&c, &hash(4), true);
        assert_eq!(forward.len(), 2);
        assert_eq!((forward[0].mask_x, forward[0].mask_z), ([0b1], [0b1]));
        assert_eq!((forward[1].mask_x, forward[1].mask_z), ([0], [0b10]));
        // Reverse order, same masks (an adjoint's delta set is the same
        // subspace).
        assert_eq!((adjoint[0].mask_x, adjoint[0].mask_z), ([0], [0b10]));
        assert_eq!((adjoint[1].mask_x, adjoint[1].mask_z), ([0b1], [0b1]));
    }

    #[test]
    fn circuit_generators_of_an_empty_circuit_is_empty() {
        assert!(circuit_generators(&Circuit::<1>::new(4), &hash(4), false).is_empty());
    }

    #[test]
    fn layer_locality_is_count_remote_deltas_as_a_predicate() {
        const N: usize = 8;
        let c = tfim_open_chain(N);
        let h = hash(N);
        // A z-only cut between qubits 3 and 4: every `X` rotation is local
        // and the one bond crossing it is the only remote layer.
        let rows = PartitionRows::<1>::cut(N, &[vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);

        for adjoint in [false, true] {
            let want: Vec<bool> = count_remote_deltas(&c, &h, &rows, adjoint)
                .into_iter()
                .map(|(_, remote)| remote == 0)
                .collect();
            let got = layer_locality(&c, &rows, &h, adjoint);
            assert_eq!(got, want, "adjoint {adjoint}");
            assert_eq!(got.len(), c.channels.len());
            // Exactly one layer — the bond crossing the cut — is remote.
            assert_eq!(got.iter().filter(|b| !**b).count(), 1, "adjoint {adjoint}");
        }
    }

    #[test]
    fn layer_locality_without_rows_is_all_local() {
        let c = tfim_open_chain(6);
        let rows = PartitionRows::<1>::none(6);
        assert!(layer_locality(&c, &rows, &hash(6), false)
            .into_iter()
            .all(|b| b));
    }

    // ---- property: the reported split is the rows' own verdict ----
}
