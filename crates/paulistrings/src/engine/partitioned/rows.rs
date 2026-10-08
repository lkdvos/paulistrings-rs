//! The circuit's generator masks.
//!
//! A partitioning splits keys by `part(v) = R·v` over GF(2), one bit per row `r = (rx, rz)` of `R` (ARCHITECTURE.md §Partitioning).
//! A prepared channel moves a term by a key delta `d`, so the layer is local for that delta iff
//!
//! ```text
//! ⟨r, d⟩ = parity(rx & dx) ^ parity(rz & dz) = 0     for every row r,
//! ```
//!
//! which is `PartitionRows::partition_of` returning 0.
//! Remoteness is a property of the mask alone, so a row set is judged by which of the circuit's delta masks it is orthogonal to; [`circuit_generators`] lists those masks with the number of layers carrying each.
//!
//! For 1- and 2-local generators a good row set is a graph cut: a z-only row makes every single-qubit `X` rotation local (its mask has no z-bits), and a bond `ZZ(i, j)` is remote exactly when the edge crosses the cut — that row set is `PartitionRows::cut`.
//! A row orthogonal to *every* mask is a conserved quantity — no term ever changes partition and the other partitions stay empty — so a useful row set always leaves some generator remote.

use std::collections::HashMap;

use crate::channel::prepared::Prepared;
use crate::circuit::Circuit;
use crate::pauli_sum::hash::Gf2Hash;

/// One key delta mask the circuit produces, with how much of the circuit carries it.
///
/// A row `r` leaves this mask local iff `⟨r, mask⟩ = 0`; `weight` is how many layers that buys (ARCHITECTURE.md §Partitioning).
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorWeight<const W: usize> {
    /// X-half of the key delta.
    pub mask_x: [u64; W],
    /// Z-half of the key delta.
    pub mask_z: [u64; W],
    /// Number of layers carrying this mask, as produced by [`circuit_generators`].
    /// A caller building the list itself may weight it any way it likes — only the ordering and the sums are read.
    pub weight: f64,
    /// [`Channel::debug_name`](crate::Channel::debug_name) of the first layer that produced this mask. Diagnostics only.
    pub name: &'static str,
}

/// Every non-identity key delta mask `circuit` produces, with the number of layers carrying it.
///
/// Layers are walked in **application order** — circuit order for `adjoint == false`, reverse order for `adjoint == true`, matching [`Direction::Heisenberg`](crate::Direction) — and each layer is prepared against `hash` exactly as `propagate` prepares it, so the masks are the engine's own.
/// A `Prepared::Rotation` contributes its generator mask; a `Prepared::Local` contributes each delta entry's mask.
/// Duplicates across layers are merged and their weights added, so a Trotter circuit of `k` identical steps reports each mask once with weight `k`.
///
/// The identity delta (mask 0) is dropped: `part(0) = 0` unconditionally, so it constrains nothing.
///
/// The result is in first-appearance order, which makes it deterministic but carries no other meaning.
///
/// # Panics
///
/// Panics if any channel declines [`Channel::prepare`](crate::Channel::prepare), the same condition on which `propagate` panics.
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
        // One layer's masks are distinct by construction (a `LocalPtm`'s entries are keyed by distinct `local_delta`s), so a mask seen twice here is a mask seen in two layers.
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

/// `(x, z)` halves of a key delta, as `PartitionRows` numbers the columns.
type Mask<const W: usize> = ([u64; W], [u64; W]);

#[inline]
fn mask_is_zero<const W: usize>(m: &Mask<W>) -> bool {
    m.0.iter().chain(m.1.iter()).all(|w| *w == 0)
}

#[cfg(test)]
mod tests;
