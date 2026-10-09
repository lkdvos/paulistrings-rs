//! The circuit's key delta masks with the number of layers carrying each, for judging partition rows (ARCHITECTURE.md §Partitioning).
//! A row `r` leaves a delta local iff `⟨r, d⟩ = parity(rx & dx) ^ parity(rz & dz) = 0`.

use std::collections::HashMap;

use crate::channel::prepared::Prepared;
use crate::circuit::Circuit;
use crate::pauli_sum::hash::Gf2Hash;

/// One key delta mask the circuit produces, with how many layers carry it.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratorWeight<const W: usize> {
    /// X-half of the key delta.
    pub mask_x: [u64; W],
    /// Z-half of the key delta.
    pub mask_z: [u64; W],
    /// Layers carrying this mask; a caller may weight it any other way.
    pub weight: f64,
    /// [`Channel::debug_name`](crate::Channel::debug_name) of the first layer that produced this mask.
    pub name: &'static str,
}

/// Every non-identity key delta mask of `circuit` prepared against `hash` in application order, merged across layers, in first-appearance order.
///
/// # Panics
///
/// If any channel declines [`Channel::prepare`](crate::Channel::prepare).
pub fn circuit_generators<const W: usize>(
    circuit: &Circuit<W>,
    hash: &Gf2Hash<W>,
    adjoint: bool,
) -> Vec<GeneratorWeight<W>> {
    let n = circuit.channels.len();
    let mut out: Vec<GeneratorWeight<W>> = Vec::new();
    let mut seen: HashMap<Mask<W>, usize> = HashMap::new();

    for k in 0..n {
        let index = if adjoint { n - 1 - k } else { k };
        let channel = &circuit.channels[index];
        let prepared = channel.prepare(hash, adjoint).unwrap_or_else(|| {
            panic!("circuit_generators: layer {index} declined Channel::prepare")
        });
        // One layer's masks are distinct, so a repeat is a second layer.
        let masks: Vec<Mask<W>> = match &prepared {
            Prepared::Local(ptm) => ptm.deltas().iter().map(|d| d.mask()).collect(),
            Prepared::Rotation(rotation) => vec![rotation.generator_mask()],
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
                        name: channel.debug_name(),
                    });
                }
            }
        }
    }
    out
}

/// `(x, z)` halves of a key delta.
type Mask<const W: usize> = ([u64; W], [u64; W]);

fn mask_is_zero<const W: usize>(m: &Mask<W>) -> bool {
    m.0.iter().chain(m.1.iter()).all(|w| *w == 0)
}

#[cfg(test)]
mod tests;
