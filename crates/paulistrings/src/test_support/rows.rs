//! Partition-row diagnostics: a circuit's key delta masks with their layer counts, and per-layer local/remote delta counts under given rows.
//! A row `r` leaves a delta local iff `⟨r, d⟩ = parity(rx & dx) ^ parity(rz & dz) = 0`.

use std::collections::HashMap;

use crate::channel::prepared::Prepared;
use crate::channel::Prepare;
use crate::circuit::Circuit;
use crate::engine::plan::PartitionPlan;
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};

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
/// If any channel's preparation declines.
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
        let prepared = channel
            .prepare(hash, adjoint)
            .unwrap_or_else(|| panic!("circuit_generators: layer {index} declined preparation"));
        // One layer's masks are distinct, so a repeat is a second layer.
        for m in prepared.key_delta_masks() {
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

/// Per layer in application order, `(local, remote)` delta counts of `circuit`'s channels prepared against `hash` under `rows`.
///
/// # Panics
///
/// If any channel's preparation declines.
pub fn count_remote_deltas<const W: usize>(
    circuit: &Circuit<W>,
    hash: &Gf2Hash<W>,
    rows: &PartitionRows<W>,
    adjoint: bool,
) -> Vec<(usize, usize)> {
    let n = circuit.channels.len();
    (0..n)
        .map(|k| if adjoint { n - 1 - k } else { k })
        .map(|index| {
            let prepared = circuit.channels[index].prepare(hash, adjoint).unwrap_or_else(|| {
                panic!("count_remote_deltas: layer {index} declined preparation")
            });
            let plan = PartitionPlan::new(&prepared, rows, 0);
            let local = plan.local_entries.iter().filter(|b| **b).count();
            (local, plan.remote.len())
        })
        .collect()
}

// The engine only ever rebuilds a plan in place; tests build one fresh.
impl PartitionPlan {
    /// A fresh plan of `prepared` under `rows`, as seen from partition `rank`.
    pub(crate) fn new<const W: usize>(
        prepared: &Prepared<W>,
        rows: &PartitionRows<W>,
        rank: u32,
    ) -> Self {
        let mut plan = Self::default();
        plan.rebuild(prepared, rows, rank);
        plan
    }
}

#[cfg(test)]
mod tests;
