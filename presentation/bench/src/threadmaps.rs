//! Baseline 3a — "per-thread dictionaries, merged at the end of every layer".
//!
//! A reconstruction of the first multithreading attempt: the resident terms are
//! split into chunks, every Rayon task applies the channel into a private
//! `HashMap<key, coeff>`, and the private maps are merged into one at the end of
//! the layer (a tree reduction). Truncation is applied after the merge, on the
//! summed coefficients, exactly like `engine/direct.rs` — the single-map model
//! this generalises. The snapshot of the map into a `Vec` (Rayon needs an
//! indexable input) is part of the design's honest cost.
//!
//! Uses the real `Channel::apply_adjoint` and `TruncationPolicy::keep_term`, so
//! the arithmetic is identical to the engine's; `tests/agreement.rs` checks it.

use hashbrown::HashMap;
use num_complex::Complex64;
use paulistrings::truncation::CoefficientThreshold;
use paulistrings::{Channel, Circuit, OutputBuffer, PauliSum, TruncationPolicy};
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::time::Instant;

use crate::common::{materialize, Key, RunResult, ZERO};
use crate::workload::W;

type Map = HashMap<Key, Complex64, FxBuildHasher>;

fn apply_chunk(ch: &dyn Channel<W>, rows: &[(Key, Complex64)]) -> Map {
    let fanout = ch.max_fanout().max(1);
    let mut bx = vec![[0u64; W]; fanout];
    let mut bz = vec![[0u64; W]; fanout];
    let mut bc = vec![ZERO; fanout];
    let mut m = Map::with_capacity_and_hasher(rows.len() * fanout, FxBuildHasher);
    for &((x, z), c) in rows {
        let mut len = 0usize;
        let mut out = OutputBuffer::<W> { x: &mut bx, z: &mut bz, coeff: &mut bc, len: &mut len };
        ch.apply_adjoint(&x, &z, c, &mut out);
        for i in 0..len {
            *m.entry((bx[i], bz[i])).or_insert(ZERO) += bc[i];
        }
    }
    m
}

fn merge(mut a: Map, b: Map) -> Map {
    if a.len() < b.len() {
        return merge(b, a);
    }
    for (k, v) in b {
        *a.entry(k).or_insert(ZERO) += v;
    }
    a
}

pub fn run(circuit: &Circuit<W>, observable: &PauliSum<W>, eps: f64, threads: usize) -> RunResult {
    let policy = CoefficientThreshold(eps);
    let mut live: Map = observable.iter().map(|(x, z, c)| ((*x, *z), c)).collect();
    let mut terms_out = Vec::with_capacity(circuit.len());
    let t0 = Instant::now();
    for ch in circuit.channels.iter().rev() {
        let rows: Vec<(Key, Complex64)> = live.iter().map(|(k, c)| (*k, *c)).collect();
        // Enough chunks for stealing, but not so many that the merge tree explodes.
        let chunk = (rows.len() / (threads * 4)).max(1024);
        let mut merged = rows
            .par_chunks(chunk)
            .map(|chunk| apply_chunk(ch.as_ref(), chunk))
            .reduce(Map::default, merge);
        merged.retain(|&(x, z), c| *c != ZERO && policy.keep_term(&x, &z, *c));
        live = merged;
        terms_out.push(live.len());
    }
    let wall_ns = t0.elapsed().as_nanos() as u64;
    let n = live.len();
    let sum = materialize(observable.num_qubits(), live.into_iter().map(|(k, c)| (k, c)).collect::<Vec<_>>().into_iter());
    debug_assert_eq!(sum.len(), n);
    RunResult { sum, wall_ns, terms_out, layer_wall_ns: None, phase: None, buckets: None }
}
