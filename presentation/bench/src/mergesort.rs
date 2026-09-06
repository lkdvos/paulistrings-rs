//! Baseline 3b — "one flat array, parallel mergesort, merge equal keys".
//!
//! A reconstruction of the second multithreading attempt: the sum is one flat
//! `Vec<(x, z, c)>` of 48-byte rows. Every layer, Rayon tasks apply the channel
//! to their chunk and emit output rows; the whole output array is sorted in
//! parallel (`par_sort_unstable_by` — the array is a concatenation of unsorted
//! chunks, so unlike the engine's gather runs there is no run structure for an
//! adaptive stable sort to exploit; see `engine/merge.rs` for that choice), and a
//! segmented reduction merges equal keys, drops zeros and applies `keep_term`.
//! The reduction is the global merge the talk blames — it walks every output row
//! once more, through memory.

use num_complex::Complex64;
use paulistrings::truncation::CoefficientThreshold;
use paulistrings::{Channel, Circuit, OutputBuffer, PauliSum, TruncationPolicy};
use rayon::prelude::*;
use std::time::Instant;

use crate::common::{materialize, Key, RunResult, ZERO};
use crate::workload::W;

type Row = (Key, Complex64);

fn apply_chunk(ch: &dyn Channel<W>, rows: &[Row]) -> Vec<Row> {
    let fanout = ch.max_fanout().max(1);
    let mut bx = vec![[0u64; W]; fanout];
    let mut bz = vec![[0u64; W]; fanout];
    let mut bc = vec![ZERO; fanout];
    let mut out_rows = Vec::with_capacity(rows.len() * fanout);
    for &((x, z), c) in rows {
        let mut len = 0usize;
        let mut out = OutputBuffer::<W> { x: &mut bx, z: &mut bz, coeff: &mut bc, len: &mut len };
        ch.apply_adjoint(&x, &z, c, &mut out);
        for i in 0..len {
            out_rows.push(((bx[i], bz[i]), bc[i]));
        }
    }
    out_rows
}

/// Sum equal-key runs of a key-sorted array in place; keep the survivors.
fn reduce_sorted(rows: &mut Vec<Row>, policy: &CoefficientThreshold) {
    let mut w = 0usize;
    let mut i = 0usize;
    while i < rows.len() {
        let key = rows[i].0;
        let mut c = rows[i].1;
        let mut j = i + 1;
        while j < rows.len() && rows[j].0 == key {
            c += rows[j].1;
            j += 1;
        }
        if c != ZERO && policy.keep_term(&key.0, &key.1, c) {
            rows[w] = (key, c);
            w += 1;
        }
        i = j;
    }
    rows.truncate(w);
}

pub fn run(circuit: &Circuit<W>, observable: &PauliSum<W>, eps: f64, threads: usize) -> RunResult {
    let policy = CoefficientThreshold(eps);
    let mut live: Vec<Row> = observable.iter().map(|(x, z, c)| ((*x, *z), c)).collect();
    live.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut terms_out = Vec::with_capacity(circuit.len());
    let t0 = Instant::now();
    for ch in circuit.channels.iter().rev() {
        let chunk = (live.len() / (threads * 4)).max(1024);
        let mut out: Vec<Row> = live
            .par_chunks(chunk)
            .flat_map_iter(|chunk| apply_chunk(ch.as_ref(), chunk))
            .collect();
        out.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        reduce_sorted(&mut out, &policy);
        live = out;
        terms_out.push(live.len());
    }
    let wall_ns = t0.elapsed().as_nanos() as u64;
    let n = live.len();
    let sum = materialize(observable.num_qubits(), live.into_iter());
    debug_assert_eq!(sum.len(), n);
    RunResult { sum, wall_ns, terms_out, layer_wall_ns: None, phase: None, buckets: None }
}
