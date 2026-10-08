//! The opt-in small-sum layer path: the sum held in a hash map across layers, one `Channel::apply` per term, no `prepare` and no sort.
//!
//! It is the algorithm of `test_support::naive_apply_layer`, and it applies channels of any support width.
//! Research/FINDINGS.md §Direct-apply path for small sums.

use hashbrown::HashMap;
use num_complex::Complex64;
use rustc_hash::FxBuildHasher;

use crate::channel::{Channel, OutputBuffer};
use crate::pauli_string::PauliString;
use crate::pauli_sum::hash::Gf2Hash;
use crate::pauli_sum::storage::{desired_bits, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN};
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// A Pauli sum held as a hash map, carrying the entering hash so [`Self::to_sum`] hands back the same rows at no fewer bits.
pub(crate) struct DirectSum<const W: usize> {
    /// The resident terms, never an exact zero.
    live: HashMap<PauliString<W>, Complex64, FxBuildHasher>,
    /// The layer's output; output keys can collide with unvisited input keys, so it cannot be done in place.
    next: HashMap<PauliString<W>, Complex64, FxBuildHasher>,
    /// `Channel::apply`'s output columns, sized to the widest `max_fanout` seen.
    buf_x: Vec<[u64; W]>,
    buf_z: Vec<[u64; W]>,
    buf_c: Vec<Complex64>,
    hash: Gf2Hash<W>,
    num_qubits: usize,
}

impl<const W: usize> DirectSum<W> {
    /// Ingest a bucketed sum.
    pub(crate) fn from_sum(sum: PauliSum<W>) -> Self {
        let n = sum.len();
        let mut live = HashMap::with_capacity_and_hasher(n, FxBuildHasher);
        for (x, z, c) in sum.iter() {
            live.insert(PauliString::<W> { x: *x, z: *z }, c);
        }
        Self {
            live,
            next: HashMap::with_capacity_and_hasher(n, FxBuildHasher),
            buf_x: Vec::new(),
            buf_z: Vec::new(),
            buf_c: Vec::new(),
            hash: sum.hash().clone(),
            num_qubits: sum.num_qubits(),
        }
    }

    /// Resident term count.
    pub(crate) fn len(&self) -> usize {
        self.live.len()
    }

    /// Apply one channel layer, then drop exact-zero sums and apply `keep_term` to each summed coefficient, as the merge does.
    pub(crate) fn apply_layer<T>(&mut self, ch: &dyn Channel<W>, policy: &T, adjoint: bool)
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let Self {
            live,
            next,
            buf_x,
            buf_z,
            buf_c,
            ..
        } = self;

        let fanout = ch.max_fanout().max(1);
        if buf_x.len() < fanout {
            buf_x.resize(fanout, [0u64; W]);
            buf_z.resize(fanout, [0u64; W]);
            buf_c.resize(fanout, ZERO);
        }

        next.clear();
        next.reserve(live.len());

        for (p, &c) in live.iter() {
            let mut len = 0usize;
            {
                let mut out = OutputBuffer::<W> {
                    x: buf_x,
                    z: buf_z,
                    coeff: buf_c,
                    len: &mut len,
                };
                if adjoint {
                    ch.apply_adjoint(&p.x, &p.z, c, &mut out);
                } else {
                    ch.apply(&p.x, &p.z, c, &mut out);
                }
            }
            for i in 0..len {
                let key = PauliString::<W> {
                    x: buf_x[i],
                    z: buf_z[i],
                };
                *next.entry(key).or_insert(ZERO) += buf_c[i];
            }
        }

        next.retain(|p, c| *c != ZERO && policy.keep_term(&p.x, &p.z, *c));
        std::mem::swap(live, next);
    }

    /// Materialize a [`PauliSum`] under the entering hash, grown as [`PauliSum::rebucket`] would, leaving the map intact.
    pub(crate) fn to_sum(&self) -> PauliSum<W> {
        let mut entries: Vec<(PauliString<W>, Complex64)> =
            self.live.iter().map(|(p, c)| (*p, *c)).collect();
        entries.sort_unstable_by(|a, b| (&a.0.x, &a.0.z).cmp(&(&b.0.x, &b.0.z)));
        let n = entries.len();
        let mut x = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut coeff = Vec::with_capacity(n);
        for (p, c) in entries {
            x.push(p.x);
            z.push(p.z);
            coeff.push(c);
        }
        let bits =
            desired_bits(n, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS).max(self.hash.bits());
        let hash = if bits == self.hash.bits() {
            self.hash.clone()
        } else {
            Gf2Hash::new(self.num_qubits, bits, self.hash.seed())
        };
        PauliSum::from_key_sorted(&x, &z, &coeff, hash, self.num_qubits)
    }

    /// Re-ingest a sum after the `finalize_layer` round trip, keeping the carried hash.
    pub(crate) fn reload(&mut self, sum: &PauliSum<W>) {
        self.live.clear();
        self.live.reserve(sum.len());
        for (x, z, c) in sum.iter() {
            self.live.insert(PauliString::<W> { x: *x, z: *z }, c);
        }
    }
}

/// Run leading layers on the direct path until one leaves the sum above the threshold, returning the sum and the layer count applied.
// Out of line so it stays out of `propagate_with`'s inlined layer loop (CLAUDE.md §Performance discipline).
// The trace records and the `DEBUG` line must match the sorting loop's exactly, since tooling parses them.
#[inline(never)]
pub(crate) fn run_direct_prefix<const W: usize, T>(
    circuit: &crate::circuit::Circuit<W>,
    sum: PauliSum<W>,
    policy: &T,
    direction: super::Direction,
    scratch: &mut super::LayerScratch<W>,
    options: super::PropagateOptions,
) -> (PauliSum<W>, usize)
where
    T: TruncationPolicy<W> + ?Sized,
{
    let n = circuit.channels.len();
    let adjoint = matches!(direction, super::Direction::Heisenberg);
    let tracing = scratch.term_trace.is_some();
    let gate_tracing = scratch.gate_trace.is_some();
    let finalizes = policy.finalizes_layer();

    let mut direct = DirectSum::from_sum(sum);
    let mut applied = 0usize;

    while applied < n {
        let idx = match direction {
            super::Direction::Forward => applied,
            super::Direction::Heisenberg => n - 1 - applied,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();
        let application_index = applied;

        let debug_on = log::log_enabled!(target: super::LOG_TARGET, log::Level::Debug);
        let want_timer = gate_tracing || debug_on;
        let layer_t0 = want_timer.then(std::time::Instant::now);
        let terms_before = direct.len();

        direct.apply_layer(ch, policy, adjoint);

        if finalizes {
            let mut materialized = direct.to_sum();
            policy.finalize_layer(&mut materialized);
            direct.reload(&materialized);
        }

        let terms_after = direct.len();
        applied += 1;

        if tracing {
            super::record_layer_terms(scratch, terms_before, terms_after);
        }
        if want_timer {
            let dt = layer_t0
                .expect("want_timer implies layer_t0 is Some")
                .elapsed();
            if gate_tracing {
                super::record_gate_trace(
                    scratch,
                    idx as u32,
                    application_index as u32,
                    ch.debug_name(),
                    terms_before,
                    terms_after,
                    dt,
                );
            }
            if debug_on {
                log::debug!(
                    target: super::LOG_TARGET,
                    "layer {}/{} [{}]: {} -> {} terms, {:.1} ms",
                    applied,
                    n,
                    ch.debug_name(),
                    terms_before,
                    terms_after,
                    dt.as_secs_f64() * 1e3,
                );
            }
        }

        if terms_after > options.small_sum_threshold {
            break;
        }
    }

    (direct.to_sum(), applied)
}

#[cfg(test)]
mod tests;
