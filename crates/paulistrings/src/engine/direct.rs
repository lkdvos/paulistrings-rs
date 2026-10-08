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
    buffer_x: Vec<[u64; W]>,
    buffer_z: Vec<[u64; W]>,
    buffer_coeff: Vec<Complex64>,
    hash: Gf2Hash<W>,
    num_qubits: usize,
}

impl<const W: usize> DirectSum<W> {
    /// Ingest a bucketed sum.
    pub(crate) fn from_sum(sum: PauliSum<W>) -> Self {
        let len = sum.len();
        let mut live = HashMap::with_capacity_and_hasher(len, FxBuildHasher);
        for (x, z, c) in sum.iter() {
            live.insert(PauliString::<W> { x: *x, z: *z }, c);
        }
        Self {
            live,
            next: HashMap::with_capacity_and_hasher(len, FxBuildHasher),
            buffer_x: Vec::new(),
            buffer_z: Vec::new(),
            buffer_coeff: Vec::new(),
            hash: sum.hash().clone(),
            num_qubits: sum.num_qubits(),
        }
    }

    /// Resident term count.
    pub(crate) fn len(&self) -> usize {
        self.live.len()
    }

    /// Apply one channel layer, then drop exact-zero sums and apply `keep_term` to each summed coefficient, as the merge does.
    pub(crate) fn apply_layer<T>(&mut self, channel: &dyn Channel<W>, policy: &T, adjoint: bool)
    where
        T: TruncationPolicy<W> + ?Sized,
    {
        let Self {
            live,
            next,
            buffer_x,
            buffer_z,
            buffer_coeff,
            ..
        } = self;

        let fanout = channel.max_fanout().max(1);
        if buffer_x.len() < fanout {
            buffer_x.resize(fanout, [0u64; W]);
            buffer_z.resize(fanout, [0u64; W]);
            buffer_coeff.resize(fanout, ZERO);
        }

        next.clear();
        next.reserve(live.len());

        for (key, &coeff) in live.iter() {
            let mut len = 0usize;
            {
                let mut out = OutputBuffer::<W> {
                    x: buffer_x,
                    z: buffer_z,
                    coeff: buffer_coeff,
                    len: &mut len,
                };
                if adjoint {
                    channel.apply_adjoint(&key.x, &key.z, coeff, &mut out);
                } else {
                    channel.apply(&key.x, &key.z, coeff, &mut out);
                }
            }
            for i in 0..len {
                let key = PauliString::<W> {
                    x: buffer_x[i],
                    z: buffer_z[i],
                };
                *next.entry(key).or_insert(ZERO) += buffer_coeff[i];
            }
        }

        next.retain(|key, coeff| *coeff != ZERO && policy.keep_term(&key.x, &key.z, *coeff));
        std::mem::swap(live, next);
    }

    /// Materialize a [`PauliSum`] under the entering hash, grown as [`PauliSum::rebucket`] would, leaving the map intact.
    pub(crate) fn to_sum(&self) -> PauliSum<W> {
        let mut entries: Vec<(PauliString<W>, Complex64)> = self
            .live
            .iter()
            .map(|(key, coeff)| (*key, *coeff))
            .collect();
        entries.sort_unstable_by(|a, b| (&a.0.x, &a.0.z).cmp(&(&b.0.x, &b.0.z)));
        let len = entries.len();
        let mut x = Vec::with_capacity(len);
        let mut z = Vec::with_capacity(len);
        let mut coeff = Vec::with_capacity(len);
        for (key, value) in entries {
            x.push(key.x);
            z.push(key.z);
            coeff.push(value);
        }
        let bits =
            desired_bits(len, DEFAULT_TARGET_BUCKET_LEN, DEFAULT_MIN_BUCKETS).max(self.hash.bits());
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
    let num_channels = circuit.channels.len();
    let adjoint = matches!(direction, super::Direction::Heisenberg);
    let tracing = scratch.term_trace.is_some();
    let gate_tracing = scratch.gate_trace.is_some();
    let finalizes = policy.finalizes_layer();

    let mut direct = DirectSum::from_sum(sum);
    let mut applied = 0usize;

    while applied < num_channels {
        let circuit_index = match direction {
            super::Direction::Forward => applied,
            super::Direction::Heisenberg => num_channels - 1 - applied,
        };
        let channel: &dyn Channel<W> = circuit.channels[circuit_index].as_ref();
        let application_index = applied;

        let debug_on = log::log_enabled!(target: super::LOG_TARGET, log::Level::Debug);
        let want_timer = gate_tracing || debug_on;
        let layer_started = want_timer.then(std::time::Instant::now);
        let terms_before = direct.len();

        direct.apply_layer(channel, policy, adjoint);

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
            let elapsed = layer_started
                .expect("want_timer implies layer_started is Some")
                .elapsed();
            if gate_tracing {
                super::record_gate_trace(
                    scratch,
                    circuit_index as u32,
                    application_index as u32,
                    channel.debug_name(),
                    terms_before,
                    terms_after,
                    elapsed,
                );
            }
            if debug_on {
                log::debug!(
                    target: super::LOG_TARGET,
                    "layer {}/{} [{}]: {} -> {} terms, {:.1} ms",
                    applied,
                    num_channels,
                    channel.debug_name(),
                    terms_before,
                    terms_after,
                    elapsed.as_secs_f64() * 1e3,
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
