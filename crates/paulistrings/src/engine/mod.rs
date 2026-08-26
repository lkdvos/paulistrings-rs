//! The propagation engine: front door [`propagate`], pipeline in
//! [`sort_merge`].
//!
//! See design doc §8 (loop) and §5 (sort-merge pipeline).

pub mod bucketed;
pub mod sort_merge;

use crate::bucket::hash::Gf2Hash;
use crate::bucket::sum::{desired_bits, BucketedSum, DEFAULT_HASH_SEED, DEFAULT_TARGET_BUCKET_LEN};
use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;
use bucketed::{apply_layer_bucketed, LayerScratch};

/// Propagation direction.
///
/// [`Direction::Forward`] applies channels in order; [`Direction::Heisenberg`]
/// iterates in reverse and applies adjoints (for backpropagating
/// observables).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Apply channels in the order they were pushed onto the [`Circuit`].
    Forward,
    /// Apply channels in reverse order, using each channel's
    /// [`Channel::apply_adjoint`].
    Heisenberg,
}

/// Propagate `initial` through `circuit` under `policy`.
///
/// Iterates the circuit's channels — in order for [`Direction::Forward`], in
/// reverse for [`Direction::Heisenberg`], calling [`Channel::apply_adjoint`]
/// in the latter case (default = self-adjoint; overridden on
/// [`PauliRotation`](crate::channel::PauliRotation) and
/// [`Clifford1Q`](crate::channel::Clifford1Q)).
///
/// # Examples
///
/// ```
/// use paulistrings::{
///     BuildAccumulator, Circuit, Direction, PauliString, Phase, TruncationPolicy,
///     channel::Clifford1Q, propagate,
/// };
/// use num_complex::Complex64;
///
/// let mut acc = BuildAccumulator::<1>::new(1);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
/// let observable = acc.finalize();
///
/// let mut circuit = Circuit::<1>::new(1);
/// circuit.push(Clifford1Q::h(0));
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {}
///
/// // H conjugates Z to X, so propagating Z₀ through H gives X₀.
/// let evolved = propagate(&circuit, observable, &KeepAll, Direction::Heisenberg);
/// assert_eq!(evolved.len(), 1);
/// assert_eq!(evolved.x()[0], [1]);
/// assert_eq!(evolved.z()[0], [0]);
/// ```
///
/// See design doc §8.1.
pub fn propagate<const W: usize, T>(
    circuit: &Circuit<W>,
    initial: PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W>
where
    T: TruncationPolicy<W>,
{
    let n = circuit.channels.len();
    if n == 0 {
        // No layers: return the input untouched, bit for bit.
        return initial;
    }

    let num_qubits = initial.num_qubits();

    // Size the partition once, then let `rebucket` keep it in band. Note the
    // bucket count depends on the thread count through the parallelism floor —
    // which is safe only because the engine's output is bitwise independent of
    // the bucket count (v0.2 §9.1), a property the tests pin directly.
    let min_buckets = default_min_buckets();
    let bits = desired_bits(initial.len(), DEFAULT_TARGET_BUCKET_LEN, min_buckets);
    let hash = Gf2Hash::<W>::new(num_qubits, bits, DEFAULT_HASH_SEED);

    let mut sum = BucketedSum::from_sum(&initial, hash);
    drop(initial);
    let mut scratch = LayerScratch::<W>::new();
    propagate_bucketed(circuit, &mut sum, policy, direction, &mut scratch);
    sum.into_sum()
}

/// Bucket-count floor: enough buckets that Rayon has slack to load-balance.
pub fn default_min_buckets() -> usize {
    4 * rayon::current_num_threads().max(1)
}

/// Propagate a sum that is **already bucketed**, in place.
///
/// This is what [`propagate`] does between its two conversions, exposed so a
/// caller can avoid paying them repeatedly.
///
/// # When you want this
///
/// [`propagate`] converts in and out once per call, which is free when amortized
/// over a long circuit and is not when it isn't. Measured at 10⁶ terms the round
/// trip is ~126 ms, against ~7 ms for a rotation layer — so a *short* circuit
/// applied repeatedly to a large sum is much better served by converting once and
/// calling this in a loop. `research/notes/2026-08-26-v0.2-results.md` §5 has the
/// numbers and the crossover.
///
/// A driver stepping an observable through many Trotter steps is exactly that
/// shape: one `BucketedSum`, one `LayerScratch`, one conversion at each end.
///
/// # Examples
///
/// ```
/// use paulistrings::bucket::{BucketedSum, Gf2Hash, DEFAULT_HASH_SEED};
/// use paulistrings::engine::bucketed::LayerScratch;
/// use paulistrings::engine::{default_min_buckets, propagate_bucketed};
/// use paulistrings::{BuildAccumulator, Circuit, Direction, PauliString, Phase, TruncationPolicy};
/// use num_complex::Complex64;
///
/// struct KeepAll;
/// impl<const W: usize> TruncationPolicy<W> for KeepAll {}
///
/// let mut acc = BuildAccumulator::<1>::with_capacity(8, 1);
/// acc.add_term(PauliString::<1>::z(0), Phase::ONE, Complex64::new(1.0, 0.0));
/// let initial = acc.finalize();
///
/// let mut circuit = Circuit::<1>::new(8);
/// circuit.push(paulistrings::channel::Clifford1Q::h(0));
///
/// let hash = Gf2Hash::<1>::new(8, 4, DEFAULT_HASH_SEED);
/// let mut sum = BucketedSum::from_sum(&initial, hash);
/// let mut scratch = LayerScratch::<1>::new();
///
/// // Two steps, one conversion each end rather than two round trips.
/// for _ in 0..2 {
///     propagate_bucketed(&circuit, &mut sum, &KeepAll, Direction::Forward, &mut scratch);
/// }
/// // H applied twice is the identity, so we are back to Z.
/// let out = sum.into_sum();
/// assert_eq!((out.x()[0], out.z()[0]), ([0], [1]));
/// ```
pub fn propagate_bucketed<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &mut BucketedSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut LayerScratch<W>,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    let n = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    let min_buckets = default_min_buckets();

    for k in 0..n {
        let idx = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();

        sum.rebucket(DEFAULT_TARGET_BUCKET_LEN, min_buckets);

        match ch.prepare(sum.hash(), adjoint) {
            Some(prep) => apply_layer_bucketed(sum, &prep, policy, scratch),
            None => {
                // The channel declined to be bucketed — support wider than the
                // local maximum, or it writes outside its declared support. Fall
                // back to the whole-sum v0.1 pipeline for this layer only. Correct,
                // just not bucketed.
                let flat = sum.to_sum();
                let out = if adjoint {
                    sort_merge::apply_layer_adjoint(&flat, ch, policy)
                } else {
                    sort_merge::apply_layer(&flat, ch, policy)
                };
                sum.refill_from_sum(&out);
            }
        }

        policy.finalize_layer_bucketed(sum);
    }
}
