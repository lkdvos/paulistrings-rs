//! The propagation engine: front door [`propagate`], pipeline in
//! [`sort_merge`].
//!
//! See design doc §8 (loop) and §5 (sort-merge pipeline).

pub mod bucketed;
pub(crate) mod coset;
pub mod sort_merge;

use crate::bucket::sum::DEFAULT_TARGET_BUCKET_LEN;
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
/// The sum is propagated in its bucketed form throughout — there is no
/// conversion at either end, so calling this repeatedly on the same sum (a
/// Trotter driver stepping an observable) costs nothing beyond the layers
/// themselves; per-bucket storage capacity is retained inside the returned
/// sum across calls. The bucket count is re-normalized against
/// [`desired_bits`](crate::bucket::desired_bits) before every layer.
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
/// let (x, z, _c) = evolved.bucket(0);
/// assert_eq!(x[0], [1]);
/// assert_eq!(z[0], [0]);
/// ```
///
/// See design doc §8.1.
pub fn propagate<const W: usize, T>(
    circuit: &Circuit<W>,
    mut sum: PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W>
where
    T: TruncationPolicy<W> + ?Sized,
{
    let n = circuit.channels.len();
    let adjoint = matches!(direction, Direction::Heisenberg);
    // The bucket count `B` is a deterministic function of the term count alone
    // (v0.3 §1): bitwise independence of the engine's output from `B` is still
    // tested (v0.2 §9.1), but it is no longer what makes this safe.
    let min_buckets = default_min_buckets();
    let mut scratch = LayerScratch::<W>::new();

    for k in 0..n {
        let idx = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();

        sum.rebucket(DEFAULT_TARGET_BUCKET_LEN, min_buckets);

        match ch.prepare(sum.hash(), adjoint) {
            Some(prep) => apply_layer_bucketed(&mut sum, &prep, policy, &mut scratch),
            None => {
                // The channel declined to be bucketed — support wider than the
                // local maximum, or it writes outside its declared support. Fall
                // back to the whole-sum v0.1 pipeline for this layer only
                // (`apply_layer` flattens, runs the flat pipeline, and scatters
                // back under the same hash). Correct, just not bucketed.
                sum = if adjoint {
                    sort_merge::apply_layer_adjoint(&sum, ch, policy)
                } else {
                    sort_merge::apply_layer(&sum, ch, policy)
                };
            }
        }

        policy.finalize_layer(&mut sum);
    }
    sum
}

/// Bucket-count floor: enough buckets that Rayon has slack to load-balance.
///
/// Fixed (v0.3 §1), not derived from `rayon::current_num_threads`: see
/// [`crate::bucket::sum::DEFAULT_MIN_BUCKETS`] for why a thread-independent
/// floor is what we want here.
pub fn default_min_buckets() -> usize {
    crate::bucket::sum::DEFAULT_MIN_BUCKETS
}
