//! The propagation engine. See §8.

pub mod sort_merge;

use crate::channel::Channel;
use crate::circuit::Circuit;
use crate::pauli_sum::PauliSum;
use crate::truncation::TruncationPolicy;

/// Propagation direction.
///
/// `Forward` applies channels in order; `Heisenberg` iterates in reverse and
/// applies adjoints (for backpropagating observables).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Heisenberg,
}

/// Propagate `initial` through `circuit` under `policy`. See §8.1.
///
/// Iterates the circuit's channels — in order for `Direction::Forward`, in
/// reverse for `Direction::Heisenberg`, calling `Channel::apply_adjoint` in
/// the latter case (default = self-adjoint; overridden on `PauliRotation`
/// and `Clifford1Q`).
pub fn propagate<const W: usize, T>(
    circuit: &Circuit<W>,
    initial: PauliSum<W>,
    policy: &T,
    direction: Direction,
) -> PauliSum<W>
where
    T: TruncationPolicy<W>,
{
    let mut sum = initial;
    let n = circuit.channels.len();
    for k in 0..n {
        let idx = match direction {
            Direction::Forward => k,
            Direction::Heisenberg => n - 1 - k,
        };
        let ch: &dyn Channel<W> = circuit.channels[idx].as_ref();
        sum = match direction {
            Direction::Forward => sort_merge::apply_layer(&sum, ch, policy),
            Direction::Heisenberg => sort_merge::apply_layer_adjoint(&sum, ch, policy),
        };
        policy.finalize_layer(&mut sum);
    }
    sum
}
