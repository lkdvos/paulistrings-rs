//! The propagation engine. See §8.

#![allow(unused)]

pub mod sort_merge;

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
pub fn propagate<const W: usize, T>(
    _circuit: &Circuit<W>,
    _initial: PauliSum<W>,
    _policy: &T,
    _direction: Direction,
) -> PauliSum<W>
where
    T: TruncationPolicy<W>,
{
    todo!("§8.1: iterate channels (reversed for Heisenberg), call sort_merge::apply_layer per channel")
}
