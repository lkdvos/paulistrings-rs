//! Prototype: the propagation engine over hash-table buckets updated in place, for comparison with the sorted-column engine.
//!
//! One partition, host only, and per-term truncation only: a policy with a layer pass panics.

// Pattern-indexed tables read clearer indexed, and a plan is built once per layer.
#![allow(clippy::needless_range_loop, clippy::large_enum_variant)]

mod fused;
mod layer;
mod open_table;
mod orbit;
mod table;

pub use layer::{apply_layer_hashed, HashScratch, Kernel, PassTimes, WARM};
pub use table::HashSum;

/// Set the walk prefetch distance in slots, 0 disabling it; a prototype knob.
pub fn set_prefetch_slots(slots: usize) {
    open_table::PREFETCH_SLOTS.store(slots, std::sync::atomic::Ordering::Relaxed);
}

use crate::channel::Prepare;
use crate::circuit::Circuit;
use crate::engine::Direction;
use crate::truncation::TruncationPolicy;

/// Bucket sizing for [`propagate_hashed`]; unlike [`PropagateOptions`](crate::PropagateOptions), any `min_buckets >= 1` is accepted, so one bucket is reachable.
#[derive(Clone, Copy, Debug)]
pub struct HashOptions {
    /// Terms per bucket the rebucketing targets.
    pub target_bucket_len: usize,
    /// Floor on the bucket count once the sum is worth splitting.
    pub min_buckets: usize,
    /// The per-coset kernel.
    pub kernel: Kernel,
}

/// Propagate `sum` through `circuit` in place under a per-term `policy`.
///
/// # Panics
///
/// If `policy` has a layer pass, or a channel's preparation declines.
pub fn propagate_hashed<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &mut HashSum<W>,
    policy: &T,
    direction: Direction,
    scratch: &mut HashScratch<W>,
    options: HashOptions,
) where
    T: TruncationPolicy<W> + ?Sized,
{
    assert!(!policy.finalizes_layer(), "the hash-kernel prototype supports per-term policies only");
    let adjoint = matches!(direction, Direction::Heisenberg);
    // The orbit and fused kernels never revisit a term no layer touches, so the input must already pass the filter.
    if options.kernel != Kernel::Scatter {
        sum.retain(|x, z, c| policy.keep_term(x, z, c));
    }
    let n = circuit.channels.len();
    for application_index in 0..n {
        let circuit_index = match direction {
            Direction::Forward => application_index,
            Direction::Heisenberg => n - 1 - application_index,
        };
        sum.rebucket(options.target_bucket_len, options.min_buckets);
        let prepared = circuit.channels[circuit_index]
            .prepare(&sum.hash, adjoint)
            .expect("channel preparation declined");
        apply_layer_hashed(sum, &prepared, policy, scratch, options.kernel);
    }
}

#[cfg(test)]
mod tests;
