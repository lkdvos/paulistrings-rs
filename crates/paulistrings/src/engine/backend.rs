//! The storage seam under `run_layers`, [`PartitionStorage`] and [`PartitionBackend`], and its host-memory implementation [`HostPartition`].

use super::partitioned::layer::{
    apply_layer_partitioned_with_plan, LayerExchangeCounts, PartitionState,
};
use super::plan::PartitionPlan;
use super::transport::Transport;
use crate::channel::prepared::Prepared;
use crate::coefficient::Coefficient;
use crate::collectives::Collectives;
use crate::engine::bucketed::{apply_rotation_run, max_fanout, LayerScratch, RowEmitter, Run};
#[cfg(feature = "phase-timing")]
use crate::engine::stats::PhaseStats;
use crate::engine::trace::PartitionLayerRow;
use crate::pauli_sum::hash::{Gf2Hash, PartitionRows};
use crate::pauli_sum::storage::{rebucketed_bits, PauliSum};
use crate::truncation::TruncationPolicy;

/// The policy-independent half of a partition.
// `pub` only because the `pub` methods of `PartitionedSum` and `DistributedSum` bound their backend parameter on it.
#[doc(hidden)]
pub trait PartitionStorage<const W: usize, C: Coefficient = num_complex::Complex64>:
    Send + Sized
{
    fn len(&self) -> usize;
    fn hash(&self) -> &Gf2Hash<W>;
    fn refine(&mut self);
    /// Halve the bucket count until `bits` remain; called only when the count this partition settles on for a layer is below the one it holds.
    fn coarsen_to(&mut self, bits: u8);
    /// The bucket bits this partition proposes for the layer; a backend may raise the host formula but never lower it.
    fn proposed_bits(
        &self,
        prepared: &Prepared<W, C>,
        target_bucket_len: usize,
        min_buckets: usize,
    ) -> u8 {
        let _ = prepared;
        crate::pauli_sum::storage::rebucketed_bits(
            self.len(),
            self.hash().bits(),
            target_bucket_len,
            min_buckets,
        )
        .min(self.hash().max_bits())
    }
    /// Refine to the host bucket policy without the prepared channel and return `true`, or return `false` for a backend whose proposal needs it.
    fn refine_unprepared(&mut self, target_bucket_len: usize, min_buckets: usize) -> bool {
        let _ = (target_bucket_len, min_buckets);
        false
    }
    /// End a call: apply any truncation the last layer pass deferred; the driver calls it after the last layer.
    fn settle(&mut self) {}
    /// Move the partition out, leaving an empty one under the same hash so the driver stays consistent if the work is never handed back.
    fn detach(&mut self) -> Self;
    #[cfg(feature = "phase-timing")]
    fn stats(&mut self) -> &mut PhaseStats;
    /// The run-total counters so far, busy time a backend holds apart from [`Self::stats`] included.
    #[cfg(feature = "phase-timing")]
    fn phase_snapshot(&mut self) -> PhaseStats {
        *self.stats()
    }
}

/// One layer under policy `T`; an implementor must issue exactly the transport calls the host layer issues, in the same order, or the group falls out of step; the device backend's one addition is the exchange vote (`engine/gpu/export.rs`), legal only because every partition of a group runs the same backend.
pub(crate) trait PartitionBackend<const W: usize, C: Coefficient, T: ?Sized>:
    PartitionStorage<W, C>
{
    /// `clocked` asks for the counts' phase times, which a backend may leave zero; `ahead` is the next layer's one remote delta, prepared under the current hash, which a backend may count for that layer's export.
    #[allow(clippy::too_many_arguments)]
    fn apply_layer<X: Transport>(
        &mut self,
        prepared: &Prepared<W, C>,
        plan: &PartitionPlan,
        rows: &PartitionRows<W>,
        policy: &T,
        transport: &X,
        clocked: bool,
        ahead: Option<&RowEmitter<'_, W, C>>,
    ) -> LayerExchangeCounts;
    fn finalize_layer(&mut self, policy: &T, collectives: &dyn Collectives);
    /// Whether [`apply_run`](Self::apply_run) is implemented; the driver asks before it looks ahead for a run.
    fn takes_runs(&self) -> bool {
        false
    }
    /// Apply `run` in one pass; called only when [`takes_runs`](Self::takes_runs) and the policy has no layer pass.
    fn apply_run(&mut self, run: Run<'_, W, C>, policy: &T) {
        let _ = (run, policy);
        unreachable!("the backend takes no runs");
    }
    /// Amend the layer's trace row with what the backend counted itself; called only when tracing.
    fn trace_layer(&self, row: &mut PartitionLayerRow) {
        let _ = row;
    }
}

/// A partition in host memory with its retained scratch.
// `pub` only because it is the default backend parameter of `PartitionedSum` and `DistributedSum`.
#[doc(hidden)]
#[derive(Debug)]
pub struct HostPartition<const W: usize, C: Coefficient = num_complex::Complex64> {
    pub(crate) sum: PauliSum<W, C>,
    pub(super) state: PartitionState<W, C>,
}

impl<const W: usize, C: Coefficient> HostPartition<W, C> {
    pub(crate) fn new(sum: PauliSum<W, C>) -> Self {
        Self::with_layer_scratch(sum, LayerScratch::default())
    }

    pub(crate) fn with_layer_scratch(sum: PauliSum<W, C>, layer: LayerScratch<W, C>) -> Self {
        Self { sum, state: PartitionState { layer, ..PartitionState::default() } }
    }

    pub(crate) fn into_parts(self) -> (PauliSum<W, C>, LayerScratch<W, C>) {
        (self.sum, self.state.layer)
    }
}

impl<const W: usize, C: Coefficient> PartitionStorage<W, C> for HostPartition<W, C> {
    /// The length once a deferred cut is applied.
    fn len(&self) -> usize {
        self.state.layer.octaves.len_after_cut(self.sum.len())
    }

    fn hash(&self) -> &Gf2Hash<W> {
        self.sum.hash()
    }

    fn refine(&mut self) {
        self.state.layer.octaves.apply_pending(&mut self.sum);
        self.sum.refine();
    }

    fn coarsen_to(&mut self, bits: u8) {
        self.state.layer.octaves.apply_pending(&mut self.sum);
        self.sum.coarsen_to(bits);
    }

    fn refine_unprepared(&mut self, target_bucket_len: usize, min_buckets: usize) -> bool {
        let bits = self.sum.hash().bits();
        let wanted = rebucketed_bits(self.len(), bits, target_bucket_len, min_buckets)
            .min(self.sum.hash().max_bits());
        if wanted != bits {
            self.state.layer.octaves.apply_pending(&mut self.sum);
            self.sum.rebucket(target_bucket_len, min_buckets);
        }
        true
    }

    fn settle(&mut self) {
        self.state.layer.octaves.settle(&mut self.sum);
    }

    fn detach(&mut self) -> Self {
        let placeholder = PauliSum::empty_with_hash(self.sum.num_qubits(), self.sum.hash().clone());
        Self {
            sum: std::mem::replace(&mut self.sum, placeholder),
            state: std::mem::take(&mut self.state),
        }
    }

    #[cfg(feature = "phase-timing")]
    fn stats(&mut self) -> &mut PhaseStats {
        &mut self.state.layer.stats
    }

    #[cfg(feature = "phase-timing")]
    fn phase_snapshot(&mut self) -> PhaseStats {
        self.state.layer.peek_phase_stats()
    }
}

impl<const W: usize, C: Coefficient, T> PartitionBackend<W, C, T> for HostPartition<W, C>
where
    T: TruncationPolicy<W, C> + ?Sized,
{
    fn apply_layer<X: Transport>(
        &mut self,
        prepared: &Prepared<W, C>,
        plan: &PartitionPlan,
        rows: &PartitionRows<W>,
        policy: &T,
        transport: &X,
        clocked: bool,
        ahead: Option<&RowEmitter<'_, W, C>>,
    ) -> LayerExchangeCounts {
        self.state.layer.octaves.begin_layer(
            max_fanout(prepared),
            policy.fused_approx_top_n(),
            plan.has_remote(),
        );
        apply_layer_partitioned_with_plan(
            &mut self.sum,
            prepared,
            plan,
            rows,
            policy,
            &mut self.state,
            transport,
            clocked,
            ahead,
        )
    }

    fn takes_runs(&self) -> bool {
        true
    }

    fn apply_run(&mut self, run: Run<'_, W, C>, policy: &T) {
        match run {
            Run::Rotations(layers) => {
                self.state.layer.octaves.begin_layer(1, policy.fused_approx_top_n(), false);
                apply_rotation_run(&mut self.sum, layers, policy, &mut self.state.layer);
            }
        }
    }

    fn finalize_layer(&mut self, policy: &T, collectives: &dyn Collectives) {
        match policy.fused_approx_top_n() {
            Some(fused) => self.state.layer.octaves.finalize(&mut self.sum, collectives, fused),
            None => {
                if policy.finalizes_layer() {
                    self.state.layer.octaves.drop_ahead();
                }
                policy.finalize_layer_partitioned(&mut self.sum, collectives);
            }
        }
    }
}

#[cfg(test)]
mod tests;
