//! [`GpuPauliSum`], a sum propagated on one CUDA device through the shared partitioned layer loop, and the [`propagate_gpu`] front door.

use std::time::Instant;

use super::error::GpuError;
use super::layer::{GpuKernelMs, GpuLayerCounters, GpuLayerOptions};
use super::partition::DevicePartition;
use super::sum::GpuSum;
use crate::bucket::hash::PartitionRows;
use crate::circuit::Circuit;
use crate::engine::partitioned::driver::{run_layers, PartitionCtx, PartitionWork};
use crate::engine::partitioned::trace::{assemble, PartitionTrace};
use crate::engine::partitioned::transport::InProcessTransport;
use crate::engine::partitioned::truncation::PartitionedTruncation;
use crate::engine::{Direction, PropagateOptions};
use crate::pauli_sum::PauliSum;

const LOG_TARGET: &str = "paulistrings::propagate";

/// A [`PauliSum`] resident on one CUDA device, propagated layer by layer with the fused device layer (ARCHITECTURE.md §GPU-Readiness).
///
/// Built by [`Self::from_host`], stepped by [`Self::propagate`], read back by [`Self::to_host`].
/// The layer loop is the partitioned engine's at `P = 1`, so the bucket-count schedule, trace and log lines are those of [`PartitionedSum`](crate::engine::partitioned::PartitionedSum).
/// Only per-term policies with a device form run here: [`TruncationPolicy::device_policy`](crate::TruncationPolicy::device_policy) must return `Some`, or `propagate` fails with [`GpuError::Unsupported`] before touching the device.
///
/// A device error mid-run leaves the sum in the state of the last completed layer and is returned from `propagate`.
pub struct GpuPauliSum<const W: usize> {
    part: DevicePartition<W>,
    rows: PartitionRows<W>,
    trace: Option<PartitionTrace>,
}

impl<const W: usize> GpuPauliSum<W> {
    /// Upload `sum` to device `ordinal`.
    pub fn from_host(sum: &PauliSum<W>, ordinal: u32) -> Result<Self, GpuError> {
        Self::wrap(GpuSum::from_host(sum, ordinal)?, sum.num_qubits())
    }

    /// As [`Self::from_host`] with extra NVRTC options, the `-DFP_BITS=<b>` collision hook.
    #[doc(hidden)]
    pub fn from_host_with_options(
        sum: &PauliSum<W>,
        ordinal: u32,
        extra_options: &[String],
    ) -> Result<Self, GpuError> {
        Self::wrap(
            GpuSum::from_host_with_options(sum, ordinal, extra_options)?,
            sum.num_qubits(),
        )
    }

    fn wrap(sum: GpuSum<W>, num_qubits: usize) -> Result<Self, GpuError> {
        Ok(Self {
            part: DevicePartition::new(sum, GpuLayerOptions::default())?,
            rows: PartitionRows::none(num_qubits),
            trace: None,
        })
    }

    /// Propagate through `circuit` with the default [`PropagateOptions`].
    pub fn propagate<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
    ) -> Result<(), GpuError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        self.propagate_with_options(circuit, policy, direction, PropagateOptions::default())
    }

    /// Propagate through `circuit`; `options` drive the host-side bucket schedule, the device policy in [`Self::set_layer_options`] refines on top of it.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unsupported`] before any layer if `policy` has no device form; any device error from a layer, after which the sum holds the last completed layer's output.
    pub fn propagate_with_options<T>(
        &mut self,
        circuit: &Circuit<W>,
        policy: &T,
        direction: Direction,
        options: PropagateOptions,
    ) -> Result<(), GpuError>
    where
        T: PartitionedTruncation<W> + ?Sized,
    {
        let keep = policy.device_policy().ok_or(GpuError::Unsupported(
            "truncation policy without a device form",
        ))?;
        if policy.finalizes_layer() {
            return Err(GpuError::Unsupported("truncation policy with a layer pass"));
        }
        self.part.keep = keep;
        self.part.take_error()?;
        let n = circuit.channels.len();
        let terms_in = self.len();
        let started = Instant::now();
        log::info!(
            target: LOG_TARGET,
            "propagate_gpu: {terms_in} terms through {n} channels ({direction:?}) on device {}",
            self.device(),
        );
        if n > 0 {
            let tracing = self.trace.is_some();
            let mut work = PartitionWork::take(&mut self.part, n, tracing);
            let transports = InProcessTransport::group(1);
            let ctx = PartitionCtx {
                rows: &self.rows,
                rank: 0,
                size: 1,
                tracing,
            };
            run_layers(
                circuit,
                policy,
                direction,
                options,
                ctx,
                &mut work,
                &transports[0],
            );
            self.part = work.local;
            if let Some(trace) = self.trace.as_mut() {
                assemble(trace, vec![work.rows]);
            }
        }
        log::info!(
            target: LOG_TARGET,
            "propagate_gpu: {n} layers applied, {terms_in} -> {} terms, {:.3} s",
            self.len(),
            started.elapsed().as_secs_f64(),
        );
        self.part.take_error()
    }

    /// Download the sum, each bucket re-sorted to the host's order.
    pub fn to_host(&self) -> Result<PauliSum<W>, GpuError> {
        self.part.sum().to_host()
    }

    /// Terms on the device.
    pub fn len(&self) -> usize {
        self.part.sum().len()
    }

    /// `true` if the sum has no terms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Active bucket bits.
    pub fn bits(&self) -> u8 {
        self.part.sum().bits()
    }

    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.part.sum().num_qubits()
    }

    /// The device ordinal.
    pub fn device(&self) -> u32 {
        self.part.sum().device()
    }

    /// The device layer's knobs; takes effect from the next layer.
    pub fn set_layer_options(&mut self, options: GpuLayerOptions) {
        self.part.scratch_mut().options = options;
    }

    /// Record CUDA-event timings per kernel family into [`Self::take_kernel_ms`]; costs a synchronization per kernel.
    pub fn set_kernel_timing(&mut self, on: bool) {
        self.part.scratch_mut().time_kernels = on;
    }

    /// Kernel milliseconds accumulated since the last call, zeroing the counters.
    pub fn take_kernel_ms(&mut self) -> GpuKernelMs {
        std::mem::take(&mut self.part.scratch_mut().kernel_ms)
    }

    /// The counters of the most recent layer.
    pub fn last_layer_counters(&self) -> GpuLayerCounters {
        self.part.counters()
    }

    /// Start recording one [`PartitionTrace`] record per layer.
    pub fn enable_trace(&mut self) {
        self.trace.get_or_insert_with(PartitionTrace::default);
    }

    /// Drain the per-layer records, `Some` iff tracing is on.
    pub fn take_trace(&mut self) -> Option<PartitionTrace> {
        self.trace.as_mut().map(std::mem::take)
    }
}

/// Propagate `sum` through `circuit` on device `device` and return the result on the host.
///
/// One-shot convenience over [`GpuPauliSum`]; a caller stepping repeatedly should hold the resident sum instead.
pub fn propagate_gpu<const W: usize, T>(
    circuit: &Circuit<W>,
    sum: &PauliSum<W>,
    policy: &T,
    direction: Direction,
    device: u32,
) -> Result<PauliSum<W>, GpuError>
where
    T: PartitionedTruncation<W> + ?Sized,
{
    let mut dev = GpuPauliSum::from_host(sum, device)?;
    dev.propagate(circuit, policy, direction)?;
    dev.to_host()
}
