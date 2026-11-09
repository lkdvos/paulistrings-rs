//! Runs on the device: a permutation run walked per string in one pass (`kernels/permute_run.cu`), and a rotation run as fused layers over pairs of its one-qubit tables.
//!
//! Contracts: a permutation run agrees with `apply_permutation_run` to floating-point tolerance and drops every intermediate string exactly where it does (engine/bucketed/permutation_run.rs).
//! A rotation run joins two one-qubit tables on different qubits into one table ([`LocalPtm::tensor`]) only under a keep program with no per-term filter, since the joined table asks it once, about the pair's output; so it agrees with its layers one by one to floating-point tolerance.

use std::borrow::Cow;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use crate::engine::gpu::module::launch_grid;

use super::fast_paths::permute_threads;
use super::rotation_run::rotation_run_fused;
use super::{apply_layer_device, timed_transfer, DeviceLayer, LayerScratch, Transfer};
use crate::channel::prepared::{LocalPtm, Prepared, LOCAL_DIM};
use crate::coefficient::Coefficient;
use crate::engine::bucketed::PermutationStep;
use crate::engine::gpu::columns::grow;
use crate::engine::gpu::error::GpuError;
use crate::engine::gpu::fingerprint::FingerprintRows;
use crate::engine::gpu::length::Growth;
use crate::engine::gpu::scan::exclusive_scan;
use crate::engine::gpu::sum::GpuSum;
use crate::engine::gpu::truncation::KeepProgram;
use crate::engine::plan::PartitionPlan;
use crate::engine::timing::timed;
use crate::engine::transport::SoloTransport;
use crate::pauli_sum::hash::Gf2Hash;
use std::sync::Arc;

/// Words of one step in `info`; matches `RUN_INFO_WORDS` in `kernels/permute_run.cu`.
const RUN_INFO_WORDS: usize = 24;

/// A permutation run's step tables on the device, and the scatter's per-bucket cursors; grow-only.
pub(crate) struct RunBuffers {
    info: CudaSlice<u32>,
    fingerprint: CudaSlice<u64>,
    amplitude: CudaSlice<u8>,
    cursor: CudaSlice<u32>,
}

impl RunBuffers {
    fn new(stream: &Arc<CudaStream>) -> Result<Self, GpuError> {
        Ok(Self {
            info: stream.alloc_zeros(1)?,
            fingerprint: stream.alloc_zeros(1)?,
            amplitude: stream.alloc_zeros(1)?,
            cursor: stream.alloc_zeros(1)?,
        })
    }
}

/// `steps` in `kernels/permute_run.cu`'s layout, every delta's bucket and fingerprint under `hash` and `fingerprints`.
fn step_tables<const W: usize, C: Coefficient>(
    steps: &[PermutationStep<W, C>],
    hash: &Gf2Hash<W>,
    fingerprints: &FingerprintRows<W>,
) -> (Vec<u32>, Vec<u64>, Vec<C>) {
    let mut info = Vec::with_capacity(steps.len() * RUN_INFO_WORDS);
    let mut fingerprint = Vec::with_capacity(steps.len() * LOCAL_DIM);
    let mut amplitude = Vec::with_capacity(steps.len() * LOCAL_DIM);
    for step in steps {
        let walk = step.walk_table();
        let [q0, q1] = walk.qubits;
        info.extend([
            q0 as u32,
            q1 as u32,
            if walk.two_qubit { 0b11 } else { 0 },
            walk.inert.into(),
        ]);
        info.extend(
            walk.delta.chunks_exact(4).map(|d| u32::from_le_bytes([d[0], d[1], d[2], d[3]])),
        );
        for &d in walk.delta {
            let (mut mask_x, mut mask_z) = ([0u64; W], [0u64; W]);
            for (bit, (qubit, column)) in
                [(q0, 0), (q0, 1), (q1, 0), (q1, 1)].into_iter().enumerate()
            {
                let mask = if column == 0 { &mut mask_x } else { &mut mask_z };
                mask[qubit / 64] ^= u64::from((d >> bit) & 1) << (qubit % 64);
            }
            info.push(hash.bucket_of(&mask_x, &mask_z));
            fingerprint.push(fingerprints.fingerprint(&mask_x, &mask_z));
        }
        amplitude.extend_from_slice(walk.amplitude);
    }
    (info, fingerprint, amplitude)
}

/// Apply the permutation run `steps` to `sum` in one walk, into the spare columns as a tight CSR in bucket order.
pub(crate) fn permutation_run_device<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    steps: &[PermutationStep<W, C>],
    keep: &KeepProgram,
    target_bits: u8,
    scratch: &mut LayerScratch<W, C>,
) -> Result<(), GpuError> {
    // A run, unlike a layer, never raises the driver's `target_bits`.
    if target_bits > sum.hash.bits() {
        scratch.refine(sum, target_bits)?;
    }
    let stream = sum.stream.clone();
    let kernels = sum.kernels.clone();
    let ordinal = sum.device();
    let b = sum.hash.num_buckets();
    let n_in = sum.len();
    let (info, fingerprint, amplitude) = step_tables(steps, &sum.hash, &sum.fingerprints);
    let amplitude = bytemuck::cast_slice::<C, u8>(&amplitude);
    let mut buffers = match scratch.run.take() {
        Some(buffers) => buffers,
        None => RunBuffers::new(&stream)?,
    };
    grow(&stream, &mut buffers.info, info.len(), ordinal)?;
    grow(&stream, &mut buffers.fingerprint, fingerprint.len(), ordinal)?;
    grow(&stream, &mut buffers.amplitude, amplitude.len(), ordinal)?;
    grow(&stream, &mut buffers.cursor, b, ordinal)?;
    let mut out = sum.take_spare(n_in, b)?;
    timed_transfer(&stream, &mut scratch.stats, Transfer::H2d, || {
        stream.memcpy_htod(&info, &mut buffers.info.slice_mut(0..info.len()))?;
        stream
            .memcpy_htod(&fingerprint, &mut buffers.fingerprint.slice_mut(0..fingerprint.len()))?;
        stream.memcpy_htod(amplitude, &mut buffers.amplitude.slice_mut(0..amplitude.len()))?;
        Ok(())
    })?;
    let (b32, steps32) = (b as u32, steps.len() as u32);
    let block_per_bucket = launch_grid(b32, permute_threads(n_in, b, W), 0);
    timed!(let t0 = scratch.event(sum)?);
    stream.memset_zeros(&mut out.lengths.slice_mut(0..b))?;
    // SAFETY: arguments match `k_permute_run_count` in permute_run.cu; the step buffers hold `steps32` steps and `out.lengths` holds `b` zeroed counters.
    unsafe {
        stream
            .launch_builder(&kernels.permute_run_count)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buffers.info)
            .arg(&buffers.fingerprint)
            .arg(&buffers.amplitude)
            .arg(&steps32)
            .arg(keep)
            .arg(&mut out.lengths)
            .launch(block_per_bucket)?;
    }
    exclusive_scan(
        &stream,
        &kernels,
        &out.lengths.slice(0..b),
        &mut out.start.slice_mut(0..b + 1),
        b,
        &mut scratch.scan,
        &mut scratch.totals,
    )?;
    stream.memcpy_dtod(&out.start.slice(0..b), &mut buffers.cursor.slice_mut(0..b))?;
    timed!(scratch.lap(sum, t0, |stats| &mut stats.gather_ns)?);
    timed!(let t1 = scratch.event(sum)?);
    // SAFETY: arguments match `k_permute_run_scatter`; `cursor` starts at `out.start`, and `out` has room for every input row, the scan's total at most.
    unsafe {
        stream
            .launch_builder(&kernels.permute_run_scatter)
            .arg(&sum.columns.x)
            .arg(&sum.columns.z)
            .arg(&sum.columns.coeff)
            .arg(&sum.columns.fingerprint)
            .arg(&sum.columns.start)
            .arg(&sum.columns.lengths)
            .arg(&buffers.info)
            .arg(&buffers.fingerprint)
            .arg(&buffers.amplitude)
            .arg(&steps32)
            .arg(keep)
            .arg(&mut buffers.cursor)
            .arg(&mut out.x)
            .arg(&mut out.z)
            .arg(&mut out.coeff)
            .arg(&mut out.fingerprint)
            .launch(block_per_bucket)?;
    }
    timed!(scratch.lap(sum, t1, |stats| &mut stats.merge_ns)?);
    scratch.run = Some(buffers);
    sum.swap_in(out, n_in);
    sum.record_step(Growth::Regroup, Some(&scratch.totals))?;
    scratch.extent = sum.len();
    scratch.record.records = n_in as u64;
    scratch.record.permuted = true;
    sum.debug_check();
    scratch.resolve(sum)
}

/// Apply the rotation run `layers` to `sum` as fused layers, a pair of one-qubit tables on different qubits joined into one when `keep` has no per-term filter; its layers are local, so a group member runs them as a lone partition would.
pub(crate) fn rotation_run_device<const W: usize, C: Coefficient>(
    sum: &mut GpuSum<W, C>,
    layers: &[Prepared<W, C>],
    keep: &KeepProgram,
    target_bits: u8,
    scratch: &mut LayerScratch<W, C>,
) -> Result<(), GpuError> {
    if *keep != KeepProgram::KEEP && rotation_run_fused(sum, layers, keep, target_bits, scratch)? {
        return Ok(());
    }
    let joins = *keep == KeepProgram::KEEP;
    let plan = PartitionPlan::default();
    let mut rest = layers;
    while let Some((first, after)) = rest.split_first() {
        let pair = match (first, after.first().filter(|_| joins)) {
            (Prepared::Local(a), Some(Prepared::Local(b))) => {
                LocalPtm::tensor(a, b, &sum.hash).map(Prepared::Local)
            }
            _ => None,
        };
        let (prepared, taken) = match pair {
            Some(pair) => (Cow::Owned(pair), 2),
            None => (Cow::Borrowed(first), 1),
        };
        let layer = DeviceLayer { prepared: &prepared, plan: &plan, keep, target_bits };
        apply_layer_device(sum, &layer, scratch, &SoloTransport)?;
        rest = &rest[taken..];
    }
    Ok(())
}

#[cfg(test)]
mod tests;
