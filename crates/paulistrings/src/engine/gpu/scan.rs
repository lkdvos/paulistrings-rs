//! Device-wide exclusive scan of a `u32` column, the three launches of `kernels/scan.cu`.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use super::error::GpuError;
use super::module::KernelSet;

/// Elements per scan block; must match `SCAN_BLOCK` in `kernels/scan.cu`.
const SCAN_BLOCK: usize = 4096;
const SCAN_THREADS: u32 = 1024;

/// `out[i] = Σ_{j<i} input[j]` for `i ≤ n`, so `out[n]` is the total; enqueued on `stream`, not synchronized.
pub(crate) fn exclusive_scan(
    stream: &Arc<CudaStream>,
    k: &KernelSet,
    input: &CudaView<'_, u32>,
    out: &mut CudaViewMut<'_, u32>,
    n: usize,
) -> Result<(), GpuError> {
    exclusive_scan_with_max(stream, k, input, out, n).map(|_| ())
}

/// [`exclusive_scan`] that also returns a two-element device buffer `[total, max]` of `input[0..n]`.
pub(crate) fn exclusive_scan_with_max(
    stream: &Arc<CudaStream>,
    k: &KernelSet,
    input: &CudaView<'_, u32>,
    out: &mut CudaViewMut<'_, u32>,
    n: usize,
) -> Result<CudaSlice<u32>, GpuError> {
    let nb = n.div_ceil(SCAN_BLOCK).max(1);
    if nb > SCAN_BLOCK {
        return Err(GpuError::Unsupported("scan of more than 2^24 elements"));
    }
    debug_assert!(input.len() >= n && out.len() > n);
    let mut block_sum = stream.alloc_zeros::<u32>(nb)?;
    let mut block_max = stream.alloc_zeros::<u32>(nb)?;
    let mut tot_max = stream.alloc_zeros::<u32>(2)?;
    let (n32, nb32) = (n as u32, nb as u32);
    let block = (SCAN_THREADS, 1, 1);
    // SAFETY: argument lists match the `extern "C"` signatures in scan.cu, and every buffer holds the elements indexed.
    unsafe {
        stream
            .launch_builder(&k.scan_block)
            .arg(input)
            .arg(&mut *out)
            .arg(&mut block_sum)
            .arg(&mut block_max)
            .arg(&n32)
            .launch(LaunchConfig {
                grid_dim: (nb32, 1, 1),
                block_dim: block,
                shared_mem_bytes: 0,
            })?;
        stream
            .launch_builder(&k.scan_single)
            .arg(&mut block_sum)
            .arg(&block_max)
            .arg(&nb32)
            .arg(&mut *out)
            .arg(&n32)
            .arg(&mut tot_max)
            .launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: block,
                shared_mem_bytes: 0,
            })?;
        stream
            .launch_builder(&k.scan_add)
            .arg(&mut *out)
            .arg(&block_sum)
            .arg(&n32)
            .launch(LaunchConfig {
                grid_dim: ((n as u32).div_ceil(SCAN_THREADS).max(1), 1, 1),
                block_dim: block,
                shared_mem_bytes: 0,
            })?;
    }
    Ok(tot_max)
}
