//! Peer access and copy bandwidth between every pair of visible devices, as the device exchange sees them.
//!
//! `cargo run --release --features cuda --example gpu_peer -- [MiB]` (default 1024).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{
    sys, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use paulistrings::engine::gpu::{self, PeerAccess};

const REPS: u32 = 5;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn gbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 * REPS as f64 / secs / 1e9
}

/// GB/s of `REPS` copies of `src` into `dst` on `dst`'s stream, after one untimed copy.
fn dtod(stream: &Arc<CudaStream>, src: &CudaSlice<u64>, dst: &mut CudaSlice<u64>) -> Res<f64> {
    stream.memcpy_dtod(src, dst)?;
    stream.synchronize()?;
    let t = Instant::now();
    for _ in 0..REPS {
        stream.memcpy_dtod(src, dst)?;
    }
    stream.synchronize()?;
    Ok(gbps(src.len() * 8, t.elapsed().as_secs_f64()))
}

/// A grid-stride copy that dereferences `src` from the launching device, so it reads peer memory over the link.
const COPY_KERNEL: &str = r#"
extern "C" __global__ void copy_u64(const unsigned long long* src, unsigned long long* dst, unsigned long long n) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
        dst[i] = src[i];
}
"#;

/// GB/s of `REPS` runs of `op` on `stream`, after one untimed run.
fn timed(stream: &Arc<CudaStream>, bytes: usize, mut op: impl FnMut() -> Res<()>) -> Res<f64> {
    op()?;
    stream.synchronize()?;
    let t = Instant::now();
    for _ in 0..REPS {
        op()?;
    }
    stream.synchronize()?;
    Ok(gbps(bytes, t.elapsed().as_secs_f64()))
}

/// The raw device address of `buf`, for copies that bypass cudarc's per-context stream bookkeeping.
fn addr(buf: &CudaSlice<u64>, stream: &Arc<CudaStream>) -> u64 {
    let (p, _sync) = buf.device_ptr(stream);
    p
}

/// `cuMemcpyDtoDAsync` over unified addresses and a copy kernel run on `on`, from `src` into `dst`.
fn raw_paths(
    on: &Arc<CudaStream>,
    kernel: &CudaFunction,
    src: u64,
    dst: u64,
    len: usize,
) -> Res<(f64, f64)> {
    let bytes = len * 8;
    on.context().bind_to_thread()?;
    // SAFETY: both addresses are live allocations of `bytes` bytes, and peer access is enabled for the pair.
    let uva = timed(on, bytes, || {
        Ok(unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, on.cu_stream()) }.result()?)
    })?;
    let n = len as u64;
    let cfg = LaunchConfig {
        grid_dim: (1024, 1, 1),
        block_dim: (512, 1, 1),
        shared_mem_bytes: 0,
    };
    let kern = timed(on, bytes, || {
        let mut b = on.launch_builder(kernel);
        b.arg(&src).arg(&dst).arg(&n);
        // SAFETY: the kernel reads and writes `n` u64s at addresses that hold them.
        unsafe { b.launch(cfg) }?;
        Ok(())
    })?;
    Ok((uva, kern))
}

fn main() -> Res<()> {
    let mib: usize = std::env::args().nth(1).map_or(Ok(1024), |s| s.parse())?;
    let len = mib << 17;
    let n = gpu::device_count();
    println!("gpu_peer: {n} device(s), {mib} MiB per copy, {REPS} reps");
    let ctxs: Vec<Arc<CudaContext>> = (0..n).map(CudaContext::new).collect::<Result<_, _>>()?;
    let streams: Vec<Arc<CudaStream>> = ctxs.iter().map(|c| c.default_stream()).collect();
    let ptx = cudarc::nvrtc::compile_ptx(COPY_KERNEL)?;
    let kernels: Vec<CudaFunction> = ctxs
        .iter()
        .map(|c| Ok(c.load_module(ptx.clone())?.load_function("copy_u64")?))
        .collect::<Res<_>>()?;
    let mut bufs: Vec<[CudaSlice<u64>; 2]> = streams
        .iter()
        .map(|s| Ok([s.alloc_zeros::<u64>(len)?, s.alloc_zeros::<u64>(len)?]))
        .collect::<Res<_>>()?;

    for d in 0..n {
        let [a, b] = &mut bufs[d];
        let same = dtod(&streams[d], a, b)?;
        let (_, same_kernel) = raw_paths(
            &streams[d],
            &kernels[d],
            addr(a, &streams[d]),
            addr(b, &streams[d]),
            len,
        )?;
        // SAFETY: the pinned buffer is written by the first copy before anything reads it.
        let mut host = unsafe { ctxs[d].alloc_pinned::<u64>(len)? };
        streams[d].memcpy_dtoh(a, &mut host)?;
        streams[d].synchronize()?;
        let t = Instant::now();
        for _ in 0..REPS {
            streams[d].memcpy_dtoh(a, &mut host)?;
        }
        streams[d].synchronize()?;
        let d2h = gbps(len * 8, t.elapsed().as_secs_f64());
        let t = Instant::now();
        for _ in 0..REPS {
            streams[d].memcpy_htod(&host, b)?;
        }
        streams[d].synchronize()?;
        let h2d = gbps(len * 8, t.elapsed().as_secs_f64());
        println!("device {d}: same-device {same:.1} GB/s (copy kernel {same_kernel:.1}), pinned d2h {d2h:.1} GB/s, pinned h2d {h2d:.1} GB/s");
    }

    for dst in 0..n {
        for src in 0..n {
            if dst == src {
                continue;
            }
            let access = gpu::peer_access(dst as u32, src as u32)?;
            let (lo, hi) = bufs.split_at_mut(dst.max(src));
            let (s, d) = if src < dst {
                (&lo[src][0], &mut hi[0][1])
            } else {
                (&hi[0][0], &mut lo[dst][1])
            };
            let bw = dtod(&streams[dst], s, d)?;
            let (sa, da) = (addr(s, &streams[src]), addr(d, &streams[dst]));
            let note = if access == PeerAccess::Enabled {
                ""
            } else {
                "  (host-staged)"
            };
            println!("{src} -> {dst}: {access:?}, cuMemcpyPeerAsync {bw:.1} GB/s{note}");
            // The pushing kernel needs the reverse mapping, which the pair loop may not have reached yet.
            let reverse = gpu::peer_access(src as u32, dst as u32)?;
            if access == PeerAccess::Enabled && reverse == PeerAccess::Enabled {
                let (uva, pull) = raw_paths(&streams[dst], &kernels[dst], sa, da, len)?;
                let (_, push) = raw_paths(&streams[src], &kernels[src], sa, da, len)?;
                println!("{src} -> {dst}: cuMemcpyDtoDAsync {uva:.1} GB/s, kernel on {dst} pulling {pull:.1} GB/s, kernel on {src} pushing {push:.1} GB/s");
            }
        }
    }

    if n >= 2 {
        // Both directions at once, as a two-partition exchange runs them.
        let (lo, hi) = bufs.split_at_mut(1);
        streams[1].memcpy_dtod(&lo[0][0], &mut hi[0][1])?;
        streams[0].memcpy_dtod(&hi[0][0], &mut lo[0][1])?;
        streams[0].synchronize()?;
        streams[1].synchronize()?;
        let t = Instant::now();
        for _ in 0..REPS {
            streams[1].memcpy_dtod(&lo[0][0], &mut hi[0][1])?;
            streams[0].memcpy_dtod(&hi[0][0], &mut lo[0][1])?;
        }
        streams[0].synchronize()?;
        streams[1].synchronize()?;
        let bw = gbps(2 * len * 8, t.elapsed().as_secs_f64());
        println!("0 <-> 1 concurrently: {bw:.1} GB/s aggregate");
    }
    Ok(())
}
