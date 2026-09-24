//! Peer access and copy bandwidth between every pair of visible devices, as the device exchange sees them.
//!
//! `cargo run --release --features cuda --example gpu_peer -- [MiB]` (default 1024).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
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

fn main() -> Res<()> {
    let mib: usize = std::env::args().nth(1).map_or(Ok(1024), |s| s.parse())?;
    let len = mib << 17;
    let n = gpu::device_count();
    println!("gpu_peer: {n} device(s), {mib} MiB per copy, {REPS} reps");
    let ctxs: Vec<Arc<CudaContext>> = (0..n).map(CudaContext::new).collect::<Result<_, _>>()?;
    let streams: Vec<Arc<CudaStream>> = ctxs.iter().map(|c| c.default_stream()).collect();
    let mut bufs: Vec<[CudaSlice<u64>; 2]> = streams
        .iter()
        .map(|s| Ok([s.alloc_zeros::<u64>(len)?, s.alloc_zeros::<u64>(len)?]))
        .collect::<Res<_>>()?;

    for d in 0..n {
        let [a, b] = &mut bufs[d];
        let same = dtod(&streams[d], a, b)?;
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
        println!("device {d}: same-device {same:.1} GB/s, pinned d2h {d2h:.1} GB/s, pinned h2d {h2d:.1} GB/s");
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
            let note = if access == PeerAccess::Enabled {
                ""
            } else {
                "  (host-staged)"
            };
            println!("{src} -> {dst}: {access:?}, {bw:.1} GB/s{note}");
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
