//! The NCCL bring-up probe: the communicator init and warm-up `gpu::MpiGpuSum` performs at scatter, one variant per process, every wait bounded, one line per rank.
//! `mpirun -n N nccl_probe --variant <name> [--shape all|per-peer|ring] [--blocking] [--device local-rank|comm|<csv>] [--timeout-s S] [--teardown-s S]`
//! Built with `--features nccl,test-utils`; `scripts/slurm/nccl-probe.sbatch` runs the variant matrix, one bounded `srun` step each.

use std::io::Write;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use paulistrings::engine::partitioned::Collectives;
use paulistrings::gpu::{
    local_device_for_comm, local_device_for_rank, nccl_available, pending_aborts, NcclComm,
    WarmUpShape,
};
use paulistrings::mpi::{rsmpi, MpiTransport};
use rsmpi::topology::Communicator;

const USAGE: &str = "usage: nccl_probe [--variant NAME] [--shape all|per-peer|ring] [--blocking] \
                     [--device local-rank|comm|D0,D1,...] [--timeout-s S] [--teardown-s S]";

/// NCCL and CUDA variables worth seeing beside a rank's outcome.
const KNOBS: [&str; 9] = [
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_CUMEM_ENABLE",
    "NCCL_PROTO",
    "NCCL_RUNTIME_CONNECT",
    "NCCL_SHM_DISABLE",
    "NCCL_LAUNCH_MODE",
    "CUDA_VISIBLE_DEVICES",
    "PAULISTRINGS_NCCL_TIMEOUT_S",
];

enum DevicePick {
    LocalRank,
    Comm,
    Explicit(Vec<u32>),
}

struct Args {
    variant: String,
    shape: WarmUpShape,
    blocking: bool,
    device: DevicePick,
    timeout: Duration,
    teardown: Duration,
}

fn usage(msg: &str) -> ! {
    eprintln!("nccl_probe: {msg}\n{USAGE}");
    std::process::exit(2)
}

fn parse(mut it: impl Iterator<Item = String>) -> Args {
    let env_timeout = std::env::var("PAULISTRINGS_NCCL_TIMEOUT_S")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| *s > 0.0)
        .unwrap_or(300.0);
    let mut a = Args {
        variant: "probe".to_string(),
        shape: WarmUpShape::AllPeers,
        blocking: false,
        device: DevicePick::LocalRank,
        timeout: Duration::from_secs_f64(env_timeout),
        teardown: Duration::from_secs(30),
    };
    let value = |flag: &str, it: &mut dyn Iterator<Item = String>| -> String {
        it.next()
            .unwrap_or_else(|| usage(&format!("{flag} needs a value")))
    };
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--variant" => a.variant = value(&flag, &mut it),
            "--shape" => {
                a.shape = match value(&flag, &mut it).as_str() {
                    "all" => WarmUpShape::AllPeers,
                    "per-peer" => WarmUpShape::PerPeer,
                    "ring" => WarmUpShape::Ring,
                    other => usage(&format!("unknown shape {other}")),
                }
            }
            "--blocking" => a.blocking = true,
            "--device" => {
                a.device = match value(&flag, &mut it).as_str() {
                    "local-rank" => DevicePick::LocalRank,
                    "comm" => DevicePick::Comm,
                    csv => DevicePick::Explicit(
                        csv.split(',')
                            .map(|d| {
                                d.trim()
                                    .parse::<u32>()
                                    .unwrap_or_else(|_| usage(&format!("bad device list {csv}")))
                            })
                            .collect(),
                    ),
                }
            }
            "--timeout-s" => {
                a.timeout = Duration::from_secs_f64(
                    value(&flag, &mut it)
                        .parse()
                        .unwrap_or_else(|_| usage("--timeout-s needs seconds")),
                )
            }
            "--teardown-s" => {
                a.teardown = Duration::from_secs_f64(
                    value(&flag, &mut it)
                        .parse()
                        .unwrap_or_else(|_| usage("--teardown-s needs seconds")),
                )
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0)
            }
            other => usage(&format!("unknown argument {other}")),
        }
    }
    a
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "?".to_string())
}

fn cpus_allowed() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "?".to_string())
}

fn knobs() -> String {
    let set: Vec<String> = KNOBS
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| format!("{k}={v}")))
        .collect();
    if set.is_empty() {
        String::new()
    } else {
        format!(" env={}", set.join(","))
    }
}

fn ms(since: Instant) -> String {
    format!("{:.0}", since.elapsed().as_secs_f64() * 1e3)
}

fn quoted<T: std::fmt::Display>(r: &Result<(), T>) -> String {
    match r {
        Ok(()) => "ok".to_string(),
        Err(e) => format!("\"{e}\""),
    }
}

fn flush() {
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
}

/// One all-reduce of every rank's failure flag: the ranks that failed, in the same order everywhere.
fn failed_ranks(coll: &dyn Collectives, failed: bool) -> Vec<usize> {
    let mut flags = vec![0u64; coll.size() as usize];
    flags[coll.rank() as usize] = u64::from(failed);
    coll.allreduce_sum_u64(&mut flags);
    flags
        .iter()
        .enumerate()
        .filter(|(_, &f)| f != 0)
        .map(|(r, _)| r)
        .collect()
}

fn main() {
    let args = parse(std::env::args().skip(1));
    let Some((universe, _)) = rsmpi::initialize_with_threading(rsmpi::Threading::Serialized) else {
        eprintln!("nccl_probe: MPI is already initialized in this process");
        std::process::exit(2);
    };
    let world = universe.world();
    let (rank, size) = (world.rank() as u32, world.size() as u32);
    let tag = format!(
        "nccl_probe {} rank={rank}/{size} host={}",
        args.variant,
        hostname()
    );
    if rank == 0 {
        println!(
            "nccl_probe {}: {size} rank(s), shape {:?}, blocking {}, wait bound {:?}, teardown bound {:?}, libnccl {}",
            args.variant,
            args.shape,
            args.blocking,
            args.timeout,
            args.teardown,
            if nccl_available() { "present" } else { "missing" }
        );
    }
    let transport = MpiTransport::from_communicator(&world);

    let picked = match &args.device {
        DevicePick::LocalRank => local_device_for_rank(rank),
        DevicePick::Comm => local_device_for_comm(&world),
        DevicePick::Explicit(list) => Ok(list[rank as usize % list.len()]),
    };
    let ctx = picked.map_err(|e| e.to_string()).and_then(|d| {
        CudaContext::new(d as usize)
            .map(|c| (d, c))
            .map_err(|e| e.to_string())
    });
    let bad = failed_ranks(&transport, ctx.is_err());
    let (device, ctx) = match ctx {
        Ok(pair) if bad.is_empty() => pair,
        Ok(_) => {
            println!("{tag} device=ok peers_without_device={bad:?}");
            flush();
            std::process::exit(1);
        }
        Err(e) => {
            println!("{tag} device=none error=\"{e}\"");
            flush();
            std::process::exit(1);
        }
    };

    let t = Instant::now();
    let comm = NcclComm::init_with(&transport, &ctx, args.timeout, args.blocking);
    let init_ms = ms(t);
    let init = quoted(&comm.as_ref().map(|_| ()));
    let stream = ctx.new_stream().unwrap_or_else(|e| {
        println!("{tag} device={device} stream=\"{e}\"");
        flush();
        std::process::exit(1)
    });
    let t = Instant::now();
    let warm = comm.as_ref().map(|c| c.warm_up_shape(&stream, args.shape));
    let warm_ms = ms(t);
    let warm_str = match &warm {
        Ok(r) => quoted(r),
        Err(_) => "skipped".to_string(),
    };
    let failed_here = !matches!(warm, Ok(Ok(())));
    let failed = failed_ranks(&transport, failed_here);
    // As the engine's bootstrap does: a communicator whose peers failed is aborted, never finalized against them.
    if let (Ok(c), false) = (&comm, failed.is_empty()) {
        c.abort();
    }
    println!(
        "{tag} device={device} cpus={} shape={:?} blocking={} init={init} init_ms={init_ms} warmup={warm_str} warmup_ms={warm_ms} failed_ranks={failed:?}{}",
        cpus_allowed(),
        args.shape,
        args.blocking,
        knobs()
    );
    if rank == 0 {
        println!(
            "nccl_probe {} summary: {} of {size} rank(s) failed{}",
            args.variant,
            failed.len(),
            if failed.is_empty() {
                String::new()
            } else {
                format!(" ({failed:?})")
            }
        );
    }
    flush();
    // MPI is done with before the teardown, so a rank whose teardown sticks holds nobody else.
    drop(transport);
    drop(world);
    drop(universe);

    let t = Instant::now();
    let (tx, rx) = mpsc::channel();
    let owned = ctx.clone();
    std::thread::spawn(move || {
        let _ = owned.bind_to_thread();
        drop(comm);
        drop(stream);
        let _ = tx.send(());
    });
    let teardown = match rx.recv_timeout(args.teardown) {
        Ok(()) => "ok",
        Err(_) => "stuck",
    };
    println!(
        "{tag} teardown={teardown} teardown_ms={} aborts_pending={}",
        ms(t),
        pending_aborts()
    );
    flush();
    let code = i32::from(!failed.is_empty());
    if teardown == "stuck" {
        // No exit handlers behind a driver call that has not returned.
        #[cfg(target_os = "linux")]
        // SAFETY: `_exit` ends the process without touching any Rust state.
        unsafe {
            libc::_exit(3)
        }
        #[cfg(not(target_os = "linux"))]
        std::process::exit(3)
    }
    std::process::exit(code)
}
