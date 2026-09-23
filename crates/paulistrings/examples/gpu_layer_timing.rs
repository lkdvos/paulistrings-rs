//! Kernel timings of the fused device layer: warm-up growth layer, then medians over repeated applications on the saturated sum.
//! `cargo run --release --features cuda,test-utils --example gpu_layer_timing -- [--n 1000000] [--reps 5] [--cells su4,cnot,rotation_zz] [--policy records|fixed]`

use std::time::Instant;

use paulistrings::channel::{Channel, Clifford2Q, GeneralUnitary2Q};
use paulistrings::gpu::{
    cuda_available, GpuBucketPolicy, GpuLayerOptions, GpuPauliSum, DEFAULT_ARENA_BYTES,
};
use paulistrings::test_support::{haar_su4_matrix, rand_sum, zz_rotation, KeepAll};
use paulistrings::{Circuit, Direction};

const QUBITS: usize = 128;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn cell(name: &str) -> Box<dyn Channel<2>> {
    match name {
        "su4" => Box::new(GeneralUnitary2Q::from_matrix(0, 1, haar_su4_matrix())),
        "cnot" => Box::new(Clifford2Q::cnot(0, 1)),
        "rotation_zz" => Box::new(zz_rotation::<2>(0, 1, 0.1)),
        other => panic!("unknown cell {other}"),
    }
}

fn used_gb() -> f64 {
    let ctx = cudarc::driver::CudaContext::new(0).expect("context");
    let (free, total) = ctx.mem_get_info().expect("mem info");
    (total - free) as f64 / 1e9
}

fn main() {
    if !cuda_available() {
        eprintln!("no CUDA device");
        return;
    }
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str, default: &str| -> String {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
            .unwrap_or_else(|| default.to_string())
    };
    let n: usize = get("--n", "1000000").parse().expect("--n");
    let reps: usize = get("--reps", "5").parse().expect("--reps");
    let cells = get("--cells", "su4,cnot,rotation_zz");
    let policy = match get("--policy", "records").as_str() {
        "records" => GpuBucketPolicy::default(),
        "fixed" => GpuBucketPolicy::TermsPerBucket(256),
        other => panic!("unknown policy {other}"),
    };
    println!(
        "W=2 qubits={QUBITS} n={n} reps={reps} policy={policy:?} arena={:.1} GB",
        DEFAULT_ARENA_BYTES as f64 / 1e9
    );
    println!("| cell | m_in | m_steady | bits | n_cap | batches | dense | K1 | K2 | K3 | K4 | refine | kernel total ms | wall ms | ns/steady term (kernel) | ns/steady term (wall) | peak GB | fallback hi/key |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for name in cells.split(',') {
        let ch = cell(name);
        let mut circuit = Circuit::<2>::new(QUBITS);
        circuit.channels.push(ch);
        let input = rand_sum::<2>(n, QUBITS, 0xCAFE);
        let mut dev = GpuPauliSum::from_host(&input, 0).expect("upload");
        dev.set_layer_options(GpuLayerOptions {
            bucket_policy: policy,
            arena_bytes: DEFAULT_ARENA_BYTES,
        });
        dev.propagate(&circuit, &KeepAll, Direction::Forward)
            .expect("growth layer");
        let steady = dev.len();
        dev.set_kernel_timing(true);
        let _ = dev.take_kernel_ms();
        let mut kernel = Vec::new();
        let mut wall = Vec::new();
        let mut parts = Vec::new();
        let mut peak = used_gb();
        for _ in 0..reps {
            dev.set_kernel_timing(false);
            let t0 = Instant::now();
            dev.propagate(&circuit, &KeepAll, Direction::Forward)
                .expect("steady layer");
            wall.push(t0.elapsed().as_secs_f64() * 1e3);
            dev.set_kernel_timing(true);
            dev.propagate(&circuit, &KeepAll, Direction::Forward)
                .expect("timed layer");
            let ms = dev.take_kernel_ms();
            kernel.push(ms.count + ms.sizes + ms.layer + ms.compact + ms.refine);
            parts.push(ms);
            peak = peak.max(used_gb());
        }
        let c = dev.last_layer_counters();
        let k = median(&mut kernel);
        let w = median(&mut wall);
        let med = |f: fn(&paulistrings::gpu::GpuKernelMs) -> f64| {
            let mut v: Vec<f64> = parts.iter().map(f).collect();
            median(&mut v)
        };
        println!(
            "| {name} | {n} | {steady} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {k:.2} | {w:.2} | {:.2} | {:.2} | {peak:.2} | {}/{} |",
            c.bits,
            c.n_cap,
            c.batches,
            c.dense,
            med(|m| m.count),
            med(|m| m.sizes),
            med(|m| m.layer),
            med(|m| m.compact),
            med(|m| m.refine),
            k * 1e6 / steady as f64,
            w * 1e6 / steady as f64,
            c.fallback_hi,
            c.fallback_key,
        );
    }
}
