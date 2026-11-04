//! Operator-Lanczos probe: the in-place recursion `O_{n−1} ← i[H, O_n] − b_n·O_{n−1}` for a Heisenberg model with a Z field, on in-process partitions or one partition per MPI rank, printing one row per step and writing one JSON line per step and process.
//! Every step is one `add_liouvillian`, whose trace layer carries the exchange volume and the phases (`PartitionLayerRecord`).

use std::io::Write as _;
use std::time::Instant;

use paulistrings::{
    BuildAccumulator, CoefficientThreshold, PartitionConfig, PartitionLayerRecord,
    PartitionRowPolicy, PartitionRuntime, PartitionTrace, PartitionedSum, PauliString, PauliSum,
    Placement, ScatterOptions, ScatterRows,
};
use serde_json::json;

const USAGE: &str = "\
lanczos_probe: operator Lanczos on a Heisenberg model, one in-place add_liouvillian per step

  --model chain:L | square:WxH   lattice, open boundaries (default chain:40)
  --field F                      Z field on every site (default 0.3)
  --steps S                      Lanczos steps (default 14)
  --threshold T                  coefficient threshold per step (default 1e-12)
  --rows seeded | blocks | xblocks
                                 partition rows: seeded GF(2) rows, or one contiguous block of
                                 sites per partition (strips of rows on a square) labelled by the
                                 z-bits (`PartitionRows::cut`) or the x-bits of its sites; a Z
                                 field moves every z-labelled term, so xblocks is the local choice
                                 (default seeded)
  --partitions P                 in-process partitions, a power of two (default 1)
  --unpinned                     unpinned in-process partitions instead of one per NUMA node
  --threads T                    threads per partition (default: the placement's CPUs)
  --mpi                          one partition per MPI rank (needs the `mpi` feature)
  --json-out PATH                one JSON line per step; under --mpi each rank writes PATH.rank<N>
  --label NAME                   echoed into every JSON line
";

#[derive(Clone, Copy, Debug)]
enum Model {
    Chain(usize),
    Square(usize, usize),
}

impl Model {
    fn num_qubits(self) -> usize {
        match self {
            Model::Chain(l) => l,
            Model::Square(w, h) => w * h,
        }
    }

    /// Nearest-neighbour bonds, sites numbered row-major on a square.
    fn bonds(self) -> Vec<(usize, usize)> {
        match self {
            Model::Chain(l) => (0..l - 1).map(|j| (j, j + 1)).collect(),
            Model::Square(w, h) => {
                let mut bonds = Vec::new();
                for y in 0..h {
                    for x in 0..w {
                        let site = y * w + x;
                        if x + 1 < w {
                            bonds.push((site, site + 1));
                        }
                        if y + 1 < h {
                            bonds.push((site, site + w));
                        }
                    }
                }
                bonds
            }
        }
    }

    fn label(self) -> String {
        match self {
            Model::Chain(l) => format!("chain:{l}"),
            Model::Square(w, h) => format!("square:{w}x{h}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Rows {
    Seeded,
    Blocks,
    XBlocks,
}

struct Config {
    model: Model,
    field: f64,
    steps: usize,
    threshold: f64,
    rows: Rows,
    partitions: usize,
    unpinned: bool,
    threads: Option<usize>,
    mpi: bool,
    json_out: Option<String>,
    label: String,
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut cfg = Config {
        model: Model::Chain(40),
        field: 0.3,
        steps: 14,
        threshold: 1e-12,
        rows: Rows::Seeded,
        partitions: 1,
        unpinned: false,
        threads: None,
        mpi: false,
        json_out: None,
        label: String::new(),
    };
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        let number = |s: &str| s.parse::<f64>().map_err(|_| format!("{flag}: not a number: {s}"));
        let count = |s: &str| s.parse::<usize>().map_err(|_| format!("{flag}: not a count: {s}"));
        match flag.as_str() {
            "--model" => {
                let spec = value()?;
                cfg.model = if let Some(l) = spec.strip_prefix("chain:") {
                    Model::Chain(count(l)?)
                } else if let Some((w, h)) =
                    spec.strip_prefix("square:").and_then(|dims| dims.split_once('x'))
                {
                    Model::Square(count(w)?, count(h)?)
                } else {
                    return Err(format!("--model: expected chain:L or square:WxH, got {spec}"));
                };
            }
            "--field" => cfg.field = number(value()?)?,
            "--steps" => cfg.steps = count(value()?)?,
            "--threshold" => cfg.threshold = number(value()?)?,
            "--rows" => {
                cfg.rows = match value()?.as_str() {
                    "seeded" => Rows::Seeded,
                    "blocks" => Rows::Blocks,
                    "xblocks" => Rows::XBlocks,
                    other => {
                        return Err(format!("--rows: expected seeded, blocks or xblocks, got {other}"))
                    }
                }
            }
            "--partitions" => cfg.partitions = count(value()?)?,
            "--unpinned" => cfg.unpinned = true,
            "--threads" => cfg.threads = Some(count(value()?)?),
            "--mpi" => cfg.mpi = true,
            "--json-out" => cfg.json_out = Some(value()?.clone()),
            "--label" => cfg.label = value()?.clone(),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let n = cfg.model.num_qubits();
    if !(2..=64).contains(&n) {
        return Err(format!("{} has {n} sites; the probe takes 2 to 64", cfg.model.label()));
    }
    if !cfg.partitions.is_power_of_two() {
        return Err(format!("--partitions {} is not a power of two", cfg.partitions));
    }
    if cfg.mpi && cfg.partitions != 1 {
        return Err("--mpi takes one partition per rank; drop --partitions".into());
    }
    Ok(cfg)
}

/// The Hamiltonian and the seed `Z` on the central site, both real.
fn model<const W: usize>(cfg: &Config) -> (PauliSum<W, f64>, PauliSum<W, f64>) {
    let n = cfg.model.num_qubits();
    let mut h = BuildAccumulator::<W, f64>::new(n);
    for (i, j) in cfg.model.bonds() {
        for f in [PauliString::<W>::x, PauliString::<W>::y, PauliString::<W>::z] {
            let mut p = f(i);
            p.mul_assign(&f(j));
            h.add_term(p, 1.0);
        }
    }
    for j in 0..n {
        h.add_term(PauliString::<W>::z(j), cfg.field);
    }
    let centre = match cfg.model {
        Model::Chain(l) => l / 2,
        Model::Square(w, h) => (h / 2) * w + w / 2,
    };
    let mut seed = BuildAccumulator::<W, f64>::new(n);
    seed.add_term(PauliString::<W>::z(centre), 1.0);
    (h.finalize(), seed.finalize())
}

/// One contiguous run of sites per partition, in site order.
fn blocks(cfg: &Config, partitions: usize) -> Vec<Vec<usize>> {
    let n = cfg.model.num_qubits();
    (0..partitions).map(|p| (p * n / partitions..(p + 1) * n / partitions).collect()).collect()
}

fn row_policy(cfg: &Config, partitions: usize) -> PartitionRowPolicy {
    match cfg.rows {
        Rows::Seeded => PartitionRowPolicy::Seeded(Some(0x5EED)),
        Rows::Blocks => PartitionRowPolicy::Cut(blocks(cfg, partitions)),
        // Row `i` reads the x-bits of every block whose index has bit `i` set, as `cut` does with z-bits.
        Rows::XBlocks => {
            let n = cfg.model.num_qubits();
            let rows = (0..partitions.trailing_zeros())
                .map(|i| {
                    let mut flags = vec![false; 2 * n];
                    for (b, block) in blocks(cfg, partitions).iter().enumerate() {
                        if (b >> i) & 1 == 1 {
                            block.iter().for_each(|&q| flags[q] = true);
                        }
                    }
                    flags
                })
                .collect();
            PartitionRowPolicy::Matrix(rows)
        }
    }
}

fn runtime(cfg: &Config, partitions: usize) -> std::sync::Arc<PartitionRuntime> {
    let placement = if cfg.unpinned {
        Placement::Unpinned { partitions }
    } else {
        Placement::Auto { max_partitions: Some(partitions) }
    };
    let config = PartitionConfig { placement, bind_memory: true, threads_per_partition: cfg.threads };
    let runtime = PartitionRuntime::new(&config).unwrap_or_else(|err| {
        eprintln!("lanczos_probe: cannot resolve the placement: {err}");
        std::process::exit(2);
    });
    if runtime.num_partitions() != partitions {
        eprintln!(
            "lanczos_probe: {partitions} partitions resolved to {}; pass --unpinned, or run on a host with that many NUMA nodes",
            runtime.num_partitions()
        );
        std::process::exit(2);
    }
    runtime
}

/// The recursion's view of a held split, in one process or across ranks.
trait Krylov {
    fn step(&mut self, h: &PauliSum<1, f64>, current: &Self, beta: f64, policy: &CoefficientThreshold);
    fn norm(&self) -> f64;
    fn scale(&mut self, c: f64);
    fn total_len(&self) -> usize;
    fn take_layer(&mut self) -> PartitionLayerRecord;
    /// Sum `buffer` over the group; a no-op in one process, which holds every partition.
    fn sum_over_group(&self, buffer: &mut [u64]);
}

fn only_layer(trace: Option<PartitionTrace>) -> PartitionLayerRecord {
    let mut trace = trace.expect("tracing was enabled");
    assert_eq!(trace.layers.len(), 1, "one product per step");
    trace.layers.pop().expect("one layer")
}

impl Krylov for PartitionedSum<1, f64> {
    fn step(&mut self, h: &PauliSum<1, f64>, current: &Self, beta: f64, policy: &CoefficientThreshold) {
        self.add_liouvillian(h, current, 1.0, beta, policy);
    }
    fn norm(&self) -> f64 {
        self.overlap(self).sqrt()
    }
    fn scale(&mut self, c: f64) {
        PartitionedSum::scale(self, c);
    }
    fn total_len(&self) -> usize {
        self.len()
    }
    fn take_layer(&mut self) -> PartitionLayerRecord {
        only_layer(self.take_trace())
    }
    fn sum_over_group(&self, _buffer: &mut [u64]) {}
}

#[cfg(feature = "mpi")]
impl Krylov for paulistrings::DistributedSum<1, paulistrings::mpi::MpiTransport, f64> {
    fn step(&mut self, h: &PauliSum<1, f64>, current: &Self, beta: f64, policy: &CoefficientThreshold) {
        self.add_liouvillian(h, current, 1.0, beta, policy);
    }
    fn norm(&self) -> f64 {
        self.overlap(self).sqrt()
    }
    fn scale(&mut self, c: f64) {
        paulistrings::DistributedSum::scale(self, c);
    }
    fn total_len(&self) -> usize {
        self.len()
    }
    fn take_layer(&mut self) -> PartitionLayerRecord {
        only_layer(self.take_trace())
    }
    fn sum_over_group(&self, buffer: &mut [u64]) {
        use paulistrings::Collectives;
        self.transport().allreduce_sum_u64(buffer);
    }
}

fn peak_rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let line = status.lines().find_map(|line| line.strip_prefix("VmHWM:"));
    line.and_then(|rest| rest.split_whitespace().next()).and_then(|v| v.parse().ok()).unwrap_or(0)
}

/// The recursion, `previous` and `current` both holding the normalized seed, with `slot` this process's first partition and `group` the group's partition count.
fn recurse<S: Krylov>(
    cfg: &Config,
    h: &PauliSum<1, f64>,
    (mut previous, mut current): (S, S),
    (slot, group, rank): (usize, usize, Option<u32>),
) {
    let policy = CoefficientThreshold(cfg.threshold);
    let mut out = cfg.json_out.as_ref().map(|path| {
        let path = match rank {
            Some(rank) => format!("{path}.rank{rank}"),
            None => path.clone(),
        };
        std::fs::File::create(&path).unwrap_or_else(|err| {
            eprintln!("lanczos_probe: cannot write {path}: {err}");
            std::process::exit(2);
        })
    });
    let first = rank.unwrap_or(0) == 0;
    if first {
        println!(
            "# {} field {} threshold {:e}, {group} partition(s), {:?} rows",
            cfg.model.label(),
            cfg.field,
            cfg.threshold,
            cfg.rows
        );
        println!(
            "{:>4} {:>12} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9}",
            "step", "terms", "b", "step_ms", "prod_ms", "xchg_ms", "shipped", "imbal", "sent_GB",
            "rss_GB"
        );
    }
    let mut b: Vec<f64> = Vec::new();
    for step in 0..cfg.steps {
        let started = Instant::now();
        previous.step(h, &current, -b.last().copied().unwrap_or(0.0), &policy);
        let norm = previous.norm();
        previous.scale(1.0 / norm);
        let wall_ns = started.elapsed().as_nanos() as u64;
        b.push(norm);
        let layer = previous.take_layer();
        std::mem::swap(&mut previous, &mut current);

        // By slot: every partition's input terms, product time, export and exchange time, and peak RSS; then the group's sent rows and bytes.
        let mut buffer = vec![0u64; 4 * group + 2];
        for (k, &terms) in layer.terms_in.iter().enumerate() {
            buffer[slot + k] = terms as u64;
            buffer[group + slot + k] = layer.nanos[k];
            buffer[2 * group + slot + k] = layer.phases[k].export_ns + layer.phases[k].exchange_ns;
        }
        buffer[3 * group + slot] = peak_rss_kb();
        buffer[4 * group] = layer.rows_sent.iter().flatten().sum();
        buffer[4 * group + 1] = layer.bytes_sent.iter().flatten().sum();
        let terms = current.total_len();
        current.sum_over_group(&mut buffer);
        let terms_in = &buffer[..group];
        let input: u64 = terms_in.iter().sum();
        let mean = input as f64 / group as f64;
        let imbalance = if mean > 0.0 { *terms_in.iter().max().unwrap_or(&0) as f64 / mean } else { 1.0 };
        let product_ns = buffer[group..2 * group].iter().copied().max().unwrap_or(0);
        let exchange_ns = buffer[2 * group..3 * group].iter().copied().max().unwrap_or(0);
        let rss_kb: u64 = buffer[3 * group..4 * group].iter().sum();
        let (rows_sent, bytes_sent) = (buffer[4 * group], buffer[4 * group + 1]);
        let shipped = if input > 0 { rows_sent as f64 / input as f64 } else { 0.0 };
        if first {
            println!(
                "{:>4} {:>12} {:>9.4} {:>9.1} {:>9.1} {:>9.1} {:>8.3} {:>8.3} {:>9.3} {:>9.2}",
                step + 1,
                terms,
                norm,
                wall_ns as f64 / 1e6,
                product_ns as f64 / 1e6,
                exchange_ns as f64 / 1e6,
                shipped,
                imbalance,
                bytes_sent as f64 / 1e9,
                rss_kb as f64 / 1e6,
            );
        }
        if let Some(file) = out.as_mut() {
            let phases: Vec<_> = layer
                .phases
                .iter()
                .map(|p| {
                    json!({
                        "collective_ns": p.collective_ns, "rebucket_ns": p.rebucket_ns,
                        "prepare_ns": p.prepare_ns, "export_ns": p.export_ns,
                        "exchange_ns": p.exchange_ns, "accumulate_ns": p.coset_loop_ns,
                        "finalize_ns": p.finalize_ns,
                    })
                })
                .collect();
            let line = json!({
                "label": cfg.label, "model": cfg.model.label(), "field": cfg.field,
                "threshold": cfg.threshold, "rows": format!("{:?}", cfg.rows).to_lowercase(),
                "partitions": group, "rank": rank, "step": step + 1, "b": norm,
                "terms": terms, "terms_in_group": input, "wall_ns": wall_ns,
                "product_ns_max": product_ns, "export_exchange_ns_max": exchange_ns, "shipped": shipped, "imbalance": imbalance,
                "rows_sent_group": rows_sent, "bytes_sent_group": bytes_sent,
                "rss_kb_group": rss_kb, "bits": layer.bits, "remote_deltas": layer.remote_deltas,
                "terms_in": layer.terms_in, "terms_out": layer.terms_out,
                "nanos": layer.nanos, "rows_sent": layer.rows_sent,
                "bytes_sent": layer.bytes_sent, "rows_received": layer.rows_received,
                "phases": phases, "vmhwm_kb": peak_rss_kb(),
            });
            writeln!(file, "{line}").expect("the sidecar is writable");
        }
    }
}

fn normalized(mut seed: PauliSum<1, f64>) -> PauliSum<1, f64> {
    let norm = seed.overlap(&seed).sqrt();
    seed.scale(1.0 / norm);
    seed
}

fn in_process(cfg: &Config) {
    let (h, seed) = model::<1>(cfg);
    let seed = normalized(seed);
    let p = cfg.partitions;
    let runtime = runtime(cfg, p);
    let scatter = |sum: PauliSum<1, f64>| {
        let rows = ScatterRows::Policy(row_policy(cfg, p));
        PartitionedSum::scatter_with(sum, ScatterOptions { runtime: runtime.clone(), rows })
    };
    let (mut previous, current) = (scatter(seed.clone()), scatter(seed));
    previous.enable_trace();
    let mut current = current;
    current.enable_trace();
    recurse(cfg, &h, (previous, current), (0, p, None));
}

#[cfg(feature = "mpi")]
fn over_ranks(cfg: &Config) {
    use paulistrings::mpi::{rsmpi, MpiTransport};
    use paulistrings::DistributedSum;
    use rsmpi::traits::Communicator;

    let (universe, threading) = rsmpi::initialize_with_threading(rsmpi::Threading::Serialized)
        .expect("MPI is initialized once");
    if threading < rsmpi::Threading::Serialized {
        eprintln!("lanczos_probe: MPI provided {threading:?}, below the SERIALIZED the engine needs");
    }
    let world = universe.world();
    let (rank, size) = (world.rank() as u32, world.size() as usize);
    let (h, seed) = model::<1>(cfg);
    let seed = normalized(seed);
    let runtime = runtime(cfg, 1);
    let scatter = |sum: PauliSum<1, f64>| {
        let rows = ScatterRows::Policy(row_policy(cfg, size));
        let options = ScatterOptions { runtime: runtime.clone(), rows };
        let mut split =
            DistributedSum::scatter_with(sum, MpiTransport::from_communicator(&world), options);
        split.enable_trace();
        split
    };
    let pair = (scatter(seed.clone()), scatter(seed));
    recurse(cfg, &h, pair, (rank as usize, size, Some(rank)));
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return;
    }
    let cfg = parse_args(&args).unwrap_or_else(|msg| {
        eprintln!("lanczos_probe: {msg}\n\n{USAGE}");
        std::process::exit(2);
    });
    if cfg.mpi {
        #[cfg(feature = "mpi")]
        over_ranks(&cfg);
        #[cfg(not(feature = "mpi"))]
        {
            eprintln!("lanczos_probe: --mpi needs the `mpi` cargo feature");
            std::process::exit(2);
        }
    } else {
        in_process(&cfg);
    }
}
