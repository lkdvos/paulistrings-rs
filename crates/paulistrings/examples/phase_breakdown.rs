//! Per-phase timing / memory probe for the bucketed propagation engine.
//!
//! Drives [`propagate_with_scratch_and_options`] over a menu of layers (rotation, Clifford,
//! general-unitary, noise, Trotter, and partitioned/distributed variants) across a matrix of
//! thread and partition counts, and prints the [`PhaseStats`] breakdown the `phase-timing`
//! feature exposes.
//!
//! ```bash
//! cargo run --release --features phase-timing --example phase_breakdown -- --help
//! ```

use std::time::Instant;

use num_complex::Complex64;
use paulistrings::bucket::hash::B_MAX_BITS;
use paulistrings::bucket::sum::{
    DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
};
use paulistrings::channel::{Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation};
use paulistrings::engine::partitioned::{
    circuit_generators, count_remote_deltas, CpuSet, GeneratorWeight, PartitionConfig,
    PartitionPhaseStats, PartitionRuntime, PartitionTrace, PartitionedSum, PartitionedTruncation,
    Placement, BITS_AGREE_EVERY,
};
use paulistrings::engine::stats::TIMER_READ_OVERHEAD_NS;
use paulistrings::test_support::{haar_su4_matrix, low_weight_sum, rand_sum};
use paulistrings::truncation::{ApproxTopN, CoefficientThreshold, TopN};
use paulistrings::{
    propagate_with_scratch_and_options, BuildAccumulator, Circuit, Direction, Gf2Hash,
    LayerScratch, PartitionRows, PauliString, PauliSum, Phase, PhaseStats, PropagateOptions,
    TruncationPolicy,
};

const USAGE: &str = "\
Usage: phase_breakdown [OPTIONS]

Measures the phase-timing breakdown of propagate_with_scratch_and_options across a
menu of layers and thread counts.

Options:
  --n <usize>              Target input term count (default: 1000000)
  --qubits <usize>         Qubit count; picks W = ceil(qubits/64), W in {1,2}
                            (default: 128)
  --threads <csv>          Comma-separated thread counts (default: 1,8,16,32;
                            16 = the reference host's physical-core count)
  --layers <csv>           Comma-separated layers, from:
                              rotation_zz, rotation_local, rotation_remote,
                              cnot, gu2q, su4, depolarizing, trotter,
                              tfim_step, heavyhex_step
                            (default: rotation_zz,cnot,gu2q,depolarizing,trotter
                            — su4 is opt-in, being the heaviest cell per --n:
                            a dense 16x16 PTM, so ~16x the fanout of gu2q's
                            sqrt(SWAP) and a ~16x larger closed key set)
  --reps <usize>           Channel repetitions per cell, ignored by trotter;
                            Trotter STEPS for tfim_step / heavyhex_step
                            (default: 8)
                            NOTE: trotter also ignores --n above 100 (see
                            TROTTER_MAX_N in the source) — it is 64 distinct
                            generators under no truncation, so growth is
                            combinatorial rather than bounded; an uncapped
                            large --n has been measured driving it to tens
                            of GB of RSS.
  --seed <u64|0xHEX>       RNG seed for the input sum (default: 0xC0FFEE)
  --hash-seed <u64|0xHEX>  Seed for the partitioning hash H (default: the
                            library's DEFAULT_HASH_SEED). Changing it re-draws
                            H's rows, which changes the rank of the layer's
                            bucket-delta span and therefore the coset
                            dimension `r` — see
                            research/FINDINGS.md.
  --bucket-bits <u8>       Pre-refine the input sum to this many bucket bits
                            (B = 2^bits) before propagating. 0 (default)
                            leaves the sum as built, i.e. the engine's own
                            `desired_bits` policy alone decides B. Because
                            `rebucket` is grow-only, a value above what the
                            policy would pick sticks for the whole run: this
                            is the knob that measures what raising
                            `desired_bits`'s parallelism floor would buy,
                            without changing the policy.
  --target-bucket-len <n>  Terms per bucket the engine's per-layer partition
                            targets (default: 1024, the library default).
  --min-buckets <n>        Floor on the per-layer bucket count once the sum is
                            worth splitting (default: 128, the library
                            default; must be >= 16).
                            NOTE: --bucket-bits pre-refines the *input* sum
                            and, `rebucket` being grow-only, can only ADD
                            buckets. These two change the engine's own
                            per-layer policy and are the only way to get
                            FEWER. Both have to move together: above the
                            floor, raising --target-bucket-len alone is inert.
  --truncation <spec>      Truncation policy for every cell, one of:
                              keep          no truncation (default)
                              coeff:<t>     CoefficientThreshold(t): a
                                            keep_term filter in the merge, no
                                            finalize_layer pass
                              topn:<N>      TopN(N): all cost in
                                            finalize_layer. N is absolute and
                                            must be BELOW the cell's
                                            steady-state term count, else
                                            TopN returns immediately.
                                            REJECTED for a partitioned cell:
                                            no PartitionedTruncation impl.
                              atopn:<N>     ApproxTopN(N): the same, with a
                                            histogram threshold instead of a
                                            selection. Keeps <= N, so its
                                            steady-state m differs from
                                            topn:<N>'s -- compare per-term.
  --partitions <csv>       Comma-separated partition counts, each a power of
                            two (default: 1). P > 1 runs the cell through the
                            partitioned engine; P = 1 runs the unpartitioned
                            one.
                            NOTE: --threads is the TOTAL thread count, so each
                            partition's pool gets threads/P workers. Every
                            --threads value must be divisible by every
                            --partitions value.
  --partition-cpus <spec>  Placement for P > 1, one of:
                              auto       (default) one partition per NUMA node
                                         in the affinity mask, capped at P
                              <l>;<l>    Placement::Explicit, one Linux cpulist
                                         per partition '0-7,16-23;8-15,24-31';
                                         the count must equal every P > 1
                              unpinned   Placement::Unpinned: partition shape,
                                         no pinning (laptops, shared boxes)
                            A partitioned cell must run under NO external
                            placement prefix (taskset/numactl) -- see the note
                            at the top of scripts/host-topology.sh.
  --bind-memory 0|1        Bind each partition's allocations to its NUMA node
                            (default: 1). 0 keeps the CPU pinning and drops
                            the memory policy.
  --partition-seed <u64|0xHEX>
                           Seed picking the GF(2) partition rows. Default: the
                            driver's own choice (the sum's hash seed). Only
                            --partition-rows random reads it.
  --partition-rows <spec>  How the log2(P) partition rows are chosen:
                              random  (default) PartitionRows::from_seed, i.e.
                                      the driver's own draw -- roughly half of
                                      a two-qubit generator's deltas cross
                              cut     log2(P) z-only rows labelling P
                                      contiguous qubit blocks: every
                                      single-qubit rotation is local, and a
                                      ZZ(i,j) is remote iff the edge crosses
                                      the cut. The blocks minimise crossed
                                      edges (of the layer's own lattice: a
                                      chain for tfim_step, the heavy-hex map
                                      for heavyhex_step, none -- so an even
                                      index split -- otherwise) subject to
                                      +-25% size balance; the cut and its
                                      crossing count are echoed to stderr
                            cut rows are low weight, so they collide with H's
                            own rows far more often than a random draw: they
                            are checked with is_independent_of against the
                            first 10 rows H will grow into, and the hash is
                            re-seeded until they pass (said so on stderr, since
                            it moves the coset dimension too).
  --initial random|z0      The cell's input sum before the warm-up call:
                              random  rand_sum(--n, --qubits, --seed), the
                                      dense steady-state input; the default
                                      for every layer but the two Trotter
                                      steps
                              z0      one term, Z on qubit --qubits/2, whose
                                      support and term count then grow step by
                                      step; the default for tfim_step and
                                      heavyhex_step, and the input under which
                                      the partition imbalance of a cut row
                                      means anything (a dense random sum is
                                      balanced under any row by construction)
                            --n is not a size under z0, and trotter ignores
                            this flag entirely (it keeps its low-weight input).
  --format table|json|tsv  Output format (default: table)
  --json-out FILE          Also append one JSON line per cell to FILE,
                           regardless of --format (input for scripts/perf-viz.py)
  -h, --help               Print this message
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerKind {
    RotationZz,
    /// A `ZZ` rotation whose deltas all stay inside their partition; the same cell as `RotationZz` at `P = 1`.
    RotationLocal,
    /// A `ZZ` rotation with at least one delta crossing partitions every layer.
    RotationRemote,
    Cnot,
    Gu2q,
    Su4,
    /// Haar SU(4) on a pair whose deltas are all local under the partition rows.
    Su4Local,
    Depolarizing,
    Trotter,
    /// One kicked-Ising Trotter step on an open chain; see [`tfim_step_circuit`].
    TfimStep,
    /// One kicked-Ising Trotter step on the 127-qubit heavy-hex lattice; see [`heavy_hex_step_circuit`].
    HeavyHexStep,
}

impl LayerKind {
    fn name(self) -> &'static str {
        match self {
            LayerKind::RotationZz => "rotation_zz",
            LayerKind::RotationLocal => "rotation_local",
            LayerKind::RotationRemote => "rotation_remote",
            LayerKind::Cnot => "cnot",
            LayerKind::Gu2q => "gu2q",
            LayerKind::Su4 => "su4",
            LayerKind::Su4Local => "su4_local",
            LayerKind::Depolarizing => "depolarizing",
            LayerKind::Trotter => "trotter",
            LayerKind::TfimStep => "tfim_step",
            LayerKind::HeavyHexStep => "heavyhex_step",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "rotation_zz" => Ok(LayerKind::RotationZz),
            "rotation_local" => Ok(LayerKind::RotationLocal),
            "rotation_remote" => Ok(LayerKind::RotationRemote),
            "cnot" => Ok(LayerKind::Cnot),
            "gu2q" => Ok(LayerKind::Gu2q),
            "su4" => Ok(LayerKind::Su4),
            "su4_local" => Ok(LayerKind::Su4Local),
            "depolarizing" => Ok(LayerKind::Depolarizing),
            "trotter" => Ok(LayerKind::Trotter),
            "tfim_step" => Ok(LayerKind::TfimStep),
            "heavyhex_step" => Ok(LayerKind::HeavyHexStep),
            other => Err(format!(
                "unknown layer '{other}' (expected one of: rotation_zz, rotation_local, \
                 rotation_remote, cnot, gu2q, su4, su4_local, depolarizing, trotter, tfim_step, \
                 heavyhex_step)"
            )),
        }
    }

    /// The two Trotter-step workloads, whose input grows from a single-site `Z` observable.
    fn is_trotter_step(self) -> bool {
        matches!(self, LayerKind::TfimStep | LayerKind::HeavyHexStep)
    }

    /// The layer's two-qubit generator graph for `--partition-rows cut`; empty when it has none.
    fn cut_edges(self, num_qubits: usize) -> Vec<(u32, u32)> {
        match self {
            LayerKind::TfimStep => chain_edges(num_qubits),
            LayerKind::HeavyHexStep => heavy_hex_127_edges(),
            _ => Vec::new(),
        }
    }

    /// Whether the layer's generator qubits are chosen per cell ([`choose_generator`]) rather than fixed at `(0, 1)`.
    fn picks_generator(self) -> bool {
        matches!(
            self,
            LayerKind::RotationLocal | LayerKind::RotationRemote | LayerKind::Su4Local
        )
    }
}

fn default_layers() -> Vec<LayerKind> {
    vec![
        LayerKind::RotationZz,
        LayerKind::Cnot,
        LayerKind::Gu2q,
        LayerKind::Depolarizing,
        LayerKind::Trotter,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Table,
    Json,
    Tsv,
}

/// Which [`TruncationPolicy`] every cell runs under, kept as a spec so `run` dispatches it into one monomorphization per variant.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TruncSpec {
    Keep,
    Coeff(f64),
    TopN(usize),
    ApproxTopN(usize),
}

impl TruncSpec {
    /// The spec as it was written on the command line, echoed into every output format.
    fn label(self) -> String {
        match self {
            TruncSpec::Keep => "keep".to_string(),
            TruncSpec::Coeff(t) => format!("coeff:{t}"),
            TruncSpec::TopN(n) => format!("topn:{n}"),
            TruncSpec::ApproxTopN(n) => format!("atopn:{n}"),
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        let t = s.trim();
        if t == "keep" {
            return Ok(TruncSpec::Keep);
        }
        if let Some(v) = t.strip_prefix("coeff:") {
            let thr = v.parse::<f64>().map_err(|_| {
                format!("--truncation coeff:<t> expects a float threshold, got '{v}'")
            })?;
            if !thr.is_finite() || thr < 0.0 {
                return Err(format!(
                    "--truncation coeff:<t> expects a finite, non-negative threshold, got '{v}'"
                ));
            }
            return Ok(TruncSpec::Coeff(thr));
        }
        if let Some(v) = t.strip_prefix("topn:") {
            return Ok(TruncSpec::TopN(parse_usize(v, "--truncation topn:<N>")?));
        }
        if let Some(v) = t.strip_prefix("atopn:") {
            return Ok(TruncSpec::ApproxTopN(parse_usize(
                v,
                "--truncation atopn:<N>",
            )?));
        }
        Err(format!(
            "--truncation expects keep | coeff:<t> | topn:<N> | atopn:<N>, got '{s}'"
        ))
    }
}

/// `--partition-cpus`, kept as the spec so it can be echoed into the sidecar and turned into a [`Placement`] once per cell.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PartitionCpus {
    /// One partition per NUMA node in the mask, capped at the cell's `P`.
    Auto,
    /// No pinning at all — the partition shape without its placement.
    Unpinned,
    /// One Linux cpulist per partition, exactly as written.
    Explicit(Vec<String>),
}

impl PartitionCpus {
    /// The spec as written, echoed into every output format.
    fn label(&self) -> String {
        match self {
            PartitionCpus::Auto => "auto".to_string(),
            PartitionCpus::Unpinned => "unpinned".to_string(),
            PartitionCpus::Explicit(lists) => lists.join(";"),
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        let t = s.trim();
        if t == "auto" {
            return Ok(PartitionCpus::Auto);
        }
        if t == "unpinned" {
            return Ok(PartitionCpus::Unpinned);
        }
        let lists: Vec<String> = t
            .split(';')
            .map(str::trim)
            .filter(|tok| !tok.is_empty())
            .map(str::to_string)
            .collect();
        if lists.is_empty() {
            return Err(
                "--partition-cpus expects auto | unpinned | '<cpulist>;<cpulist>;...', got an \
                 empty list"
                    .to_string(),
            );
        }
        for list in &lists {
            CpuSet::parse(list).map_err(|err| format!("--partition-cpus '{list}': {err}"))?;
        }
        Ok(PartitionCpus::Explicit(lists))
    }

    /// The [`Placement`] for one cell: `partitions` partitions sharing `threads` workers in total.
    fn placement(&self, partitions: usize, threads: usize) -> Placement {
        match self {
            PartitionCpus::Auto => Placement::Auto {
                max_partitions: Some(partitions),
            },
            PartitionCpus::Unpinned => Placement::Unpinned {
                partitions,
                threads_per_partition: Some((threads / partitions).max(1)),
            },
            PartitionCpus::Explicit(lists) => Placement::Explicit(
                lists
                    .iter()
                    .map(|list| CpuSet::parse(list).expect("validated at parse time"))
                    .collect(),
            ),
        }
    }
}

/// What a cell's input sum is, before the warm-up call: a dense steady-state sum, or a single observable that grows step by step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Initial {
    /// `test_support::rand_sum(--n, --qubits, --seed)`.
    Random,
    /// A single `Z` on qubit `--qubits / 2`, coefficient 1. `--n` is then not a size at all.
    Z0,
}

impl Initial {
    fn label(self) -> &'static str {
        match self {
            Initial::Random => "random",
            Initial::Z0 => "z0",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "random" => Ok(Initial::Random),
            "z0" => Ok(Initial::Z0),
            other => Err(format!("--initial expects random | z0, got '{other}'")),
        }
    }

    /// The input a layer takes when `--initial` is absent.
    fn default_for(layer: LayerKind) -> Self {
        if layer.is_trotter_step() {
            Initial::Z0
        } else {
            Initial::Random
        }
    }
}

/// `--partition-rows`: how the `log2(P)` GF(2) partition rows are chosen (`ARCHITECTURE.md §Partitioning`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartitionRowSpec {
    /// `PartitionRows::from_seed`, the driver's own default.
    Random,
    /// `log2(P)` z-only rows labelling `P` contiguous qubit blocks. See [`cut_rows`].
    Cut,
}

impl PartitionRowSpec {
    fn label(self) -> &'static str {
        match self {
            PartitionRowSpec::Random => "random",
            PartitionRowSpec::Cut => "cut",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "random" => Ok(PartitionRowSpec::Random),
            "cut" => Ok(PartitionRowSpec::Cut),
            other => Err(format!(
                "--partition-rows expects random | cut, got '{other}'"
            )),
        }
    }
}

struct Config {
    n: usize,
    qubits: usize,
    /// Seed for the partitioning hash `H`. See `--hash-seed`.
    hash_seed: u64,
    /// Bucket bits to pre-refine the input sum to; 0 = leave it to the engine's own policy.
    bucket_bits: u8,
    /// Engine's per-layer target terms per bucket. See `--target-bucket-len`.
    target_bucket_len: usize,
    /// Engine's per-layer bucket-count floor. See `--min-buckets`.
    min_buckets: usize,
    /// TOTAL thread counts; a partitioned cell splits one of these over its partitions.
    threads: Vec<usize>,
    /// Partition counts to sweep. See `--partitions`.
    partitions: Vec<usize>,
    /// `--mpi`: run each cell on the distributed driver, one partition per rank. Only settable
    /// when the `mpi` feature is on.
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    mpi: bool,
    /// Placement spec for the partitioned cells. See `--partition-cpus`.
    partition_cpus: PartitionCpus,
    /// Whether each partition binds its allocations to its NUMA node.
    bind_memory: bool,
    /// Seed for the partition rows, or `None` for the driver's own choice.
    partition_seed: Option<u64>,
    /// How the partition rows are chosen. See `--partition-rows`.
    partition_rows: PartitionRowSpec,
    /// `--initial`, or `None` for each layer's own default ([`Initial::default_for`]).
    initial: Option<Initial>,
    layers: Vec<LayerKind>,
    reps: usize,
    seed: u64,
    truncation: TruncSpec,
    format: Format,
    /// Sidecar file that gets one JSON line appended per cell, regardless of `--format`.
    json_out: Option<String>,
}

fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("{flag} expects a non-negative integer, got '{s}'"))
}

fn parse_seed(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)
            .map_err(|_| format!("--seed expects a hex (0x...) or decimal integer, got '{s}'"))
    } else {
        t.parse::<u64>()
            .map_err(|_| format!("--seed expects a hex (0x...) or decimal integer, got '{s}'"))
    }
}

fn parse_csv_usize(s: &str, flag: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .map(|tok| parse_usize(tok, flag))
        .collect()
}

fn parse_csv_layers(s: &str) -> Result<Vec<LayerKind>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .map(LayerKind::parse)
        .collect()
}

fn parse_format(s: &str) -> Result<Format, String> {
    match s {
        "table" => Ok(Format::Table),
        "json" => Ok(Format::Json),
        "tsv" => Ok(Format::Tsv),
        other => Err(format!(
            "--format expects one of table|json|tsv, got '{other}'"
        )),
    }
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut n: usize = 1_000_000;
    let mut qubits: usize = 128;
    let mut threads: Vec<usize> = vec![1, 8, 16, 32];
    let mut layers: Vec<LayerKind> = default_layers();
    let mut reps: usize = 8;
    let mut seed: u64 = 0xC0FFEE;
    let mut hash_seed: u64 = DEFAULT_HASH_SEED;
    let mut bucket_bits: u8 = 0;
    let mut target_bucket_len: usize = DEFAULT_TARGET_BUCKET_LEN;
    let mut min_buckets: usize = DEFAULT_MIN_BUCKETS;
    let mut truncation = TruncSpec::Keep;
    let mut format = Format::Table;
    let mut json_out: Option<String> = None;
    let mut partitions: Vec<usize> = vec![1];
    let mut partition_cpus = PartitionCpus::Auto;
    let mut bind_memory = true;
    let mut partition_seed: Option<u64> = None;
    let mut partition_rows = PartitionRowSpec::Random;
    let mut initial: Option<Initial> = None;
    let mut mpi = false;

    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        // The one flag with no value.
        if flag == "--mpi" {
            if cfg!(not(feature = "mpi")) {
                return Err(
                    "--mpi needs the `mpi` cargo feature (cargo run --release --features \
                     phase-timing,mpi --example phase_breakdown)"
                        .to_string(),
                );
            }
            mpi = true;
            i += 1;
            continue;
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--n" => n = parse_usize(value, "--n")?,
            "--qubits" => qubits = parse_usize(value, "--qubits")?,
            "--threads" => threads = parse_csv_usize(value, "--threads")?,
            "--layers" => layers = parse_csv_layers(value)?,
            "--reps" => reps = parse_usize(value, "--reps")?,
            "--seed" => seed = parse_seed(value)?,
            "--hash-seed" => hash_seed = parse_seed(value)?,
            "--bucket-bits" => {
                let b = parse_usize(value, "--bucket-bits")?;
                if b > B_MAX_BITS as usize {
                    return Err(format!(
                        "--bucket-bits must be at most B_MAX_BITS ({B_MAX_BITS}), got {b}"
                    ));
                }
                bucket_bits = b as u8;
            }
            "--target-bucket-len" => target_bucket_len = parse_usize(value, "--target-bucket-len")?,
            "--min-buckets" => min_buckets = parse_usize(value, "--min-buckets")?,
            "--truncation" => truncation = TruncSpec::parse(value)?,
            "--partitions" => partitions = parse_csv_usize(value, "--partitions")?,
            "--partition-cpus" => partition_cpus = PartitionCpus::parse(value)?,
            "--bind-memory" => {
                bind_memory = match value.trim() {
                    "0" => false,
                    "1" => true,
                    other => return Err(format!("--bind-memory expects 0 or 1, got '{other}'")),
                }
            }
            "--partition-seed" => partition_seed = Some(parse_seed(value)?),
            "--partition-rows" => partition_rows = PartitionRowSpec::parse(value)?,
            "--initial" => initial = Some(Initial::parse(value)?),
            "--format" => format = parse_format(value)?,
            "--json-out" => json_out = Some(value.clone()),
            other => return Err(format!("unknown flag '{other}' (see --help)")),
        }
        i += 2;
    }

    if qubits == 0 {
        return Err("--qubits must be at least 1".to_string());
    }
    if threads.is_empty() {
        return Err("--threads must list at least one thread count".to_string());
    }
    if threads.contains(&0) {
        return Err("--threads entries must be positive".to_string());
    }
    if layers.is_empty() {
        return Err("--layers must list at least one layer".to_string());
    }
    if reps == 0 {
        return Err("--reps must be at least 1".to_string());
    }
    if target_bucket_len == 0 {
        return Err("--target-bucket-len must be at least 1".to_string());
    }
    // `desired_bits`'s "worth splitting" gate is non-monotone below 16.
    if min_buckets < 16 {
        return Err(format!(
            "--min-buckets must be at least 16, got {min_buckets}"
        ));
    }
    if partitions.is_empty() {
        return Err("--partitions must list at least one partition count".to_string());
    }

    // The two Trotter-step workloads run on a lattice of their own.
    if layers.contains(&LayerKind::HeavyHexStep) && qubits < HEAVY_HEX_QUBITS {
        return Err(format!(
            "--layers heavyhex_step needs --qubits >= {HEAVY_HEX_QUBITS} (the Eagle r3 lattice \
             names qubits 0..{}), got {qubits}",
            HEAVY_HEX_QUBITS - 1,
        ));
    }
    if layers.contains(&LayerKind::TfimStep) && qubits < 2 {
        return Err(format!(
            "--layers tfim_step needs --qubits >= 2 (a chain has n-1 bonds), got {qubits}"
        ));
    }

    // The partition axis, checked before a single cell runs.
    let explicit = match &partition_cpus {
        PartitionCpus::Explicit(lists) => Some(lists.len()),
        _ => None,
    };
    for &p in &partitions {
        if p == 0 {
            return Err("--partitions entries must be positive".to_string());
        }
        if !p.is_power_of_two() {
            return Err(format!(
                "--partitions entries must be powers of two (a partition index is log2(P) GF(2) \
                 hash rows), got {p}"
            ));
        }
        for &t in &threads {
            if p > t {
                return Err(format!(
                    "--partitions {p} needs at least {p} threads, but --threads lists {t}: \
                     --threads is the TOTAL thread count, split threads/P per partition"
                ));
            }
            if t % p != 0 {
                return Err(format!(
                    "--threads {t} is not divisible by --partitions {p}: --threads is the TOTAL \
                     thread count, so each partition's pool gets threads/P workers"
                ));
            }
        }
        if p > 1 {
            if let Some(sets) = explicit {
                if sets != p {
                    return Err(format!(
                        "--partition-cpus lists {sets} CPU set(s) but --partitions includes {p}: \
                         an explicit placement needs exactly one cpulist per partition"
                    ));
                }
            }
        }
    }
    // `TopN` has no `PartitionedTruncation` impl, so reject it here.
    let any_partitioned = partitions.iter().any(|&p| p > 1);
    if any_partitioned {
        if let TruncSpec::TopN(topn) = truncation {
            return Err(format!(
                "--truncation topn:{topn} cannot run a partitioned cell: TopN's exact selection \
                 has no PartitionedTruncation impl (the trait bound rejects it statically). Use \
                 --truncation atopn:{topn} — its collective octave histogram is the partitioned \
                 form of the same policy — or drop --partitions"
            ));
        }
    }

    // `--mpi` is the distributed shape: the rank is the partition (D = 1).
    if mpi && partitions != vec![1] {
        return Err(
            "--mpi runs one partition per rank (D = 1), so --partitions must stay at its default \
             of 1: there is no domains-per-rank hybrid"
                .to_string(),
        );
    }
    if mpi {
        if let TruncSpec::TopN(topn) = truncation {
            return Err(format!(
                "--truncation topn:{topn} cannot run a distributed cell: TopN has no \
                 PartitionedTruncation impl. Use --truncation atopn:{topn}"
            ));
        }
    }

    Ok(Config {
        n,
        qubits,
        hash_seed,
        bucket_bits,
        target_bucket_len,
        min_buckets,
        threads,
        partitions,
        mpi,
        partition_cpus,
        bind_memory,
        partition_seed,
        partition_rows,
        initial,
        layers,
        reps,
        seed,
        truncation,
        format,
        json_out,
    })
}

/// A weight-2 `ZZ` rotation, verbatim from `benches/pauli_ops.rs::zz_rotation`.
fn zz_rotation<const W: usize>(q0: u32, q1: u32, theta: f64) -> PauliRotation<W> {
    let mut gen = PauliString::<W> {
        x: [0u64; W],
        z: [0u64; W],
    };
    gen.z[(q0 as usize) / 64] |= 1u64 << (q0 % 64);
    gen.z[(q1 as usize) / 64] |= 1u64 << (q1 % 64);
    PauliRotation::new(gen, theta)
}

/// sqrt(SWAP) on `(q0, q1)`, verbatim from `benches/pauli_ops.rs::sqrt_swap`.
fn sqrt_swap(q0: u32, q1: u32) -> GeneralUnitary2Q {
    let h = Complex64::new(0.5, 0.5);
    let hc = Complex64::new(0.5, -0.5);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    GeneralUnitary2Q::from_matrix(
        q0,
        q1,
        [
            [one, zero, zero, zero],
            [zero, h, hc, zero],
            [zero, hc, h, zero],
            [zero, zero, zero, one],
        ],
    )
}

/// One fixed Haar-random SU(4) block on `(q0, q1)`, the probe's stand-in for the general matrix-gate path.
///
/// Unlike [`sqrt_swap`], a generic SU(4) keeps a dense PTM under repeated application rather than cycling.
fn haar_su4_block(q0: u32, q1: u32) -> GeneralUnitary2Q {
    GeneralUnitary2Q::from_matrix(q0, q1, haar_su4_matrix())
}

/// Qubit count of the fixed [`trotter_circuit`] chain.
const TROTTER_QUBITS: usize = 32;

/// Safety cap on `trotter`'s own input size, overriding `--n`: 64 distinct generators under no
/// truncation grow combinatorially rather than closing to a bounded key set, so an uncapped `--n`
/// would try to materialize far too many terms.
const TROTTER_MAX_N: usize = 100;

/// The 64-channel TFIM Trotter step: 32 `ZZ` bond rotations (periodic boundary) then 32
/// transverse-field `X` rotations, fixed at [`TROTTER_QUBITS`] qubits regardless of `--qubits`.
fn trotter_circuit<const W: usize>() -> Circuit<W> {
    let num_qubits = TROTTER_QUBITS;
    let theta = 0.1;
    let mut circuit = Circuit::<W>::new(num_qubits);
    for q in 0..num_qubits {
        let q0 = q as u32;
        let q1 = ((q + 1) % num_qubits) as u32;
        circuit.push(zz_rotation::<W>(q0, q1, 2.0 * theta));
    }
    for q in 0..num_qubits {
        let qq = q as u32;
        let gen = PauliString::<W>::x(qq);
        circuit.push(PauliRotation::new(gen, 2.0 * theta));
    }
    circuit
}

/// Qubits in the heavy-hex lattice [`heavy_hex_127_edges`] describes.
const HEAVY_HEX_QUBITS: usize = 127;

/// The `ZZ` angle of the kicked-Ising workload: `-pi/2`, a Clifford entangler.
const THETA_ZZ: f64 = -std::f64::consts::FRAC_PI_2;

/// The transverse-field kick angle of the same workload: `5·pi/16`, the non-Clifford point.
const THETA_H: f64 = 5.0 * std::f64::consts::PI / 16.0;

/// The bonds of a 1D **open** chain: `n - 1` edges `(i, i+1)`.
fn chain_edges(num_qubits: usize) -> Vec<(u32, u32)> {
    (0..num_qubits.saturating_sub(1))
        .map(|i| (i as u32, i as u32 + 1))
        .collect()
}

/// The 127-qubit heavy-hex coupling map, from `test_support`.
fn heavy_hex_127_edges() -> Vec<(u32, u32)> {
    paulistrings::test_support::heavy_hex_127_edges()
}

/// Greedy first-fit edge coloring in sorted edge order; a color is a set of disjoint-support
/// edges, i.e. one hardware layer.
fn edge_coloring(edges: &[(u32, u32)]) -> Vec<Vec<(u32, u32)>> {
    let n = edges
        .iter()
        .map(|&(a, b)| a.max(b) as usize + 1)
        .max()
        .unwrap_or(0);
    let mut used: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut classes: Vec<Vec<(u32, u32)>> = Vec::new();
    for &(a, b) in edges {
        let mut color = 0usize;
        while used[a as usize].contains(&color) || used[b as usize].contains(&color) {
            color += 1;
        }
        while classes.len() <= color {
            classes.push(Vec::new());
        }
        classes[color].push((a, b));
        used[a as usize].push(color);
        used[b as usize].push(color);
    }
    classes
}

/// A `PauliRotation` about `X_q`.
fn x_rotation<const W: usize>(q: u32, theta: f64) -> PauliRotation<W> {
    PauliRotation::new(PauliString::<W>::x(q), theta)
}

/// `steps` Trotter steps of the kicked transverse-field Ising model on an open chain of
/// `num_qubits` qubits, one channel per gate: the `ZZ` layer then the `X` layer, `2n - 1`
/// channels per step. Uses [`THETA_ZZ`] / [`THETA_H`]; not step-comparable with
/// [`heavy_hex_step_circuit`], whose `X`-then-`ZZ` order differs.
fn tfim_step_circuit<const W: usize>(num_qubits: usize, steps: usize) -> Circuit<W> {
    let mut c = Circuit::<W>::new(num_qubits);
    for _ in 0..steps {
        for (a, b) in chain_edges(num_qubits) {
            c.push(zz_rotation::<W>(a, b, THETA_ZZ));
        }
        for q in 0..num_qubits as u32 {
            c.push(x_rotation::<W>(q, THETA_H));
        }
    }
    c
}

/// `steps` Trotter steps of the 127-qubit heavy-hex kicked-Ising circuit, one channel per gate:
/// the `X` layer then the `ZZ` layer in [`edge_coloring`] order, 271 channels per step.
/// `num_qubits` must be at least [`HEAVY_HEX_QUBITS`]; anything above is a spectator.
fn heavy_hex_step_circuit<const W: usize>(num_qubits: usize, steps: usize) -> Circuit<W> {
    assert!(
        num_qubits >= HEAVY_HEX_QUBITS,
        "heavy_hex_step_circuit: the lattice needs {HEAVY_HEX_QUBITS} qubits, got {num_qubits}",
    );
    let zz_order: Vec<(u32, u32)> = edge_coloring(&heavy_hex_127_edges())
        .into_iter()
        .flatten()
        .collect();
    let mut c = Circuit::<W>::new(num_qubits);
    for _ in 0..steps {
        for q in 0..HEAVY_HEX_QUBITS as u32 {
            c.push(x_rotation::<W>(q, THETA_H));
        }
        for &(a, b) in &zz_order {
            c.push(zz_rotation::<W>(a, b, THETA_ZZ));
        }
    }
    c
}

/// The single-term observable `Z_q` on `num_qubits` qubits, coefficient 1.
fn z_observable<const W: usize>(num_qubits: usize, q: u32) -> PauliSum<W> {
    let mut acc = BuildAccumulator::<W>::with_capacity(num_qubits, 1);
    acc.add_term(PauliString::<W>::z(q), Phase::ONE, Complex64::new(1.0, 0.0));
    acc.finalize()
}

/// A truncation policy that never drops anything.
struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// [`AlwaysKeep`] for a partitioned cell.
///
/// A separate type rather than an impl on `AlwaysKeep`: flipping `finalizes_layer()` to `false`
/// on `AlwaysKeep` itself would change what the unpartitioned cell measures
/// (`PropagateOptions::starts_direct` reads it).
struct AlwaysKeepPartitioned;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeepPartitioned {
    fn finalizes_layer(&self) -> bool {
        false
    }
}
impl<const W: usize> PartitionedTruncation<W> for AlwaysKeepPartitioned {}

/// Builds a cell's circuit. `gen_qubits` is the `(q0, q1)` pair the `rotation_*` layers rotate about.
fn build_circuit<const W: usize>(
    layer: LayerKind,
    qubits: usize,
    reps: usize,
    gen_qubits: (u32, u32),
) -> Circuit<W> {
    let theta = 0.1;
    match layer {
        LayerKind::RotationZz | LayerKind::RotationLocal | LayerKind::RotationRemote => {
            let (q0, q1) = gen_qubits;
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(zz_rotation::<W>(q0, q1, theta));
            }
            c
        }
        LayerKind::Cnot => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(Clifford2Q::cnot(0, 1));
            }
            c
        }
        LayerKind::Gu2q => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(sqrt_swap(0, 1));
            }
            c
        }
        LayerKind::Su4 | LayerKind::Su4Local => {
            let (q0, q1) = gen_qubits;
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(haar_su4_block(q0, q1));
            }
            c
        }
        LayerKind::Depolarizing => {
            let mut c = Circuit::<W>::new(qubits);
            for _ in 0..reps {
                c.push(Depolarizing {
                    support: [3],
                    p: 0.05,
                });
            }
            c
        }
        LayerKind::Trotter => trotter_circuit::<W>(),
        LayerKind::TfimStep => tfim_step_circuit::<W>(qubits, reps),
        LayerKind::HeavyHexStep => heavy_hex_step_circuit::<W>(qubits, reps),
    }
}

struct CellResult {
    layer: &'static str,
    truncation: String,
    threads: usize,
    n: usize,
    reps: usize,
    qubits: usize,
    seed: u64,
    hash_seed: u64,
    bucket_bits: u8,
    target_bucket_len: usize,
    min_buckets: usize,
    wall_ns: u64,
    /// The cell's phase breakdown; for a partitioned cell, [`fold_partition_stats`]'s cell-level view.
    stats: PhaseStats,
    vmrss_kb: u64,
    vmhwm_kb: u64,
    /// Partitions the cell ran on: `1` for an unpartitioned cell.
    partitions: usize,
    /// `--partition-cpus` echoed back (`"auto"`, `"unpinned"`, or the list).
    partition_cpus: String,
    /// `--bind-memory`, as the `pin_memory` field of the sidecar.
    pin_memory: bool,
    /// The `(q0, q1)` the `rotation_*` layers rotated about.
    gen_qubits: (u32, u32),
    /// `--initial` as it applied to this layer (each layer has its own default).
    initial: &'static str,
    /// `--partition-rows` echoed back; written on every row for a single schema.
    partition_rows: &'static str,
    /// What those rows cost this cell's circuit; all-zero on an unpartitioned row.
    row_stats: RowChoiceStats,
    /// Everything only a partitioned cell has, `None` at `P = 1` classic.
    partitioned: Option<PartitionCellStats>,
    /// `(rank, ranks)` for a `--mpi` cell; every field above is then this rank's own.
    mpi: Option<(u32, u32)>,
}

/// The partition-axis numbers of one cell, from the timed call's
/// [`PartitionTrace`] and its per-partition [`PhaseStats`].
struct PartitionCellStats {
    /// Layers with no remote delta, so no export and no transport call.
    local_layers: usize,
    /// Layers with at least one remote delta.
    remote_layers: usize,
    /// Collective calls over the timed run, summed over layers, excluding the exchanges.
    collectives: u64,
    /// Rows moved across partitions, summed over layers and senders.
    rows_exported: u64,
    /// Wire bytes for those rows.
    bytes_exported: u64,
    /// Sum over layers of each partition's input term count, by rank.
    terms_in: Vec<usize>,
    /// `max / mean` of [`Self::terms_in`]: 1.0 is a perfectly even split.
    imbalance: f64,
    /// The same ratio per layer, in application order: shows how fast terms mix between partitions.
    imbalance_by_layer: Vec<f64>,
    /// Total terms in, per layer, over the whole group.
    terms_by_layer: Vec<usize>,
    /// Each partition's `coset_loop_ns`, by rank.
    coset_loop_ns: Vec<u64>,
}

/// Read `VmRSS`/`VmHWM` (kB) from `/proc/self/status`; `0` for either field not found.
fn read_proc_status_kb() -> (u64, u64) {
    let mut rss = 0u64;
    let mut hwm = 0u64;
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                rss = parse_kb_field(rest);
            } else if let Some(rest) = line.strip_prefix("VmHWM:") {
                hwm = parse_kb_field(rest);
            }
        }
    }
    (rss, hwm)
}

fn parse_kb_field(s: &str) -> u64 {
    s.split_whitespace()
        .next()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

/// The cell's input sum, before any propagation: the seeded generator the layer asks for,
/// re-hashed and pre-refined per `--hash-seed` / `--bucket-bits`. Shared by the unpartitioned
/// and the partitioned path, so a `P = 1` and a `P = 2` cell propagate the same terms.
fn build_base_sum<const W: usize>(layer: LayerKind, cfg: &Config) -> PauliSum<W> {
    if layer.is_trotter_step() && cfg.truncation == TruncSpec::Keep {
        eprintln!(
            "phase_breakdown: warning: {} under --truncation keep does not converge — a \
             kicked-Ising step is a fresh set of distinct generators every step, so the term \
             count grows without bound. Pass a coefficient threshold, e.g. \
             --truncation coeff:1.220703125e-4 (2^-13, the presentation's working point) or \
             --truncation coeff:3.90625e-3 (2^-8) for a quick run.",
            layer.name(),
        );
    }
    // `trotter` is 64 distinct generators, unlike the other layers' one generator repeated, so a
    // dense input can blow up combinatorially. It gets a low-weight input sized to its own fixed
    // qubit count instead of `--qubits`, and its own `--n` cap (see `TROTTER_MAX_N`).
    let base = match layer {
        LayerKind::Trotter => {
            if cfg.n > TROTTER_MAX_N {
                eprintln!(
                    "phase_breakdown: note: trotter ignores --n above {TROTTER_MAX_N} \
                     (requested {}) — two chained, untruncated 64-layer passes grow \
                     combinatorially, and an uncapped --n has been measured driving this \
                     layer past tens of GB of RSS. Capping this cell's input to {TROTTER_MAX_N}.",
                    cfg.n,
                );
            }
            low_weight_sum::<W>(cfg.n.min(TROTTER_MAX_N), TROTTER_QUBITS, 3, cfg.seed)
        }
        // Everything else takes `--initial`, defaulting per layer.
        _ => match cfg.initial.unwrap_or_else(|| Initial::default_for(layer)) {
            Initial::Random => rand_sum::<W>(cfg.n, cfg.qubits, cfg.seed),
            Initial::Z0 => z_observable::<W>(cfg.qubits, (cfg.qubits / 2) as u32),
        },
    };
    // `--hash-seed` re-draws H's rows, which changes the coset dimension `r` (research/FINDINGS.md).
    // `with_hash` rescatters at zero bucket bits; `--bucket-bits` then refines.
    let mut base = if cfg.hash_seed == DEFAULT_HASH_SEED {
        base
    } else {
        let nq = base.num_qubits();
        base.with_hash(Gf2Hash::<W>::new(nq, 0, cfg.hash_seed))
    };
    while base.hash().bits() < cfg.bucket_bits {
        base.refine();
    }
    base
}

fn run_cell<const W: usize, P>(
    layer: LayerKind,
    threads: usize,
    cfg: &Config,
    policy: &P,
) -> CellResult
where
    P: TruncationPolicy<W>,
{
    let base = build_base_sum::<W>(layer, cfg);
    // Nothing is remote without partitions, so `rotation_local`/`rotation_remote` are `rotation_zz` here.
    let gen_qubits = (0u32, 1u32);
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build a rayon thread pool");

    // `rebucket` is grow-only, so this only means anything if the warm-up never grew past it.
    let options = PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        ..PropagateOptions::default()
    };

    let (steady_n, wall_ns, stats) = pool.install(|| {
        let mut scratch = LayerScratch::<W>::new();

        // Untimed warm-up drives the input to its steady state, so the timed call measures that, not first-layer growth.
        let warmed = propagate_with_scratch_and_options(
            &circuit,
            base.clone(),
            policy,
            Direction::Forward,
            &mut scratch,
            options,
        );
        let _ = scratch.take_stats(); // discard warm-up counters

        let steady_n = warmed.len();
        let start = Instant::now();
        let output = propagate_with_scratch_and_options(
            &circuit,
            warmed,
            policy,
            Direction::Forward,
            &mut scratch,
            options,
        );
        let wall_ns = start.elapsed().as_nanos() as u64;
        let stats = scratch.take_stats();
        std::hint::black_box(&output);

        (steady_n, wall_ns, stats)
    });

    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();

    CellResult {
        layer: layer.name(),
        truncation: cfg.truncation.label(),
        threads,
        n: steady_n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        hash_seed: cfg.hash_seed,
        bucket_bits: cfg.bucket_bits,
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
        partitions: 1,
        partition_cpus: cfg.partition_cpus.label(),
        pin_memory: cfg.bind_memory,
        gen_qubits,
        initial: cfg
            .initial
            .unwrap_or_else(|| Initial::default_for(layer))
            .label(),
        partition_rows: cfg.partition_rows.label(),
        row_stats: RowChoiceStats::default(),
        // No split, so no partition numbers on this row.
        partitioned: None,
        mpi: None,
    }
}

/// The `(q0, q1)` a `rotation_local` / `rotation_remote` cell rotates about: the smallest `q1 > 0`
/// whose one-channel `ZZ(0, q1)` layer has no remote delta (`local`), respectively at least one
/// (`remote`), under `rows`.
///
/// # Panics
///
/// If no qubit in `1..qubits` gives the requested class (possible for `remote` on a pathological
/// row draw); the message names `--partition-seed` as the knob.
fn choose_generator<const W: usize>(
    layer: LayerKind,
    qubits: usize,
    base: &PauliSum<W>,
    rows: &PartitionRows<W>,
) -> (u32, u32) {
    if !layer.picks_generator() {
        return (0, 1);
    }
    let want_remote = layer == LayerKind::RotationRemote;
    // A dense SU(4) needs all 15 deltas local, so it scans every pair, not just those touching 0.
    let pairs: Vec<(u32, u32)> = if layer == LayerKind::Su4Local {
        (0..qubits as u32)
            .flat_map(|q0| ((q0 + 1)..qubits as u32).map(move |q1| (q0, q1)))
            .collect()
    } else {
        (1..qubits as u32).map(|q| (0, q)).collect()
    };
    for (q0, q1) in pairs {
        let mut probe = Circuit::<W>::new(qubits);
        if layer == LayerKind::Su4Local {
            probe.push(haar_su4_block(q0, q1));
        } else {
            probe.push(zz_rotation::<W>(q0, q1, 0.1));
        }
        let (_, remote) = count_remote_deltas(&probe, base.hash(), rows, false)[0];
        if (remote > 0) == want_remote {
            return (q0, q1);
        }
    }
    panic!(
        "phase_breakdown: no {}(q0, q1) layer on {qubits} qubits is {} under these partition \
         rows — try another --partition-seed or more --qubits",
        if layer == LayerKind::Su4Local {
            "SU4"
        } else {
            "ZZ"
        },
        if want_remote { "remote" } else { "local" },
    );
}

/// `P` contiguous qubit blocks cutting as few of `edges` as possible; returns the blocks' end
/// positions.
///
/// Exact by dynamic program, minimizing cut edges first and then size imbalance, with block
/// sizes additionally held within ±25% of `num_qubits / partitions` (without that bound, the
/// heavy-hex lattice's minimum is a 4/123 split whose smaller half holds almost nothing).
fn cut_blocks(num_qubits: usize, partitions: usize, edges: &[(u32, u32)]) -> Vec<usize> {
    let n = num_qubits;
    assert!(partitions >= 1 && partitions <= n);

    // internal[l * (n + 1) + r] = edges with both endpoints inside the block [l, r).
    let mut internal = vec![0i64; (n + 1) * (n + 1)];
    let mut row = vec![0i64; n + 1];
    for l in (0..n).rev() {
        row.iter_mut().for_each(|v| *v = 0);
        for &(a, b) in edges {
            if a as usize == l && (b as usize) < n {
                row[b as usize + 1] += 1;
            }
        }
        let mut running = 0i64;
        for r in 0..=n {
            running += row[r];
            internal[l * (n + 1) + r] = internal[(l + 1) * (n + 1) + r] + running;
        }
    }

    // Fewest cut edges dominates; evenness breaks the ties.
    let weight = 4 * (n as i64) * (partitions as i64) + 1;
    let target = n as f64 / partitions as f64;
    let lo = ((target * 0.75).floor() as usize).max(1);
    let hi = ((target * 1.25).ceil() as usize).max(lo);

    const UNSET: i64 = i64::MAX;
    let mut cost = vec![UNSET; (partitions + 1) * (n + 1)];
    let mut from = vec![usize::MAX; (partitions + 1) * (n + 1)];
    cost[0] = 0;
    for k in 1..=partitions {
        for pos in 1..=n {
            let first = pos.saturating_sub(hi);
            let last = pos.saturating_sub(lo);
            for prev in first..=last {
                let before = cost[(k - 1) * (n + 1) + prev];
                if before == UNSET {
                    continue;
                }
                let size = (pos - prev) as i64;
                let deviation = (size * partitions as i64 - n as i64).abs();
                let c = before - internal[prev * (n + 1) + pos] * weight + deviation;
                let slot = k * (n + 1) + pos;
                if cost[slot] == UNSET || c < cost[slot] {
                    cost[slot] = c;
                    from[slot] = prev;
                }
            }
        }
    }
    assert!(
        cost[partitions * (n + 1) + n] != UNSET,
        "cut_blocks: no {partitions}-block split of {n} qubits fits the ±25% size window",
    );

    let mut ends = vec![0usize; partitions];
    let mut pos = n;
    for k in (1..=partitions).rev() {
        ends[k - 1] = pos;
        pos = from[k * (n + 1) + pos];
    }
    ends
}

/// The `log2(P)` z-only "cut" rows labelling the blocks [`cut_blocks`] found, plus the number of
/// `edges` the cut crosses. Every single-qubit rotation is then local, and a `ZZ(i, j)` rotation
/// is remote exactly when the edge crosses the cut.
fn cut_rows<const W: usize>(
    num_qubits: usize,
    partitions: usize,
    edges: &[(u32, u32)],
) -> (PartitionRows<W>, Vec<usize>, usize) {
    let ends = cut_blocks(num_qubits, partitions, edges);

    // label[q] = the partition label of q's block.
    let mut label = vec![0u32; num_qubits];
    let mut blocks: Vec<Vec<u32>> = Vec::with_capacity(partitions);
    let mut start = 0usize;
    for (block, &end) in ends.iter().enumerate() {
        label[start..end].iter_mut().for_each(|l| *l = block as u32);
        blocks.push((start as u32..end as u32).collect());
        start = end;
    }

    let crossed = edges
        .iter()
        .filter(|&&(a, b)| {
            (a as usize) < num_qubits
                && (b as usize) < num_qubits
                && label[a as usize] != label[b as usize]
        })
        .count();
    (PartitionRows::cut(num_qubits, &blocks), ends, crossed)
}

/// What the chosen rows cost the circuit, for the sidecar.
#[derive(Clone, Copy, Debug, Default)]
struct RowChoiceStats {
    /// Distinct key-delta masks the rows leave remote.
    remote_gens: usize,
    /// Their total weight: the export-and-exchange count a run of this circuit will pay.
    remote_weight: f64,
}

/// How many bucket bits [`choose_partition_rows`] checks row independence at (ten bits is 1024
/// buckets, roughly where the default target puts a million-term sum).
const INDEPENDENCE_PROBE_BITS: u8 = 10;

/// Fresh hash seeds tried when the chosen rows are dependent on `H`'s.
const INDEPENDENCE_RETRIES: usize = 16;

/// The partition rows one cell runs under, the sum they will be scattered from, and what they
/// cost the circuit.
///
/// A non-random row set that is dependent on `H`'s active rows costs load balance, so it gets the
/// hash re-seeded until [`PartitionRows::is_independent_of`] at [`INDEPENDENCE_PROBE_BITS`]
/// passes; this changes the coset dimension too, so the cell reports the seed it ended up with.
fn choose_partition_rows<const W: usize>(
    layer: LayerKind,
    cfg: &Config,
    base: PauliSum<W>,
    partitions: usize,
) -> (PauliSum<W>, PartitionRows<W>, RowChoiceStats) {
    let num_qubits = base.num_qubits();
    let bits = partitions.trailing_zeros() as u8;
    // The generator scan needs the circuit. A layer whose generator qubits
    // come from the rows themselves (`rotation_local` and friends) is
    // circular, so the scan sees that layer's `(0, 1)` variant — harmless,
    // since those layers exist to probe a *given* row set.
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, (0, 1));
    let gens = circuit_generators(&circuit, base.hash(), false);

    let rows = match cfg.partition_rows {
        PartitionRowSpec::Random => PartitionRows::<W>::from_seed(
            num_qubits,
            bits,
            cfg.partition_seed.unwrap_or_else(|| base.hash().seed()),
        ),
        PartitionRowSpec::Cut => {
            let edges = layer.cut_edges(num_qubits);
            let (rows, ends, crossed) = cut_rows::<W>(num_qubits, partitions, &edges);
            if partitions > 1 {
                eprintln!(
                    "phase_breakdown: note: --partition-rows cut on {} at P={partitions}: blocks \
                     ending at {ends:?}, crossing {crossed} of {} two-qubit generators (every \
                     single-qubit rotation is local by construction).",
                    layer.name(),
                    edges.len(),
                );
            }
            rows
        }
    };

    // `random` rows are effectively never dependent, so they skip the re-seed check.
    let base = if cfg.partition_rows == PartitionRowSpec::Random || bits == 0 {
        base
    } else {
        reseed_hash_until_independent(cfg, base, &rows, layer, partitions)
    };

    let remote: Vec<&GeneratorWeight<W>> = gens
        .iter()
        .filter(|g| rows.partition_of(&g.mask_x, &g.mask_z) != 0)
        .collect();
    let stats = RowChoiceStats {
        remote_gens: remote.len(),
        remote_weight: remote.iter().map(|g| g.weight).sum(),
    };
    (base, rows, stats)
}

/// Re-seed the sum's hash until `rows` is independent of the rows `H` will have at
/// [`INDEPENDENCE_PROBE_BITS`], or until the tries run out. Re-hashing rescatters the sum at
/// zero bucket bits, so `--bucket-bits`'s pre-refinement is re-applied afterwards.
fn reseed_hash_until_independent<const W: usize>(
    cfg: &Config,
    base: PauliSum<W>,
    rows: &PartitionRows<W>,
    layer: LayerKind,
    partitions: usize,
) -> PauliSum<W> {
    let num_qubits = base.num_qubits();
    // Rows can only be independent while they fit in the `2n` key columns.
    let headroom = (2 * num_qubits).saturating_sub(rows.bits() as usize);
    let probe_bits = INDEPENDENCE_PROBE_BITS
        .min(B_MAX_BITS)
        .min(headroom.min(u8::MAX as usize) as u8);
    let independent_at =
        |seed: u64| rows.is_independent_of(&Gf2Hash::<W>::new(num_qubits, probe_bits, seed));
    let seed = base.hash().seed();
    if independent_at(seed) {
        return base;
    }

    // Splitmix64's increment: a deterministic full-period walk, identical on every rank.
    let mut candidate = seed;
    for attempt in 1..=INDEPENDENCE_RETRIES {
        candidate = candidate.wrapping_add(0x9E37_79B9_7F4A_7C15);
        if !independent_at(candidate) {
            continue;
        }
        eprintln!(
            "phase_breakdown: note: the {} rows for {} at P={partitions} are NOT independent of \
             H's first {probe_bits} rows under hash seed {seed:#x} — the joint bucket would carry \
             fewer bits than it claims and the split would be unbalanced. Re-seeded the hash to \
             {candidate:#x} (attempt {attempt}). This changes the coset dimension too, so do not \
             compare this cell's phase timings against a differently seeded one \
             (research/FINDINGS.md).",
            cfg.partition_rows.label(),
            layer.name(),
        );
        let mut base = base.with_hash(Gf2Hash::<W>::new(num_qubits, 0, candidate));
        while base.hash().bits() < cfg.bucket_bits {
            base.refine();
        }
        return base;
    }
    eprintln!(
        "phase_breakdown: warning: the {} rows for {} at P={partitions} are dependent on H's \
         first {probe_bits} rows, and {INDEPENDENCE_RETRIES} re-seeds did not fix it. Running \
         anyway: dependence costs load balance, not correctness.",
        cfg.partition_rows.label(),
        layer.name(),
    );
    base
}

/// One partitioned cell: scatter (untimed), warm up, drain, time one `propagate`, and read the
/// trace and the per-partition counters. Mirrors [`run_cell`]'s shape; the two differ only in
/// the engine underneath.
fn run_cell_partitioned<const W: usize, P>(
    layer: LayerKind,
    threads: usize,
    partitions: usize,
    cfg: &Config,
    policy: &P,
) -> CellResult
where
    P: PartitionedTruncation<W>,
{
    let base = build_base_sum::<W>(layer, cfg);

    let config = PartitionConfig {
        placement: cfg.partition_cpus.placement(partitions, threads),
        bind_memory: cfg.bind_memory,
        partition_row_seed: cfg.partition_seed,
    };
    // `--threads` is the TOTAL, so P pools of threads/P compare against one pool of threads.
    let runtime = PartitionRuntime::with_threads_per_partition(&config, Some(threads / partitions))
        .unwrap_or_else(|err| {
            eprintln!("phase_breakdown: cannot resolve the partition placement: {err}");
            std::process::exit(2);
        });
    // `Auto` can silently resolve to fewer partitions than the host has NUMA nodes; refuse rather than mislabel every row.
    if runtime.num_partitions() != partitions {
        eprintln!(
            "phase_breakdown: --partitions {partitions} resolved to {} partitions: an `auto` \
             placement takes one partition per NUMA node in the affinity mask (rounded down to a \
             power of two). Pass --partition-cpus with {partitions} cpulists, or \
             --partition-cpus unpinned, to get {partitions} partitions on this host.",
            runtime.num_partitions(),
        );
        std::process::exit(2);
    }

    // Built here so the generator scan below sees exactly the split the run will use.
    let (base, rows, row_stats) = choose_partition_rows::<W>(layer, cfg, base, partitions);
    let gen_qubits = choose_generator::<W>(layer, cfg.qubits, &base, &rows);
    if layer.picks_generator() {
        eprintln!(
            "phase_breakdown: note: {} at P={partitions} acts on {}({}, {}) — the smallest \
             pair whose deltas are {} under these partition rows.",
            layer.name(),
            if layer == LayerKind::Su4Local {
                "SU4"
            } else {
                "ZZ"
            },
            gen_qubits.0,
            gen_qubits.1,
            if layer == LayerKind::RotationRemote {
                "remote"
            } else {
                "local"
            },
        );
    }
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);

    let options = PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        ..PropagateOptions::default()
    };

    let split_hash_seed = base.hash().seed();
    let mut split = PartitionedSum::scatter_with_rows(base, rows, runtime);
    split.enable_trace();

    // Untimed warm-up, then its counters discarded — same contract as the unpartitioned cell.
    split.propagate_with_options(&circuit, policy, Direction::Forward, options);
    let _ = split.take_trace();
    let _ = split.take_stats();

    let steady_n = split.len();
    let started = Instant::now();
    split.propagate_with_options(&circuit, policy, Direction::Forward, options);
    let wall_ns = started.elapsed().as_nanos() as u64;
    let trace = split.take_trace().expect("tracing was enabled");
    let per_partition = split.take_stats();
    std::hint::black_box(&split);

    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();
    let stats = fold_partition_stats(&per_partition);
    let summary = summarize_partitions(partitions, &trace, &per_partition, &stats);

    CellResult {
        layer: layer.name(),
        truncation: cfg.truncation.label(),
        threads,
        n: steady_n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        // The seed actually used, which `choose_partition_rows` may have re-drawn.
        hash_seed: split_hash_seed,
        bucket_bits: cfg.bucket_bits,
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
        partitions,
        partition_cpus: cfg.partition_cpus.label(),
        pin_memory: cfg.bind_memory,
        gen_qubits,
        initial: cfg
            .initial
            .unwrap_or_else(|| Initial::default_for(layer))
            .label(),
        partition_rows: cfg.partition_rows.label(),
        row_stats,
        partitioned: Some(summary),
        mpi: None,
    }
}

/// One cell on the distributed driver: this process is one rank's partition, and the numbers it
/// reports are its own. The MPI counterpart of [`run_cell_partitioned`] — same shape, but the
/// runtime holds exactly one partition, placed by the launcher's affinity mask, over the world
/// communicator. Collective: every rank runs the identical cell matrix in the identical order.
/// The headline is `vmhwm_kb`, peak resident set per rank including the export/receive transients.
#[cfg(feature = "mpi")]
fn run_cell_mpi<const W: usize, P>(
    layer: LayerKind,
    threads: usize,
    cfg: &Config,
    policy: &P,
) -> CellResult
where
    P: PartitionedTruncation<W>,
{
    use paulistrings::engine::partitioned::{Collectives, DistributedSum};
    use paulistrings::mpi::{rsmpi, MpiTransport};
    use rsmpi::topology::{Communicator, SimpleCommunicator};

    let world = SimpleCommunicator::world();
    let rank = world.rank() as u32;
    let ranks = world.size() as u32;
    if !ranks.is_power_of_two() {
        if rank == 0 {
            eprintln!(
                "phase_breakdown: --mpi needs a power-of-two rank count (a partition is named by \
                 log2(P) GF(2) rows), got {ranks}",
            );
        }
        std::process::exit(2);
    }

    let base = build_base_sum::<W>(layer, cfg);

    // D = 1: one partition over whatever CPUs the launcher left us.
    let config = PartitionConfig {
        placement: cfg.partition_cpus.placement(1, threads),
        bind_memory: cfg.bind_memory,
        partition_row_seed: cfg.partition_seed,
    };
    let runtime = PartitionRuntime::with_threads_per_partition(&config, Some(threads))
        .unwrap_or_else(|err| {
            eprintln!("phase_breakdown: rank {rank} cannot resolve its placement: {err}");
            std::process::exit(2);
        });

    // Same inputs on every rank, so every rank derives the same rows with nothing agreed at run time.
    let (base, rows, row_stats) = choose_partition_rows::<W>(layer, cfg, base, ranks as usize);
    let gen_qubits = choose_generator::<W>(layer, cfg.qubits, &base, &rows);
    if layer.picks_generator() && rank == 0 {
        eprintln!(
            "phase_breakdown: note: {} on {ranks} ranks acts on ({}, {}).",
            layer.name(),
            gen_qubits.0,
            gen_qubits.1,
        );
    }
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);

    let options = PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        ..PropagateOptions::default()
    };

    let transport = MpiTransport::from_communicator(&world);
    let split_hash_seed = base.hash().seed();
    let mut split = DistributedSum::scatter_with_rows(base, transport, runtime, rows);
    split.enable_trace();

    // Untimed warm-up, counters discarded; a barrier so the timed call starts together.
    split.propagate_with_options(&circuit, policy, Direction::Forward, options);
    let _ = split.take_trace();
    let _ = split.take_stats();
    split.transport().barrier();

    let steady_n = split.len_local();
    let started = Instant::now();
    split.propagate_with_options(&circuit, policy, Direction::Forward, options);
    let wall_ns = started.elapsed().as_nanos() as u64;
    let trace = split.take_trace().expect("tracing was enabled");
    let per_partition = split.take_stats();
    std::hint::black_box(&split);

    let (vmrss_kb, vmhwm_kb) = read_proc_status_kb();
    let stats = fold_partition_stats(&per_partition);
    // One "partition" here: this rank.
    let summary = summarize_partitions(1, &trace, &per_partition, &stats);

    CellResult {
        layer: layer.name(),
        truncation: cfg.truncation.label(),
        threads,
        n: steady_n,
        reps: cfg.reps,
        qubits: cfg.qubits,
        seed: cfg.seed,
        // The seed actually used, which `choose_partition_rows` may have re-drawn.
        hash_seed: split_hash_seed,
        bucket_bits: cfg.bucket_bits,
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        wall_ns,
        stats,
        vmrss_kb,
        vmhwm_kb,
        partitions: 1,
        partition_cpus: cfg.partition_cpus.label(),
        pin_memory: cfg.bind_memory,
        gen_qubits,
        initial: cfg
            .initial
            .unwrap_or_else(|| Initial::default_for(layer))
            .label(),
        partition_rows: cfg.partition_rows.label(),
        row_stats,
        partitioned: Some(summary),
        mpi: Some((rank, ranks)),
    }
}

/// The cell-level [`PhaseStats`] of a partitioned run: wall-clock fields are the maximum over
/// partitions (the critical path), busy-time and counter fields are sums. `layers` is the
/// driver's layer count, not a sum — every partition drove the same layers.
fn fold_partition_stats(stats: &PartitionPhaseStats) -> PhaseStats {
    let mut out = PhaseStats::default();
    for s in &stats.per_partition {
        // Wall-clock phases: max over partitions.
        out.rebucket_ns = out.rebucket_ns.max(s.rebucket_ns);
        out.prepare_ns = out.prepare_ns.max(s.prepare_ns);
        out.rescale_ns = out.rescale_ns.max(s.rescale_ns);
        out.span_plan_ns = out.span_plan_ns.max(s.span_plan_ns);
        out.permute_ns = out.permute_ns.max(s.permute_ns);
        out.coset_loop_ns = out.coset_loop_ns.max(s.coset_loop_ns);
        out.unpermute_ns = out.unpermute_ns.max(s.unpermute_ns);
        out.recount_ns = out.recount_ns.max(s.recount_ns);
        out.finalize_ns = out.finalize_ns.max(s.finalize_ns);
        out.collective_ns = out.collective_ns.max(s.collective_ns);
        out.export_ns = out.export_ns.max(s.export_ns);
        out.exchange_ns = out.exchange_ns.max(s.exchange_ns);
        // Worker busy time and counters: sums over the group.
        out.swap_ns += s.swap_ns;
        out.size_ns += s.size_ns;
        out.gather_ns += s.gather_ns;
        out.sort_ns += s.sort_ns;
        out.merge_ns += s.merge_ns;
        out.clear_ns += s.clear_ns;
        out.cosets += s.cosets;
        out.runs += s.runs;
        out.rows_gathered += s.rows_gathered;
        out.rows_sorted += s.rows_sorted;
        out.rows_id += s.rows_id;
        out.terms_in += s.terms_in;
        out.terms_out += s.terms_out;
        out.rows_exported += s.rows_exported;
        out.recv_rows += s.recv_rows;
        out.append_ns += s.append_ns;
        out.chunk_wait_ns += s.chunk_wait_ns;
    }
    out.layers = stats.layers;
    out
}

/// The partition-axis summary of one timed call. `rows_exported` and `bytes_exported` are the
/// same send traffic counted in rows and in wire bytes, respectively.
fn summarize_partitions(
    partitions: usize,
    trace: &PartitionTrace,
    per_partition: &PartitionPhaseStats,
    folded: &PhaseStats,
) -> PartitionCellStats {
    let mut terms_in = vec![0usize; partitions];
    for layer in &trace.layers {
        for (rank, &t) in layer.terms_in.iter().enumerate() {
            terms_in[rank] += t;
        }
    }
    let total: usize = terms_in.iter().sum();
    let imbalance = if total == 0 {
        1.0
    } else {
        let mean = total as f64 / partitions as f64;
        terms_in.iter().copied().max().unwrap_or(0) as f64 / mean
    };

    PartitionCellStats {
        local_layers: trace.local_layers(),
        remote_layers: trace.remote_layers(),
        collectives: trace.total_collectives(),
        rows_exported: folded.rows_exported,
        bytes_exported: trace
            .layers
            .iter()
            .flat_map(|layer| layer.bytes_sent.iter())
            .flat_map(|row| row.iter())
            .sum(),
        terms_in,
        imbalance,
        imbalance_by_layer: trace.imbalance(),
        terms_by_layer: trace
            .layers
            .iter()
            .map(|layer| layer.terms_in.iter().sum())
            .collect(),
        coset_loop_ns: per_partition
            .per_partition
            .iter()
            .map(|s| s.coset_loop_ns)
            .collect(),
    }
}

/// Always printed first for every cell, in every format: `n=` and `layers=` are a contract other
/// scripts grep for (machine contract (b) in `benchmarks/PROFILING.md`).
fn print_cell_line(cell: &CellResult) {
    // Empty for every non-`--mpi` cell, so the line is byte-identical to before the flag existed.
    let mpi = match cell.mpi {
        Some((rank, ranks)) => format!(" rank={rank}/{ranks} vmhwm_kb={}", cell.vmhwm_kb),
        None => String::new(),
    };
    println!(
        "cell layer={} threads={} n={} layers={} wall_ms={:.3} trunc={} partitions={}{mpi}",
        cell.layer,
        cell.threads,
        cell.n,
        cell.stats.layers,
        cell.wall_ns as f64 / 1e6,
        cell.truncation,
        cell.partitions,
    );
}

type PhaseGetter = fn(&PhaseStats) -> u64;

const WALL_PHASES: [(&str, PhaseGetter); 9] = [
    ("rebucket", |s| s.rebucket_ns),
    ("prepare", |s| s.prepare_ns),
    ("rescale", |s| s.rescale_ns),
    ("span_plan", |s| s.span_plan_ns),
    ("permute", |s| s.permute_ns),
    ("coset_loop", |s| s.coset_loop_ns),
    ("unpermute", |s| s.unpermute_ns),
    ("recount", |s| s.recount_ns),
    ("finalize", |s| s.finalize_ns),
];

const BUSY_PHASES: [(&str, PhaseGetter); 6] = [
    ("gather", |s| s.gather_ns),
    ("sort", |s| s.sort_ns),
    ("merge", |s| s.merge_ns),
    ("swap", |s| s.swap_ns),
    ("size", |s| s.size_ns),
    ("clear", |s| s.clear_ns),
];

fn print_table(cell: &CellResult) {
    let s = &cell.stats;
    let wall_total = s.wall_total_ns();

    println!(
        "  {:<12} {:>10} {:>12} {:>8}",
        "phase", "ms", "ms/layer", "% wall"
    );
    for (name, get) in WALL_PHASES {
        let ns = get(s);
        if ns == 0 {
            continue;
        }
        let ms = ns as f64 / 1e6;
        let ms_per_layer = if s.layers > 0 {
            ms / s.layers as f64
        } else {
            0.0
        };
        let pct = if wall_total > 0 {
            ns as f64 / wall_total as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<12} {:>10.3} {:>12.4} {:>7.1}%",
            name, ms, ms_per_layer, pct
        );
    }

    let busy_total = s.busy_total_ns();
    println!("  busy (summed over every coset task / worker):");
    for (name, get) in BUSY_PHASES {
        let ns = get(s);
        let ms = ns as f64 / 1e6;
        let pct = if busy_total > 0 {
            ns as f64 / busy_total as f64 * 100.0
        } else {
            0.0
        };
        println!("    {:<10} {:>10.3} ms {:>7.1}% of busy", name, ms, pct);
    }
    let efficiency = if s.coset_loop_ns > 0 && cell.threads > 0 {
        busy_total as f64 / (s.coset_loop_ns as f64 * cell.threads as f64)
    } else {
        0.0
    };
    println!(
        "    sum(busy) = {:.3} ms, parallel efficiency (busy / (coset_loop * threads)) = {:.2}",
        busy_total as f64 / 1e6,
        efficiency
    );

    let wall_s = cell.wall_ns as f64 / 1e9;
    let strings_per_s = if wall_s > 0.0 {
        s.terms_in as f64 / wall_s
    } else {
        0.0
    };
    let overhead_ns = s.timer_reads() * TIMER_READ_OVERHEAD_NS;
    let overhead_pct = if cell.wall_ns > 0 {
        overhead_ns as f64 / cell.wall_ns as f64 * 100.0
    } else {
        0.0
    };
    println!("  strings/s          = {strings_per_s:.3e}");
    println!(
        "  timer overhead est = {:.3} us ({} reads x {} ns) = {:.2}% of wall",
        overhead_ns as f64 / 1e3,
        s.timer_reads(),
        TIMER_READ_OVERHEAD_NS,
        overhead_pct
    );
    println!(
        "  VmRSS = {} kB   VmHWM = {} kB",
        cell.vmrss_kb, cell.vmhwm_kb
    );
    println!(
        "  target_bucket_len  = {}   min_buckets = {}",
        cell.target_bucket_len, cell.min_buckets
    );
    print_partition_block(cell);
    println!();
}

/// The partition block of the table format, printed only for a cell that ran through the partitioned engine.
fn print_partition_block(cell: &CellResult) {
    let Some(p) = cell.partitioned.as_ref() else {
        return;
    };
    let s = &cell.stats;
    println!(
        "  partitions         = {} on {} (pin_memory = {}, gen_qubits = {},{})",
        cell.partitions,
        cell.partition_cpus,
        u8::from(cell.pin_memory),
        cell.gen_qubits.0,
        cell.gen_qubits.1,
    );
    println!(
        "    layers: {} local / {} remote   rows exported = {} ({:.3} MiB on the wire)",
        p.local_layers,
        p.remote_layers,
        p.rows_exported,
        p.bytes_exported as f64 / (1024.0 * 1024.0),
    );
    println!(
        "    collectives = {} over {} layers (bucket-count schedule every {} layers, \
         plus every remote layer and any the policy runs)",
        p.collectives,
        p.local_layers + p.remote_layers,
        BITS_AGREE_EVERY,
    );
    println!(
        "    export = {:.3} ms   exchange = {:.3} ms   barrier (bucket-count all-reduce) = \
         {:.3} ms   [max over partitions]",
        s.export_ns as f64 / 1e6,
        s.exchange_ns as f64 / 1e6,
        s.collective_ns as f64 / 1e6,
    );
    println!(
        "      received rows appended into the rest streams = {:.3} ms (of which {:.3} ms waiting \
         for a chunk to land) [worker busy time, part of gather]",
        s.append_ns as f64 / 1e6,
        s.chunk_wait_ns as f64 / 1e6,
    );
    println!(
        "    terms in per partition = {:?}   imbalance (max/mean) = {:.3}",
        p.terms_in, p.imbalance,
    );
    let coset_ms: Vec<String> = p
        .coset_loop_ns
        .iter()
        .map(|ns| format!("{:.3}", *ns as f64 / 1e6))
        .collect();
    println!(
        "    coset_loop ms per partition = [{}]",
        coset_ms.join(", ")
    );
    // The human format prints the per-layer series' shape; the full arrays are in the JSON/TSV row.
    if let (Some(first), Some(last)) = (p.imbalance_by_layer.first(), p.imbalance_by_layer.last()) {
        let worst = p
            .imbalance_by_layer
            .iter()
            .copied()
            .fold(f64::MIN, f64::max);
        println!(
            "    imbalance per layer over {} layers: first {:.3}, worst {:.3}, last {:.3}   \
             (rows = {})",
            p.imbalance_by_layer.len(),
            first,
            worst,
            last,
            cell.partition_rows,
        );
    }
}

fn print_json(cell: &CellResult) {
    println!("{}", json_line(cell));
}

/// `[a, b, c]` for a JSON array of integers, `[]` when empty.
fn json_u64_array<T: std::fmt::Display>(values: &[T]) -> String {
    let items: Vec<String> = values.iter().map(T::to_string).collect();
    format!("[{}]", items.join(","))
}

/// `[a, b, c]` for a JSON array of floats at six decimals, `[]` when empty.
fn json_f64_array(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{v:.6}")).collect();
    format!("[{}]", items.join(","))
}

/// `a|b|c` for the TSV spelling of the same array of floats.
fn tsv_f64_array(values: &[f64]) -> String {
    let items: Vec<String> = values.iter().map(|v| format!("{v:.6}")).collect();
    items.join("|")
}

/// `a|b|c` — the TSV spelling of the same array, `` (empty) when empty.
fn tsv_array<T: std::fmt::Display>(values: &[T]) -> String {
    let items: Vec<String> = values.iter().map(T::to_string).collect();
    items.join("|")
}

/// One cell as a single JSON line, shared by `--format json` (stdout) and `--json-out` (sidecar).
/// The partition fields are written on every row so a campaign has one schema; `barrier_ns` is
/// the sidecar's name for the engine's `collective_ns`. See machine contract (a) in
/// `benchmarks/PROFILING.md`.
fn json_line(cell: &CellResult) -> String {
    let s = &cell.stats;
    let empty = Vec::new();
    let terms_in = cell
        .partitioned
        .as_ref()
        .map_or(&empty, |p| &p.terms_in)
        .as_slice();
    let empty_ns = Vec::new();
    let coset_loop_ns = cell
        .partitioned
        .as_ref()
        .map_or(&empty_ns, |p| &p.coset_loop_ns)
        .as_slice();
    let partition_fields =
        format!(
        ",\"partitions\":{},\"partition_cpus\":\"{}\",\"pin_memory\":{},\"gen_qubits\":[{},{}],\
         \"local_layers\":{},\"remote_layers\":{},\"collectives\":{},\"rows_exported\":{},\
         \"bytes_exported\":{},\
         \"partition_terms_in\":{},\"partition_imbalance\":{:.6},\"export_ns\":{},\
         \"exchange_ns\":{},\"barrier_ns\":{},\"partition_coset_loop_ns\":{},\
         \"append_ns\":{},\"chunk_wait_ns\":{},\"initial\":\"{}\",\"partition_rows\":\"{}\",\
         \"partition_imbalance_by_layer\":{},\"terms_by_layer\":{},\
         \"rows_remote_gens\":{},\"rows_remote_weight\":{:.1}",
        cell.partitions,
        cell.partition_cpus,
        u8::from(cell.pin_memory),
        cell.gen_qubits.0,
        cell.gen_qubits.1,
        cell.partitioned.as_ref().map_or(0, |p| p.local_layers),
        cell.partitioned.as_ref().map_or(0, |p| p.remote_layers),
        cell.partitioned.as_ref().map_or(0, |p| p.collectives),
        cell.partitioned.as_ref().map_or(0, |p| p.rows_exported),
        cell.partitioned.as_ref().map_or(0, |p| p.bytes_exported),
        json_u64_array(terms_in),
        cell.partitioned.as_ref().map_or(1.0, |p| p.imbalance),
        s.export_ns,
        s.exchange_ns,
        s.collective_ns,
        json_u64_array(coset_loop_ns),
        s.append_ns,
        s.chunk_wait_ns,
        cell.initial,
        cell.partition_rows,
        json_f64_array(cell.partitioned.as_ref().map_or(&[][..], |p| &p.imbalance_by_layer)),
        json_u64_array(
            cell.partitioned
                .as_ref()
                .map_or(&[][..], |p| &p.terms_by_layer)
        ),
        cell.row_stats.remote_gens,
        cell.row_stats.remote_weight,
    );
    let core = format!(
        "{{\"layer\":\"{}\",\"truncation\":\"{}\",\"threads\":{},\"n\":{},\"reps\":{},\
         \"qubits\":{},\"seed\":{},\"hash_seed\":{},\"bucket_bits\":{},\
         \"wall_ns\":{},\"rebucket_ns\":{},\"prepare_ns\":{},\"rescale_ns\":{},\
         \"span_plan_ns\":{},\"permute_ns\":{},\"coset_loop_ns\":{},\"unpermute_ns\":{},\
         \"recount_ns\":{},\"finalize_ns\":{},\"swap_ns\":{},\"size_ns\":{},\
         \"gather_ns\":{},\"sort_ns\":{},\"merge_ns\":{},\"clear_ns\":{},\"layers\":{},\
         \"cosets\":{},\"runs\":{},\"rows_gathered\":{},\"rows_sorted\":{},\"rows_id\":{},\"terms_in\":{},\"terms_out\":{},\"vmrss_kb\":{},\
         \"vmhwm_kb\":{},\"target_bucket_len\":{},\"min_buckets\":{}",
        cell.layer,
        cell.truncation,
        cell.threads,
        cell.n,
        cell.reps,
        cell.qubits,
        cell.seed,
        cell.hash_seed,
        cell.bucket_bits,
        cell.wall_ns,
        s.rebucket_ns,
        s.prepare_ns,
        s.rescale_ns,
        s.span_plan_ns,
        s.permute_ns,
        s.coset_loop_ns,
        s.unpermute_ns,
        s.recount_ns,
        s.finalize_ns,
        s.swap_ns,
        s.size_ns,
        s.gather_ns,
        s.sort_ns,
        s.merge_ns,
        s.clear_ns,
        s.layers,
        s.cosets,
        s.runs,
        s.rows_gathered,
        s.rows_sorted,
        s.rows_id,
        s.terms_in,
        s.terms_out,
        cell.vmrss_kb,
        cell.vmhwm_kb,
        cell.target_bucket_len,
        cell.min_buckets,
    );
    let mpi_fields = match cell.mpi {
        Some((rank, ranks)) => format!(",\"rank\":{rank},\"ranks\":{ranks}"),
        None => String::new(),
    };
    format!("{core}{partition_fields}{mpi_fields}}}")
}

const TSV_HEADER: &str =
    "layer\ttruncation\tthreads\tn\treps\tqubits\tseed\twall_ns\trebucket_ns\tprepare_ns\t\
rescale_ns\tspan_plan_ns\tpermute_ns\tcoset_loop_ns\tunpermute_ns\trecount_ns\tfinalize_ns\t\
swap_ns\tsize_ns\tgather_ns\tsort_ns\tmerge_ns\tclear_ns\tlayers\tcosets\truns\trows_gathered\trows_sorted\trows_id\t\
terms_in\tterms_out\tvmrss_kb\tvmhwm_kb\ttarget_bucket_len\tmin_buckets\tpartitions\t\
partition_cpus\tpin_memory\tgen_qubits\tlocal_layers\tremote_layers\tcollectives\t\
rows_exported\t\
bytes_exported\tpartition_terms_in\tpartition_imbalance\texport_ns\texchange_ns\tbarrier_ns\t\
partition_coset_loop_ns\tappend_ns\tchunk_wait_ns\tinitial\tpartition_rows\t\
partition_imbalance_by_layer\tterms_by_layer\trows_remote_gens\trows_remote_weight";

fn print_tsv_row(cell: &CellResult) {
    let s = &cell.stats;
    let p = cell.partitioned.as_ref();
    let empty: Vec<usize> = Vec::new();
    let empty_ns: Vec<u64> = Vec::new();
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
         {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
         {}\t{}\t{}\t{}|{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t\
         {}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.1}",
        cell.layer,
        cell.truncation,
        cell.threads,
        cell.n,
        cell.reps,
        cell.qubits,
        cell.seed,
        cell.wall_ns,
        s.rebucket_ns,
        s.prepare_ns,
        s.rescale_ns,
        s.span_plan_ns,
        s.permute_ns,
        s.coset_loop_ns,
        s.unpermute_ns,
        s.recount_ns,
        s.finalize_ns,
        s.swap_ns,
        s.size_ns,
        s.gather_ns,
        s.sort_ns,
        s.merge_ns,
        s.clear_ns,
        s.layers,
        s.cosets,
        s.runs,
        s.rows_gathered,
        s.rows_sorted,
        s.rows_id,
        s.terms_in,
        s.terms_out,
        cell.vmrss_kb,
        cell.vmhwm_kb,
        cell.target_bucket_len,
        cell.min_buckets,
        cell.partitions,
        cell.partition_cpus,
        u8::from(cell.pin_memory),
        cell.gen_qubits.0,
        cell.gen_qubits.1,
        p.map_or(0, |p| p.local_layers),
        p.map_or(0, |p| p.remote_layers),
        p.map_or(0, |p| p.collectives),
        p.map_or(0, |p| p.rows_exported),
        p.map_or(0, |p| p.bytes_exported),
        tsv_array(p.map_or(&empty, |p| &p.terms_in)),
        p.map_or(1.0, |p| p.imbalance),
        s.export_ns,
        s.exchange_ns,
        // `barrier_ns` in every output format is the engine's `collective_ns`.
        s.collective_ns,
        tsv_array(p.map_or(&empty_ns, |p| &p.coset_loop_ns)),
        s.append_ns,
        s.chunk_wait_ns,
        cell.initial,
        cell.partition_rows,
        tsv_f64_array(p.map_or(&[][..], |p| &p.imbalance_by_layer)),
        tsv_array(p.map_or(&[][..], |p| &p.terms_by_layer)),
        cell.row_stats.remote_gens,
        cell.row_stats.remote_weight,
    );
}

/// Dispatch `--truncation` into exactly one monomorphization of [`run_cells`], so the policy's
/// `keep_term` inlines into the merge the way a real caller's does. Each spec supplies two policy
/// values: the unpartitioned one and its [`PartitionedTruncation`] form (`topn` has none, already
/// rejected by [`parse_args`]).
fn run<const W: usize>(cfg: &Config) {
    match cfg.truncation {
        TruncSpec::Keep => run_cells::<W, _, _>(cfg, &AlwaysKeep, Some(&AlwaysKeepPartitioned)),
        TruncSpec::Coeff(t) => run_cells::<W, _, _>(
            cfg,
            &CoefficientThreshold(t),
            Some(&CoefficientThreshold(t)),
        ),
        TruncSpec::TopN(n) => run_cells::<W, _, ApproxTopN>(cfg, &TopN(n), None),
        TruncSpec::ApproxTopN(n) => run_cells::<W, _, _>(cfg, &ApproxTopN(n), Some(&ApproxTopN(n))),
    }
}

fn run_cells<const W: usize, P, PP>(cfg: &Config, policy: &P, partitioned_policy: Option<&PP>)
where
    P: TruncationPolicy<W>,
    PP: PartitionedTruncation<W>,
{
    if cfg.format == Format::Tsv {
        println!("{TSV_HEADER}");
    }

    let mut sidecar = cfg.json_out.as_ref().map(|path| {
        // Under `--mpi` every rank writes its own file, suffixed by rank, to avoid interleaving.
        let path = mpi_sidecar_path(path, cfg);
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|e| {
                eprintln!("phase_breakdown: cannot open --json-out '{path}': {e}");
                std::process::exit(2);
            })
    });

    for &layer in &cfg.layers {
        for &partitions in &cfg.partitions {
            for &threads in &cfg.threads {
                #[cfg(feature = "mpi")]
                let cell = if cfg.mpi {
                    let policy = partitioned_policy.expect(
                        "an --mpi cell without a partitioned policy — parse_args rejects the one \
                         spec (topn) that has none",
                    );
                    run_cell_mpi::<W, PP>(layer, threads, cfg, policy)
                } else if partitions > 1 {
                    let policy = partitioned_policy.expect(
                        "a partitioned cell without a partitioned policy — parse_args rejects \
                         the one spec (topn) that has none",
                    );
                    run_cell_partitioned::<W, PP>(layer, threads, partitions, cfg, policy)
                } else {
                    run_cell::<W, P>(layer, threads, cfg, policy)
                };
                #[cfg(not(feature = "mpi"))]
                let cell = if partitions > 1 {
                    let policy = partitioned_policy.expect(
                        "a partitioned cell without a partitioned policy — parse_args rejects \
                         the one spec (topn) that has none",
                    );
                    run_cell_partitioned::<W, PP>(layer, threads, partitions, cfg, policy)
                } else {
                    run_cell::<W, P>(layer, threads, cfg, policy)
                };

                print_cell_line(&cell);
                match cfg.format {
                    Format::Table => print_table(&cell),
                    Format::Json => print_json(&cell),
                    Format::Tsv => print_tsv_row(&cell),
                }
                if let Some(f) = sidecar.as_mut() {
                    use std::io::Write;
                    writeln!(f, "{}", json_line(&cell)).unwrap_or_else(|e| {
                        eprintln!("phase_breakdown: writing --json-out failed: {e}");
                        std::process::exit(2);
                    });
                }
            }
        }
    }
}

/// The `--json-out` path this process writes: unchanged without `--mpi`, and
/// suffixed `.rank<N>` with it.
#[cfg(feature = "mpi")]
fn mpi_sidecar_path(path: &str, cfg: &Config) -> String {
    if !cfg.mpi {
        return path.to_string();
    }
    use paulistrings::mpi::rsmpi::topology::{Communicator, SimpleCommunicator};
    format!("{path}.rank{}", SimpleCommunicator::world().rank())
}

/// Without the `mpi` feature there is no rank, so the path is the path.
#[cfg(not(feature = "mpi"))]
fn mpi_sidecar_path(path: &str, _cfg: &Config) -> String {
    path.to_string()
}

/// The extra `--help` lines the `mpi` feature adds.
#[cfg(feature = "mpi")]
const MPI_USAGE: &str = "\
  --mpi                    Run each cell on the distributed driver: one
                            partition per rank of MPI_COMM_WORLD (D = 1),
                            placed by the launcher's affinity mask. Every rank
                            runs the whole matrix and reports its own numbers,
                            `vmhwm_kb` being peak RSS per rank including the
                            exchange transients. --partitions must stay at 1.
                            Each rank appends to its own --json-out sidecar,
                            suffixed `.rank<N>`, whose lines carry `rank` and
                            `ranks` fields. Launch it, e.g.:
                              mpirun -n 4 --map-by ppr:1:numa --bind-to numa \\
                                target/release/examples/phase_breakdown --mpi";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{USAGE}");
        #[cfg(feature = "mpi")]
        println!("{MPI_USAGE}");
        return;
    }

    let cfg = match parse_args(&args) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("phase_breakdown: {msg}");
            eprintln!();
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };

    // The universe outlives every cell; `Universe`'s drop is `MPI_Finalize`. The library never
    // creates one, so the probe does it here.
    #[cfg(feature = "mpi")]
    let _universe = cfg.mpi.then(|| {
        use paulistrings::mpi::rsmpi;
        // SERIALIZED, not FUNNELED: the layer loop runs inside a Rayon pool.
        let (universe, threading) = rsmpi::initialize_with_threading(rsmpi::Threading::Serialized)
            .unwrap_or_else(|| {
                eprintln!("phase_breakdown: MPI is already initialized in this process");
                std::process::exit(2);
            });
        if threading < rsmpi::Threading::Serialized {
            eprintln!(
                "phase_breakdown: warning: MPI provided {threading:?}, below the SERIALIZED the \
                 engine needs",
            );
        }
        universe
    });

    let words = cfg.qubits.div_ceil(64);
    match words {
        1 => run::<1>(&cfg),
        2 => run::<2>(&cfg),
        w => {
            eprintln!(
                "phase_breakdown: --qubits {} needs W={w} 64-bit words, but this probe only \
                 supports W in {{1, 2}} (qubits <= 128). Pass --qubits <= 128.",
                cfg.qubits,
            );
            std::process::exit(2);
        }
    }
}
