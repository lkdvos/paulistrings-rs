//! Command-line options and their cross-flag validation.

use paulistrings::test_support::{
    B_MAX_BITS, DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
};
use paulistrings::{CpuSet, Placement};

use crate::workloads::{default_layers, LayerKind, HEAVY_HEX_QUBITS};

pub(crate) const USAGE: &str = "\
Usage: phase_breakdown [OPTIONS]

Measures the phase-timing breakdown of propagate_with across a
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
  --occupancy-at <rep>     0-based rep index to sample bucket occupancy at (diagnostic, opt-in;
                            absent by default). Runs ONE real cfg.reps-deep trajectory from the
                            cell's built initial sum, one propagate_with call
                            per rep (no untimed warm-up pass first: warm-up-then-repeat would
                            silently run the dynamics to depth 2 * --reps, which is fine for a
                            periodic single-gate layer but wrong for a growing Trotter-step
                            trajectory, since it would decouple the sampled step index from the
                            requested one). wall_ns/stats sum over that one trajectory via
                            PhaseStats::add. After the named rep, records num_buckets(), an
                            explicit empty-bucket count, and median/p95/max bucket_len() over
                            non-empty buckets, into the JSON sidecar. Not supported for `trotter`
                            (its circuit ignores --reps). Must be less than --reps.
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
                                            partitioned exact TopN is not
                                            yet supported.
                              atopn:<N>     ApproxTopN(N): the same, with a
                                            histogram boundary instead of a
                                            selection. Also keeps exactly N,
                                            but not the N largest.
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
  --format table|json      Output format (default: table)
  --json-out FILE          Also append one JSON line per cell to FILE,
                           regardless of --format (input for scripts/perf-viz.py)
  -h, --help               Print this message
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Table,
    Json,
}

/// Which [`TruncationPolicy`] every cell runs under, kept as a spec so `run` dispatches it into one monomorphization per variant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TruncSpec {
    Keep,
    Coefficient(f64),
    TopN(usize),
    ApproxTopN(usize),
}

impl TruncSpec {
    /// The spec as it was written on the command line, echoed into every output format.
    pub(crate) fn label(self) -> String {
        match self {
            TruncSpec::Keep => "keep".to_string(),
            TruncSpec::Coefficient(t) => format!("coeff:{t}"),
            TruncSpec::TopN(n) => format!("topn:{n}"),
            TruncSpec::ApproxTopN(n) => format!("atopn:{n}"),
        }
    }

    pub(crate) fn parse(s: &str) -> Result<Self, String> {
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
            return Ok(TruncSpec::Coefficient(thr));
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
pub(crate) enum PartitionCpus {
    /// One partition per NUMA node in the mask, capped at the cell's `P`.
    Auto,
    /// No pinning at all — the partition shape without its placement.
    Unpinned,
    /// One Linux cpulist per partition, exactly as written.
    Explicit(Vec<String>),
}

impl PartitionCpus {
    /// The spec as written, echoed into every output format.
    pub(crate) fn label(&self) -> String {
        match self {
            PartitionCpus::Auto => "auto".to_string(),
            PartitionCpus::Unpinned => "unpinned".to_string(),
            PartitionCpus::Explicit(lists) => lists.join(";"),
        }
    }

    pub(crate) fn parse(s: &str) -> Result<Self, String> {
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

    /// The [`Placement`] for one cell of `partitions` partitions.
    pub(crate) fn placement(&self, partitions: usize) -> Placement {
        match self {
            PartitionCpus::Auto => Placement::Auto {
                max_partitions: Some(partitions),
            },
            PartitionCpus::Unpinned => Placement::Unpinned { partitions },
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
pub(crate) enum Initial {
    /// `test_support::rand_sum(--n, --qubits, --seed)`.
    Random,
    /// A single `Z` on qubit `--qubits / 2`, coefficient 1. `--n` is then not a size at all.
    Z0,
}

impl Initial {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Initial::Random => "random",
            Initial::Z0 => "z0",
        }
    }

    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "random" => Ok(Initial::Random),
            "z0" => Ok(Initial::Z0),
            other => Err(format!("--initial expects random | z0, got '{other}'")),
        }
    }

    /// The input a layer takes when `--initial` is absent.
    pub(crate) fn default_for(layer: LayerKind) -> Self {
        if layer.is_trotter_step() {
            Initial::Z0
        } else {
            Initial::Random
        }
    }
}

/// `--partition-rows`: how the `log2(P)` GF(2) partition rows are chosen (`ARCHITECTURE.md §Partitioning`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PartitionRowSpec {
    /// `PartitionRows::from_seed`, the driver's own default.
    Random,
    /// `log2(P)` z-only rows labelling `P` contiguous qubit blocks. See [`cut_rows`].
    Cut,
}

impl PartitionRowSpec {
    pub(crate) fn label(self) -> &'static str {
        match self {
            PartitionRowSpec::Random => "random",
            PartitionRowSpec::Cut => "cut",
        }
    }

    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "random" => Ok(PartitionRowSpec::Random),
            "cut" => Ok(PartitionRowSpec::Cut),
            other => Err(format!(
                "--partition-rows expects random | cut, got '{other}'"
            )),
        }
    }
}

pub(crate) struct Config {
    pub(crate) n: usize,
    pub(crate) qubits: usize,
    /// Seed for the partitioning hash `H`. See `--hash-seed`.
    pub(crate) hash_seed: u64,
    /// Bucket bits to pre-refine the input sum to; 0 = leave it to the engine's own policy.
    pub(crate) bucket_bits: u8,
    /// Engine's per-layer target terms per bucket. See `--target-bucket-len`.
    pub(crate) target_bucket_len: usize,
    /// Engine's per-layer bucket-count floor. See `--min-buckets`.
    pub(crate) min_buckets: usize,
    /// Rep index (0-based) to sample bucket occupancy at. See `--occupancy-at`.
    pub(crate) occupancy_at: Option<usize>,
    /// TOTAL thread counts; a partitioned cell splits one of these over its partitions.
    pub(crate) threads: Vec<usize>,
    /// Partition counts to sweep. See `--partitions`.
    pub(crate) partitions: Vec<usize>,
    /// `--mpi`: run each cell on the distributed driver, one partition per rank. Only settable
    /// when the `mpi` feature is on.
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    pub(crate) mpi: bool,
    /// `--device <csv|auto>`: run each cell on CUDA devices. Only settable when the `cuda` feature is on.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) device: Option<DeviceSpec>,
    /// `--gpu-partitions`: virtual device partitions of a `--device` cell.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) gpu_partitions: usize,
    /// Placement spec for the partitioned cells. See `--partition-cpus`.
    pub(crate) partition_cpus: PartitionCpus,
    /// Whether each partition binds its allocations to its NUMA node.
    pub(crate) bind_memory: bool,
    /// Seed for the partition rows, or `None` for the driver's own choice.
    pub(crate) partition_seed: Option<u64>,
    /// How the partition rows are chosen. See `--partition-rows`.
    pub(crate) partition_rows: PartitionRowSpec,
    /// `--initial`, or `None` for each layer's own default ([`Initial::default_for`]).
    pub(crate) initial: Option<Initial>,
    pub(crate) layers: Vec<LayerKind>,
    pub(crate) reps: usize,
    pub(crate) seed: u64,
    pub(crate) truncation: TruncSpec,
    pub(crate) format: Format,
    /// Sidecar file that gets one JSON line appended per cell, regardless of `--format`.
    pub(crate) json_out: Option<String>,
}

/// `--device`: explicit ordinals, or every visible device (`auto`; under `--mpi`, the rank's local device).
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) enum DeviceSpec {
    Ordinals(Vec<u32>),
    Auto,
}

impl DeviceSpec {
    pub(crate) fn parse(s: &str) -> Result<Self, String> {
        if s.trim() == "auto" {
            return Ok(Self::Auto);
        }
        let ordinals = s
            .split(',')
            .map(|v| {
                v.trim()
                    .parse::<u32>()
                    .map_err(|_| format!("--device expects ordinals or 'auto', got '{s}'"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ordinals.is_empty() {
            return Err("--device must list at least one ordinal".to_string());
        }
        Ok(Self::Ordinals(ordinals))
    }
}

pub(crate) fn parse_usize(s: &str, flag: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|_| format!("{flag} expects a non-negative integer, got '{s}'"))
}

pub(crate) fn parse_seed(s: &str) -> Result<u64, String> {
    let t = s.trim();
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16)
            .map_err(|_| format!("--seed expects a hex (0x...) or decimal integer, got '{s}'"))
    } else {
        t.parse::<u64>()
            .map_err(|_| format!("--seed expects a hex (0x...) or decimal integer, got '{s}'"))
    }
}

pub(crate) fn parse_csv_usize(s: &str, flag: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .map(|tok| parse_usize(tok, flag))
        .collect()
}

pub(crate) fn parse_csv_layers(s: &str) -> Result<Vec<LayerKind>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .map(LayerKind::parse)
        .collect()
}

pub(crate) fn parse_format(s: &str) -> Result<Format, String> {
    match s {
        "table" => Ok(Format::Table),
        "json" => Ok(Format::Json),
        other => Err(format!("--format expects table or json, got '{other}'")),
    }
}

pub(crate) fn parse_args(args: &[String]) -> Result<Config, String> {
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
    let mut occupancy_at: Option<usize> = None;
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
    let mut device: Option<DeviceSpec> = None;
    let mut gpu_partitions: usize = 1;

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
            "--occupancy-at" => occupancy_at = Some(parse_usize(value, "--occupancy-at")?),
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
            "--device" => {
                if cfg!(not(feature = "cuda")) {
                    return Err(
                        "--device needs the `cuda` cargo feature (cargo run --release --features \
                         phase-timing,cuda --example phase_breakdown)"
                            .to_string(),
                    );
                }
                device = Some(DeviceSpec::parse(value)?);
            }
            "--gpu-partitions" => gpu_partitions = parse_usize(value, "--gpu-partitions")?,
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
    if let Some(step) = occupancy_at {
        if step >= reps {
            return Err(format!(
                "--occupancy-at {step} must be less than --reps ({reps}): it names a 0-based rep \
                 index to sample after"
            ));
        }
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
    // Partitioned exact `TopN` is not yet supported, so reject it here.
    let any_partitioned = partitions.iter().any(|&p| p > 1);
    if any_partitioned {
        if let TruncSpec::TopN(topn) = truncation {
            return Err(format!(
                "--truncation topn:{topn} cannot run a partitioned cell: partitioned exact TopN \
                 is not yet supported. Use \
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
                "--truncation topn:{topn} cannot run a distributed cell: partitioned exact \
                 TopN is not yet supported. Use --truncation atopn:{topn}"
            ));
        }
    }

    // The device axis: one process, `--gpu-partitions` virtual partitions on one device, no exact selection.
    if gpu_partitions == 0 || !gpu_partitions.is_power_of_two() {
        return Err(format!(
            "--gpu-partitions {gpu_partitions} must be a power of two"
        ));
    }
    if gpu_partitions > 1 && device.is_none() {
        return Err("--gpu-partitions needs --device".to_string());
    }
    if let Some(spec) = &device {
        if partitions != vec![1] {
            return Err(
                "--device places partitions with --gpu-partitions, so --partitions must stay at 1"
                    .to_string(),
            );
        }
        if mpi {
            if matches!(spec, DeviceSpec::Ordinals(v) if v.len() > 1) {
                return Err(
                    "--mpi --device takes one ordinal or 'auto': a rank drives one device"
                        .to_string(),
                );
            }
            if gpu_partitions > 1 {
                return Err(
                    "--mpi --device runs one device partition per rank, so --gpu-partitions must \
                     stay at 1"
                        .to_string(),
                );
            }
        } else if let DeviceSpec::Ordinals(v) = spec {
            check_device_partitions(v.len(), gpu_partitions, &layers)
                .map_err(|msg| format!("--device: {msg}"))?;
        }
        let one_device =
            !mpi && gpu_partitions == 1 && matches!(spec, DeviceSpec::Ordinals(v) if v.len() == 1);
        if let (TruncSpec::TopN(topn), false) = (truncation, one_device) {
            return Err(format!(
                "--truncation topn:{topn} runs on one device partition only (the n-th largest of \
                 a split sum has no collective form); use --truncation atopn:{topn}"
            ));
        }
        if occupancy_at.is_some() {
            return Err("--occupancy-at is not supported for a device cell".to_string());
        }
    }
    Ok(Config {
        n,
        qubits,
        hash_seed,
        bucket_bits,
        target_bucket_len,
        min_buckets,
        occupancy_at,
        threads,
        partitions,
        mpi,
        device,
        gpu_partitions,
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

/// `devices × gpu_partitions` must be a power of two, and above one when `layers` needs a remote delta.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn check_device_partitions(
    devices: usize,
    gpu_partitions: usize,
    layers: &[LayerKind],
) -> Result<(), String> {
    let total = devices * gpu_partitions;
    if !total.is_power_of_two() {
        return Err(format!(
            "{devices} device(s) x --gpu-partitions {gpu_partitions} = {total} partitions, which is not a power of two"
        ));
    }
    if total == 1 && layers.contains(&LayerKind::RotationRemote) {
        return Err("--layers rotation_remote needs more than one device partition".to_string());
    }
    Ok(())
}

/// The extra `--help` lines the `mpi` feature adds.
#[cfg(feature = "mpi")]
pub(crate) const MPI_USAGE: &str = "\
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

/// The extra `--help` lines the `cuda` feature adds.
#[cfg(feature = "cuda")]
pub(crate) const DEVICE_USAGE: &str = "\
  --device <csv|auto>      Run each cell on CUDA devices through GpuPartitionedSum, one
                            partition per ordinal (or auto, every visible device),
                            and with --mpi one device per rank (MpiGpuSum, one ordinal
                            or auto). The upload and download are untimed (upload_ns,
                            download_ns); wall_ns is the second call on the resident sum.
                            One cell per layer at --threads[0]; topn:<N> needs one
                            ordinal at one partition. benchmarks/PROFILING.md maps the
                            kernels onto the phases.
  --gpu-partitions <n>     With --device: <n> virtual partitions on each device
                            (devices x n a power of two, default 1).";
