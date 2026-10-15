//! Command-line options and their cross-flag validation.

use paulistrings::{CpuSet, Placement};

use crate::workloads::{default_layers, LayerKind, HEAVY_HEX_QUBITS};

pub(crate) const USAGE: &str = "\
Usage: phase_breakdown [OPTIONS]

Times one warmed-up propagate_with call per cell (layer x partitions x threads) and prints its
phase breakdown. benchmarks/PROFILING.md documents every flag, the output contracts and the
measurement protocol.

Options:
  --n <usize>              Input term count (default: 1000000; trotter caps it at 100)
  --qubits <usize>         Qubit count, at most 128 (default: 128)
  --threads <csv>          Total thread counts (default: 1,8,16,32)
  --layers <csv>           rotation_zz, rotation_local, rotation_remote, cnot, gu2q, su4,
                           su4_local, depolarizing, trotter, tfim_step, heavyhex_step
                           (default: rotation_zz,cnot,gu2q,depolarizing,trotter)
  --reps <usize>           Channel repetitions per cell, Trotter steps for tfim_step and
                           heavyhex_step, ignored by trotter (default: 8)
  --seed <u64|0xHEX>       Seed of the input sum (default: 0xC0FFEE)
  --truncation <spec>      keep (default), coeff:<t>, topn:<N> or atopn:<N>; topn runs at
                           one partition only
  --partitions <csv>       Partition counts, powers of two dividing every --threads entry
                           (default: 1)
  --partition-cpus <spec>  auto (default, one partition per NUMA node), unpinned, or one
                           cpulist per partition, ';'-separated; run under no taskset/numactl
  --bind-memory 0|1        Bind each partition's memory to its NUMA node (default: 1)
  --partition-seed <seed>  Seed of random partition rows (default: the sum's hash seed)
  --partition-rows <spec>  random (default) or cut: z-only rows over contiguous qubit blocks
                           crossing the fewest of the layer's lattice edges
  --initial random|z0      Input sum: random, or Z on qubit --qubits/2 (the default for
                           tfim_step and heavyhex_step)
  --format table|json      Output format (default: table)
  --json-out FILE          Also append one JSON line per cell to FILE
  -h, --help               Print this message";

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
            return Ok(TruncSpec::ApproxTopN(parse_usize(v, "--truncation atopn:<N>")?));
        }
        Err(format!("--truncation expects keep | coeff:<t> | topn:<N> | atopn:<N>, got '{s}'"))
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
        let lists: Vec<String> =
            t.split(';').map(str::trim).filter(|tok| !tok.is_empty()).map(str::to_string).collect();
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
            PartitionCpus::Auto => Placement::Auto { max_partitions: Some(partitions) },
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

/// `--partition-rows`: how the `log2(P)` GF(2) partition rows are chosen (`PartitionRows`).
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
            other => Err(format!("--partition-rows expects random | cut, got '{other}'")),
        }
    }
}

pub(crate) struct Config {
    pub(crate) n: usize,
    pub(crate) qubits: usize,
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
    s.parse::<usize>().map_err(|_| format!("{flag} expects a non-negative integer, got '{s}'"))
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
    s.split(',').map(str::trim).filter(|tok| !tok.is_empty()).map(LayerKind::parse).collect()
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
                return Err("--mpi needs the `mpi` cargo feature (cargo run --release --features \
                     phase-timing,mpi --example phase_breakdown)"
                    .to_string());
            }
            mpi = true;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).ok_or_else(|| format!("{flag} requires a value"))?;
        match flag {
            "--n" => n = parse_usize(value, "--n")?,
            "--qubits" => qubits = parse_usize(value, "--qubits")?,
            "--threads" => threads = parse_csv_usize(value, "--threads")?,
            "--layers" => layers = parse_csv_layers(value)?,
            "--reps" => reps = parse_usize(value, "--reps")?,
            "--seed" => seed = parse_seed(value)?,
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
        return Err(format!("--gpu-partitions {gpu_partitions} must be a power of two"));
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
                return Err("--mpi --device takes one ordinal or 'auto': a rank drives one device"
                    .to_string());
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
    }
    Ok(Config {
        n,
        qubits,
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
pub(crate) const MPI_USAGE: &str =
    "  --mpi                    One partition per rank of MPI_COMM_WORLD, --threads per rank;
                           each rank writes its own sidecar, FILE.rank<N>";

/// The extra `--help` lines the `cuda` feature adds.
#[cfg(feature = "cuda")]
pub(crate) const DEVICE_USAGE: &str =
    "  --device <csv|auto>      Run each layer on these CUDA devices, one cell at --threads[0];
                           with --mpi, one ordinal or auto per rank
  --gpu-partitions <n>     Partitions per device (default: 1)";
