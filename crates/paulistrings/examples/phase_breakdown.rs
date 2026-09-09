//! Per-phase timing / memory probe for the bucketed propagation engine.
//!
//! Drives [`propagate_with_scratch_and_options`] over a small menu of single-channel
//! circuits (plus a multi-channel Trotter step) at a matrix of thread
//! counts, and prints the [`PhaseStats`] breakdown the `phase-timing`
//! feature exposes: per-layer wall-clock phases, per-coset worker busy
//! time, and derived throughput / overhead / memory figures.
//!
//! This binary only builds with `--features phase-timing` (see the
//! `required-features` entry in `Cargo.toml`); without the feature,
//! [`LayerScratch::take_stats`] does not exist, so there would be nothing
//! to read.
//!
//! Run with:
//! ```bash
//! cargo run --release --features phase-timing --example phase_breakdown -- \
//!     [--n 1000000] [--qubits 128] [--threads 1,8,16,32] \
//!     [--layers rotation_zz,cnot,gu2q,su4,depolarizing,trotter] [--reps 8] \
//!     [--seed 0xC0FFEE] [--truncation keep] [--format table|json|tsv] \
//!     [--partitions 1,2] [--partition-cpus '0-7,16-23;8-15,24-31'] \
//!     [--bind-memory 1] [--partition-seed <u64>] [--p1-path classic] \
//!     [--mpi]
//! ```
//!
//! `--qubits` picks the const-generic width `W` by `ceil(qubits / 64)`;
//! this probe only supports `W ∈ {1, 2}` (qubits ≤ 128), which is the same
//! set the crate's Python bindings restrict to for the smallest two
//! widths. Wider requests fail with a clear message rather than silently
//! truncating.
//!
//! Every layer except `trotter` is a single-channel circuit repeated
//! `--reps` times; `trotter` is a fixed 64-channel TFIM Trotter step (32
//! `ZZ` bond rotations + 32 transverse-field `X` rotations, copied from
//! `benches/pauli_ops.rs::bench_propagate_trotter`) and ignores `--reps`.
//! It also ignores `--n` above [`TROTTER_MAX_N`]: 64 *distinct* generators
//! under `AlwaysKeep` (no truncation) grow combinatorially rather than
//! closing to a bounded key set, and this was measured driving RSS into the
//! tens of GB well before `--n`'s default of 1,000,000 — see
//! `TROTTER_MAX_N`'s doc comment for the measurement. A capped cell prints
//! a note to stderr.
//!
//! `--truncation` selects the [`TruncationPolicy`] every cell runs under,
//! statically (one monomorphization per spec, so `keep_term` inlines into the
//! merge exactly as it does in a real caller — a `dyn` policy would change the
//! thing being measured):
//!
//! - `keep` (default) — `AlwaysKeep`, no filter and no finalize pass.
//! - `coeff:<t>` — [`CoefficientThreshold`]`(t)`: a `keep_term` filter inside
//!   the merge, and *no* `finalize_layer` work at all.
//! - `topn:<N>` — [`TopN`]`(N)`: no `keep_term`, all the cost in
//!   `finalize_layer` (three O(m) passes + a `select_nth_unstable`), which
//!   lands in the probe's `finalize` row. `N` is absolute, so pick it *below*
//!   the cell's steady-state term count — `TopN` returns immediately when
//!   `len <= N` and would otherwise be measured as free. Pairing a `coeff:0.0`
//!   run with a `topn:<N>` run at the same `--n` isolates the selection cost:
//!   the former's `finalize` is zero by construction.
//! - `atopn:<N>` — [`ApproxTopN`]`(N)`: the same shape of work as `topn`, but
//!   an octave histogram and a threshold instead of a candidate array and a
//!   selection, so `topn:<N>` versus `atopn:<N>` at one `--n` is the two
//!   policies' `finalize` cost side by side *on the same binary*. Note the two
//!   do not keep the same number of terms (`ApproxTopN` keeps `<= N`, short by
//!   at most one octave's population), so the steady-state `m` differs a
//!   little between the pair and the per-term figures are the ones to compare.
//!
//! Each `(layer, thread count)` cell runs the circuit twice inside a
//! dedicated Rayon thread pool of that width: an untimed warm-up call
//! (which, for the fanout-bounded channels, drives the input to its closed
//! key set so the timed call measures steady-state cost, not first-layer
//! growth), then the timed call whose input is the warm-up's output. Its
//! `PhaseStats` are read via [`LayerScratch::take_stats`] and its
//! `/proc/self/status` `VmRSS` / `VmHWM` are sampled right after.
//!
//! # Partitioned cells
//!
//! `--partitions <csv>` (default `1`) adds a partition axis to the matrix. A
//! cell with `P > 1` runs the same circuit through
//! [`PartitionedSum`] on a [`PartitionRuntime`] of `P` pinned pools instead of
//! one Rayon pool: the sum is scattered once *outside* the timed region, the
//! warm-up and the timed call are the same two calls as above, and the
//! per-partition [`PartitionTrace`] and [`PhaseStats`] are drained after each.
//!
//! **`--threads` stays the TOTAL thread count**: each partition's pool gets
//! `threads / P` workers (`PartitionRuntime::with_threads_per_partition`
//! overrides the width the placement would derive, leaving the CPU sets
//! alone), so `--threads 32 --partitions 1` and `--threads 32 --partitions 2`
//! put the same number of workers on the machine. A `--threads` value not
//! divisible by every `--partitions` value is an error, not a rounding.
//!
//! `P = 1` runs the *unpartitioned* path — today's `propagate_with_scratch_
//! and_options`, byte for byte — unless `--p1-path partitioned` is given, in
//! which case it goes through `PartitionedSum` with one partition. The pair
//! `--partitions 1 --p1-path partitioned` against `--partitions 1` (default)
//! is the machinery's own overhead: a pool build, a thread scope, a transport
//! group and a per-layer all-reduce of one number, over a code path the
//! driver's tests pin as bitwise identical.
//!
//! Placement comes from `--partition-cpus`:
//!
//! - absent (default) — `Placement::Auto { max_partitions: Some(P) }`: one
//!   partition per NUMA node in the affinity mask, rounded down to `P`.
//! - `'<list>;<list>;...'` — `Placement::Explicit`, one Linux cpulist per
//!   partition (`scripts/host-topology.sh`'s `PARTITION_CPUS` writes exactly
//!   this string). The list count must equal every `P > 1` in `--partitions`.
//! - `unpinned` — `Placement::Unpinned`: the shape of a partitioned run with
//!   no pinning at all, for a laptop or a shared box.
//!
//! `--bind-memory 0` drops the per-partition `set_mempolicy` binding (pinning
//! stays); `--partition-seed <u64>` fixes the partition rows instead of
//! letting the driver derive them from the sum's hash seed.
//!
//! # Distributed cells (`--mpi`, needs the `mpi` feature)
//!
//! `--mpi` swaps the in-process partition axis for ranks: every process holds
//! **one** partition (`D = 1`) and the group is `MPI_COMM_WORLD`, so
//! `--partitions` must stay at its default of 1. The cell shape is otherwise
//! `run_cell_partitioned`'s — scatter outside the timed region, one warm-up,
//! one timed call — and `--threads` is this rank's pool width, not a total,
//! because the placement comes from the launcher:
//!
//! ```bash
//! cargo build --release --features phase-timing,mpi --example phase_breakdown
//! mpirun -n 4 --map-by ppr:1:numa --bind-to numa \
//!     target/release/examples/phase_breakdown --mpi --threads 16 \
//!     --layers su4 --json-out results/mpi.jsonl
//! ```
//!
//! Every rank runs the whole matrix and reports **its own** numbers: one `cell`
//! line and one JSON object per rank per cell, each carrying `rank` and `ranks`
//! fields, and each rank appending to its own `--json-out` sidecar suffixed
//! `.rank<N>` (separate processes have no shared file position). The headline
//! is `vmhwm_kb` — peak resident set per rank, including the export and receive
//! transients, which is the capacity metric the distributed engine exists to
//! report; `partition_terms_in` and `partition_coset_loop_ns` have a single
//! entry (this rank's), and comparing ranks is the caller's job.
//!
//! Without `--mpi` nothing about the output changes, so a `--mpi`-capable
//! binary is still the binary for every other cell.
//!
//! The two partition-aware layers are `rotation_local` and `rotation_remote`:
//! a `ZZ` rotation on `(0, q)` where `q` is the smallest qubit in
//! `1..--qubits` whose layer has, respectively, no remote delta and at least
//! one — decided once per cell by [`count_remote_deltas`] against the hash the
//! sum carries and the partition rows the scatter will use (remoteness is a
//! property of the delta's key mask and those rows alone, so the bucket bits
//! the run later grows to do not enter). Both collapse to `rotation_zz`'s
//! `(0, 1)` at `P = 1`, where nothing is remote. The chosen pair is echoed to
//! stderr and recorded as `gen_qubits` in the JSON/TSV rows.
//!
//! `--truncation topn:<N>` is rejected for a partitioned cell: `TopN`'s exact
//! selection has no [`PartitionedTruncation`] impl (the bound rejects it at
//! compile time), and `atopn:<N>` is the collective form of the same policy.
//!
//! The input generators (`Xs64`, `rand_sum`, `low_weight_sum`) come from
//! `paulistrings::test_support`, shared with `benches/pauli_ops.rs` and the
//! crate's own tests. The per-layer channel recipes below are still duplicated
//! from the bench, since they are bench-shaped fixtures rather than fixtures
//! the library's tests use.

use std::time::Instant;

use num_complex::Complex64;
use paulistrings::bucket::hash::B_MAX_BITS;
use paulistrings::bucket::sum::{
    DEFAULT_HASH_SEED, DEFAULT_MIN_BUCKETS, DEFAULT_TARGET_BUCKET_LEN,
};
use paulistrings::channel::{Clifford2Q, Depolarizing, GeneralUnitary2Q, PauliRotation};
use paulistrings::engine::partitioned::{
    count_remote_deltas, CpuSet, PartitionConfig, PartitionPhaseStats, PartitionRuntime,
    PartitionTrace, PartitionedSum, PartitionedTruncation, Placement,
};
use paulistrings::engine::stats::TIMER_READ_OVERHEAD_NS;
use paulistrings::test_support::{haar_su4_matrix, low_weight_sum, rand_sum};
use paulistrings::truncation::{ApproxTopN, CoefficientThreshold, TopN};
use paulistrings::{
    propagate_with_scratch_and_options, Circuit, Direction, Gf2Hash, LayerScratch, PartitionRows,
    PauliString, PauliSum, PhaseStats, PropagateOptions, TruncationPolicy,
};

// ---------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------

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
                              cnot, gu2q, su4, depolarizing, trotter
                            (default: rotation_zz,cnot,gu2q,depolarizing,trotter
                            — su4 is opt-in, being the heaviest cell per --n:
                            a dense 16x16 PTM, so ~16x the fanout of gu2q's
                            sqrt(SWAP) and a ~16x larger closed key set)
  --reps <usize>           Channel repetitions per cell, ignored by trotter
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
                            research/notes/2026-09-01-bucket-cliff.md.
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
                            one unless --p1-path partitioned.
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
                            driver's own choice (the sum's hash seed).
  --p1-path classic|partitioned
                           Which path a P = 1 cell takes (default: classic,
                            i.e. today's propagate_with_scratch_and_options).
                            `partitioned` sends it through PartitionedSum with
                            one partition, which measures the machinery's
                            overhead against an otherwise identical run.
  --format table|json|tsv  Output format (default: table)
  --json-out FILE          Also append one JSON line per cell to FILE,
                           regardless of --format (input for scripts/perf-viz.py)
  -h, --help               Print this message
";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayerKind {
    RotationZz,
    /// A `ZZ` rotation chosen so the layer's deltas all stay inside their
    /// partition: the partitioned engine's best case, and at `P = 1` the same
    /// cell as [`LayerKind::RotationZz`].
    RotationLocal,
    /// A `ZZ` rotation chosen so at least one delta crosses partitions: the
    /// cell that pays for an export + exchange every layer.
    RotationRemote,
    Cnot,
    Gu2q,
    Su4,
    /// Haar SU(4) on a pair whose 15 non-identity deltas are all local under the
    /// partition rows: the dense, bandwidth-bound class with zero exchange.
    Su4Local,
    Depolarizing,
    Trotter,
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
            other => Err(format!(
                "unknown layer '{other}' (expected one of: rotation_zz, rotation_local, \
                 rotation_remote, cnot, gu2q, su4, su4_local, depolarizing, trotter)"
            )),
        }
    }

    /// Whether the layer's generator qubits are chosen per cell from the
    /// partition rows ([`choose_generator`]) rather than fixed at `(0, 1)`.
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

/// Which [`TruncationPolicy`] every cell runs under. Kept as a *spec* rather
/// than a boxed policy so `run` can dispatch it into one monomorphization per
/// variant: `keep_term` has to inline into the merge for the measurement to
/// mean anything.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TruncSpec {
    Keep,
    Coeff(f64),
    TopN(usize),
    ApproxTopN(usize),
}

impl TruncSpec {
    /// The spec as it was written on the command line — echoed into every
    /// output format so a raw log identifies its own policy.
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

/// `--partition-cpus`, kept as the *spec* the command line carried so it can
/// be echoed verbatim into the sidecar (`partition_cpus`, machine contract (a)
/// in `benchmarks/PROFILING.md`) and turned into a [`Placement`] once per
/// cell, where the partition count and the thread budget are known.
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

    /// The [`Placement`] for one cell: `partitions` partitions sharing
    /// `threads` workers in total.
    ///
    /// The caller has already checked the list count against `partitions`
    /// (see [`parse_args`]), so the `Explicit` arm cannot mismatch here.
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

/// What a `P = 1` cell measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum P1Path {
    /// Today's `propagate_with_scratch_and_options` on one Rayon pool — the
    /// default, and the reference point every partitioned number is read
    /// against.
    Classic,
    /// `PartitionedSum` with one partition: the same layer code with the
    /// driver's machinery (pool build, thread scope, transport group, the
    /// per-layer all-reduce) around it.
    Partitioned,
}

impl P1Path {
    fn parse(s: &str) -> Result<Self, String> {
        match s.trim() {
            "classic" => Ok(P1Path::Classic),
            "partitioned" => Ok(P1Path::Partitioned),
            other => Err(format!(
                "--p1-path expects classic | partitioned, got '{other}'"
            )),
        }
    }
}

struct Config {
    n: usize,
    qubits: usize,
    /// Seed for the partitioning hash `H`. See `--hash-seed`.
    hash_seed: u64,
    /// Bucket bits to pre-refine the input sum to; 0 = leave it to the
    /// engine's own `desired_bits` policy. See `--bucket-bits`.
    bucket_bits: u8,
    /// Engine's per-layer target terms per bucket. See `--target-bucket-len`.
    target_bucket_len: usize,
    /// Engine's per-layer bucket-count floor. See `--min-buckets`.
    min_buckets: usize,
    /// TOTAL thread counts; a partitioned cell splits one of these over its
    /// partitions. See `--threads` / `--partitions`.
    threads: Vec<usize>,
    /// Partition counts to sweep. See `--partitions`.
    partitions: Vec<usize>,
    /// `--mpi`: run each cell on the distributed driver, one partition per
    /// rank of `MPI_COMM_WORLD`. Only settable when the `mpi` feature is on —
    /// `parse_args` rejects the flag otherwise, so without the feature the
    /// field is always false and nothing reads it.
    #[cfg_attr(not(feature = "mpi"), allow(dead_code))]
    mpi: bool,
    /// Placement spec for the partitioned cells. See `--partition-cpus`.
    partition_cpus: PartitionCpus,
    /// Whether each partition binds its allocations to its NUMA node.
    bind_memory: bool,
    /// Seed for the partition rows, or `None` for the driver's own choice.
    partition_seed: Option<u64>,
    /// Which path a `P = 1` cell takes. See `--p1-path`.
    p1_path: P1Path,
    layers: Vec<LayerKind>,
    reps: usize,
    seed: u64,
    truncation: TruncSpec,
    format: Format,
    /// Sidecar file that gets one JSON line appended per cell, regardless of
    /// the stdout `--format` — the input `scripts/perf-viz.py` renders.
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
    let mut p1_path = P1Path::Classic;
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
            "--p1-path" => p1_path = P1Path::parse(value)?,
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
    // `desired_bits`'s "worth splitting" gate is non-monotone below 16
    // (crates/paulistrings/src/bucket/sum.rs), so the core documents the same
    // bound on `PropagateOptions::min_buckets`.
    if min_buckets < 16 {
        return Err(format!(
            "--min-buckets must be at least 16, got {min_buckets}"
        ));
    }
    if partitions.is_empty() {
        return Err("--partitions must list at least one partition count".to_string());
    }

    // The partition axis, checked against everything it interacts with before
    // a single cell runs: a campaign that dies on its fourth cell wastes the
    // three that ran.
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
    if p1_path == P1Path::Partitioned && partitions.contains(&1) {
        if let Some(sets) = explicit {
            if sets != 1 {
                return Err(format!(
                    "--p1-path partitioned runs the P = 1 cell through the partitioned engine, \
                     but --partition-cpus lists {sets} CPU sets: give one set, or drop \
                     --partition-cpus for the P = 1 cell"
                ));
            }
        }
    }
    // `TopN`'s exact selection needs a global view of the coefficients and has
    // no `PartitionedTruncation` impl — the bound rejects it at compile time,
    // so the probe has to reject it here rather than dispatch into a
    // partitioned cell that cannot exist.
    let any_partitioned = partitions
        .iter()
        .any(|&p| p > 1 || p1_path == P1Path::Partitioned);
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

    // `--mpi` is the distributed shape: the rank *is* the partition (D = 1), so
    // the in-process axis has to stay at one.
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
        p1_path,
        layers,
        reps,
        seed,
        truncation,
        format,
        json_out,
    })
}

// ---------------------------------------------------------------------
// Per-layer channel recipes (duplicated from benches/pauli_ops.rs)
// ---------------------------------------------------------------------

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

/// One fixed Haar-random SU(4) block on `(q0, q1)` — the probe's stand-in for
/// the *general matrix-gate* path.
///
/// [`sqrt_swap`] is a poor proxy for that path in two ways, both measured: its
/// PTM is sparse (steady-state fanout 3.65 rows gathered per input term at
/// `--qubits 128`, against a dense PTM's 16), and `sqrt(SWAP)^2 = SWAP` is
/// Clifford, so repeating it drives the term count into a **period-2 cycle**
/// (10 000 -> 32 503 -> 10 000 ... at `--n 10000`) rather than to a fixed
/// point. A generic SU(4) has neither property: every `U^k` stays generic, so
/// the PTM stays dense and the closed key set is a fixed ~16x the number of
/// distinct off-support key patterns.
///
/// The matrix is `test_support::haar_su4_matrix` — shared with the crate's
/// tests and `examples/delta_span_diagnostics.rs`, which need the same dense
/// PTM. See its doc comment for provenance.
fn haar_su4_block(q0: u32, q1: u32) -> GeneralUnitary2Q {
    GeneralUnitary2Q::from_matrix(q0, q1, haar_su4_matrix())
}

/// Qubit count of the fixed [`trotter_circuit`] chain — also the qubit
/// count `run_cell` uses when building `trotter`'s low-weight input, so
/// every weight-3 excitation lands inside the circuit's actual support.
const TROTTER_QUBITS: usize = 32;

/// Safety cap on `trotter`'s own input size, overriding `--n`.
///
/// `trotter` chains two full 64-layer circuit applications (the warm-up and
/// the timed call — see the "chain them" step in `run_cell`) under
/// `AlwaysKeep`, i.e. no truncation. Unlike rotation_zz/cnot/gu2q (a single
/// generator repeated, which provably closes to a bounded key set — see
/// `run_cell`), trotter is 64 *distinct* generators, and per-pass growth is a
/// roughly n-independent multiplicative factor
/// (measured ~600-700x per pass at `weight = 3`, `TROTTER_QUBITS = 32`, so
/// ~4-5×10^5x over the two chained passes). Left uncapped, `--n`'s default
/// of 1_000_000 would try to materialize on the order of 10^11 terms.
/// Measured directly at this cap: ~1×10^7 output terms, ~1 GB peak RSS,
/// finishes in single-digit seconds — see `run_cell` for the warning this
/// triggers when `--n` requests more.
const TROTTER_MAX_N: usize = 100;

/// The 64-channel TFIM Trotter step from
/// `benches/pauli_ops.rs::bench_propagate_trotter`: 32 `ZZ` bond rotations
/// (periodic boundary conditions) followed by 32 transverse-field `X`
/// rotations. Fixed shape ([`TROTTER_QUBITS`] qubits), independent of
/// `--qubits`/`--reps` — `--qubits` only sizes the input sum for every other
/// layer; `trotter`'s own input is sized off `TROTTER_QUBITS` instead (see
/// `run_cell`).
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

/// A truncation policy that never drops anything — mirrors the `AlwaysKeep`
/// helper used throughout the engine's own tests and `benches/pauli_ops.rs`.
struct AlwaysKeep;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeep {}

/// [`AlwaysKeep`] for a partitioned cell.
///
/// A separate type rather than an impl on `AlwaysKeep`, because
/// [`PartitionedTruncation`]'s default body rejects a policy whose
/// `finalizes_layer()` is the trait's conservative `true`, and flipping that
/// on `AlwaysKeep` would change what the *unpartitioned* cell measures: the
/// engine reads `finalizes_layer` when it decides between the bucketed and the
/// small-sum direct path (`PropagateOptions::starts_direct`). The two policies
/// mean the same thing — keep every term, do nothing per layer.
struct AlwaysKeepPartitioned;
impl<const W: usize> TruncationPolicy<W> for AlwaysKeepPartitioned {
    fn finalizes_layer(&self) -> bool {
        false
    }
}
impl<const W: usize> PartitionedTruncation<W> for AlwaysKeepPartitioned {}

/// Builds a cell's circuit. `gen_qubits` is the `(q0, q1)` pair the
/// `rotation_*` layers rotate about — `(0, 1)` for `rotation_zz`, and whatever
/// [`choose_generator`] picked for `rotation_local` / `rotation_remote`.
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
            // `su4` acts on (0, 1); `su4_local` on the pair `choose_generator`
            // picked so that all 15 non-identity deltas are local — the dense,
            // bandwidth-bound class with zero exchange, i.e. the pure NUMA cell.
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
    }
}

// ---------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------

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
    /// The cell's phase breakdown. For a partitioned cell this is
    /// [`fold_partition_stats`]'s cell-level view of the per-partition
    /// counters, not any one partition's.
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
    /// Everything only a partitioned cell has, `None` at `P = 1` classic.
    partitioned: Option<PartitionCellStats>,
    /// `(rank, ranks)` for a `--mpi` cell, `None` otherwise. Every field above
    /// is then **this rank's**: the numbers are per process, not per group, and
    /// `vmhwm_kb` in particular is the capacity metric a distributed run exists
    /// to report. `None` keeps the non-MPI output byte-identical.
    mpi: Option<(u32, u32)>,
}

/// The partition-axis numbers of one cell, from the timed call's
/// [`PartitionTrace`] and its per-partition [`PhaseStats`].
struct PartitionCellStats {
    /// Layers with no remote delta, so no export and no transport call.
    local_layers: usize,
    /// Layers with at least one remote delta.
    remote_layers: usize,
    /// Rows moved across partitions, summed over layers and senders.
    rows_exported: u64,
    /// Wire bytes for those rows.
    bytes_exported: u64,
    /// Σ over layers of each partition's input term count, by rank.
    terms_in: Vec<usize>,
    /// `max / mean` of [`Self::terms_in`]: 1.0 is a perfectly even split.
    imbalance: f64,
    /// Each partition's `coset_loop_ns`, by rank — the spread that says
    /// whether the exchange waits are imbalance or traffic.
    coset_loop_ns: Vec<u64>,
}

/// Read `VmRSS`/`VmHWM` (kB) from `/proc/self/status`. Linux-only, like the
/// rest of this crate's deployment targets; returns `0` for either field it
/// cannot find (e.g. running the example on a non-Linux host).
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

/// The cell's input sum, before any propagation: the seeded generator the
/// layer asks for, re-hashed and pre-refined per `--hash-seed` /
/// `--bucket-bits`.
///
/// Shared by the unpartitioned and the partitioned path, so a `P = 1` and a
/// `P = 2` cell of the same `(layer, --n, --seed)` propagate the same terms.
fn build_base_sum<const W: usize>(layer: LayerKind, cfg: &Config) -> PauliSum<W> {
    // `trotter` is 64 *distinct* generators applied once each, not one
    // generator repeated — the latter provably closes to a bounded key set,
    // which is what keeps rotation_zz/cnot/gu2q/su4 bounded here. A dense input
    // can anticommute with most of 64 distinct generators and blow up
    // combinatorially (`benches/pauli_ops.rs` puts it at "up to 2^64", and
    // benches only low-weight inputs for that reason): a dense `rand_sum`
    // input was measured driving this layer past 50 GB of RSS in well under a
    // minute at only n = 2000. So trotter alone gets a low-weight input sized
    // to its own fixed qubit count instead of `--qubits`, and its own `--n`
    // cap (see `TROTTER_MAX_N`).
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
        _ => rand_sum::<W>(cfg.n, cfg.qubits, cfg.seed),
    };
    // `--hash-seed` re-draws H's rows. It is *not* cosmetic: the rank of the
    // layer's bucket-delta span `h(D)` — hence the coset dimension `r`, hence
    // the sort's comparison count — depends on which rows H happens to have
    // (research/notes/2026-09-01-bucket-cliff.md). `with_hash` rescatters at
    // zero bucket bits; `--bucket-bits` then refines, and the engine's
    // grow-only `rebucket` keeps whatever it finds.
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
    // Nothing is remote without partitions, so `rotation_local` and
    // `rotation_remote` are both `rotation_zz` here (documented at the top of
    // the file, and echoed in the cell's `gen_qubits`).
    let gen_qubits = (0u32, 1u32);
    let circuit = build_circuit::<W>(layer, cfg.qubits, cfg.reps, gen_qubits);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build a rayon thread pool");

    // The engine's per-layer bucket policy for both the warm-up and the timed
    // call: `rebucket` is grow-only, so a coarse policy only means anything if
    // the warm-up never grew the partition past it.
    let options = PropagateOptions {
        target_bucket_len: cfg.target_bucket_len,
        min_buckets: cfg.min_buckets,
        ..PropagateOptions::default()
    };

    let (steady_n, wall_ns, stats) = pool.install(|| {
        let mut scratch = LayerScratch::<W>::new();

        // Untimed warm-up: for rotation_zz/cnot/gu2q/su4 this drives the input
        // to its closed key set (or, under a truncating policy, to the
        // steady state that policy admits), so the timed call below measures
        // steady-state cost rather than first-layer growth; for
        // depolarizing/trotter it just warms scratch/buffer capacity.
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
        // No split, so no partition numbers: the sidecar's partition fields
        // stay zero/empty on this row (machine contract (a)).
        partitioned: None,
        mpi: None,
    }
}

/// The `(q0, q1)` a `rotation_local` / `rotation_remote` cell rotates about:
/// the smallest `q1 > 0` whose one-channel `ZZ(0, q1)` layer has no remote
/// delta (`local`), respectively at least one (`remote`), under `rows`.
///
/// [`count_remote_deltas`] does the deciding, on a one-channel circuit and the
/// hash the sum currently carries. The verdict does not depend on the bucket
/// count: a delta is remote when the *partition rows* map its key mask off
/// zero, and those rows are fixed before the scatter, so a hash that later
/// grows bucket bits reclassifies nothing. The scan is `O(qubits)` prepares of
/// a single channel, once per cell, entirely outside the timed region.
///
/// # Panics
///
/// If no qubit in `1..qubits` gives the requested class — impossible for
/// `local` (the rows have `log2(P)` rows over `qubits` coordinates, so most
/// pairs are local) and possible in principle for `remote` on a pathological
/// row draw, in which case the message names `--partition-seed` as the knob.
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
    // A rotation's single non-identity delta is local for most pairs, so `(0, q)`
    // suffices. A dense SU(4) needs all 15 deltas on the pair local, i.e. the
    // partition rows zero on both qubits' x and z columns (1 in 16 pairs under
    // one random row), so it scans every pair — including ones not touching 0.
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

/// One partitioned cell: scatter (untimed), warm up, drain, time one
/// `propagate`, and read the trace and the per-partition counters.
///
/// The shape mirrors [`run_cell`] exactly — same input sum, same warm-up
/// rationale, same `PropagateOptions` — so the two differ only in the engine
/// underneath. Everything outside the timed call (building the runtime and its
/// `P` pinned pools, deriving the partition rows, choosing the generator,
/// scattering) happens before the clock starts.
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
    let num_qubits = base.num_qubits();

    let config = PartitionConfig {
        placement: cfg.partition_cpus.placement(partitions, threads),
        bind_memory: cfg.bind_memory,
        partition_row_seed: cfg.partition_seed,
    };
    // `--threads` is the TOTAL: the placement decides *where* each partition
    // runs, this decides how many workers it gets, so P pools of threads/P
    // compare against one pool of threads on the same machine.
    let runtime = PartitionRuntime::with_threads_per_partition(&config, Some(threads / partitions))
        .unwrap_or_else(|err| {
            eprintln!("phase_breakdown: cannot resolve the partition placement: {err}");
            std::process::exit(2);
        });
    // `Auto` reads the machine: it rounds the NUMA node count down to a power
    // of two and caps it at `max_partitions`, so asking for more partitions
    // than the host has nodes silently gets fewer. Fewer partitions than the
    // cell claims would mislabel every row, so refuse instead.
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

    // The rows the scatter would derive on its own — built here so the
    // generator scan below sees exactly the split the run will use.
    let rows = PartitionRows::<W>::from_seed(
        num_qubits,
        partitions.trailing_zeros() as u8,
        cfg.partition_seed.unwrap_or_else(|| base.hash().seed()),
    );
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

    let mut split = PartitionedSum::scatter_with_rows(base, rows, runtime);
    split.enable_trace();

    // Untimed warm-up, then its counters discarded — same contract as the
    // unpartitioned cell.
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
        hash_seed: cfg.hash_seed,
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
        partitioned: Some(summary),
        mpi: None,
    }
}

/// One cell on the **distributed** driver: this process is one rank's
/// partition, and the numbers it reports are its own.
///
/// The MPI counterpart of [`run_cell_partitioned`], and deliberately the same
/// shape — one untimed warm-up, one timed call, the counters drained between —
/// so a rank's row is comparable with an in-process partition's. What differs
/// is what a "partition" is: the runtime holds exactly one, placed by the
/// launcher's affinity mask (`mpirun --map-by ppr:1:numa --bind-to numa`), and
/// the group is the world communicator.
///
/// **Collective.** Every rank runs the identical cell matrix in the identical
/// order; the input comes from the same seed on every rank and the partition
/// rows from the same draw, so nothing here needs to be agreed at run time.
///
/// The headline is `vmhwm_kb`: peak resident set **per rank, including the
/// export and receive transients**, which is the capacity metric a multi-node
/// run exists to report.
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
    let num_qubits = base.num_qubits();

    // D = 1: one partition over whatever CPUs the launcher left us. `Auto`
    // reads that mask, so `--bind-to numa` is what places this rank.
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

    // The rows every rank derives, built here so the generator scan below sees
    // the split the run will use. Same seed on every rank, so same rows.
    let rows = PartitionRows::<W>::from_seed(
        num_qubits,
        ranks.trailing_zeros() as u8,
        cfg.partition_seed.unwrap_or_else(|| base.hash().seed()),
    );
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
    let mut split = DistributedSum::scatter_with_rows(base, transport, runtime, rows);
    split.enable_trace();

    // Untimed warm-up, counters discarded — the same contract as every other
    // cell. A barrier after it so the timed call starts together and
    // `exchange_ns` measures traffic rather than a straggling warm-up.
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
    // One "partition" here: this rank. The trace's per-layer vectors carry a
    // single entry for the same reason.
    let summary = summarize_partitions(1, &trace, &per_partition, &stats);

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
        partitioned: Some(summary),
        mpi: Some((rank, ranks)),
    }
}

/// The cell-level [`PhaseStats`] of a partitioned run: **wall-clock fields are
/// the maximum over partitions** (the critical path — the group is only as
/// fast as its slowest partition, and a partition's own wall phases already
/// sum to about its layer time), **busy-time and counter fields are sums**
/// (they are per-worker or per-term totals, and the cell's total work is the
/// group's).
///
/// `layers` is the *driver's* layer count, not a sum: every partition drove
/// the same layers, so summing would report `P × n` and break the per-layer
/// figures (and the `layers=` field of the `cell` line, which
/// `scripts/perf-stat.sh` reads).
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
        // Sub-phases of the two above, folded the same way so a row's parts
        // still add up to the whole on the partition that defined it.
        out.export_count_ns = out.export_count_ns.max(s.export_count_ns);
        out.export_fill_ns = out.export_fill_ns.max(s.export_fill_ns);
        out.send_post_ns = out.send_post_ns.max(s.send_post_ns);
        out.hdr_wait_ns = out.hdr_wait_ns.max(s.hdr_wait_ns);
        out.recv_alloc_ns = out.recv_alloc_ns.max(s.recv_alloc_ns);
        out.data_wait_ns = out.data_wait_ns.max(s.data_wait_ns);
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

/// The partition-axis summary of one timed call.
///
/// `rows_exported` comes from the folded counters and `bytes_exported` from
/// the trace's `bytes_sent`; the two are the send side of the same traffic
/// counted in rows and in wire bytes, by the export pass and the trace
/// respectively.
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
        rows_exported: folded.rows_exported,
        bytes_exported: trace
            .layers
            .iter()
            .flat_map(|layer| layer.bytes_sent.iter())
            .flat_map(|row| row.iter())
            .sum(),
        terms_in,
        imbalance,
        coset_loop_ns: per_partition
            .per_partition
            .iter()
            .map(|s| s.coset_loop_ns)
            .collect(),
    }
}

// ---------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------

/// Always printed first for every cell, in every format: `n=` and
/// `layers=` are a contract other scripts grep for (`scripts/perf-stat.sh`'s
/// awk; machine contract (b) in `benchmarks/PROFILING.md`). `trunc=` and then
/// `partitions=` are appended last so those greps keep matching unchanged.
fn print_cell_line(cell: &CellResult) {
    // The MPI suffix is empty for every non-`--mpi` cell, so the line is
    // byte-identical to what it was before the flag existed.
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

/// The partition block of the table format, printed only for a cell that ran
/// through the partitioned engine — an unpartitioned cell's table is exactly
/// what it was before the partition axis existed.
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
        "    export = {:.3} ms   exchange = {:.3} ms   barrier (bucket-count all-reduce) = \
         {:.3} ms   [max over partitions]",
        s.export_ns as f64 / 1e6,
        s.exchange_ns as f64 / 1e6,
        s.collective_ns as f64 / 1e6,
    );
    println!(
        "      export  = count {:.3} + fill {:.3} ms",
        s.export_count_ns as f64 / 1e6,
        s.export_fill_ns as f64 / 1e6,
    );
    println!(
        "      exchange = send/post {:.3} + header+early wait {:.3} + recv alloc {:.3} + \
         residual data wait {:.3} ms",
        s.send_post_ns as f64 / 1e6,
        s.hdr_wait_ns as f64 / 1e6,
        s.recv_alloc_ns as f64 / 1e6,
        s.data_wait_ns as f64 / 1e6,
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
}

fn print_json(cell: &CellResult) {
    println!("{}", json_line(cell));
}

/// `[a, b, c]` for a JSON array of integers, `[]` when empty.
fn json_u64_array<T: std::fmt::Display>(values: &[T]) -> String {
    let items: Vec<String> = values.iter().map(T::to_string).collect();
    format!("[{}]", items.join(","))
}

/// `a|b|c` — the TSV spelling of the same array, `` (empty) when empty.
fn tsv_array<T: std::fmt::Display>(values: &[T]) -> String {
    let items: Vec<String> = values.iter().map(T::to_string).collect();
    items.join("|")
}

/// One cell as a single JSON line — shared by `--format json` (stdout) and
/// `--json-out` (sidecar file for `scripts/perf-viz.py`).
///
/// The partition fields are written on **every** row, `partitions` included, so
/// a campaign mixing partitioned and unpartitioned cells has one schema; an
/// unpartitioned row carries `partitions: 1`, zero counters and empty arrays.
/// `barrier_ns` is the sidecar's name for the engine's `collective_ns` — the
/// driver's per-layer bucket-count all-reduce, the one unconditional collective
/// a layer makes. See machine contract (a) in `benchmarks/PROFILING.md`.
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
    let partition_fields = format!(
        ",\"partitions\":{},\"partition_cpus\":\"{}\",\"pin_memory\":{},\"gen_qubits\":[{},{}],\
         \"local_layers\":{},\"remote_layers\":{},\"rows_exported\":{},\"bytes_exported\":{},\
         \"partition_terms_in\":{},\"partition_imbalance\":{:.6},\"export_ns\":{},\
         \"exchange_ns\":{},\"barrier_ns\":{},\"partition_coset_loop_ns\":{},\
         \"export_count_ns\":{},\"export_fill_ns\":{},\"send_post_ns\":{},\"hdr_wait_ns\":{},\
         \"recv_alloc_ns\":{},\"data_wait_ns\":{},\"append_ns\":{},\
         \"chunk_wait_ns\":{}",
        cell.partitions,
        cell.partition_cpus,
        u8::from(cell.pin_memory),
        cell.gen_qubits.0,
        cell.gen_qubits.1,
        cell.partitioned.as_ref().map_or(0, |p| p.local_layers),
        cell.partitioned.as_ref().map_or(0, |p| p.remote_layers),
        cell.partitioned.as_ref().map_or(0, |p| p.rows_exported),
        cell.partitioned.as_ref().map_or(0, |p| p.bytes_exported),
        json_u64_array(terms_in),
        cell.partitioned.as_ref().map_or(1.0, |p| p.imbalance),
        s.export_ns,
        s.exchange_ns,
        s.collective_ns,
        json_u64_array(coset_loop_ns),
        s.export_count_ns,
        s.export_fill_ns,
        s.send_post_ns,
        s.hdr_wait_ns,
        s.recv_alloc_ns,
        s.data_wait_ns,
        s.append_ns,
        s.chunk_wait_ns,
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
partition_cpus\tpin_memory\tgen_qubits\tlocal_layers\tremote_layers\trows_exported\t\
bytes_exported\tpartition_terms_in\tpartition_imbalance\texport_ns\texchange_ns\tbarrier_ns\t\
partition_coset_loop_ns\texport_count_ns\texport_fill_ns\tsend_post_ns\thdr_wait_ns\t\
recv_alloc_ns\tdata_wait_ns\tappend_ns\tchunk_wait_ns";

fn print_tsv_row(cell: &CellResult) {
    let s = &cell.stats;
    let p = cell.partitioned.as_ref();
    let empty: Vec<usize> = Vec::new();
    let empty_ns: Vec<u64> = Vec::new();
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
         {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\
         {}\t{}\t{}\t{}|{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t\
         {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
        p.map_or(0, |p| p.rows_exported),
        p.map_or(0, |p| p.bytes_exported),
        tsv_array(p.map_or(&empty, |p| &p.terms_in)),
        p.map_or(1.0, |p| p.imbalance),
        s.export_ns,
        s.exchange_ns,
        // `barrier_ns` in every output format is the engine's `collective_ns`.
        s.collective_ns,
        tsv_array(p.map_or(&empty_ns, |p| &p.coset_loop_ns)),
        s.export_count_ns,
        s.export_fill_ns,
        s.send_post_ns,
        s.hdr_wait_ns,
        s.recv_alloc_ns,
        s.data_wait_ns,
        s.append_ns,
        s.chunk_wait_ns,
    );
}

/// Dispatch `--truncation` into exactly one monomorphization of
/// [`run_cells`], so the policy's `keep_term` inlines into the merge the way a
/// real caller's does. A `&dyn TruncationPolicy` would be one line shorter and
/// would change the thing being measured.
///
/// Each spec supplies two policy values: the one an unpartitioned cell runs
/// under, and its [`PartitionedTruncation`] form for a partitioned cell. They
/// are the same value for every spec that has both — only `keep` needs a
/// separate type (see [`AlwaysKeepPartitioned`]) — and `topn` has no
/// partitioned form at all, which [`parse_args`] has already rejected by the
/// time this runs.
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
        // Under `--mpi` every rank writes its own file: the ranks are separate
        // processes with no shared file position, and appending to one path
        // would interleave partial lines. The suffix is the rank, so a harness
        // globs `<path>.rank*` and each line carries its own `rank` field.
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
                } else if partitions > 1 || cfg.p1_path == P1Path::Partitioned {
                    let policy = partitioned_policy.expect(
                        "a partitioned cell without a partitioned policy — parse_args rejects \
                         the one spec (topn) that has none",
                    );
                    run_cell_partitioned::<W, PP>(layer, threads, partitions, cfg, policy)
                } else {
                    run_cell::<W, P>(layer, threads, cfg, policy)
                };
                #[cfg(not(feature = "mpi"))]
                let cell = if partitions > 1 || cfg.p1_path == P1Path::Partitioned {
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

    // The universe outlives every cell: the transports hold duplicates of its
    // communicator, and `Universe`'s drop is `MPI_Finalize`. The library never
    // creates one (see `paulistrings::mpi`), so the probe does it here.
    #[cfg(feature = "mpi")]
    let _universe = cfg.mpi.then(|| {
        use paulistrings::mpi::rsmpi;
        // SERIALIZED, not FUNNELED: the layer loop runs inside a Rayon pool, so
        // the thread issuing MPI calls is a pool worker — one at a time, but
        // not the main thread.
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
