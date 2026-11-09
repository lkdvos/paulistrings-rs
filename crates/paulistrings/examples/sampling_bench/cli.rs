//! Command-line options.

use crate::models::{IsingQuench, Observable};

pub(crate) const USAGE: &str = "\
Usage: sampling_bench [OPTIONS]

Propagates one observable through STEPS Trotter steps of one model, one propagate call per step
on a resident sum, and prints the wall time per step, the term counts per layer and the readout
<0...0|O|0...0>. Input generation, the upload and the final download are outside the timed steps.

Options:
  --model kicked-ising|tfim  kicked Ising on heavy-hex (theta_h = pi/4, Clifford ZZ, Z_62) or the
                             open square-lattice TFIM (J = 1, h = 3.04, dt = 0.1, second order,
                             magnetization) (default: kicked-ising)
  --lattice RxC              heavy-hex rows x width (default 7x15, the 127-qubit Eagle) or square
                             Lx x Ly (default 16x16)
  --observable site:Q|magnetization  (default: the model's)
  --steps N                  Trotter steps (default: 20)
  --backend cpu|cpus|gpu|gpus|mpi-gpu
                             the host engine; one sum split over --partitions host partitions
                             (PartitionedSum, --threads shared evenly); one CUDA device; one sum
                             split over --devices in this process (GpuPartitionedSum); or one
                             device per MPI rank over NCCL (MpiGpuSum, features cuda and mpi,
                             launched under mpirun or srun) (default: cpu)
  --threads N                host threads (default: all available)
  --device N|auto            the ordinal under gpu (default 0), or this rank's under mpi-gpu
                             (default auto: a device near the rank's CPUs)
  --devices LIST             gpus: comma-separated ordinals, one partition each, repeats allowed
                             (default: 0)
  --partitions N             cpus: host partitions (default: 2)
  --rows seeded|cut|quadrant|xparity:K|zparity:K
                             split backends: partition rows drawn from the sum's hash seed; a cut
                             of the qubits into equal contiguous blocks, under which every R_X is
                             local and only the bonds crossing blocks exchange; the same over four
                             lattice quadrants meeting at the observable's site (the lattice
                             centre for the magnetization), at 4 partitions; or log2(P) rows,
                             row i the parity of the x-bits (z-bits) of the i-th K qubits in
                             breadth-first order from the observable's site (the lattice centre
                             for the magnetization), under which every R_ZZ is local and only the
                             R_X on those qubits exchange (only the bonds leaving those groups)
                             (default: cut)
  --dtype f64|f32            coefficient type (default: f64)
  --threshold EPS            drop |c| <= EPS
  --approx-topn N            keep N terms per layer (octave cut)
  --partition-approx-topn N  keep N terms per layer on each partition alone, with no collective
                             (PerPartition(ApproxTopN(N)); the same as --approx-topn N unsplit)
  --collapse                 CollapseSample, with --cache N --samples M --seed S
  --trace                    after the timed run, repeat it with the layer trace on and report the per-kind,
                             phase, per-partition and pass figures from that run (it settles exact device term
                             counts, so its wall time is reported apart and is slower)
  --layers-out FILE          write one TSV row per applied layer of the traced run (needs --trace)
  --json-out FILE            append one JSON line with the run summary (rank 0 only under mpi-gpu)
  -h, --help                 print this message

Exactly one of --threshold, --approx-topn, --partition-approx-topn and --collapse is required.
The partition count of gpus and the rank count of mpi-gpu must be powers of two.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Model {
    KickedIsing,
    Tfim,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    Cpu,
    /// One sum over `partitions` host partitions of this process.
    Cpus,
    Gpu,
    /// One sum over `devices.len()` partitions of this process.
    Gpus,
    /// One partition per MPI rank, each on its own device.
    MpiGpu,
}

impl Backend {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cpus => "cpus",
            Backend::Gpu => "gpu",
            Backend::Gpus => "gpus",
            Backend::MpiGpu => "mpi-gpu",
        }
    }
}

/// How a split run assigns keys to partitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rows {
    Seeded,
    Cut,
    /// Each row the x-parity of `K` qubits nearest the observable's site.
    XParity(usize),
    /// Each row the z-parity of `K` qubits nearest the observable's site.
    ZParity(usize),
    /// Z-only rows labelling four lattice quadrants that meet at the observable's site.
    Quadrant,
}

impl Rows {
    pub(crate) fn name(self) -> String {
        match self {
            Rows::Seeded => "seeded".to_string(),
            Rows::Cut => "cut".to_string(),
            Rows::XParity(k) => format!("xparity:{k}"),
            Rows::ZParity(k) => format!("zparity:{k}"),
            Rows::Quadrant => "quadrant".to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Truncation {
    Threshold(f64),
    ApproxTopN(usize),
    /// `PerPartition(ApproxTopN(n))`: each partition keeps `n` on its own, with no collective.
    PartitionApproxTopN(usize),
    Collapse {
        cache: usize,
        samples: usize,
        seed: u64,
    },
}

impl Truncation {
    pub(crate) fn label(self) -> String {
        match self {
            Truncation::Threshold(eps) => format!("threshold:{eps:e}"),
            Truncation::ApproxTopN(n) => format!("approx-topn:{n}"),
            Truncation::PartitionApproxTopN(n) => format!("partition-approx-topn:{n}"),
            Truncation::Collapse { cache, samples, seed } => {
                format!("collapse:cache={cache},samples={samples},seed={seed}")
            }
        }
    }
}

pub(crate) struct Config {
    pub(crate) model: Model,
    pub(crate) lattice: Option<(usize, usize)>,
    pub(crate) observable: Option<Observable>,
    pub(crate) steps: usize,
    pub(crate) backend: Backend,
    pub(crate) threads: usize,
    /// `--device`; `None` is ordinal 0 under `gpu` and the rank's nearest device under `mpi-gpu`.
    pub(crate) device: Option<u32>,
    pub(crate) devices: Vec<u32>,
    pub(crate) partitions: usize,
    pub(crate) rows: Rows,
    pub(crate) dtype: &'static str,
    pub(crate) truncation: Truncation,
    pub(crate) trace: bool,
    pub(crate) layers_out: Option<String>,
    pub(crate) json_out: Option<String>,
}

impl Config {
    pub(crate) fn quench(&self) -> IsingQuench {
        match self.model {
            Model::KickedIsing => {
                let (rows, width) = self.lattice.unwrap_or((7, 15));
                let site = match self.observable {
                    Some(Observable::Site(q)) => Some(q),
                    Some(Observable::Magnetization) | None => None,
                };
                let mut quench = IsingQuench::kicked_ising(rows, width, site);
                if self.observable == Some(Observable::Magnetization) {
                    quench.observable = Observable::Magnetization;
                }
                quench
            }
            Model::Tfim => {
                let (lx, ly) = self.lattice.unwrap_or((16, 16));
                IsingQuench::square_tfim(
                    lx,
                    ly,
                    self.observable.unwrap_or(Observable::Magnetization),
                )
            }
        }
    }
}

fn value<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i).map(String::as_str).ok_or_else(|| format!("{flag} needs a value"))
}

fn number<T: std::str::FromStr>(text: &str, flag: &str) -> Result<T, String> {
    text.parse().map_err(|_| format!("{flag}: cannot parse '{text}'"))
}

pub(crate) fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut cfg = Config {
        model: Model::KickedIsing,
        lattice: None,
        observable: None,
        steps: 20,
        backend: Backend::Cpu,
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        device: None,
        devices: vec![0],
        partitions: 2,
        rows: Rows::Cut,
        dtype: "f64",
        truncation: Truncation::Threshold(0.0),
        trace: false,
        layers_out: None,
        json_out: None,
    };
    let mut truncations = 0;
    let mut collapse = false;
    let (mut cache, mut samples, mut seed) = (None, None, 0u64);
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "--model" => {
                cfg.model = match value(args, &mut i, flag)? {
                    "kicked-ising" => Model::KickedIsing,
                    "tfim" => Model::Tfim,
                    other => return Err(format!("unknown model '{other}'")),
                }
            }
            "--lattice" => {
                let text = value(args, &mut i, flag)?;
                let (a, b) = text.split_once('x').ok_or("--lattice takes AxB")?;
                cfg.lattice = Some((number(a, flag)?, number(b, flag)?));
            }
            "--observable" => {
                let text = value(args, &mut i, flag)?;
                cfg.observable = Some(match text.strip_prefix("site:") {
                    Some(q) => Observable::Site(number(q, flag)?),
                    None if text == "magnetization" => Observable::Magnetization,
                    None => return Err(format!("unknown observable '{text}'")),
                });
            }
            "--steps" => cfg.steps = number(value(args, &mut i, flag)?, flag)?,
            "--backend" => {
                cfg.backend = match value(args, &mut i, flag)? {
                    "cpu" => Backend::Cpu,
                    "cpus" => Backend::Cpus,
                    "gpu" => Backend::Gpu,
                    "gpus" => Backend::Gpus,
                    "mpi-gpu" => Backend::MpiGpu,
                    other => return Err(format!("unknown backend '{other}'")),
                }
            }
            "--threads" => cfg.threads = number(value(args, &mut i, flag)?, flag)?,
            "--device" => {
                cfg.device = match value(args, &mut i, flag)? {
                    "auto" => None,
                    text => Some(number(text, flag)?),
                }
            }
            "--devices" => {
                cfg.devices = value(args, &mut i, flag)?
                    .split(',')
                    .map(|d| number(d.trim(), flag))
                    .collect::<Result<_, _>>()?;
            }
            "--partitions" => cfg.partitions = number(value(args, &mut i, flag)?, flag)?,
            "--rows" => {
                let text = value(args, &mut i, flag)?;
                cfg.rows = match (text, text.split_once(':')) {
                    ("seeded", _) => Rows::Seeded,
                    ("cut", _) => Rows::Cut,
                    ("quadrant", _) => Rows::Quadrant,
                    (_, Some(("xparity", k))) => Rows::XParity(number(k, flag)?),
                    (_, Some(("zparity", k))) => Rows::ZParity(number(k, flag)?),
                    _ => return Err(format!("unknown rows '{text}'")),
                };
                if matches!(cfg.rows, Rows::XParity(0) | Rows::ZParity(0)) {
                    return Err("--rows xparity:K and zparity:K need K >= 1".into());
                }
            }
            "--dtype" => {
                cfg.dtype = match value(args, &mut i, flag)? {
                    "f64" => "f64",
                    "f32" => "f32",
                    other => return Err(format!("unknown dtype '{other}'")),
                }
            }
            "--threshold" => {
                cfg.truncation = Truncation::Threshold(number(value(args, &mut i, flag)?, flag)?);
                truncations += 1;
            }
            "--approx-topn" => {
                let n: f64 = number(value(args, &mut i, flag)?, flag)?;
                cfg.truncation = Truncation::ApproxTopN(n as usize);
                truncations += 1;
            }
            "--partition-approx-topn" => {
                let n: f64 = number(value(args, &mut i, flag)?, flag)?;
                cfg.truncation = Truncation::PartitionApproxTopN(n as usize);
                truncations += 1;
            }
            "--collapse" => {
                collapse = true;
                truncations += 1;
            }
            "--cache" => {
                let n: f64 = number(value(args, &mut i, flag)?, flag)?;
                cache = Some(n as usize);
            }
            "--samples" => {
                let n: f64 = number(value(args, &mut i, flag)?, flag)?;
                samples = Some(n as usize);
            }
            "--seed" => seed = number(value(args, &mut i, flag)?, flag)?,
            "--trace" => cfg.trace = true,
            "--layers-out" => cfg.layers_out = Some(value(args, &mut i, flag)?.to_string()),
            "--json-out" => cfg.json_out = Some(value(args, &mut i, flag)?.to_string()),
            other => return Err(format!("unknown option '{other}'")),
        }
        i += 1;
    }
    if truncations != 1 {
        return Err("exactly one of --threshold, --approx-topn, --partition-approx-topn and --collapse is required".into());
    }
    if collapse {
        let cache = cache.ok_or("--collapse needs --cache")?;
        cfg.truncation = Truncation::Collapse { cache, samples: samples.unwrap_or(cache), seed };
    } else if cache.is_some() || samples.is_some() {
        return Err("--cache and --samples apply to --collapse only".into());
    }
    if cfg.layers_out.is_some() && !cfg.trace {
        return Err("--layers-out needs --trace".into());
    }
    if cfg.steps == 0 || cfg.threads == 0 {
        return Err("--steps and --threads must be positive".into());
    }
    if cfg.backend == Backend::Cpus && !cfg.partitions.is_power_of_two() {
        return Err(format!("--partitions {} is not a power of two", cfg.partitions));
    }
    if cfg.backend == Backend::Gpus && !cfg.devices.len().is_power_of_two() {
        return Err(format!(
            "--devices names {} partitions, not a power of two",
            cfg.devices.len()
        ));
    }
    Ok(cfg)
}
