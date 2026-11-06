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
  --backend cpu|gpu          host engine or one CUDA device (default: cpu)
  --threads N                host threads (default: all available)
  --device N                 CUDA ordinal (default: 0)
  --dtype f64|f32            coefficient type (default: f64)
  --threshold EPS            drop |c| <= EPS
  --approx-topn N            keep N terms per layer (octave cut)
  --collapse                 CollapseSample, with --cache N --samples M --seed S
  --readout-every-step       read <O> after every step, outside the timed region
  --one-call                 propagate all steps in one call; step times come from the trace
  --layers-out FILE          write one TSV row per applied layer
  --json-out FILE            append one JSON line with the run summary
  -h, --help                 print this message

Exactly one of --threshold, --approx-topn and --collapse is required.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Model {
    KickedIsing,
    Tfim,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Truncation {
    Threshold(f64),
    ApproxTopN(usize),
    Collapse { cache: usize, samples: usize, seed: u64 },
}

impl Truncation {
    pub(crate) fn label(self) -> String {
        match self {
            Truncation::Threshold(eps) => format!("threshold:{eps:e}"),
            Truncation::ApproxTopN(n) => format!("approx-topn:{n}"),
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
    pub(crate) device: u32,
    pub(crate) dtype: &'static str,
    pub(crate) truncation: Truncation,
    pub(crate) readout_every_step: bool,
    pub(crate) one_call: bool,
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
        device: 0,
        dtype: "f64",
        truncation: Truncation::Threshold(0.0),
        readout_every_step: false,
        one_call: false,
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
                    "gpu" => Backend::Gpu,
                    other => return Err(format!("unknown backend '{other}'")),
                }
            }
            "--threads" => cfg.threads = number(value(args, &mut i, flag)?, flag)?,
            "--device" => cfg.device = number(value(args, &mut i, flag)?, flag)?,
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
            "--readout-every-step" => cfg.readout_every_step = true,
            "--one-call" => cfg.one_call = true,
            "--layers-out" => cfg.layers_out = Some(value(args, &mut i, flag)?.to_string()),
            "--json-out" => cfg.json_out = Some(value(args, &mut i, flag)?.to_string()),
            other => return Err(format!("unknown option '{other}'")),
        }
        i += 1;
    }
    if truncations != 1 {
        return Err("exactly one of --threshold, --approx-topn and --collapse is required".into());
    }
    if collapse {
        let cache = cache.ok_or("--collapse needs --cache")?;
        cfg.truncation = Truncation::Collapse { cache, samples: samples.unwrap_or(cache), seed };
    } else if cache.is_some() || samples.is_some() {
        return Err("--cache and --samples apply to --collapse only".into());
    }
    if cfg.steps == 0 || cfg.threads == 0 {
        return Err("--steps and --threads must be positive".into());
    }
    if cfg.one_call && cfg.readout_every_step {
        return Err("--one-call has no step boundaries to read out at".into());
    }
    Ok(cfg)
}
