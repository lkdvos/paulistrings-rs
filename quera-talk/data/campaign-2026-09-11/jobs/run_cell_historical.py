"""Single-cell driver for the four historical Rust-revision variants (E1/E2 prep).

CLI: `python run_cell_historical.py <cell.json> --out-dir <dir>`. Unlike
`run_cell.py` (which only runs `variant_id="bucketed_current"` against the live
checkout), every `variant_id` this driver accepts names a historical commit in
`tasks/T01-variants.json` (`VARIANT_REGISTRY` below), built in a throwaway
`git worktree` -- never the live checkout -- because each commit's Python API
is incompatible with `examples/common/` in a different way (see decisions.md's
historical-variants entry for the full investigation):

- `naive_baseline` (d410f4e): the PyO3 bindings are `todo!()`-stubbed at this
  commit. Driven via a small Rust example (`_NAIVE_EXAMPLE_SRC` below),
  compiled with `cargo build --release --example`, run as a subprocess, and
  its one JSON stdout line parsed for the timing/term-count fields. Because
  this commit predates `Circuit::rx`/`rz`/`cnot` sugar, the ZZ interaction is
  built with the engine's native two-qubit `PauliRotation` generator directly
  -- not the CNOT-RZ-CNOT sandwich the other three variants use -- so final
  term counts are not expected to match those three exactly (a real, disclosed
  confounder, not a bug: truncation is applied per-channel, and the two
  decompositions expose a different intermediate circuit to it).
- `direct_small_sum_path` (e56f021), `bucketed_engine_serial` (f08db7d),
  `bucketed_engine_parallel` (ef03701): built via `maturin develop --release`
  into a per-variant venv, then run through `_HANDROLLED_WORKLOAD_SRC` below,
  a hand-built kicked-Ising circuit using only the gate methods each of these
  commits actually exposes (`Circuit.rx`/`.cnot`/`.rz`; no generic two-qubit
  rotation binding exists yet, hence the CNOT sandwich). `examples/common/`
  does not exist at `bucketed_engine_serial`/`bucketed_engine_parallel`, and
  `direct_small_sum_path`'s `PauliSum.propagate` has no `engine=` kwarg (that
  backport from current HEAD's `sum.rs`, decisions.md's option (a), was ruled
  out of scope for this pass -- see decisions.md) so `direct_small_sum_path`
  runs its default (sorted) engine only; the direct-apply path it is meant to
  demonstrate is not exercised. All three share one venv-build strategy and
  one workload script.

Every cell shares theta_h=7pi/32/theta_zz=-pi/2 built into the workload
(`THETA_H`/`THETA_ZZ` module constants). `n_qubits`, `trotter_steps`, and
`min_abs_coeff` are per-cell: `cell.json`'s `n_qubits` (default
`DEFAULT_N_QUBITS=12`), `trotter_steps` (default `TROTTER_STEPS=2`, both kept
for backward compatibility with the original toy-scale cells), and
`min_abs_coeff` (a float, or a list of floats to sweep in one invocation --
see `HistoricalCellSpec`). This linear-chain construction scaled up (not the
heavy-hex topology) is a deliberate, documented choice for the 2026-09-14
full-scale extension: it is already how the toy-scale comparison works, just
bigger, and building the actual heavy-hex 127-qubit task is unavailable to
three of the four historical commits anyway (see the strategy notes above).

**Real feasibility finding (2026-09-14, non-genoa workstation, see
decisions.md's "quera-talk full-scale historical sweep" entry for the full
table)**: at the original `trotter_steps=2` depth, wall time and final term
count are *flat* in `n_qubits` (12 through 127) and in `min_abs_coeff` alike
-- a backward (Heisenberg) propagation of a local single-site observable has
a light cone that stays local after only 2 layers, so scaling qubit count
alone produces a trivial, cost-flat curve for every variant, not the "cost
grows, weak variants fall behind" story the recurring figure needs. Real
depth, not qubit count, is what drives cost here. The 2026-09-14 full-scale
cells therefore fix `n_qubits=127` for every variant (verified free: n_qubits
doesn't move the needle at any depth tested) and raise `trotter_steps=10` to
get a real, cutoff-sensitive term-count curve; `naive_baseline`'s unbucketed
serial engine already costs ~130s locally for its single loosest-cutoff point
at that depth, so its cutoff list is deliberately just one point while the
other three (and `bucketed_current`) get the full 4-point grid -- see
`jobs/campaign-genoa-historical.sbatch` for the exact per-variant plan. This
is NOT the frozen 127-qubit canonical task (`contract.md`, which is 20 Trotter
steps on the heavy-hex lattice); it exists to let each variant run to
completion for a fair relative comparison at the largest scale/depth it can
actually reach in a bounded Slurm wall-time cap.

Writes one schema-v1 run record (`analysis/schema.py::validate_run`) per
(cell, cutoff) point to `<out-dir>/runs.jsonl` (append). No gate-trace record: none of the four
strategies exposes a per-gate `PropagationStats`-equivalent object (the Rust
harness has none by construction; the three PyO3 builds predate
`propagate_with_stats`), so `trace_enabled` is always `False` and
`gate_trace_path` always `None` -- mirroring the precedent already set by
`run_cell_julia.py` for its own genuinely-unavailable per-gate trace.

Preflight-gated like `run_cell.py`/`run_cell_julia.py`: an `invalid_hardware`
record on a non-`genoa` host, no worktree/build attempted. Set
`PS_HIST_SKIP_PREFLIGHT=1` to bypass this for **local plumbing validation
only** (this driver was developed and locally validated on a non-genoa
workstation) -- the checked-in `campaign-genoa-historical.sbatch` never sets
this, so a real cluster run always goes through the genuine hardware gate.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_JOBS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _JOBS_DIR.parents[2]
sys.path.insert(0, str(_JOBS_DIR))

import preflight  # noqa: E402

# NOT `from run_cell import ...`: run_cell.py's module-level `from common
# import circuits` chain requires `paulistrings` to already be importable
# (examples/common/circuits.py imports it at module scope), which is exactly
# what this driver cannot assume -- it orchestrates its own per-variant
# throwaway builds and is invoked by the bare interpreter before any of them
# exist. A real cluster run hit this as `ModuleNotFoundError: No module named
# 'paulistrings'` on every variant (job 7033257) because subagent-local
# testing had `.venv` (with paulistrings already installed for
# `bucketed_current`) active, masking it. These four helpers are pure stdlib
# and identical to run_cell.py's; RUN_FIELDS must be kept in sync with that
# module's copy by hand (schema.py's RUN_FIELDS constant is the actual
# validator both sides answer to, so a drift here fails loudly there).
RUN_FIELDS = (
    "schema_version", "campaign_id", "run_id", "task_id", "config_id",
    "variant_id", "repetition_index", "pair_index", "source_commit", "dirty",
    "build_features", "compiler_version", "runtime_version", "n_qubits",
    "direction", "state", "min_abs_coeff", "max_weight", "policy", "engine",
    "partitions", "threads", "ranks", "slurm_job_id", "node_class",
    "hardware_contract_id", "hardware_valid", "trace_enabled", "wall_time_s",
    "setup_time_s", "scatter_time_s", "gather_time_s", "initial_terms",
    "final_terms", "peak_terms", "peak_rss_kb", "peak_rss_provenance",
    "status", "failure_reason", "log_path", "gate_trace_path",
    "partition_row_policy",
)


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _compiler_version() -> str | None:
    """`rustc`'s own version string, or `None` if it's not on `PATH`."""
    try:
        out = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _slurm_job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID") or None

SCHEMA_VERSION = 1
CAMPAIGN_ID = "campaign-2026-09-11"

# Default n_qubits for a cell.json that omits the field, kept equal to the
# original toy scale (decisions.md #28) for backward compatibility. The
# 2026-09-14 feasibility pass (decisions.md, "quera-talk full-scale historical
# sweep") picks a real per-variant n_qubits well above this default for three
# of the four variants -- see that entry for the measured table.
DEFAULT_N_QUBITS = 12
TROTTER_STEPS = 2
THETA_H = 0.6872233929727672  # 7*pi/32, contract.md's primary point
THETA_ZZ = -1.5707963267948966  # -pi/2, contract.md's fixed value

#: The campaign-wide 4-point cutoff grid (campaign-genoa-convergence.sbatch,
#: campaign-genoa-e8.sbatch): 2^-12, 2^-14, 2^-16, 2^-18.
CUTOFF_GRID = (2.44140625e-04, 6.103515625e-05, 1.5258789e-05, 3.8146973e-06)


@dataclass(frozen=True)
class VariantEntry:
    commit_sha: str
    strategy: str  # "rust_harness" | "pyo3_handrolled"
    notes: str


#: variant_id -> historical commit + build/run strategy. Every commit_sha here
#: is the one T01/contract.md already froze (`tasks/T01-variants.json`).
VARIANT_REGISTRY: dict[str, VariantEntry] = {
    "naive_baseline": VariantEntry(
        commit_sha="d410f4e5985ad917146867be31511143fde8f893",
        strategy="rust_harness",
        notes=(
            "PyO3 bindings are todo!()-stubbed at this commit; driven via a "
            "Rust example, ZZ built with the native PauliRotation generator "
            "directly (no rx/rz/cnot sugar exists yet)."
        ),
    ),
    "direct_small_sum_path": VariantEntry(
        commit_sha="e56f021e54f3f64c3ddb8e2f688c39d91433d721",
        strategy="pyo3_handrolled",
        notes=(
            "PyO3 propagate() has no engine= kwarg at this commit (that is "
            "current HEAD's later addition); runs the default (sorted) "
            "engine only. The direct-apply path this variant is meant to "
            "demonstrate is NOT exercised -- see decisions.md."
        ),
    ),
    "bucketed_engine_serial": VariantEntry(
        commit_sha="f08db7df8bcd771f25383db0120e111cfb018bd2",
        strategy="pyo3_handrolled",
        notes="examples/common/ does not exist yet; circuit built by hand via Circuit.rx/.cnot/.rz.",
    ),
    "bucketed_engine_parallel": VariantEntry(
        commit_sha="ef037012e645d4f63013f26eaf3dbd6ce6299660",
        strategy="pyo3_handrolled",
        notes="Same API shape as bucketed_engine_serial; real rayon par_iter_mut parallelism confirmed by inspection.",
    ),
}

#: Field order, reused verbatim from run_cell.py so runs.jsonl stays one file
#: with one schema regardless of which driver wrote a given line.
_RUN_FIELDS = RUN_FIELDS


@dataclass(frozen=True)
class HistoricalCellSpec:
    """One point of the historical-variant matrix, as read from `cell.json`.

    `min_abs_coeff` is a *list* of cutoffs to sweep -- normalized from either
    a bare float or a list by `from_dict` -- so one invocation of this driver
    builds a variant's worktree/venv once and runs it once per cutoff, rather
    than paying a full rebuild per tolerance point (2026-09-14 full-scale
    extension; see decisions.md). `run_cell` returns one record per cutoff.
    """

    variant_id: str
    min_abs_coeff: tuple[float, ...]
    direction: str
    repetition_index: int
    n_qubits: int = DEFAULT_N_QUBITS
    trotter_steps: int = TROTTER_STEPS
    task_id: str = "T02-canonical-historical-reduced"
    config_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoricalCellSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"cell.json has unknown fields {sorted(unknown)}")
        data = dict(data)
        coeffs = data.get("min_abs_coeff")
        if coeffs is None:
            raise ValueError("cell.json must set min_abs_coeff (a float or a list of floats)")
        if isinstance(coeffs, (int, float)):
            data["min_abs_coeff"] = (float(coeffs),)
        else:
            data["min_abs_coeff"] = tuple(float(c) for c in coeffs)
        return cls(**data)


# --------------------------------------------------------------------------
# Rust-harness strategy (naive_baseline)
# --------------------------------------------------------------------------

_NAIVE_EXAMPLE_SRC = """\
//! Reduced-scale kicked-Ising harness for naive_baseline, generated by
//! run_cell_historical.py -- see that module's docstring for why this exists.
use num_complex::Complex64;
use paulistrings::accumulator::BuildAccumulator;
use paulistrings::channel::PauliRotation;
use paulistrings::circuit::Circuit;
use paulistrings::engine::{{propagate, Direction}};
use paulistrings::pauli_string::PauliString;
use paulistrings::phase::Phase;
use paulistrings::truncation::CoefficientThreshold;
use std::time::Instant;

const N_QUBITS: usize = {n_qubits};
const TROTTER_STEPS: usize = {trotter_steps};
const THETA_H: f64 = {theta_h};
const THETA_ZZ: f64 = {theta_zz};
// min_abs_coeff is a runtime CLI arg (argv[1]), not a compile-time const, so
// a tolerance sweep reuses one build across every cutoff -- see
// run_cell_historical.py's HistoricalCellSpec docstring.

fn x_rotation(q: usize) -> PauliRotation<1> {{
    let mut gen_x = [0u64; 1];
    gen_x[0] |= 1u64 << q;
    PauliRotation::<1> {{ support: vec![q as u32], gen_x, gen_z: [0u64; 1], theta: THETA_H }}
}}

fn zz_rotation(i: usize, j: usize) -> PauliRotation<1> {{
    let mut gen_z = [0u64; 1];
    gen_z[0] |= (1u64 << i) | (1u64 << j);
    PauliRotation::<1> {{ support: vec![i as u32, j as u32], gen_x: [0u64; 1], gen_z, theta: THETA_ZZ }}
}}

fn main() {{
    let min_abs_coeff: f64 = std::env::args()
        .nth(1)
        .expect("usage: historical_smoke <min_abs_coeff>")
        .parse()
        .expect("min_abs_coeff must parse as f64");

    let mut circuit = Circuit::<1>::new(N_QUBITS);
    for _ in 0..TROTTER_STEPS {{
        for q in 0..N_QUBITS {{
            circuit.push(x_rotation(q));
        }}
        for i in 0..N_QUBITS - 1 {{
            circuit.push(zz_rotation(i, i + 1));
        }}
    }}

    let mid = N_QUBITS / 2;
    let mut acc = BuildAccumulator::<1>::new(N_QUBITS);
    let mut ps = PauliString::<1> {{ x: [0], z: [0] }};
    ps.z[0] |= 1u64 << mid;
    acc.add_term(ps, Phase::ONE, Complex64::new(1.0, 0.0));
    let initial = acc.finalize();
    let initial_terms = initial.len();

    let policy = CoefficientThreshold(min_abs_coeff);
    let start = Instant::now();
    let evolved = propagate(&circuit, initial, &policy, Direction::Heisenberg);
    let wall_time_s = start.elapsed().as_secs_f64();

    println!(
        "{{{{\\"wall_time_s\\": {{}}, \\"initial_terms\\": {{}}, \\"final_terms\\": {{}}}}}}",
        wall_time_s, initial_terms, evolved.len()
    );
}}
"""


def _build_naive_rust_harness(spec: HistoricalCellSpec, worktree: Path, scratch: Path) -> Path:
    """Writes and builds the harness once (n_qubits is compile-time, min_abs_coeff is not).

    Returns the built binary's path so the caller can invoke it once per cutoff.
    """
    examples_dir = worktree / "crates" / "paulistrings" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    src = _NAIVE_EXAMPLE_SRC.format(
        n_qubits=spec.n_qubits,
        trotter_steps=spec.trotter_steps,
        theta_h=THETA_H,
        theta_zz=THETA_ZZ,
    )
    (examples_dir / "historical_smoke.rs").write_text(src)

    target_dir = scratch / "target"
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    subprocess.run(
        ["cargo", "build", "--release", "-p", "paulistrings", "--example", "historical_smoke"],
        cwd=worktree,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return target_dir / "release" / "examples" / "historical_smoke"


def _run_naive_rust_harness_once(binary: Path, min_abs_coeff: float) -> dict[str, Any]:
    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = subprocess.run([str(binary), str(min_abs_coeff)], check=True, capture_output=True, text=True)
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    # ru_maxrss is in KB on Linux; RUSAGE_CHILDREN accumulates, so the delta
    # is a lower bound on this subprocess's own peak, same caveat harness.py
    # documents for /proc/self/status's process-lifetime VmHWM.
    payload["peak_rss_kb"] = max(rusage_after - rusage_before, 0) or rusage_after
    return payload


# --------------------------------------------------------------------------
# PyO3-handrolled strategy (direct_small_sum_path, bucketed_engine_serial/parallel)
# --------------------------------------------------------------------------

_HANDROLLED_WORKLOAD_SRC = """\
import json
import time
import paulistrings as ps

N_QUBITS = {n_qubits}
TROTTER_STEPS = {trotter_steps}
THETA_H = {theta_h}
THETA_ZZ = {theta_zz}
MIN_ABS_COEFF = {min_abs_coeff}
EDGES = [(i, i + 1) for i in range(N_QUBITS - 1)]


def build_circuit():
    c = ps.Circuit(N_QUBITS)
    for _ in range(TROTTER_STEPS):
        for q in range(N_QUBITS):
            c.rx(THETA_H, q)
        for i, j in EDGES:
            c.cnot(i, j)
            c.rz(THETA_ZZ, j)
            c.cnot(i, j)
    return c


def main():
    mid = N_QUBITS // 2
    key = "".join("Z" if q == mid else "I" for q in range(N_QUBITS))
    observable = ps.PauliSum.from_strings({{key: 1.0}}, num_qubits=N_QUBITS)
    circuit = build_circuit()
    policy = ps.truncation.coeff(MIN_ABS_COEFF)

    start = time.perf_counter()
    evolved = observable.propagate(circuit, policy, direction="{direction}")
    wall_time_s = time.perf_counter() - start

    print(json.dumps({{"wall_time_s": wall_time_s, "initial_terms": 1, "final_terms": len(evolved)}}))


if __name__ == "__main__":
    main()
"""


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python3"


def _build_pyo3_handrolled(spec: HistoricalCellSpec, worktree: Path, scratch: Path) -> Path:
    """Builds the per-variant venv once via `maturin develop --release`.

    Returns the venv's python so the caller can run the (cheap-to-rewrite,
    no-recompile) workload script once per cutoff.
    """
    venv_dir = scratch / "venv"
    target_dir = scratch / "target"
    python_bin = shutil.which("python3.11") or sys.executable
    if not venv_dir.exists():
        subprocess.run([python_bin, "-m", "venv", str(venv_dir)], check=True, capture_output=True, text=True)
        pip = venv_dir / "bin" / "pip"
        subprocess.run([str(pip), "install", "--quiet", "--upgrade", "pip"], check=True, capture_output=True, text=True)
        subprocess.run(
            [str(pip), "install", "--quiet", "maturin>=1.5,<2.0", "numpy"],
            check=True,
            capture_output=True,
            text=True,
        )

    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = f"{venv_dir / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    subprocess.run(
        [str(venv_dir / "bin" / "maturin"), "develop", "--release", "-m", "crates/paulistrings-py/Cargo.toml"],
        cwd=worktree,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return _venv_python(venv_dir)


def _run_pyo3_handrolled_once(
    python_bin: Path, spec: HistoricalCellSpec, scratch: Path, min_abs_coeff: float
) -> dict[str, Any]:
    script = scratch / "workload.py"
    script.write_text(
        _HANDROLLED_WORKLOAD_SRC.format(
            n_qubits=spec.n_qubits,
            trotter_steps=spec.trotter_steps,
            theta_h=THETA_H,
            theta_zz=THETA_ZZ,
            min_abs_coeff=min_abs_coeff,
            direction=spec.direction,
        )
    )

    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = subprocess.run([str(python_bin), str(script)], check=True, capture_output=True, text=True)
    rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["peak_rss_kb"] = max(rusage_after - rusage_before, 0) or rusage_after
    return payload


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _empty_run_record(
    spec: HistoricalCellSpec,
    run_id: str,
    status: str,
    failure_reason: str,
    commit_sha: str | None,
    node_class: str = "unknown",
    min_abs_coeff: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "run_id": run_id,
        "task_id": spec.task_id,
        "config_id": spec.config_id,
        "variant_id": spec.variant_id,
        "repetition_index": spec.repetition_index,
        "pair_index": None,
        "source_commit": commit_sha or "unknown",
        "dirty": False,
        "build_features": [],
        "compiler_version": _compiler_version(),
        # schema.py requires a non-null str; this cell never got far enough to
        # import a built extension in this process (per-variant venvs live in
        # a worktree that's already been removed by the time this record is
        # written on a failure path).
        "runtime_version": "unknown",
        "n_qubits": spec.n_qubits,
        "direction": spec.direction,
        "state": "z+",
        "min_abs_coeff": min_abs_coeff if min_abs_coeff is not None else spec.min_abs_coeff[0],
        "max_weight": None,
        "policy": None,
        "engine": "unpartitioned",
        "partitions": 1,
        "threads": 1,
        "ranks": 1,
        "slurm_job_id": _slurm_job_id(),
        "node_class": node_class,
        "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
        "hardware_valid": False,
        "trace_enabled": False,
        "wall_time_s": None,
        "wall_time_s_reason": failure_reason,
        "setup_time_s": None,
        "scatter_time_s": None,
        "gather_time_s": None,
        "initial_terms": None,
        "final_terms": None,
        "peak_terms": None,
        "peak_rss_kb": None,
        "peak_rss_provenance": None,
        "status": status,
        "failure_reason": failure_reason,
        "log_path": None,
        "gate_trace_path": None,
        "partition_row_policy": None,
    }


def run_cell(spec: HistoricalCellSpec, out_dir: Path, scratch_root: Path) -> list[dict[str, Any]]:
    """Run one historical-variant cell across every cutoff in `spec.min_abs_coeff`.

    Builds the variant's worktree/venv exactly once (n_qubits is fixed at
    build time; min_abs_coeff is a runtime argument on both strategies), then
    runs once per cutoff. Returns one schema-v1 record per cutoff -- never
    raises for a cell that legitimately cannot run (bad hardware, unknown
    variant, a build failure); only a genuine bug propagates. A build failure
    or hardware-gate failure yields one record per requested cutoff, all
    carrying the same failure reason, so `len(records) == len(spec.min_abs_coeff)`
    always holds regardless of outcome.
    """
    entry = VARIANT_REGISTRY.get(spec.variant_id)
    if entry is None:
        reason = (
            f"variant_id={spec.variant_id!r} is not in VARIANT_REGISTRY "
            f"(known: {sorted(VARIANT_REGISTRY)})"
        )
        return [
            _empty_run_record(spec, str(uuid.uuid4()), "other", reason, None, min_abs_coeff=c)
            for c in spec.min_abs_coeff
        ]

    if os.environ.get("PS_HIST_SKIP_PREFLIGHT"):
        hardware = {"preflight_passed": True, "node_class_guess": "genoa (skip-preflight override)"}
    else:
        hardware = preflight.run_preflight()
    if not hardware["preflight_passed"]:
        reason = (
            f"preflight failed: node_class_guess={hardware['node_class_guess']!r} "
            f"(contract wants {preflight.HARDWARE_CONTRACT_ID!r})"
        )
        return [
            _empty_run_record(
                spec, str(uuid.uuid4()), "invalid_hardware", reason, entry.commit_sha,
                node_class=str(hardware["node_class_guess"]), min_abs_coeff=c,
            )
            for c in spec.min_abs_coeff
        ]

    scratch = scratch_root / spec.variant_id
    scratch.mkdir(parents=True, exist_ok=True)
    worktree = scratch / "src"

    if worktree.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=_REPO_ROOT, check=False)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), entry.commit_sha],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        if entry.strategy == "rust_harness":
            binary = _build_naive_rust_harness(spec, worktree, scratch)
            run_once = lambda c: _run_naive_rust_harness_once(binary, c)  # noqa: E731
        elif entry.strategy == "pyo3_handrolled":
            python_bin = _build_pyo3_handrolled(spec, worktree, scratch)
            run_once = lambda c: _run_pyo3_handrolled_once(python_bin, spec, scratch, c)  # noqa: E731
        else:
            raise ValueError(f"unknown strategy {entry.strategy!r} for variant {spec.variant_id!r}")
    except subprocess.CalledProcessError as exc:
        reason = (
            f"{entry.strategy} build failed for {spec.variant_id}@{entry.commit_sha[:12]}: "
            f"{exc.cmd} exit={exc.returncode} stderr_tail={exc.stderr[-2000:] if exc.stderr else ''}"
        )
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=_REPO_ROOT, check=False)
        return [
            _empty_run_record(spec, str(uuid.uuid4()), "build_failure", reason, entry.commit_sha, min_abs_coeff=c)
            for c in spec.min_abs_coeff
        ]

    records: list[dict[str, Any]] = []
    try:
        for min_abs_coeff in spec.min_abs_coeff:
            run_id = str(uuid.uuid4())
            try:
                payload = run_once(min_abs_coeff)
            except subprocess.CalledProcessError as exc:
                reason = (
                    f"{entry.strategy} run failed for {spec.variant_id}@{entry.commit_sha[:12]} "
                    f"min_abs_coeff={min_abs_coeff}: {exc.cmd} exit={exc.returncode} "
                    f"stderr_tail={exc.stderr[-2000:] if exc.stderr else ''}"
                )
                records.append(
                    _empty_run_record(
                        spec, run_id, "build_failure", reason, entry.commit_sha, min_abs_coeff=min_abs_coeff
                    )
                )
                continue

            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "campaign_id": CAMPAIGN_ID,
                    "run_id": run_id,
                    "task_id": spec.task_id,
                    "config_id": spec.config_id,
                    "variant_id": spec.variant_id,
                    "repetition_index": spec.repetition_index,
                    "pair_index": None,
                    "source_commit": entry.commit_sha,
                    "dirty": False,
                    "build_features": [],
                    "compiler_version": _compiler_version(),
                    # The historical Cargo.toml's `version.workspace = true` resolves to
                    # the same "0.1.0" at every one of these four commits (checked via
                    # `git show <sha>:crates/paulistrings-py/Cargo.toml`); this is real
                    # provenance, not a guess, just not independently queryable from this
                    # process (the built extension lives in a now-removed worktree venv).
                    "runtime_version": "0.1.0",
                    "n_qubits": spec.n_qubits,
                    "direction": spec.direction,
                    "state": "z+",
                    "min_abs_coeff": min_abs_coeff,
                    "max_weight": None,
                    "policy": f"CoefficientThreshold({min_abs_coeff})",
                    "engine": "unpartitioned",
                    "partitions": 1,
                    "threads": 1,
                    "ranks": 1,
                    "slurm_job_id": _slurm_job_id(),
                    "node_class": hardware["node_class_guess"],
                    "hardware_contract_id": preflight.HARDWARE_CONTRACT_ID,
                    "hardware_valid": True,
                    "trace_enabled": False,
                    "wall_time_s": payload["wall_time_s"],
                    "setup_time_s": None,
                    "scatter_time_s": None,
                    "gather_time_s": None,
                    "initial_terms": payload["initial_terms"],
                    "final_terms": payload["final_terms"],
                    "peak_terms": None,
                    "peak_rss_kb": payload.get("peak_rss_kb"),
                    "peak_rss_provenance": "rusage_children_maxrss_delta",
                    "status": "completed",
                    "failure_reason": None,
                    "log_path": None,
                    "gate_trace_path": None,
                    "partition_row_policy": None,
                    "extra": {"variant_notes": entry.notes},
                }
            )
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=_REPO_ROOT, check=False)

    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cell_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(os.environ.get("TMPDIR", "/tmp")) / "quera-historical-scratch",
        help="Where worktrees/venvs/target dirs are built (never the live checkout).",
    )
    args = parser.parse_args(argv)

    cell_data = json.loads(args.cell_json.read_text())
    spec = HistoricalCellSpec.from_dict(cell_data)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = run_cell(spec, args.out_dir, args.scratch_root)
    _append_jsonl(args.out_dir / "runs.jsonl", records)

    print(
        json.dumps(
            [{"run_id": r["run_id"], "min_abs_coeff": r["min_abs_coeff"], "status": r["status"]} for r in records]
        )
    )
    return 0 if all(r["status"] == "completed" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
