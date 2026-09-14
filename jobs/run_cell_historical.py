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

Every cell here uses a small, fixed reduced scale (n_qubits=12, 2 Trotter
steps, theta_h=7pi/32, theta_zz=-pi/2 built into the workload, min_abs_coeff
from cell.json) chosen so even the unbucketed serial `naive_baseline` finishes
in well under a second -- see decisions.md for the exact rationale. This is
NOT the frozen 127-qubit canonical task (`contract.md`); it exists only to let
all variants run to completion for a fair relative comparison.

Writes one schema-v1 run record (`analysis/schema.py::validate_run`) per cell
to `<out-dir>/runs.jsonl` (append). No gate-trace record: none of the four
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
from run_cell import RUN_FIELDS, _append_jsonl, _compiler_version, _slurm_job_id  # noqa: E402

SCHEMA_VERSION = 1
CAMPAIGN_ID = "campaign-2026-09-11"

# Reduced-scale workload, shared by every historical cell (decisions.md's
# "historical-variants reduced scale" entry). Not the frozen 127-qubit
# canonical task -- see module docstring.
N_QUBITS = 12
TROTTER_STEPS = 2
THETA_H = 0.6872233929727672  # 7*pi/32, contract.md's primary point
THETA_ZZ = -1.5707963267948966  # -pi/2, contract.md's fixed value


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
    """One point of the historical-variant matrix, as read from `cell.json`."""

    variant_id: str
    min_abs_coeff: float
    direction: str
    repetition_index: int
    task_id: str = "T02-canonical-historical-reduced"
    config_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoricalCellSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"cell.json has unknown fields {sorted(unknown)}")
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
const MIN_ABS_COEFF: f64 = {min_abs_coeff};

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

    let policy = CoefficientThreshold(MIN_ABS_COEFF);
    let start = Instant::now();
    let evolved = propagate(&circuit, initial, &policy, Direction::Heisenberg);
    let wall_time_s = start.elapsed().as_secs_f64();

    println!(
        "{{{{\\"wall_time_s\\": {{}}, \\"initial_terms\\": {{}}, \\"final_terms\\": {{}}}}}}",
        wall_time_s, initial_terms, evolved.len()
    );
}}
"""


def _run_naive_rust_harness(spec: HistoricalCellSpec, worktree: Path, scratch: Path) -> dict[str, Any]:
    examples_dir = worktree / "crates" / "paulistrings" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    src = _NAIVE_EXAMPLE_SRC.format(
        n_qubits=N_QUBITS,
        trotter_steps=TROTTER_STEPS,
        theta_h=THETA_H,
        theta_zz=THETA_ZZ,
        min_abs_coeff=spec.min_abs_coeff,
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

    binary = target_dir / "release" / "examples" / "historical_smoke"
    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
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


def _run_pyo3_handrolled(spec: HistoricalCellSpec, worktree: Path, scratch: Path) -> dict[str, Any]:
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

    script = scratch / "workload.py"
    script.write_text(
        _HANDROLLED_WORKLOAD_SRC.format(
            n_qubits=N_QUBITS,
            trotter_steps=TROTTER_STEPS,
            theta_h=THETA_H,
            theta_zz=THETA_ZZ,
            min_abs_coeff=spec.min_abs_coeff,
            direction=spec.direction,
        )
    )

    rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result = subprocess.run(
        [str(_venv_python(venv_dir)), str(script)], check=True, capture_output=True, text=True
    )
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
        "n_qubits": N_QUBITS,
        "direction": spec.direction,
        "state": "z+",
        "min_abs_coeff": spec.min_abs_coeff,
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


def run_cell(spec: HistoricalCellSpec, out_dir: Path, scratch_root: Path) -> dict[str, Any]:
    """Run one historical-variant cell. Never raises for a cell that legitimately
    cannot run (bad hardware, unknown variant); only a genuine bug propagates.
    """
    run_id = str(uuid.uuid4())

    entry = VARIANT_REGISTRY.get(spec.variant_id)
    if entry is None:
        return _empty_run_record(
            spec, run_id, "other",
            f"variant_id={spec.variant_id!r} is not in VARIANT_REGISTRY "
            f"(known: {sorted(VARIANT_REGISTRY)})",
            None,
        )

    if os.environ.get("PS_HIST_SKIP_PREFLIGHT"):
        hardware = {"preflight_passed": True, "node_class_guess": "genoa (skip-preflight override)"}
    else:
        hardware = preflight.run_preflight()
    if not hardware["preflight_passed"]:
        return _empty_run_record(
            spec, run_id, "invalid_hardware",
            f"preflight failed: node_class_guess={hardware['node_class_guess']!r} "
            f"(contract wants {preflight.HARDWARE_CONTRACT_ID!r})",
            entry.commit_sha,
            node_class=str(hardware["node_class_guess"]),
        )

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
            payload = _run_naive_rust_harness(spec, worktree, scratch)
        elif entry.strategy == "pyo3_handrolled":
            payload = _run_pyo3_handrolled(spec, worktree, scratch)
        else:
            raise ValueError(f"unknown strategy {entry.strategy!r} for variant {spec.variant_id!r}")
    except subprocess.CalledProcessError as exc:
        reason = (
            f"{entry.strategy} failed for {spec.variant_id}@{entry.commit_sha[:12]}: "
            f"{exc.cmd} exit={exc.returncode} stderr_tail={exc.stderr[-2000:] if exc.stderr else ''}"
        )
        return _empty_run_record(spec, run_id, "build_failure", reason, entry.commit_sha)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=_REPO_ROOT, check=False)

    record = {
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
        "n_qubits": N_QUBITS,
        "direction": spec.direction,
        "state": "z+",
        "min_abs_coeff": spec.min_abs_coeff,
        "max_weight": None,
        "policy": f"CoefficientThreshold({spec.min_abs_coeff})",
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
    return record


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
    record = run_cell(spec, args.out_dir, args.scratch_root)
    _append_jsonl(args.out_dir / "runs.jsonl", [record])

    print(json.dumps({"run_id": record["run_id"], "status": record["status"]}))
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
