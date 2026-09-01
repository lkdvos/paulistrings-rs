"""Subprocess wrapper around the PauliPropagation.jl baseline runner.

The Julia baseline is deliberately **subprocess-only**. There is no PyJulia /
juliacall dependency anywhere in this repo, and there is not going to be:
``bench_baseline.py`` records the same exclusion for ``PauliStrings.jl``
("calling Julia from pytest would pull in PyJulia and isn't worth the
wiring"), and the adapted plan restates it as decision D6. This module writes
a task-JSON file, shells out to::

    julia --project=benchmarks/julia benchmarks/julia/runner.jl <task.json>

and parses the runner's single-line JSON result off stdout. Nothing here is
imported by CI: CI's python job runs ``pytest python/paulistrings/tests``
only.

Interchange format
------------------

The task JSON is schema v1, frozen in
``research/notes/2026-09-01-python-api-extensions.md`` §A5. :func:`make_task`
builds and validates one; :func:`run_task` executes it. The same dict drives
both engines, which is what makes the parity gate in ``test_julia_parity.py``
meaningful.

Graceful skipping
-----------------

* :func:`julia_available` — cheap boolean.
* :func:`skip_reason` — ``None`` when runnable, else a human-readable reason.
* :func:`importorskip_julia` — pytest-style skip (import pytest lazily), the
  analogue of ``pytest.importorskip`` for a binary rather than a module.

Standalone use::

    python benchmarks/python/julia_baseline.py path/to/task.json
    python benchmarks/python/julia_baseline.py --self-test
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --- Layout -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
JULIA_PROJECT = REPO_ROOT / "benchmarks" / "julia"
RUNNER = JULIA_PROJECT / "runner.jl"

# Julia is not on the default PATH of every Flatiron host; juliaup's shim
# directory is the other place it reliably lives. Lmod also provides Julia
# (``module load julia``), which puts it on PATH, so PATH is checked first.
_JULIA_FALLBACKS = (Path.home() / ".juliaup" / "bin" / "julia",)

DEFAULT_TIMEOUT_S = 3600.0


class JuliaBaselineError(RuntimeError):
    """The runner could not be started, failed, or produced unparseable output."""


# --- Discovery --------------------------------------------------------------


def find_julia() -> str | None:
    """Absolute path to a ``julia`` binary, or ``None`` if there is none."""
    env = os.environ.get("JULIA_BINARY")
    if env:
        return env if Path(env).exists() else None
    found = shutil.which("julia")
    if found:
        return found
    for candidate in _JULIA_FALLBACKS:
        if candidate.exists():
            return str(candidate)
    return None


def skip_reason() -> str | None:
    """``None`` if the baseline can run here, else why it cannot."""
    if find_julia() is None:
        return (
            "no `julia` binary found (checked $JULIA_BINARY, PATH, "
            f"{_JULIA_FALLBACKS[0]}); on Flatiron hosts try `module load julia`"
        )
    if not RUNNER.exists():
        return f"runner not found at {RUNNER}"
    if not (JULIA_PROJECT / "Manifest.toml").exists():
        return (
            f"{JULIA_PROJECT}/Manifest.toml is missing — the pinned "
            "PauliPropagation.jl environment is not checked out"
        )
    return None


def julia_available() -> bool:
    return skip_reason() is None


def importorskip_julia():
    """Skip the calling pytest test unless the Julia baseline can run.

    Mirrors ``pytest.importorskip``'s role for an external binary. Returns the
    julia path so a test can echo it into a report.
    """
    reason = skip_reason()
    if reason is not None:
        import pytest

        pytest.skip(reason, allow_module_level=False)
    return find_julia()


# --- Schema v1 task construction -------------------------------------------

#: Gate name -> the set of required fields beyond ``name``, and the expected
#: number of qubits (``None`` = derived from another field). Frozen schema v1;
#: keep in lockstep with ``benchmarks/julia/runner.jl``.
GATE_SPEC: Mapping[str, tuple[tuple[str, ...], int | None]] = {
    "h": ((), 1),
    "s": ((), 1),
    "x": ((), 1),
    "y": ((), 1),
    "z": ((), 1),
    "cnot": ((), 2),
    "cz": ((), 2),
    "swap": ((), 2),
    "rz": (("theta",), 1),
    "rx": (("theta",), 1),
    "ry": (("theta",), 1),
    "pauli_rotation": (("pauli", "theta"), None),
    "depolarize": (("p",), 1),
    "dephase": (("p",), 1),
    "amplitude_damping": (("gamma",), 1),
    "pauli_channel": (("px", "py", "pz"), 1),
    "depolarize2": (("p",), 2),
    "unitary_1q": (("matrix",), 1),
    "unitary_2q": (("matrix",), 2),
}

DIRECTIONS = ("forward", "heisenberg")


def gate(name: str, qubits: Sequence[int], **fields: Any) -> dict[str, Any]:
    """One schema-v1 gate object, validated.

    One gate object is one channel on both engines — the one-gate-per-channel
    parity rule is structural in this format (adapted plan §5, D10).
    """
    if name not in GATE_SPEC:
        raise ValueError(
            f"unknown gate name {name!r}; schema v1 vocabulary: {sorted(GATE_SPEC)}"
        )
    required, nq = GATE_SPEC[name]
    qs = [int(q) for q in qubits]
    if nq is not None and len(qs) != nq:
        raise ValueError(f"gate {name!r} takes {nq} qubit(s), got {qs}")
    if len(set(qs)) != len(qs):
        raise ValueError(f"gate {name!r} qubit indices must be distinct, got {qs}")
    missing = [k for k in required if k not in fields]
    if missing:
        raise ValueError(f"gate {name!r} is missing field(s) {missing}")
    extra = [k for k in fields if k not in required]
    if extra:
        raise ValueError(f"gate {name!r} does not take field(s) {extra}")
    if name == "pauli_rotation":
        pauli = str(fields["pauli"])
        if not pauli or any(ch not in "XYZ" for ch in pauli):
            raise ValueError(
                f"gate 'pauli_rotation' field 'pauli' must be non-empty over XYZ "
                f"(identity positions are expressed by omission), got {pauli!r}"
            )
        if len(pauli) != len(qs):
            raise ValueError(
                f"gate 'pauli_rotation' has {len(pauli)} Pauli letter(s) and {len(qs)} qubit(s)"
            )
    out: dict[str, Any] = {"name": name, "qubits": qs}
    out.update(fields)
    return out


def _coeff_json(value: Any) -> Any:
    """Serialize a coefficient as a number, or ``[re, im]`` when complex."""
    c = complex(value)
    if c.imag == 0.0:
        return c.real
    return [c.real, c.imag]


@dataclass(frozen=True)
class Task:
    """A schema-v1 task, plus the file it was written to (if any)."""

    payload: dict[str, Any]
    path: Path | None = None

    @property
    def n_qubits(self) -> int:
        return int(self.payload["n_qubits"])

    @property
    def gates(self) -> list[dict[str, Any]]:
        return list(self.payload["circuit"]["gates"])

    def write(self, path: str | os.PathLike[str]) -> Task:
        p = Path(path)
        p.write_text(json.dumps(self.payload, indent=2, sort_keys=False) + "\n")
        return Task(self.payload, p)


def make_task(
    *,
    n_qubits: int,
    gates: Iterable[Mapping[str, Any]],
    observable: Mapping[str, Any],
    direction: str,
    max_weight: int | None = None,
    min_abs_coeff: float | None = None,
    threads: int = 1,
    state: str | None = None,
) -> Task:
    """Build a validated schema-v1 task.

    ``direction`` is required and never defaulted — the README's stale
    "Heisenberg by default" claim is exactly the trap this guards (adapted
    plan D9). ``observable`` keys are full-length Pauli strings in the
    Hermitian-Y convention, leftmost character = qubit 0.
    """
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be positive, got {n_qubits}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    gate_list = [dict(g) for g in gates]
    for g in gate_list:
        for q in g["qubits"]:
            if not 0 <= q < n_qubits:
                raise ValueError(
                    f"gate {g['name']!r} qubit {q} out of range for n_qubits={n_qubits}"
                )
    obs: dict[str, Any] = {}
    for label, coeff in observable.items():
        if len(label) != n_qubits:
            raise ValueError(
                f"observable key {label!r} has length {len(label)}, expected {n_qubits}"
            )
        if any(ch not in "IXYZ" for ch in label):
            raise ValueError(f"observable key {label!r} must be over I, X, Y, Z")
        obs[label] = _coeff_json(coeff)
    if not obs:
        raise ValueError("observable must contain at least one term")

    payload: dict[str, Any] = {
        "version": 1,
        "n_qubits": int(n_qubits),
        "circuit": {"gates": gate_list},
        "observable": obs,
    }
    truncation: dict[str, Any] = {}
    if max_weight is not None:
        truncation["max_weight"] = int(max_weight)
    if min_abs_coeff is not None:
        truncation["min_abs_coeff"] = float(min_abs_coeff)
    if truncation:
        payload["truncation"] = truncation
    run: dict[str, Any] = {"direction": direction, "threads": int(threads)}
    if state is not None:
        run["state"] = state
    payload["run"] = run
    return Task(payload)


# --- Execution --------------------------------------------------------------


@dataclass
class JuliaResult:
    """Parsed runner output, with the raw payload kept for provenance."""

    raw: dict[str, Any]
    stderr: str = ""

    @property
    def expectation(self) -> complex | None:
        e = self.raw["result"]["expectation"]
        return None if e is None else complex(e["re"], e["im"])

    @property
    def final_terms(self) -> int:
        return int(self.raw["result"]["final_terms"])

    @property
    def per_layer_terms(self) -> list[int] | None:
        v = self.raw["result"]["per_layer_terms"]
        return None if v is None else [int(x) for x in v]

    @property
    def peak_terms(self) -> int | None:
        v = self.raw["result"]["peak_terms"]
        return None if v is None else int(v)

    @property
    def terms(self) -> dict[str, complex] | None:
        """Evolved sum term-by-term, when ``emit_terms`` asked for it."""
        v = self.raw["result"].get("terms")
        if v is None:
            return None
        return {label: complex(pair[0], pair[1]) for label, pair in v.items()}

    @property
    def wall_cold_s(self) -> float:
        return float(self.raw["timing"]["wall_cold_s"])

    @property
    def wall_warm_s(self) -> float | None:
        v = self.raw["timing"]["wall_warm_s"]
        return None if v is None else float(v)

    @property
    def versions(self) -> dict[str, str]:
        return dict(self.raw["versions"])

    @property
    def notes(self) -> list[str]:
        return list(self.raw.get("notes", []))


def run_task(
    task: Task | Mapping[str, Any],
    *,
    threads: int | None = None,
    warm_repeats: int | None = None,
    layer_counts: bool = True,
    backend: str | None = None,
    fused: bool = False,
    emit_terms: int = 0,
    timeout: float = DEFAULT_TIMEOUT_S,
    keep_task_file: str | os.PathLike[str] | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> JuliaResult:
    """Write ``task``, invoke the runner, parse the result JSON.

    ``threads`` overrides the julia worker count (``julia -t N``); it defaults
    to the task's own ``run.threads``, so a single-threaded task really does
    get a single-threaded Julia. Everything else maps onto the runner's
    documented environment knobs.
    """
    reason = skip_reason()
    if reason is not None:
        raise JuliaBaselineError(reason)
    julia = find_julia()
    assert julia is not None  # skip_reason() already checked

    payload = task.payload if isinstance(task, Task) else dict(task)
    n_threads = threads if threads is not None else int(payload["run"].get("threads", 1))

    env = dict(os.environ)
    env["PP_LAYER_COUNTS"] = "1" if layer_counts else "0"
    env["PP_FUSED"] = "1" if fused else "0"
    env["PP_EMIT_TERMS"] = str(int(emit_terms))
    if warm_repeats is not None:
        env["PP_WARM_REPEATS"] = str(int(warm_repeats))
    if backend is not None:
        env["PP_BACKEND"] = backend
    if extra_env:
        env.update(extra_env)

    with tempfile.TemporaryDirectory(prefix="ps-julia-task-") as tmp:
        task_path = Path(keep_task_file) if keep_task_file else Path(tmp) / "task.json"
        task_path.write_text(json.dumps(payload, indent=2) + "\n")
        cmd = [
            julia,
            f"--project={JULIA_PROJECT}",
            f"-t{n_threads}",
            str(RUNNER),
            str(task_path),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env
            )
        except FileNotFoundError as exc:  # pragma: no cover - discovery already ran
            raise JuliaBaselineError(f"could not execute {julia}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise JuliaBaselineError(
                f"runner timed out after {timeout}s: {' '.join(cmd)}"
            ) from exc

    if proc.returncode != 0:
        raise JuliaBaselineError(
            f"runner exited {proc.returncode}\ncmd: {' '.join(cmd)}\n"
            f"--- stderr ---\n{proc.stderr}\n--- stdout ---\n{proc.stdout}"
        )
    # The runner prints exactly one JSON line on stdout; take the last
    # non-empty line so a stray precompilation notice cannot break parsing.
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise JuliaBaselineError(
            f"runner produced no stdout\n--- stderr ---\n{proc.stderr}"
        )
    try:
        raw = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise JuliaBaselineError(
            f"could not parse runner output as JSON: {exc}\nlast line: {lines[-1]!r}\n"
            f"--- stderr ---\n{proc.stderr}"
        ) from exc
    return JuliaResult(raw, proc.stderr)


# --- CLI --------------------------------------------------------------------


def _self_test() -> int:
    """Run a 2-qubit task end-to-end and print the parsed result."""
    task = make_task(
        n_qubits=2,
        gates=[
            gate("h", [0]),
            gate("cnot", [0, 1]),
            gate("rz", [1], theta=0.3),
        ],
        observable={"ZI": 1.0, "IZ": 0.5},
        direction="heisenberg",
        min_abs_coeff=1e-12,
        state="z+",
    )
    res = run_task(task, warm_repeats=1)
    print(json.dumps(res.raw, indent=2))
    print()
    print(f"expectation      = {res.expectation}")
    print(f"final_terms      = {res.final_terms}")
    print(f"per_layer_terms  = {res.per_layer_terms}")
    print(f"wall cold / warm = {res.wall_cold_s:.4f} s / {res.wall_warm_s:.6f} s")
    return 0


def main(argv: Sequence[str]) -> int:
    reason = skip_reason()
    if reason is not None:
        print(f"julia baseline unavailable: {reason}", file=sys.stderr)
        return 77  # conventional "skipped" exit code
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: julia_baseline.py <task.json> | --self-test")
        return 0 if argv else 2
    if argv[0] == "--self-test":
        return _self_test()
    payload = json.loads(Path(argv[0]).read_text())
    res = run_task(Task(payload))
    print(json.dumps(res.raw, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
