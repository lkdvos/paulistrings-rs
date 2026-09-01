"""Results schema, JSON writer, and plot helpers for the examples/benchmarks suite.

Per `research/plans/2026-08-31-examples-benchmarks-suite.md` Part 0.5. This module
defines the machine-readable record a benchmark or showcase run produces
(`RunRecord`, wrapping a `Provenance` block), a JSON writer that appends into a
`benchmarks/results/<date>-<host>/`-style directory, and a set of matplotlib plot
helpers. `harness.py` (a later handoff item, not part of this file) is expected to
be the sole *producer* of `RunRecord`s; everything here just consumes or
serializes them, so this module has no dependency on `harness.py`, `circuits.py`,
`observables.py`, or `oracles.py`.

Results schema
---------------

``Provenance`` — one snapshot of "what produced this run", mirroring the spirit of
the provenance headers `scripts/bench-campaign.sh` writes into
`benchmarks/results/<date>-<host>/<name>.txt`:

- ``commit`` — ``git rev-parse HEAD`` (or ``"unknown"`` outside a git checkout).
- ``dirty`` — ``True`` iff the tracked worktree differs from ``HEAD`` (unstaged or
  staged changes), matching `scripts/bench-campaign.sh`'s
  ``git diff --quiet`` / ``git diff --cached --quiet`` check. ``None`` when git
  itself is unavailable. Untracked files never make this ``True`` — concurrent
  work elsewhere in a shared worktree must not poison every run's provenance.
- ``cpu_model`` — first ``model name`` line of ``/proc/cpuinfo`` (Linux only; a
  portable fallback is used elsewhere).
- ``python_version`` — ``platform.python_version()``.
- ``rustc_version`` — ``rustc -V`` output, or ``None`` if the toolchain isn't on
  ``PATH`` (pure-Python runs, e.g. a Julia-only comparison, legitimately lack it).
- ``library_versions`` — free-form ``{name: version}``, seeded with this
  environment's installed ``paulistrings`` version (when importable) and merged
  with whatever the caller supplies (qiskit, stim, PauliPropagation.jl, ...).
- ``seeds`` — free-form ``{purpose: seed}``, e.g. ``{"circuit": 1234}``; the caller
  owns naming since only it knows what was seeded.
- ``thread_count`` — the actual thread count the run executed under (harness.py's
  `assert_single_threaded`-verified count, or an explicit multi-thread count for a
  labeled scaling report per CLAUDE.md §Performance discipline). ``None`` if the
  caller didn't supply one.
- ``hostname`` — short hostname (``hostname -s`` equivalent).
- ``date`` — ISO calendar date the run was recorded.

``RunRecord`` — one benchmark/showcase run:

- ``engine`` — e.g. ``"paulistrings"``, ``"PauliPropagation.jl"``.
- ``engine_version`` — free-form version string for that engine.
- ``n_qubits`` — system size.
- ``direction`` — ``"forward"`` or ``"heisenberg"`` (never omitted, per the
  adapted plan's D9 rule that every suite call passes ``direction=`` explicitly).
- ``truncation`` — the truncation parameters actually used, as a plain dict (e.g.
  ``{"max_weight": 6, "min_abs_coeff": 1e-6}`` or ``{}`` for no truncation) —
  mirrors `harness.py`'s planned `make_policy` aliases (research/notes/
  2026-09-01-python-api-extensions.md, A7), not the policy object itself.
- ``propagation_time_s`` — warm wall time for the propagation step alone.
- ``contraction_time_s`` — warm wall time for the contraction/expectation step
  alone (``None`` when not measured separately, e.g. a run that only reports
  term counts).
- ``peak_terms`` — peak resident unique-term count during propagation (A2
  semantics: max over ``terms_in[0]`` and every ``terms_out[k]``), or ``None`` if
  unavailable.
- ``final_terms`` — final unique-term count after the last layer.
- ``expectation_value`` — the observable's expectation value for this run, or
  ``None`` when the run doesn't compute one (e.g. a pure term-growth benchmark).
  Kept real-valued: every observable in this suite is Hermitian, so its
  expectation is real to the tolerance the oracle assumes.
- ``absolute_error`` — ``abs(expectation_value - oracle_value)``, or ``None`` when
  no oracle was available for this run.
- ``peak_memory_kb`` — peak resident set size in KiB (``/proc/self/status``
  ``VmHWM``, the same source `phase_breakdown.rs` samples), or ``None``.
- ``provenance`` — the ``Provenance`` block above.
- ``extra`` — free-form ``{key: value}`` for parameters specific to one benchmark
  (e.g. ``theta_h``, ``jz``, a Trotter step count) that don't warrant a dedicated
  field on every run across the whole suite. Values must be JSON-serializable.

Deliberately absent: no exact-bitwise-output field. Per CLAUDE.md's determinism
policy, agreement is to floating-point tolerance; nothing in this schema is a
tripwire for output-bit stability.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, Mapping, Sequence


# --- Results schema ---------------------------------------------------------


@dataclass
class Provenance:
    """Reproducibility metadata for one run. See the module docstring."""

    commit: str
    dirty: bool | None
    cpu_model: str
    python_version: str
    rustc_version: str | None
    library_versions: dict[str, str] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    thread_count: int | None = None
    hostname: str = ""
    date: str = ""


@dataclass
class RunRecord:
    """One benchmark/showcase run. See the module docstring for field semantics."""

    engine: str
    engine_version: str
    n_qubits: int
    direction: str
    truncation: dict[str, Any]
    propagation_time_s: float
    final_terms: int
    provenance: Provenance
    contraction_time_s: float | None = None
    peak_terms: int | None = None
    expectation_value: float | None = None
    absolute_error: float | None = None
    peak_memory_kb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_time_s(self) -> float:
        """``propagation_time_s`` plus ``contraction_time_s`` (0 if unmeasured)."""
        return self.propagation_time_s + (self.contraction_time_s or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunRecord":
        data = dict(data)
        prov = data.pop("provenance")
        return cls(provenance=Provenance(**prov), **data)


# --- Provenance collection ---------------------------------------------------

_CPUINFO_PATH = Path("/proc/cpuinfo")


def _git_commit_and_dirty(repo_root: str | Path | None = None) -> tuple[str, bool | None]:
    """Mirror `scripts/bench-campaign.sh`'s commit/dirty check.

    ``dirty`` reflects tracked changes only (unstaged or staged vs ``HEAD``);
    untracked files never set it, so concurrent work elsewhere in a shared
    worktree doesn't poison provenance. Returns ``("unknown", None)`` outside a
    git checkout or when the ``git`` binary is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", None

    dirty = False
    for diff_args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        try:
            result = subprocess.run(
                ["git", *diff_args], cwd=repo_root, capture_output=True, text=True
            )
        except (FileNotFoundError, OSError):
            return commit, None
        if result.returncode != 0:
            dirty = True
    return commit, dirty


def _cpu_model() -> str:
    if _CPUINFO_PATH.exists():
        try:
            with _CPUINFO_PATH.open("r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _rustc_version() -> str | None:
    try:
        result = subprocess.run(["rustc", "-V"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return result.stdout.strip()


def _paulistrings_version() -> str | None:
    try:
        import importlib.metadata as _metadata

        return _metadata.version("paulistrings")
    except Exception:
        return None


def _short_hostname() -> str:
    return platform.node().split(".", 1)[0]


def collect_provenance(
    *,
    seeds: Mapping[str, int] | None = None,
    thread_count: int | None = None,
    extra_library_versions: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> Provenance:
    """Build a `Provenance` block for the calling process.

    `seeds` and `thread_count` are run-specific and only the caller knows them
    (`thread_count` should come from `harness.py`'s single-thread-pin
    verification, or an explicit value for a labeled scaling report); everything
    else is discovered from the environment.
    """
    commit, dirty = _git_commit_and_dirty(repo_root)
    library_versions: dict[str, str] = {}
    ps_version = _paulistrings_version()
    if ps_version is not None:
        library_versions["paulistrings"] = ps_version
    if extra_library_versions:
        library_versions.update(extra_library_versions)
    return Provenance(
        commit=commit,
        dirty=dirty,
        cpu_model=_cpu_model(),
        python_version=platform.python_version(),
        rustc_version=_rustc_version(),
        library_versions=library_versions,
        seeds=dict(seeds) if seeds else {},
        thread_count=thread_count,
        hostname=_short_hostname(),
        date=_date.today().isoformat(),
    )


def default_results_dir(base: str | Path = "benchmarks/results") -> Path:
    """`<base>/<today>-<short-hostname>`, matching `scripts/bench-campaign.sh`'s
    `benchmarks/results/$(date +%F)-$(hostname -s)` convention. Does not create
    the directory; `write_results` does that.
    """
    return Path(base) / f"{_date.today().isoformat()}-{_short_hostname()}"


# --- JSON writer -------------------------------------------------------------


def write_results(
    records: Sequence[RunRecord], output_dir: str | Path, name: str = "results"
) -> Path:
    """Append `records` as JSON to `<output_dir>/<name>.json`.

    `output_dir` is caller-supplied and expected to already follow the
    `benchmarks/results/<date>-<host>/` convention (see `default_results_dir`
    and `benchmarks/PROFILING.md`); this function only creates that directory
    if missing and writes into it. Existing file contents are loaded and
    extended, never overwritten — the same append-on-rerun discipline
    `bench-campaign.sh` uses for its own `.txt`/`.json` outputs. Returns the
    path written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"

    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            raise ValueError(f"{path} does not contain a JSON array; refusing to append")

    existing.extend(record.to_dict() for record in records)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(existing, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp_path, path)
    return path


def read_results(path: str | Path) -> list[RunRecord]:
    """Inverse of `write_results`: load every `RunRecord` from a results file."""
    with Path(path).open("r") as f:
        raw = json.load(f)
    return [RunRecord.from_dict(entry) for entry in raw]


# --- Plot helpers -------------------------------------------------------------
#
# matplotlib is imported lazily (inside each function) so importing this module
# stays possible in a numpy-only environment (`pyproject.toml`'s `examples`
# extra, which pulls matplotlib, is optional). Figures are saved as SVG.
#
# Color assignment follows the dataviz skill's categorical-by-identity rule:
# each engine name is assigned the next unused slot from the validated
# 8-color palette (research/... dataviz skill, `references/palette.md`), in
# first-seen order, cached per-process so the same engine keeps the same color
# across every plot in one script run. Axes never use a second (right-hand) y
# scale — genuinely different measures (time vs. memory) get their own subplot
# instead of a dual axis.

_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
_GRID_COLOR = "#e1e0d9"
_MUTED_TEXT = "#898781"

_engine_color_cache: dict[str, str] = {}


def _color_for_engine(engine: str) -> str:
    if engine not in _engine_color_cache:
        idx = len(_engine_color_cache) % len(_PALETTE)
        _engine_color_cache[engine] = _PALETTE[idx]
    return _engine_color_cache[engine]


def _group_by_engine(records: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    grouped: dict[str, list[RunRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.engine, []).append(rec)
    return grouped


def _style_axes(ax) -> None:
    """Recessive grid/axes per the dataviz skill: hairline grid, muted spines."""
    ax.grid(True, color=_GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED_TEXT)
    ax.tick_params(colors=_MUTED_TEXT)


def _save_or_return(fig, save_path: str | Path | None):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="svg", bbox_inches="tight")
    return fig


def plot_error_vs_runtime(
    records: Sequence[RunRecord],
    *,
    ax=None,
    save_path: str | Path | None = None,
):
    """One curve per engine: total wall time (x, log) vs. absolute error (y, log).

    Records with `absolute_error is None` (no oracle available) are skipped.
    Log-log axes since both quantities typically span orders of magnitude
    across a truncation sweep.
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    grouped = _group_by_engine(records)
    for engine, recs in grouped.items():
        points = sorted(
            (
                (r.total_time_s, r.absolute_error)
                for r in recs
                if r.absolute_error is not None and r.absolute_error > 0
            )
        )
        if not points:
            continue
        xs, ys = zip(*points)
        color = _color_for_engine(engine)
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=color, label=engine)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("absolute error vs. oracle")
    _style_axes(ax)
    if grouped:
        ax.legend(frameon=False)

    return _save_or_return(fig, save_path)


def plot_term_count_vs_truncation(
    records: Sequence[RunRecord],
    *,
    truncation_key: str = "min_abs_coeff",
    term_field: str = "final_terms",
    xscale: str = "log",
    ax=None,
    save_path: str | Path | None = None,
):
    """One curve per engine: a truncation parameter (x) vs. term count (y).

    `truncation_key` selects which entry of each record's `truncation` dict is
    the x value (default `"min_abs_coeff"`; pass `"max_weight"` and
    `xscale="linear"` for a weight-cap sweep). `term_field` selects
    `"final_terms"` or `"peak_terms"` on `RunRecord`. Records missing the key
    (or with a `None` term field) are skipped.
    """
    import matplotlib.pyplot as plt

    if term_field not in ("final_terms", "peak_terms"):
        raise ValueError(f"term_field must be 'final_terms' or 'peak_terms', got {term_field!r}")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    grouped = _group_by_engine(records)
    for engine, recs in grouped.items():
        points = []
        for r in recs:
            x = r.truncation.get(truncation_key)
            y = getattr(r, term_field)
            if x is None or y is None:
                continue
            points.append((x, y))
        points.sort()
        if not points:
            continue
        xs, ys = zip(*points)
        color = _color_for_engine(engine)
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=color, label=engine)

    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(truncation_key)
    ax.set_ylabel(term_field.replace("_", " "))
    _style_axes(ax)
    if grouped:
        ax.legend(frameon=False)

    return _save_or_return(fig, save_path)


def plot_time_and_memory_vs_size(
    records: Sequence[RunRecord],
    *,
    save_path: str | Path | None = None,
):
    """Two side-by-side subplots (never a dual y-axis): wall time vs. `n_qubits`
    and peak memory vs. `n_qubits`, one curve per engine on each. Records
    missing `peak_memory_kb` are skipped in the memory panel only.
    """
    import matplotlib.pyplot as plt

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(9, 4))

    grouped = _group_by_engine(records)
    any_time = False
    any_mem = False
    for engine, recs in grouped.items():
        color = _color_for_engine(engine)

        time_points = sorted((r.n_qubits, r.total_time_s) for r in recs)
        if time_points:
            any_time = True
            xs, ys = zip(*time_points)
            ax_time.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=color, label=engine)

        mem_points = sorted(
            (r.n_qubits, r.peak_memory_kb) for r in recs if r.peak_memory_kb is not None
        )
        if mem_points:
            any_mem = True
            xs, ys = zip(*mem_points)
            ax_mem.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=color, label=engine)

    ax_time.set_yscale("log")
    ax_time.set_xlabel("n qubits")
    ax_time.set_ylabel("wall time (s)")
    _style_axes(ax_time)
    if any_time:
        ax_time.legend(frameon=False)

    ax_mem.set_yscale("log")
    ax_mem.set_xlabel("n qubits")
    ax_mem.set_ylabel("peak memory (KiB)")
    _style_axes(ax_mem)
    if any_mem:
        ax_mem.legend(frameon=False)

    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_convergence_panel(
    records: Sequence[RunRecord],
    *,
    truncation_key: str = "min_abs_coeff",
    reference_value: float | None = None,
    xscale: str = "log",
    ax=None,
    save_path: str | Path | None = None,
):
    """One curve per engine: truncation strength (x) vs. `expectation_value` (y).

    Per the adapted plan's global rule 4 ("every real-time-dynamics or
    truncated result ships with a convergence panel"). `reference_value`, when
    given, is drawn as a horizontal dashed line (e.g. an oracle or a
    self-converged reference) so convergence is visible rather than assumed.
    Records with `expectation_value is None` or a missing `truncation_key` are
    skipped.
    """
    import matplotlib.pyplot as plt

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    grouped = _group_by_engine(records)
    for engine, recs in grouped.items():
        points = []
        for r in recs:
            x = r.truncation.get(truncation_key)
            if x is None or r.expectation_value is None:
                continue
            points.append((x, r.expectation_value))
        points.sort()
        if not points:
            continue
        xs, ys = zip(*points)
        color = _color_for_engine(engine)
        ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.5, color=color, label=engine)

    if reference_value is not None:
        ax.axhline(reference_value, color=_MUTED_TEXT, linewidth=1.2, linestyle="--", label="reference")

    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(truncation_key)
    ax.set_ylabel("expectation value")
    _style_axes(ax)
    if grouped or reference_value is not None:
        ax.legend(frameon=False)

    return _save_or_return(fig, save_path)
