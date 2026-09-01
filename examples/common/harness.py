"""The one runner for every Part-A benchmark and every timed Part-B showcase.

Handoff item P0d; see `research/plans/2026-08-31-examples-benchmarks-suite.md`
§6 Part 0.4 for the adapted specification and
`research/notes/2026-09-01-python-api-extensions.md` A2 (`propagate_with_stats`)
and A7 (truncation aliases + thread pinning) for the API this builds on.

What lives here
---------------

- `make_policy` / `TruncationSpec` — the jl-compatible `(max_weight,
  min_abs_coeff)` knobs, the only truncation vocabulary comparative runs may
  use. `topn` is deliberately not offered (plan §2, D3: PauliPropagation.jl has
  no equivalent, and `topn` on the right of an `|` composition is silently
  inert).
- `assert_single_threaded` — the A7 thread-pin gate.
- `run_propagation` — one run in, one `report.RunRecord` out: separately timed
  propagation and contraction, term counts from `PropagationStats`, peak RSS,
  absolute error against a caller-supplied oracle.
- `diff_pauli_sums` / `check_term_parity` / `require_parity` — the **blocking**
  parity gate. Plan §7 global rule 2: no cross-engine timing may be reported
  for a run whose evolved sums diverge term-for-term at matched truncation.
  Call `require_parity` *before* any comparative number is printed or written.
- `convergence_sweep` and `time_to_accuracy` — deterministic sweeps over a
  caller-supplied truncation grid, feeding `report.plot_convergence_panel` and
  `report.plot_error_vs_runtime` respectively.

Import path
-----------

`examples/` is not an installed package; import this module as part of the
`common` package after putting the repo's `examples/` directory on `sys.path`::

    sys.path.insert(0, str(repo_root / "examples"))
    from common import harness

Two process-global hazards, both by design of the underlying library
--------------------------------------------------------------------

**Threads.** `RAYON_NUM_THREADS=1` must be exported **before the interpreter
starts**::

    RAYON_NUM_THREADS=1 python benchmarks/python/bench_whatever.py

`run_propagation(..., threads=1)` asserts the pin (`assert_single_threaded`);
`threads=None` (the default) skips the assert and records the thread count the
run actually executed under, so a record is always self-describing. Every
comparative Part-A run must pass `threads=1` explicitly; multi-thread numbers
belong in a separate, labeled scaling report (CLAUDE.md §Performance
discipline, plan §7 rule 3).

Two corrections to A7's measured thread facts, re-measured on ccqlin038
(2026-08-31, this build) — A7 says the Rayon pool spawns at
`import paulistrings` and that a pinned process shows exactly 2 threads:

1. **The 31-or-so threads visible right after `import paulistrings` are
   numpy's, not Rayon's.** `paulistrings` imports numpy transitively (via
   `interop`), and numpy's OpenBLAS spawns one thread per core at import: a
   bare `import numpy` alone takes the process from 1 thread to 32 on a
   32-core host, and `import paulistrings` afterwards adds none. An absolute
   "at most 2 threads" bound is therefore unreachable for any process in this
   suite, which is why `assert_single_threaded` checks the count **relative to
   `IMPORT_THREAD_COUNT`** instead.
2. **Rayon's pool is built at the first `propagate`, not at import.** Unset,
   the first propagate adds one worker per core (32 → 64 threads); with
   `RAYON_NUM_THREADS=1` it adds exactly one (32 → 33). Setting
   `os.environ["RAYON_NUM_THREADS"] = "1"` after `import paulistrings` but
   before the first propagate was measured to work. Do not rely on it: any
   import, oracle, or helper may propagate first, and the pool is built once
   and never resized. Export the variable before the interpreter starts, which
   is what this module requires.

**Logging.** An enabled DEBUG filter on the engine's progress logger adds a
clock read per layer (CLAUDE.md §Performance discipline: "Run campaigns with
`RUST_LOG` unset"). `run_propagation` therefore refuses to time anything while
`paulistrings.propagate` is DEBUG-enabled, and never installs a handler itself.
Pass `require_quiet_logging=False` to time a run anyway — its numbers are then
diagnostic, not comparison-grade. Remember `paulistrings.reset_log_cache()`
after changing levels mid-process (`pyo3-log` caches each logger's level).

Memory accounting
-----------------

`VmHWM` from `/proc/self/status` is the **process-lifetime** high-water mark,
not a per-run figure: it never decreases, so a later run in the same process
inherits an earlier one's peak. Every record therefore carries both readings —
`peak_memory_kb` (after the run) plus `extra["baseline_peak_memory_kb"]` and
`extra["peak_memory_kb_delta"]` — and a size sweep that wants clean per-point
memory should run one point per process (or read the delta, which is a lower
bound on this run's own footprint).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Any

from paulistrings import truncation as _ps_truncation
from paulistrings._paulistrings import Truncation as _Truncation

from .report import Provenance, RunRecord, collect_provenance

__all__ = [
    "DIRECTIONS",
    "IMPORT_THREAD_COUNT",
    "PINNED_RAYON_THREADS",
    "AccuracyResult",
    "ParityError",
    "ParityResult",
    "SumDiff",
    "TruncationSpec",
    "assert_logging_quiet",
    "assert_single_threaded",
    "check_term_parity",
    "convergence_sweep",
    "current_memory_kb",
    "diff_pauli_sums",
    "logging_is_quiet",
    "make_policy",
    "observed_thread_count",
    "peak_memory_kb",
    "rayon_worker_estimate",
    "require_parity",
    "run_propagation",
    "time_to_accuracy",
]

#: The two propagation directions. Never defaulted anywhere in this suite
#: (plan §8, D9: the library's default is `"forward"` and the README's
#: "Heisenberg by default" claim was stale), so `run_propagation` takes
#: `direction` as a required argument.
DIRECTIONS = ("forward", "heisenberg")

#: Threads a *pinned* Rayon pool adds to the process: exactly one worker
#: (measured on ccqlin038, 2026-08-31 — with `RAYON_NUM_THREADS=1` the first
#: propagate takes the count from 32 to 33; unset it goes to 64 on a 32-core
#: host). See the module docstring for why the check is relative.
PINNED_RAYON_THREADS = 1

_STATUS_PATH = Path("/proc/self/status")
_PROPAGATE_LOGGER = "paulistrings.propagate"
_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Truncation knobs (A7)
# --------------------------------------------------------------------------


def make_policy(
    max_weight: int | None = None,
    min_abs_coeff: float | None = None,
):
    """Build the `paulistrings` truncation policy for the two shared knobs.

    Returns `truncation.weight(max_weight) & truncation.coeff(min_abs_coeff)`
    when both are given, the single policy when one is, and `None` — meaning
    "no per-term filtering" — when neither is. These two knobs are exactly the
    ones PauliPropagation.jl also has, which is why they are the only
    truncation vocabulary a comparative run may use (plan §5).

    **Boundary semantics, the parity-relevant detail.** `coeff(eps)` drops a
    term when ``abs(c) <= eps`` — the comparison is *inclusive*, so a
    coefficient exactly equal to the cutoff is discarded, not kept
    (`crates/paulistrings/src/truncation/builtin.rs:22`). Whether the reference
    implementation's boundary is inclusive or strict must be probed empirically
    with boundary-straddling fixtures before any cross-engine timing is
    recorded; dyadic cutoffs (2⁻¹⁴, 2⁻¹⁶, …) make an exact straddle plausible.
    Any divergence is reported as a finding, never fudged (plan §5).

    `weight(k)` keeps terms of Pauli weight ``<= k``. Composition is `&`
    (both must accept), evaluated after every channel — this engine truncates
    per channel, so suite circuits are built one gate per channel to make that
    schedule match a per-gate-truncating reference (plan §5, D10).

    `topn` is deliberately absent: there is no reference-implementation
    equivalent, and it is silently inert on the right of an `|` composition
    (plan §2, D3). Showcases that want it should call
    `paulistrings.truncation.topn` directly and say so in their narrative.
    """
    parts = []
    if max_weight is not None:
        if isinstance(max_weight, bool) or not isinstance(max_weight, int):
            raise ValueError(f"max_weight must be an int or None, got {max_weight!r}")
        if max_weight < 0:
            raise ValueError(f"max_weight must be non-negative, got {max_weight}")
        parts.append(_ps_truncation.weight(max_weight))
    if min_abs_coeff is not None:
        coeff = float(min_abs_coeff)
        if coeff < 0.0:
            raise ValueError(f"min_abs_coeff must be non-negative, got {coeff}")
        parts.append(_ps_truncation.coeff(coeff))

    if not parts:
        return None
    policy = parts[0]
    for extra_part in parts[1:]:
        policy = policy & extra_part
    return policy


@dataclass(frozen=True)
class TruncationSpec:
    """One point of a truncation grid: the knobs, not the policy object.

    Carrying the knobs (rather than an opaque `Truncation`) is what lets a
    `RunRecord` record *which* truncation produced it, which in turn is what
    `report.plot_term_count_vs_truncation` and `report.plot_convergence_panel`
    key their x axis on (`record.truncation["min_abs_coeff"]`).
    """

    max_weight: int | None = None
    min_abs_coeff: float | None = None

    def policy(self):
        """The `paulistrings` policy object (`None` when both knobs are unset)."""
        return make_policy(self.max_weight, self.min_abs_coeff)

    def as_dict(self) -> dict[str, Any]:
        """`RunRecord.truncation` form: unset knobs are *omitted*, not `None`.

        Omission is what the plot helpers test for when they skip a record, so
        an unset knob must not appear as a `None` value.
        """
        out: dict[str, Any] = {}
        if self.max_weight is not None:
            out["max_weight"] = self.max_weight
        if self.min_abs_coeff is not None:
            out["min_abs_coeff"] = self.min_abs_coeff
        return out

    def __str__(self) -> str:
        if not self.as_dict():
            return "no truncation"
        return ", ".join(f"{k}={v!r}" for k, v in self.as_dict().items())

    @classmethod
    def coerce(cls, value: Any) -> TruncationSpec:
        """Accept `None`, a `TruncationSpec`, a `(max_weight, min_abs_coeff)`
        pair, or a mapping of those two keys. Anything else — including a
        ready-made `paulistrings` `Truncation`, which has no readable knobs —
        is an error, so a grid entry can never silently lose its labels.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            unknown = set(value) - {"max_weight", "min_abs_coeff"}
            if unknown:
                raise ValueError(
                    f"unknown truncation keys {sorted(unknown)}; only 'max_weight' and "
                    "'min_abs_coeff' are allowed in a comparative grid (see make_policy)"
                )
            return cls(
                max_weight=value.get("max_weight"),
                min_abs_coeff=value.get("min_abs_coeff"),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError(
                    "a truncation grid entry given as a sequence must be "
                    f"(max_weight, min_abs_coeff); got {len(value)} items: {value!r}"
                )
            return cls(max_weight=value[0], min_abs_coeff=value[1])
        raise TypeError(
            f"cannot read truncation knobs from {value!r}; pass None, a TruncationSpec, "
            "a (max_weight, min_abs_coeff) pair, or a mapping of those keys"
        )


def _policy_and_labels(policy: Any) -> tuple[Any, dict[str, Any]]:
    """Split a `run_propagation` `policy` argument into (policy object, labels).

    A ready-made `paulistrings.Truncation` is passed through untouched, with
    its `repr` as the only label — such a record cannot feed the
    truncation-keyed plots, which is why the knob forms are preferred.
    """
    if isinstance(policy, _Truncation):
        return policy, {"policy": repr(policy)}
    spec = TruncationSpec.coerce(policy)
    return spec.policy(), spec.as_dict()


# --------------------------------------------------------------------------
# Process introspection: threads, memory, logging (A7)
# --------------------------------------------------------------------------


def _status_field(name: str) -> str | None:
    """The value of one `/proc/self/status` field, or `None` if unreadable."""
    try:
        with _STATUS_PATH.open("r") as f:
            prefix = f"{name}:"
            for line in f:
                if line.startswith(prefix):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def observed_thread_count() -> int | None:
    """This process's `Threads:` count, or `None` outside Linux/procfs."""
    raw = _status_field("Threads")
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except (ValueError, IndexError):
        return None


#: The process's thread count when this module was imported. Importing
#: `paulistrings` pulls numpy in, so this already includes OpenBLAS's
#: one-thread-per-core pool; Rayon's pool is *not* in it, because that is built
#: lazily at the first propagate. `assert_single_threaded` and
#: `rayon_worker_estimate` both work against this baseline.
IMPORT_THREAD_COUNT = observed_thread_count()


def _status_kb(name: str) -> float | None:
    raw = _status_field(name)
    if raw is None:
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


def peak_memory_kb() -> float | None:
    """`VmHWM` in KiB — peak RSS over the **whole process lifetime**.

    The same source `crates/paulistrings/examples/phase_breakdown.rs` samples.
    It is monotone: it never falls when memory is released, so it is a per-run
    figure only in a process that has done exactly one run. See the module
    docstring's memory-accounting note.
    """
    return _status_kb("VmHWM")


def current_memory_kb() -> float | None:
    """`VmRSS` in KiB — resident set size right now."""
    return _status_kb("VmRSS")


def rayon_worker_estimate() -> int | None:
    """Threads this process gained since `harness` was imported.

    Once a propagate has run, that gain *is* Rayon's pool: nothing else in this
    suite spawns threads after import (numpy's OpenBLAS pool is already in
    `IMPORT_THREAD_COUNT`). `None` when procfs is unavailable; `0` before the
    first propagate, since the pool is built lazily.
    """
    threads = observed_thread_count()
    if threads is None or IMPORT_THREAD_COUNT is None:
        return None
    return max(0, threads - IMPORT_THREAD_COUNT)


def assert_single_threaded() -> None:
    """Raise `RuntimeError` unless this process is pinned to one Rayon worker.

    Two independent conditions, both required:

    1. `RAYON_NUM_THREADS` is exactly `"1"`. This is the only actual control —
       Rayon reads it when it builds the global pool, once, and never resizes.
    2. The process has gained at most `PINNED_RAYON_THREADS` threads since this
       module was imported. Checked *relative* to `IMPORT_THREAD_COUNT` because
       numpy's OpenBLAS pool already accounts for one thread per core (see the
       module docstring's thread note); an absolute bound would be unreachable.

    Condition 1 alone would pass a pool that was built before the variable was
    set; condition 2 alone would pass a process whose pool has not spawned yet.
    """
    env = os.environ.get("RAYON_NUM_THREADS")
    threads = observed_thread_count()
    gained = rayon_worker_estimate()

    problems = []
    if env != "1":
        problems.append(f"RAYON_NUM_THREADS is {env!r}, not '1'")
    if gained is not None and gained > PINNED_RAYON_THREADS:
        problems.append(
            f"the process has gained {gained} threads since importing this module "
            f"({threads} now vs {IMPORT_THREAD_COUNT} at import), but a pinned Rayon "
            f"pool adds at most {PINNED_RAYON_THREADS}"
        )
    if not problems:
        return

    raise RuntimeError(
        "this run asked for threads=1 but is not single-threaded: "
        + "; ".join(problems)
        + ".\nRayon builds its global pool once, at the first propagate, and never "
        "resizes it, so the variable must be in the environment before anything in "
        "the process propagates. Export it before the interpreter starts:\n"
        "    RAYON_NUM_THREADS=1 python <script>\n"
        "(measured thread behaviour: this module's docstring, correcting "
        "research/notes/2026-09-01-python-api-extensions.md §A7)"
    )


def logging_is_quiet() -> bool:
    """`True` when the engine's per-layer DEBUG records are switched off."""
    return not logging.getLogger(_PROPAGATE_LOGGER).isEnabledFor(logging.DEBUG)


def assert_logging_quiet() -> None:
    """Raise `RuntimeError` if `paulistrings.propagate` is DEBUG-enabled.

    An enabled DEBUG filter costs a clock read per layer, so a timed run under
    it is not comparable with one without (CLAUDE.md §Performance discipline:
    campaigns run with `RUST_LOG` unset, where the per-layer logging is one
    static level check that allocates nothing).
    """
    if logging_is_quiet():
        return
    logger = logging.getLogger(_PROPAGATE_LOGGER)
    raise RuntimeError(
        f"logger {_PROPAGATE_LOGGER!r} is DEBUG-enabled (effective level "
        f"{logging.getLevelName(logger.getEffectiveLevel())}); a timed run must not be. "
        "The engine's per-layer progress records cost a clock read per layer when the "
        "filter is on (CLAUDE.md §Performance discipline). Raise the level (and call "
        "paulistrings.reset_log_cache(), since pyo3-log caches it) before timing, or "
        "pass require_quiet_logging=False to accept diagnostic-only timings."
    )


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


@cache
def _cached_provenance(
    thread_count: int | None,
    seeds: tuple[tuple[str, int], ...],
    library_versions: tuple[tuple[str, str], ...],
) -> Provenance:
    """`collect_provenance`, memoized: a 20-point sweep must not shell out to
    `git rev-parse` and `rustc -V` twenty times.
    """
    return collect_provenance(
        seeds=dict(seeds) or None,
        thread_count=thread_count,
        extra_library_versions=dict(library_versions) or None,
        repo_root=_REPO_ROOT,
    )


def _provenance(
    thread_count: int | None,
    seeds: Mapping[str, int] | None,
    library_versions: Mapping[str, str] | None,
) -> Provenance:
    cached = _cached_provenance(
        thread_count,
        tuple(sorted((seeds or {}).items())),
        tuple(sorted((library_versions or {}).items())),
    )
    # Hand out a copy so a caller mutating one record's provenance cannot
    # reach into every other record built from the same cache entry.
    return replace(
        cached,
        library_versions=dict(cached.library_versions),
        seeds=dict(cached.seeds),
    )


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


def run_propagation(
    circuit,
    observable,
    policy: Any,
    direction: str,
    *,
    state: str | None = None,
    contract: Callable[[Any], complex] | None = None,
    warmup: bool = True,
    oracle_value: float | None = None,
    engine: str = "paulistrings",
    engine_version: str | None = None,
    threads: int | None = None,
    seeds: Mapping[str, int] | None = None,
    library_versions: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
    require_quiet_logging: bool = True,
) -> RunRecord:
    """Propagate `observable` through `circuit` and return one `RunRecord`.

    Arguments
    ---------
    `circuit`, `observable`
        A `paulistrings.Circuit` and the `PauliSum` to evolve. `propagate`
        takes `&self` and clones internally, so `observable` is *not* consumed
        or mutated: the warmup and the timed run start from byte-identical
        input, and the caller may reuse the same observable across a sweep.
    `policy`
        A `TruncationSpec`, a `(max_weight, min_abs_coeff)` pair, a mapping of
        those keys, `None`, or a ready-made `paulistrings` `Truncation`. The
        knob forms are strongly preferred: only they can label
        `RunRecord.truncation` in the form the truncation-keyed plot helpers
        read.
    `direction`
        `"forward"` or `"heisenberg"`. Required — never defaulted anywhere in
        this suite (plan §8, D9).
    `state`
        Product state for the contraction step (`"x+"`, `"y+"`, `"z+"`, or an
        A4 per-qubit label string). `None` skips contraction entirely:
        `contraction_time_s` and `expectation_value` stay `None`.
    `contract`
        Alternative contraction: any callable taking the evolved `PauliSum` and
        returning a number, for observables contracted by `overlap` rather
        than a product state. Mutually exclusive with `state`.
    `warmup`
        Run propagation (and contraction) once untimed before the timed pass,
        so the reported number is a *warm* time — the same discipline
        `benchmarks/PROFILING.md` requires of both engines. Set `False` only
        for a run so long that a doubled cost is prohibitive; say so in the
        narrative when you do.
    `oracle_value`
        Reference value; when given together with a contraction, the record
        carries `absolute_error`. Plan §7 rule 1: every numeric claim comes
        from an oracle or a provenance-tagged reference file.
    `threads`
        `1` asserts the single-thread pin before doing anything else
        (`assert_single_threaded`); any other value is recorded as-is, for a
        labeled scaling report; `None` (default) skips the assert and records
        `rayon_worker_estimate()` instead, so an unpinned run is still
        self-describing rather than silently passing for a pinned one.

    Timing
    ------
    Propagation and contraction are timed separately with `time.perf_counter`,
    around `propagate_with_stats` and the contraction call respectively. The
    stats object costs two `usize` reads per layer on the calling thread (A2),
    so the propagation time is comparable with a plain `propagate`.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if state is not None and contract is not None:
        raise ValueError("pass either state= or contract=, not both")
    if threads == 1:
        assert_single_threaded()
    if require_quiet_logging:
        assert_logging_quiet()

    policy_obj, truncation_labels = _policy_and_labels(policy)
    baseline_peak_kb = peak_memory_kb()

    if warmup:
        warm_evolved, _ = observable.propagate_with_stats(
            circuit, policy_obj, direction=direction
        )
        if state is not None:
            warm_evolved.expectation(state)
        elif contract is not None:
            contract(warm_evolved)
        del warm_evolved

    start = time.perf_counter()
    evolved, stats = observable.propagate_with_stats(
        circuit, policy_obj, direction=direction
    )
    propagation_time_s = time.perf_counter() - start

    expectation_value: float | None = None
    contraction_time_s: float | None = None
    record_extra: dict[str, Any] = dict(extra or {})

    if state is not None or contract is not None:
        start = time.perf_counter()
        raw = evolved.expectation(state) if state is not None else contract(evolved)
        contraction_time_s = time.perf_counter() - start
        value = complex(raw)
        expectation_value = value.real
        # Every observable in this suite is Hermitian, so a non-negligible
        # imaginary part means something is wrong; surface it instead of
        # discarding it silently.
        if abs(value.imag) > 1e-9 * max(1.0, abs(value.real)):
            record_extra["expectation_imag"] = value.imag

    absolute_error: float | None = None
    if oracle_value is not None and expectation_value is not None:
        absolute_error = abs(expectation_value - float(oracle_value))

    run_peak_kb = peak_memory_kb()
    if baseline_peak_kb is not None:
        record_extra.setdefault("baseline_peak_memory_kb", baseline_peak_kb)
        if run_peak_kb is not None:
            record_extra.setdefault(
                "peak_memory_kb_delta", run_peak_kb - baseline_peak_kb
            )
    rss_kb = current_memory_kb()
    if rss_kb is not None:
        record_extra.setdefault("rss_kb_after", rss_kb)
    if state is not None:
        record_extra.setdefault("state", state)
    observed = observed_thread_count()
    if observed is not None:
        record_extra.setdefault("observed_threads", observed)

    # `thread_count` in the provenance block means "how many workers did this
    # run use", so the fallback is the Rayon-pool estimate, not the raw process
    # count (which carries OpenBLAS's whole pool — see the module docstring).
    thread_count = threads if threads is not None else rayon_worker_estimate()
    provenance = _provenance(thread_count, seeds, library_versions)
    if engine_version is None:
        engine_version = provenance.library_versions.get(engine, "unknown")

    return RunRecord(
        engine=engine,
        engine_version=engine_version,
        n_qubits=observable.num_qubits,
        direction=direction,
        truncation=truncation_labels,
        propagation_time_s=propagation_time_s,
        final_terms=stats.final_terms,
        provenance=provenance,
        contraction_time_s=contraction_time_s,
        peak_terms=stats.peak_terms,
        expectation_value=expectation_value,
        absolute_error=absolute_error,
        peak_memory_kb=run_peak_kb,
        extra=record_extra,
    )


# --------------------------------------------------------------------------
# The parity gate (plan §7 rule 2)
# --------------------------------------------------------------------------


class ParityError(RuntimeError):
    """Raised by `require_parity` when two runs are not term-for-term equal.

    This is a *blocking* condition: the plan forbids reporting cross-engine
    timings for runs whose evolved sums diverge at matched truncation.
    """


Key = tuple[tuple[int, ...], tuple[int, ...]]


def _strip_trailing_zero_words(row: Iterable[Any]) -> tuple[int, ...]:
    """Canonicalize one symplectic half-key by dropping trailing zero words.

    Two sums monomorphized at the same `num_qubits` always share a width, so
    this is normally a no-op; doing it anyway makes the comparison independent
    of how many `u64` words the exporter happened to emit.
    """
    words = [int(word) for word in row]
    while words and words[-1] == 0:
        words.pop()
    return tuple(words)


def _term_dict(pauli_sum) -> dict[Key, complex]:
    """`{(x_words, z_words): coefficient}` — an order-independent view.

    Built from `x_array`/`z_array`/`coefficients_array`, so it works for any
    object exposing that numpy export (both engines' Python wrappers do).
    Duplicate keys, which a canonical `PauliSum` never has, are summed rather
    than silently overwritten.
    """
    x_rows = pauli_sum.x_array()
    z_rows = pauli_sum.z_array()
    coefficients = pauli_sum.coefficients_array()
    out: dict[Key, complex] = {}
    for x_row, z_row, coeff in zip(x_rows, z_rows, coefficients):
        key = (_strip_trailing_zero_words(x_row), _strip_trailing_zero_words(z_row))
        out[key] = out.get(key, 0j) + complex(coeff)
    return out


def _format_key(key: Key) -> str:
    x_words, z_words = key
    x_text = ",".join(f"0x{w:x}" for w in x_words) or "0x0"
    z_text = ",".join(f"0x{w:x}" for w in z_words) or "0x0"
    return f"x=[{x_text}] z=[{z_text}]"


@dataclass(frozen=True)
class SumDiff:
    """Term-for-term comparison of two evolved Pauli sums."""

    tol: float
    terms_a: int
    terms_b: int
    matched: int
    only_in_a: list[tuple[Key, complex]] = field(default_factory=list)
    only_in_b: list[tuple[Key, complex]] = field(default_factory=list)
    max_abs_delta: float = 0.0
    max_delta_key: Key | None = None

    @property
    def is_match(self) -> bool:
        return (
            not self.only_in_a
            and not self.only_in_b
            and self.max_abs_delta <= self.tol
        )

    def describe(self, *, max_listed: int = 5) -> str:
        lines = [
            (
                f"{self.terms_a} vs {self.terms_b} terms, {self.matched} shared keys, "
                f"max |Δcoeff| = {self.max_abs_delta:.3e} (tol {self.tol:.3e})"
            )
        ]
        if self.max_delta_key is not None and self.max_abs_delta > self.tol:
            lines.append(f"  worst key: {_format_key(self.max_delta_key)}")
        for label, missing in (("only in A", self.only_in_a), ("only in B", self.only_in_b)):
            if not missing:
                continue
            lines.append(f"  {len(missing)} keys {label} (|coeff| > tol):")
            for key, coeff in missing[:max_listed]:
                lines.append(f"    {_format_key(key)}  coeff={coeff!r}")
            if len(missing) > max_listed:
                lines.append(f"    ... and {len(missing) - max_listed} more")
        return "\n".join(lines)


def diff_pauli_sums(sum_a, sum_b, tol: float = 1e-12) -> SumDiff:
    """Compare two evolved Pauli sums term for term, order-independently.

    Both sums are turned into `{symplectic key: coefficient}` dicts, so
    storage order — which this repo explicitly does not pin (CLAUDE.md
    §Determinism policy) — cannot affect the result.

    A key present in one sum only counts as a divergence **unless** its
    coefficient is within `tol` of zero: a term that survived one engine's
    truncation boundary with a negligible coefficient and lost the other's is
    agreement to floating-point tolerance, which is the correctness bar. Shared
    keys contribute `max |a - b|` as `max_abs_delta`.
    """
    if tol < 0.0:
        raise ValueError(f"tol must be non-negative, got {tol}")
    n_a, n_b = sum_a.num_qubits, sum_b.num_qubits
    if n_a != n_b:
        raise ValueError(f"cannot diff sums over different qubit counts ({n_a} vs {n_b})")

    terms_a = _term_dict(sum_a)
    terms_b = _term_dict(sum_b)

    max_abs_delta = 0.0
    max_delta_key: Key | None = None
    matched = 0
    for key, coeff_a in terms_a.items():
        if key not in terms_b:
            continue
        matched += 1
        delta = abs(coeff_a - terms_b[key])
        if delta > max_abs_delta:
            max_abs_delta = delta
            max_delta_key = key

    only_in_a = sorted(
        ((key, coeff) for key, coeff in terms_a.items() if key not in terms_b and abs(coeff) > tol),
        key=lambda item: -abs(item[1]),
    )
    only_in_b = sorted(
        ((key, coeff) for key, coeff in terms_b.items() if key not in terms_a and abs(coeff) > tol),
        key=lambda item: -abs(item[1]),
    )

    return SumDiff(
        tol=tol,
        terms_a=len(terms_a),
        terms_b=len(terms_b),
        matched=matched,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        max_abs_delta=max_abs_delta,
        max_delta_key=max_delta_key,
    )


@dataclass(frozen=True)
class ParityResult:
    """Outcome of `check_term_parity`: `ok`, plus every reason it is not."""

    ok: bool
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.ok:
            return "parity holds"
        return "parity FAILED:\n" + "\n".join(f"  - {reason}" for reason in self.reasons)


def check_term_parity(
    record_a: RunRecord, record_b: RunRecord, coeff_tol: float = 1e-12
) -> ParityResult:
    """Compare two `RunRecord`s: matched setup, equal term counts, equal value.

    The record-level half of the parity gate (the sum-level half is
    `diff_pauli_sums`). Checks, in order: same `n_qubits`, same `direction`,
    **same truncation labels** — parity is only meaningful at matched
    truncation, so differing knobs are themselves a failure — then
    `final_terms`, `peak_terms` (when both records have it) and, when both
    records carry an expectation value, agreement within `coeff_tol`.

    Term counts must be *equal*, not close: at matched truncation two engines
    that agree on the semantics keep the same set of terms. A near-miss (say
    one term out of a million) is exactly the boundary-straddling cutoff
    divergence plan §5 requires be investigated and reported, not tolerated.
    """
    reasons: list[str] = []
    if record_a.n_qubits != record_b.n_qubits:
        reasons.append(f"n_qubits differ: {record_a.n_qubits} vs {record_b.n_qubits}")
    if record_a.direction != record_b.direction:
        reasons.append(f"direction differs: {record_a.direction!r} vs {record_b.direction!r}")
    if record_a.truncation != record_b.truncation:
        reasons.append(
            f"truncation differs: {record_a.truncation} vs {record_b.truncation} "
            "(parity is only defined at matched truncation)"
        )
    if record_a.final_terms != record_b.final_terms:
        reasons.append(
            f"final_terms differ: {record_a.final_terms} ({record_a.engine}) vs "
            f"{record_b.final_terms} ({record_b.engine})"
        )
    if (
        record_a.peak_terms is not None
        and record_b.peak_terms is not None
        and record_a.peak_terms != record_b.peak_terms
    ):
        reasons.append(
            f"peak_terms differ: {record_a.peak_terms} ({record_a.engine}) vs "
            f"{record_b.peak_terms} ({record_b.engine})"
        )
    if record_a.expectation_value is not None and record_b.expectation_value is not None:
        delta = abs(record_a.expectation_value - record_b.expectation_value)
        if delta > coeff_tol:
            reasons.append(
                f"expectation values differ by {delta:.3e} > {coeff_tol:.3e}: "
                f"{record_a.expectation_value!r} ({record_a.engine}) vs "
                f"{record_b.expectation_value!r} ({record_b.engine})"
            )
    return ParityResult(ok=not reasons, reasons=reasons)


def require_parity(a, b, *, coeff_tol: float = 1e-12, label: str | None = None):
    """Blocking parity gate: raise `ParityError` with a diagnostic dump.

    Accepts either two `RunRecord`s (dispatches to `check_term_parity`) or two
    Pauli sums (dispatches to `diff_pauli_sums`). Returns the `ParityResult` /
    `SumDiff` on success, so a caller can log the matched counts.

    Call this **before** printing or writing any cross-engine number — plan §7
    rule 2 makes a parity failure block the timing report, not annotate it.
    """
    prefix = f"[{label}] " if label else ""
    if isinstance(a, RunRecord) and isinstance(b, RunRecord):
        result = check_term_parity(a, b, coeff_tol=coeff_tol)
        if not result.ok:
            raise ParityError(prefix + result.describe())
        return result
    if hasattr(a, "x_array") and hasattr(b, "x_array"):
        diff = diff_pauli_sums(a, b, tol=coeff_tol)
        if not diff.is_match:
            raise ParityError(prefix + "evolved sums diverge:\n" + diff.describe())
        return diff
    raise TypeError(
        "require_parity takes two RunRecords or two Pauli sums, got "
        f"{type(a).__name__} and {type(b).__name__}"
    )


# --------------------------------------------------------------------------
# Truncation sweeps
# --------------------------------------------------------------------------

#: A caller-supplied factory: one `TruncationSpec` in, one finished
#: `RunRecord` out. Typically a closure over a fixed circuit/observable, e.g.
#: `lambda spec: run_propagation(circuit, obs, spec, "heisenberg", state="z+",
#: threads=1, oracle_value=ref)`.
BuildRun = Callable[[TruncationSpec], RunRecord]


def _run_grid(
    build_run: BuildRun,
    truncation_grid: Sequence[Any],
    oracle_value: float | None,
    stop_when: Callable[[RunRecord], bool] | None = None,
) -> tuple[list[TruncationSpec], list[RunRecord]]:
    specs = [TruncationSpec.coerce(entry) for entry in truncation_grid]
    if not specs:
        raise ValueError("truncation_grid is empty")

    records: list[RunRecord] = []
    for spec in specs:
        record = build_run(spec)
        if not isinstance(record, RunRecord):
            raise TypeError(
                f"build_run must return a report.RunRecord, got {type(record).__name__}"
            )
        expected = spec.as_dict()
        if expected and any(record.truncation.get(k) != v for k, v in expected.items()):
            raise ValueError(
                f"build_run ignored its spec: asked for {expected}, record says "
                f"{record.truncation}. The sweep's x axis comes from "
                "RunRecord.truncation, so build_run must pass the spec it is given "
                "through to run_propagation."
            )
        if oracle_value is not None and record.expectation_value is not None:
            record.absolute_error = abs(record.expectation_value - float(oracle_value))
        records.append(record)
        if stop_when is not None and stop_when(record):
            break
    return specs[: len(records)], records


def convergence_sweep(
    build_run: BuildRun,
    truncation_grid: Sequence[Any],
    *,
    oracle_value: float | None = None,
) -> list[RunRecord]:
    """Run one `RunRecord` per truncation-grid point, in the caller's order.

    This is the collector behind plan §7 rule 4 ("every real-time-dynamics or
    truncated result ships with a convergence panel"): feed the returned list
    straight to `report.plot_convergence_panel`, which reads
    `record.truncation[truncation_key]` for x and `record.expectation_value`
    for y — so `build_run` must contract (pass `state=` or `contract=` to
    `run_propagation`) or the panel will be empty.

    `oracle_value`, when given, (re)computes `absolute_error` on every record
    from that single reference, which is what makes a mixed sweep's
    error-vs-runtime curve comparable point to point.

    Nothing adaptive happens here: the grid is the caller's, evaluated in
    order, every point run.
    """
    _, records = _run_grid(build_run, truncation_grid, oracle_value)
    return records


@dataclass(frozen=True)
class AccuracyResult:
    """The outcome of `time_to_accuracy`: the whole sweep plus two selections."""

    epsilon: float
    oracle_value: float
    specs: list[TruncationSpec]
    records: list[RunRecord]
    first_index: int | None
    cheapest_index: int | None

    @property
    def achieved(self) -> bool:
        """Did any grid point reach `|error| < epsilon`?"""
        return self.first_index is not None

    @property
    def first(self) -> RunRecord | None:
        """The earliest grid point (caller's order) meeting the accuracy bar."""
        return None if self.first_index is None else self.records[self.first_index]

    @property
    def cheapest(self) -> RunRecord | None:
        """The fastest (`total_time_s`) grid point meeting the accuracy bar."""
        return None if self.cheapest_index is None else self.records[self.cheapest_index]

    @property
    def first_spec(self) -> TruncationSpec | None:
        return None if self.first_index is None else self.specs[self.first_index]

    @property
    def cheapest_spec(self) -> TruncationSpec | None:
        return None if self.cheapest_index is None else self.specs[self.cheapest_index]

    def describe(self) -> str:
        lines = [
            (
                f"time to |error| < {self.epsilon:.3e} vs oracle {self.oracle_value!r} "
                f"({len(self.records)} grid points run)"
            )
        ]
        for index, (spec, record) in enumerate(zip(self.specs, self.records)):
            marks = "".join(
                (
                    "*" if index == self.first_index else "",
                    "$" if index == self.cheapest_index else "",
                )
            )
            error = "n/a" if record.absolute_error is None else f"{record.absolute_error:.3e}"
            lines.append(
                f"  [{index}]{marks:<2} {spec}: error={error} "
                f"time={record.total_time_s:.4g}s terms={record.final_terms}"
            )
        if not self.achieved:
            lines.append("  no grid point met the bar (* = first pass, $ = cheapest pass)")
        else:
            lines.append("  * = first pass in grid order, $ = cheapest pass by wall time")
        return "\n".join(lines)


def time_to_accuracy(
    build_run: BuildRun,
    oracle_value: float,
    epsilon: float,
    truncation_grid: Sequence[Any],
    *,
    stop_early: bool = False,
) -> AccuracyResult:
    """Sweep a truncation grid and report the cheapest run that hits `epsilon`.

    "Time to fixed accuracy" is the comparison the suite actually cares about:
    not how fast one truncation runs, but how long an engine needs to reach a
    stated error bar. `build_run(spec) -> RunRecord` does one run (see
    `BuildRun`); the whole grid is evaluated, in the caller's order, so the
    result also carries the full error-vs-runtime curve for
    `report.plot_error_vs_runtime`.

    Selection is deterministic and hand-checkable — no adaptive refinement, no
    interpolation, no re-running:

    - `first` / `first_index`: the earliest grid point with
      `absolute_error < epsilon`. Order the grid loosest-to-tightest and this
      is "the cheapest truncation that suffices"; the ordering contract is the
      caller's, not this driver's.
    - `cheapest` / `cheapest_index`: among *all* passing points, the smallest
      `total_time_s` (ties broken by the earlier index). Wall time carries
      benchmark noise (±5–8% single-threaded on the reference host, CLAUDE.md),
      so treat a near-tie between two passing configurations as a tie.

    The bar is strict (`<`, not `<=`) so `epsilon` reads as "error below this".
    A record without an `expectation_value` cannot be scored and is an error,
    not a silent skip. `stop_early=True` stops at the first passing point,
    leaving the curve partial — for grids where the tight end is expensive.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    oracle_value = float(oracle_value)

    def _passes(record: RunRecord) -> bool:
        return record.absolute_error is not None and record.absolute_error < epsilon

    specs, records = _run_grid(
        build_run,
        truncation_grid,
        oracle_value,
        stop_when=_passes if stop_early else None,
    )

    unscored = [
        index for index, record in enumerate(records) if record.absolute_error is None
    ]
    if unscored:
        raise ValueError(
            f"grid points {unscored} produced no expectation value, so their error "
            "against the oracle is undefined; build_run must contract (pass state= or "
            "contract= to run_propagation)"
        )

    passing = [index for index, record in enumerate(records) if _passes(record)]
    first_index = passing[0] if passing else None
    cheapest_index = (
        min(passing, key=lambda index: (records[index].total_time_s, index))
        if passing
        else None
    )

    return AccuracyResult(
        epsilon=epsilon,
        oracle_value=oracle_value,
        specs=specs,
        records=records,
        first_index=first_index,
        cheapest_index=cheapest_index,
    )
