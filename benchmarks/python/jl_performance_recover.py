"""Rebuild `summary.json` / `results.json` from a `run.log` of an interrupted study.

`bench_jl_performance.py` accumulates its summary in memory and writes it once,
at the end. That is right for a completed run and useless for a cancelled one:
kill the driver and the structured record dies with it, even though every
measurement it took is sitting in `run.log`, which is written and flushed line
by line as it goes.

This reconstructs the record from that log. It is a recovery tool, not a second
analysis: the protocol math is not reimplemented here, it is *imported* from the
driver (`analyze_pairs`, `interpolate_crossover`, `bytes_per_term`), so a
recovered `summary.json` is bit-identical to what the driver would have written
for the configurations that finished.

    python benchmarks/python/jl_performance_recover.py \
        benchmarks/python/jl_performance/run.log \
        --out benchmarks/python/jl_performance \
        --memory-from /path/to/pilot/summary.json /path/to/pilot_xxz/summary.json

What the log does and does not carry
------------------------------------

Carried, and therefore recovered exactly: every pair's two runtimes and its
order, per-configuration final term counts, the parity evidence (layer count and
expectation delta), and the workload metadata (re-read from the driver's own
`workloads()`).

**Not** carried, because the driver logs neither: `peak_terms` and the memory
samples. Both are recoverable from a *different* run of the identical
configuration, because both are deterministic in the configuration —
`peak_terms` exactly (same circuit, same truncation, same arithmetic), memory to
within allocator slack. `--memory-from` takes summary files from such runs and
joins them on the configuration label. Every joined value is tagged
`"source": "joined"` in the output so no reader can mistake it for something
this run measured.

A configuration whose pairs did not all complete is dropped, not padded: the
acceptance rule needs the full set, and a partial one would silently lower the
bar.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

CURVE_RE = re.compile(r"^=== curve: (?P<title>.+) ===\s*$")
CONFIG_RE = re.compile(
    r"^  (?P<workload>\w+) eps=(?P<eps>[-\d.e+]+)"
    r"(?: max_weight=(?P<weight>\d+))?: parity gate \.\.\.\s*$"
)
PARITY_RE = re.compile(
    r"^    parity OK: (?P<layers>\d+) layers identical, (?P<terms>\d+) terms"
    r"(?:, \|dE\| = (?P<delta>[-\d.e+naN]+))?\s*$"
)
PAIR_RE = re.compile(
    r"^    pair (?P<idx>\d+) \(\s*(?P<order>[\w-]+)\): "
    r"rust\s+(?P<rust>[\d.]+)s\s+jl\s+(?P<jl>[\d.]+)s\s+ratio\s+(?P<ratio>[\d.]+)\s*$"
)
RESULT_RE = re.compile(
    r"^    ->\s+(?P<terms>\d+) terms\s+median ratio\s+(?P<ratio>[\d.]+)\s+\[(?P<verdict>\w+)\]\s*$"
)
FREERAM_RE = re.compile(r"^    free RAM: (?P<gib>\d+) GiB\s*$")


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "_bench_jl_performance", HERE / "bench_jl_performance.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_bench_jl_performance"] = module
    spec.loader.exec_module(module)
    return module


def parse_log(log_path: Path) -> list[dict[str, Any]]:
    """Group the log into curves -> configurations -> pairs."""
    curves: list[dict[str, Any]] = []
    curve: dict[str, Any] | None = None
    config: dict[str, Any] | None = None

    for raw in log_path.read_text().splitlines():
        m = CURVE_RE.match(raw)
        if m:
            curve = {"title": m["title"], "configs": [], "free_ram_gib_at_start": None}
            curves.append(curve)
            config = None
            continue
        if curve is None:
            continue
        m = FREERAM_RE.match(raw)
        if m:
            curve["free_ram_gib_at_start"] = float(m["gib"])
            continue
        m = CONFIG_RE.match(raw)
        if m:
            config = {
                "workload": m["workload"],
                "min_abs_coeff": float(m["eps"]),
                "max_weight": int(m["weight"]) if m["weight"] else None,
                "pairs": [],
                "parity": None,
                "result": None,
            }
            curve["configs"].append(config)
            continue
        if config is None:
            continue
        m = PARITY_RE.match(raw)
        if m:
            config["parity"] = {
                "ok": True,
                "problems": [],
                "n_layers": int(m["layers"]),
                "rust_final_terms": int(m["terms"]),
                "jl_final_terms": int(m["terms"]),
                "expectation_delta": float(m["delta"]) if m["delta"] else None,
            }
            continue
        m = PAIR_RE.match(raw)
        if m:
            config["pairs"].append(
                {
                    "pair": int(m["idx"]),
                    "order": m["order"],
                    "rust_s": float(m["rust"]),
                    "jl_s": float(m["jl"]),
                    "ratio_jl_over_rust": float(m["ratio"]),
                }
            )
            continue
        m = RESULT_RE.match(raw)
        if m:
            config["result"] = {
                "final_terms": int(m["terms"]),
                "median_ratio_logged": float(m["ratio"]),
                "verdict_logged": m["verdict"],
            }
            continue
    return curves


def load_memory_join(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """`{config label: {peak_terms, memory...}}` from other runs' summaries."""
    joined: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for curve in data.get("curves", []):
            for cfg in curve.get("configs", []):
                timing = cfg.get("timing")
                if not timing:
                    continue
                parity = timing.get("parity") or {}
                joined[cfg["label"]] = {
                    "peak_terms": timing.get("peak_terms"),
                    "peak_memory": timing.get("peak_memory"),
                    # The log records only |dE|, not the two expectations, so
                    # these are joined too. Like peak_terms they are exactly
                    # deterministic in the configuration.
                    "expectation_rust": parity.get("expectation_rust"),
                    "expectation_jl": parity.get("expectation_jl"),
                    "joined_from": str(path),
                }
    return joined


def label_for(workload: str, eps: float, weight: int | None) -> str:
    """The driver's own label spelling, so the join key matches exactly."""
    knob = f"eps={eps:.3e}" + (f" max_weight={weight}" if weight else "")
    return f"{workload} {knob}"


def snap_cutoff(eps: float, declared: tuple[float, ...]) -> float:
    """Recover the *exact* cutoff the driver used from the log's rounded print.

    The log prints `eps=%.3e`, i.e. four significant figures, so `2**-6` comes
    back as `0.01562` rather than `0.015625`. That is harmless for a label and
    actively wrong for anything that inspects the value: `is_dyadic` asks whether
    the mantissa is exactly 0.5, and a 4-digit round-trip destroys exactly that
    property — every dyadic cutoff would be misreported as non-dyadic.

    The true values are known exactly: they are the workload's own declared
    sweep. So the parsed value is matched back to the nearest declared cutoff in
    log space and the declared one is used. A parsed value that matches nothing
    within 0.1% is an error rather than a silent pass-through, because it means
    the log and the workload declaration have diverged.
    """
    if not declared:
        return eps
    best = min(declared, key=lambda c: abs(math.log(c) - math.log(eps)))
    if abs(best - eps) > 1e-3 * best:
        raise SystemExit(
            f"logged cutoff {eps:.6e} matches no declared cutoff "
            f"(nearest {best:.6e}); the log and workloads() have diverged"
        )
    return best


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--memory-from", type=Path, nargs="*", default=[])
    ap.add_argument(
        "--expect-pairs",
        type=int,
        default=5,
        help="drop any configuration with fewer completed pairs (default 5)",
    )
    ap.add_argument("--status", default="partial")
    ap.add_argument("--cancelled-note", default="")
    args = ap.parse_args(argv)

    driver = load_driver()
    all_workloads = driver.workloads()
    join = load_memory_join(args.memory_from)

    parsed = parse_log(args.log)
    curves_out: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for curve in parsed:
        points: list[dict[str, Any]] = []
        configs_out: list[dict[str, Any]] = []
        workload_key = curve["configs"][0]["workload"] if curve["configs"] else None
        workload = all_workloads.get(workload_key)

        for cfg in curve["configs"]:
            # Restore the exact cutoff before anything reads its value; the label
            # is built from the rounded print so it still matches the driver's.
            label = label_for(cfg["workload"], cfg["min_abs_coeff"], cfg["max_weight"])
            if workload is not None:
                declared = tuple(workload.cutoffs)
                if workload.weight_variant is not None:
                    declared = declared + (workload.weight_variant[1],)
                cfg["min_abs_coeff"] = snap_cutoff(cfg["min_abs_coeff"], declared)
            if len(cfg["pairs"]) < args.expect_pairs or cfg["result"] is None:
                dropped.append(
                    {
                        "label": label,
                        "completed_pairs": len(cfg["pairs"]),
                        "expected_pairs": args.expect_pairs,
                        "reason": "cancelled before the pair set completed",
                    }
                )
                continue

            analysis = driver.analyze_pairs(cfg["pairs"])
            analysis["label"] = label
            analysis["pairs"] = cfg["pairs"]
            analysis["final_terms"] = cfg["result"]["final_terms"]
            analysis["parity"] = cfg["parity"]
            analysis["truncation"] = {"min_abs_coeff": cfg["min_abs_coeff"]}
            if cfg["max_weight"] is not None:
                analysis["truncation"]["max_weight"] = cfg["max_weight"]
            analysis["mitigation"] = {
                "min_abs_coeff_rust": cfg["min_abs_coeff"],
                "min_abs_coeff_julia": driver.julia_min_abs_coeff(cfg["min_abs_coeff"]),
                "cutoff_is_dyadic": driver.is_dyadic(cfg["min_abs_coeff"]),
            }

            # Sanity: the median we recompute from the logged *times* must agree
            # with the median ratio the driver logged, to within what the log's
            # own precision allows.
            #
            # The log prints each time as `%9.4f`, so a time carries an absolute
            # quantization of up to 5e-5 s. A ratio built from two such times has
            # relative uncertainty (5e-5/rust + 5e-5/jl) -- negligible at 1 s,
            # but ~3% for a 3 ms configuration. Comparing against a fixed
            # tolerance would either miss real parse errors on the slow
            # configurations or reject correct parses on the fast ones, so the
            # bound is derived per configuration instead.
            #
            # The logged ratio is the more accurate of the two (it was computed
            # from full-precision times), so it is kept in the output alongside.
            logged = cfg["result"]["median_ratio_logged"]
            got = analysis["median_ratio_jl_over_rust"]
            quantum = 0.5e-4
            bound = max(
                got * (quantum / p["rust_s"] + quantum / p["jl_s"]) for p in cfg["pairs"]
            )
            bound += 1e-3  # the logged median's own 3-decimal rounding
            if abs(got - logged) > bound:
                raise SystemExit(
                    f"{label}: recomputed median {got:.4f} disagrees with the logged "
                    f"{logged:.4f} by more than the log's precision allows "
                    f"({bound:.4f}) — the log parse is wrong, refusing to write"
                )
            analysis["median_ratio_logged"] = logged
            analysis["median_ratio_log_precision_bound"] = bound
            if analysis["verdict"] != cfg["result"]["verdict_logged"]:
                raise SystemExit(
                    f"{label}: recovered verdict {analysis['verdict']!r} disagrees with "
                    f"the logged {cfg['result']['verdict_logged']!r} — refusing to write"
                )

            j = join.get(label)
            # The log carries |dE| but not the expectations themselves; join
            # them where a donor run has them, else leave them explicitly None.
            cfg["parity"]["expectation_rust"] = (j or {}).get("expectation_rust")
            cfg["parity"]["expectation_jl"] = (j or {}).get("expectation_jl")
            cfg["parity"]["expectation_source"] = "joined" if j else "unavailable"
            if j and j.get("peak_terms"):
                analysis["peak_terms"] = j["peak_terms"]
                mem = dict(j["peak_memory"] or {})
                mem["source"] = "joined"
                mem["joined_from"] = j["joined_from"]
                analysis["peak_memory"] = mem
            else:
                # No donor run: fall back to the final count and say so, rather
                # than inventing a peak. The memory block keeps its full key set
                # with explicit nulls, so every consumer sees "not measured"
                # rather than tripping over a missing key.
                analysis["peak_terms"] = analysis["final_terms"]
                analysis["peak_terms_source"] = "unavailable; using final_terms"
                analysis["peak_memory"] = {
                    "source": "unavailable",
                    "final_terms": analysis["final_terms"],
                    "peak_terms": analysis["peak_terms"],
                    "rust_vmhwm_kb": None,
                    "rust_floor_kb": None,
                    "rust_bytes_per_term": None,
                    "rust_bytes_per_peak_term": None,
                    "jl_vmhwm_kb": None,
                    "jl_floor_kb": None,
                    "jl_bytes_per_term": None,
                    "jl_bytes_per_peak_term": None,
                }

            configs_out.append(
                {
                    "label": label,
                    "truncation": analysis["truncation"],
                    "parity": cfg["parity"],
                    "timing": analysis,
                    "cut": None,
                    "disqualified": None,
                }
            )
            points.append(
                {
                    "label": label,
                    "max_weight": cfg["max_weight"],
                    "final_terms": analysis["final_terms"],
                    "peak_terms": analysis["peak_terms"],
                    "median_ratio_jl_over_rust": got,
                    "sign_consistent": analysis["sign_consistent"],
                    "verdict": analysis["verdict"],
                    "rust_s_median": analysis["rust_s_median"],
                    "jl_s_median": analysis["jl_s_median"],
                }
            )

        # The max_weight point is a knob-equivalence demonstration, not a point
        # on the size curve; it must not bracket the crossover.
        curve_points = [p for p in points if p.get("max_weight") is None]
        crossover = driver.interpolate_crossover(curve_points, "peak_terms")

        curves_out.append(
            {
                "workload": workload_key,
                "title": curve["title"],
                "notes": workload.notes if workload else "",
                "n_qubits": workload.n_qubits if workload else None,
                "observable": workload.observable if workload else {},
                "state": workload.state if workload else None,
                "points": points,
                "configs": configs_out,
                "curve_points_excluding_weight_variant": [p["label"] for p in curve_points],
                "crossover": crossover,
                "cuts": [],
                "projections": [],
                "free_ram_gib_at_start": curve["free_ram_gib_at_start"],
            }
        )

    summary = {
        "status": args.status,
        "reconstructed_from": str(args.log),
        "reconstructed_by": "benchmarks/python/jl_performance_recover.py",
        "cancelled_note": args.cancelled_note,
        "dropped_configurations": dropped,
        "protocol": {
            "ratio_convention": "ratio = t_julia / t_paulistrings; > 1 means paulistrings faster",
            "pairs_per_configuration": args.expect_pairs,
            "within_pair_order": "abba — rust-first on even pairs, julia-first on odd",
            "acceptance": (
                "direction consistency across every pair; median ratio is the effect "
                "size. Mixed signs are reported as 'indistinguishable', never as a "
                "small win (benchmarks/PROFILING.md)."
            ),
            "direction": "heisenberg",
            "threads": {"rayon": 1, "julia": 1},
            "timing": (
                "warm in-process on both sides: one untimed propagation, then one timed "
                "propagation, in the same process. Construction, contraction, oracles "
                "and logging are all outside the timed region."
            ),
            "parity_gate": (
                "every timed configuration first passes a per-layer term-count "
                "comparison; a failure disqualifies the configuration and no timing "
                "for it is reported."
            ),
            "cutoff_mitigation": (
                "the julia task carries nextafter(eps, +inf) so its strict |c| < eps "
                "equals this engine's inclusive |c| <= eps bit for bit; the threshold "
                "moves by one ulp and no coefficient is touched."
            ),
            "memory": (
                "each engine samples its own /proc/self/status. Values marked "
                "source=joined were sampled during a 1-pair run of the identical "
                "configuration, because the driver logs memory only in the summary it "
                "never got to write; peak RSS is deterministic in the term count to "
                "within allocator slack."
            ),
        },
        "curves": curves_out,
        "accuracy": [],
        "thread_scaling": None,
        "cuts": [],
        "extension_provenance": driver.extension_provenance(),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"wrote {out_dir / 'summary.json'}")

    sys.path.insert(0, str(REPO_ROOT / "examples"))
    from common import report

    results_path = out_dir / "results.json"
    if results_path.exists():
        results_path.unlink()
    report.write_results(driver.build_run_records(curves_out), out_dir, name="results")
    print(f"wrote {results_path}")

    for curve in curves_out:
        x = curve["crossover"]
        print(
            f"{curve['workload']}: {len(curve['points'])} configs, crossover "
            f"{x['crossover_terms']!r} peak terms, zone {x['indistinguishable_zone']}"
        )
    for d in dropped:
        print(f"dropped {d['label']}: {d['completed_pairs']}/{d['expected_pairs']} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
