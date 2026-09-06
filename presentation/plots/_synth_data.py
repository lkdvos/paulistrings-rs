"""Synthetic data generator for developing/visually-checking figures F1-F7.

The real bench driver that will produce `presentation/data/*.jsonl` is being
written concurrently on this branch (see the sibling agent's work under
`presentation/bench/`) and hasn't landed numbers yet. This script fabricates
plausible JSONL files with the *exact* schema the figure scripts expect, so
each `figN_*.py` can be written, run, and rasterized for a visual check before
real data exists.

Every fabricated file goes under `<data-dir>/_synth/` -- never into
`<data-dir>/` itself -- so nothing here is mistaken for a collected result.
Pass `--data-dir` to point at a different root and `--out-suffix` to tag the
run (both accepted for parity with the real collection scripts, per the task
brief; this generator only ever writes under `_synth/`).

The physical numbers below are invented to be *qualitatively* right for what
the talk claims: naive is slow and effectively single-threaded, the two old
multithreading attempts (per-thread maps, parallel mergesort) scale poorly,
bucketed is fast and scales well, fine buckets beat coarse buckets and are
superlinear over the coarse baseline at low thread counts (smaller working
sets fit cache), the bucket-size sweep has a U-shaped ns/term-layer curve
with L2/LLC miss rates rising as buckets grow past cache size, and
`target-cpu=native` gives a modest, mostly-consistent-sign speedup. None of
these numbers are measurements -- they exist only so the plotting code has
something to draw and check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

QUBITS = 127
LAYERS = 1355
STEPS = 5
DEFAULT_TARGET_BUCKET_LEN = 512
COARSE_TARGET_BUCKET_LEN = 16384
MIN_BUCKETS = 16
REPS = 5

DEFAULT_EPS = 2**-10  # moderate sum, ~2e5 peak terms -- used by ladder/thread_scaling
LARGE_EPS = 2**-15  # ~4e6 peak terms -- used by thread_scaling_large

THREAD_GRID = [1, 2, 4, 8, 16, 32]

PROVENANCE = {
    "commit": "deadbeef0123456789abcdef0123456789abcdef",
    "rustc": "rustc 1.94.0 (aaaaaaaaaa 2026-08-01)",
    "host": "ccqlin038",
    "date": "2026-09-06",
    "cpu_model": "AMD EPYC 7742 64-Core Processor",
    "governor": "performance",
}

_BUCKETED_PHASE_FIELDS = [
    "rebucket_ns",
    "prepare_ns",
    "coset_loop_ns",
    "finalize_ns",
    "gather_ns",
    "sort_ns",
    "merge_ns",
    "busy_total_ns",
    "cosets",
    "runs",
    "rows_gathered",
    "rows_sorted",
    "rows_id",
    "terms_in",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _reps(rng: np.random.Generator, median: float, rel_spread: float, k: int = REPS) -> list[float]:
    """`k` samples scattered around `median` by relative Gaussian noise."""
    noise = rng.normal(0.0, rel_spread, size=k)
    return [max(1.0, median * (1.0 + float(x))) for x in noise]


def _write(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"# provenance: {json.dumps(PROVENANCE)}\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records):5d} rows -> {path}")


def _efficiency(variant: str, threads: int) -> float:
    """Parallel efficiency (fraction of ideal linear speedup) at `threads`."""
    if threads <= 1:
        return 1.0
    log2t = math.log2(threads)
    if variant in ("bucketed", "bucketed-coarse"):
        base = 0.94 if variant == "bucketed" else 0.88
        return max(0.35, base - 0.006 * log2t)
    if variant == "threadmaps":
        return max(0.10, 0.62 - 0.11 * log2t)
    if variant == "mergesort":
        return max(0.18, 0.72 - 0.09 * log2t)
    raise ValueError(variant)


def _ns_per_term_layer(variant: str) -> float:
    return {
        "naive": 520.0,
        "threadmaps": 480.0,
        "mergesort": 430.0,
        "bucketed": 150.0,
        "bucketed-coarse": 185.0,
    }[variant]


def _ideal_wall_ns(variant: str, n: int, layers: int = LAYERS) -> float:
    return _ns_per_term_layer(variant) * n * layers


def _wall_ns(variant: str, threads: int, n: int, layers: int = LAYERS) -> float:
    return _ideal_wall_ns(variant, n, layers) / (threads * _efficiency(variant, threads))


def _phase_stats(rng: np.random.Generator, wall_ns: float, threads: int, n: int, variant: str = "bucketed") -> dict:
    """Plausible PhaseStats breakdown for a bucketed row, roughly summing to wall_ns."""
    coset_loop_frac = 0.72
    coset_loop_ns = wall_ns * coset_loop_frac
    rebucket_ns = wall_ns * 0.06
    prepare_ns = wall_ns * 0.05
    finalize_ns = wall_ns * 0.04
    remainder = wall_ns - (coset_loop_ns + rebucket_ns + prepare_ns + finalize_ns)
    gather_ns = remainder * 0.35
    sort_ns = remainder * 0.40
    merge_ns = remainder * 0.25
    busy_total_ns = coset_loop_ns * _efficiency(variant, threads) * threads
    cosets = max(threads, 8)
    runs = cosets * int(rng.integers(3, 7))
    # PhaseStats totals accumulate across every layer of the run, not just one
    # layer's worth of terms -- an average n terms in per layer, times LAYERS.
    terms_in_total = int(n * LAYERS * rng.uniform(0.55, 0.7))
    rows_gathered = terms_in_total * 2
    rows_sorted = rows_gathered
    rows_id = int(terms_in_total * rng.uniform(0.55, 0.7))
    return {
        "rebucket_ns": int(rebucket_ns),
        "prepare_ns": int(prepare_ns),
        "coset_loop_ns": int(coset_loop_ns),
        "finalize_ns": int(finalize_ns),
        "gather_ns": int(gather_ns),
        "sort_ns": int(sort_ns),
        "merge_ns": int(merge_ns),
        "busy_total_ns": int(busy_total_ns),
        "cosets": cosets,
        "runs": runs,
        "rows_gathered": rows_gathered,
        "rows_sorted": rows_sorted,
        "rows_id": rows_id,
        "terms_in": terms_in_total,
    }


def _row(
    *,
    layer: str,
    tag: str,
    threads: int,
    eps: float,
    n: int,
    rep: int,
    wall_ns: float,
    target_bucket_len: int | None = None,
    min_buckets: int | None = None,
    terms_out=None,
    layer_wall_ns=None,
    vmrss_kb: int | None = None,
    vmhwm_kb: int | None = None,
    phase: dict | None = None,
) -> dict:
    final_terms = int(n * 0.4)
    row = {
        "layer": layer,
        "tag": tag,
        "threads": threads,
        "target_bucket_len": target_bucket_len,
        "min_buckets": min_buckets,
        "eps": eps,
        "steps": STEPS,
        "qubits": QUBITS,
        "layers": LAYERS,
        "n": n,
        "rep": rep,
        "wall_ns": int(wall_ns),
        "region_s": wall_ns / 1e9,
        "peak_terms": n,
        "final_terms": final_terms,
        "terms_out": terms_out,
        "layer_wall_ns": layer_wall_ns,
        "vmrss_kb": vmrss_kb if vmrss_kb is not None else int(40_000 + n * 0.00015),
        "vmhwm_kb": vmhwm_kb if vmhwm_kb is not None else int(45_000 + n * 0.00020),
    }
    if phase is None:
        for key in _BUCKETED_PHASE_FIELDS:
            row[key] = None
    else:
        row.update(phase)
    return row


def _bucketed_row(rng, *, tag, threads, eps, n, rep, target_bucket_len, **kw):
    variant = "bucketed-coarse" if target_bucket_len == COARSE_TARGET_BUCKET_LEN else "bucketed"
    wall_ns = _wall_ns(variant, threads, n) * (1.0 + rng.normal(0.0, 0.04))
    phase = _phase_stats(rng, wall_ns, threads, n, variant=variant)
    return _row(
        layer="bucketed",
        tag=tag,
        threads=threads,
        eps=eps,
        n=n,
        rep=rep,
        wall_ns=wall_ns,
        target_bucket_len=target_bucket_len,
        min_buckets=MIN_BUCKETS,
        phase=phase,
        **kw,
    )


def _synth_layer_profile(rng, n_final: int) -> tuple[list[int], list[float]]:
    """A staircase terms_out curve over LAYERS channels, and matching per-layer ms."""
    n_steps = LAYERS // (LAYERS // STEPS) if False else STEPS
    channels_per_step = LAYERS // STEPS
    terms_out = []
    cur = 8.0
    for layer_idx in range(LAYERS):
        step = layer_idx // channels_per_step
        growth = 1.0 + 0.02 * (1 + 0.3 * math.sin(step))
        cur = min(cur * growth, n_final)
        terms_out.append(int(cur))
    layer_wall_ms = [
        (t * _ns_per_term_layer("bucketed") / 1e6) * rng.uniform(0.9, 1.1) for t in terms_out
    ]
    return terms_out, layer_wall_ms


# --------------------------------------------------------------------------
# engine_ladder.jsonl
# --------------------------------------------------------------------------


def gen_engine_ladder(out_dir: Path) -> None:
    rng = _rng(1)
    n = 200_000
    rows = []

    for rep in range(REPS):
        rows.append(_row(layer="naive", tag="default", threads=1, eps=DEFAULT_EPS, n=n, rep=rep,
                          wall_ns=_wall_ns("naive", 1, n) * (1.0 + rng.normal(0.0, 0.05))))

    for variant, layer_name in (("threadmaps", "threadmaps"), ("mergesort", "mergesort")):
        for threads in THREAD_GRID:
            for rep in range(REPS):
                wall = _wall_ns(variant, threads, n) * (1.0 + rng.normal(0.0, 0.05))
                rows.append(_row(layer=layer_name, tag="default", threads=threads, eps=DEFAULT_EPS,
                                  n=n, rep=rep, wall_ns=wall))

    for rep in range(REPS):
        rows.append(_bucketed_row(rng, tag="default", threads=1, eps=DEFAULT_EPS, n=n, rep=rep,
                                   target_bucket_len=DEFAULT_TARGET_BUCKET_LEN))

    for target_bucket_len in (COARSE_TARGET_BUCKET_LEN, DEFAULT_TARGET_BUCKET_LEN):
        for rep in range(REPS):
            rows.append(_bucketed_row(rng, tag="default", threads=32, eps=DEFAULT_EPS, n=n, rep=rep,
                                       target_bucket_len=target_bucket_len))

    # Give the bucketed 1-thread, rep-0 row a full per-layer profile for fig7.
    terms_out, layer_wall_ms = _synth_layer_profile(rng, n)
    for r in rows:
        if r["layer"] == "bucketed" and r["threads"] == 1 and r["rep"] == 0:
            r["terms_out"] = terms_out
            r["layer_wall_ns"] = [ms * 1e6 for ms in layer_wall_ms]

    _write(rows, out_dir / "engine_ladder.jsonl")


# --------------------------------------------------------------------------
# thread_scaling.jsonl / thread_scaling_large.jsonl
# --------------------------------------------------------------------------


def gen_thread_scaling(out_dir: Path, *, name: str, eps: float, n: int, include_old: bool) -> None:
    rng = _rng(2 if include_old else 3)
    rows = []

    if include_old:
        for variant in ("threadmaps", "mergesort"):
            for threads in THREAD_GRID:
                for rep in range(REPS):
                    wall = _wall_ns(variant, threads, n) * (1.0 + rng.normal(0.0, 0.05))
                    rows.append(_row(layer=variant, tag="default", threads=threads, eps=eps,
                                      n=n, rep=rep, wall_ns=wall))

    for target_bucket_len in (DEFAULT_TARGET_BUCKET_LEN, COARSE_TARGET_BUCKET_LEN):
        for threads in THREAD_GRID:
            for rep in range(REPS):
                rows.append(_bucketed_row(rng, tag="default", threads=threads, eps=eps, n=n, rep=rep,
                                           target_bucket_len=target_bucket_len))

    _write(rows, out_dir / name)


# --------------------------------------------------------------------------
# bucket_sweep.jsonl / bucket_sweep_perf.jsonl
# --------------------------------------------------------------------------


def _bucket_sweep_grid() -> list[int]:
    return [2**k for k in range(5, 19)]  # 32 .. 262144


def _ns_per_term_for_bucket(target_bucket_len: int, threads: int) -> float:
    """U-shaped curve: overhead-dominated when small, cache-miss-dominated when large."""
    optimum = 1024.0
    log_ratio = math.log2(target_bucket_len / optimum)
    small_side = max(0.0, -log_ratio) * 18.0
    large_side = max(0.0, log_ratio) ** 1.6 * 9.0
    base = 95.0 + small_side + large_side
    # Parallel efficiency loss folded in as a flat per-thread-count multiplier.
    eff = _efficiency("bucketed", threads)
    return base / eff


def gen_bucket_sweep(out_dir: Path) -> None:
    rng = _rng(4)
    n = 800_000
    rows = []
    for target_bucket_len in _bucket_sweep_grid():
        for threads in (1, 16):
            ns_per = _ns_per_term_for_bucket(target_bucket_len, threads)
            wall_ns_median = ns_per * n * LAYERS / threads
            for rep, wall_ns in enumerate(_reps(rng, wall_ns_median, 0.04)):
                phase = _phase_stats(rng, wall_ns, threads, n)
                rows.append(_row(
                    layer="bucketed", tag="default", threads=threads, eps=DEFAULT_EPS,
                    n=n, rep=rep, wall_ns=wall_ns, target_bucket_len=target_bucket_len,
                    min_buckets=MIN_BUCKETS, phase=phase,
                ))
    _write(rows, out_dir / "bucket_sweep.jsonl")


def gen_bucket_sweep_perf(out_dir: Path) -> None:
    rng = _rng(5)
    n = 800_000
    rows = []
    for target_bucket_len in _bucket_sweep_grid():
        for threads in (1, 16):
            ns_per = _ns_per_term_for_bucket(target_bucket_len, threads)
            wall_ns_median = ns_per * n * LAYERS / threads

            # L2 crossover: gather run = 2 * terms_per_bucket * 48B vs 1 MiB.
            # LLC crossover: same quantity vs 24.75 MiB (research/notes bandwidth fact sheet).
            terms_per_bucket = target_bucket_len
            gather_bytes = 2 * terms_per_bucket * 48
            l2_x = gather_bytes / (1024**2)
            llc_x = gather_bytes / (24.75 * 1024**2)
            l2_miss_rate = 1.0 / (1.0 + math.exp(-4.0 * (math.log2(max(l2_x, 1e-6)))))
            llc_miss_rate = 0.02 + 0.25 / (1.0 + math.exp(-4.0 * (math.log2(max(llc_x, 1e-6)))))
            l2_miss_rate = min(0.55, max(0.01, l2_miss_rate * 0.5))
            llc_miss_rate = min(0.30, llc_miss_rate)

            for rep in range(3):
                wall_ns = wall_ns_median * (1.0 + rng.normal(0.0, 0.04))
                instructions = int(n * LAYERS * 6.5 / threads)
                cycles = int(instructions / 1.35)
                l2_refs = int(n * LAYERS * 0.9 / threads)
                l2_misses = int(l2_refs * l2_miss_rate * (1.0 + rng.normal(0.0, 0.05)))
                llc_loads = int(l2_misses * 1.05)
                llc_load_misses = int(llc_loads * llc_miss_rate * (1.0 + rng.normal(0.0, 0.05)))
                phase = _phase_stats(rng, wall_ns, threads, n)
                row = _row(
                    layer="bucketed", tag="default", threads=threads, eps=DEFAULT_EPS, n=n,
                    rep=rep, wall_ns=wall_ns, target_bucket_len=target_bucket_len,
                    min_buckets=MIN_BUCKETS, phase=phase,
                )
                row.update({
                    "cycles": cycles,
                    "instructions": instructions,
                    "l2_refs": l2_refs,
                    "l2_misses": l2_misses,
                    "llc_loads": llc_loads,
                    "llc_load_misses": llc_load_misses,
                    "l2_miss_rate": l2_misses / l2_refs,
                    "llc_miss_rate": llc_load_misses / llc_loads,
                    "ipc": instructions / cycles,
                })
                rows.append(row)
    _write(rows, out_dir / "bucket_sweep_perf.jsonl")


# --------------------------------------------------------------------------
# memory.jsonl
# --------------------------------------------------------------------------


def gen_memory(out_dir: Path) -> None:
    rng = _rng(6)
    n = 200_000
    rows = []
    baseline_vmrss = 38_000

    per_term_bytes = {
        "naive": 96.0,  # hash map overhead atop the 48B SoA payload
        "threadmaps": 130.0,  # one map per thread, some duplication
        "mergesort": 80.0,
        "bucketed": 52.0,
        "bucketed-coarse": 58.0,
    }
    threads_for = {"naive": 1, "threadmaps": 32, "mergesort": 32, "bucketed": 32, "bucketed-coarse": 32}
    target_bucket_len_for = {"bucketed": DEFAULT_TARGET_BUCKET_LEN, "bucketed-coarse": COARSE_TARGET_BUCKET_LEN}

    for variant, per_term in per_term_bytes.items():
        layer_name = "bucketed" if variant.startswith("bucketed") else variant
        threads = threads_for[variant]
        for rep in range(REPS):
            hwm_kb = baseline_vmrss + n * per_term / 1024.0 * (1.0 + rng.normal(0.0, 0.03))
            rss_kb = baseline_vmrss + n * per_term / 1024.0 * 0.85
            wall_ns = _wall_ns(variant, threads, n) * (1.0 + rng.normal(0.0, 0.05))
            rows.append(_row(
                layer=layer_name, tag="default", threads=threads, eps=DEFAULT_EPS, n=n, rep=rep,
                wall_ns=wall_ns, vmrss_kb=int(rss_kb), vmhwm_kb=int(hwm_kb),
                target_bucket_len=target_bucket_len_for.get(variant),
                min_buckets=MIN_BUCKETS if variant.startswith("bucketed") else None,
                phase=_phase_stats(rng, wall_ns, threads, n, variant=variant) if variant.startswith("bucketed") else None,
            ))
    _write(rows, out_dir / "memory.jsonl")


# --------------------------------------------------------------------------
# targetcpu_default.jsonl / targetcpu_native.jsonl
# --------------------------------------------------------------------------


def gen_targetcpu(out_dir: Path) -> None:
    rng_d = _rng(7)
    rng_n = _rng(8)
    n = 200_000
    threads_for = {"naive": 1, "threadmaps": 32, "mergesort": 32, "bucketed": 32, "bucketed-coarse": 32}
    # A native speedup per variant, close to consistent in sign across reps
    # (occasional sign flips model measurement noise, per fig2's "k/N pairs
    # same sign" framing).
    native_speedup = {
        "naive": 0.05,
        "threadmaps": 0.03,
        "mergesort": 0.04,
        "bucketed": 0.07,
        "bucketed-coarse": 0.06,
    }

    default_rows, native_rows = [], []
    for variant, speedup in native_speedup.items():
        layer_name = "bucketed" if variant.startswith("bucketed") else variant
        threads = threads_for[variant]
        target_bucket_len = {"bucketed": DEFAULT_TARGET_BUCKET_LEN, "bucketed-coarse": COARSE_TARGET_BUCKET_LEN}.get(variant)
        for rep in range(REPS):
            base_wall = _wall_ns(variant, threads, n) * (1.0 + rng_d.normal(0.0, 0.05))
            # native is drawn as a perturbation of the *same* base_wall so the
            # pairing is meaningful, with a small chance of a sign flip.
            local_noise = rng_n.normal(0.0, 0.02)
            native_wall = base_wall * (1.0 - speedup + local_noise)

            kw = dict(target_bucket_len=target_bucket_len,
                      min_buckets=MIN_BUCKETS if variant.startswith("bucketed") else None)
            phase_d = _phase_stats(rng_d, base_wall, threads, n, variant=variant) if variant.startswith("bucketed") else None
            phase_n = _phase_stats(rng_n, native_wall, threads, n, variant=variant) if variant.startswith("bucketed") else None

            default_rows.append(_row(layer=layer_name, tag="default", threads=threads, eps=DEFAULT_EPS,
                                      n=n, rep=rep, wall_ns=base_wall, phase=phase_d, **kw))
            native_rows.append(_row(layer=layer_name, tag="native", threads=threads, eps=DEFAULT_EPS,
                                     n=n, rep=rep, wall_ns=native_wall, phase=phase_n, **kw))

    _write(default_rows, out_dir / "targetcpu_default.jsonl")
    _write(native_rows, out_dir / "targetcpu_native.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="presentation/data")
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()

    out_dir = Path(args.data_dir) / "_synth"
    if args.out_suffix:
        out_dir = out_dir.with_name(out_dir.name + args.out_suffix)

    gen_engine_ladder(out_dir)
    gen_thread_scaling(out_dir, name="thread_scaling.jsonl", eps=DEFAULT_EPS, n=200_000, include_old=True)
    gen_thread_scaling(out_dir, name="thread_scaling_large.jsonl", eps=LARGE_EPS, n=4_000_000, include_old=False)
    gen_bucket_sweep(out_dir)
    gen_bucket_sweep_perf(out_dir)
    gen_memory(out_dir)
    gen_targetcpu(out_dir)


if __name__ == "__main__":
    main()
