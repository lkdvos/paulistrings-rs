"""Collect `run_b8.py` JSON into per-configuration mean, std and standard error; optionally plot S_delta against eta.

    python aggregate.py results/ppmc results/mpi --csv summary.csv --plot ole_vs_eta.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ole  # noqa: E402

KEY = ("L", "eta", "delta", "policy", "cache", "eps", "max_weight", "mpi_ranks")


def load(roots) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for root in roots:
        for path in sorted(Path(root).rglob("seed_*.json")):
            rec = json.loads(path.read_text())
            cfg = rec["config"]
            groups[tuple(round(cfg[k], 6) if isinstance(cfg[k], float) else cfg[k] for k in KEY)].append(rec)
    return groups


def summarize(groups) -> list[dict]:
    rows = []
    for key, recs in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        row = dict(zip(KEY, key))
        row["n"] = len(recs)
        for field in ("S_diag", "S_exact"):
            vals = np.array([r[field] for r in recs if r.get(field) is not None], dtype=float)
            if len(vals) == 0:
                continue
            row[f"{field}_mean"] = float(vals.mean())
            row[f"{field}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
            row[f"{field}_stderr"] = row[f"{field}_std"] / math.sqrt(len(vals)) if len(vals) > 1 else math.nan
        row["norm_mean"] = float(np.mean([r["norm"] for r in recs]))
        row["collapses_mean"] = float(np.mean([r["collapses"] or 0 for r in recs]))
        row["propagate_s_mean"] = float(np.mean([r["propagate_s"] for r in recs]))
        row["peak_rss_gb_max"] = max(r["peak_rss_kb_max_rank"] for r in recs) / 1e6
        rows.append(row)
    return rows


def plot(rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = defaultdict(list)
    for r in rows:
        if "S_diag_mean" in r:
            series[(r["L"], r["policy"], r["cache"], r["mpi_ranks"])].append(r)
    for (L, policy, cache, ranks), rs in sorted(series.items(), key=lambda kv: str(kv[0])):
        rs.sort(key=lambda r: r["eta"])
        label = f"L={L} {policy}" + (f" M={cache:.0e}" if cache else "") + (f" ranks={ranks}" if ranks > 1 else "")
        ax.errorbar(
            [r["eta"] for r in rs],
            [r["S_diag_mean"] for r in rs],
            yerr=[0.0 if math.isnan(r.get("S_diag_std", math.nan)) else r["S_diag_std"] for r in rs],
            marker="o",
            capsize=3,
            label=label,
        )
    ref = ole.spec()["references"]["table_I"]
    eta_ref = ole.eta_of_alpha(0.15)
    for row in ref["rows"]:
        if row[1] is not None:
            ax.errorbar([eta_ref], [row[1][0]], yerr=[row[1][1]], marker="s", color="k", ms=4, capsize=2)
            ax.annotate(f"exp L={row[0]}", (eta_ref, row[1][0]), textcoords="offset points", xytext=(5, 0), fontsize=7)
    ax.axhline(ole.spec()["references"]["full_scrambling"]["value"], ls=":", color="grey", label="full scrambling")
    ax.set_xlabel(r"scattering strength $\eta$")
    ax.set_ylabel(r"$S_{\delta=0.3}$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args(argv)
    rows = summarize(load(args.roots))
    for r in rows:
        s = f"S_diag={r['S_diag_mean']:.4f}±{r.get('S_diag_stderr', math.nan):.4f} (std {r.get('S_diag_std', math.nan):.4f})"
        if "S_exact_mean" in r:
            s += f" S_exact={r['S_exact_mean']:.4f}"
        print(f"L={r['L']} eta={r['eta']:.4f} {r['policy']} M={r['cache']} eps={r['eps']} n={r['n']}: {s} norm={r['norm_mean']:.4g}")
    if args.csv:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in KEY, k))
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    if args.plot:
        plot(rows, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
