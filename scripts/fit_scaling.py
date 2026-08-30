#!/usr/bin/env python3
"""Fit thread-scaling models (Amdahl's law and the Universal Scalability Law)
to criterion benchmark results.

Criterion 0.5 writes, for each benchmark id ``<t>`` inside a group
``<group>``::

    target/criterion/<group>/<t>/new/estimates.json

with the shape ``{"median": {"point_estimate": <ns>}, ...}``.  In this repo
the thread-scaling groups (``thread_scaling_rotation_1e6``,
``thread_scaling_bucketed_rotation_1e6``, ``thread_scaling_bucketed_gu2q``,
...) use the thread count as the benchmark id.

For each group this script:

  1. Collects ``(threads, median_ns)`` pairs.
  2. Computes measured speedup ``S(t) = median(1) / median(t)``.
  3. Fits Amdahl's law:  ``S(p) = 1 / (s + (1 - s) / p)``, ``s`` in [0, 1].
  4. Fits the USL:       ``S(p) = p / (1 + sigma*(p-1) + kappa*p*(p-1))``,
     ``sigma, kappa >= 0``.
  5. Reports R^2 for both fits and prints a paste-ready markdown table plus
     a short interpretation (contention- vs coherence-dominated).

Usage:
    scripts/fit_scaling.py
    scripts/fit_scaling.py --group thread_scaling_bucketed_gu2q
    scripts/fit_scaling.py --group all
    scripts/fit_scaling.py --group all --criterion-dir target/criterion
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

DEFAULT_GROUP = "thread_scaling_bucketed_rotation_1e6"
DEFAULT_CRITERION_DIR = "target/criterion"
GROUP_PREFIX = "thread_scaling"
MIN_FIT_POINTS = 3


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def discover_all_groups(criterion_dir):
    """Return sorted names of every top-level dir under criterion_dir whose
    name starts with GROUP_PREFIX."""
    if not os.path.isdir(criterion_dir):
        return []
    names = []
    for name in os.listdir(criterion_dir):
        if name.startswith(GROUP_PREFIX) and os.path.isdir(
            os.path.join(criterion_dir, name)
        ):
            names.append(name)
    return sorted(names)


def load_group_medians(criterion_dir, group):
    """Return a dict {thread_count: median_ns} for a given group directory.

    Non-integer benchmark-id subdirectories (e.g. criterion's own "report")
    are skipped. A benchmark-id directory missing "new/estimates.json" is
    skipped with a warning on stderr (e.g. an incomplete/aborted run).
    """
    group_dir = os.path.join(criterion_dir, group)
    if not os.path.isdir(group_dir):
        return {}

    medians = {}
    for entry in sorted(os.listdir(group_dir)):
        entry_path = os.path.join(group_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            threads = int(entry)
        except ValueError:
            continue  # e.g. "report"

        estimates_path = os.path.join(entry_path, "new", "estimates.json")
        if not os.path.isfile(estimates_path):
            print(
                f"warning: {group}/{entry}: missing new/estimates.json, skipping",
                file=sys.stderr,
            )
            continue

        try:
            with open(estimates_path) as f:
                data = json.load(f)
            median_ns = float(data["median"]["point_estimate"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(
                f"warning: {group}/{entry}: could not parse estimates.json ({exc}), skipping",
                file=sys.stderr,
            )
            continue

        medians[threads] = median_ns

    return medians


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def amdahl(p, s):
    """Amdahl's law: S(p) = 1 / (s + (1 - s) / p)."""
    return 1.0 / (s + (1.0 - s) / p)


def usl(p, sigma, kappa):
    """Universal Scalability Law: S(p) = p / (1 + sigma*(p-1) + kappa*p*(p-1))."""
    return p / (1.0 + sigma * (p - 1.0) + kappa * p * (p - 1.0))


def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_amdahl(ts, speedups):
    """Returns (params_or_None, r2_or_None, error_message_or_None)."""
    try:
        popt, _ = curve_fit(
            amdahl, ts, speedups, p0=[0.05], bounds=(0.0, 1.0), maxfev=10000
        )
    except RuntimeError:
        return None, None, "fit did not converge"
    pred = amdahl(np.asarray(ts, dtype=float), *popt)
    return popt, r_squared(speedups, pred), None


def fit_usl(ts, speedups):
    try:
        popt, _ = curve_fit(
            usl,
            ts,
            speedups,
            p0=[0.05, 1e-3],
            bounds=([0.0, 0.0], [1.0, np.inf]),
            maxfev=10000,
        )
    except RuntimeError:
        return None, None, "fit did not converge"
    pred = usl(np.asarray(ts, dtype=float), *popt)
    return popt, r_squared(speedups, pred), None


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def pick_unit(reference_ns):
    """Pick a human-scaled time unit based on the magnitude of a reference
    (typically the t=1 median)."""
    if reference_ns >= 1e9:
        return 1e9, "s"
    if reference_ns >= 1e6:
        return 1e6, "ms"
    if reference_ns >= 1e3:
        return 1e3, "µs"
    return 1.0, "ns"


def fmt_num(x, decimals=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x:.{decimals}f}"


def build_report(group, medians):
    """Build the paste-ready markdown report for one group.

    Returns (report_text, status) where status is one of:
      "ok", "no_data", "missing_baseline".
    """
    lines = [f"## {group}"]

    if not medians:
        lines.append("")
        lines.append(f"No data found for group `{group}`.")
        return "\n".join(lines), "no_data"

    if 1 not in medians:
        lines.append("")
        lines.append(
            f"Error: group `{group}` has no t=1 sample; cannot compute speedup "
            "(median(1) is the baseline for S(t) = median(1) / median(t))."
        )
        return "\n".join(lines), "missing_baseline"

    ts = sorted(medians.keys())
    base_ns = medians[1]
    speedups = [base_ns / medians[t] for t in ts]

    divisor, unit = pick_unit(base_ns)

    n = len(ts)
    do_fit = n >= MIN_FIT_POINTS

    amdahl_params = amdahl_r2 = amdahl_err = None
    usl_params = usl_r2 = usl_err = None
    if do_fit:
        amdahl_params, amdahl_r2, amdahl_err = fit_amdahl(ts, speedups)
        usl_params, usl_r2, usl_err = fit_usl(ts, speedups)

    def amdahl_pred(t):
        if amdahl_params is None:
            return None
        return amdahl(float(t), *amdahl_params)

    def usl_pred(t):
        if usl_params is None:
            return None
        return usl(float(t), *usl_params)

    lines.append("")
    lines.append(f"| t | median ({unit}) | speedup | Amdahl pred | USL pred |")
    lines.append("|---:|---:|---:|---:|---:|")
    for t, s in zip(ts, speedups):
        median_scaled = medians[t] / divisor
        a_pred = amdahl_pred(t)
        u_pred = usl_pred(t)
        lines.append(
            f"| {t} | {fmt_num(median_scaled, 3)} | {fmt_num(s, 2)} | "
            f"{fmt_num(a_pred, 2)} | {fmt_num(u_pred, 2)} |"
        )

    lines.append("")

    if not do_fit:
        lines.append(
            f"Fewer than {MIN_FIT_POINTS} data points (N={n}); skipping model fits."
        )
        return "\n".join(lines), "ok"

    if amdahl_err is not None:
        lines.append(f"Amdahl fit: {amdahl_err}.")
    else:
        (s_param,) = amdahl_params
        lines.append(
            f"Amdahl serial fraction s={fmt_num(s_param, 4)} (R²={fmt_num(amdahl_r2, 3)})"
        )

    if usl_err is not None:
        lines.append(f"USL fit: {usl_err}.")
    else:
        sigma, kappa = usl_params
        lines.append(
            f"USL σ={fmt_num(sigma, 4)}, κ={fmt_num(kappa, 6)} "
            f"(R²={fmt_num(usl_r2, 3)})"
        )

    if usl_err is None:
        sigma, kappa = usl_params
        # sigma >> kappa: contention dominates and the model behaves like
        # a queueing bottleneck with no eventual retrograde falloff.
        # kappa significant relative to sigma: coherence/crosstalk costs
        # dominate and the model predicts a scaling peak.
        contention_dominated = kappa <= 1e-9 or sigma > 20.0 * kappa
        if contention_dominated:
            lines.append(
                "Interpretation: contention-dominated (σ ≫ κ); "
                "no predicted retrograde peak in the fitted range."
            )
        else:
            peak_line = (
                "Interpretation: coherence-dominated (κ significant)"
            )
            if kappa > 0.0 and sigma < 1.0:
                p_star = math.sqrt((1.0 - sigma) / kappa)
                peak_line += f" → predicted peak at p*={fmt_num(p_star, 2)}"
            lines.append(peak_line + ".")

    return "\n".join(lines), "ok"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit Amdahl/USL thread-scaling models to criterion results."
    )
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=(
            "Criterion group name to fit, or 'all' to fit every group whose "
            f"name starts with '{GROUP_PREFIX}' (default: {DEFAULT_GROUP})"
        ),
    )
    parser.add_argument(
        "--criterion-dir",
        default=DEFAULT_CRITERION_DIR,
        help=f"Path to criterion's output directory (default: {DEFAULT_CRITERION_DIR})",
    )
    args = parser.parse_args(argv)

    if args.group == "all":
        groups = discover_all_groups(args.criterion_dir)
        if not groups:
            print(
                f"No groups matching '{GROUP_PREFIX}*' found under "
                f"'{args.criterion_dir}'."
            )
            return 1
    else:
        groups = [args.group]

    any_no_data = False
    reports = []
    for group in groups:
        medians = load_group_medians(args.criterion_dir, group)
        report, status = build_report(group, medians)
        reports.append(report)
        if status == "no_data":
            any_no_data = True

    print("\n\n".join(reports))

    return 1 if any_no_data else 0


if __name__ == "__main__":
    sys.exit(main())
