#!/usr/bin/env python3
"""Turn one benchmark campaign's data files into a self-contained HTML report.

Usage:
    python3 scripts/perf-viz.py <dir>/<campaign-name> [--compare OLD_SNAPSHOT.json]

``<dir>/<campaign-name>`` is a path *prefix*: this tool looks for, and
renders whichever of the following exist (see the docstrings on each
``load_*`` function for the exact shape expected):

    <prefix>.txt                       campaign log (provenance header only)
    <prefix>.json                      criterion snapshot (criterion-report.py format)
    <prefix>-probe.json                one JSON object per line, engine phase timings
    <prefix>-scaling-<placement>.json  criterion snapshot, thread-scaling groups
    bandwidth.txt (same directory)     memory-bandwidth ceiling sections

Output is written to ``<prefix>-report.html`` and its path printed on
success. Exits 1 only when *none* of the input files exist.

Stdlib only; no cargo, no benchmarks are run by this script.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------
# Palette (fixed per phase name, consistent across every chart)
# --------------------------------------------------------------------------

WALL_PHASES = [
    "rebucket", "prepare", "rescale", "span_plan", "permute",
    "coset_loop", "unpermute", "recount", "finalize", "fallback",
]
BUSY_PHASES = ["gather", "sort", "merge", "swap", "size", "clear"]

# A single fixed color per phase name, shared between the wall and busy
# palettes where names coincide is not required (the two groups are
# disjoint) but each name maps to exactly one color everywhere it appears.
_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#86bcb6", "#d37295",
]

PHASE_COLOR = {}
for _i, _name in enumerate(WALL_PHASES + BUSY_PHASES):
    PHASE_COLOR[_name] = _PALETTE[_i % len(_PALETTE)]

LINE_COLORS = [
    "#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1",
    "#76b7b2", "#edc948", "#ff9da7", "#9c755f",
]

GRID_COLOR = "#d8d8d8"
TEXT_COLOR = "#222222"
MUTED_COLOR = "#666666"

# --------------------------------------------------------------------------
# Number formatting helpers
# --------------------------------------------------------------------------


def human_ns(ns: Optional[float]) -> str:
    """Format a nanosecond duration as ns / µs / ms / s, whichever reads best."""
    if ns is None:
        return "n/a"
    a = abs(ns)
    if a < 1e3:
        return f"{ns:.2f} ns"
    if a < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if a < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def engineering(x: Optional[float], unit: str = "") -> str:
    """Format ``x`` in engineering notation, e.g. 3.2e8 (exponent a multiple of 3)."""
    if x is None or x != x:  # NaN check
        return "n/a"
    if x == 0:
        return f"0{unit}"
    sign = "-" if x < 0 else ""
    x = abs(x)
    exp = int(math.floor(math.log10(x)))
    eng_exp = exp - (exp % 3)
    mantissa = x / (10 ** eng_exp)
    return f"{sign}{mantissa:.2f}e{eng_exp}{unit}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2f}%"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# --------------------------------------------------------------------------
# Input loaders — each degrades to None / empty on a missing or bad file.
# --------------------------------------------------------------------------


def load_provenance(txt_path: Path, missing: list, malformed: dict) -> Optional[dict]:
    """Parse only the header (first ~8 lines) plus any 'determinism gate' lines.

    Header shape (see any <prefix>.txt in benchmarks/results/):
        <title line>
        date: ...
        commit: ... (dirty)?
        rustc ...
        threads (nproc): ...
        governor: ...
        cpu: ...
        load at start: ...
    """
    if not txt_path.exists():
        missing.append(txt_path.name)
        return None
    try:
        lines = txt_path.read_text(errors="replace").splitlines()
    except OSError:
        missing.append(txt_path.name)
        return None

    prov: dict = {"title": None, "host": None, "gates": []}
    header = lines[:8]
    if header:
        title_line = header[0]
        prov["title"] = title_line
        m = re.search(r"[—-]{1,3}\s*(\S+)\s*$", title_line)
        if m:
            prov["host"] = m.group(1)
    for line in header[1:]:
        m = re.match(r"^\s*([A-Za-z][A-Za-z ()]*):\s*(.*)$", line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        prov[key] = val

    for line in lines:
        m = re.match(r"^determinism gate:\s*(PASS|FAIL)\s*$", line.strip())
        if m:
            prov["gates"].append(m.group(1))

    return prov


def load_criterion_json(path: Path, missing: list) -> Optional[dict]:
    """Load a criterion-report.py snapshot: {full_id: {median_ns, ...}}."""
    if not path.exists():
        missing.append(path.name)
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        missing.append(f"{path.name} (malformed JSON)")
        return None


def load_probe(path: Path, missing: list, malformed: dict) -> list:
    """Load <prefix>-probe.json: one JSON object per line."""
    if not path.exists():
        missing.append(path.name)
        return []
    rows = []
    bad = 0
    try:
        text = path.read_text(errors="replace")
    except OSError:
        missing.append(path.name)
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        malformed[path.name] = bad
    return rows


def load_scaling_files(prefix: Path, missing: list) -> dict:
    """Glob <prefix>-scaling-<placement>.json; return {placement: snapshot}."""
    pattern = f"{prefix.name}-scaling-*.json"
    found = sorted(glob.glob(str(prefix.parent / pattern)))
    result = {}
    if not found:
        missing.append(f"{prefix.name}-scaling-*.json")
        return result
    for fp in found:
        p = Path(fp)
        m = re.match(rf"^{re.escape(prefix.name)}-scaling-(.+)\.json$", p.name)
        placement = m.group(1) if m else p.stem
        try:
            result[placement] = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            missing.append(f"{p.name} (malformed JSON)")
    return result


def load_bandwidth(dir_path: Path, missing: list) -> list:
    """Parse bandwidth.txt into a list of (section_label, {kernel: best_gbps})."""
    path = dir_path / "bandwidth.txt"
    if not path.exists():
        missing.append("bandwidth.txt")
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        missing.append("bandwidth.txt")
        return []

    sections = []
    current_label = None
    current_kernels: dict = {}
    section_re = re.compile(r"^===\s*(.+?)\s*===$")
    kernel_re = re.compile(
        r"kernel=(\S+)\s+threads=(\d+)\s+mib=(\d+)\s+reps=(\d+)\s+"
        r"best_gbps=([\d.]+)\s+avg_gbps=([\d.]+)"
    )
    for line in lines:
        sm = section_re.match(line.strip())
        if sm:
            if current_label is not None and current_kernels:
                sections.append((current_label, current_kernels))
            current_label = sm.group(1)
            current_kernels = {}
            continue
        km = kernel_re.search(line)
        if km and current_label is not None:
            kernel = km.group(1)
            best_gbps = float(km.group(5))
            current_kernels[kernel] = best_gbps
    if current_label is not None and current_kernels:
        sections.append((current_label, current_kernels))
    return sections


# --------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------


def svg_open(width: int, height: int, extra_class: str = "") -> str:
    cls = f' class="{esc(extra_class)}"' if extra_class else ""
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}"'
        f' xmlns="http://www.w3.org/2000/svg"{cls}>'
    )


def svg_text(x, y, text, size=11, anchor="start", fill=TEXT_COLOR, weight="normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}">{esc(text)}</text>'
    )


# --------------------------------------------------------------------------
# Section 1: header / provenance
# --------------------------------------------------------------------------


def render_header(campaign_name: str, prov: Optional[dict]) -> str:
    out = []
    out.append(f"<h1>{esc(campaign_name)}</h1>")
    out.append('<p class="subtitle">paulistrings perf report</p>')
    if prov is None:
        out.append('<p class="note">No campaign log (.txt) found — provenance unavailable.</p>')
        return "\n".join(out)

    commit = prov.get("commit", "n/a")
    date = prov.get("date", "n/a")
    host = prov.get("host") or "n/a"
    cpu = prov.get("cpu", "n/a")
    governor = prov.get("governor", "n/a")
    nproc = prov.get("threads (nproc)", "n/a")
    load = prov.get("load at start", "n/a")

    out.append('<dl class="provenance">')
    for label, val in [
        ("commit", commit),
        ("date", date),
        ("host", host),
        ("cpu", cpu),
        ("governor", governor),
        ("threads (nproc)", nproc),
        ("load at start", load),
    ]:
        out.append(f"<dt>{esc(label)}</dt><dd>{esc(val)}</dd>")
    out.append("</dl>")

    gates = prov.get("gates", [])
    if gates:
        out.append('<p class="gates">')
        for i, g in enumerate(gates):
            cls = "chip-pass" if g == "PASS" else "chip-fail"
            label = f"determinism gate {i + 1}: {g}" if len(gates) > 1 else f"determinism gate: {g}"
            out.append(f'<span class="chip {cls}">{esc(label)}</span>')
        out.append("</p>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Section 2: phase breakdown
# --------------------------------------------------------------------------


def render_legend() -> str:
    parts = ['<div class="legend">']
    parts.append('<div class="legend-group"><span class="legend-title">wall phases</span>')
    for name in WALL_PHASES:
        parts.append(
            f'<span class="legend-item"><span class="swatch" '
            f'style="background:{PHASE_COLOR[name]}"></span>{esc(name)}</span>'
        )
    parts.append("</div>")
    parts.append('<div class="legend-group"><span class="legend-title">busy (all workers)</span>')
    for name in BUSY_PHASES:
        parts.append(
            f'<span class="legend-item"><span class="swatch" '
            f'style="background:{PHASE_COLOR[name]}"></span>{esc(name)}</span>'
        )
    parts.append("</div>")
    parts.append("</div>")
    return "\n".join(parts)


def _stacked_bar_svg(
    width: int,
    height: int,
    segments: list,
    total: float,
    label_prefix: str,
) -> str:
    """segments: list of (name, value_ns_or_units, per_layer_ms_for_title)."""
    parts = [svg_open(width, height)]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f2f2f2"/>')
    if total <= 0:
        parts.append(svg_text(4, height / 2 + 4, "n/a", size=10, fill=MUTED_COLOR))
        parts.append("</svg>")
        return "\n".join(parts)

    x = 0.0
    for name, value, per_ms in segments:
        frac = value / total
        seg_w = frac * width
        if seg_w < width * 0.004:
            # too thin to render, but still counted in total (hover info lost
            # for this one — acceptable, it's sub-0.4%)
            continue
        color = PHASE_COLOR.get(name, "#999999")
        title = f"{name}: {per_ms:.2f} ms/layer ({frac * 100:.1f}% of {label_prefix})"
        parts.append(
            f'<rect x="{x:.2f}" y="0" width="{seg_w:.2f}" height="{height}" '
            f'fill="{color}"><title>{esc(title)}</title></rect>'
        )
        x += seg_w
    parts.append("</svg>")
    return "\n".join(parts)


def render_phase_breakdown(probe_rows: list) -> str:
    if not probe_rows:
        return '<p class="note">No probe sidecar (.-probe.json) found — phase breakdown unavailable.</p>'

    groups: dict = {}
    for row in probe_rows:
        layer = row.get("layer", "?")
        groups.setdefault(layer, []).append(row)

    out = [render_legend()]

    bar_w = 560
    bar_h = 26
    busy_h = 16

    for layer in sorted(groups):
        rows = sorted(groups[layer], key=lambda r: r.get("threads", 0))
        out.append(f'<h3 class="layer-name">{esc(layer)}</h3>')
        for row in rows:
            threads = row.get("threads", "?")
            wall_ns = row.get("wall_ns", 0) or 0
            layers = row.get("layers", 1) or 1
            terms_in = row.get("terms_in", 0) or 0
            vmhwm_kb = row.get("vmhwm_kb", 0) or 0

            wall_ms_per_layer = wall_ns / 1e6 / layers if layers else 0.0

            wall_segments = []
            for name in WALL_PHASES:
                v = row.get(f"{name}_ns", 0) or 0
                per_ms = (v / 1e6) / layers if layers else 0.0
                wall_segments.append((name, v, per_ms))

            busy_names = BUSY_PHASES
            busy_values = {n: (row.get(f"{n}_ns", 0) or 0) for n in busy_names}
            busy_total = sum(busy_values.values())
            threads_n = row.get("threads", 1) or 1
            busy_segments = []
            for name in busy_names:
                v = busy_values[name]
                per_ms = (v / 1e6) / layers if layers else 0.0
                busy_segments.append((name, v, per_ms))

            wall_bar = _stacked_bar_svg(bar_w, bar_h, wall_segments, wall_ns, "wall")
            busy_bar = _stacked_bar_svg(bar_w, busy_h, busy_segments, busy_total, "busy total")

            strings_per_s = terms_in / (wall_ns / 1e9) if wall_ns > 0 else None
            coset_loop_ns = row.get("coset_loop_ns", 0) or 0
            par_eff = None
            if coset_loop_ns > 0:
                par_eff = busy_total / (coset_loop_ns * threads_n)

            vmhwm_mb = vmhwm_kb / 1024.0

            out.append('<div class="phase-row">')
            out.append(f'<div class="phase-threads">threads = {esc(threads)}</div>')
            out.append('<div class="phase-bars">')
            out.append(f'<div class="wall-bar">{wall_bar}</div>')
            out.append(f'<div class="busy-bar-wrap"><div class="busy-bar">{busy_bar}</div></div>')
            out.append("</div>")
            out.append('<div class="phase-stats">')
            out.append(f'<div>wall: {wall_ms_per_layer:.3f} ms/layer</div>')
            out.append(f'<div>strings/s: {engineering(strings_per_s)}</div>')
            if par_eff is not None:
                out.append(f'<div>parallel eff.: {par_eff * 100:.1f}%</div>')
            out.append(f'<div>VmHWM: {vmhwm_mb:.1f} MB</div>')
            out.append("</div>")
            out.append("</div>")

    return "\n".join(out)


# --------------------------------------------------------------------------
# Section 3: throughput vs threads (from probe rows)
# --------------------------------------------------------------------------


def render_throughput_chart(probe_rows: list) -> str:
    if not probe_rows:
        return '<p class="note">No probe sidecar found — throughput-vs-threads chart unavailable.</p>'

    by_layer: dict = {}
    thread_set = set()
    for row in probe_rows:
        layer = row.get("layer", "?")
        threads = row.get("threads")
        wall_ns = row.get("wall_ns", 0) or 0
        terms_in = row.get("terms_in", 0) or 0
        if threads is None or wall_ns <= 0:
            continue
        strings_per_s = terms_in / (wall_ns / 1e9)
        by_layer.setdefault(layer, {})[threads] = strings_per_s
        thread_set.add(threads)

    if not thread_set:
        return '<p class="note">Probe data present but no usable (threads, wall_ns) points.</p>'

    threads_sorted = sorted(thread_set)
    n_t = len(threads_sorted)

    width, height = 640, 380
    margin_l, margin_r, margin_t, margin_b = 60, 90, 20, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_vals = [v for series in by_layer.values() for v in series.values() if v > 0]
    if not all_vals:
        return '<p class="note">No positive throughput values to chart.</p>'
    y_min_exp = int(math.floor(math.log10(min(all_vals))))
    y_max_exp = int(math.ceil(math.log10(max(all_vals))))
    if y_min_exp == y_max_exp:
        y_max_exp += 1

    def x_for(t):
        idx = threads_sorted.index(t)
        if n_t == 1:
            return margin_l + plot_w / 2
        return margin_l + idx * plot_w / (n_t - 1)

    def y_for(v):
        if v <= 0:
            return margin_t + plot_h
        exp = math.log10(v)
        frac = (exp - y_min_exp) / (y_max_exp - y_min_exp)
        return margin_t + plot_h - frac * plot_h

    parts = [svg_open(width, height)]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    # gridlines + y labels (powers of ten)
    for exp in range(y_min_exp, y_max_exp + 1):
        y = y_for(10 ** exp)
        parts.append(
            f'<line x1="{margin_l}" y1="{y:.1f}" x2="{margin_l + plot_w}" y2="{y:.1f}" '
            f'stroke="{GRID_COLOR}" stroke-width="1"/>'
        )
        parts.append(svg_text(margin_l - 6, y + 3, f"1e{exp}", size=10, anchor="end", fill=MUTED_COLOR))

    # x axis labels
    for t in threads_sorted:
        x = x_for(t)
        parts.append(svg_text(x, margin_t + plot_h + 16, str(t), size=10, anchor="middle"))
    parts.append(svg_text(margin_l + plot_w / 2, height - 6, "threads", size=11, anchor="middle", fill=MUTED_COLOR))

    for i, (layer, series) in enumerate(sorted(by_layer.items())):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        pts = [(t, series[t]) for t in threads_sorted if t in series and series[t] > 0]
        if not pts:
            continue
        path_pts = " ".join(f"{x_for(t):.1f},{y_for(v):.1f}" for t, v in pts)
        parts.append(f'<polyline points="{path_pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for t, v in pts:
            title = f"{esc(layer)}: {engineering(v)} strings/s @ {t} threads"
            parts.append(
                f'<circle cx="{x_for(t):.1f}" cy="{y_for(v):.1f}" r="3" fill="{color}">'
                f'<title>{title}</title></circle>'
            )
        last_t, last_v = pts[-1]
        parts.append(
            svg_text(x_for(last_t) + 6, y_for(last_v) + 3, layer, size=10, fill=color, weight="bold")
        )

    parts.append("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Section 4: thread scaling (criterion)
# --------------------------------------------------------------------------


def render_thread_scaling(scaling: dict) -> str:
    if not scaling:
        return '<p class="note">No thread-scaling snapshots (-scaling-*.json) found.</p>'

    # group full_id -> group name + thread count, per placement
    # key like "thread_scaling_bucketed_rotation_1e6/8"
    groups: dict = {}  # group_name -> placement -> {t: median_ns}
    for placement, snapshot in scaling.items():
        for full_id, metrics in snapshot.items():
            m = re.match(r"^(.*)/(\d+)$", full_id)
            if not m:
                continue
            group_name, t_str = m.group(1), m.group(2)
            t = int(t_str)
            median_ns = (metrics or {}).get("median_ns")
            if median_ns is None:
                continue
            groups.setdefault(group_name, {}).setdefault(placement, {})[t] = median_ns

    if not groups:
        return '<p class="note">Scaling snapshots present but no parseable thread-scaling groups.</p>'

    out = []
    width, height = 640, 380
    margin_l, margin_r, margin_t, margin_b = 60, 90, 20, 40
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    for group_name in sorted(groups):
        placements = groups[group_name]
        usable = {p: series for p, series in placements.items() if 1 in series}
        skipped = sorted(set(placements) - set(usable))
        if not usable:
            out.append(f'<h3 class="layer-name">{esc(group_name)}</h3>')
            out.append('<p class="note">All placements missing t=1 baseline — skipped.</p>')
            continue

        speedups: dict = {}
        for p, series in usable.items():
            base = series[1]
            speedups[p] = {t: (base / v if v > 0 else None) for t, v in series.items()}

        all_t = sorted({t for s in speedups.values() for t in s})
        max_t = max(all_t) if all_t else 1
        max_s = max(
            (v for s in speedups.values() for v in s.values() if v is not None),
            default=1.0,
        )
        max_axis = max(max_t, max_s, 1.0)

        def x_for(t, _max_axis=max_axis):
            return margin_l + (t / _max_axis) * plot_w

        def y_for(s, _max_axis=max_axis):
            return margin_t + plot_h - (s / _max_axis) * plot_h

        out.append(f'<h3 class="layer-name">{esc(group_name)}</h3>')
        if skipped:
            out.append(
                f'<p class="note">Placement(s) without t=1 skipped: {esc(", ".join(skipped))}</p>'
            )

        parts = [svg_open(width, height)]
        parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

        n_ticks = 5
        for i in range(n_ticks + 1):
            val = max_axis * i / n_ticks
            x = x_for(val)
            y = y_for(val)
            parts.append(
                f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" '
                f'stroke="{GRID_COLOR}" stroke-width="1"/>'
            )
            parts.append(
                f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" '
                f'y2="{margin_t + plot_h}" stroke="{GRID_COLOR}" stroke-width="1"/>'
            )
            parts.append(svg_text(x, margin_t + plot_h + 16, f"{val:.0f}", size=9, anchor="middle", fill=MUTED_COLOR))
            parts.append(svg_text(margin_l - 6, y + 3, f"{val:.1f}", size=9, anchor="end", fill=MUTED_COLOR))

        # ideal y=x dashed line
        parts.append(
            f'<line x1="{x_for(0):.1f}" y1="{y_for(0):.1f}" x2="{x_for(max_axis):.1f}" '
            f'y2="{y_for(max_axis):.1f}" stroke="{MUTED_COLOR}" stroke-width="1.5" '
            f'stroke-dasharray="5,4"><title>ideal (linear speedup)</title></line>'
        )

        for i, placement in enumerate(sorted(speedups)):
            series = speedups[placement]
            color = LINE_COLORS[i % len(LINE_COLORS)]
            pts = [(t, s) for t, s in sorted(series.items()) if s is not None]
            if not pts:
                continue
            path_pts = " ".join(f"{x_for(t):.1f},{y_for(s):.1f}" for t, s in pts)
            parts.append(f'<polyline points="{path_pts}" fill="none" stroke="{color}" stroke-width="2"/>')
            for t, s in pts:
                title = f"{esc(placement)}: {s:.2f}x @ {t} threads"
                parts.append(
                    f'<circle cx="{x_for(t):.1f}" cy="{y_for(s):.1f}" r="3" fill="{color}">'
                    f'<title>{title}</title></circle>'
                )
            last_t, last_s = pts[-1]
            parts.append(svg_text(x_for(last_t) + 6, y_for(last_s) + 3, placement, size=10, fill=color, weight="bold"))
        parts.append(svg_text(margin_l + plot_w / 2, height - 6, "threads", size=11, anchor="middle", fill=MUTED_COLOR))
        parts.append("</svg>")
        out.append("\n".join(parts))

        # medians table
        out.append('<div class="table-wrap"><table>')
        out.append("<thead><tr><th>threads</th>")
        for placement in sorted(usable):
            out.append(f"<th>{esc(placement)} (ms)</th>")
        out.append("</tr></thead><tbody>")
        for t in all_t:
            out.append(f"<tr><td>{t}</td>")
            for placement in sorted(usable):
                v = usable[placement].get(t)
                out.append(f"<td>{v / 1e6:.3f}</td>" if v is not None else "<td>n/a</td>")
            out.append("</tr>")
        out.append("</tbody></table></div>")

    return "\n".join(out)


# --------------------------------------------------------------------------
# Section 5: criterion microbenchmarks
# --------------------------------------------------------------------------


def render_criterion_table(snapshot: Optional[dict], compare_snapshot: Optional[dict]) -> str:
    if not snapshot:
        return '<p class="note">No criterion snapshot (.json) found.</p>'

    out = ['<div class="table-wrap"><table>']
    header = "<tr><th>bench id</th><th>median</th><th>Melem/s</th>"
    if compare_snapshot is not None:
        header += "<th>old median</th><th>&Delta;%</th>"
    header += "</tr>"
    out.append(f"<thead>{header}</thead><tbody>")

    only_old = []
    ids = sorted(snapshot)
    old_ids = set(compare_snapshot) if compare_snapshot is not None else set()

    for bench_id in ids:
        metrics = snapshot.get(bench_id) or {}
        median_ns = metrics.get("median_ns")
        melem = metrics.get("melem_per_s")
        row = f"<tr><td>{esc(bench_id)}</td><td>{esc(human_ns(median_ns))}</td>"
        row += f"<td>{melem:.2f}</td>" if melem is not None else "<td>n/a</td>"
        if compare_snapshot is not None:
            old_metrics = compare_snapshot.get(bench_id)
            if old_metrics is None:
                row += "<td>n/a</td><td>n/a</td>"
            else:
                old_median = old_metrics.get("median_ns")
                if old_median is None or median_ns is None or old_median == 0:
                    row += f"<td>{esc(human_ns(old_median))}</td><td>n/a</td>"
                else:
                    delta = (median_ns - old_median) / old_median * 100.0
                    cls = ""
                    if delta > 5.0:
                        cls = "delta-bad"
                    elif delta < -5.0:
                        cls = "delta-good"
                    row += f"<td>{esc(human_ns(old_median))}</td><td class=\"{cls}\">{esc(fmt_pct(delta))}</td>"
        row += "</tr>"
        out.append(row)
    out.append("</tbody></table></div>")

    if compare_snapshot is not None:
        only_old = sorted(set(compare_snapshot) - set(snapshot))
        only_new = sorted(set(snapshot) - set(compare_snapshot))
        if only_old or only_new:
            out.append('<p class="note">')
            if only_new:
                out.append(f"Only in this snapshot: {esc(', '.join(only_new))}. ")
            if only_old:
                out.append(f"Only in compare snapshot: {esc(', '.join(only_old))}.")
            out.append("</p>")

    return "\n".join(out)


# --------------------------------------------------------------------------
# Section 6: bandwidth ceiling
# --------------------------------------------------------------------------


def render_bandwidth(sections: list) -> str:
    if not sections:
        return '<p class="note">No bandwidth.txt found — bandwidth ceiling unavailable.</p>'

    kernels = []
    for _, kmap in sections:
        for k in kmap:
            if k not in kernels:
                kernels.append(k)

    out = ['<div class="table-wrap"><table>']
    out.append("<thead><tr><th>placement</th>")
    for k in kernels:
        out.append(f"<th>{esc(k)} (GB/s)</th>")
    out.append("</tr></thead><tbody>")
    for label, kmap in sections:
        out.append(f"<tr><td>{esc(label)}</td>")
        for k in kernels:
            v = kmap.get(k)
            out.append(f"<td>{v:.2f}</td>" if v is not None else "<td>-</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    out.append(
        '<p class="note">Reference rows: <em>1 core</em> sections are the ceiling for '
        "serial phases (rebucket, prepare, rescale, finalize); "
        "<em>one-socket</em> sections (e.g. “node0, 8 physical”) are the ceiling "
        "for the coset loop when a layer is placed on one socket, and "
        "<em>both sockets</em> sections are the ceiling when placed across both.</p>"
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------


def render_footer(consumed: list, missing: list, malformed: dict) -> str:
    out = ["<hr/>", '<footer>']
    if consumed:
        out.append(f'<p><strong>Files consumed:</strong> {esc(", ".join(consumed))}</p>')
    if missing:
        out.append(f'<p><strong>Files missing:</strong> {esc(", ".join(missing))}</p>')
    if malformed:
        parts = [f"{name} ({count} bad line(s) skipped)" for name, count in malformed.items()]
        out.append(f'<p><strong>Malformed lines skipped:</strong> {esc(", ".join(parts))}</p>')
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    out.append(f"<p>Generated {esc(ts)}. See <code>benchmarks/PROFILING.md</code> for methodology.</p>")
    out.append("</footer>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Page assembly
# --------------------------------------------------------------------------

STYLE = """
:root {
  color-scheme: light;
}
* { box-sizing: border-box; }
body {
  background: #ffffff;
  color: #222222;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px 60px;
  line-height: 1.5;
}
h1 { margin-bottom: 2px; font-size: 1.7em; }
.subtitle { color: #666666; margin-top: 0; margin-bottom: 20px; }
h2 { border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; margin-top: 40px; }
h3.layer-name { margin-top: 24px; margin-bottom: 6px; font-size: 1.05em; color: #333333; }
dl.provenance {
  display: grid;
  grid-template-columns: max-content 1fr;
  column-gap: 12px;
  row-gap: 4px;
  font-size: 0.92em;
  max-width: 640px;
}
dl.provenance dt { color: #666666; font-weight: 600; }
dl.provenance dd { margin: 0; }
.chip {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.85em;
  font-weight: 600;
  margin-right: 6px;
}
.chip-pass { background: #e2f4e5; color: #1e7d34; }
.chip-fail { background: #fce4e4; color: #b3261e; }
.note { color: #666666; font-size: 0.9em; }
.legend {
  display: flex;
  gap: 28px;
  flex-wrap: wrap;
  margin-bottom: 12px;
  font-size: 0.85em;
}
.legend-group { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.legend-title { color: #666666; font-weight: 600; margin-right: 4px; }
.legend-item { display: inline-flex; align-items: center; gap: 4px; }
.swatch { width: 10px; height: 10px; display: inline-block; border-radius: 2px; }
.phase-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
  flex-wrap: wrap;
}
.phase-threads { width: 90px; font-weight: 600; font-size: 0.9em; flex-shrink: 0; }
.phase-bars { flex: 1 1 auto; min-width: 300px; }
.wall-bar { line-height: 0; }
.busy-bar-wrap { margin-left: 24px; margin-top: 3px; line-height: 0; }
.phase-stats {
  font-size: 0.82em;
  color: #333333;
  font-variant-numeric: tabular-nums;
  display: grid;
  grid-template-columns: repeat(2, max-content);
  gap: 2px 18px;
  flex-shrink: 0;
}
.table-wrap { overflow-x: auto; margin: 10px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.88em; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 10px; border-bottom: 1px solid #eeeeee; white-space: nowrap; }
th { color: #666666; font-weight: 600; background: #fafafa; }
.delta-bad { color: #b3261e; font-weight: 600; }
.delta-good { color: #1e7d34; font-weight: 600; }
footer { color: #666666; font-size: 0.82em; }
footer p { margin: 4px 0; }
code { background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }
"""


def build_report(prefix: Path, compare_path: Optional[Path]) -> str:
    campaign_name = prefix.name
    dir_path = prefix.parent

    missing: list = []
    malformed: dict = {}
    consumed: list = []

    txt_path = prefix.with_name(prefix.name + ".txt")
    json_path = prefix.with_name(prefix.name + ".json")
    probe_path = prefix.with_name(prefix.name + "-probe.json")

    prov = load_provenance(txt_path, missing, malformed)
    if prov is not None:
        consumed.append(txt_path.name)

    snapshot = load_criterion_json(json_path, missing)
    if snapshot is not None:
        consumed.append(json_path.name)

    probe_rows = load_probe(probe_path, missing, malformed)
    if probe_rows:
        consumed.append(probe_path.name)

    scaling = load_scaling_files(prefix, missing)
    if scaling:
        consumed.extend(f"{prefix.name}-scaling-{p}.json" for p in sorted(scaling))

    bandwidth_sections = load_bandwidth(dir_path, missing)
    if bandwidth_sections:
        consumed.append("bandwidth.txt")

    compare_snapshot = None
    if compare_path is not None:
        compare_snapshot = load_criterion_json(compare_path, missing)
        if compare_snapshot is not None:
            consumed.append(str(compare_path))

    any_input = any([prov, snapshot, probe_rows, scaling, bandwidth_sections])
    if not any_input:
        raise SystemExit(
            f"perf-viz: no input files found for prefix {prefix} "
            f"(looked for {txt_path.name}, {json_path.name}, {probe_path.name}, "
            f"{prefix.name}-scaling-*.json, bandwidth.txt in {dir_path})"
        )

    parts = []
    title = f"{campaign_name} — paulistrings perf report"
    parts.append(
        "<!-- No <!DOCTYPE>/<html>/<head>/<body>: this fragment is meant to be "
        "embedded by tooling that supplies its own document shell; the title "
        "and style below are hoisted into the implied head. -->"
    )
    parts.append(f"<title>{esc(title)}</title>")
    parts.append(f"<style>{STYLE}</style>")

    parts.append(render_header(campaign_name, prov))

    parts.append("<h2>Phase breakdown</h2>")
    parts.append(render_phase_breakdown(probe_rows))

    parts.append("<h2>Throughput vs threads</h2>")
    parts.append(render_throughput_chart(probe_rows))

    parts.append("<h2>Thread scaling (criterion)</h2>")
    parts.append(render_thread_scaling(scaling))

    parts.append("<h2>Criterion microbenchmarks</h2>")
    parts.append(render_criterion_table(snapshot, compare_snapshot))

    parts.append("<h2>Bandwidth ceiling</h2>")
    parts.append(render_bandwidth(bandwidth_sections))

    parts.append(render_footer(consumed, missing, malformed))

    return "\n".join(parts) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="perf-viz.py",
        description=(
            "Render one benchmark campaign's data files into a self-contained "
            "HTML report with inline-SVG charts (no cargo/benchmarks are run)."
        ),
    )
    parser.add_argument("prefix", help="Path prefix, e.g. benchmarks/results/2026-08-30-host/campaign-name")
    parser.add_argument(
        "--compare", metavar="OLD_SNAPSHOT.json", default=None,
        help="Old criterion snapshot JSON to diff the current one against.",
    )
    args = parser.parse_args(argv)

    prefix = Path(args.prefix)
    compare_path = Path(args.compare) if args.compare else None

    try:
        report = build_report(prefix, compare_path)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1

    out_path = prefix.with_name(prefix.name + "-report.html")
    out_path.write_text(report)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
