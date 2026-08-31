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
    bandwidth.txt (same directory)     memory-bandwidth ceiling sections; an optional
                                        leading "# ceiling-map: ..." header (see
                                        scripts/bandwidth.sh) picks which section is the
                                        roofline ceiling per thread count, else a
                                        hard-coded ccqlin038-shaped table is used

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

# The one-bar-per-cell layout (v0.4 redesign) splits each row into a serial
# family (the calling thread, drawn muted grey-blue) and a parallel family
# (the coset-loop workers, drawn saturated) plus a hatched "idle" segment.
# Every name below maps to exactly one color everywhere it appears in the
# report — extending, not replacing, the phase->color contract.
SERIAL_PHASES = [p for p in WALL_PHASES if p != "coset_loop"]
OTHER_SERIAL = "other (serial)"
BUSY_SUB = ["gather", "sort", "merge"]
OTHER_BUSY = "other busy"
IDLE = "idle (imbalance)"

# Serial family: one hue (blue-grey), stepped dark->light — these segments
# are almost always tiny slivers (see OTHER_SERIAL merge rule below), so
# mutual distinguishability matters less than reading as "not the busy work".
_SERIAL_COLORS = {
    "rebucket": "#3f5c73",
    "prepare": "#4f6d87",
    "rescale": "#5f7992",
    "span_plan": "#6f88a0",
    "permute": "#7d93a8",
    "unpermute": "#86a0b3",
    "recount": "#93a7b8",
    "finalize": "#9fb0bd",
    "fallback": "#acb9c2",
    OTHER_SERIAL: "#c7ced4",
}
# Parallel family: saturated categorical hues (blue/orange/aqua/violet),
# validated CVD-safe in this adjacent order via the dataviz skill's
# validate_palette.js (worst adjacent CVD deltaE 9.2, normal-vision 27.6).
_BUSY_COLORS = {
    "gather": "#2a78d6",
    "sort": "#eb6834",
    "merge": "#1baf7a",
    OTHER_BUSY: "#4a3aa7",
}
IDLE_BASE = "#e3e1db"
IDLE_STRIPE = "#c3c2b7"

PHASE_COLOR = {}
PHASE_COLOR.update(_SERIAL_COLORS)
PHASE_COLOR.update(_BUSY_COLORS)
PHASE_COLOR[IDLE] = IDLE_BASE

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
    return dedupe_probe_rows(rows)


def dedupe_probe_rows(rows: list) -> list:
    """Keep only the LAST line for each (layer, threads) key.

    ``<prefix>-probe.json`` may contain multiple lines for the same
    (layer, threads) pair from appended re-runs (e.g. a campaign resumed
    after a partial run). Newer lines carry an extra ``rows_gathered``
    field; older lines lack it, which ``dict.get`` already treats as
    ``None`` — no special-casing needed here beyond picking the last line.
    Order is preserved: rows are emitted in the position of each key's
    last occurrence, so unrelated (layer, threads) groups keep their
    original relative order in the file.
    """
    last_idx: dict = {}
    for i, row in enumerate(rows):
        key = (row.get("layer"), row.get("threads"))
        last_idx[key] = i
    keep = sorted(last_idx.values())
    return [rows[i] for i in keep]


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


def parse_ceiling_map(line: str) -> Optional[dict]:
    """Parse a ``# ceiling-map: <key>=<label>;...`` header line (emitted as
    the first line of bandwidth.txt by scripts/bandwidth.sh) into
    ``{threads_int: section_label, ..., "default": section_label}``.

    Keys are thread counts or the literal ``default``; labels must match a
    ``=== <label> ===`` section name elsewhere in the same file (not
    validated here — an unmatched label just means that ceiling stays
    unavailable, handled by ``triad_ceiling_gbps``). Unparseable entries
    (bad int, no ``=``, empty label) are skipped rather than raising, so a
    partially-malformed header still contributes whatever it can. Returns
    ``None`` if the line isn't a ceiling-map header at all, or nothing
    usable was parsed from it.
    """
    m = re.match(r"^#\s*ceiling-map:\s*(.+)$", line.strip())
    if not m:
        return None
    result: dict = {}
    for entry in m.group(1).split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        key, _, label = entry.partition("=")
        key = key.strip()
        label = label.strip()
        if not label:
            continue
        if key == "default":
            result["default"] = label
        else:
            try:
                result[int(key)] = label
            except ValueError:
                continue  # unknown/malformed key: ignore, keep the rest
    return result or None


def load_bandwidth(dir_path: Path, missing: list) -> tuple:
    """Parse bandwidth.txt into (sections, ceiling_map).

    ``sections`` is a list of (section_label, {kernel: best_gbps}).
    ``ceiling_map`` is the parsed ``# ceiling-map:`` header (see
    ``parse_ceiling_map``) or ``None`` when the file has no such header —
    callers fall back to the hard-coded ``_BW_SECTION_BY_THREADS`` /
    ``_BW_SECTION_DEFAULT`` in that case, so older campaign dirs render the
    same as before this map became data-driven.
    """
    path = dir_path / "bandwidth.txt"
    if not path.exists():
        missing.append("bandwidth.txt")
        return [], None
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        missing.append("bandwidth.txt")
        return [], None

    ceiling_map = parse_ceiling_map(lines[0]) if lines else None

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
    return sections, ceiling_map


# --------------------------------------------------------------------------
# DRAM traffic model (% of bandwidth ceiling)
# --------------------------------------------------------------------------

# Fallback bandwidth.txt section chosen by thread count, used only when the
# file has no ``# ceiling-map:`` header (see parse_ceiling_map) — i.e. older
# campaign dirs, or a bandwidth.sh predating that header. Matched against the
# "===" labels load_bandwidth() parses; this hard-codes ccqlin038's topology,
# which is exactly the hidden coupling the header line exists to remove.
_BW_SECTION_BY_THREADS = [
    (1, "1 core, node0 local"),
    (8, "node0, 8 physical"),
    (16, "both sockets, 16 physical"),
]
_BW_SECTION_DEFAULT = "both sockets, 32 threads"


def bandwidth_section_label(threads: int, ceiling_map: Optional[dict] = None) -> str:
    """Which bandwidth.txt section is the roofline ceiling for ``threads``.

    When ``ceiling_map`` is given (parsed from a bandwidth.txt
    ``# ceiling-map:`` header — see ``parse_ceiling_map``), it entirely
    replaces the hard-coded table below: numeric keys are matched the same
    way (smallest key >= threads), falling back to its own ``"default"``
    entry. Absent a map (old campaign dirs, or bandwidth.txt with no
    header), the hard-coded ccqlin038-shaped table is used, unchanged.
    """
    if ceiling_map:
        numeric_keys = sorted(k for k in ceiling_map if isinstance(k, int))
        for max_t in numeric_keys:
            if threads <= max_t:
                return ceiling_map[max_t]
        return ceiling_map.get("default", _BW_SECTION_DEFAULT)
    for max_t, label in _BW_SECTION_BY_THREADS:
        if threads <= max_t:
            return label
    return _BW_SECTION_DEFAULT


def triad_ceiling_gbps(bandwidth_sections: list, threads: int, ceiling_map: Optional[dict] = None):
    """Return (best_gbps, section_label) for the TRIAD kernel at this thread
    count, or (None, section_label) if bandwidth.txt is absent, missing that
    section, the section has no triad kernel line, or (defensively) the
    triad line reads zero — all of those mean "no usable ceiling", not a
    crash."""
    label = bandwidth_section_label(threads, ceiling_map)
    for sec_label, kmap in bandwidth_sections:
        if sec_label == label:
            triad = kmap.get("triad")
            return (triad if triad else None), label
    return None, label


def dram_traffic_bytes_per_term(qubits: Optional[int]) -> Optional[int]:
    """T = 16*ceil(qubits/64) + 16: key words (2 x [u64; W]) + a 16-byte
    Complex64 coefficient, W = ceil(qubits/64)."""
    if not qubits or qubits <= 0:
        return None
    w = math.ceil(qubits / 64)
    return 16 * w + 16


def dram_metric(row: dict, bandwidth_sections: list, ceiling_map: Optional[dict] = None) -> Optional[dict]:
    """Model DRAM traffic per layer for one probe row (Change 3).

    Returns None if the metric cannot be modeled at all (no qubits/wall_ns).
    Otherwise returns a dict with ``gbps`` (float or None), ``pct`` (float or
    None, requires a matched bandwidth.txt section), and ``title`` — the
    formula with this cell's numbers substituted, for a hover tooltip.
    ``gbps`` is None when the byte model itself is inapplicable (neither the
    coset-loop nor the in-place-rescale traffic shape matches this row).
    """
    qubits = row.get("qubits")
    wall_ns = row.get("wall_ns", 0) or 0
    layers = row.get("layers", 1) or 1
    if not qubits or wall_ns <= 0 or layers <= 0:
        return None

    T = dram_traffic_bytes_per_term(qubits)
    terms_in = row.get("terms_in", 0) or 0
    terms_out = row.get("terms_out", 0) or 0
    rows_gathered = row.get("rows_gathered")  # None on pre-v0.4 probe lines
    coset_loop_ns = row.get("coset_loop_ns", 0) or 0
    rescale_ns = row.get("rescale_ns", 0) or 0
    threads = row.get("threads", 1) or 1

    if coset_loop_ns > 0 and rows_gathered is not None and rows_gathered > 0:
        rows_sorted = row.get("rows_sorted")
        rows_id = row.get("rows_id")  # None before v0.6 G1d
        if rows_sorted is not None:
            # v0.5 model: no tag byte; a gathered row is written by gather and
            # read by merge (2T); the sorted subset is additionally read and
            # rewritten by the sort (2T more). From v0.6 G1d (`rows_id`
            # present), a dense identity row materializes only its 16-byte
            # coefficient — its keys are borrowed from the source bucket in
            # place (modeled coset-cache-resident) — so those rows are priced
            # at 2×16 instead of 2×T.
            id_borrowed = rows_id or 0
            bytes_per_layer = (
                (terms_in / layers) * T
                + 2 * ((rows_gathered - id_borrowed) / layers) * T
                + 2 * (id_borrowed / layers) * 16
                + 2 * (rows_sorted / layers) * T
                + (terms_out / layers) * T
            )
            id_note = (
                f" + 2×({id_borrowed}/{layers})×16 [coeff-only id rows, keys borrowed]"
                if rows_id is not None
                else ""
            )
            formula = (
                f"({terms_in}/{layers})×{T} [gather in] + "
                f"2×({rows_gathered - id_borrowed}/{layers})×{T} [gather w + merge r]"
                f"{id_note} + "
                f"2×({rows_sorted}/{layers})×{T} [sort r/w] + "
                f"({terms_out}/{layers})×{T} [merge out] = {bytes_per_layer:,.0f} B/layer"
            )
        else:
            # pre-v0.5 probe lines: tag byte + every row through the sort.
            bytes_per_layer = (
                (terms_in / layers) * T
                + 4 * (rows_gathered / layers) * (T + 1)
                + (terms_out / layers) * T
            )
            formula = (
                f"({terms_in}/{layers})×{T} [gather in] + "
                f"4×({rows_gathered}/{layers})×({T}+1) [tag r/w + sort r/w] + "
                f"({terms_out}/{layers})×{T} [merge out] = {bytes_per_layer:,.0f} B/layer"
            )
    elif rescale_ns > 0 and coset_loop_ns == 0:
        bytes_per_layer = 2 * (terms_in / layers) * T
        formula = f"2×({terms_in}/{layers})×{T} [in-place r/w] = {bytes_per_layer:,.0f} B/layer"
    else:
        return {"gbps": None, "pct": None, "ceiling": None, "section": None, "title": None}

    gbps = bytes_per_layer * layers / (wall_ns / 1e9) / 1e9
    ceiling, section = triad_ceiling_gbps(bandwidth_sections, threads, ceiling_map)
    # A ceiling of exactly 0 is nonsensical (and would divide-by-zero below);
    # treat it the same as "no ceiling found" rather than propagating a
    # falsy-but-not-None value that would otherwise dodge the `is not None`
    # checks downstream and format a None pct.
    if not ceiling or ceiling <= 0:
        ceiling = None
    pct = (gbps / ceiling * 100.0) if ceiling else None

    title = f"T = 16×ceil({qubits}/64)+16 = {T} B/term. bytes/layer = {formula}. "
    title += f"GB/s = bytes×layers/(wall_ns/1e9)/1e9 = {gbps:.2f} GB/s."
    if ceiling is not None and pct is not None:
        title += f" Ceiling: '{section}' triad best_gbps = {ceiling:.2f} GB/s -> {pct:.1f}% of ceiling."
        if pct > 100.0:
            title += (
                " Over 100% means most of the modeled traffic is served from cache,"
                " not DRAM (the per-coset working set fits in L2/L3): this phase is"
                " not DRAM-bound."
            )
    else:
        title += f" No bandwidth.txt section '{section}' found — ceiling unavailable."

    return {"gbps": gbps, "pct": pct, "ceiling": ceiling, "section": section, "title": title}


def meter_bar_html(pct: Optional[float]) -> str:
    """A thin 0-100% inline meter; clamps visually past 100% with a distinct
    'over-full' color (the bar stays at full width, only the fill color
    changes) since some layers genuinely exceed the modeled ceiling."""
    if pct is None:
        return ""
    clamped = max(0.0, min(pct, 100.0))
    cls = "meter-fill-over" if pct > 100.0 else "meter-fill"
    return (
        '<span class="dram-meter">'
        f'<span class="{cls}" style="width:{clamped:.1f}%"></span>'
        "</span>"
    )


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
    parts.append('<div class="legend-group"><span class="legend-title">serial (calling thread)</span>')
    for name in SERIAL_PHASES + [OTHER_SERIAL]:
        parts.append(
            f'<span class="legend-item"><span class="swatch" '
            f'style="background:{PHASE_COLOR[name]}"></span>{esc(name)}</span>'
        )
    parts.append("</div>")
    parts.append(
        '<div class="legend-group"><span class="legend-title">'
        "parallel region (per-worker average)</span>"
    )
    for name in BUSY_SUB + [OTHER_BUSY]:
        parts.append(
            f'<span class="legend-item"><span class="swatch" '
            f'style="background:{PHASE_COLOR[name]}"></span>{esc(name)}</span>'
        )
    parts.append(
        '<span class="legend-item"><span class="swatch swatch-idle"></span>'
        f"{esc(IDLE)}</span>"
    )
    parts.append("</div>")
    parts.append("</div>")
    parts.append(
        '<p class="note phase-legend-note">Each bar is one layer’s wall time '
        "<em>× thread count</em> (CPU time), so under perfect scaling every row in "
        "a group is the same length and growth relative to the 1t bar is scaling "
        "loss — and the high-thread breakdowns stay readable. The right-hand label "
        "is the actual wall ms/layer. The parallel region is subdivided by what its "
        "worker threads spent time on (averaged over threads); ‘idle’ is load "
        "imbalance.</p>"
    )
    return "\n".join(parts)


def _text_color_for(bg_hex: str) -> str:
    """White text on dark segment fills, near-black text on light ones."""
    h = bg_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if luminance < 140 else "#1a1a1a"


def _slug(*parts) -> str:
    s = "-".join(str(p) for p in parts)
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s)


def _segment_rect(
    x: float, w: float, height: float, fill: str, title: str, name: str, per_ms: Optional[float] = None
) -> list:
    """One colored segment: a <rect> with a hover <title>, plus an inline
    text label ("name" or, if there's more room, "name X ms") once the
    segment is wide enough to hold it (~48px)."""
    parts = [
        f'<rect x="{x:.2f}" y="0" width="{w:.2f}" height="{height:.2f}" fill="{fill}">'
        f"<title>{esc(title)}</title></rect>"
    ]
    label = None
    if w >= 90 and per_ms is not None:
        label = f"{name} {per_ms:.2f} ms"
    elif w >= 48:
        label = name
    if label:
        text_fill = "#1a1a1a" if fill.startswith("url(") else _text_color_for(fill)
        parts.append(
            svg_text(x + w / 2, height / 2 + 4, label, size=10, anchor="middle", fill=text_fill)
        )
    return parts


def _phase_cell_svg(row: dict, group_max_cpu_ms: float, uid: str) -> str:
    """One cell's bar: length ∝ wall ms/layer × thread count (CPU time),
    scaled to the group's max CPU ms/layer, segmented into serial phases then
    the coset-loop's busy/idle breakdown. CPU-time scaling keeps rows at high
    thread counts readable (a wall-time scale collapses the 32t row to a
    sliver) and makes growth vs the 1t bar read directly as scaling loss.
    See CLAUDE.md / the v0.4 perf-viz redesign for the segment layout."""
    track_w = 460.0
    label_room = 84.0
    height = 24.0
    width = track_w + label_room

    wall_ns = row.get("wall_ns", 0) or 0
    layers = row.get("layers", 1) or 1
    threads_n = row.get("threads", 1) or 1
    wall_ms = (wall_ns / 1e6 / layers) if layers else 0.0
    cpu_ms = wall_ms * threads_n

    bar_frac = 0.0
    if group_max_cpu_ms > 0 and wall_ns > 0:
        bar_frac = max(0.0, min(1.0, cpu_ms / group_max_cpu_ms))
    bar_w = bar_frac * track_w

    parts = [svg_open(width, height)]
    # faint background rail spanning the group's full scale
    parts.append(f'<rect x="0" y="0" width="{track_w:.1f}" height="{height:.1f}" fill="#f2f2f2"/>')

    if wall_ns <= 0 or bar_w <= 0:
        parts.append(svg_text(4, height / 2 + 4, "n/a", size=10, fill=MUTED_COLOR))
        parts.append("</svg>")
        return "\n".join(parts)

    x = 0.0

    # --- 1. serial phases, in fixed order; sub-0.4%-of-wall ones merge ---
    big = []
    small = []
    for name in SERIAL_PHASES:
        v = row.get(f"{name}_ns", 0) or 0
        frac = v / wall_ns
        per_ms = (v / 1e6) / layers if layers else 0.0
        (big if frac >= 0.004 else small).append((name, v, per_ms, frac))

    for name, v, per_ms, frac in big:
        seg_w = frac * bar_w
        title = f"{name}: {per_ms:.3f} ms/layer ({frac * 100:.2f}% of wall)"
        parts.extend(_segment_rect(x, seg_w, height, PHASE_COLOR[name], title, name, per_ms))
        x += seg_w

    small_total_ns = sum(v for _, v, _, _ in small)
    if small_total_ns > 0:
        frac = small_total_ns / wall_ns
        seg_w = frac * bar_w
        per_ms_total = (small_total_ns / 1e6) / layers if layers else 0.0
        detail = ", ".join(f"{n} {m:.3f} ms" for n, _, m, _ in small if m > 0)
        title = (
            f"{OTHER_SERIAL}: {per_ms_total:.3f} ms/layer ({frac * 100:.2f}% of wall)"
            + (f" — includes {detail} (each < 0.4% of wall)" if detail else "")
        )
        parts.extend(
            _segment_rect(x, seg_w, height, PHASE_COLOR[OTHER_SERIAL], title, OTHER_SERIAL, per_ms_total)
        )
        x += seg_w

    # --- 2. the coset-loop portion, subdivided by what workers were doing ---
    coset_loop_ns = row.get("coset_loop_ns", 0) or 0
    if coset_loop_ns > 0:
        coset_frac = coset_loop_ns / wall_ns
        coset_w = coset_frac * bar_w

        other_busy_ns = sum((row.get(f"{n}_ns", 0) or 0) for n in BUSY_PHASES if n not in BUSY_SUB)
        busy_raw = {n: (row.get(f"{n}_ns", 0) or 0) for n in BUSY_SUB}
        busy_raw[OTHER_BUSY] = other_busy_ns
        busy_total_ns = sum(busy_raw.values())

        cx = x
        for name in BUSY_SUB + [OTHER_BUSY]:
            avg_ns = busy_raw[name] / threads_n
            sub_frac = (avg_ns / coset_loop_ns) if coset_loop_ns else 0.0
            sub_w = sub_frac * coset_w
            per_ms = (avg_ns / 1e6) / layers if layers else 0.0
            wall_frac = avg_ns / wall_ns if wall_ns else 0.0
            title = (
                f"{name}: {per_ms:.3f} ms/layer ({wall_frac * 100:.2f}% of wall, "
                f"per-worker average over {threads_n} thread(s))"
            )
            parts.extend(_segment_rect(cx, sub_w, height, PHASE_COLOR[name], title, name, per_ms))
            cx += sub_w

        idle_ns = max(0.0, coset_loop_ns - busy_total_ns / threads_n)
        idle_w = max(0.0, x + coset_w - cx)
        if idle_w > 0.05:
            par_eff = busy_total_ns / (coset_loop_ns * threads_n) if coset_loop_ns else None
            idle_per_ms = (idle_ns / 1e6) / layers if layers else 0.0
            idle_wall_frac = idle_ns / wall_ns if wall_ns else 0.0
            title = (
                f"{IDLE}: {idle_per_ms:.3f} ms/layer ({idle_wall_frac * 100:.2f}% of wall); "
                f"parallel efficiency = {par_eff:.2f} (busy / (coset_loop × threads))"
            )
            pattern_id = f"idlehatch-{uid}"
            parts.insert(
                1,
                f'<defs><pattern id="{pattern_id}" width="6" height="6" '
                'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="6" height="6" fill="{IDLE_BASE}"/>'
                f'<rect width="3" height="6" fill="{IDLE_STRIPE}"/>'
                "</pattern></defs>",
            )
            parts.extend(
                _segment_rect(cx, idle_w, height, f"url(#{pattern_id})", title, "idle", idle_per_ms)
            )
        x += coset_w

    # value label at the bar's right end
    parts.append(
        svg_text(x + 6, height / 2 + 4, f"{wall_ms:.2f} ms", size=10.5, fill=TEXT_COLOR, weight="600")
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_phase_breakdown(probe_rows: list, bandwidth_sections: list, ceiling_map: Optional[dict] = None) -> str:
    if not probe_rows:
        return '<p class="note">No probe sidecar (.-probe.json) found — phase breakdown unavailable.</p>'

    groups: dict = {}
    for row in probe_rows:
        layer = row.get("layer", "?")
        groups.setdefault(layer, []).append(row)

    # Precompute each row's DRAM metric once (reused below by identity, since
    # `groups` holds the same row dicts as `probe_rows`) and use the pass to
    # decide up front whether any ceiling was actually available — missing
    # bandwidth.txt, or a ceiling map/section that doesn't cover the thread
    # counts this campaign used, degrades every row to "GB/s only, no % of
    # ceiling" rather than crashing; say so once instead of 32 times over.
    metrics_by_id = {id(row): dram_metric(row, bandwidth_sections, ceiling_map) for row in probe_rows}
    any_dram = any(m is not None and m.get("gbps") is not None for m in metrics_by_id.values())
    any_ceiling = any(m is not None and m.get("ceiling") is not None for m in metrics_by_id.values())

    out = [render_legend()]
    out.append(
        '<p class="note">How to read the DRAM figure: &gt;100% of ceiling means the '
        "modeled traffic is largely served from cache, not DRAM (not bandwidth-bound); "
        "&asymp;100% means the phase is at the memory wall; a low percentage alongside "
        "high wall time means it is latency-, serial-, or imbalance-bound instead, not "
        "bandwidth-bound.</p>"
    )
    if any_dram and not any_ceiling:
        out.append(
            '<p class="note">Bandwidth ceilings unavailable for this campaign (no '
            "bandwidth.txt, or none of its sections match the thread counts used here) "
            "— DRAM figures below show modeled GB/s only, with no % of ceiling.</p>"
        )

    for layer in sorted(groups):
        rows = sorted(groups[layer], key=lambda r: r.get("threads", 0))
        out.append(f'<h3 class="layer-name">{esc(layer)}</h3>')

        # Group scale is CPU time (wall × threads), not wall time: see
        # _phase_cell_svg. The 1t row usually sets the scale; rows only
        # exceed it by their scaling loss.
        group_max_cpu_ms = 0.0
        for row in rows:
            wall_ns = row.get("wall_ns", 0) or 0
            layers = row.get("layers", 1) or 1
            threads_n = row.get("threads", 1) or 1
            if wall_ns > 0 and layers:
                group_max_cpu_ms = max(group_max_cpu_ms, wall_ns / 1e6 / layers * threads_n)

        for row in rows:
            threads = row.get("threads", "?")
            wall_ns = row.get("wall_ns", 0) or 0
            layers = row.get("layers", 1) or 1
            terms_in = row.get("terms_in", 0) or 0
            vmhwm_kb = row.get("vmhwm_kb", 0) or 0
            wall_ms_per_layer = wall_ns / 1e6 / layers if layers else 0.0
            strings_per_s = terms_in / (wall_ns / 1e9) if wall_ns > 0 else None
            vmhwm_mb = vmhwm_kb / 1024.0

            uid = _slug(layer, threads)
            bar_svg = _phase_cell_svg(row, group_max_cpu_ms, uid)

            metric = metrics_by_id[id(row)]
            if metric is None or metric.get("gbps") is None:
                dram_html = '<span class="dram-na">DRAM: —</span>'
            elif metric["ceiling"] is None:
                dram_html = (
                    f'<span title="{esc(metric["title"])}">DRAM: {metric["gbps"]:.1f} GB/s</span>'
                )
            else:
                over = " (cache-served)" if metric["pct"] > 100.0 else ""
                dram_html = (
                    f'<span class="dram-metric" title="{esc(metric["title"])}">'
                    f'DRAM: {metric["gbps"]:.1f} GB/s = {metric["pct"]:.0f}% of ceiling{over}'
                    f"{meter_bar_html(metric['pct'])}</span>"
                )

            out.append('<div class="phase-row">')
            out.append(f'<div class="phase-threads">{esc(threads)} t</div>')
            out.append(f'<div class="phase-bar">{bar_svg}</div>')
            out.append('<div class="phase-stats">')
            out.append(f"<div>wall: {wall_ms_per_layer:.3f} ms/layer</div>")
            out.append(f"<div>strings/s: {engineering(strings_per_s)}</div>")
            out.append(f"<div>{dram_html}</div>")
            out.append(f"<div>VmHWM: {vmhwm_mb:.1f} MB</div>")
            out.append("</div>")
            out.append("</div>")

    out.append(
        '<p class="note">DRAM figure is a traffic model (see '
        "<code>benchmarks/PROFILING.md</code>), not a measurement; ceiling = membench "
        "triad at a comparable core count (see the how-to-read note above for what the "
        "percentage means).</p>"
    )
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
    note = (
        '<p class="note">How to read: one line per layer, log-scale y axis '
        "(strings/sec, from probe wall time). A line that flattens or turns "
        "down past some thread count has stopped scaling there.</p>"
    )
    return "\n".join(parts) + "\n" + note


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

    out = [
        '<p class="note">How to read: solid lines are speedup relative to each '
        "placement's own 1-thread median; the dashed diagonal is ideal (linear) "
        "speedup — the gap below it at a given thread count is scaling loss.</p>"
    ]
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

    out = [
        '<p class="note">How to read: median is per-call wall time, Melem/s is '
        "throughput (higher is better). With a --compare snapshot, &Delta;% is "
        "new vs. old median: red (&gt;+5%) is a regression, green (&lt;-5%) an "
        "improvement — treat it as a prompt to look, not a pass/fail gate.</p>"
    ]
    out.append('<div class="table-wrap"><table>')
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

    out = [
        '<p class="note">How to read: each row is a placement (thread/NUMA affinity), '
        "each column a STREAM-style kernel's best measured GB/s; these are the "
        "ceilings the phase-breakdown section's DRAM% figures divide into.</p>"
    ]
    out.append('<div class="table-wrap"><table>')
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
.swatch-idle {
  width: 10px; height: 10px; display: inline-block; border-radius: 2px;
  background: repeating-linear-gradient(45deg, #e3e1db, #e3e1db 2px, #c3c2b7 2px, #c3c2b7 4px);
}
.phase-legend-note { margin-top: 0; margin-bottom: 14px; }
.phase-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 6px 0;
  border-bottom: 1px solid #f0f0f0;
  flex-wrap: wrap;
}
.phase-threads {
  width: 40px;
  font-weight: 600;
  font-size: 0.85em;
  flex-shrink: 0;
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.phase-bar { line-height: 0; flex: 0 0 auto; }
.phase-stats {
  font-size: 0.82em;
  color: #333333;
  font-variant-numeric: tabular-nums;
  display: flex;
  flex-direction: column;
  gap: 1px;
  flex: 0 0 auto;
}
.phase-stats > div { white-space: nowrap; }
.dram-na { color: #999999; }
.dram-metric { display: inline-flex; align-items: center; }
.dram-meter {
  display: inline-block;
  position: relative;
  width: 56px;
  height: 6px;
  margin-left: 6px;
  border-radius: 3px;
  background: #e2e2e2;
  overflow: hidden;
  vertical-align: middle;
}
.meter-fill, .meter-fill-over {
  position: absolute;
  left: 0; top: 0; bottom: 0;
  border-radius: 3px;
}
.meter-fill { background: #2a78d6; }
.meter-fill-over { background: #d03b3b; }
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

    bandwidth_sections, ceiling_map = load_bandwidth(dir_path, missing)
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
    parts.append(render_phase_breakdown(probe_rows, bandwidth_sections, ceiling_map))

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
