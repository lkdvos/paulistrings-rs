// Talk-specific helpers on top of theme/organic.typ and theme/extensions.typ.
// Kept small on purpose: this is not a general slide framework, just the few
// things this deck's 37 conceptual slides need in common. See sections/*.typ
// for content and slides.json for the id/section/build manifest this mirrors.

#import "theme/organic.typ": *
#import "theme/extensions.typ": *

// ── Build selection ──────────────────────────────────────────────────────────
// mode is "prototype" (the six polished concepts, for fast iteration and the
// P1-P6 visual-vocabulary review) or "full" (all 37 conceptual slides).
// Prototype coverage: P3 and P4 each polish two adjacent conceptual ids, so
// the six concepts touch eight stable ids in total (handoff, "Six prototype
// concepts": "may produce more than six PDF pages" — the same is true of
// which ids are drawn, not only how many pages a given id emits).
#let prototype-ids = (
  "intro-title", // P1
  "pauli-encoding", // P2
  "bit-flip", "two-sorted-streams", // P3
  "rotation-bucket-pairs", "coset-closure", // P4
  "baseline-performance", // P5
  "querakit-architecture", // P6
)

#let active(id, mode) = mode == "full" or prototype-ids.contains(id)

// ── Progressive builds ───────────────────────────────────────────────────────
// A conceptual id keeps one identity across n PDF pages. `make-stage` is
// called once per stage with the 1-indexed stage number and must return a
// full page (i.e. call a theme slide layout, which already wraps in `stage`).
// Object positions must stay fixed across stages: use `reveal` inside
// `make-stage` rather than changing a layout between calls.
#let progressive-slide(id, mode, n-stages, make-stage) = {
  if active(id, mode) {
    for s in range(1, n-stages + 1) {
      make-stage(s)
    }
  }
}

#let slide(id, mode, make) = progressive-slide(id, mode, 1, _ => make())

// Show `body` only once `cond` holds; otherwise reserve its layout space with
// `hide` so later stages don't reflow earlier ones (handoff: "Reserve the
// space of unrevealed objects where necessary so content does not jump").
#let reveal(cond, body) = if cond { body } else { hide(body) }

// ── Recurring two-panel performance figure ──────────────────────────────────
// The one evolving figure carried from slide 10 through 28 (outline,
// "Recurring performance figure"). Every appearance is the same two frames at
// the same size — only which variants are revealed changes — so the figure
// never has to be re-authored or rescaled once real data lands; only the
// `variants-through` argument changes per call site. See figure-contract.json
// for the frozen bounding boxes, typography and variant styling this mirrors.

// (label, stroke, dash) — grayscale- and colorblind-safe: neutral ramp step,
// distinct dash pattern and endpoint marker per variant, never color alone.
#let figure-variants = (
  (id: "baseline", label: "Baseline (external reference)", color: neutral-600, dash: none),
  (id: "kernel", label: "Kernel-optimized baseline", color: neutral-800, dash: none),
  (id: "threading-attempt", label: "Attempted threading", color: neutral-500, dash: "dotted"),
  (id: "bucketed-1t", label: "Bucketed, 1 thread", color: accent-600, dash: none),
  (id: "bucketed-mt", label: "Bucketed, multithread", color: accent-800, dash: none),
  (id: "distributed", label: "Distributed (multi-node)", color: accent2-750, dash: "densely-dotted"),
)

// One axis frame: hairline box, kicker-size axis labels, a pending-data
// watermark. `variants-through` is a count into `figure-variants` — the
// legend accumulates exactly like the real curves will.
#let figure-panel(title, xlabel, ylabel, w, h, variants-through: 0) = block(
  width: w, height: h, radius: radius-md, stroke: (paint: divider, thickness: 1pt), inset: 24pt,
)[
  #set text(font: font-body, size: t-kicker, fill: ink-55)
  #stack(
    dir: ttb, spacing: 12pt,
    text(font: font-heading, weight: heading-weight, size: t-small, fill: ink, title),
    v(0pt),
    box(width: 100%, height: h - 150pt, stroke: (bottom: 1pt + divider, left: 1pt + divider))[
      #place(center + horizon, text(fill: ink-55, style: "italic", "Benchmark data pending"))
    ],
    text(ylabel + " " + sym.arrow.t + "  ·  " + xlabel + " " + sym.arrow.r),
    ..figure-variants.slice(0, variants-through).map(v => stack(
      dir: ltr, spacing: 10pt,
      line(length: 28pt, stroke: (paint: v.color, thickness: 3pt, dash: v.dash)),
      text(fill: ink-70, v.label),
    )),
  )
]

// ── Categorical color language ───────────────────────────────────────────────
// Two-way group contrast (bit-flip, two-sorted-streams, whole-bucket-movement):
// a calm neutral for "the other side" and one solid, unambiguously-terracotta
// swatch for the flipped/highlighted side — no pale accent-100/accent2-100
// washes, which read as pink/mint rather than as this system's warm terracotta
// and sage. Text color follows fill lightness so contrast always holds.
#let pair-a-fill = neutral-200
#let pair-a-text = ink
#let pair-b-fill = accent-500
#let pair-b-text = bg

// Eight-way categorical palette (rotation-bucket-pairs' full 4-bit partition):
// alternating terracotta/sage ramp steps at increasing saturation, so hue
// alone never has to carry more than one step of the same color. Labels
// (the binary strings themselves) remain the primary identifier; color is
// support, per figure-contract.json's "never color alone" rule.
#let coset-palette = (
  (fill: accent-300, text: ink),
  (fill: accent2-300, text: ink),
  (fill: accent-500, text: bg),
  (fill: accent2-600, text: bg),
  (fill: accent-400, text: ink),
  (fill: accent2-400, text: ink),
  (fill: accent-700, text: bg),
  (fill: accent2-750, text: bg),
)

// ── Binary chips ─────────────────────────────────────────────────────────────
// Monospace, fixed-width pill for an n-bit binary label — used wherever the
// bucket story needs to stay legible as bit patterns rather than decimal
// integers (handoff: "show a list of bit-integers in binary").
#let bin(v, bits) = {
  let s = ""
  let n = v
  for _ in range(bits) {
    s = str(calc.rem(n, 2)) + s
    n = calc.div-euclid(n, 2)
  }
  s
}

// No calc.xor in this Typst version — bit-by-bit over `bits` positions.
#let xor-int(a, b, bits) = {
  let result = 0
  let pow = 1
  for _ in range(bits) {
    let abit = calc.rem(calc.div-euclid(a, pow), 2)
    let bbit = calc.rem(calc.div-euclid(b, pow), 2)
    result = result + calc.rem(abit + bbit, 2) * pow
    pow = pow * 2
  }
  result
}

#let bit-chip(v, bits, fill: pair-a-fill, tcol: pair-a-text) = box(
  fill: fill, radius: 999pt, inset: (x: 16pt, y: 8pt),
  text(font: "Source Code Pro", size: t-small, fill: tcol, features: ("tnum",), bin(v, bits)),
)
#let bit-chip-row(values, bits, fill: pair-a-fill, tcol: pair-a-text) = stack(
  dir: ltr, spacing: 14pt, ..values.map(v => bit-chip(v, bits, fill: fill, tcol: tcol)),
)

// ── Small diagrams ───────────────────────────────────────────────────────────
// Hand-authored with native primitives (line/circle/place) per the handoff:
// "mathematical diagrams must be authored as precise editable code, not
// generated bitmap illustrations." Each is a fixed-size canvas so it drops
// into a slide the same way at every stage.

// A small illustrative circuit: three qubit wires, a few single-qubit gates,
// one two-qubit link, and the observable at the right end the Heisenberg
// picture propagates backward from. Schematic, not the benchmark's actual
// gate sequence (that is kicked-ising-benchmark's job, with real parameters).
#let circuit-diagram(width: 1000pt, height: 280pt) = {
  let wires = (50pt, 140pt, 230pt)
  let span = width - 200pt
  let gate(x, y, label) = place(top + left, dx: x - 26pt, dy: y - 26pt, block(
    width: 52pt, height: 52pt, radius: radius-md, fill: neutral-200,
  )[#align(center + horizon, text(font: font-heading, weight: heading-weight, size: t-small, fill: ink, label))])
  box(width: width, height: height)[
    #for y in wires {
      place(top + left, dy: y, line(end: (span, 0pt), stroke: 1.5pt + divider))
    }
    #gate(100pt, wires.at(0), "R")
    #gate(100pt, wires.at(2), "R")
    #gate(240pt, wires.at(1), "R")
    #gate(380pt, wires.at(0), "R")
    #let lx = 480pt
    #place(top + left, dx: lx, dy: wires.at(1), line(end: (0pt, wires.at(2) - wires.at(1)), stroke: 1.5pt + ink))
    #place(top + left, dx: lx - 7pt, dy: wires.at(1) - 7pt, circle(radius: 7pt, fill: ink))
    #place(top + left, dx: lx - 7pt, dy: wires.at(2) - 7pt, circle(radius: 7pt, fill: ink))
    #gate(600pt, wires.at(2), "R")
    #place(top + left, dx: span + 40pt, dy: wires.at(1) - 32pt, block(
      width: 64pt, height: 64pt, radius: 999pt, fill: accent-500,
    )[#align(center + horizon, text(font: font-heading, weight: heading-weight, size: t-body, fill: bg, "O"))])
    #place(top + left, dx: 40pt, dy: 0pt, text(size: t-kicker, fill: accent-700)[#sym.arrow.l propagate backward])
  ]
}

// A small heavy-hex-style patch: hexagon edges plus a flag qubit at every
// edge midpoint (the actual distinguishing feature of IBM's heavy-hex lattice
// versus a plain hex lattice) — a schematic patch, not the full device map.
#let hex-lattice-diagram(cells, R: 52pt) = {
  let deg = calc.pi / 180
  let hex-center(q, r) = (R * calc.sqrt(3) * (q + r / 2), R * 1.5 * r)
  let hex-vertices(cx, cy) = range(6).map(k => {
    let ang = (60 * k + 30) * deg
    (cx + R * calc.cos(ang), cy + R * calc.sin(ang))
  })
  let centers = cells.map(c => hex-center(c.at(0), c.at(1)))
  let xs = centers.map(c => hex-vertices(c.at(0), c.at(1)).map(v => v.at(0))).flatten()
  let ys = centers.map(c => hex-vertices(c.at(0), c.at(1)).map(v => v.at(1))).flatten()
  let minx = calc.min(..xs) - R
  let miny = calc.min(..ys) - R
  let maxx = calc.max(..xs) + R
  let maxy = calc.max(..ys) + R
  box(width: maxx - minx, height: maxy - miny)[
    #for c in cells {
      let (cx, cy) = hex-center(c.at(0), c.at(1))
      let verts = hex-vertices(cx, cy)
      for i in range(6) {
        let a = verts.at(i)
        let b = verts.at(calc.rem(i + 1, 6))
        place(top + left, dx: a.at(0) - minx, dy: a.at(1) - miny, line(
          end: (b.at(0) - a.at(0), b.at(1) - a.at(1)), stroke: 1.5pt + divider,
        ))
        let mx = (a.at(0) + b.at(0)) / 2 - minx
        let my = (a.at(1) + b.at(1)) / 2 - miny
        place(top + left, dx: mx - 5pt, dy: my - 5pt, circle(radius: 5pt, fill: accent2-600))
      }
      for v in verts {
        place(top + left, dx: v.at(0) - minx - 9pt, dy: v.at(1) - miny - 9pt, circle(radius: 9pt, fill: accent-600))
      }
    }
  ]
}
#let heavy-hex-flower = ((0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))

// Clifford (no branching) beside a Pauli rotation's commute/anticommute split.
#let split-diagram() = box(width: 1500pt, height: 260pt)[
  #let node(x, y, w, label, fill: neutral-200, tcol: ink) = place(top + left, dx: x - w / 2, dy: y - 25pt, block(
    width: w, height: 50pt, radius: radius-md, fill: fill,
  )[#align(center + horizon, text(font: "Source Code Pro", size: t-small, fill: tcol, label))])
  #let link(x0, y0, x1, y1) = place(top + left, dx: x0, dy: y0, line(end: (x1 - x0, y1 - y0), stroke: 1.5pt + ink-55))
  // Clifford, left half
  #node(190pt, 40pt, 70pt, "p")
  #node(190pt, 200pt, 70pt, "p")
  #link(190pt, 65pt, 190pt, 175pt)
  #place(top + left, dx: 230pt, dy: 100pt, text(size: t-kicker, fill: ink-55, "up to sign"))
  #place(top + left, dx: 60pt, dy: 230pt, text(font: font-heading, weight: heading-weight, size: t-small, fill: ink, "Clifford"))
  // divider
  #place(top + left, dx: 560pt, dy: 10pt, line(end: (0pt, 240pt), stroke: 1pt + divider))
  // Pauli rotation, right half
  #node(900pt, 40pt, 70pt, "p")
  #node(760pt, 200pt, 90pt, "p")
  #node(1060pt, 200pt, 190pt, [p #sym.plus.o g], fill: accent-500, tcol: bg)
  #link(900pt, 65pt, 760pt, 175pt)
  #link(900pt, 65pt, 1060pt, 175pt)
  #place(top + left, dx: 650pt, dy: 100pt, text(size: t-kicker, fill: ink-55, "commute"))
  #place(top + left, dx: 950pt, dy: 100pt, text(size: t-kicker, fill: accent-700, "anticommute"))
  #place(top + left, dx: 660pt, dy: 230pt, text(font: font-heading, weight: heading-weight, size: t-small, fill: ink, "Pauli rotation"))
]

// Sorted-storage merge vs. hash-map collision, side by side.
#let sum-structure-diagram() = grid(columns: (1fr, 1fr), column-gutter: 96pt,
  box(width: 100%, height: 260pt)[
    #let row(labels, y) = place(top + left, dx: 0pt, dy: y, stack(
      dir: ltr, spacing: 10pt,
      ..labels.map(l => box(fill: neutral-200, radius: 8pt, inset: (x: 16pt, y: 10pt),
        text(font: "Source Code Pro", size: t-kicker, l))),
    ))
    #row(("a", "b", "c", "c", "d"), 30pt)
    #place(top + left, dx: 190pt, dy: 80pt, text(size: 18pt, fill: accent-700, sym.arrow.b))
    #row(("a", "b", "c", "d"), 140pt)
    #place(top + left, dx: 0pt, dy: 210pt, text(size: t-kicker, fill: ink-55, "adjacent equal labels -> linear merge"))
  ],
  box(width: 100%, height: 260pt)[
    #let bucket(x, label, items) = {
      place(top + left, dx: x, dy: 30pt, box(
        width: 90pt, height: 40pt, radius: radius-md, stroke: 1pt + divider,
      )[#align(center + horizon, text(size: t-kicker, fill: ink-55, label))])
      for (i, it) in items.enumerate() {
        place(top + left, dx: x + 10pt, dy: 85pt + i * 44pt, box(
          fill: if items.len() > 1 { accent-500 } else { neutral-200 },
          radius: 8pt, inset: (x: 14pt, y: 8pt),
        )[#text(font: "Source Code Pro", size: t-kicker, fill: if items.len() > 1 { bg } else { ink }, it)])
      }
    }
    #bucket(0pt, "0", ("a",))
    #bucket(120pt, "1", ("b", "e"))
    #bucket(240pt, "2", ())
    #bucket(360pt, "3", ("d",))
    #place(top + left, dx: 0pt, dy: 210pt, text(size: t-kicker, fill: ink-55, "collision -> lookup + update per contribution"))
  ],
)

// Appearance schedule from the outline's "Figure appearances" table: which
// slide reveals which legend row. Shared across sections/02-memory.typ and
// sections/04-results.typ so the two files can't drift on the figure's state.
#let fig-stage = (
  baseline-performance: 1, // baseline
  kernel-optimization: 2, // + kernel
  threading-obstacle: 3, // + attempted threading
  memory-budget: 3, // annotate only, no new curve
  performance-bucketed: 4, // + bucketed 1 thread
  bucketing-vs-threading: 5, // + bucketed multithread
  distributed-and-hashing: 6, // + distributed
)

// Stage content area is 1632pt wide (1920pt stage minus 144pt side margins);
// 750 + 750 + 96pt gutter = 1596pt, leaving a small margin either side. See
// figure-contract.json for the frozen bounding box this mirrors exactly.
#let recurring-figure(variants-through: 0) = grid(
  columns: (750pt, 750pt), column-gutter: 96pt,
  figure-panel(
    "Processing efficiency", "strings entering the update", "updates / s (log)",
    750pt, 420pt, variants-through: variants-through,
  ),
  figure-panel(
    "Full calculation", "truncation tolerance (decreasing, log)", "wall time / s (log)",
    750pt, 420pt, variants-through: variants-through,
  ),
)
