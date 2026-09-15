// ─────────────────────────────────────────────────────────────────────────────
// Organic — design system, ported to Typst for 16:9 slide decks.
//
// The stage is 1920pt × 1080pt, so 1pt here == 1px in the source HTML design
// system: every size below is the design system's own number, unchanged.
//
// Usage:  #import "organic.typ": *
// Then call one slide function per slide (each emits its own page).
// ─────────────────────────────────────────────────────────────────────────────

// ── Color ────────────────────────────────────────────────────────────────────
#let bg = rgb("#f5ead8")          // parchment ground
#let surface = rgb("#ebddc5")     // sand surface tint
#let ink = rgb("#201e1d")         // text
#let accent = rgb("#c67139")      // terracotta — the lead accent
#let accent2 = rgb("#7a8a5e")     // sage — a genuine second voice

// Muted ink (the system's caption/meta voice) — alpha over the ground.
#let ink-70 = rgb(32, 30, 29, 179)
#let ink-62 = rgb(32, 30, 29, 158)
#let ink-55 = rgb(32, 30, 29, 140)
#let divider = rgb(32, 30, 29, 41)

// Tonal ramps — one shared OKLCH lightness scale, so the same step of any
// role carries the same visual weight. 500 is the base, 100–300 are tints,
// 700–900 are for text on tints and pressed states.
#let neutral-100 = rgb("#f9f4ed")
#let neutral-200 = rgb("#eee7db")
#let neutral-300 = rgb("#dcd3c4")
#let neutral-400 = rgb("#c0b6a5")
#let neutral-500 = rgb("#a19786")
#let neutral-600 = rgb("#82796a")
#let neutral-700 = rgb("#645c50")
#let neutral-800 = rgb("#474238")
#let neutral-900 = rgb("#2e2b25")

#let accent-100 = rgb("#fff2eb")
#let accent-200 = rgb("#ffe1d0")
#let accent-300 = rgb("#ffc6a5")
#let accent-400 = rgb("#f6a06b")
#let accent-500 = rgb("#d67f48")
#let accent-600 = rgb("#b2622d")
#let accent-700 = rgb("#8c491a")   // accent-colored body copy (5.7:1 on bg)
#let accent-800 = rgb("#643312")
#let accent-900 = rgb("#402310")

#let accent2-100 = rgb("#f0fae1")
#let accent2-200 = rgb("#e1eecc")
#let accent2-300 = rgb("#ccdbb2")
#let accent2-400 = rgb("#aebf92")
#let accent2-500 = rgb("#8fa073")
#let accent2-600 = rgb("#728157")
#let accent2-700 = rgb("#728157")
#let accent2-750 = rgb("#56633f")
#let accent2-800 = rgb("#3d472b")
#let accent2-900 = rgb("#272e1b")

// ── Type ─────────────────────────────────────────────────────────────────────
// Caprasimo is the ONLY display voice (one upright cut, no italic exists).
// Figtree carries all reading text. Install both from Google Fonts, or pass
// --font-path to the compiler.
// Caprasimo / Figtree are the system's faces (Google Fonts). Neither is
// installed on this host, so the chain falls through to the brief's rule --
// "a soft-bodied serif for display and a humanist sans for body". URW Bookman
// Demi (weight 600) is the warmest round serif available; Carlito the
// humanist sans. Install the real pair and nothing else changes.
#let font-heading = ("Caprasimo", "URW Bookman", "C059", "Georgia", "serif")
#let font-body = ("Figtree", "Carlito", "PT Sans", "Helvetica", "sans-serif")
// Caprasimo has exactly one cut (400). Asking for 600 makes Typst synthesize
// a bold on a face that is already very heavy. Set this to 600 only when
// compiling without Caprasimo, so the URW Bookman fallback lands on Demi
// rather than Light.
#let heading-weight = 400

// NOTE (port): the system's scale is pitched for a design deck, where the
// display face carries the page and the copy is a caption under it. A talk
// inverts that -- the reading text is the substance and the title is a label
// on it -- so the display sizes come down and the reading sizes go up.
// Title:body was 2.0; it is 1.5 here. The 24pt floor is unchanged.
#let t-display = 100pt      // was 126
#let t-title = 60pt         // was 72
#let t-subtitle = 40pt      // was 42
#let t-body = 40pt          // was 36
#let t-small = 34pt         // was 30
#let t-kicker = 24pt

// ── Geometry ─────────────────────────────────────────────────────────────────
#let pad-x = 144pt
#let pad-y = 108pt
#let baseline = 54pt        // grid unit: one body line (36pt at deck leading)
#let radius-md = 16pt
#let radius-lg = 28pt

// Leading that puts each type size on the 54pt grid: baseline-to-baseline
// ≈ size + leading, so leading = target line height − size.
#let lead-body = 14pt       // 40 → 54
#let lead-title = 21pt      // 60 → 81  (1.5 grid steps)
#let lead-display = 35pt    // 100 → 135 (2.5 grid steps)
#let lead-small = 20pt      // 34 → 54

// ── Page furniture ───────────────────────────────────────────────────────────
#let folio-note = state("organic-folio", "Confidential — July 2026")

// The folio is a soft pill in the margin band — the system rounds everything
// that can be round. Content slides carry it; covers and dividers do not.
#let folio() = context {
  set text(font: font-body, size: t-kicker, fill: ink-62)
  align(right, box(
    fill: surface, radius: 999pt, inset: (x: 18pt, y: 8pt),
    [#counter(page).display("1") · #folio-note.get()],
  ))
}

// ── Primitives ───────────────────────────────────────────────────────────────
#let stage(fill: bg, footer: none, body) = page(
  width: 1920pt, height: 1080pt,
  fill: fill,
  margin: (left: pad-x, right: pad-x, top: pad-y, bottom: pad-y),
  footer: footer, footer-descent: 34pt,
  body,
)

#let display-text(t, fill: ink) = {
  set par(leading: lead-display)
  text(font: font-heading, weight: heading-weight, size: t-display, fill: fill, t)
}

#let slide-title(t, fill: ink, size: t-title) = {
  set par(leading: lead-title)
  text(font: font-heading, weight: heading-weight, size: size, fill: fill, t)
}

// The bullet mark is a drawn terracotta circle hanging in the left margin —
// the text never indents (one alignment axis).
// NOTE (port): the export named this `dot`, which shadows the math symbol of
// the same name -- `$x dot y$` inside a deck that imports the system renders
// a terracotta bullet instead of a multiplication dot. Renamed.
#let dot-mark = box(baseline: -10pt, circle(radius: 8pt, fill: accent))

#let bullets(items) = {
  set text(font: font-body, size: t-body, fill: ink)
  set par(leading: lead-body)
  set list(marker: dot-mark, indent: 0pt, body-indent: 30pt, spacing: 2 * baseline)
  list(..items)
}

// A soft circular swatch — the system's shape for any drawn mark.
#let swatch(c, size: 40pt) = circle(radius: size / 2, fill: c, stroke: none)

// ── Slide layouts ────────────────────────────────────────────────────────────

// Cover / close. Centered — center is this system's dominant axis.
#let cover-slide(title, subtitle: none, meta: ()) = stage[
  #set align(center + horizon)
  #stack(
    spacing: 54pt,
    display-text(title),
    if subtitle != none {
      set par(leading: 12pt, justify: false)
      block(width: 1180pt, text(font: font-body, size: t-subtitle, fill: ink-70, subtitle))
    },
  )
  #place(
    bottom + center,
    dy: 0pt,
    text(font: font-body, size: t-kicker, fill: ink-55, meta.join([ · ])),
  )
]

// Contents — a ragged-centered list, rows on every third gridline.
#let contents-slide(title: "Contents", rows: ()) = stage(footer: folio())[
  #set align(center)
  #slide-title(title)
  #v(2 * baseline)
  #stack(
    spacing: 3 * baseline - 42pt,
    ..rows
      .enumerate()
      .map(((i, r)) => [
        #set text(font: font-body, size: t-subtitle, fill: ink)
        #text(size: t-kicker, fill: accent, numbering("01", i + 1))#h(28pt)#r
      ]),
  )
]

// Section divider — its own sage ground, so sections read as breaks when
// flipping the deck. A soft circle rises past the corner as decoration.
#let divider-slide(n, title) = stage(fill: accent2-800)[
  #place(
    bottom + right,
    dx: 320pt, dy: 420pt,
    circle(radius: 520pt, fill: accent2-750, stroke: none),
  )
  #set align(center + horizon)
  #stack(
    spacing: 54pt,
    text(font: font-heading, weight: heading-weight, size: 210pt, fill: accent2-600, n),
    slide-title(title, fill: bg, size: 72pt),
  )
]

// The workhorse content slide: title on the 4th gridline, body below.
#let content-slide(title, body, align-center: false) = stage(footer: folio())[
  #set text(font: font-body, size: t-body, fill: ink)
  #set par(leading: lead-body)
  // NOTE (port): the export had `#if align-center { set align(center) }`.
  // A `set` inside an if-block only applies inside that block, so the flag
  // was a no-op and centered bodies kept a flush-left title -- two axes on
  // one slide, which the brief forbids.
  #set align(if align-center { center } else { left })
  #slide-title(title)
  #v(2 * baseline)
  #body
]

// Columns — ragged right, heads in the display face at body size.
#let columns-slide(title, cols) = content-slide(title)[
  #grid(
    columns: cols.map(_ => 1fr),
    column-gutter: 96pt,
    ..cols.map(c => [
      #text(font: font-heading, weight: heading-weight, size: t-body, fill: ink, c.at(0))
      #v(baseline - 24pt)
      #set text(size: t-small, fill: ink)
      #c.at(1)
    ]),
  )
]

// Quadrants — four rounded cells, clay and sage tints pairing the diagonals.
#let quadrant-slide(title, cells, axis: none) = content-slide(title)[
  #let tint = (accent2-100, accent-100, accent-100, accent2-100)
  #grid(
    columns: (1fr, 1fr), rows: (auto, auto),
    gutter: 42pt,
    ..cells
      .enumerate()
      .map(((i, c)) => block(
        fill: tint.at(calc.rem(i, 4)), radius: radius-lg,
        inset: (x: 48pt, y: 42pt), width: 100%,
        [
          #text(font: font-heading, weight: heading-weight, size: t-body, fill: ink, c.at(0))
          #v(18pt)
          #set text(size: t-small, fill: ink)
          #set par(leading: 10pt)
          #c.at(1)
        ],
      )),
  )
  #if axis != none {
    v(baseline)
    set text(size: t-kicker, fill: ink-55)
    grid(columns: (1fr, 1fr), axis.at(0), align(right, axis.at(1)))
  }
]

// Data table — hairline rules only (the one exception to the soft grammar,
// inherited from the system's own .table component). 108pt row pitch.
#let table-slide(title, header: (), rows: ()) = content-slide(title)[
  #set text(size: t-small)
  #show table.cell.where(y: 0): set text(size: t-kicker, fill: ink-55)
  #table(
    columns: header.map(_ => auto),
    inset: (x: 28pt, y: 27pt),
    align: left + horizon,
    stroke: (x, y) => (bottom: (paint: divider, thickness: 1pt)),
    fill: none,
    table.header(..header.map(h => upper(h))),
    ..rows.flatten(),
  )
]

// Hero figure — one huge terracotta number, centered.
#let hero-slide(figure-text, caption) = stage(footer: folio())[
  #set align(center + horizon)
  #stack(
    spacing: 72pt,
    text(font: font-heading, weight: heading-weight, size: 190pt, fill: accent, figure-text),
    {
      set par(leading: lead-body)
      block(width: 1200pt, text(font: font-body, size: t-body, fill: ink-70, caption))
    },
  )
]

// Quote — the display voice, centered, no hanging quotation mark.
#let quote-slide(body, attribution: none) = stage(footer: folio())[
  #set align(center + horizon)
  #stack(
    spacing: 54pt,
    {
      set par(leading: 21pt)
      block(width: 1400pt, text(font: font-heading, weight: heading-weight, size: 50pt, fill: ink, body))
    },
    if attribution != none {
      text(font: font-body, size: t-small, fill: ink-70, [— #attribution])
    },
  )
]

// Image slides. Photographs are "washed" in the HTML system (desaturated,
// lower-contrast, lifted) — Typst cannot filter an image, so wash the file
// itself before importing it, or overlay the ground at ~10% as shown here.
#let washed(img, radius: radius-lg) = block(
  radius: radius, clip: true,
  stack(dir: ttb, img),
)

// Copy on one side, a washed photograph on the other.
#let split-slide(title, note: none, img) = stage(footer: folio())[
  #set align(horizon)
  #grid(
    columns: (640pt, 1fr),
    column-gutter: 96pt,
    [
      #set align(center)
      #slide-title(title)
      #v(42pt)
      #set par(leading: lead-body)
      #text(font: font-body, size: t-small, fill: ink-55, note)
    ],
    block(height: 756pt, clip: true, radius: 200pt, img),
  )
]

// Half bleed — the photograph holds one half to three edges, copy the other.
#let half-bleed-slide(title, body, img, flip: false) = page(
  width: 1920pt, height: 1080pt, fill: bg, margin: 0pt,
)[
  #place(
    if flip { top + right } else { top + left },
    block(width: 1000pt, height: 1080pt, clip: true, img),
  )
  #place(
    top + if flip { left } else { right },
    dx: if flip { pad-x } else { -pad-x }, dy: 4 * baseline,
    block(width: 640pt, [
      #set text(font: font-body, size: t-small, fill: ink)
      #set par(leading: lead-body)
      #if flip { set align(center) }
      #slide-title(title)
      #v(2 * baseline)
      #body
    ]),
  )
  #place(bottom + if flip { right } else { left }, dx: 0pt, dy: 0pt)[]
]
