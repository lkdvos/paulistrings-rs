// Extensions to the Organic system for a technical talk.
//
// The shipped system has no layout for a chart beside copy, no mono voice and
// no math setup -- a design-system deck does not usually carry any of the
// three. Everything here follows the system's own grammar (rounded
// containers, sand tints, one alignment axis per slide, nothing below 24pt),
// but it is an extension, not the system. See BRIEF.md.

#import "organic.typ": *

// ── Math ─────────────────────────────────────────────────────────────────────
// STIX Two Math is the warmest full math face installed here; it sits with a
// Bookman/Caprasimo display without clashing. `size: 1em` is deliberate --
// pinning a math size breaks every caption that contains a symbol.
#let math-setup(body) = {
  show math.equation: set text(font: "STIX Two Math", size: 1em)
  body
}

// ── Mono ─────────────────────────────────────────────────────────────────────
// The system has no monospace voice. Code rests in a sand pill, never on the
// bare ground, so it reads as an inset object rather than as body copy.
#let code-block(body, size: t-small) = block(
  fill: surface, radius: radius-lg, inset: (x: 42pt, y: 36pt), width: 100%,
  {
    set text(font: ("Source Code Pro", "DejaVu Sans Mono"), size: size, fill: ink)
    set par(leading: 14pt)
    body
  },
)

#let code-slide(title, code, note: none) = content-slide(title)[
  #code-block(code)
  #if note != none {
    v(baseline)
    text(font: font-body, size: t-small, fill: ink-70, note)
  }
]

// ── Figures ──────────────────────────────────────────────────────────────────
// Centered figure + centered caption: one axis, the system's dominant one.
// The image is placed at its authored size so a 24pt tick label is 24pt on
// the stage -- never scale a chart to fit, re-author it at the width it gets.
#let figure-slide(title, img, caption: none, width: none) = stage(footer: folio())[
  #set align(center)
  // Typst's default paragraph spacing (1.2em = 43pt at this body size) lands
  // between the title, the image and the caption -- three gaps that push a
  // 560pt figure off the stage. The rhythm here is explicit, on the grid.
  #set par(spacing: 0pt)
  #slide-title(title)
  #v(baseline)
  #if width != none { box(width: width, img) } else { img }
  #if caption != none {
    v(27pt)
    block(width: 1300pt, text(font: font-body, size: 30pt, fill: ink-70, caption))
  }
]

// Chart beside copy. Flush-left throughout: the brief allows the left axis
// for charts and hanging lists, and forbids mixing the two axes on one slide.
#let chart-copy-slide(title, img, copy, chart-width: 900pt) = content-slide(title)[
  #grid(
    columns: (chart-width, 1fr), column-gutter: 96pt, align: horizon,
    img,
    block({
      set text(font: font-body, size: t-body, fill: ink)
      set par(leading: lead-body)
      copy
    }),
  )
]

// ── Numbers ──────────────────────────────────────────────────────────────────
// `hero-slide` carries one number. A talk that turns on a comparison needs a
// row of them; same voice, three columns, terracotta leads and sage answers.
#let stat-row(items) = grid(
  columns: items.map(_ => 1fr), column-gutter: 42pt, align: center,
  ..items.map(it => block({
    text(font: font-heading, weight: heading-weight, size: 84pt, fill: it.at(2), it.at(0))
    v(21pt)
    set par(leading: lead-small)
    text(font: font-body, size: t-small, fill: ink-70, it.at(1))
  })),
)

// ── Tables ───────────────────────────────────────────────────────────────────
// The shipped `table-slide` left-aligns every cell, so a column of numbers
// does not line up. The brief asks for tabular figures wherever digits sit in
// columns; this variant right-aligns the numeric columns and fills the stage.
// `data-table` is the component on its own, for slides that pair it with copy.
#let data-table(header: (), rows: (), numeric-from: 2, cols: none) = {
  set text(size: t-small, features: ("tnum",))
  show table.cell.where(y: 0): set text(size: t-kicker, fill: ink-55)
  table(
    columns: if cols != none { cols } else { (auto, 1fr) + header.slice(2).map(_ => auto) },
    inset: (x: 28pt, y: 27pt),
    align: (col, _) => if col >= numeric-from { right + horizon } else { left + horizon },
    stroke: (x, y) => (bottom: (paint: divider, thickness: 1pt)),
    fill: none,
    table.header(..header.map(h => upper(h))),
    ..rows.flatten(),
  )
}

#let data-table-slide(title, header: (), rows: (), numeric-from: 2, unit: none, cols: none) = content-slide(title)[
  #data-table(header: header, rows: rows, numeric-from: numeric-from, cols: cols)
  #if unit != none {
    v(27pt)
    text(font: font-body, size: t-kicker, fill: ink-55, unit)
  }
]

// ── Columns ──────────────────────────────────────────────────────────────────
// `columns-slide` owns its title; this is the same grid on its own, for a
// slide that needs copy above or below the columns.
#let columns-body(cols) = grid(
  columns: cols.map(_ => 1fr),
  column-gutter: 96pt,
  ..cols.map(c => [
    #text(font: font-heading, weight: heading-weight, size: t-body, fill: ink, c.at(0))
    #v(baseline - 24pt)
    #set text(size: t-small, fill: ink)
    #set par(leading: lead-small)
    #c.at(1)
  ]),
)
