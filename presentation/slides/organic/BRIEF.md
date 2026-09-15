# Organic — design system brief for a Typst slide deck

Hand this whole folder to the agent. It contains everything needed to build a
deck that looks like the Organic design system: this brief, `organic.typ`
(tokens + slide layouts, ready to `#import`), `deck.typ` (a worked example),
and `assets/photo.jpg` (the reference photograph).

**Stage:** 1920pt × 1080pt (16:9). 1pt in Typst == 1px in the source HTML
system, so every number below is the system's own value, unconverted.

**Fonts:** Caprasimo (display) and Figtree (body), both Google Fonts. Install
them locally or compile with `--font-path ./fonts`. If neither is available,
fall back to a soft-bodied serif for display and a humanist sans for body —
never a condensed or geometric display face.

---

## The voice

Warm, rounded and a little playful. A cream-and-sand ground with a terracotta
lead accent and a sage second accent. Everything that can be round is round.
Photographs are washed so they sit back into the warm page instead of on top
of it. Air matters — rounded shapes need space to read as soft.

**Do**

- Over-round: 28pt radii on containers, fully pill (999pt) on small elements,
  circles for every drawn mark (bullets, timeline nodes, swatches, chart nodes,
  chart bar tops).
- Use soft circles and blobs as decoration and as image masks.
- Treat the sage as a genuine second voice — a second data series, a divider
  ground — not a highlight.
- Center. Center is this system's **dominant** axis: cover, close, contents,
  dividers, hero figures, quotes and split copy all center. Flush-left is for
  slides where centering is dishonest: bulleted lists (hanging marks need one
  left axis), tables, charts, timelines.
- One axis per slide. Never mix centered and left-aligned blocks on one slide.

**Don't**

- No sharp corners, no hairline-only geometry (the data table's rules are the
  single sanctioned exception — they belong to the system's table component).
- Don't desaturate into greys; warmth is the point.
- Don't crowd. Don't add typographic eyebrows/kickers above titles — the title
  carries the slide.
- No gradients, no drop shadows on type, no emoji.

---

## Color

| Role | Value | Notes |
| --- | --- | --- |
| Ground | `#f5ead8` | every content slide |
| Surface | `#ebddc5` | folio pill, tinted panels |
| Text | `#201e1d` | 13.9:1 on the ground |
| Terracotta (accent) | `#c67139` | 3.0:1 — chrome, large type, marks, not body copy |
| Sage (accent 2) | `#7a8a5e` | 3.1:1 — the second voice |
| Terracotta 700 | `#8c491a` | 5.7:1 — use this for accent-colored *reading* text |
| Sage 800 | `#3d472b` | the divider ground |

Each role has a 100–900 ramp on one shared perceptual lightness scale (all in
`organic.typ`). 100–300 are tints and fills, 500 is the base, 700–900 carry
text on tints. Prefer a ramp step over an ad-hoc mix.

Muted ink for captions and meta: the text color at 55% (`ink-55`), 62% for
the folio, 70% for de-emphasised values.

---

## Type

- Display (cover/close): Caprasimo 126pt, line height 135pt.
- Slide title: Caprasimo 72pt, line height 81pt.
- Subtitle: 42pt. Body: 36pt/54pt. Small (captions, column copy): 30pt.
  Kicker (folio, legends, axis labels, meta): 24pt.
- Never below 24pt anywhere on the stage.
- Headings and column heads take Caprasimo; everything readable takes Figtree.
  Opt into tabular figures (`"tnum"`) for anything in columns — tables, chart
  axes, contents numbers.

---

## Layout grid

- Margins: 144pt left/right, 108pt top/bottom.
- **54pt baseline grid**, measured from the top edge. Text sits on it:
  title baseline on the 4th gridline, first body block two gridlines below,
  list items every other gridline. Compact rhythms (table rows, chart ticks)
  use the 27pt half step.
- Pinned bottom content (meta rows, captions) lands its last baseline on the
  18th gridline (972pt).
- Table rows: 108pt pitch (two gridlines).
- Column gutter: 96pt. Quadrant gap: 42pt.

---

## Slide types (all implemented in `organic.typ`)

| Function | What it is |
| --- | --- |
| `cover-slide` | Centered display title + subtitle, meta row pinned bottom |
| `contents-slide` | Ragged-centered rows, terracotta 2-digit numbers, no rules |
| `divider-slide` | Sage 800 ground, 280pt ghost numeral in sage 600, parchment title, soft circle past the corner |
| `content-slide` | Title + any body — the workhorse |
| `bullets` | Hanging terracotta dots, flush-left, 108pt item pitch |
| `columns-slide` | 2 or 3 ragged-right columns, Caprasimo heads at body size |
| `quadrant-slide` | Four rounded tinted cells, clay/sage on the diagonals, muted axis poles |
| `table-slide` | Hairline rules, uppercase kicker header, optional circular swatches |
| `hero-slide` | One 300pt terracotta figure + centered caption |
| `quote-slide` | Caprasimo 60pt centered, em-dash attribution, no hanging quote mark |
| `split-slide` | Centered copy beside a washed photograph in a soft blob crop |
| `half-bleed-slide` | Photograph holds one half to three edges; copy in the other. `flip: true` mirrors |

Page furniture: content slides carry a folio — a surface-tinted pill at the
bottom right, `«page» · «note»` at 24pt in 62% ink. Covers and dividers don't.
Set the note with `#folio-note.update("Confidential — Month Year")`.

## Charts

Use `cetz` (or hand-placed Typst shapes) and follow the system's dataviz
grammar:

- Bars: neutral 600 by default, terracotta for the highlighted bar, **pill
  tops** (round the top corners at 28pt, keep bottoms square on the axis).
- Lines: terracotta leads solid at 4pt with round caps; sage answers **beaded**
  (a dotted round-cap rhythm) so the pair separates in grayscale too.
- Nodes: filled circles, sage for waypoints, terracotta for the current one.
- Grid and axis: 1pt hairlines in the divider color. Reference lines dashed in
  terracotta 700, labelled in the same.
- Axis and value text: Figtree 24pt in 55% ink, tabular figures.
- Captions below the figure at 27pt in muted ink, max ~82 characters.

## Imagery

Every photograph is *washed* — desaturated, lower-contrast, slightly lifted.
Typst cannot filter images, so **pre-process the file** (e.g. reduce saturation
to ~60%, contrast to ~85%, brightness ~110%) before importing, or overlay the
ground color at ~10%. Crop on-page figures in soft organic shapes (large
asymmetric radii), never a hard box. Edge-bleeding photographs stay flush to
the stage edges and curve only their *inner* edge. Type never sits on a
photograph — it rests on the parchment ground.

## Compile

```
typst compile deck.typ deck.pdf --font-path ./fonts
```
