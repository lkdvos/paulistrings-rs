# The talk in the Organic design system

`talk.typ` is the full deck (57 slides) — the same argument as `../talk.typ`, restructured
claim-first because this system's density demands it.

```bash
# from the repo root
typst compile --root . presentation/slides/organic/talk.typ
./.venv/bin/python presentation/plots/organic_figures.py    # all 19 figures
```

| file | what |
|---|---|
| `BRIEF.md` | the system's own brief, as shipped in `Organic.zip` |
| `organic.typ` | the shipped tokens and layouts, with three fixes (below) |
| `extensions.typ` | what a technical talk needs and a design-system deck does not |
| `talk.typ` | the deck |
| `../../plots/organic_style.py` | the system's palette and chart grammar for matplotlib |
| `../../plots/organic_figures.py` | regenerates every figure the deck uses |
| `../../figures/organic/sizes.json` | each figure's true size in stage points |

## Fonts

The system's faces are **Caprasimo** (display) and **Figtree** (body), both Google Fonts, and
both are installed under `~/.local/share/fonts/organic/`. The deck and the figures use them.

`heading-weight` in `organic.typ` is **400**, because Caprasimo has exactly one cut and asking
for 600 makes Typst synthesize a bold on a face that is already very heavy. If you compile
somewhere without Caprasimo, the chain falls through to **URW Bookman** and **Carlito** — the
brief's own rule, "a soft-bodied serif for display and a humanist sans for body" — and you want
`heading-weight: 600` there so Bookman lands on Demi rather than Light.

To install the pair on another machine:

```bash
mkdir -p ~/.local/share/fonts/organic && cd ~/.local/share/fonts/organic
curl -sH 'User-Agent: Mozilla/4.0' \
  'https://fonts.googleapis.com/css2?family=Caprasimo&family=Figtree:wght@400;600;700' \
  | grep -oE 'https://[^)]+\.ttf' | sort -u | xargs -n1 curl -sOL
fc-cache -f ~/.local/share/fonts && fc-list | grep -Ei 'caprasimo|figtree'
```

The old `User-Agent` is what makes Google serve TrueType instead of woff2, which Typst cannot
read. If the directory ends up empty, fetch them from the `google/fonts` repository instead
(`ofl/caprasimo/`, `ofl/figtree/`). Once `fc-list` shows both, recompile — no flag needed,
since fontconfig finds them; `--font-path <dir>` also works if you keep them out of the
system font path.

## Three fixes to the shipped `organic.typ`

Each is marked `NOTE (port)` at the site.

1. **`dot` shadowed the math symbol.** The bullet mark was exported as `dot`, so `$x dot y$` in
   any deck that imports the system renders a terracotta bullet instead of a multiplication
   dot. Renamed to `dot-mark`.
2. **`content-slide`'s `align-center` was a no-op.** `#if align-center { set align(center) }`
   scopes the `set` to the if-block, so centered bodies kept a flush-left title — the two-axis
   mix the brief forbids.
3. **The cover subtitle was 547pt wide** on a 1632pt stage: `38em * 0.4` resolves `em` against
   the 36pt body, not the 42pt subtitle.

`deck.typ` in the zip also does not compile: `[\##201e1d]` reads as an escaped hash followed by
a live `#`, and Typst reports `invalid number suffix: d` on three rows of the color table.

## Type scale — a deliberate deviation from the brief

The brief's scale (display 126, title 72, body 36, small 30) is pitched for a design deck,
where the display face carries the page and the copy is a caption under it. A talk inverts
that: the reading text is the substance and the title is a label on it. So the display sizes
come down and the reading sizes go up — title 60, body 40, small 34, display 100 — which takes
title:body from 2.0 to 1.5. The 54pt baseline grid is unchanged (the leadings were recomputed),
and 24pt is still the floor. Reading copy that the brief tinted at 55 % ink is at 70 %; 55 %
stays for true meta — the folio, axis labels, table headers.

Figure type moved with it: `organic_style.T_AXIS` is 28pt, about 70 % of body.

## Extensions

The system ships no math setup, no monospace voice, no chart-beside-copy layout, and a table
component that left-aligns numerals. `extensions.typ` adds them, each following the system's own
grammar — rounded containers, sand tints, one alignment axis per slide, nothing below 24pt:

`math-setup` · `code-block` / `code-slide` · `figure-slide` · `chart-copy-slide` · `stat-row` ·
`data-table` / `data-table-slide` · `columns-body`

## Figures

The stage is 1920×1080pt and Typst's pt is matplotlib's pt, so a figure authored at
`stage_figsize(1180, 520)` is 1:1 on the slide and a 24pt tick label is 24pt on the stage.
Place every image at the size in `sizes.json`; never scale one to fit — re-author it at the
width it gets. Three things the port had to sweep:

- `bbox_inches="tight"` crops, so a fixed-width placement silently rescales the type. The
  driver intercepts `savefig` and keeps the authored size.
- The figure scripts hardcode `fontsize=8..9` on legends and annotations, and matplotlib's
  default 6pt marker. Both are lifted to the system's floor.
- `docs/figures/*.py` do not use `plots/common.py`; their module-level color constants are
  rebound by hand, and `baseline_ops.LIBS` captures its colors at import time, so the tuple is
  rewritten too.

Five engine variants against a two-accent system: the rejected attempts take neutral and sage,
the shipped engine owns terracotta in two tones.
