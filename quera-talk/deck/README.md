# Pauli propagation at scale — QuEra technical interview talk

Lukas Devos's technical interview talk, built on the Organic design system (`theme/`, adapted
from the attached archive — see `theme/organic.typ`'s header comments and `fonts/FONTS.md` for
the three documented port fixes this project preserves). 37 conceptual slides, 6 of them
polished as prototypes; the rest are an editable scaffold. See `CHANGES.md` for what shipped in
this pass and what's still open.

## Build

```bash
scripts/build.sh prototype   # the 6 polished concepts (P1-P6) -> build/prototype.pdf, 17 pages
scripts/build.sh full        # all 37 conceptual slides         -> build/full.pdf, 46 pages
scripts/build.sh full --strict   # full build + fail if a required figure/slot is still pending
scripts/watch.sh prototype   # live recompile on save
```

Everything resolves from this directory (`--root .`); no reference to `/presentation/...` or
anyone's home directory. Fonts come from the project-local `fonts/` via `--font-path fonts` —
`scripts/check-fonts.sh` (run automatically by `build.sh`) fails loudly if Caprasimo, Figtree,
STIX Two Math, or Source Code Pro don't resolve, so a build never silently falls back without
saying so.

Typst version used: `typst 0.14.2`. Pin this (or note the drift) if a future Typst release
changes layout behavior — nothing here depends on unstable features, but page-break behavior
around oversized content (see `CHANGES.md`) is exactly the kind of thing a version bump could
shift.

**No external Typst packages are used** — `cetz` and friends were deliberately avoided per the
handoff ("avoid optional packages when the provided theme already handles the need"); the
recurring figure's placeholder frame is drawn with native `block`/`line`/`grid` primitives.

## Layout

```
main.typ                 entry point: mode selection, section order, imports
theme/organic.typ        preserved theme (unmodified from the attached archive's fixed version)
theme/extensions.typ     preserved technical extensions (math/mono/figure/table layouts)
components.typ           talk-specific helpers: build-stage selection, reveal(), recurring-figure
sections/00-06-*.typ     one #let render(mode) per section, in outline order
figures/                 figure-contract.json's counterpart: manifest + placeholders, no real assets yet
fonts/                   portable Caprasimo/Figtree/STIX Two Math/Source Code Pro + licenses
scripts/                 build.sh, watch.sh, check-fonts.sh, check-figures.sh
slides.json              id/section/build-count/status manifest (37 conceptual ids)
notes.md                 speaker cues keyed by stable id, not page number
figure-contract.json     frozen dimensions/typography/colors for the benchmark agent
build/                   generated PDFs (gitignored if this project is committed to a repo that tracks that)
```

`main.typ` selects build mode via `--input mode=prototype|full` (default `full`) and an optional
`--input date=2026-XX-XX` for the cover's interview date slot. Sections never read `sys.inputs`
themselves — `main.typ` is the single place that decides mode and passes it down, so a section
file is easy to reason about in isolation.

## Editing

- **Content:** edit the relevant `sections/NN-*.typ`. Each slide is one `slide(id, mode, () =>
  ...)` or `progressive-slide(id, mode, n, stage => ...)` call — see `components.typ` for both.
- **Stable ids:** never rename an id in `slide(...)`/`progressive-slide(...)` without updating the
  matching row in `slides.json` and the matching heading in `notes.md` — physical page numbers
  will keep moving as builds change, but ids don't.
- **Progressive builds:** wrap content that should appear later in `reveal(stage >= k, body)`
  (from `components.typ`) — it reserves the object's layout space via `hide()` even before it's
  shown, so later stages never reflow earlier ones.
- **A slide overflowing to a second physical page** is a real bug (Typst inserts a silent
  continuation page rather than erroring) — see `CHANGES.md`'s "what to watch for" section before
  adding dense content to any slide.
- **[slot: ...]** markers (rendered in terracotta) mark a fact Lukas needs to supply — a
  benchmark parameter, a project name, a measured number. `scripts/check-figures.sh --release`
  refuses to pass while any remain, which is the intended behavior before a real delivery, not a
  bug during scaffolding.
- **Replacing a placeholder figure:** once the benchmark/data agent delivers an asset per
  `figure-contract.json`, drop it at the path `figures/figure-manifest.json` names, flip that
  entry's `status` to `"real-asset"` with a `provenance` block, and swap the corresponding
  `recurring-figure(...)` call for an `image(...)` placement at the exact contracted size — no
  slide layout should need to change.

## Fonts

See `fonts/FONTS.md` for full provenance, license files, and the fallback chain if a font is ever
unavailable on a compiling machine. Verified present on this build host and copied into
`fonts/`: Caprasimo, Figtree (Regular/SemiBold/Bold), STIX Two Math, Source Code Pro
(Regular/Bold).

## Known limitations

See `CHANGES.md`.
