# Change summary and known limitations

Scaffolding pass against `quera-typst-scaffolding-agent-handoff.md`. Read this before treating
either build as finished.

## What's real

- Both builds compile cleanly: `build/prototype.pdf` (17 pages, the 6 prototype concepts) and
  `build/full.pdf` (46 physical pages across the 37 conceptual slides — the difference is
  progressive builds, see `slides.json`'s `totals` block for the exact page arithmetic).
- All four required fonts (Caprasimo, Figtree, STIX Two Math, Source Code Pro) are present on
  this build host and copied into `fonts/` with their licenses; `scripts/check-fonts.sh` verifies
  this on every build rather than assuming it.
- The theme is the archive's already-fixed version (`dot-mark` rename, scoped `align-center`,
  corrected cover-subtitle width) — copied unmodified, not re-derived.
- The six prototypes were visually inspected at the rendered PDF level (not just "it compiles"):
  checked for clipping, correct font substitution, fixed object positions across build stages, and
  correct arithmetic in the bit-flip/coset examples.
- `figure-contract.json`'s panel dimensions (750×420pt per panel, 96pt gutter) were derived by
  actually rendering the first attempt at 900×620pt, finding it bled off the 1632pt content width
  invisibly (see "what to watch for" below), and fixing it — not guessed from the brief's numbers
  alone.

## Deviations from the type scale in BRIEF.md

None beyond what the archive's own `theme/organic.typ` already documents (the "NOTE (port)" type
scale change: display 126→100, title 72→60, body 36→40, small 30→34 — a deliberate talk-vs-deck
inversion, already present in the supplied theme, not introduced here).

## What to watch for (a real bug class, not a style note)

Typst does not error when a fixed-size `stage()` page (1920×1080pt) overflows — it silently
starts a continuation page. Early in this pass, several slides (the first draft of
`pauli-encoding`, `two-sorted-streams`, `coset-closure`, `baseline-performance`,
`end-of-gate-truncation`, `profiling-results`, `memory-budget`, `sort-merge-work-unit`) did
exactly this, most from ordinary Typst paragraph spacing compounding across several stacked
`text()`/equation blocks rather than any one long paragraph. Root cause each time was default
`par(spacing:)` (≈1.2× body size, applied above *and* below every top-level text/equation block)
adding up across several stacked elements, plus the recurring-figure's first pass being
authored 264pt too wide for the 1632pt content area — content pushed that far off-stage is
present in the PDF text layer but invisible in a rendered raster, which is the more dangerous
failure mode (it doesn't look broken at a glance).

Fixed by: `#set par(spacing: 0pt)` at the top of any slide body stacking more than ~2 text/equation
elements, explicit `v()` calls for the intended rhythm instead, and (for `coset-closure`)
shrinking the pseudocode block from `t-small` to `t-kicker`. **Before adding dense content to any
slide, render it and check the actual PDF page count matches the declared build count** —
`slides.json`'s `totals` block is the reference; `pdfinfo build/*.pdf | grep Pages` is the check.

## Explicitly not done here (owned by another agent, or genuinely blocked)

- **No benchmark data, no figures.** `figures/` holds only a manifest and a contract; every
  `recurring-figure(...)` call renders the same-size "Benchmark data pending" placeholder.
  `scripts/check-figures.sh --release` fails on this by design — that failure is expected while
  data is pending, not a defect.
- **No cluster access, no Slurm submission.** Nothing here runs `sbatch` or touches the actual
  measurement campaign; that is explicitly the benchmark agent's scope per the handoff.
- **~28 `[slot: ...]` markers remain** across `sections/*.typ` (bracketed facts only Lukas can
  supply: circuit parameters, project names, measured numbers, current affiliation, interview
  date). `scripts/check-figures.sh` reports the count; grep `sections/*.typ` for `\[slot:` to find
  them all.
- **`thread-scaling-tuning` and `profiling-results`** use a plain placeholder block (no
  `recurring-figure`) since they're not part of the recurring two-panel figure's appearance
  schedule — they'll take a one-off chart/profile crop image instead, sized to fit the block
  currently reserving their space.
- **28b hash-engineering split**: the outline flags this as a possible second build once real
  data/rehearsal timing is known. Currently `distributed-and-hashing` is one build carrying both
  the distributed capacity story and the hash-communication evidence; `slides.json` notes this
  explicitly so it isn't mistaken for an oversight.

## Font provenance caveat

Caprasimo and Figtree's `LICENSE.txt` files reproduce the standard OFL 1.1 text with the
copyright line read from each project's current `google/fonts` `METADATA.pb` — this project does
not pin an exact commit/hash for either font. STIX Two Math and Source Code Pro's licenses are
extracted verbatim from the installed Rocky Linux package (version numbers in `fonts/FONTS.md`),
which is exact and reproducible. Re-verify the Google Fonts pair before a real interview delivery
if that matters for the audience.

## If picking this back up

1. Read `README.md` for the build commands and editing model.
2. Read `notes.md` for speaker cues and the rehearsal budget.
3. Grep `sections/*.typ` for `\[slot:` and fill in what you can from your own knowledge — the rest
   needs actual benchmark results.
4. Once `figures/*.svg` land per `figure-contract.json`, flip `figures/figure-manifest.json`
   statuses and swap the placeholder `recurring-figure(...)` calls for real `image(...)` calls at
   the same authored size.
