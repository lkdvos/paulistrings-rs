# Figure manifest — deck v2 restyle (2026-09-14)

Every figure below is styling/assembly over already-real, already-accepted campaign data.
No number was invented, altered, or re-derived; every point cites the `raw/` run record(s)
and the `evidence.md`/`decisions.md` entry it comes from. Nothing here touches the in-flight
Slurm jobs 7033945/7033946/7034671 or any Slurm command.

## Deck sizing and theme (applies to every figure below)

- Deck stage: 960x540 PDF points. Full-figure variant target box: **900x340pt**. Compact
  variant: **half-width (450x340pt)** for the wide single/dual-line plots (thread scaling,
  hash-cut, consistency), or **half-height (900x170pt)** for the two-panel/annotated plots
  (baseline, ranks-capacity) where halving the width would crowd two panels or long
  annotation text. Both choices are per-figure judgment calls, not a fixed rule — check
  against the real slide before trusting them.
- Export sizing: `export_deck_figure()` (`figures/make_compact_figures.py`) sets
  `fig.set_size_inches(width_pt/72, height_pt/72)` before saving. Since matplotlib's font-size
  unit and the PDF point are the same 1/72in unit, a figure exported this way and placed on
  the deck **unscaled** reproduces its matplotlib font sizes as identical PDF points — this is
  why every figure below uses one absolute font size (18pt body, ~13-14pt tick/legend) for
  BOTH the full and compact variant, rather than scaling text up for the smaller box: both
  variants are meant to sit, unscaled, in their own correctly-sized box on the same 960pt-wide
  stage, so the same matplotlib point size already lands correctly on each. If a variant is
  later placed at a size other than the box it was exported for, text will scale with it like
  any other vector image — re-export at the actual target size instead of trusting a stretch.
- PNG preview DPI: 200 (SVG/PDF are vector, unaffected). Pixel sizes below are DPI x inches.
- Colors: white background (`savefig(transparent=True)`), body text/spines/ticks
  `#1c2954` (dark navy), gridlines `#d7dbe6` (light gray-blue), `axes.facecolor="none"`.
- Fonts: **DejaVu Sans** (matplotlib's own bundled sans-serif) for all text — checked via
  `matplotlib.font_manager.fontManager.ttflist` in this environment; neither Founders Grotesk
  nor Tiempos Headline is installed here, and no font was extracted from the deck PDF (that
  would copy a proprietary font). This is an honestly-labeled fallback, not a claim of the
  real deck typeface. Math text uses matplotlib's built-in `mathtext.fontset="stix"`, a real
  license-clean STIX approximation needing no external font file.
- Series are distinguished by BOTH color and marker/linestyle (never color alone): the
  `_DECK_SERIES` palette in `figures/make_compact_figures.py` pairs each color with a
  distinct marker shape and line dash pattern.
- No in-figure titles duplicating the deck's own slide title; where a short caption is useful
  it is passed via the `title=` kwarg at a small size, never baked in by default.

## Figures

### 1. baseline (deck page 16 — "Recurring two-panel baseline plus external libraries")

Two-panel figure: left = efficiency (input-string throughput vs. strings entering a gate),
right = wall time vs. coefficient tolerance, from `make_recurring_figure(..., stage=1,
highlight_variant="naive_baseline")` — this IS the existing two-panel baseline function
already used for `recurring_stage6.png`; no new plotting function was written for this view.

- Files: `baseline_v2.{svg,pdf,png}` (900x340pt, 2500x944px @200dpi),
  `baseline_v2_compact.{svg,pdf,png}` (900x170pt half-height, 2500x472px @200dpi).
- Left (efficiency) panel: **honestly empty**, with an explicit "no per-gate efficiency data
  for this variant" annotation — `naive_baseline` predates the engine's per-gate stats
  plumbing (same disclosed gap as `recurring_stage6.png`, `evidence.md`'s E4/E5 row).
- Right (cost) panel, internal-baseline point: `naive_baseline`, `min_abs_coeff=1e-6`,
  `wall_time_s=2.9561e-05`, `final_terms=14`, `peak_rss_kb=426248` — job 7033716, toy scale
  (`n_qubits=12`, 2 Trotter steps). Source: `raw/2026-09-14-worker7165-historical/runs.jsonl`,
  `evidence.md`'s E4/E5-historical-baseline row, `decisions.md` #32. Same record already
  plotted (parchment theme) in `figures/real/recurring_stage6.png`.
- Right panel, external-library reference points (drawn as stars, NOT part of the same
  tolerance sweep — different circuit scale, disclosed via each point's annotation):
  - `bucketed_current` (Rust, this engine), `min_abs_coeff=2^-16 (1.5258789e-05)`,
    `wall_time_s=1629.9046` (one of 3 repetitions of job 7030090, range 1629.9-1675.6s),
    `n_qubits=127` canonical, `threads=1`. Source: `raw/2026-09-13-worker7277/runs.jsonl`.
  - `PauliPropagation.jl` (external, dict backend), same `min_abs_coeff=2^-16`,
    `wall_time_s=4874.939709082`, `n_qubits=127` canonical, `threads=1`, job 7033031.
    Source: `raw/2026-09-14-worker7169-julia/runs.jsonl`, `decisions.md` #27.
  - Both share `final_terms=38,791,220` (cross-engine correctness check).
  - **Limitation, stated explicitly**: there is no real per-gate efficiency data for the
    Julia leg (`decisions.md` #10: "the efficiency-panel per-gate series for
    PauliPropagation.jl stays marked missing"), so the external reference appears only on
    the cost panel, never the efficiency panel.

### 2. threads (deck page 31 — Rust-only reveal)

`make_thread_scaling_figure(rows, theme="deck")`, restyled only — same function used for the
existing `thread_scaling.png`. Rust-only: the Julia overlay (`other_rows=`) is deliberately
NOT included, since that data is mid-flight in job 7034671.

- Files: `thread_scaling_v2.{svg,pdf,png}` (900x340pt, 2500x944px), `thread_scaling_v2_compact.*`
  (450x340pt half-width, 1250x944px).
- Data: `normalize.thread_scaling()` over `bucketed_current`, `min_abs_coeff=2^-16
  (1.5258789e-05)`, single-node (`ranks=1`) runs only, from
  `raw/2026-09-13-worker7277/runs.jsonl` (threads=1, 96),
  `raw/2026-09-13-worker7207/runs.jsonl` (threads=2,4,8,16,32,48),
  `raw/2026-09-13-worker7206/runs.jsonl` (threads=64). All genoa, job 7030090/7031793/7032105.
  `raw/2026-09-13-distributed-4ranks/runs.jsonl`'s 4-rank/48-threads-per-rank point was
  deliberately EXCLUDED from this load: `thread_scaling()` has no ranks filter, and mixing a
  multi-node topology into a single-node scaling curve at the same nominal thread count would
  silently conflate two different regimes.

### 3. ranks-capacity (deck page 32 — new figure, `make_distributed_capacity_figure`)

No figure previously existed for this asset. New function
`make_distributed_capacity_figure(rows, theme=..., figsize_pt=..., title=...)` in
`figures/make_compact_figures.py`: wall time vs. ranks, one line per `min_abs_coeff`,
non-completed rows drawn as distinct sentinel markers at an axes-fraction y (never a
fabricated wall time — enforced by a `ValueError` if a non-completed row carries one).

- Files: `distributed_capacity_v2.{svg,pdf,png}` (900x340pt, 2500x944px),
  `distributed_capacity_v2_compact.*` (900x170pt half-height, 2500x472px).
- eps=2^-16 (1.5258789e-05), **overlap** check: ranks=1, `wall_time_s=41.628`,
  `final_terms=38,791,220` (job 7033147); ranks=4, `wall_time_s=69.542`,
  `peak_terms=45,418,768`, `peak_rss_kb=14,377,788` (job 7031873).
  Source: `raw/2026-09-13-distributed-4ranks/runs.jsonl`, `raw/2026-09-14-worker7169/runs.jsonl`.
- eps=2^-18 (3.8146973e-06), **E6 overlap-consistency** (`evidence.md` E6, `decisions.md` #19):
  ranks=1 (single-node, 96 threads), `wall_time_s=482.571` (job 7031790, one of 3 reps,
  range 470.9-482.6s); ranks=4, `wall_time_s=468.772`, `peak_terms=635,371,364`,
  `peak_rss_kb=199,972,704` (job 7032059) — matches the single-node run's `final_terms`
  exactly, confirming no correctness loss going distributed at this scale.
  Source: `raw/2026-09-13-worker7327/runs.jsonl`, `raw/2026-09-13-distributed-4ranks/runs.jsonl`.
- eps=2^-20 (9.5367432e-07), **E7 beyond-single-node capacity** (`evidence.md` E7,
  `decisions.md` #22-23): ranks=16, `wall_time_s=1658.816`, `peak_terms=8,923,556,570`,
  `peak_rss_kb=3,041,410,140` (≈3.04TB decimal, job 7032351/16-rank sbatch).
  Source: `raw/2026-09-13-distributed-16ranks/runs.jsonl`.
  - **ranks=1 at this eps is marked "untested"**, not measured or capacity-extended: job
    7031792 ("eps20-t96") was submitted but never reached a final status in
    `job-ledger.jsonl` — there is no completed, failed, or timed-out record for it.
  - **ranks=8 at this eps is marked "oom"** — job 7032060, a REAL measured OOM crash, not an
    inferred boundary: `sacct`/`seff` show `MaxRSS=1257805740K` (~1.20 TiB) on the single
    worst rank vs. `AveRSS=365811870K` (~349 GiB) average, 343% imbalance, root-caused in
    `decisions.md` #22 to the random partition-row draw, not a code bug. No `wall_time_s` is
    plotted for this row (it never completed) — enforced by the figure function's own guard.
  - **The "genoa node's 1.5TB" capacity-boundary line quoted in the eps=2^-20/ranks=16
    annotation is INFERRED from the node's nominal installed RAM (an infrastructure spec:
    CLAUDE.md's node table), NOT a measured single-node OOM in this campaign's data.** The
    one *measured* OOM in this dataset is the ranks=8 point above, which is a different
    memory allocation (8-way distributed, ~2.73TB requested across ranks per `seff`) from the
    single-node 1.5TB figure the E7 narrative compares the 16-rank total against. Both facts
    are stated explicitly in the figure's own point annotations, not just here.
  - No genuine multi-point curve exists for eps=2^-20 alone (only one completed point, one
    OOM, one untested) — per the task's own allowance, this is reported as the honest,
    minimal comparison the real data supports, not padded with invented intermediate points.

### 4. hash-cut (existing `hash_communication.png`)

`make_hash_communication_vs_cutoff_figure(rows, theme="deck")`, restyled only — same
cutoff-sweep data already plotted (parchment theme) in the current `hash_communication.png`.

- Files: `hash_communication_v2.{svg,pdf,png}` (900x340pt, 2500x944px),
  `hash_communication_v2_compact.*` (450x340pt half-width, 1250x944px).
- Data: `normalize.hash_communication()` over `raw/2026-09-14-worker7162-e8/runs.jsonl` +
  `gates.rank-0.jsonl`, job 7033663 (`evidence.md` E8, `decisions.md` #36): random vs. cut
  partition-row policy, `total_rows_exported` across `min_abs_coeff` in
  {2^-12, 2^-14, 2^-16, 2^-18}. Real ratios per decision #36: 10.7x, 10.8x, 11.1x, 11.2x
  (cut's advantage holds, and grows slightly, across 4 orders of magnitude).

### 5. consistency (existing `convergence.png`)

`make_convergence_figure(rows, julia_rows=..., theme="deck")`, restyled only — same
trajectory + single-endpoint Julia overlay already plotted (parchment theme) in the current
`convergence.png`. Job 7033946's fuller Julia trajectory is mid-flight and NOT included here.

- Files: `convergence_v2.{svg,pdf,png}` (900x340pt, 2500x944px), `convergence_v2_compact.*`
  (450x340pt half-width, 1250x944px).
- Rust data: `raw/2026-09-14-worker7172-convergence/convergence.jsonl`, job 7033650
  (`evidence.md` E9, `decisions.md` #33): `<O>` at every Trotter step 1-20, for
  `min_abs_coeff` in {2^-12, 2^-14, 2^-16, 2^-18}, 96 threads, n_qubits=127 canonical.
- Julia overlay: the one real single-endpoint record, `min_abs_coeff=2^-16`, `trotter_step=20`
  (assigned per `decisions.md` #27/#33's convention — the record carries no `trotter_step` of
  its own), `expectation_re=0.3971653299846826`, from
  `raw/2026-09-14-worker7169-julia/runs.jsonl` (job 7033031 — the SAME run record used as the
  "PauliPropagation.jl" external reference point in the baseline figure above; it serves both
  evidence slots). Agrees with the Rust trajectory's own step-20 endpoint at that cutoff to
  `abs_delta=1.67e-16` per `decisions.md` #27.

## Test coverage

`quera-talk-data/campaign-2026-09-11/figures/tests/test_make_figures.py` gained:
`test_thread_scaling_deck_theme_uses_navy_spines_and_exact_size`,
`test_hash_communication_vs_cutoff_deck_theme_distinguishes_series_by_marker_too`,
`test_convergence_deck_theme_still_builds_and_omits_default_legacy_title`,
`test_recurring_figure_deck_theme_with_external_points_draws_stars`,
`test_distributed_capacity_figure_empty_input_raises_clear_error`,
`test_distributed_capacity_figure_rejects_fabricated_runtime_on_failed_row`,
`test_distributed_capacity_figure_marks_completed_oom_and_untested_distinctly`,
`test_distributed_capacity_figure_deck_theme_exports_at_exact_size`.

Full suite: `pytest quera-talk-data/campaign-2026-09-11/figures/tests/` — 33 passed (25
pre-existing + 8 new; a `make_attempts_figure`/`attempts.*` figure and its tests present in
this working tree belong to concurrent, unrelated work in this shared worktree, not this task).
