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

### 1. baseline (deck page 16 — superseded 2026-09-14, see "baseline-pivot" below)

**Superseded.** The `naive_baseline`-sweep framing below (`make_recurring_figure(..., stage=1,
highlight_variant="naive_baseline")`) is kept here for provenance only — `decisions.md` #42
found that commit's merge phase never wires truncation into its merge at all, so every point
of a sweep comes back with the identical `final_terms`. It is truncation-inert, not a real
"before this work" baseline, and `baseline_v2*` files are no longer page 16's asset (see the
new "baseline-pivot" entry, `baseline_v3*`, immediately after this one).

- Files (kept, not regenerated): `baseline_v2.{svg,pdf,png}` (900x340pt, 2500x944px @200dpi),
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

### 1b. baseline-pivot (deck page 16 — real 3-point comparison, 2026-09-14)

New function `make_baseline_pivot_figure(rows, theme=..., figsize_pt=..., title=...)` in
`figures/make_compact_figures.py`: a plain 3-bar chart, not a sweep — the user's own framing
("I'm happy to use the Julia data (both threaded and not), along with the current engine with
a single bucket") replaces the truncation-inert `naive_baseline` story above. All three points
share the same `eps=2^-16 (1.5258789e-05)`, 127-qubit canonical config.

- Files: `baseline_v3.{svg,pdf,png}` (900x340pt, 2500x944px @200dpi), `baseline_v3_compact.*`
  (450x340pt half-width, 1250x944px @200dpi) — half-width, matching this MANIFEST's existing
  convention for simple single-panel plots (thread scaling, hash-cut), since this is one plain
  bar chart, not a two-panel/annotated view.
- Real numbers plotted:
  - Julia, 1 thread, `dict` backend: `wall_time_s=4874.939709082`, `final_terms=38,791,220`,
    job 7033031. Source: `raw/2026-09-14-worker7169-julia/runs.jsonl`.
  - Julia, 96 threads, `vector` backend: `wall_time_s=899.49`, `final_terms=38,791,220`, job
    7034021. Source: `decisions.md` #38 / `job-ledger.jsonl` (no standalone `runs.jsonl` was
    produced for this one-off point; `decisions.md` #38 is the citable record, at the 2-decimal
    precision it reports).
  - Current engine, single-thread, forced to one bucket (`min_buckets=1`): `wall_time_s=
    1967.578165213985`, `final_terms=38,791,220`, job 7036526. Source:
    `raw/2026-09-14-worker7160-single-bucket/runs.jsonl`.
  - All three share `final_terms=38,791,220`.
- **Honesty contract, load-bearing for this figure's design**: the x-axis is categorical
  (three labels), never a numeric/log thread axis. The two Julia bars ARE a directly
  comparable pair (same engine, same code, 1 vs. 96 threads, annotated `5.42x` speedup); the
  current-engine bar is a DIFFERENT implementation, back at 1 thread again — 1 -> 96 -> 1 is
  not a monotonic thread progression, and no line connects the three bars. Julia's two bars
  share one color (`_DECK_SERIES[1]`); the current-engine bar gets both a distinct color
  (`_DECK_SERIES[0]`, navy) AND a hatch pattern, so the "not part of the Julia pair" signal
  survives grayscale/color-blind viewing, not just a color difference. Each bar is also
  annotated with its own thread count directly on the bar.
- Test coverage: `test_baseline_pivot_figure_empty_input_raises_clear_error`,
  `test_baseline_pivot_figure_plots_one_bar_per_row_in_order`,
  `test_baseline_pivot_figure_x_axis_is_categorical_not_a_thread_count`,
  `test_baseline_pivot_figure_current_engine_bar_is_hatched_distinctly`,
  `test_baseline_pivot_figure_annotates_julia_speedup`,
  `test_baseline_pivot_figure_deck_theme_exports_at_exact_size`,
  `test_baseline_pivot_figure_compact_variant_exports_at_exact_size`.

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

### 2b. threads-with-julia (deck page 31 — build 2, Rust + PauliPropagation.jl overlay, 2026-09-14)

Job 7034671 landed, so this is the "reveal" build that follows `thread_scaling_v2` on the same
slide: `thread_scaling_v2` is kept as-is (it is the deliberate Rust-only first build, not a
draft to overwrite), and the overlay is a new `_v3` asset per the same function,
`make_thread_scaling_figure(rust_rows, other_rows=julia_rows, other_label="PauliPropagation.jl",
theme="deck", figsize_pt=(900, 340))`. `rust_rows` is the exact same `thread_scaling()` call and
`raw/` sources as the v2 entry above (unchanged, re-verified against the raw files).

- Files: `thread_scaling_v3.{svg,pdf,png}` (900x340pt, 2500x944px), `thread_scaling_v3_compact.*`
  (450x340pt half-width, 1250x944px).
- Julia data: `normalize.thread_scaling()` called with `variant_id="external_pauli_propagation_jl"`,
  `min_abs_coeff=2^-16 (1.5258789e-05)`, over 8 completed runs (`threads` in
  {1,2,4,8,16,32,48,96}, `julia_backend="vector"`) from
  `raw/2026-09-14-worker7160-julia/runs.jsonl`, job 7034671, 127-qubit canonical circuit, direct
  Julia timing (`wall_time_s`, not `extra.driver_wall_s`, per this campaign's established
  convention of timing the propagation call itself, not the whole driver process).
- Real `wall_time_s` by thread count: 1 -> 1590.163, 2 -> 1113.476, 4 -> 698.944, 8 -> 479.973,
  16 -> 382.632, 32 -> 302.335, 48 -> 334.580, 96 -> 974.958 (seconds). Speedup vs. Julia's own
  1-thread baseline: 1.00x, 1.43x, 2.28x, 3.31x, 4.16x, **5.26x (peak, at 32 threads)**, 4.75x
  (48 threads), 1.63x (96 threads).
- **Non-monotonic degradation, real and reproducible from this one job's data (not noise)**:
  Julia's `vector`-backend thread scaling is good through 32 threads, then **degrades badly
  beyond that** — 48 threads (334.58s) is slower than 32 threads (302.33s), and 96 threads
  (974.96s) is dramatically slower than 48 threads, roughly **2.9x worse** (974.958/334.580 =
  2.91x) — worse in absolute wall time than even the 4-thread point (698.94s). This is stated
  explicitly on the figure/slide, never smoothed over: the Rust curve (`thread_scaling_v2`) has
  no such collapse over the same thread range, so the overlay is itself the interesting
  Rust-vs-Julia parallel-efficiency finding for this slide, not just a speed comparison.
- Test: `figures/tests/test_make_figures.py::test_thread_scaling_julia_overlay_is_non_monotonic_past_32_threads`
  pins this shape (peak at 32, monotonic decline through 48 and 96) as a regression tripwire.

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

### 6. bucket-size (deck page 30 — "Throughput versus target bucket size and occupancy distribution")

**Superseded by "6b. bucket-size-v3" below** (job 7036867/7036934, `topn:1000000` re-sweep) —
kept here for provenance. The `coeff:2^-12` sweep this section describes reached only ~770-900
final terms, too few for `target_bucket_len` in {256..4096} to produce meaningfully different
occupancy (most buckets ended up near-empty regardless of the target).

No figure previously existed for this asset. New function
`make_bucket_size_figure(rows, theme=..., figsize_pt=..., title=...)` in
`figures/make_compact_figures.py`: two panels vs. `target_bucket_len` (x,
log2-spaced) — throughput (left) and occupancy median/p95/max plus the
empty-bucket fraction on a twin bar axis (right). `rows` are the probe's raw
JSON sidecar objects, each augmented with a `strings_per_s` key read from the
sibling `.txt` report (the JSON does not carry `strings/s` — confirmed
against the sidecar's own keys); there is no `normalize.py` step for this
shape, matching `make_distributed_capacity_figure`'s precedent of taking raw
dicts directly when no existing normalize helper fits.

- Files: `bucket_size_v2.{svg,pdf,png}` (900x340pt, 2500x944px @200dpi),
  `bucket_size_v2_compact.{svg,pdf,png}` (900x170pt half-height, 2500x472px @200dpi).
- Data: single-thread, 127-qubit heavy-hex Trotter step, `min_abs_coeff=2^-12
  (0.000244140625)`, `min_buckets=128`, `target_bucket_len` in
  {256, 512, 1024, 2048, 4096} — job 7035853. Source:
  `raw/2026-09-14-worker7202-bucketsize/bucketsize-tbl{256,512,1024,2048,4096}-probe.json`
  and the matching `.txt` phase-breakdown files (`strings/s` line only).
- Real numbers plotted (`target_bucket_len`: `strings_per_s`, `num_buckets`,
  `empty_buckets`, `occupancy_median`/`p95`/`max`):
  - 256: 5.713e7, 4096, 3374, 1/2/3
  - 512: 6.659e7, 2048, 1416, 1/2/4
  - 1024: 7.158e7, 1024, 518, 1/3/5
  - 2048: 7.438e7, 512, 107, 2/4/6
  - 4096: 7.588e7, 256, 7, 3/6/8
- **Honesty caveat, stated in the function's own docstring**: this is one
  fixed cell configuration (workload/threads/hash/cutoff/depth all fixed,
  only `target_bucket_len` varies), 5 points, one job. Throughput rises
  **monotonically across the entire tested range with no interior peak** —
  the figure/docstring never claims an optimum, a flattening, or cache
  residency; the honest statement is that the curve is still rising at the
  largest tested value (4096). A performance optimum near an estimated cache
  size would be evidence consistent with locality, not proof of cache
  residency — moot here regardless, since there is no optimum in range at all.
- Occupancy was sampled at the final step of a genuine 20-step trajectory,
  after a real depth-doubling bug in the occupancy-sampling path was found
  and fixed in `crates/paulistrings/examples/phase_breakdown.rs` (git log:
  "phase_breakdown: fix occupancy sampling's silent depth-doubling") —
  `--occupancy-at N` previously sampled after a warm-up pass plus N more
  steps (real depth `2*reps`, not `reps`), which for this growing Trotter
  trajectory had already truncated to nothing under the coefficient
  threshold while `num_buckets()` stayed at its grow-only high-water mark,
  masking the real effect of `target_bucket_len` on occupancy. Fixed by
  skipping the warm-up entirely on the occupancy-sampling path.
- The empty-bucket fraction (`empty_buckets/num_buckets`) is rendered on its
  own twin bar axis, never folded into the occupancy percentiles — a large
  empty fraction lowers the *mean* occupancy of all buckets but says nothing
  about how full the occupied ones are, which `occupancy_median/p95/max`
  already describe correctly by excluding empty buckets from their sample.
- **Compared against the `presentation` branch's `fig4b_bucket_speedup.py` /
  `fig5_bucket_sweep.py`** (the user's pointed-at reference for this slide):
  read directly from the `pres` worktree, those two figures are a genuinely
  different measurement — fig4b is 16-vs-1-thread speedup + parallel
  efficiency vs. bucket size, fig5 is ns/term-layer + L2/LLC cache-miss rate
  vs. bucket size, both from a 2026-09-06 `ccqlin038` (Cascade Lake) sweep
  (`presentation/data/bucket_sweep.jsonl` + `bucket_sweep_perf.jsonl`,
  commit `52511c0`) that predates the `--occupancy-at` flag entirely and
  carries no occupancy data of any kind. Neither is a structural match for
  the actual page-30 ask, so the two-panel throughput+occupancy layout above
  is kept as-is (it already is that content) rather than restructured to
  match either reference, and no numbers from that worktree appear here.
  One design idea *is* borrowed, cheaply and honestly: each throughput point
  is now annotated `B={num_buckets}`, the same convention fig4b uses under
  its bottom panel — `num_buckets` was already a required row field. No
  cache-crossing vertical bands (which both fig4b and fig5 have) were added:
  this campaign's host is AMD Genoa (EPYC 9474F) and `research/HARDWARE.md`
  has no measured L2/LLC size for it, only for `ccqlin038`, and this repo's
  convention is measured facts over spec numbers. The two figures' own
  `common.py` color palette was not adopted either — this deck has its own
  `_DECK_SERIES`/`_DECK_NAVY` theme that every other v2 figure in this
  campaign already uses, and switching one figure to a different deck's
  palette would break intra-deck consistency, not improve it.

### 6b. bucket-size-v3 (deck page 30 — real topn:1000000 re-sweep, single panel, 2026-09-14)

Jobs 7036867 (5-point sweep) + 7036934 (added `target_bucket_len=8192`, plus real L2 cache
capture). `make_bucket_size_figure` gained `single_panel=True` (throughput only, per user
request) and `l2_cache_bytes=`/`bytes_per_term=` (a vertical reference line at
`target_bucket_len = l2_cache_bytes / bytes_per_term`).

- Files: `bucket_size_v3.{svg,pdf,png}` (900x340pt single panel), `bucket_size_v3_compact.*`
  (450x340pt half-width).
- Real data (all 6 points hit exactly `n=1,000,000` via `topn:1000000`, ZERO empty buckets at
  every `target_bucket_len` — a real fix over the v2 sweep's near-empty buckets):

  | target_bucket_len | num_buckets | strings/s | occupancy median/p95/max |
  |---|---|---|---|
  | 256 | 4096 | 5.787e7 | 244/269/300 |
  | 512 | 2048 | 6.132e7 | 488/524/569 |
  | 1024 | 1024 | 6.322e7 | 976/1031/1079 |
  | 2048 | 512 | 6.513e7 (peak) | 1952/2029/2099 |
  | 4096 | 256 | 6.495e7 | 3905/3996/4069 |
  | 8192 | 128 (the `min_buckets` floor — the largest value that can still move) | 6.498e7 | 7807/7963/8061 |

- **Honest finding, stated in the figure/caption, not overclaimed**: per the user's request for
  "a point at larger target bucket lengths so the peak is more pronounced" — the added
  `target_bucket_len=8192` point shows the shape is a PLATEAU (2048/4096/8192 all within ~0.3%
  of each other), not a sharper peak or a real drop-off. `target_bucket_len=8192` is also the
  last point that can move at all: `num_buckets` is floored at `min_buckets=128`, so any larger
  value is identical.
- **Real L2 cache size, this exact node** (`lscpu -C`, job 7036934, NOT a substituted spec-sheet
  number — `research/HARDWARE.md` had no genoa L2 on file before this): **1 MiB per core**
  (`L2 1M 96M 8 Unified 2 2048 1 64`). At 48 B/term the reference line sits at
  `target_bucket_len ~= 21,845` (2^14.4) — well PAST where the plateau begins (~2^11) and past
  every tested point. The figure does NOT claim the plateau is an L2-residency effect; the line
  is shown as a reference only, per this function's own documented discipline (a peak/plateau
  near an estimated cache size is evidence consistent with locality, never proof of cache
  residency by itself — moot here anyway, since the plateau begins well before the L2 line).
- x-axis label is "target bucket size (terms)" (human-readable), not the code identifier
  `target_bucket_len`, per user request.
- Test coverage: `test_bucket_size_figure_single_panel_omits_occupancy_axes`,
  `test_bucket_size_figure_l2_line_sits_at_cache_bytes_over_bytes_per_term`.

### 7. bucketed-1t (deck page 29 — DROPPED 2026-09-14, see "bucketed-1t-v3" below)

**Dropped.** `decisions.md` #46: the user pointed out this figure (historical
`bucketed_engine_serial` vs. `bucketed_engine_parallel`, two DIFFERENT old commits) does not
support what page 29 is actually meant to claim — that the CURRENT engine, single-threaded,
performs worse at one bucket than at its default many-bucket configuration, isolated from
threading entirely. `bucketed_1t_v2*` files were removed from `figures/real/`; this section is
kept for provenance only (the description below is of the removed figure). The stage-6
historical view (`make_recurring_figure(..., stage=6, ...)`) itself remains real, available
backup material (e.g. for the "attempts" story, page 19), just no longer page 29's asset — see
the new "bucketed-1t-v3" entry immediately after this one for what replaced it.

`make_recurring_figure(..., stage=6, highlight_variant="bucketed_engine_serial", theme="deck")` —
the SAME two-panel function already used for `recurring_stage6.png` and the baseline figure
above; no new plotting function was written. This is the real, full-scale (n_qubits=127,
trotter_steps=10) extension of `recurring_stage6.png`'s toy-scale data, from the real genoa
cluster job 7033945 (`decisions.md` #44) — not the same job as `recurring_stage6.png`'s and not
a replacement for it.

- Files: `bucketed_1t_v2.{svg,pdf,png}` (900x340pt, 2500x944px @200dpi), `bucketed_1t_v2_compact.*`
  (900x170pt half-height, 2500x472px @200dpi).
- Source: `raw/2026-09-14-worker7150-historical/runs.jsonl`, 17 rows, all `status=completed`,
  validated 0 problems (`analysis/validate_campaign.py`).
- Real numbers plotted (`variant_id`: eps=2^-12 -> 2^-14 -> 2^-16 -> 2^-18 `wall_time_s`,
  `final_terms` identical across `direct_small_sum_path`/`bucketed_engine_serial`/
  `bucketed_engine_parallel` at each cutoff: 232,432 / 696,172 / 1,791,652 / 3,936,794):
  - `naive_baseline`: one point only, eps=2^-12, `65.406s` (3,018,683 terms; truncation-inert at
    this commit, `decisions.md` #42).
  - `direct_small_sum_path`: `0.750s -> 0.805s -> 0.920s -> 1.150s`.
  - `bucketed_engine_serial`: `11.988s -> 21.858s -> 41.132s -> 66.113s`.
  - `bucketed_engine_parallel`: `14.781s -> 39.356s -> 79.512s -> 140.263s`.
- **Confirmed, prominently disclosed finding**: `bucketed_engine_parallel` is SLOWER than
  `bucketed_engine_serial` at every one of the 4 tolerance points, on real full-scale genoa
  cluster hardware — ratio 1.23x, 1.80x, 1.93x, 2.12x at eps=2^-12/2^-14/2^-16/2^-18
  respectively, growing more pronounced at tighter tolerances. `final_terms` match exactly
  between the two at every cutoff, so this is purely a wall-clock effect, not a correctness bug.
  This confirms, at real full scale, the anomaly first seen only locally/at toy scale
  (`decisions.md` #34) — a genuine "adding threads made it slower" result, stated here and via
  an in-plot text note (full variant only), never softened or buried.
- **Adaptation needed, disclosed**: `normalize.runtime_tolerance()`'s row shape is otherwise used
  directly, but every historical row here has `peak_terms=None` (`trace_enabled=false` at these
  commits) — the point-annotation field falls back to `final_terms` when `peak_terms` is null,
  the same precedent `make_distributed_capacity_figure`'s single-rank reference point already
  established.
- **Highlighting**: `highlight_variant="bucketed_engine_serial"` at `stage=6` (so
  `bucketed_engine_parallel` is drawn too, not hidden by the stage cutoff); a thin
  generation-script-level post-process un-mutes `bucketed_engine_parallel`'s line from the
  standard 0.35 "introduced but not yet highlighted" alpha to full opacity — both lines must stay
  legible side by side for the finding above to read clearly, so this figure deliberately departs
  from the usual single-highlight convention.
- No `title=` kwarg, matching production practice already confirmed by reading the real exported
  `baseline_v2.png`/`baseline_v2_compact.png` (neither carries an in-figure title — the deck
  slide's own title covers that). The finding is instead a small in-plot text note in an
  empirically empty band of the log-log cost panel, full variant only; the half-height compact
  variant omits it (matches `make_distributed_capacity_figure`'s established precedent of leaving
  qualifying remarks to the presenter verbally on a space-constrained compact export).
- **Circuit-mismatch caveat, stated explicitly, not papered over**: a same-scale
  `bucketed_current` overlay was also run in this job (4 points, `3.369s`/189,845 terms through
  `2812.324s`/288,715,006 terms) but is DELIBERATELY NOT PLOTTED on this figure's axes.
  `run_cell.py` drives a different circuit/observable (`theta_h=0.6872233929727672`,
  `observable=debug_single_z`, heavy-hex-adjacent) than `run_cell_historical.py`'s four historical
  variants (`direction=heisenberg`, linear-chain-scaled circuit — three of the four historical
  commits cannot build heavy-hex at all), so `bucketed_current`'s `final_terms` are not
  comparable 1:1 to the historical variants' at the same nominal eps. This mismatch is a known,
  already-accepted campaign convention (the historical comparison is about algorithm/
  implementation cost trends, not an apples-to-apples circuit match) — plotting its roughly
  1000x-larger term counts on the same cost axis would compress the very serial-vs-parallel
  comparison this figure exists to show, so the numbers are disclosed here and in `decisions.md`
  #44 instead of plotted.
- Efficiency (left) panel is honestly empty: none of these four historical commits expose
  per-gate stats, the same disclosed gap as `recurring_stage6.png`/the baseline figure above.

### 7b. bucketed-1t-v3 (deck page 29 — real 2-point comparison, 2026-09-14)

New function `make_single_bucket_comparison_figure(rows, theme=..., figsize_pt=...,
title=...)` in `figures/make_compact_figures.py`: a plain 2-bar chart — current engine,
single-thread, default bucket config vs. forced to a single bucket, at the SAME
`eps=2^-16 (1.5258789e-05)`, 127-qubit canonical config.

- Files: `bucketed_1t_v3.{svg,pdf,png}` (900x340pt, 2500x944px @200dpi), `bucketed_1t_v3_compact.*`
  (450x340pt half-width, 1250x944px @200dpi).
- Real numbers plotted:
  - Default bucket config (`target_bucket_len=1024`, `min_buckets=128`, the library defaults):
    `wall_time_s=1629.9046` (one of 3 repetitions of job 7030090, range 1629.9-1675.6s),
    `final_terms=38,791,220`. Source: `raw/2026-09-13-worker7277/runs.jsonl`.
  - Forced single bucket (`min_buckets=1`, `target_bucket_len=1e9`): `wall_time_s=
    1967.578165213985`, `final_terms=38,791,220`, job 7036526. Source:
    `raw/2026-09-14-worker7160-single-bucket/runs.jsonl`.
- **Real, clean finding**: forcing a single bucket is **~20.7% slower** than the default
  many-bucket configuration, with `final_terms` matching exactly (38,791,220 both), a pure
  wall-clock effect isolated from threading (both runs are single-thread) and from any
  historical-commit confound (both runs are the SAME current-engine commit) — a much more
  direct result than the dropped `bucketed_1t_v2` figure. Stated on the figure itself via the
  `+20.7%` annotation and a `final_terms identical: 38,791,220` caption.
- **`expectation_re` cross-check, disclosed**: job 7030090's own `runs.jsonl` rows carry no
  `extra.expectation_re` field at all (only `wall_time_s`/`final_terms`/etc.). The default-bucket
  config's expectation value is instead cited from the sibling E9 convergence sweep's own
  `trotter_step=20` point at the same config (job 7033650, same `task_id=T02-canonical`, same
  `eps=2^-16`): `expectation_re=0.39716532998468246`, which also reproduces
  `final_terms=38,791,220` exactly — matching the single-bucket run's own
  `expectation_re=0.3971653299846819` to floating-point tolerance (~3e-15), the repo's
  determinism-policy bar. Not plotted on the figure (only `final_terms` is), cited here for the
  record.
- Test coverage: `test_single_bucket_comparison_figure_empty_input_raises_clear_error`,
  `test_single_bucket_comparison_figure_plots_two_bars_in_order`,
  `test_single_bucket_comparison_figure_single_bucket_is_slower_by_about_21_percent`,
  `test_single_bucket_comparison_figure_annotates_percent_difference`,
  `test_single_bucket_comparison_figure_states_matching_final_terms`,
  `test_single_bucket_comparison_figure_deck_theme_exports_at_exact_size`,
  `test_single_bucket_comparison_figure_compact_variant_exports_at_exact_size`.

## Test coverage

`quera-talk-data/campaign-2026-09-11/figures/tests/test_make_figures.py` gained:
`test_thread_scaling_deck_theme_uses_navy_spines_and_exact_size`,
`test_hash_communication_vs_cutoff_deck_theme_distinguishes_series_by_marker_too`,
`test_convergence_deck_theme_still_builds_and_omits_default_legacy_title`,
`test_recurring_figure_deck_theme_with_external_points_draws_stars`,
`test_distributed_capacity_figure_empty_input_raises_clear_error`,
`test_distributed_capacity_figure_rejects_fabricated_runtime_on_failed_row`,
`test_distributed_capacity_figure_marks_completed_oom_and_untested_distinctly`,
`test_distributed_capacity_figure_deck_theme_exports_at_exact_size`,
`test_bucket_size_figure_annotates_realised_bucket_count`.

Full suite: `pytest quera-talk-data/campaign-2026-09-11/figures/tests/` — 33 passed (25
pre-existing + 8 new; a `make_attempts_figure`/`attempts.*` figure and its tests present in
this working tree belong to concurrent, unrelated work in this shared worktree, not this task).

The bucket-size figure (section 6 above) added
`test_bucket_size_figure_empty_input_raises_clear_error`,
`test_bucket_size_figure_builds_two_panels`,
`test_bucket_size_figure_throughput_panel_plots_all_five_points_in_order`,
`test_bucket_size_figure_throughput_has_no_peak_in_tested_range`,
`test_bucket_size_figure_empty_fraction_never_folded_into_occupancy_percentiles`,
`test_bucket_size_figure_deck_theme_exports_at_exact_size`. Full suite after this addition:
39 passed (33 pre-existing + 6 new).

The bucketed-1t figure (section 7 above) added `test_bucketed_1t_parallel_slower_than_serial_at_every_cutoff`
(pins the real numbers as a regression tripwire), `test_bucketed_1t_figure_reuses_recurring_figure_stage6_unmodified`,
`test_bucketed_1t_figure_unmuted_parallel_line_stays_fully_legible`,
`test_bucketed_1t_figure_deck_theme_exports_at_exact_size`. Full suite after this addition:
46 passed (42 pre-existing + 4 new; run via
`.venv/bin/python -m pytest quera-talk-data/campaign-2026-09-11/figures/tests/`).

### 8. memory (deck page 17 — "Memory & bandwidth diagnosis")

No figure previously existed for this asset. New function
`make_memory_diagnosis_figure(phase_rows, traffic, theme=..., figsize_pt=..., title=...)`
in `figures/make_compact_figures.py`: two panels — (A) a phase time-SHARE
horizontal stacked bar at 1 and 96 threads (`permute`/`coset_loop`/`unpermute`/
`recount`/`other (serial)`, the same "fold small serial phases together"
convention the probe's own `.txt`/HTML report uses), each bar annotated with
its real ms/layer total so the ~3.4x absolute-time difference between the two
thread counts is not lost by normalizing to a 0-100% share; (B) three
DISTINCT "stat tile" numbers — payload, modeled traffic, peak resident memory
— deliberately NOT drawn as bars on one shared axis, since they are different
units (B/term, GB/s, kB) and a shared axis would visually imply comparable
magnitudes.

- Files: `memory_v2.{svg,pdf,png}` (900x340pt, 2500x944px @200dpi),
  `memory_v2_compact.{svg,pdf,png}` (900x170pt half-height, 2500x472px
  @200dpi) — half-height chosen over half-width because this is a two-panel
  view (a half-width box left neither panel legible in a real export test;
  the two-panel/annotated-plot half-height precedent this MANIFEST already
  states for `baseline_v2_compact`/`distributed_capacity_v2_compact` fits
  here too). The compact variant drops panel B's sub-captions and the
  bandwidth-unavailable prose note (same "compact drops qualifying detail,
  presenter states it verbally" precedent `make_distributed_capacity_figure`
  established) — the three headline numbers themselves stay, only their
  supporting text is cut for space.
- Data source: real cluster job **7035691**, `raw/2026-09-14-worker7183-memory/`
  (host `worker7183.cm.cluster`, AMD EPYC 9474F "Genoa", commit
  `9bbb65196f40ed809af836fccce93a8e03873c0a`, 2026-09-14). Circuit:
  `heavyhex_step`, 127 qubits, `--truncation coeff:1.5258789e-05`, `n=229269`,
  `layers=5420`, threads swept at {1, 96}.
  - **theta_h caveat, stated here explicitly per the task**: `phase_breakdown`
    hard-codes `theta_h = 5*pi/16` for its `heavyhex_step` layer, NOT this
    campaign's primary `theta_h = 7*pi/32` working point — this is this
    campaign's own synthetic benchmark circuit, not the canonical Python
    task, and a known, already-accepted gap (same probe used for the earlier
    bucket-size figure, section 6).
  - `perf-stat.sh` failed for this job ("Workload failed: No such file or
    directory") and is marked non-fatal in the job log — perf counters are
    blocked on this shared cluster account, so there is no flame graph and no
    hardware-counter evidence anywhere in this figure. Nothing here claims
    bandwidth saturation from the phase-timing breakdown alone; the breakdown
    only identifies which phases cost time.
- **Panel A real numbers** (probe `.txt`, ms/layer and % of that row's own
  wall time):
  - 1 thread (wall 51.832 ms/layer): permute 5.9934 (11.6%), coset_loop
    43.5192 (84.0%), unpermute 1.9688 (3.8%), recount 0.3393 (0.7%), other
    (serial) 0.0100 (~0.0%).
  - 96 threads (wall 15.152 ms/layer): permute 7.4094 (48.9%), coset_loop
    2.2446 (14.8%), unpermute 5.1791 (34.2%), recount 0.3085 (2.0%), other
    (serial) 0.0090 (~0.0%). Parallel efficiency (busy / (coset_loop ×
    threads)) drops from 0.89 at 1 thread to 0.33 at 96 -- `coset_loop`
    itself is no longer the dominant phase at 96 threads, `permute` and
    `unpermute` (the serial repacking either side of it) are.
- **Panel B: three DISTINCT numbers, kept separate by construction**:
  1. **Payload (fixed fact)**: 48 B/term for `W=2`, `Complex64`
     (`T=16W+16`) — independent of any measurement.
  2. **Modeled traffic**: **1.79 GB/s**, **142 B/term-update** (~3.0x the raw
     payload), scoped ONLY to the **96-thread `coset_loop` phase**
     (`coset_loop_ns=12,165,898,487` at 96 threads) — the one phase/thread
     count this campaign treats as a genuinely valid rate comparison, since
     it is the parallel, steady-state region; the serial `permute`/
     `unpermute` phases and the 1-thread cell are NOT used for this ratio.
     Derivation (same `T=16W+16` traffic model `perf-viz.py` already
     implements, confirmed against the probe's own auto-rendered HTML
     report): `bytes/layer = (terms_in/layers)*48 [gather-in]
     + 2*(rows_sorted/layers)*48 [gather-w + merge-r]
     + 2*(terms_in/layers)*16 [coeff-only id rows, keys borrowed]
     + 2*(rows_sorted/layers)*48 [sort r/w]
     + (terms_out/layers)*48 [merge-out] = 4,011,262.75 B/layer`, using the
     probe's real `terms_in=153,312,746`, `rows_sorted=11,083,394`,
     `terms_out=153,083,599`, `layers=5420`. `GB/s = bytes/layer * layers /
     (coset_loop_ns/1e9) / 1e9 = 1.787`; `B/term-update = bytes/layer /
     (terms_out/layers) = 142.02`.
  3. **Peak resident memory (`VmHWM`)**: **16,806,572 kB ≈ 16.81 GB** —
     identical in both the 1- and 96-thread probe rows (one process, one
     high-water mark across both cells of this run); reported on its own,
     never divided by anything or folded into the traffic number above.
- **No bandwidth-ceiling comparison is made, and the figure/caption say so
  explicitly**: `bandwidth.txt` for this job shows the sweep never produced
  any measurement — `scripts/bandwidth.sh` failed to build `membench` on
  worker7183 (`bandwidth.stderr.log`: `target/release/membench: No such file
  or directory`), so there is no genoa ceiling at 1 OR 96 threads. The probe's
  own auto-rendered HTML report reaches the identical conclusion
  independently ("Bandwidth ceilings unavailable for this campaign ... DRAM
  figures below show modeled GB/s only, with no % of ceiling."). Per the
  task's explicit instruction, `research/HARDWARE.md`'s Cascade Lake
  (`ccqlin038`) bandwidth ceilings are a **different architecture** and are
  **not substituted in** — `make_memory_diagnosis_figure` renders the
  `bandwidth_unavailable_reason` string instead of a fabricated percentage
  whenever `bandwidth_ceiling_gbps` is `None`, which is the real, honest
  state for this campaign.
- Generation script: ad hoc (not checked in, matching this repo's existing
  precedent for `bucket_size_v2`/`distributed_capacity_v2` — no `jobs/
  generate_figures.py` exists for any of these); the exact call and traffic
  derivation above are reproducible from the numbers in this entry.

`quera-talk-data/campaign-2026-09-11/figures/tests/test_make_figures.py` gained (memory
diagnosis, section 8 above): `test_memory_diagnosis_figure_empty_input_raises_clear_error`,
`test_memory_diagnosis_figure_builds_two_panels`,
`test_memory_diagnosis_figure_phase_shares_sum_close_to_100_percent`,
`test_memory_diagnosis_figure_annotates_real_wall_ms_per_layer`,
`test_memory_diagnosis_figure_keeps_payload_traffic_peak_rss_as_distinct_numbers`,
`test_memory_diagnosis_figure_states_bandwidth_unavailable_reason_when_ceiling_is_none`,
`test_memory_diagnosis_figure_with_real_ceiling_states_percent_of_ceiling`,
`test_memory_diagnosis_figure_deck_theme_exports_at_exact_size`,
`test_memory_diagnosis_figure_compact_variant_exports_at_exact_size`. Full suite after this
addition: **55 passed** (46 pre-existing + 9 new; run via
`.venv/bin/python -m pytest quera-talk-data/campaign-2026-09-11/figures/tests/`).

The baseline-pivot figure (section 1b above) added
`test_baseline_pivot_figure_empty_input_raises_clear_error`,
`test_baseline_pivot_figure_plots_one_bar_per_row_in_order`,
`test_baseline_pivot_figure_x_axis_is_categorical_not_a_thread_count`,
`test_baseline_pivot_figure_current_engine_bar_is_hatched_distinctly`,
`test_baseline_pivot_figure_annotates_julia_speedup`,
`test_baseline_pivot_figure_deck_theme_exports_at_exact_size`,
`test_baseline_pivot_figure_compact_variant_exports_at_exact_size`. The bucketed-1t-v3 figure
(section 7b above) added `test_single_bucket_comparison_figure_empty_input_raises_clear_error`,
`test_single_bucket_comparison_figure_plots_two_bars_in_order`,
`test_single_bucket_comparison_figure_single_bucket_is_slower_by_about_21_percent`,
`test_single_bucket_comparison_figure_annotates_percent_difference`,
`test_single_bucket_comparison_figure_states_matching_final_terms`,
`test_single_bucket_comparison_figure_deck_theme_exports_at_exact_size`,
`test_single_bucket_comparison_figure_compact_variant_exports_at_exact_size`. Full suite after
both additions: **69 passed** (55 pre-existing + 14 new; run via
`.venv/bin/python -m pytest quera-talk-data/campaign-2026-09-11/figures/tests/`).
