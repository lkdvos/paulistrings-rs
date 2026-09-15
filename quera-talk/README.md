# QuEra talk

Everything for the QuEra technical talk on Pauli propagation lives under this one folder.

- `data/campaign-2026-09-11/` — the benchmark campaign: job scripts, decisions log, evidence, analysis, and the real figures it produced. See its own `README.md`, `decisions.md`, and `figures/real/MANIFEST.md`. Raw job output (`raw/`, gitignored, several GB) is not part of this history; it lives on the filesystem wherever the campaign was actually run.
- `deck/` — the Typst slide deck. Currently a scaffold: every figure in `deck/figures/figure-manifest.json` is still `"pending"`; wiring in the real figures from `data/campaign-2026-09-11/figures/real/` is the next step.
- `outline.md` — the talk outline.
- `handoff/` — agent handoff notes from earlier work on the benchmark campaign and the Typst scaffolding.
- `reference/` — reference material for the deck's visual style (the `organic` theme).

This folder consolidates work that was previously split across three worktrees/branches (`presentation-work`, `presentation-slides`, and an older superseded `presentation` branch, now `archive/presentation-2026-08`). Engine-level changes made along the way (raising `P_MAX_BITS`, the distributed row-policy/`cut` support, etc.) were extracted separately onto `engine/distributed-scaling-p64` for evaluation against `main` rather than folded in here.
