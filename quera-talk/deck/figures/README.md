# Figures — handoff note for the benchmark/data agent

This directory is empty of real assets on purpose. The deck compiles in **draft mode** with a
built-in placeholder (same size, same axes, "Benchmark data pending") wherever a real figure
belongs — see `components.typ:recurring-figure`. Nothing here blocks a compile; nothing here is
allowed to become a fabricated curve.

## What to read first

`../figure-contract.json` is the actual contract: exact panel sizes in stage points (750×420pt
per panel, 96pt gutter, 1596pt total — already checked against the 1632pt content width by
rendering the prototype, not guessed), typography, colors, variant styling, and the seven
recurring-figure stage filenames plus the two standalone figures (hash-communication, accuracy).
Author each asset at its listed size — `figsize_inches = (width_pt/72, height_pt/72)` in
matplotlib — and do not let anything downstream rescale it; the brief and the archive's own port
notes both call out a clipped/rescaled chart as the class of bug to avoid here.

## What "done" looks like for one asset

1. File lands at the exact path in `figure-manifest.json`, at the exact size in
   `figure-contract.json`.
2. Text is live (embedded), not rasterized, unless the fallback note in the contract applies —
   and if it does, say so in `provenance`.
3. `figure-manifest.json`'s entry flips from `"pending"` to `"real-asset"` with a `provenance`
   block (commit, host, date, generating script).
4. The recurring-figure stage files (`baseline.svg` … `distributed.svg`) each carry the **full
   cumulative set** of variants revealed through that stage, not just the one new curve — see
   `figure-contract.json`'s `stage_filenames_note`.

## Ownership boundary

This project (the slide agent) owns placement, typography, and this contract. It does not own
data, computation, or scientific plotting choices — those belong to whichever agent or process
produces the actual benchmark results (see `../../quera-benchmark-agent-handoff.md` if present in
your workspace). Neither side edits the other's files.
