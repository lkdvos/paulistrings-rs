# `docs/` — the documentation site

The published site at
[lkdvos.github.io/paulistrings-rs](https://lkdvos.github.io/paulistrings-rs/) is
an [mdBook](https://rust-lang.github.io/mdBook/) whose source is
[`book/`](book/), with the crate's rustdoc mounted underneath it at `/api/`.
Both are built and deployed by [`.github/workflows/docs.yml`](../.github/workflows/docs.yml)
on every push to `main`, published from the `gh-pages` branch (GitHub Pages'
"Deploy from a branch" source, not the Actions-based deployment — see "Previews"
below for why).

Every fenced `python` code block in the book runs for real in that same
workflow, against a built (debug-profile) extension, before anything deploys — see
[`scripts/test-doc-snippets.py`](../scripts/test-doc-snippets.py).
A block preceded by a `<!-- doctest: skip -->` line is excluded: MPI/NUMA
snippets this runner has no hardware for, and case-study excerpts already
covered by `pytest examples/tests`.
Run it locally the same way CI does, against an activated venv:

```bash
python scripts/test-doc-snippets.py            # every page
python scripts/test-doc-snippets.py --list      # what would run/skip, no execution
python scripts/test-doc-snippets.py FILE ...    # just these pages
```

This is an exit-code check, not an accuracy guarantee: a snippet that runs and
prints a wrong number still passes. It catches a broken import, a renamed
method or a stale signature — not stale prose around a correct-looking output.

## Previews

Every same-repo pull request that touches `docs/**` gets its build published
under `pr-preview/pr-<n>/` on the `gh-pages` branch, with the link
auto-commented on the PR by
[`rossjrw/pr-preview-action`](https://github.com/rossjrw/pr-preview-action);
the preview is removed again when the PR closes. This only works for pull
requests from this repository (not forks), since a fork's `GITHUB_TOKEN` has
no write access — same restriction the main deploy always had. This is also
why the site deploys via a `gh-pages` branch rather than the newer
Actions-based Pages deployment: a preview subtree living alongside the main
site on one branch has no equivalent in the Actions method, which publishes
exactly one artifact per deployment.

## Building it locally

```bash
cargo install mdbook --locked --version 0.5.4         # once; or grab the release binary
cargo install mdbook-katex --locked --version 0.10.0  # once; no prebuilt binary published
./docs/sync-assets.sh                                 # refresh the figure links
mdbook build docs/book                                # renders to docs/book/site/ (gitignored)
mdbook serve docs/book --open                         # live-reloading preview
```

Pin the same mdBook and mdbook-katex versions the workflow uses (`MDBOOK_VERSION` and
`MDBOOK_KATEX_VERSION` there) so a local build and CI cannot disagree about rendering.

Math is `$...$` inline and `$$ ... $$` on its own lines for a display equation,
rendered by the [`mdbook-katex`](https://github.com/lzanini/mdbook-katex)
preprocessor (`[preprocessor.katex]` in [`book/book.toml`](book/book.toml)). It
depends on a CDN-hosted KaTeX stylesheet, so a local `mdbook serve` needs network
access to see it styled.

`create-missing = false` in [`book/book.toml`](book/book.toml) makes a
`SUMMARY.md` entry with no file behind it an **error** rather than a silently
created empty page — the site is assembled from committed material only.

One link is expected to be dead in a local build: the landing page's **API
reference** points at `api/paulistrings/index.html`, which the workflow fills in
by rendering `cargo doc` alongside the book. Everything else resolves locally.

## What is in it

The book is organized into four areas, Python-only — the Rust interface's home
is the crate's own rustdoc and `ARCHITECTURE.md`.

| section | contents |
|---|---|
| `book/src/index.md`, `installation.md` | landing page and install |
| `book/src/manual/` | Operators, Circuits, Engine and propagation (direction, truncation, validation, incremental propagation, stats/memory/logging, the engine internals, NUMA partitions, MPI ranks), Measurements |
| `book/src/examples/` | a guided first propagation, `showcases/` (the Part-B applications) and `benchmarks/` (the Part-A benchmarks plus engine performance), and comparisons against other tools |
| `book/src/library/` | terse Python API reference: `PauliSum`, `Circuit`, `propagate`, truncation, direction, measurement, module helpers |

## Two rules the content follows

**The pitch paragraph is single-sourced.** `book/src/index.md` pulls it out of the
repository `README.md` with mdBook's `{{#include ../../../README.md:pitch}}`,
between the `ANCHOR: pitch` / `ANCHOR_END: pitch` comments there. Do not
paraphrase it into the book — edit the README and both follow.

**Every number on the site is traceable.** A page is the writeup of record for
its study, and cites two things: the committed results artifact it draws from
(`results*.json` or `.csv`, next to the script that produced it) and the
provenance block in that study's README. No page introduces a measurement of its
own, and nothing on the site regenerates itself. When a benchmark is rerun, the
artifact, the README's tables and the page are updated in the same commit.

## Figures

Every figure is a committed matplotlib SVG living next to the script that
produced it. mdBook only copies assets from inside its `src/` tree, so
[`sync-assets.sh`](sync-assets.sh) maintains **relative symlinks** under
`book/src/assets/<group>/` pointing back at those originals — no duplicated
bytes, and no way for a page to show a figure that has drifted from its source.
mdBook resolves the links at build time, so the published site contains real
files.

Run the script after adding a figure to a page or renaming a source directory; it
is idempotent, and it exits non-zero on a missing source or a dangling link
rather than shipping a broken image.
