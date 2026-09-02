# `docs/` — the documentation site

The published site at
[lkdvos.github.io/paulistrings-rs](https://lkdvos.github.io/paulistrings-rs/) is
an [mdBook](https://rust-lang.github.io/mdBook/) whose source is
[`book/`](book/), with the crate's rustdoc mounted underneath it at `/api/`.
Both are built and deployed by [`.github/workflows/docs.yml`](../.github/workflows/docs.yml)
on every push to `main`.

## Building it locally

```bash
cargo install mdbook --locked --version 0.5.4   # once; or grab the release binary
./docs/sync-assets.sh                           # refresh the figure links
mdbook build docs/book                          # renders to docs/book/site/ (gitignored)
mdbook serve docs/book --open                   # live-reloading preview
```

Pin the same mdBook version the workflow uses (`MDBOOK_VERSION` there) so a local
build and CI cannot disagree about rendering.

`create-missing = false` in [`book/book.toml`](book/book.toml) makes a
`SUMMARY.md` entry with no file behind it an **error** rather than a silently
created empty page — the site is assembled from committed material only.

One link is expected to be dead in a local build: the landing page's **API
reference** points at `api/paulistrings/index.html`, which the workflow fills in
by rendering `cargo doc` alongside the book. Everything else resolves locally.

## What is in it

| page | contents |
|---|---|
| `book/src/index.md` | landing page: what Pauli propagation is, what the library is and is not |
| `book/src/getting-started.md` | install, both quickstarts, truncation, direction semantics, threads |
| `book/src/showcases/` | one page per Part-B showcase: setup, method, measured result |
| `book/src/benchmarks/` | one page per Part-A benchmark A–E, setup → oracle → result |
| `book/src/comparisons.md` | `PauliPropagation.jl` parity and crossover; state-vector, stabilizer and MPO methods |

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
