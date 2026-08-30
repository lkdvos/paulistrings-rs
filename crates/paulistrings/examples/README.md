# Examples

Runnable end-to-end simulations. Each example has a companion narrative
under `crates/paulistrings/docs/examples/` that's embedded into the
rustdoc at [`paulistrings::examples`](../src/lib.rs).

| Example | Source | Narrative |
|---|---|---|
| 2D Ising quench | `ising_2d_quench.rs` | `../docs/examples/ising_2d_quench.md` |

## Run

From the repo root, in release mode:

```bash
cargo run --example ising_2d_quench --release
```

Output (CSVs) lands under `examples/output/` and is committed to the
repo. The committed CSV is what the plot script reads, and the committed
SVG is what docs.rs / GitHub Pages serves — so contributors who just
want to regenerate the plot don't have to rerun the simulation.

## Plot

Plots are generated with matplotlib via the companion script and written
to `crates/paulistrings/docs/examples/img/`:

```bash
source .venv/bin/activate     # or your usual python env
python crates/paulistrings/examples/plot_ising_quench.py
```

If you change the Rust example, regenerate **both** the CSV and the SVG
in the same commit so the embedded plot matches the data.

## Why CSVs and SVGs are committed

docs.rs and the GitHub Pages preview render rustdoc without running any
code; the plot is loaded from `raw.githubusercontent.com` and so must
exist in the tree. Keeping the CSV alongside makes the plot regeneration
a one-line Python invocation instead of a multi-minute Rust run.

## Regenerating the committed output

`output/ising_{4x4,6x6}.csv` come from `cargo run --release --example
ising_2d_quench`; `docs/examples/img/ising_quench.svg` comes from
`python plot_ising_quench.py` (needs matplotlib, e.g. via `./scripts/setup.sh`).

**Known staleness:** the CSVs were regenerated in v0.3 §3, when `TopN(n)`
changed from "keep exactly `n`, key-ascending tiebreak" to "keep at most
`n`, discarding the whole boundary tie group rather than splitting it" (see
the "How sensitive is the answer" section of
`../docs/examples/ising_2d_quench.md` for the full three-way comparison and
numbers). The SVG was regenerated in the same pass via
`module load modules/2.5-beta1 python/3.12.13` (Lmod-provided matplotlib
3.11.0), so it matches the current CSVs.
