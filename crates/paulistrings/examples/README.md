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
