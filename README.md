# paulistrings-rs

Rust library for Pauli string manipulation, with Python bindings. Inspired by
[`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl).

## Layout

```
crates/
  paulistrings/         # core Rust library
    benches/            # criterion microbenchmarks
  paulistrings-py/      # PyO3 bindings (cdylib `_paulistrings`)
python/
  paulistrings/         # Python package; re-exports the extension module
benchmarks/
  python/               # pytest-benchmark suites + cross-library comparisons
  results/              # raw benchmark output (gitignored)
research/
  ideas/  plans/  notes/  literature/
```

## Quickstart

```bash
# One-time setup: creates .venv, installs maturin/pytest/etc., builds the extension.
./scripts/setup.sh
source .venv/bin/activate

# Rust
cargo build --release
cargo test
cargo bench -p paulistrings

# Python
maturin develop --release -m crates/paulistrings-py/Cargo.toml   # rebuild after Rust changes
pytest python/paulistrings/tests
pytest benchmarks/python --benchmark-only
```

## Toolchain

- **Rust** is pinned via `rust-toolchain.toml` (currently 1.94.0, with `rustfmt`
  and `clippy`). `rustup` picks this up automatically.
- **Python** 3.9+ is required. `scripts/setup.sh` defaults to
  `/usr/bin/python3.11`; override via `PYTHON=/path/to/python3.x ./scripts/setup.sh`.
- The venv lives at `./.venv` (gitignored). Re-run `scripts/setup.sh` any time
  to refresh dependencies — it is idempotent.
- After Rust changes, rebuild the extension with
  `maturin develop --release -m crates/paulistrings-py/Cargo.toml`.

## Goals

- Fast — competitive with or faster than existing Pauli-string libraries on
  representative workloads.
- Ergonomic from both Rust and Python.
- Honest benchmarks. See `benchmarks/README.md`.

## Status

v0.1 scaffolding. See `research/plans/` for the roadmap.
