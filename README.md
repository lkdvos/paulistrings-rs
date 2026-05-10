# paulistrings-rs

[![CI](https://github.com/lkdvos/paulistrings-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/lkdvos/paulistrings-rs/actions/workflows/ci.yml)
[![docs](https://github.com/lkdvos/paulistrings-rs/actions/workflows/docs.yml/badge.svg)](https://lkdvos.github.io/paulistrings-rs/)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

Rust library for Pauli string manipulation, with Python bindings. Inspired by
[`PauliStrings.jl`](https://github.com/nicolasloizeau/PauliStrings.jl).

**Preview docs:** [lkdvos.github.io/paulistrings-rs](https://lkdvos.github.io/paulistrings-rs/) (rebuilt on every push to `main`).

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
cargo doc --no-deps -p paulistrings --open   # render rustdoc

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

## License

Dual-licensed under the [MIT License](https://opensource.org/licenses/MIT) or
[Apache License 2.0](https://opensource.org/licenses/Apache-2.0), at your
option.
