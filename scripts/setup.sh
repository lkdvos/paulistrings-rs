#!/usr/bin/env bash
# Idempotent dev environment bootstrap for paulistrings-rs.
# Creates ./.venv, installs Python tooling, and builds the PyO3 extension.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-/usr/bin/python3.11}
VENV=.venv

if [[ ! -x $PYTHON ]]; then
  echo "error: $PYTHON not found or not executable" >&2
  echo "       set PYTHON=/path/to/python3.11 and re-run" >&2
  exit 1
fi

if [[ ! -d $VENV ]]; then
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip

# Dev tooling. Installed directly (not via `pip install -e .[dev]`) so we don't
# trigger the maturin build backend twice — `maturin develop` below handles the
# Rust build with the flags we want.
pip install 'maturin>=1.5,<2.0' 'pytest>=7' 'pytest-benchmark>=4' ruff 'numpy>=1.24'

# Optional: cross-library benchmark deps. Best-effort — failures here shouldn't
# block the rest of the setup.
if ! pip install qiskit openfermion stim; then
  echo "warning: optional benchmark deps (qiskit/openfermion/stim) failed to install — skipping" >&2
fi

# Build the Rust extension into the venv (release mode for perf parity).
maturin develop --release -m crates/paulistrings-py/Cargo.toml

cat <<EOF

Setup complete.

  Activate:  source $VENV/bin/activate
  Rust:      cargo build --release && cargo test
  Python:    pytest python/paulistrings/tests
  Rebuild:   maturin develop --release -m crates/paulistrings-py/Cargo.toml
EOF
