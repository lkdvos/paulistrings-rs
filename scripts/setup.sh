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

# Optional: cross-library benchmark deps, plus what the examples suite's oracles
# and report plots need (`[examples]` in pyproject.toml — qiskit-aer runs the
# dense reference in `examples/common/oracles.py`). Best-effort: failures here
# shouldn't block the rest of the setup, and every test that touches these
# importorskips them. But leaving qiskit-aer out means a fresh venv silently
# *skips* every statevector cross-check rather than failing, so it belongs here.
if ! pip install qiskit qiskit-aer openfermion stim matplotlib; then
  echo "warning: optional deps (qiskit/qiskit-aer/openfermion/stim/matplotlib) failed to install — skipping" >&2
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
