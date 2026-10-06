#!/usr/bin/env bash
# Create or refresh the dev venv (./.venv, or $UV_PROJECT_ENVIRONMENT) with uv: dev tools, the examples extra, and the extension in release mode.
# Afterwards `uv run <cmd>` keeps both current, rebuilding the extension whenever a Rust source changes.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || { echo "uv not found: module load uv, or see https://docs.astral.sh/uv/" >&2; exit 2; }

# Two steps because the extension builds without isolation against the venv's maturin, which uv before 0.8 does not install first.
uv sync --no-install-project "$@"
# The examples extra is best-effort, but without qiskit-aer every statevector cross-check silently skips.
if ! uv sync --extra examples "$@"; then
    echo "warning: the examples extra failed to install; continuing without it" >&2
    uv sync "$@"
fi

cat <<'EOF'

Setup complete.

  Python:  uv run pytest python/paulistrings/tests
  Rust:    cargo test --workspace
  Shell:   source .venv/bin/activate   # a plain `python` does not rebuild the extension; `uv run python` does
EOF
