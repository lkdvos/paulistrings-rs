#!/usr/bin/env bash
# Populate a Rust toolchain + crate registry on the shared filesystem for cluster jobs.
#
# On the CCQ workstations `~/.cargo` and `~/.rustup` are symlinks into the local NVMe `/home`,
# which cluster nodes do not mount, so rustup's `cargo` proxy is a dangling symlink there
# ("cargo: command not found" in the job log). Run this once on a host WITH network access
# (a workstation or login node); the sbatch templates then use the result via
#   SHARED_RUST=$HOME/.local/rust-shared   (override with the env var of the same name)
#   RUSTUP_HOME=$SHARED_RUST/rustup  CARGO_HOME=$SHARED_RUST/cargo  PATH=$CARGO_HOME/bin:$PATH
# Re-run after bumping rust-toolchain.toml or adding dependencies (it is idempotent).
set -euo pipefail
cd "$(dirname "$0")/../.."
SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
mkdir -p "$CARGO_HOME/bin" "$RUSTUP_HOME"
rustup_bin=$(command -v rustup || readlink -f "$HOME/.cargo/bin/rustup")
[ -x "$rustup_bin" ] || { echo "rustup not found; install it first" >&2; exit 1; }
cp -f "$rustup_bin" "$CARGO_HOME/bin/rustup"
for proxy in cargo rustc rustdoc rustfmt cargo-fmt cargo-clippy clippy-driver; do
  ln -sf rustup "$CARGO_HOME/bin/$proxy"
done
export PATH="$CARGO_HOME/bin:$PATH"
toolchain=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
rustup toolchain install "$toolchain" --profile minimal --no-self-update
rustup default "$toolchain" >/dev/null
echo "== toolchain"; cargo --version; rustc --version
echo "== registry (cargo fetch, all workspace targets)"
cargo fetch
echo "== offline check"
CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-shared-check-$$" cargo check --offline -p paulistrings --features phase-timing --example phase_breakdown
if command -v mpicc >/dev/null 2>&1 && [ -n "${LIBCLANG_PATH:-}" ]; then
  echo "== offline check, mpi feature (rsmpi + bindgen build-time crates)"
  CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-shared-check-$$" cargo check --offline -p paulistrings --features phase-timing,mpi --example phase_breakdown --tests
else
  echo "note: mpicc/LIBCLANG_PATH not set — load openmpi + llvm modules and re-run to verify the mpi feature builds offline" >&2
fi
rm -rf "${TMPDIR:-/tmp}/paulistrings-shared-check-$$"
echo "ready: RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME ($(du -sh "$SHARED_RUST" | cut -f1))"
