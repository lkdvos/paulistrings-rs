# Sourced by the Slurm templates after `cd` to the repo root: the shared toolchain, a pinned private checkout, and helpers every template needs.
# Never invokes any Slurm command itself.

repo=$PWD
_checkouts=()

# Rust from the shared filesystem: on the workstations ~/.cargo and ~/.rustup point into local /home, which cluster nodes do not mount.
use_shared_rust() {
    SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
    export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
    export PATH="$CARGO_HOME/bin:$PATH"
    command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
        echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
        exit 3
    }
    unset RUST_LOG RAYON_NUM_THREADS
}

# checkout_rev <rev> <dir>: a detached worktree of <rev>, removed on exit, whose locked crates must all be in the shared cache.
# Builds go through it, never the live checkout, so editing the checkout while jobs are queued cannot break them.
checkout_rev() {
    git -C "$repo" worktree add --detach "$2" "$1" >/dev/null
    _checkouts+=("$2")
    trap _remove_checkouts EXIT
    # A rev older than the committed lock gets the checkout's, so the offline build resolves to cached versions.
    [ -f "$2/Cargo.lock" ] || cp "$repo/Cargo.lock" "$2/Cargo.lock"
    (cd "$2" && cargo fetch --offline --locked >/dev/null) || {
        echo "shared crate cache lacks a Cargo.lock entry: re-run scripts/slurm/setup-shared-toolchain.sh on a networked host" >&2
        exit 3
    }
}

_remove_checkouts() {
    local dir
    for dir in "${_checkouts[@]}"; do git -C "$repo" worktree remove --force "$dir" 2>/dev/null || true; done
    [ "${REMOVE_TARGET_DIR:-0}" = 1 ] && rm -rf "$CARGO_TARGET_DIR"
    return 0
}

# results_dir: this node's dated results directory, created.
results_dir() {
    local dir="$repo/benchmarks/results/$(date +%Y-%m-%d)-$(hostname -s)"
    mkdir -p "$dir"
    echo "$dir"
}

# test_binary <features> <test>: builds that integration test (errors stay visible) and prints its executable.
test_binary() {
    cargo test --offline --release --features "$1" -p paulistrings --test "$2" --no-run >&2
    cargo test --offline --release --features "$1" -p paulistrings --test "$2" --no-run --message-format=json \
        | python3 -c 'import json, sys
for line in sys.stdin:
    try: r = json.loads(line)
    except ValueError: continue
    if r.get("reason") == "compiler-artifact" and r["target"]["name"] == sys.argv[1] and r.get("executable"): print(r["executable"])' "$2"
}
