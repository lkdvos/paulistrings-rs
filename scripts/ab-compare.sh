#!/usr/bin/env bash
# Interleaved A/B measurement harness for paulistrings-rs.
#
# Usage: scripts/ab-compare.sh <name> --a <git-rev|.> --b <git-rev|.> \
#            --probe '<phase_breakdown args>' [--pairs N] [--order abab|abba] \
#            [--features <cargo features>] [--keep-worktrees]
#
# Why this exists: this box's single-shot campaign noise is ~±5-8% at 1
# thread and ~±10-26% at 8/32 threads -- untouched code has moved that much
# between campaigns (see benchmarks/PROFILING.md, Interleaved A/B protocol).
# Comparing "one campaign per build" therefore cannot resolve the ~5-10%
# effects the engine work actually produces. This script implements the
# protocol that can: build *both* sides up front into two separate
# binaries, then alternate them adjacent in time for several pairs, and
# report paired per-run deltas (scripts/ab-report.py) rather than a
# difference of two independently-noisy campaign means.
#
# Writes into benchmarks/results/<date>-<hostname>/:
#   <name>-ab.log              provenance header + every run's stdout + report
#   <name>-a.probe.jsonl       phase_breakdown --json-out sidecar, side A
#   <name>-b.probe.jsonl       ditto, side B
#   <name>-{a,b}-<sha|worktree>[-dirty]   the exact binaries that were run
# Pre-existing sidecars from an earlier run of the same name/day are rotated
# aside (never appended to), because the report pairs runs by their order of
# appearance in the file.
#
# Design notes:
#   - Probe args are passed through *verbatim* (word-split via `eval` into a
#     bash array, so quoting inside --probe works). No flags are injected --
#     not even --seed: the probe already defaults to a fixed seed, and adding
#     a flag the pinned binary might not accept would break the A side, the
#     B side, or both. If you want a specific seed, put it in --probe.
#   - A failing single run is logged and the campaign continues; ab-report.py
#     pairs up to the minimum run count per cell, so a lost run costs one
#     pair, not the whole comparison. Build failures, malformed usage and
#     unresolvable revs abort -- those are caller mistakes, not flaky
#     measurements.
#   - Each side builds in its own tree (a detached `git worktree` for a rev,
#     the working tree for `.`), so the two builds cannot evict each other
#     from a shared target dir mid-campaign. The working tree's Cargo.lock,
#     if present, is copied into each worktree so the two sides differ only
#     in source, not in resolved dependency versions.
#
# Never invokes any Slurm command.

set -euo pipefail

cd "$(dirname "$0")/.."

ORIG_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage: scripts/ab-compare.sh <name> --a <git-rev|.> --b <git-rev|.> \
           --probe '<phase_breakdown args>' [options]

Builds two phase_breakdown binaries and alternates them adjacent in time,
then reports paired per-run deltas. This is the protocol for effects
smaller than this host's campaign-to-campaign noise (±5-8% at 1 thread,
±10-26% at 8/32 threads).

Required:
  <name>                  Campaign name; prefixes every output file.
  --a <git-rev|.>         Baseline side. A git revision (tag, branch, SHA,
                          HEAD~1, ...) or "." for the current working tree
                          (uncommitted changes included).
  --b <git-rev|.>         Candidate side, same syntax.
  --probe '<args>'        Arguments for the phase_breakdown probe, passed
                          through verbatim, e.g.
                            --probe '--n 1000000 --threads 1,32 --layers rotation_zz'
                          See `cargo run --release --features phase-timing \
                          --example phase_breakdown -- --help`.

Options:
  --pairs N               Number of A/B pairs to run (default: 3). Direction
                          consistency across all N pairs is the acceptance
                          criterion -- see the report's closing note.
  --order abab|abba       abab (default): every pair runs A then B.
                          abba: alternate the within-pair order per pair, so
                          a monotone drift in machine state cannot masquerade
                          as a consistent B-is-faster signal.
  --features <list>       Cargo features for both builds
                          (default: phase-timing -- the probe requires it).
  --keep-worktrees        Don't remove the temporary git worktrees after
                          building (for inspecting exactly what was built).

Outputs (benchmarks/results/<date>-<hostname>/):
  <name>-ab.log, <name>-{a,b}.probe.jsonl, <name>-{a,b}-<sha|worktree>[-dirty]

The report can be re-run at any time on the archived sidecars:
  python3 scripts/ab-report.py <name>-a.probe.jsonl <name>-b.probe.jsonl \
      [--field wall_ns] [--all-phases]

Never invokes any Slurm command.
EOF
}

die() {
  echo "error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

if [[ $# -eq 0 ]]; then
  # A bare invocation is malformed, not a help request: every argument this
  # script needs is one you cannot guess a default for.
  die "no arguments (need at least <name> --a REV --b REV --probe 'ARGS')"
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

name=$1
shift
if [[ -z "$name" || "$name" == -* ]]; then
  die "first argument must be a campaign name, got '${name}'"
fi

a_rev=""
b_rev=""
probe=""
pairs=3
order="abab"
features="phase-timing"
keep_worktrees=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --a | --b | --probe | --pairs | --order | --features)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --a) a_rev=$2 ;;
        --b) b_rev=$2 ;;
        --probe) probe=$2 ;;
        --pairs) pairs=$2 ;;
        --order) order=$2 ;;
        --features) features=$2 ;;
      esac
      shift 2
      ;;
    --keep-worktrees)
      keep_worktrees=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument '$1'"
      ;;
  esac
done

[[ -n "$a_rev" ]] || die "--a is required"
[[ -n "$b_rev" ]] || die "--b is required"
[[ -n "$probe" ]] || die "--probe is required (the phase_breakdown args to measure)"
[[ "$pairs" =~ ^[1-9][0-9]*$ ]] || die "--pairs must be a positive integer, got '${pairs}'"
[[ "$order" == "abab" || "$order" == "abba" ]] || die "--order must be abab or abba, got '${order}'"
[[ -n "$features" ]] || die "--features must not be empty (the probe requires phase-timing)"

# Verbatim pass-through of the probe args: `eval` so that quoting *inside*
# --probe is honored, unlike bench-campaign.sh's plain word-splitting.
eval "probe_args=( $probe )"
if [[ ${#probe_args[@]} -eq 0 ]]; then
  die "--probe expanded to no arguments"
fi

out_dir="benchmarks/results/$(date +%F)-$(hostname -s)"
mkdir -p "$out_dir"
log_file="$out_dir/${name}-ab.log"
sidecar_a="$out_dir/${name}-a.probe.jsonl"
sidecar_b="$out_dir/${name}-b.probe.jsonl"
wt_root="target/ab-worktrees"

log() {
  printf '%s\n' "$*" | tee -a "$log_file"
}

# ---------------------------------------------------------------------------
# Temporary worktree bookkeeping
# ---------------------------------------------------------------------------

worktrees=()

cleanup_worktrees() {
  if [[ $keep_worktrees -eq 1 ]]; then
    return 0
  fi
  local d
  for d in ${worktrees[@]+"${worktrees[@]}"}; do
    [[ -n "$d" && -d "$d" ]] || continue
    # --force: the build left untracked artifacts (target/) behind, which
    # plain `git worktree remove` refuses to discard.
    git worktree remove --force "$d" >/dev/null 2>&1 || rm -rf "$d"
  done
}
trap cleanup_worktrees EXIT

# ---------------------------------------------------------------------------
# Revision resolution
# ---------------------------------------------------------------------------

tree_is_dirty() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo yes
  else
    echo no
  fi
}

# Validated before any substitution below, so an unresolvable rev prints the
# usage heredoc and exits nonzero from the *script*, not from a subshell.
require_rev() {
  local rev=$1
  if [[ "$rev" == "." ]]; then
    return 0
  fi
  if ! git rev-parse --verify --quiet "${rev}^{commit}" >/dev/null; then
    die "cannot resolve revision '${rev}' to a commit (--a/--b take a git rev or '.')"
  fi
}

resolve_sha() {
  local rev=$1
  if [[ "$rev" == "." ]]; then
    git rev-parse HEAD
  else
    git rev-parse --verify "${rev}^{commit}"
  fi
}

require_rev "$a_rev"
require_rev "$b_rev"
a_sha=$(resolve_sha "$a_rev")
b_sha=$(resolve_sha "$b_rev")
wt_dirty=$(tree_is_dirty)
a_dirty=no
b_dirty=no
[[ "$a_rev" == "." ]] && a_dirty=$wt_dirty
[[ "$b_rev" == "." ]] && b_dirty=$wt_dirty

# ---------------------------------------------------------------------------
# Provenance header
# ---------------------------------------------------------------------------

if [[ -f "$log_file" ]]; then
  {
    echo
    echo "############################################################"
    echo "# RERUN: $(date -Iseconds)"
    echo "############################################################"
    echo
  } >>"$log_file"
fi

hostname_full=$(hostname -f 2>/dev/null || hostname)
governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')

log "${name} interleaved A/B comparison — ${hostname_full}"
log "date: $(date -Iseconds)"
log "command: $(printf '%q ' "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"})"
log "side A: rev '${a_rev}' -> ${a_sha} (dirty: ${a_dirty})"
log "side B: rev '${b_rev}' -> ${b_sha} (dirty: ${b_dirty})"
log "probe args: ${probe_args[*]}"
log "pairs: ${pairs}   order: ${order}   features: ${features}"
log "$(rustc -V)"
log "threads (nproc): $(nproc)"
log "governor: ${governor}"
log "cpu: ${cpu_model}"
log "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# Smoke mode: identical trees on both sides measure the harness, not a change.
if [[ "$a_rev" == "." && "$b_rev" == "." ]]; then
  log "WARNING: both sides are the current working tree — this measures this"
  log "         host's own noise floor (smoke mode), not a code change."
elif [[ "$a_sha" == "$b_sha" && "$a_dirty" == "no" && "$b_dirty" == "no" ]]; then
  log "WARNING: both sides resolve to the same clean commit ${a_sha} — this"
  log "         measures this host's own noise floor, not a code change."
fi
log ""

# ---------------------------------------------------------------------------
# Build phase
# ---------------------------------------------------------------------------

# build_side <a|b> <rev> <dirty:yes|no> <sha>: builds that side and sets the
# global `built_bin` to the archived binary path. Deliberately *not* a
# command substitution -- its progress has to reach the log live, and an
# `exit 1` inside `$(...)` would only kill the subshell.
built_bin=""
build_side() {
  local side=$1 rev=$2 dirty=$3 sha=$4
  local short tag src dest wt
  built_bin=""

  if [[ "$rev" == "." ]]; then
    tag="worktree"
  else
    short=$(git rev-parse --short "$sha")
    tag="$short"
  fi
  [[ "$dirty" == "yes" ]] && tag="${tag}-dirty"
  dest="$out_dir/${name}-${side}-${tag}"

  if [[ "$rev" == "." ]]; then
    log "build[${side}]: current working tree (${sha}, dirty=${dirty}) in ./target"
    if ! (cargo build --offline --release --features "$features" \
      -p paulistrings --example phase_breakdown) 2>&1 | tee -a "$log_file"; then
      log "error: build of side ${side} (working tree) failed — aborting"
      exit 1
    fi
    src="target/release/examples/phase_breakdown"
  else
    wt="${wt_root}/${name}-${side}-$$"
    mkdir -p "$wt_root"
    log "build[${side}]: detached worktree at ${sha} -> ${wt}"
    if ! (git worktree add --detach "$wt" "$sha") 2>&1 | tee -a "$log_file"; then
      log "error: git worktree add failed for side ${side} (${rev}) — aborting"
      exit 1
    fi
    worktrees+=("$wt")
    if [[ -f Cargo.lock ]]; then
      cp Cargo.lock "$wt/Cargo.lock"
      log "  seeded Cargo.lock from the working tree (both sides resolve the same deps)"
    fi
    # Subshell cd rather than --manifest-path so the *checked-out* rev's
    # rust-toolchain.toml selects the compiler, and so the build lands in
    # the worktree's own target dir.
    if ! (cd "$wt" && cargo build --offline --release --features "$features" \
      -p paulistrings --example phase_breakdown) 2>&1 | tee -a "$log_file"; then
      log "error: build of side ${side} (${rev} @ ${sha}) failed — aborting"
      exit 1
    fi
    src="$wt/target/release/examples/phase_breakdown"
  fi

  if [[ ! -x "$src" ]]; then
    log "error: expected binary '${src}' after building side ${side} — aborting"
    exit 1
  fi
  cp "$src" "$dest"
  log "  archived: ${dest}"
  built_bin="$dest"
}

log "=== build phase ==="
build_side a "$a_rev" "$a_dirty" "$a_sha"
bin_a="$built_bin"
build_side b "$b_rev" "$b_dirty" "$b_sha"
bin_b="$built_bin"

if [[ $keep_worktrees -eq 0 ]]; then
  cleanup_worktrees
  worktrees=()
  log "  temporary worktrees removed (pass --keep-worktrees to keep them)"
elif [[ ${#worktrees[@]} -gt 0 ]]; then
  log "  temporary worktrees kept: ${worktrees[*]}"
else
  log "  no temporary worktrees were created (both sides built from the working tree)"
fi
log ""

# Both binaries now exist, so nothing after this point rebuilds anything --
# that is the whole point: the two builds cannot interleave with the runs.

# ---------------------------------------------------------------------------
# Run phase
# ---------------------------------------------------------------------------

# The report pairs runs by order of appearance per cell, so a stale sidecar
# would silently shift the pairing. Rotate instead of appending.
stamp=$(date +%Y%m%dT%H%M%S)
for sidecar in "$sidecar_a" "$sidecar_b"; do
  if [[ -e "$sidecar" ]]; then
    mv "$sidecar" "${sidecar}.prev-${stamp}"
    log "note: rotated pre-existing $(basename "$sidecar") to $(basename "$sidecar").prev-${stamp}"
  fi
done

run_side() {
  local side=$1 bin=$2 sidecar=$3 pair=$4
  log "--- pair ${pair}: side ${side} ($(basename "$bin")) ---"
  if ! ("$bin" ${probe_args[@]+"${probe_args[@]}"} --json-out "$sidecar") \
    2>&1 | tee -a "$log_file"; then
    log "  (side ${side} run in pair ${pair} exited nonzero — continuing; the"
    log "   report pairs up to the minimum run count per cell)"
  fi
}

log "=== run phase: ${pairs} pair(s), order ${order} ==="
for ((p = 1; p <= pairs; p++)); do
  log ""
  log "=== pair ${p}/${pairs} — $(date -Iseconds) — load: $(cut -d' ' -f1-3 /proc/loadavg) ==="
  if [[ "$order" == "abba" && $((p % 2)) -eq 0 ]]; then
    run_side b "$bin_b" "$sidecar_b" "$p"
    run_side a "$bin_a" "$sidecar_a" "$p"
  else
    run_side a "$bin_a" "$sidecar_a" "$p"
    run_side b "$bin_b" "$sidecar_b" "$p"
  fi
done
log ""

# ---------------------------------------------------------------------------
# Report phase
# ---------------------------------------------------------------------------

log "=== report ==="
if ! (python3 scripts/ab-report.py "$sidecar_a" "$sidecar_b" --all-phases \
  --label-a "A=${a_rev}" --label-b "B=${b_rev}") 2>&1 | tee -a "$log_file"; then
  log "  (ab-report.py exited nonzero — the sidecars are intact, re-run it by hand:"
  log "   python3 scripts/ab-report.py '${sidecar_a}' '${sidecar_b}' --all-phases)"
fi

log ""
log "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
log "log:      ${log_file}"
log "sidecars: ${sidecar_a}"
log "          ${sidecar_b}"
log "binaries: ${bin_a}"
log "          ${bin_b}"
