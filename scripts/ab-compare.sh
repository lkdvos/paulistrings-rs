#!/usr/bin/env bash
# Interleaved A/B harness: build the two sides' probes up front, alternate them adjacent in time, and report paired per-run deltas (scripts/ab-report.py).
# The protocol, and why a difference of two campaign means cannot resolve the effects it measures: benchmarks/PROFILING.md §Interleaved A/B protocol.
# Probe args pass through verbatim (`eval` into an array, so quoting inside --probe survives); nothing is injected, not even --seed, because a pinned binary may not accept it.
# A failing run is logged and the campaign continues (ab-report.py pairs up to the minimum run count per cell); a failing build, malformed usage or an unresolvable rev aborts.
# Each rev builds once, in its own detached worktree, so the two builds cannot evict each other; two sides on the same clean commit (or both `.`) share one binary.
# Never invokes any Slurm command.

set -euo pipefail

. "$(dirname "$0")/jcc-rustflags.sh"   # research/FINDINGS.md §JCC branch padding, opt-in

cd "$(dirname "$0")/.."

ORIG_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage: scripts/ab-compare.sh <name> --a <git-rev|.> --b <git-rev|.> \
           --probe '<phase_breakdown args>' [options]

Required:
  <name>                  Campaign name; prefixes every output file.
  --a <git-rev|.>         Baseline side: a git revision, or "." for the
                          working tree (uncommitted changes included).
  --b <git-rev|.>         Candidate side, same syntax.
  --probe '<args>'        phase_breakdown arguments, passed verbatim, e.g.
                          '--n 1000000 --threads 1,32 --layers rotation_zz'.

Options:
  --probe-b '<args>'      Side B's probe args (default: --probe's). With the
                          same rev on both sides this is a runtime-knob A/B,
                          e.g. '--partitions 2 ...' against '--partitions 1 ...';
                          the report then pairs on layer,threads.
  --pairs N               A/B pairs to run (default 3). Direction consistency
                          across every pair is the acceptance criterion.
  --order abab|abba       abab (default) runs A then B in every pair; abba
                          alternates the order so a monotone drift cannot pose
                          as a consistent effect.
  --features <list>       Cargo features for the builds (default phase-timing).
  --bin-a <path>          Run this prebuilt probe as side A instead of building
  --bin-b <path>          --a/--b; the rev then only labels the side.
  --launch '<prefix>'     Run every probe under this launcher, e.g.
                          'mpirun -n 2' or 'srun --mpi=pmix --ntasks 4'; the
                          probe args need --mpi. Each run's per-rank sidecars
                          (<file>.rank<N>) merge into one row per cell: every
                          *_ns time and vmhwm_kb is the maximum over ranks (the
                          critical path), every other key is rank 0's.
  --keep-worktrees        Keep the temporary build worktrees.

Outputs (benchmarks/results/<date>-<host>/):
  <name>-ab.log, <name>-{a,b}.probe.jsonl, <name>-{a,b}-<sha|worktree>[-dirty|-prebuilt]

Re-run the report on the archived sidecars at any time:
  python3 scripts/ab-report.py <name>-a.probe.jsonl <name>-b.probe.jsonl \
      [--field wall_ns] [--all-phases] [--pair-on layer,threads]
EOF
}

die() {
  echo "error: $*" >&2
  echo >&2
  usage >&2
  exit 2
}

[[ $# -gt 0 ]] || die "no arguments (need at least <name> --a REV --b REV --probe 'ARGS')"
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

name=$1
shift
[[ -n "$name" && "$name" != -* ]] || die "first argument must be a campaign name, got '${name}'"

a_rev=""
b_rev=""
probe=""
probe_b=""
probe_b_given=0
pairs=3
order="abab"
features="phase-timing"
keep_worktrees=0
bin_a_given=""
bin_b_given=""
launch=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --a | --b | --probe | --probe-b | --pairs | --order | --features | --bin-a | --bin-b | --launch)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      case "$1" in
        --a) a_rev=$2 ;;
        --b) b_rev=$2 ;;
        --probe) probe=$2 ;;
        --probe-b)
          probe_b=$2
          probe_b_given=1
          ;;
        --pairs) pairs=$2 ;;
        --order) order=$2 ;;
        --features) features=$2 ;;
        --bin-a) bin_a_given=$2 ;;
        --bin-b) bin_b_given=$2 ;;
        --launch) launch=$2 ;;
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
    *) die "unknown argument '$1'" ;;
  esac
done

[[ -n "$a_rev" ]] || die "--a is required"
[[ -n "$b_rev" ]] || die "--b is required"
[[ -n "$probe" ]] || die "--probe is required (the phase_breakdown args to measure)"
[[ "$pairs" =~ ^[1-9][0-9]*$ ]] || die "--pairs must be a positive integer, got '${pairs}'"
[[ "$order" == "abab" || "$order" == "abba" ]] || die "--order must be abab or abba, got '${order}'"
[[ -n "$features" ]] || die "--features must not be empty (the probe requires phase-timing)"
for b in "$bin_a_given" "$bin_b_given"; do
  [[ -z "$b" || -x "$b" ]] || die "'$b' is not an executable"
done
[[ $probe_b_given -eq 1 ]] || probe_b="$probe"

eval "probe_args_a=( $probe )"
[[ ${#probe_args_a[@]} -gt 0 ]] || die "--probe expanded to no arguments"
eval "probe_args_b=( $probe_b )"
[[ ${#probe_args_b[@]} -gt 0 ]] || die "--probe-b expanded to no arguments"
launch_args=()
[[ -z "$launch" ]] || eval "launch_args=( $launch )"

out_dir="benchmarks/results/$(date +%F)-$(hostname -s)"
mkdir -p "$out_dir"
log_file="$out_dir/${name}-ab.log"
sidecar_a="$out_dir/${name}-a.probe.jsonl"
sidecar_b="$out_dir/${name}-b.probe.jsonl"
target_dir="${CARGO_TARGET_DIR:-target}"
wt_root="${target_dir}/ab-worktrees"

log() {
  printf '%s\n' "$*" | tee -a "$log_file"
}

worktrees=()
cleanup_worktrees() {
  [[ $keep_worktrees -eq 0 ]] || return 0
  local d
  for d in ${worktrees[@]+"${worktrees[@]}"}; do
    [[ -n "$d" && -d "$d" ]] || continue
    # --force: the build left untracked artifacts behind, which plain `git worktree remove` refuses to discard.
    git worktree remove --force "$d" >/dev/null 2>&1 || rm -rf "$d"
  done
}
trap cleanup_worktrees EXIT

tree_is_dirty() {
  if ! git diff --quiet || ! git diff --cached --quiet; then echo yes; else echo no; fi
}

resolve_sha() {
  local rev=$1
  if [[ "$rev" == "." ]]; then
    git rev-parse HEAD
  else
    git rev-parse --verify --quiet "${rev}^{commit}" \
      || die "cannot resolve revision '${rev}' to a commit (--a/--b take a git rev or '.')"
  fi
}

a_sha=$(resolve_sha "$a_rev") || exit 2
b_sha=$(resolve_sha "$b_rev") || exit 2
wt_dirty=$(tree_is_dirty)
a_dirty=no
b_dirty=no
[[ "$a_rev" == "." ]] && a_dirty=$wt_dirty
[[ "$b_rev" == "." ]] && b_dirty=$wt_dirty

# The two sides build the same thing when both are the working tree or both the same clean commit.
same_tree=0
if [[ "$a_rev" == "." && "$b_rev" == "." ]]; then
  same_tree=1
elif [[ "$a_sha" == "$b_sha" && "$a_dirty" == "no" && "$b_dirty" == "no" ]]; then
  same_tree=1
fi

if [[ -f "$log_file" ]]; then
  printf '\n############################################################\n# RERUN: %s\n############################################################\n\n' \
    "$(date -Iseconds)" >>"$log_file"
fi

log "${name} interleaved A/B comparison — $(hostname -f 2>/dev/null || hostname)"
log "date: $(date -Iseconds)"
log "command: $(printf '%q ' "$0" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"})"
log "side A: rev '${a_rev}' -> ${a_sha} (dirty: ${a_dirty})${bin_a_given:+, prebuilt ${bin_a_given}}"
log "side B: rev '${b_rev}' -> ${b_sha} (dirty: ${b_dirty})${bin_b_given:+, prebuilt ${bin_b_given}}"
log "probe args: ${probe_args_a[*]}"
log "probe args B: ${probe_args_b[*]}"
[[ -z "$launch" ]] || log "launcher: ${launch_args[*]} (per-rank rows merged: *_ns and vmhwm_kb max over ranks, the rest rank 0's)"
log "pairs: ${pairs}   order: ${order}   features: ${features}"
log "$(rustc -V)"
log "threads (nproc): $(nproc)"
log "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
log "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
log "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
if [[ $same_tree -eq 1 && $probe_b_given -eq 1 ]]; then
  log "note: both sides share one tree, and --probe-b differs by design: a runtime-knob A/B of '${probe}' vs '${probe_b}'."
elif [[ $same_tree -eq 1 ]]; then
  log "WARNING: both sides are the same tree (${a_sha}); this measures the host's noise floor, not a change."
fi
log ""

# build_side <a|b> <rev> <dirty> <sha>: builds that side and sets the global `built_bin` to the archived binary.
# Not a command substitution: its progress must reach the log live, and an `exit 1` inside `$(...)` would only end the subshell.
built_bin=""
build_side() {
  local side=$1 rev=$2 dirty=$3 sha=$4
  local tag src dest wt
  if [[ "$rev" == "." ]]; then tag="worktree"; else tag=$(git rev-parse --short "$sha"); fi
  [[ "$dirty" == "yes" ]] && tag="${tag}-dirty"
  dest="$out_dir/${name}-${side}-${tag}"

  if [[ "$rev" == "." ]]; then
    log "build[${side}]: working tree (${sha}, dirty=${dirty}) in ${target_dir}"
    (cargo build --offline --release --features "$features" -p paulistrings --example phase_breakdown) 2>&1 \
      | tee -a "$log_file" || { log "error: build of side ${side} (working tree) failed"; exit 1; }
    src="${target_dir}/release/examples/phase_breakdown"
  else
    wt="${wt_root}/${name}-${side}-$$"
    mkdir -p "$wt_root"
    log "build[${side}]: detached worktree at ${sha} -> ${wt}"
    (git worktree add --detach "$wt" "$sha") 2>&1 | tee -a "$log_file" \
      || { log "error: git worktree add failed for side ${side} (${rev})"; exit 1; }
    worktrees+=("$wt")
    if [[ ! -f "$wt/Cargo.lock" && -f Cargo.lock ]]; then
      cp Cargo.lock "$wt/Cargo.lock"
      log "  seeded Cargo.lock from the working tree (the rev predates the committed lock)"
    fi
    # cd rather than --manifest-path, so the checked-out rev's rust-toolchain.toml picks the compiler.
    (cd "$wt" && cargo build --offline --release --features "$features" -p paulistrings --example phase_breakdown) 2>&1 \
      | tee -a "$log_file" || { log "error: build of side ${side} (${rev} @ ${sha}) failed"; exit 1; }
    src="${CARGO_TARGET_DIR:-$wt/target}/release/examples/phase_breakdown"
  fi
  [[ -x "$src" ]] || { log "error: expected binary '${src}' after building side ${side}"; exit 1; }
  cp "$src" "$dest"
  log "  archived: ${dest}"
  built_bin="$dest"
}

# archive_prebuilt <a|b> <path> <sha>: copies a caller's binary next to the sidecars, so the record says exactly what ran.
archive_prebuilt() {
  local dest="$out_dir/${name}-$1-$(git rev-parse --short "$3")-prebuilt"
  cp "$2" "$dest"
  log "build[$1]: prebuilt $2, archived: ${dest}"
  built_bin="$dest"
}

log "=== build phase ==="
if [[ -n "$bin_a_given" ]]; then archive_prebuilt a "$bin_a_given" "$a_sha"; else build_side a "$a_rev" "$a_dirty" "$a_sha"; fi
bin_a="$built_bin"
if [[ -n "$bin_b_given" ]]; then
  archive_prebuilt b "$bin_b_given" "$b_sha"
  bin_b="$built_bin"
elif [[ $same_tree -eq 1 && -z "$bin_a_given" ]]; then
  bin_b="$bin_a"
  log "build[b]: same tree as side A, reusing ${bin_a}"
else
  build_side b "$b_rev" "$b_dirty" "$b_sha"
  bin_b="$built_bin"
fi
cleanup_worktrees
[[ $keep_worktrees -eq 0 ]] && worktrees=()
log ""

# The report pairs runs by order of appearance per cell, so a stale sidecar would shift the pairing: rotate it aside.
stamp=$(date +%Y%m%dT%H%M%S)
for sidecar in "$sidecar_a" "$sidecar_b"; do
  if [[ -e "$sidecar" ]]; then
    mv "$sidecar" "${sidecar}.prev-${stamp}"
    log "note: rotated pre-existing $(basename "$sidecar") to $(basename "$sidecar").prev-${stamp}"
  fi
done

# merge_ranks <run-sidecar> <side-sidecar>: appends one row per cell from <run-sidecar>.rank<N>, then removes the rank files.
merge_ranks() {
  python3 - "$1" "$2" <<'PY' 2>&1 | tee -a "$log_file"
import glob, json, re, sys
run, out = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(run + ".rank*"), key=lambda p: int(re.sub(r".*\.rank", "", p)))
ranks = [[json.loads(line) for line in open(p) if line.strip()] for p in paths]
if not ranks:
    sys.exit(f"  no {run}.rank<N> sidecars: was --mpi in the probe args?")
if len({len(rows) for rows in ranks}) != 1:
    print(f"  warning: ranks wrote {[len(rows) for rows in ranks]} rows; merging the first {min(map(len, ranks))}")
with open(out, "a") as sink:
    for cell in zip(*ranks):
        row = dict(cell[0])
        for key, value in row.items():
            if (key.endswith("_ns") or key == "vmhwm_kb") and isinstance(value, (int, float)):
                row[key] = max(r.get(key, value) for r in cell)
        sink.write(json.dumps(row) + "\n")
print(f"  merged {len(ranks)} ranks x {min(map(len, ranks))} rows into {out}")
PY
  local status=${PIPESTATUS[0]}
  rm -f "$1".rank*
  return "$status"
}

run_side() {
  local side=$1 bin=$2 sidecar=$3 pair=$4
  local -a args
  if [[ "$side" == "a" ]]; then
    args=(${probe_args_a[@]+"${probe_args_a[@]}"})
  else
    args=(${probe_args_b[@]+"${probe_args_b[@]}"})
  fi
  log "--- pair ${pair}: side ${side} ($(basename "$bin")) ---"
  if [[ -z "$launch" ]]; then
    ("$bin" "${args[@]}" --json-out "$sidecar") 2>&1 | tee -a "$log_file" \
      || log "  (side ${side} run in pair ${pair} exited nonzero; continuing)"
  else
    local run_sidecar="${sidecar%.probe.jsonl}.run.jsonl"
    rm -f "$run_sidecar".rank*
    if ("${launch_args[@]}" "$bin" "${args[@]}" --json-out "$run_sidecar") 2>&1 | tee -a "$log_file"; then
      merge_ranks "$run_sidecar" "$sidecar" || log "  (merging side ${side}'s ranks in pair ${pair} failed; continuing)"
    else
      log "  (side ${side} run in pair ${pair} exited nonzero; continuing)"
      rm -f "$run_sidecar".rank*
    fi
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

log "=== report ==="
label_b="B=${b_rev}"
report_extra_args=()
if [[ $probe_b_given -eq 1 ]]; then
  # A and B ran different probe args (typically --partitions), so the full cell key would never match.
  label_b="B=${b_rev} (--probe-b: ${probe_b})"
  report_extra_args=(--pair-on layer,threads)
fi
(python3 scripts/ab-report.py "$sidecar_a" "$sidecar_b" --all-phases \
  --label-a "A=${a_rev}" --label-b "$label_b" ${report_extra_args[@]+"${report_extra_args[@]}"}) 2>&1 \
  | tee -a "$log_file" \
  || log "  (ab-report.py exited nonzero; the sidecars are intact, re-run it by hand: python3 scripts/ab-report.py '${sidecar_a}' '${sidecar_b}' --all-phases ${report_extra_args[*]:-})"

log ""
log "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
log "log:      ${log_file}"
log "sidecars: ${sidecar_a}"
log "          ${sidecar_b}"
log "binaries: ${bin_a}"
log "          ${bin_b}"
