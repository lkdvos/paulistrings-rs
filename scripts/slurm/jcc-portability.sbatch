#!/usr/bin/env bash
#SBATCH --job-name=ps-jcc
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Is `-Cllvm-args=-x86-branches-within-32B-boundaries` (in .cargo/config.toml)
# a win, a wash, or a cost on THIS node type?
#
# The flag mitigates the JCC erratum (SKX102), which exists only on
# Skylake-derived Intel parts. On AMD Zen and on Ice Lake and later it is pure
# overhead: ~2% more instructions for a hazard the core does not have. The
# reference workstation (ccqlin038, Cascade Lake) measures -9..-13% wall with
# it; `ccq` is mostly rome/genoa, where the sign could plausibly invert.
# See research/notes/2026-09-10-hot-path-code-size.md.
#
# DELIBERATELY NO --constraint: the whole point is to run this on several node
# types. Supply one at submit time and set ISA_TAG to match, e.g.
#
#   for c in rome genoa icelake; do
#     ISA_TAG=$c env -u SBATCH_RESERVATION sbatch --constraint="$c&rocky9" \
#       scripts/slurm/jcc-portability.sbatch
#   done
#
# (The workstation exports a stale SBATCH_RESERVATION=rocky9; `env -u` drops it.)
#
# This job doubles as the quiet-box MULTI-THREAD campaign: THREADS defaults to
# 1 and the node's physical core count, which is the datum the shared-workstation
# runs could not resolve (every 16-thread arm there read "no consistent change"
# on wall while its phase deltas held direction).
#
# Environment knobs (all optional):
#   ISA_TAG   label for the report (default: derived from the CPU model)
#   LAYERS    probe layers, space separated (default "rotation_zz cnot trotter su4")
#   NS        --n values (default "1000000")
#   THREADS   thread counts (default: "1 <physical cores>")
#   N_PER_THREAD  if set, --n becomes N_PER_THREAD x threads for each cell,
#             overriding NS. This is what a multi-thread campaign wants: a fixed
#             --n across thread counts starves the workers and measures
#             scheduling noise, not the kernel. Jobs 7018021-3 ran that way and
#             produced ZERO direction-consistent multi-thread wall cells, with
#             spreads to -33%..+33%, at ~10-20k terms per worker.
#             1000000 reproduces the 1-thread cell's per-worker work.
#             DO NOT use it with su4: su4 closes at ~14x its --n, so
#             N_PER_THREAD=1000000 on a 96-core node asks for ~1.4e9 terms
#             (tens of GB). Give su4 its own run at a fixed small NS; it is the
#             one layer that stays informative unscaled, precisely because of
#             that closure factor.
#   REPS      --reps (default 20 — the timed region must be >=200ms; the
#             governor is not pinnable and short regions run downclocked)
#   PAIRS     pairs per cell (default 7)
#   PS_REV    commit to build (default HEAD at job start)
#
# NOTE: hardware counters do NOT work on ccq nodes (perf_event_paranoid), so
# this is wall-clock paired only — no DSB share, no branch-miss attribution.
# Counter work stays on the workstation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
    echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
    exit 3
}
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
unset RUST_LOG
unset RAYON_NUM_THREADS

echo "== node $(hostname -s), job ${SLURM_JOB_ID:-?}, $(date -Is)"
lscpu | grep -E 'Model name|Vendor|CPU family|^Model:|Stepping|Socket|NUMA|Thread|^CPU\(s\)'
echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
echo "microcode: $(grep -m1 microcode /proc/cpuinfo || echo n/a)"
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"

# Does the JCC erratum apply here? Intel family 6, Skylake-derived models only.
vendor=$(grep -m1 vendor_id /proc/cpuinfo | awk '{print $3}')
family=$(grep -m1 'cpu family' /proc/cpuinfo | awk '{print $4}')
model=$(grep -m1 '^model[[:space:]]*:' /proc/cpuinfo | awk '{print $3}')
affected=no
if [ "$vendor" = "GenuineIntel" ] && [ "$family" = 6 ]; then
    case "$model" in
        78|94|85|142|158|165|166) affected=yes ;;   # SKL, SKX/CLX/CPX, KBL/CFL/WHL, CML
    esac
fi
echo "JCC erratum (SKX102) applies on this part: $affected  (vendor=$vendor family=$family model=$model)"
echo "  -> expect the padding to WIN if yes, and to cost ~2% instructions for nothing if no"

phys=$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)
: "${THREADS:=1 $phys}"
: "${LAYERS:=rotation_zz cnot trotter su4}"
: "${NS:=1000000}"
: "${REPS:=20}"
: "${PAIRS:=7}"
: "${ISA_TAG:=$(lscpu | sed -n 's/^Model name: *//p' | tr ' ' '_' | head -c 40)}"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "ISA_TAG=$ISA_TAG phys=$phys threads='$THREADS' rev=$PS_REV"
if [ -n "${N_PER_THREAD:-}" ]; then
    echo "N_PER_THREAD=$N_PER_THREAD (--n scales with the thread count; NS ignored)"
    case " $LAYERS " in
        *" su4 "*|*" su4_local "*)
            echo "!! su4 closes at ~14x --n, so N_PER_THREAD would ask for ~$(( N_PER_THREAD * phys * 14 )) terms." >&2
            echo "!! Run su4 in its own job at a fixed NS instead. Refusing." >&2
            exit 6 ;;
    esac
fi

# Build both arms from a worktree of PS_REV, never the live checkout.
wt=$(mktemp -d "${TMPDIR:-/tmp}/ps-jcc-wt.XXXXXX")
git worktree add --detach -q "$wt" "$PS_REV" >/dev/null
trap 'git worktree remove --force "$wt" 2>/dev/null || true' EXIT
cp Cargo.lock "$wt/Cargo.lock" 2>/dev/null || true

out=benchmarks/results/$(date +%F)-$(hostname -s)
mkdir -p "$out"
bin_pad="$out/jcc-${ISA_TAG}-padded"
bin_raw="$out/jcc-${ISA_TAG}-unpadded"

# The shared registry goes stale whenever Cargo.lock moves (jobs 7017996-8 died
# this way: the cache held zerocopy 0.8.48 against a lock pinning 0.8.57). Check
# once, cheaply, and say exactly what to do rather than failing inside a build.
if ! ( cd "$wt" && cargo metadata --offline --format-version 1 >/dev/null 2>&1 ); then
    echo "!! the shared cargo registry ($CARGO_HOME) cannot resolve Cargo.lock offline." >&2
    echo "!! Refresh it on a networked host (the workstation), then resubmit:" >&2
    echo "!!   RUSTUP_HOME=\$HOME/.local/rust-shared/rustup CARGO_HOME=\$HOME/.local/rust-shared/cargo cargo fetch" >&2
    exit 5
fi

# Separate target dirs: the two arms differ only in rustflags, and sharing a
# dir would make each build invalidate the other's fingerprint for no reason.
# The padding is no longer in .cargo/config.toml — the shipped default is the
# portable one and hosts with the erratum opt in (scripts/jcc-rustflags.sh), so
# the PADDED arm is now the one that adds a flag and UNPADDED is the default
# build. This inverted when the config changed; the size guard below is what
# catches it if the two ever stop differing.
echo "== building PADDED (default + the JCC padding flag)"
( cd "$wt" && CARGO_TARGET_DIR="$CARGO_TARGET_DIR-pad" \
    RUSTFLAGS="-Cllvm-args=-x86-branches-within-32B-boundaries" \
    cargo build --offline --release --features phase-timing --example phase_breakdown )
cp "$CARGO_TARGET_DIR-pad/release/examples/phase_breakdown" "$bin_pad"

# UNPADDED is simply the shipped default now. (Historical note, because it cost
# three exclusive-node allocations: when the flag *was* in the config, the
# obvious `cargo --config 'rustflags = []'` did not remove it — cargo MERGES
# config arrays rather than replacing them, so an array cannot be cleared that
# way. Jobs 7018011-3 died on that, caught by the size guard below.)
echo "== building UNPADDED (shipped default, no flag)"
( cd "$wt" && CARGO_TARGET_DIR="$CARGO_TARGET_DIR-raw" \
    cargo build --offline --release --features phase-timing --example phase_breakdown )
cp "$CARGO_TARGET_DIR-raw/release/examples/phase_breakdown" "$bin_raw"

cmp -s "$bin_pad" "$bin_raw" && {
    echo "!! the two builds are byte-identical — the padding override did not take effect."
    echo "!! Nothing to measure; fix the override before trusting any result." >&2
    exit 4
}
sz_pad=$(stat -c%s "$bin_pad"); sz_raw=$(stat -c%s "$bin_raw")
echo "padded=$sz_pad B  unpadded=$sz_raw B  (delta $((sz_pad - sz_raw)) B)"
if [ "$sz_pad" -le "$sz_raw" ]; then
    echo "!! the padded build is not larger than the unpadded one." >&2
    echo "!! Padding inserts prefixes and nops, so this means the arms are mislabelled" >&2
    echo "!! or the flag is not doing what we think. Refusing to report a direction." >&2
    exit 4
fi

for layer in $LAYERS; do
  for n_spec in $NS; do
    for t in $THREADS; do
      # N_PER_THREAD keeps the per-worker work fixed across thread counts.
      if [ -n "${N_PER_THREAD:-}" ]; then n=$(( N_PER_THREAD * t )); else n=$n_spec; fi
      a="$out/jcc-${ISA_TAG}-${layer}-n${n}-t${t}-unpadded.jsonl"
      b="$out/jcc-${ISA_TAG}-${layer}-n${n}-t${t}-padded.jsonl"
      rm -f "$a" "$b"
      probe="--n $n --qubits 128 --layers $layer --threads $t --reps $REPS"
      echo "== $layer n=$n t=$t : $PAIRS pairs, abba"
      for p in $(seq 1 "$PAIRS"); do
        if (( p % 2 )); then order="a b"; else order="b a"; fi
        for side in $order; do
          case $side in
            a) "$bin_raw" $probe --format json --json-out "$a" >/dev/null ;;
            b) "$bin_pad" $probe --format json --json-out "$b" >/dev/null ;;
          esac
        done
      done
      python3 scripts/ab-report.py "$a" "$b" --all-phases \
          --label-a "unpadded" --label-b "padded (shipping)" || true
    done
  done
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out/"
echo
echo "READING THIS: B (padded) lower = the shipping config is right for this node type."
echo "B consistently HIGHER means the flag should be narrowed from cfg(target_arch)"
echo "to something that excludes this part — see the note's follow-up section."
