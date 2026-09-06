#!/usr/bin/env bash
# The whole data campaign behind presentation/figures. Appends JSON lines to
# presentation/data/*.jsonl (each file gets one '# provenance:' header line when
# created). Stages are selectable: STAGES="ladder scaling" ./collect_all.sh
#
# Protocol (CLAUDE.md §Performance discipline): RUST_LOG unset, quiet box,
# dedicated Rayon pool per cell, one warm-up, every repetition recorded.
set -euo pipefail
cd "$(dirname "$0")/../../.."
unset RUST_LOG
B0=presentation/bench/target/release/presentation-bench
D=presentation/data
mkdir -p "$D"
STAGES=${STAGES:-"tests build calibration ladder scaling scaling_large sweep memory"}
STEPS=${STEPS:-10}                  # working point: 10 Trotter steps (2710 layers) ...
EPS=${EPS:-2.44140625e-4}           # ... at 2^-12: 1.07e6 peak terms, ~half the layers above 1e5 terms
EPS_LARGE=${EPS_LARGE:-1.220703125e-4}  # 2^-13 at 10 steps: 3.9e6 peak terms, the "large m" arm
STEPS_LARGE=${STEPS_LARGE:-10}
COARSE="--target-bucket-len 16384 --min-buckets 64"
B="$B0 "; run() { $B0 "$1" --steps "$STEPS" "${@:2}"; }

have() { [[ " $STAGES " == *" $1 "* ]]; }
load1() { cut -d' ' -f1 /proc/loadavg; }
header() {  # header FILE
  if [[ ! -s "$1" ]]; then
    echo "# provenance: $(date -Is) host=$(hostname -s) commit=$(git rev-parse --short HEAD)$(git diff --quiet -- crates presentation/bench || echo -dirty) rustc=$(rustc -V | cut -d' ' -f2) load=$(cut -d' ' -f1-3 /proc/loadavg) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)" > "$1"
  fi
}
stage() { echo; echo "=== stage $1  ($(date +%T), load $(load1))"; }

if (( $(echo "$(load1) > 2.0" | bc -l) )) && [[ -z "${FORCE:-}" ]]; then
  echo "load $(load1) > 2.0 — box is not quiet; set FORCE=1 to run anyway" >&2; exit 1
fi

if have tests; then stage tests
  cargo test --release --manifest-path presentation/bench/Cargo.toml --quiet 2>&1 | tail -3
fi
if have build; then stage build
  cargo build --release --manifest-path presentation/bench/Cargo.toml --quiet
  RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=presentation/bench/target-native \
    cargo build --release --manifest-path presentation/bench/Cargo.toml --quiet
fi
if have calibration; then stage calibration
  f=$D/calibration.jsonl; header "$f"
  for k in 12 13 14 15 16; do eps=$(python3 -c "print(2**-$k)")
    $B0 bucketed --terms-only --steps 5 --eps "$eps" | sed 's/"terms_out":\[[^]]*\]/"terms_out":null/' >> "$f"; done
  for k in 11 12 13; do eps=$(python3 -c "print(2**-$k)")
    $B0 bucketed --terms-only --steps 10 --eps "$eps" | sed 's/"terms_out":\[[^]]*\]/"terms_out":null/' >> "$f"; done
  grep -o '"eps":[^,]*,"steps":[0-9]*,"layers":[0-9]*,"peak_terms":[0-9]*' "$f"
fi
if have ladder; then stage ladder
  f=$D/engine_ladder.jsonl; header "$f"
  # naive is ~50 s/run here; threadmaps/mergesort come from the scaling stage (fig1 reads both files).
  [[ -n "${SKIP_NAIVE:-}" ]] || run naive --threads 1 --reps 3 --eps $EPS --json-out "$f" | grep '^cell'
  run bucketed   --threads 1,32     --reps 5 --eps $EPS --json-out "$f" | grep '^cell'
  run bucketed   --threads 32       --reps 5 --eps $EPS $COARSE --json-out "$f" | grep '^cell'
fi
if have scaling; then stage scaling
  f=$D/thread_scaling.jsonl; header "$f"
  T=1,2,4,8,16,32
  # The reconstructed baselines run for minutes per propagation at one thread: two reps, no warm-up.
  run threadmaps --threads $T --reps 2 --no-warmup --eps $EPS --json-out "$f" | grep '^cell'
  run mergesort  --threads $T --reps 2 --no-warmup --eps $EPS --json-out "$f" | grep '^cell'
  run bucketed   --threads $T --reps 5 --eps $EPS --json-out "$f" | grep '^cell'
  run bucketed   --threads $T --reps 5 --eps $EPS $COARSE --json-out "$f" | grep '^cell'
fi
if have scaling_large; then stage scaling_large
  f=$D/thread_scaling_large.jsonl; header "$f"
  T=1,2,4,8,16,32
  $B0 bucketed --threads $T --reps 3 --steps $STEPS_LARGE --eps $EPS_LARGE --json-out "$f" | grep '^cell'
  $B0 bucketed --threads $T --reps 3 --steps $STEPS_LARGE --eps $EPS_LARGE $COARSE --json-out "$f" | grep '^cell'
fi
if have sweep; then stage sweep
  f=$D/bucket_sweep.jsonl; header "$f"
  for cell in "256 128" "1024 128" "4096 128" "16384 64" "65536 16" "262144 16"; do set -- $cell
    run bucketed --threads 1,16 --reps 5 --eps $EPS --target-bucket-len $1 --min-buckets $2 --json-out "$f" | grep '^cell' | sed "s/^/target=$1 min=$2 /"
  done
fi
if have memory; then stage memory
  f=$D/memory.jsonl; header "$f"
  run naive      --threads 1  --reps 1 --eps $EPS --json-out "$f" | grep '^cell'
  run threadmaps --threads 8  --reps 1 --eps $EPS --json-out "$f" | grep '^cell'
  run mergesort  --threads 8  --reps 1 --eps $EPS --json-out "$f" | grep '^cell'
  run bucketed   --threads 1  --reps 1 --eps $EPS --json-out "$f" | grep '^cell'
  run bucketed   --threads 32 --reps 1 --eps $EPS --json-out "$f" | grep '^cell'
fi
echo; echo "done $(date +%T)"
