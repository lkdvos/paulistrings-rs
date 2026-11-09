# Sourced by sampling-gpus.sbatch and sampling-mpi-gpu.sbatch: the sampled-run sweep of `examples/sampling_bench`, one JSON line per cell.
# The caller sets the arrays `launch_one` (one device, the reference) and `launch_all` (the whole allocation), each a command line that sampling_bench's arguments are appended to, and `$sidecar`.
# Never invokes any Slurm command itself.

# Kicked Ising (20 steps) and the 16x16 TFIM (10 steps) under CollapseSample at samples = cache / 4, seed 1, under ApproxTopN at the same caps, and kicked Ising under coefficient thresholds.
: "${CACHES:=1e8 4e8}"
: "${REF_CACHE:=1e8}"
: "${SEED:=1}"
: "${THRESHOLDS:=2.5e-5 1e-5}"
: "${CELL_BOUND:=2400}"
# The row-choice sweep (SWEEP=rows): split rows to compare, the partition count of the split, and NCCL_P2P_LEVEL values to repeat its collapse and threshold cells under.
: "${ROWS:=cut quadrant xparity:2}"
: "${P2P_LEVELS:=}"

samples_of() { python3 -c "import sys; print(int(float(sys.argv[1]) / 4))" "$1"; }

# cell <label> <launcher> <args...>: one bounded run whose failure is reported and does not stop the sweep.
cell() {
    local label=$1
    local -n launcher=$2
    shift 2
    echo "== $label: $*"
    bounded "$CELL_BOUND" "${launcher[@]}" "$@" --json-out "$sidecar" \
        || { failed=$?; echo "== $label FAILED (exit $failed)"; leftover_gpu_processes; }
}

run_sweep() {
    local model steps cache
    for spec in "kicked-ising 20" "tfim 10"; do
        read -r model steps <<< "$spec"
        cell "reference $model" launch_one --model "$model" --steps "$steps" \
            --collapse --cache "$REF_CACHE" --samples "$(samples_of "$REF_CACHE")" --seed "$SEED"
        for cache in $CACHES; do
            cell "split $model cache $cache" launch_all --model "$model" --steps "$steps" \
                --collapse --cache "$cache" --samples "$(samples_of "$cache")" --seed "$SEED"
        done
    done
    # Seeded rows against the default cut, at the reference size.
    cell "split kicked-ising seeded rows" launch_all --rows seeded --model kicked-ising --steps 20 \
        --collapse --cache "$REF_CACHE" --samples "$(samples_of "$REF_CACHE")" --seed "$SEED"
    # The deterministic scaling cells: the same ApproxTopN cap on one device and on all of them, then the larger caps split.
    for spec in "kicked-ising 20" "tfim 10"; do
        read -r model steps <<< "$spec"
        cell "reference $model approx-topn $REF_CACHE" launch_one --model "$model" --steps "$steps" --approx-topn "$REF_CACHE"
        for cache in $CACHES; do
            cell "split $model approx-topn $cache" launch_all --model "$model" --steps "$steps" --approx-topn "$cache"
        done
    done
    # Thresholds keep every term above eps, so reference and split read out alike to rounding.
    for eps in $THRESHOLDS; do
        cell "reference kicked-ising threshold $eps" launch_one --model kicked-ising --steps 20 --threshold "$eps"
        cell "split kicked-ising threshold $eps" launch_all --model kicked-ising --steps 20 --threshold "$eps"
    done
}

# run_rows_sweep <partitions>: one-device references, then each split cell under every row choice in ROWS (quadrant only at 4 partitions, x-parity rows only on kicked Ising, whose site-centred light cone they balance),
# under the global ApproxTopN cap and its per-partition share (PerPartition(ApproxTopN(cap / P)), no layer collective), the thresholds and the collapse.
run_rows_sweep() {
    local partitions=$1 rows eps level share
    share=$(python3 -c "import sys; print(int(float(sys.argv[1]) / int(sys.argv[2])))" "$REF_CACHE" "$partitions")
    local collapse=(--collapse --cache "$REF_CACHE" --samples "$(samples_of "$REF_CACHE")" --seed "$SEED")
    cell "reference kicked-ising approx-topn $REF_CACHE" launch_one --model kicked-ising --steps 20 --approx-topn "$REF_CACHE"
    cell "reference kicked-ising collapse $REF_CACHE" launch_one --model kicked-ising --steps 20 "${collapse[@]}"
    for eps in $THRESHOLDS; do
        cell "reference kicked-ising threshold $eps" launch_one --model kicked-ising --steps 20 --threshold "$eps"
    done
    cell "reference tfim approx-topn $REF_CACHE" launch_one --model tfim --steps 10 --approx-topn "$REF_CACHE"
    for rows in $ROWS; do
        [ "$rows" = quadrant ] && [ "$partitions" -ne 4 ] && continue
        cell "split kicked-ising approx-topn $REF_CACHE rows $rows" launch_all --rows "$rows" --model kicked-ising --steps 20 --approx-topn "$REF_CACHE"
        cell "split kicked-ising partition-approx-topn $share rows $rows" launch_all --rows "$rows" --model kicked-ising --steps 20 --partition-approx-topn "$share"
        cell "split kicked-ising collapse $REF_CACHE rows $rows" launch_all --rows "$rows" --model kicked-ising --steps 20 "${collapse[@]}"
        for eps in $THRESHOLDS; do
            cell "split kicked-ising threshold $eps rows $rows" launch_all --rows "$rows" --model kicked-ising --steps 20 --threshold "$eps"
        done
        case "$rows" in xparity:*|zparity:*) continue ;; esac
        cell "split tfim approx-topn $REF_CACHE rows $rows" launch_all --rows "$rows" --model tfim --steps 10 --approx-topn "$REF_CACHE"
        cell "split tfim partition-approx-topn $share rows $rows" launch_all --rows "$rows" --model tfim --steps 10 --partition-approx-topn "$share"
    done
    for level in $P2P_LEVELS; do
        export NCCL_P2P_LEVEL=$level
        for rows in $ROWS; do
            [ "$rows" = quadrant ] && [ "$partitions" -ne 4 ] && continue
            cell "split kicked-ising collapse $REF_CACHE rows $rows NCCL_P2P_LEVEL=$level" launch_all --rows "$rows" --model kicked-ising --steps 20 "${collapse[@]}"
            cell "split kicked-ising threshold 1e-5 rows $rows NCCL_P2P_LEVEL=$level" launch_all --rows "$rows" --model kicked-ising --steps 20 --threshold 1e-5
        done
        unset NCCL_P2P_LEVEL
    done
}

# phase_timing_cell <features> <sidecar>: with PT=1 (the default), the largest kicked-Ising cell again under `phase-timing` and `--trace`, for the export and exchange split and the per-partition totals; the other cells are untraced.
phase_timing_cell() {
    [ "${PT:-1}" = "1" ] || return 0
    local largest
    largest=$(tr ' ' '\n' <<< "$CACHES" | sort -g | tail -1)
    cargo build --offline --release -p paulistrings --features "$1" --example sampling_bench
    sidecar=$2
    cell "phase-timing split kicked-ising cache $largest" launch_all --model kicked-ising --steps 20 --trace \
        --collapse --cache "$largest" --samples "$(samples_of "$largest")" --seed "$SEED"
}
