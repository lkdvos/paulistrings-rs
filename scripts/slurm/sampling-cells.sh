# Sourced by sampling-gpus.sbatch and sampling-mpi-gpu.sbatch: the sampled-run sweep of `examples/sampling_bench`, one JSON line per cell.
# The caller sets the arrays `launch_one` (one device, the reference) and `launch_all` (the whole allocation), each a command line that sampling_bench's arguments are appended to, and `$sidecar`.
# Never invokes any Slurm command itself.

# Kicked Ising (20 steps) and the 16x16 TFIM (10 steps) under CollapseSample at samples = cache / 4, seed 1.
: "${CACHES:=1e8 4e8}"
: "${REF_CACHE:=1e8}"
: "${SEED:=1}"
: "${CELL_BOUND:=2400}"

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
    # The deterministic scaling cell: the same ApproxTopN cap on one device and on all of them.
    cell "reference kicked-ising approx-topn" launch_one --model kicked-ising --steps 20 --approx-topn "$REF_CACHE"
    cell "split kicked-ising approx-topn" launch_all --model kicked-ising --steps 20 --approx-topn "$REF_CACHE"
}
