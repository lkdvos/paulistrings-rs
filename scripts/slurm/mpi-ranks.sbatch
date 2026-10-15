#!/usr/bin/env bash
#SBATCH --job-name=ps-mpi
#SBATCH --partition=ccq
#SBATCH --constraint=icelake&rocky9
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# The MPI transport's multi-rank differential net (`crates/paulistrings/tests/mpi_ranks.rs`,
# behind the `mpi` cargo feature) on a real multi-node allocation.
#
# One rank per NUMA domain: the engine holds one partition per process and takes its placement from the launcher's affinity mask, so `--cpu-bind=ldoms` pins the run.
# The rank count must be a power of two (a partition is named by log2(P) GF(2) rows), so scripts/slurm/ranks.sh rounds `nodes × domains` down and says what it picked.
#
# Environment knobs (all optional): PROBE=0 runs the net alone; LAYERS (default cnot,su4,trotter), N (default 3000000), PROBE_ARGS; PS_REV as in ab-campaign.sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
. scripts/slurm/common.sh

module purge
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"   # bindgen (rsmpi) at build time
use_shared_rust
# Every rank execs the binary from its cwd, so the build and its checkout live on the shared filesystem, not node-local /tmp.
export CARGO_TARGET_DIR="${PAULISTRINGS_BUILD_ROOT:-$HOME/.cache/paulistrings-build}/job-${SLURM_JOB_ID:-local}"
REMOVE_TARGET_DIR=1
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
checkout_rev "$PS_REV" "$CARGO_TARGET_DIR/src"
out=$(results_dir)
cd "$CARGO_TARGET_DIR/src"
. scripts/jcc-rustflags.sh   # research/FINDINGS.md §JCC branch padding, opt-in
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"

echo "== job ${SLURM_JOB_ID:-?} on $SLURM_JOB_NODELIST, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
source scripts/slurm/ranks.sh

bin=$(test_binary mpi mpi_ranks)
[ -n "$bin" ] || { echo "could not build the mpi_ranks test binary" >&2; exit 1; }
echo "== binary: $bin"

srun --ntasks="$ranks" --ntasks-per-node="$per_node" --cpus-per-task="$cpus_per_numa" \
     --cpu-bind=ldoms --mpi=pmix "$bin"

# The capacity probe, same placement: per-rank phase breakdown and peak RSS including the exchange transients.
# Each rank writes its own `<path>.rank<N>` sidecar.
if [ "${PROBE:-1}" = "1" ]; then
    cargo build --offline --release --features phase-timing,mpi -p paulistrings \
        --example phase_breakdown >/dev/null
    probe="$CARGO_TARGET_DIR/release/examples/phase_breakdown"
    sidecar="$out/mpi-ranks-${SLURM_JOB_ID:-local}.jsonl"
    echo "== probe: $ranks ranks, $cpus_per_numa threads each -> $sidecar.rank*"
    srun --ntasks="$ranks" --ntasks-per-node="$per_node" --cpus-per-task="$cpus_per_numa" \
         --cpu-bind=ldoms --mpi=pmix "$probe" --mpi --threads "$cpus_per_numa" \
         --layers "${LAYERS:-cnot,su4,trotter}" --n "${N:-3000000}" \
         --json-out "$sidecar" ${PROBE_ARGS:-}
fi
