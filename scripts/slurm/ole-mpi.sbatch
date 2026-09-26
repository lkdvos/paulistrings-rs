#!/usr/bin/env bash
#SBATCH --job-name=ole-mpi
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# The 56-qubit OLE (examples/b8_ole) with the sum distributed one partition per NUMA domain across nodes (DistributedSum over MPI).
# Seeds run one after another inside the allocation; every rank takes part in every trajectory.
# Memory: ~3 x 32 B x cache in total, spread over the ranks.
#
# POLICY  ppmc (default) | approx_topn | coeff | none     CACHE  cache size or ApproxTopN budget (default 8e9)
# ALPHAS  comma list of tracker alphas (default 0.15)     L      Floquet layers per half (default 6)
# SEEDS   'lo:hi' (default 0:4)                            EPS    optional coefficient threshold
# EXACT   1 to also compute the exact rotated_overlap      OUT    results directory (default benchmarks/results/ole)
# VENV    MPI-enabled virtualenv (default ./.venv-mpi; see CLAUDE.md "Commands")
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7 python-mpi/3.12.9
source "${VENV:-.venv-mpi}/bin/activate"
unset RUST_LOG
python -c 'import paulistrings, sys; sys.exit(0 if paulistrings.mpi_available() else 1)' || {
    echo "the extension in ${VENV:-.venv-mpi} was built without the mpi feature; rebuild it (CLAUDE.md, Commands)" >&2
    exit 3
}

echo "== job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-?}, $(date -Is), rev $(git rev-parse HEAD)"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
numa=$(ls -d /sys/devices/system/node/node* | wc -l)
cpus_per_numa=$(( SLURM_CPUS_ON_NODE / numa ))
nodes=${SLURM_JOB_NUM_NODES:-1}
# The rank count must be a power of two (log2(P) GF(2) partition rows).
total=$(( nodes * numa ))
ranks=1
while [ $(( ranks * 2 )) -le "$total" ]; do ranks=$(( ranks * 2 )); done
per_node=$(( ranks / nodes ))
[ "$per_node" -lt 1 ] && { per_node=1; ranks=$nodes; }
echo "== $ranks ranks ($per_node per node), $cpus_per_numa threads each"

extra=()
[ -n "${EPS:-}" ] && extra+=(--eps "$EPS")
[ "${EXACT:-0}" = "1" ] && extra+=(--exact-overlap)
IFS=, read -ra alphas <<< "${ALPHAS:-0.15}"
for alpha in "${alphas[@]}"; do
    RAYON_NUM_THREADS=$cpus_per_numa srun --ntasks="$ranks" --ntasks-per-node="$per_node" \
        --cpus-per-task="$cpus_per_numa" --cpu-bind=ldoms --mpi=pmix \
        python examples/b8_ole/run_b8.py --mpi --alpha "$alpha" --L "${L:-6}" \
        --policy "${POLICY:-ppmc}" --cache "${CACHE:-8e9}" --seeds "${SEEDS:-0:4}" \
        --out "${OUT:-benchmarks/results/ole}" "${extra[@]}"
done
