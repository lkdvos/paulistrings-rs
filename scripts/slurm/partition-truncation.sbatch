#!/usr/bin/env bash
#SBATCH --job-name=partition-truncation
#SBATCH --partition=ccq
#SBATCH --constraint=icelake&rocky9
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Accuracy of per-partition truncation against the global budget on the Benchmark C kicked Ising point (issue #12, part 3): benchmarks/python/bench_partition_truncation.py.
# Partitions are contiguous slices of the node's CPUs, so P = 16 on an icelake node runs four threads per partition.
#
# VENV    virtualenv with the extension built from the commit under test (default ./.venv-mpi, from scripts/sync-mpi-venv.sh)
# ARGS    extra driver arguments, e.g. ARGS="--steps 5 --budgets 1e4 1e5"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 openmpi/5.0.6 python-mpi/3.12.9
source "${VENV:-.venv-mpi}/bin/activate"
unset RUST_LOG RAYON_NUM_THREADS

python -c 'from paulistrings import truncation; truncation.partition_approx_topn' 2>/dev/null || {
    echo "the extension in ${VENV:-.venv-mpi} predates partition_approx_topn; rebuild it with scripts/sync-mpi-venv.sh" >&2
    exit 3
}

out="benchmarks/results/$(date +%Y-%m-%d)-$(hostname -s)/partition-truncation-${SLURM_JOB_ID:-local}.jsonl"
echo "== job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-?}, $(date -Is), rev $(git rev-parse HEAD) -> $out"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
# shellcheck disable=SC2086
python benchmarks/python/bench_partition_truncation.py --out "$out" ${ARGS:-}
