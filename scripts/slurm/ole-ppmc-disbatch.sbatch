#!/usr/bin/env bash
#SBATCH --job-name=ole-ppmc
#SBATCH --partition=ccq
#SBATCH --constraint=rome&rocky9
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Independent PP-MC trajectories of the 56-qubit OLE (examples/b8_ole), run by disBatch inside this allocation.
# One task line = one run_b8.py call over a block of seeds; `--ntasks` tasks run concurrently, each with `--cpus-per-task` threads.
# Size `--ntasks` so that ntasks x ~3 x 32 B x cache fits in node memory (cache 5e8: ~48 GB per task).
#
# TASKS   task file from examples/b8_ole/make_tasks.py (required)
# VENV    virtualenv with the paulistrings extension built from the commit under test (default ./.venv-mpi, which also serves non-MPI runs)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 openmpi/5.0.6 python-mpi/3.12.9 disBatch
source "${VENV:-.venv-mpi}/bin/activate"
unset RUST_LOG

TASKS=${TASKS:?set TASKS to a make_tasks.py task file}
echo "== job ${SLURM_JOB_ID:-?} on ${SLURM_JOB_NODELIST:-?}, $(date -Is)"
echo "== paulistrings rev $(git rev-parse HEAD), $(wc -l < "$TASKS") tasks from $TASKS"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
free -g | head -2

disBatch -e --prefix "benchmarks/results/disbatch-${SLURM_JOB_ID:-local}-" "$TASKS"
