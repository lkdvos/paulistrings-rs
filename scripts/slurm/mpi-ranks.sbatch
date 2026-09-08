#!/usr/bin/env bash
#SBATCH --job-name=ps-mpi
#SBATCH --partition=ccq
#SBATCH --constraint=icelake&rocky9
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/results/slurm-%j.out
#
# Phase-3 template: multi-rank differential test + the domains-vs-ranks probe comparison.
# Requires the `mpi` cargo feature (rsmpi) and `scripts/mpi-test.sh` from phase 3 of the plan;
# until they land this script only prints the topology and exits 2.
#
# Layouts it exercises (one rank per NUMA domain, threads pinned inside the domain by the engine):
#   ranks-per-node = <numa domains>, domains-per-rank D = 1      (pure MPI)
#   ranks-per-node = 1,              domains-per-rank D = <numa>  (hybrid: in-process domains + MPI across nodes)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"   # bindgen (rsmpi) at build time
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
unset RUST_LOG

echo "== job ${SLURM_JOB_ID:-?} on $SLURM_JOB_NODELIST, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
numa=$(ls -d /sys/devices/system/node/node* | wc -l)
cpus_per_numa=$(( SLURM_CPUS_ON_NODE / numa ))
echo "numa domains per node: $numa, cpus per domain: $cpus_per_numa"

if [ ! -f scripts/mpi-test.sh ]; then
    echo "phase 3 has not landed (scripts/mpi-test.sh missing); nothing to run" >&2
    exit 2
fi

cargo build --offline --release --features mpi -p paulistrings --tests >/dev/null
bin=$(cargo test --offline --release --features mpi -p paulistrings --test mpi_ranks --no-run \
      --message-format=json 2>/dev/null \
      | python3 -c 'import json,sys
for l in sys.stdin:
    r=json.loads(l)
    if r.get("reason")=="compiler-artifact" and r["target"]["name"]=="mpi_ranks" and r.get("executable"): print(r["executable"])')

echo "== pure MPI: $numa ranks/node, D=1"
srun --ntasks-per-node="$numa" --cpus-per-task="$cpus_per_numa" --cpu-bind=ldoms --mpi=pmix "$bin"
echo "== hybrid: 1 rank/node, D=$numa"
PAULISTRINGS_DOMAINS="$numa" srun --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_ON_NODE" \
    --cpu-bind=none --mpi=pmix "$bin"
