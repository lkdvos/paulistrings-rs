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
# One rank per NUMA domain, D = 1: the engine holds exactly one partition per process and takes
# its placement from the launcher's affinity mask, so `--cpu-bind=ldoms` is what pins the run
# (`Placement::Auto` then resolves to a single slot covering the domain). There is no
# domains-per-rank hybrid; `PAULISTRINGS_DOMAINS` no longer exists.
#
# The rank count must be a power of two — a partition is named by log2(P) GF(2) rows — so the
# script rounds `nodes × domains` down and says what it picked.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

module purge
module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"   # bindgen (rsmpi) at build time
# Rust from the shared filesystem: ~/.cargo and ~/.rustup are symlinks into the workstation's
# local /home, absent on cluster nodes. scripts/slurm/setup-shared-toolchain.sh populates this.
SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
    echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
    exit 3
}
export CARGO_TARGET_DIR="${TMPDIR:-/tmp}/paulistrings-target-${SLURM_JOB_ID:-local}"
unset RUST_LOG

echo "== job ${SLURM_JOB_ID:-?} on $SLURM_JOB_NODELIST, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|^CPU\(s\)'
numa=$(ls -d /sys/devices/system/node/node* | wc -l)
cpus_per_numa=$(( SLURM_CPUS_ON_NODE / numa ))
nodes=${SLURM_JOB_NUM_NODES:-1}
echo "numa domains per node: $numa, cpus per domain: $cpus_per_numa, nodes: $nodes"

# Round the total rank count down to a power of two, then spread it evenly over the nodes.
total=$(( nodes * numa ))
ranks=1
while [ $(( ranks * 2 )) -le "$total" ]; do ranks=$(( ranks * 2 )); done
per_node=$(( ranks / nodes ))
[ "$per_node" -lt 1 ] && { per_node=1; ranks=$nodes; }
echo "== running $ranks ranks ($per_node per node, D=1), from $total available domains"

bin=$(cargo test --offline --release --features mpi -p paulistrings --test mpi_ranks --no-run \
      --message-format=json 2>/dev/null \
      | python3 -c 'import json,sys
for l in sys.stdin:
    try: r=json.loads(l)
    except ValueError: continue
    if r.get("reason")=="compiler-artifact" and r["target"]["name"]=="mpi_ranks" and r.get("executable"): print(r["executable"])')
[ -n "$bin" ] || { echo "could not build the mpi_ranks test binary" >&2; exit 1; }
echo "== binary: $bin"

srun --ntasks="$ranks" --ntasks-per-node="$per_node" --cpus-per-task="$cpus_per_numa" \
     --cpu-bind=ldoms --mpi=pmix "$bin"

# The capacity probe, same placement: per-rank phase breakdown and peak RSS including the
# exchange transients (the metric phase 3 exists for). Only if the probe was built with --mpi.
probe=$CARGO_TARGET_DIR/release/examples/phase_breakdown
if [ -x "$probe" ]; then
    echo "== probe: $ranks ranks, D=1"
    srun --ntasks="$ranks" --ntasks-per-node="$per_node" --cpus-per-task="$cpus_per_numa" \
         --cpu-bind=ldoms --mpi=pmix "$probe" --mpi --threads "$cpus_per_numa" \
         --json-out "benchmarks/results/mpi-${SLURM_JOB_ID:-local}.json"
fi
