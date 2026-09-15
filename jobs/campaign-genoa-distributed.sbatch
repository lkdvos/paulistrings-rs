#!/usr/bin/env bash
#SBATCH --job-name=quera-c4-node
#SBATCH --partition=ccq
#SBATCH --constraint=genoa&rocky9
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=quera-talk-data/campaign-2026-09-11/raw/slurm-%j.out
#
# Multi-node distributed (MPI, `comm=`) stage for campaign-2026-09-11 (C4,
# E6/E7): one partition per rank, **one rank per node** (D=1, spanning both
# NUMA domains). `comm=`'s placement (`paulistrings::mpi::default_config`) is
# `Placement::Auto{max_partitions: Some(1)}`, which resolves to a single slot
# over whatever CPUs are in the process's affinity mask
# (`engine/partitioned/topology.rs::resolve_auto`): when that mask spans both
# of a node's NUMA domains, the two are merged into one un-pinned-to-a-node
# slot (`node: None`, so `bind_memory` is a no-op) with `threads = ` the full
# core count. So node-granularity placement needs no engine change at all —
# only `--cpus-per-task`/`--cpu-bind` here, giving each rank the whole node
# instead of one domain. This supersedes this file's earlier one-rank-per-
# NUMA-domain placement (`--cpu-bind=ldoms`, mirroring
# scripts/slurm/mpi-ranks.sbatch): half as many ranks for the same node
# count, each rank twice the size. Runs the real canonical task through
# `run_cell_distributed.py`, not the differential-test binary.
# `variant_id=bucketed_current` only, same deferral as `campaign-genoa.sbatch`.
#
# Multi-node builds must land on the SHARED filesystem, not node-local /tmp
# (mpi-ranks.sbatch's note on jobs 7008425/26): every rank execs the binary/
# venv from its own node, so CARGO_TARGET_DIR and the venv both live under
# $HOME/.cache, keyed by job ID.
#
# Does NOT call sbatch anywhere — submission is the user's step:
#
#   env -u SBATCH_RESERVATION sbatch quera-talk-data/campaign-2026-09-11/jobs/campaign-genoa-distributed.sbatch
#   env -u SBATCH_RESERVATION --nodes=4 sbatch .../campaign-genoa-distributed.sbatch   # 4 nodes -> more ranks
#
# Environment knobs (all optional):
#   PS_REV          commit to build (default: HEAD when the job starts), via a shared-FS worktree.
#   MIN_ABS_COEFF   ONE dyadic value (default: 2^-16, the same point campaign-genoa.sbatch already
#                    trusts, so the first distributed cell is an "overlap" comparison, not a blind
#                    leap to an unmeasured tolerance -- see contract.md's overlap-before-capacity rule).
#   TROTTER_STEPS   headline depth (default: 20).
#   THETA_H         kick angle (default: 7pi/32).
#   REPS            repetitions (default: 1 -- an MPI allocation is expensive to repeat blindly;
#                    bump once the first point's cost is known).
#   PARTITION_ROW_POLICY   "random" (default, today's behavior exactly) or "cut". E8
#                    (decisions.md #13/2026-09-13): `run_cell_distributed.py`'s `comm=` path has
#                    no explicit-rows plumbing yet (`sum.rs::parse_run_mode` refuses
#                    `partition_row_seed=`/`partition_row_blocks=` together with `comm=`), so
#                    "cut" is rejected below with a clear error rather than silently running
#                    "random" under a "cut" label. A real distributed E8 comparison needs that
#                    MPI-side plumbing built first; the in-process partitioned engine
#                    (`campaign-genoa.sbatch`/`run_cell.py`, no MPI needed) already supports both
#                    policies today.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"

module purge
module load modules/2.4-20250724
module load openmpi/5.0.6
module load llvm/19.1.7
module load python-mpi/3.12.9
export LIBCLANG_PATH="${LIBCLANG_PATH:-$(llvm-config --libdir)}"   # bindgen (rsmpi) at build time
PYTHON_BIN=$(command -v python3.12)

# Rust from the shared filesystem: ~/.cargo and ~/.rustup are symlinks into the workstation's
# local /home, absent on cluster nodes. scripts/slurm/setup-shared-toolchain.sh populates this.
SHARED_RUST=${SHARED_RUST:-$HOME/.local/rust-shared}
export RUSTUP_HOME="$SHARED_RUST/rustup" CARGO_HOME="$SHARED_RUST/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo >/dev/null && cargo --version >/dev/null 2>&1 || {
    echo "cargo not usable on this node (SHARED_RUST=$SHARED_RUST); run scripts/slurm/setup-shared-toolchain.sh on a networked host first" >&2
    exit 3
}

# shellcheck disable=SC1091
source scripts/jcc-rustflags.sh
export RUSTFLAGS="${RUSTFLAGS:-}${RUSTFLAGS:+ }${JCC_RUSTFLAGS}"

echo "== job ${SLURM_JOB_ID:-?} on $SLURM_JOB_NODELIST, $(date -Is)"
lscpu | grep -E 'Model name|Socket|NUMA|Thread|^CPU\(s\)'
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg)"
lscpu | grep -E 'Thread\(s\) per core'   # contract.md: verify, never trust Slurm's ThreadsPerCore alone

# --- worktree build on the SHARED filesystem (mpi-ranks.sbatch's lesson) ---
export CARGO_TARGET_DIR="${PAULISTRINGS_BUILD_ROOT:-$HOME/.cache/paulistrings-build}/job-${SLURM_JOB_ID:-local}"
mkdir -p "$CARGO_TARGET_DIR"
PS_REV=${PS_REV:-$(git rev-parse HEAD)}
echo "building rev $PS_REV ($(git log -1 --format=%s "$PS_REV"))"
repo=$PWD
src="$CARGO_TARGET_DIR/src"
git worktree add --detach "$src" "$PS_REV" >/dev/null
trap 'cd "$repo" && git worktree remove --force "$src" 2>/dev/null; rm -rf "$CARGO_TARGET_DIR"' EXIT
[ -f "$repo/Cargo.lock" ] && cp "$repo/Cargo.lock" "$src/Cargo.lock"

venv="$CARGO_TARGET_DIR/venv"
"$PYTHON_BIN" -m venv --system-site-packages "$venv"
# shellcheck disable=SC1091
source "$venv/bin/activate"
"$venv/bin/python" -c "import mpi4py" || {
    echo "mpi4py not visible in $venv even with --system-site-packages; check the python-mpi module" >&2
    exit 3
}
(cd "$src" && maturin develop --release --features mpi -m crates/paulistrings-py/Cargo.toml)

# --- placement: RANK_GRANULARITY selects one rank per NODE (default, both NUMA
# domains in one rank -- see decision #53) or one rank per NUMA DOMAIN (the
# original placement, kept for direct scaling comparisons -- decision #54 found
# the two are NOT equivalent at matched rank/node count, so both stay live). ---
: "${RANK_GRANULARITY:=node}"   # "node" (default) or "domain"
nodes=${SLURM_JOB_NUM_NODES:-1}
case "$RANK_GRANULARITY" in
  node)
    cpus_per_rank=$SLURM_CPUS_ON_NODE
    ranks=1
    while [ $(( ranks * 2 )) -le "$nodes" ]; do ranks=$(( ranks * 2 )); done
    per_node=1
    cpu_bind=none
    out_tag="distributed-node"
    config_suffix="-1rank-per-node"
    ;;
  domain)
    numa=$(ls -d /sys/devices/system/node/node* 2>/dev/null | wc -l)
    [ "$numa" -lt 1 ] && numa=1
    cpus_per_rank=$(( SLURM_CPUS_ON_NODE / numa ))
    total=$(( nodes * numa ))
    ranks=1
    while [ $(( ranks * 2 )) -le "$total" ]; do ranks=$(( ranks * 2 )); done
    per_node=$(( ranks / nodes ))
    [ "$per_node" -lt 1 ] && { per_node=1; ranks=$nodes; }
    cpu_bind=ldoms
    out_tag="distributed"
    config_suffix=""
    ;;
  *)
    echo "RANK_GRANULARITY must be 'node' or 'domain', got '$RANK_GRANULARITY'" >&2
    exit 3
    ;;
esac
echo "granularity=$RANK_GRANULARITY cpus per rank: $cpus_per_rank, nodes: $nodes"
echo "== running $ranks ranks ($per_node per node), from $nodes nodes available"

: "${MIN_ABS_COEFF:=1.5258789e-05}"     # 2^-16, the point campaign-genoa.sbatch already trusts
: "${TROTTER_STEPS:=20}"
: "${THETA_H:=0.6872233929727672}"      # 7*pi/32
: "${REPS:=1}"
: "${PARTITION_ROW_POLICY:=random}"
if [ "$PARTITION_ROW_POLICY" != "random" ]; then
    echo "PARTITION_ROW_POLICY=$PARTITION_ROW_POLICY: only 'random' is supported on the" \
         "distributed (comm=) path today -- run_cell_distributed.py has no explicit-rows" \
         "plumbing yet (see this file's header comment and decisions.md)" >&2
    exit 3
fi
echo "min_abs_coeff=$MIN_ABS_COEFF trotter_steps=$TROTTER_STEPS reps=$REPS partition_row_policy=$PARTITION_ROW_POLICY"

date_tag=$(date +%Y-%m-%d)
out_dir="$repo/quera-talk-data/campaign-2026-09-11/raw/${date_tag}-${out_tag}-${ranks}ranks"
mkdir -p "$out_dir"

cell_json="$CARGO_TARGET_DIR/cell.json"
config_id="genoa-c4-eps${MIN_ABS_COEFF}-r${ranks}${config_suffix}"
for rep in $(seq 0 $((REPS - 1))); do
  cat > "$cell_json" <<EOF
{
  "variant_id": "bucketed_current",
  "theta_h": $THETA_H,
  "trotter_steps": $TROTTER_STEPS,
  "min_abs_coeff": $MIN_ABS_COEFF,
  "direction": "heisenberg",
  "state": "z+",
  "observable": "canonical_z_127",
  "threads": $cpus_per_rank,
  "partitions": $ranks,
  "repetition_index": $rep,
  "task_id": "T02-canonical",
  "config_id": "$config_id",
  "partition_row_policy": "$PARTITION_ROW_POLICY"
}
EOF
  echo "== cell $config_id rep=$rep ($ranks ranks)"
  # cpu_bind=none (RANK_GRANULARITY=node): each rank's affinity mask must cover
  # the whole node so `default_config`'s Placement::Auto{max_partitions: Some(1)}
  # merges both NUMA domains into its one slot instead of resolving to just
  # one of them. cpu_bind=ldoms (RANK_GRANULARITY=domain) restricts each rank
  # right back to one domain, the original placement. RAYON_NUM_THREADS is set
  # for parity/safety on any non-pool rayon use before scatter; the partitioned
  # pool itself sizes from the resolved affinity mask, not this env var.
  RAYON_NUM_THREADS=$cpus_per_rank srun --ntasks="$ranks" --ntasks-per-node="$per_node" \
      --cpus-per-task="$cpus_per_rank" --cpu-bind="$cpu_bind" --mpi=pmix \
      "$venv/bin/python" "$repo/quera-talk-data/campaign-2026-09-11/jobs/run_cell_distributed.py" \
      "$cell_json" --out-dir "$out_dir" \
      || echo "  cell failed/invalid — see $out_dir/runs.jsonl status field"
done

echo "load at end: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "results: $out_dir/"
