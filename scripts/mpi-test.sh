#!/usr/bin/env bash
# Run the MPI transport's differential net (`tests/mpi_ranks.rs`) at several
# rank counts under `mpirun`.
#
#   scripts/mpi-test.sh                        # 2 and 4 ranks, debug profile
#   scripts/mpi-test.sh --ranks 2,4,8          # more
#   scripts/mpi-test.sh --oversubscribe        # more ranks than cores (CI, a busy box)
#   scripts/mpi-test.sh --release              # the shipping codegen
#
# The rank count must be a power of two: a partition is named by log2(P) GF(2)
# rows (ARCHITECTURE.md §Partitioning), and the test binary refuses anything
# else with exit 2.
#
# This script loads no modules. It needs `mpicc` (for rsmpi's build probe),
# `LIBCLANG_PATH` (for its bindgen) and `mpirun` on PATH, and says how to get
# them if they are missing. On a Slurm node use `scripts/slurm/mpi-ranks.sbatch`
# instead, which drives the same binary through `srun --mpi=pmix`.
set -euo pipefail
cd "$(dirname "$0")/.."

ranks="2,4"
profile=""
oversubscribe=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ranks) ranks="$2"; shift 2 ;;
        --ranks=*) ranks="${1#*=}"; shift ;;
        --oversubscribe) oversubscribe=1; shift ;;
        --release) profile="--release"; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

missing=0
command -v mpicc  >/dev/null 2>&1 || { echo "mpicc not found (rsmpi's build script probes it)" >&2; missing=1; }
command -v mpirun >/dev/null 2>&1 || { echo "mpirun not found" >&2; missing=1; }
if [ -z "${LIBCLANG_PATH:-}" ] && ! ldconfig -p 2>/dev/null | grep -q libclang; then
    echo "LIBCLANG_PATH is unset and no libclang is on the loader path (rsmpi runs bindgen)" >&2
    missing=1
fi
if [ "$missing" -ne 0 ]; then
    cat >&2 <<'EOF'

On a Flatiron host:
    module load modules/2.4-20250724 openmpi/5.0.6 llvm/19.1.7
    export LIBCLANG_PATH=$(llvm-config --libdir)
Elsewhere: install an MPI (libopenmpi-dev / openmpi-bin) and libclang-dev.
EOF
    exit 2
fi

echo "== building the mpi_ranks test binary ${profile:-(debug)}"
bin=$(cargo test -p paulistrings --features mpi --test mpi_ranks --no-run \
        ${profile:+$profile} --message-format=json 2>/dev/null \
      | python3 -c 'import json, sys
for line in sys.stdin:
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    if rec.get("reason") == "compiler-artifact" \
       and rec["target"]["name"] == "mpi_ranks" and rec.get("executable"):
        print(rec["executable"])')
if [ -z "$bin" ]; then
    echo "could not find the mpi_ranks executable; rebuilding with output:" >&2
    cargo test -p paulistrings --features mpi --test mpi_ranks --no-run ${profile:+$profile} >&2 || true
    exit 1
fi
echo "   $bin"

# Shared-memory and oversubscription knobs. `vader_single_copy_mechanism=none`
# turns off CMA/XPMEM cross-process copies, which container runtimes and
# hardened kernels forbid (`ptrace_scope`), so the run falls back to a copy
# through shared memory; `rmaps_base_oversubscribe` lets a CI runner with two
# cores host four ranks. Harmless on a workstation, load-bearing on a runner.
export OMPI_MCA_btl_vader_single_copy_mechanism=none
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl_base_warn_component_unused=0

flags=""
[ "$oversubscribe" -eq 1 ] && flags="--oversubscribe"

status=0
for n in ${ranks//,/ }; do
    echo "== mpirun -n $n $flags"
    if mpirun -n "$n" $flags "$bin"; then
        echo "== $n ranks: ok"
    else
        echo "== $n ranks: FAILED" >&2
        status=1
    fi
done
exit "$status"
