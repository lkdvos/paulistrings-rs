#!/usr/bin/env bash
# Build or refresh the MPI venv (default ./.venv-mpi) with uv: the extension with `--features mpi`, plus mpi4py compiled against the loaded MPI.
#
#   scripts/sync-mpi-venv.sh                     # ./.venv-mpi, release profile
#   scripts/sync-mpi-venv.sh --cuda              # `--features mpi,cuda`
#   scripts/sync-mpi-venv.sh --debug             # dev profile, as `scripts/mpi-test.sh --python` builds without --release
#   scripts/sync-mpi-venv.sh --venv DIR          # somewhere else
#
# The interpreter is $PYTHON, else `python3` on PATH, which must be the one the MPI modules provide (CLAUDE.md, Commands); this script loads no modules.
# The install is non-editable, so the venv pins the commit it was built from and never touches the in-tree extension ./.venv's editable install uses.
# Re-run it after any change, and use `.venv-mpi/bin/python` rather than `uv run`, which would re-sync the venv as an editable default build.
# uv caches the mpi4py build, so after switching MPI implementations run `uv cache clean mpi4py` first.
set -euo pipefail
cd "$(dirname "$0")/.."

venv="$PWD/.venv-mpi"
features="mpi"
profile=""
while [ $# -gt 0 ]; do
    case "$1" in
        --venv) venv="$2"; shift 2 ;;
        --venv=*) venv="${1#*=}"; shift ;;
        --cuda) features="mpi,cuda"; shift ;;
        --debug) profile="--profile dev"; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

command -v uv >/dev/null 2>&1 || { echo "uv not found (module load uv)" >&2; exit 2; }
command -v mpicc >/dev/null 2>&1 || { echo "mpicc not found (module load openmpi/5.0.6); rsmpi and mpi4py both build against it" >&2; exit 2; }
if [ -z "${LIBCLANG_PATH:-}" ] && ! ldconfig -p 2>/dev/null | grep -q libclang; then
    echo "LIBCLANG_PATH is unset and no libclang is on the loader path (module load llvm/19.1.7; export LIBCLANG_PATH=\$(llvm-config --libdir))" >&2
    exit 2
fi
python="${PYTHON:-$(command -v python3 || true)}"
[ -n "$python" ] || { echo "no python3 on PATH (module load python-mpi/3.12.9, or set PYTHON)" >&2; exit 2; }

echo "== syncing $venv: --features $features ${profile:-(release)}, $python"
export UV_PROJECT_ENVIRONMENT="$venv" MATURIN_PEP517_ARGS="--features $features${profile:+ $profile}"
# Two steps because the extension builds without isolation against the venv's maturin, which uv before 0.8 does not install first.
uv sync --no-editable --extra mpi --python "$python" --no-install-project
uv sync --no-editable --extra mpi --python "$python" --reinstall-package paulistrings
