#!/usr/bin/env bash
# Explicit commands for campaign-2026-09-11. Sourcing or running this script
# with no arguments prints usage and does nothing -- it never launches the
# expensive campaign on its own. Each stage is a separate, explicit command
# you copy-paste, per CLAUDE.md's "Submitting is the user's step, never an
# agent's".
set -euo pipefail
cd "$(dirname "$0")/../.."

usage() {
  cat <<'EOF'
reproduce.sh <stage>

  prepare-venv       Build the local .venv and PyO3 extension (needed to run
                     the Python-side tooling/tests, not needed on the compute
                     node -- campaign-genoa.sbatch builds its own venv).
  test-tooling       Run every unit test this campaign added, with no cluster
                     access needed (analysis/, jobs/, figures/).
  preflight          Run jobs/preflight.py on the CURRENT host and print its
                     report. Useful before and after an allocation lands.
  submit-c1c2        PRINT (never run) the exact sbatch command for the
                     single-node C1 smoke + C2 tolerance-sweep stage.
  submit-c3          PRINT the exact sbatch command for the C3 thread ladder.
  collect            Validate whatever raw/<date>-<node>/{runs,gates.rank-*}.jsonl
                     exist so far against analysis/validate_campaign.py.
  tables             Build the normalized CSV tables from validated raw data
                     (requires `collect` to have passed clean first).
  figures            Render the recurring figure's real stages plus the
                     compact figures from the tables built above -- refuses
                     to run against synthetic/fixture data.
EOF
}

PY=.venv/bin/python
CAMPAIGN=quera-talk-data/campaign-2026-09-11

case "${1:-}" in
  prepare-venv)
    module load modules/2.4-20250724 python/3.11.11
    PYTHON=$(command -v python3.11) ./scripts/setup.sh
    ;;

  test-tooling)
    "$PY" -m pytest "$CAMPAIGN/analysis/tests" "$CAMPAIGN/jobs/tests" "$CAMPAIGN/figures/tests" -v
    ;;

  preflight)
    "$PY" "$CAMPAIGN/jobs/preflight.py"
    ;;

  submit-c1c2)
    echo "This command is not run for you. Copy-paste it yourself once you have reviewed"
    echo "$CAMPAIGN/jobs/campaign-genoa.sbatch and the resource bounds in campaign.json:"
    echo
    echo "  env -u SBATCH_RESERVATION sbatch $CAMPAIGN/jobs/campaign-genoa.sbatch"
    echo
    echo "Record the returned job ID in $CAMPAIGN/job-ledger.jsonl (see job-ledger.md)."
    ;;

  submit-c3)
    echo "Not run for you. After C2 has told you which min_abs_coeff values are worth a full"
    echo "thread ladder, copy-paste (override THREADS/MIN_ABS_COEFF as needed):"
    echo
    echo "  env -u SBATCH_RESERVATION THREADS='1 2 4 8 16 32 48 96' \\"
    echo "    sbatch $CAMPAIGN/jobs/campaign-genoa.sbatch"
    ;;

  collect)
    shopt -s nullglob
    dirs=("$CAMPAIGN"/raw/*/)
    if [ ${#dirs[@]} -eq 0 ]; then
      echo "no raw/<date>-<node>/ output directories yet -- nothing to validate" >&2
      exit 1
    fi
    for d in "${dirs[@]}"; do
      echo "== $d"
      "$PY" "$CAMPAIGN/analysis/validate_campaign.py" "$d"/runs.jsonl "$d"/gates.rank-*.jsonl
    done
    ;;

  tables)
    echo "Not yet implemented as a CLI: analysis/normalize.py's functions are called from" >&2
    echo "Python. Run 'collect' first, then drive normalize.py's functions over the" >&2
    echo "validated raw/<date>-<node>/ records for the tables/ directory (T06's functions" >&2
    echo "take already-parsed record lists, not file paths -- see their docstrings)." >&2
    exit 2
    ;;

  figures)
    echo "Not yet wired to real data: make_recurring_figure.py/make_compact_figures.py take" >&2
    echo "normalized table rows as arguments (see figures/tests for the call shape). Once" >&2
    echo "'tables' produces real rows, call them from a short script and save under figures/" >&2
    echo "(never figures/_synth/, which is reserved for synthetic previews)." >&2
    exit 2
    ;;

  "")
    usage
    ;;

  *)
    usage
    echo "unknown stage: $1" >&2
    exit 1
    ;;
esac
