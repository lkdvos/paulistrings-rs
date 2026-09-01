#!/usr/bin/env bash
# Wire the docs site's figures to the committed SVGs they came from.
#
# mdBook copies non-Markdown files out of its `src/` tree into the rendered
# site, and only out of that tree — a relative image path that climbs above
# `src/` renders as a broken image. So every figure on the site needs an entry
# under `docs/book/src/assets/`.
#
# Those entries are **relative symlinks into the repository**, not copies: the
# figure a page shows is byte-identical to the one committed next to the script
# that produced it, and rerunning that script updates the site with no sync
# step. mdBook resolves the link and writes a real file into the output, so the
# published site has no symlinks in it.
#
# This script is idempotent. Run it after adding a figure to a page, or after
# renaming a source directory; it rewrites every link and reports anything it
# could not find.
#
#     ./docs/sync-assets.sh
#
# Figures are referenced from the pages as `../assets/<group>/<file>.svg`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS_DIR="$REPO_ROOT/docs/book/src/assets"

# `docs/book/src/assets/<group>/<file>` is five directories below the repo root.
UP_TO_ROOT="../../../../.."

# One record per asset group: <group>|<source directory, relative to the repo
# root>|<the figures the pages embed from it>.
ASSET_GROUPS=(
    "ising-quench|crates/paulistrings/docs/examples/img|ising_quench.svg"
    "b1|examples/b1_operator_scrambling|light_cone_1d.svg support_growth.svg otoc_1d.svg convergence_panel_1d.svg butterfly_velocity_1d.svg velocity_vs_kick_angle.svg quench_observables_2d.svg light_cone_2d.svg correlator_2d.svg convergence_panel_2d.svg"
    "b2|examples/b2_noisy_verification|terms-and-time-vs-noise.svg convergence-vs-cutoff.svg observable-decay-vs-noise.svg"
    "b5|examples/b5_operator_backpropagation|depth_vs_terms.svg convergence_panel.svg"
    "b6|examples/b6_resource_probes|theta_sweep.svg depth_sweep.svg convergence_panel.svg"
    "theta-sweep|benchmarks/python/theta_sweep|error-vs-runtime.svg error-vs-min-abs-coeff.svg error-vs-max-weight.svg parity-per-layer-terms.svg term-count-vs-truncation.svg"
    "deep-trotter|benchmarks/python/deep_trotter|error-vs-runtime.svg convergence-vs-truncation.svg term-count-vs-truncation.svg parity-per-layer-terms.svg"
    "su4|benchmarks/python/su4_staircase|term_count_vs_depth.svg error_vs_runtime.svg time_memory_vs_n.svg"
    "xxz|examples/xxz_chain/figures|term-growth.svg time-memory-vs-n.svg error-vs-runtime.svg self-convergence.svg"
)

missing=0
linked=0

for entry in "${ASSET_GROUPS[@]}"; do
    group="${entry%%|*}"
    rest="${entry#*|}"
    src_dir="${rest%%|*}"
    figures="${rest#*|}"

    mkdir -p "$ASSETS_DIR/$group"

    for figure in $figures; do
        target="$REPO_ROOT/$src_dir/$figure"
        if [[ ! -f "$target" ]]; then
            printf 'MISSING: %s/%s\n' "$src_dir" "$figure" >&2
            missing=$((missing + 1))
            continue
        fi
        ln -sfn "$UP_TO_ROOT/$src_dir/$figure" "$ASSETS_DIR/$group/$figure"
        linked=$((linked + 1))
    done
done

# Any leftover link whose target has gone away is a silent broken image on the
# site, so fail on it rather than shipping it. (A link left behind by a renamed
# figure is exactly the case this catches.)
shopt -s nullglob
for link in "$ASSETS_DIR"/*/*; do
    if [[ -L "$link" && ! -e "$link" ]]; then
        printf 'DANGLING: %s\n' "${link#"$REPO_ROOT/"}" >&2
        missing=$((missing + 1))
    fi
done
shopt -u nullglob

printf 'linked %d figure(s) under docs/book/src/assets/\n' "$linked"

if (( missing > 0 )); then
    printf '%d problem(s) — the site would render a broken image\n' "$missing" >&2
    exit 1
fi
