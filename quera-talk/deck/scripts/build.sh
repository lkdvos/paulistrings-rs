#!/usr/bin/env bash
# Compile the deck. Usage: scripts/build.sh [prototype|full] [--strict]
#
# prototype (default) -> the 6 polished concepts only, fast iteration.
# full                -> all 37 conceptual slides.
# --strict            -> also run scripts/check-figures.sh --release, which
#                        fails if a required real figure asset is still a
#                        placeholder. Never pass this while benchmark data is
#                        intentionally pending -- that failure is expected,
#                        not a scaffold defect.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mode="${1:-prototype}"
strict=false
for arg in "$@"; do
  [[ "$arg" == "--strict" ]] && strict=true
done

if [[ "$mode" != "prototype" && "$mode" != "full" ]]; then
  echo "usage: scripts/build.sh [prototype|full] [--strict]" >&2
  exit 1
fi

mkdir -p build

echo "==> checking fonts actually resolve (not the documented fallback chain)"
scripts/check-fonts.sh

echo "==> compiling mode=$mode"
typst compile main.typ "build/$mode.pdf" --font-path fonts --input "mode=$mode" --root .

pages=$(pdfinfo "build/$mode.pdf" 2>/dev/null | awk '/^Pages:/ {print $2}')
echo "==> build/$mode.pdf: $pages pages"

if $strict; then
  echo "==> --strict: checking figures"
  scripts/check-figures.sh --release
fi
