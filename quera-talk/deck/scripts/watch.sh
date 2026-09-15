#!/usr/bin/env bash
# Live-recompile on save. Usage: scripts/watch.sh [prototype|full]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-prototype}"
mkdir -p build
typst watch main.typ "build/$mode.pdf" --font-path fonts --input "mode=$mode" --root .
