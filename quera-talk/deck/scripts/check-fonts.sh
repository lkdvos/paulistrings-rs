#!/usr/bin/env bash
# Fails loudly if the build would silently fall back to organic.typ's
# documented fallback chain (URW Bookman / Carlito / DejaVu Sans Mono)
# instead of the real Caprasimo / Figtree / STIX Two Math / Source Code Pro.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

found="$(typst fonts --font-path fonts 2>/dev/null)"
missing=()
for family in "Caprasimo" "Figtree" "STIX Two Math" "Source Code Pro"; do
  echo "$found" | grep -qi "^$family\$" || missing+=("$family")
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "error: fonts/ is missing (or typst cannot see): ${missing[*]}" >&2
  echo "       the build will still compile, using organic.typ's documented" >&2
  echo "       fallback chain -- see fonts/FONTS.md before shipping a PDF." >&2
  exit 1
fi

echo "ok: Caprasimo, Figtree, STIX Two Math, Source Code Pro all resolve from fonts/"
