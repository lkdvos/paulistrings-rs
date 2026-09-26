#!/usr/bin/env bash
# Fail unless every pair of GPUs this process sees is joined by NVLink (`NV#` in `nvidia-smi topo -m`).
# Usage: check-gpu-links.sh [--warn]; with --warn a non-NVLink pair is reported but does not fail, and GPU_TOPO_FILE reads a saved `topo -m` instead.
set -euo pipefail
warn=0
[ "${1:-}" = "--warn" ] && warn=1
topo=$( { if [ -n "${GPU_TOPO_FILE:-}" ]; then cat "$GPU_TOPO_FILE"; else nvidia-smi topo -m; fi; } | sed 's/\x1b\[[0-9;]*m//g')
# The header row names the columns; a data row is `GPUi` followed by one entry per header column.
links=$(printf '%s\n' "$topo" | awk '
    !hdr && $1 == "GPU0" && $2 !~ /^X$/ { for (j = 1; j <= NF; j++) col[j] = $j; ncol = NF; hdr = 1; next }
    hdr && $1 ~ /^GPU[0-9]+$/ { for (j = 1; j <= ncol; j++) if (col[j] ~ /^GPU[0-9]+$/ && col[j] != $1) print $1 "-" col[j] "=" $(j + 1) }')
n=$(printf '%s\n' "$topo" | awk '!seen && $1 == "GPU0" && $2 !~ /^X$/ { seen = 1; next } seen && $1 ~ /^GPU[0-9]+$/ { c++ } END { print c + 0 }')
if [ "$n" -lt 2 ]; then
    echo "gpu links: $n GPU visible, nothing to check"
    exit 0
fi
bad=$(printf '%s\n' "$links" | grep -v '=NV[0-9]*$' || true)
if [ -z "$bad" ]; then
    echo "gpu links: all $n GPUs pairwise NVLink ($(printf '%s\n' "$links" | sed 's/.*=//' | sort -u | tr '\n' ' '))"
    exit 0
fi
echo "gpu links: pairs without NVLink: $(printf '%s ' $bad)" >&2
[ "$warn" = 1 ] && exit 0
echo "gpu links: refusing to measure a multi-GPU exchange over PCIe; resubmit with --exclude=$(hostname -s) or pass --warn" >&2
exit 4
