#!/usr/bin/env bash
# Sourced, never executed. Exports one variable:
#
#   JCC_RUSTFLAGS — the branch-padding flag when THIS CPU has the JCC erratum
#                   (SKX102), the empty string otherwise.
#
# Why this is not in .cargo/config.toml
# -------------------------------------
# `-Cllvm-args=-x86-branches-within-32B-boundaries` mitigates the JCC erratum:
# with the mitigating microcode loaded, a Skylake-derived core refuses to cache
# in the DSB any 32-byte fetch window whose jump crosses or ends on the
# boundary, and that window re-decodes through legacy MITE every iteration. On
# the reference host (ccqlin038, Cascade Lake) the flag takes DSB residency
# 45.8% -> 98.0% and is worth -9..-13% wall.
#
# The erratum is Skylake-derived Intel only. On AMD Zen and on Ice Lake and
# later the flag is pure overhead: measured across rome (Zen2), genoa (Zen4)
# and icelake (Ice Lake-SP), **13 of 13 direction-consistent phase results at 1
# thread show the padded build slower**, +0.6..+3.8%. `ccq` is entirely made of
# those parts. So the shipped default is portable — no flag — and the hosts
# that benefit opt in.
#
# Measurement scripts source this so the reference host cannot silently lose
# 9-13% and corrupt an A/B; that is the failure mode the default-off choice
# would otherwise create. Detection reads /proc/cpuinfo rather than a hostname,
# so it is correct on any node, including ones nobody has calibrated.
#
# Full data: research/notes/2026-09-10-hot-path-code-size.md (the mechanism)
# and research/notes/2026-09-10-jcc-portability.md (the cross-node campaign).
#
# NOTE: an exported RUSTFLAGS **replaces** cargo's config rustflags wholesale.
# Append, never assign:  RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$JCC_RUSTFLAGS"

declare -g JCC_RUSTFLAGS=""

_jcc_applies() {
    [ -r /proc/cpuinfo ] || return 1
    local vendor family model
    vendor=$(awk -F': ' '/^vendor_id/{print $2; exit}' /proc/cpuinfo)
    family=$(awk -F': ' '/^cpu family/{print $2; exit}' /proc/cpuinfo)
    model=$(awk -F': ' '/^model\t/{print $2; exit}' /proc/cpuinfo)
    [ "$vendor" = "GenuineIntel" ] && [ "$family" = 6 ] || return 1
    case "$model" in
        # SKL/SKL-X, SKX/CLX/CPX, KBL/CFL/WHL, CML. Ice Lake (106/108/125/126)
        # and later dropped the erratum and are deliberately absent.
        78|94|85|142|158|165|166) return 0 ;;
        *) return 1 ;;
    esac
}

if _jcc_applies; then
    JCC_RUSTFLAGS="-Cllvm-args=-x86-branches-within-32B-boundaries"
fi
unset -f _jcc_applies
