#!/usr/bin/env bash
# Sourced, never executed: on a CPU with the JCC erratum (SKX102) it sets JCC_RUSTFLAGS to the branch-padding flag and appends it to RUSTFLAGS once; elsewhere both are left alone.
# Why the flag is opt-in rather than in .cargo/config.toml: research/FINDINGS.md §JCC branch padding, opt-in.
# An exported RUSTFLAGS replaces .cargo/config.toml's rustflags wholesale, so this appends and never assigns; sourcing it again (a nested script) changes nothing, so cargo does not rebuild.

declare -g JCC_RUSTFLAGS=""

_jcc_applies() {
    [ -r /proc/cpuinfo ] || return 1
    local vendor family model
    vendor=$(awk -F': ' '/^vendor_id/{print $2; exit}' /proc/cpuinfo)
    family=$(awk -F': ' '/^cpu family/{print $2; exit}' /proc/cpuinfo)
    model=$(awk -F': ' '/^model\t/{print $2; exit}' /proc/cpuinfo)
    [ "$vendor" = "GenuineIntel" ] && [ "$family" = 6 ] || return 1
    case "$model" in
        # SKL/SKL-X, SKX/CLX/CPX, KBL/CFL/WHL, CML; Ice Lake and later have no erratum.
        78|94|85|142|158|165|166) return 0 ;;
        *) return 1 ;;
    esac
}

if _jcc_applies; then
    JCC_RUSTFLAGS="-Cllvm-args=-x86-branches-within-32B-boundaries"
    case " ${RUSTFLAGS:-} " in
        *" $JCC_RUSTFLAGS "*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$JCC_RUSTFLAGS" ;;
    esac
fi
unset -f _jcc_applies
