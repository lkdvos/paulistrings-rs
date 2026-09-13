"""Hardware preflight check for campaign-2026-09-11 (T07).

Verifies the *actual* CPU topology of the host this process is running on
against the frozen `genoa` hardware contract (`contract.md#hardware-contract`),
without trusting Slurm's `ThreadsPerCore` field or any hostname convention.

Two independent checks, both required to pass:

1. **Physical-core-only affinity.** Build `{logical_cpu: (socket, core)}` from
   `lscpu -p=CPU,CORE,SOCKET,ONLINE` and intersect with
   `os.sched_getaffinity(0)`. If two allowed logical CPUs map to the same
   `(socket, core)` pair, this process can run on both threads of one SMT
   pair — the actual proof of "physical cores only", not Slurm's advertised
   `ThreadsPerCore=1`.
2. **CPU model fingerprint.** `genoa` vs `rome` vs `icelake` from
   `/proc/cpuinfo`'s `vendor_id`/`cpu family`/`model` fields (not the free-text
   `model name` string, which for Intel Xeon Scalable generations does not
   reliably encode microarchitecture). See `_fingerprint_node_class` for the
   exact rule and its confidence caveats.

Importable as a function (`run_preflight()`) and runnable as a CLI
(`python preflight.py`, prints the dict as JSON and exits 1 if it fails).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HARDWARE_CONTRACT_ID = "genoa"

CPUINFO_PATH = Path("/proc/cpuinfo")

__all__ = [
    "HARDWARE_CONTRACT_ID",
    "parse_cpuinfo",
    "parse_lscpu_p",
    "fingerprint_node_class",
    "build_report",
    "run_preflight",
]


# --------------------------------------------------------------------------
# Parsing (pure functions, taking text so tests never touch the real host)
# --------------------------------------------------------------------------


def parse_cpuinfo(text: str) -> list[dict[str, str]]:
    """One dict per logical CPU (`processor` block) from `/proc/cpuinfo` text."""
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    if current:
        blocks.append(current)
    return blocks


def parse_lscpu_p(text: str) -> dict[int, tuple[int, int, bool]]:
    """`lscpu -p=CPU,CORE,SOCKET,ONLINE` text -> `{cpu: (socket, core, online)}`.

    Comment lines (`#...`) are skipped. `ONLINE` is `Y`/`N`; some `lscpu`
    versions leave it blank for CPU 0 (which cannot be offlined), treated as
    online.
    """
    out: dict[int, tuple[int, int, bool]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        cpu, core, socket = int(parts[0]), int(parts[1]), int(parts[2])
        online_raw = parts[3].strip() if len(parts) > 3 else ""
        online = online_raw != "N"
        out[cpu] = (socket, core, online)
    return out


# --------------------------------------------------------------------------
# CPU model fingerprint
# --------------------------------------------------------------------------

#: (vendor_id, cpu family, model) -> node class, plus a confidence label and
#: source note. This is deliberately NOT keyed on the free-text `model name`
#: string: Intel's Xeon Scalable `model name` field (e.g. "Intel(R) Xeon(R)
#: Gold 6244 CPU @ 3.60GHz") does not encode microarchitecture generation at
#: all, so a substring match against "Ice Lake" would silently never fire on
#: real hardware and give false confidence. `vendor_id`/`cpu family`/`model`
#: are the CPUID-derived fields the kernel itself uses to distinguish
#: microarchitectures (`arch/x86/include/asm/intel-family.h`,
#: `arch/x86/kernel/cpu/amd.c` upstream).
#:
#: Sourced 2026-09-11 via web search against secondary sources (vendor
#: launch coverage, kernel patch threads), NOT AMD's/Intel's primary
#: Processor Programming Reference or the kernel source tree directly (no
#: repo checkout of either was available in this task). Confidence is
#: therefore MEDIUM, not HIGH — see the module docstring and the T07 report's
#: explicit escalation note.
#:
#: AMD family 25 (0x19) spans multiple Zen4 products: Genoa (EPYC 9004,
#: model 0x10 engineering / 0x11 production) and Bergamo (Zen4c, EPYC 9004
#: "c" SKUs) are NOT distinguished by this table — both are family 25, and
#: this task could not verify Bergamo's model range against a primary
#: source, so a Bergamo host would currently be misreported as "genoa". This
#: is exactly the ambiguity CLAUDE.md's escalation policy asks to flag
#: rather than paper over.
_FINGERPRINTS: dict[tuple[str, int, int], tuple[str, str]] = {
    ("AuthenticAMD", 25, 16): ("genoa", "AMD family 19h model 10h: Genoa engineering sample"),
    ("AuthenticAMD", 25, 17): ("genoa", "AMD family 19h model 11h: Genoa production"),
    ("AuthenticAMD", 23, 49): ("rome", "AMD family 17h model 31h: EPYC 7002 Rome (Zen2)"),
    ("GenuineIntel", 6, 106): ("icelake", "Intel family 6 model 0x6A: Ice Lake-SP"),
    ("GenuineIntel", 6, 108): ("icelake", "Intel family 6 model 0x6C: Ice Lake-D"),
}


def fingerprint_node_class(vendor_id: str, family: int, model: int) -> tuple[str, str]:
    """`(node_class, note)`. `node_class` is `"unknown"` when the
    `(vendor_id, family, model)` triple is not in `_FINGERPRINTS` — this
    function never guesses from the free-text model-name string.
    """
    key = (vendor_id, family, model)
    if key in _FINGERPRINTS:
        return _FINGERPRINTS[key]
    return (
        "unknown",
        f"no fingerprint entry for vendor_id={vendor_id!r} family={family} model={model} "
        "(see _FINGERPRINTS' docstring: only genoa/rome/icelake are covered, at MEDIUM "
        "confidence from secondary sources)",
    )


def _int_field(raw: str | None) -> int | None:
    if raw is None:
        return None
    match = re.match(r"\s*(\d+)", raw)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def build_report(
    cpuinfo_text: str,
    lscpu_p_text: str,
    affinity: set[int],
) -> dict:
    """Pure function version of `run_preflight`, given the three raw inputs.

    Kept separate from `run_preflight` so tests can supply synthetic
    `/proc/cpuinfo`-shaped text and `lscpu -p` text without touching the host.
    """
    cpu_blocks = parse_cpuinfo(cpuinfo_text)
    lscpu_map = parse_lscpu_p(lscpu_p_text)

    cpu_model = "unknown"
    vendor_id = "unknown"
    family: int | None = None
    model: int | None = None
    for block in cpu_blocks:
        if "model name" in block and cpu_model == "unknown":
            cpu_model = block["model name"]
        if "vendor_id" in block and vendor_id == "unknown":
            vendor_id = block["vendor_id"]
        if family is None:
            family = _int_field(block.get("cpu family"))
        if model is None:
            model = _int_field(block.get("model"))
        if cpu_model != "unknown" and vendor_id != "unknown" and family is not None and model is not None:
            break

    node_class_guess, fingerprint_note = (
        fingerprint_node_class(vendor_id, family, model)
        if family is not None and model is not None
        else ("unknown", "missing 'cpu family'/'model' fields in /proc/cpuinfo")
    )

    allowed = sorted(cpu for cpu in affinity if cpu in lscpu_map)
    missing_from_lscpu = sorted(cpu for cpu in affinity if cpu not in lscpu_map)

    seen_pairs: dict[tuple[int, int], int] = {}
    smt_siblings_present = False
    for cpu in allowed:
        socket, core, _online = lscpu_map[cpu]
        pair = (socket, core)
        if pair in seen_pairs:
            smt_siblings_present = True
        else:
            seen_pairs[pair] = cpu

    physical_cores_available = len(seen_pairs)
    affinity_is_physical_core_only = bool(allowed) and not smt_siblings_present and not missing_from_lscpu

    hardware_contract_id = HARDWARE_CONTRACT_ID
    hardware_matches = node_class_guess == hardware_contract_id
    preflight_passed = bool(
        hardware_matches and affinity_is_physical_core_only and not smt_siblings_present
    )

    report = {
        "hardware_contract_id": hardware_contract_id,
        "cpu_model": cpu_model,
        "cpu_vendor_id": vendor_id,
        "cpu_family": family,
        "cpu_model_number": model,
        "node_class_guess": node_class_guess,
        "node_class_fingerprint_note": fingerprint_note,
        "physical_cores_available": physical_cores_available,
        "smt_siblings_present": smt_siblings_present,
        "affinity_is_physical_core_only": affinity_is_physical_core_only,
        "affinity_cpu_count": len(affinity),
        "affinity_cpus_missing_from_lscpu": missing_from_lscpu,
        "preflight_passed": preflight_passed,
    }
    return report


def _lscpu_p_text() -> str:
    result = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def run_preflight() -> dict:
    """The real-host preflight: reads `/proc/cpuinfo`, runs `lscpu -p`, reads
    `os.sched_getaffinity(0)`. Never touches Slurm.
    """
    import os

    cpuinfo_text = CPUINFO_PATH.read_text()
    lscpu_text = _lscpu_p_text()
    affinity = set(os.sched_getaffinity(0))
    return build_report(cpuinfo_text, lscpu_text, affinity)


def main(argv: list[str] | None = None) -> int:
    report = run_preflight()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["preflight_passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
