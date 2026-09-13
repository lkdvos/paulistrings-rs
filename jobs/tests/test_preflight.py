"""Unit tests for `preflight.py`, against synthetic fixtures only.

No real `/proc/cpuinfo`/`lscpu` access: `build_report` takes text/affinity
directly, so these tests pin the logic on hosts that are not `genoa` (this
one included) without needing a real allocation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import preflight  # noqa: E402


def _cpuinfo_block(vendor_id: str, family: int, model: int, model_name: str) -> str:
    return (
        f"vendor_id\t: {vendor_id}\n"
        f"cpu family\t: {family}\n"
        f"model\t\t: {model}\n"
        f"model name\t: {model_name}\n"
    )


def _cpuinfo_text(n_cpus: int, vendor_id: str, family: int, model: int, model_name: str) -> str:
    block = _cpuinfo_block(vendor_id, family, model, model_name)
    return "\n\n".join(f"processor\t: {i}\n{block}" for i in range(n_cpus)) + "\n"


def _lscpu_p_text(rows: list[tuple[int, int, int, str]]) -> str:
    header = (
        "# The following is the parsable format, which can be fed to other\n"
        "# programs. Each different item in every column has an unique ID\n"
        "# starting usually from zero.\n"
        "# CPU,Core,Socket,Online\n"
    )
    lines = [f"{cpu},{core},{socket},{online}" for cpu, core, socket, online in rows]
    return header + "\n".join(lines) + "\n"


def _genoa_2socket_smt(cores_per_socket: int = 4):
    """2 sockets x `cores_per_socket` physical cores x 2 SMT threads, Genoa."""
    rows = []
    cpu = 0
    # Physical threads first (0..N-1), SMT siblings after (N..2N-1) — a
    # common Linux enumeration, though not the only one lscpu can produce.
    physical = [(socket, core) for socket in range(2) for core in range(cores_per_socket)]
    for socket, core in physical:
        rows.append((cpu, core, socket, "Y"))
        cpu += 1
    for socket, core in physical:
        rows.append((cpu, core, socket, "Y"))
        cpu += 1
    return _lscpu_p_text(rows), physical


GENOA_CPUINFO = _cpuinfo_text(16, "AuthenticAMD", 25, 17, "AMD EPYC 9454 48-Core Processor")
ROME_CPUINFO = _cpuinfo_text(16, "AuthenticAMD", 23, 49, "AMD EPYC 7742 64-Core Processor")
ICELAKE_CPUINFO = _cpuinfo_text(16, "GenuineIntel", 6, 106, "Intel(R) Xeon(R) Platinum 8358")
CASCADE_LAKE_CPUINFO = _cpuinfo_text(16, "GenuineIntel", 6, 85, "Intel(R) Xeon(R) Gold 6244 CPU @ 3.60GHz")


def test_parse_cpuinfo_splits_blocks_on_blank_lines():
    blocks = preflight.parse_cpuinfo(GENOA_CPUINFO)
    assert len(blocks) == 16
    assert blocks[0]["vendor_id"] == "AuthenticAMD"
    assert blocks[0]["model"] == "17"


def test_parse_lscpu_p_reads_cpu_core_socket_online():
    text, physical = _genoa_2socket_smt(cores_per_socket=2)
    got = preflight.parse_lscpu_p(text)
    assert got[0] == (0, 0, True)
    # SMT sibling of cpu 0 is cpu len(physical), same (socket, core).
    sibling_cpu = len(physical)
    assert got[sibling_cpu][:2] == got[0][:2]


def test_fingerprint_genoa():
    node_class, _note = preflight.fingerprint_node_class("AuthenticAMD", 25, 17)
    assert node_class == "genoa"


def test_fingerprint_rome():
    node_class, _note = preflight.fingerprint_node_class("AuthenticAMD", 23, 49)
    assert node_class == "rome"


def test_fingerprint_icelake():
    node_class, _note = preflight.fingerprint_node_class("GenuineIntel", 6, 106)
    assert node_class == "icelake"


def test_fingerprint_unknown_for_unrecognized_triple():
    node_class, note = preflight.fingerprint_node_class("GenuineIntel", 6, 85)
    assert node_class == "unknown"
    assert "no fingerprint entry" in note


def test_genoa_physical_core_only_affinity_passes():
    text, physical = _genoa_2socket_smt(cores_per_socket=4)
    # Affinity restricted to the first physical thread of each core (cpus 0..7).
    affinity = set(range(len(physical)))
    report = preflight.build_report(GENOA_CPUINFO, text, affinity)
    assert report["node_class_guess"] == "genoa"
    assert report["smt_siblings_present"] is False
    assert report["affinity_is_physical_core_only"] is True
    assert report["physical_cores_available"] == len(physical)
    assert report["preflight_passed"] is True


def test_genoa_with_smt_sibling_in_affinity_fails():
    text, physical = _genoa_2socket_smt(cores_per_socket=4)
    # Affinity includes cpu 0 and its SMT sibling (cpu len(physical)).
    affinity = {0, len(physical)}
    report = preflight.build_report(GENOA_CPUINFO, text, affinity)
    assert report["smt_siblings_present"] is True
    assert report["affinity_is_physical_core_only"] is False
    assert report["preflight_passed"] is False


def test_rome_fails_contract_even_with_clean_affinity():
    text, physical = _genoa_2socket_smt(cores_per_socket=4)
    affinity = set(range(len(physical)))
    report = preflight.build_report(ROME_CPUINFO, text, affinity)
    assert report["node_class_guess"] == "rome"
    assert report["affinity_is_physical_core_only"] is True
    assert report["preflight_passed"] is False  # hardware contract mismatch


def test_icelake_fails_contract():
    text, physical = _genoa_2socket_smt(cores_per_socket=4)
    affinity = set(range(len(physical)))
    report = preflight.build_report(ICELAKE_CPUINFO, text, affinity)
    assert report["node_class_guess"] == "icelake"
    assert report["preflight_passed"] is False


def test_unfingerprinted_cascade_lake_reports_unknown_not_a_guess():
    """This repo's own reference workstation (ccqlin038, Cascade Lake) is
    exactly the case the fingerprint table does not cover: it must come back
    `unknown`, never silently misreported as one of the three contract
    classes."""
    text, physical = _genoa_2socket_smt(cores_per_socket=4)
    affinity = set(range(len(physical)))
    report = preflight.build_report(CASCADE_LAKE_CPUINFO, text, affinity)
    assert report["node_class_guess"] == "unknown"
    assert report["preflight_passed"] is False


def test_affinity_cpu_missing_from_lscpu_is_not_physical_core_only():
    text, physical = _genoa_2socket_smt(cores_per_socket=2)
    affinity = set(range(len(physical))) | {999}
    report = preflight.build_report(GENOA_CPUINFO, text, affinity)
    assert 999 in report["affinity_cpus_missing_from_lscpu"]
    assert report["affinity_is_physical_core_only"] is False


def test_empty_affinity_does_not_pass():
    text, _physical = _genoa_2socket_smt(cores_per_socket=2)
    report = preflight.build_report(GENOA_CPUINFO, text, set())
    assert report["affinity_is_physical_core_only"] is False
    assert report["preflight_passed"] is False
