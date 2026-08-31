#!/usr/bin/env python3
"""Regenerate ``heavy_hex_127.edges`` from a real IBM Eagle coupling map.

The 127-qubit heavy-hex edge list is *generated*, never hand-typed
(``research/plans/2026-08-31-examples-benchmarks-suite.md`` §6, Part 0.1): a
144-entry adjacency list transcribed by hand is a silent-wrong-answer waiting
to happen, and the generated file carries its own provenance header so a reader
can tell exactly which device map it came from.

Source of truth: the ``FakeSherbrooke`` backend snapshot shipped by
``qiskit-ibm-runtime``, i.e. the recorded configuration of IBM's 127-qubit
``ibm_sherbrooke`` (Eagle r3) processor. Qubit indices are the device's own
numbering, 0..126, which is also the numbering used by published experiments on
Eagle-family devices.

``qiskit-ibm-runtime`` is a **dev-only** dependency: nothing in
``examples/common/`` imports it, because the generated file is checked in. Run
this script only to refresh the file (e.g. after a package upgrade), and commit
the result together with any diff it produces.

Usage::

    source .venv/bin/activate
    pip install qiskit-ibm-runtime            # dev dep, not in the examples extra
    python examples/data/generate_heavy_hex.py            # writes the .edges file
    python examples/data/generate_heavy_hex.py --check     # exit 1 if stale

Provider fallbacks, tried in order (qiskit >= 2.0 dropped its bundled fake
backends, so on a modern qiskit only the first one exists):

1. ``qiskit_ibm_runtime.fake_provider.FakeSherbrooke``
2. ``qiskit.providers.fake_provider.FakeSherbrooke`` (qiskit < 1.0 layout)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Eagle r3 structural facts, asserted against whatever the provider hands back
# so a provider change cannot quietly swap in a different topology.
EAGLE_NUM_QUBITS = 127
EAGLE_NUM_EDGES = 144
EAGLE_DEGREES = {1, 2, 3}

OUTPUT_NAME = "heavy_hex_127.edges"


def _load_backend():
    """Return ``(backend, package_name, package_version)`` or raise."""
    errors = []
    try:
        import qiskit_ibm_runtime
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

        return FakeSherbrooke(), "qiskit-ibm-runtime", qiskit_ibm_runtime.__version__
    except Exception as exc:  # pragma: no cover - depends on the environment
        errors.append(f"qiskit_ibm_runtime.fake_provider: {exc!r}")

    try:  # pragma: no cover - only on qiskit < 1.0
        import qiskit
        from qiskit.providers.fake_provider import FakeSherbrooke

        return FakeSherbrooke(), "qiskit", qiskit.__version__
    except Exception as exc:  # pragma: no cover
        errors.append(f"qiskit.providers.fake_provider: {exc!r}")

    raise RuntimeError(
        "no provider supplies a 127-qubit Eagle coupling map. Tried:\n  "
        + "\n  ".join(errors)
        + "\n\nInstall the dev dependency:  pip install qiskit-ibm-runtime"
    )


def extract_edges(backend) -> list[tuple[int, int]]:
    """Undirected, deduplicated, sorted ``(lo, hi)`` edges of the coupling map."""
    coupling_map = backend.coupling_map
    if coupling_map is None:  # pragma: no cover
        raise RuntimeError(f"backend {backend.name!r} exposes no coupling map")
    edges = {(min(a, b), max(a, b)) for a, b in coupling_map.get_edges()}
    return sorted(edges)


def degree_histogram(edges) -> Counter:
    degree: Counter = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    return degree


def validate(edges, num_qubits: int) -> Counter:
    """Assert the Eagle r3 heavy-hex structure; return the degree map."""
    if num_qubits != EAGLE_NUM_QUBITS:
        raise ValueError(f"expected {EAGLE_NUM_QUBITS} qubits, backend reports {num_qubits}")
    if len(edges) != EAGLE_NUM_EDGES:
        raise ValueError(f"expected {EAGLE_NUM_EDGES} undirected edges, got {len(edges)}")
    degree = degree_histogram(edges)
    if len(degree) != num_qubits:
        missing = sorted(set(range(num_qubits)) - set(degree))
        raise ValueError(f"isolated qubits in the coupling map: {missing}")
    bad = {q: d for q, d in degree.items() if d not in EAGLE_DEGREES}
    if bad:
        raise ValueError(f"degrees outside {sorted(EAGLE_DEGREES)}: {bad}")
    return degree


def render(edges, degree: Counter, num_qubits: int, package: str, version: str) -> str:
    hist = Counter(degree.values())
    lines = [
        "# 127-qubit heavy-hex coupling map (IBM Eagle r3).",
        "#",
        "# GENERATED FILE - do not edit by hand.",
        "#   regenerate with: python examples/data/generate_heavy_hex.py",
        "#",
        f"# source package : {package} {version}",
        "# source object  : fake_provider.FakeSherbrooke().coupling_map",
        "#                  (recorded configuration of IBM's 127-qubit ibm_sherbrooke,",
        "#                   Eagle r3 processor); qubit indices are the device's own",
        "#                   numbering 0..126.",
        f"# nodes          : {num_qubits}",
        f"# undirected edges: {len(edges)}",
        "# degree histogram: "
        + ", ".join(f"degree {d}: {c} qubits" for d, c in sorted(hist.items())),
        "#",
        "# Format: one undirected edge per line, `lo hi`, both indices decimal,",
        "# sorted lexicographically. Comment lines start with '#'.",
    ]
    lines.extend(f"{a} {b}" for a, b in edges)
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the checked-in file differs in its edge list",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_NAME,
        help=f"output path (default: alongside this script, {OUTPUT_NAME})",
    )
    args = parser.parse_args(argv)

    backend, package, version = _load_backend()
    edges = extract_edges(backend)
    degree = validate(edges, backend.num_qubits)
    text = render(edges, degree, backend.num_qubits, package, version)

    if args.check:
        if not args.output.exists():
            print(f"{args.output} does not exist", file=sys.stderr)
            return 1
        # Compare edge lists, not bytes: the header carries a package version
        # that legitimately drifts without the topology changing.
        want = [ln for ln in text.splitlines() if not ln.startswith("#")]
        got = [ln for ln in args.output.read_text().splitlines() if not ln.startswith("#")]
        if want != got:
            print(f"{args.output} is stale (edge list differs)", file=sys.stderr)
            return 1
        print(f"{args.output}: up to date ({len(edges)} edges)")
        return 0

    args.output.write_text(text)
    print(f"wrote {args.output}: {backend.num_qubits} nodes, {len(edges)} edges (from {package} {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
