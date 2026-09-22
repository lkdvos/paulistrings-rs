"""Catches a `propagate`/`Circuit` kwarg or gate/noise method added, renamed, or removed without updating docs/book/src/library/.

Skipped outside a repo checkout (an installed wheel ships this test file but not `docs/`).
"""

import inspect
import re
from pathlib import Path

import pytest

import paulistrings
from paulistrings import Circuit, gates, noise

BOOK_SRC = Path(__file__).resolve().parents[3] / "docs" / "book" / "src"

pytestmark = pytest.mark.skipif(
    not BOOK_SRC.is_dir(), reason="docs/book/src not present (installed wheel, not a repo checkout)"
)


def _strip_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _table_rows(text: str, header: str) -> list[list[str]]:
    """Cells of every row in the pipe table whose header line contains `header`, past the `---` separator."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("|") and header in line), None)
    if start is None:
        raise AssertionError(f"no table with header containing {header!r} in the reference page")
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def _backtick_name(cell: str) -> str | None:
    match = re.search(r"`([^`]+)`", cell)
    if not match:
        return None
    return match.group(1).split("(")[0].strip().lstrip(".")


def _factory_name(cell: str) -> str | None:
    """`_backtick_name`, then dropping a `gates.`/`noise.` module prefix."""
    name = _backtick_name(cell)
    if name is None:
        return None
    return name.split(".", 1)[1] if "." in name else name


def test_propagate_reference_lists_exactly_the_actual_kwargs():
    doc = _strip_fences((BOOK_SRC / "library" / "propagate.md").read_text())
    rows = _table_rows(doc, "Parameter")
    documented = {_backtick_name(row[0]) for row in rows if _backtick_name(row[0])}
    documented -= {"circuit"}  # positional, listed for completeness, not a kwarg

    sig = inspect.signature(paulistrings.PauliSum.propagate)
    actual_kwargs = {
        name
        for name, param in sig.parameters.items()
        if name not in ("self", "circuit") and param.default is not inspect.Parameter.empty
    }

    assert documented == actual_kwargs, (
        f"reference/propagate.md's parameter table disagrees with propagate()'s actual kwargs: "
        f"missing {actual_kwargs - documented}, stale {documented - actual_kwargs}"
    )


def test_circuit_reference_lists_exactly_the_actual_gate_and_noise_surface():
    doc = _strip_fences((BOOK_SRC / "library" / "circuit.md").read_text())
    gate_rows = _table_rows(doc, "gates.")
    noise_rows = _table_rows(doc, "noise.")

    documented_methods = {_backtick_name(r[0]) for r in gate_rows + noise_rows if _backtick_name(r[0])}
    documented_gate_factories = {_factory_name(r[1]) for r in gate_rows if _factory_name(r[1])}
    documented_noise_factories = {_factory_name(r[1]) for r in noise_rows if _factory_name(r[1])}

    # Composition/introspection surface documented in its own prose section, not these tables.
    non_gate_methods = {"append", "extend", "adjoint", "num_qubits", "gates"}
    circuit_methods = {name for name in dir(Circuit) if not name.startswith("_") and name not in non_gate_methods}
    gate_factories = {name for name in dir(gates) if not name.startswith("_")}
    noise_factories = {name for name in dir(noise) if not name.startswith("_")}

    assert documented_methods == circuit_methods, (
        f"library/circuit.md's method columns disagree with Circuit's actual gate/noise methods: "
        f"missing {circuit_methods - documented_methods}, stale {documented_methods - circuit_methods}"
    )
    assert documented_gate_factories == gate_factories, (
        f"library/circuit.md's gates. column disagrees with the gates module: "
        f"missing {gate_factories - documented_gate_factories}, stale {documented_gate_factories - gate_factories}"
    )
    assert documented_noise_factories == noise_factories, (
        f"library/circuit.md's noise. column disagrees with the noise module: "
        f"missing {noise_factories - documented_noise_factories}, stale {documented_noise_factories - noise_factories}"
    )
