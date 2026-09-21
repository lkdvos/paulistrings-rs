"""Reference docs (docs/book/src/reference/) are handwritten and drift silently
(see the Documentation Build Handout, review finding 6). These tests catch the
direction drift actually happens in: a kwarg or a gate/noise method added to
the extension and not added to the docs.

They do not catch prose going stale, only names disappearing from the tables.
"""

import inspect
import re
from pathlib import Path

import paulistrings
from paulistrings import Circuit, gates, noise

BOOK_SRC = Path(__file__).resolve().parents[3] / "docs" / "book" / "src"


def _code_spans(text: str) -> set[str]:
    """Every `backtick`-quoted token in a markdown file, stripped of parens/args.

    Fenced code blocks are stripped first: an odd number of literal backticks
    inside one (e.g. a ```text function-signature block) throws off single-
    backtick pairing for everything that follows in the file.
    """
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    spans = set()
    for raw in re.findall(r"`([^`]+)`", text):
        name = raw.split("(")[0].strip().lstrip(".")
        if name:
            spans.add(name)
    return spans


def test_propagate_reference_lists_every_kwarg():
    doc = (BOOK_SRC / "reference" / "propagate.md").read_text()
    documented = _code_spans(doc)

    sig = inspect.signature(paulistrings.PauliSum.propagate)
    actual_kwargs = {
        name
        for name, param in sig.parameters.items()
        if name not in ("self", "circuit") and param.default is not inspect.Parameter.empty
    }

    missing = actual_kwargs - documented
    assert not missing, f"propagate() kwargs missing from reference/propagate.md: {missing}"


def test_circuit_reference_lists_every_gate_and_noise_method():
    doc = (BOOK_SRC / "reference" / "circuit.md").read_text()
    documented = _code_spans(doc)

    # Composition/introspection surface documented in its own section, not
    # the Gates/Noise tables.
    non_gate_methods = {"append", "extend", "adjoint", "num_qubits", "gates"}

    circuit_methods = {
        name
        for name in dir(Circuit)
        if not name.startswith("_") and name not in non_gate_methods
    }
    gate_factories = {name for name in dir(gates) if not name.startswith("_")}
    noise_factories = {name for name in dir(noise) if not name.startswith("_")}

    missing = (circuit_methods | gate_factories | noise_factories) - documented
    assert not missing, f"Circuit gate/noise surface missing from reference/circuit.md: {missing}"
