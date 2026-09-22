# B5 — Hybrid depth reduction

Splits a circuit near the end and back-propagates the observable through the tail classically in the Heisenberg picture, so a QPU only has to run the shorter front circuit against a modified observable.
A schema-v1 task file carries the front circuit and evolved observable to a QPU-side runner.

```bash
source .venv/bin/activate
python examples/b5_operator_backpropagation/run_b5.py
pytest examples/tests/test_showcase_b5.py
```

The script rewrites every artifact in this directory: both task JSONs, the `.npz`, `depth_vs_terms.csv`, and both SVG figures.

Full writeup, headline numbers, and provenance: https://lkdvos.github.io/paulistrings-rs/examples/showcases/b5-operator-backpropagation.html
